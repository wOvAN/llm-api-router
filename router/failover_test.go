package router

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"llm-api-router/config"
	"llm-api-router/domain"
)

// A backend that answers 429 is throttled, not wrong: the client's request is
// fine and the next attempt may serve it. The proxy buffered that answer (no
// byte of it reached the client), so the router fails over instead of relaying
// it — and a 429 buys itself no same-server replay, because a backend that said
// "back off" does not change its mind while we sleep between tries.
func TestHandleUpstream429FailsOverAtOnce(t *testing.T) {
	var primaryCalls, fallbackCalls atomic.Int32
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		primaryCalls.Add(1)
		w.Header().Set("Retry-After", "20")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"type":"rate_limit_exceeded","message":"slow down"}}`))
	}))
	defer primary.Close()

	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fallbackCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"backup-model","choices":[{"finish_reason":"stop"}]}`))
	}))
	defer fallback.Close()

	r, store, ms, _ := newTestRouter(t)
	rl := config.NewRateLimiter(5, 60*time.Second, 5*time.Minute)
	r = New(store, r.metrics, nil, rl, nil, nil)

	for _, s := range []*domain.Server{
		{ID: "primary", Name: "Primary", URL: primary.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
		{ID: "fallback", Name: "Fallback", URL: fallback.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
	} {
		if err := store.AddServer(s); err != nil {
			t.Fatalf("AddServer: %v", err)
		}
	}
	if err := store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "primary",
		Fallbacks:      []domain.FallbackEntry{{ServerID: "fallback", TargetModel: "backup-model"}},
		Enabled:        true,
	}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 from the fallback: %s", res.StatusCode, w.Body.String())
	}
	if got := primaryCalls.Load(); got != 1 {
		t.Errorf("primary hit %d times, want 1 (a 429 is not replayed)", got)
	}
	if got := fallbackCalls.Load(); got != 1 {
		t.Errorf("fallback hit %d times, want 1", got)
	}
	if got := res.Header.Get("X-Router-Attempts"); got != "2/2" {
		t.Errorf("X-Router-Attempts = %q, want 2/2", got)
	}
	// The throttle still counts: the backend said 20s, so 20s is the cooldown.
	if d := rl.CooldownRemaining("primary"); d > 21*time.Second || d <= 0 {
		t.Errorf("cooldown = %v, want the backend's Retry-After (20s)", d)
	}
	if rec := ms.Recent(); len(rec) != 1 || rec[0].StatusCode != http.StatusOK || !rec[0].WasFallback {
		t.Errorf("recent = %+v, want one successful fallback request", rec)
	}
}

// Every backend failed and none of them got a byte to the client, so the
// client's answer is the last backend's own — status, headers and body — not a
// proxy-invented 502 quoting the backends. A client that sent a 12k-token
// prompt and got "context length exceeded" from every candidate needs to read
// that reason, and 0/10 of the routers it talks to speak our 502.
func TestHandleAllBackendsFailServeLastBackendAnswer(t *testing.T) {
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Retry-After", "5")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":{"type":"server_error","message":"boom-one"}}`))
	}))
	defer primary.Close()

	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":{"type":"overloaded","message":"boom-three"}}`))
	}))
	defer fallback.Close()

	r, store, ms, _ := newTestRouter(t)
	for _, s := range []*domain.Server{
		{ID: "primary", Name: "Primary", URL: primary.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
		{ID: "fallback", Name: "Fallback", URL: fallback.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
	} {
		if err := store.AddServer(s); err != nil {
			t.Fatalf("AddServer: %v", err)
		}
	}
	if err := store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "primary",
		Fallbacks:      []domain.FallbackEntry{{ServerID: "fallback"}},
		Enabled:        true,
	}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want the last backend's 503", res.StatusCode)
	}
	if body := w.Body.String(); !strings.Contains(body, "boom-three") || strings.Contains(body, "all backends failed") {
		t.Fatalf("body = %q, want the last backend's answer verbatim", body)
	}
	if got := res.Header.Get("Content-Type"); !strings.Contains(got, "application/json") {
		t.Errorf("Content-Type = %q, want the backend's", got)
	}
	if got := res.Header.Get("X-Router-Server"); got != "fallback" {
		t.Errorf("X-Router-Server = %q, want fallback", got)
	}
	// The primary's Retry-After still cools it down — the answer was buffered,
	// not forwarded, so the proxy never saw a served response.
	if rec := ms.Recent(); len(rec) != 1 || rec[0].StatusCode != http.StatusServiceUnavailable {
		t.Errorf("recent = %+v, want one request recorded with the served status", rec)
	}
}

// The other side of the same line: a 4xx other than 429 is the client's request
// being wrong, which reads identically at every backend. Failing over would
// spend a full generation on a request that deserves the answer it got.
func TestHandleUpstream4xxStaysVerbatim(t *testing.T) {
	var primaryCalls, fallbackCalls atomic.Int32
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		primaryCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"max_tokens must be > 0"}}`))
	}))
	defer primary.Close()

	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fallbackCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`))
	}))
	defer fallback.Close()

	r, store, _, _ := newTestRouter(t)
	for _, s := range []*domain.Server{
		{ID: "primary", Name: "Primary", URL: primary.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
		{ID: "fallback", Name: "Fallback", URL: fallback.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
	} {
		if err := store.AddServer(s); err != nil {
			t.Fatalf("AddServer: %v", err)
		}
	}
	if err := store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "primary",
		Fallbacks:      []domain.FallbackEntry{{ServerID: "fallback"}},
		Enabled:        true,
	}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want the backend's 400", res.StatusCode)
	}
	if got := primaryCalls.Load(); got != 1 {
		t.Errorf("primary hit %d times, want 1", got)
	}
	if got := fallbackCalls.Load(); got != 0 {
		t.Errorf("fallback hit %d times, want 0 (a 400 is not a failover trigger)", got)
	}
}

// A 5xx the router buffered is a failure that can move to the next backend, so
// it takes the retry path — and it keeps the retry budget the operator asked
// for (an explicit num_retries replays a backend that is still warming up),
// while the generous transport-only default stays reserved for a backend that
// never answered at all.
func TestHandleBuffered5xxKeepsExplicitRetryBudget(t *testing.T) {
	var calls, fallbackCalls atomic.Int32
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) <= 2 {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":{"message":"warming up"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`))
	}))
	defer primary.Close()

	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fallbackCalls.Add(1)
	}))
	defer fallback.Close()

	r, store, _, _ := newTestRouter(t)
	for _, s := range []*domain.Server{
		{ID: "primary", Name: "Primary", URL: primary.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
		{ID: "fallback", Name: "Fallback", URL: fallback.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
	} {
		if err := store.AddServer(s); err != nil {
			t.Fatalf("AddServer: %v", err)
		}
	}
	if err := store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "primary",
		NumRetries:     2,
		Fallbacks:      []domain.FallbackEntry{{ServerID: "fallback"}},
		Enabled:        true,
	}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if got := w.Result().StatusCode; got != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", got, w.Body.String())
	}
	if got := calls.Load(); got != 3 {
		t.Errorf("primary hit %d times, want 3 (2 buffered 5xx + the success)", got)
	}
	if got := fallbackCalls.Load(); got != 0 {
		t.Errorf("fallback hit %d times, want 0", got)
	}
}
