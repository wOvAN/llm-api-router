package router

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"llm-api-router/config"
	"llm-api-router/domain"
	"llm-api-router/metrics"
	"llm-api-router/proxy"
	"llm-api-router/trace"
)

// newRouterWithSettings is newTestRouter over a config that carries settings — the
// conversion knobs (`drop_params`) are config-level, so the loss-policy fixtures
// need a store built from a file.
func newRouterWithSettings(t *testing.T, configJSON string) (*Router, *config.Store, *metrics.Store, *trace.Store) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(configJSON), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	store, err := config.NewStore(path)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	ms := metrics.New(100)
	ts := trace.New(100, domain.TraceSettings{})
	return New(store, ms, nil, nil, nil, ts, nil), store, ms, ts
}

// A backend that declares nothing is byte-for-byte passthrough: the client's path
// and body reach it untouched, and no conversion is reported.
func TestHandleNoProtocolsIsPassthrough(t *testing.T) {
	var gotPath, gotBody string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if answerProbe(w, r) {
			return
		}
		b, _ := io.ReadAll(r.Body)
		gotPath, gotBody = r.URL.Path, string(b)
		_, _ = w.Write([]byte(`{"id":"c1","model":"m","choices":[{"message":{"role":"assistant"}}]}`))
	}))
	defer backend.Close()

	r, store, ms, _ := newTestRouter(t)
	addTestRoute(t, store, "m", backend.URL)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", w.Result().StatusCode, w.Body.String())
	}
	if gotPath != "/v1/chat/completions" {
		t.Fatalf("backend path = %q, want the client's path", gotPath)
	}
	if h := w.Result().Header.Get("X-Router-Convert"); h != "" {
		t.Fatalf("X-Router-Convert = %q, want absent", h)
	}
	if !strings.Contains(gotBody, `"content":"hi"`) {
		t.Fatalf("body was rewritten: %s", gotBody)
	}
	rec := ms.Recent()
	if len(rec) != 1 || rec[0].Convert != "" || rec[0].UpstreamEndpoint != "/v1/chat/completions" {
		t.Fatalf("metric: %+v", rec)
	}
}

// A backend that speaks neither the client's endpoint nor a dialect the router can
// reach gets a 400 from us, naming what it serves — and the backend saw nothing.
// (Before the converter its 404 was forwarded as-is — 4xx is not an error — so
// nothing retried, nothing fell back, and the rule looked healthy.)
func TestHandleProtocolMismatchRefusesWithOur400(t *testing.T) {
	var hits int
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if answerProbe(w, r) {
			return
		}
		hits++
		http.Error(w, "not found", http.StatusNotFound)
	}))
	defer backend.Close()

	r, store, ms, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: backend.URL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/responses"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"}, ServerID: "s1",
		TargetModel: "claude", NumRetries: -1, Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if got := w.Result().StatusCode; got != http.StatusBadRequest {
		t.Fatalf("status %d, want 400: %s", got, w.Body.String())
	}
	if hits != 0 {
		t.Fatalf("backend got %d requests, want 0", hits)
	}
	if body := w.Body.String(); !strings.Contains(body, "/v1/responses") {
		t.Fatalf("refusal must name what the backend serves: %s", body)
	}
	rec := ms.Recent()
	if len(rec) != 1 || rec[0].StatusCode != http.StatusBadRequest {
		t.Fatalf("the refusal must be recorded: %+v", rec)
	}
}

// The hole the feature exists for: a rule whose primary cannot serve the client's
// dialect and whose fallback can. Before the converter the primary's 404 was
// forwarded as-is (4xx is not an error), the fallback was never tried, and the
// rule looked healthy.
func TestHandleProtocolMismatchConvertsFallback(t *testing.T) {
	var primaryHits int
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if answerProbe(w, r) {
			return
		}
		primaryHits++
		http.Error(w, "nope", http.StatusNotFound)
	}))
	defer primary.Close()
	var fbPath string
	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if answerProbe(w, r) {
			return
		}
		fbPath = r.URL.Path
		_, _ = w.Write([]byte(`{"id":"c","model":"claude","choices":[{"index":0,` +
			`"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer fallback.Close()

	r, store, ms, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "primary", Name: "primary", URL: primary.URL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/responses"}}}); err != nil {
		t.Fatalf("AddServer primary: %v", err)
	}
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: fallback.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI}}); err != nil {
		t.Fatalf("AddServer s1: %v", err)
	}
	// The fallback declares no protocols, so only its own `convert` can plan it
	// onto chat — which is what pins FallbackEntry.Convert being read.
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"}, ServerID: "primary",
		TargetModel: "claude", NumRetries: -1, Enabled: true,
		Fallbacks: []domain.FallbackEntry{{ServerID: "s1", Convert: "messages>chat"}}}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", w.Result().StatusCode, w.Body.String())
	}
	if primaryHits != 0 {
		t.Fatalf("the refusing primary got %d requests, want 0", primaryHits)
	}
	if fbPath != "/v1/chat/completions" {
		t.Fatalf("fallback path = %q", fbPath)
	}
	if h := w.Result().Header.Get("X-Router-Convert"); h != "messages>chat" {
		t.Fatalf("X-Router-Convert = %q", h)
	}
	rec := ms.Recent()
	if len(rec) != 1 || rec[0].ServerID != "s1" || rec[0].Convert != "messages>chat" {
		t.Fatalf("the metric must describe the serving attempt: %+v", rec)
	}
}

// A refusal is not a transport failure: no health mark (the server was never
// contacted), no rate-limit failure, no quota charge. A rule with one bad
// protocol declaration must not take a healthy backend out of rotation, and a
// request that was never sent must not be metered as if it had been.
func TestHandleRefusalLeavesHealthClean(t *testing.T) {
	var hits int
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if answerProbe(w, r) {
			return
		}
		hits++
	}))
	defer backend.Close()

	store, err := config.NewStore(filepath.Join(t.TempDir(), "config.json"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	health := config.NewHealthTracker(store, time.Hour, nil)
	rateLimit := config.NewRateLimiter(5, 60*time.Second, 5*time.Minute)
	quota := config.NewQuotaTracker()
	r := New(store, metrics.New(100), health, rateLimit, quota, trace.New(100, domain.TraceSettings{}), nil)
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: backend.URL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"r"}, ServerID: "s1",
		TargetModel: "r", NumRetries: -1, Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	health.Start() // marks the existing servers healthy; interval keeps it idle
	defer health.Stop()
	if !health.IsHealthy("s1") {
		t.Fatalf("fixture: s1 starts unhealthy")
	}
	// A responses request is no longer a refusal (it has its own spoke), so the
	// fixture refuses the family that stays refused: /v1/completions wants a
	// rendered prompt, and the chat template lives in the backend's tokenizer.
	req := httptest.NewRequest(http.MethodPost, "/v1/completions",
		strings.NewReader(`{"model":"r","prompt":"hi"}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusBadRequest || hits != 0 {
		t.Fatalf("status %d, hits %d: %s", w.Result().StatusCode, hits, w.Body.String())
	}
	if !health.IsHealthy("s1") {
		t.Fatal("a refusal must not mark a server unhealthy — it was never contacted")
	}
	if n := rateLimit.FailureCount("s1"); n != 0 {
		t.Fatalf("a refusal must not count as a rate-limit failure, got %d", n)
	}
	if skip := rateLimit.ShouldSkip("s1"); skip {
		t.Fatal("a refusal must not put a server into cooldown")
	}
	if tokens, requests := quota.Usage("s1"); tokens != 0 || requests != 0 {
		t.Fatalf("a refusal must not charge quota, got tokens=%d requests=%d", tokens, requests)
	}
}

// The mixed sequence: an earlier attempt reaches the proxy (its eager headers are
// on the writer's header map), the LAST attempt refuses. Nothing was sent for the
// refusing attempt, so its 400 must not carry the earlier attempt's conversion —
// the refusal tail reports no conversion, in the headers and in the metric alike,
// exactly like the single-attempt refusal.
func TestHandleRefusalTailReportsNoConversion(t *testing.T) {
	converted := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if answerProbe(w, req) {
			return
		}
	}))
	convertedURL := converted.URL
	converted.Close() // hijacked + closed: instant transport error after the eager headers
	refuser := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if answerProbe(w, req) {
			return
		}
	}))
	refuserURL := refuser.URL
	refuser.Close() // refused: never reached, so it must also see nothing

	r, store, ms, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "chatty", Name: "chatty", URL: convertedURL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer chatty: %v", err)
	}
	if err := store.AddServer(&domain.Server{ID: "responder", Name: "responder", URL: refuserURL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/responses"}}}); err != nil {
		t.Fatalf("AddServer responder: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"},
		ServerID: "chatty", TargetModel: "claude", NumRetries: -1, Enabled: true,
		Fallbacks: []domain.FallbackEntry{{ServerID: "responder"}}}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusBadRequest {
		t.Fatalf("status %d, want 400: %s", res.StatusCode, w.Body.String())
	}
	if got := res.Header.Get("X-Router-Convert"); got != "" {
		t.Fatalf("the refusal carries the failed attempt's conversion: X-Router-Convert=%q", got)
	}
	rec := ms.Recent()
	if len(rec) != 1 || rec[0].Convert != "" || rec[0].UpstreamEndpoint != "" {
		t.Fatalf("the refusal metric must describe no conversion: %+v", rec)
	}
}

// `convert: off` — from the rule and from the request header — sends the client's
// own request upstream, exactly as before this feature. The backend declares only
// chat, so without the off these two requests would be converted.
func TestHandleConvertOff(t *testing.T) {
	var hits int
	var gotPath, gotHeader string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if answerProbe(w, r) {
			return
		}
		hits++
		gotPath = r.URL.Path
		gotHeader = r.Header.Get("X-Router-Convert")
		_, _ = w.Write([]byte(`{"type":"message","role":"assistant","content":[],"model":"claude"}`))
	}))
	defer backend.Close()

	r, store, _, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: backend.URL,
		APITypes:  []domain.APIType{domain.APITypeAnthropic},
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"}, ServerID: "s1",
		TargetModel: "claude", NumRetries: -1, Enabled: true, Convert: "off"}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	send := func(header bool) *http.Response {
		req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
			`{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`))
		if header {
			req.Header.Set("X-Router-Convert", "off")
		}
		w := httptest.NewRecorder()
		r.Handle(w, req)
		return w.Result()
	}
	off := send(true)
	if h := off.Header.Get("X-Router-Convert"); h != "" {
		t.Fatalf("X-Router-Convert = %q, want absent", h)
	}
	send(false) // the rule's own `convert: off`, same decision
	if hits != 2 || gotPath != "/v1/messages" {
		t.Fatalf("off must send the client's own request upstream: hits=%d path=%q", hits, gotPath)
	}
	// The override is the client's instruction to the router; a backend has no
	// header by that name to honor, and the upstream request is a clone.
	if gotHeader != "" {
		t.Fatalf("the client's X-Router-Convert rode upstream: %q", gotHeader)
	}
}

// The plan describes the attempt that SERVES: an attempt whose backend can not
// speak the client's dialect is skipped (nothing sent), and the decision of the
// attempt that answered is what the client's headers and the metric report.
func TestHandleConvertReportsServingAttempt(t *testing.T) {
	var hits int
	var gotPath string
	chat := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if answerProbe(w, r) {
			return
		}
		hits++
		gotPath = r.URL.Path
		_, _ = w.Write([]byte(`{"id":"c1","model":"claude","choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer chat.Close()
	dead := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if answerProbe(w, r) {
			return
		}
	}))
	deadURL := dead.URL
	dead.Close() // hijacked + closed: instant transport error, no dial timeout

	r, store, ms, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "responder", Name: "responder", URL: deadURL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/responses"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddServer(&domain.Server{ID: "chatty", Name: "chatty", URL: chat.URL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"},
		ServerID: "responder", TargetModel: "claude", NumRetries: -1, Enabled: true,
		Fallbacks: []domain.FallbackEntry{{ServerID: "chatty"}}}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", res.StatusCode, w.Body.String())
	}
	if got := res.Header.Get("X-Router-Convert"); got != "messages>chat" {
		t.Fatalf("X-Router-Convert = %q, want messages>chat", got)
	}
	if got := res.Header.Get("X-Router-Server"); got != "chatty" {
		t.Fatalf("X-Router-Server = %q, want the serving attempt", got)
	}
	if hits != 1 || gotPath != "/v1/chat/completions" {
		t.Fatalf("the fallback must be reached at the planned path: hits=%d path=%q", hits, gotPath)
	}
	rec := ms.Recent()
	if len(rec) != 1 || rec[0].Convert != "messages>chat" || rec[0].UpstreamEndpoint != "/v1/chat/completions" {
		t.Fatalf("metric: %+v", rec)
	}
}

// The convert headers are rewritten, not just set: the header map persists across
// attempts, so a conversion reported by a failed attempt must not ride on a
// passthrough later one.
func TestHandleConvertHeadersDoNotLeakAcrossAttempts(t *testing.T) {
	converted := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if answerProbe(w, r) {
			return
		}
	}))
	convertedURL := converted.URL
	converted.Close() // refused after the eager headers were written
	plain := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if answerProbe(w, r) {
			return
		}
		_, _ = w.Write([]byte(`{"id":"c1","model":"claude","choices":[]}`))
	}))
	defer plain.Close()

	r, store, _, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "chatty", Name: "chatty", URL: convertedURL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddServer(&domain.Server{ID: "undeclared", Name: "undeclared", URL: plain.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"},
		ServerID: "chatty", TargetModel: "claude", NumRetries: -1, Enabled: true,
		Fallbacks: []domain.FallbackEntry{{ServerID: "undeclared"}}}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", res.StatusCode, w.Body.String())
	}
	if got := res.Header.Get("X-Router-Convert"); got != "" {
		t.Fatalf("X-Router-Convert = %q, want absent: the serving backend declares nothing", got)
	}
}

// stream_options is an OpenAI field: it must follow the dialect decision, not the
// client's path. An Anthropic request on an Anthropic backend must arrive with no
// stream_options in the body.
func TestHandleAnthropicStreamGetsNoStreamOptions(t *testing.T) {
	var gotBody string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if answerProbe(w, r) {
			return
		}
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		_, _ = w.Write([]byte(`{"id":"m1","type":"message","role":"assistant","content":[],"model":"claude"}`))
	}))
	defer backend.Close()

	r, store, _, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: backend.URL,
		APITypes:  []domain.APIType{domain.APITypeAnthropic},
		Protocols: []domain.ServerProtocol{{Path: "/v1/messages"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"}, ServerID: "s1",
		TargetModel: "claude", NumRetries: -1, Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","max_tokens":10,"stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	r.Handle(httptest.NewRecorder(), req)

	if strings.Contains(gotBody, "stream_options") {
		t.Fatalf("OpenAI stream_options leaked into an Anthropic request: %s", gotBody)
	}
}

// The mirror case: an OpenAI streaming request on a chat backend keeps the
// injection, now decided per attempt.
func TestHandleChatStreamStillInjectsUsage(t *testing.T) {
	var gotBody string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if answerProbe(w, r) {
			return
		}
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		_, _ = w.Write([]byte(`{"id":"c1","model":"m","choices":[]}`))
	}))
	defer backend.Close()

	r, store, _, _ := newTestRouter(t)
	addTestRoute(t, store, "m", backend.URL)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	r.Handle(httptest.NewRecorder(), req)

	if !strings.Contains(gotBody, `"include_usage":true`) {
		t.Fatalf("injection lost: %s", gotBody)
	}
}

// The first usable slice: an Anthropic client, a chat-only backend, a non-streamed
// answer. The request must reach the backend IN CHAT SHAPE (the plan moved the
// path; the transform carries the body) and the answer must come back in Anthropic
// shape, envelope, blocks and usage spelling included.
func TestHandleMessagesServedByChatBackend(t *testing.T) {
	var gotPath, gotBody string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if answerProbe(w, r) {
			return
		}
		b, _ := io.ReadAll(r.Body)
		gotPath, gotBody = r.URL.Path, string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"c1","model":"claude","choices":[{"index":0,` +
			`"message":{"role":"assistant","content":"hi there"},"finish_reason":"stop"}],` +
			`"usage":{"prompt_tokens":120,"completion_tokens":3,` +
			`"prompt_tokens_details":{"cached_tokens":100}}}`))
	}))
	defer backend.Close()

	r, store, ms, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: backend.URL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"}, ServerID: "s1",
		TargetModel: "claude", NumRetries: -1, Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","max_tokens":10,"system":"be brief",`+
			`"tools":[{"name":"fetch","input_schema":{"type":"object","properties":{"u":{"type":"number"}}}}],`+
			`"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", w.Result().StatusCode, w.Body.String())
	}
	if gotPath != "/v1/chat/completions" {
		t.Fatalf("backend path = %q", gotPath)
	}
	// The request side is the point of the spoke: an Anthropic body that merely
	// landed on a chat path is the failure the loss policy exists to prevent.
	if strings.Contains(gotBody, "input_schema") {
		t.Fatalf("the backend received an Anthropic body: %s", gotBody)
	}
	for _, want := range []string{`"type":"function"`, `"parameters"`, `"name":"fetch"`,
		`"role":"system"`, `"content":"be brief"`, `"content":"hi"`, `"max_tokens":10`} {
		if !strings.Contains(gotBody, want) {
			t.Fatalf("converted request lost %s: %s", want, gotBody)
		}
	}
	body := w.Body.String()
	if !strings.Contains(body, `"type":"message"`) || !strings.Contains(body, `"stop_reason":"end_turn"`) {
		t.Fatalf("client did not get an Anthropic message: %s", body)
	}
	// The usage convention is converted too: OpenAI's prompt_tokens (120) already
	// contains its cached tokens (100), Anthropic's input_tokens sits next to them.
	for _, want := range []string{`"type":"text"`, `"text":"hi there"`, `"input_tokens":20`,
		`"cache_read_input_tokens":100`, `"output_tokens":3`} {
		if !strings.Contains(body, want) {
			t.Fatalf("client answer lost %s: %s", want, body)
		}
	}
	if h := w.Result().Header.Get("X-Router-Convert"); h != "messages>chat" {
		t.Fatalf("X-Router-Convert = %q", h)
	}
	rec := ms.Recent()
	if len(rec) != 1 || rec[0].Convert != "messages>chat" || rec[0].UpstreamEndpoint != "/v1/chat/completions" {
		t.Fatalf("metric not annotated: %+v", rec)
	}
	// Usage is read from the native response: the translator runs after
	// metricsWriter, so extraction never sees translated bytes.
	if rec[0].PromptTokens != 120 || rec[0].CompletionTokens != 3 || rec[0].CachedTokens != 100 {
		t.Fatalf("usage lost: %+v", rec[0])
	}
}

// A request the spoke can not represent is a per-attempt refusal, exactly like a
// backend the plan can not reach: nothing is sent to that backend, the next
// attempt is tried, and when the last attempt refuses too the client gets OUR 400
// naming the field — never a backend 4xx for a body it never saw, and never a
// healthy-looking rule.
func TestHandleConvertRequestRefusal(t *testing.T) {
	// A hosted tool is always-400 class: the provider runs it, the model does not.
	const body = `{"model":"claude","max_tokens":10,` +
		`"tools":[{"name":"w","type":"web_search_20250305"}],` +
		`"messages":[{"role":"user","content":"hi"}]}`

	t.Run("falls through to the next attempt", func(t *testing.T) {
		var chatHits, plainHits int
		chat := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			if answerProbe(w, req) {
				return
			}
			chatHits++
		}))
		chatURL := chat.URL
		chat.Close() // hijacked + closed: the refusal must be decided before the dial
		plain := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			if answerProbe(w, req) {
				return
			}
			plainHits++
			_, _ = w.Write([]byte(`{"type":"message","role":"assistant","content":[],"model":"claude"}`))
		}))
		defer plain.Close()

		r, store, ms, _ := newTestRouter(t)
		if err := store.AddServer(&domain.Server{ID: "chatty", Name: "chatty", URL: chatURL,
			APITypes:  []domain.APIType{domain.APITypeOpenAI},
			Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
			t.Fatalf("AddServer: %v", err)
		}
		if err := store.AddServer(&domain.Server{ID: "undeclared", Name: "undeclared", URL: plain.URL,
			APITypes: []domain.APIType{domain.APITypeAnthropic}}); err != nil {
			t.Fatalf("AddServer: %v", err)
		}
		if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"},
			ServerID: "chatty", TargetModel: "claude", NumRetries: -1, Enabled: true,
			Fallbacks: []domain.FallbackEntry{{ServerID: "undeclared"}}}); err != nil {
			t.Fatalf("AddRule: %v", err)
		}
		req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
		w := httptest.NewRecorder()
		r.Handle(w, req)

		res := w.Result()
		if res.StatusCode != http.StatusOK {
			t.Fatalf("status %d: %s", res.StatusCode, w.Body.String())
		}
		if chatHits != 0 || plainHits != 1 {
			t.Fatalf("the refusing attempt must send nothing: chat=%d plain=%d", chatHits, plainHits)
		}
		if h := res.Header.Get("X-Router-Convert"); h != "" {
			t.Fatalf("the serving attempt is a passthrough, got X-Router-Convert=%q", h)
		}
		if rec := ms.Recent(); len(rec) != 1 || rec[0].Convert != "" {
			t.Fatalf("the metric describes the serving attempt: %+v", rec)
		}
	})

	t.Run("the last refusal is our 400", func(t *testing.T) {
		var hits int
		backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			if answerProbe(w, req) {
				return
			}
			hits++
		}))
		defer backend.Close()

		r, store, ms, _ := newTestRouter(t)
		if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: backend.URL,
			APITypes:  []domain.APIType{domain.APITypeOpenAI},
			Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
			t.Fatalf("AddServer: %v", err)
		}
		if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"}, ServerID: "s1",
			TargetModel: "claude", NumRetries: -1, Enabled: true}); err != nil {
			t.Fatalf("AddRule: %v", err)
		}
		req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
		w := httptest.NewRecorder()
		r.Handle(w, req)

		res := w.Result()
		if res.StatusCode != http.StatusBadRequest {
			t.Fatalf("status %d, want 400: %s", res.StatusCode, w.Body.String())
		}
		if hits != 0 {
			t.Fatalf("a refusal sends nothing upstream, got %d hits", hits)
		}
		if got := res.Header.Get("X-Router-Fallback-Errors"); got != "" {
			t.Fatalf("a single attempt has no fallback errors: %q", got)
		}
		if b := w.Body.String(); !strings.Contains(b, "tools") {
			t.Fatalf("the refusal must name the field: %s", b)
		}
		if rec := ms.Recent(); len(rec) != 1 || rec[0].StatusCode != http.StatusBadRequest {
			t.Fatalf("the refusal must be recorded: %+v", rec)
		}
	})
}

// A top-level field the spoke never decodes vanishes from the converted body, and
// a silent loss is what the notes channel exists to make visible (R16): the routed
// path compares what went in against what came out and reports the difference.
func TestHandleConvertReportsDroppedKeys(t *testing.T) {
	var gotBody string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if answerProbe(w, r) {
			return
		}
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		_, _ = w.Write([]byte(`{"id":"c1","model":"claude","choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer backend.Close()

	r, store, ms, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: backend.URL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"}, ServerID: "s1",
		TargetModel: "claude", NumRetries: -1, Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","max_tokens":10,"service_tier":"priority",`+
			`"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if strings.Contains(gotBody, "service_tier") {
		t.Fatalf("the spoke must not pass the field through: %s", gotBody)
	}
	notes := w.Result().Header.Get("X-Router-Convert-Notes")
	if !strings.Contains(notes, "keys:1dropped") {
		t.Fatalf("the loss of an undecoded field must be visible, notes=%q", notes)
	}
	rec := ms.Recent()
	if len(rec) != 1 || !strings.Contains(strings.Join(rec[0].ConvertNotes, ","), "keys:1dropped") {
		t.Fatalf("the metric must carry it too: %+v", rec)
	}
}

// The bulk note counts the keys NOTHING accounts for. A field the spoke renames
// (`stop_sequences`→`stop`), folds (`system` into `messages`), maps
// (`output_format`→`response_format`) or names when it drops (`top_k=dropped`) is
// accounted for: counting it too would report the plain system prompt of every
// Anthropic client as a loss, and would make the operator count a field twice.
// Only the undecoded key — `service_tier`, invisible to the spoke — is left.
func TestHandleConvertCountsOnlyUndecodedKeys(t *testing.T) {
	var gotBody string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if answerProbe(w, r) {
			return
		}
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		_, _ = w.Write([]byte(`{"id":"c1","model":"claude","choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer backend.Close()

	r, store, _, _ := newRouterWithSettings(t, `{"settings":{"convert":{"drop_params":true}}}`)
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: backend.URL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"}, ServerID: "s1",
		TargetModel: "claude", NumRetries: -1, Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","max_tokens":10,"system":"be brief","stop_sequences":["END"],`+
			`"top_k":40,"thinking":{"type":"enabled","budget_tokens":1024},`+
			`"output_format":{"type":"json_schema","name":"answer","schema":{"type":"object"}},`+
			`"service_tier":"priority","messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	for _, gone := range []string{"top_k", "thinking", "service_tier", "stop_sequences", "output_format"} {
		if strings.Contains(gotBody, `"`+gone+`"`) {
			t.Fatalf("%s must not reach the chat backend: %s", gone, gotBody)
		}
	}
	for _, kept := range []string{`"role":"system"`, `"stop":["END"]`, `"response_format"`, `"max_tokens":10`} {
		if !strings.Contains(gotBody, kept) {
			t.Fatalf("converted request lost %s: %s", kept, gotBody)
		}
	}
	notes := strings.Split(w.Result().Header.Get("X-Router-Convert-Notes"), ", ")
	for _, want := range []string{"top_k=dropped", "thinking=dropped", "keys:1dropped"} {
		if !slices.Contains(notes, want) {
			t.Fatalf("notes = %q, want %s", notes, want)
		}
	}
}

// The response direction can not refuse (the answer is already at the client), so
// notes are its only channel — and they must reach the recorded metric like the
// request side's already do (R24). A refused answer arrives as an `end_turn`
// completion, and the note is the only thing that says why it stopped; the parts
// and choices the mapping had nowhere to put stay visible the same way. Pinned on
// both response shapes, because JSON and SSE have separate producers.
func TestHandleConvertReportsResponseNotes(t *testing.T) {
	const jsonAnswer = `{"id":"c1","model":"claude","choices":[` +
		`{"index":0,"message":{"role":"assistant","content":[{"type":"text","text":"hi"},` +
		`{"type":"image_url","image_url":{"url":"data:image/png;base64,AAA"}}]},` +
		`"finish_reason":"refusal"},` +
		`{"index":1,"message":{"role":"assistant","content":"x"},"finish_reason":"stop"}]}`

	t.Run("json", func(t *testing.T) {
		backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			if answerProbe(w, req) {
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(jsonAnswer))
		}))
		defer backend.Close()
		r, store, ms, _ := newTestRouter(t)
		convertChatFixture(t, store, backend.URL)

		req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
			`{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`))
		w := httptest.NewRecorder()
		r.Handle(w, req)
		if !strings.Contains(w.Body.String(), `"stop_reason":"end_turn"`) {
			t.Fatalf("a refused answer must arrive as an end_turn completion: %s", w.Body.String())
		}
		rec := ms.Recent()
		if len(rec) != 1 {
			t.Fatalf("metrics: %+v", rec)
		}
		got := strings.Join(rec[0].ConvertNotes, ",")
		for _, want := range []string{"finish_reason=refusal", "choices=dropped:1",
			"content_part=dropped:image_url"} {
			if !strings.Contains(got, want) {
				t.Fatalf("the loss of %s must be on the metric, notes=%q", want, got)
			}
		}
	})

	t.Run("stream", func(t *testing.T) {
		backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			if answerProbe(w, req) {
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			f := w.(http.Flusher)
			for _, frame := range []string{
				`data: {"id":"c1","model":"claude","choices":[{"index":0,"delta":` +
					`{"content":[{"type":"text","text":"hi"}]}}]}` + "\n\n",
				`data: {"id":"c1","model":"claude","choices":[{"index":0,"delta":{"content":" there"},` +
					`"finish_reason":"content_filter"}]}` + "\n\n",
				"data: [DONE]\n\n",
			} {
				_, _ = w.Write([]byte(frame))
				f.Flush()
			}
		}))
		defer backend.Close()
		r, store, ms, _ := newTestRouter(t)
		convertChatFixture(t, store, backend.URL)

		req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
			`{"model":"claude","max_tokens":10,"stream":true,`+
				`"messages":[{"role":"user","content":"hi"}]}`))
		w := httptest.NewRecorder()
		r.Handle(w, req)
		out := w.Body.String()
		// A content-parts delta is translated, not forwarded native: the client reads
		// the answer, and the note is what says a part was left out.
		if !strings.Contains(out, `"text":"hi"`) || strings.Contains(out, `"content":[{`) {
			t.Fatalf("a content-parts delta did not reach the client translated:\n%s", out)
		}
		rec := ms.Recent()
		if len(rec) != 1 {
			t.Fatalf("metrics: %+v", rec)
		}
		if got := strings.Join(rec[0].ConvertNotes, ","); !strings.Contains(got, "finish_reason=content_filter") {
			t.Fatalf("the collapsed finish reason must be on the metric, notes=%q", got)
		}
	})
}

// convertChatFixture routes the incoming model `claude` to a chat-only declared
// server, i.e. a `messages → chat` conversion.
func convertChatFixture(t *testing.T, store *config.Store, url string) {
	t.Helper()
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: url,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"}, ServerID: "s1",
		TargetModel: "claude", NumRetries: -1, Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
}

// The metric's notes list is bounded (the spec's ≤ 512B): whole notes are kept,
// the tail dropped, and a list that fits is untouched.
func TestCapNotesBoundsTheTail(t *testing.T) {
	if got := proxy.CapConvertNotes(nil); len(got) != 0 {
		t.Fatalf("nil notes: %v", got)
	}
	short := []string{"thinking=off", "stop_dropped"}
	if got := proxy.CapConvertNotes(short); len(got) != 2 {
		t.Fatalf("a list that fits must be untouched: %v", got)
	}
	long := []string{}
	n := 0
	for i := 0; n <= proxy.MaxConvertNotesBytes; i++ {
		s := strings.Repeat("x", 100)
		long = append(long, s)
		n += len(s)
	}
	got := proxy.CapConvertNotes(long)
	total := 0
	for _, s := range got {
		total += len(s)
	}
	if len(got) != len(long)-1 || total > proxy.MaxConvertNotesBytes {
		t.Fatalf("CapConvertNotes kept %d of %d notes (%d bytes, limit %d)",
			len(got), len(long), total, proxy.MaxConvertNotesBytes)
	}
}

// CI's view of the responses spoke: the router plans a Responses request onto the
// chat endpoint the backend declares, reports the edge, and answers the client in
// its own dialect (sdktest pins the SDK's own reading of the same bytes).
func TestHandleResponsesServedByChatBackend(t *testing.T) {
	var gotPath, gotBody string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if answerProbe(w, r) {
			return
		}
		b, _ := io.ReadAll(r.Body)
		gotPath, gotBody = r.URL.Path, string(b)
		_, _ = w.Write([]byte(`{"id":"c1","model":"c","object":"chat.completion",` +
			`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},` +
			`"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}`))
	}))
	defer backend.Close()

	r, store, ms, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: backend.URL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"r"}, ServerID: "s1",
		TargetModel: "c", NumRetries: -1, Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(
		`{"model":"r","instructions":"be brief","input":"hi","max_output_tokens":64}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", res.StatusCode, w.Body.String())
	}
	if gotPath != "/v1/chat/completions" {
		t.Errorf("upstream path = %q, want /v1/chat/completions", gotPath)
	}
	for _, want := range []string{`"max_tokens":64`, `"role":"system"`, "be brief"} {
		if !strings.Contains(gotBody, want) {
			t.Errorf("the translated chat request lacks %s: %s", want, gotBody)
		}
	}
	if h := res.Header.Get("X-Router-Convert"); h != "responses>chat" {
		t.Errorf("X-Router-Convert = %q, want responses>chat", h)
	}
	body := w.Body.String()
	if !strings.Contains(body, `"object":"response"`) || !strings.Contains(body, `"output"`) {
		t.Fatalf("the client asked in the Responses dialect and must be answered in it: %s", body)
	}
	// The model rewrite runs above the translator, so the client's own name comes
	// back in the translated body.
	if !strings.Contains(body, `"model":"r"`) {
		t.Errorf("the response model was not rewritten back to the client's: %s", body)
	}
	rec := ms.Recent()
	if len(rec) != 1 || rec[0].Convert != "responses>chat" ||
		rec[0].UpstreamEndpoint != "/v1/chat/completions" {
		t.Fatalf("the metric must describe the serving attempt: %+v", rec)
	}
	if len(rec) == 1 && (rec[0].PromptTokens != 3 || rec[0].CompletionTokens != 2) {
		t.Fatalf("usage must be read off the backend's native bytes: %+v", rec[0])
	}
}
