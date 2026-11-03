package router

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"llm-api-router/domain"
)

// A request whose estimated prompt is bigger than the window a backend declares
// can not be served by it — the engine will answer 400 after eating the whole
// prompt. Declaring the window is the operator's word, so the router moves the
// request to the attempt that declares room for it (the last attempt is always
// tried anyway). A rule that declares no window gates nothing.
func TestHandleSkipsAttemptTooSmallForContext(t *testing.T) {
	var small, big atomic.Int32
	smallSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		small.Add(1)
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"the request context size exceeds the model's context size"}}`))
	}))
	defer smallSrv.Close()
	bigSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		big.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"long-ctx","choices":[{"finish_reason":"stop"}]}`))
	}))
	defer bigSrv.Close()

	r, store, ms, _ := newTestRouter(t)
	for _, s := range []*domain.Server{
		{ID: "small", Name: "Small", URL: smallSrv.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
		{ID: "big", Name: "Big", URL: bigSrv.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
	} {
		if err := store.AddServer(s); err != nil {
			t.Fatalf("AddServer: %v", err)
		}
	}
	if err := store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"chat"},
		TargetModel:    "chat",
		ServerID:       "small",
		ContextWindow:  1000,
		Fallbacks: []domain.FallbackEntry{
			{ServerID: "big", ContextWindow: 200000},
		},
		Enabled: true,
	}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}

	prompt := strings.Repeat("word ", 4000) // ~5k chars ≈ 1.25k tokens > 1000
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"chat","messages":[{"role":"user","content":"`+prompt+`"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if got := w.Result().StatusCode; got != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", got, w.Body.String())
	}
	if got := small.Load(); got != 0 {
		t.Errorf("the 1000-token backend was hit %d times, want 0 (the prompt is bigger)", got)
	}
	if got := big.Load(); got != 1 {
		t.Errorf("the 200k backend hit %d times, want 1", got)
	}
	if rec := ms.Recent(); len(rec) != 1 || !rec[0].WasFallback {
		t.Errorf("recent = %+v, want one fallback request", rec)
	}
}

// The other side of the gate: a request that fits is untouched, and a rule with
// no declared window routes exactly as before (an absent declaration is not a
// declaration of zero).
func TestHandleContextWindowGateStaysInert(t *testing.T) {
	var hits, fallbacks atomic.Int32
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`))
	}))
	defer primary.Close()
	backup := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fallbacks.Add(1)
	}))
	defer backup.Close()

	r, store, _, _ := newTestRouter(t)
	for _, s := range []*domain.Server{
		{ID: "s1", Name: "S1", URL: primary.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
		{ID: "s2", Name: "S2", URL: backup.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
	} {
		if err := store.AddServer(s); err != nil {
			t.Fatalf("AddServer: %v", err)
		}
	}
	if err := store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "s1",
		ContextWindow:  8000,
		Fallbacks:      []domain.FallbackEntry{{ServerID: "s2"}},
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
	if hits.Load() != 1 || fallbacks.Load() != 0 {
		t.Errorf("hits = %d, fallbacks = %d, want 1 and 0", hits.Load(), fallbacks.Load())
	}
}
