package router

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"llm-api-router/domain"
)

// A rule with `balance: "round-robin"` is a pool of interchangeable backends
// (two GPUs, two replicas of one model): successive requests start on a
// different member instead of hammering the primary until it fails. The rest of
// the list stays the failover chain, so a pool member that is down costs one
// rotation, not a request.
func TestHandleRoundRobinBalancesAcrossReplicas(t *testing.T) {
	var a, b atomic.Int32
	handler := func(counter *atomic.Int32) http.HandlerFunc {
		return func(w http.ResponseWriter, _ *http.Request) {
			counter.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`))
		}
	}
	first := httptest.NewServer(handler(&a))
	defer first.Close()
	second := httptest.NewServer(handler(&b))
	defer second.Close()

	r, store, ms, _ := newTestRouter(t)
	for _, s := range []*domain.Server{
		{ID: "a", Name: "A", URL: first.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
		{ID: "b", Name: "B", URL: second.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
	} {
		if err := store.AddServer(s); err != nil {
			t.Fatalf("AddServer: %v", err)
		}
	}
	if err := store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "a",
		Fallbacks:      []domain.FallbackEntry{{ServerID: "b"}},
		Balance:        domain.BalanceRoundRobin,
		Enabled:        true,
	}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}

	for range 4 {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
			strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`))
		w := httptest.NewRecorder()
		r.Handle(w, req)
		if got := w.Result().StatusCode; got != http.StatusOK {
			t.Fatalf("status = %d, want 200: %s", got, w.Body.String())
		}
	}
	if a.Load() != 2 || b.Load() != 2 {
		t.Errorf("hits = a %d, b %d, want 2 and 2", a.Load(), b.Load())
	}
	// A rotation is not a failover: the request the pool served on its own
	// starting slot must not count as a fallback in the metrics.
	for _, m := range ms.Recent() {
		if m.WasFallback {
			t.Errorf("request %s recorded as a fallback, want a served pool slot", m.ServerID)
		}
	}
}

// Weights are the point of balancing a pool of unequal replicas (a 4-GPU box
// beside a 1-GPU one): the start slot is dealt out in proportion, so traffic
// lands 1:3 without an operator counting anything.
func TestHandleRoundRobinHonoursWeights(t *testing.T) {
	var a, b atomic.Int32
	handler := func(counter *atomic.Int32) http.HandlerFunc {
		return func(w http.ResponseWriter, _ *http.Request) {
			counter.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`))
		}
	}
	first := httptest.NewServer(handler(&a))
	defer first.Close()
	second := httptest.NewServer(handler(&b))
	defer second.Close()

	r, store, _, _ := newTestRouter(t)
	for _, s := range []*domain.Server{
		{ID: "a", Name: "A", URL: first.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
		{ID: "b", Name: "B", URL: second.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
	} {
		if err := store.AddServer(s); err != nil {
			t.Fatalf("AddServer: %v", err)
		}
	}
	if err := store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "a",
		Fallbacks:      []domain.FallbackEntry{{ServerID: "b", Weight: 3}},
		Balance:        domain.BalanceRoundRobin,
		Enabled:        true,
	}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}

	for range 4 {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
			strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`))
		w := httptest.NewRecorder()
		r.Handle(w, req)
	}
	if a.Load() != 1 || b.Load() != 3 {
		t.Errorf("hits = a %d, b %d, want 1 and 3 (weights 1:3)", a.Load(), b.Load())
	}
}

// A rule without `balance` keeps the historical contract exactly: the primary
// takes every request it can serve and a fallback is a rescue, not a share.
func TestHandleBalanceOffKeepsFailoverOrder(t *testing.T) {
	var a, b atomic.Int32
	handler := func(counter *atomic.Int32) http.HandlerFunc {
		return func(w http.ResponseWriter, _ *http.Request) {
			counter.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`))
		}
	}
	first := httptest.NewServer(handler(&a))
	defer first.Close()
	second := httptest.NewServer(handler(&b))
	defer second.Close()

	r, store, _, _ := newTestRouter(t)
	for _, s := range []*domain.Server{
		{ID: "a", Name: "A", URL: first.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
		{ID: "b", Name: "B", URL: second.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
	} {
		if err := store.AddServer(s); err != nil {
			t.Fatalf("AddServer: %v", err)
		}
	}
	if err := store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "a",
		Fallbacks:      []domain.FallbackEntry{{ServerID: "b"}},
		Enabled:        true,
	}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}

	for range 4 {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
			strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`))
		w := httptest.NewRecorder()
		r.Handle(w, req)
	}
	if a.Load() != 4 || b.Load() != 0 {
		t.Errorf("hits = a %d, b %d, want 4 and 0 (no balancing requested)", a.Load(), b.Load())
	}
}
