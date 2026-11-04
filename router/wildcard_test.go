package router

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"llm-api-router/config"
	"llm-api-router/domain"
)

// A wildcard rule owns a model family, so it has no single target model: the
// client's own name is what the backend must receive. Rewriting it to the
// (empty) target would blank the field and every engine rejects that.
func TestHandleWildcardRuleForwardsClientModel(t *testing.T) {
	var received string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := make([]byte, 4096)
		n, _ := r.Body.Read(body)
		received = string(body[:n])
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"qwen3-32b","choices":[{"finish_reason":"stop"}]}`))
	}))
	defer backend.Close()

	r, store, _, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: backend.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"qwen3-*"},
		ServerID:       "s1",
		Enabled:        true,
	}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"qwen3-32b","messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", res.StatusCode, w.Body.String())
	}
	if !strings.Contains(received, `"model":"qwen3-32b"`) {
		t.Errorf("the client's model name never reached the backend: %s", received)
	}
	if body := w.Body.String(); !strings.Contains(body, "qwen3-32b") {
		t.Errorf("response lost the model name: %s", body)
	}
}

// /v1/models is the list of names a client may send, so a pattern is not in it
// (it is a rule, not a name) — and each listed name carries the health of the
// servers that can serve it, so a client with a model picker can see a model is
// down without sending a request to find out.
func TestListModelsSkipsPatternsAndReportsHealth(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {}))
	defer backend.Close()

	r, store, _, _ := newTestRouter(t)
	health := config.NewHealthTracker(store, time.Hour, nil)
	r = New(store, r.metrics, health, nil, nil, nil, nil)

	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: backend.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddServer(&domain.Server{ID: "s2", Name: "s2", URL: backend.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddServer(&domain.Server{ID: "s3", Name: "s3", URL: backend.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"qwen3-*", "qwen3-flagship"},
		TargetModel:    "qwen3-32b",
		ServerID:       "s1",
		Enabled:        true,
	}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"tiny"},
		TargetModel:    "tiny",
		ServerID:       "s2",
		Fallbacks:      []domain.FallbackEntry{{ServerID: "s3"}},
		Enabled:        true,
	}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	health.MarkHealthy("s1")

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	var payload struct {
		Data []struct {
			ID      string `json:"id"`
			Healthy bool   `json:"healthy"`
		} `json:"data"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode /v1/models: %v (%s)", err, w.Body.String())
	}
	got := map[string]bool{}
	for _, m := range payload.Data {
		got[m.ID] = m.Healthy
	}
	if _, ok := got["qwen3-*"]; ok {
		t.Errorf("a wildcard pattern was listed as a model: %v", w.Body.String())
	}
	if !got["qwen3-flagship"] {
		t.Errorf("qwen3-flagship: healthy = false, want true (s1 is healthy): %v", got)
	}
	if got["tiny"] {
		t.Errorf("tiny: healthy = true, want false (both of its servers are down)")
	}
}

// A 404 that names the models this router accepts is a 404 the caller can act
// on; one that only quotes its own typo is a 404 it has to go and read the
// config for.
func TestHandleNoRoutingRuleNamesRoutedModels(t *testing.T) {
	r, store, _, _ := newTestRouter(t)
	if err := store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4", "sonnet", "gpt-*"},
		ServerID:       "s1",
		Enabled:        true,
	}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"claude-3","messages":[]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", res.StatusCode)
	}
	body := w.Body.String()
	if !strings.Contains(body, "gpt-4") || !strings.Contains(body, "sonnet") {
		t.Fatalf("the 404 names nothing the caller can send: %s", body)
	}
	if strings.Contains(body, "gpt-*") {
		t.Fatalf("the 404 lists a pattern as a sendable name: %s", body)
	}
}
