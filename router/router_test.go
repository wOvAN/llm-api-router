package router

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"llm-api-router/config"
	"llm-api-router/domain"
	"llm-api-router/metrics"
)

func newTestRouter(t *testing.T) (*Router, *config.Store, *metrics.Store) {
	t.Helper()
	store, err := config.NewStore("/nonexistent/path.json")
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	ms := metrics.New(100)
	return New(store, ms, nil), store, ms
}

func TestAPITypeFromPath(t *testing.T) {
	tests := []struct {
		path string
		want domain.APIType
	}{
		{"/v1/chat/completions", domain.APITypeOpenAI},
		{"/v1/completions", domain.APITypeOpenAI},
		{"/v1/messages", domain.APITypeAnthropic},
		{"/v1/messages/stream", domain.APITypeAnthropic},
		{"/v1/embeddings", domain.APITypeOpenAI},
	}
	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			got := apiTypeFromPath(tt.path)
			if got != tt.want {
				t.Errorf("apiTypeFromPath(%q) = %q, want %q", tt.path, got, tt.want)
			}
		})
	}
}

func TestExtractModel(t *testing.T) {
	tests := []struct {
		name    string
		body    []byte
		want    string
		wantErr bool
	}{
		{
			name:    "valid model",
			body:    []byte(`{"model":"gpt-4","messages":[]}`),
			want:    "gpt-4",
			wantErr: false,
		},
		{
			name:    "model with whitespace",
			body:    []byte(`{"model":"  gpt-4  ","messages":[]}`),
			want:    "gpt-4",
			wantErr: false,
		},
		{
			name:    "missing model field",
			body:    []byte(`{"messages":[]}`),
			wantErr: true,
		},
		{
			name:    "model is not a string",
			body:    []byte(`{"model":123}`),
			wantErr: true,
		},
		{
			name:    "invalid JSON",
			body:    []byte(`not json`),
			wantErr: true,
		},
		{
			name:    "empty object",
			body:    []byte(`{}`),
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := extractModel(tt.body)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestListModels(t *testing.T) {
	t.Run("returns enabled models", func(t *testing.T) {
		r, store, _ := newTestRouter(t)
		_ = store.AddRule(&domain.RoutingRule{
			IncomingModels: []string{"gpt-4", "gpt-4-turbo"},
			ServerID:       "s1",
			Enabled:        true,
		})
		_ = store.AddRule(&domain.RoutingRule{
			IncomingModels: []string{"claude-3"},
			ServerID:       "s2",
			Enabled:        true,
		})
		_ = store.AddRule(&domain.RoutingRule{
			IncomingModels: []string{"disabled-model"},
			ServerID:       "s3",
			Enabled:        false,
		})

		req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
		w := httptest.NewRecorder()
		r.Handle(w, req)

		resp := w.Result()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("got status %d, want 200", resp.StatusCode)
		}

		var result struct {
			Object string                   `json:"object"`
			Data   []map[string]interface{} `json:"data"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			t.Fatalf("decode response: %v", err)
		}

		if result.Object != "list" {
			t.Errorf("object = %q, want %q", result.Object, "list")
		}
		if len(result.Data) != 3 {
			t.Fatalf("got %d models, want 3", len(result.Data))
		}
	})

	t.Run("no models returns empty list", func(t *testing.T) {
		r, _, _ := newTestRouter(t)
		req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
		w := httptest.NewRecorder()
		r.Handle(w, req)

		var result struct {
			Data []map[string]interface{} `json:"data"`
		}
		_ = json.NewDecoder(w.Result().Body).Decode(&result)
		if len(result.Data) != 0 {
			t.Errorf("got %d models, want 0", len(result.Data))
		}
	})
}

func TestHandleMethodNotAllowed(t *testing.T) {
	r, _, _ := newTestRouter(t)

	methods := []string{http.MethodPut, http.MethodDelete, http.MethodPatch, http.MethodOptions}
	for _, method := range methods {
		t.Run(method, func(t *testing.T) {
			req := httptest.NewRequest(method, "/v1/chat/completions", nil)
			w := httptest.NewRecorder()
			r.Handle(w, req)

			if w.Result().StatusCode != http.StatusMethodNotAllowed {
				t.Errorf("got status %d, want 405", w.Result().StatusCode)
			}
		})
	}
}

func TestHandleInvalidBody(t *testing.T) {
	r, _, _ := newTestRouter(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`not json`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusBadRequest {
		t.Errorf("got status %d, want 400", w.Result().StatusCode)
	}
}

func TestHandleMissingModel(t *testing.T) {
	r, _, _ := newTestRouter(t)
	body := strings.NewReader(`{"messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusBadRequest {
		t.Errorf("got status %d, want 400", w.Result().StatusCode)
	}
}

func TestHandleNoRoutingRule(t *testing.T) {
	r, _, _ := newTestRouter(t)
	body := strings.NewReader(`{"model":"unknown-model","messages":[]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusNotFound {
		t.Errorf("got status %d, want 404", w.Result().StatusCode)
	}
}

func TestHandleServerNotFound(t *testing.T) {
	r, store, _ := newTestRouter(t)
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		ServerID:       "nonexistent-server",
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"gpt-4","messages":[]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusInternalServerError {
		t.Errorf("got status %d, want 500", w.Result().StatusCode)
	}
}

func TestHandleAllBackendsFail(t *testing.T) {
	r, store, _ := newTestRouter(t)

	_ = store.AddServer(&domain.Server{
		ID:       "s1",
		Name:     "test-server",
		URL:      "http://localhost:1",
		APIKey:   "test-key",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "s1",
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusBadGateway {
		t.Errorf("got status %d, want 502", w.Result().StatusCode)
	}
}

func TestHandleAnthropicPath(t *testing.T) {
	r, store, _ := newTestRouter(t)

	_ = store.AddServer(&domain.Server{
		ID:       "s1",
		Name:     "test-server",
		URL:      "http://localhost:1",
		APIKey:   "test-key",
		APITypes: []domain.APIType{domain.APITypeAnthropic},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"claude-3"},
		TargetModel:    "claude-3-opus",
		ServerID:       "s1",
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"claude-3","max_tokens":100}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusBadGateway {
		t.Errorf("got status %d, want 502", w.Result().StatusCode)
	}
}

func TestMetricsAreRecorded(t *testing.T) {
	r, store, ms := newTestRouter(t)

	_ = store.AddServer(&domain.Server{
		ID:       "s1",
		Name:     "test-server",
		URL:      "http://localhost:1",
		APIKey:   "test-key",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "s1",
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	recent := ms.Recent()
	if len(recent) != 1 {
		t.Fatalf("expected 1 metric, got %d", len(recent))
	}
	if recent[0].Model != "gpt-4" {
		t.Errorf("metric model = %q, want %q", recent[0].Model, "gpt-4")
	}
	if recent[0].StatusCode != http.StatusBadGateway {
		t.Errorf("metric status = %d, want %d", recent[0].StatusCode, http.StatusBadGateway)
	}
}

func TestFallbackPreservesActualModel(t *testing.T) {
	// When fallback occurs, the response should contain the actual model
	// used by the fallback server, not the original client model.
	// Primary server fails, fallback succeeds.

	// Create fallback server that returns a known model name
	fallbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"fallback-model","choices":[{"finish_reason":"stop"}]}`)) //nolint:errcheck
	}))
	defer fallbackServer.Close()

	r, store, _ := newTestRouter(t)

	_ = store.AddServer(&domain.Server{
		ID:       "primary",
		Name:     "Primary",
		URL:      "http://localhost:1", // unreachable
		APIKey:   "test-key",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddServer(&domain.Server{
		ID:       "fallback",
		Name:     "Fallback",
		URL:      fallbackServer.URL,
		APIKey:   "test-key",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"opus"},
		TargetModel:    "gpt-4",
		ServerID:       "primary",
		Fallbacks:      []domain.FallbackEntry{{ServerID: "fallback", TargetModel: "fallback-model"}},
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"opus","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("got status %d, want 200", w.Result().StatusCode)
	}

	respBody, _ := io.ReadAll(w.Result().Body)
	var resp map[string]interface{}
	_ = json.Unmarshal(respBody, &resp)

	// Response should contain the actual fallback model, not "opus"
	model, ok := resp["model"].(string)
	if !ok {
		t.Fatalf("no 'model' field in response: %s", respBody)
	}
	if model != "fallback-model" {
		t.Errorf("response model = %q, want %q (actual fallback model)", model, "fallback-model")
	}
}

func TestPrimaryAttemptRewritesModel(t *testing.T) {
	// When primary succeeds, response should contain the original client model
	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`)) //nolint:errcheck
	}))
	defer primaryServer.Close()

	r, store, _ := newTestRouter(t)

	_ = store.AddServer(&domain.Server{
		ID:       "primary",
		Name:     "Primary",
		URL:      primaryServer.URL,
		APIKey:   "test-key",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"opus"},
		TargetModel:    "gpt-4",
		ServerID:       "primary",
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"opus","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("got status %d, want 200", w.Result().StatusCode)
	}

	respBody, _ := io.ReadAll(w.Result().Body)
	var resp map[string]interface{}
	_ = json.Unmarshal(respBody, &resp)

	model, ok := resp["model"].(string)
	if !ok {
		t.Fatalf("no 'model' field in response: %s", respBody)
	}
	// Primary attempt: response should contain original client model
	if model != "opus" {
		t.Errorf("response model = %q, want %q (original client model)", model, "opus")
	}
}

func TestNoRewriteWhenModelsMatch(t *testing.T) {
	// When targetModel == originalModel, no rewriting occurs
	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`)) //nolint:errcheck
	}))
	defer primaryServer.Close()

	r, store, _ := newTestRouter(t)

	_ = store.AddServer(&domain.Server{
		ID:       "primary",
		Name:     "Primary",
		URL:      primaryServer.URL,
		APIKey:   "test-key",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4", // Same as incoming
		ServerID:       "primary",
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("got status %d, want 200", w.Result().StatusCode)
	}

	respBody, _ := io.ReadAll(w.Result().Body)
	var resp map[string]interface{}
	_ = json.Unmarshal(respBody, &resp)

	model, ok := resp["model"].(string)
	if !ok {
		t.Fatalf("no 'model' field in response: %s", respBody)
	}
	// Models match, so response should contain gpt-4
	if model != "gpt-4" {
		t.Errorf("response model = %q, want %q", model, "gpt-4")
	}
}

func TestFallbackLogsActualModel(t *testing.T) {
	// Verify that when fallback occurs, the log message includes the actual model
	// This is a structural test to ensure wasFallback is set correctly in metrics
	r, store, ms := newTestRouter(t)

	// Create fallback server that succeeds
	fallbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"haiku","choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":10}}`)) //nolint:errcheck
	}))
	defer fallbackServer.Close()

	_ = store.AddServer(&domain.Server{
		ID:       "primary",
		Name:     "Primary",
		URL:      "http://localhost:1", // unreachable
		APIKey:   "test-key",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddServer(&domain.Server{
		ID:       "fallback",
		Name:     "Fallback",
		URL:      fallbackServer.URL,
		APIKey:   "test-key",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"opus"},
		TargetModel:    "gpt-4",
		ServerID:       "primary",
		Fallbacks:      []domain.FallbackEntry{{ServerID: "fallback", TargetModel: "haiku"}},
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"opus","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("got status %d, want 200", w.Result().StatusCode)
	}

	recent := ms.Recent()
	if len(recent) != 1 {
		t.Fatalf("expected 1 metric, got %d", len(recent))
	}
	m := recent[0]
	if !m.WasFallback {
		t.Error("WasFallback should be true")
	}
	if m.TargetModel != "haiku" {
		t.Errorf("TargetModel = %q, want %q", m.TargetModel, "haiku")
	}
	if m.ServerID != "fallback" {
		t.Errorf("ServerID = %q, want %q", m.ServerID, "fallback")
	}
	// Model in metric should be the original client model
	if m.Model != "opus" {
		t.Errorf("Model = %q, want %q", m.Model, "opus")
	}
	_ = fmt.Sprintf // suppress unused import
}
