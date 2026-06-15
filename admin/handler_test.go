package admin

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"llm-api-router/config"
	"llm-api-router/domain"
	"llm-api-router/metrics"
)

func newTestHandler(t *testing.T) (*Handler, *config.Store) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	store, err := config.NewStore(path)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	ms := metrics.New(100)
	return NewHandler(store, ms), store
}

func TestListServersEmpty(t *testing.T) {
	h, _ := newTestHandler(t)
	req := httptest.NewRequest(http.MethodGet, "/admin/api/servers", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("got status %d, want 200", w.Code)
	}

	var servers []domain.Server
	if err := json.NewDecoder(w.Body).Decode(&servers); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(servers) != 0 {
		t.Errorf("got %d servers, want 0", len(servers))
	}
}

func TestAddServer(t *testing.T) {
	h, _ := newTestHandler(t)
	body := strings.NewReader(`{"id":"s1","name":"Test","url":"http://localhost:8080","api_types":["openai"]}`)
	req := httptest.NewRequest(http.MethodPost, "/admin/api/servers", body)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusCreated {
		t.Errorf("got status %d, want 201", w.Code)
	}

	var srv domain.Server
	if err := json.NewDecoder(w.Body).Decode(&srv); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if srv.ID != "s1" {
		t.Errorf("got id %q, want %q", srv.ID, "s1")
	}
}

func TestAddServerDuplicate(t *testing.T) {
	h, store := newTestHandler(t)
	_ = store.AddServer(&domain.Server{ID: "s1", URL: "http://localhost"})

	body := strings.NewReader(`{"id":"s1","url":"http://other"}`)
	req := httptest.NewRequest(http.MethodPost, "/admin/api/servers", body)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusConflict {
		t.Errorf("got status %d, want 409", w.Code)
	}
}

func TestAddServerMissingID(t *testing.T) {
	h, _ := newTestHandler(t)
	body := strings.NewReader(`{"url":"http://localhost"}`)
	req := httptest.NewRequest(http.MethodPost, "/admin/api/servers", body)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("got status %d, want 400", w.Code)
	}
}

func TestAddServerInvalidJSON(t *testing.T) {
	h, _ := newTestHandler(t)
	body := strings.NewReader(`not json`)
	req := httptest.NewRequest(http.MethodPost, "/admin/api/servers", body)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("got status %d, want 400", w.Code)
	}
}

func TestUpdateServer(t *testing.T) {
	h, store := newTestHandler(t)
	_ = store.AddServer(&domain.Server{ID: "s1", Name: "Old", URL: "http://old.com"})

	body := strings.NewReader(`{"name":"New","url":"http://new.com","api_types":["openai"]}`)
	req := httptest.NewRequest(http.MethodPut, "/admin/api/servers/s1", body)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("got status %d, want 200", w.Code)
	}

	srv, _ := store.GetServer("s1")
	if srv.Name != "New" {
		t.Errorf("got name %q, want %q", srv.Name, "New")
	}
	if srv.URL != "http://new.com" {
		t.Errorf("got url %q, want %q", srv.URL, "http://new.com")
	}
}

func TestUpdateServerNotFound(t *testing.T) {
	h, _ := newTestHandler(t)
	body := strings.NewReader(`{"name":"New","url":"http://new.com"}`)
	req := httptest.NewRequest(http.MethodPut, "/admin/api/servers/nonexistent", body)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("got status %d, want 404", w.Code)
	}
}

func TestDeleteServer(t *testing.T) {
	h, store := newTestHandler(t)
	_ = store.AddServer(&domain.Server{ID: "s1", URL: "http://test.com"})

	req := httptest.NewRequest(http.MethodDelete, "/admin/api/servers/s1", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("got status %d, want 200", w.Code)
	}

	if _, ok := store.GetServer("s1"); ok {
		t.Error("server should be deleted")
	}
}

func TestDeleteServerNotFound(t *testing.T) {
	h, _ := newTestHandler(t)
	req := httptest.NewRequest(http.MethodDelete, "/admin/api/servers/nonexistent", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("got status %d, want 404", w.Code)
	}
}

func TestListRulesEmpty(t *testing.T) {
	h, _ := newTestHandler(t)
	req := httptest.NewRequest(http.MethodGet, "/admin/api/rules", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("got status %d, want 200", w.Code)
	}

	var rules []domain.RoutingRule
	if err := json.NewDecoder(w.Body).Decode(&rules); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(rules) != 0 {
		t.Errorf("got %d rules, want 0", len(rules))
	}
}

func TestAddRule(t *testing.T) {
	h, store := newTestHandler(t)
	_ = store.AddServer(&domain.Server{ID: "s1", URL: "http://localhost"})

	body := strings.NewReader(`{"incoming_models":["gpt-4"],"target_model":"gpt-4","server_id":"s1","enabled":true}`)
	req := httptest.NewRequest(http.MethodPost, "/admin/api/rules", body)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusCreated {
		t.Errorf("got status %d, want 201", w.Code)
	}

	cfg := store.GetConfig()
	if len(cfg.Rules) != 1 {
		t.Errorf("got %d rules, want 1", len(cfg.Rules))
	}
}

func TestAddRuleInvalidJSON(t *testing.T) {
	h, _ := newTestHandler(t)
	body := strings.NewReader(`not json`)
	req := httptest.NewRequest(http.MethodPost, "/admin/api/rules", body)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("got status %d, want 400", w.Code)
	}
}

func TestUpdateRule(t *testing.T) {
	h, store := newTestHandler(t)
	_ = store.AddServer(&domain.Server{ID: "s1", URL: "http://localhost"})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"old-model"},
		TargetModel:    "old-model",
		ServerID:       "s1",
		Enabled:        true,
	})

	body := strings.NewReader(`{"incoming_models":["new-model"],"target_model":"new-model","server_id":"s1","enabled":true}`)
	req := httptest.NewRequest(http.MethodPut, "/admin/api/rules/0", body)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("got status %d, want 200", w.Code)
	}

	rule, ok := store.GetRuleByModel("new-model")
	if !ok {
		t.Fatal("updated rule not found")
	}
	if !rule.Enabled {
		t.Error("rule should be enabled")
	}
}

func TestUpdateRuleInvalidIndex(t *testing.T) {
	h, _ := newTestHandler(t)
	body := strings.NewReader(`{"incoming_models":["m1"],"server_id":"s1"}`)
	req := httptest.NewRequest(http.MethodPut, "/admin/api/rules/abc", body)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("got status %d, want 400", w.Code)
	}
}

func TestUpdateRuleOutOfRange(t *testing.T) {
	h, _ := newTestHandler(t)
	body := strings.NewReader(`{"incoming_models":["m1"],"server_id":"s1"}`)
	req := httptest.NewRequest(http.MethodPut, "/admin/api/rules/99", body)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("got status %d, want 404", w.Code)
	}
}

func TestDeleteRule(t *testing.T) {
	h, store := newTestHandler(t)
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"m1"},
		ServerID:       "s1",
		Enabled:        true,
	})

	req := httptest.NewRequest(http.MethodDelete, "/admin/api/rules/0", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("got status %d, want 200", w.Code)
	}

	if len(store.GetConfig().Rules) != 0 {
		t.Error("rule should be deleted")
	}
}

func TestDeleteRuleOutOfRange(t *testing.T) {
	h, _ := newTestHandler(t)
	req := httptest.NewRequest(http.MethodDelete, "/admin/api/rules/99", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("got status %d, want 404", w.Code)
	}
}

func TestGetConfig(t *testing.T) {
	h, store := newTestHandler(t)
	_ = store.AddServer(&domain.Server{ID: "s1", URL: "http://test.com", APITypes: []domain.APIType{domain.APITypeOpenAI}})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "s1",
		Enabled:        true,
	})

	req := httptest.NewRequest(http.MethodGet, "/admin/api/config", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("got status %d, want 200", w.Code)
	}

	var cfg domain.Config
	if err := json.NewDecoder(w.Body).Decode(&cfg); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(cfg.Servers) != 1 {
		t.Errorf("got %d servers, want 1", len(cfg.Servers))
	}
	if len(cfg.Rules) != 1 {
		t.Errorf("got %d rules, want 1", len(cfg.Rules))
	}
}

func TestReloadConfig(t *testing.T) {
	h, _ := newTestHandler(t)
	req := httptest.NewRequest(http.MethodPost, "/admin/api/config/reload", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("got status %d, want 200", w.Code)
	}

	var result map[string]string
	if err := json.NewDecoder(w.Body).Decode(&result); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if result["status"] != "reloaded" {
		t.Errorf("got status %q, want %q", result["status"], "reloaded")
	}
}

func TestGetMetrics(t *testing.T) {
	h, _ := newTestHandler(t)
	req := httptest.NewRequest(http.MethodGet, "/admin/api/metrics", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("got status %d, want 200", w.Code)
	}
}

func TestGetRecentRequests(t *testing.T) {
	h, _ := newTestHandler(t)
	req := httptest.NewRequest(http.MethodGet, "/admin/api/metrics/recent", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("got status %d, want 200", w.Code)
	}
}

func TestResetMetrics(t *testing.T) {
	h, _ := newTestHandler(t)
	req := httptest.NewRequest(http.MethodPost, "/admin/api/metrics/reset", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("got status %d, want 200", w.Code)
	}

	var result map[string]string
	if err := json.NewDecoder(w.Body).Decode(&result); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if result["status"] != "reset" {
		t.Errorf("got status %q, want %q", result["status"], "reset")
	}
}

func TestNotFound(t *testing.T) {
	h, _ := newTestHandler(t)
	req := httptest.NewRequest(http.MethodGet, "/admin/api/nonexistent", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("got status %d, want 404", w.Code)
	}
}

func TestMethodNotAllowed(t *testing.T) {
	h, _ := newTestHandler(t)
	req := httptest.NewRequest(http.MethodPost, "/admin/api/servers/s1", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("got status %d, want 404", w.Code)
	}
}

func TestGetServerModelsNoServer(t *testing.T) {
	h, _ := newTestHandler(t)
	req := httptest.NewRequest(http.MethodGet, "/admin/api/servers/nonexistent/models", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("got status %d, want 404", w.Code)
	}
}

func TestTestServerInvalidJSON(t *testing.T) {
	h, _ := newTestHandler(t)
	body := strings.NewReader(`not json`)
	req := httptest.NewRequest(http.MethodPost, "/admin/api/servers/test", body)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("got status %d, want 400", w.Code)
	}
}

func TestGetServerModelsInvalidURL(t *testing.T) {
	h, store := newTestHandler(t)
	_ = store.AddServer(&domain.Server{
		ID:   "s1",
		URL:  string([]byte{0x7f}),
		Name: "Bad",
	})

	req := httptest.NewRequest(http.MethodGet, "/admin/api/servers/s1/models", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("got status %d, want 400", w.Code)
	}
}
