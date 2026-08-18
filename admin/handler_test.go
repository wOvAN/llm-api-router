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
	return NewHandler(store, ms, nil), store
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
	if len(cfg.Profiles[0].Rules) != 1 {
		t.Errorf("got %d rules, want 1", len(cfg.Profiles[0].Rules))
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

	if len(store.GetConfig().Profiles[0].Rules) != 0 {
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

func TestListProfiles(t *testing.T) {
	h, store := newTestHandler(t)
	if _, err := store.AddProfile("prod", false); err != nil {
		t.Fatalf("AddProfile: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/admin/api/profiles", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("got status %d, want 200", w.Code)
	}
	var resp struct {
		Profiles        []domain.RuleProfile `json:"profiles"`
		ActiveProfileID string               `json:"active_profile_id"`
	}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(resp.Profiles) != 2 {
		t.Errorf("got %d profiles, want 2", len(resp.Profiles))
	}
	if resp.ActiveProfileID != "default" {
		t.Errorf("active_profile_id = %q, want default", resp.ActiveProfileID)
	}
}

func TestAddProfile(t *testing.T) {
	h, store := newTestHandler(t)
	req := httptest.NewRequest(http.MethodPost, "/admin/api/profiles", strings.NewReader(`{"name":"prod","copy_from_active":false}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusCreated {
		t.Fatalf("got status %d, want 201", w.Code)
	}
	if len(store.GetProfiles()) != 2 {
		t.Errorf("got %d profiles, want 2", len(store.GetProfiles()))
	}

	req2 := httptest.NewRequest(http.MethodPost, "/admin/api/profiles", strings.NewReader(`{"name":"prod"}`))
	req2.Header.Set("Content-Type", "application/json")
	w2 := httptest.NewRecorder()
	h.ServeHTTP(w2, req2)
	if w2.Code != http.StatusBadRequest {
		t.Errorf("duplicate profile got status %d, want 400", w2.Code)
	}
}

func TestRenameProfile(t *testing.T) {
	h, store := newTestHandler(t)
	if _, err := store.AddProfile("prod", false); err != nil {
		t.Fatalf("AddProfile: %v", err)
	}

	req := httptest.NewRequest(http.MethodPut, "/admin/api/profiles/prod", strings.NewReader(`{"name":"production"}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("got status %d, want 200", w.Code)
	}

	req2 := httptest.NewRequest(http.MethodPut, "/admin/api/profiles/nope", strings.NewReader(`{"name":"x"}`))
	req2.Header.Set("Content-Type", "application/json")
	w2 := httptest.NewRecorder()
	h.ServeHTTP(w2, req2)
	if w2.Code != http.StatusNotFound {
		t.Errorf("unknown profile got status %d, want 404", w2.Code)
	}
}

func TestDeleteProfile(t *testing.T) {
	h, store := newTestHandler(t)
	if _, err := store.AddProfile("prod", false); err != nil {
		t.Fatalf("AddProfile: %v", err)
	}

	req := httptest.NewRequest(http.MethodDelete, "/admin/api/profiles/prod", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("got status %d, want 200", w.Code)
	}
	if len(store.GetProfiles()) != 1 {
		t.Errorf("got %d profiles, want 1", len(store.GetProfiles()))
	}

	req2 := httptest.NewRequest(http.MethodDelete, "/admin/api/profiles/default", nil)
	w2 := httptest.NewRecorder()
	h.ServeHTTP(w2, req2)
	if w2.Code != http.StatusBadRequest {
		t.Errorf("deleting last profile got status %d, want 400", w2.Code)
	}
}

func TestActivateProfile(t *testing.T) {
	h, store := newTestHandler(t)
	if _, err := store.AddProfile("prod", false); err != nil {
		t.Fatalf("AddProfile: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/admin/api/profiles/prod/activate", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("got status %d, want 200", w.Code)
	}
	if store.GetActiveProfileID() != "prod" {
		t.Errorf("active = %q, want prod", store.GetActiveProfileID())
	}

	req2 := httptest.NewRequest(http.MethodPost, "/admin/api/profiles/nope/activate", nil)
	w2 := httptest.NewRecorder()
	h.ServeHTTP(w2, req2)
	if w2.Code != http.StatusNotFound {
		t.Errorf("unknown profile got status %d, want 404", w2.Code)
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
	if len(cfg.Profiles[0].Rules) != 1 {
		t.Errorf("got %d rules, want 1", len(cfg.Profiles[0].Rules))
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
