package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"llm-api-router/domain"
)

func TestNewStore(t *testing.T) {
	t.Run("creates empty store when file missing", func(t *testing.T) {
		s, err := NewStore("/nonexistent/path.json")
		if err != nil {
			t.Fatalf("NewStore on missing file: %v", err)
		}
		cfg := s.GetConfig()
		if cfg.Servers == nil {
			t.Error("Servers map should not be nil")
		}
		if cfg.Rules == nil {
			t.Error("Rules slice should not be nil")
		}
	})

	t.Run("loads existing file", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "config.json")
		if err := os.WriteFile(path, []byte(`{"servers":{"s1":{"id":"s1","name":"test","url":"http://example.com","api_key":"key","api_types":["openai"]}},"rules":[{"incoming_models":["gpt-4"],"target_model":"gpt-4","server_id":"s1","enabled":true}]}`), 0644); err != nil {
			t.Fatal(err)
		}

		s, err := NewStore(path)
		if err != nil {
			t.Fatalf("NewStore: %v", err)
		}

		cfg := s.GetConfig()
		if len(cfg.Servers) != 1 {
			t.Errorf("expected 1 server, got %d", len(cfg.Servers))
		}
		if len(cfg.Rules) != 1 {
			t.Errorf("expected 1 rule, got %d", len(cfg.Rules))
		}
	})
}

func TestStore_AddGetServer(t *testing.T) {
	s := newEmptyStore(t)

	srv := &domain.Server{ID: "test-srv", Name: "Test", URL: "http://localhost:8080", APIKey: "abc", APITypes: []domain.APIType{domain.APITypeOpenAI}}
	if err := s.AddServer(srv); err != nil {
		t.Fatalf("AddServer: %v", err)
	}

	got, ok := s.GetServer("test-srv")
	if !ok {
		t.Fatal("GetServer: not found")
	}
	if got.Name != "Test" {
		t.Errorf("got name %q, want %q", got.Name, "Test")
	}
	if len(got.APITypes) != 1 || got.APITypes[0] != domain.APITypeOpenAI {
		t.Errorf("got api_types %v, want [openai]", got.APITypes)
	}
}

func TestStore_AddServerDuplicate(t *testing.T) {
	s := newEmptyStore(t)
	_ = s.AddServer(&domain.Server{ID: "s1", URL: "http://a.com"})
	err := s.AddServer(&domain.Server{ID: "s1", URL: "http://b.com"})
	if err == nil {
		t.Fatal("expected error for duplicate server ID")
	}
}

func TestStore_UpdateServer(t *testing.T) {
	s := newEmptyStore(t)
	_ = s.AddServer(&domain.Server{ID: "s1", URL: "http://old.com", Name: "Old"})

	if err := s.UpdateServer("s1", &domain.Server{URL: "http://new.com", Name: "New", APITypes: []domain.APIType{domain.APITypeAnthropic}}); err != nil {
		t.Fatalf("UpdateServer: %v", err)
	}

	got, _ := s.GetServer("s1")
	if got.URL != "http://new.com" {
		t.Errorf("got URL %q, want %q", got.URL, "http://new.com")
	}
	if got.Name != "New" {
		t.Errorf("got Name %q, want %q", got.Name, "New")
	}
}

func TestStore_UpdateServerNotFound(t *testing.T) {
	s := newEmptyStore(t)
	err := s.UpdateServer("nonexistent", &domain.Server{URL: "http://x.com"})
	if err == nil {
		t.Fatal("expected error for nonexistent server")
	}
}

func TestStore_DeleteServer(t *testing.T) {
	s := newEmptyStore(t)
	_ = s.AddServer(&domain.Server{ID: "s1", URL: "http://a.com"})
	if err := s.DeleteServer("s1"); err != nil {
		t.Fatalf("DeleteServer: %v", err)
	}
	if _, ok := s.GetServer("s1"); ok {
		t.Fatal("server should be deleted")
	}
}

func TestStore_DeleteServerNotFound(t *testing.T) {
	s := newEmptyStore(t)
	err := s.DeleteServer("nonexistent")
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestStore_AddGetRule(t *testing.T) {
	s := newEmptyStore(t)
	rule := &domain.RoutingRule{
		IncomingModels: []string{"gpt-4", "gpt-4-turbo"},
		TargetModel:    "gpt-4",
		ServerID:       "s1",
		Enabled:        true,
	}
	if err := s.AddRule(rule); err != nil {
		t.Fatalf("AddRule: %v", err)
	}

	got, ok := s.GetRuleByModel("gpt-4")
	if !ok {
		t.Fatal("GetRuleByModel: not found")
	}
	if got.TargetModel != "gpt-4" {
		t.Errorf("got target_model %q, want %q", got.TargetModel, "gpt-4")
	}
}

func TestStore_GetRuleByModel_Disabled(t *testing.T) {
	s := newEmptyStore(t)
	_ = s.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "s1",
		Enabled:        false,
	})
	if _, ok := s.GetRuleByModel("gpt-4"); ok {
		t.Fatal("disabled rule should not match")
	}
}

func TestStore_GetRuleByModel_NotFound(t *testing.T) {
	s := newEmptyStore(t)
	if _, ok := s.GetRuleByModel("nonexistent"); ok {
		t.Fatal("should not find nonexistent model")
	}
}

func TestStore_UpdateRule(t *testing.T) {
	s := newEmptyStore(t)
	_ = s.AddRule(&domain.RoutingRule{IncomingModels: []string{"old"}, ServerID: "s1", Enabled: true})

	if err := s.UpdateRule(0, &domain.RoutingRule{IncomingModels: []string{"new"}, ServerID: "s2", Enabled: true}); err != nil {
		t.Fatalf("UpdateRule: %v", err)
	}

	got, ok := s.GetRuleByModel("new")
	if !ok {
		t.Fatal("updated rule not found")
	}
	if got.ServerID != "s2" {
		t.Errorf("got server_id %q, want %q", got.ServerID, "s2")
	}
}

func TestStore_UpdateRuleOutOfRange(t *testing.T) {
	s := newEmptyStore(t)
	err := s.UpdateRule(5, &domain.RoutingRule{ServerID: "s1"})
	if err == nil {
		t.Fatal("expected error for out-of-range index")
	}
}

func TestStore_DeleteRule(t *testing.T) {
	s := newEmptyStore(t)
	_ = s.AddRule(&domain.RoutingRule{IncomingModels: []string{"m1"}, ServerID: "s1", Enabled: true})
	_ = s.AddRule(&domain.RoutingRule{IncomingModels: []string{"m2"}, ServerID: "s2", Enabled: true})

	if err := s.DeleteRule(0); err != nil {
		t.Fatalf("DeleteRule: %v", err)
	}

	if _, ok := s.GetRuleByModel("m1"); ok {
		t.Fatal("deleted rule should not be found")
	}
	if _, ok := s.GetRuleByModel("m2"); !ok {
		t.Fatal("remaining rule should be found")
	}
}

func TestStore_DeleteRuleOutOfRange(t *testing.T) {
	s := newEmptyStore(t)
	err := s.DeleteRule(0)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestStore_SaveAndLoad(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")

	s, err := NewStore(path)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}

	_ = s.AddServer(&domain.Server{ID: "s1", Name: "Server 1", URL: "http://s1.com", APIKey: "key1", APITypes: []domain.APIType{domain.APITypeOpenAI}})
	_ = s.AddRule(&domain.RoutingRule{IncomingModels: []string{"gpt-4"}, TargetModel: "gpt-4", ServerID: "s1", Enabled: true})

	if err := s.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	s2, err := NewStore(path)
	if err != nil {
		t.Fatalf("NewStore reload: %v", err)
	}

	cfg := s2.GetConfig()
	if len(cfg.Servers) != 1 {
		t.Errorf("got %d servers, want 1", len(cfg.Servers))
	}
	if len(cfg.Rules) != 1 {
		t.Errorf("got %d rules, want 1", len(cfg.Rules))
	}
}

func TestStore_ConcurrentAccess(t *testing.T) {
	s := newEmptyStore(t)
	var wg sync.WaitGroup

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			id := string(rune('a' + n%26))
			_ = s.AddServer(&domain.Server{ID: id, URL: "http://test.com"})
			s.GetServer(id)
			_ = s.GetConfig()
		}(i)
	}
	wg.Wait()
}

func TestMigrateOldAPIType(t *testing.T) {
	input := []byte(`{"servers":{"s1":{"api_type":"openai","url":"http://x.com"}}}`)
	output := migrateOldAPIType(input)

	var raw map[string]interface{}
	if err := json.Unmarshal(output, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	servers := raw["servers"].(map[string]interface{})
	s1 := servers["s1"].(map[string]interface{})

	if _, has := s1["api_type"]; has {
		t.Error("api_type should be removed")
	}

	apiTypes, ok := s1["api_types"].([]interface{})
	if !ok {
		t.Fatal("api_types should be an array")
	}
	if len(apiTypes) != 1 || apiTypes[0].(string) != "openai" {
		t.Errorf("got api_types %v, want [openai]", apiTypes)
	}
}

func TestMigrateOldAPIType_NoChange(t *testing.T) {
	input := []byte(`{"servers":{"s1":{"api_types":["openai"],"url":"http://x.com"}}}`)
	output := migrateOldAPIType(input)
	if string(output) != string(input) {
		t.Error("output should match input when no migration needed")
	}
}

func TestMigrateOldAPIType_NotJSON(t *testing.T) {
	input := []byte(`not json`)
	output := migrateOldAPIType(input)
	if string(output) != string(input) {
		t.Error("should return original data for invalid JSON")
	}
}

func TestLegacyFallbackServerIDs(t *testing.T) {
	data := []byte(`{"servers":{"s1":{"id":"s1","url":"http://x.com","api_key":"k","api_types":["openai"]}},"rules":[{"incoming_models":["m1"],"target_model":"m1","server_id":"s1","enabled":true,"fallback_server_ids":["s2","s3"]}]}`)

	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}

	s, err := NewStore(path)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}

	rule, ok := s.GetRuleByModel("m1")
	if !ok {
		t.Fatal("rule not found")
	}

	if len(rule.FallbackServerIDs) != 0 {
		t.Error("FallbackServerIDs should be cleared after migration")
	}

	if len(rule.Fallbacks) != 2 {
		t.Fatalf("expected 2 fallbacks, got %d", len(rule.Fallbacks))
	}
	if rule.Fallbacks[0].ServerID != "s2" || rule.Fallbacks[1].ServerID != "s3" {
		t.Errorf("got fallbacks %v, want [{s2 } {s3 }]", rule.Fallbacks)
	}
}

func TestServerGetURLForAPIType(t *testing.T) {
	tests := []struct {
		name    string
		server  domain.Server
		apiType domain.APIType
		want    string
	}{
		{
			name:    "no override returns base",
			server:  domain.Server{URL: "http://base.com"},
			apiType: domain.APITypeOpenAI,
			want:    "http://base.com",
		},
		{
			name:    "openai override",
			server:  domain.Server{URL: "http://base.com", OpenAIURL: "http://openai.com"},
			apiType: domain.APITypeOpenAI,
			want:    "http://openai.com",
		},
		{
			name:    "anthropic override",
			server:  domain.Server{URL: "http://base.com", AnthropicURL: "http://anthropic.com"},
			apiType: domain.APITypeAnthropic,
			want:    "http://anthropic.com",
		},
		{
			name:    "unrelated override falls back",
			server:  domain.Server{URL: "http://base.com", OpenAIURL: "http://openai.com"},
			apiType: domain.APITypeAnthropic,
			want:    "http://base.com",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.server.GetURLForAPIType(tt.apiType)
			if got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestServerSupportsAPIType(t *testing.T) {
	s := domain.Server{APITypes: []domain.APIType{domain.APITypeOpenAI}}
	if !s.SupportsAPIType(domain.APITypeOpenAI) {
		t.Error("should support openai")
	}
	if s.SupportsAPIType(domain.APITypeAnthropic) {
		t.Error("should not support anthropic")
	}
}

func TestRoutingRuleGetTargetModelForServer(t *testing.T) {
	rule := &domain.RoutingRule{
		TargetModel: "gpt-4",
		ServerID:    "primary",
		Fallbacks: []domain.FallbackEntry{
			{ServerID: "fb1", TargetModel: "claude-3"},
			{ServerID: "fb2"},
		},
	}

	tests := []struct {
		serverID string
		want     string
	}{
		{"primary", "gpt-4"},
		{"fb1", "claude-3"},
		{"fb2", "gpt-4"},
		{"unknown", "gpt-4"},
	}
	for _, tt := range tests {
		got := rule.GetTargetModelForServer(tt.serverID)
		if got != tt.want {
			t.Errorf("GetTargetModelForServer(%q) = %q, want %q", tt.serverID, got, tt.want)
		}
	}
}

func TestRoutingRuleAllServerIDs(t *testing.T) {
	rule := &domain.RoutingRule{
		ServerID: "primary",
		Fallbacks: []domain.FallbackEntry{
			{ServerID: "fb1"},
			{ServerID: "fb2"},
		},
	}
	ids := rule.AllServerIDs()
	want := []string{"primary", "fb1", "fb2"}
	if len(ids) != len(want) {
		t.Fatalf("got %v, want %v", ids, want)
	}
	for i, id := range ids {
		if id != want[i] {
			t.Errorf("ids[%d] = %q, want %q", i, id, want[i])
		}
	}
}

func TestGetConfigDeepCopy(t *testing.T) {
	s := newEmptyStore(t)
	_ = s.AddServer(&domain.Server{ID: "s1", URL: "http://test.com", APITypes: []domain.APIType{domain.APITypeOpenAI}})
	_ = s.AddRule(&domain.RoutingRule{IncomingModels: []string{"m1"}, ServerID: "s1", Enabled: true})

	cfg := s.GetConfig()
	cfg.Servers["s1"].URL = "http://hacked.com"
	cfg.Rules[0].ServerID = "hacked"

	original, _ := s.GetServer("s1")
	if original.URL == "http://hacked.com" {
		t.Error("GetConfig should return a deep copy")
	}

	rule, _ := s.GetRuleByModel("m1")
	if rule.ServerID == "hacked" {
		t.Error("GetConfig should return a deep copy of rules")
	}
}

func newEmptyStore(t *testing.T) *Store {
	t.Helper()
	s, err := NewStore("/nonexistent/path.json")
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	return s
}
