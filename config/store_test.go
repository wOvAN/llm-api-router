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
		if len(cfg.Profiles) != 1 {
			t.Errorf("expected 1 profile, got %d", len(cfg.Profiles))
		} else if cfg.Profiles[0].Rules == nil {
			t.Error("profile Rules slice should not be nil")
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
		if len(cfg.Profiles) != 1 {
			t.Errorf("expected 1 profile, got %d", len(cfg.Profiles))
		} else if len(cfg.Profiles[0].Rules) != 1 {
			t.Errorf("expected 1 rule, got %d", len(cfg.Profiles[0].Rules))
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
	if len(cfg.Profiles) != 1 {
		t.Errorf("got %d profiles, want 1", len(cfg.Profiles))
	} else if len(cfg.Profiles[0].Rules) != 1 {
		t.Errorf("got %d rules, want 1", len(cfg.Profiles[0].Rules))
	}
}

func TestStore_ConcurrentAccess(t *testing.T) {
	s := newEmptyStore(t)
	var wg sync.WaitGroup

	for i := range 100 {
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
	output := migrateLegacy(input)

	var raw map[string]any
	if err := json.Unmarshal(output, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	servers := raw["servers"].(map[string]any)
	s1 := servers["s1"].(map[string]any)

	if _, has := s1["api_type"]; has {
		t.Error("api_type should be removed")
	}

	apiTypes, ok := s1["api_types"].([]any)
	if !ok {
		t.Fatal("api_types should be an array")
	}
	if len(apiTypes) != 1 || apiTypes[0].(string) != "openai" {
		t.Errorf("got api_types %v, want [openai]", apiTypes)
	}
}

func TestMigrateOldAPIType_NoChange(t *testing.T) {
	input := []byte(`{"servers":{"s1":{"api_types":["openai"],"url":"http://x.com"}}}`)
	output := migrateLegacy(input)
	if string(output) != string(input) {
		t.Error("output should match input when no migration needed")
	}
}

func TestMigrateOldAPIType_NotJSON(t *testing.T) {
	input := []byte(`not json`)
	output := migrateLegacy(input)
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

func TestMigrateRulesToProfiles(t *testing.T) {
	input := []byte(`{"servers":{"s1":{"id":"s1","url":"http://x.com"}},"rules":[{"incoming_models":["m1"],"target_model":"m1","server_id":"s1","enabled":true}]}`)
	output := migrateLegacy(input)

	var raw map[string]any
	if err := json.Unmarshal(output, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, has := raw["rules"]; has {
		t.Error("rules key should be removed")
	}
	if raw["active_profile_id"] != "default" {
		t.Errorf("active_profile_id = %v, want default", raw["active_profile_id"])
	}
	profiles, ok := raw["profiles"].([]any)
	if !ok || len(profiles) != 1 {
		t.Fatalf("expected 1 profile, got %v", raw["profiles"])
	}
	p := profiles[0].(map[string]any)
	if p["id"] != "default" || p["name"] != "default" {
		t.Errorf("profile id/name = %v/%v, want default/default", p["id"], p["name"])
	}
	if rules, _ := p["rules"].([]any); len(rules) != 1 {
		t.Fatalf("expected 1 rule in profile, got %v", p["rules"])
	}
}

func TestMigrateRulesToProfiles_NoChange(t *testing.T) {
	input := []byte(`{"servers":{},"profiles":[{"id":"default","name":"default","rules":[]}],"active_profile_id":"default"}`)
	output := migrateLegacy(input)
	if string(output) != string(input) {
		t.Error("output should match input when profiles already present")
	}
}

func TestMigrateRulesToProfiles_NoRulesKey(t *testing.T) {
	input := []byte(`{"servers":{"s1":{"id":"s1","url":"http://x.com"}}}`)
	output := migrateLegacy(input)
	if string(output) != string(input) {
		t.Error("output should match input when no rules key")
	}
}

func TestLoadLegacyRulesFile(t *testing.T) {
	data := []byte(`{"servers":{"s1":{"id":"s1","url":"http://x.com","api_key":"k","api_types":["openai"]}},"rules":[{"incoming_models":["m1"],"target_model":"m1","server_id":"s1","enabled":true}]}`)
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
	s, err := NewStore(path)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	cfg := s.GetConfig()
	if len(cfg.Profiles) != 1 {
		t.Fatalf("expected 1 profile, got %d", len(cfg.Profiles))
	}
	if cfg.Profiles[0].ID != "default" || cfg.Profiles[0].Name != "default" {
		t.Errorf("profile = %q/%q, want default/default", cfg.Profiles[0].ID, cfg.Profiles[0].Name)
	}
	if cfg.ActiveProfileID != "default" {
		t.Errorf("active_profile_id = %q, want default", cfg.ActiveProfileID)
	}
	if len(cfg.Profiles[0].Rules) != 1 {
		t.Errorf("expected 1 rule, got %d", len(cfg.Profiles[0].Rules))
	}
}

func TestLoadNoRulesNoProfiles(t *testing.T) {
	data := []byte(`{"servers":{"s1":{"id":"s1","url":"http://x.com","api_types":["openai"]}}}`)
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
	s, err := NewStore(path)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	cfg := s.GetConfig()
	if len(cfg.Profiles) != 1 {
		t.Fatalf("expected 1 profile, got %d", len(cfg.Profiles))
	}
	if cfg.ActiveProfileID != cfg.Profiles[0].ID {
		t.Errorf("active_profile_id = %q, want %q", cfg.ActiveProfileID, cfg.Profiles[0].ID)
	}
}

func TestLoadBadActiveProfileID(t *testing.T) {
	data := []byte(`{"servers":{},"profiles":[{"id":"a","name":"A","rules":[{"incoming_models":["m1"],"target_model":"m1","server_id":"s1","enabled":true}]},{"id":"b","name":"B","rules":[]}],"active_profile_id":"ghost"}`)
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
	s, err := NewStore(path)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	// Unknown active id falls back to Profiles[0] (id "a"), which holds the rule.
	if _, ok := s.GetRuleByModel("m1"); !ok {
		t.Error("GetRuleByModel should fall back to Profiles[0] for an unknown active id")
	}
}

func TestProfileCRUD(t *testing.T) {
	s := newEmptyStore(t)
	_ = s.AddRule(&domain.RoutingRule{IncomingModels: []string{"m1"}, ServerID: "s1", Enabled: true})

	if id, err := s.AddProfile("prod", false); err != nil {
		t.Fatalf("AddProfile: %v", err)
	} else if id != "prod" {
		t.Errorf("id = %q, want prod", id)
	}
	if got := s.GetProfiles()[1].Rules; len(got) != 0 {
		t.Errorf("non-copy profile should have no rules, got %d", len(got))
	}

	if _, err := s.AddProfile("prod-copy", true); err != nil {
		t.Fatalf("AddProfile copy: %v", err)
	}
	if got := s.GetProfiles()[2].Rules; len(got) != 1 {
		t.Errorf("copy profile should have 1 rule, got %d", len(got))
	}

	if _, err := s.AddProfile("prod", false); err == nil {
		t.Error("duplicate profile name should error")
	}

	if err := s.RenameProfile("prod", "production"); err != nil {
		t.Fatalf("RenameProfile: %v", err)
	}
	renamed := false
	for _, p := range s.GetProfiles() {
		if p.ID == "prod" && p.Name == "production" {
			renamed = true
		}
	}
	if !renamed {
		t.Error("rename should change Name and keep ID")
	}

	if err := s.SetActiveProfile("prod"); err != nil {
		t.Fatalf("SetActiveProfile: %v", err)
	}
	if s.GetActiveProfileID() != "prod" {
		t.Errorf("active = %q, want prod", s.GetActiveProfileID())
	}

	if err := s.SetActiveProfile("nope"); err == nil {
		t.Error("SetActiveProfile unknown should error")
	}
	if err := s.RenameProfile("nope", "x"); err == nil {
		t.Error("RenameProfile unknown should error")
	}
	if err := s.DeleteProfile("nope"); err == nil {
		t.Error("DeleteProfile unknown should error")
	}

	// Deleting the active profile auto-switches to the first remaining one.
	if err := s.DeleteProfile("prod"); err != nil {
		t.Fatalf("DeleteProfile: %v", err)
	}
	if s.GetActiveProfileID() != s.GetProfiles()[0].ID {
		t.Errorf("after deleting active, active = %q, want first profile %q", s.GetActiveProfileID(), s.GetProfiles()[0].ID)
	}

	// Delete down to one profile, then deleting it must error.
	for len(s.GetProfiles()) > 1 {
		_ = s.DeleteProfile(s.GetProfiles()[0].ID)
	}
	if err := s.DeleteProfile(s.GetProfiles()[0].ID); err == nil {
		t.Error("deleting the only profile should error")
	}
}

func TestRulesScopedToActiveProfile(t *testing.T) {
	s := newEmptyStore(t)
	_ = s.AddRule(&domain.RoutingRule{IncomingModels: []string{"m1"}, ServerID: "s1", Enabled: true})

	if _, err := s.AddProfile("empty", false); err != nil {
		t.Fatalf("AddProfile: %v", err)
	}
	if err := s.SetActiveProfile("empty"); err != nil {
		t.Fatalf("SetActiveProfile: %v", err)
	}

	// m1 lives in the other profile, so it is no longer routable.
	if _, ok := s.GetRuleByModel("m1"); ok {
		t.Error("m1 should not be routable from the empty active profile")
	}

	// New rules land in the active profile.
	_ = s.AddRule(&domain.RoutingRule{IncomingModels: []string{"m2"}, ServerID: "s1", Enabled: true})
	if _, ok := s.GetRuleByModel("m2"); !ok {
		t.Error("m2 should be routable from the active profile")
	}

	// The original profile still holds m1.
	for _, p := range s.GetProfiles() {
		if p.ID == "default" && len(p.Rules) != 1 {
			t.Errorf("default profile should still have 1 rule, got %d", len(p.Rules))
		}
	}
}

func TestProfileScopedRuleCRUD(t *testing.T) {
	s := newEmptyStore(t)
	_ = s.AddRule(&domain.RoutingRule{IncomingModels: []string{"m1"}, ServerID: "s1", Enabled: true})
	if _, err := s.AddProfile("prod", false); err != nil {
		t.Fatalf("AddProfile: %v", err)
	}

	// Rules can be added to a non-active profile without activating it.
	err := s.AddRuleToProfile("prod", &domain.RoutingRule{IncomingModels: []string{"m2"}, ServerID: "s2", Enabled: true})
	if err != nil {
		t.Fatalf("AddRuleToProfile: %v", err)
	}
	if got := len(s.GetActiveRules()); got != 1 {
		t.Fatalf("active profile should still have 1 rule, got %d", got)
	}
	rules, err := s.GetRules("prod")
	if err != nil {
		t.Fatalf("GetRules: %v", err)
	}
	if len(rules) != 1 || rules[0].ServerID != "s2" {
		t.Fatalf("prod rules = %+v, want 1 rule on s2", rules)
	}

	// Update and delete by index in the target profile.
	if err := s.UpdateRuleInProfile("prod", 0, &domain.RoutingRule{IncomingModels: []string{"m2"}, ServerID: "s3", Enabled: false}); err != nil {
		t.Fatalf("UpdateRuleInProfile: %v", err)
	}
	rules, _ = s.GetRules("prod")
	if rules[0].ServerID != "s3" || rules[0].Enabled {
		t.Errorf("update not applied: %+v", rules[0])
	}
	if err := s.DeleteRuleFromProfile("prod", 0); err != nil {
		t.Fatalf("DeleteRuleFromProfile: %v", err)
	}
	if rules, _ = s.GetRules("prod"); len(rules) != 0 {
		t.Errorf("prod should be empty after delete, got %d", len(rules))
	}
	if got := len(s.GetActiveRules()); got != 1 {
		t.Errorf("active profile should be untouched, got %d rules", got)
	}

	// Unknown profile errors on every operation.
	if _, err := s.GetRules("nope"); err == nil {
		t.Error("GetRules unknown profile should error")
	}
	if err := s.AddRuleToProfile("nope", &domain.RoutingRule{}); err == nil {
		t.Error("AddRuleToProfile unknown profile should error")
	}
	if err := s.UpdateRuleInProfile("nope", 0, &domain.RoutingRule{}); err == nil {
		t.Error("UpdateRuleInProfile unknown profile should error")
	}
	if err := s.DeleteRuleFromProfile("nope", 0); err == nil {
		t.Error("DeleteRuleFromProfile unknown profile should error")
	}

	// Out-of-range index errors.
	if err := s.UpdateRuleInProfile("prod", 0, &domain.RoutingRule{}); err == nil {
		t.Error("UpdateRuleInProfile out of range should error")
	}
	if err := s.DeleteRuleFromProfile("prod", 5); err == nil {
		t.Error("DeleteRuleFromProfile out of range should error")
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

func TestGetConfigDeepCopy(t *testing.T) {
	s := newEmptyStore(t)
	_ = s.AddServer(&domain.Server{ID: "s1", URL: "http://test.com", APITypes: []domain.APIType{domain.APITypeOpenAI}})
	_ = s.AddRule(&domain.RoutingRule{IncomingModels: []string{"m1"}, ServerID: "s1", Enabled: true})

	cfg := s.GetConfig()
	cfg.Servers["s1"].URL = "http://hacked.com"
	cfg.Profiles[0].Name = "hacked"
	cfg.Profiles[0].Rules[0].ServerID = "hacked"

	original, _ := s.GetServer("s1")
	if original.URL == "http://hacked.com" {
		t.Error("GetConfig should return a deep copy")
	}

	if s.GetProfiles()[0].Name == "hacked" {
		t.Error("GetConfig should return a deep copy of profile names")
	}

	rule, _ := s.GetRuleByModel("m1")
	if rule.ServerID == "hacked" {
		t.Error("GetConfig should return a deep copy of rules")
	}
}

func newEmptyStore(t *testing.T) *Store {
	t.Helper()
	dir := t.TempDir()
	s, err := NewStore(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	return s
}

func TestServerProxyURL(t *testing.T) {
	tests := []struct {
		name   string
		server domain.Server
		want   string
	}{
		{name: "no proxy means direct", server: domain.Server{}, want: ""},
		{name: "proxy returned", server: domain.Server{Proxy: "http://base:3128"}, want: "http://base:3128"},
		{name: "disabled proxy means direct", server: domain.Server{Proxy: "http://base:3128", ProxyEnabled: boolPtr(false)}, want: ""},
		{name: "explicitly enabled proxy works", server: domain.Server{Proxy: "http://base:3128", ProxyEnabled: boolPtr(true)}, want: "http://base:3128"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.server.ProxyURL(); got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func boolPtr(b bool) *bool { return &b }
