package config

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"

	"llm-api-router/domain"
)

// Store provides thread-safe access to the configuration with file persistence.
type Store struct {
	mu       sync.RWMutex
	config   *domain.Config
	filepath string
}

// NewStore creates a new config store, loading from file if it exists.
func NewStore(filepath string) (*Store, error) {
	s := &Store{
		config: &domain.Config{
			Servers:         make(map[string]*domain.Server),
			Profiles:        []domain.RuleProfile{{ID: "default", Name: "default", Rules: make([]*domain.RoutingRule, 0)}},
			ActiveProfileID: "default",
		},
		filepath: filepath,
	}

	if err := s.Load(); err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}
	return s, nil
}

// Load reads the config from the JSON file. Creates an empty config if the file doesn't exist.
func (s *Store) Load() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	data, err := os.ReadFile(s.filepath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	migrated := migrateLegacy(data)

	var cfg domain.Config
	if err := json.Unmarshal(migrated, &cfg); err != nil {
		return fmt.Errorf("unmarshal: %w", err)
	}

	if cfg.Servers == nil {
		cfg.Servers = make(map[string]*domain.Server)
	}
	if len(cfg.Profiles) == 0 {
		cfg.Profiles = []domain.RuleProfile{{ID: "default", Name: "default", Rules: make([]*domain.RoutingRule, 0)}}
	}
	if cfg.ActiveProfileID == "" {
		cfg.ActiveProfileID = cfg.Profiles[0].ID
	}

	for i := range cfg.Profiles {
		p := &cfg.Profiles[i]
		if p.Rules == nil {
			p.Rules = make([]*domain.RoutingRule, 0)
		}
		for _, rule := range p.Rules {
			if len(rule.FallbackServerIDs) > 0 && len(rule.Fallbacks) == 0 {
				for _, id := range rule.FallbackServerIDs {
					rule.Fallbacks = append(rule.Fallbacks, domain.FallbackEntry{ServerID: id})
				}
				rule.FallbackServerIDs = nil
			}
		}
	}

	s.config = &cfg
	return nil
}

// save writes the current config to the JSON file.
// Caller must hold s.mu write lock.
// Direct write instead of rename-tmp: rename fails on Docker bind mounts
// with "device or resource busy". The mutex provides write safety.
func (s *Store) save() error {
	data, err := json.MarshalIndent(s.config, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}

	// Guard against Docker volume mount creating a directory instead of file
	if info, statErr := os.Stat(s.filepath); statErr == nil && info.IsDir() {
		return fmt.Errorf("config path %q is a directory (Docker volume mount issue — remove directory and create file)", s.filepath)
	}

	return os.WriteFile(s.filepath, data, 0644)
}

// Save writes the current config to the JSON file atomically.
func (s *Store) Save() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.save()
}

// GetConfig returns a copy of the full config.
func (s *Store) GetConfig() *domain.Config {
	s.mu.RLock()
	defer s.mu.RUnlock()

	cfg := &domain.Config{
		Servers:         make(map[string]*domain.Server, len(s.config.Servers)),
		Profiles:        make([]domain.RuleProfile, len(s.config.Profiles)),
		ActiveProfileID: s.config.ActiveProfileID,
		Settings:        s.config.Settings,
	}

	for id, srv := range s.config.Servers {
		srvCopy := *srv
		srvCopy.APITypes = make([]domain.APIType, len(srv.APITypes))
		copy(srvCopy.APITypes, srv.APITypes)
		cfg.Servers[id] = &srvCopy
	}

	for i, p := range s.config.Profiles {
		cfg.Profiles[i] = domain.RuleProfile{
			ID:    p.ID,
			Name:  p.Name,
			Rules: make([]*domain.RoutingRule, 0, len(p.Rules)),
		}
		for _, rule := range p.Rules {
			cfg.Profiles[i].Rules = append(cfg.Profiles[i].Rules, cloneRule(rule))
		}
	}

	return cfg
}

// GetSettings returns a copy of the global router settings.
func (s *Store) GetSettings() domain.Settings {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.config.Settings
}

// GetServer returns a server by ID.
// Returns a pointer to the internal server — callers must not modify it.
// The config store is the sole writer; server fields are immutable between writes.
func (s *Store) GetServer(id string) (*domain.Server, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	srv, ok := s.config.Servers[id]
	if !ok {
		return nil, false
	}
	return srv, true
}

// GetRuleByModel returns the first routing rule matching the given incoming
// model name from the active profile.
func (s *Store) GetRuleByModel(model string) (*domain.RoutingRule, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	p := s.activeProfile()
	if p == nil {
		return nil, false
	}
	for _, rule := range p.Rules {
		if !rule.Enabled {
			continue
		}
		for _, m := range rule.IncomingModels {
			if m == model {
				return cloneRule(rule), true
			}
		}
	}
	return nil, false
}

// AddServer adds a new server.
func (s *Store) AddServer(srv *domain.Server) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.config.Servers[srv.ID]; exists {
		return fmt.Errorf("server %q already exists", srv.ID)
	}
	apis := make([]domain.APIType, len(srv.APITypes))
	copy(apis, srv.APITypes)
	s.config.Servers[srv.ID] = &domain.Server{
		ID:           srv.ID,
		Name:         srv.Name,
		URL:          srv.URL,
		OpenAIURL:    srv.OpenAIURL,
		AnthropicURL: srv.AnthropicURL,
		APIKey:       srv.APIKey,
		APITypes:     apis,
	}
	return s.save()
}

// UpdateServer updates an existing server.
func (s *Store) UpdateServer(id string, srv *domain.Server) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.config.Servers[id]; !exists {
		return fmt.Errorf("server %q not found", id)
	}
	apis := make([]domain.APIType, len(srv.APITypes))
	copy(apis, srv.APITypes)
	s.config.Servers[id] = &domain.Server{
		ID:           id,
		Name:         srv.Name,
		URL:          srv.URL,
		OpenAIURL:    srv.OpenAIURL,
		AnthropicURL: srv.AnthropicURL,
		APIKey:       srv.APIKey,
		APITypes:     apis,
	}
	return s.save()
}

// DeleteServer removes a server.
func (s *Store) DeleteServer(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.config.Servers[id]; !exists {
		return fmt.Errorf("server %q not found", id)
	}
	delete(s.config.Servers, id)
	return s.save()
}

// AddRule appends a new routing rule to the active profile.
func (s *Store) AddRule(rule *domain.RoutingRule) error {
	return s.AddRuleToProfile("", rule)
}

// UpdateRule updates a rule by index in the active profile.
func (s *Store) UpdateRule(idx int, rule *domain.RoutingRule) error {
	return s.UpdateRuleInProfile("", idx, rule)
}

// DeleteRule removes a rule by index from the active profile.
func (s *Store) DeleteRule(idx int) error {
	return s.DeleteRuleFromProfile("", idx)
}

// profileByID returns the profile with the given ID, or nil if not found.
// Caller must hold s.mu (read or write).
func (s *Store) profileByID(id string) *domain.RuleProfile {
	for i := range s.config.Profiles {
		if s.config.Profiles[i].ID == id {
			return &s.config.Profiles[i]
		}
	}
	return nil
}

// resolveProfile returns the profile to operate on: the active profile when
// id is empty, otherwise the profile with that ID (nil if not found).
// Caller must hold s.mu (read or write).
func (s *Store) resolveProfile(id string) *domain.RuleProfile {
	if id == "" {
		return s.activeProfile()
	}
	return s.profileByID(id)
}

// activeProfile returns the profile pointed to by ActiveProfileID, falling
// back to Profiles[0] when the ID is unknown or empty. Caller must hold s.mu
// (read or write). Returns nil only if Profiles is empty, which the
// len(Profiles) >= 1 invariant prevents.
func (s *Store) activeProfile() *domain.RuleProfile {
	if p := s.profileByID(s.config.ActiveProfileID); p != nil {
		return p
	}
	if len(s.config.Profiles) == 0 {
		return nil
	}
	return &s.config.Profiles[0]
}

// GetActiveProfileID returns the ID of the active profile.
func (s *Store) GetActiveProfileID() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.config.ActiveProfileID
}

// GetActiveRules returns a deep copy of the active profile's rules.
func (s *Store) GetActiveRules() []*domain.RoutingRule {
	rules, _ := s.GetRules("")
	return rules
}

// GetRules returns a deep copy of the rules of the profile with the given ID.
// An empty ID selects the active profile.
func (s *Store) GetRules(id string) ([]*domain.RoutingRule, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	p := s.resolveProfile(id)
	if p == nil {
		return nil, fmt.Errorf("profile %q not found", id)
	}
	out := make([]*domain.RoutingRule, 0, len(p.Rules))
	for _, rule := range p.Rules {
		out = append(out, cloneRule(rule))
	}
	return out, nil
}

// AddRuleToProfile appends a rule to the profile with the given ID
// (empty = active profile).
func (s *Store) AddRuleToProfile(id string, rule *domain.RoutingRule) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	p := s.resolveProfile(id)
	if p == nil {
		return fmt.Errorf("profile %q not found", id)
	}
	p.Rules = append(p.Rules, cloneRule(rule))
	return s.save()
}

// UpdateRuleInProfile updates a rule by index in the profile with the given
// ID (empty = active profile).
func (s *Store) UpdateRuleInProfile(id string, idx int, rule *domain.RoutingRule) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	p := s.resolveProfile(id)
	if p == nil {
		return fmt.Errorf("profile %q not found", id)
	}
	if idx < 0 || idx >= len(p.Rules) {
		return fmt.Errorf("rule index %d out of range", idx)
	}
	p.Rules[idx] = cloneRule(rule)
	return s.save()
}

// DeleteRuleFromProfile removes a rule by index from the profile with the
// given ID (empty = active profile).
func (s *Store) DeleteRuleFromProfile(id string, idx int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	p := s.resolveProfile(id)
	if p == nil {
		return fmt.Errorf("profile %q not found", id)
	}
	if idx < 0 || idx >= len(p.Rules) {
		return fmt.Errorf("rule index %d out of range", idx)
	}
	p.Rules = append(p.Rules[:idx], p.Rules[idx+1:]...)
	return s.save()
}

// GetProfiles returns a deep copy of all profiles.
func (s *Store) GetProfiles() []domain.RuleProfile {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make([]domain.RuleProfile, len(s.config.Profiles))
	for i, p := range s.config.Profiles {
		out[i] = domain.RuleProfile{
			ID:    p.ID,
			Name:  p.Name,
			Rules: make([]*domain.RoutingRule, 0, len(p.Rules)),
		}
		for _, rule := range p.Rules {
			out[i].Rules = append(out[i].Rules, cloneRule(rule))
		}
	}
	return out
}

// AddProfile creates a new profile. Its ID is set to name. copyFromActive
// clones the active profile's rules into the new profile. The new profile is
// not activated. Returns the new profile's ID.
func (s *Store) AddProfile(name string, copyFromActive bool) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if name == "" {
		return "", fmt.Errorf("profile name is required")
	}
	for _, p := range s.config.Profiles {
		if p.Name == name || p.ID == name {
			return "", fmt.Errorf("profile %q already exists", name)
		}
	}

	p := domain.RuleProfile{ID: name, Name: name, Rules: make([]*domain.RoutingRule, 0)}
	if copyFromActive {
		if active := s.activeProfile(); active != nil {
			for _, rule := range active.Rules {
				p.Rules = append(p.Rules, cloneRule(rule))
			}
		}
	}
	s.config.Profiles = append(s.config.Profiles, p)
	return p.ID, s.save()
}

// RenameProfile changes a profile's display name. The ID is unchanged.
func (s *Store) RenameProfile(id, name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if name == "" {
		return fmt.Errorf("profile name is required")
	}
	for i := range s.config.Profiles {
		if s.config.Profiles[i].ID != id {
			continue
		}
		for _, p := range s.config.Profiles {
			if p.ID != id && (p.Name == name || p.ID == name) {
				return fmt.Errorf("profile %q already exists", name)
			}
		}
		s.config.Profiles[i].Name = name
		return s.save()
	}
	return fmt.Errorf("profile %q not found", id)
}

// DeleteProfile removes a profile. The last remaining profile cannot be
// deleted. If the active profile is deleted, the first remaining profile
// becomes active.
func (s *Store) DeleteProfile(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	idx := -1
	for i := range s.config.Profiles {
		if s.config.Profiles[i].ID == id {
			idx = i
			break
		}
	}
	if idx == -1 {
		return fmt.Errorf("profile %q not found", id)
	}
	if len(s.config.Profiles) == 1 {
		return fmt.Errorf("cannot delete the only profile")
	}

	s.config.Profiles = append(s.config.Profiles[:idx], s.config.Profiles[idx+1:]...)
	if s.config.ActiveProfileID == id {
		s.config.ActiveProfileID = s.config.Profiles[0].ID
	}
	return s.save()
}

// SetActiveProfile sets the active profile by ID.
func (s *Store) SetActiveProfile(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for i := range s.config.Profiles {
		if s.config.Profiles[i].ID == id {
			s.config.ActiveProfileID = id
			return s.save()
		}
	}
	return fmt.Errorf("profile %q not found", id)
}

// cloneRule returns a deep copy of a RoutingRule with fresh slices.
func cloneRule(rule *domain.RoutingRule) *domain.RoutingRule {
	cpy := &domain.RoutingRule{
		TargetModel:   rule.TargetModel,
		ServerID:      rule.ServerID,
		NumRetries:    rule.NumRetries,
		Enabled:       rule.Enabled,
		ContextWindow: rule.ContextWindow,
	}
	if rule.IncomingModels != nil {
		cpy.IncomingModels = make([]string, len(rule.IncomingModels))
		copy(cpy.IncomingModels, rule.IncomingModels)
	}
	if rule.Fallbacks != nil {
		cpy.Fallbacks = make([]domain.FallbackEntry, len(rule.Fallbacks))
		copy(cpy.Fallbacks, rule.Fallbacks)
	}
	if rule.FallbackServerIDs != nil {
		cpy.FallbackServerIDs = make([]string, len(rule.FallbackServerIDs))
		copy(cpy.FallbackServerIDs, rule.FallbackServerIDs)
	}
	return cpy
}

// migrateLegacy upgrades pre-profile configs before unmarshal: converts old
// "api_type": "openai" server fields to "api_types": ["openai"], and a legacy
// top-level "rules" array into a "profiles" array holding a single "default"
// profile (the new struct has no "rules" field, so without this the rules
// would silently drop). No-op when the input isn't a JSON object or nothing
// legacy is present.
func migrateLegacy(data []byte) []byte {
	var raw map[string]interface{}
	if json.Unmarshal(data, &raw) != nil {
		return data
	}

	changed := false

	if servers, ok := raw["servers"].(map[string]interface{}); ok {
		for _, srvVal := range servers {
			srv, ok := srvVal.(map[string]interface{})
			if !ok {
				continue
			}
			if oldType, has := srv["api_type"]; has {
				if str, ok := oldType.(string); ok {
					srv["api_types"] = []string{str}
					delete(srv, "api_type")
					changed = true
				}
			}
		}
	}

	if _, has := raw["profiles"]; !has {
		if rules, has := raw["rules"]; has {
			raw["profiles"] = []interface{}{
				map[string]interface{}{
					"id":    "default",
					"name":  "default",
					"rules": rules,
				},
			}
			raw["active_profile_id"] = "default"
			delete(raw, "rules")
			changed = true
		}
	}

	if !changed {
		return data
	}

	out, _ := json.MarshalIndent(raw, "", "  ")
	return out
}
