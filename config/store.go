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
		config:   &domain.Config{Servers: make(map[string]*domain.Server), Rules: make([]*domain.RoutingRule, 0)},
		filepath: filepath,
	}

	if err := s.Load(); err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}
	return s, nil
}

// Load reads the config from the JSON file. Creates an empty config if the file doesn't exist.
func (s *Store) Load() error {
	data, err := os.ReadFile(s.filepath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	migrated := migrateOldAPIType(data)

	var cfg domain.Config
	if err := json.Unmarshal(migrated, &cfg); err != nil {
		return fmt.Errorf("unmarshal: %w", err)
	}

	if cfg.Servers == nil {
		cfg.Servers = make(map[string]*domain.Server)
	}
	if cfg.Rules == nil {
		cfg.Rules = make([]*domain.RoutingRule, 0)
	}

	for _, rule := range cfg.Rules {
		if len(rule.FallbackServerIDs) > 0 && len(rule.Fallbacks) == 0 {
			for _, id := range rule.FallbackServerIDs {
				rule.Fallbacks = append(rule.Fallbacks, domain.FallbackEntry{ServerID: id})
			}
			rule.FallbackServerIDs = nil
		}
	}

	s.config = &cfg
	return nil
}

// Save writes the current config to the JSON file.
func (s *Store) Save() error {
	s.mu.RLock()
	data, err := json.MarshalIndent(s.config, "", "  ")
	s.mu.RUnlock()
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}

	return os.WriteFile(s.filepath, data, 0644)
}

// GetConfig returns a copy of the full config.
func (s *Store) GetConfig() *domain.Config {
	s.mu.RLock()
	defer s.mu.RUnlock()

	cfg := &domain.Config{
		Servers: make(map[string]*domain.Server, len(s.config.Servers)),
		Rules:   make([]*domain.RoutingRule, len(s.config.Rules)),
	}

	for id, srv := range s.config.Servers {
		srvCopy := *srv
		srvCopy.APITypes = make([]domain.APIType, len(srv.APITypes))
		copy(srvCopy.APITypes, srv.APITypes)
		cfg.Servers[id] = &srvCopy
	}

	for i, rule := range s.config.Rules {
		ruleCopy := *rule
		ruleCopy.IncomingModels = make([]string, len(rule.IncomingModels))
		copy(ruleCopy.IncomingModels, rule.IncomingModels)
		if rule.Fallbacks != nil {
			ruleCopy.Fallbacks = make([]domain.FallbackEntry, len(rule.Fallbacks))
			copy(ruleCopy.Fallbacks, rule.Fallbacks)
		}
		if rule.FallbackServerIDs != nil {
			ruleCopy.FallbackServerIDs = make([]string, len(rule.FallbackServerIDs))
			copy(ruleCopy.FallbackServerIDs, rule.FallbackServerIDs)
		}
		cfg.Rules[i] = &ruleCopy
	}

	return cfg
}

// GetServer returns a server by ID.
func (s *Store) GetServer(id string) (*domain.Server, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	srv, ok := s.config.Servers[id]
	if !ok {
		return nil, false
	}
	cpy := *srv
	cpy.APITypes = make([]domain.APIType, len(srv.APITypes))
	copy(cpy.APITypes, srv.APITypes)
	return &cpy, true
}

// GetRuleByModel returns the first routing rule matching the given incoming model name.
func (s *Store) GetRuleByModel(model string) (*domain.RoutingRule, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	for _, rule := range s.config.Rules {
		if !rule.Enabled {
			continue
		}
		for _, m := range rule.IncomingModels {
			if m == model {
				cpy := *rule
				cpy.IncomingModels = make([]string, len(rule.IncomingModels))
				copy(cpy.IncomingModels, rule.IncomingModels)
				if rule.Fallbacks != nil {
					cpy.Fallbacks = make([]domain.FallbackEntry, len(rule.Fallbacks))
					copy(cpy.Fallbacks, rule.Fallbacks)
				}
				if rule.FallbackServerIDs != nil {
					cpy.FallbackServerIDs = make([]string, len(rule.FallbackServerIDs))
					copy(cpy.FallbackServerIDs, rule.FallbackServerIDs)
				}
				return &cpy, true
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
	s.config.Servers[srv.ID] = &domain.Server{
		ID:           srv.ID,
		Name:         srv.Name,
		URL:          srv.URL,
		OpenAIURL:    srv.OpenAIURL,
		AnthropicURL: srv.AnthropicURL,
		APIKey:       srv.APIKey,
		APITypes:     srv.APITypes,
	}
	return nil
}

// UpdateServer updates an existing server.
func (s *Store) UpdateServer(id string, srv *domain.Server) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.config.Servers[id]; !exists {
		return fmt.Errorf("server %q not found", id)
	}
	s.config.Servers[id] = &domain.Server{
		ID:           id,
		Name:         srv.Name,
		URL:          srv.URL,
		OpenAIURL:    srv.OpenAIURL,
		AnthropicURL: srv.AnthropicURL,
		APIKey:       srv.APIKey,
		APITypes:     srv.APITypes,
	}
	return nil
}

// DeleteServer removes a server.
func (s *Store) DeleteServer(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.config.Servers[id]; !exists {
		return fmt.Errorf("server %q not found", id)
	}
	delete(s.config.Servers, id)
	return nil
}

// AddRule appends a new routing rule.
func (s *Store) AddRule(rule *domain.RoutingRule) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.config.Rules = append(s.config.Rules, &domain.RoutingRule{
		IncomingModels: rule.IncomingModels,
		TargetModel:    rule.TargetModel,
		ServerID:       rule.ServerID,
		Fallbacks:      rule.Fallbacks,
		Enabled:        rule.Enabled,
	})
	return nil
}

// UpdateRule updates a rule by index.
func (s *Store) UpdateRule(idx int, rule *domain.RoutingRule) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if idx < 0 || idx >= len(s.config.Rules) {
		return fmt.Errorf("rule index %d out of range", idx)
	}
	s.config.Rules[idx] = &domain.RoutingRule{
		IncomingModels: rule.IncomingModels,
		TargetModel:    rule.TargetModel,
		ServerID:       rule.ServerID,
		Fallbacks:      rule.Fallbacks,
		Enabled:        rule.Enabled,
	}
	return nil
}

// DeleteRule removes a rule by index.
func (s *Store) DeleteRule(idx int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if idx < 0 || idx >= len(s.config.Rules) {
		return fmt.Errorf("rule index %d out of range", idx)
	}
	s.config.Rules = append(s.config.Rules[:idx], s.config.Rules[idx+1:]...)
	return nil
}

// migrateOldAPIType converts old "api_type": "openai" → "api_types": ["openai"] in raw JSON.
func migrateOldAPIType(data []byte) []byte {
	var raw map[string]interface{}
	if json.Unmarshal(data, &raw) != nil {
		return data
	}

	servers, ok := raw["servers"].(map[string]interface{})
	if !ok {
		return data
	}

	changed := false
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

	if !changed {
		return data
	}

	out, _ := json.MarshalIndent(raw, "", "  ")
	return out
}
