package config

import (
	"context"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"llm-api-router/domain"
)

// HealthTracker tracks server health status with a background checker.
type HealthTracker struct {
	mu      sync.RWMutex
	status  map[string]bool // true = healthy
	store   *Store
	interval time.Duration
	stopCh  chan struct{}
}

// NewHealthTracker creates a tracker with the given check interval.
func NewHealthTracker(store *Store, interval time.Duration) *HealthTracker {
	return &HealthTracker{
		status:   make(map[string]bool),
		store:    store,
		interval: interval,
		stopCh:   make(chan struct{}),
	}
}

// Start begins the background health checker.
func (t *HealthTracker) Start() {
	// Mark all existing servers as healthy
	cfg := t.store.GetConfig()
	for id := range cfg.Servers {
		t.mu.Lock()
		t.status[id] = true
		t.mu.Unlock()
	}

	go t.loop()
}

// Stop stops the background health checker.
func (t *HealthTracker) Stop() {
	close(t.stopCh)
}

func (t *HealthTracker) loop() {
	ticker := time.NewTicker(t.interval)
	defer ticker.Stop()

	for {
		select {
		case <-t.stopCh:
			return
		case <-ticker.C:
			t.checkAll()
		}
	}
}

func (t *HealthTracker) checkAll() {
	cfg := t.store.GetConfig()
	for id, srv := range cfg.Servers {
		if !t.IsHealthy(id) {
			// Server is marked unhealthy — check if it's back
			if t.checkServer(srv) {
				log.Printf("[health] %s is back", srv.Name)
				t.MarkHealthy(id)
			}
		}
	}
}

// checkServer pings the server's /v1/models endpoint.
func (t *HealthTracker) checkServer(srv *domain.Server) bool {
	rawURL := srv.GetURLForAPIType(domain.APITypeOpenAI)
	if !strings.Contains(rawURL, "://") {
		rawURL = "https://" + rawURL
	}

	modelsURL := rawURL + "/v1/models"
	modelsURL = strings.ReplaceAll(modelsURL, "/v1/v1", "/v1")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, modelsURL, nil)
	if err != nil {
		return false
	}
	if srv.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+srv.APIKey)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close() //nolint:errcheck
	return resp.StatusCode == http.StatusOK
}

// IsHealthy reports whether the server is currently considered healthy.
func (t *HealthTracker) IsHealthy(id string) bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.status[id]
}

// MarkHealthy marks a server as healthy.
func (t *HealthTracker) MarkHealthy(id string) {
	t.mu.Lock()
	t.status[id] = true
	t.mu.Unlock()
}

// MarkUnhealthy marks a server as unhealthy.
func (t *HealthTracker) MarkUnhealthy(id string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.status[id] {
		log.Printf("[health] %s is down", id)
	}
	t.status[id] = false
}

// GetStatus returns a copy of the current health status map.
func (t *HealthTracker) GetStatus() map[string]bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	out := make(map[string]bool, len(t.status))
	for k, v := range t.status {
		out[k] = v
	}
	return out
}
