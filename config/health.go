package config

import (
	"context"
	"maps"
	"net/http"
	"strings"
	"sync"
	"time"

	"llm-api-router/domain"
	"llm-api-router/pkg/log"
	"llm-api-router/proxy"
)

// HealthTracker tracks server health status with a background checker.
type HealthTracker struct {
	mu       sync.RWMutex
	status   map[string]bool // true = healthy
	store    *Store
	interval time.Duration
	stopCh   chan struct{}
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
		t.mu.Lock()
		if _, exists := t.status[id]; !exists {
			t.status[id] = true
		}
		t.mu.Unlock()
		if !t.IsHealthy(id) {
			// Server is marked unhealthy — check if it's back
			if t.checkServer(srv) {
				log.Infof("[health] %s is back", srv.Name)
				t.MarkHealthy(id)
			}
		}
	}
}

// ModelsURL returns the server's /v1/models probe URL: the protocol-specific
// URL override for OpenAI, scheme defaulted to https, /v1 suffix dedup'd
// against the probe path.
func ModelsURL(srv *domain.Server) string {
	rawURL := srv.GetURLForAPIType(domain.APITypeOpenAI)
	if !strings.Contains(rawURL, "://") {
		rawURL = "https://" + rawURL
	}
	u := strings.TrimRight(rawURL, "/") + "/v1/models"
	return strings.ReplaceAll(u, "/v1/v1", "/v1")
}

// checkServer pings the server's /v1/models endpoint.
func (t *HealthTracker) checkServer(srv *domain.Server) bool {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, ModelsURL(srv), nil)
	if err != nil {
		log.Debugf("[health] %s: failed to create probe request: %v", srv.Name, err)
		return false
	}
	// Probe through the server's configured proxy (same one real OpenAI traffic
	// uses) — a server only reachable via proxy must not be marked unhealthy.
	transport, err := proxy.TransportFor(srv.GetProxyForAPIType(domain.APITypeOpenAI))
	if err != nil {
		log.Warnf("[health] %s: %v", srv.Name, err)
		return false
	}
	if srv.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+srv.APIKey)
	}

	resp, err := (&http.Client{Transport: transport}).Do(req)
	if err != nil {
		log.Debugf("[health] %s: probe failed: %v", srv.Name, err)
		return false
	}
	if err := resp.Body.Close(); err != nil {
		log.Warnf("[health] %s: failed to close probe response body: %v", srv.Name, err)
	}
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
		log.Warnf("[health] %s is down", id)
	}
	t.status[id] = false
}

// GetStatus returns a copy of the current health status map.
func (t *HealthTracker) GetStatus() map[string]bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	out := make(map[string]bool, len(t.status))
	maps.Copy(out, t.status)
	return out
}
