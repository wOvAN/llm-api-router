package config

import (
	"context"
	"io"
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
	mu        sync.RWMutex
	status    map[string]bool      // true = healthy
	throttled map[string]time.Time // a 429 answer's Retry-After window: do not probe inside it
	caps      *CapabilitiesCache   // the probe's /v1/models listing lands here
	store     *Store
	interval  time.Duration
	stopCh    chan struct{}
}

// NewHealthTracker creates a tracker with the given check interval. caps (nil
// allowed) is the capability cache the probe's /v1/models body feeds, so a
// server's listing is refreshed by the probe that already happens instead of by
// a second walk to the same URL.
func NewHealthTracker(store *Store, interval time.Duration, caps *CapabilitiesCache) *HealthTracker {
	return &HealthTracker{
		status:    make(map[string]bool),
		throttled: make(map[string]time.Time),
		caps:      caps,
		store:     store,
		interval:  interval,
		stopCh:    make(chan struct{}),
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
		// A server the tracker has never probed starts healthy (the historical
		// behavior: only a probe failure marks one down), and a healthy server is
		// not re-probed — only the ones that are down are watched for recovery.
		// (Start() marks every configured server healthy once, which is what
		// gives a server added later through the admin API the same start.)
		if t.IsHealthy(id) {
			continue
		}
		t.mu.RLock()
		throttled := time.Now().Before(t.throttled[id])
		t.mu.RUnlock()
		if throttled {
			// The backend asked not to be probed inside this window. Probing it
			// every tick is how a health check becomes the reason it stays 429.
			log.Debugf("[health] %s: inside its Retry-After window, probe skipped", srv.Name)
			continue
		}
		if t.checkServer(srv) {
			log.Infof("[health] %s is back", srv.Name)
			t.MarkHealthy(id)
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
	transport, err := proxy.TransportFor(srv.ProxyURL())
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
	defer func() {
		if err := resp.Body.Close(); err != nil {
			log.Warnf("[health] %s: failed to close probe response body: %v", srv.Name, err)
		}
	}()

	// An HTTP answer is not an outage. 401/403/429 say the backend is up and
	// refusing us — a missing/expired api_key, a throttled probe endpoint — and
	// the rate limiter reports that with the status as its reason, where the GUI
	// can show it. Marking it down instead leaves a red "down" server that has
	// been answering every probe all along.
	switch resp.StatusCode {
	case http.StatusOK:
		// The probe has the model listing in hand; the capability cache walks to
		// this same URL for the same payload, so it gets the answer here — and a
		// server that just came back has just re-discovered its models.
		if t.caps != nil {
			if body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20)); err == nil {
				t.caps.Observe(srv.ID, body)
			}
		}
		return true
	case http.StatusTooManyRequests:
		// The backend is throttling the probe. Obey it: probing on every tick is
		// how a health check becomes the reason the server stays 429.
		if d := proxy.ParseRetryAfter(resp.Header.Get("Retry-After")); d > 0 {
			t.mu.Lock()
			t.throttled[srv.ID] = time.Now().Add(d)
			t.mu.Unlock()
		}
		log.Warnf("[health] %s is up but answers %s to the probe — check its api_key",
			srv.Name, http.StatusText(resp.StatusCode))
		return true
	case http.StatusUnauthorized, http.StatusForbidden:
		log.Warnf("[health] %s is up but answers %s to the probe — check its api_key",
			srv.Name, http.StatusText(resp.StatusCode))
		return true
	}
	return false
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
	defer t.mu.Unlock()
	t.status[id] = true
	delete(t.throttled, id) // a success ends the throttle conversation
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
