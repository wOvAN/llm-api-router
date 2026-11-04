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
	alerter   Alerter              // edge events (down/back) leave the process here
	store     *Store
	interval  time.Duration
	timeout   time.Duration
	probePath string
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
		timeout:   healthProbeTimeout,
		probePath: healthProbePath,
		stopCh:    make(chan struct{}),
	}
}

// Defaults for the probe, overridable through settings.health.
const (
	healthProbePath    = "/v1/models"
	healthProbeTimeout = 5 * time.Second
)

// SetProbe tunes the probe's endpoint and per-request bound ("" / 0 keep the
// defaults). Installed separately from the constructor so a config reload can move
// it, and so a backend that does not serve /v1/models can be probed at the route
// it does: a probe against a path the backend lacks reads as an outage.
func (t *HealthTracker) SetProbe(path string, timeout time.Duration) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if path != "" {
		t.probePath = path
	}
	if timeout > 0 {
		t.timeout = timeout
	}
}

// ProbeConfigured reports the probe's current endpoint, per-request bound and
// tick interval (a test's window on the three knobs settings.health tunes).
func (t *HealthTracker) ProbeConfigured(path string, timeout, interval time.Duration) bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.probePath == path && t.timeout == timeout && t.interval == interval
}

// SetInterval moves the probe interval (0 keeps it). The loop reads it each tick,
// so a reload applies on the next one.
func (t *HealthTracker) SetInterval(interval time.Duration) {
	if interval <= 0 {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.interval = interval
}

// SetAlerter installs the out-of-band notice for health transitions (nil = none).
// Installed separately from the constructor so a config reload can move the
// webhook without rebuilding the tracker.
func (t *HealthTracker) SetAlerter(a Alerter) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.alerter = a
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
	for {
		// The interval is read every tick, so SetInterval applies without a
		// restart (a ticker built at start would freeze the startup value).
		t.mu.RLock()
		interval := t.interval
		t.mu.RUnlock()

		select {
		case <-t.stopCh:
			return
		case <-time.After(interval):
			t.checkAll()
		}
	}
}

func (t *HealthTracker) checkAll() {
	cfg := t.store.GetConfig()
	t.forgetRemoved(cfg)
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

// forgetRemoved drops the tracker's entries for servers the config no longer
// defines (the sweep runs on every tick, so a delete can not leave a ghost
// behind). Caller must not hold t.mu.
func (t *HealthTracker) forgetRemoved(cfg *domain.Config) {
	t.mu.Lock()
	defer t.mu.Unlock()
	for id := range t.status {
		if _, ok := cfg.Servers[id]; !ok {
			delete(t.status, id)
			delete(t.throttled, id)
		}
	}
}

// ModelsURL returns the server's /v1/models probe URL: the protocol-specific
// URL override for OpenAI, scheme defaulted to https, /v1 suffix dedup'd
// against the probe path.
func ModelsURL(srv *domain.Server) string {
	return probeURL(srv, healthProbePath)
}

// probeURL joins a server's base URL and a probe path, applying the /v1 dedup.
func probeURL(srv *domain.Server, path string) string {
	rawURL := srv.GetURLForAPIType(domain.APITypeOpenAI)
	if !strings.Contains(rawURL, "://") {
		rawURL = "https://" + rawURL
	}
	u := strings.TrimRight(rawURL, "/") + path
	return strings.ReplaceAll(u, "/v1/v1", "/v1")
}

// checkServer pings the server's /v1/models endpoint.
func (t *HealthTracker) checkServer(srv *domain.Server) bool {
	t.mu.RLock()
	timeout, probePath := t.timeout, t.probePath
	t.mu.RUnlock()

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, probeURL(srv, probePath), nil)
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
	// A server the tracker has never seen (added through the admin API after
	// Start()) has no status at all; it is not "back" from anything.
	healthy, known := t.status[id]
	wasDown := known && !healthy
	t.status[id] = true
	delete(t.throttled, id) // a success ends the throttle conversation
	alerter := t.alerter
	t.mu.Unlock()
	// Edge-triggered: the router calls this on every successful request, so
	// without the edge test a healthy backend alerts once per request.
	if wasDown && alerter != nil {
		alerter.Notify("server " + id + " is back")
	}
}

// MarkUnhealthy marks a server as unhealthy.
func (t *HealthTracker) MarkUnhealthy(id string) {
	t.mu.Lock()
	wasUp := t.status[id]
	t.status[id] = false
	alerter := t.alerter
	t.mu.Unlock()
	if wasUp {
		log.Warnf("[health] %s is down", id)
		if alerter != nil {
			alerter.Notify("server " + id + " is down")
		}
	}
}

// GetStatus returns a copy of the current health status map.
func (t *HealthTracker) GetStatus() map[string]bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	out := make(map[string]bool, len(t.status))
	maps.Copy(out, t.status)
	return out
}
