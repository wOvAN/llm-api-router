package router

import (
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"llm-api-router/config"
	"llm-api-router/domain"
	"llm-api-router/metrics"
	"llm-api-router/trace"
)

// A port that listened and stopped listening answers connect with RST, so the
// dial fails with ECONNREFUSED — the shape of "this backend is not running".
// (A port that never listened black-holes on this host, which would test the
// dial-timeout path instead.)
func refusedURL(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen: %v", err)
	}
	url := "http://" + ln.Addr().String()
	if err := ln.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	return url
}

func newHealthRouter(t *testing.T) (*Router, *config.Store, *config.HealthTracker) {
	t.Helper()
	store, err := config.NewStore(filepath.Join(t.TempDir(), "config.json"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	health := config.NewHealthTracker(store, time.Hour)
	return New(store, metrics.New(100), health, nil, nil, trace.New(100, domain.TraceSettings{})), store, health
}

func chatOK(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"id":"c1","model":"m","choices":[{"index":0,` +
		`"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
}

// Nothing is listening: the same server can not recover inside one request, so
// retrying it is pure latency in front of a fallback that answers. One attempt,
// server marked unhealthy, next attempt tried.
func TestHandleConnectionRefusedFailsOverAtOnce(t *testing.T) {
	live := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		chatOK(w)
	}))
	defer live.Close()

	r, store, health := newHealthRouter(t)
	if err := store.AddServer(&domain.Server{ID: "dead", Name: "dead", URL: refusedURL(t),
		APITypes: []domain.APIType{domain.APITypeOpenAI}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddServer(&domain.Server{ID: "live", Name: "live", URL: live.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	// num_retries unset: the generous default (10) applies, which is what the
	// fast-fail must override for a refused connection.
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"m"},
		ServerID: "dead", TargetModel: "m", Enabled: true,
		Fallbacks: []domain.FallbackEntry{{ServerID: "live"}}}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	health.Start()
	defer health.Stop()

	start := time.Now()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"m","messages":[]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if res := w.Result(); res.StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", res.StatusCode, w.Body.String())
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Fatalf("a refused backend burned %v of same-server retries; the fallback "+
			"should be tried after one attempt", elapsed)
	}
	if health.IsHealthy("dead") {
		t.Fatal("a backend that refuses connections must be marked unhealthy")
	}
}

// Every proxied response carries a request id, and the recorded metric carries
// the same one: the number a client quotes in a bug report, available without
// arming the trace window. A client that already has an id is echoed, so its own
// logs and the router's agree.
func TestHandleRequestIDHeaderAndMetric(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		chatOK(w)
	}))
	defer backend.Close()

	r, store, ms, _ := newTestRouter(t)
	addTestRoute(t, store, "m", backend.URL)

	run := func(clientID string) (string, string) {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
			strings.NewReader(`{"model":"m","messages":[]}`))
		if clientID != "" {
			req.Header.Set("X-Request-ID", clientID)
		}
		w := httptest.NewRecorder()
		r.Handle(w, req)
		if res := w.Result(); res.StatusCode != http.StatusOK {
			t.Fatalf("status %d: %s", res.StatusCode, w.Body.String())
		}
		recent := ms.Recent()
		return w.Result().Header.Get("X-Router-Request-Id"), recent[len(recent)-1].RequestID
	}

	generated, inMetric := run("")
	if len(generated) < 16 {
		t.Fatalf("X-Router-Request-Id = %q, want a generated id", generated)
	}
	if inMetric != generated {
		t.Fatalf("the metric recorded %q, the client was told %q", inMetric, generated)
	}
	echoed, echoedMetric := run("req-42")
	if echoed != "req-42" || echoedMetric != "req-42" {
		t.Fatalf("the client's own id must be reused: header=%q metric=%q", echoed, echoedMetric)
	}
}
