package config

import (
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"llm-api-router/domain"
)

func newHealthStore(t *testing.T, url string) *Store {
	t.Helper()
	store, err := NewStore(filepath.Join(t.TempDir(), "config.json"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: url}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	return store
}

// A probe that answers 401/403/429 is a backend that is up and refusing us (a
// missing or expired api_key, a throttled /v1/models) — not an outage. Marking
// it down hides the real cause: the GUI prints "down", the router skips it, and
// nobody sees the key problem. Real traffic still fails on its 401, and the
// status-aware rate limiter cools it down with a reason that says so.
func TestHealthProbeAcceptsRefusalAsUp(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
		want   bool
	}{
		{"ok", http.StatusOK, true},
		{"unauthorized", http.StatusUnauthorized, true},
		{"forbidden", http.StatusForbidden, true},
		{"rate limited", http.StatusTooManyRequests, true},
		{"server error", http.StatusInternalServerError, false},
		{"not found", http.StatusNotFound, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.status)
			}))
			defer backend.Close()

			tt := NewHealthTracker(newHealthStore(t, backend.URL), time.Hour)
			tt.MarkUnhealthy("s1") // the probe only re-checks a server it thinks is down
			tt.checkAll()

			if got := tt.IsHealthy("s1"); got != tc.want {
				t.Errorf("probe answered %d: healthy = %v, want %v", tc.status, got, tc.want)
			}
		})
	}
}

// The other side of the same gate: a probe that gets no HTTP answer at all is an
// outage, and must stay down.
func TestHealthProbeMarksUnreachableServerDown(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen: %v", err)
	}
	url := "http://" + ln.Addr().String()
	if err := ln.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	tt := NewHealthTracker(newHealthStore(t, url), time.Hour)
	tt.MarkUnhealthy("s1")
	tt.checkAll()

	if tt.IsHealthy("s1") {
		t.Error("a server that refuses connections must stay unhealthy")
	}
}
