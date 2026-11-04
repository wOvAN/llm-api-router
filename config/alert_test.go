package config

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

// recorder is a test Alerter that keeps the events it was given.
type recorder struct {
	mu     sync.Mutex
	events []string
}

func (r *recorder) Notify(event string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, event)
}

func (r *recorder) got() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]string, len(r.events))
	copy(out, r.events)
	return out
}

// Health alerts are edge-triggered: one event when a backend goes down, one when
// it comes back, nothing while it sits in either state. The router probes every
// tick, so an alert on state would turn one dead GPU into a webhook per 30s,
// which is how alerting gets muted on the first bad night.
func TestHealthTrackerAlertsOnTransitions(t *testing.T) {
	store := newHealthStore(t, "http://127.0.0.1:1")
	tt := NewHealthTracker(store, time.Hour, nil)
	rec := &recorder{}
	tt.SetAlerter(rec)

	tt.MarkHealthy("s1") // the baseline Start() installs: a known-healthy server
	tt.MarkUnhealthy("s1")
	tt.MarkUnhealthy("s1")
	tt.MarkHealthy("s1")
	tt.MarkHealthy("s1")

	want := []string{"server s1 is down", "server s1 is back"}
	got := rec.got()
	if len(got) != len(want) {
		t.Fatalf("events = %q, want %q (one per transition)", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("event %d = %q, want %q", i, got[i], want[i])
		}
	}
}

// A cooldown start is an event the operator cannot see anywhere else in time: the
// GUI shows it only while looking, and the router quietly stopped sending traffic
// to that backend. Re-entering an existing cooldown is not a new event.
func TestRateLimiterAlertsOnCooldownStart(t *testing.T) {
	rl := NewRateLimiter(5, 60*time.Second, time.Hour)
	rec := &recorder{}
	rl.SetAlerter(rec)

	if !rl.RecordFailure("s1", http.StatusTooManyRequests) {
		t.Fatal("fixture: the first 429 did not cool the server down")
	}
	rl.RecordFailure("s1", http.StatusTooManyRequests) // already cooling: no new event

	got := rec.got()
	if len(got) != 1 {
		t.Fatalf("events = %q, want exactly one (the cooldown start)", got)
	}
	if got[0] != "server s1 cooling down (status Too Many Requests)" {
		t.Errorf("event = %q, want the cooldown to name its reason", got[0])
	}
}

// The webhook itself: one JSON POST per event, and a repeated event inside the
// dedup window costs one POST.
func TestWebhookAlerterPostsEachEvent(t *testing.T) {
	var mu sync.Mutex
	var bodies []map[string]string
	done := make(chan struct{}, 4)
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		body, _ := io.ReadAll(req.Body)
		var payload map[string]string
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Errorf("webhook body is not JSON: %s", body)
		}
		mu.Lock()
		bodies = append(bodies, payload)
		mu.Unlock()
		done <- struct{}{}
	}))
	defer backend.Close()

	a := NewWebhookAlerter(backend.URL)
	a.Notify("server s1 is down")
	a.Notify("server s1 is down") // deduped
	a.Notify("server s1 is back")
	for range 2 {
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Fatal("the webhook never received an event")
		}
	}

	mu.Lock()
	defer mu.Unlock()
	if len(bodies) != 2 {
		t.Fatalf("webhook got %d POSTs, want 2 (the repeat is deduped): %v", len(bodies), bodies)
	}
	// Delivery is fire-and-forget (one goroutine per POST), so arrival order is
	// not the send order — assert the set.
	seen := map[string]bool{}
	for _, b := range bodies {
		seen[b["event"]] = true
		if b["at"] == "" {
			t.Error("the payload carries no timestamp")
		}
	}
	for _, want := range []string{"server s1 is down", "server s1 is back"} {
		if !seen[want] {
			t.Errorf("webhook never got %q: %v", want, bodies)
		}
	}
}

// No webhook configured means no alerts, and no nil dereference on the way.
func TestNewWebhookAlerterIsEmptyWhenUnset(t *testing.T) {
	if a := NewWebhookAlerter(""); a != nil {
		t.Errorf("NewWebhookAlerter(\"\") = %T, want nil", a)
	}
}
