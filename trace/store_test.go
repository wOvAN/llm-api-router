package trace

import (
	"testing"
	"time"

	"llm-api-router/domain"
)

func TestStoreRingOverwritesOldest(t *testing.T) {
	s := New(3, domain.TraceSettings{})
	for i := range 4 {
		s.Add(Entry{ID: string(rune('a' + i))})
	}
	if s.Len() != 3 {
		t.Fatalf("len = %d, want 3", s.Len())
	}
	list := s.List()
	if list[0].ID != "b" || list[2].ID != "d" {
		t.Errorf("ring order = %q..%q, want b..d (chronological, oldest overwritten)", list[0].ID, list[2].ID)
	}
	if _, ok := s.Get("a"); ok {
		t.Error("overwritten entry still readable")
	}
	if e, ok := s.Get("c"); !ok || e.ID != "c" {
		t.Error("Get by id must find a live entry")
	}
}

func TestStoreStartArmsWindowAndFilters(t *testing.T) {
	s := New(10, domain.TraceSettings{})
	if s.Armed("gpt", "s1") {
		t.Fatal("fresh store must not be armed")
	}
	s.Start(0, []string{"gpt"}, []string{"s1"})
	if !s.Armed("gpt", "s1") {
		t.Error("matching model+server must be armed")
	}
	if s.Armed("other", "s1") {
		t.Error("model filter must exclude")
	}
	if s.Armed("gpt", "s2") {
		t.Error("server filter must exclude")
	}
	s.Start(0, nil, nil)
	if !s.Armed("anything", "anything") {
		t.Error("empty filters must match everything")
	}
	s.Stop()
	if s.Armed("anything", "anything") {
		t.Error("Stop must disarm")
	}
}

func TestStoreWindowExpiresLazily(t *testing.T) {
	now := time.Now()
	s := New(10, domain.TraceSettings{})
	s.now = func() time.Time { return now }
	s.Start(60, nil, nil)
	w := s.Window()
	if !w.Enabled || w.ExpiresAt == nil || !w.ExpiresAt.Equal(now.Add(time.Minute)) {
		t.Fatalf("window = %+v, want enabled, expiring in 60s", w)
	}
	now = now.Add(61 * time.Second)
	if s.Armed("m", "s") {
		t.Error("expired window must not arm")
	}
	if w := s.Window(); w.Enabled {
		t.Error("Window() must report the lazy expiry")
	}
}

func TestStoreNewArmsFromConfigDefaults(t *testing.T) {
	s := New(10, domain.TraceSettings{Enabled: true, Seconds: 300, Models: []string{"m"}})
	if !s.Armed("m", "any") {
		t.Error("config defaults must arm the window at startup")
	}
	if s.Armed("other", "any") {
		t.Error("config default filter must apply")
	}
}

func TestStoreResetKeepsWindow(t *testing.T) {
	s := New(10, domain.TraceSettings{})
	s.Start(0, nil, nil)
	s.Add(Entry{ID: "a"})
	s.Reset()
	if s.Len() != 0 {
		t.Fatal("reset must empty the ring")
	}
	if !s.Armed("m", "s") {
		t.Error("reset must not touch the live window")
	}
}
