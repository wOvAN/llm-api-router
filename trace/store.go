package trace

import (
	"sync"
	"time"

	"llm-api-router/domain"
)

// Entry is one traced request: the bodies as they went on the wire (binaries
// replaced by stubs) plus the derived views of them.
type Entry struct {
	ID          string    `json:"id"`
	Timestamp   time.Time `json:"timestamp"`
	Method      string    `json:"method"`
	Path        string    `json:"path"`
	Model       string    `json:"model"`
	TargetModel string    `json:"target_model,omitempty"`
	ServerID    string    `json:"server_id,omitempty"`
	Class       string    `json:"class,omitempty"`
	Features    []string  `json:"features,omitempty"`
	StatusCode  int       `json:"status_code"`
	LatencyMs   int64     `json:"latency_ms"`
	TTFBMs      int64     `json:"ttfb_ms"`
	JevChoice   string    `json:"jev_choice,omitempty"`

	Request  string `json:"request,omitempty"`
	Response string `json:"response,omitempty"`
	Digest   Digest `json:"digest"`

	RequestTruncated  bool `json:"request_truncated,omitempty"`
	ResponseTruncated bool `json:"response_truncated,omitempty"`
}

// Window is the live capture state: what is being traced right now.
type Window struct {
	Enabled   bool       `json:"enabled"`
	ExpiresAt *time.Time `json:"expires_at,omitempty"`
	Models    []string   `json:"models"`
	Servers   []string   `json:"servers"`
}

// Store is the trace ring buffer plus the live window. The ring overwrites by
// index like metrics.Store's recent requests; the window expires lazily, on the
// next read, so an idle router has no timer.
type Store struct {
	mu      sync.Mutex
	entries []Entry
	ringIdx int
	count   int
	max     int

	enabled   bool
	expiresAt time.Time
	models    []string
	servers   []string

	// now is the clock seam: tests advance it instead of sleeping.
	now func() time.Time
}

// New creates an empty ring and arms the window from the config defaults.
func New(max int, st domain.TraceSettings) *Store {
	s := &Store{
		entries: make([]Entry, max),
		max:     max,
		models:  st.Models,
		servers: st.Servers,
		now:     time.Now,
	}
	if st.Enabled {
		s.enabled = true
		if st.Seconds > 0 {
			s.expiresAt = s.now().Add(time.Duration(st.Seconds) * time.Second)
		}
	}
	return s
}

// Start arms the window for the next `seconds`, 0 = until Stop.
func (s *Store) Start(seconds int, models, servers []string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.enabled = true
	s.models, s.servers = models, servers
	if seconds > 0 {
		s.expiresAt = s.now().Add(time.Duration(seconds) * time.Second)
	} else {
		s.expiresAt = time.Time{}
	}
}

// Stop disarms the window; existing entries stay readable until Reset.
func (s *Store) Stop() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.enabled = false
	s.expiresAt = time.Time{}
}

// Window reports the live state, expiring a finished window as it reports it.
func (s *Store) Window() Window {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.expireLocked()
	w := Window{Enabled: s.enabled, Models: s.models, Servers: s.servers}
	if !s.expiresAt.IsZero() {
		at := s.expiresAt
		w.ExpiresAt = &at
	}
	return w
}

// Armed reports whether a request for `model` on `serverID` is traced. An empty
// filter matches everything.
func (s *Store) Armed(model, serverID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.expireLocked() {
		return false
	}
	return matches(s.models, model) && matches(s.servers, serverID)
}

// expireLocked disarms a finished window; it reports whether tracing is off.
// Callers hold s.mu.
func (s *Store) expireLocked() bool {
	if !s.enabled {
		return true
	}
	if !s.expiresAt.IsZero() && !s.now().Before(s.expiresAt) {
		s.enabled = false
		s.expiresAt = time.Time{}
		return true
	}
	return false
}

func matches(filter []string, v string) bool {
	if len(filter) == 0 {
		return true
	}
	for _, f := range filter {
		if f == v {
			return true
		}
	}
	return false
}

// Add writes an entry into the ring.
func (s *Store) Add(e Entry) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries[s.ringIdx] = e
	s.ringIdx = (s.ringIdx + 1) % s.max
	if s.count < s.max {
		s.count++
	}
}

// List returns the ring in chronological order.
func (s *Store) List() []Entry {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Entry, s.count)
	if s.count == 0 {
		return out
	}
	if s.count < s.max {
		copy(out, s.entries[:s.count])
	} else {
		n := copy(out, s.entries[s.ringIdx:])
		copy(out[n:], s.entries[:s.ringIdx])
	}
	return out
}

// Get returns one entry by ID (newest wins on an id collision).
func (s *Store) Get(id string) (Entry, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := s.count - 1; i >= 0; i-- {
		if s.entries[i].ID == id {
			return s.entries[i], true
		}
	}
	return Entry{}, false
}

// Len reports how many entries the ring holds.
func (s *Store) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.count
}

// Reset clears the ring; the live window is untouched (clearing the ring is not
// switching collection off).
func (s *Store) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries = make([]Entry, s.max)
	s.ringIdx = 0
	s.count = 0
}
