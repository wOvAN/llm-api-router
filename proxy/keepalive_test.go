package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// syncWriter is a mutex-guarded http.ResponseWriter for tests where the
// heartbeat timer writes from a separate goroutine while the relay loop writes
// from its own.
type syncWriter struct {
	mu sync.Mutex
	b  strings.Builder
}

func (s *syncWriter) Header() http.Header { return make(http.Header) }
func (s *syncWriter) WriteHeader(int)     {}
func (s *syncWriter) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}
func (s *syncWriter) Flush() {}
func (s *syncWriter) content() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

func TestKeepAliveWriterPingsAfterIdle(t *testing.T) {
	w := &syncWriter{}
	k := newKeepAliveWriter(w, []byte(": keep-alive\n\n"), 20*time.Millisecond)
	k.Start()
	defer k.Stop()

	if _, err := k.Write([]byte("data: {\"a\":1}\n\n")); err != nil {
		t.Fatalf("Write: %v", err)
	}
	time.Sleep(80 * time.Millisecond)
	if !strings.Contains(w.content(), ": keep-alive") {
		t.Errorf("expected heartbeat after idle, got %q", w.content())
	}
}

func TestKeepAliveWriterNoPingBeforeStart(t *testing.T) {
	w := &syncWriter{}
	k := newKeepAliveWriter(w, []byte(": keep-alive\n\n"), 10*time.Millisecond)
	defer k.Stop()

	if _, err := k.Write([]byte("data: {\"a\":1}\n\n")); err != nil {
		t.Fatalf("Write: %v", err)
	}
	time.Sleep(50 * time.Millisecond)
	if strings.Contains(w.content(), "keep-alive") {
		t.Error("heartbeat must not fire before Start")
	}
}

func TestKeepAliveWriterWriteResetsIdle(t *testing.T) {
	w := &syncWriter{}
	k := newKeepAliveWriter(w, []byte(": keep-alive\n\n"), 40*time.Millisecond)
	k.Start()
	defer k.Stop()

	// Writes every 30ms — well below the 40ms idle — must keep pushing the
	// heartbeat back.
	for i := 0; i < 3; i++ {
		if _, err := k.Write([]byte("data: {\"a\":1}\n\n")); err != nil {
			t.Fatalf("Write: %v", err)
		}
		time.Sleep(30 * time.Millisecond)
	}
	if strings.Contains(w.content(), "keep-alive") {
		t.Fatalf("heartbeat fired despite steady writes: %q", w.content())
	}
	time.Sleep(80 * time.Millisecond)
	if !strings.Contains(w.content(), "keep-alive") {
		t.Errorf("heartbeat should fire after the last write: %q", w.content())
	}
}

func TestKeepAliveWriterStopPreventsPings(t *testing.T) {
	w := &syncWriter{}
	k := newKeepAliveWriter(w, []byte(": keep-alive\n\n"), 20*time.Millisecond)
	k.Start()

	if _, err := k.Write([]byte("data: {\"a\":1}\n\n")); err != nil {
		t.Fatalf("Write: %v", err)
	}
	time.Sleep(60 * time.Millisecond)
	before := w.content()
	if !strings.Contains(before, "keep-alive") {
		t.Fatalf("expected heartbeat before Stop: %q", before)
	}

	k.Stop()
	time.Sleep(60 * time.Millisecond)
	if w.content() != before {
		t.Errorf("heartbeat fired after Stop: %q", w.content())
	}
}

// TestStreamProxyKeepAliveDuringBackendPause verifies the full relay injects
// heartbeats into the client stream while the backend pauses between SSE
// frames for longer than KeepAliveIdle.
func TestStreamProxyKeepAliveDuringBackendPause(t *testing.T) {
	old := KeepAliveIdle
	KeepAliveIdle = 30 * time.Millisecond
	defer func() { KeepAliveIdle = old }()

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"first\"}}]}\n\n"))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		time.Sleep(100 * time.Millisecond) // pause longer than the heartbeat idle
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"second\"}}]}\n\n"))
	}))
	defer backend.Close()

	w := &syncWriter{}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	if _, err := StreamProxy(req.Context(), backend.URL, "key", req, w, "gpt-4", "gpt-4", nil, false, []byte(": keep-alive\n\n")); err != nil {
		t.Fatalf("StreamProxy: %v", err)
	}

	got := w.content()
	first := strings.Index(got, "first")
	ping := strings.Index(got, ": keep-alive")
	second := strings.Index(got, "second")
	if first < 0 || second < 0 {
		t.Fatalf("stream incomplete:\n%s", got)
	}
	if ping < 0 {
		t.Fatalf("no heartbeat injected during backend pause:\n%s", got)
	}
	if !(first < ping && ping < second) {
		t.Errorf("expected heartbeat between the two frames:\n%s", got)
	}
}

// TestStreamProxyKeepAliveSkipsNonSSE verifies heartbeats are never injected
// into non-streaming (JSON) responses.
func TestStreamProxyKeepAliveSkipsNonSSE(t *testing.T) {
	old := KeepAliveIdle
	KeepAliveIdle = 20 * time.Millisecond
	defer func() { KeepAliveIdle = old }()

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{"model":"gpt-4"}`))
		time.Sleep(60 * time.Millisecond)
	}))
	defer backend.Close()

	w := &syncWriter{}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	if _, err := StreamProxy(req.Context(), backend.URL, "key", req, w, "gpt-4", "gpt-4", nil, false, []byte(": keep-alive\n\n")); err != nil {
		t.Fatalf("StreamProxy: %v", err)
	}
	if strings.Contains(w.content(), "keep-alive") {
		t.Errorf("non-SSE response must not receive heartbeats: %q", w.content())
	}
}

// TestKeepAliveWriterNoPingWithoutPingBytes guards the passthrough path: when
// StreamProxy receives a nil ping, no heartbeat machinery is created at all.
func TestStreamProxyNoKeepAliveWhenPingNil(t *testing.T) {
	old := KeepAliveIdle
	KeepAliveIdle = 20 * time.Millisecond
	defer func() { KeepAliveIdle = old }()

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n"))
		time.Sleep(60 * time.Millisecond)
	}))
	defer backend.Close()

	w := &syncWriter{}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	if _, err := StreamProxy(req.Context(), backend.URL, "key", req, w, "gpt-4", "gpt-4", nil, false, nil); err != nil {
		t.Fatalf("StreamProxy: %v", err)
	}
	if strings.Contains(w.content(), "keep-alive") {
		t.Errorf("heartbeat must not be sent when ping is nil: %q", w.content())
	}
}
