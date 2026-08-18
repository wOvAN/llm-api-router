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
	k := newSilenceWriter(w, 20*time.Millisecond, []byte(": keep-alive\n\n"))
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
	k := newSilenceWriter(w, 10*time.Millisecond, []byte(": keep-alive\n\n"))
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
	k := newSilenceWriter(w, 40*time.Millisecond, []byte(": keep-alive\n\n"))
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
	k := newSilenceWriter(w, 20*time.Millisecond, []byte(": keep-alive\n\n"))
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
	if first >= ping || ping >= second {
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

// TestStreamProxyKeepAliveOverrideHeader verifies X-Router-KeepAlive tunes the
// heartbeat cadence per request: absent falls back to the global idle, "0"
// disables heartbeats for that request.
func TestStreamProxyKeepAliveOverrideHeader(t *testing.T) {
	old := KeepAliveIdle
	KeepAliveIdle = 40 * time.Millisecond
	defer func() { KeepAliveIdle = old }()

	// Backend pauses 150ms between frames — longer than the 40ms global idle,
	// so a heartbeat fires unless the request disables it.
	newBackend := func() *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(200)
			_, _ = w.Write([]byte("data: {\"a\":1}\n\n"))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			time.Sleep(150 * time.Millisecond)
			_, _ = w.Write([]byte("data: {\"a\":2}\n\n"))
		}))
	}

	// No header: global 40ms idle fires a heartbeat during the pause.
	b1 := newBackend()
	w1 := &syncWriter{}
	req1 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	if _, err := StreamProxy(req1.Context(), b1.URL, "key", req1, w1, "gpt-4", "gpt-4", nil, false, []byte(": keep-alive\n\n")); err != nil {
		b1.Close()
		t.Fatalf("StreamProxy: %v", err)
	}
	b1.Close()
	if !strings.Contains(w1.content(), "keep-alive") {
		t.Errorf("no header: expected global heartbeat, got %q", w1.content())
	}

	// Header "0": heartbeats disabled, none during the same pause.
	b2 := newBackend()
	w2 := &syncWriter{}
	req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	req2.Header.Set("X-Router-KeepAlive", "0")
	if _, err := StreamProxy(req2.Context(), b2.URL, "key", req2, w2, "gpt-4", "gpt-4", nil, false, []byte(": keep-alive\n\n")); err != nil {
		b2.Close()
		t.Fatalf("StreamProxy: %v", err)
	}
	b2.Close()
	if strings.Contains(w2.content(), "keep-alive") {
		t.Errorf("X-Router-KeepAlive: 0 should disable heartbeats, got %q", w2.content())
	}
}

// TestStreamProxyKeepAliveServerOverride verifies the server-configured
// keep-alive interval (rh.KeepAliveIdle) beats the global default, and the
// per-request X-Router-KeepAlive header beats the server value.
func TestStreamProxyKeepAliveServerOverride(t *testing.T) {
	old := KeepAliveIdle
	KeepAliveIdle = 500 * time.Millisecond // global is slow: won't fire in the pause
	defer func() { KeepAliveIdle = old }()

	newBackend := func() *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(200)
			_, _ = w.Write([]byte("data: {\"a\":1}\n\n"))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			time.Sleep(150 * time.Millisecond)
			_, _ = w.Write([]byte("data: {\"a\":2}\n\n"))
		}))
	}

	rh := &RouterHeaders{KeepAliveIdle: 40 * time.Millisecond} // server override: fast

	// Server override (40ms) fires during the 150ms pause, even though the
	// global (500ms) would not.
	b1 := newBackend()
	w1 := &syncWriter{}
	req1 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	if _, err := StreamProxy(req1.Context(), b1.URL, "key", req1, w1, "gpt-4", "gpt-4", rh, false, []byte(": keep-alive\n\n")); err != nil {
		b1.Close()
		t.Fatalf("StreamProxy: %v", err)
	}
	b1.Close()
	if !strings.Contains(w1.content(), "keep-alive") {
		t.Errorf("server override: expected heartbeat at 40ms, got %q", w1.content())
	}

	// Header "0" beats the server value: no heartbeat.
	b2 := newBackend()
	w2 := &syncWriter{}
	req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	req2.Header.Set("X-Router-KeepAlive", "0")
	if _, err := StreamProxy(req2.Context(), b2.URL, "key", req2, w2, "gpt-4", "gpt-4", rh, false, []byte(": keep-alive\n\n")); err != nil {
		b2.Close()
		t.Fatalf("StreamProxy: %v", err)
	}
	b2.Close()
	if strings.Contains(w2.content(), "keep-alive") {
		t.Errorf("header 0 should beat the server override, got %q", w2.content())
	}
}
