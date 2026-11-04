package proxy

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// A backend that answers, streams a piece, and then goes silent is the one
// failure none of the existing bounds cover: ResponseHeaderTimeout bounded a header
// wait that already succeeded, and the drain bound applies only once the client is
// gone. Without a bound the stream stays open forever — the client is kept alive by
// our own heartbeats, so it never notices either, and the router holds a connection
// and a goroutine on a generation nobody is producing.
func TestStreamProxyStreamIdleTimeoutEndsAWedgedStream(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		flusher.Flush()
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"partial answer\"}}]}\n\n")
		flusher.Flush()
		<-req.Context().Done() // wedge: never finish, never close
	}))
	defer backend.Close()

	original := StreamIdleTimeout
	t.Cleanup(func() { StreamIdleTimeout = original })
	StreamIdleTimeout = 200 * time.Millisecond

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	w := httptest.NewRecorder()
	start := time.Now()
	_, err := StreamProxy(req.Context(), backend.URL, "", req, w, "gpt-4", "gpt-4", nil, false, nil)
	elapsed := time.Since(start)

	var mse *MidStreamError
	if !errors.As(err, &mse) {
		t.Fatalf("err = %v, want a *MidStreamError (headers are already sent, so no fallback is possible)", err)
	}
	if elapsed > 10*time.Second {
		t.Errorf("the wedged stream took %v to end, want the idle bound to cut it", elapsed)
	}
	body := w.Body.String()
	if !strings.Contains(body, "partial answer") {
		t.Errorf("the client lost the tokens it was given: %s", body)
	}
	if !strings.Contains(body, "error") {
		t.Errorf("the client got a closed stream with no error frame: %s", body)
	}
}

// The bound is an idle *gap*, not a total request timeout: a JSON answer that
// takes longer than the gap to compute is a normal slow generation, and cutting it
// would turn a thinking model into an error.
func TestStreamProxyStreamIdleTimeoutIgnoresSlowJSON(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(400 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`))
	}))
	defer backend.Close()

	original := StreamIdleTimeout
	t.Cleanup(func() { StreamIdleTimeout = original })
	StreamIdleTimeout = 150 * time.Millisecond

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	w := httptest.NewRecorder()
	if _, err := StreamProxy(req.Context(), backend.URL, "", req, w, "gpt-4", "gpt-4", nil, false, nil); err != nil {
		t.Fatalf("StreamProxy: %v (a slow JSON answer must not be cut by a stream idle bound)", err)
	}
	if !strings.Contains(w.Body.String(), "finish_reason") {
		t.Errorf("body = %s, want the backend's answer relayed", w.Body.String())
	}
}

// The watchdog bounds a stream with a client watching it. Once the client is
// gone, the drain owns that connection and has its own, far longer, bound: a
// watchdog that keeps firing there cancels the very generation the drain exists
// to protect, which is the opposite of why the drain was built.
func TestStreamProxyStreamIdleTimeoutStandsDownForDrain(t *testing.T) {
	oldIdle, oldDrain := StreamIdleTimeout, UpstreamDrainTimeout
	t.Cleanup(func() { StreamIdleTimeout, UpstreamDrainTimeout = oldIdle, oldDrain })
	StreamIdleTimeout = 100 * time.Millisecond
	UpstreamDrainTimeout = 5 * time.Second

	completed := make(chan struct{})
	var earlyAbort atomic.Bool
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer close(completed)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"1\"}}]}\n\n"))
		w.(http.Flusher).Flush()
		for range 3 {
			select {
			case <-req.Context().Done():
				earlyAbort.Store(true)
				return
			case <-time.After(150 * time.Millisecond):
			}
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n"))
			w.(http.Flusher).Flush()
		}
	}))
	defer backend.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	ctx, cancel := context.WithCancel(req.Context())
	defer cancel()
	req = req.WithContext(ctx)

	_, err := StreamProxy(req.Context(), backend.URL, "", req,
		&failWriter{w: httptest.NewRecorder(), onWrite: cancel}, "gpt-4", "gpt-4", nil, false, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	select {
	case <-completed:
	case <-time.After(3 * time.Second):
		t.Fatal("the drain did not run to completion")
	}
	if earlyAbort.Load() {
		t.Error("the stream idle watchdog fired during the post-disconnect drain and cut the generation")
	}
}
