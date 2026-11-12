package router

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// X-Router-* are the client's instructions to US. A backend has no idea what a
// conversion edge, a jev bypass, a retry budget or a heartbeat interval named in
// one of them means, and on a converted attempt the upstream request is a clone
// of the client's — so the namespace is deleted before routing and never rides
// upstream.
func TestHandleClientRouterHeadersDoNotRideUpstream(t *testing.T) {
	var seen http.Header
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if answerProbe(w, req) {
			return
		}
		seen = req.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`))
	}))
	defer backend.Close()

	r, store, _, _ := newTestRouter(t)
	addTestRoute(t, store, "gpt-4", backend.URL)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("X-Router-Retries", "3")
	req.Header.Set("X-Router-KeepAlive", "5")
	req.Header.Set("X-Router-Jev", "off")
	req.Header.Set("X-Router-Convert", "off")
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if got := w.Result().StatusCode; got != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", got, w.Body.String())
	}
	for _, h := range []string{"X-Router-Retries", "X-Router-KeepAlive", "X-Router-Jev", "X-Router-Convert"} {
		if v := seen.Get(h); v != "" {
			t.Errorf("%s reached the backend as %q, want it stripped", h, v)
		}
	}
}

// A client-set retry budget is a request for patience, not a licence to spend
// days: an unbounded number in front of a backend that accepts the connection
// and drops it is ~1s of backoff per retry, forever.
func TestClientRetriesOverrideIsClamped(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want int
		ok   bool
	}{
		{"5", 5, true},
		{"0", 0, true},
		{"100", maxClientRetries, true},
		{"1000000", maxClientRetries, true},
		{"-3", 0, false},
		{"abc", 0, false},
		{"", 0, false},
	} {
		got, ok := clientRetries(tc.in)
		if ok != tc.ok || (ok && got != tc.want) {
			t.Errorf("clientRetries(%q) = %d,%v, want %d,%v", tc.in, got, ok, tc.want, tc.ok)
		}
	}
}

// The inbound limiter refuses to read X-Forwarded-For for the same reason the
// metric must not: the header is the client's word for where it came from, so
// honouring it lets any client sign its traffic as anyone else's, and the recent
// request list is the operator's view of who is hammering the box.
func TestHandleClientIPIgnoresForgedForwardedHeaders(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if answerProbe(w, req) {
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`))
	}))
	defer backend.Close()

	for _, name := range []string{"X-Forwarded-For", "X-Real-IP"} {
		r, store, ms, _ := newTestRouter(t)
		addTestRoute(t, store, "gpt-4", backend.URL)

		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
			strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`))
		req.Header.Set(name, "203.0.113.9, 10.0.0.1")
		w := httptest.NewRecorder()
		r.Handle(w, req)
		if got := w.Result().StatusCode; got != http.StatusOK {
			t.Fatalf("%s: status = %d, want 200: %s", name, got, w.Body.String())
		}
		recent := ms.Recent()
		if len(recent) == 0 {
			t.Fatalf("%s: no metric recorded", name)
		}
		if ip := recent[0].ClientIP; ip != "192.0.2.1" { // httptest's peer address
			t.Errorf("%s: ClientIP = %q, want the peer address the connection arrived on", name, ip)
		}
	}
}
