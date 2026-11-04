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
