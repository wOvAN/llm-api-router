package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// A backend behind a gateway that demands a fixed header (a tenant id, a
// required user-agent, an internal routing tag) can not be served by the client's
// own headers: the operator's word wins, so a server's static headers are stamped
// on the upstream request, overwriting whatever the client sent under the same
// name.
func TestStreamProxyAppliesStaticHeaders(t *testing.T) {
	var seen http.Header
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		seen = req.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[]}`))
	}))
	defer backend.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	req.Header.Set("X-Scope-Id", "whatever-the-client-claims")
	w := httptest.NewRecorder()

	rh := &RouterHeaders{Headers: map[string]string{"X-Scope-Id": "tenant-7", "X-Gateway-Tag": "llm"}}
	if _, err := StreamProxy(req.Context(), backend.URL, "", req, w, "gpt-4", "gpt-4", rh, false, nil); err != nil {
		t.Fatalf("StreamProxy: %v", err)
	}
	if got := seen.Get("X-Scope-Id"); got != "tenant-7" {
		t.Errorf("X-Scope-Id = %q, want the configured value to win", got)
	}
	if got := seen.Get("X-Gateway-Tag"); got != "llm" {
		t.Errorf("X-Gateway-Tag = %q, want the configured header on the upstream request", got)
	}
}
