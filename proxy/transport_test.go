package proxy

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// newRecordingProxy returns an httptest server that answers every request
// itself. A plain HTTP proxy receives the absolute-form request target from
// Go's client, so the recorded URL proves the request was routed through it.
func newRecordingProxy(t *testing.T, seen *string, hits *atomic.Int64) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		*seen = r.URL.String()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"cmpl-1","model":"m"}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestTransportForDirectAndCached(t *testing.T) {
	tr, err := TransportFor("")
	if err != nil {
		t.Fatalf("TransportFor(\"\"): %v", err)
	}
	if tr != http.RoundTripper(upstreamTransport) {
		t.Error("empty proxy must return the shared direct transport")
	}

	first, err := TransportFor("http://proxy.test:3128")
	if err != nil {
		t.Fatalf("TransportFor: %v", err)
	}
	second, err := TransportFor("http://proxy.test:3128")
	if err != nil {
		t.Fatalf("TransportFor: %v", err)
	}
	if first != second {
		t.Error("same proxy URL must reuse one transport (connection pooling)")
	}

	schemeless, err := TransportFor("proxy.test:3128")
	if err != nil {
		t.Fatalf("TransportFor(schemeless): %v", err)
	}
	if schemeless != first {
		t.Error("scheme-less proxy URL should normalize to http:// and share the transport")
	}

	if _, err := TransportFor("://bad proxy"); err == nil {
		t.Error("expected error for invalid proxy URL")
	}
}

func TestStreamProxyUsesPerServerProxy(t *testing.T) {
	var seen string
	var hits atomic.Int64
	p := newRecordingProxy(t, &seen, &hits)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"m"}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	pm, err := StreamProxy(context.Background(), "http://origin.invalid/v1", "key", req, w,
		"m", "m", &RouterHeaders{ServerID: "s1", Proxy: p.URL}, false, nil)
	if err != nil {
		t.Fatalf("StreamProxy: %v", err)
	}
	if hits.Load() != 1 {
		t.Fatalf("proxy hits = %d, want 1 (request bypassed the proxy)", hits.Load())
	}
	if seen != "http://origin.invalid/v1/chat/completions" {
		t.Errorf("proxy saw %q, want the absolute origin URL", seen)
	}
	if pm.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200", pm.StatusCode)
	}
}

func TestStreamProxyInvalidProxyURLFailsBeforeHeaders(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"m"}`))
	w := httptest.NewRecorder()

	_, err := StreamProxy(context.Background(), "http://origin.invalid/v1", "", req, w,
		"m", "m", &RouterHeaders{Proxy: "://bad proxy"}, false, nil)
	if err == nil {
		t.Fatal("expected error for invalid proxy URL")
	}
	if w.Body.Len() != 0 {
		t.Errorf("client must see no bytes on pre-response failure, got %q", w.Body.String())
	}
}
