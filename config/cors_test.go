package config

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"llm-api-router/domain"
)

// A browser client (OpenWebUI, a custom dashboard) can read the answer but not
// the routing headers, because nothing exposes them cross-origin. The middleware
// is off until an operator allows origins, and the X-Router-* namespace is the
// point of allowing one.
func TestCORSMiddleware(t *testing.T) {
	c := NewCORS(domain.COSSettings{Enabled: true, Origins: []string{"https://ui.example"}})
	h := c.Middleware(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Router-Server", "s1")
		_, _ = w.Write([]byte("served"))
	}))

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("Origin", "https://ui.example")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if got := w.Header().Get("Access-Control-Allow-Origin"); got != "https://ui.example" {
		t.Errorf("allow-origin = %q, want the allowed origin echoed", got)
	}
	exposed := w.Header().Get("Access-Control-Expose-Headers")
	for _, name := range []string{"X-Router-Server", "X-Router-Request-Id", "X-Router-Convert"} {
		if !strings.Contains(exposed, name) {
			t.Errorf("expose-headers = %q, want %s readable by a browser client", exposed, name)
		}
	}

	// An origin nobody allowed gets the answer with no CORS grant.
	other := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	other.Header.Set("Origin", "https://evil.example")
	w2 := httptest.NewRecorder()
	h.ServeHTTP(w2, other)
	if w2.Header().Get("Access-Control-Allow-Origin") != "" {
		t.Error("an origin outside the allowlist was granted CORS")
	}

	// A preflight never reaches a backend.
	pre := httptest.NewRequest(http.MethodOptions, "/v1/chat/completions", nil)
	pre.Header.Set("Origin", "https://ui.example")
	pre.Header.Set("Access-Control-Request-Method", "POST")
	pre.Header.Set("Access-Control-Request-Headers", "X-Router-Retries, Content-Type")
	w3 := httptest.NewRecorder()
	h.ServeHTTP(w3, pre)
	if w3.Code != http.StatusNoContent {
		t.Fatalf("preflight status = %d, want 204 (no backend was contacted)", w3.Code)
	}
	if !strings.Contains(w3.Header().Get("Access-Control-Allow-Headers"), "X-Router-Retries") {
		t.Errorf("preflight allow-headers = %q, want the client's requested headers granted",
			w3.Header().Get("Access-Control-Allow-Headers"))
	}
}

// An empty origin list is "any origin", the shape a laptop box wants.
func TestCORSWildcardOrigin(t *testing.T) {
	c := NewCORS(domain.COSSettings{Enabled: true})
	h := c.Middleware(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("served"))
	}))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("Origin", "https://anything")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if got := w.Header().Get("Access-Control-Allow-Origin"); got != "https://anything" {
		t.Errorf("allow-origin = %q, want the origin echoed (any is allowed)", got)
	}
	if w.Header().Get("Vary") != "Origin" {
		t.Error("echoing an origin without Vary: Origin poisons shared caches")
	}
}

func TestCORSOffByDefault(t *testing.T) {
	c := NewCORS(domain.COSSettings{})
	if c != nil {
		t.Fatal("disabled CORS must create no middleware")
	}
	var h = (*CORS)(nil).Middleware(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("served"))
	}))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("Origin", "https://ui.example")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Header().Get("Access-Control-Allow-Origin") != "" {
		t.Error("an unconfigured router granted CORS")
	}
	if w.Body.String() != "served" {
		t.Errorf("body = %q, want the request served untouched", w.Body.String())
	}
}
