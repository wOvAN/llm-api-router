package config

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"llm-api-router/domain"
)

// One GPU and one runaway agent loop: the router's own rate limiter protects
// backends from us, and nothing protected the router from clients. The limiter is
// off by default (rps 0), and a denial says so in the shape a client already
// parses, with the headers that tell it when to come back.
func TestInboundLimiterDeniesOverBudget(t *testing.T) {
	// 2 rps, burst 2: two immediately, the third inside the window is denied for
	// about half a second (one token's refill).
	l := NewInboundLimiter(domain.InboundSettings{RPS: 2, Burst: 2, PerIP: true})
	if ok, _ := l.Allow("1.2.3.4"); !ok {
		t.Fatal("the first request of a fresh bucket was denied")
	}
	if ok, _ := l.Allow("1.2.3.4"); !ok {
		t.Fatal("the burst of 2 was not served")
	}
	ok, retry := l.Allow("1.2.3.4")
	if ok {
		t.Fatal("the third request inside one second was served")
	}
	if retry <= 0 || retry > time.Second {
		t.Errorf("retry-after = %v, want a positive sub-second wait", retry)
	}

	// Keys are separate buckets: a second client is not punished for the first.
	if ok, _ := l.Allow("5.6.7.8"); !ok {
		t.Error("a fresh client was denied")
	}

	time.Sleep(600 * time.Millisecond)
	if ok, _ := l.Allow("1.2.3.4"); !ok {
		t.Error("the bucket never refilled")
	}
}

func TestInboundLimiterMiddlewareHeaders(t *testing.T) {
	l := NewInboundLimiter(domain.InboundSettings{RPS: 1, Burst: 1, PerIP: true})
	h := l.Middleware(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("served"))
	}))

	first := httptest.NewRecorder()
	h.ServeHTTP(first, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil))
	if first.Code != http.StatusOK || first.Body.String() != "served" {
		t.Fatalf("first request: %d %q, want 200 served", first.Code, first.Body.String())
	}
	if first.Header().Get("X-RateLimit-Remaining") == "" {
		t.Error("the served response carries no rate-limit headers")
	}

	second := httptest.NewRecorder()
	h.ServeHTTP(second, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil))
	if second.Code != http.StatusTooManyRequests {
		t.Fatalf("second request: %d, want 429", second.Code)
	}
	if second.Header().Get("Retry-After") == "" {
		t.Error("a 429 without Retry-After leaves the client to guess")
	}
	var payload map[string]map[string]string
	if err := json.Unmarshal(second.Body.Bytes(), &payload); err != nil || payload["error"]["type"] != "rate_limit_error" {
		t.Errorf("body = %s, want the rate_limit_error envelope", strings.TrimSpace(second.Body.String()))
	}
}

// A limiter that was never configured must not exist: every request passing
// through a nil check is the historical behavior, untouched.
func TestInboundLimiterOffByDefault(t *testing.T) {
	l := NewInboundLimiter(domain.InboundSettings{})
	if l != nil {
		t.Fatal("rps 0 must create no limiter")
	}
	var handler = (*InboundLimiter)(nil).Middleware(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("served"))
	}))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
}

// One shared bucket (per_ip false) is the whole deployment's budget, which is the
// right shape for a single GPU box; per-IP is the shape for a shared router.
func TestInboundLimiterSharedKey(t *testing.T) {
	l := NewInboundLimiter(domain.InboundSettings{RPS: 1, Burst: 1})
	if ok, _ := l.Allow(""); !ok {
		t.Fatal("shared bucket: first request denied")
	}
	if ok, _ := l.Allow(""); ok {
		t.Error("shared bucket: second request served")
	}
}

// Shedding traffic has to be visible to the operator, not only to the client
// that got the 429: a limiter that fires in silence is indistinguishable from a
// backend that is merely slow.
func TestInboundLimiterCountsDenials(t *testing.T) {
	l := NewInboundLimiter(domain.InboundSettings{RPS: 1, Burst: 2})
	for range 5 {
		l.Allow("1.2.3.4") // burst 2, so three of these are over budget
	}
	if got := l.Denied(); got != 3 {
		t.Errorf("Denied() = %d, want 3 (allowed requests are not counted)", got)
	}
	l.Allow("1.2.3.4")
	var nilLimiter *InboundLimiter
	if got := nilLimiter.Denied(); got != 0 {
		t.Errorf("Denied() on a disabled limiter = %d, want 0", got)
	}
}
