package router

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"llm-api-router/config"
	"llm-api-router/domain"
	"llm-api-router/metrics"
	"llm-api-router/trace"
)

// The TPM reservation is released by quota.Complete on every exit path, and
// Complete is the only release there is (no TTL). A client that gives up while
// the router is sleeping in the retry backoff returns through the
// context.Done() branch of the backoff select, which is an exit path too.
func TestHandleClientDisconnectInBackoffReleasesQuotaReservation(t *testing.T) {
	var hits atomic.Int32
	dead := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		hijacker, ok := w.(http.Hijacker)
		if !ok {
			t.Fatal("response writer is not a hijacker")
		}
		conn, _, err := hijacker.Hijack()
		if err != nil {
			t.Fatalf("Hijack: %v", err)
		}
		_ = conn.Close() // accepted then closed: a transport error, retried
	}))
	defer dead.Close()

	store, err := config.NewStore(t.TempDir() + "/config.json")
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	quota := config.NewQuotaTracker()
	quota.SetLimitOverride(func(string) (int64, int64) { return 1000, 0 })
	r := New(store, metrics.New(100), nil, nil, quota, trace.New(100, domain.TraceSettings{}), nil)

	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: dead.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"gpt-4"},
		ServerID: "s1", TargetModel: "gpt-4", NumRetries: 5, Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4","max_tokens":1000,`+
			`"messages":[{"role":"user","content":"hi"}]}`))
	req = req.WithContext(ctx)
	// The first attempt fails on the closed connection in milliseconds, so the
	// cancel lands while the router is in its 100ms backoff.
	time.AfterFunc(20*time.Millisecond, cancel)
	r.Handle(httptest.NewRecorder(), req)

	if hits.Load() == 0 {
		t.Fatal("the backend was never contacted; the test proves nothing")
	}
	// Nothing completed, so the whole window is free again: a request that
	// declares the full budget must be admitted.
	if !quota.Reserve("s1", 1000) {
		t.Error("the disconnected request left its 1000-token reservation on the server, " +
			"want the TPM window free (a leaked reservation silently erodes capacity)")
	}
}
