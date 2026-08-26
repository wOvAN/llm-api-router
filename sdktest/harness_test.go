// Package sdktest validates the wire-protocol compliance of the llm-api-router
// proxy by using the official OpenAI (openai-go) and Anthropic
// (anthropic-sdk-go) SDKs as the consumers of proxied responses.
//
// It is a separate Go module (own go.mod) so the main router module keeps its
// zero-dependency promise: the SDK imports live only here. The SDK sources are
// pulled from the .info/ reference clones via replace directives, so this
// module only builds where .info/ is populated (it is gitignored reference
// material, not part of the main build). `go build ./...` / `go test ./...`
// from the repo root never enter this directory.
package sdktest

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"
	"github.com/openai/openai-go/v3"
	openaioption "github.com/openai/openai-go/v3/option"

	"llm-api-router/config"
	"llm-api-router/domain"
	"llm-api-router/metrics"
	"llm-api-router/router"
)

const (
	// incomingModel is what the OpenAI SDK client requests.
	incomingModel = "gpt-test"
	// anthropicModel is what the Anthropic SDK client requests on /v1/messages.
	anthropicModel = "claude-test"
	// targetModel is the model name the proxy rewrites the request to before
	// sending it upstream, and that fake backends echo back in responses.
	targetModel = "backend-model-xyz"
)

// stack wires the real router in front of a fake backend and exposes official
// SDK clients pointed at the proxy.
type stack struct {
	t         *testing.T
	backend   *httptest.Server
	proxy     *httptest.Server
	ms        *metrics.Store
	store     *config.Store
	openai    openai.Client
	anthropic anthropic.Client

	mu       sync.Mutex
	lastReq  *http.Request
	lastBody []byte
}

func newStack(t *testing.T, backend http.Handler) *stack {
	t.Helper()

	st := &stack{t: t, ms: metrics.New(100)}

	st.backend = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("fake backend: read request body: %v", err)
		}
		st.mu.Lock()
		st.lastReq = r.Clone(context.Background())
		st.lastBody = body
		st.mu.Unlock()
		r.Body = io.NopCloser(bytes.NewReader(body))
		backend.ServeHTTP(w, r)
	}))
	t.Cleanup(st.backend.Close)

	var err error
	st.store, err = config.NewStore(filepath.Join(t.TempDir(), "config.json"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	if err := st.store.AddServer(&domain.Server{
		ID:       "srv1",
		Name:     "fake",
		URL:      st.backend.URL,
		APIKey:   "upstream-key",
		APITypes: []domain.APIType{domain.APITypeOpenAI, domain.APITypeAnthropic},
	}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}

	r := router.New(st.store, st.ms, nil, nil, nil)
	st.proxy = httptest.NewServer(http.HandlerFunc(r.Handle))
	t.Cleanup(st.proxy.Close)

	st.openai = openai.NewClient(
		openaioption.WithBaseURL(st.proxy.URL+"/v1"),
		openaioption.WithAPIKey("client-key"),
	)
	st.anthropic = anthropic.NewClient(
		anthropicoption.WithBaseURL(st.proxy.URL),
		anthropicoption.WithAPIKey("client-key"),
	)
	return st
}

// addRule routes the given incoming model names to target on srv1.
func (st *stack) addRule(incoming []string, target string) {
	st.t.Helper()
	if err := st.store.AddRule(&domain.RoutingRule{
		IncomingModels: incoming,
		TargetModel:    target,
		ServerID:       "srv1",
		Enabled:        true,
	}); err != nil {
		st.t.Fatalf("AddRule: %v", err)
	}
}

// lastRequest returns the most recent request the fake backend saw.
func (st *stack) lastRequest() (*http.Request, []byte) {
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.lastReq, st.lastBody
}

// assertUpstream verifies the proxy rewrote the request model to wantModel and
// replaced the client's credentials with the server's key, in both header
// conventions the upstream may check (Authorization, x-api-key).
func (st *stack) assertUpstream(t *testing.T, wantModel string) {
	t.Helper()
	req, body := st.lastRequest()
	if req == nil {
		t.Fatal("fake backend saw no request")
	}
	var m struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("upstream request body is not valid JSON: %v\nbody: %s", err, body)
	}
	if m.Model != wantModel {
		t.Errorf("upstream model = %q, want %q (request rewrite)", m.Model, wantModel)
	}
	if got := req.Header.Get("Authorization"); got != "Bearer upstream-key" {
		t.Errorf("upstream Authorization = %q, want %q", got, "Bearer upstream-key")
	}
	if got := req.Header.Get("x-api-key"); got != "upstream-key" {
		t.Errorf("upstream x-api-key = %q, want %q", got, "upstream-key")
	}
}

// lastMetric returns the most recently recorded request metric, waiting for
// the router to record it. The router records after the handler returns, but
// an OpenAI SDK client can already have finished its stream at `data: [DONE]`
// — reading the store immediately is racy.
func (st *stack) lastMetric() *domain.RequestMetric {
	st.t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		recent := st.ms.Recent()
		if len(recent) > 0 {
			return &recent[len(recent)-1]
		}
		if time.Now().After(deadline) {
			st.t.Fatal("no metrics recorded")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// jsonBackend serves a fixed JSON body.
func jsonBackend(status int, body string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		fmt.Fprint(w, body)
	})
}

// sseBackend serves complete SSE frames ("...\n\n" each) one write+flush at a
// time, like a real streaming backend.
func sseBackend(frames ...string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		fl, ok := w.(http.Flusher)
		if !ok {
			panic("no flusher")
		}
		for _, f := range frames {
			if _, err := fmt.Fprint(w, f); err != nil {
				return
			}
			fl.Flush()
		}
	})
}
