package router

import (
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
)

// Anthropic clients ask what a prompt costs before they send it — Claude Code
// fires bursts of these to decide when to compact. Nothing about the answer
// needs a generation, and most backends do not serve the endpoint at all (an
// OpenAI-only backend answers 404, which the client can only log), so the
// router answers it itself with the estimate it already uses for the
// context-window gate.
func TestHandleCountTokensAnsweredLocally(t *testing.T) {
	var hits atomic.Int32
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
	}))
	defer backend.Close()

	r, store, _, _ := newTestRouter(t)
	addTestRoute(t, store, "gpt-4", backend.URL)

	prompt := strings.Repeat("word ", 400) // 2000 chars ≈ 500 tokens
	body := `{"model":"gpt-4","max_tokens":64,"messages":[{"role":"user","content":"` + prompt + `"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.Handle(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", resp.StatusCode, w.Body.String())
	}
	if got := hits.Load(); got != 0 {
		t.Errorf("backend hit %d times, want 0 (a token count needs no generation)", got)
	}
	want := `{"input_tokens":` + strconv.Itoa(len(body)/4) + `}`
	if got := strings.TrimSpace(w.Body.String()); got != want {
		t.Errorf("body = %s, want %s", got, want)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("content-type = %q, want json", ct)
	}
}

// The router answering is not the request disappearing: Claude Code fires these in
// bursts, so the most frequent caller in the room would otherwise be invisible in
// the metrics, the summary tables and active_requests.
func TestHandleCountTokensRecordsAMetric(t *testing.T) {
	r, store, ms, _ := newTestRouter(t)
	addTestRoute(t, store, "gpt-4", "http://127.0.0.1:1")

	req := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens",
		strings.NewReader(`{"model":"gpt-4","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")
	r.Handle(httptest.NewRecorder(), req)

	recent := ms.Recent()
	if len(recent) != 1 {
		t.Fatalf("recent = %d requests, want the answered count_tokens to be visible", len(recent))
	}
	m := recent[0]
	if m.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200", m.StatusCode)
	}
	if m.Model != "gpt-4" {
		t.Errorf("model = %q, want the model the request named", m.Model)
	}
	if m.Class != "count_tokens" {
		t.Errorf("class = %q, want its own class (so the summary table can show it)", m.Class)
	}
	if m.PromptTokens == 0 {
		t.Error("prompt_tokens = 0, want the estimate the router answered with")
	}
	if _, ok := ms.Summaries()["server:router"]; !ok {
		t.Error("no server:router summary, so the answer is invisible where an operator looks")
	}
}
