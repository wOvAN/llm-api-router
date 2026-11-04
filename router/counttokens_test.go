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
