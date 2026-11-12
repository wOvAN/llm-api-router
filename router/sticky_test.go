package router

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"llm-api-router/domain"
)

// A conversation that rotates across pool members pays a cold prompt cache on
// every turn: each replica has its own KV cache, and turn N's prefix was cached
// on the replica that served turn N-1. So a conversation sticks to the member
// its first turn landed on (the pin LiteLLM calls deployment_affinity and olla
// calls sticky_sessions), while conversations that never share a key keep
// rotating over the whole pool.
func TestHandleRoundRobinPinsConversationToMember(t *testing.T) {
	var a, b atomic.Int32
	handler := func(counter *atomic.Int32) http.HandlerFunc {
		return func(w http.ResponseWriter, req *http.Request) {
			if answerProbe(w, req) {
				return
			}
			counter.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`))
		}
	}
	first := httptest.NewServer(handler(&a))
	defer first.Close()
	second := httptest.NewServer(handler(&b))
	defer second.Close()

	r, store, _, _ := newTestRouter(t)
	for _, s := range []*domain.Server{
		{ID: "a", Name: "A", URL: first.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
		{ID: "b", Name: "B", URL: second.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
	} {
		if err := store.AddServer(s); err != nil {
			t.Fatalf("AddServer: %v", err)
		}
	}
	if err := store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "a",
		Fallbacks:      []domain.FallbackEntry{{ServerID: "b"}},
		Balance:        domain.BalanceRoundRobin,
		Enabled:        true,
	}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}

	send := func(sessionID string) {
		t.Helper()
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
			strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`))
		req.Header.Set("X-Claude-Code-Session-Id", sessionID)
		w := httptest.NewRecorder()
		r.Handle(w, req)
		if got := w.Result().StatusCode; got != http.StatusOK {
			t.Fatalf("status = %d, want 200: %s", got, w.Body.String())
		}
	}

	send("conv-1") // rotates: lands on a, and conv-1 is pinned there
	send("conv-2") // rotates: lands on b
	send("conv-1") // pinned: a again, no rotation spent
	send("conv-1")

	if gotA, gotB := a.Load(), b.Load(); gotA != 3 || gotB != 1 {
		t.Errorf("hits a=%d b=%d, want 3 and 1 (one conversation stays on one member, "+
			"the pool still spreads across members)", gotA, gotB)
	}
}

// The client does not have to send a header: the Responses API's
// prompt_cache_key is the same fact, and the pin works off it too.
func TestHandleRoundRobinPinsByPromptCacheKey(t *testing.T) {
	var a, b atomic.Int32
	handler := func(counter *atomic.Int32) http.HandlerFunc {
		return func(w http.ResponseWriter, req *http.Request) {
			if answerProbe(w, req) {
				return
			}
			counter.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`))
		}
	}
	first := httptest.NewServer(handler(&a))
	defer first.Close()
	second := httptest.NewServer(handler(&b))
	defer second.Close()

	r, store, _, _ := newTestRouter(t)
	for _, s := range []*domain.Server{
		{ID: "a", Name: "A", URL: first.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
		{ID: "b", Name: "B", URL: second.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
	} {
		if err := store.AddServer(s); err != nil {
			t.Fatalf("AddServer: %v", err)
		}
	}
	if err := store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "a",
		Fallbacks:      []domain.FallbackEntry{{ServerID: "b"}},
		Balance:        domain.BalanceRoundRobin,
		Enabled:        true,
	}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}

	send := func(key string) {
		t.Helper()
		req := httptest.NewRequest(http.MethodPost, "/v1/responses",
			strings.NewReader(`{"model":"gpt-4","prompt_cache_key":"`+key+`","input":[{"role":"user","content":"hi"}]}`))
		w := httptest.NewRecorder()
		r.Handle(w, req)
		if got := w.Result().StatusCode; got != http.StatusOK {
			t.Fatalf("status = %d, want 200: %s", got, w.Body.String())
		}
	}

	send("k1")
	send("k2")
	send("k1")

	if gotA, gotB := a.Load(), b.Load(); gotA != 2 || gotB != 1 {
		t.Errorf("hits a=%d b=%d, want 2 and 1", gotA, gotB)
	}
}

// The key cascade: a session header outranks the body, a body-level key outranks
// the content fingerprint, and the fingerprint separates two conversations that
// open differently while it joins a conversation's follow-up turns (whose text
// is tool_result chatter, not the opening turn).
func TestConversationKeyCascade(t *testing.T) {
	newReq := func(headers map[string]string) *http.Request {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		for k, v := range headers {
			req.Header.Set(k, v)
		}
		return req
	}

	bodyA := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"refactor the parser"},{"role":"user","content":"and the tests"}]}`)
	bodyAChatty := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"refactor the parser"},{"role":"assistant","content":"ok"},{"role":"user","content":"and the tests"}]}`)
	bodyB := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"write me a haiku"}]}`)
	bodyKeyed := []byte(`{"model":"gpt-4","prompt_cache_key":"conv-9","messages":[{"role":"user","content":"refactor the parser"}]}`)

	headA, headB := conversationKey(newReq(nil), bodyA), conversationKey(newReq(nil), bodyAChatty)
	if headA == "" || headA != headB {
		t.Errorf("fingerprint of a conversation's turns = %q vs %q, want one non-empty key", headA, headB)
	}
	if other := conversationKey(newReq(nil), bodyB); other == headA {
		t.Error("two different conversations share one key")
	}
	if keyed := conversationKey(newReq(nil), bodyKeyed); keyed == "" || keyed == headA {
		t.Errorf("prompt_cache_key = %q, want its own key (distinct from the fingerprint %q)", keyed, headA)
	}
	fromHeader := conversationKey(newReq(map[string]string{"X-Claude-Code-Session-Id": "s-1"}), bodyA)
	if fromHeader == "" || fromHeader == keyedHeaderless(t, bodyKeyed) {
		t.Error("a session header must outrank the body's own key")
	}
	if other := conversationKey(newReq(map[string]string{"X-Claude-Code-Session-Id": "s-2"}), bodyA); other == fromHeader {
		t.Error("two session ids share one key")
	}
}

func keyedHeaderless(t *testing.T, body []byte) string {
	t.Helper()
	return conversationKey(httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil), body)
}
