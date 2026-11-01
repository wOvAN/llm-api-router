package router

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"llm-api-router/domain"
)

// A backend that declares nothing is byte-for-byte passthrough: the client's path
// and body reach it untouched, and no conversion is reported.
func TestHandleNoProtocolsIsPassthrough(t *testing.T) {
	var gotPath, gotBody string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		gotPath, gotBody = r.URL.Path, string(b)
		_, _ = w.Write([]byte(`{"id":"c1","model":"m","choices":[{"message":{"role":"assistant"}}]}`))
	}))
	defer backend.Close()

	r, store, ms, _ := newTestRouter(t)
	addTestRoute(t, store, "m", backend.URL)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", w.Result().StatusCode, w.Body.String())
	}
	if gotPath != "/v1/chat/completions" {
		t.Fatalf("backend path = %q, want the client's path", gotPath)
	}
	if h := w.Result().Header.Get("X-Router-Convert"); h != "" {
		t.Fatalf("X-Router-Convert = %q, want absent", h)
	}
	if !strings.Contains(gotBody, `"content":"hi"`) {
		t.Fatalf("body was rewritten: %s", gotBody)
	}
	rec := ms.Recent()
	if len(rec) != 1 || rec[0].Convert != "" || rec[0].UpstreamEndpoint != "/v1/chat/completions" {
		t.Fatalf("metric: %+v", rec)
	}
}

// A backend that speaks neither the client's endpoint nor a dialect the router can
// reach gets a 400 from us, naming what it serves — and the backend saw nothing.
// (Before the converter its 404 was forwarded as-is — 4xx is not an error — so
// nothing retried, nothing fell back, and the rule looked healthy.)
func TestHandleProtocolMismatchRefusesWithOur400(t *testing.T) {
	var hits int
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		http.Error(w, "not found", http.StatusNotFound)
	}))
	defer backend.Close()

	r, store, ms, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: backend.URL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/responses"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"}, ServerID: "s1",
		TargetModel: "claude", NumRetries: -1, Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if got := w.Result().StatusCode; got != http.StatusBadRequest {
		t.Fatalf("status %d, want 400: %s", got, w.Body.String())
	}
	if hits != 0 {
		t.Fatalf("backend got %d requests, want 0", hits)
	}
	if body := w.Body.String(); !strings.Contains(body, "/v1/responses") {
		t.Fatalf("refusal must name what the backend serves: %s", body)
	}
	rec := ms.Recent()
	if len(rec) != 1 || rec[0].StatusCode != http.StatusBadRequest {
		t.Fatalf("the refusal must be recorded: %+v", rec)
	}
}

// `convert: off` — from the rule and from the request header — sends the client's
// own request upstream, exactly as before this feature. The backend declares only
// chat, so without the off these two requests would be converted.
func TestHandleConvertOff(t *testing.T) {
	var hits int
	var gotPath string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		gotPath = r.URL.Path
		_, _ = w.Write([]byte(`{"type":"message","role":"assistant","content":[],"model":"claude"}`))
	}))
	defer backend.Close()

	r, store, _, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: backend.URL,
		APITypes:  []domain.APIType{domain.APITypeAnthropic},
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"}, ServerID: "s1",
		TargetModel: "claude", NumRetries: -1, Enabled: true, Convert: "off"}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	send := func(header bool) *http.Response {
		req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
			`{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`))
		if header {
			req.Header.Set("X-Router-Convert", "off")
		}
		w := httptest.NewRecorder()
		r.Handle(w, req)
		return w.Result()
	}
	off := send(true)
	if h := off.Header.Get("X-Router-Convert"); h != "" {
		t.Fatalf("X-Router-Convert = %q, want absent", h)
	}
	send(false) // the rule's own `convert: off`, same decision
	if hits != 2 || gotPath != "/v1/messages" {
		t.Fatalf("off must send the client's own request upstream: hits=%d path=%q", hits, gotPath)
	}
}

// The plan describes the attempt that SERVES: an attempt whose backend can not
// speak the client's dialect is skipped (nothing sent), and the decision of the
// attempt that answered is what the client's headers and the metric report.
func TestHandleConvertReportsServingAttempt(t *testing.T) {
	var hits int
	var gotPath string
	chat := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		gotPath = r.URL.Path
		_, _ = w.Write([]byte(`{"id":"c1","model":"claude","choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer chat.Close()
	dead := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	deadURL := dead.URL
	dead.Close() // hijacked + closed: instant transport error, no dial timeout

	r, store, ms, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "responder", Name: "responder", URL: deadURL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/responses"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddServer(&domain.Server{ID: "chatty", Name: "chatty", URL: chat.URL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"},
		ServerID: "responder", TargetModel: "claude", NumRetries: -1, Enabled: true,
		Fallbacks: []domain.FallbackEntry{{ServerID: "chatty"}}}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", res.StatusCode, w.Body.String())
	}
	if got := res.Header.Get("X-Router-Convert"); got != "messages>chat" {
		t.Fatalf("X-Router-Convert = %q, want messages>chat", got)
	}
	if got := res.Header.Get("X-Router-Server"); got != "chatty" {
		t.Fatalf("X-Router-Server = %q, want the serving attempt", got)
	}
	if hits != 1 || gotPath != "/v1/chat/completions" {
		t.Fatalf("the fallback must be reached at the planned path: hits=%d path=%q", hits, gotPath)
	}
	rec := ms.Recent()
	if len(rec) != 1 || rec[0].Convert != "messages>chat" || rec[0].UpstreamEndpoint != "/v1/chat/completions" {
		t.Fatalf("metric: %+v", rec)
	}
}

// The convert headers are rewritten, not just set: the header map persists across
// attempts, so a conversion reported by a failed attempt must not ride on a
// passthrough later one.
func TestHandleConvertHeadersDoNotLeakAcrossAttempts(t *testing.T) {
	converted := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	convertedURL := converted.URL
	converted.Close() // refused after the eager headers were written
	plain := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"c1","model":"claude","choices":[]}`))
	}))
	defer plain.Close()

	r, store, _, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "chatty", Name: "chatty", URL: convertedURL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddServer(&domain.Server{ID: "undeclared", Name: "undeclared", URL: plain.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"},
		ServerID: "chatty", TargetModel: "claude", NumRetries: -1, Enabled: true,
		Fallbacks: []domain.FallbackEntry{{ServerID: "undeclared"}}}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", res.StatusCode, w.Body.String())
	}
	if got := res.Header.Get("X-Router-Convert"); got != "" {
		t.Fatalf("X-Router-Convert = %q, want absent: the serving backend declares nothing", got)
	}
}

// stream_options is an OpenAI field: it must follow the dialect decision, not the
// client's path. An Anthropic request on an Anthropic backend must arrive with no
// stream_options in the body.
func TestHandleAnthropicStreamGetsNoStreamOptions(t *testing.T) {
	var gotBody string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		_, _ = w.Write([]byte(`{"id":"m1","type":"message","role":"assistant","content":[],"model":"claude"}`))
	}))
	defer backend.Close()

	r, store, _, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: backend.URL,
		APITypes:  []domain.APIType{domain.APITypeAnthropic},
		Protocols: []domain.ServerProtocol{{Path: "/v1/messages"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"}, ServerID: "s1",
		TargetModel: "claude", NumRetries: -1, Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","max_tokens":10,"stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	r.Handle(httptest.NewRecorder(), req)

	if strings.Contains(gotBody, "stream_options") {
		t.Fatalf("OpenAI stream_options leaked into an Anthropic request: %s", gotBody)
	}
}

// The mirror case: an OpenAI streaming request on a chat backend keeps the
// injection, now decided per attempt.
func TestHandleChatStreamStillInjectsUsage(t *testing.T) {
	var gotBody string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		_, _ = w.Write([]byte(`{"id":"c1","model":"m","choices":[]}`))
	}))
	defer backend.Close()

	r, store, _, _ := newTestRouter(t)
	addTestRoute(t, store, "m", backend.URL)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	r.Handle(httptest.NewRecorder(), req)

	if !strings.Contains(gotBody, `"include_usage":true`) {
		t.Fatalf("injection lost: %s", gotBody)
	}
}

// The first usable slice: an Anthropic client, a chat-only backend, a non-streamed
// answer. The client must receive an Anthropic message object.
func TestHandleMessagesServedByChatBackend(t *testing.T) {
	var gotPath, gotBody string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		gotPath, gotBody = r.URL.Path, string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"c1","model":"claude","choices":[{"index":0,` +
			`"message":{"role":"assistant","content":"hi there"},"finish_reason":"stop"}],` +
			`"usage":{"prompt_tokens":12,"completion_tokens":3}}`))
	}))
	defer backend.Close()

	r, store, ms, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: backend.URL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"}, ServerID: "s1",
		TargetModel: "claude", NumRetries: -1, Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", w.Result().StatusCode, w.Body.String())
	}
	if gotPath != "/v1/chat/completions" {
		t.Fatalf("backend path = %q", gotPath)
	}
	if !strings.Contains(gotBody, `"max_tokens":10`) {
		t.Fatalf("anthropic max_tokens not carried over: %s", gotBody)
	}
	body := w.Body.String()
	if !strings.Contains(body, `"type":"message"`) || !strings.Contains(body, `"stop_reason":"end_turn"`) {
		t.Fatalf("client did not get an Anthropic message: %s", body)
	}
	if h := w.Result().Header.Get("X-Router-Convert"); h != "messages>chat" {
		t.Fatalf("X-Router-Convert = %q", h)
	}
	rec := ms.Recent()
	if len(rec) != 1 || rec[0].Convert != "messages>chat" || rec[0].UpstreamEndpoint != "/v1/chat/completions" {
		t.Fatalf("metric not annotated: %+v", rec)
	}
	// Usage is read from the native response: the translator runs after
	// metricsWriter, so extraction never sees translated bytes.
	if rec[0].PromptTokens != 12 || rec[0].CompletionTokens != 3 {
		t.Fatalf("usage lost: %+v", rec[0])
	}
}

// The metric's notes list is bounded (the spec's ≤ 512B): whole notes are kept,
// the tail dropped, and a list that fits is untouched.
func TestCapNotesBoundsTheTail(t *testing.T) {
	if got := capNotes(nil); len(got) != 0 {
		t.Fatalf("nil notes: %v", got)
	}
	short := []string{"thinking=off", "stop_dropped"}
	if got := capNotes(short); len(got) != 2 {
		t.Fatalf("a list that fits must be untouched: %v", got)
	}
	long := []string{}
	n := 0
	for i := 0; n <= maxConvertNotesBytes; i++ {
		s := strings.Repeat("x", 100)
		long = append(long, s)
		n += len(s)
	}
	got := capNotes(long)
	total := 0
	for _, s := range got {
		total += len(s)
	}
	if len(got) != len(long)-1 || total > maxConvertNotesBytes {
		t.Fatalf("capNotes kept %d of %d notes (%d bytes, limit %d)",
			len(got), len(long), total, maxConvertNotesBytes)
	}
}
