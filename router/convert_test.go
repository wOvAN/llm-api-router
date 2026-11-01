package router

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"llm-api-router/domain"
	"llm-api-router/proxy"
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
// answer. The request must reach the backend IN CHAT SHAPE (the plan moved the
// path; the transform carries the body) and the answer must come back in Anthropic
// shape, envelope, blocks and usage spelling included.
func TestHandleMessagesServedByChatBackend(t *testing.T) {
	var gotPath, gotBody string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		gotPath, gotBody = r.URL.Path, string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"c1","model":"claude","choices":[{"index":0,` +
			`"message":{"role":"assistant","content":"hi there"},"finish_reason":"stop"}],` +
			`"usage":{"prompt_tokens":120,"completion_tokens":3,` +
			`"prompt_tokens_details":{"cached_tokens":100}}}`))
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
		`{"model":"claude","max_tokens":10,"system":"be brief",`+
			`"tools":[{"name":"fetch","input_schema":{"type":"object","properties":{"u":{"type":"number"}}}}],`+
			`"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", w.Result().StatusCode, w.Body.String())
	}
	if gotPath != "/v1/chat/completions" {
		t.Fatalf("backend path = %q", gotPath)
	}
	// The request side is the point of the spoke: an Anthropic body that merely
	// landed on a chat path is the failure the loss policy exists to prevent.
	if strings.Contains(gotBody, "input_schema") {
		t.Fatalf("the backend received an Anthropic body: %s", gotBody)
	}
	for _, want := range []string{`"type":"function"`, `"parameters"`, `"name":"fetch"`,
		`"role":"system"`, `"content":"be brief"`, `"content":"hi"`, `"max_tokens":10`} {
		if !strings.Contains(gotBody, want) {
			t.Fatalf("converted request lost %s: %s", want, gotBody)
		}
	}
	body := w.Body.String()
	if !strings.Contains(body, `"type":"message"`) || !strings.Contains(body, `"stop_reason":"end_turn"`) {
		t.Fatalf("client did not get an Anthropic message: %s", body)
	}
	// The usage convention is converted too: OpenAI's prompt_tokens (120) already
	// contains its cached tokens (100), Anthropic's input_tokens sits next to them.
	for _, want := range []string{`"type":"text"`, `"text":"hi there"`, `"input_tokens":20`,
		`"cache_read_input_tokens":100`, `"output_tokens":3`} {
		if !strings.Contains(body, want) {
			t.Fatalf("client answer lost %s: %s", want, body)
		}
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
	if rec[0].PromptTokens != 120 || rec[0].CompletionTokens != 3 || rec[0].CachedTokens != 100 {
		t.Fatalf("usage lost: %+v", rec[0])
	}
}

// A request the spoke can not represent is a per-attempt refusal, exactly like a
// backend the plan can not reach: nothing is sent to that backend, the next
// attempt is tried, and when the last attempt refuses too the client gets OUR 400
// naming the field — never a backend 4xx for a body it never saw, and never a
// healthy-looking rule.
func TestHandleConvertRequestRefusal(t *testing.T) {
	// A hosted tool is always-400 class: the provider runs it, the model does not.
	const body = `{"model":"claude","max_tokens":10,` +
		`"tools":[{"name":"w","type":"web_search_20250305"}],` +
		`"messages":[{"role":"user","content":"hi"}]}`

	t.Run("falls through to the next attempt", func(t *testing.T) {
		var chatHits, plainHits int
		chat := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			chatHits++
		}))
		chatURL := chat.URL
		chat.Close() // hijacked + closed: the refusal must be decided before the dial
		plain := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			plainHits++
			_, _ = w.Write([]byte(`{"type":"message","role":"assistant","content":[],"model":"claude"}`))
		}))
		defer plain.Close()

		r, store, ms, _ := newTestRouter(t)
		if err := store.AddServer(&domain.Server{ID: "chatty", Name: "chatty", URL: chatURL,
			APITypes:  []domain.APIType{domain.APITypeOpenAI},
			Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
			t.Fatalf("AddServer: %v", err)
		}
		if err := store.AddServer(&domain.Server{ID: "undeclared", Name: "undeclared", URL: plain.URL,
			APITypes: []domain.APIType{domain.APITypeAnthropic}}); err != nil {
			t.Fatalf("AddServer: %v", err)
		}
		if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"},
			ServerID: "chatty", TargetModel: "claude", NumRetries: -1, Enabled: true,
			Fallbacks: []domain.FallbackEntry{{ServerID: "undeclared"}}}); err != nil {
			t.Fatalf("AddRule: %v", err)
		}
		req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
		w := httptest.NewRecorder()
		r.Handle(w, req)

		res := w.Result()
		if res.StatusCode != http.StatusOK {
			t.Fatalf("status %d: %s", res.StatusCode, w.Body.String())
		}
		if chatHits != 0 || plainHits != 1 {
			t.Fatalf("the refusing attempt must send nothing: chat=%d plain=%d", chatHits, plainHits)
		}
		if h := res.Header.Get("X-Router-Convert"); h != "" {
			t.Fatalf("the serving attempt is a passthrough, got X-Router-Convert=%q", h)
		}
		if rec := ms.Recent(); len(rec) != 1 || rec[0].Convert != "" {
			t.Fatalf("the metric describes the serving attempt: %+v", rec)
		}
	})

	t.Run("the last refusal is our 400", func(t *testing.T) {
		var hits int
		backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			hits++
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
		req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
		w := httptest.NewRecorder()
		r.Handle(w, req)

		res := w.Result()
		if res.StatusCode != http.StatusBadRequest {
			t.Fatalf("status %d, want 400: %s", res.StatusCode, w.Body.String())
		}
		if hits != 0 {
			t.Fatalf("a refusal sends nothing upstream, got %d hits", hits)
		}
		if got := res.Header.Get("X-Router-Fallback-Errors"); got != "" {
			t.Fatalf("a single attempt has no fallback errors: %q", got)
		}
		if b := w.Body.String(); !strings.Contains(b, "tools") {
			t.Fatalf("the refusal must name the field: %s", b)
		}
		if rec := ms.Recent(); len(rec) != 1 || rec[0].StatusCode != http.StatusBadRequest {
			t.Fatalf("the refusal must be recorded: %+v", rec)
		}
	})
}

// A top-level field the spoke never decodes vanishes from the converted body, and
// a silent loss is what the notes channel exists to make visible (R16): the routed
// path compares what went in against what came out and reports the difference.
func TestHandleConvertReportsDroppedKeys(t *testing.T) {
	var gotBody string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		_, _ = w.Write([]byte(`{"id":"c1","model":"claude","choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
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
		`{"model":"claude","max_tokens":10,"service_tier":"priority",`+
			`"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if strings.Contains(gotBody, "service_tier") {
		t.Fatalf("the spoke must not pass the field through: %s", gotBody)
	}
	notes := w.Result().Header.Get("X-Router-Convert-Notes")
	if !strings.Contains(notes, "keys:1dropped") {
		t.Fatalf("the loss of an undecoded field must be visible, notes=%q", notes)
	}
	rec := ms.Recent()
	if len(rec) != 1 || !strings.Contains(strings.Join(rec[0].ConvertNotes, ","), "keys:1dropped") {
		t.Fatalf("the metric must carry it too: %+v", rec)
	}
}

// The metric's notes list is bounded (the spec's ≤ 512B): whole notes are kept,
// the tail dropped, and a list that fits is untouched.
func TestCapNotesBoundsTheTail(t *testing.T) {
	if got := proxy.CapConvertNotes(nil); len(got) != 0 {
		t.Fatalf("nil notes: %v", got)
	}
	short := []string{"thinking=off", "stop_dropped"}
	if got := proxy.CapConvertNotes(short); len(got) != 2 {
		t.Fatalf("a list that fits must be untouched: %v", got)
	}
	long := []string{}
	n := 0
	for i := 0; n <= proxy.MaxConvertNotesBytes; i++ {
		s := strings.Repeat("x", 100)
		long = append(long, s)
		n += len(s)
	}
	got := proxy.CapConvertNotes(long)
	total := 0
	for _, s := range got {
		total += len(s)
	}
	if len(got) != len(long)-1 || total > proxy.MaxConvertNotesBytes {
		t.Fatalf("CapConvertNotes kept %d of %d notes (%d bytes, limit %d)",
			len(got), len(long), total, proxy.MaxConvertNotesBytes)
	}
}
