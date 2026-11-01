package router

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"llm-api-router/domain"
)

// The client-dialect pins for a converted stream: what the emitter arms in Task 8
// must keep an Anthropic client's stream free of Chat Completions frames, on the
// paths where the proxy itself writes into the chain.

// A converted stream that dies mid-generation is closed by the proxy's own
// Anthropic-shaped error frame (tw.passthrough() runs before sendMidStreamError):
// the client must not also receive the native control frames the emitter absorbs.
func TestHandleConvertStreamMidStreamBreakStaysInDialect(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		_, _ = w.Write([]byte("data: {\"id\":\"c1\",\"model\":\"claude\",\"choices\":" +
			"[{\"index\":0,\"delta\":{\"content\":\"par\"}}]}\n\n"))
		f.Flush()
		conn, _, err := w.(http.Hijacker).Hijack()
		if err == nil {
			_ = conn.Close() // the generation dies mid-stream, no finish_reason, no [DONE]
		}
	}))
	defer backend.Close()

	r, store, _, _ := newTestRouter(t)
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
		`{"model":"claude","max_tokens":10,"stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	out := w.Body.String()
	if !strings.Contains(out, "event: error") {
		t.Fatalf("the break must be reported in the client's dialect:\n%s", out)
	}
	if !strings.Contains(out, `"text":"par"`) {
		t.Fatalf("the content produced before the break must survive:\n%s", out)
	}
	if strings.Contains(out, "data: [DONE]") || strings.Contains(out, `"object":"chat.completion`) {
		t.Fatalf("native chat frames leaked into a translated stream:\n%s", out)
	}
	if strings.Contains(out, "event: message_stop") {
		t.Fatalf("after an error frame the emitter must not append a normal close:\n%s", out)
	}
}

// include_usage makes a chat backend append a usage-only chunk. On a converted
// attempt the emitter is the filter (the router stops stripping there), so it must
// read that chunk and emit nothing: an OpenAI usage frame in an Anthropic stream
// is unparseable, and message_delta is where its numbers belong.
func TestHandleConvertStreamUsageChunkStaysInsideTheEmitter(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		_, _ = w.Write([]byte("data: {\"id\":\"c1\",\"model\":\"claude\",\"choices\":" +
			"[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n" +
			"data: {\"id\":\"c1\",\"model\":\"claude\",\"choices\":[]," +
			"\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":4}}\n\n" +
			"data: [DONE]\n\n"))
		f.Flush()
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
		`{"model":"claude","max_tokens":10,"stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	out := w.Body.String()
	if strings.Contains(out, `"object":"chat.completion`) || strings.Contains(out, `"usage":{"prompt_tokens"`) {
		t.Fatalf("the native usage frame reached an Anthropic client:\n%s", out)
	}
	// The stream ended with no finish_reason chunk, so the envelope closes with the
	// usage the emitter read from that chunk.
	last := out[strings.LastIndex(out, "event: message_delta"):]
	if !strings.Contains(last, `"input_tokens":9`) || !strings.Contains(last, `"output_tokens":4`) {
		t.Fatalf("the closing message_delta lost the usage:\n%s", out)
	}
	if rec := ms.Recent(); len(rec) != 1 || rec[0].PromptTokens != 9 || rec[0].CompletionTokens != 4 {
		t.Fatalf("native usage not read: %+v", rec)
	}
}

// A dying backend answers 5xx with the SSE frames it had buffered. The proxy
// refuses to forward that (it is unparseable as a response), reports it as a
// failure, and the client gets OUR status: nothing is translated, because nothing
// of it is a stream. Pinned so the shape of that answer is a decision, not an
// accident — the body quotes native frames (see the Task 8 report's findings).
func TestHandleConvertStreamFormErrorBodyIsOurFailure(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("data: {\"id\":\"c1\",\"model\":\"claude\",\"choices\":" +
			"[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n"))
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
		`{"model":"claude","max_tokens":10,"stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusBadGateway {
		t.Fatalf("status %d, want our 502: %s", res.StatusCode, w.Body.String())
	}
	if ct := res.Header.Get("Content-Type"); !strings.Contains(ct, "text/plain") {
		t.Fatalf("a failure is not a stream, content-type = %q", ct)
	}
	// Nothing of it is a stream, so nothing of an Anthropic envelope may appear
	// either: a client must never see a half-message (deltas with no
	// message_start, a close with no open) because an attempt failed.
	if b := w.Body.String(); strings.Contains(b, "message_start") ||
		strings.Contains(b, "content_block_delta") || strings.Contains(b, "message_stop") {
		t.Fatalf("a failed attempt authored a partial Anthropic stream:\n%s", b)
	}
	if rec := ms.Recent(); len(rec) != 1 || rec[0].StatusCode != http.StatusBadGateway {
		t.Fatalf("the failure must be recorded: %+v", rec)
	}
}

// The relay frames on "\n\n", "\r\n\r\n" and "\r\r", so a CRLF-framed backend is a
// real shape on the converted path: every event must reach the client as it is
// produced, not as one native dump at stream end.
func TestHandleConvertStreamAcceptsCRLFFraming(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		// One event per write+flush: the relay hands the emitter a frame per read,
		// so each CRLF event must be translated and delivered as it arrives.
		_, _ = w.Write([]byte("data: {\"id\":\"c1\",\"model\":\"claude\",\"choices\":" +
			"[{\"index\":0,\"delta\":{\"content\":\"He\"}}]}\r\n\r\n"))
		f.Flush()
		_, _ = w.Write([]byte("data: {\"id\":\"c1\",\"model\":\"claude\",\"choices\":" +
			"[{\"index\":0,\"delta\":{\"content\":\"llo\"},\"finish_reason\":\"stop\"}]}\r\n\r\n"))
		f.Flush()
	}))
	defer backend.Close()

	r, store, _, _ := newTestRouter(t)
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
		`{"model":"claude","max_tokens":10,"stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	out := w.Body.String()
	for _, want := range []string{"event: message_start", `"text":"He"`, `"text":"llo"`,
		"event: message_stop", `"stop_reason":"end_turn"`} {
		if !strings.Contains(out, want) {
			t.Fatalf("a CRLF-framed stream lost %s:\n%s", want, out)
		}
	}
	if strings.Contains(out, "data: {\"id\"") {
		t.Fatalf("native frames forwarded to an Anthropic client:\n%s", out)
	}
}

// A backend that answers 200 and then puts its own error object into the stream is
// the frame the emitter must not swallow: an Anthropic client left with no error
// event reads a truncated answer as a complete one.
func TestHandleConvertStreamForwardsAnUpstreamErrorChunk(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		_, _ = w.Write([]byte("data: {\"id\":\"c1\",\"model\":\"claude\",\"choices\":" +
			"[{\"index\":0,\"delta\":{\"content\":\"par\"}}]}\n\n" +
			"data: {\"error\":{\"type\":\"server_error\",\"message\":\"worker died\"}}\n\n"))
		f.Flush()
	}))
	defer backend.Close()

	r, store, _, _ := newTestRouter(t)
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
		`{"model":"claude","max_tokens":10,"stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	out := w.Body.String()
	i := strings.Index(out, "event: error")
	if i < 0 {
		t.Fatalf("the backend's error object never reached the client:\n%s", out)
	}
	if !strings.Contains(out[i:], `"type":"server_error"`) || !strings.Contains(out[i:], `"message":"worker died"`) {
		t.Fatalf("the error is not in the client's spelling:\n%s", out[i:])
	}
	if strings.Contains(out[i:], "message_stop") {
		t.Fatalf("an error event is terminal, no normal close may follow:\n%s", out[i:])
	}
}

// The stream-usage strip decision belongs to the attempt that SERVES. A rule whose
// first attempt converts (and fails) and whose second is a plain chat backend must
// strip on the second: a client that never asked for the usage chunk must not
// receive it because an earlier attempt spoke a different dialect.
func TestHandleConvertStripGateFollowsServingAttempt(t *testing.T) {
	converted := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {}))
	convertedURL := converted.URL
	converted.Close() // hijacked + closed: instant transport error, nothing written
	plain := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		_, _ = w.Write([]byte("data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":" +
			"[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n" +
			"data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[]," +
			"\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":4}}\n\n"))
		f.Flush()
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
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"gpt-4"},
		ServerID: "chatty", TargetModel: "gpt-4", NumRetries: -1, Enabled: true,
		Fallbacks: []domain.FallbackEntry{{ServerID: "undeclared"}}}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"gpt-4","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	res := w.Result()
	if got := res.Header.Get("X-Router-Convert"); got != "" {
		t.Fatalf("the serving attempt converts nothing, got X-Router-Convert=%q", got)
	}
	if body := w.Body.String(); strings.Contains(body, `"usage":{"prompt_tokens"`) {
		t.Fatalf("the passthrough attempt must strip the injected usage chunk:\n%s", body)
	}
	if body := w.Body.String(); !strings.Contains(body, `"content":"hi"`) {
		t.Fatalf("the answer was lost: %s", body)
	}
}
