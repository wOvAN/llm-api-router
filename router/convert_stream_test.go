package router

import (
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"llm-api-router/domain"
	"llm-api-router/proxy"
)

// The client-dialect pins for a converted stream: what the emitter arms in Task 8
// must keep an Anthropic client's stream free of Chat Completions frames, on the
// paths where the proxy itself writes into the chain. The one non-streamed pin
// here is the forwarded backend error body, which the ruling on the router's own
// 5xx shape argues about.

// A converted stream that dies mid-generation is closed by the proxy's own
// Anthropic-shaped error frame (tw.passthrough() runs before sendMidStreamError):
// the client must not also receive the native control frames the emitter absorbs.
func TestHandleConvertStreamMidStreamBreakStaysInDialect(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if answerProbe(w, req) {
			return
		}
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
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if answerProbe(w, req) {
			return
		}
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
// refuses to forward that (an SSE client can not parse it as a response), reports
// it as a failure, and the client gets OUR status: a text/plain 502 whose text
// quotes the backend's native frames.
//
// RULING (a decision, not an accident to fix): a router-authored failure keeps the
// router's own shape in every dialect. The spec's translated-error-body
// requirement is about the body we FORWARD — a backend's `{"error":{…}}` JSON on a
// 4xx/5xx becomes `{"type":"error","error":{…}}`, which the response spoke does on
// the normal path. A body we AUTHOR is not a forwarded one, and router-authored
// failures are text/plain by design across the product (400 for a conversion
// refusal, 404 for an unknown model, 502 when every attempt failed). A non-200
// makes every SDK raise an API error carrying the raw
// body: the Anthropic SDK parses its own envelope to build a typed message and
// falls back to the raw text when the body is not that shape, so the client is
// never left with a silent failure — it reads what the backend actually said, the
// most useful text available. Translating this one path into an Anthropic envelope
// would dress up the same quote, and would make a router 502 speak a dialect no
// other router failure speaks.
//
// Pinned on all four counts: our status, our content-type, the quote itself, and
// the absence of any half-built Anthropic envelope.
func TestHandleConvertStreamFormErrorBodyIsOurFailure(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if answerProbe(w, req) {
			return
		}
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
	b := w.Body.String()
	// The quote: the backend's status and the frames it died with.
	if !strings.Contains(b, "503") || !strings.Contains(b, "partial") {
		t.Fatalf("the failure stopped quoting the backend:\n%s", b)
	}
	// Nothing of it is a stream, so nothing of an Anthropic envelope may appear
	// either: a client must never see a half-message (deltas with no
	// message_start, a close with no open) because an attempt failed.
	if strings.Contains(b, "message_start") ||
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
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if answerProbe(w, req) {
			return
		}
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
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if answerProbe(w, req) {
			return
		}
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
	converted := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if answerProbe(w, req) {
			return
		}
	}))
	convertedURL := converted.URL
	converted.Close() // hijacked + closed: instant transport error, nothing written
	plain := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if answerProbe(w, req) {
			return
		}
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

// The other half of the 5xx ruling above: a backend's own error body, which the
// proxy FORWARDS (4xx/5xx are not errors here), is translated into the client's
// envelope — an Anthropic SDK builds a typed API error from
// `{"type":"error","error":{…}}` and reads an OpenAI `{"error":{…}}` body as an
// opaque string. The spoke does this on the JSON path; pinned where the client
// reads it, because the unit test cannot show it survived the writer chain.
func TestHandleConvertForwardedErrorBodyIsTranslated(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if answerProbe(w, req) {
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"type":"rate_limit_exceeded","message":"slow down"}}`))
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

	res := w.Result()
	if res.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("the backend's status must be forwarded: %d", res.StatusCode)
	}
	body := w.Body.String()
	if !strings.Contains(body, `"type":"error"`) || !strings.Contains(body, `"type":"rate_limit_exceeded"`) ||
		!strings.Contains(body, `"message":"slow down"`) {
		t.Fatalf("the forwarded error body is not in the client's spelling:\n%s", body)
	}
	if rec := ms.Recent(); len(rec) != 1 || rec[0].StatusCode != http.StatusTooManyRequests {
		t.Fatalf("metric: %+v", rec)
	}
}

// clientEvents lists the event types a client stream carries, in the order its SDK
// dispatches on them.
func clientEvents(s string) []string {
	var out []string
	for _, line := range strings.Split(s, "\n") {
		if name, ok := strings.CutPrefix(line, "event: "); ok {
			out = append(out, name)
		}
	}
	return out
}

// The shape a chat backend actually closes with most often: deltas, then
// `data: [DONE]` with no finish_reason anywhere near it (vLLM and llama.cpp put the
// reason on a chunk that carries content, and a client-driven drain ends on neither).
// An Anthropic client must still get the whole envelope with message_stop last, and
// one delivery must carry exactly one event — the relay flushes per frame, so a
// delivery gluing one event's tail onto the next is a client-side parse bug
// (opencode #7692). The frames are deliberately split across writes so a partial
// frame can not be what proves the shape.
func TestHandleConvertStreamEndsOnDoneWithNoFinishReason(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if answerProbe(w, req) {
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		_, _ = w.Write([]byte("data: {\"id\":\"c1\",\"model\":\"claude\",\"choices\":" +
			"[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"}}]}\n"))
		f.Flush()
		_, _ = w.Write([]byte("\ndata: {\"id\":\"c1\",\"model\":\"claude\",\"choices\":" +
			"[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}\n\ndata: [DONE]\n\n"))
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
	want := []string{"message_start", "content_block_start", "content_block_delta",
		"content_block_delta", "content_block_stop", "message_delta", "message_stop"}
	if got := clientEvents(out); !slices.Equal(got, want) {
		t.Fatalf("event sequence = %v, want %v:\n%s", got, want, out)
	}
	if strings.Contains(out, "data: [DONE]") || strings.Contains(out, `"object":"chat.completion`) {
		t.Fatalf("native frames leaked into a translated stream:\n%s", out)
	}
	// One event per delivery: every blank-line-separated block is an event, so a
	// block that opens with a data: line is a frame that was split or glued.
	for _, ev := range strings.Split(strings.TrimRight(out, "\n"), "\n\n") {
		if !strings.HasPrefix(ev, "event: ") {
			t.Fatalf("delivered chunk is not one event: %q", ev)
		}
	}
	if rec := ms.Recent(); len(rec) != 1 || rec[0].Convert != "messages>chat" {
		t.Fatalf("metric: %+v", rec)
	}
}

// A stream that ends with neither a finish_reason nor `[DONE]` — the backend closed
// the body cleanly after its last delta — is the same promise: the message the
// client was left with must be a closed one, not an open block and a dead socket.
func TestHandleConvertStreamEndsWithoutDone(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if answerProbe(w, req) {
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"c1\",\"model\":\"claude\",\"choices\":" +
			"[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\n"))
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
	want := []string{"message_start", "content_block_start", "content_block_delta",
		"content_block_stop", "message_delta", "message_stop"}
	if got := clientEvents(out); !slices.Equal(got, want) {
		t.Fatalf("event sequence = %v, want %v:\n%s", got, want, out)
	}
}

// A stream that thinks, answers, and answers twice with parallel tool calls whose
// arguments arrive in fragments: the block lifecycle that keeps them apart has to
// survive the relay, the model rewrite and the metrics writer, all of which sit on
// the native side of the emitter. Pinned where the client reads it — the emitter's
// own unit test cannot show that a frame survived the writers above it.
func TestHandleConvertStreamThinkingAndParallelToolCalls(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if answerProbe(w, req) {
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		for _, frame := range []string{
			`data: {"id":"c1","model":"claude","choices":[{"index":0,"delta":{"reasoning_content":"hm"}}]}` + "\n\n",
			`data: {"id":"c1","model":"claude","choices":[{"index":0,"delta":{"content":"both"}}]}` + "\n\n",
			`data: {"id":"c1","model":"claude","choices":[{"index":0,"delta":{"tool_calls":[` +
				`{"index":0,"id":"call_a","function":{"name":"a","arguments":"{\"x\":"}},` +
				`{"index":1,"id":"call_b","function":{"name":"b","arguments":"{\"y\":"}}]}}]}` + "\n\n",
			`data: {"id":"c1","model":"claude","choices":[{"index":0,"delta":{"tool_calls":[` +
				`{"index":1,"function":{"arguments":"2}"}},` +
				`{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":"tool_calls"}]}` + "\n\n",
			"data: [DONE]\n\n",
		} {
			_, _ = w.Write([]byte(frame))
			f.Flush()
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
	// One block per kind, in the order the deltas arrived: thinking, text, then one
	// tool_use per chat tool_calls index — and every block that opened is closed.
	if got := strings.Count(out, `"type":"content_block_start"`); got != 4 {
		t.Fatalf("content_block_start x%d, want 4 (thinking, text, tool_use x2):\n%s", got, out)
	}
	if got := strings.Count(out, `"content_block_stop"`); got != 4 {
		t.Fatalf("content_block_stop x%d, want 4:\n%s", got, out)
	}
	for _, want := range []string{`"thinking":"hm"`, `"text":"both"`,
		`"id":"call_a"`, `"id":"call_b"`, `"partial_json":"{\"x\":"`, `"partial_json":"1}"`,
		`"partial_json":"{\"y\":"`, `"partial_json":"2}"`} {
		if !strings.Contains(out, want) {
			t.Fatalf("a converted stream lost %s:\n%s", want, out)
		}
	}
	last := out[strings.LastIndex(out, "event: message_delta"):]
	if !strings.Contains(last, `"stop_reason":"tool_use"`) {
		t.Fatalf("the closing message_delta downgraded the stop reason:\n%s", out)
	}
	if !strings.Contains(out, "event: message_stop") {
		t.Fatalf("the message was never closed:\n%s", out)
	}
	if strings.Contains(out, `"tool_calls"`) || strings.Contains(out, "data: [DONE]") {
		t.Fatalf("native frames leaked into a translated stream:\n%s", out)
	}
}

// A runaway generation the loop detector cuts short is a control event, so it lands
// in the client's dialect — and the emitter, which sits below the detector, must
// stay silent behind it: the proxy returns as soon as the frame is written, so a
// normal close after the error is a stream that contradicts itself.
func TestHandleConvertStreamLoopDetectionStaysInDialect(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if answerProbe(w, req) {
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		chunk := "data: {\"id\":\"c1\",\"model\":\"claude\",\"choices\":" +
			"[{\"index\":0,\"delta\":{\"content\":\"same \"}}]}\n\n"
		for range 40 {
			if _, err := w.Write([]byte(chunk)); err != nil {
				return
			}
			f.Flush()
		}
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
	w := newSyncRecorder()
	r.Handle(w, req)

	out := w.content()
	i := strings.Index(out, "event: error")
	if i < 0 {
		t.Fatalf("the loop must be reported in the client's dialect:\n%s", out)
	}
	if !strings.Contains(out[i:], "stuck in a loop") {
		t.Fatalf("the loop frame is not the client's dialect:\n%s", out[i:])
	}
	if strings.Contains(out, "data: [DONE]") || strings.Contains(out[i:], "message_stop") {
		t.Fatalf("a cut stream must not be closed as a normal one:\n%s", out)
	}
	if rec := ms.Recent(); len(rec) != 1 {
		t.Fatalf("the cut stream must be recorded: %+v", rec)
	}
}

// A heartbeat inside a converted stream: the ping writers sit BELOW the emitter, so
// an Anthropic client gets Anthropic's `event: ping` and the emitter never sees it —
// a comment frame reaching the emitter is dropped, and a ping translated as a chunk
// would open a content block out of nothing.
func TestHandleConvertStreamKeepAliveStaysInDialect(t *testing.T) {
	old := proxy.KeepAliveIdle
	proxy.KeepAliveIdle = 30 * time.Millisecond
	defer func() { proxy.KeepAliveIdle = old }()

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if answerProbe(w, req) {
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		_, _ = w.Write([]byte("data: {\"id\":\"c1\",\"model\":\"claude\",\"choices\":" +
			"[{\"index\":0,\"delta\":{\"content\":\"He\"}}]}\n\n"))
		f.Flush()
		time.Sleep(120 * time.Millisecond) // a backend pause longer than the heartbeat
		_, _ = w.Write([]byte("data: {\"id\":\"c1\",\"model\":\"claude\",\"choices\":" +
			"[{\"index\":0,\"delta\":{\"content\":\"llo\"},\"finish_reason\":\"stop\"}]}\n\n" +
			"data: [DONE]\n\n"))
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
	w := newSyncRecorder()
	r.Handle(w, req)

	out := w.content()
	start := strings.Index(out, "event: message_start")
	ping := strings.Index(out, "event: ping")
	stop := strings.Index(out, "event: message_stop")
	if start < 0 || stop < 0 {
		t.Fatalf("the envelope did not reach the client:\n%s", out)
	}
	if ping < 0 {
		t.Fatalf("a converted stream got no heartbeat:\n%s", out)
	}
	if start >= ping || ping >= stop {
		t.Fatalf("the heartbeat must ride inside the stream it belongs to:\n%s", out)
	}
	// The emitter resumed after the pause: the second half of the answer and the
	// close are its own doing, not the heartbeat's.
	if !strings.Contains(out, `"text":"He"`) || !strings.Contains(out, `"text":"llo"`) {
		t.Fatalf("the stream did not resume after the heartbeat:\n%s", out)
	}
	if strings.Contains(out, ": keep-alive") {
		t.Fatalf("the OpenAI heartbeat comment leaked into an Anthropic stream:\n%s", out)
	}
}
