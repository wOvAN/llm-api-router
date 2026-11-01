package sdktest

import (
	"context"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go/v2"
	"github.com/anthropics/anthropic-sdk-go/v2/packages/ssestream"
)

const anthropicMessageJSON = `{
	"id": "msg_fake_1",
	"type": "message",
	"role": "assistant",
	"model": "` + targetModel + `",
	"content": [{"type": "text", "text": "Bonjour from fake backend"}],
	"stop_reason": "end_turn",
	"stop_sequence": null,
	"usage": {"input_tokens": 10, "output_tokens": 5}
}`

// Canonical Anthropic stream event frames, full shape (official API).
var anthropicFullFrames = []string{
	"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_fake_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"" + targetModel + "\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n",
	"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Bon\"}}\n\n",
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"jour\"}}\n\n",
	"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
	"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"input_tokens\":10,\"output_tokens\":5}}\n\n",
	"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
}

// llama.cpp /v1/messages stream shape: same events minus message_start (and no
// OpenAI-style [DONE] sentinel). Backends in the wild (llama.cpp) emit exactly
// this, so official SDK clients must survive the missing message_start.
var anthropicLlamaCppFrames = []string{
	"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Bon\"}}\n\n",
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"jour\"}}\n\n",
	"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
	"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"input_tokens\":10,\"output_tokens\":5}}\n\n",
	"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
}

func anthropicParams() anthropic.MessageNewParams {
	return anthropic.MessageNewParams{
		Model:     anthropicModel,
		MaxTokens: 100,
		Messages:  []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock("hi"))},
	}
}

func TestAnthropicMessagesJSON(t *testing.T) {
	st := newStack(t, jsonBackend(http.StatusOK, anthropicMessageJSON))
	st.addRule([]string{anthropicModel}, targetModel)

	resp, err := st.anthropic.Messages.New(context.Background(), anthropicParams())
	if err != nil {
		t.Fatalf("Messages.New: %v", err)
	}
	st.assertUpstream(t, targetModel)

	if resp.Model != anthropicModel {
		t.Errorf("Model = %q, want %q (response rewrite back to client's name)", resp.Model, anthropicModel)
	}
	if resp.ID != "msg_fake_1" {
		t.Errorf("ID = %q, want msg_fake_1", resp.ID)
	}
	if len(resp.Content) != 1 {
		t.Fatalf("content blocks = %d, want 1", len(resp.Content))
	}
	if got := resp.Content[0].AsText().Text; got != "Bonjour from fake backend" {
		t.Errorf("content = %q, want %q", got, "Bonjour from fake backend")
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("stop_reason = %q, want end_turn", resp.StopReason)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 5 {
		t.Errorf("usage = in %d out %d, want 10/5", resp.Usage.InputTokens, resp.Usage.OutputTokens)
	}

	m := st.lastMetric()
	if m.StatusCode != 200 {
		t.Errorf("metric status = %d, want 200", m.StatusCode)
	}
	if m.PromptTokens != 10 || m.CompletionTokens != 5 {
		t.Errorf("metric tokens = %d/%d, want 10/5 (Anthropic usage)", m.PromptTokens, m.CompletionTokens)
	}
}

// accumulateStream iterates a NewStreaming result into an anthropic.Message,
// the pattern the SDK documents for stream consumers.
func accumulateStream(t *testing.T, stream *ssestream.Stream[anthropic.MessageStreamEventUnion]) *anthropic.Message {
	t.Helper()
	msg := &anthropic.Message{}
	for stream.Next() {
		if err := msg.Accumulate(stream.Current()); err != nil {
			t.Fatalf("Accumulate: %v", err)
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream.Err: %v", err)
	}
	return msg
}

func TestAnthropicStreamFull(t *testing.T) {
	st := newStack(t, sseBackend(anthropicFullFrames...))
	st.addRule([]string{anthropicModel}, targetModel)

	stream := st.anthropic.Messages.NewStreaming(context.Background(), anthropicParams())
	msg := accumulateStream(t, stream)
	st.assertUpstream(t, targetModel)

	if msg.Model != anthropicModel {
		t.Errorf("streamed Model = %q, want %q (rewrite inside message_start)", msg.Model, anthropicModel)
	}
	if msg.ID != "msg_fake_1" {
		t.Errorf("streamed ID = %q, want msg_fake_1", msg.ID)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("content blocks = %d, want 1", len(msg.Content))
	}
	if got := msg.Content[0].AsText().Text; got != "Bonjour" {
		t.Errorf("streamed content = %q, want %q", got, "Bonjour")
	}
	if msg.StopReason != "end_turn" {
		t.Errorf("stop_reason = %q, want end_turn", msg.StopReason)
	}
	if msg.Usage.InputTokens != 10 || msg.Usage.OutputTokens != 5 {
		t.Errorf("usage = in %d out %d, want 10/5", msg.Usage.InputTokens, msg.Usage.OutputTokens)
	}
}

func TestAnthropicStreamLlamaCpp(t *testing.T) {
	st := newStack(t, sseBackend(anthropicLlamaCppFrames...))
	st.addRule([]string{anthropicModel}, targetModel)

	stream := st.anthropic.Messages.NewStreaming(context.Background(), anthropicParams())
	msg := accumulateStream(t, stream)

	// Without message_start the SDK cannot know the model name; everything
	// else must still accumulate.
	if len(msg.Content) != 1 {
		t.Fatalf("content blocks = %d, want 1", len(msg.Content))
	}
	if got := msg.Content[0].AsText().Text; got != "Bonjour" {
		t.Errorf("streamed content = %q, want %q", got, "Bonjour")
	}
	if msg.StopReason != "end_turn" {
		t.Errorf("stop_reason = %q, want end_turn", msg.StopReason)
	}
	if msg.Usage.InputTokens != 10 || msg.Usage.OutputTokens != 5 {
		t.Errorf("usage = in %d out %d, want 10/5 (from message_delta)", msg.Usage.InputTokens, msg.Usage.OutputTokens)
	}
}

// Official Anthropic streams and our proxy's keep-alive heartbeats both use
// the ping event; SDK clients must ignore it mid-stream.
func TestAnthropicStreamIgnoresPing(t *testing.T) {
	frames := []string{
		anthropicFullFrames[0],
		anthropicFullFrames[1],
		anthropicFullFrames[2],
		"event: ping\ndata: {\"type\":\"ping\"}\n\n",
		anthropicFullFrames[3],
		"event: ping\ndata: {\"type\":\"ping\"}\n\n",
		anthropicFullFrames[4],
		anthropicFullFrames[5],
		anthropicFullFrames[6],
	}
	st := newStack(t, sseBackend(frames...))
	st.addRule([]string{anthropicModel}, targetModel)

	stream := st.anthropic.Messages.NewStreaming(context.Background(), anthropicParams())
	msg := accumulateStream(t, stream)
	if got := msg.Content[0].AsText().Text; got != "Bonjour" {
		t.Errorf("streamed content = %q, want %q", got, "Bonjour")
	}
}

func TestAnthropicErrorPassthrough(t *testing.T) {
	st := newStack(t, jsonBackend(http.StatusBadRequest,
		`{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens too big"}}`))
	st.addRule([]string{anthropicModel}, targetModel)

	_, err := st.anthropic.Messages.New(context.Background(), anthropicParams())
	if err == nil {
		t.Fatal("Messages.New: want error for upstream 400, got nil")
	}
	if !strings.Contains(err.Error(), "max_tokens too big") {
		t.Errorf("error %q does not carry the upstream error body", err)
	}
}

// A chat-only backend, answering in Chat Completions shape.
const chatCompletionJSON = `{"id":"chatcmpl-1","model":"` + targetModel + `",` +
	`"object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant",` +
	`"content":"Bonjour from fake backend"},"finish_reason":"stop"}],` +
	`"usage":{"prompt_tokens":10,"completion_tokens":5}}`

var chatStreamFrames = []string{
	"data: {\"id\":\"chatcmpl-1\",\"model\":\"" + targetModel +
		"\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0," +
		"\"delta\":{\"role\":\"assistant\",\"content\":\"Bon\"}}]}\n\n",
	"data: {\"id\":\"chatcmpl-1\",\"model\":\"" + targetModel +
		"\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0," +
		"\"delta\":{\"content\":\"jour\"},\"finish_reason\":\"stop\"}]," +
		"\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n",
	"data: [DONE]\n\n",
}

// The phase-2 promise, checked by the official SDK: an Anthropic client, a
// chat-only backend, and the SDK's own JSON decoding accepts the translated
// answer — including the model name rewritten back to the client's.
func TestAnthropicMessagesJSONServedByChatBackend(t *testing.T) {
	st := newStack(t, jsonBackend(http.StatusOK, chatCompletionJSON))
	st.addRule([]string{anthropicModel}, targetModel)
	st.declareProtocols("/v1/chat/completions")

	resp, err := st.anthropic.Messages.New(context.Background(), anthropicParams())
	if err != nil {
		t.Fatalf("Messages.New: %v", err)
	}
	req, body := st.lastRequest()
	if req == nil {
		t.Fatal("fake backend saw no request")
	}
	if req.URL.Path != "/v1/chat/completions" {
		t.Errorf("upstream path = %q, want /v1/chat/completions", req.URL.Path)
	}
	if !strings.Contains(string(body), `"max_tokens":100`) {
		t.Errorf("client max_tokens not carried into the chat request: %s", body)
	}
	if resp.Model != anthropicModel {
		t.Errorf("Model = %q, want %q (response rewrite)", resp.Model, anthropicModel)
	}
	if len(resp.Content) != 1 {
		t.Fatalf("content blocks = %d, want 1: %+v", len(resp.Content), resp.Content)
	}
	if got := resp.Content[0].AsText().Text; got != "Bonjour from fake backend" {
		t.Errorf("content = %q", got)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("stop_reason = %q, want end_turn", resp.StopReason)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 5 {
		t.Errorf("usage = in %d out %d, want 10/5", resp.Usage.InputTokens, resp.Usage.OutputTokens)
	}

	m := st.lastMetric()
	if m.Convert != "messages>chat" || m.UpstreamEndpoint != "/v1/chat/completions" {
		t.Errorf("metric convert = %q upstream = %q, want messages>chat / /v1/chat/completions",
			m.Convert, m.UpstreamEndpoint)
	}
	// Usage is read from the native bytes, above the translator.
	if m.PromptTokens != 10 || m.CompletionTokens != 5 {
		t.Errorf("metric tokens = %d/%d, want 10/5", m.PromptTokens, m.CompletionTokens)
	}
}

// The stream half: every frame the emitter authored must survive
// `Message.Accumulate`, and `stream.Err()` is the gate that says the SDK parsed
// them — a frame with a wrong event name or a missing field fails there.
func TestAnthropicStreamServedByChatBackend(t *testing.T) {
	st := newStack(t, sseBackend(chatStreamFrames...))
	st.addRule([]string{anthropicModel}, targetModel)
	st.declareProtocols("/v1/chat/completions")

	stream := st.anthropic.Messages.NewStreaming(context.Background(), anthropicParams())
	msg := accumulateStream(t, stream)

	if msg.Model != anthropicModel {
		t.Errorf("streamed Model = %q, want %q", msg.Model, anthropicModel)
	}
	if msg.ID != "chatcmpl-1" {
		t.Errorf("streamed ID = %q, want chatcmpl-1", msg.ID)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("content blocks = %d, want 1: %+v", len(msg.Content), msg.Content)
	}
	if got := msg.Content[0].AsText().Text; got != "Bonjour" {
		t.Errorf("streamed content = %q, want Bonjour", got)
	}
	if msg.StopReason != "end_turn" {
		t.Errorf("stop_reason = %q, want end_turn", msg.StopReason)
	}
	if msg.Usage.OutputTokens != 5 {
		t.Errorf("output_tokens = %d, want 5 (from message_delta)", msg.Usage.OutputTokens)
	}
	if msg.Usage.InputTokens != 10 {
		t.Errorf("input_tokens = %d, want 10 (restated in message_delta)", msg.Usage.InputTokens)
	}
}

// A backend that dies mid-generation must surface an error to the SDK client.
// Before the fix, the proxy appended an OpenAI-style bare data: frame, which
// the Anthropic SDK silently drops (unrecognized event type) — the client saw
// a truncated message with stream.Err() == nil.
func TestAnthropicMidStreamError(t *testing.T) {
	backend := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		fmt.Fprint(w, anthropicLlamaCppFrames[0])
		fl.Flush()
		fmt.Fprint(w, anthropicLlamaCppFrames[1])
		fl.Flush()
		// Kill the connection without terminating the chunked response.
		if conn, _, err := w.(http.Hijacker).Hijack(); err == nil {
			conn.Close()
		}
	})
	st := newStack(t, backend)
	st.addRule([]string{anthropicModel}, targetModel)

	stream := st.anthropic.Messages.NewStreaming(context.Background(), anthropicParams())
	msg := &anthropic.Message{}
	for stream.Next() {
		_ = msg.Accumulate(stream.Current())
	}
	if stream.Err() == nil {
		t.Fatal("stream ended without error after upstream disconnect (mid-stream error invisible to the SDK client)")
	}
	if len(msg.Content) != 1 {
		t.Fatalf("content blocks = %d, want 1", len(msg.Content))
	}
	// ContentBlockUnion.Text holds the accumulated deltas; AsText() would read
	// the block's raw wire JSON, which is only refreshed by the stop events a
	// cut-off stream never delivers.
	if got := msg.Content[0].Text; got != "Bon" {
		t.Errorf("partial content = %q, want %q", got, "Bon")
	}
}

// streamReport is what an official Anthropic client actually received from a
// converted stream: the events its SDK dispatched, the message Accumulate built
// from them, and what stream.Err() says. A frame the SDK's decoder skips is
// simply absent from Events — that absence, next to the bytes the proxy really
// wrote, is the evidence.
type streamReport struct {
	Events []string
	Msg    anthropic.Message
	Err    error
}

func (r streamReport) text() string {
	var out string
	for _, b := range r.Msg.Content {
		out += b.Text
	}
	return out
}

// readStream drives the SDK's documented stream consumer by hand (accumulateStream
// fatals on a stream error, this must report it) and reports the observation.
func readStream(st *stack) streamReport {
	var rep streamReport
	stream := st.anthropic.Messages.NewStreaming(context.Background(), anthropicParams())
	for stream.Next() {
		ev := stream.Current()
		rep.Events = append(rep.Events, ev.Type)
		if err := rep.Msg.Accumulate(ev); err != nil {
			rep.Err = fmt.Errorf("Accumulate(%s): %w", ev.Type, err)
			break
		}
	}
	if rep.Err == nil {
		rep.Err = stream.Err()
	}
	return rep
}

// PARKED ITEM (task review of the streaming emitter), resolved at the client layer:
// the emitter's fail-open used to forward a data: frame it could not read as a
// chat chunk VERBATIM, and an unlabelled data: frame is exactly what the Anthropic
// SDK ignores (packages/ssestream drops every event with no name,
// `case "": continue`), so a client was handed a closed, complete-looking message
// whose content was missing that piece — the truncated answer with no error the
// spec forbids. The emitter now classifies frames by dialect (R22): a frame in no
// dialect it speaks ENDS the stream with the client's own `event: error`, which
// this SDK raises. Pinned here as the client's view: the raw frame never reaches
// the wire, the SDK reports an error, and the envelope is not closed as complete.
func TestAnthropicStreamServedByChatBackendUnparseableFrame(t *testing.T) {
	// The tail of the answer, in frames the emitter cannot read.
	const foreignTailFrame = `data: {"type":"response.output_text.delta","item_id":"msg_1",` +
		`"output_index":0,"content_index":0,"delta":"jour"}` + "\n\n"

	t.Run("garbage json", func(t *testing.T) {
		st := newStack(t, sseBackend(
			chatStreamFrames[0],
			"data: the worker died here, and this is not JSON\n\n",
			chatStreamFrames[1],
			chatStreamFrames[2],
		))
		st.addRule([]string{anthropicModel}, targetModel)
		st.declareProtocols("/v1/chat/completions")

		rep := readStream(st)
		t.Logf("events=%v text=%q stop=%q err=%v", rep.Events, rep.text(),
			rep.Msg.StopReason, rep.Err)
		// The frame no longer rides through: an unlabelled data: frame is dropped on
		// the client's floor, so forwarding it WAS the silent truncation.
		if got := string(st.clientBytes()); strings.Contains(got, "the worker died here") {
			t.Errorf("the emitter forwarded a frame in no dialect it speaks:\n%s", got)
		}
		if rep.Err == nil {
			t.Fatal("stream.Err = nil, want the failure reported (a client must not read a cut stream as a complete answer)")
		}
		if slices.Contains(rep.Events, "message_stop") {
			t.Errorf("an error event is terminal, the emitter closed the message anyway: %v", rep.Events)
		}
		if got := rep.text(); got != "Bon" {
			t.Errorf("text = %q, want Bon (the stream ended at the bad frame)", got)
		}
	})

	t.Run("foreign frame carries the answer", func(t *testing.T) {
		st := newStack(t, sseBackend(
			chatStreamFrames[0],
			foreignTailFrame,
			chatStreamFrames[2],
		))
		st.addRule([]string{anthropicModel}, targetModel)
		st.declareProtocols("/v1/chat/completions")

		rep := readStream(st)
		t.Logf("events=%v text=%q stop=%q err=%v", rep.Events, rep.text(),
			rep.Msg.StopReason, rep.Err)
		if got := string(st.clientBytes()); strings.Contains(got, "response.output_text.delta") {
			t.Errorf("the emitter forwarded a frame in no dialect it speaks:\n%s", got)
		}
		// VERDICT (the spec's refusals item 3): the answer's tail rides a frame the
		// converter cannot assemble, so the content IS lost — what must not be lost
		// is the fact. The client's dialect ends the stream with an error event, the
		// SDK raises it, and no normal close follows the hole.
		if rep.Err == nil {
			t.Fatal("stream.Err = nil, want the upstream failure (the answer's tail was untranslatable)")
		}
		if slices.Contains(rep.Events, "message_stop") {
			t.Errorf("the truncated answer was reported as complete: %v", rep.Events)
		}
		if got := rep.text(); got != "Bon" {
			t.Errorf("text = %q, want Bon (the tail rode the untranslatable frame)", got)
		}
	})

	// A multimodal chat backend streams delta.content as a CONTENT-PARTS ARRAY. The
	// emitter used to decide "is a chunk" by the inner field types, so every chunk
	// of the answer failed its decode, went out native, and this SDK dropped them
	// all: a message closed as a normal end_turn with no content at all.
	t.Run("content-parts delta is translated", func(t *testing.T) {
		st := newStack(t, sseBackend(
			`data: {"id":"chatcmpl-1","model":"`+targetModel+`","object":"chat.completion.chunk",`+
				`"choices":[{"index":0,"delta":{"role":"assistant",`+
				`"content":[{"type":"text","text":"Bon"}]}}]}`+"\n\n",
			`data: {"id":"chatcmpl-1","model":"`+targetModel+`","object":"chat.completion.chunk",`+
				`"choices":[{"index":0,"delta":{"content":[{"type":"text","text":"jour"}]},`+
				`"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`+"\n\n",
			"data: [DONE]\n\n",
		))
		st.addRule([]string{anthropicModel}, targetModel)
		st.declareProtocols("/v1/chat/completions")

		rep := readStream(st)
		t.Logf("events=%v text=%q stop=%q err=%v", rep.Events, rep.text(),
			rep.Msg.StopReason, rep.Err)
		if rep.Err != nil {
			t.Fatalf("stream.Err = %v, want nil", rep.Err)
		}
		if got := rep.text(); got != "Bonjour" {
			t.Errorf("text = %q, want Bonjour (the answer went out untranslated)", got)
		}
		if !slices.Contains(rep.Events, "message_stop") {
			t.Errorf("events = %v, want the message closed normally", rep.Events)
		}
	})

	t.Run("openai error object is translated", func(t *testing.T) {
		st := newStack(t, sseBackend(
			chatStreamFrames[0],
			`data: {"error":{"type":"server_error","message":"worker died"}}`+"\n\n",
			chatStreamFrames[1],
			chatStreamFrames[2],
		))
		st.addRule([]string{anthropicModel}, targetModel)
		st.declareProtocols("/v1/chat/completions")

		rep := readStream(st)
		t.Logf("events=%v text=%q stop=%q err=%v", rep.Events, rep.text(),
			rep.Msg.StopReason, rep.Err)
		// The one shape among these the emitter does translate: an in-stream
		// `error` object becomes the client's dialect's error event, the SDK
		// raises it, and no normal close follows.
		if rep.Err == nil {
			t.Fatal("stream.Err = nil, want the upstream error (an Anthropic client read a truncated answer as complete)")
		}
		if !strings.Contains(rep.Err.Error(), "worker died") {
			t.Errorf("error %q lost the upstream message", rep.Err)
		}
		if slices.Contains(rep.Events, "message_stop") {
			t.Errorf("an error event is terminal, the emitter closed the message anyway: %v", rep.Events)
		}
	})
}
