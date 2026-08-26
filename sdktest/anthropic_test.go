package sdktest

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
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
