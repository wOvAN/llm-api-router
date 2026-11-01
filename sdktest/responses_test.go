package sdktest

import (
	"context"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/responses"
)

// The phase-4 promise, checked by the official SDK: an OpenAI Responses client, a
// chat-only backend, and the SDK's own decoders accept the translated answer —
// the response object for JSON, the event stream for a generation.

func responsesParams() responses.ResponseNewParams {
	return responses.ResponseNewParams{
		Model:           incomingModel,
		Input:           responses.ResponseNewParamsInputUnion{OfString: openai.String("Bonjour")},
		MaxOutputTokens: openai.Int(100),
	}
}

func TestOpenAIResponsesJSONServedByChatBackend(t *testing.T) {
	st := newStack(t, jsonBackend(http.StatusOK, chatCompletionJSON))
	st.addRule([]string{incomingModel}, targetModel)
	st.declareProtocols("/v1/chat/completions")

	resp, err := st.openai.Responses.New(context.Background(), responsesParams())
	if err != nil {
		t.Fatalf("Responses.New: %v", err)
	}
	req, body := st.lastRequest()
	if req == nil {
		t.Fatal("fake backend saw no request")
	}
	if req.URL.Path != "/v1/chat/completions" {
		t.Errorf("upstream path = %q, want /v1/chat/completions", req.URL.Path)
	}
	if !strings.Contains(string(body), `"max_tokens":100`) {
		t.Errorf("client max_output_tokens not carried into the chat request: %s", body)
	}
	if !strings.Contains(string(body), "Bonjour") {
		t.Errorf("client input not carried into the chat request: %s", body)
	}

	if resp.ID != "chatcmpl-1" {
		t.Errorf("ID = %q, want chatcmpl-1", resp.ID)
	}
	if resp.Model != incomingModel {
		t.Errorf("Model = %q, want %q (response rewrite)", resp.Model, incomingModel)
	}
	if resp.Status != "completed" {
		t.Errorf("status = %q, want completed", resp.Status)
	}
	if len(resp.Output) != 1 {
		t.Fatalf("output items = %d, want 1: %+v", len(resp.Output), resp.Output)
	}
	if got := resp.Output[0].Content[0].Text; got != "Bonjour from fake backend" {
		t.Errorf("output text = %q", got)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 5 {
		t.Errorf("usage = in %d out %d, want 10/5", resp.Usage.InputTokens, resp.Usage.OutputTokens)
	}

	m := st.lastMetric()
	if m.Convert != "responses>chat" || m.UpstreamEndpoint != "/v1/chat/completions" {
		t.Errorf("metric convert = %q upstream = %q, want responses>chat / /v1/chat/completions",
			m.Convert, m.UpstreamEndpoint)
	}
	// Usage is read from the native bytes, above the translator.
	if m.PromptTokens != 10 || m.CompletionTokens != 5 {
		t.Errorf("metric tokens = %d/%d, want 10/5", m.PromptTokens, m.CompletionTokens)
	}
}

// The stream half: every event the emitter authors must decode as the SDK's own
// event union, and the terminal event must carry the whole response object
// (usage included) — that is what a Responses client reads as the answer.
func TestOpenAIResponsesStreamServedByChatBackend(t *testing.T) {
	st := newStack(t, sseBackend(chatStreamFrames...))
	st.addRule([]string{incomingModel}, targetModel)
	st.declareProtocols("/v1/chat/completions")

	events, terminal, text := collectResponses(t,
		st.openai.Responses.NewStreaming(context.Background(), responsesParams()))

	for _, want := range []string{"response.created", "response.in_progress",
		"response.output_item.added", "response.content_part.added",
		"response.output_text.delta", "response.output_text.done",
		"response.content_part.done", "response.output_item.done", "response.completed"} {
		if !slices.Contains(events, want) {
			t.Errorf("stream lacks %s: %v", want, events)
		}
	}
	if text != "Bonjour" {
		t.Errorf("streamed text = %q, want Bonjour", text)
	}
	if terminal == nil {
		t.Fatal("no terminal event reached the SDK client")
	}
	if terminal.Status != "completed" {
		t.Errorf("terminal status = %q, want completed", terminal.Status)
	}
	if terminal.Model != incomingModel {
		t.Errorf("terminal model = %q, want %q (response rewrite)", terminal.Model, incomingModel)
	}
	if terminal.ID != "chatcmpl-1" {
		t.Errorf("terminal id = %q, want chatcmpl-1", terminal.ID)
	}
	if len(terminal.Output) != 1 || terminal.Output[0].Content[0].Text != "Bonjour" {
		t.Fatalf("terminal output = %+v, want the message item with the answer", terminal.Output)
	}
	if terminal.Usage.InputTokens != 10 || terminal.Usage.OutputTokens != 5 {
		t.Errorf("terminal usage = in %d out %d, want 10/5 (restated on the terminal event)",
			terminal.Usage.InputTokens, terminal.Usage.OutputTokens)
	}
	if !strings.Contains(string(st.clientBytes()), "data: [DONE]") {
		t.Error("the stream was not terminated with [DONE]")
	}

	m := st.lastMetric()
	if m.Convert != "responses>chat" {
		t.Errorf("metric convert = %q, want responses>chat", m.Convert)
	}
	if m.PromptTokens != 10 || m.CompletionTokens != 5 {
		t.Errorf("metric tokens = %d/%d, want 10/5", m.PromptTokens, m.CompletionTokens)
	}
}

// The agent loop is the shape that matters most on this dialect: the call the
// client answers next is keyed by the item's `call_id`, and an answer that loses
// it strands the loop.
func TestOpenAIResponsesStreamToolCallsServedByChatBackend(t *testing.T) {
	st := newStack(t, sseBackend(
		"data: {\"id\":\"chatcmpl-1\",\"model\":\""+targetModel+"\",\"object\":\"chat.completion.chunk\","+
			"\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\","+
			"\"function\":{\"name\":\"fetch\",\"arguments\":\"{\\\"a\\\":1}\"}}]}}]}\n\n",
		"data: {\"id\":\"chatcmpl-1\",\"model\":\""+targetModel+"\",\"object\":\"chat.completion.chunk\","+
			"\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
		"data: [DONE]\n\n",
	))
	st.addRule([]string{incomingModel}, targetModel)
	st.declareProtocols("/v1/chat/completions")

	events, terminal, _ := collectResponses(t,
		st.openai.Responses.NewStreaming(context.Background(), responsesParams()))

	for _, want := range []string{"response.output_item.added",
		"response.function_call_arguments.delta", "response.function_call_arguments.done",
		"response.completed"} {
		if !slices.Contains(events, want) {
			t.Errorf("stream lacks %s: %v", want, events)
		}
	}
	if terminal == nil {
		t.Fatal("no terminal event reached the SDK client")
	}
	var call *responses.ResponseOutputItemUnion
	for i := range terminal.Output {
		if terminal.Output[i].Type == "function_call" {
			call = &terminal.Output[i]
		}
	}
	if call == nil {
		t.Fatalf("terminal output lost the function call: %+v", terminal.Output)
	}
	if call.Name != "fetch" || call.CallID != "call_1" {
		t.Errorf("call = %q/%q, want fetch/call_1", call.Name, call.CallID)
	}
	if got := call.Arguments.OfString; got != `{"a":1}` {
		t.Errorf("arguments = %q, want {\"a\":1}", got)
	}
}

// collectResponses iterates a Responses stream the way the SDK documents, and
// reports the event names it saw, the terminal event's response object, and the
// output text the delta events streamed.
func collectResponses(t *testing.T,
	stream *ssestream.Stream[responses.ResponseStreamEventUnion],
) (events []string, terminal *responses.Response, text string) {
	t.Helper()
	var prose strings.Builder
	for stream.Next() {
		ev := stream.Current()
		events = append(events, ev.Type)
		if ev.Type == "response.output_text.delta" {
			prose.WriteString(ev.Delta)
		}
		if ev.Type == "response.completed" || ev.Type == "response.incomplete" ||
			ev.Type == "response.failed" {
			terminal = &ev.Response
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream.Err: %v\nevents: %v", err, events)
	}
	return events, terminal, prose.String()
}
