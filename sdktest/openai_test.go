package sdktest

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3"

	"llm-api-router/domain"
)

// llama.cpp/vllm responses carry provider-specific extra fields (llama.cpp
// native `timings`, vLLM `served_model` and `metrics`) that official SDKs
// must tolerate.
const openAIChatJSON = `{
	"id": "chatcmpl-fake-1",
	"object": "chat.completion",
	"created": 1700000000,
	"model": "` + targetModel + `",
	"choices": [
		{"index": 0, "message": {"role": "assistant", "content": "Hello from fake backend"}, "finish_reason": "stop"}
	],
	"usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
	"timings": {"prompt_n": 7, "predicted_n": 5, "prompt_per_second": 250.5, "predicted_per_second": 42.25, "prompt_ms": 123.4, "predicted_ms": 567.8, "cache_n": 4},
	"served_model": "served-by-vllm",
	"metrics": {"time_to_first_token_ms": 12.5, "generation_time_ms": 100.25, "queue_time_ms": 2.0, "mean_itl_ms": 25.0, "tokens_per_second": 48.0}
}`

func oaChunk(delta, finishReason string) string {
	fr := "null"
	if finishReason != "" {
		fr = `"` + finishReason + `"`
	}
	return "data: {\"id\":\"chatcmpl-fake-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"" + targetModel + "\",\"choices\":[{\"index\":0,\"delta\":" + delta + ",\"finish_reason\":" + fr + "}]}\n\n"
}

const oaUsageChunk = "data: {\"id\":\"chatcmpl-fake-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"" + targetModel + "\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":5,\"total_tokens\":12}}\n\n"

const oaDoneChunk = "data: [DONE]\n\n"

// openAIStreamBackend behaves like a real OpenAI-compatible backend: it emits
// the standard chunk sequence, and the trailing usage chunk only when the
// request set stream_options.include_usage (llama.cpp / vLLM both do).
func openAIStreamBackend() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req struct {
			StreamOptions struct {
				IncludeUsage bool `json:"include_usage"`
			} `json:"stream_options"`
		}
		_ = json.Unmarshal(body, &req)

		frames := []string{
			oaChunk(`{"role":"assistant","content":"Hel"}`, ""),
			oaChunk(`{"content":"lo"}`, ""),
			oaChunk(`{"content":" world"}`, ""),
			oaChunk(`{}`, "stop"),
		}
		if req.StreamOptions.IncludeUsage {
			frames = append(frames, oaUsageChunk)
		}
		frames = append(frames, oaDoneChunk)
		sseBackend(frames...).ServeHTTP(w, r)
	})
}

func TestOpenAIChatJSON(t *testing.T) {
	st := newStack(t, jsonBackend(http.StatusOK, openAIChatJSON))
	st.addRule([]string{incomingModel}, targetModel)

	resp, err := st.openai.Chat.Completions.New(context.Background(), openai.ChatCompletionNewParams{
		Model:    incomingModel,
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("ChatCompletions.New: %v", err)
	}
	st.assertUpstream(t, targetModel)

	if resp.Model != incomingModel {
		t.Errorf("Model = %q, want %q (response rewrite back to client's name)", resp.Model, incomingModel)
	}
	if got := resp.Choices[0].Message.Content; got != "Hello from fake backend" {
		t.Errorf("content = %q, want %q", got, "Hello from fake backend")
	}
	if resp.Usage.PromptTokens != 7 || resp.Usage.CompletionTokens != 5 || resp.Usage.TotalTokens != 12 {
		t.Errorf("usage = %+v, want 7/5/12", resp.Usage)
	}

	// Proxy-side metric capture must see tokens and the llama.cpp native
	// timings extension, and record a success.
	m := st.lastMetric()
	if m.StatusCode != 200 {
		t.Errorf("metric status = %d, want 200", m.StatusCode)
	}
	if m.PromptTokens != 7 || m.CompletionTokens != 5 || m.TotalTokens != 12 {
		t.Errorf("metric tokens = %d/%d/%d, want 7/5/12", m.PromptTokens, m.CompletionTokens, m.TotalTokens)
	}
	if m.CachedTokens != 4 {
		t.Errorf("metric cached_tokens = %d, want 4 (llama.cpp timings.cache_n)", m.CachedTokens)
	}
	if m.NativePromptTokPerSec != 250.5 || m.NativeDecodeTokPerSec != 42.25 {
		t.Errorf("metric native timings = %.2f/%.2f, want 250.50/42.25", m.NativePromptTokPerSec, m.NativeDecodeTokPerSec)
	}
}

func TestOpenAIChatStream(t *testing.T) {
	st := newStack(t, openAIStreamBackend())
	st.addRule([]string{incomingModel}, targetModel)

	params := openai.ChatCompletionNewParams{
		Model:    incomingModel,
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
	}

	stream := st.openai.Chat.Completions.NewStreaming(context.Background(), params)

	var sb strings.Builder
	finish := ""
	emptyChoiceChunk := false
	for stream.Next() {
		chunk := stream.Current()
		if chunk.Model != incomingModel {
			t.Fatalf("chunk Model = %q, want %q (stream rewrite)", chunk.Model, incomingModel)
		}
		if len(chunk.Choices) == 0 {
			emptyChoiceChunk = true // the injected usage artifact leaked to the client
			continue
		}
		sb.WriteString(chunk.Choices[0].Delta.Content)
		if chunk.Choices[0].FinishReason != "" {
			finish = chunk.Choices[0].FinishReason
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream.Err: %v", err)
	}

	st.assertUpstream(t, targetModel)
	if got := sb.String(); got != "Hello world" {
		t.Errorf("streamed content = %q, want %q", got, "Hello world")
	}
	if finish != "stop" {
		t.Errorf("finish_reason = %q, want %q", finish, "stop")
	}
	if emptyChoiceChunk {
		t.Error("injected stream-usage chunk (empty choices) was not stripped from the client stream")
	}

	// The proxy injects include_usage into the upstream request for metrics.
	if _, body := st.lastRequest(); !strings.Contains(string(body), `"include_usage":true`) {
		t.Error("proxy did not inject stream_options.include_usage into the upstream request")
	}

	m := st.lastMetric()
	if m.PromptTokens != 7 || m.CompletionTokens != 5 {
		t.Errorf("metric stream tokens = %d/%d, want 7/5", m.PromptTokens, m.CompletionTokens)
	}
}

func TestOpenAIChatStreamIncludeUsage(t *testing.T) {
	st := newStack(t, openAIStreamBackend())
	st.addRule([]string{incomingModel}, targetModel)

	params := openai.ChatCompletionNewParams{
		Model: incomingModel,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("hi"),
		},
	}
	params.StreamOptions.IncludeUsage = openai.Bool(true) // client opts in: usage chunk must reach the client

	stream := st.openai.Chat.Completions.NewStreaming(context.Background(), params)
	var usage openai.CompletionUsage
	for stream.Next() {
		chunk := stream.Current()
		if chunk.Usage.TotalTokens != 0 {
			usage = chunk.Usage
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream.Err: %v", err)
	}
	if usage.TotalTokens != 12 {
		t.Errorf("client-visible stream usage = %+v, want total 12", usage)
	}
}

func TestOpenAIChatStreamToolCalls(t *testing.T) {
	frames := []string{
		oaChunk(`{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}`, ""),
		oaChunk(`{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc"}}]}`, ""),
		oaChunk(`{"tool_calls":[{"index":0,"function":{"arguments":"ation\":\"Paris\"}"}}]}`, ""),
		oaChunk(`{}`, "tool_calls"),
		oaDoneChunk,
	}
	st := newStack(t, sseBackend(frames...))
	st.addRule([]string{incomingModel}, targetModel)

	params := openai.ChatCompletionNewParams{
		Model:    incomingModel,
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("weather?")},
	}

	stream := st.openai.Chat.Completions.NewStreaming(context.Background(), params)
	var args strings.Builder
	toolName, finish := "", ""
	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) == 0 {
			continue
		}
		for _, tc := range chunk.Choices[0].Delta.ToolCalls {
			// The name/id ride only on the first tool_call delta; later
			// argument-fragment deltas carry empty fields.
			if tc.Function.Name != "" {
				toolName = tc.Function.Name
			}
			args.WriteString(tc.Function.Arguments)
		}
		if chunk.Choices[0].FinishReason != "" {
			finish = chunk.Choices[0].FinishReason
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream.Err: %v", err)
	}
	if toolName != "get_weather" {
		t.Errorf("tool name = %q, want %q", toolName, "get_weather")
	}
	if finish != "tool_calls" {
		t.Errorf("finish_reason = %q, want %q", finish, "tool_calls")
	}
	var call struct {
		Location string `json:"location"`
	}
	if err := json.Unmarshal([]byte(args.String()), &call); err != nil {
		t.Fatalf("tool arguments %q are not valid JSON: %v", args.String(), err)
	}
	if call.Location != "Paris" {
		t.Errorf("tool argument location = %q, want %q", call.Location, "Paris")
	}
}

// A backend that dies mid-generation appends an OpenAI-style error chunk to
// the stream; the SDK client must parse it as a regular chunk carrying the
// error note.
func TestOpenAIMidStreamError(t *testing.T) {
	backend := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		fmt.Fprint(w, oaChunk(`{"role":"assistant","content":"Par"}`, ""))
		fl.Flush()
		// Kill the connection without terminating the chunked response.
		if conn, _, err := w.(http.Hijacker).Hijack(); err == nil {
			conn.Close()
		}
	})
	st := newStack(t, backend)
	st.addRule([]string{incomingModel}, targetModel)

	params := openai.ChatCompletionNewParams{
		Model:    incomingModel,
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
	}
	stream := st.openai.Chat.Completions.NewStreaming(context.Background(), params)
	var sb strings.Builder
	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) == 0 {
			continue
		}
		sb.WriteString(chunk.Choices[0].Delta.Content)
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream.Err: %v", err)
	}
	if !strings.Contains(sb.String(), "[error:") {
		t.Errorf("client did not receive the appended mid-stream error event, got %q", sb.String())
	}
}

func TestOpenAIListModels(t *testing.T) {
	st := newStack(t, nil) // /v1/models is served by the router, not the backend
	st.addRule([]string{incomingModel, "gpt-test-2"}, targetModel)
	// Disabled rules must not be listed.
	if err := st.store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"disabled-model"},
		TargetModel:    targetModel,
		ServerID:       "srv1",
		Enabled:        false,
	}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}

	page, err := st.openai.Models.List(context.Background())
	if err != nil {
		t.Fatalf("Models.List: %v", err)
	}
	var ids []string
	for _, m := range page.Data {
		ids = append(ids, m.ID)
	}
	want := map[string]bool{incomingModel: false, "gpt-test-2": false}
	for _, id := range ids {
		if id == "disabled-model" {
			t.Errorf("disabled model %q was listed", id)
		}
		if _, ok := want[id]; ok {
			want[id] = true
		}
	}
	for id, seen := range want {
		if !seen {
			t.Errorf("/v1/models via SDK missing model %q (got %v)", id, ids)
		}
	}
}
