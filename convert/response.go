package convert

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

// Response translates a non-streamed answer from the dialect the backend answered
// in to the dialect the client asked in — the reverse of the plan's request
// direction, so the spoke test is `Dialect == Chat && From == Messages`. An
// upstream error envelope is translated too (the proxy forwards 4xx/5xx bodies
// as-is, and an Anthropic SDK can not read OpenAI's shape).
func Response(p Plan, body []byte) ([]byte, []string, error) {
	if p.Dialect == p.From || p.From == "" {
		return body, nil, nil
	}
	if p.Dialect == Chat && p.From == Messages {
		if b, ok, err := chatErrorToMessages(body); ok || err != nil {
			return b, nil, err
		}
		return ChatToMessages(body)
	}
	return nil, nil, &Error{Code: CodeNoDialect, Msg: "response conversion " + p.Edge() + " is not supported"}
}

// chatErrorToMessages rewrites an OpenAI error envelope into Anthropic's. ok is
// false when the body is not one, leaving it to the answer path. `type` is
// copied when present (invalid_request_error / rate_limit_exceeded / …), else
// api_error, the Anthropic SDK's catch-all.
func chatErrorToMessages(body []byte) ([]byte, bool, error) {
	var in struct {
		Error *struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &in) != nil || in.Error == nil {
		return nil, false, nil
	}
	typ := in.Error.Type
	if typ == "" {
		typ = "api_error"
	}
	b, err := json.Marshal(map[string]any{"type": "error",
		"error": map[string]any{"type": typ, "message": in.Error.Message}})
	return b, true, err
}

type cResponse struct {
	ID      string    `json:"id"`
	Model   string    `json:"model"`
	Choices []cChoice `json:"choices"`
	Usage   cUsage    `json:"usage"`
}

type cChoice struct {
	Message struct {
		Content   json.RawMessage `json:"content"`
		Reasoning string          `json:"reasoning_content"`
		ToolCalls []cToolCall     `json:"tool_calls"`
	} `json:"message"`
	FinishReason string `json:"finish_reason"`
}

type cToolCall struct {
	ID       string `json:"id"`
	Function struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

type cUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	PromptDetails    struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

func ChatToMessages(body []byte) ([]byte, []string, error) {
	var c cResponse
	if err := json.Unmarshal(body, &c); err != nil {
		return nil, nil, &Error{Code: CodeField, Field: "body", Msg: "chat response is not JSON: " + err.Error()}
	}
	if len(c.Choices) == 0 {
		return nil, nil, &Error{Code: CodeField, Field: "choices", Msg: "chat response carries no choices"}
	}
	ch := c.Choices[0]
	blocks := make([]any, 0, 4)
	if ch.Message.Reasoning != "" {
		blocks = append(blocks, map[string]any{"type": "thinking", "thinking": ch.Message.Reasoning})
	}
	blocks = append(blocks, contentBlocks(ch.Message.Content)...)
	for i, tc := range ch.Message.ToolCalls {
		id := tc.ID
		if id == "" {
			id = "toolu_" + shortHash(fmt.Sprintf("%s:%d", c.ID, i))
		}
		blocks = append(blocks, map[string]any{"type": "tool_use", "id": id,
			"name": tc.Function.Name, "input": toolArguments(tc.Function.Arguments)})
	}
	out := map[string]any{
		"id":            c.ID,
		"type":          "message",
		"role":          "assistant",
		"model":         c.Model,
		"content":       blocks,
		"stop_reason":   stopReason(ch.FinishReason),
		"stop_sequence": nil,
		"usage": map[string]any{
			// OpenAI's prompt_tokens already contains cached tokens; Anthropic's
			// input_tokens sits next to cache_read_input_tokens. Adding the two
			// conventions is the double count the metrics layer fights.
			"input_tokens":            max(0, c.Usage.PromptTokens-c.Usage.PromptDetails.CachedTokens),
			"output_tokens":           c.Usage.CompletionTokens,
			"cache_read_input_tokens": c.Usage.PromptDetails.CachedTokens,
		},
	}
	b, err := json.Marshal(out)
	if err != nil {
		return nil, nil, &Error{Code: CodeField, Field: "body", Msg: "encode: " + err.Error()}
	}
	return b, nil, nil
}

func stopReason(fr string) string {
	switch fr {
	case "length":
		return "max_tokens"
	case "tool_calls", "function_call":
		return "tool_use"
	case "content_filter":
		return "refusal"
	}
	return "end_turn"
}

// contentBlocks accepts a Chat Completions content string, an array of parts, or
// nothing, and returns Anthropic content blocks.
func contentBlocks(raw json.RawMessage) []any {
	if len(raw) == 0 {
		return nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		if s == "" {
			return nil
		}
		return []any{map[string]any{"type": "text", "text": s}}
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) != nil {
		return nil
	}
	out := make([]any, 0, len(parts))
	for _, p := range parts {
		if p.Text != "" {
			out = append(out, map[string]any{"type": "text", "text": p.Text})
		}
	}
	return out
}

// toolArguments decodes a Chat Completions `arguments` field into an Anthropic
// tool_use `input`. On the wire that field is a JSON-ENCODED STRING
// ("arguments": "{\"u\":1}"), so the first decode unwraps the string and the
// second parses the object inside it; a backend that already sends an object
// takes the same path. Arguments that parse to nothing become `{}`: Anthropic's
// `input` is an object, and a tool_use block with a string input is not one.
func toolArguments(raw json.RawMessage) any {
	if len(raw) == 0 {
		return map[string]any{}
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		raw = []byte(s)
	}
	var input any
	if json.Unmarshal(raw, &input) != nil {
		return map[string]any{}
	}
	return input
}

func shortHash(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])[:20]
}
