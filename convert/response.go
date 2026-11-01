package convert

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
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
// api_error, the Anthropic SDK's catch-all. The `error` field arrives both as an
// object (`{"error":{"message":…}}`) and as a bare string
// (`{"error":"upstream timeout"}`) — a backend's shorthand, and the one shape a
// mid-stream frame can take.
func chatErrorToMessages(body []byte) ([]byte, bool, error) {
	var in struct {
		Error json.RawMessage `json:"error"`
	}
	if json.Unmarshal(body, &in) != nil || len(in.Error) == 0 {
		return nil, false, nil
	}
	typ, msg := "api_error", ""
	var obj struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	}
	switch {
	case json.Unmarshal(in.Error, &obj) == nil && (obj.Type != "" || obj.Message != ""):
		msg = obj.Message
		if obj.Type != "" {
			typ = obj.Type
		}
	case json.Unmarshal(in.Error, &msg) != nil || msg == "":
		return nil, false, nil
	}
	b, err := json.Marshal(map[string]any{"type": "error",
		"error": map[string]any{"type": typ, "message": msg}})
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
	// An `n>1` answer is one Anthropic message: the extra choices have nowhere to
	// go, so the count is reported (the spec's mapping table maps the answer, and
	// notes are the only channel a mapped response has).
	var notes []string
	if len(c.Choices) > 1 {
		notes = append(notes, "choices=dropped:"+strconv.Itoa(len(c.Choices)-1))
	}
	ch := c.Choices[0]
	if n := finishNote(ch.FinishReason); n != "" {
		notes = append(notes, n)
	}
	blocks := make([]any, 0, 4)
	if ch.Message.Reasoning != "" {
		blocks = append(blocks, map[string]any{"type": "thinking", "thinking": ch.Message.Reasoning})
	}
	texts, dropped := contentTexts(ch.Message.Content)
	// One distinct note per distinct loss (the emitter's `note()` is the same rule):
	// the cap at the consumer keeps the HEAD of the list, so twelve identical
	// `content_part` notes would evict every other loss from the metric and the GUI.
	for _, n := range dropped {
		if !slices.Contains(notes, n) {
			notes = append(notes, n)
		}
	}
	for _, t := range texts {
		blocks = append(blocks, map[string]any{"type": "text", "text": t})
	}
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
	return b, notes, nil
}

// finishNote reports a Chat Completions finish reason the mapping table has no
// Anthropic spelling for. `stop`/`length`/`tool_calls` arrive under the mapping's
// own rows; everything a real engine adds — `content_filter`, `refusal`,
// `eos_token`, `abort` — collapses to a milder `stop_reason` (content_filter
// aside, the answer reads as a normal end_turn), and the one thing an operator
// must see is that the answer did not end the way the envelope says. The spec's
// loss table names this class explicitly: refusal/content_filter are lost
// information, written to notes, never to silence.
func finishNote(fr string) string {
	switch fr {
	case "", "stop", "length", "tool_calls", "function_call":
		return ""
	}
	return "finish_reason=" + fr
}

// contentTexts reads a Chat Completions `content` value — a string, an array of
// content parts (what a multimodal backend answers), a null, a value that is
// none of these — and returns the text pieces it carries plus one note per part
// the spoke cannot read as text. An image part answering an Anthropic client is
// content that left, and the response direction has no refusal to say so with.
func contentTexts(raw json.RawMessage) (texts []string, dropped []string) {
	if len(raw) == 0 {
		return nil, nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		if s == "" {
			return nil, nil
		}
		return []string{s}, nil
	}
	var parts []struct {
		Type    string `json:"type"`
		Text    string `json:"text"`
		Refusal string `json:"refusal"`
	}
	if json.Unmarshal(raw, &parts) != nil {
		return nil, []string{"content_part=dropped:content"}
	}
	for _, p := range parts {
		switch {
		case p.Text != "":
			texts = append(texts, p.Text)
		case p.Refusal != "":
			texts = append(texts, p.Refusal)
		case p.Type == "" || p.Type == "text":
			// An empty text part carries nothing; it is not a lost part.
		default:
			dropped = append(dropped, "content_part=dropped:"+p.Type)
		}
	}
	return texts, dropped
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
