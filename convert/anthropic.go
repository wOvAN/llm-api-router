package convert

import (
	"encoding/json"
	"strconv"
	"strings"
)

// The messages→chat spoke. Every field is either mapped, refused, or — with
// Options.DropParams — dropped and reported; nothing is ever dropped quietly. The
// reference for the mapping is LiteLLM's anthropic adapter
// (llms/anthropic/experimental_pass_through/adapters/transformation.py) and the
// inverse shim llama.cpp ships in tools/server/server-chat.cpp.

type aRequest struct {
	Model         string          `json:"model"`
	System        json.RawMessage `json:"system"`
	Messages      []aMsg          `json:"messages"`
	MaxTokens     *int            `json:"max_tokens"`
	StopSequences []string        `json:"stop_sequences"`
	Temperature   *float64        `json:"temperature"`
	TopP          *float64        `json:"top_p"`
	TopK          *int            `json:"top_k"`
	Stream        bool            `json:"stream,omitempty"`
	Tools         []aTool         `json:"tools"`
	ToolChoice    json.RawMessage `json:"tool_choice"`
	OutputFormat  json.RawMessage `json:"output_format"`
	Thinking      json.RawMessage `json:"thinking"`
	Metadata      *aMetadata      `json:"metadata"`
}

type aMetadata struct {
	UserID string `json:"user_id"`
}

type aMsg struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// aBlock is the union of the Anthropic content-block shapes this spoke reads
// (text, image, tool_use, tool_result, plus the fields tools declare).
type aBlock struct {
	Type        string          `json:"type"`
	Text        string          `json:"text"`
	ID          string          `json:"id"`
	Name        string          `json:"name"`
	Input       json.RawMessage `json:"input"`
	ToolUseID   string          `json:"tool_use_id"`
	Content     json.RawMessage `json:"content"`
	Source      *aSource        `json:"source"`
	Description string          `json:"description"`
}

type aSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

type aTool struct {
	Name        string          `json:"name"`
	Type        string          `json:"type"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

func MessagesToChat(body []byte, o Options) ([]byte, []string, error) {
	var in aRequest
	if err := json.Unmarshal(body, &in); err != nil {
		return nil, nil, &Error{Code: CodeField, Field: "body", Msg: "invalid JSON body: " + err.Error()}
	}
	var notes []string
	// drop refuses by default (a field with no representation must stop the
	// request: the client asked for something the backend can not do) and reports
	// a note when the operator opted into silent drops.
	drop := func(field, detail string) error {
		if !o.DropParams {
			return &Error{Code: CodeField, Field: field, Msg: field + " has no Chat Completions " +
				"representation (" + detail + "). Set settings.convert.drop_params to drop it."}
		}
		notes = append(notes, field+"=dropped")
		return nil
	}

	out := map[string]any{"model": in.Model}
	if in.Stream {
		out["stream"] = true
	}
	if in.Temperature != nil {
		out["temperature"] = *in.Temperature
	}
	if in.TopP != nil {
		out["top_p"] = *in.TopP
	}
	if len(in.StopSequences) > 0 {
		out["stop"] = in.StopSequences
	}
	if in.MaxTokens != nil {
		out["max_tokens"] = *in.MaxTokens
	} else if o.MaxTokens > 0 {
		out["max_tokens"] = o.MaxTokens
		notes = append(notes, "max_tokens=injected:"+strconv.Itoa(o.MaxTokens))
	} else if err := drop("max_tokens", "the client sent none and no default is configured"); err != nil {
		return nil, nil, err
	}
	if in.TopK != nil {
		if err := drop("top_k", "chat has no top_k"); err != nil {
			return nil, nil, err
		}
	}
	if len(in.Thinking) > 0 {
		// Anthropic signs its thinking blocks; a signature can not be reproduced
		// from OpenAI reasoning text, so a multi-turn tool loop over a converted
		// request would break mid-conversation. Emulating it is never permitted.
		if err := drop("thinking", "signed thinking can not round-trip"); err != nil {
			return nil, nil, err
		}
	}
	if len(in.OutputFormat) > 0 {
		out["response_format"] = in.OutputFormat
		notes = append(notes, "output_format=response_format")
	}

	msgs := make([]any, 0, len(in.Messages)+1)
	if sys := systemText(in.System); sys != "" {
		msgs = append(msgs, map[string]any{"role": "system", "content": sys})
	}
	for _, m := range in.Messages {
		converted, err := convertMessage(m, drop)
		if err != nil {
			return nil, nil, err
		}
		msgs = append(msgs, converted...)
	}
	out["messages"] = msgs

	if len(in.Tools) > 0 {
		tools := make([]any, 0, len(in.Tools))
		for _, tl := range in.Tools {
			if tl.Type != "" {
				// web_search_20250305, text_editor_20241022, computer_*: the
				// provider runs these, the model does not.
				if err := drop("tools", "hosted tool "+tl.Type+" is provider-specific"); err != nil {
					return nil, nil, err
				}
				continue
			}
			tools = append(tools, map[string]any{"type": "function", "function": map[string]any{
				"name": tl.Name, "description": tl.Description, "parameters": tl.InputSchema}})
		}
		if len(tools) > 0 {
			out["tools"] = tools
		}
	}
	if tc := chatToolChoice(in.ToolChoice); tc != nil {
		out["tool_choice"] = tc
	}
	if in.Metadata != nil && in.Metadata.UserID != "" {
		out["user"] = in.Metadata.UserID
	}

	b, err := json.Marshal(out)
	if err != nil {
		return nil, nil, &Error{Code: CodeField, Field: "body", Msg: "encode: " + err.Error()}
	}
	return b, notes, nil
}

// systemText flattens Anthropic's system field — a string or an array of text
// blocks — into the single system message Chat Completions has.
func systemText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var blocks []aBlock
	if json.Unmarshal(raw, &blocks) != nil {
		return ""
	}
	parts := make([]string, 0, len(blocks))
	for _, b := range blocks {
		if b.Type == "text" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n")
}

// convertMessage maps one Anthropic message to one or more Chat Completions
// messages: an assistant turn carrying tool_use becomes a message with tool_calls,
// and a tool_result becomes a role:"tool" message per call.
func convertMessage(m aMsg, drop func(string, string) error) ([]any, error) {
	var text string
	var blocks []aBlock
	if json.Unmarshal(m.Content, &text) != nil {
		if json.Unmarshal(m.Content, &blocks) != nil {
			return nil, &Error{Code: CodeField, Field: "content", Msg: "content must be a string or blocks"}
		}
	}
	if len(blocks) == 0 {
		return []any{map[string]any{"role": m.Role, "content": text}}, nil
	}

	var out []any
	var calls []any
	var parts []any
	for _, b := range blocks {
		switch b.Type {
		case "text":
			parts = append(parts, map[string]any{"type": "text", "text": b.Text})
		case "image":
			if b.Source == nil || b.Source.Type != "base64" {
				if err := drop("image", "only base64 image sources are proxied"); err != nil {
					return nil, err
				}
				continue
			}
			parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]any{
				"url": "data:" + b.Source.MediaType + ";base64," + b.Source.Data}})
		case "tool_use":
			calls = append(calls, map[string]any{"id": b.ID, "type": "function",
				"function": map[string]any{"name": b.Name, "arguments": b.Input}})
		case "tool_result":
			body, err := toolResultContent(b, drop)
			if err != nil {
				return nil, err
			}
			out = append(out, map[string]any{"role": "tool", "tool_call_id": b.ToolUseID, "content": body})
		default:
			if err := drop(b.Type, "content block has no Chat Completions representation"); err != nil {
				return nil, err
			}
		}
	}
	if len(calls) > 0 {
		am := map[string]any{"role": "assistant", "tool_calls": calls}
		if len(parts) > 0 {
			am["content"] = parts
		}
		return append([]any{am}, out...), nil
	}
	// A pure tool_result turn (or one whose blocks were all dropped) is fully
	// represented by the tool messages: adding the client's own role back on top
	// would send a second, content-less message per turn.
	if len(parts) == 0 {
		return out, nil
	}
	// One text part degrades to the plain string form every client understands.
	if len(parts) == 1 {
		if tx, ok := parts[0].(map[string]any)["text"].(string); ok {
			return append(out, map[string]any{"role": m.Role, "content": tx}), nil
		}
	}
	return append(out, map[string]any{"role": m.Role, "content": parts}), nil
}

func toolResultContent(b aBlock, drop func(string, string) error) (string, error) {
	var s string
	if json.Unmarshal(b.Content, &s) == nil {
		return s, nil
	}
	var blocks []aBlock
	if json.Unmarshal(b.Content, &blocks) != nil {
		return "", nil
	}
	parts := make([]string, 0, len(blocks))
	for _, blk := range blocks {
		if blk.Type == "text" {
			parts = append(parts, blk.Text)
			continue
		}
		if err := drop("tool_result."+blk.Type, "a tool result must be text for Chat Completions"); err != nil {
			return "", err
		}
	}
	return strings.Join(parts, "\n"), nil
}

// chatToolChoice maps Anthropic's tool_choice (a string or {type,name}) onto the
// Chat Completions values: "auto", "required", a named function, or "none".
func chatToolChoice(raw json.RawMessage) any {
	if len(raw) == 0 {
		return nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		if s == "any" {
			return "required"
		}
		return s
	}
	var obj struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if json.Unmarshal(raw, &obj) != nil {
		return nil
	}
	switch obj.Type {
	case "any":
		if obj.Name == "" {
			return "required"
		}
		return map[string]any{"type": "function", "function": map[string]any{"name": obj.Name}}
	case "tool":
		return map[string]any{"type": "function", "function": map[string]any{"name": obj.Name}}
	case "auto", "none":
		return obj.Type
	}
	return nil
}
