package convert

import (
	"encoding/json"
	"reflect"
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

// DecodedRequestKeys is aRequest's key set, derived from its own json tags: the
// top-level fields MessagesToChat handles, each one mapped, folded, renamed or
// reported. A spoke marshals a whitelist, so a client key OUTSIDE this set leaves
// with nothing to say so — which is what the router's `keys:<n>dropped` note
// counts. It is derived, not transcribed, so a new field on the decoder is
// accounted for the moment it exists.
var DecodedRequestKeys = decodedRequestKeys()

func decodedRequestKeys() map[string]bool {
	typ := reflect.TypeOf(aRequest{})
	out := make(map[string]bool, typ.NumField())
	for i := 0; i < typ.NumField(); i++ {
		name, _, _ := strings.Cut(typ.Field(i).Tag.Get("json"), ",")
		if name != "" && name != "-" {
			out[name] = true
		}
	}
	return out
}

// DroppedKeys compares what the client sent against what the backend will see and
// reports the TOP-LEVEL keys that left with no per-field note to explain them. A
// spoke marshals a whitelist, so a key it never decodes vanishes with nothing to
// say so: no note, no refusal. The spec forbids silent loss, and a refusal would
// 400 real traffic on every field the providers ship next month (leaving
// `convert: off` as the only escape), so the path that counts them reports
// `keys:<n>dropped`.
//
// The baseline is the spoke's own decoder set (`DecodedRequestKeys`): a key on it
// is a key the spoke handled — mapped straight through, folded into another key
// (`system` into `messages`), renamed (`stop_sequences` into `stop`), or dropped
// under a `field=dropped` note. Counting it would report the plain system prompt
// of every Anthropic client as a loss and make the operator count a field twice,
// and a nested loss is already visible through its own block-level note
// (`document=dropped`, `thinking=dropped`), so counting its container on top of
// that note would double-report one loss (R25).
func DroppedKeys(in, out []byte) []string {
	inKeys, outKeys := keySet(in), keySet(out)
	if inKeys == nil {
		return nil
	}
	n := 0
	for k := range inKeys {
		if _, kept := outKeys[k]; kept || DecodedRequestKeys[k] {
			continue
		}
		n++
	}
	if n == 0 {
		return nil
	}
	return []string{"keys:" + strconv.Itoa(n) + "dropped"}
}

func keySet(body []byte) map[string]json.RawMessage {
	var m map[string]json.RawMessage
	if json.Unmarshal(body, &m) != nil {
		return nil
	}
	return m
}

type aMetadata struct {
	UserID string `json:"user_id"`
}

type aMsg struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// aBlock is the union of the Anthropic content-block shapes this spoke reads
// (text, image, document, audio, tool_use, tool_result).
type aBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
	ToolUseID string          `json:"tool_use_id"`
	Content   json.RawMessage `json:"content"`
	Source    *aSource        `json:"source"`
}

type aSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
	URL       string `json:"url"`
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
	// The always-400 class: content the target dialect cannot carry AND cannot
	// lose quietly (a document, an audio block, a hosted tool run — the answer
	// would be about material that vanished). drop_params can not buy these off,
	// and their refusal must not invite it.
	refuseMsg := func(field, detail string) string {
		return field + " has no Chat Completions representation (" + detail + ")."
	}
	refuse := func(field, detail string) error {
		return &Error{Code: CodeField, Field: field, Msg: refuseMsg(field, detail)}
	}
	// drop refuses by default (a field with no representation must stop the
	// request: the client asked for something the backend can not do) and reports
	// a note when the operator opted into silent drops.
	drop := func(field, detail string) error {
		if !o.DropParams {
			return &Error{Code: CodeField, Field: field,
				Msg: refuseMsg(field, detail) + " Set settings.convert.drop_params to drop it."}
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
		// request loses the chain of thought. A reported drop, per the plan.
		if err := drop("thinking", "signed thinking can not round-trip"); err != nil {
			return nil, nil, err
		}
	}
	if len(in.OutputFormat) > 0 {
		rf, err := chatResponseFormat(in.OutputFormat, drop)
		if err != nil {
			return nil, nil, err
		}
		if rf != nil {
			out["response_format"] = rf
		}
	}

	msgs := make([]any, 0, len(in.Messages)+1)
	if sys := systemText(in.System); sys != "" {
		msgs = append(msgs, map[string]any{"role": "system", "content": sys})
	}
	for _, m := range in.Messages {
		converted, err := convertMessage(m, drop, refuse)
		if err != nil {
			return nil, nil, err
		}
		msgs = append(msgs, converted...)
	}
	out["messages"] = msgs

	if len(in.Tools) > 0 {
		tools := make([]any, 0, len(in.Tools))
		seen := make(map[string]bool, len(in.Tools))
		for _, tl := range in.Tools {
			if tl.Type != "" {
				// web_search_20250305, file_search, code_interpreter, mcp:
				// the provider runs these, the model does not. An agent whose
				// declared tool vanished answers without it — always-400 class.
				return nil, nil, refuse("tools", "hosted tool "+tl.Type+" is provider-specific")
			}
			if detail := toolNameDefect(tl.Name, seen); detail != "" {
				// The name never reaches the backend and never comes back: refused
				// by default (Field "tools", the message naming the tool), a
				// reported drop naming it when the operator opted into them.
				name := tl.Name
				if name == "" {
					name = "<unnamed>"
				}
				if !o.DropParams {
					return nil, nil, refuse("tools", "tool "+strconv.Quote(name)+" "+detail)
				}
				notes = append(notes, "tools."+name+"=dropped")
				continue
			}
			seen[tl.Name] = true
			fn := map[string]any{"name": tl.Name, "description": tl.Description}
			if len(tl.InputSchema) > 0 {
				fn["parameters"] = tl.InputSchema
			}
			tools = append(tools, map[string]any{"type": "function", "function": fn})
		}
		if len(tools) > 0 {
			out["tools"] = tools
		}
	}
	tc, parallel, err := chatToolChoice(in.ToolChoice, drop)
	if err != nil {
		return nil, nil, err
	}
	if tc != nil {
		out["tool_choice"] = tc
	}
	if parallel != nil {
		out["parallel_tool_calls"] = !*parallel
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

// maxChatToolNameBytes is Chat Completions' function-name cap. Anthropic allows
// [a-zA-Z0-9_-]{1,128}; the intersection is what survives the round trip.
const maxChatToolNameBytes = 64

// toolNameDefect names what makes a tool name unable to survive the round trip,
// or "" when it can. Nothing upstream validates either limit, so a name the
// backend rejects used to leave this proxy intact and come back as the backend's
// own 400 — forwarded as-is, no retry, no fallback, health green. A Claude Code
// MCP-prefixed name (`mcp__<server>__<tool>`) over 64 bytes is routine, which is
// exactly the hole this spoke exists to close: refuse it here, naming the tool.
func toolNameDefect(name string, seen map[string]bool) string {
	switch {
	case name == "":
		return "has no name"
	case len(name) > maxChatToolNameBytes:
		return "is over Chat Completions' " + strconv.Itoa(maxChatToolNameBytes) +
			"-byte tool-name cap (Anthropic allows 128)"
	case !isToolNameChars(name):
		return "carries characters outside [a-zA-Z0-9_-]"
	case seen[name]:
		return "is declared twice"
	}
	return ""
}

func isToolNameChars(name string) bool {
	for i := 0; i < len(name); i++ {
		c := name[i]
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '_', c == '-':
		default:
			return false
		}
	}
	return true
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
func convertMessage(m aMsg, drop, refuse func(string, string) error) ([]any, error) {
	var text string
	var blocks []aBlock
	content := m.Content
	if len(content) == 0 {
		// A message with no content key at all behaves like content: null (which
		// unmarshals to the empty string), not like a malformed one.
		content = json.RawMessage(`""`)
	}
	if json.Unmarshal(content, &text) != nil {
		if json.Unmarshal(content, &blocks) != nil {
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
			switch {
			case b.Source != nil && b.Source.Type == "base64" && b.Source.Data != "":
				parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]any{
					"url": "data:" + b.Source.MediaType + ";base64," + b.Source.Data}})
			case b.Source != nil && b.Source.Type == "url" && b.Source.URL != "":
				// Chat's image_url.url takes an http(s) URL directly: lossless.
				parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]any{
					"url": b.Source.URL}})
			default:
				if err := drop("image", "the image source is neither a base64 payload nor a url"); err != nil {
					return nil, err
				}
			}
		case "document", "audio":
			// A PDF page or an audio clip the chat backend can not ingest:
			// always-400 class, drop_params can not buy it off.
			return nil, refuse(b.Type, b.Type+" content can not be ingested by a Chat Completions backend")
		case "tool_use":
			// Chat's arguments is a JSON string on the wire (official SDK:
			// Arguments string); forwarding the raw object makes strict backends
			// 4xx, which the relay forwards as-is — no retry, no fallback.
			args := "{}"
			if len(b.Input) > 0 && string(b.Input) != "null" {
				args = string(b.Input)
			}
			calls = append(calls, map[string]any{"id": b.ID, "type": "function",
				"function": map[string]any{"name": b.Name, "arguments": args}})
		case "tool_result":
			body, err := toolResultContent(b, drop)
			if err != nil {
				return nil, err
			}
			out = append(out, map[string]any{"role": "tool", "tool_call_id": b.ToolUseID, "content": body})
		default:
			name := b.Type
			if name == "" {
				name = "content_block"
			}
			if err := drop(name, "content block has no Chat Completions representation"); err != nil {
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

// chatResponseFormat rebuilds Anthropic's flat output_format
// ({type,name,schema,strict}) as Chat Completions' response_format, whose
// json_schema payload nests one level deeper: copying the object across would
// leave the schema where the backend looks for a {name,schema,strict} object,
// silently un-constraining the answer.
func chatResponseFormat(raw json.RawMessage, drop func(string, string) error) (any, error) {
	var of struct {
		Type   string          `json:"type"`
		Name   string          `json:"name"`
		Schema json.RawMessage `json:"schema"`
		Strict *bool           `json:"strict"`
	}
	if json.Unmarshal(raw, &of) != nil {
		if err := drop("output_format", "output_format is not an object"); err != nil {
			return nil, err
		}
		return nil, nil
	}
	switch of.Type {
	case "json_schema":
		if len(of.Schema) == 0 {
			if err := drop("output_format", "a json_schema output_format carries no schema"); err != nil {
				return nil, err
			}
			return nil, nil
		}
		name := of.Name
		if name == "" {
			name = "response" // chat requires a schema name, Anthropic makes it optional
		}
		js := map[string]any{"name": name, "schema": of.Schema}
		if of.Strict != nil {
			js["strict"] = *of.Strict
		}
		return map[string]any{"type": "json_schema", "json_schema": js}, nil
	case "json_object", "text":
		return map[string]any{"type": of.Type}, nil
	}
	if err := drop("output_format", "output_format type "+strconv.Quote(of.Type)+
		" has no Chat Completions representation"); err != nil {
		return nil, err
	}
	return nil, nil
}

// chatToolChoice maps Anthropic's tool_choice (a string or {type,name,
// disable_parallel_tool_use}) onto Chat Completions: the value plus the parallel
// flag it carries. The loss policy governs the rest: a value with no chat
// representation (a bare "tool", the provider-side mcp_tool, an undecodable
// value) goes through drop — refused by default, a reported drop under
// drop_params — never a quiet vanish.
func chatToolChoice(raw json.RawMessage, drop func(string, string) error) (any, *bool, error) {
	if len(raw) == 0 {
		return nil, nil, nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		switch s {
		case "any":
			return "required", nil, nil
		case "auto", "required", "none":
			return s, nil, nil
		}
		if err := drop("tool_choice", "a bare tool_choice "+strconv.Quote(s)+" is not a Chat Completions value"); err != nil {
			return nil, nil, err
		}
		return nil, nil, nil
	}
	var obj struct {
		Type            string `json:"type"`
		Name            string `json:"name"`
		DisableParallel *bool  `json:"disable_parallel_tool_use"`
	}
	if json.Unmarshal(raw, &obj) != nil {
		if err := drop("tool_choice", "tool_choice has no Chat Completions representation"); err != nil {
			return nil, nil, err
		}
		return nil, nil, nil
	}
	var tc any
	switch obj.Type {
	case "any":
		if obj.Name == "" {
			tc = "required"
		} else {
			tc = namedFunction(obj.Name)
		}
	case "tool":
		tc = namedFunction(obj.Name)
	case "auto", "none":
		tc = obj.Type
	default:
		// mcp_tool and the other provider-side choices.
		if err := drop("tool_choice", "tool_choice type "+obj.Type+" is provider-specific"); err != nil {
			return nil, nil, err
		}
	}
	return tc, obj.DisableParallel, nil
}

func namedFunction(name string) any {
	return map[string]any{"type": "function", "function": map[string]any{"name": name}}
}
