package convert

import (
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"time"
)

// The responses→chat spoke. A Responses client on a chat-only backend carries a
// real conversation — `input` items, a `function_call`/`function_call_output`
// tool loop, `instructions` as the system prompt — and the answer must come from
// that conversation, so the request is translated item by item.
//
// Three fates, as on the messages spoke: a field is mapped, refused (a
// `drop_params` hint when it can be bought off), or reported. The fields the
// proxy simply does not execute (`store`, `include`, `truncation`, `background`,
// …) are REPORTED, never refused: LiteLLM-style clients set `store` on every
// request, and a 400 there stops real traffic over a field that changes nothing
// about the answer.

type rRequest struct {
	Model              string          `json:"model"`
	Instructions       string          `json:"instructions"`
	Input              json.RawMessage `json:"input"`
	MaxOutputTokens    *int            `json:"max_output_tokens"`
	Temperature        *float64        `json:"temperature"`
	TopP               *float64        `json:"top_p"`
	Stream             bool            `json:"stream,omitempty"`
	Tools              []rTool         `json:"tools"`
	ToolChoice         json.RawMessage `json:"tool_choice"`
	Text               json.RawMessage `json:"text"`
	Reasoning          json.RawMessage `json:"reasoning"`
	ParallelToolCalls  *bool           `json:"parallel_tool_calls"`
	PreviousResponseID string          `json:"previous_response_id"`
	User               string          `json:"user"`
	PromptCacheKey     string          `json:"prompt_cache_key"`
	ServiceTier        string          `json:"service_tier"`
	Store              json.RawMessage `json:"store"`
	Include            json.RawMessage `json:"include"`
	Truncation         json.RawMessage `json:"truncation"`
	Background         json.RawMessage `json:"background"`
	SafetyIdentifier   json.RawMessage `json:"safety_identifier"`
	Metadata           json.RawMessage `json:"metadata"`
	TopLogprobs        json.RawMessage `json:"top_logprobs"`
	MaxToolCalls       json.RawMessage `json:"max_tool_calls"`
}

// DecodedResponsesKeys is rRequest's key set, derived from its own json tags — the
// baseline DroppedKeys compares a Responses request against. Same rule as
// DecodedRequestKeys: derived, not transcribed.
var DecodedResponsesKeys = decodedKeys(reflect.TypeOf(rRequest{}))

func decodedKeys(typ reflect.Type) map[string]bool {
	out := make(map[string]bool, typ.NumField())
	for i := 0; i < typ.NumField(); i++ {
		name, _, _ := strings.Cut(typ.Field(i).Tag.Get("json"), ",")
		if name != "" && name != "-" {
			out[name] = true
		}
	}
	return out
}

// rItem is the union of the Responses input-item shapes this spoke reads: a
// message item (role + content), a function_call, a function_call_output, and the
// item kinds it can only report.
type rItem struct {
	Type      string          `json:"type"`
	Role      string          `json:"role"`
	ID        string          `json:"id"`
	Content   json.RawMessage `json:"content"`
	CallID    string          `json:"call_id"`
	Name      string          `json:"name"`
	Arguments string          `json:"arguments"`
	Output    json.RawMessage `json:"output"`
}

type rPart struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	Refusal  string `json:"refusal"`
	ImageURL string `json:"image_url"`
}

type rTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      *bool           `json:"strict"`
	Format      json.RawMessage `json:"format"`
}

func ResponsesToChat(body []byte, o Options) ([]byte, []string, error) {
	var in rRequest
	if err := json.Unmarshal(body, &in); err != nil {
		return nil, nil, &Error{Code: CodeField, Field: "body", Msg: "invalid JSON body: " + err.Error()}
	}
	if len(in.Input) == 0 || string(in.Input) == "null" {
		return nil, nil, &Error{Code: CodeField, Field: "input",
			Msg: "the request carries no input: a Responses call with no conversation is not a question"}
	}
	// The router stores nothing, so a continuation would silently answer a
	// truncated conversation. Refused before the backend is contacted, and the text
	// says what to send instead (the spec's note on this field).
	if in.PreviousResponseID != "" {
		return nil, nil, &Error{Code: CodeState, Field: "previous_response_id",
			Msg: "this proxy stores no responses: the router can not continue " +
				strconv.Quote(in.PreviousResponseID) + ". Send the whole conversation in `input`."}
	}
	var notes []string
	refuseMsg := func(field, detail string) string {
		return field + " has no Chat Completions representation (" + detail + ")."
	}
	refuse := func(field, detail string) error {
		return &Error{Code: CodeField, Field: field, Msg: refuseMsg(field, detail)}
	}
	drop := func(field, detail string) error {
		if !o.DropParams {
			return &Error{Code: CodeField, Field: field,
				Msg: refuseMsg(field, detail) + " Set settings.convert.drop_params to drop it."}
		}
		notes = append(notes, field+"=dropped")
		return nil
	}
	// Reported, never refused: the proxy does not execute the field, and the answer
	// does not depend on it.
	report := func(field string) { notes = append(notes, field+"=dropped") }

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
	if in.MaxOutputTokens != nil {
		out["max_tokens"] = *in.MaxOutputTokens
	}
	if in.ParallelToolCalls != nil {
		out["parallel_tool_calls"] = *in.ParallelToolCalls
	}
	if in.User != "" {
		out["user"] = in.User
	}
	if in.PromptCacheKey != "" {
		out["prompt_cache_key"] = in.PromptCacheKey
	}
	if in.ServiceTier != "" {
		out["service_tier"] = in.ServiceTier
	}
	for _, f := range []struct {
		name string
		raw  json.RawMessage
	}{
		{"store", in.Store}, {"include", in.Include}, {"truncation", in.Truncation},
		{"background", in.Background}, {"safety_identifier", in.SafetyIdentifier},
		{"metadata", in.Metadata}, {"top_logprobs", in.TopLogprobs},
		{"max_tool_calls", in.MaxToolCalls},
	} {
		if len(f.raw) > 0 && string(f.raw) != "null" {
			report(f.name)
		}
	}
	if len(in.Reasoning) > 0 {
		effort, summary, err := chatReasoningEffort(in.Reasoning, drop)
		if err != nil {
			return nil, nil, err
		}
		if effort != "" {
			out["reasoning_effort"] = effort
		}
		_ = summary
	}
	if len(in.Text) > 0 {
		rf, err := chatResponseFormat(textFormat(in.Text), drop)
		if err != nil {
			return nil, nil, err
		}
		if rf != nil {
			out["response_format"] = rf
		}
	}

	sys := []string{}
	if in.Instructions != "" {
		sys = append(sys, in.Instructions)
	}
	var msgs []any
	var calls []any
	flush := func() {
		if len(calls) == 0 {
			return
		}
		msgs = append(msgs, map[string]any{"role": "assistant", "tool_calls": calls})
		calls = nil
	}
	var items []rItem
	if json.Unmarshal(in.Input, &items) != nil {
		var s string
		if json.Unmarshal(in.Input, &s) != nil {
			return nil, nil, &Error{Code: CodeField, Field: "input",
				Msg: "input must be a string or a list of items"}
		}
		items = []rItem{{Role: "user", Content: json.RawMessage(strconv.Quote(s))}}
	}
	for _, it := range items {
		switch it.Type {
		case "function_call":
			// Chat's arguments is a JSON string, and Responses' is one too: the
			// value rides along verbatim. An empty one is an empty object.
			args := it.Arguments
			if args == "" {
				args = "{}"
			}
			id := it.CallID
			if id == "" {
				id = "call_" + shortHash(it.Name)
			}
			calls = append(calls, map[string]any{"id": id, "type": "function",
				"function": map[string]any{"name": it.Name, "arguments": args}})
			continue
		case "function_call_output":
			flush()
			body, err := itemOutput(it, drop)
			if err != nil {
				return nil, nil, err
			}
			msgs = append(msgs, map[string]any{"role": "tool", "tool_call_id": it.CallID, "content": body})
			continue
		case "reasoning", "item_reference", "image_generation_call",
			"web_search_call", "file_search_call", "computer_call", "computer_call_output",
			"code_interpreter_call", "mcp_call", "mcp_list_tools", "mcp_approval_request",
			"custom_tool_call", "custom_tool_call_output", "local_shell_call",
			"function_approval_call", "compaction":
			// The model's own chain of thought (encrypted, chat has no place for
			// it) and the hosted runs (the provider executed them, the chat backend
			// can not). Reported as a dropped item, never silently skipped.
			if err := dropItem(&notes, o, it.Type); err != nil {
				return nil, nil, err
			}
			continue
		}
		role := it.Role
		if role == "" {
			if it.Type != "" {
				if err := dropItem(&notes, o, it.Type); err != nil {
					return nil, nil, err
				}
				continue
			}
			role = "user"
		}
		text, parts, err := convertInputItem(it, drop)
		if err != nil {
			return nil, nil, err
		}
		// Responses has no system turn of its own: `instructions` is the system
		// prompt, and a client that puts a system/developer item in `input` means
		// the same thing. Chat has one system message, so they concatenate.
		if role == "system" || role == "developer" {
			if text != "" {
				sys = append(sys, text)
			}
			continue
		}
		flush()
		switch {
		case len(parts) > 0:
			msgs = append(msgs, map[string]any{"role": role, "content": parts})
		default:
			msgs = append(msgs, map[string]any{"role": role, "content": text})
		}
	}
	flush()
	if len(sys) > 0 {
		msgs = append([]any{map[string]any{"role": "system", "content": strings.Join(sys, "\n")}}, msgs...)
	}
	out["messages"] = msgs

	if len(in.Tools) > 0 {
		tools := make([]any, 0, len(in.Tools))
		seen := make(map[string]bool, len(in.Tools))
		for _, tl := range in.Tools {
			if tl.Type != "" && tl.Type != "function" {
				return nil, nil, refuse("tools", "hosted tool "+tl.Type+" is provider-specific")
			}
			if detail := toolNameDefect(tl.Name, seen); detail != "" {
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
			if len(tl.Parameters) > 0 {
				fn["parameters"] = tl.Parameters
			}
			if tl.Strict != nil {
				fn["strict"] = *tl.Strict
			}
			tools = append(tools, map[string]any{"type": "function", "function": fn})
		}
		if len(tools) > 0 {
			out["tools"] = tools
		}
	}
	if tc, err := chatToolChoiceResponses(in.ToolChoice, drop); err != nil {
		return nil, nil, err
	} else if tc != nil {
		out["tool_choice"] = tc
	}

	b, err := json.Marshal(out)
	if err != nil {
		return nil, nil, &Error{Code: CodeField, Field: "body", Msg: "encode: " + err.Error()}
	}
	return b, notes, nil
}

// dropItem reports one input item the target dialect has no turn for.
func dropItem(notes *[]string, o Options, kind string) error {
	if !o.DropParams {
		return &Error{Code: CodeField, Field: "input",
			Msg: "input item " + strconv.Quote(kind) + " has no Chat Completions representation. " +
				"Set settings.convert.drop_params to drop it."}
	}
	*notes = append(*notes, "input_item=dropped:"+kind)
	return nil
}

// convertInputItem flattens a message item: a string content, or a list of content
// parts, of which only text and images have a chat representation.
func convertInputItem(it rItem, drop func(string, string) error) (text string, parts []any, err error) {
	content := it.Content
	if len(content) == 0 {
		return "", nil, nil
	}
	if json.Unmarshal(content, &text) == nil {
		return text, nil, nil
	}
	var rp []rPart
	if json.Unmarshal(content, &rp) != nil {
		return "", nil, &Error{Code: CodeField, Field: "content",
			Msg: "content must be a string or a list of content parts"}
	}
	for _, p := range rp {
		switch p.Type {
		case "input_text", "output_text", "text", "refusal", "summary_text":
			s := p.Text
			if s == "" {
				s = p.Refusal
			}
			if s == "" {
				continue
			}
			parts = append(parts, map[string]any{"type": "text", "text": s})
		case "input_image":
			if p.ImageURL == "" {
				if err := drop("input_image", "the image is a stored file id, which a Chat Completions backend can not fetch"); err != nil {
					return "", nil, err
				}
				continue
			}
			parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]any{"url": p.ImageURL}})
		case "file":
			// A PDF/document: the always-400 class, as on the messages spoke.
			return "", nil, &Error{Code: CodeField, Field: "file",
				Msg: refuseFileMsg()}
		default:
			name := p.Type
			if name == "" {
				name = "content_part"
			}
			if err := drop(name, "content part has no Chat Completions representation"); err != nil {
				return "", nil, err
			}
		}
	}
	// One text part degrades to the plain string form every client understands.
	if len(parts) == 1 {
		if tx, ok := parts[0].(map[string]any)["text"].(string); ok {
			return tx, nil, nil
		}
	}
	return "", parts, nil
}

func refuseFileMsg() string {
	return "file has no Chat Completions representation (a stored file has no bytes for the backend to read)"
}

// ChatToResponses is the JSON direction of the responses spoke: a Chat
// Completions answer rebuilt as a Responses answer. A Responses SDK decodes
// `output` ITEMS — a message item with `output_text` parts, one function_call item
// per call, a reasoning item for a chat backend's plaintext chain of thought — and
// reads `status`/`incomplete_details`, not `choices`/`finish_reason`.
//
// Like every response direction it never refuses: the answer is already at the
// client, so a loss is a note.
func ChatToResponses(body []byte) ([]byte, []string, error) {
	// The Responses API reports errors in OpenAI's own envelope, so an upstream
	// 4xx/5xx body is already in the client's dialect: translating it is the damage.
	var env struct {
		Error json.RawMessage `json:"error"`
	}
	if json.Unmarshal(body, &env) == nil && len(env.Error) > 0 {
		return body, nil, nil
	}
	var c cResponse
	if err := json.Unmarshal(body, &c); err != nil {
		return nil, nil, &Error{Code: CodeField, Field: "body", Msg: "chat response is not JSON: " + err.Error()}
	}
	if len(c.Choices) == 0 {
		return nil, nil, &Error{Code: CodeField, Field: "choices", Msg: "chat response carries no choices"}
	}
	var notes []string
	if len(c.Choices) > 1 {
		notes = append(notes, "choices=dropped:"+strconv.Itoa(len(c.Choices)-1))
	}
	ch := c.Choices[0]
	if n := finishNote(ch.FinishReason); n != "" {
		notes = append(notes, n)
	}
	items := make([]any, 0, 3)
	if ch.Message.Reasoning != "" {
		// Chat reports reasoning as plaintext with no signature; Responses' reasoning
		// item carries a summary, which is the closest faithful home (and the shape
		// vLLM/SGLang stream natively for it).
		items = append(items, map[string]any{"type": "reasoning",
			"id":      "rs_" + shortHash(c.ID+":"+ch.Message.Reasoning),
			"summary": []any{map[string]any{"type": "summary_text", "text": ch.Message.Reasoning}},
			"status":  "completed"})
	}
	texts, dropped := contentTexts(ch.Message.Content)
	// One distinct note per distinct loss (the emitter's `note()` is the same rule):
	// the cap at the consumer keeps the HEAD of the list.
	for _, n := range dropped {
		if !slices.Contains(notes, n) {
			notes = append(notes, n)
		}
	}
	parts := make([]any, 0, len(texts))
	for _, t := range texts {
		if t == "" {
			continue
		}
		parts = append(parts, map[string]any{"type": "output_text", "text": t, "annotations": []any{}})
	}
	if len(parts) > 0 {
		items = append(items, map[string]any{"type": "message",
			"id": "msg_" + shortHash(c.ID+":message"), "role": "assistant",
			"status": "completed", "content": parts})
	}
	for i, tc := range ch.Message.ToolCalls {
		id := tc.ID
		if id == "" {
			id = "call_" + shortHash(fmt.Sprintf("%s:%d", c.ID, i))
		}
		items = append(items, map[string]any{"type": "function_call", "id": id, "call_id": id,
			"name": tc.Function.Name, "arguments": string(toolArgumentText(tc.Function.Arguments)),
			"status": "completed"})
	}
	status, reason := responsesStatus(ch.FinishReason)
	in := max(0, c.Usage.PromptTokens-c.Usage.PromptDetails.CachedTokens)
	out := map[string]any{
		"id":         c.ID,
		"object":     "response",
		"created_at": time.Now().Unix(),
		"status":     status,
		"model":      c.Model,
		"output":     items,
		"usage": map[string]any{
			"input_tokens":  in,
			"output_tokens": c.Usage.CompletionTokens,
			"input_tokens_details": map[string]any{
				"cached_tokens": c.Usage.PromptDetails.CachedTokens},
			"total_tokens": in + c.Usage.CompletionTokens,
		},
	}
	if reason != "" {
		out["incomplete_details"] = map[string]any{"reason": reason}
	}
	b, err := json.Marshal(out)
	if err != nil {
		return nil, nil, &Error{Code: CodeField, Field: "body", Msg: "encode: " + err.Error()}
	}
	return b, notes, nil
}

// responsesStatus maps a Chat Completions finish reason onto Responses'
// `status`/`incomplete_details.reason`. A reason with no row is a completed
// response plus a note — `finishNote` reports it.
func responsesStatus(fr string) (string, string) {
	switch fr {
	case "length":
		return "incomplete", "max_output_tokens"
	case "content_filter", "refusal":
		return "incomplete", "content_filter"
	}
	return "completed", ""
}

// toolArgumentText renders a Chat Completions `arguments` value as the JSON string
// Responses' `arguments` field is: the wire form is already a string, and a
// backend that sends an object is marshalled back to one.
func toolArgumentText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return "{}"
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	return string(raw)
}

// itemOutput flattens a function_call_output's `output` — a string, or a list of
// output parts (text and images: a tool that answers with a screenshot has a chat
// representation, since chat's tool content is a string, so the image is reported).
func itemOutput(it rItem, drop func(string, string) error) (string, error) {
	if len(it.Output) == 0 {
		return "", nil
	}
	var s string
	if json.Unmarshal(it.Output, &s) == nil {
		return s, nil
	}
	var parts []rPart
	if json.Unmarshal(it.Output, &parts) != nil {
		return string(it.Output), nil
	}
	texts := make([]string, 0, len(parts))
	for _, p := range parts {
		if p.Type == "output_text" || p.Type == "input_text" || p.Type == "text" {
			texts = append(texts, p.Text)
			continue
		}
		name := p.Type
		if name == "" {
			name = "tool_output_part"
		}
		if err := drop(name, "a tool result must be text for Chat Completions"); err != nil {
			return "", err
		}
	}
	return strings.Join(texts, ""), nil
}

// chatReasoningEffort maps Responses' `reasoning` object onto chat's
// `reasoning_effort`. The summary request (the provider returns reasoning text the
// proxy has no turn for) is reported.
func chatReasoningEffort(raw json.RawMessage, drop func(string, string) error) (string, bool, error) {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s, false, nil
	}
	var obj struct {
		Effort  string `json:"effort"`
		Summary string `json:"summary"`
	}
	if json.Unmarshal(raw, &obj) != nil {
		return "", false, drop("reasoning", "reasoning is neither a string nor an object")
	}
	if obj.Summary != "" {
		if err := drop("reasoning.summary", "chat streams no reasoning summary"); err != nil {
			return "", false, err
		}
	}
	return obj.Effort, obj.Summary != "", nil
}

// textFormat unwraps Responses' `text: {format: {…}}`. The inner object has
// Anthropic's flat output_format shape, so the same builder maps it.
func textFormat(raw json.RawMessage) json.RawMessage {
	var t struct {
		Format json.RawMessage `json:"format"`
	}
	if json.Unmarshal(raw, &t) == nil && len(t.Format) > 0 {
		return t.Format
	}
	return raw
}

// chatToolChoiceResponses maps Responses' tool_choice: the three strings chat
// shares, a named function (Responses names it at the top level, chat nests it),
// and everything else — `allowed_tools`, `custom`, the hosted choices — through the
// loss policy.
func chatToolChoiceResponses(raw json.RawMessage, drop func(string, string) error) (any, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		switch s {
		case "auto", "none", "required":
			return s, nil
		}
		return nil, drop("tool_choice", "a bare tool_choice "+strconv.Quote(s)+
			" is not a Chat Completions value")
	}
	var obj struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if json.Unmarshal(raw, &obj) != nil || obj.Type == "" {
		return nil, drop("tool_choice", "tool_choice has no Chat Completions representation")
	}
	if obj.Type == "function" {
		return namedFunction(obj.Name), nil
	}
	return nil, drop("tool_choice", "tool_choice type "+strconv.Quote(obj.Type)+
		" constrains a tool set Chat Completions can not express")
}
