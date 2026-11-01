package convert

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func decode(t *testing.T, b []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("not JSON: %v (%s)", err, b)
	}
	return m
}

func fieldErr(err error, want string) (code, field string, ok bool) {
	var ce *Error
	if !errors.As(err, &ce) {
		return "", "", false
	}
	if want != "" && ce.Field != want {
		return ce.Code, ce.Field, false
	}
	return ce.Code, ce.Field, true
}

func TestMessagesToChatCore(t *testing.T) {
	in := `{"model":"claude","system":[{"type":"text","text":"be terse"}],
		"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]},
			{"role":"assistant","content":[{"type":"tool_use","id":"tu_1","name":"fetch","input":{"url":"u"}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":"ok"}]}],
		"max_tokens":50,"stop_sequences":["END"],"temperature":0.2,
		"tools":[{"name":"fetch","description":"d","input_schema":{"type":"object"}}],
		"tool_choice":{"type":"any"},"metadata":{"user_id":"u-1"}}`

	out, notes, err := MessagesToChat([]byte(in), Options{})
	if err != nil {
		t.Fatalf("MessagesToChat: %v", err)
	}
	m := decode(t, out)
	if m["system"] != nil {
		t.Fatalf("system must become a message, not stay top-level: %v", m["system"])
	}
	// system + user + assistant(tool_calls) + tool
	msgs, _ := m["messages"].([]any)
	if len(msgs) != 4 {
		t.Fatalf("got %d messages, want 4 (system, user, assistant, tool): %+v", len(msgs), msgs)
	}
	first, _ := msgs[0].(map[string]any)
	if first["role"] != "system" || first["content"] != "be terse" {
		t.Fatalf("system message: %+v", msgs[0])
	}
	asst, _ := msgs[2].(map[string]any)
	if asst["role"] != "assistant" {
		t.Fatalf("msgs[2] must be the assistant turn: %+v", msgs[2])
	}
	calls, _ := asst["tool_calls"].([]any)
	if len(calls) != 1 {
		t.Fatalf("tool_use not mapped to tool_calls: %+v", msgs[2])
	}
	call, _ := calls[0].(map[string]any)
	if call["id"] != "tu_1" || call["type"] != "function" {
		t.Fatalf("tool call shape: %+v", call)
	}
	cfn, _ := call["function"].(map[string]any)
	if cfn["arguments"] != `{"url":"u"}` {
		t.Fatalf("tool_call arguments must be the serialized JSON string, not the raw object: %#v", cfn["arguments"])
	}
	toolMsg, _ := msgs[3].(map[string]any)
	if toolMsg["role"] != "tool" || toolMsg["tool_call_id"] != "tu_1" || toolMsg["content"] != "ok" {
		t.Fatalf("tool_result not mapped to a role:tool message: %+v", msgs[3])
	}
	if notes != nil {
		t.Fatalf("a lossless request must produce no notes: %v", notes)
	}
	if got, _ := m["stop"].([]any); len(got) != 1 || got[0] != "END" {
		t.Fatalf("stop_sequences not mapped to stop: %v", m["stop"])
	}
	if m["user"] != "u-1" {
		t.Fatalf("metadata.user_id not mapped to user: %v", m["user"])
	}
	if m["tool_choice"] != "required" {
		t.Fatalf("tool_choice any must map to required: %v", m["tool_choice"])
	}
	tools, _ := m["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools not mapped: %v", m["tools"])
	}
	tl, _ := tools[0].(map[string]any)
	fn, _ := tl["function"].(map[string]any)
	if fn["parameters"] == nil {
		t.Fatalf("input_schema not mapped to parameters: %+v", tl)
	}
}

func TestMessagesToChatRequiresMaxTokensBudget(t *testing.T) {
	// Anthropic requires max_tokens, OpenAI clients may omit it: the injected
	// default comes from Options, and with no budget configured we refuse.
	_, _, err := MessagesToChat([]byte(`{"model":"m","messages":[]}`), Options{})
	if _, _, ok := fieldErr(err, "max_tokens"); !ok {
		t.Fatalf("refusal must name max_tokens: %v", err)
	}
	out, notes, err := MessagesToChat([]byte(`{"model":"m","messages":[]}`), Options{MaxTokens: 4096})
	if err != nil {
		t.Fatalf("with a budget configured: %v", err)
	}
	if m := decode(t, out); m["max_tokens"] != float64(4096) {
		t.Fatalf("budget not injected: %v", m)
	}
	if len(notes) != 1 {
		t.Fatalf("an injected default must be reported: %v", notes)
	}
}

func TestMessagesToChatRefusesUnrepresentable(t *testing.T) {
	cases := map[string]string{
		"thinking":    `{"model":"m","max_tokens":1,"messages":[],"thinking":{"type":"enabled","budget_tokens":100}}`,
		"top_k":       `{"model":"m","max_tokens":1,"top_k":5,"messages":[]}`,
		"document":    `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"text","media_type":"text/plain","data":"x"}}]}]}`,
		"hosted tool": `{"model":"m","max_tokens":1,"messages":[],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`,
		"tool image":  `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAA"}}]}]}]}`,
	}
	for name, in := range cases {
		code, field, ok := fieldErr(mustFail(t, in), "")
		if !ok || code != CodeField || field == "" {
			t.Fatalf("%s: want *convert.Error{%s} naming a field, got code=%q field=%q",
				name, CodeField, code, field)
		}
	}
	// drop_params turns the refusal into a reported drop.
	_, notes, err := MessagesToChat([]byte(`{"model":"m","max_tokens":1,"top_k":5,"messages":[]}`),
		Options{DropParams: true})
	if err != nil {
		t.Fatalf("drop_params: %v", err)
	}
	if len(notes) != 1 || notes[0] != "top_k=dropped" {
		t.Fatalf("drop not reported: %v", notes)
	}
	// thinking is a reported drop too (amended plan, R14), spelled like every
	// other drop.
	_, notes, err = MessagesToChat([]byte(
		`{"model":"m","max_tokens":1,"messages":[],"thinking":{"type":"enabled","budget_tokens":100}}`),
		Options{DropParams: true})
	if err != nil || len(notes) != 1 || notes[0] != "thinking=dropped" {
		t.Fatalf("thinking under drop_params must be a reported drop: %v %v", notes, err)
	}
}

func mustFail(t *testing.T, body string) error {
	t.Helper()
	_, _, err := MessagesToChat([]byte(body), Options{})
	if err == nil {
		t.Fatalf("%s: want refusal, got nil", body)
	}
	return err
}

func TestMessagesToChatImagesBecomeDataURLs(t *testing.T) {
	in := `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[
		{"type":"text","text":"what is in this?"},
		{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAA"}}]}]}`
	out, notes, err := MessagesToChat([]byte(in), Options{})
	if err != nil {
		t.Fatalf("MessagesToChat: %v", err)
	}
	if notes != nil {
		t.Fatalf("images map losslessly: %v", notes)
	}
	m := decode(t, out)
	msgs, _ := m["messages"].([]any)
	um, _ := msgs[0].(map[string]any)
	parts, _ := um["content"].([]any)
	if len(parts) != 2 {
		t.Fatalf("want text + image parts: %+v", um["content"])
	}
	part, _ := parts[1].(map[string]any)
	iu, _ := part["image_url"].(map[string]any)
	if iu["url"] != "data:image/png;base64,AAA" {
		t.Fatalf("image not mapped to an OpenAI image_url data URL: %+v", part)
	}
}

// OpenAI's tool_calls[].function.arguments is a JSON string on the wire: a
// vLLM-shaped backend 4xxs on a raw object, and 4xx rides the relay (no retry,
// no fallback), so the object must be serialized here.
func TestMessagesToChatToolCallArgumentsSerialized(t *testing.T) {
	in := `{"model":"m","max_tokens":1,"messages":[{"role":"assistant","content":[
		{"type":"tool_use","id":"t1","name":"f"},
		{"type":"tool_use","id":"t2","name":"g","input":{"a":1,"b":[2]}}]}]}`
	out, notes, err := MessagesToChat([]byte(in), Options{})
	if err != nil {
		t.Fatalf("MessagesToChat: %v", err)
	}
	if notes != nil {
		t.Fatalf("lossless request: %v", notes)
	}
	m := decode(t, out)
	msgs, _ := m["messages"].([]any)
	am, _ := msgs[0].(map[string]any)
	calls, _ := am["tool_calls"].([]any)
	if len(calls) != 2 {
		t.Fatalf("want 2 tool calls: %+v", am["tool_calls"])
	}
	c0, _ := calls[0].(map[string]any)
	f0, _ := c0["function"].(map[string]any)
	if f0["arguments"] != "{}" {
		t.Fatalf("a tool_use with no input must serialize to {}: %#v", f0["arguments"])
	}
	c1, _ := calls[1].(map[string]any)
	f1, _ := c1["function"].(map[string]any)
	if f1["arguments"] != `{"a":1,"b":[2]}` {
		t.Fatalf("input must ride as its exact JSON string: %#v", f1["arguments"])
	}
}

// The always-400 class (spec R14): a field whose meaning the backend can not
// recover stays a refusal even under drop_params — shipping a request whose PDF,
// hosted tool run or audio clip silently vanished is worse than a 400, and the
// drop_params invitation must not be offered for it. (thinking is NOT in this
// class: the amended plan makes it a reported drop, thinking=dropped.)
func TestMessagesToChatAlwaysRefusedClass(t *testing.T) {
	cases := map[string]string{
		"document":    `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"file","media_type":"application/pdf","file_id":"f"}}]}]}`,
		"audio":       `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"type":"audio","source":{"type":"base64","media_type":"audio/wav","data":"AAA"}}]}]}`,
		"hosted tool": `{"model":"m","max_tokens":1,"messages":[],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`,
	}
	for name, in := range cases {
		_, notes, err := MessagesToChat([]byte(in), Options{DropParams: true})
		code, field, ok := fieldErr(err, "")
		if !ok || code != CodeField || field == "" {
			t.Fatalf("%s: drop_params must not smuggle an always-refused field through: code=%q field=%q err=%v", name, code, field, err)
		}
		if notes != nil {
			t.Fatalf("%s: a refused field must not also be noted: %v", name, notes)
		}
		if strings.Contains(err.Error(), "drop_params") {
			t.Fatalf("%s: refusal must not invite drop_params: %v", name, err)
		}
	}
}

// Anthropic's output_format is flat ({type,name,schema,strict}); Chat's
// response_format nests the payload under json_schema. A copy-across would leave
// the schema where the backend looks for a {name,schema,strict} object:
// structured output silently unconstrained.
func TestMessagesToChatOutputFormatNesting(t *testing.T) {
	out, notes, err := MessagesToChat([]byte(
		`{"model":"m","max_tokens":1,"messages":[],"output_format":{"type":"json_schema","name":"recipe","strict":true,"schema":{"type":"object"}}}`), Options{})
	if err != nil {
		t.Fatalf("MessagesToChat: %v", err)
	}
	if notes != nil {
		t.Fatalf("the json_schema nesting is lossless: %v", notes)
	}
	m := decode(t, out)
	rf, _ := m["response_format"].(map[string]any)
	if rf == nil || rf["type"] != "json_schema" {
		t.Fatalf("response_format: %+v", m["response_format"])
	}
	if _, has := rf["schema"]; has {
		t.Fatalf("Anthropic's flat schema leaked into the chat response_format: %+v", rf)
	}
	js, _ := rf["json_schema"].(map[string]any)
	if js["name"] != "recipe" || js["strict"] != true || js["schema"] == nil {
		t.Fatalf("json_schema nesting: %+v", rf)
	}
	// text form and the omitted optional name
	out, notes, err = MessagesToChat([]byte(
		`{"model":"m","max_tokens":1,"messages":[],"output_format":{"type":"json_schema","schema":{"type":"object"}}}`), Options{})
	if err != nil || notes != nil {
		t.Fatalf("json_schema without a name: %v %v", err, notes)
	}
	rf, _ = decode(t, out)["response_format"].(map[string]any)
	js, _ = rf["json_schema"].(map[string]any)
	if js["name"] != "response" {
		t.Fatalf("a chat json_schema must carry a name: %+v", rf)
	}
	if _, has := js["strict"]; has {
		t.Fatalf("strict must not be invented: %+v", rf)
	}
	out, _, err = MessagesToChat([]byte(
		`{"model":"m","max_tokens":1,"messages":[],"output_format":{"type":"text"}}`), Options{})
	if err != nil {
		t.Fatalf("text output_format: %v", err)
	}
	rf, _ = decode(t, out)["response_format"].(map[string]any)
	if rf["type"] != "text" {
		t.Fatalf("text form: %+v", rf)
	}
	// a type chat has no shape for: the droppable path
	_, _, err = MessagesToChat([]byte(
		`{"model":"m","max_tokens":1,"messages":[],"output_format":{"type":"grammar","grammar":"x"}}`), Options{})
	if _, _, ok := fieldErr(err, "output_format"); !ok {
		t.Fatalf("an output_format type chat has no shape for must name the field: %v", err)
	}
	_, notes, err = MessagesToChat([]byte(
		`{"model":"m","max_tokens":1,"messages":[],"output_format":{"type":"grammar","grammar":"x"}}`),
		Options{DropParams: true})
	if err != nil || len(notes) != 1 || notes[0] != "output_format=dropped" {
		t.Fatalf("the output_format drop must be reported: %v %v", notes, err)
	}
}

// tool_choice: the four chat values, the named-function forms, and — the part the
// loss policy guards — nothing may vanish quietly: disable_parallel_tool_use maps
// to parallel_tool_calls, and any value with no chat representation (a bare
// "tool", mcp_tool, an undecodable value) is refused (droppable with a note).
func TestMessagesToChatToolChoicePolicy(t *testing.T) {
	// parallel_tool_calls mapping is lossless: no note.
	out, notes, err := MessagesToChat([]byte(
		`{"model":"m","max_tokens":1,"messages":[],"tools":[{"name":"f","input_schema":{}}],"tool_choice":{"type":"auto","disable_parallel_tool_use":true}}`), Options{})
	if err != nil || notes != nil {
		t.Fatalf("disable_parallel_tool_use: %v %v", err, notes)
	}
	m := decode(t, out)
	if m["tool_choice"] != "auto" || m["parallel_tool_calls"] != false {
		t.Fatalf("tool_choice/parallel_tool_calls: %v %v", m["tool_choice"], m["parallel_tool_calls"])
	}
	// a bare string is validated against the chat values, not forwarded verbatim
	out, notes, err = MessagesToChat([]byte(`{"model":"m","max_tokens":1,"messages":[],"tool_choice":"auto"}`), Options{})
	if err != nil || notes != nil {
		t.Fatalf("bare auto: %v %v", err, notes)
	}
	if m := decode(t, out); m["tool_choice"] != "auto" {
		t.Fatalf("bare auto: %v", m["tool_choice"])
	}
	if _, _, ok := fieldErr(mustFail(t, `{"model":"m","max_tokens":1,"messages":[],"tool_choice":"tool"}`), "tool_choice"); !ok {
		t.Fatal("a bare tool_choice the chat dialect has no value for must name the field")
	}
	// provider-side object types (mcp_tool) and undecodable values: refused by
	// default, a reported drop under drop_params.
	for _, tc := range []string{`{"type":"mcp_tool","name":"m"}`, `["auto"]`} {
		in := `{"model":"m","max_tokens":1,"messages":[],"tool_choice":` + tc + `}`
		if _, _, ok := fieldErr(mustFail(t, in), "tool_choice"); !ok {
			t.Fatalf("%s: must refuse naming tool_choice", tc)
		}
		_, notes, err := MessagesToChat([]byte(in), Options{DropParams: true})
		if err != nil || len(notes) != 1 || notes[0] != "tool_choice=dropped" {
			t.Fatalf("%s: drop_params must report the drop: %v %v", tc, notes, err)
		}
	}
}

// An Anthropic image with a url source rides chat's image_url losslessly — it
// must not 400, and it must not depend on drop_params.
func TestMessagesToChatImageSourceURL(t *testing.T) {
	in := `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[
		{"type":"text","text":"what is in this?"},
		{"type":"image","source":{"type":"url","url":"https://x.example/y.png"}}]}]}`
	out, notes, err := MessagesToChat([]byte(in), Options{})
	if err != nil {
		t.Fatalf("MessagesToChat: %v", err)
	}
	if notes != nil {
		t.Fatalf("url images map losslessly: %v", notes)
	}
	m := decode(t, out)
	msgs, _ := m["messages"].([]any)
	um, _ := msgs[0].(map[string]any)
	parts, _ := um["content"].([]any)
	if len(parts) != 2 {
		t.Fatalf("want text + image parts: %+v", um["content"])
	}
	part, _ := parts[1].(map[string]any)
	iu, _ := part["image_url"].(map[string]any)
	if iu["url"] != "https://x.example/y.png" {
		t.Fatalf("url source not carried over: %+v", part)
	}
}

// The small fields: a tool with no input_schema omits parameters (a null
// parameters is a schema vLLM rejects), an untyped content block refuses under a
// non-empty field name, and a message with no content key converts like
// content:"" (not an "invalid content" refusal for a key that was never sent).
func TestMessagesToChatSmallFields(t *testing.T) {
	out, notes, err := MessagesToChat([]byte(
		`{"model":"m","max_tokens":1,"messages":[],"tools":[{"name":"f","description":"d"}]}`), Options{})
	if err != nil || notes != nil {
		t.Fatalf("MessagesToChat: %v %v", err, notes)
	}
	m := decode(t, out)
	tools, _ := m["tools"].([]any)
	tl, _ := tools[0].(map[string]any)
	fn, _ := tl["function"].(map[string]any)
	if _, has := fn["parameters"]; has {
		t.Fatalf("a tool with no input_schema must omit parameters: %+v", fn)
	}

	code, field, ok := fieldErr(mustFail(t,
		`{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"text":"hi"}]}]}`), "")
	if !ok || code != CodeField || field == "" {
		t.Fatalf("an untyped content block must refuse under a non-empty field: code=%q field=%q", code, field)
	}
	_, notes, err = MessagesToChat([]byte(
		`{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"text":"hi"}]}]}`),
		Options{DropParams: true})
	if err != nil || len(notes) != 1 || notes[0] != field+"=dropped" {
		t.Fatalf("the untyped block drop must be reported under the same name: %v %v", notes, err)
	}

	out, notes, err = MessagesToChat([]byte(`{"model":"m","max_tokens":1,"messages":[{"role":"user"}]}`), Options{})
	if err != nil || notes != nil {
		t.Fatalf("a message with no content key: %v %v", err, notes)
	}
	m = decode(t, out)
	msgs, _ := m["messages"].([]any)
	um, _ := msgs[0].(map[string]any)
	if um["content"] != "" {
		t.Fatalf("no content key must convert like an empty message: %+v", um)
	}
}

// A tool name that can not survive the round trip is refused, not forwarded:
// Anthropic allows `[a-zA-Z0-9_-]{1,128}`, Chat Completions caps function names at
// 64 bytes, and nothing upstream validates. A Claude Code MCP-prefixed name
// (`mcp__<server>__<tool>`) is routinely over the cap, so the request used to
// leave with the name intact and come back as the backend's own 400 — forwarded,
// no fallback, health green. drop_params buys it off with a note naming the tool.
func TestMessagesToChatToolNameConstraint(t *testing.T) {
	long := strings.Repeat("m", 65)
	cases := map[string]string{
		"over the chat cap": `{"model":"m","max_tokens":1,"messages":[],"tools":[{"name":"` + long +
			`","input_schema":{}}]}`,
		"illegal characters": `{"model":"m","max_tokens":1,"messages":[],"tools":[{"name":"mcp__srv__tool.v2",` +
			`"input_schema":{}}]}`,
		"empty":    `{"model":"m","max_tokens":1,"messages":[],"tools":[{"name":"","input_schema":{}}]}`,
		"collides": `{"model":"m","max_tokens":1,"messages":[],"tools":[{"name":"f","input_schema":{}},{"name":"f","input_schema":{}}]}`,
	}
	for name, in := range cases {
		err := mustFail(t, in)
		code, field, ok := fieldErr(err, "tools")
		if !ok || code != CodeField {
			t.Fatalf("%s: want *convert.Error{%s, tools}, got code=%q field=%q", name, CodeField, code, field)
		}
	}
	// The refusal names the offending tool, so the client can fix it.
	err := mustFail(t, `{"model":"m","max_tokens":1,"messages":[],"tools":[{"name":"mcp__srv__tool.v2",`+
		`"input_schema":{}}]}`)
	if !strings.Contains(err.Error(), "mcp__srv__tool.v2") {
		t.Fatalf("the refusal must name the tool: %v", err)
	}
	// drop_params: the tool is dropped, reported, and its siblings survive.
	out, notes, err := MessagesToChat([]byte(`{"model":"m","max_tokens":1,"messages":[],"tools":[`+
		`{"name":"keep","input_schema":{}},{"name":"bad.name","input_schema":{}}]}`),
		Options{DropParams: true})
	if err != nil {
		t.Fatalf("drop_params: %v", err)
	}
	if len(notes) != 1 || notes[0] != "tools.bad.name=dropped" {
		t.Fatalf("the dropped tool must be reported: %v", notes)
	}
	m := decode(t, out)
	tools, _ := m["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("only the offending tool may leave: %+v", m["tools"])
	}
	// 64 bytes is the cap, not 63, and Anthropic's 128 must not be the limit we
	// check against.
	if _, _, err := MessagesToChat([]byte(`{"model":"m","max_tokens":1,"messages":[],"tools":[`+
		`{"name":"`+strings.Repeat("m", 64)+`","input_schema":{}}]}`), Options{}); err != nil {
		t.Fatalf("a 64-byte name is legal: %v", err)
	}
}

func TestRequestDispatch(t *testing.T) {
	body := []byte(`{"model":"m","messages":[]}`)
	// Identity plans return the bytes untouched — no re-marshal, no key reorder.
	got, notes, err := Request(Plan{From: Chat, Dialect: Chat, Path: "/v1/chat/completions"}, body, Options{})
	if err != nil || string(got) != string(body) || notes != nil {
		t.Fatalf("identity must be a no-op: %s %v %v", got, notes, err)
	}
	// A dialectless plan (audio, embeddings) likewise.
	got, notes, err = Request(Plan{Path: "/v1/audio/speech"}, body, Options{})
	if err != nil || string(got) != string(body) || notes != nil {
		t.Fatalf("dialectless plan: %s %v %v", got, notes, err)
	}
	if _, _, err := Request(Plan{From: Responses, Dialect: Chat}, body, Options{}); err == nil {
		t.Fatal("an unimplemented pair must refuse, not pass through")
	}
}
