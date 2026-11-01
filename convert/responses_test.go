package convert

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"
)

// The responses→chat spoke. A Responses client on a chat-only backend is the
// highest-value edge the spec names: the client's provider is asking for the
// Responses API, the local engines only serve Chat Completions, and the request
// carries a real conversation, so the answer must come from that conversation.

func decodeMap(t *testing.T, b []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("not JSON: %v\n%s", err, b)
	}
	return m
}

func TestResponsesToChatRequest(t *testing.T) {
	in := `{"model":"claude","instructions":"be brief","max_output_tokens":64,` +
		`"temperature":0.2,"top_p":0.9,"stream":true,"parallel_tool_calls":true,` +
		`"input":[{"role":"system","content":"today is sunday"},` +
		`{"role":"user","content":[{"type":"input_text","text":"hi"},` +
		`{"type":"input_image","image_url":"https://x/y.png"}]}]}`
	out, notes, err := ResponsesToChat([]byte(in), Options{})
	if err != nil {
		t.Fatalf("ResponsesToChat: %v", err)
	}
	m := decodeMap(t, out)
	if m["model"] != "claude" || m["stream"] != true {
		t.Fatalf("envelope: %+v", m)
	}
	if m["max_tokens"] != float64(64) || m["temperature"] != 0.2 || m["top_p"] != 0.9 {
		t.Fatalf("sampling/limit fields not mapped: %+v", m)
	}
	if m["parallel_tool_calls"] != true {
		t.Fatalf("parallel_tool_calls not mapped: %+v", m)
	}
	msgs, _ := m["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("want system + user, got %+v", m["messages"])
	}
	sys, _ := msgs[0].(map[string]any)
	if sys["role"] != "system" || sys["content"] != "be brief\ntoday is sunday" {
		t.Fatalf("instructions must fold into the system message: %+v", msgs[0])
	}
	user, _ := msgs[1].(map[string]any)
	parts, _ := user["content"].([]any)
	if len(parts) != 2 {
		t.Fatalf("text + image parts expected: %+v", user["content"])
	}
	p0, _ := parts[0].(map[string]any)
	p1, _ := parts[1].(map[string]any)
	if p0["type"] != "text" || p0["text"] != "hi" {
		t.Fatalf("input_text not mapped: %+v", p0)
	}
	img, _ := p1["image_url"].(map[string]any)
	if p1["type"] != "image_url" || img["url"] != "https://x/y.png" {
		t.Fatalf("input_image not mapped: %+v", p1)
	}
	if len(notes) != 0 {
		t.Fatalf("notes: %v", notes)
	}
}

// A plain string `input` is the shape every SDK's convenience builder sends, and
// `instructions` is absent there.
func TestResponsesToChatStringInput(t *testing.T) {
	out, notes, err := ResponsesToChat([]byte(`{"model":"m","input":"ping"}`), Options{})
	if err != nil {
		t.Fatalf("ResponsesToChat: %v", err)
	}
	m := decodeMap(t, out)
	msgs, _ := m["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("want one user message: %+v", m["messages"])
	}
	msg, _ := msgs[0].(map[string]any)
	if msg["role"] != "user" || msg["content"] != "ping" {
		t.Fatalf("string input not mapped: %+v", msg)
	}
	if len(notes) != 0 {
		t.Fatalf("notes: %v", notes)
	}
}

// The tool loop is the reason the edge matters: an agent's whole state arrives as
// `function_call`/`function_call_output` items, and chat represents them as an
// assistant `tool_calls` message followed by role:"tool" messages. `call_id` is
// the key that ties them, so it must survive on both sides.
func TestResponsesToChatToolLoop(t *testing.T) {
	in := `{"model":"m","input":[` +
		`{"role":"user","content":"what time is it"},` +
		`{"type":"function_call","call_id":"call_1","name":"clock","arguments":"{\"tz\":1}"},` +
		`{"type":"function_call","call_id":"call_2","name":"clock","arguments":"{\"tz\":2}"},` +
		`{"type":"function_call_output","call_id":"call_1","output":"12:00"},` +
		`{"type":"function_call_output","call_id":"call_2","output":[{"type":"output_text","text":"5:00"}]},` +
		`{"role":"user","content":"thanks"}]}`
	out, _, err := ResponsesToChat([]byte(in), Options{})
	if err != nil {
		t.Fatalf("ResponsesToChat: %v", err)
	}
	m := decodeMap(t, out)
	msgs, _ := m["messages"].([]any)
	if len(msgs) != 5 {
		t.Fatalf("want user, assistant(tool_calls), tool, tool, user: %+v", m["messages"])
	}
	am, _ := msgs[1].(map[string]any)
	calls, _ := am["tool_calls"].([]any)
	if am["role"] != "assistant" || len(calls) != 2 {
		t.Fatalf("parallel calls must merge into one assistant turn: %+v", msgs[1])
	}
	c0, _ := calls[0].(map[string]any)
	fn0, _ := c0["function"].(map[string]any)
	if c0["id"] != "call_1" || c0["type"] != "function" ||
		fn0["name"] != "clock" || fn0["arguments"] != `{"tz":1}` {
		t.Fatalf("function_call not mapped: %+v", c0)
	}
	t0, _ := msgs[2].(map[string]any)
	t1, _ := msgs[3].(map[string]any)
	if t0["role"] != "tool" || t0["tool_call_id"] != "call_1" || t0["content"] != "12:00" {
		t.Fatalf("function_call_output not mapped: %+v", msgs[2])
	}
	if t1["tool_call_id"] != "call_2" || t1["content"] != "5:00" {
		t.Fatalf("an output of content parts must flatten to text: %+v", msgs[3])
	}
}

func TestResponsesToChatToolsAndFormat(t *testing.T) {
	in := `{"model":"m","input":"x","tools":[` +
		`{"type":"function","name":"fetch","description":"d","parameters":{"type":"object"},"strict":true}],` +
		`"tool_choice":{"type":"function","name":"fetch"},` +
		`"text":{"format":{"type":"json_schema","name":"answer","schema":{"type":"object"},"strict":true}},` +
		`"reasoning":{"effort":"high"}}`
	out, notes, err := ResponsesToChat([]byte(in), Options{})
	if err != nil {
		t.Fatalf("ResponsesToChat: %v", err)
	}
	m := decodeMap(t, out)
	tools, _ := m["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools: %+v", m["tools"])
	}
	tl, _ := tools[0].(map[string]any)
	fn, _ := tl["function"].(map[string]any)
	if tl["type"] != "function" || fn["name"] != "fetch" || fn["description"] != "d" ||
		fn["parameters"] == nil || fn["strict"] != true {
		t.Fatalf("tool not nested as chat expects: %+v", tl)
	}
	tc, _ := m["tool_choice"].(map[string]any)
	tcFn, _ := tc["function"].(map[string]any)
	if tc["type"] != "function" || tcFn["name"] != "fetch" {
		t.Fatalf("tool_choice not mapped: %+v", m["tool_choice"])
	}
	rf, _ := m["response_format"].(map[string]any)
	js, _ := rf["json_schema"].(map[string]any)
	if rf["type"] != "json_schema" || js["name"] != "answer" || js["schema"] == nil || js["strict"] != true {
		t.Fatalf("text.format must nest one level deeper: %+v", m["response_format"])
	}
	if m["reasoning_effort"] != "high" {
		t.Fatalf("reasoning.effort not mapped: %+v", m["reasoning_effort"])
	}
	if len(notes) != 0 {
		t.Fatalf("notes: %v", notes)
	}
}

// A Responses client whose tool list carries a hosted tool asked for a search the
// chat backend can not run: the answer would be about a tool call that never
// happened. Always-400 class, drop_params can not buy it off.
func TestResponsesToChatRefusesHostedTool(t *testing.T) {
	for _, drop := range []bool{false, true} {
		_, _, err := ResponsesToChat([]byte(`{"model":"m","input":"x","tools":[{"type":"web_search"}]}`),
			Options{DropParams: drop})
		var e *Error
		if err == nil || !asError(err, &e) || e.Code != CodeField || e.Field != "tools" {
			t.Fatalf("hosted tool must stay a field refusal, got %v (drop_params=%v)", err, drop)
		}
		if strings.Contains(e.Msg, "drop_params") {
			t.Fatalf("an always-refused field must not invite drop_params: %s", e.Msg)
		}
	}
}

// The fields the proxy does not execute are reported, never refused: LiteLLM
// clients set `store` on every request, and a 400 there would stop real traffic
// on a field that changes nothing about the answer.
func TestResponsesToChatReportsUnexecutedFields(t *testing.T) {
	out, notes, err := ResponsesToChat([]byte(`{"model":"m","input":"x","store":true,`+
		`"include":["reasoning.summary"],"truncation":"auto","background":false,`+
		`"safety_identifier":"u1","metadata":{"k":"v"},"top_logprobs":3}`), Options{})
	if err != nil {
		t.Fatalf("ResponsesToChat: %v", err)
	}
	if len(out) == 0 {
		t.Fatal("no body")
	}
	for _, want := range []string{"store=dropped", "include=dropped", "truncation=dropped",
		"background=dropped", "safety_identifier=dropped", "metadata=dropped", "top_logprobs=dropped"} {
		if !slices.Contains(notes, want) {
			t.Fatalf("notes = %v, want %s", notes, want)
		}
	}
}

// A chain of thought the client encrypted for a Responses backend has no chat
// representation, and the signed Anthropic form can not be rebuilt either:
// refused by default, a reported drop on request.
func TestResponsesToChatRefusesReasoningItems(t *testing.T) {
	in := `{"model":"m","input":[` +
		`{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"because"}]},` +
		`{"role":"user","content":"go"}]}`
	_, _, err := ResponsesToChat([]byte(in), Options{})
	var e *Error
	if err == nil || !asError(err, &e) || e.Code != CodeField || !strings.Contains(e.Msg, "reasoning") {
		t.Fatalf("a reasoning item must be refused by default: %v", err)
	}
	out, notes, err := ResponsesToChat([]byte(in), Options{DropParams: true})
	if err != nil {
		t.Fatalf("with drop_params: %v", err)
	}
	m := decodeMap(t, out)
	if msgs, _ := m["messages"].([]any); len(msgs) != 1 {
		t.Fatalf("the dropped item must leave no message behind: %+v", m["messages"])
	}
	if !slices.Contains(notes, "input_item=dropped:reasoning") {
		t.Fatalf("notes = %v, want the dropped item reported", notes)
	}
}

// `previous_response_id` is a promise the router can not keep: it stores nothing,
// so a continuation would silently answer a truncated conversation. The refusal
// has to tell the client what to send instead.
func TestResponsesToChatRefusesPreviousResponseID(t *testing.T) {
	_, _, err := ResponsesToChat([]byte(`{"model":"m","input":"x","previous_response_id":"resp_9"}`),
		Options{DropParams: true})
	var e *Error
	if err == nil || !asError(err, &e) || e.Code != CodeState || e.Field != "previous_response_id" {
		t.Fatalf("previous_response_id must refuse with the state code: %v", err)
	}
	if !strings.Contains(e.Msg, "input") {
		t.Fatalf("the refusal must say what to send instead: %s", e.Msg)
	}
}

func TestResponsesToChatToolChoiceStrings(t *testing.T) {
	for _, tc := range []string{"auto", "none", "required"} {
		out, _, err := ResponsesToChat([]byte(`{"model":"m","input":"x","tool_choice":"`+tc+`"}`), Options{})
		if err != nil {
			t.Fatalf("%s: %v", tc, err)
		}
		if got := decodeMap(t, out)["tool_choice"]; got != tc {
			t.Fatalf("%s: tool_choice = %v", tc, got)
		}
	}
	// allowed_tools / custom / hosted choices constrain a tool set chat cannot
	// express: refused by default, a reported drop under drop_params.
	for _, tc := range []string{`{"type":"allowed_tools","mode":"required","tools":[]}`,
		`{"type":"custom","format":{"type":"text"}}`} {
		_, notes, err := ResponsesToChat([]byte(`{"model":"m","input":"x","tool_choice":`+tc+`}`),
			Options{DropParams: true})
		if err != nil {
			t.Fatalf("%s: %v", tc, err)
		}
		if !slices.Contains(notes, "tool_choice=dropped") {
			t.Fatalf("%s: notes = %v, want the drop reported", tc, notes)
		}
	}
}

func TestResponsesToChatRefusesDocumentPart(t *testing.T) {
	_, _, err := ResponsesToChat([]byte(`{"model":"m","input":[{"role":"user","content":`+
		`[{"type":"file","file":{"file_id":"f-1"}}]}]}`), Options{DropParams: true})
	var e *Error
	if err == nil || !asError(err, &e) || e.Code != CodeField || e.Field != "file" {
		t.Fatalf("a file part is the always-400 class: %v", err)
	}
}

func TestResponsesToChatIdentityFields(t *testing.T) {
	out, _, err := ResponsesToChat([]byte(`{"model":"m","input":"x","user":"u1",`+
		`"prompt_cache_key":"pk","service_tier":"default"}`), Options{})
	if err != nil {
		t.Fatalf("ResponsesToChat: %v", err)
	}
	m := decodeMap(t, out)
	if m["user"] != "u1" || m["prompt_cache_key"] != "pk" || m["service_tier"] != "default" {
		t.Fatalf("identity fields must ride along (jev stickiness reads them): %+v", m)
	}
}

func TestResponsesToChatRefusesGarbage(t *testing.T) {
	if _, _, err := ResponsesToChat([]byte(`{`), Options{}); err == nil {
		t.Fatal("a body that is not JSON must refuse")
	}
	if _, _, err := ResponsesToChat([]byte(`{"model":"m"}`), Options{}); err == nil {
		t.Fatal("a request with no input refuses: an empty conversation is not a question")
	}
}

func asError(err error, target **Error) bool {
	e, ok := err.(*Error)
	if ok {
		*target = e
	}
	return ok
}

// The JSON direction of the same spoke: a chat answer has to arrive as a
// Responses answer, because a Responses SDK decodes `output` ITEMS (a message item
// with `output_text` parts, one function_call item per call) and reads `status`,
// not `choices` and `finish_reason`.

func TestChatToResponsesResponse(t *testing.T) {
	in := `{"id":"chatcmpl-9","model":"claude","object":"chat.completion",` +
		`"choices":[{"index":0,"message":{"role":"assistant",` +
		`"content":[{"type":"text","text":"hi there"}],` +
		`"tool_calls":[{"id":"call_1","type":"function","function":{"name":"fetch","arguments":"{\"u\":1}"}}]},` +
		`"finish_reason":"tool_calls"}],` +
		`"usage":{"prompt_tokens":120,"completion_tokens":17,` +
		`"prompt_tokens_details":{"cached_tokens":100}}}`
	out, notes, err := ChatToResponses([]byte(in))
	if err != nil {
		t.Fatalf("ChatToResponses: %v", err)
	}
	m := decodeMap(t, out)
	if m["object"] != "response" || m["id"] != "chatcmpl-9" || m["model"] != "claude" {
		t.Fatalf("envelope: %+v", m)
	}
	if m["status"] != "completed" {
		t.Fatalf("status = %v, want completed", m["status"])
	}
	if _, ok := m["created_at"].(float64); !ok {
		t.Fatalf("created_at missing: %+v", m["created_at"])
	}
	items, _ := m["output"].([]any)
	if len(items) != 2 {
		t.Fatalf("want a message item and a function_call item: %+v", m["output"])
	}
	msg, _ := items[0].(map[string]any)
	if msg["type"] != "message" || msg["role"] != "assistant" || msg["status"] != "completed" {
		t.Fatalf("message item: %+v", msg)
	}
	parts, _ := msg["content"].([]any)
	if len(parts) != 1 {
		t.Fatalf("output_text part: %+v", msg["content"])
	}
	part, _ := parts[0].(map[string]any)
	if part["type"] != "output_text" || part["text"] != "hi there" {
		t.Fatalf("output part: %+v", part)
	}
	call, _ := items[1].(map[string]any)
	if call["type"] != "function_call" || call["call_id"] != "call_1" ||
		call["name"] != "fetch" || call["arguments"] != `{"u":1}` {
		t.Fatalf("function_call item: %+v", call)
	}
	usage, _ := m["usage"].(map[string]any)
	// Responses spells usage like Anthropic does: input_tokens sits next to the
	// cached count, so the conventions must not be added together.
	if usage["input_tokens"] != float64(20) || usage["output_tokens"] != float64(17) {
		t.Fatalf("usage convention not converted: %+v", usage)
	}
	details, _ := usage["input_tokens_details"].(map[string]any)
	if details["cached_tokens"] != float64(100) || usage["total_tokens"] != float64(37) {
		t.Fatalf("usage details: %+v", usage)
	}
	if len(notes) != 0 {
		t.Fatalf("notes: %v", notes)
	}
}

// Chat's `reasoning_content` is plaintext with no signature, and Responses'
// reasoning item carries a summary: that is where a chat backend's chain of
// thought belongs, and it is the shape vLLM/SGLang emit natively for it.
func TestChatToResponsesReasoningBecomesAReasoningItem(t *testing.T) {
	out, _, err := ChatToResponses([]byte(`{"id":"c","model":"m","choices":[{"finish_reason":"stop",` +
		`"message":{"role":"assistant","content":"answer","reasoning_content":"because"}}]}`))
	if err != nil {
		t.Fatalf("ChatToResponses: %v", err)
	}
	items, _ := decodeMap(t, out)["output"].([]any)
	if len(items) != 2 {
		t.Fatalf("want a reasoning item and a message item: %+v", items)
	}
	rs, _ := items[0].(map[string]any)
	if rs["type"] != "reasoning" {
		t.Fatalf("first item = %v, want reasoning", rs["type"])
	}
	sum, _ := rs["summary"].([]any)
	s0, _ := sum[0].(map[string]any)
	if len(sum) != 1 || s0["type"] != "summary_text" || s0["text"] != "because" {
		t.Fatalf("reasoning summary: %+v", rs["summary"])
	}
	if id, _ := rs["id"].(string); id == "" {
		t.Fatalf("a reasoning item with no id: %+v", rs)
	}
}

// `status` is a Responses client's whole view of why the answer stopped:
// `length` must arrive as an incomplete response with the reason the dialect
// spells, and an unmapped finish reason as a note (the response direction can not
// refuse — the answer is already at the client).
func TestChatToResponsesStatusFromFinishReason(t *testing.T) {
	for fr, want := range map[string][2]string{"stop": {"completed", ""}, "tool_calls": {"completed", ""},
		"": {"completed", ""}, "function_call": {"completed", ""}, "eos_token": {"completed", ""},
		"length": {"incomplete", "max_output_tokens"}, "content_filter": {"incomplete", "content_filter"},
		"refusal": {"incomplete", "content_filter"}, "abort": {"completed", ""}} {
		out, _, err := ChatToResponses([]byte(`{"id":"c","model":"m","choices":[{"message":` +
			`{"role":"assistant","content":"t"},"finish_reason":"` + fr + `"}]}`))
		if err != nil {
			t.Fatalf("%s: %v", fr, err)
		}
		m := decodeMap(t, out)
		if m["status"] != want[0] {
			t.Fatalf("%s: status = %v, want %s", fr, m["status"], want[0])
		}
		details, _ := m["incomplete_details"].(map[string]any)
		if want[1] == "" {
			if m["incomplete_details"] != nil {
				t.Fatalf("%s: an incomplete_details on a %s response: %+v", fr, want[0], m["incomplete_details"])
			}
			continue
		}
		if details["reason"] != want[1] {
			t.Fatalf("%s: incomplete_details = %+v, want reason %s", fr, m["incomplete_details"], want[1])
		}
	}
	// The rows the mapping table has lose nothing; a reason it has no row for is a
	// loss, and notes are the only channel a mapped response has.
	for _, fr := range []string{"stop", "length", "tool_calls", "function_call"} {
		_, nts, _ := ChatToResponses([]byte(`{"id":"c","model":"m","choices":[{"message":` +
			`{"role":"assistant","content":"t"},"finish_reason":"` + fr + `"}]}`))
		if len(nts) != 0 {
			t.Fatalf("%q: notes = %v (nothing is lost here)", fr, nts)
		}
	}
	for _, fr := range []string{"refusal", "eos_token", "abort", "content_filter"} {
		_, notes, _ := ChatToResponses([]byte(`{"id":"c","model":"m","choices":[{"message":` +
			`{"role":"assistant","content":"t"},"finish_reason":"` + fr + `"}]}`))
		if !slices.Contains(notes, "finish_reason="+fr) {
			t.Fatalf("%s: notes = %v, want the collapsed finish reason reported", fr, notes)
		}
	}
}

func TestChatToResponsesReportsWhatItLost(t *testing.T) {
	_, notes, err := ChatToResponses([]byte(`{"id":"c","model":"m","choices":[` +
		`{"index":0,"message":{"role":"assistant","content":[{"type":"text","text":"hi"},` +
		`{"type":"image_url","image_url":{"url":"data:image/png;base64,AAA"}}]},` +
		`"finish_reason":"stop"},` +
		`{"index":1,"message":{"role":"assistant","content":"second"},"finish_reason":"stop"}]}`))
	if err != nil {
		t.Fatalf("ChatToResponses: %v", err)
	}
	for _, want := range []string{"choices=dropped:1", "content_part=dropped:image_url"} {
		if !slices.Contains(notes, want) {
			t.Fatalf("notes = %v, want %s", notes, want)
		}
	}
}

// An upstream 4xx/5xx body is already in the client's dialect: the Responses API
// reports errors in OpenAI's envelope, so translating it would be the damage.
func TestChatToResponsesLeavesErrorEnvelopeAlone(t *testing.T) {
	body := `{"error":{"message":"model not found","type":"invalid_request_error"}}`
	out, notes, err := ChatToResponses([]byte(body))
	if err != nil {
		t.Fatalf("ChatToResponses: %v", err)
	}
	if string(out) != body {
		t.Fatalf("an OpenAI error body must reach a Responses client untouched: %s", out)
	}
	if len(notes) != 0 {
		t.Fatalf("notes: %v", notes)
	}
}

func TestResponseDispatchToResponses(t *testing.T) {
	body := []byte(`{"id":"c","model":"m","choices":[{"message":{"role":"assistant","content":"t"}}]}`)
	got, _, err := Response(Plan{From: Responses, Dialect: Chat}, body)
	if err != nil {
		t.Fatalf("Response: %v", err)
	}
	if m := decodeMap(t, got); m["object"] != "response" {
		t.Fatalf("not a Responses answer: %s", got)
	}
	if _, _, err := Response(Plan{From: Chat, Dialect: Responses}, body); err == nil {
		t.Fatal("the chat-client-on-responses-backend direction is not implemented")
	}
}
