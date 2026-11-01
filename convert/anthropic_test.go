package convert

import (
	"encoding/json"
	"errors"
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
