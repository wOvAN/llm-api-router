package convert

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"
)

func TestChatToMessagesResponse(t *testing.T) {
	in := `{"id":"chatcmpl-9","model":"claude","object":"chat.completion",` +
		`"choices":[{"index":0,"message":{"role":"assistant",` +
		`"content":[{"type":"text","text":"hi there"}],` +
		`"tool_calls":[{"id":"call_1","type":"function","function":{"name":"fetch","arguments":"{\"u\":1}"}}]},` +
		`"finish_reason":"tool_calls"}],` +
		`"usage":{"prompt_tokens":120,"completion_tokens":17,` +
		`"prompt_tokens_details":{"cached_tokens":100}}}`

	out, notes, err := ChatToMessages([]byte(in))
	if err != nil {
		t.Fatalf("ChatToMessages: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("not JSON: %v %s", err, out)
	}
	if m["type"] != "message" || m["role"] != "assistant" || m["model"] != "claude" {
		t.Fatalf("envelope: %+v", m)
	}
	if m["stop_reason"] != "tool_use" {
		t.Fatalf("stop_reason = %v, want tool_use", m["stop_reason"])
	}
	blocks, _ := m["content"].([]any)
	if len(blocks) != 2 {
		t.Fatalf("want a text block and a tool_use block: %+v", m["content"])
	}
	blk, _ := blocks[1].(map[string]any)
	if blk["type"] != "tool_use" || blk["id"] != "call_1" || blk["name"] != "fetch" {
		t.Fatalf("tool_use block: %+v", blk)
	}
	input, _ := blk["input"].(map[string]any)
	if input["u"] == nil {
		t.Fatalf("arguments not parsed into input: %+v", blk)
	}
	usage, _ := m["usage"].(map[string]any)
	// OpenAI's prompt_tokens already contains its cached tokens; Anthropic's
	// input_tokens sits next to cache_read_input_tokens.
	if usage["input_tokens"] != float64(20) || usage["cache_read_input_tokens"] != float64(100) ||
		usage["output_tokens"] != float64(17) {
		t.Fatalf("usage convention not converted: %+v", usage)
	}
	if len(notes) != 0 {
		t.Fatalf("notes: %v", notes)
	}
}

func TestChatToMessagesFinishReasonMap(t *testing.T) {
	for fr, want := range map[string]string{"stop": "end_turn", "length": "max_tokens",
		"content_filter": "refusal", "tool_calls": "tool_use", "": "end_turn"} {
		out, _, err := ChatToMessages([]byte(`{"id":"x","model":"m","choices":[{"finish_reason":"` + fr +
			`","message":{"role":"assistant","content":"t"}}]}`))
		if err != nil {
			t.Fatalf("%s: %v", fr, err)
		}
		var m map[string]any
		if err := json.Unmarshal(out, &m); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if m["stop_reason"] != want {
			t.Fatalf("%q → %v, want %s", fr, m["stop_reason"], want)
		}
	}
}

func TestChatToMessagesReasoningBecomesThinking(t *testing.T) {
	out, _, err := ChatToMessages([]byte(`{"id":"x","model":"m","choices":[{"finish_reason":"stop",` +
		`"message":{"role":"assistant","content":"answer","reasoning_content":"because"}}]}`))
	if err != nil {
		t.Fatalf("ChatToMessages: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("decode: %v", err)
	}
	blocks, _ := m["content"].([]any)
	if len(blocks) != 2 {
		t.Fatalf("want thinking + text: %+v", m["content"])
	}
	first, _ := blocks[0].(map[string]any)
	if first["type"] != "thinking" || first["thinking"] != "because" {
		t.Fatalf("reasoning_content not mapped to a thinking block: %+v", first)
	}
}

// The response direction can not refuse — the answer is already at the client —
// so notes are its only channel, and the mapping loses real information in three
// places: an `n>1` answer maps `choices[0]` only, a finish reason the mapping has
// no row for collapses to a milder `stop_reason`, and a content part it cannot
// read as text leaves entirely. Every one of them is a loss the operator must see
// (R24), and none of them was visible.
func TestChatToMessagesResponseReportsWhatItLost(t *testing.T) {
	out, notes, err := ChatToMessages([]byte(`{"id":"c","model":"m","choices":[` +
		`{"index":0,"message":{"role":"assistant","content":[{"type":"text","text":"hi"},` +
		`{"type":"image_url","image_url":{"url":"data:image/png;base64,AAA"}}]},` +
		`"finish_reason":"content_filter"},` +
		`{"index":1,"message":{"role":"assistant","content":"second"},"finish_reason":"stop"}]}`))
	if err != nil {
		t.Fatalf("ChatToMessages: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("not JSON: %v", err)
	}
	if m["stop_reason"] != "refusal" {
		t.Fatalf("stop_reason = %v, want refusal", m["stop_reason"])
	}
	blocks, _ := m["content"].([]any)
	if len(blocks) != 1 {
		t.Fatalf("only the text part maps: %+v", m["content"])
	}
	for _, want := range []string{"choices=dropped:1", "finish_reason=content_filter",
		"content_part=dropped:image_url"} {
		if !slices.Contains(notes, want) {
			t.Fatalf("notes = %v, want %s", notes, want)
		}
	}
}

// Notes are bounded at 512 bytes at the consumer and the cap keeps the head, so
// one repeated loss must be one note: a multimodal answer carrying a dozen image
// parts would otherwise evict every other note from the metric and the GUI
// tooltip (the emitter's own `note()` already dedups — this direction must too).
func TestChatToMessagesResponseNotesAreReportedOnce(t *testing.T) {
	part := `{"type":"image_url","image_url":{"url":"data:image/png;base64,AAA"}},`
	body := `{"id":"c","model":"m","choices":[{"index":0,"message":{"role":"assistant",` +
		`"content":[` + strings.Repeat(part, 12) +
		`{"type":"text","text":"hi"}]},"finish_reason":"content_filter"},` +
		`{"index":1,"message":{"role":"assistant","content":"x"},"finish_reason":"stop"}]}`
	_, notes, err := ChatToMessages([]byte(body))
	if err != nil {
		t.Fatalf("ChatToMessages: %v", err)
	}
	if len(notes) != 3 {
		t.Fatalf("a repeated loss must be reported once, notes = %v", notes)
	}
}

// A refused answer is the case the spec's loss table names: it must arrive as an
// `end_turn` completion WITH a note, never as a silent end_turn — the client's
// whole view of why the answer stopped is that field.
func TestChatToMessagesResponseNotesACollapsedFinishReason(t *testing.T) {
	for _, fr := range []string{"refusal", "eos_token", "abort"} {
		out, notes, err := ChatToMessages([]byte(`{"id":"c","model":"m","choices":[{"message":` +
			`{"role":"assistant","content":"partial"},"finish_reason":"` + fr + `"}]}`))
		if err != nil {
			t.Fatalf("%s: %v", fr, err)
		}
		var m map[string]any
		if err := json.Unmarshal(out, &m); err != nil {
			t.Fatalf("%s: decode: %v", fr, err)
		}
		if m["stop_reason"] != "end_turn" {
			t.Fatalf("%s: stop_reason = %v, want end_turn", fr, m["stop_reason"])
		}
		if !slices.Contains(notes, "finish_reason="+fr) {
			t.Fatalf("%s: the collapsed finish reason must be reported: %v", fr, notes)
		}
	}
	// The mapping's own rows lose nothing, and a backend that reported no finish
	// reason has nothing to report: no note.
	for _, fr := range []string{"", "stop", "length", "tool_calls", "function_call"} {
		_, notes, err := ChatToMessages([]byte(`{"id":"c","model":"m","choices":[{"message":` +
			`{"role":"assistant","content":"t"},"finish_reason":"` + fr + `"}]}`))
		if err != nil || len(notes) != 0 {
			t.Fatalf("%q: notes = %v, err = %v (nothing is lost here)", fr, notes, err)
		}
	}
}

func TestResponseDispatch(t *testing.T) {
	body := []byte(`{"choices":[]}`)
	got, _, err := Response(Plan{From: Chat, Dialect: Chat}, body)
	if err != nil || string(got) != string(body) {
		t.Fatalf("identity response must be a no-op: %s %v", got, err)
	}
	if _, _, err := Response(Plan{From: Chat, Dialect: Responses}, body); err == nil {
		t.Fatal("an unimplemented response pair must refuse")
	}
}

// A backend's 4xx/5xx body rides the same JSON path as an answer (HTTP 5xx is not
// an error to this proxy — it is forwarded as-is). An Anthropic SDK can not read
// OpenAI's `{"error":{"message":…}}`, so an untranslated error body reaches the
// client as a parse failure that hides the backend's actual reason.
func TestResponseTranslatesErrorEnvelope(t *testing.T) {
	out, _, err := Response(Plan{From: Messages, Dialect: Chat},
		[]byte(`{"error":{"message":"model not found","type":"invalid_request_error"}}`))
	if err != nil {
		t.Fatalf("Response: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("not JSON: %s", out)
	}
	if m["type"] != "error" {
		t.Fatalf("not an Anthropic error envelope: %s", out)
	}
	e, _ := m["error"].(map[string]any)
	if e["type"] != "invalid_request_error" || e["message"] != "model not found" {
		t.Fatalf("error fields lost: %s", out)
	}
	// A body that is not an error envelope goes to the normal path.
	if _, ok, _ := chatErrorToMessages([]byte(`{"choices":[]}`)); ok {
		t.Fatal("a chat answer must not match the error path")
	}
}

// OpenAI's `arguments` is a JSON-encoded string; a backend may also send an
// object. Both must arrive as an Anthropic object `input`, unparseable arguments
// must arrive as `{}` (a tool_use block with a string input is not a tool_use
// block), and a tool call with no id keeps a synthetic one — an Anthropic client
// keys its tool_result on that id.
func TestChatToMessagesToolArguments(t *testing.T) {
	for _, args := range []string{`"{\"u\":1}"`, `{"u": 1}`, `""`, `"not json"`} {
		out, _, err := ChatToMessages([]byte(`{"id":"c","model":"m","choices":[{"message":` +
			`{"role":"assistant","content":"t","tool_calls":[{"function":{"name":"f","arguments":` +
			args + `}}]}}]}`))
		if err != nil {
			t.Fatalf("%s: %v", args, err)
		}
		var m map[string]any
		if err := json.Unmarshal(out, &m); err != nil {
			t.Fatalf("decode: %v", err)
		}
		blocks, _ := m["content"].([]any)
		if len(blocks) != 2 {
			t.Fatalf("%s: want a text block and a tool_use block: %s", args, out)
		}
		blk, _ := blocks[1].(map[string]any)
		if _, ok := blk["input"].(map[string]any); !ok {
			t.Fatalf("arguments %s → input %#v, want an object", args, blk["input"])
		}
		if id, _ := blk["id"].(string); id == "" {
			t.Fatalf("arguments %s: a tool call with no id lost its synthetic one: %s", args, out)
		}
	}
}
