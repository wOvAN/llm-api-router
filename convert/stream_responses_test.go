package convert

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"
)

// The streamed direction of the responses spoke: a Chat Completions chunk stream
// rebuilt as Responses events. A Responses client reads a lifecycle the chat
// stream has no equivalent for — items added and done, a content part inside a
// message item, argument deltas per function call, a terminal event carrying the
// whole response object — so the emitter allocates items as content kinds appear
// and closes them at the end.

func TestChatToResponsesStreamText(t *testing.T) {
	out := emit(NewChatToResponses(),
		`{"id":"c1","model":"claude","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}`,
		`{"id":"c1","model":"claude","choices":[{"index":0,"delta":{"content":"lo"},`+
			`"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":2,`+
			`"prompt_tokens_details":{"cached_tokens":3}}}`,
		`[DONE]`)

	want := []string{"response.created", "response.output_item.added", "response.content_part.added",
		"response.output_text.delta", "response.output_text.delta",
		"response.output_text.done", "response.content_part.done",
		"response.output_item.done", "response.completed"}
	names := eventNames(out)
	for _, w := range want {
		if !slices.Contains(names, w) {
			t.Fatalf("stream lacks %s: %v\n%s", w, names, out)
		}
	}
	if !strings.Contains(out, `"delta":"Hel"`) || !strings.Contains(out, `"delta":"lo"`) {
		t.Fatalf("text deltas lost:\n%s", out)
	}
	if !strings.Contains(out, `"text":"Hello"`) {
		t.Fatalf("the .done event must carry the part's whole text:\n%s", out)
	}
	last := names[len(names)-1]
	if !strings.Contains(out, "data: [DONE]") || !strings.Contains(out, `"type":"response.completed"`) {
		t.Fatalf("a Responses client needs the terminal event and the terminator: %v", names)
	}
	if last != "data: [DONE]" {
		t.Fatalf("the stream must end on the [DONE] frame, got %q", last)
	}
	// The response object is the metering channel a Responses client reads.
	tail := out[strings.Index(out, "event: response.completed"):]
	for _, want := range []string{`"input_tokens":4`, `"output_tokens":2`, `"cached_tokens":3`,
		`"total_tokens":6`, `"status":"completed"`, `"id":"c1"`} {
		if !strings.Contains(tail, want) {
			t.Fatalf("response.completed lacks %s:\n%s", want, tail)
		}
	}
	for _, f := range strings.Split(strings.TrimRight(out, "\n"), "\n\n") {
		if strings.HasPrefix(f, "data: [DONE]") {
			continue
		}
		if !strings.HasPrefix(f, "event: ") || !strings.Contains(f, "\ndata: ") {
			t.Fatalf("incomplete frame emitted: %q", f)
		}
	}
}

// `sequence_number` is required on every Responses event, and monotonic: the SDK's
// accumulator and every debugger on that stream order events by it.
func TestChatToResponsesStreamNumbersEveryEvent(t *testing.T) {
	out := emit(NewChatToResponses(),
		`{"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"a"}}]}`,
		`{"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"b"},"finish_reason":"stop"}]}`,
		`[DONE]`)
	prev := int64(-1)
	n := 0
	for _, f := range strings.Split(strings.TrimRight(out, "\n"), "\n\n") {
		if strings.HasPrefix(f, "data: [DONE]") {
			continue
		}
		_, data, ok := strings.Cut(f, "\n")
		if !ok {
			t.Fatalf("frame lacks a data line: %q", f)
		}
		payload := strings.TrimPrefix(strings.TrimSpace(data), "data:")
		if strings.TrimSpace(payload) == "[DONE]" {
			continue
		}
		var ev struct {
			Type           string `json:"type"`
			SequenceNumber *int64 `json:"sequence_number"`
		}
		if err := json.Unmarshal([]byte(payload), &ev); err != nil {
			t.Fatalf("frame payload is not JSON: %v\n%s", err, payload)
		}
		if ev.SequenceNumber == nil {
			t.Fatalf("%s carries no sequence_number:\n%s", ev.Type, payload)
		}
		if *ev.SequenceNumber <= prev {
			t.Fatalf("sequence_number is not monotonic: %d after %d (%s)", *ev.SequenceNumber, prev, ev.Type)
		}
		prev, n = *ev.SequenceNumber, n+1
	}
	if n < 3 {
		t.Fatalf("expected a created/added/delta/done sequence, got %d events:\n%s", n, out)
	}
}

func TestChatToResponsesStreamClosesAStreamEndedByDone(t *testing.T) {
	out := emit(NewChatToResponses(),
		`{"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"partial"}}]}`,
		`[DONE]`)
	names := eventNames(out)
	for _, want := range []string{"response.output_text.done", "response.output_item.done", "response.completed"} {
		if !slices.Contains(names, want) {
			t.Fatalf("a stream that ended without a finish_reason must still close: %v", names)
		}
	}
}

// The tool loop is the shape that matters most: parallel calls interleave their
// argument deltas, so each call needs its own item keyed on the chunk's
// `tool_calls[].index`, and the call's `call_id` is what ties it to the
// function_call_output the client will send next.
func TestChatToResponsesStreamToolCalls(t *testing.T) {
	out := emit(NewChatToResponses(),
		`{"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":`+
			`[{"index":0,"id":"call_a","function":{"name":"fetch","arguments":"{\"a\":"}}]}}]}`,
		`{"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":`+
			`[{"index":1,"id":"call_b","function":{"name":"store","arguments":"{\"b\":"}}]}}]}`,
		`{"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":`+
			`[{"index":0,"function":{"arguments":"1}"}}]}}]}`,
		`{"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":`+
			`[{"index":1,"function":{"arguments":"2}"}}]},"finish_reason":"tool_calls"}]}`,
		`[DONE]`)
	names := eventNames(out)
	for _, want := range []string{"response.output_item.added", "response.function_call_arguments.delta",
		"response.function_call_arguments.done", "response.output_item.done", "response.completed"} {
		if !slices.Contains(names, want) {
			t.Fatalf("stream lacks %s: %v\n%s", want, names, out)
		}
	}
	if !strings.Contains(out, `"call_id":"call_a"`) || !strings.Contains(out, `"call_id":"call_b"`) {
		t.Fatalf("call ids lost:\n%s", out)
	}
	var a, b string
	for _, f := range strings.Split(out, "\n\n") {
		if !strings.Contains(f, "response.function_call_arguments.done") {
			continue
		}
		_, data, _ := strings.Cut(f, "\n")
		var ev struct {
			CallID    string `json:"call_id"`
			Arguments string `json:"arguments"`
		}
		if json.Unmarshal([]byte(strings.TrimPrefix(strings.TrimSpace(data), "data:")), &ev) != nil {
			t.Fatalf("arguments.done is not JSON: %s", f)
		}
		switch ev.CallID {
		case "call_a":
			a = ev.Arguments
		case "call_b":
			b = ev.Arguments
		}
	}
	if a != `{"a":1}` || b != `{"b":2}` {
		t.Fatalf("interleaved arguments merged into the wrong call: %q %q", a, b)
	}
	// The terminal event carries the items, so an SDK that never saw the deltas
	// still reads the calls.
	tail := out[strings.Index(out, "event: response.completed"):]
	if !strings.Contains(tail, `"name":"fetch"`) || !strings.Contains(tail, `"name":"store"`) {
		t.Fatalf("response.completed must carry every item:\n%s", tail)
	}
}

func TestChatToResponsesStreamThinkingBecomesReasoning(t *testing.T) {
	out := emit(NewChatToResponses(),
		`{"id":"c1","model":"m","choices":[{"index":0,"delta":{"reasoning_content":"because"}}]}`,
		`{"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"answer"},`+
			`"finish_reason":"stop"}]}`,
		`[DONE]`)
	names := eventNames(out)
	for _, want := range []string{"response.reasoning_summary_text.delta",
		"response.reasoning_summary_text.done", "response.output_text.delta", "response.completed"} {
		if !slices.Contains(names, want) {
			t.Fatalf("stream lacks %s: %v\n%s", want, names, out)
		}
	}
	if !strings.Contains(out, `"delta":"because"`) {
		t.Fatalf("reasoning lost:\n%s", out)
	}
	if !strings.Contains(out, `"type":"reasoning"`) {
		t.Fatalf("the reasoning item was never added:\n%s", out)
	}
}

// `length` is a truncated answer, and a Responses client reads that as an
// incomplete response with a reason — `response.completed` would tell it the
// answer is finished.
func TestChatToResponsesStreamStatusFromFinishReason(t *testing.T) {
	out := emit(NewChatToResponses(),
		`{"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"cut off"},`+
			`"finish_reason":"length"}]}`,
		`[DONE]`)
	if !strings.Contains(out, `"type":"response.incomplete"`) ||
		!strings.Contains(out, `"reason":"max_output_tokens"`) {
		t.Fatalf("a truncated answer must arrive as an incomplete response:\n%s", out)
	}
	// A finish reason the mapping has no row for is a loss the notes carry.
	s := NewChatToResponses()
	emit(s, `{"id":"c","model":"m","choices":[{"index":0,"delta":{"content":"x"},`+
		`"finish_reason":"content_filter"}]}`, `[DONE]`)
	if notes := s.Notes(); !slices.Contains(notes, "finish_reason=content_filter") {
		t.Fatalf("notes = %v, want the collapsed finish reason reported", notes)
	}
}

// The proxy's own loop-detector frame is authored in the client's dialect, and for
// an OpenAI-family client that is a chunk with `finish_reason:"error"`. A Responses
// client has no such value: it must land as a failed response, not a completed one.
func TestChatToResponsesStreamFailsAResponseThatEndedInError(t *testing.T) {
	out := emit(NewChatToResponses(),
		`{"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"loop"},`+
			`"finish_reason":"error"}]}`)
	if !strings.Contains(out, `"type":"response.failed"`) {
		t.Fatalf("a stream the proxy ended must fail the response:\n%s", out)
	}
	if strings.Contains(out, "response.completed") {
		t.Fatalf("a failed stream must not also complete:\n%s", out)
	}
}

func TestChatToResponsesStreamForwardsAnUpstreamErrorChunk(t *testing.T) {
	s := NewChatToResponses()
	frames := s.WriteFrame([]byte("data: " + `{"error":{"message":"boom","type":"server_error"}}` + "\n\n"))
	if len(frames) != 1 || !strings.HasPrefix(string(frames[0]), "event: error\n") {
		t.Fatalf("an upstream error must become the client's error event: %q", frames)
	}
	if !strings.Contains(string(frames[0]), `"message":"boom"`) {
		t.Fatalf("the backend's reason is lost: %s", frames[0])
	}
	if len(s.FinishFrames()) != 0 {
		t.Fatal("a stream that ended in an error gets no terminal envelope")
	}
	if !slices.Contains(s.Notes(), "finish_reason=error") {
		t.Fatalf("notes = %v, want the failure reported", s.Notes())
	}
}

// A frame in no dialect the emitter speaks ends the stream with the client's error
// event — the same fail-open the Anthropic emitter has. Forwarding it is the silent
// truncation: a Responses SDK ignores an event it has no name for, so the client
// reads a complete response whose content is missing that piece.
func TestChatToResponsesStreamEndsOnAFrameNoDialectOwns(t *testing.T) {
	s := NewChatToResponses()
	frames := s.WriteFrame([]byte("data: " + `{"type":"content_block_delta","index":0,` +
		`"delta":{"type":"text_delta","text":"tail"}}` + "\n\n"))
	if len(frames) != 1 || !strings.HasPrefix(string(frames[0]), "event: error\n") {
		t.Fatalf("an Anthropic event on a Responses stream must fail open, got %q", frames)
	}
	if !slices.Contains(s.Notes(), "frame=untranslated") {
		t.Fatalf("notes = %v, want the untranslated frame reported", s.Notes())
	}
	if len(s.WriteFrame([]byte("data: "+`{"choices":[{"delta":{"content":"after"}}]}`+"\n\n"))) != 0 {
		t.Fatal("nothing may reach the client after the stream ended in an error")
	}
}

// A dual-protocol backend writing its own Responses events is already in the
// client's dialect: forwarded as it arrived, and never decoded as a chunk.
func TestChatToResponsesStreamForwardsResponsesEvents(t *testing.T) {
	s := NewChatToResponses()
	frames := s.WriteFrame([]byte("event: response.output_text.delta\ndata: " +
		`{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,` +
		`"delta":"native","sequence_number":3}` + "\n\n"))
	if len(frames) != 1 || string(frames[0]) != "event: response.output_text.delta\ndata: "+
		`{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,`+
		`"delta":"native","sequence_number":3}`+"\n\n" {
		t.Fatalf("a native Responses event must be forwarded untouched: %q", frames)
	}
	if notes := s.Notes(); len(notes) != 0 {
		t.Fatalf("a forwarded event is not a loss: %v", notes)
	}
}

func TestChatToResponsesStreamDropsAKeylessFrame(t *testing.T) {
	out := emit(NewChatToResponses(), `{"model":"m","choices":[{"index":0,"delta":{"content":"x"}}]}`, `[DONE]`)
	if strings.Contains(out, "response.failed") || strings.Contains(out, "event: error") {
		t.Fatalf("a keep-alive/usage-shaped frame must stay inert:\n%s", out)
	}
}

func TestChatToResponsesStreamMixedTerminators(t *testing.T) {
	s := NewChatToResponses()
	s.WriteFrame([]byte("data: " + `{"id":"c","model":"m","choices":[{"index":0,"delta":{"content":"a"}}]}` + "\n\n"))
	frames := s.WriteFrame([]byte("data: " + `{"id":"c","model":"m","choices":[{"index":0,"delta":{"content":"b"}}]}` + "\r\n\r\n"))
	joined := strings.Join(stringSlice(frames), "")
	if strings.Count(joined, "event: response.output_text.delta") != 1 {
		t.Fatalf("a CRLF-framed chunk must stay one event: %q", frames)
	}
}

func TestChatToResponsesStreamFlushesUnterminatedTail(t *testing.T) {
	s := NewChatToResponses()
	s.WriteFrame([]byte(`{"id":"c","model":"m","choices":[{"index":0,"delta":{"content":"a"}}]}`))
	out := s.FinishFrames()
	joined := strings.Join(stringSlice(out), "")
	if !strings.Contains(joined, `"delta":"a"`) || !strings.Contains(joined, "response.completed") {
		t.Fatalf("a tail handed over without a terminator must still be translated: %q", out)
	}
}

func TestNewStreamResponsesDirection(t *testing.T) {
	if s, ok := NewStream(Plan{From: Responses, Dialect: Chat}); !ok || s == nil {
		t.Fatal("a responses client on a chat backend needs an emitter")
	}
	if _, ok := NewStream(Plan{From: Chat, Dialect: Responses}); ok {
		t.Fatal("the chat-client-on-a-responses-backend direction is not implemented")
	}
	if _, ok := NewStream(Plan{From: Chat, Dialect: Chat}); ok {
		t.Fatal("an identity plan arms nothing")
	}
}

func stringSlice(frames [][]byte) []string {
	out := make([]string, 0, len(frames))
	for _, f := range frames {
		out = append(out, string(f))
	}
	return out
}
