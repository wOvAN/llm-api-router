package convert

import (
	"encoding/json"
	"strings"
	"testing"
)

func emit(s Stream, chunks ...string) string {
	var b strings.Builder
	for _, c := range chunks {
		for _, f := range s.WriteFrame([]byte("data: " + c + "\n\n")) {
			b.Write(f)
		}
	}
	for _, f := range s.FinishFrames() {
		b.Write(f)
	}
	return b.String()
}

func TestChatToMessagesStreamText(t *testing.T) {
	out := emit(NewChatToMessages(),
		`{"id":"c1","model":"claude","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}`,
		`{"id":"c1","model":"claude","choices":[{"index":0,"delta":{"content":"lo"},`+
			`"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":2}}`,
		`[DONE]`)

	for _, want := range []string{
		"event: message_start", `"type":"message_start"`,
		"event: content_block_start", `"index":0`, `"type":"text"`,
		`"text":"Hel"`, `"text":"lo"`,
		"event: content_block_stop", "event: message_delta", `"stop_reason":"end_turn"`,
		"event: message_stop",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("stream lacks %s:\n%s", want, out)
		}
	}
	if strings.Contains(out, "[DONE]") {
		t.Fatalf("[DONE] must not reach an Anthropic client:\n%s", out)
	}
	// Every emitted frame is one complete SSE event: the client's parser splits on
	// blank lines, and a half frame glues itself onto the next (opencode #7692).
	for _, f := range strings.Split(strings.TrimRight(out, "\n"), "\n\n") {
		if !strings.HasPrefix(f, "event: ") || !strings.Contains(f, "\ndata: ") {
			t.Fatalf("incomplete frame emitted: %q", f)
		}
	}
	if !strings.Contains(out, `"output_tokens":2`) {
		t.Fatalf("usage must ride message_delta:\n%s", out)
	}
	// A chat backend learns prompt tokens only at the end, and Anthropic reports
	// them in message_start: message_delta must restate them, or a client that
	// shows cost reads a permanent zero.
	if !strings.Contains(out, `"input_tokens":7`) {
		t.Fatalf("input tokens lost:\n%s", out)
	}
}

func TestChatToMessagesStreamToolCallsAndThinking(t *testing.T) {
	out := emit(NewChatToMessages(),
		`{"choices":[{"index":0,"delta":{"reasoning_content":"hm"}}]}`,
		`{"choices":[{"index":0,"delta":{"content":"ok"}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1",`+
			`"function":{"name":"fetch","arguments":"{\"u\":"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,`+
			`"function":{"arguments":"1}"}}]},"finish_reason":"tool_calls"}]}`)

	for _, want := range []string{`"type":"thinking"`, `"thinking_delta"`, `"type":"tool_use"`,
		`"partial_json":"{\"u\":"`, `"partial_json":"1}"`, `"stop_reason":"tool_use"`} {
		if !strings.Contains(out, want) {
			t.Fatalf("lacks %s:\n%s", want, out)
		}
	}
	// A kind change closes the previous block: indices are distinct and ordered,
	// and every block opened is closed — thinking, text and tool_use each open
	// once, and the finish_reason closes the last one. The counted spelling is the
	// payload's `"type"` field, one per event, not the `event:` line as well.
	if got := strings.Count(out, `"content_block_start"`); got != 3 {
		t.Fatalf("content_block_start x%d, want 3 (thinking, text, tool_use):\n%s", got, out)
	}
	if got := strings.Count(out, `"content_block_stop"`); got != 3 {
		t.Fatalf("content_block_stop x%d, want 3:\n%s", got, out)
	}
	// A stream accumulator keeps the LAST message_delta, so the frame that closes
	// the message must restate the real reason: closing with the default would
	// turn a tool call into an end_turn answer.
	if i := strings.LastIndex(out, "event: message_delta"); i < 0 ||
		!strings.Contains(out[i:], `"stop_reason":"tool_use"`) {
		t.Fatalf("the closing message_delta downgraded the stop reason:\n%s", out)
	}
}

func TestChatToMessagesStreamPassesThroughWhatItCannotTranslate(t *testing.T) {
	s := NewChatToMessages()
	// A frame that is not a Chat Completions chunk is not ours to translate: an
	// upstream-authored error frame, or the Anthropic mid-stream error the proxy
	// injects in the client's dialect. Swallowing it would leave the client a
	// truncated message and no error. Comments (heartbeats) and [DONE] carry
	// nothing an Anthropic client needs, so those are dropped.
	frames := s.WriteFrame([]byte("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\"}}\n\n"))
	if len(frames) != 1 || !strings.Contains(string(frames[0]), "event: error") {
		t.Fatalf("error frame swallowed: %q", frames)
	}
	if got := s.WriteFrame([]byte(": keep-alive\n\n")); len(got) != 0 {
		t.Fatalf("heartbeat forwarded to the translator output: %q", got)
	}
	if got := s.WriteFrame([]byte("data: [DONE]\n\n")); len(got) != 0 {
		t.Fatalf("[DONE] forwarded: %q", got)
	}
	out := emit(s, `{"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":"stop"}]}`)
	if strings.Count(out, `"content_block_delta"`) != 1 {
		t.Fatalf("contentless frames produced deltas:\n%s", out)
	}
	// A stream that never carried message_start must not be closed with an
	// Anthropic envelope out of thin air.
	if got := NewChatToMessages().FinishFrames(); got != nil {
		t.Fatalf("unstarted stream produced frames: %q", got)
	}
}

// The relay hands over a trailing frame the backend closed without its blank
// line (sseRelay.finish), so the emitter must not swallow it at stream end.
func TestChatToMessagesStreamFlushesUnterminatedTail(t *testing.T) {
	s := NewChatToMessages()
	s.WriteFrame([]byte("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}}]}\n\n"))
	if got := len(s.WriteFrame([]byte("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"b\"}}]}"))); got != 0 {
		t.Fatalf("a partial frame was forwarded: %d frames", got)
	}
	var out strings.Builder
	for _, f := range s.FinishFrames() {
		out.Write(f)
	}
	if !strings.Contains(out.String(), `"text":"b"`) {
		t.Fatalf("the unterminated tail was dropped:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "event: message_stop") {
		t.Fatalf("the stream was not closed:\n%s", out.String())
	}
}

// A backend answering with parallel tool calls interleaves the argument deltas of
// both calls (a1, b1, a2, b2 is a legal delta order, and a call carries its id
// only on its first delta). Anthropic numbers its blocks, so every chat
// `tool_calls[].index` needs a block of its own, and a fragment must land in the
// block its call belongs to — appending b's tail to a's arguments corrupts both.
func TestChatToMessagesStreamParallelToolCalls(t *testing.T) {
	out := emit(NewChatToMessages(),
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a",`+
			`"function":{"name":"a","arguments":"{\"x\":"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b",`+
			`"function":{"name":"b","arguments":"{\"y\":"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,`+
			`"function":{"arguments":"1}"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,`+
			`"function":{"arguments":"2}"}}]},"finish_reason":"tool_calls"}]}`)

	// id → the arguments delivered to its block; stopped → blocks that were closed.
	idOf := map[int]string{}
	args := map[string]string{}
	stopped := map[string]bool{}
	for _, ev := range strings.Split(strings.TrimRight(out, "\n"), "\n\n") {
		var e struct {
			Type         string `json:"type"`
			Index        int    `json:"index"`
			ContentBlock *struct {
				Type string `json:"type"`
				ID   string `json:"id"`
			} `json:"content_block"`
			Delta *struct {
				Type        string `json:"type"`
				PartialJSON string `json:"partial_json"`
			} `json:"delta"`
		}
		data := ev[strings.Index(ev, "data: ")+6:]
		if json.Unmarshal([]byte(data), &e) != nil {
			t.Fatalf("emitted frame is not JSON: %q", ev)
		}
		switch e.Type {
		case "content_block_start":
			if e.ContentBlock != nil && e.ContentBlock.Type == "tool_use" {
				idOf[e.Index] = e.ContentBlock.ID
			}
		case "content_block_delta":
			args[idOf[e.Index]] += e.Delta.PartialJSON
		case "content_block_stop":
			stopped[idOf[e.Index]] = true
		}
	}
	if len(idOf) != 2 {
		t.Fatalf("parallel calls must open one block each, got %d:\n%s", len(idOf), out)
	}
	if args["call_a"] != `{"x":1}` {
		t.Fatalf("call_a arguments = %q:\n%s", args["call_a"], out)
	}
	if args["call_b"] != `{"y":2}` {
		t.Fatalf("call_b arguments = %q:\n%s", args["call_b"], out)
	}
	if !stopped["call_a"] || !stopped["call_b"] {
		t.Fatalf("both blocks must be closed: %v\n%s", stopped, out)
	}
}

// A backend can answer 200 and then put its OWN error object in the stream
// (vLLM/llama.cpp do it, and a proxy in front of the model does it): the payload is
// valid JSON, so an emitter that only forwards what fails to PARSE swallows it and
// the client gets a truncated answer closed as a normal end_turn, with no error at
// all. It is foreign content in the client's stream, so it goes out as Anthropic's
// error event — and it is terminal, so the normal close must not follow it.
func TestChatToMessagesStreamForwardsAnUpstreamErrorChunk(t *testing.T) {
	run := func(prefix string) string {
		s := NewChatToMessages()
		var out strings.Builder
		if prefix != "" {
			for _, f := range s.WriteFrame([]byte("data: " + prefix + "\n\n")) {
				out.Write(f)
			}
		}
		for _, f := range s.WriteFrame([]byte("data: {\"error\":{\"type\":\"server_error\"," +
			"\"message\":\"worker died\"}}\n\n")) {
			out.Write(f)
		}
		for _, f := range s.FinishFrames() {
			out.Write(f)
		}
		return out.String()
	}

	for _, tc := range []struct{ name, prefix string }{
		{"after a delta", `{"choices":[{"index":0,"delta":{"content":"par"}}]}`},
		{"as the first frame", ``},
	} {
		got := run(tc.prefix)
		i := strings.Index(got, "event: error")
		if i < 0 {
			t.Fatalf("%s: the upstream error frame was swallowed:\n%s", tc.name, got)
		}
		if !strings.Contains(got[i:], `"type":"error"`) ||
			!strings.Contains(got[i:], `"type":"server_error"`) ||
			!strings.Contains(got[i:], `"message":"worker died"`) {
			t.Fatalf("%s: the upstream error is not in the client's spelling:\n%s", tc.name, got)
		}
		if strings.Contains(got[i:], "message_stop") || strings.Contains(got[i:], `"text":"par"`) {
			t.Fatalf("%s: an error frame is terminal, nothing may follow it:\n%s", tc.name, got[i:])
		}
	}
}

// A JSON object with none of a Chat Completions chunk's keys is not ours to
// translate — an upstream-authored Anthropic frame, a keep-alive object — so it
// goes through untouched instead of being parsed into nothing and emitted as
// nothing.
func TestChatToMessagesStreamForwardsAFrameThatIsNotAChunk(t *testing.T) {
	s := NewChatToMessages()
	got := s.WriteFrame([]byte("data: {\"type\":\"content_block_delta\",\"index\":0}\n\n"))
	if len(got) != 1 || !strings.Contains(string(got[0]), `"type":"content_block_delta"`) {
		t.Fatalf("a foreign JSON frame was swallowed: %q", got)
	}
}

// The relay frames on "\n\n", "\r\n\r\n" and "\r\r" and hands the terminator to the
// translator inside the frame, and it flushes per frame so the client sees events as
// they are produced. A scanner that knows only "\n\n" buffers a CRLF-framed
// backend's whole answer and emits it once at stream end — right answer, wrong
// timing: every token of the generation arrives in one lump after it finished.
func TestChatToMessagesStreamAcceptsCRLFFraming(t *testing.T) {
	for _, sep := range []string{"\r\n\r\n", "\r\r"} {
		s := NewChatToMessages()
		first := s.WriteFrame([]byte("data: {\"id\":\"c1\",\"choices\":" +
			"[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}" + sep))
		if len(first) == 0 {
			t.Fatalf("frames framed with %q were buffered instead of emitted", sep)
		}
		var out strings.Builder
		for _, f := range first {
			out.Write(f)
		}
		for _, f := range s.WriteFrame([]byte("data: {\"id\":\"c1\",\"choices\":" +
			"[{\"index\":0,\"delta\":{\"content\":\" there\"},\"finish_reason\":\"stop\"}]}" + sep)) {
			out.Write(f)
		}
		for _, f := range s.FinishFrames() {
			out.Write(f)
		}
		got := out.String()
		if !strings.Contains(got, "event: message_start") ||
			!strings.Contains(got, `"text":"Hi"`) || !strings.Contains(got, `"text":" there"`) {
			t.Fatalf("frames framed with %q were not translated:\n%s", sep, got)
		}
		if strings.Contains(got, "data: {\"id\"") {
			t.Fatalf("native frames forwarded to the client:\n%s", got)
		}
	}
}

// llama.cpp and friends stream `content` in the same delta as `tool_calls`, so a
// text delta can land between two argument fragments of a call that is still
// streaming. It closes the prose block, not the call's block: closing every open
// block would end the tool_use block mid-arguments and re-open it under a
// synthetic id, splitting the JSON the client has to parse.
func TestChatToMessagesStreamContentBetweenToolArgumentsKeepsTheBlock(t *testing.T) {
	out := emit(NewChatToMessages(),
		`{"choices":[{"index":0,"delta":{"content":"Sure"}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1",`+
			`"function":{"name":"fetch","arguments":"{\"u\":"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"content":" now"}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,`+
			`"function":{"arguments":"1}"}}]},"finish_reason":"tool_calls"}]}`)

	if got := strings.Count(out, `"type":"tool_use"`); got != 1 {
		t.Fatalf("the call was split across %d tool_use blocks:\n%s", got, out)
	}
	if !strings.Contains(out, `"id":"call_1","input":{},"name":"fetch"`) {
		t.Fatalf("the call's id/name were lost:\n%s", out)
	}
	// Prose and arguments each stayed in their own block, in order.
	if !strings.Contains(out, `"text":"Sure"`) || !strings.Contains(out, `"text":" now"`) {
		t.Fatalf("the prose was lost:\n%s", out)
	}
	if strings.Count(out, `"partial_json"`) != 2 {
		t.Fatalf("argument fragments did not both reach the call's block:\n%s", out)
	}
}

func TestNewStreamIdentityArmsNothing(t *testing.T) {
	if s, ok := NewStream(Plan{From: Chat, Dialect: Chat}); ok || s != nil {
		t.Fatal("an identity plan must arm no translator (hot path pays nothing)")
	}
	if _, ok := NewStream(Plan{From: Responses, Dialect: Chat}); ok {
		t.Fatal("unimplemented pairs must report no translator")
	}
	s, ok := NewStream(Plan{From: Messages, Dialect: Chat, Path: "/v1/chat/completions"})
	if !ok || s == nil {
		t.Fatal("chat answered, messages asked: must translate")
	}
}
