package convert

import (
	"encoding/json"
	"slices"
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

// eventNames lists the emitted frames' event types, in the order a client reads
// them.
func eventNames(s string) []string {
	var out []string
	for _, ev := range strings.Split(strings.TrimRight(s, "\n"), "\n\n") {
		line, _, _ := strings.Cut(ev, "\n")
		out = append(out, strings.TrimPrefix(line, "event: "))
	}
	return out
}

// A chat stream's last frame is `data: [DONE]`, and plenty of backends put no
// finish_reason anywhere near it (the reason rides on a chunk that carries nothing
// else, or on none at all). The Anthropic envelope must still open, close every
// block and close the MESSAGE: message_stop is the stream end's job, so a client
// that waits for the terminal event never waits forever. Pinned event by event
// because the shape was otherwise only visible through message_delta.
func TestChatToMessagesStreamClosesAStreamEndedByDone(t *testing.T) {
	out := emit(NewChatToMessages(),
		`{"id":"c1","model":"claude","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}`,
		`{"id":"c1","model":"claude","choices":[{"index":0,"delta":{"content":"lo"}}]}`,
		`[DONE]`)

	want := []string{"message_start", "content_block_start", "content_block_delta",
		"content_block_delta", "content_block_stop", "message_delta", "message_stop"}
	if got := eventNames(out); !slices.Equal(got, want) {
		t.Fatalf("event sequence = %v, want %v:\n%s", got, want, out)
	}
	// With no finish_reason the emitter keeps its default reason, and the closing
	// frame is a restatement of the message it already streamed, not a new one.
	if !strings.Contains(out, `"stop_reason":"end_turn"`) {
		t.Fatalf("a stream with no finish_reason must still report a stop reason:\n%s", out)
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

// R22's branch 4: a frame whose `type` names an Anthropic stream event is already
// in the client's dialect (a dual-protocol backend, the proxy's own frame), so it
// goes out as it arrived — and it must not touch the emitter's recorded usage.
// `message_delta` is the ONE Anthropic event that carries usage, so an emitter
// that classifies by "has a usage key" reads the event as a chat chunk, emits
// zero frames (no `choices`) and overwrites the real counts with zeros.
func TestChatToMessagesStreamForwardsAnthropicEventsAndKeepsUsage(t *testing.T) {
	s := NewChatToMessages()
	s.WriteFrame([]byte(`data: {"id":"c1","choices":[{"index":0,"delta":{"content":"x"}}],` +
		`"usage":{"prompt_tokens":7,"completion_tokens":2}}` + "\n\n"))
	got := s.WriteFrame([]byte(`data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"},` +
		`"usage":{"input_tokens":100,"output_tokens":200}}` + "\n\n"))
	if len(got) != 1 || !strings.Contains(string(got[0]), `"stop_reason":"max_tokens"`) {
		t.Fatalf("an upstream-authored Anthropic event was swallowed: %q", got)
	}
	var out strings.Builder
	for _, f := range s.FinishFrames() {
		out.Write(f)
	}
	if !strings.Contains(out.String(), `"input_tokens":7`) ||
		!strings.Contains(out.String(), `"output_tokens":2`) {
		t.Fatalf("the forwarded event overwrote the recorded usage:\n%s", out.String())
	}
}

// R22's branch 7: a JSON object with no keys at all (a keep-alive `{}`) carries
// nothing an Anthropic client can read, so it is inert — forwarding it puts an
// unlabelled frame in the stream that the client's parser drops on the floor.
func TestChatToMessagesStreamDropsAKeylessFrame(t *testing.T) {
	s := NewChatToMessages()
	for _, frame := range []string{"data: {}\n\n", "data:   \n\n"} {
		if got := s.WriteFrame([]byte(frame)); len(got) != 0 {
			t.Fatalf("%q forwarded to an Anthropic client: %q", frame, got)
		}
	}
}

// R22's branch 8: a frame in NO dialect the emitter speaks (garbage after
// `data:`, or a foreign family like the Responses API's
// `response.output_text.delta`) is not ours to forward: an unlabelled `data:`
// frame is exactly what an Anthropic SDK ignores (`case "": continue`), so
// forwarding the frame that carries the answer's tail leaves the client a
// complete-looking message whose content is missing that piece and
// `stream.Err() == nil` — the silent truncation the spec forbids. It ends the
// stream with the client's own error event, and the note makes a backend that
// does this every stream visible in the GUI.
func TestChatToMessagesStreamEndsOnAFrameNoDialectOwns(t *testing.T) {
	for _, frame := range []string{
		`data: the worker died here, and this is not JSON` + "\n\n",
		`data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"jour"}` + "\n\n",
	} {
		s := NewChatToMessages()
		var out strings.Builder
		for _, f := range s.WriteFrame([]byte(`data: {"id":"c1","choices":[{"index":0,` +
			`"delta":{"content":"Bon"}}]}` + "\n\n")) {
			out.Write(f)
		}
		for _, f := range s.WriteFrame([]byte(frame)) {
			out.Write(f)
		}
		for _, f := range s.WriteFrame([]byte("data: [DONE]\n\n")) {
			out.Write(f)
		}
		for _, f := range s.FinishFrames() {
			out.Write(f)
		}
		got := out.String()
		i := strings.Index(got, "event: error")
		if i < 0 {
			t.Fatalf("%s: a frame in no dialect the emitter speaks was swallowed:\n%s", frame, got)
		}
		if !strings.Contains(got[i:], `"type":"api_error"`) {
			t.Fatalf("%s: the failure is not in the client's spelling:\n%s", frame, got[i:])
		}
		if strings.Contains(got[i:], "message_stop") {
			t.Fatalf("%s: an error event is terminal, nothing may follow it:\n%s", frame, got[i:])
		}
		if !slices.Contains(s.Notes(), "frame=untranslated") {
			t.Fatalf("%s: the untranslated frame must be reported: %v", frame, s.Notes())
		}
	}
}

// The string-shaped in-stream error is the other shape a backend gives up in
// (`{"error":"upstream timeout"}`); it has no chunk content, so it must reach the
// client as the client's own error event, not as a forwarded frame it ignores.
func TestChatToMessagesStreamTranslatesAStringErrorChunk(t *testing.T) {
	s := NewChatToMessages()
	got := s.WriteFrame([]byte(`data: {"error":"upstream timeout"}` + "\n\n"))
	if len(got) != 1 || !strings.Contains(string(got[0]), "event: error") {
		t.Fatalf("a string-shaped in-stream error was not translated: %q", got)
	}
	if !strings.Contains(string(got[0]), `"message":"upstream timeout"`) {
		t.Fatalf("the upstream error message was lost: %q", got)
	}
	for _, f := range s.WriteFrame([]byte("data: [DONE]\n\n")) {
		t.Fatalf("a stream that ended in an error kept writing: %q", f)
	}
	if got := s.FinishFrames(); len(got) != 0 {
		t.Fatalf("the normal close followed an error event: %q", got)
	}
}

// A multimodal chat backend streams `delta.content` as a CONTENT-PARTS ARRAY
// (`[{"type":"text","text":"…"}]`, llama.cpp and the OpenAI-fork builds do it).
// Deciding "is a chunk" by the inner field types made such a chunk fail the typed
// decode, so the whole chunk went out native: the client read an empty assistant
// message closed as a normal end_turn, metered correctly in our own metrics and
// invisible everywhere. Translate the text/refusal parts; note the rest.
func TestChatToMessagesStreamTranslatesAContentPartsDelta(t *testing.T) {
	s := NewChatToMessages()
	out := emit(s,
		`{"id":"c1","choices":[{"index":0,"delta":{"content":[{"type":"text","text":"Hi"}]}}]}`,
		`{"id":"c1","choices":[{"index":0,"delta":{"content":[{"type":"refusal","text":" no"},`+
			`{"type":"image_url","image_url":{"url":"data:image/png;base64,AAA"}}]},`+
			`"finish_reason":"stop"}]}`)

	if !strings.Contains(out, `"text":"Hi"`) || !strings.Contains(out, `"text":" no"`) {
		t.Fatalf("a content-parts delta was not translated:\n%s", out)
	}
	if strings.Contains(out, "image_url") || strings.Contains(out, `"object":"chat.completion.chunk"`) {
		t.Fatalf("native frames leaked into a translated stream:\n%s", out)
	}
	if !strings.Contains(out, `"stop_reason":"end_turn"`) || !strings.Contains(out, "event: message_stop") {
		t.Fatalf("the message was not closed:\n%s", out)
	}
	if !slices.Contains(s.Notes(), "content_part=dropped:image_url") {
		t.Fatalf("the part the translator cannot read must be reported: %v", s.Notes())
	}
}

// The `event:` branch must judge the frame by its NAME. A backend fronting a
// Responses-family model answers a converted request with named frames of a third
// dialect, and the Anthropic SDK dispatches an unknown event name into the message
// accumulator: a foreign payload carrying `usage`/`content` clobbers the answer the
// emitter built, which is the same silent-corruption class the fail-open exists to
// close. The proxy's own loop-detector frame is named `error`, so it keeps passing.
func TestChatToMessagesStreamFailsOpenOnAForeignEventName(t *testing.T) {
	s := NewChatToMessages()
	s.WriteFrame([]byte(`data: {"id":"c1","choices":[{"index":0,"delta":{"content":"Bon"}}]}` + "\n\n"))
	got := s.WriteFrame([]byte("event: response.output_text.delta\ndata: {\"delta\":\"jour\"}\n\n"))
	if len(got) != 1 || !strings.Contains(string(got[0]), "event: error") ||
		!strings.Contains(string(got[0]), `"type":"api_error"`) {
		t.Fatalf("a named frame of a third dialect was forwarded into the client's accumulator: %q", got)
	}
	if !slices.Contains(s.Notes(), "frame=untranslated") {
		t.Fatalf("the untranslated frame must be reported: %v", s.Notes())
	}
	for _, f := range s.WriteFrame([]byte("event: message_stop\ndata: {}\n\n")) {
		if strings.Contains(string(f), "message_stop") {
			t.Fatalf("a stream that failed open is terminal: %q", f)
		}
	}
}

// A JSON object with none of a Chat Completions chunk's keys is not ours to
// translate — an upstream-authored Anthropic frame — so it goes through untouched
// instead of being parsed into nothing and emitted as nothing.
func TestChatToMessagesStreamForwardsAFrameThatIsNotAChunk(t *testing.T) {
	s := NewChatToMessages()
	got := s.WriteFrame([]byte("data: {\"type\":\"content_block_delta\",\"index\":0}\n\n"))
	if len(got) != 1 || !strings.Contains(string(got[0]), `"type":"content_block_delta"`) {
		t.Fatalf("a foreign JSON frame was swallowed: %q", got)
	}
	// A forwarded frame carries exactly ONE terminator: the relay hands the
	// terminator over inside the frame, so a re-terminating emitter that appends
	// blindly writes the client a doubled blank line.
	if got := string(got[0]); !strings.HasSuffix(got, "\n\n") || strings.HasSuffix(got, "\n\n\n\n") {
		t.Fatalf("forwarded frame does not end in exactly one blank line: %q", got)
	}
}

// A backend can mix terminators inside one stream (a proxy in front of the model
// normalizing CRLF, a chunked writer flushing mid-event), and the relay hands the
// translator the bytes as it framed them. Taking the FIRST terminator SEARCHED
// rather than the leftmost one glues the LF-framed event onto the CRLF-framed
// one: the glued blob parses as nothing, both events fall to the fail-open, and
// the first event's content reaches the client native — silently lost.
func TestChatToMessagesStreamMixedTerminators(t *testing.T) {
	s := NewChatToMessages()
	var out strings.Builder
	for _, f := range s.WriteFrame([]byte(
		`data: {"id":"c1","choices":[{"index":0,"delta":{"content":"Hi"}}]}` + "\n\n" +
			`data: {"id":"c1","choices":[{"index":0,"delta":{"content":" there"},` +
			`"finish_reason":"stop"}]}` + "\r\n\r\n")) {
		out.Write(f)
	}
	for _, f := range s.FinishFrames() {
		out.Write(f)
	}
	got := out.String()
	if !strings.Contains(got, `"text":"Hi"`) {
		t.Fatalf("the LF-framed event was glued to the next one and lost:\n%s", got)
	}
	if !strings.Contains(got, `"text":" there"`) {
		t.Fatalf("the CRLF-framed event was lost:\n%s", got)
	}
	if n := strings.Count(got, "event: content_block_delta"); n != 2 {
		t.Fatalf("content_block_delta x%d, want one per upstream event:\n%s", n, got)
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
