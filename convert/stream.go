package convert

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"strconv"
)

// Stream converts a streamed answer from the backend's dialect to the client's.
// It is stateful per request on purpose: Anthropic opens and closes numbered
// content blocks, and a Chat Completions chunk carries no block index, so the
// emitter allocates them as content kinds appear.
type Stream interface {
	// WriteFrame takes one or more complete upstream SSE frames and returns the
	// complete frames to write to the client. The returned slice aliases the
	// emitter's own buffer: write the frames out before calling again.
	WriteFrame(frame []byte) [][]byte
	// FinishFrames closes the stream: everything the emitter still owes at the end
	// of the response (an open block, the closing events, a tail held across
	// reads).
	FinishFrames() [][]byte
	// Notes lists what the translation cost, in the same channel as the request
	// and JSON directions' notes.
	Notes() []string
}

// NewStream returns the translator for a plan's response direction (the backend
// answered in p.Dialect, the client asked in p.From), and false when none exists:
// identity, a dialectless request, or a pair this build does not implement. The
// caller arms the relay only on true, so an unconverted request pays nothing.
func NewStream(p Plan) (Stream, bool) {
	if p.Dialect == Chat && p.From == Messages {
		return NewChatToMessages(), true
	}
	return nil, false
}

func NewChatToMessages() Stream {
	return &chatToMessages{open: -1, inputTok: -1, stop: "end_turn"}
}

type chatToMessages struct {
	id, model  string
	open       int    // index of the open text/thinking block; -1 = none
	kind       string // that block's kind: text | thinking ("" = none open)
	tools      []toolBlock
	next       int    // the next block index to allocate
	stop       string // the stop reason reported so far (the stream end restates it)
	buf        []byte // held tail of a frame split across reads
	out        [][]byte
	inputTok   int // -1 until a chunk reports usage
	outputToks int
	started    bool
	failed     bool // an upstream error frame ended the message
}

// toolBlock is one open tool_use block, keyed by the Chat Completions
// `tool_calls[].index` it streams: a backend answering with parallel calls
// interleaves their argument deltas, so a block cannot be keyed on the "last one
// opened" — a fragment would land in the other call's arguments.
type toolBlock struct {
	idx int // the chat delta's tool_calls index
	blk int // the Anthropic content block index
	id  string
}

func (c *chatToMessages) Notes() []string { return nil }

// WriteFrame takes one or more complete upstream SSE frames and returns the
// complete frames to write to the client.
func (c *chatToMessages) WriteFrame(frame []byte) [][]byte {
	c.buf = append(c.buf, frame...)
	out := c.out[:0]
	for {
		i, n := frameEnd(c.buf)
		if i < 0 {
			break
		}
		out = append(out, c.handle(c.buf[:i])...)
		c.buf = c.buf[i+n:]
	}
	c.out = out
	return out
}

// frameEnd returns the index and length of the first SSE frame terminator in b.
// The relay the emitter is fed by frames on all three of these (and hands the
// terminator over inside the frame), so a scanner that knows only "\n\n" would
// buffer a CRLF-framed backend's whole answer and dump it native at stream end.
func frameEnd(b []byte) (int, int) {
	if i := bytes.Index(b, []byte("\r\n\r\n")); i >= 0 {
		return i, 4
	}
	if i := bytes.Index(b, []byte("\n\n")); i >= 0 {
		return i, 2
	}
	if i := bytes.Index(b, []byte("\r\r")); i >= 0 {
		return i, 2
	}
	return -1, 0
}

func (c *chatToMessages) handle(frame []byte) [][]byte {
	data := bytes.TrimSpace(frame)
	if len(data) == 0 {
		return nil
	}
	// SSE comments are heartbeats; Anthropic's are injected below the translator.
	if bytes.HasPrefix(data, []byte(":")) {
		return nil
	}
	data = bytes.TrimSpace(bytes.TrimPrefix(data, []byte("data:")))
	if bytes.Equal(data, []byte("[DONE]")) {
		return nil
	}
	// A frame the emitter cannot read is not ours to translate: an upstream author's
	// Anthropic frame, a keep-alive object. Swallowing one would leave the client a
	// truncated message and no error, so it goes out as it arrived (re-terminated,
	// because the frame arrives without its blank line).
	var keys map[string]json.RawMessage
	if json.Unmarshal(data, &keys) != nil {
		return reframe(frame)
	}
	if c.failed {
		return nil // an error event is terminal; the rest of the stream is silence
	}
	if _, ok := keys["error"]; ok {
		// A backend's in-stream `error` object is valid JSON with no chunk content,
		// so it is the frame a parse-only test would forward raw and a naive
		// classifier would drop. It is the client's dialect's error event, and it
		// ends the message: no stream-end envelope may follow it.
		if b, _, err := chatErrorToMessages(data); err == nil {
			c.failed = true
			return [][]byte{append(append([]byte("event: error\ndata: "), b...), '\n', '\n')}
		}
	}
	if !isChunkShape(keys) {
		return reframe(frame)
	}
	var chunk struct {
		ID      string          `json:"id"`
		Model   string          `json:"model"`
		Choices []cStreamChoice `json:"choices"`
		Usage   *cUsage         `json:"usage"`
	}
	if json.Unmarshal(data, &chunk) != nil {
		return reframe(frame)
	}
	var frames [][]byte
	if chunk.ID != "" {
		c.id = chunk.ID
	}
	if chunk.Model != "" {
		c.model = chunk.Model
	}
	if u := chunk.Usage; u != nil {
		c.inputTok = u.PromptTokens - u.PromptDetails.CachedTokens
		c.outputToks = u.CompletionTokens
	}
	for _, ch := range chunk.Choices {
		if u := ch.Usage; u != nil {
			c.inputTok = u.PromptTokens - u.PromptDetails.CachedTokens
			c.outputToks = u.CompletionTokens
		}
		if !c.started {
			frames = append(frames, c.start())
		}
		frames = append(frames, c.text(ch.Delta.Reasoning, "thinking")...)
		frames = append(frames, c.text(ch.Delta.Content, "text")...)
		for _, tc := range ch.Delta.ToolCalls {
			frames = append(frames, c.tool(tc.Index, tc)...)
		}
		if ch.FinishReason != "" {
			frames = append(frames, c.finish(ch.FinishReason)...)
		}
	}
	// A chunk that carried nothing (a role-only delta, a bare usage chunk) opened no
	// block: emit nothing rather than an empty Anthropic event.
	return frames
}

// reframe re-terminates a frame the emitter forwards untouched (it arrives without
// its blank line, and the client's parser needs one).
func reframe(frame []byte) [][]byte {
	return [][]byte{append(append([]byte{}, frame...), '\n', '\n')}
}

// isChunkShape reports whether a JSON object carries anything a Chat Completions
// chunk carries. An object with none of these keys is not a chunk — an
// Anthropic-shaped frame authored upstream or by the proxy, a keep-alive object —
// and forwarding it whole is the fail-open answer.
func isChunkShape(o map[string]json.RawMessage) bool {
	for _, k := range [...]string{"choices", "id", "model", "usage"} {
		if _, ok := o[k]; ok {
			return true
		}
	}
	return false
}

type cStreamChoice struct {
	Delta struct {
		Role      string          `json:"role"`
		Content   string          `json:"content"`
		Reasoning string          `json:"reasoning_content"`
		ToolCalls []deltaToolCall `json:"tool_calls"`
	} `json:"delta"`
	FinishReason string  `json:"finish_reason"`
	Usage        *cUsage `json:"usage"`
}

type deltaToolCall struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

func (c *chatToMessages) start() []byte {
	c.started = true
	id := c.id
	if id == "" {
		// A backend that sent no id leaves us to invent one — per stream, because
		// a constant would give every converted answer the same message id.
		var b [10]byte
		if _, err := rand.Read(b[:]); err != nil {
			id = "msg_" + shortHash("stream")
		} else {
			id = "msg_" + hex.EncodeToString(b[:])
		}
	}
	input := c.inputTok
	if input < 0 {
		input = 0
	}
	return frame("message_start", map[string]any{"type": "message_start", "message": map[string]any{
		"id": id, "type": "message", "role": "assistant", "model": c.model,
		"content": []any{},
		"usage":   map[string]any{"input_tokens": input, "cache_read_input_tokens": 0},
	}})
}

func frame(event string, payload any) []byte {
	b, _ := json.Marshal(payload) // values built here always marshal
	return append(append([]byte("event: "+event+"\ndata: "), b...), '\n', '\n')
}

// text appends a delta to the open block of that kind, opening (and closing every
// block before it) when the content kind changed.
func (c *chatToMessages) text(s, kind string) [][]byte {
	if s == "" {
		return nil
	}
	var out [][]byte
	if c.kind != kind {
		// The prose block changes; a tool call that is still streaming keeps its
		// block (closing it here would split its arguments across two blocks).
		out = append(out, c.closeText()...)
		c.kind, c.open = kind, c.next
		c.next++
		out = append(out, frame("content_block_start", map[string]any{"type": "content_block_start",
			"index": c.open, "content_block": blockStart(kind)}))
	}
	field, deltaType := "text", "text_delta"
	if kind == "thinking" {
		field, deltaType = "thinking", "thinking_delta"
	}
	return append(out, frame("content_block_delta", map[string]any{"type": "content_block_delta",
		"index": c.open, "delta": map[string]any{"type": deltaType, field: s}}))
}

// tool streams one Chat Completions tool-call delta into its own tool_use block.
// A call is new when its `tool_calls` index was never seen, or when the index
// reappears with a fresh id (a second call reusing the index); everything else is
// a continuation of the block that index already owns.
func (c *chatToMessages) tool(idx int, tc deltaToolCall) [][]byte {
	var out [][]byte
	blk, id := -1, ""
	for _, t := range c.tools {
		if t.idx == idx {
			blk, id = t.blk, t.id
		}
	}
	if blk >= 0 && (tc.ID == "" || tc.ID == id) {
		return c.toolArgs(blk, tc.Function.Arguments)
	}
	if blk >= 0 {
		out = append(out, blockStop(blk))
		c.dropTool(idx)
	} else {
		// The text/thinking block, if open, is over; the other calls' blocks stay
		// open — their arguments are still arriving, interleaved with this one's.
		out = append(out, c.closeText()...)
	}
	blk = c.next
	c.next++
	if tc.ID == "" {
		tc.ID = "toolu_" + strconv.Itoa(blk)
	}
	c.tools = append(c.tools, toolBlock{idx: idx, blk: blk, id: tc.ID})
	out = append(out, frame("content_block_start", map[string]any{"type": "content_block_start",
		"index": blk, "content_block": map[string]any{"type": "tool_use",
			"id": tc.ID, "name": tc.Function.Name, "input": map[string]any{}}}))
	return append(out, c.toolArgs(blk, tc.Function.Arguments)...)
}

func (c *chatToMessages) toolArgs(blk int, args string) [][]byte {
	if args == "" {
		return nil
	}
	return [][]byte{frame("content_block_delta", map[string]any{"type": "content_block_delta",
		"index": blk, "delta": map[string]any{"type": "input_json_delta",
			"partial_json": args}})}
}

func (c *chatToMessages) dropTool(idx int) {
	kept := c.tools[:0]
	for _, t := range c.tools {
		if t.idx != idx {
			kept = append(kept, t)
		}
	}
	c.tools = kept
}

// closeText closes the open text/thinking block, if any: a tool call starts while
// the answer's prose is still open, and the calls that are already streaming keep
// their blocks.
func (c *chatToMessages) closeText() [][]byte {
	if c.kind == "" {
		return nil
	}
	c.kind = ""
	return [][]byte{blockStop(c.open)}
}

// close closes every open block, lowest index first: a content kind change ends
// the blocks that came before it, and so does the end of the stream.
func (c *chatToMessages) close() [][]byte {
	out := c.closeText()
	for _, t := range c.tools {
		out = append(out, blockStop(t.blk))
	}
	c.tools = nil
	return out
}

func blockStop(index int) []byte {
	return frame("content_block_stop", map[string]any{"type": "content_block_stop", "index": index})
}

func blockStart(kind string) map[string]any {
	if kind == "thinking" {
		return map[string]any{"type": "thinking", "thinking": ""}
	}
	return map[string]any{"type": "text", "text": ""}
}

// finish closes the open block and reports the stop reason and the usage. The
// message itself is not closed here: a chat stream continues to [DONE] after the
// finish_reason chunk, and message_stop is the stream end's job (FinishFrames) —
// so a message_delta arriving twice (finish_reason, then stream end) is the
// normal shape, and an Anthropic client reads the same stop reason both times.
func (c *chatToMessages) finish(fr string) [][]byte {
	c.stop = stopReason(fr)
	out := c.close()
	return append(out, frame("message_delta", map[string]any{"type": "message_delta",
		"delta": map[string]any{"stop_reason": c.stop, "stop_sequence": nil},
		"usage": c.deltaUsage()}))
}

// deltaUsage carries the output count, and the input count when a usage chunk
// supplied one. Anthropic's own streams report input tokens in message_start,
// which a chat backend only knows at the end — so message_delta restates it and a
// client that shows cost never reads a permanent zero.
func (c *chatToMessages) deltaUsage() map[string]any {
	u := map[string]any{"output_tokens": c.outputToks}
	if c.inputTok >= 0 {
		u["input_tokens"] = c.inputTok
	}
	return u
}

// FinishFrames closes a stream that ended without a finish_reason (a backend that
// died mid-generation, a client-driven drain) so an Anthropic SDK never sees an
// unterminated message, and delivers a tail frame the relay handed over without
// its blank-line terminator.
func (c *chatToMessages) FinishFrames() [][]byte {
	out := c.out[:0]
	if len(bytes.TrimSpace(c.buf)) > 0 {
		out = append(out, c.handle(c.buf)...)
		c.buf = nil
	}
	if !c.started || c.failed {
		// A stream that never opened a message gets no Anthropic envelope, and one
		// that ended in an upstream error frame gets no normal close — but a tail
		// frame the relay handed over is not ours to keep in either case.
		c.out = out
		return out
	}
	out = append(out, c.close()...)
	out = append(out, frame("message_delta", map[string]any{"type": "message_delta",
		"delta": map[string]any{"stop_reason": c.stop, "stop_sequence": nil},
		"usage": c.deltaUsage()}))
	out = append(out, frame("message_stop", map[string]any{"type": "message_stop"}))
	c.out = out
	return out
}
