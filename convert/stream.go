package convert

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"slices"
	"strconv"
	"strings"
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
	notes      []string
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

// Notes lists what the translation cost: a frame in no dialect the emitter
// speaks, a content part it cannot read as text, a finish reason whose Anthropic
// counterpart says something milder.
func (c *chatToMessages) Notes() []string { return c.notes }

// note records a loss once per distinct note — a backend that fails on every
// frame of a stream is one note in the GUI, not a thousand.
func (c *chatToMessages) note(s string) {
	if !slices.Contains(c.notes, s) {
		c.notes = append(c.notes, s)
	}
}

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

// frameEnd returns the index and length of the LEFTMOST SSE frame terminator in
// b. The relay the emitter is fed by frames on all three of these (and hands the
// terminator over inside the frame), so a scanner that knows only "\n\n" would
// buffer a CRLF-framed backend's whole answer and dump it native at stream end.
// The leftmost, not the first one searched: a stream that mixes terminators
// (event A framed "\n\n", event B framed "\r\n\r\n") must stay two frames, and
// searching for the CRLF terminator across the whole buffer first glues B's
// terminator onto A, which parses as nothing and falls to the fail-open.
// proxy.findSSEFrameEnd is the same rule on the relay side.
func frameEnd(b []byte) (int, int) {
	best, size := -1, 0
	for _, term := range [...][]byte{[]byte("\r\n\r\n"), []byte("\n\n"), []byte("\r\r")} {
		if i := bytes.Index(b, term); i >= 0 && (best < 0 || i < best) {
			best, size = i, len(term)
		}
	}
	return best, size
}

func (c *chatToMessages) handle(frame []byte) [][]byte {
	// Checked before any parsing: a stream that ended in an error event ends, and
	// nothing after it — not even a frame this emitter cannot read — reaches the
	// client.
	if c.failed {
		return nil
	}
	data := bytes.TrimSpace(frame)
	if len(data) == 0 {
		return nil
	}
	// A frame that carries an `event:` line is already in the client's dialect: the
	// proxy's own loop-detector frame, an Anthropic backend authoring its own
	// events. It must be the FIRST branch, because the prefix makes the frame as a
	// whole fail JSON parsing — a classifier that parses first judges it by its
	// payload and drops it.
	if bytes.HasPrefix(data, []byte("event:")) {
		return reframe(frame)
	}
	// SSE comments are heartbeats; Anthropic's are injected below the translator.
	if bytes.HasPrefix(data, []byte(":")) {
		return nil
	}
	data = bytes.TrimSpace(bytes.TrimPrefix(data, []byte("data:")))
	// An empty payload (a bare `data:` line) carries nothing, like a comment.
	if len(data) == 0 {
		return nil
	}
	if bytes.Equal(data, []byte("[DONE]")) {
		return nil
	}
	var keys map[string]json.RawMessage
	if json.Unmarshal(data, &keys) != nil {
		return c.failOpen()
	}
	// An Anthropic stream event authored upstream: forwarded as it arrived, and
	// NOT decoded as a chunk — `message_delta` is the one event that carries
	// usage, so reading it as a chunk (it has a `usage` key) emits zero frames and
	// overwrites the counts the chat chunks reported with its own.
	if t, ok := keys["type"]; ok && isAnthropicEvent(t) {
		return reframe(frame)
	}
	// A backend's in-stream `error` (the object shape, and the string shape
	// `{"error":"upstream timeout"}`) is valid JSON with no chunk content, so it is
	// the frame a parse-only test would forward raw and a naive classifier would
	// drop. It is the client's dialect's error event, and it ends the message: no
	// stream-end envelope may follow it.
	if _, ok := keys["error"]; ok {
		if b, isErr, err := chatErrorToMessages(data); isErr && err == nil {
			c.failed = true
			return [][]byte{append(append([]byte("event: error\ndata: "), b...), '\n', '\n')}
		}
		return c.failOpen()
	}
	// A keep-alive object carries nothing an Anthropic client can read: inert.
	if len(keys) == 0 {
		return nil
	}
	// A Chat Completions chunk, classified by KEY NAMES only — the content fields
	// are decoded leniently below (cStreamDelta) precisely so that "is a chunk" is
	// never decided by inner field types.
	if !isChunkShape(keys) {
		return c.failOpen()
	}
	var chunk struct {
		ID      string          `json:"id"`
		Model   string          `json:"model"`
		Choices []cStreamChoice `json:"choices"`
		Usage   *cUsage         `json:"usage"`
	}
	if json.Unmarshal(data, &chunk) != nil {
		return c.failOpen()
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
		text, reasoning, dropped := ch.Delta.prose()
		for _, n := range dropped {
			c.note(n)
		}
		if !c.started {
			frames = append(frames, c.start())
		}
		frames = append(frames, c.text(reasoning, "thinking")...)
		frames = append(frames, c.text(text, "text")...)
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

// failOpen ends the stream on a frame that belongs to no dialect this emitter
// speaks: garbage after `data:`, a chunk whose keys promise more than its shape
// delivers, a foreign event family (the Responses API's `response.output_text.delta`
// carrying the answer's tail). Forwarding it is the silent truncation the spec
// forbids — an unlabelled `data:` frame is exactly what an Anthropic SDK ignores
// (packages/ssestream drops every event with no `event:` name), so the client
// reads a closed, complete-looking message whose content is missing that piece,
// with no error. The note makes a backend that does this on every stream visible.
func (c *chatToMessages) failOpen() [][]byte {
	c.failed = true
	c.note("frame=untranslated")
	return [][]byte{frame("error", map[string]any{
		"type": "error",
		"error": map[string]any{
			"type":    "api_error",
			"message": "upstream sent a stream frame this proxy cannot translate",
		},
	})}
}

// anthropicEvents are the `type` values Anthropic's streaming protocol defines: a
// frame carrying one is already in the client's dialect.
var anthropicEvents = map[string]bool{
	"message_start": true, "content_block_start": true, "content_block_delta": true,
	"content_block_stop": true, "message_delta": true, "message_stop": true,
	"ping": true, "error": true,
}

func isAnthropicEvent(raw json.RawMessage) bool {
	var t string
	return json.Unmarshal(raw, &t) == nil && anthropicEvents[t]
}

// reframe re-terminates a frame the emitter forwards untouched. The relay hands
// the frame over WITH its terminator (it slices past it) and FinishFrames hands
// over a tail without one, so the terminator is normalized, not appended: a
// client's parser reads the doubled blank line as an empty event.
func reframe(frame []byte) [][]byte {
	return [][]byte{append(append([]byte{}, bytes.TrimRight(frame, "\r\n")...), '\n', '\n')}
}

// isChunkShape reports whether a JSON object carries anything a Chat Completions
// chunk carries. An object with none of these keys is not a chunk, and the
// branches around this one decide what it is instead.
func isChunkShape(o map[string]json.RawMessage) bool {
	for _, k := range [...]string{"choices", "id", "model", "usage"} {
		if _, ok := o[k]; ok {
			return true
		}
	}
	return false
}

type cStreamChoice struct {
	Delta        cStreamDelta `json:"delta"`
	FinishReason string       `json:"finish_reason"`
	Usage        *cUsage      `json:"usage"`
}

// cStreamDelta is a Chat Completions stream delta with its prose left RAW: a
// multimodal chat backend streams `delta.content` as a content-parts array
// (`[{"type":"text","text":"…"}]`), and a typed string decode fails the WHOLE
// chunk on it — which is how a backend speaking valid Chat Completions came to be
// classified as "not a chunk" and handed to the client native.
type cStreamDelta struct {
	Content   json.RawMessage `json:"content"`
	Reasoning json.RawMessage `json:"reasoning_content"`
	ToolCalls []deltaToolCall `json:"tool_calls"`
}

// prose flattens the delta's two prose channels and reports the parts that carry
// no text this dialect can forward.
func (d cStreamDelta) prose() (text, reasoning string, dropped []string) {
	ts, dropped := contentTexts(d.Content)
	rs, more := contentTexts(d.Reasoning)
	return strings.Join(ts, ""), strings.Join(rs, ""), append(dropped, more...)
}

type deltaToolCall struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
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
	if n := finishNote(fr); n != "" {
		c.note(n)
	}
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
