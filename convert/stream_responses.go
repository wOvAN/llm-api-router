package convert

import (
	"bytes"
	"encoding/json"
	"slices"
	"strconv"
	"strings"
	"time"
)

// The streamed responses spoke: a Chat Completions chunk stream rebuilt as
// Responses events. The two dialects disagree about the shape of a stream: a chat
// chunk carries prose with no address, while a Responses stream addresses every
// piece — an item at an `output_index`, a content part at a `content_index`,
// argument deltas per function call — and ends with a terminal event carrying the
// whole response object. So this emitter allocates items as content kinds appear,
// numbers every event, and closes what it opened.
//
// The classification order is the Anthropic emitter's with the dialects swapped
// (see chatToMessages.handle for why each branch is where it is): a named frame is
// judged by its event NAME before anything is parsed, and only the `response.*`
// namespace is already in the client's dialect. A frame in no dialect this emitter
// speaks ends the stream with the client's error event: forwarding it is the
// silent truncation the spec forbids, because a Responses SDK ignores an event it
// has no name for, so the client reads a complete response whose content is
// missing that piece.

func NewChatToResponses() Stream {
	return &chatToResponses{inputTok: -1, msgItem: -1, rsItem: -1}
}

type chatToResponses struct {
	id, model string
	seq       int64
	buf       []byte
	out       [][]byte
	notes     []string

	started  bool // response.created emitted
	finished bool // a terminal event emitted
	failed   bool // an upstream error ended the stream

	inputTok, outputToks, cachedTok int
	stop                            string // the finish reason reported so far

	text      strings.Builder // the answer's prose, accumulated for the .done events
	reasoning strings.Builder
	msgItem   int // output_index of the message item; -1 until added
	rsItem    int // output_index of the reasoning item
	calls     []*respCall
	next      int // the next output_index to allocate
}

// respCall is one streamed function call, keyed by the Chat Completions
// `tool_calls[].index` it streams: parallel calls interleave their argument
// deltas, so each needs its own item and its own argument buffer.
type respCall struct {
	idx  int
	item int
	id   string // the client's call_id (the key of the next request's output)
	name string
	args strings.Builder
}

func (r *chatToResponses) Notes() []string { return r.notes }

func (r *chatToResponses) note(s string) {
	if !slices.Contains(r.notes, s) {
		r.notes = append(r.notes, s)
	}
}

func (r *chatToResponses) WriteFrame(frame []byte) [][]byte {
	r.buf = append(r.buf, frame...)
	out := r.out[:0]
	for {
		i, n := frameEnd(r.buf)
		if i < 0 {
			break
		}
		out = append(out, r.handle(r.buf[:i])...)
		r.buf = r.buf[i+n:]
	}
	r.out = out
	return out
}

// ev numbers and authors one Responses event. Every event in this dialect carries
// `type` and a monotonic `sequence_number`.
func (r *chatToResponses) ev(event string, payload map[string]any) []byte {
	r.seq++
	payload["type"] = event
	payload["sequence_number"] = r.seq
	return frame(event, payload)
}

func (r *chatToResponses) handle(data []byte) [][]byte {
	if r.failed || r.finished {
		return nil
	}
	data = bytes.TrimSpace(data)
	if len(data) == 0 {
		return nil
	}
	if bytes.HasPrefix(data, []byte("event:")) {
		name := data[len("event:"):]
		if i := bytes.IndexAny(name, "\r\n"); i >= 0 {
			name = name[:i]
		}
		if isResponsesEvent(string(bytes.TrimSpace(name))) {
			return reframe(data)
		}
		return r.failOpen()
	}
	if bytes.HasPrefix(data, []byte(":")) {
		return nil
	}
	data = bytes.TrimSpace(bytes.TrimPrefix(data, []byte("data:")))
	if len(data) == 0 {
		return nil
	}
	if bytes.Equal(data, []byte("[DONE]")) {
		return nil
	}
	var keys map[string]json.RawMessage
	if json.Unmarshal(data, &keys) != nil {
		return r.failOpen()
	}
	// A data:-framed event of the client's own dialect (a backend that writes the
	// Responses API without `event:` lines): forwarded, never decoded as a chunk —
	// a `response.completed` frame has a `usage` key, and reading it as a chunk
	// would zero the counts the chat chunks reported.
	var named struct {
		Type string `json:"type"`
	}
	if json.Unmarshal(data, &named) == nil && isResponsesEvent(named.Type) {
		return reframe(data)
	}
	// An Anthropic stream event is the OTHER spoke's dialect: this client has no
	// name for it, so it ends the stream.
	if json.Unmarshal(data, &named) == nil && anthropicEvents[named.Type] {
		return r.failOpen()
	}
	if _, ok := keys["error"]; ok {
		code, msg := upstreamError(data)
		r.note("finish_reason=error")
		r.failed = true
		return [][]byte{r.ev("error", map[string]any{"code": code, "message": msg, "param": nil})}
	}
	if len(keys) == 0 {
		return nil
	}
	if !isChunkShape(keys) {
		return r.failOpen()
	}
	var chunk struct {
		ID      string          `json:"id"`
		Model   string          `json:"model"`
		Choices []cStreamChoice `json:"choices"`
		Usage   *cUsage         `json:"usage"`
	}
	if json.Unmarshal(data, &chunk) != nil {
		return r.failOpen()
	}
	var frames [][]byte
	if chunk.ID != "" {
		r.id = chunk.ID
	}
	if chunk.Model != "" {
		r.model = chunk.Model
	}
	if u := chunk.Usage; u != nil {
		r.readUsage(u)
	}
	for _, ch := range chunk.Choices {
		if u := ch.Usage; u != nil {
			r.readUsage(u)
		}
		if !r.started {
			frames = append(frames, r.start()...)
		}
		text, reasoning, dropped := ch.Delta.prose()
		for _, n := range dropped {
			r.note(n)
		}
		frames = append(frames, r.reasoningDelta(reasoning)...)
		frames = append(frames, r.textDelta(text)...)
		for _, tc := range ch.Delta.ToolCalls {
			frames = append(frames, r.call(tc)...)
		}
		if ch.FinishReason != "" {
			frames = append(frames, r.finish(ch.FinishReason)...)
		}
	}
	return frames
}

func (r *chatToResponses) readUsage(u *cUsage) {
	r.inputTok = u.PromptTokens - u.PromptDetails.CachedTokens
	r.outputToks = u.CompletionTokens
	r.cachedTok = u.PromptDetails.CachedTokens
}

// failOpen ends the stream on a frame no dialect this emitter speaks owns.
func (r *chatToResponses) failOpen() [][]byte {
	r.failed = true
	r.note("frame=untranslated")
	return [][]byte{r.ev("error", map[string]any{
		"code":    "server_error",
		"message": "upstream sent a stream frame this proxy cannot translate",
		"param":   nil,
	})}
}

// isResponsesEvent reports the Responses API's event namespace: everything the
// dialect defines, including the hosted-tool families this proxy never emits, so a
// dual-protocol backend's own events are forwarded rather than destroyed.
func isResponsesEvent(name string) bool {
	return strings.HasPrefix(name, "response.")
}

// upstreamError reads a backend's in-stream error in either shape it arrives in.
func upstreamError(data []byte) (code, msg string) {
	code, msg = "server_error", "the upstream stream ended with an error"
	var obj struct {
		Error struct {
			Type    string `json:"type"`
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(data, &obj) == nil {
		if obj.Error.Message != "" {
			msg = obj.Error.Message
		}
		if obj.Error.Type != "" {
			code = obj.Error.Type
		} else if obj.Error.Code != "" {
			code = obj.Error.Code
		}
		return code, msg
	}
	var s string
	var env struct {
		Error string `json:"error"`
	}
	if json.Unmarshal(data, &env) == nil && env.Error != "" {
		s = env.Error
	}
	if s != "" {
		msg = s
	}
	return code, msg
}

func (r *chatToResponses) start() [][]byte {
	r.started = true
	if r.id == "" {
		r.id = "resp_" + shortHash(strconv.FormatInt(time.Now().UnixNano(), 10))
	}
	partial := map[string]any{"id": r.id, "object": "response", "model": r.model,
		"created_at": time.Now().Unix(), "status": "in_progress", "output": []any{}}
	return [][]byte{
		r.ev("response.created", map[string]any{"response": partial}),
		r.ev("response.in_progress", map[string]any{"response": partial}),
	}
}

func (r *chatToResponses) itemID(kind string) string {
	switch kind {
	case "reasoning":
		return "rs_" + shortHash(r.id+":reasoning")
	case "message":
		return "msg_" + shortHash(r.id+":message")
	}
	return "fc_" + shortHash(r.id+":"+kind)
}

// textDelta streams one piece of the answer into the message item's output_text
// part, adding the item and the part on the first piece.
func (r *chatToResponses) textDelta(s string) [][]byte {
	if s == "" {
		return nil
	}
	var out [][]byte
	if r.msgItem < 0 {
		r.msgItem = r.next
		r.next++
		out = append(out, r.ev("response.output_item.added", map[string]any{
			"output_index": r.msgItem,
			"item": map[string]any{"id": r.itemID("message"), "type": "message",
				"role": "assistant", "status": "in_progress", "content": []any{}}}))
		out = append(out, r.ev("response.content_part.added", map[string]any{
			"item_id": r.itemID("message"), "output_index": r.msgItem, "content_index": 0,
			"part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}}}))
	}
	r.text.WriteString(s)
	return append(out, r.ev("response.output_text.delta", map[string]any{
		"item_id": r.itemID("message"), "output_index": r.msgItem, "content_index": 0,
		"delta": s, "logprobs": []any{}}))
}

func (r *chatToResponses) reasoningDelta(s string) [][]byte {
	if s == "" {
		return nil
	}
	var out [][]byte
	if r.rsItem < 0 {
		r.rsItem = r.next
		r.next++
		out = append(out, r.ev("response.output_item.added", map[string]any{
			"output_index": r.rsItem,
			"item": map[string]any{"id": r.itemID("reasoning"), "type": "reasoning",
				"summary": []any{}, "status": "in_progress"}}))
	}
	r.reasoning.WriteString(s)
	return append(out, r.ev("response.reasoning_summary_text.delta", map[string]any{
		"item_id": r.itemID("reasoning"), "output_index": r.rsItem,
		"summary_index": 0, "delta": s}))
}

// call streams one Chat Completions tool-call delta into its own function_call
// item, keyed on the chunk's `tool_calls[].index`.
func (r *chatToResponses) call(tc deltaToolCall) [][]byte {
	var out [][]byte
	for i, c := range r.calls {
		if c.idx != tc.Index {
			continue
		}
		if tc.ID != "" && tc.ID != c.id {
			// The index reappeared with a fresh id: a new call, not a continuation.
			out = append(out, r.closeCall(i)...)
			break
		}
		if tc.Function.Arguments == "" {
			return out
		}
		c.args.WriteString(tc.Function.Arguments)
		return append(out, r.ev("response.function_call_arguments.delta", map[string]any{
			"item_id": r.callItemID(c.id), "output_index": c.item,
			"call_id": c.id, "delta": tc.Function.Arguments}))
	}
	item := r.next
	r.next++
	id := tc.ID
	if id == "" {
		id = "call_" + strconv.Itoa(item)
	}
	r.calls = append(r.calls, &respCall{idx: tc.Index, item: item, id: id, name: tc.Function.Name})
	out = append(out, r.ev("response.output_item.added", map[string]any{
		"output_index": item,
		"item": map[string]any{"id": r.callItemID(id), "type": "function_call",
			"call_id": id, "name": tc.Function.Name, "arguments": "", "status": "in_progress"}}))
	if tc.Function.Arguments == "" {
		return out
	}
	r.calls[len(r.calls)-1].args.WriteString(tc.Function.Arguments)
	return append(out, r.ev("response.function_call_arguments.delta", map[string]any{
		"item_id": r.callItemID(id), "output_index": item,
		"call_id": id, "delta": tc.Function.Arguments}))
}

// callItemID is the function_call item's id, stable across the item.added, the
// argument deltas, the item.done and the terminal response object: a client keys
// its accumulator on it.
func (r *chatToResponses) callItemID(id string) string { return "fc_" + shortHash(r.id+":"+id) }

func (r *chatToResponses) closeCall(i int) [][]byte {
	c := r.calls[i]
	out := [][]byte{r.ev("response.function_call_arguments.done", map[string]any{
		"item_id": r.callItemID(c.id), "output_index": c.item,
		"call_id": c.id, "arguments": c.args.String()})}
	return append(out, r.ev("response.output_item.done", map[string]any{
		"output_index": c.item,
		"item": map[string]any{"id": r.callItemID(c.id), "type": "function_call",
			"call_id": c.id, "name": c.name, "arguments": c.args.String(), "status": "completed"}}))
}

// closeItems closes every open item: the prose parts first (a Responses client
// reads the parts inside the item, so the part events precede the item's done).
func (r *chatToResponses) closeItems() [][]byte {
	var out [][]byte
	if r.msgItem >= 0 {
		out = append(out,
			r.ev("response.output_text.done", map[string]any{"item_id": r.itemID("message"),
				"output_index": r.msgItem, "content_index": 0, "text": r.text.String(),
				"logprobs": []any{}}),
			r.ev("response.content_part.done", map[string]any{"item_id": r.itemID("message"),
				"output_index": r.msgItem, "content_index": 0,
				"part": map[string]any{"type": "output_text", "text": r.text.String(),
					"annotations": []any{}}}),
			r.ev("response.output_item.done", map[string]any{"output_index": r.msgItem,
				"item": map[string]any{"id": r.itemID("message"), "type": "message",
					"role": "assistant", "status": "completed",
					"content": []any{map[string]any{"type": "output_text",
						"text": r.text.String(), "annotations": []any{}}}}}))
	}
	if r.rsItem >= 0 {
		out = append(out,
			r.ev("response.reasoning_summary_text.done", map[string]any{
				"item_id": r.itemID("reasoning"), "output_index": r.rsItem,
				"summary_index": 0, "text": r.reasoning.String()}),
			r.ev("response.output_item.done", map[string]any{"output_index": r.rsItem,
				"item": map[string]any{"id": r.itemID("reasoning"), "type": "reasoning",
					"status": "completed",
					"summary": []any{map[string]any{"type": "summary_text",
						"text": r.reasoning.String()}}}}))
	}
	for i := range r.calls {
		out = append(out, r.closeCall(i)...)
	}
	return out
}

// finish closes the items and reports the terminal event. A stream that continues
// to [DONE] after the finish_reason chunk must not complete twice, so the terminal
// event is emitted once (FinishFrames honours `finished`).
func (r *chatToResponses) finish(fr string) [][]byte {
	if n := finishNote(fr); n != "" {
		r.note(n)
	}
	if fr == "error" {
		r.note("finish_reason=error")
		r.failed = true
		r.finished = true
		return [][]byte{r.terminal()}
	}
	r.stop = fr
	out := r.closeItems()
	r.finished = true
	return append(out, r.terminal())
}

func (r *chatToResponses) terminal() []byte {
	in := r.inputTok
	if in < 0 {
		in = 0
	}
	resp := map[string]any{
		"id": r.id, "object": "response", "model": r.model,
		"created_at": time.Now().Unix(), "output": r.items(),
		"usage": map[string]any{
			"input_tokens":  in,
			"output_tokens": r.outputToks,
			"input_tokens_details": map[string]any{
				"cached_tokens": r.cachedTok},
			"total_tokens": in + r.outputToks,
		},
	}
	if r.failed {
		resp["status"] = "failed"
		resp["error"] = map[string]any{"code": "server_error",
			"message": "the upstream stream ended with an error"}
		return r.ev("response.failed", map[string]any{"response": resp})
	}
	status, reason := responsesStatus(r.stop)
	resp["status"] = status
	if reason != "" {
		resp["incomplete_details"] = map[string]any{"reason": reason}
		return r.ev("response.incomplete", map[string]any{"response": resp})
	}
	return r.ev("response.completed", map[string]any{"response": resp})
}

// items is the response's `output`: the same items the item events streamed, in
// the order they appeared, so an SDK that joined mid-stream still reads the whole
// answer.
func (r *chatToResponses) items() []any {
	out := make([]any, 0, 3)
	if r.rsItem >= 0 {
		out = append(out, map[string]any{"id": r.itemID("reasoning"), "type": "reasoning",
			"status": "completed",
			"summary": []any{map[string]any{"type": "summary_text",
				"text": r.reasoning.String()}}})
	}
	if r.msgItem >= 0 {
		out = append(out, map[string]any{"id": r.itemID("message"), "type": "message",
			"role": "assistant", "status": "completed",
			"content": []any{map[string]any{"type": "output_text",
				"text": r.text.String(), "annotations": []any{}}}})
	}
	for _, c := range r.calls {
		out = append(out, map[string]any{"id": r.callItemID(c.id), "type": "function_call",
			"call_id": c.id, "name": c.name, "arguments": c.args.String(), "status": "completed"})
	}
	return out
}

// FinishFrames closes a stream that ended without a finish_reason and delivers a
// tail frame the relay handed over without its blank-line terminator, then the
// Responses terminator the client's parser waits for.
func (r *chatToResponses) FinishFrames() [][]byte {
	out := r.out[:0]
	if len(bytes.TrimSpace(r.buf)) > 0 {
		out = append(out, r.handle(r.buf)...)
		r.buf = nil
	}
	if !r.started || r.failed {
		r.out = out
		return out
	}
	if !r.finished {
		out = append(out, r.closeItems()...)
		out = append(out, r.terminal())
		r.finished = true
	}
	return append(out, r.done())
}

func (r *chatToResponses) done() []byte {
	return []byte("data: [DONE]\n\n")
}
