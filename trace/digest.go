package trace

import (
	"bytes"
	"encoding/json"
	"strings"
)

// DigestResponse reads a response body once, when the entry is written to the ring.
// SSE bodies are stored as the raw frames the backend sent, so the deltas are
// reassembled and the event names counted here (OpenAI chat, OpenAI Responses,
// Anthropic messages, audio.cpp's audio SSE) — the compact view the GUI shows
// before a human expands the raw stream.
func DigestResponse(raw []byte, contentType string) Digest {
	ct := strings.ToLower(contentType)
	switch {
	case strings.Contains(ct, "text/event-stream"):
		return digestStream(raw)
	case strings.Contains(ct, "json"):
		return digestJSONResponse(raw)
	default:
		// TTS WAV, raw PCM: nothing to parse, the capture renders the stub.
		return Digest{}
	}
}

// digestStream reassembles deltas and counts event names on one pass. A frame
// that fails to parse is counted and skipped — the digest is a view of the
// stream, never a reason to lose the entry.
func digestStream(raw []byte) Digest {
	d := Digest{}
	d.Events = map[string]int{}
	var body, think strings.Builder
	var event string

	forEachLine(raw, func(line []byte) {
		if bytes.HasPrefix(line, []byte("event:")) {
			event = string(bytes.TrimSpace(line[len("event:"):]))
			d.Events[event]++
			return
		}
		if !bytes.HasPrefix(line, []byte("data:")) {
			return
		}
		payload := bytes.TrimSpace(line[len("data:"):])
		if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
			return
		}
		var obj map[string]any
		if err := json.Unmarshal(payload, &obj); err != nil {
			d.Events["unparsed"]++
			return
		}
		name := event
		if t, _ := obj["type"].(string); t != "" {
			name = t
		}
		if name == "" {
			name = "data"
		}
		if event == "" || name != event {
			d.Events[name]++
		}

		switch name {
		case "response.output_text.delta", "transcript.text.delta":
			if s, _ := obj["delta"].(string); s != "" {
				body.WriteString(s)
			}
		case "speech.audio.delta":
			d.AudioDeltas++
		case "content_block_delta":
			if delta, _ := obj["delta"].(map[string]any); delta != nil {
				writeStrings(&body, delta["text"])
				writeStrings(&think, delta["thinking"])
			}
		case "message_delta", "response.completed":
			// Only a real answer overwrites: the frames after message_delta
			// carry no reason and must not erase it.
			if fr := firstFinishReason(obj); fr != "" {
				d.FinishReason = fr
			}
		}
		if _, ok := obj["choices"]; ok {
			openAIChunkText(&body, &think, &d, obj)
		}
		event = ""
	})

	d.Text = joinChannels(body.String(), think.String())
	if len(d.Events) == 0 {
		d.Events = nil
	}
	return d
}

// openAIChunkText pulls text and the finish reason out of OpenAI's
// `choices[]` shape (chat completions and legacy completions).
func openAIChunkText(body, think *strings.Builder, d *Digest, obj map[string]any) {
	choices, _ := obj["choices"].([]any)
	if len(choices) == 0 {
		return
	}
	c, _ := choices[0].(map[string]any)
	if c == nil {
		return
	}
	if delta, _ := c["delta"].(map[string]any); delta != nil {
		writeStrings(body, delta["content"])
		writeStrings(think, delta["reasoning_content"])
	}
	if msg, _ := c["message"].(map[string]any); msg != nil {
		writeStrings(body, msg["content"])
	}
	if fr, _ := c["finish_reason"].(string); fr != "" {
		d.FinishReason = fr
	}
}

// joinChannels renders the completion: the answer, then the model's reasoning
// after it, separated so the two channels never glue into one unreadable word.
func joinChannels(body, think string) string {
	if think == "" {
		return body
	}
	if body == "" {
		return think
	}
	return body + " " + think
}

// firstFinishReason reads the stop reason where each dialect puts it: Anthropic
// nests it in the message_delta's delta, Responses in the completed event's
// response.
func firstFinishReason(obj map[string]any) string {
	if inner, _ := obj["delta"].(map[string]any); inner != nil {
		if s, _ := inner["stop_reason"].(string); s != "" {
			return s
		}
	}
	if inner, _ := obj["response"].(map[string]any); inner != nil {
		if s, _ := inner["status"].(string); s != "" {
			return s
		}
	}
	return ""
}

// writeStrings appends every string it finds, flattening the array form
// (Responses' output_text is an array of strings).
func writeStrings(b *strings.Builder, vals ...any) {
	for _, v := range vals {
		switch t := v.(type) {
		case string:
			b.WriteString(t)
		case []any:
			for _, item := range t {
				if s, ok := item.(string); ok {
					b.WriteString(s)
				}
			}
		}
	}
}

// digestJSONResponse reads a non-streaming response: the completion text
// (OpenAI `choices[].message.content`, Anthropic `content[]` blocks, Responses
// `output_text`) and its finish reason.
func digestJSONResponse(raw []byte) Digest {
	var d Digest
	var obj map[string]any
	if err := json.Unmarshal(raw, &obj); err != nil {
		return d
	}
	var body, think strings.Builder
	if _, ok := obj["choices"]; ok {
		openAIChunkText(&body, &think, &d, obj)
	}
	writeStrings(&body, obj["output_text"])
	switch blocks := obj["content"].(type) {
	case string:
		body.WriteString(blocks)
	case []any:
		for _, blk := range blocks {
			m, _ := blk.(map[string]any)
			if m == nil {
				continue
			}
			writeStrings(&body, m["text"])
			writeStrings(&think, m["thinking"])
		}
	}
	d.Text = joinChannels(body.String(), think.String())
	if fr := firstFinishReason(obj); fr != "" {
		d.FinishReason = fr
	}
	if s, _ := obj["finish_reason"].(string); s != "" {
		d.FinishReason = s
	}
	if s, _ := obj["stop_reason"].(string); s != "" {
		d.FinishReason = s
	}
	return d
}

func forEachLine(body []byte, fn func(line []byte)) {
	for offset := 0; offset < len(body); {
		nl := bytes.IndexByte(body[offset:], '\n')
		var line []byte
		if nl == -1 {
			line = body[offset:]
			offset = len(body)
		} else {
			line = body[offset : offset+nl]
			offset += nl + 1
		}
		line = bytes.TrimSpace(line)
		if len(line) > 0 {
			fn(line)
		}
	}
}
