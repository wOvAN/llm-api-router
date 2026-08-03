package proxy

import (
	"bytes"
	"encoding/json"
	"net/http"
)

// EnsureStreamUsage injects stream_options.include_usage=true into OpenAI
// chat completion streaming request bodies so backends report token usage
// (which our metrics then track), even when the client did not opt in.
//
// Injection mirrors LiteLLM's default stream-usage tracking: only OpenAI
// streaming chat completion requests are modified, and a request that already
// sets include_usage=true (or that is not streaming) is returned unchanged.
//
// Returns the (possibly rewritten) body and whether the injected usage chunk
// should be stripped from the client stream. When alwaysInclude is true the
// usage chunk is forwarded to the client (match LiteLLM's
// always_include_stream_usage setting); otherwise the injected artifact is
// hidden while still being captured in metrics.
func EnsureStreamUsage(body []byte, alwaysInclude bool) ([]byte, bool) {
	var obj map[string]interface{}
	if err := json.Unmarshal(body, &obj); err != nil {
		return body, false
	}

	streaming, _ := obj["stream"].(bool)
	if !streaming {
		return body, false
	}

	streamOptions, _ := obj["stream_options"].(map[string]interface{})
	if inc, ok := streamOptions["include_usage"].(bool); ok && inc {
		// Client already asked for usage — leave the request and response untouched.
		return body, false
	}

	if streamOptions == nil {
		streamOptions = map[string]interface{}{}
		obj["stream_options"] = streamOptions
	}
	streamOptions["include_usage"] = true

	out, err := json.Marshal(obj)
	if err != nil {
		return body, false
	}
	return out, !alwaysInclude
}

// usageStripper filters injected stream-usage artifacts out of an SSE response
// as it streams to the client. It sits between the metrics writer and the
// client so metric capture (which reads unfiltered response bytes) is
// unaffected while the client never sees the injected usage chunk.
type usageStripper struct {
	http.ResponseWriter
	pending []byte
	dropped bool // true once we have stripped at least one frame
}

func newUsageStripper(w http.ResponseWriter) *usageStripper {
	return &usageStripper{ResponseWriter: w}
}

// Write buffers incoming SSE bytes and forwards it, minus usage-only frames.
// A frame is dropped when all of its choices are empty (no finish_reason, no
// logprobs, no populated delta) or when it has an empty choices array — the
// exact shape of the injected usage / prompt-filter chunk.
func (s *usageStripper) Write(data []byte) (int, error) {
	s.pending = append(s.pending, data...)

	total := 0
	for {
		end := findSSEFrameEnd(s.pending)
		if end < 0 {
			break
		}
		frame := s.pending[:end]
		s.pending = s.pending[end:]
		if isUsageArtifact(frame) {
			s.dropped = true
			continue
		}
		// A dropped frame's bytes still count as "written" from the caller's
		// perspective (the metrics layer already accounted for them separately).
		n, err := s.ResponseWriter.Write(frame)
		total += n
		if err != nil {
			return total, err
		}
	}
	return len(data), nil
}

// Flush flushes any remaining buffered bytes, then the underlying flusher.
func (s *usageStripper) Flush() {
	if len(s.pending) > 0 {
		_, _ = s.ResponseWriter.Write(s.pending)
		s.pending = nil
	}
	if flusher, ok := s.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

// findSSEFrameEnd returns the index just past the end of the next complete SSE
// frame (a blank-line-delimited event: "\n\n", "\r\n\r\n" or "\r\r"), or -1
// when no complete frame is buffered yet.
func findSSEFrameEnd(b []byte) int {
	if i := bytes.Index(b, []byte("\r\n\r\n")); i >= 0 {
		return i + 4
	}
	if i := bytes.Index(b, []byte("\n\n")); i >= 0 {
		return i + 2
	}
	if i := bytes.Index(b, []byte("\r\r")); i >= 0 {
		return i + 2
	}
	return -1
}

// isUsageArtifact reports whether an SSE frame is an injected stream-usage
// artifact: a JSON payload whose choices are all empty, or whose choices array
// is empty (the exact byte shape appended by upstreams when include_usage is
// requested). Non-JSON frames (e.g. "data: [DONE]") are never artifacts.
func isUsageArtifact(frame []byte) bool {
	line := bytes.TrimSpace(frame)
	line = bytes.TrimPrefix(line, []byte("data:"))
	line = bytes.TrimSpace(line)
	if len(line) == 0 || bytes.Equal(line, []byte("[DONE]")) {
		return false
	}
	var obj map[string]interface{}
	if err := json.Unmarshal(line, &obj); err != nil {
		return false
	}
	choices, ok := obj["choices"]
	if !ok {
		return false
	}
	arr, ok := choices.([]interface{})
	if !ok {
		return false
	}
	if len(arr) == 0 {
		return true
	}
	for _, c := range arr {
		cm, ok := c.(map[string]interface{})
		if !ok || !choiceIsEmpty(cm) {
			return false
		}
	}
	return true
}

// choiceIsEmpty reports whether a single streaming choice carries no payload:
// it has no finish_reason, no logprobs, and its delta is absent or all-null.
func choiceIsEmpty(c map[string]interface{}) bool {
	if fr, ok := c["finish_reason"]; ok && fr != nil {
		return false
	}
	if lp, ok := c["logprobs"]; ok && lp != nil {
		return false
	}
	delta, has := c["delta"]
	if !has {
		return true
	}
	dm, ok := delta.(map[string]interface{})
	if !ok {
		return false
	}
	for _, v := range dm {
		if v != nil {
			return false
		}
	}
	return true
}
