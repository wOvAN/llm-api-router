package proxy

import (
	"bytes"
	"encoding/json"
	"net/http"

	"llm-api-router/pkg/log"
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
	var obj map[string]any
	if err := json.Unmarshal(body, &obj); err != nil {
		log.Debugf("EnsureStreamUsage: failed to parse request body: %v", err)
		return body, false
	}

	streaming, _ := obj["stream"].(bool)
	if !streaming {
		return body, false
	}

	streamOptions, _ := obj["stream_options"].(map[string]any)
	if inc, ok := streamOptions["include_usage"].(bool); ok && inc {
		// Client already asked for usage — leave the request and response untouched.
		return body, false
	}

	if streamOptions == nil {
		streamOptions = map[string]any{}
		obj["stream_options"] = streamOptions
	}
	streamOptions["include_usage"] = true

	out, err := json.Marshal(obj)
	if err != nil {
		log.Errorf("EnsureStreamUsage: failed to re-marshal request body: %v", err)
		return body, false
	}
	return out, !alwaysInclude
}

// sseRelay delivers SSE frames to the client one at a time. Incoming bytes are
// buffered until a complete frame (blank-line-terminated event) is available,
// then each frame is written and flushed separately; partial frames are never
// forwarded mid-stream. With strip=true it additionally filters injected
// stream-usage artifacts out of the stream (frames whose choices are all
// empty) while metrics capture the unfiltered bytes upstream of it.
type sseRelay struct {
	http.ResponseWriter
	pending []byte
	strip   bool
	dropped bool // true once at least one artifact frame was stripped
}

func newSSERelay(w http.ResponseWriter, strip bool) *sseRelay {
	return &sseRelay{ResponseWriter: w, strip: strip}
}

// Write buffers incoming SSE bytes and forwards complete frames, minus
// usage-only frames when stripping. A frame is dropped when all of its choices
// are empty (no finish_reason, no logprobs, no populated delta) or when it has
// an empty choices array — the exact shape of the injected usage /
// prompt-filter chunk.
func (s *sseRelay) Write(data []byte) (int, error) {
	s.pending = append(s.pending, data...)

	total := 0
	for {
		end := findSSEFrameEnd(s.pending)
		if end < 0 {
			break
		}
		frame := s.pending[:end]
		s.pending = s.pending[end:]
		if s.strip && isUsageArtifact(frame) {
			s.dropped = true
			continue
		}
		validateSSEFrame(frame, "sseRelay")
		// A dropped frame's bytes still count as "written" from the caller's
		// perspective (the metrics layer already accounted for them separately).
		n, err := s.ResponseWriter.Write(frame)
		total += n
		// Flush per frame: the upstream may coalesce several frames into one
		// read (llama.cpp batches finish+usage, TCP merges token chunks), and
		// clients that treat one delivery as one SSE event mis-split frames
		// that arrive in a single write.
		s.flush()
		if err != nil {
			return total, err
		}
	}
	return len(data), nil
}

func (s *sseRelay) flush() {
	if flusher, ok := s.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

// Flush forwards the flush downstream without releasing incomplete frames.
// Mid-stream flushes (one per streamed chunk) must not push a partial data
// line to the client — that is exactly the fragmentation that makes clients
// glue the tail of one frame onto the next. Trailing bytes are delivered by
// finish() once the stream ends.
func (s *sseRelay) Flush() {
	s.flush()
}

// finish writes any remaining buffered bytes (a trailing frame the backend
// closed without a blank-line terminator) and flushes. Called once, at the end
// of the stream.
func (s *sseRelay) finish() {
	if len(s.pending) > 0 {
		if _, err := s.ResponseWriter.Write(s.pending); err != nil {
			log.Errorf("sseRelay: failed to flush buffered SSE frames to client: %v", err)
		}
		s.pending = nil
	}
	s.flush()
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
	var obj map[string]any
	if err := json.Unmarshal(line, &obj); err != nil {
		log.Debugf("usageStripper: failed to parse SSE frame: %v", err)
		return false
	}
	choices, ok := obj["choices"]
	if !ok {
		return false
	}
	arr, ok := choices.([]any)
	if !ok {
		return false
	}
	if len(arr) == 0 {
		return true
	}
	for _, c := range arr {
		cm, ok := c.(map[string]any)
		if !ok || !choiceIsEmpty(cm) {
			return false
		}
	}
	return true
}

// choiceIsEmpty reports whether a single streaming choice carries no payload:
// it has no finish_reason, no logprobs, and its delta is absent or all-null.
func choiceIsEmpty(c map[string]any) bool {
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
	dm, ok := delta.(map[string]any)
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

// validateSSEFrame logs a warning when a complete SSE frame carries a data
// line that is not valid JSON. The frame is still forwarded unchanged — the
// router is a byte-pass-through and must never drop or re-frame upstream
// bytes — but the warning surfaces the corruption at the exact hop that
// produced it: a malformed frame in the logs proves the bytes were already
// broken upstream (the router never reorders, splits or duplicates bytes),
// which pinpoints a buggy backend build or an intermediate proxy.
func validateSSEFrame(frame []byte, writer string) {
	for line := range bytes.SplitSeq(frame, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, []byte("data:")) {
			continue
		}
		payload := bytes.TrimSpace(line[len("data:"):])
		if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
			continue
		}
		var obj map[string]any
		if err := json.Unmarshal(payload, &obj); err != nil {
			log.Warnf("%s: malformed SSE data line is not valid JSON: %v; frame: %s",
				writer, err, truncateBytes(payload, 512))
		}
	}
}
