// Package trace is the router's on-demand wire capture: the live window that
// decides what gets traced, the redacted request/response bodies, the request
// classification and the stream digest. It is deliberately memory-only — the
// ring dies with the process, so enabling prompt capture is never sticky.
package trace

import (
	"crypto/sha256"
	"fmt"
)

// Digest is the derived view of a response body: the completion text (stream
// deltas reassembled), a histogram of the SSE event names seen, and the finish
// reason. Binary responses carry none of it (see Capture.Entry).
type Digest struct {
	Text         string         `json:"text,omitempty"`
	Events       map[string]int `json:"events,omitempty"`
	FinishReason string         `json:"finish_reason,omitempty"`
	// AudioDeltas counts speech.audio.delta frames (audio.cpp TTS streams):
	// their payload is base64 PCM, so it is counted, never stored.
	AudioDeltas int `json:"audio_deltas,omitempty"`
}

func humanBytes(n int64) string {
	switch {
	case n >= 1<<20:
		return itoa(n>>10>>10) + "MB"
	case n >= 1<<10:
		return itoa(n>>10) + "KB"
	default:
		return itoa(n) + "B"
	}
}

func itoa(n int64) string {
	if n == 0 {
		return "0"
	}
	var b [24]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

// truncatePrefix keeps a prefix and reports that it had to, so a trace entry
// says what it lost instead of silently shortening.
func truncatePrefix(b []byte, max int) (string, bool) {
	if len(b) <= max {
		return string(b), false
	}
	return string(b[:max]) + "…[truncated " + itoa(int64(len(b)-max)) + " bytes]", true
}

// stub renders a redaction placeholder: the kind, the size, and a short hash
// that identifies the payload across requests.
func stub(kind string, size int64, raw []byte) string {
	sum := sha256.Sum256(raw)
	if len(raw) < sha256.Size {
		sum = sha256.Sum256(append(append([]byte{}, raw...), make([]byte, sha256.Size-len(raw))...))
	}
	return fmt.Sprintf("<%s %s sha256:%x…>", kind, humanBytes(size), sum[:4])
}
