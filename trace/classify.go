package trace

import (
	"bytes"
	"strings"
)

// EndpointLabel names the API endpoint of a request path. It is the single
// source of the vocabulary the metrics, the trace and the GUI all speak.
func EndpointLabel(path string) string {
	switch {
	case strings.Contains(path, "/messages"):
		return "messages"
	case strings.Contains(path, "/chat/completions"):
		return "chat"
	case strings.Contains(path, "/responses"):
		return "responses"
	case strings.Contains(path, "/embeddings"):
		return "embeddings"
	case strings.Contains(path, "/audio/speech"):
		return "speech"
	case strings.Contains(path, "/audio/transcriptions"):
		return "transcriptions"
	case strings.Contains(path, "/audio/voices"):
		return "voices"
	case strings.HasSuffix(path, "/tts"):
		return "tts"
	case strings.Contains(path, "/completions"):
		return "completions"
	}
	if _, after, ok := strings.Cut(path, "/v1/"); ok {
		if seg, _, _ := strings.Cut(after, "/"); seg != "" {
			return seg
		}
	}
	return "unknown"
}

// Classify labels a request and lists the secondary signs a byte scan can see.
// The body is scanned, never parsed: the same discipline as the router's model
// extraction and the multipart rewriter (an agent turn is megabytes, and the
// features are heuristics for a human reading the trace, not a parser).
func Classify(path, contentType string, body []byte) (string, []string) {
	class := EndpointLabel(path)
	var f []string
	if len(body) == 0 {
		return class, f
	}
	if jsonKeyIsTrue(body, "stream") {
		f = append(f, "stream")
	}
	// ponytail: tool declarations counted by their dialect's marker — OpenAI
	// sends {"type":"function"}, Anthropic sends input_schema. A count, not a
	// parse: good enough to spot "the client sent 40 tools".
	if n := bytes.Count(body, []byte(`"input_schema"`)) + bytes.Count(body, []byte(`"type":"function"`)); n > 0 {
		f = append(f, "tools:"+itoa(int64(n)))
	}
	if bytes.Contains(body, []byte(`"tool_result"`)) {
		f = append(f, "tool_result")
	}
	// The three shapes an image request takes on the wire: a data URL (both
	// dialects), OpenAI's image_url field, Anthropic's base64 source (which
	// names media_type instead of a URL).
	if n := max(bytes.Count(body, []byte("data:image/")),
		bytes.Count(body, []byte(`"image_url"`)),
		bytes.Count(body, []byte(`"media_type"`))); n > 0 {
		f = append(f, "images:"+itoa(int64(n)))
	}
	if start, end, ok := topLevelValueSpan(body, "system"); ok {
		f = append(f, "system:"+itoa(int64(end-start)))
	}
	if bytes.Contains(body, []byte(`"reasoning`)) || bytes.Contains(body, []byte(`"thinking"`)) {
		f = append(f, "reasoning")
	}
	if strings.HasPrefix(contentType, "multipart/") {
		f = append(f, "audio-upload")
	} else if strings.HasPrefix(contentType, "application/json") || contentType == "" {
		f = append(f, "json:"+itoa(int64(len(body))))
	}
	return class, f
}

// jsonKeyIsTrue reports whether the top-level key's value is the literal true.
func jsonKeyIsTrue(body []byte, key string) bool {
	start, end, ok := topLevelValueSpan(body, key)
	return ok && bytes.Equal(bytes.TrimSpace(body[start:end]), []byte("true"))
}

// topLevelValueSpan locates the value of a top-level JSON key, scanning bytes:
// it walks quoted strings (so a `"system"` inside a prompt is not a key) and
// container depth, then spans the value. For a string the span excludes the
// quotes, so `system:<chars>` is the text length.
func topLevelValueSpan(body []byte, key string) (start, end int, ok bool) {
	needle := []byte(`"` + key + `"`)
	depth := 0
	for i := 0; i < len(body); {
		switch body[i] {
		case '{', '[':
			depth++
			i++
		case '}', ']':
			depth--
			i++
		case '"':
			isKey := depth == 1 && bytes.HasPrefix(body[i:], needle)
			j := i + len(needle)
			if isKey && j < len(body) && isNameByte(body[j]) {
				isKey = false // a longer key: "system_prompt" is not "system"
			}
			if !isKey {
				i = skipJSONString(body, i)
				continue
			}
			i = j
			for i < len(body) && isSpaceByte(body[i]) {
				i++
			}
			if i >= len(body) || body[i] != ':' {
				i++
				continue
			}
			i++
			for i < len(body) && isSpaceByte(body[i]) {
				i++
			}
			start = i
			end = valueEnd(body, i)
			if start < len(body) && body[start] == '"' {
				start++ // the span is the text, not its quotes
			}
			return start, end, true
		default:
			i++
		}
	}
	return 0, 0, false
}

// valueEnd spans a JSON value from start: a string to its closing quote
// (exclusive), a container to the brace that closes it, anything else to the
// next comma or closing brace.
func valueEnd(body []byte, start int) int {
	if start >= len(body) {
		return start
	}
	switch body[start] {
	case '"':
		return skipJSONString(body, start) - 1
	case '{', '[':
		depth := 0
		for i := start; i < len(body); i++ {
			switch body[i] {
			case '{', '[':
				depth++
			case '}', ']':
				depth--
				if depth == 0 {
					return i + 1
				}
			case '"':
				i = skipJSONString(body, i) - 1
			}
		}
		return len(body)
	default:
		end := start
		for end < len(body) && body[end] != ',' && body[end] != '}' && body[end] != ']' {
			end++
		}
		return end
	}
}

func skipJSONString(body []byte, i int) int {
	i++
	for i < len(body) {
		switch body[i] {
		case '\\':
			i += 2
		case '"':
			return i + 1
		default:
			i++
		}
	}
	return len(body)
}

func isNameByte(b byte) bool {
	return b == '_' || b == '-' || (b >= '0' && b <= '9') || (b|0x20 >= 'a' && b|0x20 <= 'z')
}

func isSpaceByte(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}
