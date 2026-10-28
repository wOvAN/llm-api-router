package trace

import (
	"bytes"
	"encoding/json"
	"path"
	"strings"
)

// MaxRequestBytes and MaxResponseBytes bound what one trace entry keeps per
// body. 512 KiB is generous for a prompt (a 100k-token context is a megabyte of
// text) and small enough that a full ring is tens of megabytes.
const (
	MaxRequestBytes  = 512 << 10
	MaxResponseBytes = 512 << 10
)

// RedactRequest returns the request body as it may live in the trace: prompt
// text untouched, every inlined binary replaced by `<kind SIZE sha256:xxxx…>`.
// The bool reports a body truncated past MaxRequestBytes.
//
// The hash runs over the payload as stored — for a data URL / base64 field that
// is the base64 text, which identifies the payload as well as decoding it, and
// keeps a decoder off the request path.
func RedactRequest(body []byte, contentType string) (string, bool) {
	var out string
	switch {
	case len(body) == 0:
		return "", false
	case strings.HasPrefix(contentType, "multipart/"):
		out = redactMultipart(body)
	default:
		out = redactJSON(body)
	}
	return truncatePrefix([]byte(out), MaxRequestBytes)
}

// redactJSON rewrites the binary leaves of a JSON body. A body that is not JSON
// (plain text, a truncated upload) is returned as-is: redaction is best-effort
// and must never eat the request it documents.
func redactJSON(body []byte) string {
	var v any
	if err := json.Unmarshal(body, &v); err != nil {
		return string(body)
	}
	if !walkRedact(v) {
		return string(body)
	}
	// An encoder with HTML escaping off, so the stub's `<audio …>` survives
	// re-serialisation readable (json.Marshal would write it as <audio).
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return string(body)
	}
	return strings.TrimRight(buf.String(), "\n")
}

func walkRedact(v any) bool {
	changed := false
	switch t := v.(type) {
	case map[string]any:
		for k, val := range t {
			if s, ok := val.(string); ok {
				if kind, size, raw, isBin := binaryString(k, s); isBin {
					t[k] = stub(kind, size, raw)
					changed = true
					continue
				}
			}
			if walkRedact(val) {
				changed = true
			}
		}
	case []any:
		for _, val := range t {
			if walkRedact(val) {
				changed = true
			}
		}
	}
	return changed
}

// binaryString recognises an inlined binary in a JSON string: a data URL (under
// any key), or a long base64 run under an image/audio-flavoured key — the
// Anthropic `source.data` shape, which carries raw base64 with no data: prefix.
func binaryString(key, s string) (kind string, size int64, raw []byte, ok bool) {
	if i := strings.Index(s, ";base64,"); strings.HasPrefix(s, "data:") && i > 0 {
		payload := s[i+len(";base64,"):]
		return coarseKind(s[len("data:"):i]), decodedLen(len(payload)), []byte(payload), true
	}
	kind = kindForKey(key)
	if kind == "" || len(s) < 512 {
		return "", 0, nil, false
	}
	for i := range 64 {
		if i >= len(s) {
			break
		}
		if !isBase64Byte(s[i]) {
			return "", 0, nil, false
		}
	}
	return kind, decodedLen(len(s)), []byte(s), true
}

// kindForKey names a stub after the JSON field the binary came from.
func kindForKey(key string) string {
	switch key {
	case "audio", "audio_data", "input_audio":
		return "audio"
	case "data", "image", "image_data":
		return "image"
	case "pdf_data":
		return "document"
	default:
		return ""
	}
}

// coarseKind maps a media type to a stub kind ("image/png" → image,
// "audio/wav" → audio): the kind a human scans for, not the exact subtype.
func coarseKind(mediaType string) string {
	base, _, _ := strings.Cut(strings.ToLower(strings.TrimSpace(mediaType)), ";")
	top, sub, hasSub := strings.Cut(base, "/")
	switch top {
	case "image", "audio", "video", "text":
		return top
	}
	switch {
	case !hasSub:
		return "binary"
	case sub == "wav" || sub == "mp3" || sub == "ogg" || sub == "opus" || sub == "flac" || sub == "pcm" || sub == "mpeg":
		return "audio"
	default:
		return sub
	}
}

func decodedLen(n int) int64 { return int64(n) * 3 / 4 }

func isBase64Byte(b byte) bool {
	return (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') || (b >= '0' && b <= '9') ||
		b == '+' || b == '/' || b == '='
}

// redactMultipart keeps the part headers and the `model` field and replaces
// every file part's payload with a stub.
//
// ponytail: naive boundary split, the discipline of proxy/multipart.go — the
// form is never parsed, so megabytes of audio are copied zero times.
func redactMultipart(body []byte) string {
	nl := bytes.IndexByte(body, '\n')
	if nl < 0 {
		return string(body)
	}
	delim := string(bytes.TrimRight(body[:nl], "\r\n"))
	var out bytes.Buffer
	for _, part := range bytes.Split(body[nl+1:], []byte("\n"+delim)) {
		out.WriteString(delim)
		head, payload, ok := splitPart(part)
		if !ok {
			out.Write(part)
			continue
		}
		out.Write(head) // the head block ends with its own blank line
		kind := partKind(head)
		if kind == "" {
			out.Write(payload)
			continue
		}
		data := bytes.TrimRight(payload, "\r\n")
		out.WriteString(stub(kind, int64(len(data)), data))
		out.WriteString("\r\n")
	}
	return out.String()
}

// splitPart splits a multipart part into its header block (through the blank
// line, separators included) and its payload.
func splitPart(part []byte) (head, payload []byte, ok bool) {
	for _, sep := range [][]byte{[]byte("\r\n\r\n"), []byte("\n\n")} {
		if i := bytes.Index(part, sep); i >= 0 {
			return part[:i+len(sep)], part[i+len(sep):], true
		}
	}
	return nil, nil, false
}

// partKind names a file part after its Content-Type, falling back to its
// filename extension, and returns "" for a plain form field like `model`. The
// fallback is not paranoia: Go's multipart writer resolves the media type from
// the host's mime table, so on a host whose /etc/mime.types lacks `.wav` the
// part arrives as application/octet-stream and the extension is all there is.
func partKind(head []byte) string {
	lower := bytes.ToLower(head)
	if i := bytes.Index(lower, []byte("content-type:")); i >= 0 {
		value := string(head[i+len("content-type:"):])
		value, _, _ = strings.Cut(value, "\n")
		switch kind := coarseKind(strings.TrimSpace(value)); kind {
		case "binary", "octet-stream":
		default:
			return kind
		}
	}
	if i := bytes.Index(lower, []byte("filename=")); i >= 0 {
		if kind := kindForExt(quotedValue(head[i+len("filename="):])); kind != "" {
			return kind
		}
		return "file"
	}
	return ""
}

// kindForExt names a file part after its filename extension.
func kindForExt(name string) string {
	switch strings.ToLower(path.Ext(name)) {
	case ".wav", ".mp3", ".ogg", ".oga", ".opus", ".flac", ".pcm", ".m4a", ".aac", ".amr":
		return "audio"
	case ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff":
		return "image"
	case ".mp4", ".mov", ".avi", ".mkv", ".webm":
		return "video"
	case ".pdf":
		return "document"
	default:
		return ""
	}
}

// quotedValue reads a quoted multipart header parameter up to its terminator.
func quotedValue(s []byte) string {
	for _, sep := range [][]byte{[]byte(";"), []byte("\r"), []byte("\n")} {
		if i := bytes.Index(s, sep); i >= 0 {
			s = s[:i]
		}
	}
	return string(bytes.Trim(bytes.TrimSpace(s), `"`))
}
