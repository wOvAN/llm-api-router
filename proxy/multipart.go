package proxy

import (
	"bytes"
	"fmt"
	"strings"
)

// Audio endpoints deliver the model name in a multipart/form-data field, next to
// the audio part: OpenAI/llama.cpp `POST /v1/audio/transcriptions`, llama.cpp
// `POST /tts` (speaker reference), audio.cpp `/v1/audio/alignments` and
// `/v1/batches/transcriptions`. An upload is megabytes, so the field is located
// by scanning bytes — parsing the form would copy the whole audio part.

// multipartModelSpan locates the value of the "model" form field. The lookup
// mirrors the naive boundary split the audio backends themselves use (audio.cpp's
// multipart parser is one): the part's headers end at the first blank line after
// its Content-Disposition, the value ends at the boundary delimiter line that
// follows it. Quoted and unquoted `name` spellings are both accepted.
func multipartModelSpan(body []byte) (int, int, bool) {
	for _, marker := range []string{`name="model"`, `name=model`} {
		i := bytes.Index(body, []byte(marker))
		if i < 0 {
			continue
		}
		rest := i + len(marker)
		start := -1
		if j := bytes.Index(body[rest:], []byte("\r\n\r\n")); j >= 0 {
			start = rest + j + 4
		} else if j := bytes.Index(body[rest:], []byte("\n\n")); j >= 0 {
			start = rest + j + 2
		}
		if start < 0 {
			continue
		}
		end := -1
		if j := bytes.Index(body[start:], []byte("\r\n--")); j >= 0 {
			end = start + j
		} else if j := bytes.Index(body[start:], []byte("\n--")); j >= 0 {
			end = start + j
		}
		if end <= start {
			continue
		}
		return start, end, true
	}
	return 0, 0, false
}

// ExtractModelFromMultipart returns the "model" form field of a multipart body.
func ExtractModelFromMultipart(body []byte) (string, bool) {
	start, end, ok := multipartModelSpan(body)
	if !ok {
		return "", false
	}
	model := strings.TrimSpace(string(body[start:end]))
	return model, model != ""
}

// RewriteModelInMultipart replaces the "model" form field value, leaving every
// other byte — the audio part above all — untouched. An empty newModel is a
// no-op: audio rules are plain pass-through, and emptying the field would break
// a single-model backend that reads it only to disambiguate in router mode.
func RewriteModelInMultipart(body []byte, newModel string) ([]byte, error) {
	if newModel == "" {
		return body, nil
	}
	start, end, ok := multipartModelSpan(body)
	if !ok {
		return nil, fmt.Errorf("missing 'model' field")
	}
	out := make([]byte, 0, len(body)-(end-start)+len(newModel))
	out = append(out, body[:start]...)
	out = append(out, newModel...)
	out = append(out, body[end:]...)
	return out, nil
}
