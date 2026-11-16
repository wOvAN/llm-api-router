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

// WAVDurationMs returns the length, in milliseconds, of the first WAV audio in
// a request body - a raw WAV upload, or the audio part of a multipart form,
// which is what an STT upload is. It reads the RIFF chunk chain (fmt 's
// byte-rate over the data chunk's byte count), so a compressed upload is
// measured by its stored bytes like the backend measures it by its samples.
//
// This is the only source for the length of a STREAMED recognition: audio.cpp
// builds the wall_ms/audio_duration_ms/rtf timing object - and the
// X-AudioCPP-* headers - on the non-streaming paths only; a streamed answer's
// terminal event carries a timing object with a single ttft_ms key, and a
// diarized stream carries the turns. The upload is in hand, so the number that
// the backend chose not to serialize is the one the router can state.
func WAVDurationMs(body []byte) (int64, bool) {
	for at := 0; ; {
		j := bytes.Index(body[at:], []byte("RIFF"))
		if j < 0 {
			return 0, false
		}
		i := at + j
		// Every reject advances past the magic: a form field that happens to
		// carry the four bytes RIFF (an AIFF or a FLAC upload does) must not
		// send the scan back to the same offset, which is a core pegged per
		// request.
		at = i + 4
		if i+12 > len(body) || string(body[i+8:i+12]) != "WAVE" {
			continue
		}
		var byteRate uint32
		var dataBytes uint64
		for p := i + 12; p+8 <= len(body); {
			size := le32(body[p+4:])
			bodyStart := p + 8
			switch string(body[p : p+4]) {
			case "fmt ":
				if bodyStart+16 <= len(body) {
					byteRate = le32(body[bodyStart+8:])
				}
			case "data":
				dataBytes = uint64(size)
			}
			if dataBytes > 0 && byteRate > 0 {
				return int64(dataBytes * 1000 / uint64(byteRate)), true
			}
			// Chunks are word-aligned: an odd size pads with a fill byte.
			p = bodyStart + int(size) + int(size&1)
		}
	}
}

func le32(b []byte) uint32 {
	if len(b) < 4 {
		return 0
	}
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}
