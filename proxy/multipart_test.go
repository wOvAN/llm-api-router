package proxy

import (
	"bytes"
	"mime/multipart"
	"strings"
	"testing"
)

func TestExtractModelFromMultipart(t *testing.T) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	if err := w.WriteField("model", "whisper"); err != nil {
		t.Fatalf("WriteField: %v", err)
	}
	if err := w.WriteField("language", "ru"); err != nil {
		t.Fatalf("WriteField: %v", err)
	}
	fw, err := w.CreateFormFile("file", "a.wav")
	if err != nil {
		t.Fatalf("CreateFormFile: %v", err)
	}
	if _, err := fw.Write([]byte("RIFFxxxxWAVEfmt ")); err != nil {
		t.Fatalf("write audio: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	body := buf.Bytes()

	got, ok := ExtractModelFromMultipart(body)
	if !ok || got != "whisper" {
		t.Fatalf("ExtractModelFromMultipart = %q,%v; want whisper,true", got, ok)
	}

	out, err := RewriteModelInMultipart(body, "whisper-large-v3")
	if err != nil {
		t.Fatalf("RewriteModelInMultipart: %v", err)
	}
	if got, ok := ExtractModelFromMultipart(out); !ok || got != "whisper-large-v3" {
		t.Errorf("rewritten model = %q,%v; want whisper-large-v3", got, ok)
	}
	// Everything else — sibling fields, the audio part, the closing boundary —
	// must survive byte-for-byte.
	for _, want := range []string{"language", "ru", "RIFF", "WAVEfmt", w.FormDataContentType()[len("multipart/form-data; boundary="):]} {
		if !bytes.Contains(out, []byte(want)) {
			t.Errorf("rewritten body lost %q: %s", want, out)
		}
	}
	if len(out) <= len(body) {
		t.Errorf("rewritten body not longer than the original: %d <= %d", len(out), len(body))
	}
	// A pass-through rule (no target model) must leave the upload untouched.
	same, err := RewriteModelInMultipart(body, "")
	if err != nil || !bytes.Equal(same, body) {
		t.Errorf("empty target model must be a no-op (err %v)", err)
	}

	// LF-only body with an unquoted field name (hand-written clients).
	lf := "--B\nContent-Disposition: form-data; name=model\n\nvoxtral\n--B--\n"
	if got, ok := ExtractModelFromMultipart([]byte(lf)); !ok || got != "voxtral" {
		t.Errorf("LF/unquoted multipart = %q,%v; want voxtral,true", got, ok)
	}

	if _, err := RewriteModelInMultipart([]byte("--B\nContent-Disposition: form-data; name=file\n\nRIFF\n--B--\n"), "m"); err == nil {
		t.Error("missing model field must be an error")
	}
	if strings.Contains(string(body), "whisper-large-v3") {
		t.Error("original body mutated")
	}
}
