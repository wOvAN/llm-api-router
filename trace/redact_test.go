package trace

import (
	"bytes"
	"mime/multipart"
	"strings"
	"testing"
)

func TestRedactRequestDataURLAndAnthropicSource(t *testing.T) {
	body := []byte(`{"model":"opus","messages":[{"role":"user","content":[` +
		`{"type":"image_url","image_url":{"url":"data:image/png;base64,` + strings.Repeat("A", 4000) + `"}},` +
		`{"type":"text","text":"what is in the picture?"}` +
		`]}]}`)
	got, trunc := RedactRequest(body, "application/json")
	if trunc {
		t.Error("small body must not be truncated")
	}
	if strings.Contains(got, strings.Repeat("A", 100)) {
		t.Error("base64 payload must be replaced by a stub")
	}
	if !strings.Contains(got, "<image ") || !strings.Contains(got, "sha256:") {
		t.Errorf("stub missing: %s", got)
	}
	if !strings.Contains(got, "what is in the picture?") {
		t.Error("prompt text must survive untouched")
	}
}

func TestRedactRequestRawBase64Source(t *testing.T) {
	body := []byte(`{"model":"sonnet","source":{"type":"base64","media_type":"image/png","data":"` +
		strings.Repeat("QUJD", 300) + `"}}`)
	got, _ := RedactRequest(body, "application/json")
	if strings.Contains(got, strings.Repeat("QUJD", 4)) {
		t.Error("raw base64 source.data must be replaced by a stub")
	}
	if !strings.Contains(got, "<image ") {
		t.Errorf("stub missing: %s", got)
	}
}

func TestRedactRequestTextUntouched(t *testing.T) {
	body := []byte(`{"model":"gpt","messages":[{"role":"user","content":"hello"}]}`)
	got, _ := RedactRequest(body, "application/json")
	if got != string(body) {
		t.Errorf("text-only request changed:\n%s", got)
	}
}

func TestRedactRequestTruncatesOversizedBody(t *testing.T) {
	body := bytes.Repeat([]byte("x"), MaxRequestBytes+4096)
	got, trunc := RedactRequest(body, "text/plain")
	if !trunc {
		t.Error("oversized body must report truncation")
	}
	if !strings.HasSuffix(got, "…[truncated 4096 bytes]") {
		t.Errorf("truncation marker missing: …%s", got[len(got)-40:])
	}
	if len(got) <= MaxRequestBytes {
		t.Errorf("len = %d, want the prefix plus the marker", len(got))
	}
}

func TestRedactRequestMultipartAudio(t *testing.T) {
	audio := append([]byte("RIFF"), bytes.Repeat([]byte{0x01}, 4000)...)
	var b bytes.Buffer
	mw := multipart.NewWriter(&b)
	_ = mw.WriteField("model", "whisper")
	part, err := mw.CreateFormFile("file", "clip.wav")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(audio); err != nil {
		t.Fatal(err)
	}
	if err := mw.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	got, _ := RedactRequest(b.Bytes(), mw.FormDataContentType())
	if !bytes.Contains(b.Bytes(), audio) {
		t.Fatal("test fixture lost the audio")
	}
	if strings.Contains(got, "\x01\x01\x01") {
		t.Error("audio bytes must be replaced by a stub")
	}
	if !strings.Contains(got, "<audio ") || !strings.Contains(got, "sha256:") {
		t.Errorf("stub missing: %s", got)
	}
	if !strings.Contains(got, `name="model"`) || !strings.Contains(got, "whisper") {
		t.Error("the model field and part headers must stay readable")
	}
}
