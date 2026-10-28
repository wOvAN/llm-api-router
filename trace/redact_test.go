package trace

import (
	"bytes"
	"mime/multipart"
	"net/textproto"
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

// A text part is prompt content, not a payload: the JSONL file of a batch
// upload, an inline JSON tool spec. Stubbing it away deletes the one thing a
// person opens the trace to read.
func TestRedactRequestMultipartTextPartIsVerbatim(t *testing.T) {
	const jsonl = `{"custom_id":"1","body":{"model":"gpt","messages":[]}}`
	var b bytes.Buffer
	mw := multipart.NewWriter(&b)
	part, err := mw.CreatePart(textproto.MIMEHeader{
		"Content-Type": []string{"application/json"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write([]byte(jsonl)); err != nil {
		t.Fatal(err)
	}
	file, err := mw.CreateFormFile("file", "clip.wav")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.Write(bytes.Repeat([]byte{0x01}, 2000)); err != nil {
		t.Fatal(err)
	}
	if err := mw.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	got, _ := RedactRequest(b.Bytes(), mw.FormDataContentType())
	if !strings.Contains(got, jsonl) {
		t.Errorf("text part must be stored verbatim, got:\n%s", got)
	}
	if !strings.Contains(got, "<audio ") {
		t.Error("the audio part in the same form must still be stubbed")
	}
}

// Redaction must not re-serialise: a pretty-printed prompt keeps its layout, and
// the body the operator reads is the body the client sent.
func TestRedactRequestKeepsJSONLayout(t *testing.T) {
	body := "{\n  \"model\": \"opus\",\n  \"messages\": [{\n    \"role\": \"user\",\n    \"content\": \"data:image/png;base64," +
		strings.Repeat("A", 2000) + "\"\n  }]\n}"
	got, _ := RedactRequest([]byte(body), "application/json")
	if strings.Contains(got, strings.Repeat("A", 100)) {
		t.Error("base64 payload must be replaced by a stub")
	}
	if !strings.Contains(got, "\"model\": \"opus\"") || !strings.Contains(got, "\n  }]\n}") {
		t.Errorf("body re-serialised, layout lost:\n%s", got)
	}
}
