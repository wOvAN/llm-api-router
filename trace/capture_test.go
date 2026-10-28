package trace

import (
	"strings"
	"testing"
)

func TestCaptureStoresRedactedRequestAndRawResponse(t *testing.T) {
	c := NewCapture("POST", "/v1/chat/completions", "gpt", "application/json",
		[]byte(`{"model":"gpt","messages":[{"role":"user","content":"hi"}],"stream":true}`))
	if c.ID() == "" {
		t.Fatal("capture must carry a trace id")
	}
	c.ResponseContentType("text/event-stream")
	c.WriteResponse([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"He\"}}]}\n\n"))
	c.WriteResponse([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"y\"},\"finish_reason\":\"stop\"}]}\n\n"))

	e := c.Entry()
	if e.Request == "" || !strings.Contains(e.Request, "hi") {
		t.Errorf("request = %q, want the prompt text", e.Request)
	}
	if !strings.Contains(e.Response, "data: {") {
		t.Errorf("response = %q, want the frames as they arrived", e.Response)
	}
	if e.Digest.Text != "Hey" || e.Digest.FinishReason != "stop" {
		t.Errorf("digest = %+v, want the reassembled text and finish_reason", e.Digest)
	}
	if len(e.Features) == 0 {
		t.Error("features from the request body must be carried")
	}
}

func TestCaptureBinaryResponseIsAStub(t *testing.T) {
	c := NewCapture("POST", "/v1/audio/speech", "tts", "application/json", []byte(`{"model":"tts","input":"hi"}`))
	c.ResponseContentType("audio/wav")
	c.WriteResponse([]byte("RIFF"))
	c.WriteResponse([]byte(strings.Repeat("\x00", 4096)))

	e := c.Entry()
	if !strings.HasPrefix(e.Response, "<audio ") || !strings.Contains(e.Response, "sha256:") {
		t.Errorf("response = %q, want an audio stub with size and hash", e.Response)
	}
	if strings.Contains(e.Response, "\x00") {
		t.Error("audio bytes must not be buffered")
	}
	if e.Digest.Text != "" {
		t.Error("a binary response has nothing to digest")
	}
	if !c.HasResponse() {
		t.Error("HasResponse must report the bytes seen")
	}
}

func TestCaptureCapsResponse(t *testing.T) {
	c := NewCapture("POST", "/v1/chat/completions", "gpt", "application/json", nil)
	c.ResponseContentType("application/json")
	chunk := strings.Repeat("x", 128<<10)
	for range (MaxResponseBytes/(128<<10) + 2) {
		c.WriteResponse([]byte(chunk))
	}
	e := c.Entry()
	if !e.ResponseTruncated {
		t.Error("response over the cap must report truncation")
	}
	if !strings.HasSuffix(e.Response, "bytes]") {
		t.Errorf("response tail = %q, want the truncation marker", e.Response[len(e.Response)-20:])
	}
}

func TestCaptureWithoutResponseIsEmpty(t *testing.T) {
	c := NewCapture("POST", "/v1/chat/completions", "gpt", "application/json", nil)
	if c.HasResponse() {
		t.Error("no chunks means no response")
	}
	if e := c.Entry(); e.Response != "" {
		t.Errorf("response = %q, want empty so the router can fill the error body", e.Response)
	}
}
