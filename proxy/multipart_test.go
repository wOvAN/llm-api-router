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

func TestWAVDurationMs(t *testing.T) {
	// 16 kHz mono 16-bit: 16000 samples = 1s of audio.
	wav := func(samples int) []byte {
		var b bytes.Buffer
		le32w := func(v uint32) { b.Write([]byte{byte(v), byte(v >> 8), byte(v >> 16), byte(v >> 24)}) }
		le16w := func(v uint16) { b.Write([]byte{byte(v), byte(v >> 8)}) }
		b.WriteString("RIFF")
		le32w(uint32(36 + samples*2))
		b.WriteString("WAVEfmt ")
		le32w(16)
		le16w(1) // PCM
		le16w(1) // mono
		le32w(16000)
		le32w(32000) // byte rate
		le16w(2)
		le16w(16)
		b.WriteString("data")
		le32w(uint32(samples * 2))
		b.Write(bytes.Repeat([]byte{0x01}, samples*2))
		return b.Bytes()
	}

	t.Run("raw WAV upload", func(t *testing.T) {
		if ms, ok := WAVDurationMs(wav(16000)); !ok || ms != 1000 {
			t.Errorf("WAVDurationMs = %d,%v, want 1000,true", ms, ok)
		}
	})

	t.Run("audio part of a multipart form", func(t *testing.T) {
		var buf bytes.Buffer
		w := multipart.NewWriter(&buf)
		if err := w.WriteField("model", "whisper"); err != nil {
			t.Fatalf("WriteField: %v", err)
		}
		part, err := w.CreateFormFile("file", "take.wav")
		if err != nil {
			t.Fatalf("CreateFormFile: %v", err)
		}
		if _, err := part.Write(wav(48000)); err != nil { // 3s
			t.Fatalf("Write: %v", err)
		}
		if err := w.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
		if ms, ok := WAVDurationMs(buf.Bytes()); !ok || ms != 3000 {
			t.Errorf("WAVDurationMs = %d,%v, want 3000,true", ms, ok)
		}
	})

	t.Run("no audio", func(t *testing.T) {
		if ms, ok := WAVDurationMs([]byte(`{"model":"m","messages":[]}`)); ok || ms != 0 {
			t.Errorf("WAVDurationMs = %d,%v, want 0,false", ms, ok)
		}
	})

	t.Run("a RIFF that is not a WAVE container terminates", func(t *testing.T) {
		// An AIFF/FLAC upload carries the four bytes RIFF in its own header and
		// in its audio data. A scan that re-reads the same magic loops forever.
		body := append(bytes.Repeat([]byte("RIFF"), 100000), []byte("AIFF")...)
		if ms, ok := WAVDurationMs(body); ok || ms != 0 {
			t.Errorf("WAVDurationMs = %d,%v, want 0,false", ms, ok)
		}
	})

	t.Run("WAVE with no data chunk stays unknown", func(t *testing.T) {
		head := wav(16000)
		dataChunk := 12 + 8 + 16 // RIFF header + 'fmt ' chunk header + its 16 bytes
		if ms, ok := WAVDurationMs(head[:dataChunk]); ok || ms != 0 {
			t.Errorf("WAVDurationMs = %d,%v, want 0,false", ms, ok)
		}
	})
}
