package trace

import (
	"strings"
	"testing"
)

func containsFeature(t *testing.T, f []string, want string) {
	t.Helper()
	for _, v := range f {
		if v == want {
			return
		}
	}
	t.Errorf("features %q missing %q", f, want)
}

func TestClassifyClasses(t *testing.T) {
	cases := map[string]string{
		"/v1/chat/completions":     "chat",
		"/v1/messages":             "messages",
		"/v1/responses":            "responses",
		"/v1/embeddings":           "embeddings",
		"/v1/completions":          "completions",
		"/v1/audio/speech":         "speech",
		"/v1/audio/transcriptions": "transcriptions",
		"/v1/audio/voices":         "voices",
		"/tts":                     "tts",
		"/v1/systemone":            "systemone",
	}
	for path, want := range cases {
		if got, _ := Classify(path, "application/json", nil); got != want {
			t.Errorf("%s = %q, want %q", path, got, want)
		}
	}
}

func TestClassifyChatFeatures(t *testing.T) {
	body := []byte(`{"model":"gpt","stream":true,"reasoning":{"effort":"low"},` +
		`"system":"you are helpful","tools":[{"type":"function"},{"type":"function"}]}`)
	class, f := Classify("/v1/chat/completions", "application/json", body)
	if class != "chat" {
		t.Fatalf("class = %q, want chat", class)
	}
	containsFeature(t, f, "stream")
	containsFeature(t, f, "tools:2")
	containsFeature(t, f, "system:15")
	containsFeature(t, f, "reasoning")
	containsFeature(t, f, "json:"+itoa(int64(len(body))))
}

func TestClassifyAgentTurnFeatures(t *testing.T) {
	body := []byte(`{"model":"opus","max_tokens":100,"messages":[{"role":"user","content":[` +
		`{"type":"tool_result","content":"ok"},` +
		`{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}]}]}`)
	_, f := Classify("/v1/messages", "application/json", body)
	containsFeature(t, f, "tool_result")
	containsFeature(t, f, "images:1")
}

func TestClassifyAudioUpload(t *testing.T) {
	_, f := Classify("/v1/audio/transcriptions", "multipart/form-data; boundary=x", []byte("--x\r\n"))
	containsFeature(t, f, "audio-upload")
	for _, v := range f {
		if strings.HasPrefix(v, "json:") {
			t.Errorf("multipart body must not be labeled json, got %q", v)
		}
	}
}

func TestClassifyEmptyBody(t *testing.T) {
	class, f := Classify("/v1/audio/voices", "", nil)
	if class != "voices" || len(f) != 0 {
		t.Errorf("GET voice list = %q/%q, want voices with no features", class, f)
	}
}
