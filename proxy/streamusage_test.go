package proxy

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestEnsureStreamUsageInjectsIncludeUsage(t *testing.T) {
	body := []byte(`{"model":"gpt-4","stream":true}`)
	rewritten, strip := EnsureStreamUsage(body, false)
	if !strip {
		t.Fatal("expected strip=true when injecting for a client that did not request usage")
	}

	var obj map[string]interface{}
	if err := json.Unmarshal(rewritten, &obj); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	so, ok := obj["stream_options"].(map[string]interface{})
	if !ok {
		t.Fatal("stream_options not injected")
	}
	if inc, _ := so["include_usage"].(bool); !inc {
		t.Error("include_usage should be true")
	}
}

func TestEnsureStreamUsagePreservesExistingStreamOptions(t *testing.T) {
	body := []byte(`{"model":"gpt-4","stream":true,"stream_options":{"foo":"bar"}}`)
	rewritten, _ := EnsureStreamUsage(body, false)

	var obj map[string]interface{}
	_ = json.Unmarshal(rewritten, &obj)
	so := obj["stream_options"].(map[string]interface{})
	if so["foo"] != "bar" {
		t.Errorf("existing stream_options.foo lost: %v", so)
	}
	if inc, _ := so["include_usage"].(bool); !inc {
		t.Error("include_usage should be injected alongside existing options")
	}
}

func TestEnsureStreamUsage_ClientAlreadyRequested(t *testing.T) {
	body := []byte(`{"model":"gpt-4","stream":true,"stream_options":{"include_usage":true}}`)
	rewritten, strip := EnsureStreamUsage(body, false)
	if strip {
		t.Error("should not strip when the client requested include_usage itself")
	}
	if string(rewritten) != string(body) {
		t.Errorf("body should be unchanged, got: %s", rewritten)
	}
}

func TestEnsureStreamUsage_NonStreamingUntouched(t *testing.T) {
	body := []byte(`{"model":"gpt-4","stream":false}`)
	rewritten, strip := EnsureStreamUsage(body, false)
	if strip {
		t.Error("non-streaming request must not enable strip")
	}
	if string(rewritten) != string(body) {
		t.Errorf("non-streaming body should be unchanged, got: %s", rewritten)
	}
}

func TestEnsureStreamUsage_AlwaysIncludeForwards(t *testing.T) {
	body := []byte(`{"model":"gpt-4","stream":true}`)
	rewritten, strip := EnsureStreamUsage(body, true)
	if strip {
		t.Error("always_include_stream_usage=true should forward the usage chunk, not strip")
	}
	var obj map[string]interface{}
	_ = json.Unmarshal(rewritten, &obj)
	so := obj["stream_options"].(map[string]interface{})
	if inc, _ := so["include_usage"].(bool); !inc {
		t.Error("include_usage should still be injected upstream when always-include is set")
	}
}

func TestEnsureStreamUsage_MalformedBodyUntouched(t *testing.T) {
	body := []byte(`not json`)
	rewritten, strip := EnsureStreamUsage(body, false)
	if strip {
		t.Error("malformed body should not enable strip")
	}
	if string(rewritten) != string(body) {
		t.Error("malformed body must be returned unchanged")
	}
}

func TestUsageStripperRemovesInjectedArtifact(t *testing.T) {
	rec := httptest.NewRecorder()
	st := newUsageStripper(rec)

	stream := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"hello"}}]}`,
		`data: {"choices":[{"delta":{"content":" world"}}]}`,
		`data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
		`data: [DONE]`,
	}, "\n\n") + "\n\n"

	if _, err := st.Write([]byte(stream)); err != nil {
		t.Fatalf("write: %v", err)
	}
	st.Flush()

	out := rec.Body.String()
	if strings.Contains(out, "usage") {
		t.Errorf("injected usage artifact not stripped:\n%s", out)
	}
	if !strings.Contains(out, "hello") || !strings.Contains(out, " world") {
		t.Errorf("real content missing:\n%s", out)
	}
	if !strings.Contains(out, "[DONE]") {
		t.Errorf("[DONE] frame should be forwarded:\n%s", out)
	}
}

func TestUsageStripperSplitFrames(t *testing.T) {
	rec := httptest.NewRecorder()
	st := newUsageStripper(rec)

	// Usage artifact split across two Write calls mid-frame.
	prefix := `data: {"choices":[{"delta":{"content":"ok"}}]}` + "\n\ndata: {\"choices\":[],\"us"
	mid := `age":{"prompt_tokens":1}}`
	suffix := "\n\n"

	if _, err := st.Write([]byte(prefix)); err != nil {
		t.Fatal(err)
	}
	if _, err := st.Write([]byte(mid)); err != nil {
		t.Fatal(err)
	}
	if _, err := st.Write([]byte(suffix)); err != nil {
		t.Fatal(err)
	}
	st.Flush()

	out := rec.Body.String()
	if strings.Contains(out, "usage") {
		t.Errorf("usage artifact split across writes not stripped:\n%s", out)
	}
	if !strings.Contains(out, `"ok"`) {
		t.Errorf("content before artifact missing:\n%s", out)
	}
}

func TestUsageStripperDroppedCounting(t *testing.T) {
	rec := httptest.NewRecorder()
	st := newUsageStripper(rec)
	in := []string{`data: {"choices":[{"delta":{"content":"x"}}]}`, `data: {"choices":[],"usage":{}}`}
	payload := strings.Join(in, "\n\n") + "\n\n"

	n, err := st.Write([]byte(payload))
	if err != nil {
		t.Fatalf("write: %v", err)
	}
	if n != len(payload) {
		t.Errorf("Write should report full length even when dropping frames, got %d want %d", n, len(payload))
	}
}

// TestStreamProxyStripUsageIntegration verifies end-to-end that a streamed
// response with an injected usage artifact reaches the client without the usage
// chunk, while ProxyMetrics still reports usage.
func TestStreamProxyStripUsageIntegration(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		body := strings.Join([]string{
			`data: {"choices":[{"delta":{"content":"hi"}}]}`,
			`data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`,
			`data: [DONE]`,
		}, "\n\n") + "\n\n"
		_, _ = w.Write([]byte(body))
	}))

	defer backend.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	w := httptest.NewRecorder()

	pm, err := StreamProxy(req.Context(), backend.URL, "key", req, w, "gpt-4", "gpt-4", nil, true)
	if err != nil {
		t.Fatalf("StreamProxy error: %v", err)
	}
	if pm == nil {
		t.Fatal("pm is nil")
	}

	if strings.Contains(w.Body.String(), "usage") {
		t.Errorf("client stream contains usage artifact:\n%s", w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "hi") {
		t.Errorf("client stream missing content:\n%s", w.Body.String())
	}

	if pm.PromptTokens != 3 || pm.CompletionTokens != 1 {
		t.Errorf("metrics should capture usage despite stripping, got: %+v", pm)
	}
}

func TestEnsureStreamUsageRouterIntegration(t *testing.T) {
	// The router chain: EnsureStreamUsage must be called before RewriteModelInBody
	// so the stream_options survive the per-attempt model rewrite. Exercise at
	// the proxy level: rewrite a body that already has injected stream_options.
	body, _ := EnsureStreamUsage([]byte(`{"model":"gpt-4","stream":true}`), false)
	rewritten, err := RewriteModelInBody(body, "sonnet")
	if err != nil {
		t.Fatalf("rewrite after inject: %v", err)
	}
	var obj map[string]interface{}
	_ = json.Unmarshal(rewritten, &obj)
	so, ok := obj["stream_options"].(map[string]interface{})
	if !ok {
		t.Fatal("stream_options lost during model rewrite")
	}
	if inc, _ := so["include_usage"].(bool); !inc {
		t.Error("include_usage lost during model rewrite")
	}
	if obj["model"] != "sonnet" {
		t.Errorf("model not rewritten, got %v", obj["model"])
	}
}

var _ = context.Background
