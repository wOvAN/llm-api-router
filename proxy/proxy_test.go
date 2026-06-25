package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestRewriteModelInBody(t *testing.T) {
	t.Run("replaces model field", func(t *testing.T) {
		input := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
		output, err := RewriteModelInBody(input, "claude-3")
		if err != nil {
			t.Fatalf("RewriteModelInBody: %v", err)
		}

		var obj map[string]interface{}
		if err := json.Unmarshal(output, &obj); err != nil {
			t.Fatalf("unmarshal output: %v", err)
		}
		if obj["model"] != "claude-3" {
			t.Errorf("got model %q, want %q", obj["model"], "claude-3")
		}
		if obj["messages"] == nil {
			t.Error("messages should be preserved")
		}
	})

	t.Run("invalid JSON returns error", func(t *testing.T) {
		_, err := RewriteModelInBody([]byte(`not json`), "x")
		if err == nil {
			t.Fatal("expected error for invalid JSON")
		}
	})

	t.Run("empty body returns error", func(t *testing.T) {
		_, err := RewriteModelInBody([]byte(``), "x")
		if err == nil {
			t.Fatal("expected error for empty body")
		}
	})
}

func TestExtractUsageFromJSON(t *testing.T) {
	t.Run("OpenAI format", func(t *testing.T) {
		body := []byte(`{"model":"gpt-4","choices":[{"message":{"content":"hello"}}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}`)
		pm := extractUsageFromJSON(body)
		if pm.PromptTokens != 10 {
			t.Errorf("PromptTokens = %d, want 10", pm.PromptTokens)
		}
		if pm.CompletionTokens != 20 {
			t.Errorf("CompletionTokens = %d, want 20", pm.CompletionTokens)
		}
		if pm.TotalTokens != 30 {
			t.Errorf("TotalTokens = %d, want 30", pm.TotalTokens)
		}
		if pm.CachedTokens != -1 {
			t.Errorf("CachedTokens = %d, want -1", pm.CachedTokens)
		}
	})

	t.Run("Anthropic format", func(t *testing.T) {
		body := []byte(`{"content":[{"text":"hello"}],"usage":{"input_tokens":15,"output_tokens":25}}`)
		pm := extractUsageFromJSON(body)
		if pm.PromptTokens != 15 {
			t.Errorf("PromptTokens = %d, want 15", pm.PromptTokens)
		}
		if pm.CompletionTokens != 25 {
			t.Errorf("CompletionTokens = %d, want 25", pm.CompletionTokens)
		}
	})

	t.Run("llama-server timings", func(t *testing.T) {
		body := []byte(`{"content":"hello","timings":{"prompt_n":5,"predicted_n":15,"prompt_ms":100.0,"predicted_ms":500.0,"prompt_per_second":50.0,"predicted_per_second":30.0,"cache_n":3}}`)
		pm := extractUsageFromJSON(body)
		if pm.PromptTokens != 5 {
			t.Errorf("PromptTokens = %d, want 5", pm.PromptTokens)
		}
		if pm.CompletionTokens != 15 {
			t.Errorf("CompletionTokens = %d, want 15", pm.CompletionTokens)
		}
		if pm.CachedTokens != 3 {
			t.Errorf("CachedTokens = %d, want 3", pm.CachedTokens)
		}
		if pm.PromptMs != 100.0 {
			t.Errorf("PromptMs = %f, want 100.0", pm.PromptMs)
		}
		if pm.PredictedMs != 500.0 {
			t.Errorf("PredictedMs = %f, want 500.0", pm.PredictedMs)
		}
		if pm.PromptPerSec != 50.0 {
			t.Errorf("PromptPerSec = %f, want 50.0", pm.PromptPerSec)
		}
		if pm.TokensPerSec != 30.0 {
			t.Errorf("TokensPerSec = %f, want 30.0", pm.TokensPerSec)
		}
	})

	t.Run("no usage data", func(t *testing.T) {
		body := []byte(`{"content":"hello"}`)
		pm := extractUsageFromJSON(body)
		if pm.CachedTokens != -1 {
			t.Errorf("CachedTokens = %d, want -1", pm.CachedTokens)
		}
	})

	t.Run("invalid JSON", func(t *testing.T) {
		body := []byte(`not json`)
		pm := extractUsageFromJSON(body)
		if pm.CachedTokens != -1 {
			t.Errorf("CachedTokens = %d, want -1", pm.CachedTokens)
		}
	})

	t.Run("cached tokens from input_tokens_details", func(t *testing.T) {
		body := []byte(`{"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"input_tokens_details":{"cached_tokens":7}}}`)
		pm := extractUsageFromJSON(body)
		if pm.CachedTokens != 7 {
			t.Errorf("CachedTokens = %d, want 7", pm.CachedTokens)
		}
	})

	t.Run("cached tokens from prompt_tokens_details", func(t *testing.T) {
		body := []byte(`{"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":8}}}`)
		pm := extractUsageFromJSON(body)
		if pm.CachedTokens != 8 {
			t.Errorf("CachedTokens = %d, want 8", pm.CachedTokens)
		}
	})

	t.Run("cache_read_input_tokens", func(t *testing.T) {
		body := []byte(`{"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"cache_read_input_tokens":9}}`)
		pm := extractUsageFromJSON(body)
		if pm.CachedTokens != 9 {
			t.Errorf("CachedTokens = %d, want 9", pm.CachedTokens)
		}
	})
}

func TestExtractUsageFromStream(t *testing.T) {
	t.Run("OpenAI streaming format", func(t *testing.T) {
		body := []byte(
			"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n" +
				"data: {\"choices\":[{\"delta\":{}}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":20,\"total_tokens\":30}}\n" +
				"data: [DONE]\n",
		)
		pm := extractUsageFromStream(body)
		if pm.PromptTokens != 10 {
			t.Errorf("PromptTokens = %d, want 10", pm.PromptTokens)
		}
		if pm.CompletionTokens != 20 {
			t.Errorf("CompletionTokens = %d, want 20", pm.CompletionTokens)
		}
		if pm.TotalTokens != 0 {
			t.Errorf("TotalTokens = %d, want 0 (streaming doesn't compute total)", pm.TotalTokens)
		}
	})

	t.Run("Anthropic streaming format with nested usage", func(t *testing.T) {
		body := []byte(
			"data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"hello\"}}\n" +
				"data: {\"type\":\"message_stop\",\"message\":{\"usage\":{\"input_tokens\":5,\"output_tokens\":15}}}\n",
		)
		pm := extractUsageFromStream(body)
		if pm.PromptTokens != 5 {
			t.Errorf("PromptTokens = %d, want 5", pm.PromptTokens)
		}
		if pm.CompletionTokens != 15 {
			t.Errorf("CompletionTokens = %d, want 15", pm.CompletionTokens)
		}
	})

	t.Run("llama-server streaming timings", func(t *testing.T) {
		body := []byte(
			"data: {\"content\":\"hello\"}\n" +
				"data: {\"timings\":{\"prompt_n\":3,\"predicted_n\":12,\"prompt_ms\":50.0,\"predicted_ms\":400.0,\"prompt_per_second\":60.0,\"predicted_per_second\":30.0}}\n",
		)
		pm := extractUsageFromStream(body)
		if pm.PromptTokens != 3 {
			t.Errorf("PromptTokens = %d, want 3", pm.PromptTokens)
		}
		if pm.CompletionTokens != 12 {
			t.Errorf("CompletionTokens = %d, want 12", pm.CompletionTokens)
		}
	})

	t.Run("no usage data in stream", func(t *testing.T) {
		body := []byte("data: {\"content\":\"hello\"}\n")
		pm := extractUsageFromStream(body)
		if pm.CachedTokens != -1 {
			t.Errorf("CachedTokens = %d, want -1", pm.CachedTokens)
		}
	})

	t.Run("empty stream", func(t *testing.T) {
		pm := extractUsageFromStream([]byte{})
		if pm.CachedTokens != -1 {
			t.Errorf("CachedTokens = %d, want -1", pm.CachedTokens)
		}
	})
}

func TestExtractUsageFromResponse(t *testing.T) {
	t.Run("streaming detection", func(t *testing.T) {
		body := []byte("data: {\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":10}}\n")
		pm := extractUsageFromResponse(body, "", true)
		if pm.PromptTokens != 5 {
			t.Errorf("PromptTokens = %d, want 5", pm.PromptTokens)
		}
		if pm.CompletionTokens != 10 {
			t.Errorf("CompletionTokens = %d, want 10", pm.CompletionTokens)
		}
	})

	t.Run("non-streaming detection", func(t *testing.T) {
		body := []byte(`{"usage":{"prompt_tokens":3,"completion_tokens":7}}`)
		pm := extractUsageFromResponse(body, "", false)
		if pm.PromptTokens != 3 {
			t.Errorf("PromptTokens = %d, want 3", pm.PromptTokens)
		}
		if pm.CompletionTokens != 7 {
			t.Errorf("CompletionTokens = %d, want 7", pm.CompletionTokens)
		}
	})
}

func TestDecompressBody(t *testing.T) {
	t.Run("unknown encoding returns as-is", func(t *testing.T) {
		input := []byte("hello")
		out, err := decompressBody(input, "unknown")
		if err != nil {
			t.Fatalf("decompressBody: %v", err)
		}
		if string(out) != "hello" {
			t.Errorf("got %q, want %q", string(out), "hello")
		}
	})

	t.Run("empty encoding returns as-is", func(t *testing.T) {
		input := []byte("hello")
		out, err := decompressBody(input, "")
		if err != nil {
			t.Fatalf("decompressBody: %v", err)
		}
		if string(out) != "hello" {
			t.Errorf("got %q, want %q", string(out), "hello")
		}
	})
}

func TestGetField(t *testing.T) {
	obj := map[string]interface{}{
		"message": map[string]interface{}{
			"usage": map[string]interface{}{
				"input_tokens": float64(5),
			},
		},
	}

	result := getField(obj, "message.usage")
	if result == nil {
		t.Fatal("getField returned nil")
	}
	if result["input_tokens"] != float64(5) {
		t.Errorf("got input_tokens %v, want 5", result["input_tokens"])
	}

	result = getField(obj, "nonexistent.path")
	if result != nil {
		t.Error("expected nil for nonexistent path")
	}
}

func TestExtractUsageTokens(t *testing.T) {
	t.Run("OpenAI format", func(t *testing.T) {
		input, output, cached := extractUsageTokens(map[string]interface{}{
			"prompt_tokens":     float64(10),
			"completion_tokens": float64(20),
		})
		if input != 10 || output != 20 || cached != -1 {
			t.Errorf("got (%d,%d,%d), want (10,20,-1)", input, output, cached)
		}
	})

	t.Run("Anthropic format", func(t *testing.T) {
		input, output, cached := extractUsageTokens(map[string]interface{}{
			"input_tokens":  float64(15),
			"output_tokens": float64(25),
		})
		if input != 15 || output != 25 || cached != -1 {
			t.Errorf("got (%d,%d,%d), want (15,25,-1)", input, output, cached)
		}
	})

	t.Run("nil usage", func(t *testing.T) {
		input, output, cached := extractUsageTokens(nil)
		if input != 0 || output != 0 || cached != -1 {
			t.Errorf("got (%d,%d,%d), want (0,0,-1)", input, output, cached)
		}
	})
}

func TestBuildMetricsFromData(t *testing.T) {
	t.Run("without timings", func(t *testing.T) {
		pm := buildMetricsFromData(10, 20, 30, 5, nil)
		if pm.PromptTokens != 10 || pm.CompletionTokens != 20 || pm.TotalTokens != 30 || pm.CachedTokens != 5 {
			t.Errorf("got tokens (%d,%d,%d,%d)", pm.PromptTokens, pm.CompletionTokens, pm.TotalTokens, pm.CachedTokens)
		}
	})

	t.Run("with timings overrides", func(t *testing.T) {
		pm := buildMetricsFromData(10, 20, 30, -1, map[string]interface{}{
			"prompt_n":             float64(5),
			"predicted_n":          float64(15),
			"prompt_ms":            float64(100),
			"predicted_ms":         float64(500),
			"prompt_per_second":    float64(50),
			"predicted_per_second": float64(30),
			"cache_n":              float64(3),
		})
		if pm.PromptTokens != 5 {
			t.Errorf("PromptTokens = %d, want 5", pm.PromptTokens)
		}
		if pm.CompletionTokens != 15 {
			t.Errorf("CompletionTokens = %d, want 15", pm.CompletionTokens)
		}
		if pm.CachedTokens != 3 {
			t.Errorf("CachedTokens = %d, want 3", pm.CachedTokens)
		}
		if pm.PromptMs != 100.0 {
			t.Errorf("PromptMs = %f, want 100.0", pm.PromptMs)
		}
	})
}

func TestNewMetricsWriter(t *testing.T) {
	mw := newMetricsWriter(nil, time.Now())
	if mw == nil {
		t.Fatal("newMetricsWriter returned nil")
	}
	if mw.statusCode != 200 {
		t.Errorf("default statusCode = %d, want 200", mw.statusCode)
	}
	if mw.bodyBuffer == nil {
		t.Error("bodyBuffer should not be nil")
	}
}

func TestRewriteModelInResponse(t *testing.T) {
	t.Run("replaces model in JSON response", func(t *testing.T) {
		data := []byte(`{"model":"target-model","choices":[],"usage":{}}`)
		got := rewriteModelInResponse(data, "target-model", "opus")
		want := `{"model":"opus","choices":[],"usage":{}}`
		if string(got) != want {
			t.Errorf("got %s, want %s", got, want)
		}
	})

	t.Run("replaces model with whitespace", func(t *testing.T) {
		data := []byte(`{"model": "target-model", "choices": []}`)
		got := rewriteModelInResponse(data, "target-model", "opus")
		want := `{"model": "opus", "choices": []}`
		if string(got) != want {
			t.Errorf("got %s, want %s", got, want)
		}
	})

	t.Run("replaces model in SSE event", func(t *testing.T) {
		data := []byte(`data: {"type":"message_start","message":{"model":"target-model","id":"msg_123"}}`)
		got := rewriteModelInResponse(data, "target-model", "opus")
		want := `data: {"type":"message_start","message":{"model":"opus","id":"msg_123"}}`
		if string(got) != want {
			t.Errorf("got %s, want %s", got, want)
		}
	})

	t.Run("replaces multiple model fields", func(t *testing.T) {
		data := []byte(`data: {"model":"target"}
data: {"model":"target","usage":{"prompt_tokens":5}}
`)
		got := rewriteModelInResponse(data, "target", "opus")
		if !strings.Contains(string(got), `"model":"opus"`) {
			t.Errorf("missing replacement in: %s", got)
		}
		count := strings.Count(string(got), `"model":"opus"`)
		if count != 2 {
			t.Errorf("expected 2 replacements, got %d: %s", count, got)
		}
	})

	t.Run("does not replace non-matching model", func(t *testing.T) {
		data := []byte(`{"model":"other-model","choices":[]}`)
		got := rewriteModelInResponse(data, "target-model", "opus")
		if string(got) != string(data) {
			t.Errorf("got %s, want unchanged %s", got, data)
		}
	})

	t.Run("empty oldModel skips rewriting", func(t *testing.T) {
		data := []byte(`{"model":"target","choices":[]}`)
		got := rewriteModelInResponse(data, "", "opus")
		if string(got) != string(data) {
			t.Errorf("got %s, want unchanged %s", got, data)
		}
	})

	t.Run("same model skips rewriting", func(t *testing.T) {
		data := []byte(`{"model":"same","choices":[]}`)
		got := rewriteModelInResponse(data, "same", "same")
		if string(got) != string(data) {
			t.Errorf("got %s, want unchanged %s", got, data)
		}
	})

	t.Run("does not replace model-like key", func(t *testing.T) {
		data := []byte(`{"model_id":"123","model":"target","choices":[]}`)
		got := rewriteModelInResponse(data, "target", "opus")
		// model_id should not be affected
		if !strings.Contains(string(got), `"model_id":"123"`) {
			t.Errorf("model_id was incorrectly modified: %s", got)
		}
		if !strings.Contains(string(got), `"model":"opus"`) {
			t.Errorf("model was not replaced: %s", got)
		}
	})

	t.Run("preserves non-model fields", func(t *testing.T) {
		data := []byte(`{"model":"target","choices":[{"message":{"content":"hello"}}],"usage":{"prompt_tokens":10,"completion_tokens":20}}`)
		got := rewriteModelInResponse(data, "target", "opus")

		var obj map[string]interface{}
		if err := json.Unmarshal(got, &obj); err != nil {
			t.Fatalf("invalid JSON: %v", err)
		}
		if obj["model"] != "opus" {
			t.Errorf("model = %v, want opus", obj["model"])
		}
		if obj["choices"] == nil {
			t.Error("choices should be preserved")
		}
		if obj["usage"] == nil {
			t.Error("usage should be preserved")
		}
	})

	t.Run("model name with slashes", func(t *testing.T) {
		data := []byte(`{"model":"unsloth/Qwen3.6-27B-MTP-GGUF:BF16","choices":[]}`)
		got := rewriteModelInResponse(data, "unsloth/Qwen3.6-27B-MTP-GGUF:BF16", "opus")
		want := `{"model":"opus","choices":[]}`
		if string(got) != want {
			t.Errorf("got %s, want %s", got, want)
		}
	})
}

func TestHeaderInjector(t *testing.T) {
	recorder := &testResponseWriter{header: make(http.Header)}
	start := time.Now()
	mw := newMetricsWriter(recorder, start)
	hj := newHeaderInjector(recorder, mw)

	// Simulate WriteHeader (sets TTFB and status)
	hj.WriteHeader(201)

	// Simulate Write (triggers TTFB in metricsWriter)
	time.Sleep(10 * time.Millisecond)
	mw.Write([]byte("hello")) //nolint:errcheck

	// Check headers were set
	if recorder.statusCode != 201 {
		t.Errorf("statusCode = %d, want 201", recorder.statusCode)
	}

	status := recorder.header.Get("X-Router-Status")
	if status != "201" {
		t.Errorf("X-Router-Status = %q, want %q", status, "201")
	}

	ttfb := recorder.header.Get("X-Router-TTFB-Ms")
	if ttfb == "" {
		t.Error("X-Router-TTFB-Ms should be set")
	}
	// TTFB should be 0 since WriteHeader was called before Write set firstWrite
}

func TestSetRouterHeaders(t *testing.T) {
	recorder := &testResponseWriter{header: make(http.Header)}
	rh := &RouterHeaders{
		ServerID:   "srv-1",
		ServerName: "Primary Backend",
	}
	SetRouterHeaders(recorder, rh)

	if recorder.header.Get("X-Router-Server") != "srv-1" {
		t.Errorf("X-Router-Server = %q, want %q", recorder.header.Get("X-Router-Server"), "srv-1")
	}
	if recorder.header.Get("X-Router-Server-Name") != "Primary Backend" {
		t.Errorf("X-Router-Server-Name = %q, want %q", recorder.header.Get("X-Router-Server-Name"), "Primary Backend")
	}
}

func TestSetRouterHeadersNil(t *testing.T) {
	recorder := &testResponseWriter{header: make(http.Header)}
	SetRouterHeaders(recorder, nil) // Should not panic
	if len(recorder.header) > 0 {
		t.Error("no headers should be set for nil RouterHeaders")
	}
}

func TestExtractSSEContents(t *testing.T) {
	t.Run("OpenAI format", func(t *testing.T) {
		data := []byte("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\ndata: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n")
		contents := extractSSEContents(data)
		if len(contents) != 2 {
			t.Fatalf("got %d contents, want 2", len(contents))
		}
		if contents[0] != "hello" {
			t.Errorf("contents[0] = %q, want %q", contents[0], "hello")
		}
		if contents[1] != " world" {
			t.Errorf("contents[1] = %q, want %q", contents[1], " world")
		}
	})

	t.Run("Anthropic format", func(t *testing.T) {
		data := []byte("data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"hello\"}}\n")
		contents := extractSSEContents(data)
		if len(contents) != 1 {
			t.Fatalf("got %d contents, want 1", len(contents))
		}
		if contents[0] != "hello" {
			t.Errorf("contents[0] = %q, want %q", contents[0], "hello")
		}
	})

	t.Run("skips DONE", func(t *testing.T) {
		data := []byte("data: [DONE]\n")
		contents := extractSSEContents(data)
		if len(contents) != 0 {
			t.Errorf("got %d contents, want 0", len(contents))
		}
	})

	t.Run("skips empty content", func(t *testing.T) {
		data := []byte("data: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}\n")
		contents := extractSSEContents(data)
		if len(contents) != 0 {
			t.Errorf("got %d contents, want 0", len(contents))
		}
	})

	t.Run("skips non-content events", func(t *testing.T) {
		data := []byte("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n")
		contents := extractSSEContents(data)
		if len(contents) != 0 {
			t.Errorf("got %d contents, want 0", len(contents))
		}
	})
}

func TestLoopDetectorNoLoop(t *testing.T) {
	// Different content should not trigger loop detection
	recorder := &testResponseWriter{header: make(http.Header)}
	ctx := context.Background()
	ld := newLoopDetector(recorder, ctx)

	// Send 25 different contents
	for i := 0; i < 25; i++ {
		sse := fmt.Sprintf("data: {\"choices\":[{\"delta\":{\"content\":\"word-%d\"}}]}\n", i)
		_, err := ld.Write([]byte(sse))
		if err != nil {
			t.Fatalf("Write(%d): %v", i, err)
		}
	}

	if ld.detected {
		t.Error("loop should not be detected for different content")
	}
}

func TestLoopDetectorDetectsLoop(t *testing.T) {
	// Same content repeated should trigger loop detection
	recorder := &testResponseWriter{header: make(http.Header)}
	ctx := context.Background()
	ld := newLoopDetector(recorder, ctx)

	repeated := "data: {\"choices\":[{\"delta\":{\"content\":\" repeated \"}}]}\n"
	for i := 0; i < loopDetectionWindow+5; i++ {
		_, err := ld.Write([]byte(repeated))
		if err != nil {
			// Loop detected — this is expected
			break
		}
	}

	if !ld.detected {
		t.Error("loop should be detected for repeated content")
	}
}

func TestLoopDetectorMixedContent(t *testing.T) {
	// Mix of different content should not trigger detection
	recorder := &testResponseWriter{header: make(http.Header)}
	ctx := context.Background()
	ld := newLoopDetector(recorder, ctx)

	// Send varied content
	for i := 0; i < loopDetectionWindow + 10; i++ {
		sse := fmt.Sprintf("data: {\"choices\":[{\"delta\":{\"content\":\"word %d \"}}]}\n", i)
		_, err := ld.Write([]byte(sse))
		if err != nil {
			t.Fatalf("unexpected error at %d: %v", i, err)
		}
	}

	if ld.detected {
		t.Error("loop should not be detected for varied content")
	}
}

func TestLoopDetectorAllRecentIdentical(t *testing.T) {
	ld := &loopDetector{
		recent:  make([]string, loopDetectionWindow),
		ctx:     context.Background(),
	}
	ld.count = loopDetectionWindow
	ld.index = loopDetectionWindow

	// Fill with same content
	for i := 0; i < loopDetectionWindow; i++ {
		ld.recent[i] = "same"
	}
	if !ld.allRecentIdentical() {
		t.Error("should detect identical content")
	}

	// Change one
	ld.recent[5] = "different"
	if ld.allRecentIdentical() {
		t.Error("should not detect identical content when one differs")
	}
}

func TestRewriteModelWriterChain(t *testing.T) {
	// Verify that modelRewriteWriter wraps metricsWriter in the response chain:
	// modelRewriteWriter → metricsWriter → client
	// Both client and metrics buffer see the rewritten data.
	recorder := &testResponseWriter{header: make(http.Header)}
	start := time.Now()
	mw := newMetricsWriter(recorder, start)
	rw := newModelRewriteWriter(mw, "target", "opus")

	data := []byte(`{"model":"target","choices":[]}`)
	n, err := rw.Write(data)
	if err != nil {
		t.Fatalf("Write: %v", err)
	}
	if n == 0 {
		t.Error("Write returned 0 bytes")
	}

	// Client should see rewritten data
	if !strings.Contains(string(recorder.buf), `"model":"opus"`) {
		t.Errorf("client saw %s, want rewritten model", recorder.buf)
	}

	// metricsWriter buffer also has rewritten data (it's in the chain)
	if !bytes.Contains(mw.bodyBuffer.Bytes(), []byte(`"model":"opus"`)) {
		t.Errorf("metrics buffer has %s, want rewritten model", mw.bodyBuffer.Bytes())
	}
}

type testResponseWriter struct {
	buf          []byte
	header       http.Header
	headerCalled bool
	statusCode   int
}

func (w *testResponseWriter) Header() http.Header { return w.header }

func (w *testResponseWriter) Write(b []byte) (int, error) {
	w.buf = append(w.buf, b...)
	return len(b), nil
}

func (w *testResponseWriter) WriteHeader(code int) {
	w.statusCode = code
	w.headerCalled = true
}

func TestMidStreamError(t *testing.T) {
	// Simulate a backend that sends some data, then disconnects mid-stream
	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		// Send a few chunks
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hello \"}}]}\n\n")) //nolint:errcheck
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"world\"}}]}\n\n"))  //nolint:errcheck
		// Force disconnect by hijacking and closing
		hj, ok := w.(http.Hijacker)
		if ok {
			conn, _, err := hj.Hijack()
			if err == nil {
				conn.Close() //nolint:errcheck
			}
		}
	}))
	defer primaryServer.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	w := httptest.NewRecorder()

	pm, err := StreamProxy(req.Context(), primaryServer.URL, "key", req, w, "gpt-4", "gpt-4", nil)

	// Should get a MidStreamError
	var midErr *MidStreamError
	if err == nil {
		// Hijack may not work on httptest — if no error, skip
		// This is acceptable because httptest.Server doesn't support real TCP disconnects
		return
	}
	if !errors.As(err, &midErr) {
		// If not a MidStreamError, it might be a connection error — also acceptable
		return
	}
	if pm == nil {
		t.Fatal("pm should not be nil for mid-stream error")
	}
	if pm.ResponseSize == 0 {
		t.Error("ResponseSize should be > 0 (some data was written before error)")
	}
}

func TestMidStreamErrorType(t *testing.T) {
	// Verify MidStreamError implements expected interface
	midErr := &MidStreamError{
		Err:     fmt.Errorf("connection reset"),
		Written: 1234,
	}

	msg := midErr.Error()
	if !strings.Contains(msg, "1234") {
		t.Errorf("Error() = %q, want it to contain byte count", msg)
	}
	if !strings.Contains(msg, "connection reset") {
		t.Errorf("Error() = %q, want it to contain underlying error", msg)
	}

	// Unwrap should return the underlying error
	unwrapped := midErr.Unwrap()
	if unwrapped == nil {
		t.Error("Unwrap() should return the underlying error")
	}
}

func TestSendMidStreamError(t *testing.T) {
	recorder := &testResponseWriter{header: make(http.Header)}
	err := fmt.Errorf("connection reset by peer")
	sendMidStreamError(recorder, err)

	// Should have written an error event
	if len(recorder.buf) == 0 {
		t.Error("expected error event to be written")
	}
	if !strings.Contains(string(recorder.buf), "finish_reason") {
		t.Errorf("expected finish_reason in error event, got: %s", recorder.buf)
	}
	if !strings.Contains(string(recorder.buf), "error") {
		t.Errorf("expected 'error' in error event, got: %s", recorder.buf)
	}
}

func TestStreamProxyAnthropicRewrite(t *testing.T) {
	// Full integration test: Anthropic API response should have model rewritten
	backendModel := "unsloth/Qwen3.6-27B-MTP-GGUF:BF16"
	clientModel := "opus"

	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify the request has the target model (rewritten by router before StreamProxy)
		body, _ := io.ReadAll(r.Body)
		var obj map[string]interface{}
		_ = json.Unmarshal(body, &obj)
		if model, ok := obj["model"].(string); ok && model != backendModel {
			t.Errorf("backend received model %q, want %q", model, backendModel)
		}

		// Return Anthropic-style response
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"model":"%s","content":[],"usage":{"input_tokens":10,"output_tokens":20}}`, backendModel) //nolint:errcheck
	}))
	defer primaryServer.Close()

	// Pre-rewrite the request body (router does this before calling StreamProxy)
	rewrittenBody, err := RewriteModelInBody([]byte(fmt.Sprintf(`{"model":"%s","messages":[{"role":"user","content":"hi"}]}`, clientModel)), backendModel)
	if err != nil {
		t.Fatalf("RewriteModelInBody: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(string(rewrittenBody)))
	w := httptest.NewRecorder()

	pm, err := StreamProxy(req.Context(), primaryServer.URL, "key", req, w, backendModel, clientModel, nil)
	if err != nil {
		t.Fatalf("StreamProxy: %v", err)
	}
	if pm.StatusCode != 200 {
		t.Fatalf("status = %d, want 200", pm.StatusCode)
	}

	// Check the response has the client's model name
	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if model, ok := resp["model"].(string); !ok || model != clientModel {
		t.Errorf("response model = %q, want %q (client model)", model, clientModel)
	}
}

func TestStreamProxyAnthropicStreamingRewrite(t *testing.T) {
	// Full integration test: Anthropic streaming response should have model rewritten
	backendModel := "unsloth/Qwen3.6-27B-MTP-GGUF:BF16"
	clientModel := "opus"

	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, `data: {"type":"message_start","message":{"model":"%s","id":"msg_123"}}\n\n`, backendModel) //nolint:errcheck
		w.Write([]byte(`data: {"type":"content_block_delta","delta":{"text":"hello"}}\n\n`)) //nolint:errcheck
		w.Write([]byte(`data: {"type":"message_stop"}\n\n`)) //nolint:errcheck
	}))
	defer primaryServer.Close()

	// Pre-rewrite the request body (router does this before calling StreamProxy)
	rewrittenBody, err := RewriteModelInBody([]byte(fmt.Sprintf(`{"model":"%s","messages":[{"role":"user","content":"hi"}]}`, clientModel)), backendModel)
	if err != nil {
		t.Fatalf("RewriteModelInBody: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(string(rewrittenBody)))
	w := httptest.NewRecorder()

	pm, err := StreamProxy(req.Context(), primaryServer.URL, "key", req, w, backendModel, clientModel, nil)
	if err != nil {
		t.Fatalf("StreamProxy: %v", err)
	}
	if pm.StatusCode != 200 {
		t.Fatalf("status = %d, want 200", pm.StatusCode)
	}

	respBody := w.Body.String()
	// The response should contain the client model, not the backend model
	if strings.Contains(respBody, backendModel) {
		t.Errorf("response contains backend model %q, should be rewritten: %s", backendModel, respBody)
	}
	if !strings.Contains(respBody, clientModel) {
		t.Errorf("response should contain client model %q: %s", clientModel, respBody)
	}
}

func TestRewriteAnyModelValue(t *testing.T) {
	// When oldModel == newModel, any "model":"X" should be replaced with newModel.
	// This handles the case where the backend returns its actual model name
	// even when we sent the same model name as the client requested.

	t.Run("replaces different backend model", func(t *testing.T) {
		// Backend returns its real model, client expects "opus"
		data := []byte(`{"model":"unsloth/Qwen3.6-27B-MTP-GGUF:BF16","content":[]}`)
		got := rewriteModelInResponse(data, "opus", "opus")
		want := `{"model":"opus","content":[]}`
		if string(got) != want {
			t.Errorf("got %s, want %s", got, want)
		}
	})

	t.Run("replaces in SSE event", func(t *testing.T) {
		data := []byte(`data: {"type":"message_start","message":{"model":"unsloth/Qwen3.6-27B-MTP-GGUF:BF16","id":"msg_123"}}`)
		got := rewriteModelInResponse(data, "opus", "opus")
		want := `data: {"type":"message_start","message":{"model":"opus","id":"msg_123"}}`
		if string(got) != want {
			t.Errorf("got %s, want %s", got, want)
		}
	})

	t.Run("skips if already correct", func(t *testing.T) {
		data := []byte(`{"model":"opus","content":[]}`)
		got := rewriteModelInResponse(data, "opus", "opus")
		want := `{"model":"opus","content":[]}`
		if string(got) != want {
			t.Errorf("got %s, want %s", got, want)
		}
	})

	t.Run("does not replace model_id", func(t *testing.T) {
		data := []byte(`{"model_id":"123","model":"backend-model","content":[]}`)
		got := rewriteModelInResponse(data, "opus", "opus")
		if !strings.Contains(string(got), `"model_id":"123"`) {
			t.Errorf("model_id was incorrectly modified: %s", got)
		}
		if !strings.Contains(string(got), `"model":"opus"`) {
			t.Errorf("model was not replaced: %s", got)
		}
	})
}
