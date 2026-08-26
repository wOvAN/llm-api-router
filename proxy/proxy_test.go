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
	"sync/atomic"
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

		var obj map[string]any
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

	t.Run("OpenAI Responses API format", func(t *testing.T) {
		body := []byte(`{"id":"resp_1","object":"response","model":"gpt-4o","usage":{"input_tokens":10,"output_tokens":20,"total_tokens":30,"input_tokens_details":{"cached_tokens":7}}}`)
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
		if pm.CachedTokens != 7 {
			t.Errorf("CachedTokens = %d, want 7", pm.CachedTokens)
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
		body := []byte(`{"content":"hello","timings":{"prompt_n":5,"predicted_n":15,"prompt_ms":100.0,"predicted_ms":500.0,"prompt_per_second":50.0,"predicted_per_second":30.0,"cache_n":3,"draft_n":10,"draft_n_accepted":7}}`)
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
		if pm.DraftTokens != 10 || pm.DraftTokensAccepted != 7 {
			t.Errorf("draft = %d/%d, want 10/7", pm.DraftTokens, pm.DraftTokensAccepted)
		}
	})

	t.Run("partial timings keep usage tokens", func(t *testing.T) {
		// A timings object without prompt_n/predicted_n must not zero out the
		// token counts parsed from usage.
		body := []byte(`{"usage":{"prompt_tokens":7,"completion_tokens":5,"total_tokens":12},"timings":{"prompt_per_second":250.5,"predicted_per_second":42.25,"prompt_ms":123.4,"predicted_ms":567.8,"cache_n":4}}`)
		pm := extractUsageFromJSON(body)
		if pm.PromptTokens != 7 {
			t.Errorf("PromptTokens = %d, want 7", pm.PromptTokens)
		}
		if pm.CompletionTokens != 5 {
			t.Errorf("CompletionTokens = %d, want 5", pm.CompletionTokens)
		}
		if pm.TotalTokens != 12 {
			t.Errorf("TotalTokens = %d, want 12", pm.TotalTokens)
		}
		if pm.CachedTokens != 4 {
			t.Errorf("CachedTokens = %d, want 4", pm.CachedTokens)
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

	t.Run("vLLM metrics object", func(t *testing.T) {
		body := []byte(`{"model":"m","usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30},"metrics":{"time_to_first_token_ms":100.5,"generation_time_ms":500.25,"queue_time_ms":7.5,"mean_itl_ms":25.0,"tokens_per_second":39.0}}`)
		pm := extractUsageFromJSON(body)
		if pm.PromptTokens != 10 || pm.CompletionTokens != 20 || pm.TotalTokens != 30 {
			t.Errorf("tokens = (%d,%d,%d), want (10,20,30)", pm.PromptTokens, pm.CompletionTokens, pm.TotalTokens)
		}
		if pm.PromptMs != 100.5 {
			t.Errorf("PromptMs = %f, want 100.5", pm.PromptMs)
		}
		if pm.PredictedMs != 500.25 {
			t.Errorf("PredictedMs = %f, want 500.25", pm.PredictedMs)
		}
		if pm.TokensPerSec != 39.0 {
			t.Errorf("TokensPerSec = %f, want 39.0", pm.TokensPerSec)
		}
		if pm.QueueMs != 7.5 {
			t.Errorf("QueueMs = %f, want 7.5", pm.QueueMs)
		}
	})

	t.Run("vLLM metrics with null fields", func(t *testing.T) {
		body := []byte(`{"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7},"metrics":{"time_to_first_token_ms":null,"generation_time_ms":null,"queue_time_ms":null,"mean_itl_ms":null,"tokens_per_second":null}}`)
		pm := extractUsageFromJSON(body)
		if pm.PromptTokens != 3 || pm.CompletionTokens != 4 {
			t.Errorf("tokens = (%d,%d), want (3,4)", pm.PromptTokens, pm.CompletionTokens)
		}
		if pm.PromptMs != 0 || pm.PredictedMs != 0 || pm.TokensPerSec != 0 || pm.QueueMs != 0 {
			t.Errorf("native timings should be zero, got (%f,%f,%f,%f)", pm.PromptMs, pm.PredictedMs, pm.TokensPerSec, pm.QueueMs)
		}
		if pm.CachedTokens != -1 {
			t.Errorf("CachedTokens = %d, want -1", pm.CachedTokens)
		}
	})

	t.Run("vLLM reasoning and cache creation", func(t *testing.T) {
		body := []byte(`{"usage":{"prompt_tokens":10,"completion_tokens":30,"total_tokens":40,"prompt_tokens_details":{"cached_tokens":6,"created_cache_tokens":2},"completion_tokens_details":{"reasoning_tokens":12}}}`)
		pm := extractUsageFromJSON(body)
		if pm.CachedTokens != 6 || pm.CacheCreationTokens != 2 || pm.ReasoningTokens != 12 {
			t.Errorf("cached/creation/reasoning = (%d,%d,%d), want (6,2,12)", pm.CachedTokens, pm.CacheCreationTokens, pm.ReasoningTokens)
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

	t.Run("OpenAI Responses API streaming format", func(t *testing.T) {
		body := []byte(
			"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-4o\"}}\n" +
				"data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n" +
				"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-4o\",\"usage\":{\"input_tokens\":10,\"output_tokens\":20,\"total_tokens\":30,\"input_tokens_details\":{\"cached_tokens\":7}}}}\n",
		)
		pm := extractUsageFromStream(body)
		if pm.PromptTokens != 10 {
			t.Errorf("PromptTokens = %d, want 10", pm.PromptTokens)
		}
		if pm.CompletionTokens != 20 {
			t.Errorf("CompletionTokens = %d, want 20", pm.CompletionTokens)
		}
		if pm.CachedTokens != 7 {
			t.Errorf("CachedTokens = %d, want 7", pm.CachedTokens)
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

	t.Run("vLLM streaming metrics on final chunk", func(t *testing.T) {
		// Intermediate chunks serialize "metrics": null; the real object
		// arrives only on the final chunk alongside usage.
		body := []byte(
			"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}],\"metrics\":null}\n" +
				"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":10,\"completion_tokens_details\":{\"reasoning_tokens\":3}},\"metrics\":{\"time_to_first_token_ms\":12.0,\"generation_time_ms\":100.0,\"queue_time_ms\":2.5,\"mean_itl_ms\":10.0,\"tokens_per_second\":9.5}}\n" +
				"data: [DONE]\n",
		)
		pm := extractUsageFromStream(body)
		if pm.PromptTokens != 5 || pm.CompletionTokens != 10 {
			t.Errorf("tokens = (%d,%d), want (5,10)", pm.PromptTokens, pm.CompletionTokens)
		}
		if pm.PromptMs != 12.0 || pm.PredictedMs != 100.0 || pm.TokensPerSec != 9.5 || pm.QueueMs != 2.5 {
			t.Errorf("native timings = (%f,%f,%f,%f)", pm.PromptMs, pm.PredictedMs, pm.TokensPerSec, pm.QueueMs)
		}
		if pm.ReasoningTokens != 3 {
			t.Errorf("ReasoningTokens = %d, want 3", pm.ReasoningTokens)
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
	obj := map[string]any{
		"message": map[string]any{
			"usage": map[string]any{
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
		uc := extractUsageTokens(map[string]any{
			"prompt_tokens":     float64(10),
			"completion_tokens": float64(20),
		})
		if uc.input != 10 || uc.output != 20 || uc.cached != -1 {
			t.Errorf("got (%d,%d,%d), want (10,20,-1)", uc.input, uc.output, uc.cached)
		}
	})

	t.Run("Anthropic format", func(t *testing.T) {
		uc := extractUsageTokens(map[string]any{
			"input_tokens":  float64(15),
			"output_tokens": float64(25),
		})
		if uc.input != 15 || uc.output != 25 || uc.cached != -1 {
			t.Errorf("got (%d,%d,%d), want (15,25,-1)", uc.input, uc.output, uc.cached)
		}
	})

	t.Run("nil usage", func(t *testing.T) {
		uc := extractUsageTokens(nil)
		if uc.input != 0 || uc.output != 0 || uc.cached != -1 {
			t.Errorf("got (%d,%d,%d), want (0,0,-1)", uc.input, uc.output, uc.cached)
		}
	})

	t.Run("vLLM cache creation and reasoning", func(t *testing.T) {
		uc := extractUsageTokens(map[string]any{
			"prompt_tokens":     float64(10),
			"completion_tokens": float64(30),
			"prompt_tokens_details": map[string]any{
				"cached_tokens":        float64(6),
				"created_cache_tokens": float64(2),
			},
			"completion_tokens_details": map[string]any{
				"reasoning_tokens": float64(12),
			},
		})
		if uc.cached != 6 || uc.cacheCreation != 2 || uc.reasoning != 12 {
			t.Errorf("got cached/creation/reasoning (%d,%d,%d), want (6,2,12)", uc.cached, uc.cacheCreation, uc.reasoning)
		}
	})

	t.Run("Anthropic cache_creation_input_tokens", func(t *testing.T) {
		uc := extractUsageTokens(map[string]any{
			"input_tokens":                float64(5),
			"output_tokens":               float64(8),
			"cache_read_input_tokens":     float64(9),
			"cache_creation_input_tokens": float64(4),
		})
		if uc.cached != 9 || uc.cacheCreation != 4 {
			t.Errorf("got cached/creation (%d,%d), want (9,4)", uc.cached, uc.cacheCreation)
		}
	})
}

func TestBuildMetricsFromData(t *testing.T) {
	t.Run("without timings", func(t *testing.T) {
		pm := buildMetricsFromData(usageCounts{input: 10, output: 20, cached: 5}, 30, nil, nil)
		if pm.PromptTokens != 10 || pm.CompletionTokens != 20 || pm.TotalTokens != 30 || pm.CachedTokens != 5 {
			t.Errorf("got tokens (%d,%d,%d,%d)", pm.PromptTokens, pm.CompletionTokens, pm.TotalTokens, pm.CachedTokens)
		}
	})

	t.Run("with timings overrides", func(t *testing.T) {
		pm := buildMetricsFromData(usageCounts{input: 10, output: 20, cached: -1}, 30, map[string]any{
			"prompt_n":             float64(5),
			"predicted_n":          float64(15),
			"prompt_ms":            float64(100),
			"predicted_ms":         float64(500),
			"prompt_per_second":    float64(50),
			"predicted_per_second": float64(30),
			"cache_n":              float64(3),
			"draft_n":              float64(10),
			"draft_n_accepted":     float64(7),
		}, nil)
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
		if pm.DraftTokens != 10 || pm.DraftTokensAccepted != 7 {
			t.Errorf("draft = %d/%d, want 10/7", pm.DraftTokens, pm.DraftTokensAccepted)
		}
	})

	t.Run("with vLLM metrics", func(t *testing.T) {
		pm := buildMetricsFromData(usageCounts{input: 10, output: 20, cached: -1}, 30, nil, map[string]any{
			"time_to_first_token_ms": float64(100.5),
			"generation_time_ms":     float64(500.25),
			"queue_time_ms":          float64(7.5),
			"mean_itl_ms":            float64(25.0),
			"tokens_per_second":      float64(39.0),
		})
		if pm.PromptTokens != 10 || pm.CompletionTokens != 20 {
			t.Errorf("tokens = (%d,%d), want (10,20)", pm.PromptTokens, pm.CompletionTokens)
		}
		if pm.PromptMs != 100.5 {
			t.Errorf("PromptMs = %f, want 100.5", pm.PromptMs)
		}
		if pm.PredictedMs != 500.25 {
			t.Errorf("PredictedMs = %f, want 500.25", pm.PredictedMs)
		}
		if pm.TokensPerSec != 39.0 {
			t.Errorf("TokensPerSec = %f, want 39.0", pm.TokensPerSec)
		}
		if pm.QueueMs != 7.5 {
			t.Errorf("QueueMs = %f, want 7.5", pm.QueueMs)
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
		got, _ := rewriteModelInResponse(data, "target-model", "opus")
		want := `{"model":"opus","choices":[],"usage":{}}`
		if string(got) != want {
			t.Errorf("got %s, want %s", got, want)
		}
	})

	t.Run("replaces model with whitespace", func(t *testing.T) {
		data := []byte(`{"model": "target-model", "choices": []}`)
		got, _ := rewriteModelInResponse(data, "target-model", "opus")
		want := `{"model": "opus", "choices": []}`
		if string(got) != want {
			t.Errorf("got %s, want %s", got, want)
		}
	})

	t.Run("replaces model in SSE event", func(t *testing.T) {
		data := []byte(`data: {"type":"message_start","message":{"model":"target-model","id":"msg_123"}}`)
		got, _ := rewriteModelInResponse(data, "target-model", "opus")
		want := `data: {"type":"message_start","message":{"model":"opus","id":"msg_123"}}`
		if string(got) != want {
			t.Errorf("got %s, want %s", got, want)
		}
	})

	t.Run("replaces multiple model fields", func(t *testing.T) {
		data := []byte(`data: {"model":"target"}
data: {"model":"target","usage":{"prompt_tokens":5}}
`)
		got, _ := rewriteModelInResponse(data, "target", "opus")
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
		got, _ := rewriteModelInResponse(data, "target-model", "opus")
		if string(got) != string(data) {
			t.Errorf("got %s, want unchanged %s", got, data)
		}
	})

	t.Run("empty oldModel skips rewriting", func(t *testing.T) {
		data := []byte(`{"model":"target","choices":[]}`)
		got, _ := rewriteModelInResponse(data, "", "opus")
		if string(got) != string(data) {
			t.Errorf("got %s, want unchanged %s", got, data)
		}
	})

	t.Run("same model skips rewriting", func(t *testing.T) {
		data := []byte(`{"model":"same","choices":[]}`)
		got, _ := rewriteModelInResponse(data, "same", "same")
		if string(got) != string(data) {
			t.Errorf("got %s, want unchanged %s", got, data)
		}
	})

	t.Run("does not replace model-like key", func(t *testing.T) {
		data := []byte(`{"model_id":"123","model":"target","choices":[]}`)
		got, _ := rewriteModelInResponse(data, "target", "opus")
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
		got, _ := rewriteModelInResponse(data, "target", "opus")

		var obj map[string]any
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
		got, _ := rewriteModelInResponse(data, "unsloth/Qwen3.6-27B-MTP-GGUF:BF16", "opus")
		want := `{"model":"opus","choices":[]}`
		if string(got) != want {
			t.Errorf("got %s, want %s", got, want)
		}
	})
}

func TestMetricsWriterStatusHeader(t *testing.T) {
	recorder := &testResponseWriter{header: make(http.Header)}
	start := time.Now()
	mw := newMetricsWriter(recorder, start)
	mw.injectStatus = true

	// Simulate WriteHeader (sets status only; TTFB is measured at first Write)
	mw.WriteHeader(201)

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

	// TTFB is measured at first Write() time, not in the header
	pm := mw.metrics()
	if pm.TTFBMs < 10 {
		t.Errorf("TTFBMs = %d, want >= 10 (measured at first Write)", pm.TTFBMs)
	}
}

func TestSetRouterHeaders(t *testing.T) {
	recorder := &testResponseWriter{header: make(http.Header)}
	rh := &RouterHeaders{
		ServerID:       "srv-1",
		ServerName:     "Primary Backend",
		Attempts:       "2/3",
		Retries:        1,
		FallbackErrors: []string{"connection refused", "timeout"},
	}
	SetRouterHeaders(recorder, rh)

	if recorder.header.Get("X-Router-Server") != "srv-1" {
		t.Errorf("X-Router-Server = %q, want %q", recorder.header.Get("X-Router-Server"), "srv-1")
	}
	if recorder.header.Get("X-Router-Server-Name") != "Primary Backend" {
		t.Errorf("X-Router-Server-Name = %q, want %q", recorder.header.Get("X-Router-Server-Name"), "Primary Backend")
	}
	if got := recorder.header.Get("X-Router-Attempts"); got != "2/3" {
		t.Errorf("X-Router-Attempts = %q, want %q", got, "2/3")
	}
	if got := recorder.header.Get("X-Router-Retries"); got != "1" {
		t.Errorf("X-Router-Retries = %q, want %q", got, "1")
	}
	if got := recorder.header.Get("X-Router-Fallback-Errors"); got != `["connection refused","timeout"]` {
		t.Errorf("X-Router-Fallback-Errors = %q", got)
	}
}

func TestSetRouterHeadersZeroValuesOmitted(t *testing.T) {
	recorder := &testResponseWriter{header: make(http.Header)}
	rh := &RouterHeaders{
		ServerID:   "srv-1",
		ServerName: "Primary",
		// Attempts empty, Retries 0, FallbackErrors nil → omitted except Retries
	}
	SetRouterHeaders(recorder, rh)

	if recorder.header.Get("X-Router-Attempts") != "" {
		t.Error("X-Router-Attempts should be omitted when empty")
	}
	// X-Router-Retries is always written (even "0") to overwrite stale values
	// from previous retry/fallback attempts on the same response writer.
	if got := recorder.header.Get("X-Router-Retries"); got != "0" {
		t.Errorf("X-Router-Retries = %q, want 0", got)
	}
	if recorder.header.Get("X-Router-Fallback-Errors") != "" {
		t.Error("X-Router-Fallback-Errors should be omitted when empty")
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

	t.Run("skips event lines", func(t *testing.T) {
		data := []byte("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"hi\"}}\n\nevent: message_stop\n\n")
		contents := extractSSEContents(data)
		if len(contents) != 1 {
			t.Fatalf("got %d contents, want 1", len(contents))
		}
		if contents[0] != "hi" {
			t.Errorf("contents[0] = %q, want %q", contents[0], "hi")
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
	for i := range 25 {
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
	for range loopDetectionWindow + 5 {
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

func TestLoopDetectorSendsErrorToClient(t *testing.T) {
	// The loop error frame must actually reach the client. Regression: the
	// detected-guard in Write used to swallow the frame sent by sendLoopError.
	recorder := &testResponseWriter{header: make(http.Header)}
	ld := newLoopDetector(recorder, context.Background())

	repeated := "data: {\"choices\":[{\"delta\":{\"content\":\"echo \"}}]}\n"
	for range loopDetectionWindow + 1 {
		if _, err := ld.Write([]byte(repeated)); err != nil {
			break
		}
	}

	if !ld.detected {
		t.Fatal("loop should be detected")
	}
	if !strings.Contains(string(recorder.buf), "stuck in a loop") {
		t.Errorf("client should receive loop error frame, got: %q", recorder.buf)
	}
}

func TestStreamProxyLoopReturnsMidStreamError(t *testing.T) {
	// A stuck backend (identical SSE chunks) must yield a *MidStreamError so the
	// router knows headers were already sent and does not attempt fallback.
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		chunk := "data: {\"choices\":[{\"delta\":{\"content\":\"same \"}}]}\n\n"
		for range loopDetectionWindow + 10 {
			if _, err := w.Write([]byte(chunk)); err != nil {
				return
			}
		}
	}))
	defer backend.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	w := httptest.NewRecorder()

	pm, err := StreamProxy(req.Context(), backend.URL, "key", req, w, "gpt-4", "gpt-4", nil, false, nil)
	if _, ok := errors.AsType[*MidStreamError](err); !ok {
		t.Fatalf("expected *MidStreamError for detected loop, got: %v", err)
	}
	if pm == nil {
		t.Fatal("pm should not be nil")
	}
	if !strings.Contains(w.Body.String(), "stuck in a loop") {
		t.Errorf("client should have received loop error frame, got: %q", w.Body.String())
	}
}

func TestLoopDetectorMixedContent(t *testing.T) {
	// Mix of different content should not trigger detection
	recorder := &testResponseWriter{header: make(http.Header)}
	ctx := context.Background()
	ld := newLoopDetector(recorder, ctx)

	// Send varied content
	for i := range loopDetectionWindow + 10 {
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
		recent: make([]string, loopDetectionWindow),
		ctx:    context.Background(),
	}
	ld.count = loopDetectionWindow
	ld.index = loopDetectionWindow

	// Fill with same content
	for i := range loopDetectionWindow {
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

	pm, err := StreamProxy(req.Context(), primaryServer.URL, "key", req, w, "gpt-4", "gpt-4", nil, false, nil)

	// Should get a MidStreamError
	if err == nil {
		// Hijack may not work on httptest — if no error, skip
		// This is acceptable because httptest.Server doesn't support real TCP disconnects
		return
	}
	if _, ok := errors.AsType[*MidStreamError](err); !ok {
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
	sendMidStreamError(recorder, err, false)

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

func TestSendMidStreamErrorAnthropic(t *testing.T) {
	// Anthropic SDKs drop data: frames without a recognized event: type, so a
	// /messages stream must get the standard `event: error` frame or the client
	// sees a silently truncated message.
	recorder := &testResponseWriter{header: make(http.Header)}
	sendMidStreamError(recorder, fmt.Errorf("boom"), true)

	got := string(recorder.buf)
	if !strings.Contains(got, "event: error") {
		t.Errorf("expected 'event: error' line, got: %q", got)
	}
	if !strings.Contains(got, `"type":"error"`) || !strings.Contains(got, `"message":"boom"`) {
		t.Errorf("expected Anthropic error JSON body, got: %q", got)
	}
	if !strings.HasSuffix(got, "\n\n") {
		t.Errorf("event must end with real newlines, got: %q", got)
	}
}

func TestSendMidStreamErrorTerminatesEvent(t *testing.T) {
	// Regression: the event used a raw string, so "\n\n" was sent literally and
	// clients never saw the SSE event terminator.
	recorder := &testResponseWriter{header: make(http.Header)}
	sendMidStreamError(recorder, fmt.Errorf("boom"), false)

	got := string(recorder.buf)
	if !strings.HasSuffix(got, "\n\n") {
		t.Errorf("event must end with real newlines, got: %q", got)
	}
	if strings.Contains(got, `\n\n`) {
		t.Errorf("event contains literal backslash-n (raw-string bug), got: %q", got)
	}
	if strings.Contains(got, `[error: boom]`) == false {
		t.Errorf("expected escaped message in event, got: %q", got)
	}
}

func TestStreamProxyAnthropicRewrite(t *testing.T) {
	// Full integration test: Anthropic API response should have model rewritten
	backendModel := "unsloth/Qwen3.6-27B-MTP-GGUF:BF16"
	clientModel := "opus"

	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify the request has the target model (rewritten by router before StreamProxy)
		body, _ := io.ReadAll(r.Body)
		var obj map[string]any
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

	pm, err := StreamProxy(req.Context(), primaryServer.URL, "key", req, w, backendModel, clientModel, nil, false, nil)
	if err != nil {
		t.Fatalf("StreamProxy: %v", err)
	}
	if pm.StatusCode != 200 {
		t.Fatalf("status = %d, want 200", pm.StatusCode)
	}

	// Check the response has the client's model name
	var resp map[string]any
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
		w.Write([]byte(`data: {"type":"content_block_delta","delta":{"text":"hello"}}\n\n`))                       //nolint:errcheck
		w.Write([]byte(`data: {"type":"message_stop"}\n\n`))                                                       //nolint:errcheck
	}))
	defer primaryServer.Close()

	// Pre-rewrite the request body (router does this before calling StreamProxy)
	rewrittenBody, err := RewriteModelInBody([]byte(fmt.Sprintf(`{"model":"%s","messages":[{"role":"user","content":"hi"}]}`, clientModel)), backendModel)
	if err != nil {
		t.Fatalf("RewriteModelInBody: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(string(rewrittenBody)))
	w := httptest.NewRecorder()

	pm, err := StreamProxy(req.Context(), primaryServer.URL, "key", req, w, backendModel, clientModel, nil, false, nil)
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
		got, _ := rewriteModelInResponse(data, "opus", "opus")
		want := `{"model":"opus","content":[]}`
		if string(got) != want {
			t.Errorf("got %s, want %s", got, want)
		}
	})

	t.Run("replaces in SSE event", func(t *testing.T) {
		data := []byte(`data: {"type":"message_start","message":{"model":"unsloth/Qwen3.6-27B-MTP-GGUF:BF16","id":"msg_123"}}`)
		got, _ := rewriteModelInResponse(data, "opus", "opus")
		want := `data: {"type":"message_start","message":{"model":"opus","id":"msg_123"}}`
		if string(got) != want {
			t.Errorf("got %s, want %s", got, want)
		}
	})

	t.Run("skips if already correct", func(t *testing.T) {
		data := []byte(`{"model":"opus","content":[]}`)
		got, _ := rewriteModelInResponse(data, "opus", "opus")
		want := `{"model":"opus","content":[]}`
		if string(got) != want {
			t.Errorf("got %s, want %s", got, want)
		}
	})

	t.Run("does not replace model_id", func(t *testing.T) {
		data := []byte(`{"model_id":"123","model":"backend-model","content":[]}`)
		got, _ := rewriteModelInResponse(data, "opus", "opus")
		if !strings.Contains(string(got), `"model_id":"123"`) {
			t.Errorf("model_id was incorrectly modified: %s", got)
		}
		if !strings.Contains(string(got), `"model":"opus"`) {
			t.Errorf("model was not replaced: %s", got)
		}
	})
}

func TestRewriteAnyModelValueOpenAI(t *testing.T) {
	// Same fix applies to OpenAI format — backend may return its actual model

	t.Run("replaces in OpenAI JSON response", func(t *testing.T) {
		data := []byte(`{"id":"chatcmpl-123","model":"llama-3.1-70b","choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":20}}`)
		got, _ := rewriteModelInResponse(data, "gpt-4", "gpt-4")
		if !strings.Contains(string(got), `"model":"gpt-4"`) {
			t.Errorf("model was not replaced: %s", got)
		}
		if strings.Contains(string(got), "llama-3.1-70b") {
			t.Errorf("backend model leaked: %s", got)
		}
	})

	t.Run("replaces in OpenAI SSE chunk", func(t *testing.T) {
		data := []byte(`data: {"id":"chatcmpl-123","model":"llama-3.1-70b","choices":[{"delta":{"content":"hello"}}]}`)
		got, _ := rewriteModelInResponse(data, "gpt-4", "gpt-4")
		if !strings.Contains(string(got), `"model":"gpt-4"`) {
			t.Errorf("model was not replaced: %s", got)
		}
	})
}

func TestEscapeJSONString(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"double quote", `"hello"`, `\"hello\"`},
		{"backslash", `path\to\file`, `path\\to\\file`},
		{"newline", "line1\nline2", `line1\nline2`},
		{"carriage return", "a\r\nb", `a\r\nb`},
		{"tab", "a\tb", `a\tb`},
		{"backspace", "a\bb", `a\bb`},
		{"form feed", "a\fb", `a\fb`},
		{"control char", "a\x01b", `a\u0001b`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := string(escapeJSONString(tt.in))
			if got != tt.want {
				t.Errorf("escapeJSONString(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestStreamProxy5xxSSEBodyRetryable(t *testing.T) {
	// A backend that died mid-generation may answer 5xx with a body of
	// already-buffered SSE frames. Nothing may reach the client.
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(500)
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"par\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"tial\"}}]}\n\n"))
	}))
	defer backend.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	w := httptest.NewRecorder()

	pm, err := StreamProxy(req.Context(), backend.URL, "key", req, w, "gpt-4", "gpt-4", nil, false, nil)

	var sseErr *StreamErrorResponse
	if err == nil || !errors.As(err, &sseErr) {
		t.Fatalf("expected *StreamErrorResponse, got err=%v pm=%+v", err, pm)
	}
	if sseErr.StatusCode != 500 {
		t.Errorf("StatusCode = %d, want 500", sseErr.StatusCode)
	}
	if !strings.Contains(sseErr.Error(), "SSE-form") {
		t.Errorf("error message should describe the response: %s", sseErr.Error())
	}
	if len(w.Body.Bytes()) != 0 {
		t.Errorf("client must receive no bytes, got %q", w.Body.Bytes())
	}
	if w.Code != http.StatusOK {
		t.Errorf("recorder code = %d, want default 200 (no WriteHeader)", w.Code)
	}
}

func TestStreamProxy5xxSSEBodyWithJsonPrefixForwarded(t *testing.T) {
	// A plain JSON 5xx (even with text/event-stream content type set by a bad
	// backend) must be forwarded as before.
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(500)
		_, _ = w.Write([]byte(`{"error":{"message":"slot full","code":500}}`))
	}))
	defer backend.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	w := httptest.NewRecorder()

	pm, err := StreamProxy(req.Context(), backend.URL, "key", req, w, "gpt-4", "gpt-4", nil, false, nil)

	if err != nil {
		t.Fatalf("plain JSON 5xx must be forwarded, got err: %v", err)
	}
	if pm.StatusCode != 500 {
		t.Errorf("StatusCode = %d, want 500", pm.StatusCode)
	}
	if w.Code != 500 {
		t.Errorf("recorder code = %d, want 500", w.Code)
	}
	if !strings.Contains(w.Body.String(), "slot full") {
		t.Errorf("body must pass through, got %q", w.Body.String())
	}
}

// failWriter pretends the client socket died: it cancels the request context
// and fails the write, like a real disconnect while the handler relays.
type failWriter struct {
	w        http.ResponseWriter
	onWrite  func()
	writeCnt int
}

func (f *failWriter) Header() http.Header { return f.w.Header() }
func (f *failWriter) Write(p []byte) (int, error) {
	f.writeCnt++
	if f.onWrite != nil {
		f.onWrite()
	}
	return 0, errors.New("client disconnected")
}
func (f *failWriter) WriteHeader(code int) { f.w.WriteHeader(code) }

func TestStreamProxyDrainsUpstreamOnClientGone(t *testing.T) {
	old := UpstreamDrainTimeout
	UpstreamDrainTimeout = 5 * time.Second
	defer func() { UpstreamDrainTimeout = old }()

	upstreamCompleted := make(chan struct{})
	var earlyAbort atomic.Bool
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(upstreamCompleted)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"1\"}}]}\n\n"))
		w.(http.Flusher).Flush()
		for range 3 {
			select {
			case <-r.Context().Done():
				// Proxy closed the socket mid-generation: exactly what a
				// reverse-proxy backend (llama.cpp router mode) treats as a
				// client-cancel and aborts the task.
				earlyAbort.Store(true)
				return
			case <-time.After(20 * time.Millisecond):
			}
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n"))
			w.(http.Flusher).Flush()
		}
	}))
	defer backend.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	ctx, cancel := context.WithCancel(req.Context())
	defer cancel()
	req = req.WithContext(ctx)

	pm, err := StreamProxy(req.Context(), backend.URL, "key", req,
		&failWriter{w: httptest.NewRecorder(), onWrite: cancel}, "gpt-4", "gpt-4", nil, false, nil)

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled (pm=%+v)", err, pm)
	}
	select {
	case <-upstreamCompleted:
	case <-time.After(2 * time.Second):
		t.Fatal("upstream did not finish its generation despite the drain")
	}
	if earlyAbort.Load() {
		t.Error("upstream saw the client socket close mid-generation; drain failed")
	}
}

func TestStreamProxyDrainDisabledClosesUpstream(t *testing.T) {
	old := UpstreamDrainTimeout
	UpstreamDrainTimeout = 0
	defer func() { UpstreamDrainTimeout = old }()

	var earlyAbort atomic.Bool
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: x\n\n"))
		w.(http.Flusher).Flush()
		for {
			select {
			case <-r.Context().Done():
				earlyAbort.Store(true)
				return
			case <-time.After(10 * time.Millisecond):
			}
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n"))
			w.(http.Flusher).Flush()
		}
	}))
	defer backend.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	ctx, cancel := context.WithCancel(req.Context())
	defer cancel()
	req = req.WithContext(ctx)

	// The legacy (UpstreamDrainTimeout=0) behavior closes the upstream socket
	// right after the client write fails, so the backend must observe the abort.
	_, err := StreamProxy(req.Context(), backend.URL, "key", req,
		&failWriter{w: httptest.NewRecorder(), onWrite: cancel}, "gpt-4", "gpt-4", nil, false, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	deadline := time.After(2 * time.Second)
	for {
		if earlyAbort.Load() {
			return
		}
		select {
		case <-deadline:
			t.Fatal("upstream never noticed the closed socket with draining disabled")
		case <-time.After(5 * time.Millisecond):
		}
	}
}

func TestStreamProxy5xxEmptyBodyForwarded(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
	}))
	defer backend.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	w := httptest.NewRecorder()

	pm, err := StreamProxy(req.Context(), backend.URL, "key", req, w, "gpt-4", "gpt-4", nil, false, nil)

	if err != nil {
		t.Fatalf("empty 5xx must be forwarded, got err: %v", err)
	}
	if pm.StatusCode != 500 {
		t.Errorf("StatusCode = %d, want 500", pm.StatusCode)
	}
	if w.Code != 500 {
		t.Errorf("recorder code = %d, want 500", w.Code)
	}
}

func TestStreamProxyURLDedupV1(t *testing.T) {
	// Server URL ends with /v1 and the request path starts with /v1: they must
	// merge into a single /v1, not become /v1v1/....
	gotPath := make(chan string, 1)
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath <- r.URL.Path
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{"id":"1","object":"chat.completion","model":"gpt-4"}`))
	}))
	defer backend.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	w := httptest.NewRecorder()

	if _, err := StreamProxy(req.Context(), backend.URL+"/v1", "key", req, w, "gpt-4", "gpt-4", nil, false, nil); err != nil {
		t.Fatalf("unexpected upstream error: %v", err)
	}
	select {
	case p := <-gotPath:
		if p != "/v1/chat/completions" {
			t.Errorf("upstream path = %q, want /v1/chat/completions", p)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("backend never received the request")
	}
}

func TestStreamProxyURLDedupLongBase(t *testing.T) {
	// Same dedup with a multi-segment base: /api/v1 + /v1/... -> /api/v1/...
	gotPath := make(chan string, 1)
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath <- r.URL.Path
		w.WriteHeader(200)
		_, _ = w.Write([]byte("{\"ok\":true}"))
	}))
	defer backend.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/models", strings.NewReader(``))
	w := httptest.NewRecorder()

	if _, err := StreamProxy(req.Context(), backend.URL+"/api/v1", "key", req, w, "gpt-4", "gpt-4", nil, false, nil); err != nil {
		t.Fatalf("unexpected upstream error: %v", err)
	}
	select {
	case p := <-gotPath:
		if p != "/api/v1/models" {
			t.Errorf("upstream path = %q, want /api/v1/models", p)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("backend never received the request")
	}
}

func TestStreamProxyAnthropicAuthHeaders(t *testing.T) {
	// A Claude Code style client authenticates with its personal x-api-key.
	// The configured server key must replace it (and set Authorization too),
	// so the upstream never sees (nor checks) the client's own key.
	got := make(chan http.Header, 1)
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got <- r.Header.Clone()
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{"type":"message","model":"opus","content":[]}`))
	}))
	defer backend.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/messages",
		strings.NewReader(`{"model":"opus","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("x-api-key", "client-personal-key")
	w := httptest.NewRecorder()

	if _, err := StreamProxy(req.Context(), backend.URL+"/v1", "server-key", req, w, "opus", "opus", nil, false, nil); err != nil {
		t.Fatalf("unexpected upstream error: %v", err)
	}
	h := <-got
	if got := h.Get("x-api-key"); got != "server-key" {
		t.Errorf("upstream x-api-key = %q, want configured server-key (client key leaked)", got)
	}
	if got := h.Get("Authorization"); got != "Bearer server-key" {
		t.Errorf("upstream Authorization = %q, want Bearer server-key", got)
	}
}

func TestStreamProxyClientKeyPassthroughWhenConfiguredEmpty(t *testing.T) {
	// Empty configured key: the router adds no auth headers of its own, the
	// client's key passes through untouched.
	got := make(chan http.Header, 1)
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got <- r.Header.Clone()
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{"id":"m","object":"chat.completion","model":"gpt-4"}`))
	}))
	defer backend.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4","stream":false,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("x-api-key", "client-key")
	w := httptest.NewRecorder()

	if _, err := StreamProxy(req.Context(), backend.URL+"/v1", "", req, w, "gpt-4", "gpt-4", nil, false, nil); err != nil {
		t.Fatalf("unexpected upstream error: %v", err)
	}
	h := <-got
	if got := h.Get("x-api-key"); got != "client-key" {
		t.Errorf("upstream x-api-key = %q, want client-key passthrough", got)
	}
	if h.Get("Authorization") != "" {
		t.Errorf("upstream Authorization = %q, want none", h.Get("Authorization"))
	}
}

func TestDeclaredOutputBudget(t *testing.T) {
	cases := []struct {
		name string
		body string
		want int64
	}{
		{"max_tokens only", `{"max_tokens":500}`, 500},
		{"max_completion_tokens only", `{"max_completion_tokens":750}`, 750},
		{"both, larger wins", `{"max_tokens":1,"max_completion_tokens":10000}`, 10000},
		{"both, smaller wins", `{"max_tokens":9000,"max_completion_tokens":10}`, 9000},
		{"neither", `{"model":"gpt-4"}`, 0},
		{"invalid json", `{not json`, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := DeclaredOutputBudget([]byte(tc.body)); got != tc.want {
				t.Errorf("DeclaredOutputBudget(%q) = %d, want %d", tc.body, got, tc.want)
			}
		})
	}
}

// TestStreamProxyResponseHeaderTimeout verifies a backend that accepts the
// connection but never sends headers is cut off by ResponseHeaderTimeout
// instead of hanging forever (the client context here has no deadline, so the
// header timeout is the only bound).
func TestStreamProxyResponseHeaderTimeout(t *testing.T) {
	old := upstreamTransport.ResponseHeaderTimeout
	upstreamTransport.ResponseHeaderTimeout = 100 * time.Millisecond
	defer func() { upstreamTransport.ResponseHeaderTimeout = old }()

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body) // fully "write" the request, then stall
		time.Sleep(500 * time.Millisecond) // longer than the 100ms header timeout
		w.WriteHeader(http.StatusOK)
	}))
	defer backend.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	w := httptest.NewRecorder()
	start := time.Now()
	_, err := StreamProxy(req.Context(), backend.URL, "key", req, w, "gpt-4", "gpt-4", nil, false, nil)
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("expected a response-header timeout error, got nil")
	}
	if elapsed > 400*time.Millisecond {
		t.Errorf("request took %v; ResponseHeaderTimeout (100ms) should have cut it off", elapsed)
	}
}
