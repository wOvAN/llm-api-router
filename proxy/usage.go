package proxy

import (
	"bytes"
	"compress/flate"
	"compress/gzip"
	"encoding/json"
	"io"
	"strings"

	"llm-api-router/pkg/log"
)

// usagePaths lists JSON paths where usage can appear in SSE events.
var usagePaths = []string{"usage", "response.usage", "message.usage"}

// extractUsageFromResponse parses token usage and timings from the response body.
func extractUsageFromResponse(body []byte, contentEncoding string, isStream bool) ProxyMetrics {
	if contentEncoding != "" {
		decompressed, err := decompressBody(body, contentEncoding)
		if err != nil {
			log.Warnf("extractUsage: failed to decompress %q response body: %v", contentEncoding, err)
			return ProxyMetrics{CachedTokens: -1}
		}
		body = decompressed
	}

	if isStream {
		return extractUsageFromStream(body)
	}
	return extractUsageFromJSON(body)
}

// forEachDataLine splits body into lines and invokes fn for each non-empty
// `data:`-prefixed line, passing the trimmed JSON payload. fn returns true to
// stop iteration early.
func forEachDataLine(body []byte, fn func(jsonData []byte) bool) {
	prefix := []byte("data:")
	for offset := 0; offset < len(body); {
		nl := bytes.IndexByte(body[offset:], '\n')
		var line []byte
		if nl == -1 {
			line = body[offset:]
			offset = len(body)
		} else {
			line = body[offset : offset+nl]
			offset += nl + 1
		}
		line = bytes.TrimSpace(line)
		if len(line) == 0 || !bytes.HasPrefix(line, prefix) {
			continue
		}
		payload := bytes.TrimSpace(line[len(prefix):])
		if len(payload) == 0 {
			continue
		}
		if fn(payload) {
			return
		}
	}
}

// extractUsageFromStream parses SSE events looking for usage and timings.
func extractUsageFromStream(body []byte) ProxyMetrics {
	var (
		uc          usageCounts
		timings     map[string]any
		vllmMetrics map[string]any
		hasAny      bool
	)
	uc.cached = -1

	forEachDataLine(body, func(data []byte) bool {
		if bytes.Equal(data, []byte("[DONE]")) {
			return false
		}

		var obj map[string]any
		if err := json.Unmarshal(data, &obj); err != nil {
			log.Debugf("extractUsage from stream: failed to parse SSE data line: %v", err)
			return false
		}

		for _, path := range usagePaths {
			usage := getField(obj, path)
			if usage == nil {
				continue
			}
			c := extractUsageTokens(usage)
			if c.input > 0 {
				uc.input = c.input
			}
			if c.output > 0 {
				uc.output = c.output
			}
			if c.cached >= 0 {
				uc.cached = c.cached
			}
			if c.cacheCreation > 0 {
				uc.cacheCreation = c.cacheCreation
			}
			if c.reasoning > 0 {
				uc.reasoning = c.reasoning
			}
			hasAny = true
		}

		// llama.cpp native timings; vLLM's `metrics` object arrives on the
		// final stream chunk (intermediate chunks serialize it as null).
		if t, ok := obj["timings"].(map[string]any); ok {
			timings = t
			hasAny = true
		}
		if m, ok := obj["metrics"].(map[string]any); ok {
			vllmMetrics = m
			hasAny = true
		}
		return false
	})

	if !hasAny {
		return ProxyMetrics{CachedTokens: -1}
	}

	return buildMetricsFromData(uc, 0, timings, vllmMetrics)
}

// extractUsageFromJSON parses a non-streaming JSON response.
func extractUsageFromJSON(body []byte) ProxyMetrics {
	var obj map[string]any
	if err := json.Unmarshal(body, &obj); err != nil {
		log.Debugf("extractUsage from JSON: failed to parse response body: %v", err)
		return ProxyMetrics{CachedTokens: -1}
	}

	var usage map[string]any
	if u, ok := obj["usage"].(map[string]any); ok {
		usage = u
	}
	var timings map[string]any
	if t, ok := obj["timings"].(map[string]any); ok {
		timings = t
	}
	var vllmMetrics map[string]any
	if m, ok := obj["metrics"].(map[string]any); ok {
		vllmMetrics = m
	}

	if usage == nil && timings == nil && vllmMetrics == nil {
		return ProxyMetrics{CachedTokens: -1}
	}

	uc := extractUsageTokens(usage)
	total := intToFloat64(usage["total_tokens"])

	return buildMetricsFromData(uc, int64(total), timings, vllmMetrics)
}

// getField traverses a dotted JSON path in a map.
func getField(obj map[string]any, path string) map[string]any {
	parts := strings.Split(path, ".")
	var current any = obj
	for _, p := range parts {
		m, ok := current.(map[string]any)
		if !ok {
			return nil
		}
		current = m[p]
	}
	if current == nil {
		return nil
	}
	result, ok := current.(map[string]any)
	if !ok {
		return nil
	}
	return result
}

// usageCounts holds token counts extracted from a usage object.
type usageCounts struct {
	input         int
	output        int
	cached        int // -1 = not reported
	cacheCreation int // vLLM created_cache_tokens / Anthropic cache_creation_input_tokens
	reasoning     int // completion_tokens_details.reasoning_tokens
}

// extractUsageTokens reads token counts from a usage map.
func extractUsageTokens(usage map[string]any) usageCounts {
	uc := usageCounts{cached: -1}
	if usage == nil {
		return uc
	}

	uc.input = intToFloat64(usage["prompt_tokens"]) + intToFloat64(usage["input_tokens"])
	uc.output = intToFloat64(usage["completion_tokens"]) + intToFloat64(usage["output_tokens"])

	if v, ok := usage["cache_read_input_tokens"]; ok {
		uc.cached = intToFloat64(v)
	}
	if uc.cached < 0 {
		if details, ok := usage["input_tokens_details"].(map[string]any); ok {
			uc.cached = intToFloat64(details["cached_tokens"])
		}
	}
	if details, ok := usage["prompt_tokens_details"].(map[string]any); ok {
		if uc.cached < 0 {
			uc.cached = intToFloat64(details["cached_tokens"])
		}
		if v, ok := details["created_cache_tokens"]; ok {
			uc.cacheCreation = intToFloat64(v)
		}
	}
	if v, ok := usage["cache_creation_input_tokens"]; ok {
		uc.cacheCreation = intToFloat64(v)
	}
	if details, ok := usage["completion_tokens_details"].(map[string]any); ok {
		uc.reasoning = intToFloat64(details["reasoning_tokens"])
	}

	return uc
}

// buildMetricsFromData composes ProxyMetrics from token counts and optional
// backend-native timing objects: llama.cpp's top-level `timings` and vLLM's
// top-level `metrics` (requires --enable-per-request-metrics on the server).
// A backend emits at most one of the two, so the timings block wins when both
// are present.
func buildMetricsFromData(uc usageCounts, totalTokens int64, timings, vllmMetrics map[string]any) ProxyMetrics {
	pm := ProxyMetrics{
		PromptTokens:        uc.input,
		CompletionTokens:    uc.output,
		TotalTokens:         int(totalTokens),
		CachedTokens:        uc.cached,
		ReasoningTokens:     uc.reasoning,
		CacheCreationTokens: uc.cacheCreation,
	}

	// vLLM `metrics`: time_to_first_token_ms is prefill (queue wait excluded),
	// generation_time_ms is the decode interval (first to last output token),
	// tokens_per_second is the backend's output throughput.
	if vllmMetrics != nil {
		if ms := floatToFloat64(vllmMetrics["time_to_first_token_ms"]); ms > 0 {
			pm.PromptMs = ms
		}
		if ms := floatToFloat64(vllmMetrics["generation_time_ms"]); ms > 0 {
			pm.PredictedMs = ms
		}
		if tps := floatToFloat64(vllmMetrics["tokens_per_second"]); tps > 0 {
			pm.TokensPerSec = tps
		}
		pm.QueueMs = floatToFloat64(vllmMetrics["queue_time_ms"])
	}

	if timings != nil {
		// Native counts from llama.cpp's `timings` take precedence, but only
		// when present: a partial timings object must not zero out the
		// usage-derived token counts.
		if n := intToFloat64(timings["prompt_n"]); n > 0 {
			pm.PromptTokens = n
		}
		if n := intToFloat64(timings["predicted_n"]); n > 0 {
			pm.CompletionTokens = n
		}
		pm.PromptPerSec = floatToFloat64(timings["prompt_per_second"])
		pm.TokensPerSec = floatToFloat64(timings["predicted_per_second"])
		pm.PromptMs = floatToFloat64(timings["prompt_ms"])
		pm.PredictedMs = floatToFloat64(timings["predicted_ms"])
		if cv := intToFloat64(timings["cache_n"]); cv >= 0 {
			pm.CachedTokens = cv
		}
		// Speculative decoding stats, present only when a draft model is used.
		pm.DraftTokens = intToFloat64(timings["draft_n"])
		pm.DraftTokensAccepted = intToFloat64(timings["draft_n_accepted"])
	}

	return pm
}

// decompressBody decompresses gzip or deflate encoded data.
func decompressBody(body []byte, encoding string) ([]byte, error) {
	switch strings.ToLower(strings.TrimSpace(encoding)) {
	case "gzip":
		reader, err := gzip.NewReader(bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		defer func() {
			if err := reader.Close(); err != nil {
				log.Errorf("usage: failed to close gzip reader: %v", err)
			}
		}()
		return io.ReadAll(reader)
	case "deflate":
		reader := flate.NewReader(bytes.NewReader(body))
		defer func() {
			if err := reader.Close(); err != nil {
				log.Errorf("usage: failed to close flate reader: %v", err)
			}
		}()
		return io.ReadAll(reader)
	default:
		return body, nil
	}
}

func intToFloat64(v any) int {
	if v == nil {
		return 0
	}
	f, ok := v.(float64)
	if !ok {
		return 0
	}
	return int(f)
}

func floatToFloat64(v any) float64 {
	if v == nil {
		return 0
	}
	f, ok := v.(float64)
	if !ok {
		return 0
	}
	return f
}
