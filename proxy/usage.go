package proxy

import (
	"bytes"
	"compress/flate"
	"compress/gzip"
	"encoding/json"
	"io"
	"strings"
)

// usagePaths lists JSON paths where usage can appear in SSE events.
var usagePaths = []string{"usage", "response.usage", "message.usage"}

// extractUsageFromResponse parses token usage and timings from the response body.
func extractUsageFromResponse(body []byte, contentEncoding string, isStream bool) ProxyMetrics {
	if contentEncoding != "" {
		decompressed, err := decompressBody(body, contentEncoding)
		if err != nil {
			return ProxyMetrics{CachedTokens: -1}
		}
		body = decompressed
	}

	if isStream {
		return extractUsageFromStream(body)
	}
	return extractUsageFromJSON(body)
}

// extractUsageFromStream parses SSE events looking for usage and timings.
func extractUsageFromStream(body []byte) ProxyMetrics {
	var (
		inputTokens  int64
		outputTokens int64
		cachedTokens int64 = -1
		hasAny       bool
		timings      map[string]interface{}
	)

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
		data := bytes.TrimSpace(line[len(prefix):])
		if len(data) == 0 || bytes.Equal(data, []byte("[DONE]")) {
			continue
		}

		var obj map[string]interface{}
		if err := json.Unmarshal(data, &obj); err != nil {
			continue
		}

		for _, path := range usagePaths {
			usage := getField(obj, path)
			if usage == nil {
				continue
			}
			i, o, c := extractUsageTokens(usage)
			if i > 0 {
				inputTokens = int64(i)
			}
			if o > 0 {
				outputTokens = int64(o)
			}
			if c >= 0 {
				cachedTokens = int64(c)
			}
			hasAny = true
		}

		if t, ok := obj["timings"].(map[string]interface{}); ok {
			timings = t
			hasAny = true
		}
	}

	if !hasAny {
		return ProxyMetrics{CachedTokens: -1}
	}

	return buildMetricsFromData(inputTokens, outputTokens, 0, cachedTokens, timings)
}

// extractUsageFromJSON parses a non-streaming JSON response.
func extractUsageFromJSON(body []byte) ProxyMetrics {
	var obj map[string]interface{}
	if err := json.Unmarshal(body, &obj); err != nil {
		return ProxyMetrics{CachedTokens: -1}
	}

	var usage map[string]interface{}
	if u, ok := obj["usage"].(map[string]interface{}); ok {
		usage = u
	}
	var timings map[string]interface{}
	if t, ok := obj["timings"].(map[string]interface{}); ok {
		timings = t
	}

	if usage == nil && timings == nil {
		return ProxyMetrics{CachedTokens: -1}
	}

	input, output, cached := extractUsageTokens(usage)
	total := intToFloat64(usage["total_tokens"])

	return buildMetricsFromData(int64(input), int64(output), int64(total), int64(cached), timings)
}

// getField traverses a dotted JSON path in a map.
func getField(obj map[string]interface{}, path string) map[string]interface{} {
	parts := strings.Split(path, ".")
	var current interface{} = obj
	for _, p := range parts {
		m, ok := current.(map[string]interface{})
		if !ok {
			return nil
		}
		current = m[p]
	}
	if current == nil {
		return nil
	}
	result, ok := current.(map[string]interface{})
	if !ok {
		return nil
	}
	return result
}

// extractUsageTokens reads token counts from a usage map.
func extractUsageTokens(usage map[string]interface{}) (input, output, cached int) {
	cached = -1
	if usage == nil {
		return
	}

	input = intToFloat64(usage["prompt_tokens"]) + intToFloat64(usage["input_tokens"])
	output = intToFloat64(usage["completion_tokens"]) + intToFloat64(usage["output_tokens"])

	if v, ok := usage["cache_read_input_tokens"]; ok {
		cached = intToFloat64(v)
	}
	if cached < 0 {
		if details, ok := usage["input_tokens_details"].(map[string]interface{}); ok {
			cached = intToFloat64(details["cached_tokens"])
		}
	}
	if cached < 0 {
		if details, ok := usage["prompt_tokens_details"].(map[string]interface{}); ok {
			cached = intToFloat64(details["cached_tokens"])
		}
	}

	return
}

// buildMetricsFromData composes ProxyMetrics from token counts and optional timings.
func buildMetricsFromData(inputTokens, outputTokens, totalTokens int64, cachedTokens int64, timings map[string]interface{}) ProxyMetrics {
	pm := ProxyMetrics{
		PromptTokens:     int(inputTokens),
		CompletionTokens: int(outputTokens),
		TotalTokens:      int(totalTokens),
		CachedTokens:     int(cachedTokens),
	}

	if timings != nil {
		pm.PromptTokens = intToFloat64(timings["prompt_n"])
		pm.CompletionTokens = intToFloat64(timings["predicted_n"])
		pm.PromptPerSec = floatToFloat64(timings["prompt_per_second"])
		pm.TokensPerSec = floatToFloat64(timings["predicted_per_second"])
		pm.PromptMs = floatToFloat64(timings["prompt_ms"])
		pm.PredictedMs = floatToFloat64(timings["predicted_ms"])
		if cv := intToFloat64(timings["cache_n"]); cv >= 0 {
			pm.CachedTokens = cv
		}
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
		defer reader.Close() //nolint:errcheck
		return io.ReadAll(reader)
	case "deflate":
		reader := flate.NewReader(bytes.NewReader(body))
		defer reader.Close() //nolint:errcheck
		return io.ReadAll(reader)
	default:
		return body, nil
	}
}

func intToFloat64(v interface{}) int {
	if v == nil {
		return 0
	}
	f, ok := v.(float64)
	if !ok {
		return 0
	}
	return int(f)
}

func floatToFloat64(v interface{}) float64 {
	if v == nil {
		return 0
	}
	f, ok := v.(float64)
	if !ok {
		return 0
	}
	return f
}
