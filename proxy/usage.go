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

// llamaTimingsPaths and vllmMetricsPaths list where the native backend timing
// objects can appear in an SSE event: top level for chat/completions, and
// nested for the Responses API, where llama.cpp attaches `timings` to the
// event's `data` object and vLLM carries `metrics` inside `response`.
var (
	llamaTimingsPaths = []string{"timings", "response.timings", "data.timings"}
	vllmMetricsPaths  = []string{"metrics", "response.metrics", "data.metrics"}
)

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
		for _, path := range llamaTimingsPaths {
			if t := getField(obj, path); t != nil {
				timings = t
				hasAny = true
			}
		}
		for _, path := range vllmMetricsPaths {
			if m := getField(obj, path); m != nil {
				vllmMetrics = m
				hasAny = true
			}
		}
		return false
	})

	if !hasAny {
		return ProxyMetrics{CachedTokens: -1}
	}

	// SGLang never emits meta_info in stream chunks (return_meta_info requires
	// stream=false), so the stream path has no native timing source.
	return buildMetricsFromData(uc, 0, timings, vllmMetrics, nil)
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
	sglangMeta := sglangMetaInfo(obj)

	if usage == nil && timings == nil && vllmMetrics == nil && sglangMeta == nil {
		return ProxyMetrics{CachedTokens: -1}
	}

	uc := extractUsageTokens(usage)
	total := intToFloat64(usage["total_tokens"])

	return buildMetricsFromData(uc, int64(total), timings, vllmMetrics, sglangMeta)
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

// sglangMetaInfo returns SGLang's per-choice meta_info object from a
// non-streaming OpenAI chat response (choices[0].meta_info), or nil if absent.
// SGLang only includes the timing fields there when the client sends
// return_meta_info=true and the server runs with --enable-metrics; it is never
// present in stream chunks.
func sglangMetaInfo(obj map[string]any) map[string]any {
	choices, ok := obj["choices"].([]any)
	if !ok || len(choices) == 0 {
		return nil
	}
	choice, ok := choices[0].(map[string]any)
	if !ok {
		return nil
	}
	meta, _ := choice["meta_info"].(map[string]any)
	return meta
}

// usageCounts holds token counts and backend-reported rates from a usage object.
type usageCounts struct {
	input         int
	output        int
	cached        int     // -1 = not reported
	cacheCreation int     // created_cache_tokens / cache_creation_input_tokens / cache_write_tokens
	reasoning     int     // reasoning_tokens: nested details, or (SGLang) on usage itself
	promptPerSec  float64 // TabbyAPI prompt_tokens_per_sec / oMLX prompt_tokens_per_second
	tokensPerSec  float64 // TabbyAPI completion_tokens_per_sec / oMLX generation_tokens_per_second
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
	}

	// Written-cache and reasoning tokens are spelled three ways: vLLM nests them
	// in prompt/completion_tokens_details, Anthropic puts cache creation on the
	// usage object itself, and the Responses API uses input/output_tokens_details.
	// SGLang reports reasoning tokens directly on usage.
	uc.cacheCreation = firstPositive(0,
		intToFloat64(usage["cache_creation_input_tokens"]),
		nestedInt(usage, "prompt_tokens_details", "created_cache_tokens"),
		nestedInt(usage, "input_tokens_details", "cache_write_tokens"))
	uc.reasoning = firstPositive(0,
		nestedInt(usage, "completion_tokens_details", "reasoning_tokens"),
		nestedInt(usage, "output_tokens_details", "reasoning_tokens"),
		intToFloat64(usage["reasoning_tokens"]))

	// TabbyAPI and oMLX report the backend's own throughput inside usage, using
	// different suffixes for the same two numbers.
	uc.promptPerSec = firstPositiveF(0,
		floatToFloat64(usage["prompt_tokens_per_sec"]),
		floatToFloat64(usage["prompt_tokens_per_second"]))
	uc.tokensPerSec = firstPositiveF(0,
		floatToFloat64(usage["completion_tokens_per_sec"]),
		floatToFloat64(usage["generation_tokens_per_second"]))

	return uc
}

// nestedInt reads obj[container][key] as a number, 0 when either is absent.
func nestedInt(obj map[string]any, container, key string) int {
	m, _ := obj[container].(map[string]any)
	return intToFloat64(m[key])
}

// firstPositive returns the first value greater than zero, else fallback.
func firstPositive(fallback int, vals ...int) int {
	for _, v := range vals {
		if v > 0 {
			return v
		}
	}
	return fallback
}

func firstPositiveF(fallback float64, vals ...float64) float64 {
	for _, v := range vals {
		if v > 0 {
			return v
		}
	}
	return fallback
}

// buildMetricsFromData composes ProxyMetrics from token counts and optional
// backend-native timing objects: llama.cpp's top-level `timings`, vLLM's
// top-level `metrics` (requires --enable-per-request-metrics), and SGLang's
// `choices[0].meta_info` (requires return_meta_info + --enable-metrics). A
// backend emits at most one of the three, so the timings block wins when both
// are present. Rates embedded in the usage object (TabbyAPI, oMLX) seed the
// throughput fields and lose to all three.
func buildMetricsFromData(uc usageCounts, totalTokens int64, timings, vllmMetrics, sglangMeta map[string]any) ProxyMetrics {
	pm := ProxyMetrics{
		PromptTokens:        uc.input,
		CompletionTokens:    uc.output,
		TotalTokens:         int(totalTokens),
		CachedTokens:        uc.cached,
		ReasoningTokens:     uc.reasoning,
		CacheCreationTokens: uc.cacheCreation,
		// TabbyAPI/oMLX rates ride inside usage; a real timings/metrics object
		// below overrides them.
		PromptPerSec: uc.promptPerSec,
		TokensPerSec: uc.tokensPerSec,
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
		// Speculative decoding draft counts (needs
		// --per-request-spec-decode-metrics; n==1 only, null for n>1), mapped
		// like llama.cpp's draft_n / draft_n_accepted.
		if spec, ok := vllmMetrics["speculative_decoding"].(map[string]any); ok {
			if n := intToFloat64(spec["num_draft_tokens"]); n > 0 {
				pm.DraftTokens = n
			}
			if n := intToFloat64(spec["num_accepted_draft_tokens"]); n > 0 {
				pm.DraftTokensAccepted = n
			}
		}
	}

	// SGLang `meta_info`: first_token_latency is TTFT/prefill, decode_throughput
	// is the backend's decode tok/s, queue_time is the scheduler queue wait — all
	// in seconds, so scale to ms where the field is ms-based.
	if sglangMeta != nil {
		if s := floatToFloat64(sglangMeta["first_token_latency"]); s > 0 {
			pm.PromptMs = s * 1000
		}
		if tps := floatToFloat64(sglangMeta["decode_throughput"]); tps > 0 {
			pm.TokensPerSec = tps
		}
		if s := floatToFloat64(sglangMeta["queue_time"]); s > 0 {
			pm.QueueMs = s * 1000
		}
		// SGLang reports cached tokens in meta_info (not the standard usage
		// object); only fill it when the usage-derived value is absent.
		if cv := intToFloat64(sglangMeta["cached_tokens"]); cv >= 0 && pm.CachedTokens < 0 {
			pm.CachedTokens = cv
		}
		// Speculative decoding: proposed vs. accepted draft token counts.
		if n := intToFloat64(sglangMeta["spec_num_proposed_drafts"]); n > 0 {
			pm.DraftTokens = n
		}
		if n := intToFloat64(sglangMeta["spec_num_correct_drafts"]); n > 0 {
			pm.DraftTokensAccepted = n
		}
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
		// Guarded so a partial timings object cannot erase usage-reported rates.
		if v := floatToFloat64(timings["prompt_per_second"]); v > 0 {
			pm.PromptPerSec = v
		}
		if v := floatToFloat64(timings["predicted_per_second"]); v > 0 {
			pm.TokensPerSec = v
		}
		pm.PromptMs = floatToFloat64(timings["prompt_ms"])
		pm.PredictedMs = floatToFloat64(timings["predicted_ms"])
		// Presence-guarded: llama.cpp's per-chunk timings carry no cache_n, and a
		// missing key must not erase the usage-derived count.
		if v, ok := timings["cache_n"]; ok {
			if cv := intToFloat64(v); cv >= 0 {
				pm.CachedTokens = cv
			}
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
