package metrics

import (
	"strconv"
	"strings"
)

// PrometheusMetrics renders the aggregated summaries in Prometheus text
// exposition format (version 0.0.4), zero external dependencies.
//
// Each summary group is emitted twice: once labeled `group="model"` +
// `model="<name>"`, and once `group="server"` + `server="<id>"`. Counter
// families count toward the current ring-buffer window (reset on rotation /
// Store.Reset), so treat long-term values as "recent window" figures.
func (s *Store) PrometheusMetrics() string {
	summaries := s.Summaries()

	var sb strings.Builder
	for groupKey, sum := range summaries {
		group, key := parseGroupKey(groupKey)
		if group == "" {
			continue
		}
		lbl := group + "=\"" + escapeLabel(key) + "\""

		emit := func(name, help, typ, value string) {
			sb.WriteString("# HELP " + name + " " + help + "\n")
			sb.WriteString("# TYPE " + name + " " + typ + "\n")
			sb.WriteString(name + "{" + lbl + "} " + value + "\n")
		}

		emit("llm_router_requests_total", "Total requests routed.", "counter", strconv.Itoa(sum.TotalRequests))
		emit("llm_router_requests_success_total", "Successful (2xx) requests.", "counter", strconv.Itoa(sum.SuccessCount))
		emit("llm_router_requests_errors_total", "Errored requests (non-2xx).", "counter", strconv.Itoa(sum.ErrorCount))
		emit("llm_router_requests_fallback_total", "Requests served by a fallback attempt.", "counter", strconv.Itoa(sum.FallbackCount))
		emit("llm_router_latency_avg_ms", "Average request latency in milliseconds.", "gauge", formatFloat(sum.AvgLatencyMs))
		emit("llm_router_latency_min_ms", "Minimum request latency in milliseconds.", "gauge", strconv.FormatInt(sum.MinLatencyMs, 10))
		emit("llm_router_latency_max_ms", "Maximum request latency in milliseconds.", "gauge", strconv.FormatInt(sum.MaxLatencyMs, 10))
		emit("llm_router_ttfb_avg_ms", "Average time-to-first-byte in milliseconds.", "gauge", formatFloat(sum.AvgTTFBMs))
		emit("llm_router_response_bytes_total", "Total response bytes.", "counter", strconv.FormatInt(sum.TotalBytes, 10))
		emit("llm_router_prompt_tokens_total", "Total prompt tokens.", "counter", strconv.Itoa(sum.TotalPromptTok))
		emit("llm_router_completion_tokens_total", "Total completion tokens.", "counter", strconv.Itoa(sum.TotalCompleteTok))
		emit("llm_router_tokens_total", "Total tokens (prompt + completion).", "counter", strconv.Itoa(sum.TotalTokens))
		emit("llm_router_cached_tokens_total", "Total cached (context) tokens.", "counter", strconv.Itoa(sum.TotalCachedTok))
		emit("llm_router_prefill_tok_per_sec", "Average prefill throughput in tokens/sec.", "gauge", formatFloat(sum.AvgPrefillTokSec))
		emit("llm_router_decode_tok_per_sec", "Average decode throughput in tokens/sec.", "gauge", formatFloat(sum.AvgDecodeTokSec))
	}

	return sb.String()
}

// parseGroupKey splits a summary key into (group, label). Summary keys are
// either a bare model name or "server:<id>".
func parseGroupKey(key string) (group, label string) {
	if after, ok := strings.CutPrefix(key, "server:"); ok {
		return "server", after
	}
	if key != "" {
		return "model", key
	}
	return "", ""
}

// escapeLabel escapes a Prometheus label value.
func escapeLabel(v string) string {
	r := strings.NewReplacer("\\", "\\\\", "\"", "\\\"", "\n", "\\n")
	return r.Replace(v)
}

// formatFloat formats a float for Prometheus (guards against NaN/Inf).
func formatFloat(f float64) string {
	if f != f || f > 1.7976931348623157e308 {
		return "0"
	}
	return strconv.FormatFloat(f, 'g', -1, 64)
}
