package metrics

import (
	"sort"
	"strconv"
	"strings"

	"llm-api-router/domain"
)

// PrometheusMetrics renders the aggregated summaries in Prometheus text
// exposition format (version 0.0.4), zero external dependencies.
//
// Each summary group is emitted with the labels of its key: `model="<name>"`,
// `server="<id>"`, `server="<id>",target_model="<model>"`, `class="<label>"`.
// Counter families count toward the current ring-buffer window (reset on
// rotation / Store.Reset), so treat long-term values as "recent window" figures.
//
// Latency and TTFB are also exported as histograms: avg/min/max hide the tail,
// and paging on a streaming backend is about p95 time-to-first-token, which
// needs buckets to interpolate over.
func (s *Store) PrometheusMetrics() string {
	summaries := s.Summaries()

	var sb strings.Builder
	for groupKey, sum := range summaries {
		lbl := groupLabels(groupKey)
		if lbl == "" {
			continue
		}

		emit := func(name, help, typ, labels, value string) {
			sb.WriteString("# HELP " + name + " " + help + "\n")
			sb.WriteString("# TYPE " + name + " " + typ + "\n")
			sb.WriteString(name + "{" + labels + "} " + value + "\n")
		}

		emit("llm_router_requests_total", "Total requests routed.", "counter", lbl, strconv.Itoa(sum.TotalRequests))
		emit("llm_router_requests_success_total", "Successful (2xx) requests.", "counter", lbl, strconv.Itoa(sum.SuccessCount))
		emit("llm_router_requests_errors_total", "Errored requests (non-2xx).", "counter", lbl, strconv.Itoa(sum.ErrorCount))
		emit("llm_router_requests_fallback_total", "Requests served by a fallback attempt.", "counter", lbl, strconv.Itoa(sum.FallbackCount))
		emit("llm_router_latency_avg_ms", "Average request latency in milliseconds.", "gauge", lbl, formatFloat(sum.AvgLatencyMs))
		emit("llm_router_latency_min_ms", "Minimum request latency in milliseconds.", "gauge", lbl, strconv.FormatInt(sum.MinLatencyMs, 10))
		emit("llm_router_latency_max_ms", "Maximum request latency in milliseconds.", "gauge", lbl, strconv.FormatInt(sum.MaxLatencyMs, 10))
		emit("llm_router_ttfb_avg_ms", "Average time-to-first-byte in milliseconds.", "gauge", lbl, formatFloat(sum.AvgTTFBMs))
		emit("llm_router_response_bytes_total", "Total response bytes.", "counter", lbl, strconv.FormatInt(sum.TotalBytes, 10))
		emit("llm_router_prompt_tokens_total", "Total prompt tokens.", "counter", lbl, strconv.Itoa(sum.TotalPromptTok))
		emit("llm_router_completion_tokens_total", "Total completion tokens.", "counter", lbl, strconv.Itoa(sum.TotalCompleteTok))
		emit("llm_router_tokens_total", "Total tokens (prompt + completion).", "counter", lbl, strconv.Itoa(sum.TotalTokens))
		emit("llm_router_cached_tokens_total", "Total cached (context) tokens.", "counter", lbl, strconv.Itoa(sum.TotalCachedTok))
		emit("llm_router_prefill_tok_per_sec", "Average prefill throughput in tokens/sec.", "gauge", lbl, formatFloat(sum.AvgPrefillTokSec))
		emit("llm_router_decode_tok_per_sec", "Average decode throughput in tokens/sec.", "gauge", lbl, formatFloat(sum.AvgDecodeTokSec))

		// Status classes: "backend is throttling us" (429, many 4xx) and "backend
		// is dying" (5xx, transport) need opposite answers, and the single
		// error counter can not tell them apart.
		classes := make([]string, 0, len(sum.StatusClassCount))
		for c := range sum.StatusClassCount {
			classes = append(classes, c)
		}
		sort.Strings(classes)
		for _, c := range classes {
			emit("llm_router_requests_by_status_total", "Requests by HTTP status class.", "counter",
				lbl+`,status_class="`+escapeLabel(c)+`"`, strconv.Itoa(sum.StatusClassCount[c]))
		}

		histogram := func(name, help string, buckets []int, sumMs int64) {
			if len(buckets) == 0 {
				return
			}
			sb.WriteString("# HELP " + name + " " + help + "\n")
			sb.WriteString("# TYPE " + name + " histogram\n")
			for i, bound := range domain.LatencyBucketsMs {
				sb.WriteString(name + "_bucket{" + lbl + `,le="` + strconv.FormatInt(bound, 10) + `"} ` +
					strconv.Itoa(buckets[i]) + "\n")
			}
			sb.WriteString(name + "_bucket{" + lbl + `,le="+Inf"} ` + strconv.Itoa(buckets[len(buckets)-1]) + "\n")
			sb.WriteString(name + "_sum{" + lbl + "} " + strconv.FormatInt(sumMs, 10) + "\n")
			sb.WriteString(name + "_count{" + lbl + "} " + strconv.Itoa(buckets[len(buckets)-1]) + "\n")
		}
		histogram("llm_router_latency_ms", "Request latency distribution in milliseconds.",
			sum.LatencyMsBuckets, sum.LatencyMsSum)
		histogram("llm_router_ttfb_ms", "Time-to-first-byte distribution in milliseconds.",
			sum.TTFBMsBuckets, sum.TTFBMsSum)
	}

	return sb.String()
}

// groupLabels turns a summary key into its Prometheus label set. Keys are
// "<group>:<rest>": a target key carries the server and the target model
// (splitting them is what makes the series selectable on either), a class key
// carries the request class, and a bare key is the incoming model name.
func groupLabels(key string) string {
	if after, ok := strings.CutPrefix(key, "server:"); ok {
		return `server="` + escapeLabel(after) + `"`
	}
	if after, ok := strings.CutPrefix(key, "target:"); ok {
		server, target, found := strings.Cut(after, ":")
		if !found {
			return `server="` + escapeLabel(after) + `"`
		}
		return `server="` + escapeLabel(server) + `",target_model="` + escapeLabel(target) + `"`
	}
	if after, ok := strings.CutPrefix(key, "class:"); ok {
		return `class="` + escapeLabel(after) + `"`
	}
	if key != "" {
		return `model="` + escapeLabel(key) + `"`
	}
	return ""
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
