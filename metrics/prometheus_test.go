package metrics

import (
	"strings"
	"testing"
	"time"

	"llm-api-router/domain"
)

func TestPrometheusMetricsRendersSeries(t *testing.T) {
	s := New(100)
	s.Add(domain.RequestMetric{
		Timestamp:        time.Now(),
		Model:            "gpt-4",
		ServerID:         "s1",
		StatusCode:       200,
		LatencyMs:        100,
		TTFBMs:           50,
		ResponseSize:     1024,
		PromptTokens:     10,
		CompletionTokens: 20,
		TotalTokens:      30,
		CachedTokens:     5,
	})
	// Second request: fallback + error status, so counts diverge.
	s.Add(domain.RequestMetric{
		Timestamp:    time.Now(),
		Model:        "gpt-4",
		ServerID:     "s2",
		StatusCode:   502,
		LatencyMs:    200,
		TTFBMs:       100,
		WasFallback:  true,
		TotalTokens:  50,
		CachedTokens: 0,
	})

	out := s.PrometheusMetrics()

	// Both groups must be present with both label shapes.
	for _, want := range []string{
		`llm_router_requests_total{model="gpt-4"} 2`,
		`llm_router_requests_success_total{model="gpt-4"} 1`,
		`llm_router_requests_errors_total{model="gpt-4"} 1`,
		`llm_router_requests_fallback_total{model="gpt-4"} 1`,
		`llm_router_requests_total{server="s1"} 1`,
		`llm_router_requests_total{server="s2"} 1`,
		`llm_router_prompt_tokens_total{model="gpt-4"} 10`,
		`llm_router_completion_tokens_total{model="gpt-4"} 20`,
		`llm_router_tokens_total{model="gpt-4"} 80`,
		`llm_router_response_bytes_total{model="gpt-4"} 1024`,
		`llm_router_latency_avg_ms{model="gpt-4"} 150`,
		`llm_router_latency_min_ms{model="gpt-4"} 100`,
		`llm_router_latency_max_ms{model="gpt-4"} 200`,
		`llm_router_ttfb_avg_ms{model="gpt-4"} 75`,
		`llm_router_cached_tokens_total{server="s1"} 5`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q\n---\n%s", want, out)
		}
	}

	// HELP and TYPE lines for at least one metric.
	if !strings.Contains(out, "# HELP llm_router_requests_total") ||
		!strings.Contains(out, "# TYPE llm_router_requests_total counter") {
		t.Error("missing HELP/TYPE for llm_router_requests_total")
	}
	if !strings.Contains(out, "# TYPE llm_router_latency_avg_ms gauge") {
		t.Error("missing TYPE gauge for latency_avg_ms")
	}
}

func TestPrometheusMetricsLabelEscaping(t *testing.T) {
	s := New(100)
	s.Add(domain.RequestMetric{
		Timestamp:   time.Now(),
		Model:       `model"with\quote`,
		ServerID:    "s1",
		StatusCode:  200,
		TotalTokens: 1,
	})

	out := s.PrometheusMetrics()
	if !strings.Contains(out, `model="model\"with\\quote"`) {
		t.Errorf("label value not escaped properly:\n%s", out)
	}
}

func TestPrometheusMetricsEmpty(t *testing.T) {
	s := New(100)
	if out := s.PrometheusMetrics(); out != "" {
		t.Errorf("expected empty output for empty store, got %q", out)
	}
}
