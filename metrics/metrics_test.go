package metrics

import (
	"testing"
	"time"

	"llm-api-router/domain"
)

func TestNew(t *testing.T) {
	s := New(100)
	if s == nil {
		t.Fatal("New returned nil")
	}
	recent := s.Recent()
	if len(recent) != 0 {
		t.Errorf("expected empty recent, got %d", len(recent))
	}
	summaries := s.Summaries()
	if len(summaries) != 0 {
		t.Errorf("expected empty summaries, got %d", len(summaries))
	}
}

func TestAddAndRecent(t *testing.T) {
	s := New(100)

	now := time.Now()
	m := domain.RequestMetric{
		Timestamp:        now,
		Model:            "gpt-4",
		TargetModel:      "gpt-4",
		ServerID:         "s1",
		StatusCode:       200,
		LatencyMs:        1000,
		TTFBMs:           200,
		ResponseSize:     5000,
		WasFallback:      false,
		PromptTokens:     50,
		CompletionTokens: 100,
		TotalTokens:      150,
		CachedTokens:     10,
	}
	s.Add(m)

	recent := s.Recent()
	if len(recent) != 1 {
		t.Fatalf("expected 1 recent, got %d", len(recent))
	}

	got := recent[0]
	if got.Model != "gpt-4" {
		t.Errorf("Model = %q, want %q", got.Model, "gpt-4")
	}
	if got.LatencyMs != 1000 {
		t.Errorf("LatencyMs = %d, want 1000", got.LatencyMs)
	}
	if got.TTFBMs != 200 {
		t.Errorf("TTFBMs = %d, want 200", got.TTFBMs)
	}
	if got.PrefillTimeMs != 200 {
		t.Errorf("PrefillTimeMs = %d, want 200", got.PrefillTimeMs)
	}
	if got.DecodeTimeMs != 800 {
		t.Errorf("DecodeTimeMs = %d, want 800", got.DecodeTimeMs)
	}
}

func TestAddComputedFields(t *testing.T) {
	s := New(100)

	s.Add(domain.RequestMetric{
		Model:            "gpt-4",
		ServerID:         "s1",
		StatusCode:       200,
		LatencyMs:        1000,
		TTFBMs:           200,
		PromptTokens:     50,
		CompletionTokens: 100,
	})

	recent := s.Recent()
	got := recent[0]

	if got.PrefillTokPerSec < 249 || got.PrefillTokPerSec > 251 {
		t.Errorf("PrefillTokPerSec = %f, want ~250", got.PrefillTokPerSec)
	}
	if got.DecodeTokPerSec < 124.5 || got.DecodeTokPerSec > 125.5 {
		t.Errorf("DecodeTokPerSec = %f, want ~125", got.DecodeTokPerSec)
	}
}

func TestAddNegativeDecodeTime(t *testing.T) {
	s := New(100)
	s.Add(domain.RequestMetric{
		Model:      "gpt-4",
		ServerID:   "s1",
		StatusCode: 200,
		LatencyMs:  100,
		TTFBMs:     200,
	})
	recent := s.Recent()
	if recent[0].DecodeTimeMs != 0 {
		t.Errorf("DecodeTimeMs = %d, want 0", recent[0].DecodeTimeMs)
	}
}

func TestAddNoTokens(t *testing.T) {
	s := New(100)
	s.Add(domain.RequestMetric{
		Model:      "gpt-4",
		ServerID:   "s1",
		StatusCode: 200,
		LatencyMs:  100,
		TTFBMs:     50,
	})
	recent := s.Recent()
	if recent[0].PrefillTokPerSec != 0 {
		t.Errorf("PrefillTokPerSec = %f, want 0", recent[0].PrefillTokPerSec)
	}
	if recent[0].DecodeTokPerSec != 0 {
		t.Errorf("DecodeTokPerSec = %f, want 0", recent[0].DecodeTokPerSec)
	}
}

func TestRingBuffer(t *testing.T) {
	s := New(3)

	for i := 0; i < 10; i++ {
		s.Add(domain.RequestMetric{
			Model:    "gpt-4",
			ServerID: "s1",
		})
	}

	recent := s.Recent()
	if len(recent) != 3 {
		t.Errorf("expected 3 recent, got %d", len(recent))
	}
}

func TestSummaries(t *testing.T) {
	s := New(100)

	s.Add(domain.RequestMetric{Model: "gpt-4", ServerID: "s1", StatusCode: 200, LatencyMs: 100, TTFBMs: 50, PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30, ResponseSize: 100})
	s.Add(domain.RequestMetric{Model: "gpt-4", ServerID: "s1", StatusCode: 200, LatencyMs: 200, TTFBMs: 100, PromptTokens: 20, CompletionTokens: 40, TotalTokens: 60, ResponseSize: 200})
	s.Add(domain.RequestMetric{Model: "gpt-4", ServerID: "s1", StatusCode: 500, LatencyMs: 300, TTFBMs: 150, PromptTokens: 0, CompletionTokens: 0, TotalTokens: 0, ResponseSize: 50})

	summaries := s.Summaries()

	modelSum, ok := summaries["gpt-4"]
	if !ok {
		t.Fatal("summary for gpt-4 not found")
	}

	if modelSum.TotalRequests != 3 {
		t.Errorf("TotalRequests = %d, want 3", modelSum.TotalRequests)
	}
	if modelSum.SuccessCount != 2 {
		t.Errorf("SuccessCount = %d, want 2", modelSum.SuccessCount)
	}
	if modelSum.ErrorCount != 1 {
		t.Errorf("ErrorCount = %d, want 1", modelSum.ErrorCount)
	}
	if modelSum.MinLatencyMs != 100 {
		t.Errorf("MinLatencyMs = %d, want 100", modelSum.MinLatencyMs)
	}
	if modelSum.MaxLatencyMs != 300 {
		t.Errorf("MaxLatencyMs = %d, want 300", modelSum.MaxLatencyMs)
	}
	if modelSum.TotalPromptTok != 30 {
		t.Errorf("TotalPromptTok = %d, want 30", modelSum.TotalPromptTok)
	}
	if modelSum.TotalCompleteTok != 60 {
		t.Errorf("TotalCompleteTok = %d, want 60", modelSum.TotalCompleteTok)
	}
	if modelSum.TotalBytes != 350 {
		t.Errorf("TotalBytes = %d, want 350", modelSum.TotalBytes)
	}
}

func TestSummariesFallbackCount(t *testing.T) {
	s := New(100)
	s.Add(domain.RequestMetric{Model: "gpt-4", ServerID: "s1", StatusCode: 200, WasFallback: false})
	s.Add(domain.RequestMetric{Model: "gpt-4", ServerID: "s2", StatusCode: 200, WasFallback: true})

	summaries := s.Summaries()
	modelSum := summaries["gpt-4"]
	if modelSum.FallbackCount != 1 {
		t.Errorf("FallbackCount = %d, want 1", modelSum.FallbackCount)
	}

	serverSum := summaries["server:s1"]
	if serverSum.FallbackCount != 0 {
		t.Errorf("server s1 FallbackCount = %d, want 0", serverSum.FallbackCount)
	}

	serverSum2 := summaries["server:s2"]
	if serverSum2.FallbackCount != 1 {
		t.Errorf("server s2 FallbackCount = %d, want 1", serverSum2.FallbackCount)
	}
}

func TestSummariesNegativeCachedTokens(t *testing.T) {
	s := New(100)
	s.Add(domain.RequestMetric{Model: "gpt-4", ServerID: "s1", StatusCode: 200, CachedTokens: -1})
	s.Add(domain.RequestMetric{Model: "gpt-4", ServerID: "s1", StatusCode: 200, CachedTokens: 5})

	summaries := s.Summaries()
	modelSum := summaries["gpt-4"]
	if modelSum.TotalCachedTok != 5 {
		t.Errorf("TotalCachedTok = %d, want 5", modelSum.TotalCachedTok)
	}
}

func TestSummariesWeightedAverages(t *testing.T) {
	s := New(100)
	s.Add(domain.RequestMetric{
		Model: "gpt-4", ServerID: "s1", StatusCode: 200,
		LatencyMs: 1000, TTFBMs: 100,
		PromptTokens: 100, CompletionTokens: 200,
	})
	s.Add(domain.RequestMetric{
		Model: "gpt-4", ServerID: "s1", StatusCode: 200,
		LatencyMs: 2000, TTFBMs: 200,
		PromptTokens: 200, CompletionTokens: 400,
	})

	summaries := s.Summaries()
	modelSum := summaries["gpt-4"]

	if modelSum.AvgLatencyMs != 1500 {
		t.Errorf("AvgLatencyMs = %f, want 1500", modelSum.AvgLatencyMs)
	}

	if modelSum.AvgTTFBMs != 150 {
		t.Errorf("AvgTTFBMs = %f, want 150", modelSum.AvgTTFBMs)
	}
}

func TestReset(t *testing.T) {
	s := New(100)
	s.Add(domain.RequestMetric{Model: "gpt-4", ServerID: "s1", StatusCode: 200})
	s.Reset()

	recent := s.Recent()
	if len(recent) != 0 {
		t.Errorf("expected empty after reset, got %d", len(recent))
	}
	summaries := s.Summaries()
	if len(summaries) != 0 {
		t.Errorf("expected empty summaries after reset, got %d", len(summaries))
	}
}

func TestRecentReturnsCopy(t *testing.T) {
	s := New(100)
	s.Add(domain.RequestMetric{Model: "gpt-4", ServerID: "s1", LatencyMs: 100})

	recent := s.Recent()
	recent[0].LatencyMs = 999

	recent2 := s.Recent()
	if recent2[0].LatencyMs == 999 {
		t.Error("Recent() should return a copy")
	}
}

func TestSummariesReturnsCopy(t *testing.T) {
	s := New(100)
	s.Add(domain.RequestMetric{Model: "gpt-4", ServerID: "s1", StatusCode: 200, LatencyMs: 100})

	summaries := s.Summaries()
	sum := summaries["gpt-4"]
	sum.TotalRequests = 999

	summaries2 := s.Summaries()
	if summaries2["gpt-4"].TotalRequests == 999 {
		t.Error("Summaries() should return a copy")
	}
}

func TestDecodeTokSingleToken(t *testing.T) {
	s := New(100)
	s.Add(domain.RequestMetric{
		Model:            "gpt-4",
		ServerID:         "s1",
		StatusCode:       200,
		LatencyMs:        1000,
		TTFBMs:           200,
		PromptTokens:     50,
		CompletionTokens: 1, // single-token response
	})

	recent := s.Recent()
	got := recent[0]
	// DecodeTimeMs = 1000 - 200 = 800ms
	// DecodeTokPerSec = 1 / 0.8 = 1.25
	if got.DecodeTokPerSec <= 0 {
		t.Errorf("DecodeTokPerSec = %f, want > 0 (single-token response should have non-zero decode speed)", got.DecodeTokPerSec)
	}
	if got.DecodeTokPerSec < 1.24 || got.DecodeTokPerSec > 1.26 {
		t.Errorf("DecodeTokPerSec = %f, want ~1.25", got.DecodeTokPerSec)
	}
}

func TestNativeTimingsPreferred(t *testing.T) {
	s := New(100)
	s.Add(domain.RequestMetric{
		Model:               "gpt-4",
		ServerID:            "s1",
		StatusCode:          200,
		LatencyMs:           1000,
		TTFBMs:              300,
		PromptTokens:        500,
		CompletionTokens:    200,
		NativePromptMs:      150,         // server reports 150ms for prompt processing
		NativePredictedMs:   600,         // server reports 600ms for token generation
		NativePromptTokPerSec: 3333.33,  // 500 / 0.15
		NativeDecodeTokPerSec: 333.33,   // 200 / 0.6
	})

	recent := s.Recent()
	got := recent[0]

	// Should use native timings, not wall-clock TTFB
	if got.PrefillTimeMs != 150 {
		t.Errorf("PrefillTimeMs = %d, want 150 (from native timings)", got.PrefillTimeMs)
	}
	if got.DecodeTimeMs != 600 {
		t.Errorf("DecodeTimeMs = %d, want 600 (from native timings)", got.DecodeTimeMs)
	}
	// Should use native tok/s, not wall-clock calculated
	if got.PrefillTokPerSec < 3333 || got.PrefillTokPerSec > 3334 {
		t.Errorf("PrefillTokPerSec = %f, want ~3333.33 (from native)", got.PrefillTokPerSec)
	}
	if got.DecodeTokPerSec < 333 || got.DecodeTokPerSec > 334 {
		t.Errorf("DecodeTokPerSec = %f, want ~333.33 (from native)", got.DecodeTokPerSec)
	}
}

func TestNativeTimingsFallbackToWallClock(t *testing.T) {
	s := New(100)
	s.Add(domain.RequestMetric{
		Model:            "gpt-4",
		ServerID:         "s1",
		StatusCode:       200,
		LatencyMs:        1000,
		TTFBMs:           200,
		PromptTokens:     50,
		CompletionTokens: 100,
		// No native timings — should fall back to wall-clock
	})

	recent := s.Recent()
	got := recent[0]

	if got.PrefillTimeMs != 200 {
		t.Errorf("PrefillTimeMs = %d, want 200 (TTFB fallback)", got.PrefillTimeMs)
	}
	if got.DecodeTimeMs != 800 {
		t.Errorf("DecodeTimeMs = %d, want 800 (latency - TTFB fallback)", got.DecodeTimeMs)
	}
}
