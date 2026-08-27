package metrics

import (
	"sync"
	"sync/atomic"

	"llm-api-router/domain"
)

// Store holds metrics in memory with a ring buffer for recent requests
// and aggregated summaries.
type Store struct {
	mu             sync.RWMutex
	recentRequests []domain.RequestMetric
	ringIndex      int
	ringCount      int
	summaries      map[string]*domain.Summary
	maxRecent      int
	activeRequests atomic.Int64
}

// New creates a new metrics store.
func New(maxRecent int) *Store {
	return &Store{
		recentRequests: make([]domain.RequestMetric, maxRecent),
		summaries:      make(map[string]*domain.Summary),
		maxRecent:      maxRecent,
	}
}

// Add records a new request metric.
func (s *Store) Add(m domain.RequestMetric) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Use native llama-server timings when available; fall back to wall-clock.
	if m.NativePromptMs > 0 {
		m.PrefillTimeMs = int64(m.NativePromptMs)
	} else {
		m.PrefillTimeMs = m.TTFBMs
	}

	if m.NativePredictedMs > 0 {
		m.DecodeTimeMs = int64(m.NativePredictedMs)
	} else {
		m.DecodeTimeMs = max(m.LatencyMs-m.TTFBMs, 0)
	}

	// Use native tok/s when available; fall back to wall-clock calculation.
	if m.NativePromptTokPerSec > 0 {
		m.PrefillTokPerSec = m.NativePromptTokPerSec
	} else if m.PrefillTimeMs > 0 && m.PromptTokens > 0 {
		m.PrefillTokPerSec = float64(m.PromptTokens) / (float64(m.PrefillTimeMs) / 1000.0)
	}

	if m.NativeDecodeTokPerSec > 0 {
		m.DecodeTokPerSec = m.NativeDecodeTokPerSec
	} else if m.DecodeTimeMs > 0 && m.CompletionTokens > 0 {
		m.DecodeTokPerSec = float64(m.CompletionTokens) / (float64(m.DecodeTimeMs) / 1000.0)
	}

	s.recentRequests[s.ringIndex] = m
	s.ringIndex = (s.ringIndex + 1) % s.maxRecent
	if s.ringCount < s.maxRecent {
		s.ringCount++
	}

	s.updateSummary(m.Model, m)
	s.updateSummary("server:"+m.ServerID, m)
	// Empty target (pass-through rule) means the target is the incoming model.
	// Keyed by server+target so the same model on different servers stays
	// separate. Server ID comes first: it must not contain ':' (model names
	// may — ollama-style qwen3:32b).
	target := m.TargetModel
	if target == "" {
		target = m.Model
	}
	s.updateSummary("target:"+m.ServerID+":"+target, m)
}

func (s *Store) updateSummary(key string, m domain.RequestMetric) {
	sum, ok := s.summaries[key]
	if !ok {
		sum = &domain.Summary{}
		s.summaries[key] = sum
	}

	sum.TotalRequests++
	if m.StatusCode >= 200 && m.StatusCode < 300 {
		sum.SuccessCount++
	} else {
		sum.ErrorCount++
	}
	if m.WasFallback {
		sum.FallbackCount++
	}

	if sum.MinLatencyMs == 0 || m.LatencyMs < sum.MinLatencyMs {
		sum.MinLatencyMs = m.LatencyMs
	}
	if m.LatencyMs > sum.MaxLatencyMs {
		sum.MaxLatencyMs = m.LatencyMs
	}

	sum.AvgLatencyMs = (sum.AvgLatencyMs*float64(sum.TotalRequests-1) + float64(m.LatencyMs)) / float64(sum.TotalRequests)
	sum.AvgTTFBMs = (sum.AvgTTFBMs*float64(sum.TotalRequests-1) + float64(m.TTFBMs)) / float64(sum.TotalRequests)
	sum.TotalBytes += m.ResponseSize

	sum.TotalPromptTok += m.PromptTokens
	sum.TotalCompleteTok += m.CompletionTokens
	sum.TotalTokens += m.TotalTokens
	if m.CachedTokens >= 0 {
		sum.TotalCachedTok += m.CachedTokens
	}
	sum.TotalReasoningTok += m.ReasoningTokens
	sum.TotalCacheCreationTok += m.CacheCreationTokens

	if m.PrefillTokPerSec > 0 {
		sum.AvgPrefillTokSec = (sum.AvgPrefillTokSec*float64(sum.TotalPromptTok-m.PromptTokens) + m.PrefillTokPerSec*float64(m.PromptTokens)) / float64(sum.TotalPromptTok)
	}
	if m.DecodeTokPerSec > 0 {
		sum.AvgDecodeTokSec = (sum.AvgDecodeTokSec*float64(sum.TotalCompleteTok-m.CompletionTokens) + m.DecodeTokPerSec*float64(m.CompletionTokens)) / float64(sum.TotalCompleteTok)
	}
}

// Recent returns the last N recorded requests in chronological order.
func (s *Store) Recent() []domain.RequestMetric {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make([]domain.RequestMetric, s.ringCount)
	if s.ringCount == 0 {
		return out
	}
	if s.ringCount < s.maxRecent {
		copy(out, s.recentRequests[:s.ringCount])
	} else {
		// Ring buffer is full: read from ringIndex to end, then 0 to ringIndex-1
		n := copy(out, s.recentRequests[s.ringIndex:])
		copy(out[n:], s.recentRequests[:s.ringIndex])
	}
	return out
}

// Summaries returns all aggregated summaries.
func (s *Store) Summaries() map[string]domain.Summary {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make(map[string]domain.Summary, len(s.summaries))
	for k, v := range s.summaries {
		out[k] = *v
	}
	return out
}

// Reset clears all metrics.
func (s *Store) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.recentRequests = make([]domain.RequestMetric, s.maxRecent)
	s.ringIndex = 0
	s.ringCount = 0
	s.summaries = make(map[string]*domain.Summary)
	// NOTE: activeRequests is intentionally NOT reset here — it counts live
	// in-flight requests; zeroing it while requests run would let their deferred
	// DecrActive push the counter negative.
}

// ActiveRequests returns the current number of in-flight requests.
func (s *Store) ActiveRequests() int64 {
	return s.activeRequests.Load()
}

// IncrActive increments the active request counter.
func (s *Store) IncrActive() {
	s.activeRequests.Add(1)
}

// DecrActive decrements the active request counter.
func (s *Store) DecrActive() {
	s.activeRequests.Add(-1)
}
