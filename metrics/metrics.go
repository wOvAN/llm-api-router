package metrics

import (
	"sync"

	"llm-api-router/domain"
)

// Store holds metrics in memory with a ring buffer for recent requests
// and aggregated summaries.
type Store struct {
	mu             sync.RWMutex
	recentRequests []domain.RequestMetric
	summaries      map[string]*domain.Summary
	maxRecent      int
}

// New creates a new metrics store.
func New(maxRecent int) *Store {
	return &Store{
		recentRequests: make([]domain.RequestMetric, 0, maxRecent),
		summaries:      make(map[string]*domain.Summary),
		maxRecent:      maxRecent,
	}
}

// Add records a new request metric.
func (s *Store) Add(m domain.RequestMetric) {
	s.mu.Lock()
	defer s.mu.Unlock()

	m.PrefillTimeMs = m.TTFBMs
	m.DecodeTimeMs = m.LatencyMs - m.TTFBMs
	if m.DecodeTimeMs < 0 {
		m.DecodeTimeMs = 0
	}

	if m.PrefillTimeMs > 0 && m.PromptTokens > 0 {
		m.PrefillTokPerSec = float64(m.PromptTokens) / (float64(m.PrefillTimeMs) / 1000.0)
	}

	decodeTokens := m.CompletionTokens - 1
	if decodeTokens < 0 {
		decodeTokens = 0
	}
	if m.DecodeTimeMs > 0 && decodeTokens > 0 {
		m.DecodeTokPerSec = float64(decodeTokens) / (float64(m.DecodeTimeMs) / 1000.0)
	}

	if len(s.recentRequests) >= s.maxRecent {
		s.recentRequests = s.recentRequests[1:]
	}
	s.recentRequests = append(s.recentRequests, m)

	s.updateSummary(m.Model, m)
	s.updateSummary("server:"+m.ServerID, m)
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

	if m.PrefillTokPerSec > 0 {
		sum.AvgPrefillTokSec = (sum.AvgPrefillTokSec*float64(sum.TotalPromptTok-m.PromptTokens) + m.PrefillTokPerSec*float64(m.PromptTokens)) / float64(sum.TotalPromptTok)
	}
	if m.DecodeTokPerSec > 0 {
		sum.AvgDecodeTokSec = (sum.AvgDecodeTokSec*float64(sum.TotalCompleteTok-m.CompletionTokens) + m.DecodeTokPerSec*float64(m.CompletionTokens)) / float64(sum.TotalCompleteTok)
	}
}

// Recent returns the last N recorded requests.
func (s *Store) Recent() []domain.RequestMetric {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make([]domain.RequestMetric, len(s.recentRequests))
	copy(out, s.recentRequests)
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

	s.recentRequests = s.recentRequests[:0]
	s.summaries = make(map[string]*domain.Summary)
}
