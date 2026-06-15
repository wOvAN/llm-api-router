package domain

import "time"

// RequestMetric holds performance data for a single proxied request.
type RequestMetric struct {
	Timestamp        time.Time `json:"timestamp"`
	Model            string    `json:"model"`
	TargetModel      string    `json:"target_model"`
	ServerID         string    `json:"server_id"`
	StatusCode       int       `json:"status_code"`
	LatencyMs        int64     `json:"latency_ms"`
	TTFBMs           int64     `json:"ttfb_ms"`
	ResponseSize     int64     `json:"response_size"`
	WasFallback      bool      `json:"was_fallback"`
	PromptTokens     int       `json:"prompt_tokens"`
	CompletionTokens int       `json:"completion_tokens"`
	TotalTokens      int       `json:"total_tokens"`
	CachedTokens     int       `json:"cached_tokens"`

	PrefillTimeMs       int64   `json:"prefill_time_ms"`
	DecodeTimeMs        int64   `json:"decode_time_ms"`
	PrefillTokPerSec    float64 `json:"prefill_tok_per_sec"`
	DecodeTokPerSec     float64 `json:"decode_tok_per_sec"`
	NativePromptMs      float64 `json:"native_prompt_ms"`
	NativePredictedMs   float64 `json:"native_predicted_ms"`
	NativePromptTokPerSec float64 `json:"native_prefill_tok_per_sec"`
	NativeDecodeTokPerSec  float64 `json:"native_decode_tok_per_sec"`
}

// Summary aggregates metrics for a group (by model, server, etc.).
type Summary struct {
	TotalRequests    int     `json:"total_requests"`
	SuccessCount     int     `json:"success_count"`
	ErrorCount       int     `json:"error_count"`
	FallbackCount    int     `json:"fallback_count"`
	AvgLatencyMs     float64 `json:"avg_latency_ms"`
	MinLatencyMs     int64   `json:"min_latency_ms"`
	MaxLatencyMs     int64   `json:"max_latency_ms"`
	AvgTTFBMs        float64 `json:"avg_ttfb_ms"`
	TotalBytes       int64   `json:"total_bytes"`
	TotalPromptTok   int     `json:"total_prompt_tokens"`
	TotalCompleteTok int     `json:"total_completion_tokens"`
	TotalTokens      int     `json:"total_tokens"`
	TotalCachedTok   int     `json:"total_cached_tokens"`
	AvgPrefillTokSec float64 `json:"avg_prefill_tok_per_sec"`
	AvgDecodeTokSec  float64 `json:"avg_decode_tok_per_sec"`
}
