package config

import (
	"sync"
	"time"

	"llm-api-router/pkg/log"
)

// RateLimiter tracks per-server request failures and enforces cooldown
// periods when a server exceeds the failure threshold.
//
// Unlike HealthTracker (which pings /v1/models periodically), RateLimiter
// reacts to actual request failures. This provides faster protection
// against temporarily overloaded backends.
//
// Based on LiteLLM's fail_calls/cooldown_cache pattern.
type RateLimiter struct {
	mu         sync.RWMutex
	failures   map[string][]time.Time // per-server failure timestamps
	cooldown   map[string]time.Time   // per-server cooldown expiry
	maxFails   int                    // failures within window to trigger cooldown
	window     time.Duration          // time window for counting failures
	cooldownDur time.Duration         // how long to skip the server
}

// NewRateLimiter creates a rate limiter with the given parameters.
// Typical values: maxFails=5, window=60s, cooldownDur=5min.
func NewRateLimiter(maxFails int, window, cooldownDur time.Duration) *RateLimiter {
	return &RateLimiter{
		failures:    make(map[string][]time.Time),
		cooldown:    make(map[string]time.Time),
		maxFails:    maxFails,
		window:      window,
		cooldownDur: cooldownDur,
	}
}

// ShouldSkip reports whether the server should be skipped due to cooldown.
func (rl *RateLimiter) ShouldSkip(id string) bool {
	if rl == nil {
		return false
	}
	rl.mu.RLock()
	defer rl.mu.RUnlock()
	expiry, ok := rl.cooldown[id]
	if !ok {
		return false
	}
	return time.Now().Before(expiry)
}

// RecordSuccess records a successful request to a server, clearing its failure count.
func (rl *RateLimiter) RecordSuccess(id string) {
	if rl == nil {
		return
	}
	rl.mu.Lock()
	defer rl.mu.Unlock()
	delete(rl.failures, id)
	delete(rl.cooldown, id)
}

// RecordFailure records a failed request to a server.
// If the failure count within the window exceeds the threshold, the server
// is put into cooldown. Returns true if the server was just put into cooldown.
func (rl *RateLimiter) RecordFailure(id string) bool {
	if rl == nil {
		return false
	}
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	cutoff := now.Add(-rl.window)

	// Add this failure
	fails := rl.failures[id]
	// Remove old failures outside the window
	valid := fails[:0]
	for _, t := range fails {
		if t.After(cutoff) {
			valid = append(valid, t)
		}
	}
	valid = append(valid, now)
	rl.failures[id] = valid

	// Check if threshold exceeded
	if len(valid) >= rl.maxFails {
		expiry, wasCooling := rl.cooldown[id]
		if !wasCooling || now.After(expiry) {
			// First time or cooldown expired — re-trigger
			rl.cooldown[id] = now.Add(rl.cooldownDur)
			log.Warnf("[ratelimit] %s put into cooldown (%d failures in %v, cooldown %v)",
				id, len(valid), rl.window, rl.cooldownDur)
			return true
		}
	}
	return false
}

// ClearCooldown removes the cooldown for a server (e.g., after health check passes).
func (rl *RateLimiter) ClearCooldown(id string) {
	if rl == nil {
		return
	}
	rl.mu.Lock()
	defer rl.mu.Unlock()
	delete(rl.cooldown, id)
	delete(rl.failures, id)
}

// CooldownRemaining returns the remaining cooldown time for a server.
// Returns 0 if the server is not in cooldown.
func (rl *RateLimiter) CooldownRemaining(id string) time.Duration {
	rl.mu.RLock()
	defer rl.mu.RUnlock()
	expiry, ok := rl.cooldown[id]
	if !ok {
		return 0
	}
	remaining := time.Since(expiry)
	if remaining > 0 {
		return 0
	}
	return -remaining
}

// FailureCount returns the number of recent failures for a server.
func (rl *RateLimiter) FailureCount(id string) int {
	rl.mu.RLock()
	defer rl.mu.RUnlock()
	cutoff := time.Now().Add(-rl.window)
	count := 0
	for _, t := range rl.failures[id] {
		if t.After(cutoff) {
			count++
		}
	}
	return count
}
