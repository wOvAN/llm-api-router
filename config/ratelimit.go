package config

import (
	"net/http"
	"sync"
	"time"

	"llm-api-router/pkg/log"
)

// failKind classifies a recorded failure. Different kinds use different
// thresholds and cooldown behavior, mirroring LiteLLM's cooldown_handlers:
//
//   - kindImmediate (429/401/403): the server is rate-limited or rejects us —
//     cooldown immediately on the first occurrence.
//   - kindServer (transport errors, 5xx): server-side problems — threshold
//     is maxFails failures within the window.
//   - kindClient (other 4xx): mostly bad requests — lenient threshold
//     (maxFails4xx) so a misbehaving client can't kill a healthy server.
type failKind int

const (
	kindImmediate failKind = iota
	kindServer
	kindClient
)

// failEntry is a single recorded failure for a server.
type failEntry struct {
	at   time.Time
	kind failKind
}

// RateLimiter tracks per-server request failures and enforces cooldown
// periods when a server exceeds the failure threshold.
//
// Unlike HealthTracker (which pings /v1/models periodically), RateLimiter
// reacts to actual request failures. This provides faster protection
// against temporarily overloaded backends.
//
// Based on LiteLLM's fail_calls/cooldown_cache pattern.
type RateLimiter struct {
	mu          sync.RWMutex
	failures    map[string][]failEntry        // per-server failure entries
	cooldown    map[string]time.Time          // per-server cooldown expiry
	maxFails    int                           // server-side failures within window to trigger cooldown
	maxFails4xx int                           // lenient threshold for client-side (other 4xx) failures
	window      time.Duration                 // time window for counting failures
	cooldownDur time.Duration                 // how long to skip the server
	cooldownFor func(id string) time.Duration // optional per-server cooldown override (0 = global)
}

// NewRateLimiter creates a rate limiter with the given parameters.
// Typical values: maxFails=5, window=60s, cooldownDur=5min.
// Client-side (other 4xx) failures use a lenient threshold of 2*maxFails.
func NewRateLimiter(maxFails int, window, cooldownDur time.Duration) *RateLimiter {
	return &RateLimiter{
		failures:    make(map[string][]failEntry),
		cooldown:    make(map[string]time.Time),
		maxFails:    maxFails,
		maxFails4xx: maxFails * 2,
		window:      window,
		cooldownDur: cooldownDur,
	}
}

// SetCooldownOverride installs a lazy per-server cooldown duration lookup.
// The function is called with a server ID and returns the cooldown duration
// for that server; a non-positive value falls back to the global default.
// Because it is evaluated lazily on every cooldown trigger, runtime config
// changes (e.g. admin GUI edits) are picked up automatically.
func (rl *RateLimiter) SetCooldownOverride(fn func(id string) time.Duration) {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	rl.cooldownFor = fn
}

// cooldownDuration returns the cooldown duration for a server (override or global).
// Callers must hold rl.mu.
func (rl *RateLimiter) cooldownDuration(id string) time.Duration {
	if rl.cooldownFor != nil {
		if d := rl.cooldownFor(id); d > 0 {
			return d
		}
	}
	return rl.cooldownDur
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

// RecordFailure records a failed request to a server with its HTTP status
// code (0 for transport/network errors).
//
// Status-aware behavior:
//   - 429, 401, 403 → immediate cooldown on the first occurrence
//   - transport errors (status 0) and 5xx → cooldown after maxFails failures in window
//   - other 4xx → cooldown after maxFails4xx failures in window
//
// Returns true if the server was just put into cooldown.
func (rl *RateLimiter) RecordFailure(id string, status int) bool {
	if rl == nil {
		return false
	}
	rl.mu.Lock()
	defer rl.mu.Unlock()

	kind := classifyFailure(status)
	now := time.Now()
	cutoff := now.Add(-rl.window)

	// Add this failure
	entries := rl.failures[id]
	// Remove old failures outside the window
	valid := entries[:0]
	for _, f := range entries {
		if f.at.After(cutoff) {
			valid = append(valid, f)
		}
	}
	valid = append(valid, failEntry{at: now, kind: kind})
	rl.failures[id] = valid

	threshold := rl.thresholdFor(id, kind)
	count := 0
	for _, f := range valid {
		if f.kind == kind {
			count++
		}
	}

	// Immediate cooldown statuses trigger on the first occurrence.
	if kind == kindImmediate || count >= threshold {
		expiry, wasCooling := rl.cooldown[id]
		if !wasCooling || now.After(expiry) {
			dur := rl.cooldownDuration(id)
			rl.cooldown[id] = now.Add(dur)
			reason := "failures"
			if kind == kindImmediate {
				reason = "status " + http.StatusText(status)
			}
			log.Warnf("[ratelimit] %s put into cooldown (%s, %d in %v, cooldown %v)",
				id, reason, count, rl.window, dur)
			return true
		}
	}
	return false
}

// classifyFailure maps an HTTP status (0 = transport) to a failKind.
func classifyFailure(status int) failKind {
	switch status {
	case http.StatusTooManyRequests, http.StatusUnauthorized, http.StatusForbidden:
		return kindImmediate
	case 0: // transport / network error
		return kindServer
	}
	if status >= 500 {
		return kindServer
	}
	return kindClient
}

// thresholdFor returns the failure threshold for a server's failKind.
// Callers must hold rl.mu.
func (rl *RateLimiter) thresholdFor(id string, kind failKind) int {
	if kind == kindClient {
		return rl.maxFails4xx
	}
	return rl.maxFails
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
	if rl == nil {
		return 0
	}
	rl.mu.RLock()
	defer rl.mu.RUnlock()
	expiry, ok := rl.cooldown[id]
	if !ok {
		return 0
	}
	remaining := time.Until(expiry)
	if remaining <= 0 {
		return 0
	}
	return remaining
}

// FailureCount returns the number of recent failures for a server.
func (rl *RateLimiter) FailureCount(id string) int {
	if rl == nil {
		return 0
	}
	rl.mu.RLock()
	defer rl.mu.RUnlock()
	cutoff := time.Now().Add(-rl.window)
	count := 0
	for _, f := range rl.failures[id] {
		if f.at.After(cutoff) {
			count++
		}
	}
	return count
}
