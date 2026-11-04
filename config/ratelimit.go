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
	reason      map[string]string             // why each server is cooling down
	maxFails    int                           // server-side failures within window to trigger cooldown
	maxFails4xx int                           // lenient threshold for client-side (other 4xx) failures
	window      time.Duration                 // time window for counting failures
	cooldownDur time.Duration                 // how long to skip the server
	cooldownFor func(id string) time.Duration // optional per-server cooldown override (0 = global)
	alerter     Alerter                       // cooldown starts leave the process here
}

// NewRateLimiter creates a rate limiter with the given parameters.
// Typical values: maxFails=5, window=60s, cooldownDur=5min.
// Client-side (other 4xx) failures use a lenient threshold of 2*maxFails.
func NewRateLimiter(maxFails int, window, cooldownDur time.Duration) *RateLimiter {
	return &RateLimiter{
		failures:    make(map[string][]failEntry),
		cooldown:    make(map[string]time.Time),
		reason:      make(map[string]string),
		maxFails:    maxFails,
		maxFails4xx: maxFails * 2,
		window:      window,
		cooldownDur: cooldownDur,
	}
}

// SetThresholds moves the global failure threshold and cooldown duration (0 keeps
// the current value), so an operator can keep a flaky-but-recovering backend in
// rotation without a restart. A per-server cooldown_time still beats the global
// cooldown, and a backend's Retry-After beats the global default.
func (rl *RateLimiter) SetThresholds(maxFails int, cooldown time.Duration) {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	if maxFails > 0 {
		rl.maxFails = maxFails
		rl.maxFails4xx = maxFails * 2
	}
	if cooldown > 0 {
		rl.cooldownDur = cooldown
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

// SetAlerter installs the out-of-band notice for cooldown starts (nil = none) —
// the one router-side event an operator cannot see after the fact: the router
// quietly stopped sending traffic to that backend.
func (rl *RateLimiter) SetAlerter(a Alerter) {
	if rl == nil {
		return
	}
	rl.mu.Lock()
	defer rl.mu.Unlock()
	rl.alerter = a
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

// hasCooldownOverride reports whether the server has an explicit per-server
// cooldown configured (as opposed to falling through to the global default via a
// nil/zero override). Callers must hold rl.mu.
func (rl *RateLimiter) hasCooldownOverride(id string) bool {
	return rl.cooldownFor != nil && rl.cooldownFor(id) > 0
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
	delete(rl.reason, id)
}

// RecordFailure records a failed request to a server with its HTTP status
// code (0 for transport/network errors).
//
// Status-aware behavior:
//   - 429, 401, 403 → immediate cooldown on the first occurrence
//   - transport errors (status 0) and 5xx → cooldown after maxFails failures in window
//   - other 4xx → cooldown after maxFails4xx failures in window
//
// A trailing retryAfter is the backend's own Retry-After hint (0/absent = none):
// for an immediate-cooldown status it replaces the configured duration, because
// the backend just told us when to come back and a 5min default would bench it
// for ten times its own stated pause. Clamped to [1s, 1h] so a hostile or broken
// header can neither park a server for a day nor for nothing.
//
// Returns true if the server was just put into cooldown.
func (rl *RateLimiter) RecordFailure(id string, status int, retryAfter ...time.Duration) bool {
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
			reason := "failures"
			dur := rl.cooldownDuration(id)
			if kind == kindImmediate {
				reason = "status " + http.StatusText(status)
				// Precedence (LiteLLM's): an operator's explicit per-server
				// cooldown_time outranks a header any backend can send; the hint
				// replaces only the global default.
				if hint := clampHint(retryAfter); hint > 0 && !rl.hasCooldownOverride(id) {
					dur = hint
				}
			}
			rl.cooldown[id] = now.Add(dur)
			rl.reason[id] = reason
			log.Warnf("[ratelimit] %s put into cooldown (%s, %d in %v, cooldown %v)",
				id, reason, count, rl.window, dur)
			// Already edge-triggered: the branch above is reached only when the
			// server was not already cooling down.
			if rl.alerter != nil {
				rl.alerter.Notify("server " + id + " cooling down (" + reason + ")")
			}
			return true
		}
	}
	return false
}

// clampHint bounds a backend-supplied Retry-After hint. Callers must hold rl.mu
// (it reads no state; it lives here so the cooldown path has one clamp).
func clampHint(hints []time.Duration) time.Duration {
	if len(hints) == 0 {
		return 0
	}
	d := hints[0]
	switch {
	case d <= 0:
		return 0
	case d < time.Second:
		return time.Second
	case d > time.Hour:
		return time.Hour
	}
	return d
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
	delete(rl.reason, id)
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

// Cooldown reports when a server's cooldown expires and what put it there
// ("status Too Many Requests", "status Unauthorized", "failures"). The ok
// result is false for a server that is not cooling down. This is what the admin
// API/GUI show, so a server skipped for a bad api_key is visible as that, and
// not as an unexplained healthy-and-idle one.
func (rl *RateLimiter) Cooldown(id string) (time.Time, string, bool) {
	if rl == nil {
		return time.Time{}, "", false
	}
	rl.mu.RLock()
	defer rl.mu.RUnlock()
	expiry, ok := rl.cooldown[id]
	if !ok || !time.Now().Before(expiry) {
		return time.Time{}, "", false
	}
	return expiry, rl.reason[id], true
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
