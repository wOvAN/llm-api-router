package config

import (
	"sync"
	"time"

	"llm-api-router/pkg/log"
)

// tokenEvent is a single token-usage record for a server.
type tokenEvent struct {
	at     time.Time
	tokens int64
}

// QuotaTracker enforces per-server token (TPM) and request (RPM) rate limits
// over a sliding 60s window. Unlike RateLimiter (which reacts to failures),
// QuotaTracker prevents exceeding provider quotas: requests are blocked
// before they are sent when the server's usage is at the limit boundary.
//
// Based on LiteLLM's TPM reservation pattern (#35422).
type QuotaTracker struct {
	mu       sync.Mutex
	tokens   map[string][]tokenEvent          // per-server token events
	requests map[string][]time.Time           // per-server request timestamps
	reserved map[string]int64                 // per-server in-flight reserved tokens
	limits   func(id string) (tpm, rpm int64) // optional lazy limit lookup (0 = unlimited)
}

// NewQuotaTracker creates a quota tracker with an empty sliding window.
func NewQuotaTracker() *QuotaTracker {
	return &QuotaTracker{
		tokens:   make(map[string][]tokenEvent),
		requests: make(map[string][]time.Time),
		reserved: make(map[string]int64),
	}
}

// SetLimitOverride installs a lazy per-server TPM/RPM limit lookup. The
// function is called with a server ID and returns its tokens/min and
// requests/min limits (0 = unlimited). Because it is evaluated lazily on
// every check, runtime config changes (e.g. admin GUI edits) are picked up
// automatically.
func (q *QuotaTracker) SetLimitOverride(fn func(id string) (tpm, rpm int64)) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.limits = fn
}

// Allow reports whether a new request may be sent to the server. A server
// without limits (or with limits not yet reached) is always allowed;
// requests are blocked at the limit boundary. Only the last attempt is
// allowed to exceed the quota (handled by the router).
func (q *QuotaTracker) Allow(id string) bool {
	if q == nil {
		return true
	}
	q.mu.Lock()
	defer q.mu.Unlock()

	tpm, rpm := q.limitsFor(id)
	if tpm <= 0 && rpm <= 0 {
		return true
	}

	now := time.Now()
	cutoff := now.Add(-window60s)

	if rpm > 0 {
		reqs := pruneTimes(q.requests[id], cutoff)
		q.requests[id] = reqs
		if int64(len(reqs)) >= rpm {
			log.Debugf("[quota] %s blocked: %d requests/min (limit %d)", id, len(reqs), rpm)
			return false
		}
	}
	if tpm > 0 {
		evs := pruneEvents(q.tokens[id], cutoff)
		q.tokens[id] = evs
		var used int64
		for _, e := range evs {
			used += e.tokens
		}
		if used >= tpm {
			log.Debugf("[quota] %s blocked: %d tokens/min (limit %d)", id, used, tpm)
			return false
		}
	}
	return true
}

// RecordRequest records that a request was sent to the server.
func (q *QuotaTracker) RecordRequest(id string) {
	if q == nil {
		return
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	now := time.Now()
	q.requests[id] = append(pruneTimes(q.requests[id], now.Add(-window60s)), now)
}

// RecordTokens records the token usage of a completed request.
func (q *QuotaTracker) RecordTokens(id string, tokens int64) {
	if q == nil || tokens <= 0 {
		return
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	now := time.Now()
	q.tokens[id] = append(pruneEvents(q.tokens[id], now.Add(-window60s)), tokenEvent{at: now, tokens: tokens})
}

// Reserve admits a request whose declared output budget is estimate tokens,
// holding it as an in-flight reservation so concurrent requests cannot
// collectively exceed the server's TPM limit. It returns false (blocked,
// nothing reserved) when completed usage plus in-flight reservations plus
// estimate would exceed the TPM limit. estimate <= 0 (no declared budget) and
// servers without a TPM limit are always admitted and reserve nothing. The
// reservation is released by Complete.
func (q *QuotaTracker) Reserve(id string, estimate int64) bool {
	if q == nil || estimate <= 0 {
		return true
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	tpm, _ := q.limitsFor(id)
	if tpm <= 0 {
		return true
	}
	now := time.Now()
	evs := pruneEvents(q.tokens[id], now.Add(-window60s))
	q.tokens[id] = evs
	var used int64
	for _, e := range evs {
		used += e.tokens
	}
	if used+q.reserved[id]+estimate > tpm {
		log.Debugf("[quota] %s blocked (reserved): used=%d reserved=%d estimate=%d tpm=%d",
			id, used, q.reserved[id], estimate, tpm)
		return false
	}
	q.reserved[id] += estimate
	return true
}

// Complete releases a request's in-flight reservation (reservedTokens, 0 if it
// reserved nothing) and records its actual token usage in the sliding window.
// Call it once per sent request, on every exit path (success, mid-stream
// error, client disconnect, or fallback after retries are exhausted).
func (q *QuotaTracker) Complete(id string, reservedTokens, actual int64) {
	if q == nil {
		return
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	if reservedTokens > 0 {
		q.reserved[id] -= reservedTokens
		if q.reserved[id] < 0 {
			q.reserved[id] = 0
		}
	}
	if actual > 0 {
		now := time.Now()
		q.tokens[id] = append(pruneEvents(q.tokens[id], now.Add(-window60s)), tokenEvent{at: now, tokens: actual})
	}
}

// Usage returns the current window usage for a server (tokens, requests).
func (q *QuotaTracker) Usage(id string) (tokens, requests int64) {
	if q == nil {
		return 0, 0
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	cutoff := time.Now().Add(-window60s)
	evs := pruneEvents(q.tokens[id], cutoff)
	for _, e := range evs {
		tokens += e.tokens
	}
	requests = int64(len(pruneTimes(q.requests[id], cutoff)))
	return tokens, requests
}

// limitsFor returns the configured limits for a server (0 = unlimited).
// Callers must hold q.mu.
func (q *QuotaTracker) limitsFor(id string) (tpm, rpm int64) {
	if q.limits != nil {
		return q.limits(id)
	}
	return 0, 0
}

// window60s is the sliding window for TPM/RPM accounting (var for tests).
var window60s = 60 * time.Second

// pruneEvents removes events older than cutoff (in place) and returns the
// still-valid slice.
func pruneEvents(evs []tokenEvent, cutoff time.Time) []tokenEvent {
	valid := evs[:0]
	for _, e := range evs {
		if e.at.After(cutoff) {
			valid = append(valid, e)
		}
	}
	return valid
}

// pruneTimes removes timestamps older than cutoff (in place) and returns the
// still-valid slice.
func pruneTimes(ts []time.Time, cutoff time.Time) []time.Time {
	valid := ts[:0]
	for _, t := range ts {
		if t.After(cutoff) {
			valid = append(valid, t)
		}
	}
	return valid
}
