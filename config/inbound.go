package config

import (
	"math"
	"net"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"llm-api-router/domain"
)

// inboundSweepAt bounds the limiter's key map; past it, idle buckets are dropped.
const inboundSweepAt = 4096

// InboundLimiter is a token bucket per client (or one shared bucket) in front of
// the routed endpoints.
//
// The router's rate limiter protects backends from us. Nothing protected the
// router from clients, and on a box with a handful of GPUs one runaway agent loop
// is a full outage: every slot, every request, forever. Denied requests answer
// 429 with the headers a client already reads.
type InboundLimiter struct {
	mu     sync.Mutex
	rps    float64
	burst  float64
	perIP  bool
	keys   map[string]*inboundBucket
	denied atomic.Int64 // requests the bucket turned away (see Denied)
}

type inboundBucket struct {
	tokens float64
	last   time.Time
}

// NewInboundLimiter returns nil when the limit is off (rps <= 0), so the
// default config pays nothing for the feature and a nil limiter is a valid
// pass-through middleware.
func NewInboundLimiter(s domain.InboundSettings) *InboundLimiter {
	if s.RPS <= 0 {
		return nil
	}
	burst := float64(s.Burst)
	if burst < 1 {
		// A fractional rate (0.5 rps = one request every two seconds) rounds the
		// burst up, so the limiter always lets a single request through.
		burst = math.Ceil(s.RPS)
	}
	return &InboundLimiter{rps: s.RPS, burst: burst, perIP: s.PerIP, keys: map[string]*inboundBucket{}}
}

// Denied reports how many requests the bucket has turned away since the process
// started. Shedding traffic is invisible to the operator otherwise: the client
// that got the 429 reads the headers, and the person watching the box sees a
// router with plenty of free GPU and no explanation.
func (l *InboundLimiter) Denied() int64 {
	if l == nil {
		return 0
	}
	return l.denied.Load()
}

// Allow spends one token for a key ("" = the shared bucket). Denied requests
// report how long until a token is back.
func (l *InboundLimiter) Allow(key string) (bool, time.Duration) {
	if l == nil {
		return true, 0
	}
	l.mu.Lock()
	defer l.mu.Unlock()

	b, ok := l.keys[key]
	if !ok {
		if len(l.keys) >= inboundSweepAt {
			l.sweepLocked()
		}
		b = &inboundBucket{tokens: l.burst, last: time.Now()}
		l.keys[key] = b
	}
	now := time.Now()
	b.tokens = min(b.tokens+now.Sub(b.last).Seconds()*l.rps, l.burst)
	b.last = now
	if b.tokens >= 1 {
		b.tokens--
		return true, 0
	}
	l.denied.Add(1)
	return false, time.Duration(math.Ceil((1 - b.tokens) / l.rps * float64(time.Second)))
}

// sweepLocked drops buckets idle for two full refills — they are back at the
// burst cap, so forgetting them costs nothing. Caller holds l.mu.
func (l *InboundLimiter) sweepLocked() {
	cutoff := time.Duration(2 * float64(time.Second) * l.burst / l.rps)
	for k, b := range l.keys {
		if time.Since(b.last) > cutoff {
			delete(l.keys, k)
		}
	}
}

// Middleware enforces the bucket. X-Forwarded-For is deliberately not consulted:
// a client can name any address it likes, and per-IP fairness keyed on a
// forgable key is no fairness at all.
func (l *InboundLimiter) Middleware(next http.Handler) http.Handler {
	if l == nil {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		key := ""
		if l.perIP {
			if host, _, err := net.SplitHostPort(req.RemoteAddr); err == nil {
				key = host
			} else {
				key = req.RemoteAddr
			}
		}
		ok, retry := l.Allow(key)
		if !ok {
			seconds := max(int(retry.Seconds()+0.5), 1)
			w.Header().Set("X-RateLimit-Limit", strconv.FormatFloat(l.burst, 'f', -1, 64))
			w.Header().Set("X-RateLimit-Remaining", "0")
			w.Header().Set("X-RateLimit-Reset", strconv.Itoa(seconds))
			w.Header().Set("Retry-After", strconv.Itoa(seconds))
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"type":"rate_limit_error","message":"router rate limit reached"}}`))
			return
		}
		w.Header().Set("X-RateLimit-Limit", strconv.FormatFloat(l.burst, 'f', -1, 64))
		w.Header().Set("X-RateLimit-Remaining", "1")
		next.ServeHTTP(w, req)
	})
}
