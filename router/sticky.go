package router

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"time"
)

// --- conversation pinning (prompt-prefix-cache affinity) ---
//
// A pool of replicas of one model is a set of separate KV caches. Rotating a
// multi-turn conversation across them re-prefills the whole conversation on
// every turn, because the prefix turn N-1 cached lives on the replica that
// served turn N-1. So a conversation is pinned to the member that served its
// first turn, and only conversations that share no key rotate. (LiteLLM:
// deployment_affinity/session_affinity; olla: sticky_sessions.)

const (
	// stickyTTL is how long a conversation keeps its pin, refreshed on every
	// request (an agent loop running for an hour must not be repinned
	// mid-flight; an abandoned conversation must not hold a slot forever).
	stickyTTL = 30 * time.Minute
	// stickySweepAt bounds the store: past it, expired pins are dropped.
	stickySweepAt = 4096
)

// sessionHeaders are the header names clients put a conversation id in. The
// big-agent CLIs send their own (Claude Code: x-claude-code-session-id), the
// OpenAI-family ones send a thread/conversation id, and underscore spellings
// survive net/http canonicalization unchanged, so both forms are listed.
var sessionHeaders = []string{
	"X-Claude-Code-Session-Id",
	"X-Session-Id", "Session-Id", "Session_id",
	"X-Thread-Id", "Thread-Id",
	"X-Conversation-Id", "Conversation-Id",
	"X-Openai-Conversation-Id",
}

// conversationKey identifies the conversation a request belongs to: a session
// header, else the body's own key (prompt_cache_key, user, metadata.user_id
// session_id, first user turn — see jevSessionKey), hashed so an unbounded
// client string can not inflate the pin store. "" when the request carries
// none, which is the same as "do not pin me".
func conversationKey(req *http.Request, body []byte) string {
	for _, h := range sessionHeaders {
		if v := req.Header.Get(h); v != "" {
			return hashKey("h", v)
		}
	}
	if k := jevSessionKey(body); k != "" {
		return hashKey("b", k)
	}
	return ""
}

func hashKey(kind, key string) string {
	sum := sha256.Sum256([]byte(kind + ":" + key))
	return hex.EncodeToString(sum[:])
}

type stickyPin struct {
	serverID string
	expires  time.Time
}

// stickyGet returns the backend pinned to a conversation, "" when none. Reading
// a pin refreshes its expiry (a live conversation is a sliding window).
func (r *Router) stickyGet(key string) string {
	if key == "" {
		return ""
	}
	r.pinMu.Lock()
	defer r.pinMu.Unlock()
	p, ok := r.pins[key]
	if !ok {
		return ""
	}
	if time.Now().After(p.expires) {
		delete(r.pins, key)
		return ""
	}
	p.expires = time.Now().Add(stickyTTL)
	r.pins[key] = p
	return p.serverID
}

// stickyPut pins a conversation to the backend that served it.
func (r *Router) stickyPut(key, serverID string) {
	if key == "" || serverID == "" {
		return
	}
	r.pinMu.Lock()
	defer r.pinMu.Unlock()
	if len(r.pins) >= stickySweepAt {
		now := time.Now()
		for k, p := range r.pins {
			if now.After(p.expires) {
				delete(r.pins, k)
			}
		}
	}
	r.pins[key] = stickyPin{serverID: serverID, expires: time.Now().Add(stickyTTL)}
}
