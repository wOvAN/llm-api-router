package domain

import "strings"

// Router kinds for RoutingRule.Router. Empty (RouterStatic) routes to the
// rule's own ServerID/TargetModel/Fallbacks; RouterJev asks a System One
// decision model to pick one of the rule's Jev candidates per request.
const (
	RouterStatic = ""
	RouterJev    = "jev"
)

// BalanceRoundRobin is the one RoutingRule.Balance mode: the rule's own server
// and its enabled fallbacks are an interchangeable pool, and successive requests
// start on the next member of it.
const BalanceRoundRobin = "round-robin"

// RoutingRule maps one or more incoming model names to a backend server and target model.
type RoutingRule struct {
	IncomingModels    []string        `json:"incoming_models"`
	TargetModel       string          `json:"target_model"`
	ServerID          string          `json:"server_id"`
	Fallbacks         []FallbackEntry `json:"fallbacks,omitempty"`
	FallbackServerIDs []string        `json:"fallback_server_ids,omitempty"`
	NumRetries        int             `json:"num_retries,omitempty"`
	Enabled           bool            `json:"enabled"`
	// ContextWindow is the maximum context size in tokens, exposed via
	// /v1/models so clients (Claude Code etc.) can size their prompts.
	// 0 = unknown (omitted from /v1/models).
	ContextWindow int `json:"context_window,omitempty"`
	// Router selects how the target is chosen: "" (static) or "jev"
	// (per-request decision by a System One model, see JevRouter).
	Router string `json:"router,omitempty"`
	// Jev is the auto-routing config, used when Router == "jev".
	Jev *JevRouter `json:"jev,omitempty"`
	// Convert overrides protocol conversion on this rule's primary attempt:
	// ""/"auto" (decide by the target's declared protocols), "off" (never
	// convert), or an explicit "messages>chat".
	Convert string `json:"convert,omitempty"`
	// Balance distributes traffic across the rule's server and its enabled
	// fallbacks instead of treating them as a failover chain: empty (the
	// default, and every config that predates the field) keeps primary-first
	// order, "round-robin" starts each request on the next pool member. The
	// list order stays the failover order after that start, and
	// FallbackEntry.Weight decides how often each member starts.
	Balance string `json:"balance,omitempty"`
}

// IsJev reports whether the rule routes via a decision model.
func (r *RoutingRule) IsJev() bool { return r.Router == RouterJev && r.Jev != nil }

// ServerRef is one backend a rule names, plus the role it plays there.
type ServerRef struct {
	ID   string
	Role string
}

// ServerRefs lists every backend the rule names: its primary, its enabled
// fallbacks, and for a jev rule the decision server and the candidate pool.
// Routing reads these one at a time; the admin API and the startup audit read
// them all, so a delete that would strand a rule can be refused and a
// hand-edited config that names a dead server can be reported.
func (r *RoutingRule) ServerRefs() []ServerRef {
	refs := make([]ServerRef, 0, 1+len(r.Fallbacks))
	if r.ServerID != "" {
		refs = append(refs, ServerRef{ID: r.ServerID, Role: "primary server"})
	}
	for _, fb := range r.Fallbacks {
		refs = append(refs, ServerRef{ID: fb.ServerID, Role: "fallback server"})
	}
	if r.Jev != nil {
		if r.Jev.ServerID != "" {
			refs = append(refs, ServerRef{ID: r.Jev.ServerID, Role: "jev decision server"})
		}
		for _, c := range r.Jev.Candidates {
			refs = append(refs, ServerRef{ID: c.ServerID, Role: "jev candidate"})
		}
	}
	return refs
}

// IsModelPattern reports whether an incoming-model entry is a wildcard pattern
// (`gpt-*`, `claude-?`) rather than a model name. A pattern lets one rule own a
// whole model family; the client's own name is then what the backend gets
// (pass-through, like a rule with no target model).
func IsModelPattern(name string) bool { return strings.ContainsAny(name, "*?") }

// JevRouter routes one rule with a System One decision model (llama.cpp's
// /v1/systemone with a decision model, a local OpenJev server, or the hosted
// TypeSafe/OpenRouter decisions API). The model reads the request and answers a
// single Choice question over Candidates; the router sends the request to the
// chosen candidate.
//
// Fail-open, like every Jev integration: a missing/unreachable decision server,
// a timeout, a malformed answer, or a choice outside the pool lands on
// DefaultTier, so routing never blocks a request.
type JevRouter struct {
	// ServerID is the decision backend (its URL + APIKey are used for
	// POST /v1/systemone).
	ServerID string `json:"server_id"`
	// Model is the decision model name sent in the request ("laya",
	// "jev-latest", ...). Empty = let the backend decide.
	Model string `json:"model,omitempty"`
	// Question overrides the Choice question's instructions.
	Question string `json:"question,omitempty"`
	// Candidates is the pool the decision model chooses from.
	Candidates []JevCandidate `json:"candidates"`
	// MinConfidence is the answer confidence below which the choice is
	// clamped toward DefaultTier (0 = default 0.3).
	MinConfidence float64 `json:"min_confidence,omitempty"`
	// DefaultTier is the candidate Tier used when Jev is unavailable or unsure
	// (the pool's tier the request lands on without a usable decision).
	DefaultTier int `json:"default_tier,omitempty"`
	// TimeoutMs bounds the decision request (0 = default 1500).
	TimeoutMs int `json:"timeout_ms,omitempty"`
	// StateMaxChars caps the request text sent to the decision model
	// (0 = default 8000).
	StateMaxChars int `json:"state_max_chars,omitempty"`
	// Security adds a Noul "how harmful is this request" question to the same
	// decision call; an answer at or above the gate threshold (0.95) blocks the
	// request with HTTP 400 instead of routing it.
	Security bool `json:"security,omitempty"`
	// CacheTTL caches the decision for identical requests (same decision model,
	// question, pool, request text) for this many seconds; 0 = off.
	CacheTTL int `json:"cache_ttl,omitempty"`
	// Images forwards the last user turn's images (data URLs, max 8) to the
	// decision model, so a vision decision model (Clef/OpenJev) can judge them.
	// Off by default: image tokens make every decision expensive and non-vision
	// decision backends reject them.
	Images bool `json:"images,omitempty"`
}

// JevCandidate is one model the decision model may pick. Tier orders the pool
// cheapest-first (0 = cheapest); it is the router's policy axis, not Jev's.
type JevCandidate struct {
	ServerID    string `json:"server_id"`
	TargetModel string `json:"target_model,omitempty"`
	// Description tells the decision model what this candidate is good at.
	// Empty = the target model name alone.
	Description string `json:"description,omitempty"`
	Tier        int    `json:"tier,omitempty"`
	// ContextWindow is the candidate's context size; a jev rule without its own
	// context_window reports the pool minimum in /v1/models (0 = unknown).
	ContextWindow int `json:"context_window,omitempty"`
	// Vision marks whether this candidate accepts image input. nil = unknown
	// (kept in image-request pools), false = blind (removed from them).
	Vision *bool `json:"vision,omitempty"`
	// Enabled toggles the candidate. nil = enabled.
	Enabled *bool `json:"enabled,omitempty"`
	// Convert overrides protocol conversion on this candidate: ""/"auto" (decide by
	// the target's declared protocols), "off" (never convert), or an explicit
	// "messages>chat".
	Convert string `json:"convert,omitempty"`
}

// IsEnabled reports whether the candidate is in the pool.
func (c JevCandidate) IsEnabled() bool { return c.Enabled == nil || *c.Enabled }

// EnabledCandidates returns the pool's enabled candidates.
func (j *JevRouter) EnabledCandidates() []JevCandidate {
	out := make([]JevCandidate, 0, len(j.Candidates))
	for _, c := range j.Candidates {
		if c.IsEnabled() {
			out = append(out, c)
		}
	}
	return out
}
