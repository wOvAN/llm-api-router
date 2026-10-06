package domain

// Router kinds for RoutingRule.Router. Empty (RouterStatic) routes to the
// rule's own ServerID/TargetModel/Fallbacks; RouterJev asks a System One
// decision model to pick one of the rule's Jev candidates per request.
const (
	RouterStatic = ""
	RouterJev    = "jev"
)

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
}

// IsJev reports whether the rule routes via a decision model.
func (r *RoutingRule) IsJev() bool { return r.Router == RouterJev && r.Jev != nil }

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
	// Enabled toggles the candidate. nil = enabled.
	Enabled *bool `json:"enabled,omitempty"`
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
