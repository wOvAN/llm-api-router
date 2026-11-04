package domain

import "slices"

// APIType represents the protocol type of a backend server.
type APIType string

const (
	APITypeOpenAI    APIType = "openai"
	APITypeAnthropic APIType = "anthropic"
)

// ServerProtocol declares one endpoint a backend serves. State records whether the
// backend keeps server-side conversation state (`previous_response_id`), which no
// local engine does by default; the router never assumes it. Declared scaffolding
// for the capability-probe phase — no consumer reads it yet, and the `state`
// refusal is planned with it (the `convert.CodeState` code is its twin).
type ServerProtocol struct {
	Path  string `json:"path"`
	State bool   `json:"state,omitempty"`
}

// Server defines an LLM inference backend.
type Server struct {
	ID           string `json:"id"`
	Name         string `json:"name"`
	URL          string `json:"url"`
	OpenAIURL    string `json:"openai_url,omitempty"`
	AnthropicURL string `json:"anthropic_url,omitempty"`
	// Proxy is the HTTP/HTTPS/SOCKS proxy used to reach this server ("" = direct).
	Proxy string `json:"proxy,omitempty"`
	// ProxyEnabled toggles proxy usage. nil = disabled (default); explicit
	// true enables the proxy, so a proxy URL alone never routes traffic
	// through it.
	ProxyEnabled *bool     `json:"proxy_enabled,omitempty"`
	APIKey       string    `json:"api_key"`
	APITypes     []APIType `json:"api_types"`
	// Protocols declares the endpoints this backend natively serves, which is what
	// decides whether a request has to be converted (see convert.Decide). Empty
	// means "undeclared": the router keeps its passthrough behavior, so a config
	// written before this field exists never changes behavior.
	Protocols []ServerProtocol `json:"protocols,omitempty"`
	// CooldownTime overrides the rate-limiter cooldown duration for this server
	// (seconds). 0 = use the global default.
	CooldownTime int64 `json:"cooldown_time,omitempty"`
	// TPMLimit caps tokens-per-minute sent to this server (0 = unlimited).
	// Enforced by QuotaTracker over a sliding 60s window.
	TPMLimit int64 `json:"tpm_limit,omitempty"`
	// RPMLimit caps requests-per-minute sent to this server (0 = unlimited).
	RPMLimit int64 `json:"rpm_limit,omitempty"`
	// MaxParallelRequests is how many generations this backend serves at once
	// (0 = unlimited). A server with N slots does not refuse the N+1-th request,
	// it queues it, and queue time is invisible to the caller — so the operator's
	// knowledge of the launch command (`llama-server -np 4`) is what turns a
	// saturated replica into a routing signal.
	MaxParallelRequests int `json:"max_parallel_requests,omitempty"`
	// KeepAliveIdle overrides the SSE keep-alive heartbeat interval for this
	// server (seconds). 0 = use the global proxy.KeepAliveIdle. A per-request
	// X-Router-KeepAlive header still overrides this.
	KeepAliveIdle int `json:"keep_alive_idle,omitempty"`
	// Headers are static headers stamped on every upstream request to this server
	// (a gateway that requires a tenant id or a routing tag). The configured value
	// overwrites the client's own header of the same name; the client's other
	// headers ride upstream untouched, as before.
	Headers map[string]string `json:"headers,omitempty"`
}

// GetURLForAPIType returns the URL to use for the given API type.
func (s *Server) GetURLForAPIType(t APIType) string {
	switch t {
	case APITypeOpenAI:
		if s.OpenAIURL != "" {
			return s.OpenAIURL
		}
	case APITypeAnthropic:
		if s.AnthropicURL != "" {
			return s.AnthropicURL
		}
	}
	return s.URL
}

// ProxyURL returns the HTTP proxy to use for this server ("" = direct).
// The proxy is used only when explicitly enabled (ProxyEnabled true);
// otherwise the proxy URL is ignored.
func (s *Server) ProxyURL() string {
	if s.ProxyEnabled == nil || !*s.ProxyEnabled {
		return ""
	}
	return s.Proxy
}

// SupportsAPIType checks if the server supports the given API type.
func (s *Server) SupportsAPIType(t APIType) bool {
	return slices.Contains(s.APITypes, t)
}

// FallbackEntry defines a fallback server with an optional per-server target model.
type FallbackEntry struct {
	ServerID    string `json:"server_id"`
	TargetModel string `json:"target_model,omitempty"`
	// Enabled toggles whether this fallback is used. nil = enabled (default,
	// preserving configs that predate the field); explicit false disables it.
	Enabled *bool `json:"enabled,omitempty"`
	// Priority orders fallbacks: lower number is tried first. 0 = unset, which
	// sorts after all prioritized fallbacks (original list order kept among ties).
	Priority int `json:"priority,omitempty"`
	// Convert overrides protocol conversion on this fallback: ""/"auto" (decide by
	// the target's declared protocols), "off" (never convert), or an explicit
	// "messages>chat".
	Convert string `json:"convert,omitempty"`
	// ContextWindow is the context size this member declares (tokens). Like
	// RoutingRule.ContextWindow it is exposed in /v1/models, and the router uses
	// it to skip an attempt whose window can not fit the request's estimated
	// prompt (0 = unknown = never skipped).
	ContextWindow int `json:"context_window,omitempty"`
	// Weight is this member's share of the traffic when the rule balances
	// (0 = 1, and a rule that does not balance ignores it entirely — there,
	// order is the whole story). A 4-GPU replica beside a 1-GPU one declares 4.
	Weight int `json:"weight,omitempty"`
}

// IsEnabled reports whether the fallback is active. A nil Enabled means enabled.
func (f FallbackEntry) IsEnabled() bool {
	return f.Enabled == nil || *f.Enabled
}
