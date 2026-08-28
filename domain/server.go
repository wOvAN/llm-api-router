package domain

import "slices"

// APIType represents the protocol type of a backend server.
type APIType string

const (
	APITypeOpenAI    APIType = "openai"
	APITypeAnthropic APIType = "anthropic"
)

// Server defines an LLM inference backend.
type Server struct {
	ID           string `json:"id"`
	Name         string `json:"name"`
	URL          string `json:"url"`
	OpenAIURL    string `json:"openai_url,omitempty"`
	AnthropicURL string `json:"anthropic_url,omitempty"`
	// Proxy is the HTTP/HTTPS proxy used to reach this server ("" = direct).
	// OpenAIProxy/AnthropicProxy override it per protocol.
	Proxy          string    `json:"proxy,omitempty"`
	OpenAIProxy    string    `json:"openai_proxy,omitempty"`
	AnthropicProxy string    `json:"anthropic_proxy,omitempty"`
	APIKey         string    `json:"api_key"`
	APITypes       []APIType `json:"api_types"`
	// CooldownTime overrides the rate-limiter cooldown duration for this server
	// (seconds). 0 = use the global default.
	CooldownTime int64 `json:"cooldown_time,omitempty"`
	// TPMLimit caps tokens-per-minute sent to this server (0 = unlimited).
	// Enforced by QuotaTracker over a sliding 60s window.
	TPMLimit int64 `json:"tpm_limit,omitempty"`
	// RPMLimit caps requests-per-minute sent to this server (0 = unlimited).
	RPMLimit int64 `json:"rpm_limit,omitempty"`
	// KeepAliveIdle overrides the SSE keep-alive heartbeat interval for this
	// server (seconds). 0 = use the global proxy.KeepAliveIdle. A per-request
	// X-Router-KeepAlive header still overrides this.
	KeepAliveIdle int `json:"keep_alive_idle,omitempty"`
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

// GetProxyForAPIType returns the HTTP proxy to use for the given API type:
// the protocol-specific override if set, else the base proxy, else "" (direct).
func (s *Server) GetProxyForAPIType(t APIType) string {
	switch t {
	case APITypeOpenAI:
		if s.OpenAIProxy != "" {
			return s.OpenAIProxy
		}
	case APITypeAnthropic:
		if s.AnthropicProxy != "" {
			return s.AnthropicProxy
		}
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
}

// IsEnabled reports whether the fallback is active. A nil Enabled means enabled.
func (f FallbackEntry) IsEnabled() bool {
	return f.Enabled == nil || *f.Enabled
}
