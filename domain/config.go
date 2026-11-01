package domain

// RuleProfile is a named set of routing rules. Exactly one profile is active
// at a time; the router routes using the active profile's rules.
type RuleProfile struct {
	ID    string         `json:"id"`
	Name  string         `json:"name"`
	Rules []*RoutingRule `json:"rules"`
}

// Config holds the full router configuration.
type Config struct {
	Servers         map[string]*Server `json:"servers"`
	Profiles        []RuleProfile      `json:"profiles"`
	ActiveProfileID string             `json:"active_profile_id,omitempty"`
	Settings        Settings           `json:"settings"`
}

// Settings holds global router behavior toggles.
type Settings struct {
	// AlwaysIncludeStreamUsage when true forwards the injected usage chunk to
	// clients instead of stripping it. Default (false) strips the injected
	// usage artifact from the client stream while still tracking it in metrics.
	AlwaysIncludeStreamUsage bool `json:"always_include_stream_usage,omitempty"`

	// Trace are the startup defaults for the on-demand request trace
	// (trace.Store holds the live window; it is never persisted).
	Trace TraceSettings `json:"trace,omitempty"`

	Convert ConvertSettings `json:"convert,omitempty"`
}

// TraceSettings are the startup defaults for the on-demand request trace.
// They arm the window at process start only; the live admin toggle is kept in
// trace.Store and never persisted, so a restart can not leave prompt capture
// switched on. Seconds = 0 means "until switched off manually".
type TraceSettings struct {
	Enabled bool     `json:"enabled,omitempty"`
	Seconds int      `json:"seconds,omitempty"`
	Models  []string `json:"models,omitempty"`
	Servers []string `json:"servers,omitempty"`
}

// ConvertSettings are the global conversion knobs. MaxTokensDefault is the output
// budget injected into a converted Anthropic request whose client sent none
// (Anthropic requires the field, OpenAI clients do not send it); the default 4096
// is LiteLLM's DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS, and a negative value switches
// the injection off, so such a request is refused naming `max_tokens`. DropParams
// turns the default refusal (a 400 naming the field) into a silent drop recorded in
// RequestMetric.ConvertNotes. Probe is reserved for the capability probe (out of
// scope here); AnthropicVersion is the anthropic_version a synthesized Anthropic
// request carries (phase 3).
type ConvertSettings struct {
	MaxTokensDefault int    `json:"max_tokens_default,omitempty"`
	DropParams       bool   `json:"drop_params,omitempty"`
	Probe            bool   `json:"probe,omitempty"`
	AnthropicVersion string `json:"anthropic_version,omitempty"`
}

func (c ConvertSettings) MaxTokens() int {
	if c.MaxTokensDefault < 0 {
		// A real setting: the operator refused to invent an output budget, so a
		// converted request whose client sent no max_tokens is refused (or dropped
		// with a note) naming the field, instead of being silently capped.
		return 0
	}
	if c.MaxTokensDefault == 0 {
		return 4096
	}
	return c.MaxTokensDefault
}

func (c ConvertSettings) Version() string {
	if c.AnthropicVersion == "" {
		return "2023-06-01"
	}
	return c.AnthropicVersion
}
