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
}
