package domain

// Config holds the full router configuration.
type Config struct {
	Servers  map[string]*Server `json:"servers"`
	Rules    []*RoutingRule     `json:"rules"`
	Settings Settings           `json:"settings,omitempty"`
}

// Settings holds global router behavior toggles.
type Settings struct {
	// AlwaysIncludeStreamUsage when true forwards the injected usage chunk to
	// clients instead of stripping it. Default (false) strips the injected
	// usage artifact from the client stream while still tracking it in metrics.
	AlwaysIncludeStreamUsage bool `json:"always_include_stream_usage,omitempty"`
}
