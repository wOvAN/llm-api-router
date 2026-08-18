package domain

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
}
