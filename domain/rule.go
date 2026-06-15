package domain

// RoutingRule maps one or more incoming model names to a backend server and target model.
type RoutingRule struct {
	IncomingModels    []string        `json:"incoming_models"`
	TargetModel       string          `json:"target_model"`
	ServerID          string          `json:"server_id"`
	Fallbacks         []FallbackEntry `json:"fallbacks,omitempty"`
	FallbackServerIDs []string        `json:"fallback_server_ids,omitempty"`
	Enabled           bool            `json:"enabled"`
}

// GetTargetModelForServer returns the target model to use for the given server ID.
func (r *RoutingRule) GetTargetModelForServer(serverID string) string {
	if serverID == r.ServerID {
		return r.TargetModel
	}
	for _, fb := range r.Fallbacks {
		if fb.ServerID == serverID {
			if fb.TargetModel != "" {
				return fb.TargetModel
			}
			return r.TargetModel
		}
	}
	return r.TargetModel
}

// AllServerIDs returns all server IDs in the rule (primary + fallbacks).
func (r *RoutingRule) AllServerIDs() []string {
	ids := make([]string, 0, 1+len(r.Fallbacks)+len(r.FallbackServerIDs))
	ids = append(ids, r.ServerID)
	for _, fb := range r.Fallbacks {
		ids = append(ids, fb.ServerID)
	}
	ids = append(ids, r.FallbackServerIDs...)
	return ids
}
