package domain

// Config holds the full router configuration.
type Config struct {
	Servers map[string]*Server `json:"servers"`
	Rules   []*RoutingRule     `json:"rules"`
}
