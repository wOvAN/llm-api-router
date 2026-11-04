package config

import (
	"net/http"
	"slices"
	"strings"

	"llm-api-router/domain"
)

// routerHeaderNames are the response headers a browser client needs permission
// to read: which backend served the request, what a conversion cost, the id to
// quote in a bug report.
var routerHeaderNames = []string{
	"X-Router-Server", "X-Router-Server-Name", "X-Router-Status", "X-Router-Request-Id",
	"X-Router-Attempts", "X-Router-Retries", "X-Router-Fallback-Errors",
	"X-Router-Jev-Choice", "X-Router-Jev-Confidence", "X-Router-Jev-Ms", "X-Router-Jev-Reason",
	"X-Router-Convert", "X-Router-Convert-Notes", "X-Router-Trace-Id", "X-Router-Count-Tokens",
}

// CORS answers cross-origin requests for browser clients (OpenWebUI, a custom
// dashboard). Off until origins are allowed.
type CORS struct {
	origins []string // empty = any origin
	any     bool
}

// NewCORS returns nil when the feature is off; a nil CORS is a pass-through.
func NewCORS(s domain.COSSettings) *CORS {
	if !s.Enabled {
		return nil
	}
	return &CORS{origins: s.Origins, any: len(s.Origins) == 0}
}

func (c *CORS) allow(origin string) bool {
	if c == nil || origin == "" {
		return false
	}
	return c.any || slices.Contains(c.origins, origin)
}

func (c *CORS) Middleware(next http.Handler) http.Handler {
	if c == nil {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		origin := req.Header.Get("Origin")
		if !c.allow(origin) {
			next.ServeHTTP(w, req) // same-origin, a CLI, or a denied origin
			return
		}
		// Echoed, never "*": an echo is what lets a deployment add
		// Access-Control-Allow-Credentials later without re-planning the CORS
		// story, and it is the only spelling that works with credentials today.
		w.Header().Set("Access-Control-Allow-Origin", origin)
		w.Header().Set("Vary", "Origin")
		w.Header().Set("Access-Control-Expose-Headers", strings.Join(routerHeaderNames, ", "))

		if req.Method == http.MethodOptions {
			// A preflight is a question for the router, not a request for a
			// backend: answering it here keeps a proxy round trip out of the
			// browser's critical path (and a POST-only backend from a OPTIONS it
			// would refuse).
			w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
			if h := req.Header.Get("Access-Control-Request-Headers"); h != "" {
				w.Header().Set("Access-Control-Allow-Headers", h)
			}
			w.Header().Set("Access-Control-Max-Age", "600")
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, req)
	})
}
