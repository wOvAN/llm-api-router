package config

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"slices"
	"sync"
	"time"

	"llm-api-router/domain"
	"llm-api-router/pkg/log"
	"llm-api-router/proxy"
)

// capabilitiesTTL bounds how long a server's model-capability probe is reused.
// The probe is lazy (first image request that needs it) and negative results
// are cached too, so an unavailable or capability-silent backend is not
// re-probed on every request.
const capabilitiesTTL = 10 * time.Minute

// capabilitiesProbeTimeout bounds a single /v1/models capability probe.
const capabilitiesProbeTimeout = 2 * time.Second

// CapabilitiesCache answers "does this server's model accept image input?" from
// the backend's own /v1/models listing. It is intentionally lazy and fail-open:
// an unknown answer (no listing, no modality fields, probe failure) is reported
// as known=false, and the caller keeps the candidate in the pool.
type CapabilitiesCache struct {
	mu      sync.Mutex
	ttl     time.Duration
	entries map[string]serverCapabilities
}

type serverCapabilities struct {
	fetched time.Time
	vision  map[string]*bool // model id -> nil when the backend reports no modalities
}

// NewCapabilitiesCache creates an empty cache.
func NewCapabilitiesCache() *CapabilitiesCache {
	return &CapabilitiesCache{
		ttl:     capabilitiesTTL,
		entries: make(map[string]serverCapabilities),
	}
}

// Vision reports whether srv's model accepts image input. known=false means the
// backend did not say (absent modality fields, unknown model, probe failure),
// in which case the caller must treat the candidate as vision-unknown.
func (c *CapabilitiesCache) Vision(ctx context.Context, srv *domain.Server, model string) (vision bool, known bool) {
	if c == nil || srv == nil || model == "" {
		return false, false
	}

	c.mu.Lock()
	entry, ok := c.entries[srv.ID]
	fresh := ok && time.Since(entry.fetched) < c.ttl
	if !fresh {
		c.mu.Unlock()
		entry = probeServerCapabilities(ctx, srv)
		c.mu.Lock()
		entry.fetched = time.Now()
		c.entries[srv.ID] = entry
	}
	defer c.mu.Unlock()

	v, ok := entry.vision[model]
	if !ok || v == nil {
		return false, false
	}
	return *v, true
}

func probeServerCapabilities(ctx context.Context, srv *domain.Server) serverCapabilities {
	out := serverCapabilities{vision: make(map[string]*bool)}

	// Detach from the request context: a client disconnect must not poison the
	// capability cache with a negative entry for the next ten minutes.
	probeCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), capabilitiesProbeTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(probeCtx, http.MethodGet, ModelsURL(srv), nil)
	if err != nil {
		log.Debugf("[capabilities] %s: build probe request: %v", srv.Name, err)
		return out
	}
	if srv.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+srv.APIKey)
		req.Header.Set("x-api-key", srv.APIKey)
	}

	transport, err := proxy.TransportFor(srv.ProxyURL())
	if err != nil {
		log.Debugf("[capabilities] %s: probe transport: %v", srv.Name, err)
		return out
	}

	resp, err := (&http.Client{Transport: transport}).Do(req)
	if err != nil {
		log.Debugf("[capabilities] %s: probe failed: %v", srv.Name, err)
		return out
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			log.Debugf("[capabilities] %s: close probe response: %v", srv.Name, err)
		}
	}()
	if resp.StatusCode != http.StatusOK {
		log.Debugf("[capabilities] %s: probe returned %d", srv.Name, resp.StatusCode)
		return out
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		log.Debugf("[capabilities] %s: read probe response: %v", srv.Name, err)
		return out
	}

	var payload struct {
		Data []struct {
			ID           string `json:"id"`
			Architecture struct {
				InputModalities *[]string `json:"input_modalities"`
			} `json:"architecture"`
			SupportedModalities *[]string `json:"supported_modalities"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		log.Debugf("[capabilities] %s: parse probe response: %v", srv.Name, err)
		return out
	}

	for _, m := range payload.Data {
		if m.ID == "" {
			continue
		}
		// No modality fields at all: the backend does not report capabilities
		// (old vLLM, Anthropic) — leave the model unknown, not blindly false.
		if m.Architecture.InputModalities == nil && m.SupportedModalities == nil {
			continue
		}
		seen := containsImageModality(m.Architecture.InputModalities) ||
			containsImageModality(m.SupportedModalities)
		v := seen
		out.vision[m.ID] = &v
	}
	return out
}

func containsImageModality(modalities *[]string) bool {
	return modalities != nil && slices.Contains(*modalities, "image")
}
