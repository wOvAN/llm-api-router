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

// CapabilitiesCache answers "does this server's model accept image input?" and
// "is it an audio model?" from the backend's own /v1/models listing. It is
// intentionally lazy and fail-open: an unknown answer (no listing, no modality
// fields, probe failure) is reported as known=false, and the caller keeps the
// candidate in the pool.
type CapabilitiesCache struct {
	mu      sync.Mutex
	ttl     time.Duration
	entries map[string]serverCapabilities
}

type serverCapabilities struct {
	fetched time.Time
	vision  map[string]*bool // model id -> nil when the backend reports no modalities
	audio   map[string]bool  // model id -> present when the model declares an audio `task`
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

// Audio reports whether srv's model is an audio-domain model, as declared by the
// `task` field of its /v1/models entry (audio.cpp: tts/asr/s2s/diar/...) or by an
// `architecture.output_modalities` list with no `text` in it (llama.cpp: a
// TTS-only checkpoint). Text engines report neither, so known=false is the normal
// answer for a chat backend and the caller must treat the model as a text
// candidate.
func (c *CapabilitiesCache) Audio(ctx context.Context, srv *domain.Server, model string) (audio bool, known bool) {
	if c == nil || srv == nil || model == "" {
		return false, false
	}

	c.mu.Lock()
	entry, ok := c.entries[srv.ID]
	if !ok || time.Since(entry.fetched) >= c.ttl {
		c.mu.Unlock()
		entry = probeServerCapabilities(ctx, srv)
		c.mu.Lock()
		entry.fetched = time.Now()
		c.entries[srv.ID] = entry
	}
	defer c.mu.Unlock()
	return entry.audio[model], entry.audio[model]
}

func probeServerCapabilities(ctx context.Context, srv *domain.Server) serverCapabilities {
	out := serverCapabilities{vision: make(map[string]*bool), audio: make(map[string]bool)}

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
			Task         string `json:"task"`
			Architecture struct {
				InputModalities  *[]string `json:"input_modalities"`
				OutputModalities *[]string `json:"output_modalities"`
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
		// A declared task is audio.cpp's vocabulary (tts/asr/s2s/diar/clon/...):
		// such a model has no chat completion to serve. Text engines never send
		// the field, so they stay absent (= unknown).
		if m.Task != "" {
			out.audio[m.ID] = true
		}
		// No modality fields at all: the backend does not report capabilities
		// (old vLLM, Anthropic) — leave the model unknown, not blindly false.
		if m.Architecture.InputModalities == nil && m.SupportedModalities == nil {
			continue
		}
		// llama.cpp's listing: a model that can only emit audio (a TTS-only
		// checkpoint) has no chat completion to serve either. A chat model that
		// can also speak keeps "text" in the list and stays a candidate.
		if emits := m.Architecture.OutputModalities; emits != nil && !slices.Contains(*emits, "text") {
			out.audio[m.ID] = true
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
