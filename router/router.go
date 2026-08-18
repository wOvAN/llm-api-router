package router

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"llm-api-router/config"
	"llm-api-router/domain"
	"llm-api-router/metrics"
	"llm-api-router/pkg/log"
	"llm-api-router/proxy"
)

// Router handles incoming LLM API requests and routes them to configured backends.
type Router struct {
	store     *config.Store
	metrics   *metrics.Store
	health    *config.HealthTracker
	rateLimit *config.RateLimiter
	quota     *config.QuotaTracker
}

// New creates a new Router.
func New(store *config.Store, m *metrics.Store, health *config.HealthTracker, rateLimit *config.RateLimiter, quota *config.QuotaTracker) *Router {
	return &Router{store: store, metrics: m, health: health, rateLimit: rateLimit, quota: quota}
}

// orderedFallbacks returns the rule's enabled fallbacks in attempt order:
// sorted by priority ascending (lower number tried first). Fallbacks with
// priority 0 (unset) sort after all prioritized ones, and their original list
// order is preserved among ties (stable sort).
func orderedFallbacks(rule *domain.RoutingRule) []domain.FallbackEntry {
	fbs := make([]domain.FallbackEntry, 0, len(rule.Fallbacks))
	for _, fb := range rule.Fallbacks {
		if fb.IsEnabled() {
			fbs = append(fbs, fb)
		}
	}
	sort.SliceStable(fbs, func(i, j int) bool {
		a, b := fbs[i].Priority, fbs[j].Priority
		if a == b {
			return false
		}
		if a == 0 {
			return false // unset sorts after set
		}
		if b == 0 {
			return true
		}
		return a < b
	})
	return fbs
}

// apiTypeFromPath determines the API type from the request path.
func apiTypeFromPath(path string) domain.APIType {
	if strings.Contains(path, "/messages") {
		return domain.APITypeAnthropic
	}
	return domain.APITypeOpenAI
}

// clientIP extracts the client IP from the request, checking X-Forwarded-For
// and X-Real-IP first (for proxied traffic), then falling back to RemoteAddr.
func clientIP(req *http.Request) string {
	if fwd := req.Header.Get("X-Forwarded-For"); fwd != "" {
		if idx := strings.IndexByte(fwd, ','); idx >= 0 {
			return strings.TrimSpace(fwd[:idx])
		}
		return strings.TrimSpace(fwd)
	}
	if rip := req.Header.Get("X-Real-IP"); rip != "" {
		return strings.TrimSpace(rip)
	}
	if ip, _, err := net.SplitHostPort(req.RemoteAddr); err == nil {
		return ip
	}
	return req.RemoteAddr
}

// apiEndpointFromPath extracts the specific API endpoint from the request path.
// Returns a human-readable identifier like "chat", "responses", "messages", etc.
func apiEndpointFromPath(path string) string {
	if strings.Contains(path, "/messages") {
		return "messages"
	}
	if strings.Contains(path, "/chat/completions") {
		return "chat"
	}
	if strings.Contains(path, "/responses") {
		return "responses"
	}
	if strings.Contains(path, "/embeddings") {
		return "embeddings"
	}
	if strings.Contains(path, "/completions") {
		return "completions"
	}
	// Fallback: extract the last path segment after /v1/
	if idx := strings.Index(path, "/v1/"); idx >= 0 {
		rest := path[idx+4:]
		if seg := strings.Split(rest, "/")[0]; seg != "" {
			return seg
		}
	}
	return "unknown"
}

// Handle processes the incoming request.
func (r *Router) Handle(w http.ResponseWriter, req *http.Request) {
	if req.Method == http.MethodGet && req.URL.Path == "/v1/models" {
		r.listModels(w, req)
		return
	}

	if req.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Cap the request body to protect against memory exhaustion. LLM prompts can
	// be large (long-context requests), so 64MB is generous.
	req.Body = http.MaxBytesReader(w, req.Body, 64<<20)
	body, err := io.ReadAll(req.Body)
	if err != nil {
		log.Errorf("[%s] failed to read request body: %v", req.URL.Path, err)
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return
	}
	defer func() {
		if err := req.Body.Close(); err != nil {
			log.Errorf("[%s] failed to close request body: %v", req.URL.Path, err)
		}
	}()

	model, err := extractModel(body)
	if err != nil {
		log.Errorf("[%s] invalid request body: %v", req.URL.Path, err)
		http.Error(w, fmt.Sprintf("invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	// Track active requests for metrics display.
	r.metrics.IncrActive()
	defer r.metrics.DecrActive()

	rule, ok := r.store.GetRuleByModel(model)
	if !ok {
		log.Errorf("[%s] no routing rule for model %q", req.URL.Path, model)
		http.Error(w, fmt.Sprintf("no routing rule for model %q", model), http.StatusNotFound)
		return
	}

	primaryServer, ok := r.store.GetServer(rule.ServerID)
	if !ok {
		log.Errorf("[%s] server %q not found for model %q", req.URL.Path, rule.ServerID, model)
		http.Error(w, fmt.Sprintf("server %q not found", rule.ServerID), http.StatusInternalServerError)
		return
	}

	type serverAttempt struct {
		server      *domain.Server
		targetModel string
	}
	attempts := []serverAttempt{{server: primaryServer, targetModel: rule.TargetModel}}
	for _, fb := range orderedFallbacks(rule) {
		if srv, ok := r.store.GetServer(fb.ServerID); ok {
			tm := fb.TargetModel
			if tm == "" {
				tm = rule.TargetModel
			}
			attempts = append(attempts, serverAttempt{server: srv, targetModel: tm})
		}
	}

	requestStart := time.Now()
	// Inject stream_options.include_usage for OpenAI streaming chat completions so
	// backends report token usage even when the client didn't opt in. stripStream
	// hides the injected usage chunk from the client (unless configured otherwise).
	apiType := apiTypeFromPath(req.URL.Path)
	stripStream := false
	if apiType == domain.APITypeOpenAI && strings.Contains(req.URL.Path, "/chat/completions") {
		alwaysInclude := r.store.GetSettings().AlwaysIncludeStreamUsage
		var injected bool
		body, injected = proxy.EnsureStreamUsage(body, alwaysInclude)
		stripStream = injected
	}
	// SSE heartbeats keep streams alive when the backend pauses between tokens
	// (slow/thinking inference) — clients otherwise trip read timeouts
	// ("waiting for api responses", "api timeout"). Anthropic clients expect the
	// official ping event; OpenAI clients get a bare SSE comment line, which
	// every SSE parser ignores. StreamProxy only emits heartbeats for
	// text/event-stream responses.
	var ping []byte
	if apiType == domain.APITypeAnthropic {
		ping = []byte("event: ping\ndata: {\"type\":\"ping\"}\n\n")
	} else {
		ping = []byte(": keep-alive\n\n")
	}
	var lastErr error
	// Errors from failed attempts, surfaced to the client via
	// X-Router-Fallback-Errors (preceding attempts only).
	var attemptErrors []string
	for i, attempt := range attempts {
		srv := attempt.server
		targetModel := attempt.targetModel

		// Skip unhealthy servers (except the last attempt — try it anyway)
		if r.health != nil && !r.health.IsHealthy(srv.ID) && i < len(attempts)-1 {
			log.Warnf("[%s] model=%q — skipping unhealthy server %s (attempt %d/%d)",
				req.URL.Path, model, srv.Name, i+1, len(attempts))
			continue
		}

		// Skip rate-limited servers (except the last attempt — try it anyway)
		if r.rateLimit != nil && r.rateLimit.ShouldSkip(srv.ID) && i < len(attempts)-1 {
			remaining := r.rateLimit.CooldownRemaining(srv.ID)
			log.Warnf("[%s] model=%q — skipping rate-limited server %s (cooldown %v, attempt %d/%d)",
				req.URL.Path, model, srv.Name, remaining.Round(time.Second), i+1, len(attempts))
			continue
		}

		// Skip quota-exceeded servers (TPM/RPM at limit; except the last attempt)
		if r.quota != nil && !r.quota.Allow(srv.ID) && i < len(attempts)-1 {
			tokens, reqs := r.quota.Usage(srv.ID)
			log.Warnf("[%s] model=%q — skipping quota-exceeded server %s (tpm=%d rpm=%d, attempt %d/%d)",
				req.URL.Path, model, srv.Name, tokens, reqs, i+1, len(attempts))
			continue
		}

		rewrittenBody, err := proxy.RewriteModelInBody(body, targetModel)
		if err != nil {
			log.Errorf("[%s] failed to rewrite model %q -> %q: %v", req.URL.Path, model, targetModel, err)
			http.Error(w, fmt.Sprintf("failed to rewrite model: %v", err), http.StatusInternalServerError)
			return
		}

		// TPM pre-call reservation: hold the declared output budget so
		// concurrent requests can't collectively exceed the server's TPM limit.
		// Released by quota.Complete on every exit path below. The last attempt
		// sends even if the reservation would exceed the limit (better to try
		// than to fail); it simply reserves nothing.
		estimate := proxy.DeclaredOutputBudget(rewrittenBody)
		reservedTokens := int64(0)
		if r.quota != nil && estimate > 0 {
			if r.quota.Reserve(srv.ID, estimate) {
				reservedTokens = estimate
			} else if i < len(attempts)-1 {
				log.Warnf("[%s] model=%q — skipping TPM-reserved server %s (attempt %d/%d)",
					req.URL.Path, model, srv.Name, i+1, len(attempts))
				continue
			}
		}

		wasFallback := i > 0

		log.Infof("[%s] model=%q -> %q on %s (attempt %d/%d)",
			req.URL.Path, model, targetModel, srv.Name, i+1, len(attempts))

		apiType := apiTypeFromPath(req.URL.Path)
		serverURL := srv.GetURLForAPIType(apiType)
		// On fallback, preserve the actual model used (don't rewrite back to
		// the original) so the client knows which model actually responded.
		responseModel := model
		if wasFallback {
			responseModel = targetModel
		}
		// Log the response rewrite parameters
		if responseModel != targetModel {
			log.Debugf("[%s] response rewrite: %q → %q", req.URL.Path, targetModel, responseModel)
		} else {
			log.Debugf("[%s] response rewrite: any model → %q (backend may return its actual model)", req.URL.Path, responseModel)
		}
		// Per-server keep-alive override (seconds), clamped to the same [1,300]s
		// bounds as the per-request header. 0 = use the global default.
		var srvKeepAlive time.Duration
		if srv.KeepAliveIdle > 0 {
			n := srv.KeepAliveIdle
			if n < 1 {
				n = 1
			} else if n > 300 {
				n = 300
			}
			srvKeepAlive = time.Duration(n) * time.Second
		}
		rh := &proxy.RouterHeaders{
			ServerID:       srv.ID,
			ServerName:     srv.Name,
			Attempts:       fmt.Sprintf("%d/%d", i+1, len(attempts)),
			FallbackErrors: attemptErrors,
			KeepAliveIdle:  srvKeepAlive,
		}

		// Retry-before-fallback: on transport (pre-response) errors retry the
		// same server up to NumRetries times before moving to the next server.
		// Precedence: X-Router-Retries header > rule's num_retries.
		numRetries := rule.NumRetries
		if numRetries == 0 {
			// Rules without num_retries get a generous default: absorb a dying
			// backend (crashed worker, restart) before marking it unhealthy and
			// falling back. Disable per request with X-Router-Retries: 0.
			numRetries = defaultNumRetries
		}
		if h := req.Header.Get("X-Router-Retries"); h != "" {
			if n, parseErr := strconv.Atoi(h); parseErr == nil && n >= 0 {
				numRetries = n
			}
		}

		for retry := 0; ; retry++ {
			// The upstream client consumes req.Body on each attempt; rebuild it
			// so a retry sends a fresh body.
			req.Body = io.NopCloser(bytes.NewReader(rewrittenBody))
			req.ContentLength = int64(len(rewrittenBody))
			rh.Retries = retry

			pm, err := proxy.StreamProxy(req.Context(), serverURL, srv.APIKey, req, w, targetModel, responseModel, rh, stripStream, ping)
			if err == nil {
				// Success — mark healthy. Rate limiter is status-aware: only
				// successful (2xx) responses clear failures/cooldown; 4xx/5xx
				// responses count toward cooldown (429/401/403 immediately).
				if r.health != nil {
					r.health.MarkHealthy(srv.ID)
				}
				if r.rateLimit != nil {
					if pm.StatusCode >= 400 {
						r.rateLimit.RecordFailure(srv.ID, pm.StatusCode)
					} else {
						r.rateLimit.RecordSuccess(srv.ID)
					}
				}
				// Account quota usage (TPM/RPM) for successful responses: count
				// the request (RPM) and release the reservation, recording actual
				// usage (TPM).
				if r.quota != nil {
					r.quota.RecordRequest(srv.ID)
					r.quota.Complete(srv.ID, reservedTokens, int64(pm.TotalTokens))
				}

				if pm.StatusCode >= 400 {
					log.Errorf("[%s] model=%q -> %q on %s returned HTTP %d %s: %s",
						req.URL.Path, model, targetModel, srv.Name, pm.StatusCode, http.StatusText(pm.StatusCode), pm.ErrorBody)
				}

				latency := time.Since(requestStart).Milliseconds()
				r.metrics.Add(domain.RequestMetric{
					Timestamp:             requestStart,
					Model:                 model,
					TargetModel:           targetModel,
					BackendModel:          pm.BackendModel,
					ServerID:              srv.ID,
					StatusCode:            pm.StatusCode,
					ErrorBody:             pm.ErrorBody,
					LatencyMs:             latency,
					TTFBMs:                pm.TTFBMs,
					ResponseSize:          pm.ResponseSize,
					WasFallback:           wasFallback,
					PromptTokens:          pm.PromptTokens,
					CompletionTokens:      pm.CompletionTokens,
					TotalTokens:           pm.TotalTokens,
					CachedTokens:          pm.CachedTokens,
					NativePromptMs:        pm.PromptMs,
					NativePredictedMs:     pm.PredictedMs,
					NativePromptTokPerSec: pm.PromptPerSec,
					NativeDecodeTokPerSec: pm.TokensPerSec,
					APIType:               apiType,
					APIEndpoint:           apiEndpointFromPath(req.URL.Path),
					ClientIP:              clientIP(req),
				})
				return
			}

			// Client disconnect: stop immediately, no fallback needed
			if req.Context().Err() != nil {
				if r.quota != nil {
					r.quota.Complete(srv.ID, reservedTokens, 0)
				}
				log.Warnf("[%s] model=%q — client disconnected during proxy (attempt %d/%d)",
					req.URL.Path, model, i+1, len(attempts))
				return
			}
			// Mid-stream error: headers already sent to client, retry/fallback impossible
			if _, isMidStream := err.(*proxy.MidStreamError); isMidStream {
				if r.health != nil {
					r.health.MarkUnhealthy(srv.ID)
				}
				if r.rateLimit != nil {
					r.rateLimit.RecordFailure(srv.ID, pm.StatusCode)
				}
				if r.quota != nil {
					r.quota.Complete(srv.ID, reservedTokens, int64(pm.TotalTokens))
				}
				log.Errorf("[%s] model=%q — mid-stream error on %s (attempt %d/%d): %v",
					req.URL.Path, model, srv.Name, i+1, len(attempts), err)
				// Record whatever metrics we have
				latency := time.Since(requestStart).Milliseconds()
				r.metrics.Add(domain.RequestMetric{
					Timestamp:        requestStart,
					Model:            model,
					TargetModel:      targetModel,
					BackendModel:     pm.BackendModel,
					ServerID:         srv.ID,
					StatusCode:       pm.StatusCode,
					ErrorBody:        pm.ErrorBody,
					LatencyMs:        latency,
					TTFBMs:           pm.TTFBMs,
					ResponseSize:     pm.ResponseSize,
					WasFallback:      wasFallback,
					PromptTokens:     pm.PromptTokens,
					CompletionTokens: pm.CompletionTokens,
					TotalTokens:      pm.TotalTokens,
					CachedTokens:     pm.CachedTokens,
					APIType:          apiType,
					APIEndpoint:      apiEndpointFromPath(req.URL.Path),
					ClientIP:         clientIP(req),
				})
				return
			}

			// Transport error (pre-response): retry same server if retries left,
			// otherwise mark unhealthy and fall through to the next server.
			lastErr = err
			if retry < numRetries {
				log.Warnf("[%s] model=%q — retrying %s (%d/%d): %v",
					req.URL.Path, model, srv.Name, retry+1, numRetries, err)
				select {
				case <-time.After(retryBackoff(retry)):
				case <-req.Context().Done():
					return
				}
				continue
			}
			if r.health != nil {
				r.health.MarkUnhealthy(srv.ID)
			}
			if r.rateLimit != nil {
				r.rateLimit.RecordFailure(srv.ID, 0)
			}
			if r.quota != nil {
				r.quota.Complete(srv.ID, reservedTokens, 0)
			}
			attemptErrors = append(attemptErrors, err.Error())
			log.Errorf("[%s] fallback from %s: %v", req.URL.Path, srv.Name, err)
			break
		}
	}

	latency := time.Since(requestStart).Milliseconds()
	errorBody := ""
	if lastErr != nil {
		errorBody = lastErr.Error()
	}
	r.metrics.Add(domain.RequestMetric{
		Timestamp:    requestStart,
		Model:        model,
		TargetModel:  rule.TargetModel,
		ServerID:     primaryServer.ID,
		StatusCode:   http.StatusBadGateway,
		ErrorBody:    errorBody,
		LatencyMs:    latency,
		TTFBMs:       0,
		ResponseSize: 0,
		WasFallback:  len(attempts) > 1,
		APIType:      apiType,
		APIEndpoint:  apiEndpointFromPath(req.URL.Path),
		ClientIP:     clientIP(req),
	})

	log.Errorf("[%s] model=%q — all backends failed: %v", req.URL.Path, model, lastErr)
	http.Error(w, fmt.Sprintf("all backends failed: %v", lastErr), http.StatusBadGateway)
}

// defaultNumRetries is the effective num_retries for rules that don't set one:
// a dying backend (crashed llama.cpp worker, restart) usually recovers within
// seconds, so retry before marking the server unhealthy and falling back.
// Override per request via X-Router-Retries ("0" disables retries).
const defaultNumRetries = 10

// retryBackoff returns the delay before retry attempt n (0-based): 100ms, 250ms,
// 500ms, then capped at 1s. Keeps retries fast but avoids hammering a failing
// backend.
func retryBackoff(n int) time.Duration {
	switch {
	case n <= 0:
		return 100 * time.Millisecond
	case n == 1:
		return 250 * time.Millisecond
	case n == 2:
		return 500 * time.Millisecond
	default:
		return time.Second
	}
}

// listModels returns the list of incoming model names (what clients can request).
func (r *Router) listModels(w http.ResponseWriter, req *http.Request) {
	rules := r.store.GetActiveRules()

	models := make([]map[string]interface{}, 0)
	for _, rule := range rules {
		if !rule.Enabled {
			continue
		}
		for _, name := range rule.IncomingModels {
			m := map[string]interface{}{
				"id":       name,
				"object":   "model",
				"created":  time.Now().Unix(),
				"owned_by": "router",
				"target":   rule.TargetModel,
				"server":   rule.ServerID,
			}
			// Clients (Claude Code etc.) read context_window to size prompts.
			// Omitted entirely when unknown (malformed/0 → absent, per LiteLLM).
			if rule.ContextWindow > 0 {
				m["context_window"] = rule.ContextWindow
			}
			models = append(models, m)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"object": "list",
		"data":   models,
	}); err != nil {
		log.Errorf("[%s] failed to encode /v1/models response: %v", req.URL.Path, err)
	}
}

// findJSONStringEnd finds the closing quote of a JSON string starting at pos.
// Returns the index of the closing quote, or -1 if not found.
func findJSONStringEnd(data []byte, pos int) int {
	for pos < len(data) {
		if data[pos] == '\\' {
			pos += 2 // skip escaped character
			continue
		}
		if data[pos] == '"' {
			return pos
		}
		pos++
	}
	return -1
}

// extractModel reads the top-level "model" field from a JSON body using byte-level
// scanning. Tracks object depth and string boundaries so "model" keys nested
// inside other objects or quoted string content (e.g. a prompt discussing model
// names) are not mistaken for the request model. Avoids full JSON unmarshal
// allocation for a single field extraction.
func extractModel(body []byte) (string, error) {
	key := []byte(`"model"`)
	keyLen := len(key)
	depth := 0

	for i := 0; i < len(body); {
		switch body[i] {
		case '{':
			depth++
			i++
			continue
		case '}':
			depth--
			i++
			continue
		case '"':
			if depth == 1 && i+keyLen <= len(body) && bytes.Equal(body[i:i+keyLen], key) {
				endOfKey := i + keyLen
				// Skip longer keys like "model_id" / "model_name".
				if endOfKey == len(body) || !isNameChar(body[endOfKey]) {
					model, found, err := parseModelValue(body, endOfKey)
					if err != nil {
						return "", err
					}
					if found {
						return model, nil
					}
				}
			}
			i = skipString(body, i)
			continue
		default:
			i++
		}
	}

	return "", fmt.Errorf("missing 'model' field")
}

// parseModelValue extracts the string value of a "model" key, starting just past
// the key's closing quote. found is false when the value is not a string (the
// caller keeps scanning); an error is returned only for an unterminated value.
func parseModelValue(body []byte, afterKey int) (string, bool, error) {
	colonIdx := bytes.IndexByte(body[afterKey:], ':')
	if colonIdx < 0 {
		return "", false, nil
	}
	valueStart := afterKey + colonIdx + 1
	for valueStart < len(body) && (body[valueStart] == ' ' || body[valueStart] == '\t' || body[valueStart] == '\n' || body[valueStart] == '\r') {
		valueStart++
	}
	if valueStart >= len(body) || body[valueStart] != '"' {
		return "", false, nil
	}
	strStart := valueStart + 1
	strEnd := findJSONStringEnd(body, strStart)
	if strEnd < 0 {
		return "", false, fmt.Errorf("invalid model value")
	}
	return strings.TrimSpace(string(body[strStart:strEnd])), true, nil
}

// skipString returns the index just past the closing quote of the JSON string
// starting at i (body[i] == '"'). Handles backslash escapes; returns len(body)
// for an unterminated string.
func skipString(body []byte, i int) int {
	i++ // skip opening quote
	for i < len(body) {
		switch body[i] {
		case '\\':
			i += 2
		case '"':
			return i + 1
		default:
			i++
		}
	}
	return len(body)
}

// isNameChar reports whether b can extend a JSON object key past "model".
func isNameChar(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9') || b == '_' || b == '-'
}
