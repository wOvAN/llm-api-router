package router

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
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
}

// New creates a new Router.
func New(store *config.Store, m *metrics.Store, health *config.HealthTracker, rateLimit *config.RateLimiter) *Router {
	return &Router{store: store, metrics: m, health: health, rateLimit: rateLimit}
}

// apiTypeFromPath determines the API type from the request path.
func apiTypeFromPath(path string) domain.APIType {
	if strings.Contains(path, "/messages") {
		return domain.APITypeAnthropic
	}
	return domain.APITypeOpenAI
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

	body, err := io.ReadAll(req.Body)
	if err != nil {
		log.Errorf("[%s] failed to read request body: %v", req.URL.Path, err)
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return
	}
	defer req.Body.Close() //nolint:errcheck

	model, err := extractModel(body)
	if err != nil {
		log.Errorf("[%s] invalid request body: %v", req.URL.Path, err)
		http.Error(w, fmt.Sprintf("invalid request body: %v", err), http.StatusBadRequest)
		return
	}

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
	for _, fb := range rule.Fallbacks {
		if srv, ok := r.store.GetServer(fb.ServerID); ok {
			tm := fb.TargetModel
			if tm == "" {
				tm = rule.TargetModel
			}
			attempts = append(attempts, serverAttempt{server: srv, targetModel: tm})
		}
	}

	requestStart := time.Now()
	var lastErr error
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

		rewrittenBody, err := proxy.RewriteModelInBody(body, targetModel)
		if err != nil {
			log.Errorf("[%s] failed to rewrite model %q -> %q: %v", req.URL.Path, model, targetModel, err)
			http.Error(w, fmt.Sprintf("failed to rewrite model: %v", err), http.StatusInternalServerError)
			return
		}

		req.Body = io.NopCloser(bytes.NewReader(rewrittenBody))
		req.ContentLength = int64(len(rewrittenBody))

		wasFallback := i > 0

		log.Infof("[%s] model=%q -> %q on %s (attempt %d/%d)",
			req.URL.Path, model, targetModel, srv.Name, i+1, len(attempts))

		apiType := apiTypeFromPath(req.URL.Path)
		serverURL := srv.GetURLForAPIType(apiType)
		pm, err := proxy.StreamProxy(req.Context(), serverURL, srv.APIKey, req, w, targetModel, model)
		if err != nil {
			lastErr = err
			if r.health != nil {
				r.health.MarkUnhealthy(srv.ID)
			}
			if r.rateLimit != nil {
				r.rateLimit.RecordFailure(srv.ID)
			}
			log.Errorf("[%s] fallback from %s: %v", req.URL.Path, srv.Name, err)
			continue
		}

		// Success — mark healthy and clear rate limit
		if r.health != nil {
			r.health.MarkHealthy(srv.ID)
		}
		if r.rateLimit != nil {
			r.rateLimit.RecordSuccess(srv.ID)
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
			ServerID:              srv.ID,
			StatusCode:            pm.StatusCode,
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
		})
		return
	}

	latency := time.Since(requestStart).Milliseconds()
	r.metrics.Add(domain.RequestMetric{
		Timestamp:    requestStart,
		Model:        model,
		TargetModel:  rule.TargetModel,
		ServerID:     primaryServer.ID,
		StatusCode:   http.StatusBadGateway,
		LatencyMs:    latency,
		TTFBMs:       0,
		ResponseSize: 0,
		WasFallback:  len(attempts) > 1,
	})

	log.Errorf("[%s] model=%q — all backends failed: %v", req.URL.Path, model, lastErr)
	http.Error(w, fmt.Sprintf("all backends failed: %v", lastErr), http.StatusBadGateway)
}

// listModels returns the list of incoming model names (what clients can request).
func (r *Router) listModels(w http.ResponseWriter, req *http.Request) {
	cfg := r.store.GetConfig()

	models := make([]map[string]interface{}, 0)
	for _, rule := range cfg.Rules {
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
			models = append(models, m)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"object": "list",
		"data":   models,
	})
}

// extractModel reads the "model" field from a JSON body.
func extractModel(body []byte) (string, error) {
	var obj map[string]interface{}
	if err := json.Unmarshal(body, &obj); err != nil {
		return "", err
	}

	model, ok := obj["model"]
	if !ok {
		return "", fmt.Errorf("missing 'model' field")
	}

	modelStr, ok := model.(string)
	if !ok {
		return "", fmt.Errorf("'model' field is not a string")
	}

	return strings.TrimSpace(modelStr), nil
}
