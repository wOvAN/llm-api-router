package admin

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"llm-api-router/config"
	"llm-api-router/domain"
	"llm-api-router/metrics"
)

// Handler serves the admin API for managing servers and routing rules.
type Handler struct {
	store   *config.Store
	metrics *metrics.Store
	health  *config.HealthTracker
}

// NewHandler creates a new admin handler.
func NewHandler(store *config.Store, m *metrics.Store, health *config.HealthTracker) *Handler {
	return &Handler{store: store, metrics: m, health: health}
}

// ServeHTTP routes admin API requests.
func (h *Handler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	path := strings.TrimPrefix(req.URL.Path, "/admin/api")

	switch {
	case path == "/servers" && req.Method == http.MethodGet:
		h.listServers(w, req)
	case path == "/servers" && req.Method == http.MethodPost:
		h.addServer(w, req)
	case strings.HasPrefix(path, "/servers/") && req.Method == http.MethodPut:
		h.updateServer(w, req, strings.TrimPrefix(path, "/servers/"))
	case strings.HasPrefix(path, "/servers/") && req.Method == http.MethodDelete:
		h.deleteServer(w, req, strings.TrimPrefix(path, "/servers/"))
	case strings.HasPrefix(path, "/servers/") && strings.HasSuffix(path, "/models") && req.Method == http.MethodGet:
		h.getServerModels(w, req, strings.TrimSuffix(strings.TrimPrefix(path, "/servers/"), "/models"))
	case path == "/servers/test" && req.Method == http.MethodPost:
		h.testServer(w, req)
	case path == "/rules" && req.Method == http.MethodGet:
		h.listRules(w, req)
	case path == "/rules" && req.Method == http.MethodPost:
		h.addRule(w, req)
	case strings.HasPrefix(path, "/rules/") && req.Method == http.MethodPut:
		h.updateRule(w, req, strings.TrimPrefix(path, "/rules/"))
	case strings.HasPrefix(path, "/rules/") && req.Method == http.MethodDelete:
		h.deleteRule(w, req, strings.TrimPrefix(path, "/rules/"))
	case path == "/config" && req.Method == http.MethodGet:
		h.getConfig(w, req)
	case path == "/config/reload" && req.Method == http.MethodPost:
		h.reloadConfig(w, req)
	case path == "/metrics" && req.Method == http.MethodGet:
		h.getMetrics(w, req)
	case path == "/metrics/recent" && req.Method == http.MethodGet:
		h.getRecentRequests(w, req)
	case path == "/metrics/reset" && req.Method == http.MethodPost:
		h.resetMetrics(w, req)
	default:
		http.Error(w, "not found", http.StatusNotFound)
	}
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

// --- Servers ---

func (h *Handler) listServers(w http.ResponseWriter, req *http.Request) {
	cfg := h.store.GetConfig()
	var status map[string]bool
	if h.health != nil {
		status = h.health.GetStatus()
	}
	servers := make([]serverResponse, 0, len(cfg.Servers))
	for _, srv := range cfg.Servers {
		servers = append(servers, serverResponse{
			Server:  *srv,
			Healthy: status[srv.ID],
		})
	}
	writeJSON(w, http.StatusOK, servers)
}

// serverResponse wraps a Server with its health status.
type serverResponse struct {
	domain.Server
	Healthy bool `json:"healthy"`
}

func (h *Handler) addServer(w http.ResponseWriter, req *http.Request) {
	var srv domain.Server
	if err := json.NewDecoder(req.Body).Decode(&srv); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	if srv.ID == "" {
		writeError(w, http.StatusBadRequest, "id is required")
		return
	}
	if err := h.store.AddServer(&srv); err != nil {
		writeError(w, http.StatusConflict, err.Error())
		return
	}
	if err := h.store.Save(); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to save config")
		return
	}
	writeJSON(w, http.StatusCreated, srv)
}

func (h *Handler) updateServer(w http.ResponseWriter, req *http.Request, id string) {
	var srv domain.Server
	if err := json.NewDecoder(req.Body).Decode(&srv); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	if err := h.store.UpdateServer(id, &srv); err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	if err := h.store.Save(); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to save config")
		return
	}
	srv.ID = id
	writeJSON(w, http.StatusOK, srv)
}

func (h *Handler) deleteServer(w http.ResponseWriter, req *http.Request, id string) {
	if err := h.store.DeleteServer(id); err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	if err := h.store.Save(); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to save config")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "deleted"})
}

func (h *Handler) getServerModels(w http.ResponseWriter, req *http.Request, id string) {
	srv, ok := h.store.GetServer(id)
	if !ok {
		writeError(w, http.StatusNotFound, "server not found")
		return
	}

	rawURL := srv.GetURLForAPIType(domain.APITypeOpenAI)
	if !strings.Contains(rawURL, "://") {
		rawURL = "https://" + rawURL
	}

	target, err := url.Parse(rawURL)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid server URL: "+err.Error())
		return
	}

	modelsURL := target.Scheme + "://" + target.Host + target.Path + "/v1/models"
	modelsURL = strings.ReplaceAll(modelsURL, "/v1/v1", "/v1")

	proxyReq, err := http.NewRequestWithContext(req.Context(), http.MethodGet, modelsURL, nil)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to create request: "+err.Error())
		return
	}

	if srv.APIKey != "" {
		proxyReq.Header.Set("Authorization", "Bearer "+srv.APIKey)
	}

	client := &http.Client{}
	resp, err := client.Do(proxyReq)
	if err != nil {
		writeError(w, http.StatusBadGateway, "failed to reach server: "+err.Error())
		return
	}
	defer resp.Body.Close() //nolint:errcheck

	if resp.StatusCode != http.StatusOK {
		writeError(w, resp.StatusCode, fmt.Sprintf("server returned %d", resp.StatusCode))
		return
	}

	var result struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to parse response: "+err.Error())
		return
	}

	models := make([]string, 0, len(result.Data))
	for _, m := range result.Data {
		models = append(models, m.ID)
	}

	writeJSON(w, http.StatusOK, models)
}

func (h *Handler) testServer(w http.ResponseWriter, req *http.Request) {
	var srv domain.Server
	if err := json.NewDecoder(req.Body).Decode(&srv); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON")
		return
	}

	rawURL := srv.GetURLForAPIType(domain.APITypeOpenAI)
	if !strings.Contains(rawURL, "://") {
		rawURL = "https://" + rawURL
	}

	target, err := url.Parse(rawURL)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid URL: "+err.Error())
		return
	}

	modelsURL := target.Scheme + "://" + target.Host + target.Path + "/v1/models"
	modelsURL = strings.ReplaceAll(modelsURL, "/v1/v1", "/v1")

	testReq, err := http.NewRequestWithContext(req.Context(), http.MethodGet, modelsURL, nil)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to create request: "+err.Error())
		return
	}

	if srv.APIKey != "" {
		testReq.Header.Set("Authorization", "Bearer "+srv.APIKey)
	}

	client := &http.Client{Timeout: 10 * time.Second}
	start := time.Now()
	resp, err := client.Do(testReq)
	elapsed := time.Since(start)

	if err != nil {
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"ok":      false,
			"message": "connection failed: " + err.Error(),
		})
		return
	}
	defer resp.Body.Close() //nolint:errcheck

	var bodySnippet string
	if resp.Body != nil {
		buf := make([]byte, 200)
		n, _ := resp.Body.Read(buf)
		bodySnippet = string(buf[:n])
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"ok":               resp.StatusCode == http.StatusOK,
		"status_code":      resp.StatusCode,
		"elapsed_ms":       elapsed.Milliseconds(),
		"response_snippet": bodySnippet,
	})
}

// --- Rules ---

func (h *Handler) listRules(w http.ResponseWriter, req *http.Request) {
	cfg := h.store.GetConfig()
	writeJSON(w, http.StatusOK, cfg.Rules)
}

func (h *Handler) addRule(w http.ResponseWriter, req *http.Request) {
	var rule domain.RoutingRule
	if err := json.NewDecoder(req.Body).Decode(&rule); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	if err := h.store.AddRule(&rule); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := h.store.Save(); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to save config")
		return
	}
	writeJSON(w, http.StatusCreated, rule)
}

func (h *Handler) updateRule(w http.ResponseWriter, req *http.Request, idxStr string) {
	idx, err := strconv.Atoi(idxStr)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid index")
		return
	}

	var rule domain.RoutingRule
	if err := json.NewDecoder(req.Body).Decode(&rule); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	if err := h.store.UpdateRule(idx, &rule); err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	if err := h.store.Save(); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to save config")
		return
	}
	writeJSON(w, http.StatusOK, rule)
}

func (h *Handler) deleteRule(w http.ResponseWriter, req *http.Request, idxStr string) {
	idx, err := strconv.Atoi(idxStr)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid index")
		return
	}
	if err := h.store.DeleteRule(idx); err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	if err := h.store.Save(); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to save config")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "deleted"})
}

// --- Config ---

func (h *Handler) getConfig(w http.ResponseWriter, req *http.Request) {
	cfg := h.store.GetConfig()
	writeJSON(w, http.StatusOK, cfg)
}

func (h *Handler) reloadConfig(w http.ResponseWriter, req *http.Request) {
	if err := h.store.Load(); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to reload config: "+err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "reloaded"})
}

// --- Metrics ---

func (h *Handler) getMetrics(w http.ResponseWriter, req *http.Request) {
	summaries := h.metrics.Summaries()
	writeJSON(w, http.StatusOK, summaries)
}

func (h *Handler) getRecentRequests(w http.ResponseWriter, req *http.Request) {
	recent := h.metrics.Recent()
	writeJSON(w, http.StatusOK, recent)
}

func (h *Handler) resetMetrics(w http.ResponseWriter, req *http.Request) {
	h.metrics.Reset()
	writeJSON(w, http.StatusOK, map[string]string{"status": "reset"})
}
