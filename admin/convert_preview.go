package admin

import (
	"encoding/json"
	"errors"
	"net/http"

	"llm-api-router/config"
	"llm-api-router/convert"
)

// convertPreview runs the conversion decision and the request transform for a
// hypothetical request and reports the result: the endpoint, the dialect, the
// translated body and what the translation cost. It never contacts a backend —
// that is the point of the endpoint (LiteLLM's /utils/transform_request analogue):
// an operator can introspect a rule at zero cost.
func (h *Handler) convertPreview(w http.ResponseWriter, req *http.Request) {
	var in struct {
		Model string          `json:"model"`
		Path  string          `json:"path"`
		Body  json.RawMessage `json:"body"`
	}
	if err := json.NewDecoder(req.Body).Decode(&in); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request: "+err.Error())
		return
	}
	out, status := convertPreviewFor(h.store, in.Model, in.Path, in.Body)
	writeJSON(w, status, out)
}

func convertPreviewFor(store *config.Store, model, path string, body json.RawMessage) (map[string]any, int) {
	if model == "" || path == "" {
		return map[string]any{"error": "model and path are required"}, http.StatusBadRequest
	}
	if len(body) == 0 {
		return map[string]any{"error": "body is required"}, http.StatusBadRequest
	}
	rule, ok := store.GetRuleByModel(model)
	if !ok {
		return map[string]any{"error": "no routing rule for model " + model}, http.StatusNotFound
	}
	if rule.IsJev() {
		// A jev rule has no primary server: its backends are the decision server and
		// the candidate pool, per-request. Previewing that needs a pool selector
		// (the GUI task), so say what the preview covers instead of 404-ing an empty id.
		return map[string]any{
			"error": "jev rules route per candidate; this preview plans the primary attempt only",
			"code":  "jev_unsupported",
		}, http.StatusOK
	}
	srv, ok := store.GetServer(rule.ServerID)
	if !ok {
		return map[string]any{"error": "server " + rule.ServerID + " not found"}, http.StatusNotFound
	}
	client := convert.DialectForPath(path)
	plan, err := convert.Decide(client, path, srv.Protocols, rule.Convert)
	if err != nil {
		return map[string]any{"error": err.Error(), "code": codeOf(err)}, http.StatusOK
	}
	out := map[string]any{"from": client, "dialect": plan.Dialect, "upstream_path": plan.Path}
	if plan.Dialect != plan.From && plan.From != "" {
		out["convert"] = string(plan.From) + ">" + string(plan.Dialect)
	}
	translated, notes, err := convert.Request(plan, body, convertOptions(store))
	if err != nil {
		out["error"], out["code"] = err.Error(), codeOf(err)
		return out, http.StatusOK
	}
	out["request"] = json.RawMessage(translated)
	if len(notes) > 0 {
		out["notes"] = notes
	}
	return out, http.StatusOK
}

// convertOptions resolves the conversion knobs the same way the router does, so a
// preview and a real request can never disagree about the defaults.
func convertOptions(store *config.Store) convert.Options {
	cs := store.GetSettings().Convert
	return convert.Options{MaxTokens: cs.MaxTokens(), DropParams: cs.DropParams,
		AnthropicVersion: cs.Version()}
}

func codeOf(err error) string {
	var ce *convert.Error
	if errors.As(err, &ce) {
		return ce.Code
	}
	return ""
}
