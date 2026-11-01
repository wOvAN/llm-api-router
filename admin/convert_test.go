package admin

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"llm-api-router/config"
	"llm-api-router/domain"
)

// newPreviewHandler wires a preview handler over one chat-only server: the URL is
// deliberately unreachable (127.0.0.1:1), because a dry run must be provably free
// of side effects — a real proxy attempt would sit on it for seconds.
func newPreviewHandler(t *testing.T, convert string) *Handler {
	t.Helper()
	store, err := config.NewStore(filepath.Join(t.TempDir(), "config.json"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "n", URL: "http://127.0.0.1:1/v1",
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude", "r"},
		ServerID: "s1", TargetModel: "claude", Convert: convert, Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	return NewHandler(store, nil, nil, nil)
}

func postPreview(t *testing.T, h *Handler, body string) map[string]any {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/admin/api/convert/preview", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", w.Result().StatusCode, w.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode: %v (%s)", err, w.Body.String())
	}
	return out
}

// The preview names the endpoint and dialect a request would be planned onto.
func TestConvertPreviewNamesDialectAndEndpoint(t *testing.T) {
	out := postPreview(t, newPreviewHandler(t, ""),
		`{"model":"claude","path":"/v1/messages","body":{"model":"claude","max_tokens":10,`+
			`"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}}`)
	if out["upstream_path"] != "/v1/chat/completions" {
		t.Fatalf("upstream_path = %v, want /v1/chat/completions", out["upstream_path"])
	}
	if out["convert"] != "messages>chat" {
		t.Fatalf("convert = %v, want messages>chat", out["convert"])
	}
	// The body transform lands in Task 6; until then the plan is reported and the
	// refusal explains the missing spoke.
	if out["request"] == nil && out["code"] == nil {
		t.Fatalf("preview reported neither a body nor a refusal: %v", out)
	}
}

// A route the preview cannot serve is an answer, not a 5xx: the operator asked
// "would this survive", and "no, because" is the answer.
func TestConvertPreviewReportsRefusals(t *testing.T) {
	out := postPreview(t, newPreviewHandler(t, ""),
		`{"model":"r","path":"/v1/responses","body":{"model":"r","input":"hi"}}`)
	if out["code"] != "no_dialect" {
		t.Fatalf("preview must report the refusal code: %v", out)
	}
	if !strings.Contains(out["error"].(string), "/v1/chat/completions") {
		t.Fatalf("refusal must name what the backend serves: %v", out["error"])
	}
}

// A jev rule has no primary server — its backends are the decision server and the
// candidate pool, which the preview does not resolve. That is an answer too, never
// a 404 naming an empty server id.
func TestConvertPreviewJevRuleIsRefused(t *testing.T) {
	store, err := config.NewStore(filepath.Join(t.TempDir(), "config.json"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "n", URL: "http://127.0.0.1:1/v1",
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"jevr"}, Enabled: true,
		Router: domain.RouterJev,
		Jev: &domain.JevRouter{ServerID: "s1", Candidates: []domain.JevCandidate{
			{ServerID: "s1", TargetModel: "claude", Tier: 1}}}}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	out := postPreview(t, NewHandler(store, nil, nil, nil),
		`{"model":"jevr","path":"/v1/messages","body":{"model":"jevr","max_tokens":10,"messages":[]}}`)
	if out["code"] != "jev_unsupported" {
		t.Fatalf("a jev rule must be answered with a code, not a 4xx: %v", out)
	}
	msg, _ := out["error"].(string)
	if !strings.Contains(msg, "jev") || strings.Contains(msg, "not found") {
		t.Fatalf("refusal must name the reason: %v", out["error"])
	}
}
