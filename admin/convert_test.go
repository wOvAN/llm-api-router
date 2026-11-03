package admin

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
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
	// The fixture deliberately omits max_tokens (Anthropic requires it, the
	// router injects its configured default): the preview must apply the same
	// defaults as a real request and report what it injected.
	out := postPreview(t, newPreviewHandler(t, ""),
		`{"model":"claude","path":"/v1/messages","body":{"model":"claude",`+
			`"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}}`)
	if out["upstream_path"] != "/v1/chat/completions" {
		t.Fatalf("upstream_path = %v, want /v1/chat/completions", out["upstream_path"])
	}
	if out["convert"] != "messages>chat" {
		t.Fatalf("convert = %v, want messages>chat", out["convert"])
	}
	reqMap, ok := out["request"].(map[string]any)
	if !ok {
		t.Fatalf("preview returned no translated body: %v", out)
	}
	if reqMap["max_tokens"] != float64(4096) {
		t.Fatalf("preview must apply the same defaults as the router: %v", out["request"])
	}
	notes, _ := out["notes"].([]any)
	if len(notes) != 1 {
		t.Fatalf("the injected budget must be reported: %v", out["notes"])
	}
}

// convert: "off" on a chat-declared backend is passthrough: the preview answers
// 200, echoes the client's body back untouched, and names no conversion edge.
func TestConvertPreviewIdentityEchoesBody(t *testing.T) {
	body := `{"model":"claude","messages":[{"role":"user","content":"hi"}]}`
	out := postPreview(t, newPreviewHandler(t, "off"),
		`{"model":"claude","path":"/v1/chat/completions","body":`+body+`}`)
	reqMap, ok := out["request"].(map[string]any)
	if !ok {
		t.Fatalf("identity preview must echo the body: %v", out)
	}
	msgs, _ := reqMap["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("identity preview must not touch the body: %v", out["request"])
	}
	if _, has := out["convert"]; has {
		t.Fatalf("an identity plan must not name a conversion: %v", out)
	}
}

// A route the preview cannot serve is an answer, not a 5xx: the operator asked
// "would this survive", and "no, because" is the answer. A responses request is no
// longer one of these (it has its own spoke), so the fixture declares a backend a
// chat client has no edge to.
func TestConvertPreviewReportsRefusals(t *testing.T) {
	store, err := config.NewStore(filepath.Join(t.TempDir(), "config.json"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "n", URL: "http://127.0.0.1:1/v1",
		Protocols: []domain.ServerProtocol{{Path: "/v1/messages"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"gpt"},
		ServerID: "s1", TargetModel: "claude", Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	out := postPreview(t, NewHandler(store, nil, nil, nil),
		`{"model":"gpt","path":"/v1/chat/completions",`+
			`"body":{"model":"gpt","messages":[{"role":"user","content":"hi"}]}}`)
	if out["code"] != "no_dialect" {
		t.Fatalf("preview must report the refusal code: %v", out)
	}
	if !strings.Contains(out["error"].(string), "/v1/chat/completions") {
		t.Fatalf("refusal must name what the backend serves: %v", out["error"])
	}
}

// The responses spoke is previewable like any other: the operator sees the chat
// body their Responses request would become, and what it cost.
func TestConvertPreviewTranslatesResponses(t *testing.T) {
	out := postPreview(t, newPreviewHandler(t, ""),
		`{"model":"r","path":"/v1/responses","body":{"model":"r","input":"hi",`+
			`"instructions":"be brief","max_output_tokens":64,"stream":true}}`)
	if out["upstream_path"] != "/v1/chat/completions" {
		t.Fatalf("upstream_path = %v, want /v1/chat/completions", out["upstream_path"])
	}
	if out["convert"] != "responses>chat" {
		t.Fatalf("convert = %v, want responses>chat", out["convert"])
	}
	reqMap, ok := out["request"].(map[string]any)
	if !ok {
		t.Fatalf("preview returned no translated body: %v", out)
	}
	if reqMap["max_tokens"] != float64(64) {
		t.Fatalf("max_output_tokens not mapped onto max_tokens: %v", out["request"])
	}
	if reqMap["stream"] != true {
		t.Fatalf("stream flag lost: %v", out["request"])
	}
	msgs, _ := reqMap["messages"].([]any)
	if len(msgs) != 2 || msgs[0].(map[string]any)["role"] != "system" {
		t.Fatalf("instructions must become the leading system message: %v", out["request"])
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

// The GUI is embedded HTML: the only cheap assertion is that the fields the tag
// renders are read. A dropped field shows up as a silently missing tag.
// The preview answers "would this survive", and on a routed request an undecoded
// top-level field is reported (`keys:<n>dropped`). Dry-run and real request must
// not disagree about what left, so the preview counts the same keys.
func TestConvertPreviewReportsDroppedKeys(t *testing.T) {
	out := postPreview(t, newPreviewHandler(t, ""),
		`{"model":"claude","path":"/v1/messages","body":{"model":"claude","max_tokens":10,`+
			`"service_tier":"priority","messages":[]}}`)
	notes, _ := out["notes"].([]any)
	for _, want := range []string{"keys:1dropped"} {
		var found bool
		for _, n := range notes {
			if n == want {
				found = true
			}
		}
		if !found {
			t.Fatalf("notes = %v, want %s", notes, want)
		}
	}
}

// The preview echoes the body back in full, so the request that asks for it is
// bounded — an admin endpoint must not buffer and re-serialize an unbounded body.
func TestConvertPreviewBoundsTheBody(t *testing.T) {
	h := newPreviewHandler(t, "")
	body := `{"model":"claude","path":"/v1/messages","body":{"model":"claude","max_tokens":1,` +
		`"messages":[{"role":"user","content":"` + strings.Repeat("x", 2<<20) + `"}]}}`
	req := httptest.NewRequest(http.MethodPost, "/admin/api/convert/preview", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if got := w.Result().StatusCode; got != http.StatusBadRequest {
		t.Fatalf("an oversized preview body must be refused, got status %d (%d bytes echoed)",
			got, w.Body.Len())
	}
}

func TestGUIShowsConversion(t *testing.T) {
	b, err := os.ReadFile("static/index.html")
	if err != nil {
		t.Fatalf("read gui: %v", err)
	}
	src := string(b)
	for _, want := range []string{"r.convert", "r.upstream_endpoint", "r.convert_notes"} {
		if !strings.Contains(src, want) {
			t.Fatalf("gui never reads %q", want)
		}
	}
}

// The server form builds its PUT payload field by field, so every field it does
// not render is erased on save: declaring protocols in config.json and then
// renaming the server in the GUI silently reverts it to undeclared (passthrough).
// The fix is a merge, so the fields the form does not manage ride along.
func TestGUIServerFormPreservesUnmanagedFields(t *testing.T) {
	b, err := os.ReadFile("static/index.html")
	if err != nil {
		t.Fatalf("read gui: %v", err)
	}
	src := string(b)
	for _, want := range []string{"editingServer = s", "editingServer = null", "Object.assign({}, editingServer"} {
		if !strings.Contains(src, want) {
			t.Fatalf("gui server form drops unmanaged fields (missing %q)", want)
		}
	}
}

// A wildcard rule owns a model family, so a name no rule spells out is still a
// real route with a real plan: the preview resolves it through the same matcher
// the router uses, and a pass-through rule (no target model) must hand the
// backend the client's own name — the preview echoes what the router would send.
func TestConvertPreviewResolvesWildcardRule(t *testing.T) {
	store, err := config.NewStore(filepath.Join(t.TempDir(), "config.json"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "n", URL: "http://127.0.0.1:1/v1",
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"qwen3-*"},
		ServerID: "s1", Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}

	out := postPreview(t, NewHandler(store, nil, nil, nil),
		`{"model":"qwen3-32b","path":"/v1/chat/completions","body":{"model":"qwen3-32b","messages":[]}}`)
	reqMap, ok := out["request"].(map[string]any)
	if !ok {
		t.Fatalf("preview planned nothing for a pattern-matched model: %v", out)
	}
	if reqMap["model"] != "qwen3-32b" {
		t.Errorf("the client's model name was rewritten: %v", reqMap["model"])
	}
}
