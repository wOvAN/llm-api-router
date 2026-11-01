# API Protocol Conversion (phases 1–2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the router a protocol-conversion layer: a per-attempt decision of which dialect to speak to which backend, the observability to see it, and the first spoke — an Anthropic `messages` client served by an OpenAI `chat` backend, JSON and streaming.

**Architecture:** A new `convert` package owns dialects, the per-attempt plan (which upstream dialect/path to use, or a refusal), request/response body translation, and a stateful stream translator. The router decides **per attempt** (a fallback server and a jev candidate each carry their own declared capability); the proxy applies the translator at the very end of the writer chain — after `metricsWriter` and `modelRewriteWriter`, and after `sseRelay`'s frame buffer — so usage extraction, native timings, model rewriting and the trace keep seeing the backend's native bytes. `proxy` declares the translator as a small interface (`StreamTranslator`) and knows nothing about `convert`, exactly as it already does with `TraceTap`.

**Tech Stack:** Go 1.27, stdlib only (`encoding/json`, `net/http`, `crypto/sha1`), `net/http/httptest` for tests, embedded admin GUI (`admin/static/index.html`, vanilla JS).

**Spec:** `docs/superpowers/specs/2026-11-01-api-protocol-convert-design.md`

## Global Constraints

- Zero new external dependencies: stdlib plus this module's own packages.
- `api_types` keeps its current meaning (per-protocol URL selection in `Server.GetURLForAPIType`). Capability is declared in the NEW `Server.Protocols`; **an empty `Protocols` means byte-for-byte passthrough**, so no existing config changes behavior.
- The hub is `chat`. Phases 3–4 (`chat → messages`, `responses`) are not in this plan: no `responses` translator, no `messages ↔ responses` direct edge.
- A refused conversion (field with no representation, backend has no dialect the router can reach) is HTTP **400 naming the field**, and the request never reaches the backend. Silent dropping happens only with `settings.convert.drop_params: true`, and every dropped field lands in `RequestMetric.ConvertNotes`.
- Defaults, applied in exactly one place (the router builds `convert.Options` from `Settings.Convert`): `max_tokens_default` 4096, `anthropic_version` `2023-06-01`, `drop_params` false, `probe` false. The probe is not implemented by this plan.
- A converted request is traced like every request: `trace` sees the client's request body and the backend's native response. No second body copy exists.
- `config.json` is user-managed and gitignored — never write it. `example.config.json` is the committed reference (Task 2 edits that one only).
- `CLAUDE.md` and `AGENTS.md` are duplicates of each other — both edited in Task 12.
- New `domain` fields must be enumerated in `config/store.go`'s clone points (`AddServer`, `UpdateServer`, `cloneRule`) or admin edits silently wipe them.
- Lint gate: `task lint` reports `0 issues`; every returned error is handled (`errcheck`).
- Tests use live `httptest` servers and `NumRetries: -1` on rules (the repo's "one try, no backoff" idiom — `numRetries == 0` means the 10-retry default).

## Review Focus

- **A backend that declares nothing (`protocols` absent) keeps today's behavior exactly** — no dialect decision, no `X-Router-Convert`, no translator armed, one nil check on the relay hot path.
- **A backend that cannot serve the client's dialect**: the client must get the router's own 400 naming both dialects, and the backend must see zero requests. Today its 404 is forwarded as-is (4xx is not an error), so nothing retries, nothing falls back, and the rule looks healthy.
- **A stream split across arbitrary read boundaries**: `sseRelay` hands the translator complete `\n\n`-terminated frames; the translator must never emit a partial frame, and `FinishFrames()` must close every block it opened (a backend that dies mid-generation must not leave an unterminated Anthropic message).
- **A converted stream that dies mid-generation, or a backend that emits its own error frame**: `sendMidStreamError` authors the frame in the *client's* dialect (it keys on `req.URL.Path`, which the proxy never rewrites), so the translator must pass frames it cannot parse as Chat Completions chunks straight through — swallowing them would leave an Anthropic SDK with a truncated message and no error.
- **A request that falls back across dialects**: the plan, the headers, the metrics and the translator are per-attempt, so the plan recorded must be the serving attempt's — a converted first attempt must not leak its plan into a passthrough second attempt.
- **A converted request whose backend answers with an error**: HTTP 5xx is not an error to this proxy — the body is forwarded as-is — so after conversion it reaches the client in a shape its SDK cannot read, and the backend's actual reason is lost inside a parse failure. An envelope of the other dialect's error shape is translated, not relayed.

---

### Task 1: `convert` package — dialects, plan, refusals

**Files:**
- Create: `convert/dialect.go`
- Create: `convert/plan.go`
- Create: `convert/errors.go`
- Test: `convert/plan_test.go`

**Interfaces:**
- Consumes: `domain.ServerProtocol` — its `Path`/`State` fields are added in Task 2, so do Task 2's `domain/server.go` struct addition first if you build in one pass (the plan's tests construct that struct in one helper).
- Produces:
  - `type Dialect string` with `Chat Dialect = "chat"`, `Messages Dialect = "messages"`, `Responses Dialect = "responses"`, `Legacy Dialect = "completions"` (`convert/dialect.go`)
  - `func DialectForPath(path string) Dialect` — `""` for a path with no dialect (audio, embeddings, systemone)
  - `func PathFor(d Dialect) string` — `chat` → `/v1/chat/completions`, `messages` → `/v1/messages`, `responses` → `/v1/responses`, anything else → `""`
  - `type Plan struct{ From, Dialect Dialect; Path string; Forced bool }`
  - `func Decide(client Dialect, path string, protocols []domain.ServerProtocol, forced string) (Plan, error)` — **the function is `Decide`, not `Plan`**: Go forbids a type and a function with the same name in one package.
  - `type Error struct{ Code, Field, Msg string }` with `func (e *Error) Error() string`, codes `CodeNoDialect = "no_dialect"`, `CodeField = "unsupported_field"`, `CodeState = "state"` (`convert/errors.go`)
  - `type Options struct{ MaxTokens int; DropParams bool; AnthropicVersion string }` (`convert/plan.go`)

- [ ] **Step 1: Write the failing tests**

Create `convert/plan_test.go`:

```go
package convert

import (
	"errors"
	"strings"
	"testing"

	"llm-api-router/domain"
)

func protocols(spec ...string) []domain.ServerProtocol {
	out := make([]domain.ServerProtocol, 0, len(spec))
	for _, s := range spec {
		p, state, _ := strings.Cut(s, "=")
		out = append(out, domain.ServerProtocol{Path: p, State: state == "state"})
	}
	return out
}

func TestDecidePassthrough(t *testing.T) {
	// A server declaring nothing: today's behavior, no conversion.
	p, err := Decide(Messages, "/v1/messages", nil, "auto")
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if p.Dialect != Messages || p.Path != "/v1/messages" {
		t.Fatalf("got %+v, want passthrough to /v1/messages", p)
	}
	// A server declaring the client's own endpoint: passthrough.
	p, err = Decide(Messages, "/v1/messages", protocols("/v1/messages", "/v1/chat/completions"), "auto")
	if err != nil || p.Dialect != Messages {
		t.Fatalf("got %+v, %v, want passthrough", p, err)
	}
}

func TestDecideConvertsToDeclaredDialect(t *testing.T) {
	// A backend that only speaks chat: an Anthropic client is planned onto chat.
	p, err := Decide(Messages, "/v1/messages", protocols("/v1/chat/completions"), "auto")
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if p.Dialect != Chat || p.Path != "/v1/chat/completions" || p.From != Messages {
		t.Fatalf("got %+v, want messages→chat at /v1/chat/completions", p)
	}
}

func TestDecideRefusesWhenNoEdgeExists(t *testing.T) {
	// Responses on a chat-only backend is a phase-4 edge: refused, not guessed.
	_, err := Decide(Responses, "/v1/responses", protocols("/v1/chat/completions"), "auto")
	var ce *Error
	if !errors.As(err, &ce) || ce.Code != CodeNoDialect {
		t.Fatalf("got %v, want *convert.Error{%s}", err, CodeNoDialect)
	}
}

func TestDecideNeverConvertsLegacy(t *testing.T) {
	if _, err := Decide(Legacy, "/v1/completions", protocols("/v1/chat/completions"), "auto"); err == nil {
		t.Fatal("legacy completions must never be converted (the chat template lives in the backend)")
	}
}

func TestDecideForcedEdgeAndOff(t *testing.T) {
	p, err := Decide(Messages, "/v1/messages", nil, "messages>chat")
	if err != nil || p.Dialect != Chat || !p.Forced {
		t.Fatalf("got %+v, %v, want forced messages→chat", p, err)
	}
	p, err = Decide(Messages, "/v1/messages", protocols("/v1/chat/completions"), "off")
	if err != nil || p.Dialect != Messages {
		t.Fatalf("got %+v, %v, want passthrough with convert off", p, err)
	}
	if _, err := Decide(Messages, "/v1/messages", nil, "chat>responses"); err == nil {
		t.Fatal("an unknown forced edge must be an error")
	}
}

func TestDecideDialectlessPathsArePassthrough(t *testing.T) {
	// Audio, embeddings, systemone: no dialect, so never a conversion subject.
	p, err := Decide("", "/v1/audio/speech", protocols("/v1/chat/completions"), "auto")
	if err != nil || p.Dialect != "" || p.Path != "/v1/audio/speech" {
		t.Fatalf("got %+v, %v, want untouched passthrough", p, err)
	}
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `go test ./convert`
Expected: FAIL — `undefined: Messages` / `undefined: Decide` (the package does not exist yet).

- [ ] **Step 3: Write the implementation**

Create `convert/dialect.go`:

```go
// Package convert translates requests, responses and SSE streams between the API
// dialects the router proxies (OpenAI Chat Completions, OpenAI Responses,
// Anthropic Messages). The hub is chat: every other dialect is defined as a spoke
// around it, so N dialects cost 2N transforms, not N².
package convert

import "strings"

// Dialect names an API dialect by the endpoint that carries it.
type Dialect string

const (
	Chat      Dialect = "chat"
	Messages  Dialect = "messages"
	Responses Dialect = "responses"
	Legacy    Dialect = "completions"
)

// DialectForPath names the dialect a request path belongs to. The empty string
// means the path carries no dialect this package can translate: audio uploads,
// embeddings, llama.cpp's /v1/systemone. Those are never conversion subjects.
//
// Order matters: /v1/chat/completions contains /completions.
func DialectForPath(path string) Dialect {
	switch {
	case strings.Contains(path, "/messages"):
		return Messages
	case strings.Contains(path, "/chat/completions"):
		return Chat
	case strings.Contains(path, "/responses"):
		return Responses
	case strings.Contains(path, "/completions"):
		return Legacy
	}
	return ""
}

// PathFor returns the canonical endpoint path carrying a dialect ("" for the
// dialectless and legacy families, which are never converted).
func PathFor(d Dialect) string {
	switch d {
	case Chat:
		return "/v1/chat/completions"
	case Messages:
		return "/v1/messages"
	case Responses:
		return "/v1/responses"
	}
	return ""
}
```

Create `convert/errors.go`:

```go
package convert

// Error is a conversion refusal: the request must not reach the backend, because
// speaking the backend's dialect would silently answer a different question than
// the one the client asked.
type Error struct {
	Code  string // CodeNoDialect, CodeField, CodeState
	Field string // the request field at issue ("" when the whole dialect is the issue)
	Msg   string // the client-facing sentence
}

const (
	CodeNoDialect = "no_dialect"
	CodeField     = "unsupported_field"
	CodeState     = "state"
)

func (e *Error) Error() string { return e.Msg }
```

Create `convert/plan.go`:

```go
package convert

import (
	"fmt"
	"strings"

	"llm-api-router/domain"
)

// Plan is the per-attempt decision: which dialect and path this attempt speaks,
// and which dialect it is translating from.
type Plan struct {
	From    Dialect
	Dialect Dialect
	Path    string
	Forced  bool // `convert` named the edge explicitly
}

// Options are the conversion knobs (see domain.ConvertSettings). The router builds
// them from settings once per request; AnthropicVersion is used by the inverse
// spoke (phase 3).
type Options struct {
	MaxTokens        int
	DropParams       bool
	AnthropicVersion string
}

// Decide picks the upstream dialect for one attempt.
//
// A backend that declares no protocols keeps today's behavior exactly: the
// client's own path, no translation. Otherwise the client's endpoint is looked up
// in the declaration; a backend that does not serve it is planned onto the one
// dialect the backend does serve, for the edges this build implements
// (messages→chat). Anything else is a refusal, never a guess.
//
// forced is rule/fallback/candidate `convert`: "auto" (or "") decides by the
// declaration, "off" forces passthrough, "messages>chat" names the edge.
func Decide(client Dialect, path string, protocols []domain.ServerProtocol, forced string) (Plan, error) {
	ident := Plan{From: client, Dialect: client, Path: path}
	if forced == "off" || client == "" || path == "" {
		return ident, nil
	}
	if forced != "" && forced != "auto" {
		from, to, ok := strings.Cut(forced, ">")
		if !ok || Dialect(from) != client {
			return Plan{}, &Error{Code: CodeNoDialect, Msg: fmt.Sprintf(
				"convert %q does not apply to a %s request", forced, client)}
		}
		if from == string(Messages) && to == string(Chat) {
			return Plan{From: client, Dialect: Chat, Path: PathFor(Chat), Forced: true}, nil
		}
		return Plan{}, &Error{Code: CodeNoDialect, Msg: fmt.Sprintf(
			"conversion %s is not supported", forced)}
	}
	if client == Legacy {
		// /v1/completions wants a rendered prompt: the chat template lives in the
		// backend's tokenizer, which the proxy does not have.
		return Plan{}, &Error{Code: CodeNoDialect, Field: "path", Msg:
			"converting /v1/completions is not supported: the chat template is applied by the backend"}}
	}
	if len(protocols) == 0 {
		return ident, nil
	}
	for _, p := range protocols {
		if p.Path == path {
			return ident, nil
		}
	}
	if client == Messages {
		for _, p := range protocols {
			if DialectForPath(p.Path) == Chat {
				return Plan{From: client, Dialect: Chat, Path: p.Path}, nil
			}
		}
	}
	return Plan{}, &Error{Code: CodeNoDialect, Msg: fmt.Sprintf(
		"backend serves %s, not %s and no conversion exists for it",
		strings.Join(declaredPaths(protocols), ", "), path)}
}

func declaredPaths(protocols []domain.ServerProtocol) []string {
	out := make([]string, 0, len(protocols))
	for _, p := range protocols {
		out = append(out, p.Path)
	}
	return out
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./convert && task lint`
Expected: PASS, lint `0 issues`.

- [ ] **Step 5: Commit**

```bash
git add convert/ && git commit -m "feat(convert): dialects, per-attempt plan, refusal errors"
```

---

### Task 2: Declared capability and conversion settings

**Files:**
- Modify: `domain/server.go` (`Server`, `FallbackEntry`, new `ServerProtocol`)
- Modify: `domain/rule.go` (`RoutingRule`, `JevCandidate`)
- Modify: `domain/config.go` (`Settings`, new `ConvertSettings`)
- Modify: `config/store.go` (`AddServer`, `UpdateServer`, `cloneRule`)
- Modify: `example.config.json`
- Test: `config/store_test.go`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `type ServerProtocol struct{ Path string; State bool }` and `Server.Protocols []ServerProtocol` (`domain/server.go`)
  - `RoutingRule.Convert string`, `FallbackEntry.Convert string`, `JevCandidate.Convert string` — all `"auto"`/`""` (decide by declaration), `"off"` (never convert), or an explicit `"messages>chat"`
  - `type ConvertSettings struct{ MaxTokensDefault int; DropParams bool; Probe bool; AnthropicVersion string }`, `Settings.Convert ConvertSettings`, and the resolved getters `func (c ConvertSettings) MaxTokens() int` (0 → 4096) / `func (c ConvertSettings) Version() string` ("" → `2023-06-01`)

- [ ] **Step 1: Write the failing test**

Append to `config/store_test.go` (its imports already include `encoding/json`, `path/filepath`, `testing`, `llm-api-router/domain`):

```go
func TestServerProtocolsAndRuleConvertSurviveClone(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	s, err := NewStore(path)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	if err := s.AddServer(&domain.Server{ID: "s1", Name: "n", URL: "http://h:8080/v1",
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	got, ok := s.GetServer("s1")
	if !ok || len(got.Protocols) != 1 || got.Protocols[0].Path != "/v1/chat/completions" {
		t.Fatalf("AddServer dropped Protocols: %+v", got)
	}
	// The stored slice must be a copy: AddServer/UpdateServer take the caller's
	// pointer, so mutating the caller's slice must not reach the store.
	src := []domain.ServerProtocol{{Path: "/v1/messages", State: true}}
	if err := s.UpdateServer("s1", &domain.Server{ID: "s1", Name: "n", URL: "http://h:8080/v1",
		Protocols: src}); err != nil {
		t.Fatalf("UpdateServer: %v", err)
	}
	src[0].Path = "/tampered"
	got, _ = s.GetServer("s1")
	if got.Protocols[0].Path != "/v1/messages" {
		t.Fatalf("UpdateServer aliased the caller's slice: %+v", got.Protocols)
	}

	if err := s.AddRule(&domain.RoutingRule{IncomingModels: []string{"m"}, ServerID: "s1",
		Convert: "messages>chat", Enabled: true,
		Fallbacks: []domain.FallbackEntry{{ServerID: "s2", Convert: "off"}}}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	r, ok := s.GetRuleByModel("m")
	if !ok || r.Convert != "messages>chat" {
		t.Fatalf("cloneRule dropped rule Convert: %+v", r)
	}
	if len(r.Fallbacks) != 1 || r.Fallbacks[0].Convert != "off" {
		t.Fatalf("cloneRule dropped fallback Convert: %+v", r.Fallbacks)
	}
	// The new fields must survive the round trip through the config file.
	if err := s.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	var persisted domain.Config
	if err := json.Unmarshal(raw, &persisted); err != nil {
		t.Fatalf("config is not valid JSON: %v", err)
	}
	if p := persisted.Servers["s1"]; p == nil || len(p.Protocols) != 1 {
		t.Fatalf("Protocols not persisted: %s", raw)
	}
}

func TestConvertSettingsDefaults(t *testing.T) {
	var c domain.ConvertSettings
	if c.MaxTokens() != 4096 {
		t.Fatalf("MaxTokens() = %d, want the 4096 default", c.MaxTokens())
	}
	if c.Version() != "2023-06-01" {
		t.Fatalf("Version() = %q, want the anthropic default", c.Version())
	}
	c = domain.ConvertSettings{MaxTokensDefault: 1024, AnthropicVersion: "2024-10-22"}
	if c.MaxTokens() != 1024 || c.Version() != "2024-10-22" {
		t.Fatalf("explicit values must win: %+v", c)
	}
}
```

(`os`, `encoding/json`, `path/filepath`, `testing` and `llm-api-router/domain` are already imported by that file.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./config -run 'Protocols|ConvertSettings' -v`
Expected: FAIL — `unknown field Protocols in struct literal` / `undefined: domain.ConvertSettings`.

- [ ] **Step 3: Write the implementation**

`domain/server.go` — in the `Server` struct, right after `APITypes []APIType`:

```go
	// Protocols declares the endpoints this backend natively serves, which is what
	// decides whether a request has to be converted (see convert.Decide). Empty
	// means "undeclared": the router keeps its passthrough behavior, so a config
	// written before this field exists never changes behavior.
	Protocols []ServerProtocol `json:"protocols,omitempty"`
```

at the end of `domain/server.go`:

```go
// ServerProtocol declares one endpoint a backend serves. State records whether the
// backend keeps server-side conversation state (`previous_response_id`), which no
// local engine does by default; the router never assumes it.
type ServerProtocol struct {
	Path  string `json:"path"`
	State bool   `json:"state,omitempty"`
}
```

`FallbackEntry` (in `domain/server.go`) gains:

```go
	// Convert overrides protocol conversion on this fallback: ""/"auto" (decide by
	// the target's declared protocols), "off" (never convert), or an explicit
	// "messages>chat".
	Convert string `json:"convert,omitempty"`
```

`domain/rule.go`: `RoutingRule` gains the same field with the doc comment reading "this rule's primary attempt"; `JevCandidate` gains it with "this candidate".

`domain/config.go`: `Settings` gains

```go
	Convert ConvertSettings `json:"convert,omitempty"`
```

and, below the `TraceSettings` type:

```go
// ConvertSettings are the global conversion knobs. MaxTokensDefault is the output
// budget injected into a converted Anthropic request whose client sent none
// (Anthropic requires the field, OpenAI clients do not send it); the default 4096
// is LiteLLM's DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS. DropParams turns the default
// refusal (a 400 naming the field) into a silent drop recorded in
// RequestMetric.ConvertNotes. Probe is reserved for the capability probe (out of
// scope here); AnthropicVersion is the anthropic_version a synthesized Anthropic
// request carries (phase 3).
type ConvertSettings struct {
	MaxTokensDefault int  `json:"max_tokens_default,omitempty"`
	DropParams       bool `json:"drop_params,omitempty"`
	Probe            bool `json:"probe,omitempty"`
	AnthropicVersion string `json:"anthropic_version,omitempty"`
}

func (c ConvertSettings) MaxTokens() int {
	if c.MaxTokensDefault <= 0 {
		return 4096
	}
	return c.MaxTokensDefault
}

func (c ConvertSettings) Version() string {
	if c.AnthropicVersion == "" {
		return "2023-06-01"
	}
	return c.AnthropicVersion
}
```

`config/store.go` — in `AddServer` and `UpdateServer`, next to the existing `APITypes` deep copy:

```go
	protos := make([]domain.ServerProtocol, len(srv.Protocols))
	copy(protos, srv.Protocols)
	added.Protocols = protos   // updated.Protocols = protos in UpdateServer
```

and in `cloneRule`'s `domain.RoutingRule` literal, alongside the other copied fields:

```go
		Convert: rule.Convert,
```

(`Fallbacks`/`Candidates` are copied whole, so their new `Convert` fields ride along.)

`example.config.json`: add to the first server entry that declares both protocols:

```json
      "protocols": [{ "path": "/v1/chat/completions" }, { "path": "/v1/messages" }],
```

and inside the existing `"settings"` object, next to `"trace"`:

```json
    "convert": {
      "max_tokens_default": 4096,
      "drop_params": false,
      "probe": false,
      "anthropic_version": "2023-06-01"
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./config ./domain ./convert && task lint`
Expected: PASS, lint `0 issues`.

- [ ] **Step 5: Commit**

```bash
git add domain/ config/ example.config.json && git commit -m "feat(config): declared server protocols and convert settings"
```

---

### Task 3: Per-attempt plan wired into routing, metrics and headers

**Files:**
- Modify: `router/router.go`
- Modify: `domain/metric.go`
- Modify: `proxy/proxy.go` (`RouterHeaders`, `SetRouterHeaders`, `StreamProxy` path join)
- Test: `router/convert_test.go`

**Interfaces:**
- Consumes: `convert.Decide`, `convert.DialectForPath`, `*convert.Error`, `Server.Protocols`, `RoutingRule.Convert`, `FallbackEntry.Convert`, `JevCandidate.Convert`, `Settings.Convert`.
- Produces:
  - `RequestMetric.UpstreamEndpoint string` / `.Convert string` / `.ConvertNotes []string` (`domain/metric.go`)
  - `proxy.RouterHeaders.UpstreamPath string` (empty = the client's path), `proxy.RouterHeaders.Convert string`
  - `serverAttempt.convert string` (the attempt's own `convert` value), `reqMeta.upstreamPath/convert/convertNotes`
  - request header `X-Router-Convert: off` honored from the client; response header `X-Router-Convert`

- [ ] **Step 1: Write the failing tests**

Create `router/convert_test.go`:

```go
package router

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"llm-api-router/domain"
)

// A backend that declares nothing is byte-for-byte passthrough: the client's path
// and body reach it untouched, and no conversion is reported.
func TestHandleNoProtocolsIsPassthrough(t *testing.T) {
	var gotPath, gotBody string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		gotPath, gotBody = r.URL.Path, string(b)
		_, _ = w.Write([]byte(`{"id":"c1","model":"m","choices":[{"message":{"role":"assistant"}}]}`))
	}))
	defer backend.Close()

	r, store, ms, _ := newTestRouter(t)
	addTestRoute(t, store, "m", backend.URL)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", w.Result().StatusCode, w.Body.String())
	}
	if gotPath != "/v1/chat/completions" {
		t.Fatalf("backend path = %q, want the client's path", gotPath)
	}
	if h := w.Result().Header.Get("X-Router-Convert"); h != "" {
		t.Fatalf("X-Router-Convert = %q, want absent", h)
	}
	if !strings.Contains(gotBody, `"content":"hi"`) {
		t.Fatalf("body was rewritten: %s", gotBody)
	}
	rec := ms.Recent()
	if len(rec) != 1 || rec[0].Convert != "" || rec[0].UpstreamEndpoint != "/v1/chat/completions" {
		t.Fatalf("metric: %+v", rec)
	}
}

// A backend that speaks neither the client's endpoint nor a dialect the router can
// reach gets a 400 from us, naming what it serves — and the backend saw nothing.
// (Before the converter its 404 was forwarded as-is — 4xx is not an error — so
// nothing retried, nothing fell back, and the rule looked healthy.)
func TestHandleProtocolMismatchRefusesWithOur400(t *testing.T) {
	var hits int
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		http.Error(w, "not found", http.StatusNotFound)
	}))
	defer backend.Close()

	r, store, ms, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: backend.URL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/responses"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"}, ServerID: "s1",
		TargetModel: "claude", NumRetries: -1, Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if got := w.Result().StatusCode; got != http.StatusBadRequest {
		t.Fatalf("status %d, want 400: %s", got, w.Body.String())
	}
	if hits != 0 {
		t.Fatalf("backend got %d requests, want 0", hits)
	}
	if body := w.Body.String(); !strings.Contains(body, "/v1/responses") {
		t.Fatalf("refusal must name what the backend serves: %s", body)
	}
	rec := ms.Recent()
	if len(rec) != 1 || rec[0].StatusCode != http.StatusBadRequest {
		t.Fatalf("the refusal must be recorded: %+v", rec)
	}
}

// `convert: off` — from the rule and from the request header — sends the client's
// own request upstream, exactly as before this feature. The backend declares only
// chat, so without the off these two requests would be converted.
func TestHandleConvertOff(t *testing.T) {
	var hits int
	var gotPath string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		gotPath = r.URL.Path
		_, _ = w.Write([]byte(`{"type":"message","role":"assistant","content":[],"model":"claude"}`))
	}))
	defer backend.Close()

	r, store, _, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: backend.URL,
		APITypes:  []domain.APIType{domain.APITypeAnthropic},
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"}, ServerID: "s1",
		TargetModel: "claude", NumRetries: -1, Enabled: true, Convert: "off"}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	send := func(header bool) *http.Response {
		req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
			`{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`))
		if header {
			req.Header.Set("X-Router-Convert", "off")
		}
		w := httptest.NewRecorder()
		r.Handle(w, req)
		return w.Result()
	}
	off := send(true)
	if h := off.Header.Get("X-Router-Convert"); h != "" {
		t.Fatalf("X-Router-Convert = %q, want absent", h)
	}
	send(false) // the rule's own `convert: off`, same decision
	if hits != 2 || gotPath != "/v1/messages" {
		t.Fatalf("off must send the client's own request upstream: hits=%d path=%q", hits, gotPath)
	}
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `go test ./router -run 'Passthrough|ProtocolMismatch|ConvertOff' -v`
Expected: `TestHandleProtocolMismatchRefusesWithOur400` FAILS with status 404 and `hits == 1` (today the backend's own 404 rides to the client). The other two PASS — the undeclared-passthrough and off-passthrough guards, pinning that this feature changes nothing for a config that does not ask for it; a passing guard is their job.

- [ ] **Step 3: Write the implementation**

`domain/metric.go`, after `APIEndpoint string`:

```go
	// UpstreamEndpoint is the path the serving attempt actually used, Convert
	// names the conversion (`messages>chat`, empty = none), ConvertNotes lists
	// what a deliberate drop cost. The client's own endpoint stays in APIEndpoint:
	// the class table answers "what did the client send".
	UpstreamEndpoint string   `json:"upstream_endpoint,omitempty"`
	Convert          string   `json:"convert,omitempty"`
	ConvertNotes     []string `json:"convert_notes,omitempty"`
```

`router/router.go` — add the import `llm-api-router/convert`; extend `reqMeta` (`router.go:48`):

```go
type reqMeta struct {
	class        string
	features     []string
	capture      *trace.Capture
	retries      int
	upstreamPath string
	convert      string
	convertNotes []string
}
```

and `serverAttempt` (`router.go:56`):

```go
type serverAttempt struct {
	server      *domain.Server
	targetModel string
	convert     string // the attempt's own `convert` value ("" = "auto")
}
```

Fill it at the three construction sites: `router.go:200` (`convert: rule.Convert`), `router.go:207` (the fallback loop → the entry's `Convert`), `router/jev.go:295` (the candidate loop → the candidate's `Convert`).

In `Handle`, next to `apiType := apiTypeFromPath(req.URL.Path)` add:

```go
	clientDialect := convert.DialectForPath(req.URL.Path)
	headerConvert := req.Header.Get("X-Router-Convert")
	convSettings := r.store.GetSettings().Convert
```

Insert the decision into the attempt loop, immediately **before** `rewrittenBody, err := rewriteAttemptBody(...)` (`router.go:294`), and the body transform immediately **after** it:

```go
		// The dialect decision is per attempt: a fallback server and a jev
		// candidate each declare their own capability, and a per-request header
		// outranks the config.
		forced := attempt.convert
		if headerConvert == "off" {
			forced = "off"
		}
		plan, planErr := convert.Decide(clientDialect, req.URL.Path, srv.Protocols, forced)
		if planErr != nil {
			// A backend this attempt can not speak to is a refusal, not a
			// transport failure: nothing was sent, so no health/rate-limit/quota
			// bookkeeping happens. Try the next attempt; if the last one refuses
			// too, the client gets our 400 naming the fields.
			attemptErrors = append(attemptErrors, planErr.Error())
			if i < len(attempts)-1 {
				log.Warnf("[%s] model=%q — skipping %s: %v", req.URL.Path, model, srv.Name, planErr)
				continue
			}
			refusal = planErr
			break
		}
		// Assigned on every attempt so the recorded metric describes the attempt
		// that served, never an earlier failed one.
		meta.upstreamPath, meta.convert, meta.convertNotes = plan.Path, "", nil
		if plan.Dialect != plan.From && plan.From != "" {
			meta.convert = string(plan.From) + ">" + string(plan.Dialect)
		}
```

and after the `rewriteAttemptBody` error check:

```go
		if meta.convert != "" {
			converted, notes, convErr := convert.Request(plan, rewrittenBody, convert.Options{
				MaxTokens:        convSettings.MaxTokens(),
				DropParams:       convSettings.DropParams,
				AnthropicVersion: convSettings.Version(),
			})
			if convErr != nil {
				attemptErrors = append(attemptErrors, convErr.Error())
				if i < len(attempts)-1 {
					log.Warnf("[%s] model=%q — skipping %s: %v", req.URL.Path, model, srv.Name, convErr)
					continue
				}
				refusal = convErr
				break
			}
			rewrittenBody, meta.convertNotes = converted, notes
		}
```

Declare `var refusal error` beside `var lastErr error` (`router.go:263`). Replace `serverURL := srv.GetURLForAPIType(apiType)` (`router.go:323`) with:

```go
		// chat, responses and legacy all ride the server's OpenAI URL; only
		// messages is Anthropic.
		upstreamType := domain.APITypeOpenAI
		if plan.Dialect == convert.Messages {
			upstreamType = domain.APITypeAnthropic
		}
		serverURL := srv.GetURLForAPIType(upstreamType)
```

Add to the `rh := &proxy.RouterHeaders{...}` literal (`router.go:348`):

```go
			UpstreamPath: plan.Path,
			Convert:      meta.convert,
			ConvertNotes: meta.convertNotes,
```

Before the tail's all-backends-failed `r.record(...)`/`http.Error(...)` (`router.go:483-490`), insert:

```go
	if refusal != nil {
		r.record(req, requestStart, model, attempts[0].targetModel, attempts[0].server.ID,
			proxy.ProxyMetrics{}, http.StatusBadRequest, refusal.Error(),
			len(attempts) > 1, apiType, jev, meta)
		log.Warnf("[%s] model=%q — refused: %v", req.URL.Path, model, refusal)
		http.Error(w, refusal.Error(), http.StatusBadRequest)
		return
	}
```

and in `record`, add to the `domain.RequestMetric` literal:

```go
		UpstreamEndpoint: meta.upstreamPath,
		Convert:          meta.convert,
		ConvertNotes:     meta.convertNotes,
```

`proxy/proxy.go` — `RouterHeaders` (after `KeepAliveIdle`):

```go
	// UpstreamPath replaces the client's path when a conversion points the request
	// at a different endpoint ("" = the client's path). Not a client-visible header.
	UpstreamPath string
	// Convert names the conversion in flight (`messages>chat`) → X-Router-Convert.
	Convert string
	// ConvertNotes lists what a deliberate drop cost → X-Router-Convert-Notes.
	ConvertNotes []string
```

`SetRouterHeaders`, beside the other conditional sets:

```go
	if h.Convert != "" {
		w.Header().Set("X-Router-Convert", h.Convert)
	}
	if len(h.ConvertNotes) > 0 {
		w.Header().Set("X-Router-Convert-Notes", strings.Join(h.ConvertNotes, ", "))
	}
```

The header map persists across attempts (`X-Router-Retries` note in CLAUDE.md), so `SetRouterHeaders` must *delete* the two convert headers when the current attempt has none — a dropped field from a refused first attempt must not ride on a passthrough second attempt:

```go
	if h.Convert == "" {
		w.Header().Del("X-Router-Convert")
		w.Header().Del("X-Router-Convert-Notes")
	}
```

`StreamProxy`: the path join currently reads the client's path (`proxy.go:1139-1151`). Introduce the override once, above it:

```go
	reqPath := req.URL.Path
	if rh != nil && rh.UpstreamPath != "" {
		reqPath = rh.UpstreamPath
	}
	targetBase := strings.TrimRight(target.Path, "/")
	targetPath := targetBase + reqPath
```

and change the two `/v1` dedup branches to use `reqPath` (the `/v1` prefix test and the root-served audio suffix drop both switch from `req.URL.Path` to `reqPath`). The mid-stream and loop-detector error frames keep keying on `req.URL.Path` — that is the client's dialect, which is what the client must read.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./router ./proxy ./convert ./domain -count=1 && task lint`
Expected: PASS, lint `0 issues`.

- [ ] **Step 5: Commit**

```bash
git add router/ proxy/ domain/ && git commit -m "feat(router): per-attempt dialect plan, refusal 400, conversion observability"
```

---

### Task 4: Stream-usage injection moves into the attempt

**Files:**
- Modify: `router/router.go`
- Test: `router/convert_test.go`

**Interfaces:**
- Consumes: `plan` from Task 3.
- Produces: `stripStream` decided per attempt from the upstream dialect (the `proxy.StreamProxy(..., stripStream, ping)` call at `router.go:384` is unchanged).

`EnsureStreamUsage` injects OpenAI's `stream_options.include_usage`. On an Anthropic upstream that is a field the backend never heard of, and once one client request can be aimed at a backend of a different dialect, "is the upstream chat" is no longer derivable from the client's path.

- [ ] **Step 1: Write the failing tests**

Append to `router/convert_test.go`:

```go
// stream_options is an OpenAI field: it must follow the dialect decision, not the
// client's path. An Anthropic request on an Anthropic backend must arrive with no
// stream_options in the body.
func TestHandleAnthropicStreamGetsNoStreamOptions(t *testing.T) {
	var gotBody string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		_, _ = w.Write([]byte(`{"id":"m1","type":"message","role":"assistant","content":[],"model":"claude"}`))
	}))
	defer backend.Close()

	r, store, _, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: backend.URL,
		APITypes:  []domain.APIType{domain.APITypeAnthropic},
		Protocols: []domain.ServerProtocol{{Path: "/v1/messages"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"}, ServerID: "s1",
		TargetModel: "claude", NumRetries: -1, Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","max_tokens":10,"stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	r.Handle(httptest.NewRecorder(), req)

	if strings.Contains(gotBody, "stream_options") {
		t.Fatalf("OpenAI stream_options leaked into an Anthropic request: %s", gotBody)
	}
}

// The mirror case: an OpenAI streaming request on a chat backend keeps the
// injection, now decided per attempt.
func TestHandleChatStreamStillInjectsUsage(t *testing.T) {
	var gotBody string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		_, _ = w.Write([]byte(`{"id":"c1","model":"m","choices":[]}`))
	}))
	defer backend.Close()

	r, store, _, _ := newTestRouter(t)
	addTestRoute(t, store, "m", backend.URL)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	r.Handle(httptest.NewRecorder(), req)

	if !strings.Contains(gotBody, `"include_usage":true`) {
		t.Fatalf("injection lost: %s", gotBody)
	}
}
```

- [ ] **Step 2: Run the tests to see where the pair stands**

Run: `go test ./router -run 'StreamGetsNoStreamOptions|ChatStreamStillInjectsUsage' -v`
Expected: both PASS today — today's pre-loop check keys on the client's protocol, which is correct for both of these. They are written before the move so the move is provably behavior-preserving; the case that needs the move is the cross-dialect one Task 7 adds, and it is these two that catch a regression there.

- [ ] **Step 3: Write the implementation**

Delete the pre-loop block at `router.go:241-250` (the `stripStream := false` declaration plus the `if apiType == domain.APITypeOpenAI && strings.Contains(req.URL.Path, "/chat/completions")` injection), and put this inside the attempt loop, immediately after the conversion block from Task 3, so it injects into the body in the dialect this attempt speaks:

```go
		// stream_options.include_usage is an OpenAI field: injected into the body
		// in the dialect this attempt speaks, after any conversion. stripStream
		// hides the injected usage chunk from the client (unless configured
		// otherwise).
		stripStream := false
		if plan.Dialect == convert.Chat && strings.Contains(plan.Path, "/chat/completions") {
			var injected bool
			rewrittenBody, injected = proxy.EnsureStreamUsage(rewrittenBody,
				r.store.GetSettings().AlwaysIncludeStreamUsage)
			stripStream = injected
		}
```

`stripStream` is now per attempt; the `proxy.StreamProxy(...)` call reads it inside the retry loop. The `ping` block below it (`router.go:251-262`) keeps keying on `apiType` — heartbeats are for the client, and they are injected below the translator.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./router -count=1 && task lint`
Expected: PASS (the whole router package, including the existing streaming-usage tests), lint `0 issues`.

- [ ] **Step 5: Commit**

```bash
git add router/ && git commit -m "fix(router): decide stream-usage injection per attempt dialect"
```

---

### Task 5: `POST /admin/api/convert/preview` — dry-run transform

**Files:**
- Create: `admin/convert_preview.go`
- Modify: `admin/handler.go` (`ServeHTTP` route switch)
- Test: `admin/convert_test.go`

**Interfaces:**
- Consumes: `convert.Decide`, `convert.Request` (identity-only in this task; Task 6 gives it the spoke), `Server.Protocols`, `RoutingRule.Convert`, `Settings.Convert`.
- Produces:
  - admin route `POST /admin/api/convert/preview` — body `{"model": "...", "path": "...", "body": {...}}`, response `{"from","dialect","upstream_path","convert","request","notes"}` for a plan it can serve, `{"error","code",...}` with status 200 for a refusal
  - `func (h *Handler) convertPreview(w http.ResponseWriter, req *http.Request)`
  - `func convertPreviewFor(store *config.Store, model, path string, body json.RawMessage) (map[string]any, int)`
  - `func convertOptions(store *config.Store) convert.Options`

- [ ] **Step 1: Write the failing test**

Create `admin/convert_test.go`:

```go
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./admin -run ConvertPreview -v`
Expected: FAIL — the route 404s, surfacing as `decode: invalid character '<'`.

- [ ] **Step 3: Write the implementation**

`admin/handler.go`, add the import `llm-api-router/convert` and a case before `default:`:

```go
	case path == "/convert/preview" && req.Method == http.MethodPost:
		h.convertPreview(w, req)
```

Create `admin/convert_preview.go`:

```go
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
```

`convert/plan.go` — add the identity-only entry point now, so the preview compiles and Task 6 replaces only its body:

```go
// Request translates a request body for a plan. Identity plans return the body
// untouched. The notes list what a deliberate drop cost.
func Request(p Plan, body []byte, o Options) ([]byte, []string, error) {
	if p.Dialect == p.From || p.From == "" {
		return body, nil, nil
	}
	return nil, nil, &Error{Code: CodeNoDialect, Msg: "conversion " + string(p.From) + ">" +
		string(p.Dialect) + " is not supported"}
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./admin ./convert -count=1 && task lint`
Expected: PASS, lint `0 issues`.

- [ ] **Step 5: Commit**

```bash
git add admin/ convert/ && git commit -m "feat(admin): dry-run conversion preview endpoint"
```

---

### Task 6: `convert.Request` — the `messages → chat` spoke

**Files:**
- Create: `convert/anthropic.go`
- Modify: `convert/plan.go` (`Request` dispatch)
- Test: `convert/anthropic_test.go`, `admin/convert_test.go` (upgrade the plan-only assertion)

**Interfaces:**
- Consumes: `Plan`, `Options`, `*Error`, codes.
- Produces:
  - `func Request(p Plan, body []byte, o Options) ([]byte, []string, error)` — dispatches `Messages → Chat` to `MessagesToChat`, identity otherwise, refuses everything else
  - `func MessagesToChat(body []byte, o Options) ([]byte, []string, error)`

- [ ] **Step 1: Write the failing tests**

Create `convert/anthropic_test.go`:

```go
package convert

import (
	"encoding/json"
	"errors"
	"testing"
)

func decode(t *testing.T, b []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("not JSON: %v (%s)", err, b)
	}
	return m
}

func fieldErr(err error, want string) (code, field string, ok bool) {
	var ce *Error
	if !errors.As(err, &ce) {
		return "", "", false
	}
	if want != "" && ce.Field != want {
		return ce.Code, ce.Field, false
	}
	return ce.Code, ce.Field, true
}

func TestMessagesToChatCore(t *testing.T) {
	in := `{"model":"claude","system":[{"type":"text","text":"be terse"}],
		"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]},
			{"role":"assistant","content":[{"type":"tool_use","id":"tu_1","name":"fetch","input":{"url":"u"}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":"ok"}]}],
		"max_tokens":50,"stop_sequences":["END"],"temperature":0.2,
		"tools":[{"name":"fetch","description":"d","input_schema":{"type":"object"}}],
		"tool_choice":{"type":"any"},"metadata":{"user_id":"u-1"}}`

	out, notes, err := MessagesToChat([]byte(in), Options{})
	if err != nil {
		t.Fatalf("MessagesToChat: %v", err)
	}
	m := decode(t, out)
	if m["system"] != nil {
		t.Fatalf("system must become a message, not stay top-level: %v", m["system"])
	}
	// system + user + assistant(tool_calls) + tool
	msgs, _ := m["messages"].([]any)
	if len(msgs) != 4 {
		t.Fatalf("got %d messages, want 4 (system, user, assistant, tool): %+v", len(msgs), msgs)
	}
	first, _ := msgs[0].(map[string]any)
	if first["role"] != "system" || first["content"] != "be terse" {
		t.Fatalf("system message: %+v", msgs[0])
	}
	asst, _ := msgs[2].(map[string]any)
	if asst["role"] != "assistant" {
		t.Fatalf("msgs[2] must be the assistant turn: %+v", msgs[2])
	}
	calls, _ := asst["tool_calls"].([]any)
	if len(calls) != 1 {
		t.Fatalf("tool_use not mapped to tool_calls: %+v", msgs[2])
	}
	call, _ := calls[0].(map[string]any)
	if call["id"] != "tu_1" || call["type"] != "function" {
		t.Fatalf("tool call shape: %+v", call)
	}
	toolMsg, _ := msgs[3].(map[string]any)
	if toolMsg["role"] != "tool" || toolMsg["tool_call_id"] != "tu_1" || toolMsg["content"] != "ok" {
		t.Fatalf("tool_result not mapped to a role:tool message: %+v", msgs[3])
	}
	if notes != nil {
		t.Fatalf("a lossless request must produce no notes: %v", notes)
	}
	if got, _ := m["stop"].([]any); len(got) != 1 || got[0] != "END" {
		t.Fatalf("stop_sequences not mapped to stop: %v", m["stop"])
	}
	if m["user"] != "u-1" {
		t.Fatalf("metadata.user_id not mapped to user: %v", m["user"])
	}
	if m["tool_choice"] != "required" {
		t.Fatalf("tool_choice any must map to required: %v", m["tool_choice"])
	}
	tools, _ := m["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools not mapped: %v", m["tools"])
	}
	tl, _ := tools[0].(map[string]any)
	fn, _ := tl["function"].(map[string]any)
	if fn["parameters"] == nil {
		t.Fatalf("input_schema not mapped to parameters: %+v", tl)
	}
}

func TestMessagesToChatRequiresMaxTokensBudget(t *testing.T) {
	// Anthropic requires max_tokens, OpenAI clients may omit it: the injected
	// default comes from Options, and with no budget configured we refuse.
	_, _, err := MessagesToChat([]byte(`{"model":"m","messages":[]}`), Options{})
	if _, _, ok := fieldErr(err, "max_tokens"); !ok {
		t.Fatalf("refusal must name max_tokens: %v", err)
	}
	out, notes, err := MessagesToChat([]byte(`{"model":"m","messages":[]}`), Options{MaxTokens: 4096})
	if err != nil {
		t.Fatalf("with a budget configured: %v", err)
	}
	if m := decode(t, out); m["max_tokens"] != float64(4096) {
		t.Fatalf("budget not injected: %v", m)
	}
	if len(notes) != 1 {
		t.Fatalf("an injected default must be reported: %v", notes)
	}
}

func TestMessagesToChatRefusesUnrepresentable(t *testing.T) {
	cases := map[string]string{
		"thinking":    `{"model":"m","max_tokens":1,"messages":[],"thinking":{"type":"enabled","budget_tokens":100}}`,
		"top_k":       `{"model":"m","max_tokens":1,"top_k":5,"messages":[]}`,
		"document":    `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"text","media_type":"text/plain","data":"x"}}]}]}`,
		"hosted tool": `{"model":"m","max_tokens":1,"messages":[],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`,
		"tool image":  `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAA"}}]}]}]}`,
	}
	for name, in := range cases {
		code, field, ok := fieldErr(mustFail(t, in), "")
		if !ok || code != CodeField || field == "" {
			t.Fatalf("%s: want *convert.Error{%s} naming a field, got code=%q field=%q",
				name, CodeField, code, field)
		}
	}
	// drop_params turns the refusal into a reported drop.
	_, notes, err := MessagesToChat([]byte(`{"model":"m","max_tokens":1,"top_k":5,"messages":[]}`),
		Options{DropParams: true})
	if err != nil {
		t.Fatalf("drop_params: %v", err)
	}
	if len(notes) != 1 || notes[0] != "top_k=dropped" {
		t.Fatalf("drop not reported: %v", notes)
	}
}

func mustFail(t *testing.T, body string) error {
	t.Helper()
	_, _, err := MessagesToChat([]byte(body), Options{})
	if err == nil {
		t.Fatalf("%s: want refusal, got nil", body)
	}
	return err
}

func TestMessagesToChatImagesBecomeDataURLs(t *testing.T) {
	in := `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[
		{"type":"text","text":"what is in this?"},
		{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAA"}}]}]}`
	out, notes, err := MessagesToChat([]byte(in), Options{})
	if err != nil {
		t.Fatalf("MessagesToChat: %v", err)
	}
	if notes != nil {
		t.Fatalf("images map losslessly: %v", notes)
	}
	m := decode(t, out)
	msgs, _ := m["messages"].([]any)
	um, _ := msgs[0].(map[string]any)
	parts, _ := um["content"].([]any)
	if len(parts) != 2 {
		t.Fatalf("want text + image parts: %+v", um["content"])
	}
	part, _ := parts[1].(map[string]any)
	iu, _ := part["image_url"].(map[string]any)
	if iu["url"] != "data:image/png;base64,AAA" {
		t.Fatalf("image not mapped to an OpenAI image_url data URL: %+v", part)
	}
}

func TestRequestDispatch(t *testing.T) {
	body := []byte(`{"model":"m","messages":[]}`)
	// Identity plans return the bytes untouched — no re-marshal, no key reorder.
	got, notes, err := Request(Plan{From: Chat, Dialect: Chat, Path: "/v1/chat/completions"}, body, Options{})
	if err != nil || string(got) != string(body) || notes != nil {
		t.Fatalf("identity must be a no-op: %s %v %v", got, notes, err)
	}
	// A dialectless plan (audio, embeddings) likewise.
	got, notes, err = Request(Plan{Path: "/v1/audio/speech"}, body, Options{})
	if err != nil || string(got) != string(body) || notes != nil {
		t.Fatalf("dialectless plan: %s %v %v", got, notes, err)
	}
	if _, _, err := Request(Plan{From: Responses, Dialect: Chat}, body, Options{}); err == nil {
		t.Fatal("an unimplemented pair must refuse, not pass through")
	}
}
```

In `admin/convert_test.go`, upgrade the plan-only assertion in `TestConvertPreviewNamesDialectAndEndpoint`: replace the `out["request"] == nil && out["code"] == nil` check with the real one — the fixture request carries no `max_tokens`, so the preview must apply the same default the router does, and report it:

```go
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `go test ./convert ./admin -run 'MessagesToChat|RequestDispatch|ConvertPreview' -v`
Expected: FAIL — `undefined: MessagesToChat`; the preview test fails on `request` being absent (Task 5's placeholder `Request` refuses the pair).

- [ ] **Step 3: Write the implementation**

Create `convert/anthropic.go`:

```go
package convert

import (
	"encoding/json"
	"strconv"
	"strings"
)

// The messages→chat spoke. Every field is either mapped, refused, or — with
// Options.DropParams — dropped and reported; nothing is ever dropped quietly. The
// reference for the mapping is LiteLLM's anthropic adapter
// (llms/anthropic/experimental_pass_through/adapters/transformation.py) and the
// inverse shim llama.cpp ships in tools/server/server-chat.cpp.

type aRequest struct {
	Model         string          `json:"model"`
	System        json.RawMessage `json:"system"`
	Messages      []aMsg          `json:"messages"`
	MaxTokens     *int            `json:"max_tokens"`
	StopSequences []string        `json:"stop_sequences"`
	Temperature   *float64        `json:"temperature"`
	TopP          *float64        `json:"top_p"`
	TopK          *int            `json:"top_k"`
	Stream        bool            `json:"stream,omitempty"`
	Tools         []aTool         `json:"tools"`
	ToolChoice    json.RawMessage `json:"tool_choice"`
	OutputFormat  json.RawMessage `json:"output_format"`
	Thinking      json.RawMessage `json:"thinking"`
	Metadata      *aMetadata      `json:"metadata"`
}

type aMetadata struct {
	UserID string `json:"user_id"`
}

type aMsg struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// aBlock is the union of the Anthropic content-block shapes this spoke reads
// (text, image, tool_use, tool_result, plus the fields tools declare).
type aBlock struct {
	Type        string          `json:"type"`
	Text        string          `json:"text"`
	ID          string          `json:"id"`
	Name        string          `json:"name"`
	Input       json.RawMessage `json:"input"`
	ToolUseID   string          `json:"tool_use_id"`
	Content     json.RawMessage `json:"content"`
	Source      *aSource        `json:"source"`
	Description string          `json:"description"`
}

type aSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

type aTool struct {
	Name        string          `json:"name"`
	Type        string          `json:"type"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

func MessagesToChat(body []byte, o Options) ([]byte, []string, error) {
	var in aRequest
	if err := json.Unmarshal(body, &in); err != nil {
		return nil, nil, &Error{Code: CodeField, Field: "body", Msg: "invalid JSON body: " + err.Error()}
	}
	var notes []string
	// drop refuses by default (a field with no representation must stop the
	// request: the client asked for something the backend can not do) and reports
	// a note when the operator opted into silent drops.
	drop := func(field, detail string) error {
		if !o.DropParams {
			return &Error{Code: CodeField, Field: field, Msg: field + " has no Chat Completions " +
				"representation (" + detail + "). Set settings.convert.drop_params to drop it."}
		}
		notes = append(notes, field+"="+detail)
		return nil
	}

	out := map[string]any{"model": in.Model}
	if in.Stream {
		out["stream"] = true
	}
	if in.Temperature != nil {
		out["temperature"] = *in.Temperature
	}
	if in.TopP != nil {
		out["top_p"] = *in.TopP
	}
	if len(in.StopSequences) > 0 {
		out["stop"] = in.StopSequences
	}
	if in.MaxTokens != nil {
		out["max_tokens"] = *in.MaxTokens
	} else if o.MaxTokens > 0 {
		out["max_tokens"] = o.MaxTokens
		notes = append(notes, "max_tokens=injected:"+strconv.Itoa(o.MaxTokens))
	} else if err := drop("max_tokens", "the client sent none and no default is configured"); err != nil {
		return nil, nil, err
	}
	if in.TopK != nil {
		if err := drop("top_k", "chat has no top_k"); err != nil {
			return nil, nil, err
		}
	}
	if len(in.Thinking) > 0 {
		// Anthropic signs its thinking blocks; a signature can not be reproduced
		// from OpenAI reasoning text, so a multi-turn tool loop over a converted
		// request would break mid-conversation. Emulating it is never permitted.
		if err := drop("thinking", "signed thinking can not round-trip"); err != nil {
			return nil, nil, err
		}
	}
	if len(in.OutputFormat) > 0 {
		out["response_format"] = in.OutputFormat
		notes = append(notes, "output_format=response_format")
	}

	msgs := make([]any, 0, len(in.Messages)+1)
	if sys := systemText(in.System); sys != "" {
		msgs = append(msgs, map[string]any{"role": "system", "content": sys})
	}
	for _, m := range in.Messages {
		converted, err := convertMessage(m, drop)
		if err != nil {
			return nil, nil, err
		}
		msgs = append(msgs, converted...)
	}
	out["messages"] = msgs

	if len(in.Tools) > 0 {
		tools := make([]any, 0, len(in.Tools))
		for _, tl := range in.Tools {
			if tl.Type != "" {
				// web_search_20250305, text_editor_20241022, computer_*: the
				// provider runs these, the model does not.
				if err := drop("tools", "hosted tool "+tl.Type+" is provider-specific"); err != nil {
					return nil, nil, err
				}
				continue
			}
			tools = append(tools, map[string]any{"type": "function", "function": map[string]any{
				"name": tl.Name, "description": tl.Description, "parameters": tl.InputSchema}})
		}
		if len(tools) > 0 {
			out["tools"] = tools
		}
	}
	if tc := chatToolChoice(in.ToolChoice); tc != nil {
		out["tool_choice"] = tc
	}
	if in.Metadata != nil && in.Metadata.UserID != "" {
		out["user"] = in.Metadata.UserID
	}

	b, err := json.Marshal(out)
	if err != nil {
		return nil, nil, &Error{Code: CodeField, Field: "body", Msg: "encode: " + err.Error()}
	}
	return b, notes, nil
}

// systemText flattens Anthropic's system field — a string or an array of text
// blocks — into the single system message Chat Completions has.
func systemText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var blocks []aBlock
	if json.Unmarshal(raw, &blocks) != nil {
		return ""
	}
	parts := make([]string, 0, len(blocks))
	for _, b := range blocks {
		if b.Type == "text" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n")
}

// convertMessage maps one Anthropic message to one or more Chat Completions
// messages: an assistant turn carrying tool_use becomes a message with tool_calls,
// and a tool_result becomes a role:"tool" message per call.
func convertMessage(m aMsg, drop func(string, string) error) ([]any, error) {
	var text string
	var blocks []aBlock
	if json.Unmarshal(m.Content, &text) != nil {
		if json.Unmarshal(m.Content, &blocks) != nil {
			return nil, &Error{Code: CodeField, Field: "content", Msg: "content must be a string or blocks"}
		}
	}
	if len(blocks) == 0 {
		return []any{map[string]any{"role": m.Role, "content": text}}, nil
	}

	var out []any
	var calls []any
	var parts []any
	for _, b := range blocks {
		switch b.Type {
		case "text":
			parts = append(parts, map[string]any{"type": "text", "text": b.Text})
		case "image":
			if b.Source == nil || b.Source.Type != "base64" {
				if err := drop("image", "only base64 image sources are proxied"); err != nil {
					return nil, err
				}
				continue
			}
			parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]any{
				"url": "data:" + b.Source.MediaType + ";base64," + b.Source.Data}})
		case "tool_use":
			calls = append(calls, map[string]any{"id": b.ID, "type": "function",
				"function": map[string]any{"name": b.Name, "arguments": b.Input}})
		case "tool_result":
			body, err := toolResultContent(b, drop)
			if err != nil {
				return nil, err
			}
			out = append(out, map[string]any{"role": "tool", "tool_call_id": b.ToolUseID, "content": body})
		default:
			if err := drop(b.Type, "content block has no Chat Completions representation"); err != nil {
				return nil, err
			}
		}
	}
	if len(calls) > 0 {
		am := map[string]any{"role": "assistant", "tool_calls": calls}
		if len(parts) > 0 {
			am["content"] = parts
		}
		return append([]any{am}, out...), nil
	}
	// One text part degrades to the plain string form every client understands.
	if len(parts) == 1 {
		if tx, ok := parts[0].(map[string]any)["text"].(string); ok {
			return append(out, map[string]any{"role": m.Role, "content": tx}), nil
		}
	}
	return append(out, map[string]any{"role": m.Role, "content": parts}), nil
}

func toolResultContent(b aBlock, drop func(string, string) error) (string, error) {
	var s string
	if json.Unmarshal(b.Content, &s) == nil {
		return s, nil
	}
	var blocks []aBlock
	if json.Unmarshal(b.Content, &blocks) != nil {
		return "", nil
	}
	parts := make([]string, 0, len(blocks))
	for _, blk := range blocks {
		if blk.Type == "text" {
			parts = append(parts, blk.Text)
			continue
		}
		if err := drop("tool_result."+blk.Type, "a tool result must be text for Chat Completions"); err != nil {
			return "", err
		}
	}
	return strings.Join(parts, "\n"), nil
}

// chatToolChoice maps Anthropic's tool_choice (a string or {type,name}) onto the
// Chat Completions values: "auto", "required", a named function, or "none".
func chatToolChoice(raw json.RawMessage) any {
	if len(raw) == 0 {
		return nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		if s == "any" {
			return "required"
		}
		return s
	}
	var obj struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if json.Unmarshal(raw, &obj) != nil {
		return nil
	}
	switch obj.Type {
	case "any":
		if obj.Name == "" {
			return "required"
		}
		return map[string]any{"type": "function", "function": map[string]any{"name": obj.Name}}
	case "tool":
		return map[string]any{"type": "function", "function": map[string]any{"name": obj.Name}}
	case "auto", "none":
		return obj.Type
	}
	return nil
}
```

`convert/plan.go` — replace the placeholder body of `Request` with the dispatch:

```go
// Request translates a request body for a plan. Identity plans return the body
// untouched. The notes list what a deliberate drop cost.
func Request(p Plan, body []byte, o Options) ([]byte, []string, error) {
	if p.Dialect == p.From || p.From == "" {
		return body, nil, nil
	}
	if p.From == Messages && p.Dialect == Chat {
		return MessagesToChat(body, o)
	}
	return nil, nil, &Error{Code: CodeNoDialect, Msg: "conversion " + string(p.From) + ">" +
		string(p.Dialect) + " is not supported"}
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./convert ./admin -count=1 && task lint && gofmt -l convert admin`
Expected: PASS, lint `0 issues`, gofmt prints no files (run `task fmt` if it lists the struct tags above).

- [ ] **Step 5: Commit**

```bash
git add convert/ admin/ && git commit -m "feat(convert): messages→chat request transform with refusals and reported drops"
```

---

### Task 7: `convert.Response` — a chat answer in Anthropic shape (non-streamed)

**Files:**
- Create: `convert/response.go`
- Create: `router/convert.go`
- Modify: `proxy/proxy.go` (`StreamTranslator`, `RouterHeaders.Translate`, `translateWriter`, chain wiring)
- Modify: `router/router.go` (arm the translator, merge its notes)
- Test: `convert/response_test.go`, `router/convert_test.go`

**Interfaces:**
- Consumes: `Plan`, `Options`, `Server.Protocols`, `proxy.RouterHeaders`.
- Produces:
  - `func Response(p Plan, body []byte) ([]byte, []string, error)` and `func ChatToMessages(body []byte) ([]byte, []string, error)` (`convert/response.go`)
  - `proxy.StreamTranslator` interface: `WriteFrame(frame []byte) [][]byte`, `FinishFrames() [][]byte`, `TranslateJSON(body []byte) ([]byte, error)`, `Notes() []string`
  - `proxy.RouterHeaders.Translate StreamTranslator`
  - `newResponseTranslator(p convert.Plan) *responseTranslator` (`router/convert.go`), implementing it

- [ ] **Step 1: Write the failing test**

Create `convert/response_test.go`:

```go
package convert

import (
	"encoding/json"
	"testing"
)

func TestChatToMessagesResponse(t *testing.T) {
	in := `{"id":"chatcmpl-9","model":"claude","object":"chat.completion",` +
		`"choices":[{"index":0,"message":{"role":"assistant",` +
		`"content":[{"type":"text","text":"hi there"}],` +
		`"tool_calls":[{"id":"call_1","type":"function","function":{"name":"fetch","arguments":"{\"u\":1}"}}]},` +
		`"finish_reason":"tool_calls"}],` +
		`"usage":{"prompt_tokens":120,"completion_tokens":17,` +
		`"prompt_tokens_details":{"cached_tokens":100}}}`

	out, notes, err := ChatToMessages([]byte(in))
	if err != nil {
		t.Fatalf("ChatToMessages: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("not JSON: %v %s", err, out)
	}
	if m["type"] != "message" || m["role"] != "assistant" || m["model"] != "claude" {
		t.Fatalf("envelope: %+v", m)
	}
	if m["stop_reason"] != "tool_use" {
		t.Fatalf("stop_reason = %v, want tool_use", m["stop_reason"])
	}
	blocks, _ := m["content"].([]any)
	if len(blocks) != 2 {
		t.Fatalf("want a text block and a tool_use block: %+v", m["content"])
	}
	blk, _ := blocks[1].(map[string]any)
	if blk["type"] != "tool_use" || blk["id"] != "call_1" || blk["name"] != "fetch" {
		t.Fatalf("tool_use block: %+v", blk)
	}
	input, _ := blk["input"].(map[string]any)
	if input["u"] == nil {
		t.Fatalf("arguments not parsed into input: %+v", blk)
	}
	usage, _ := m["usage"].(map[string]any)
	// OpenAI's prompt_tokens already contains its cached tokens; Anthropic's
	// input_tokens sits next to cache_read_input_tokens.
	if usage["input_tokens"] != float64(20) || usage["cache_read_input_tokens"] != float64(100) ||
		usage["output_tokens"] != float64(17) {
		t.Fatalf("usage convention not converted: %+v", usage)
	}
	if len(notes) != 0 {
		t.Fatalf("notes: %v", notes)
	}
}

func TestChatToMessagesFinishReasonMap(t *testing.T) {
	for fr, want := range map[string]string{"stop": "end_turn", "length": "max_tokens",
		"content_filter": "refusal", "tool_calls": "tool_use", "": "end_turn"} {
		out, _, err := ChatToMessages([]byte(`{"id":"x","model":"m","choices":[{"finish_reason":"` + fr +
			`","message":{"role":"assistant","content":"t"}}]}`))
		if err != nil {
			t.Fatalf("%s: %v", fr, err)
		}
		var m map[string]any
		if err := json.Unmarshal(out, &m); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if m["stop_reason"] != want {
			t.Fatalf("%q → %v, want %s", fr, m["stop_reason"], want)
		}
	}
}

func TestChatToMessagesReasoningBecomesThinking(t *testing.T) {
	out, _, err := ChatToMessages([]byte(`{"id":"x","model":"m","choices":[{"finish_reason":"stop",` +
		`"message":{"role":"assistant","content":"answer","reasoning_content":"because"}}]}`))
	if err != nil {
		t.Fatalf("ChatToMessages: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("decode: %v", err)
	}
	blocks, _ := m["content"].([]any)
	if len(blocks) != 2 {
		t.Fatalf("want thinking + text: %+v", m["content"])
	}
	first, _ := blocks[0].(map[string]any)
	if first["type"] != "thinking" || first["thinking"] != "because" {
		t.Fatalf("reasoning_content not mapped to a thinking block: %+v", first)
	}
}

func TestResponseDispatch(t *testing.T) {
	body := []byte(`{"choices":[]}`)
	got, _, err := Response(Plan{From: Chat, Dialect: Chat}, body)
	if err != nil || string(got) != string(body) {
		t.Fatalf("identity response must be a no-op: %s %v", got, err)
	}
	if _, _, err := Response(Plan{From: Chat, Dialect: Responses}, body); err == nil {
		t.Fatal("an unimplemented response pair must refuse")
	}
}

// A backend's 4xx/5xx body rides the same JSON path as an answer (HTTP 5xx is not
// an error to this proxy — it is forwarded as-is). An Anthropic SDK can not read
// OpenAI's `{"error":{"message":…}}`, so an untranslated error body reaches the
// client as a parse failure that hides the backend's actual reason.
func TestResponseTranslatesErrorEnvelope(t *testing.T) {
	out, _, err := Response(Plan{From: Messages, Dialect: Chat},
		[]byte(`{"error":{"message":"model not found","type":"invalid_request_error"}}`))
	if err != nil {
		t.Fatalf("Response: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("not JSON: %s", out)
	}
	if m["type"] != "error" {
		t.Fatalf("not an Anthropic error envelope: %s", out)
	}
	e, _ := m["error"].(map[string]any)
	if e["type"] != "invalid_request_error" || e["message"] != "model not found" {
		t.Fatalf("error fields lost: %s", out)
	}
	// A body that is not an error envelope goes to the normal path.
	if _, ok, _ := chatErrorToMessages([]byte(`{"choices":[]}`)); ok {
		t.Fatal("a chat answer must not match the error path")
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./convert -run 'ChatToMessages|ResponseDispatch|ResponseTranslatesErrorEnvelope' -v`
Expected: FAIL — `undefined: ChatToMessages`, `undefined: Response`, `undefined: chatErrorToMessages`.

- [ ] **Step 3: Write the implementation**

Create `convert/response.go`:

```go
package convert

import (
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

// Response translates a non-streamed answer from the dialect the backend answered
// in to the dialect the client asked in. An upstream error envelope is translated
// too (the proxy forwards 4xx/5xx bodies as-is, and an Anthropic SDK can not read
// OpenAI's shape).
func Response(p Plan, body []byte) ([]byte, []string, error) {
	if p.Dialect == p.From || p.From == "" {
		return body, nil, nil
	}
	if p.Dialect == Messages && p.From == Chat {
		if b, ok, err := chatErrorToMessages(body); ok || err != nil {
			return b, nil, err
		}
		return ChatToMessages(body)
	}
	return nil, nil, &Error{Code: CodeNoDialect, Msg: "response conversion " + string(p.From) + ">" +
		string(p.Dialect) + " is not supported"}
}

// chatErrorToMessages rewrites an OpenAI error envelope into Anthropic's. ok is
// false when the body is not one, leaving it to the answer path. `type` is
// copied when present (invalid_request_error / rate_limit_exceeded / …), else
// api_error, the Anthropic SDK's catch-all.
func chatErrorToMessages(body []byte) ([]byte, bool, error) {
	var in struct {
		Error *struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &in) != nil || in.Error == nil {
		return nil, false, nil
	}
	typ := in.Error.Type
	if typ == "" {
		typ = "api_error"
	}
	b, err := json.Marshal(map[string]any{"type": "error",
		"error": map[string]any{"type": typ, "message": in.Error.Message}})
	return b, true, err
}

type cResponse struct {
	ID      string    `json:"id"`
	Model   string    `json:"model"`
	Choices []cChoice `json:"choices"`
	Usage   cUsage    `json:"usage"`
}

type cChoice struct {
	Message struct {
		Content   json.RawMessage `json:"content"`
		Reasoning string          `json:"reasoning_content"`
		ToolCalls []cToolCall     `json:"tool_calls"`
	} `json:"message"`
	FinishReason string `json:"finish_reason"`
}

type cToolCall struct {
	ID       string `json:"id"`
	Function struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

type cUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	PromptDetails    struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

func ChatToMessages(body []byte) ([]byte, []string, error) {
	var c cResponse
	if err := json.Unmarshal(body, &c); err != nil {
		return nil, nil, &Error{Code: CodeField, Field: "body", Msg: "chat response is not JSON: " + err.Error()}
	}
	if len(c.Choices) == 0 {
		return nil, nil, &Error{Code: CodeField, Field: "choices", Msg: "chat response carries no choices"}
	}
	ch := c.Choices[0]
	blocks := make([]any, 0, 4)
	if ch.Message.Reasoning != "" {
		blocks = append(blocks, map[string]any{"type": "thinking", "thinking": ch.Message.Reasoning})
	}
	blocks = append(blocks, contentBlocks(ch.Message.Content)...)
	for i, tc := range ch.Message.ToolCalls {
		var input any
		if len(tc.Function.Arguments) > 0 {
			if json.Unmarshal(tc.Function.Arguments, &input) != nil {
				input = map[string]any{}
			}
		}
		id := tc.ID
		if id == "" {
			id = "toolu_" + shortHash(fmt.Sprintf("%s:%d", c.ID, i))
		}
		blocks = append(blocks, map[string]any{"type": "tool_use", "id": id,
			"name": tc.Function.Name, "input": input})
	}
	out := map[string]any{
		"id":            c.ID,
		"type":          "message",
		"role":          "assistant",
		"model":         c.Model,
		"content":       blocks,
		"stop_reason":   stopReason(ch.FinishReason),
		"stop_sequence": nil,
		"usage": map[string]any{
			// OpenAI's prompt_tokens already contains cached tokens; Anthropic's
			// input_tokens sits next to cache_read_input_tokens. Adding the two
			// conventions is the double count the metrics layer fights.
			"input_tokens":            c.Usage.PromptTokens - c.Usage.PromptDetails.CachedTokens,
			"output_tokens":           c.Usage.CompletionTokens,
			"cache_read_input_tokens": c.Usage.PromptDetails.CachedTokens,
		},
	}
	b, err := json.Marshal(out)
	if err != nil {
		return nil, nil, &Error{Code: CodeField, Field: "body", Msg: "encode: " + err.Error()}
	}
	return b, nil, nil
}

func stopReason(fr string) string {
	switch fr {
	case "length":
		return "max_tokens"
	case "tool_calls", "function_call":
		return "tool_use"
	case "content_filter":
		return "refusal"
	}
	return "end_turn"
}

// contentBlocks accepts a Chat Completions content string, an array of parts, or
// nothing, and returns Anthropic content blocks.
func contentBlocks(raw json.RawMessage) []any {
	if len(raw) == 0 {
		return nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		if s == "" {
			return nil
		}
		return []any{map[string]any{"type": "text", "text": s}}
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) != nil {
		return nil
	}
	out := make([]any, 0, len(parts))
	for _, p := range parts {
		if p.Text != "" {
			out = append(out, map[string]any{"type": "text", "text": p.Text})
		}
	}
	return out
}

func shortHash(s string) string {
	sum := sha1.Sum([]byte(s))
	return hex.EncodeToString(sum[:])[:20]
}
```

`proxy/proxy.go` — declare the interface next to `TraceTap` (`proxy.go:297`), the field in `RouterHeaders` (beside `Trace`), and the writer:

```go
// StreamTranslator converts the response from the backend's dialect into the
// client's: whole SSE frames in, whole SSE frames out, and a whole JSON body for
// a non-streamed answer. Declared here so proxy stays ignorant of convert (no
// import, no cycle), exactly as with TraceTap; a nil translator costs one branch
// on the relay path.
type StreamTranslator interface {
	WriteFrame(frame []byte) [][]byte
	FinishFrames() [][]byte
	TranslateJSON(body []byte) ([]byte, error)
	Notes() []string
}

// MaxTranslatedBody bounds how much of a non-streamed JSON response the
// translator may buffer to rewrite it. A larger body is relayed untranslated (with
// a log line): holding a multi-megabyte answer to restyle a dialect is worse than
// the client seeing the backend's shape.
const MaxTranslatedBody = 8 << 20
```

```go
// translateWriter sits between the SSE relay (which hands it complete
// blank-line-terminated frames) and the heartbeat writers. SSE frames are
// translated frame by frame; a JSON body is buffered to the end and translated in
// one pass, because an Anthropic client can not parse a partial Chat Completions
// object.
type translateWriter struct {
	http.ResponseWriter
	tr      StreamTranslator
	json    bool
	jsonBuf []byte
	pass    bool
}

func newTranslateWriter(w http.ResponseWriter, tr StreamTranslator) *translateWriter {
	return &translateWriter{ResponseWriter: w, tr: tr}
}

// setJSON arms JSON buffering. StreamProxy already learns the response kind from
// the upstream headers and arms the relay the same way.
func (t *translateWriter) setJSON(on bool) { t.json = on && t.tr != nil }

// passthrough stops translating: an upstream-authored error frame is already in
// the client's dialect (sendMidStreamError keys on the client's path).
func (t *translateWriter) passthrough() { t.pass = true }

func (t *translateWriter) Write(data []byte) (int, error) {
	if t.tr == nil || t.pass {
		return t.ResponseWriter.Write(data)
	}
	if !t.json {
		for _, frame := range t.tr.WriteFrame(data) {
			if _, err := t.ResponseWriter.Write(frame); err != nil {
				return 0, err
			}
		}
		return len(data), nil
	}
	if len(t.jsonBuf)+len(data) > MaxTranslatedBody {
		log.Warnf("translateWriter: response exceeds %d bytes, relaying untranslated", MaxTranslatedBody)
		t.pass = true
		return t.ResponseWriter.Write(t.takeBuf())
	}
	t.jsonBuf = append(t.jsonBuf, data...)
	return len(data), nil
}

func (t *translateWriter) takeBuf() []byte {
	b := t.jsonBuf
	t.jsonBuf = nil
	return b
}

// finish delivers what the translator still owes: the frames that close open
// blocks, or the translated JSON body.
func (t *translateWriter) finish() {
	if t.tr == nil || t.pass {
		return
	}
	if !t.json {
		for _, frame := range t.tr.FinishFrames() {
			if _, err := t.ResponseWriter.Write(frame); err != nil {
				log.Warnf("translateWriter: frame write failed: %v", err)
			}
		}
	} else if b := t.takeBuf(); len(b) > 0 {
		translated, err := t.tr.TranslateJSON(b)
		if err != nil {
			log.Warnf("translateWriter: json translation failed, relaying native body: %v", err)
			translated = b
		}
		_, _ = t.ResponseWriter.Write(translated)
	}
	if f, ok := t.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}
```

Wire it into `StreamProxy` (`proxy.go:1188-1190`, `:1315`, `:1392`, `:1412-1413`):

```go
	ws := newSilenceWriter(baseW, WaitSignalIdle, ping)
	defer ws.Stop()
	var translator StreamTranslator
	if rh != nil {
		translator = rh.Translate
	}
	tw := newTranslateWriter(ws, translator)
	clientW := newSSERelay(tw, strip)
```

```go
	clientW.setRaw(!isStream)
	tw.setJSON(!isStream)
```

```go
			tw.passthrough()
			sendMidStreamError(ld, readErr, strings.Contains(req.URL.Path, "/messages"))
```

```go
	mrw.Finish()
	clientW.finish()
	tw.finish()
```

The loop detector's own error frame reaches the client through `mrw → mw → clientW → tw`; it is OpenAI-shaped and carries `choices`, so the Task 8 emitter translates it into Anthropic shapes — the desired direction, and Task 8's tests pin the pass-through rule that keeps an unparseable frame visible.

`router/convert.go` (new file):

```go
package router

import (
	"llm-api-router/convert"
	"llm-api-router/proxy"
)

// responseTranslator adapts convert's response direction to the shape
// proxy.StreamTranslator expects. Here it translates non-streamed JSON bodies;
// Task 8 gives it the stream emitter.
type responseTranslator struct {
	plan  convert.Plan
	notes []string
}

var _ proxy.StreamTranslator = (*responseTranslator)(nil)

func newResponseTranslator(p convert.Plan) *responseTranslator {
	return &responseTranslator{plan: p}
}

func (t *responseTranslator) TranslateJSON(body []byte) ([]byte, error) {
	b, notes, err := convert.Response(t.plan, body)
	if err != nil {
		return nil, err
	}
	t.notes = append(t.notes, notes...)
	return b, nil
}

// The SSE path is Task 8's: a frame is forwarded untouched, which is also the
// right answer for a pair with no stream emitter.
func (t *responseTranslator) WriteFrame(frame []byte) [][]byte { return [][]byte{frame} }

func (t *responseTranslator) FinishFrames() [][]byte { return nil }

func (t *responseTranslator) Notes() []string { return t.notes }
```

`router/router.go` — in the attempt loop, right after the `rh` literal is built (Task 3 added fields there):

```go
		if meta.convert != "" {
			rh.Translate = newResponseTranslator(plan)
			meta.translator = rh.Translate
		}
```

which needs a new field on `reqMeta` (the struct Task 3 extended): `translator proxy.StreamTranslator`, reset to `nil` on every attempt beside `meta.upstreamPath`. Then in `record`, after the `m := domain.RequestMetric{...}` literal:

```go
	if rt, ok := meta.translator.(*responseTranslator); ok {
		m.ConvertNotes = append(m.ConvertNotes, rt.Notes()...)
	}
```

- [ ] **Step 4: Add the end-to-end test, run everything**

Append to `router/convert_test.go`:

```go
// The first usable slice: an Anthropic client, a chat-only backend, a non-streamed
// answer. The client must receive an Anthropic message object.
func TestHandleMessagesServedByChatBackend(t *testing.T) {
	var gotPath, gotBody string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		gotPath, gotBody = r.URL.Path, string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"c1","model":"claude","choices":[{"index":0,` +
			`"message":{"role":"assistant","content":"hi there"},"finish_reason":"stop"}],` +
			`"usage":{"prompt_tokens":12,"completion_tokens":3}}`))
	}))
	defer backend.Close()

	r, store, ms, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: backend.URL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"}, ServerID: "s1",
		TargetModel: "claude", NumRetries: -1, Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", w.Result().StatusCode, w.Body.String())
	}
	if gotPath != "/v1/chat/completions" {
		t.Fatalf("backend path = %q", gotPath)
	}
	if !strings.Contains(gotBody, `"max_tokens":10`) {
		t.Fatalf("anthropic max_tokens not carried over: %s", gotBody)
	}
	body := w.Body.String()
	if !strings.Contains(body, `"type":"message"`) || !strings.Contains(body, `"stop_reason":"end_turn"`) {
		t.Fatalf("client did not get an Anthropic message: %s", body)
	}
	if h := w.Result().Header.Get("X-Router-Convert"); h != "messages>chat" {
		t.Fatalf("X-Router-Convert = %q", h)
	}
	rec := ms.Recent()
	if len(rec) != 1 || rec[0].Convert != "messages>chat" || rec[0].UpstreamEndpoint != "/v1/chat/completions" {
		t.Fatalf("metric not annotated: %+v", rec)
	}
	// Usage is read from the native response: the translator runs after
	// metricsWriter, so extraction never sees translated bytes.
	if rec[0].PromptTokens != 12 || rec[0].CompletionTokens != 3 {
		t.Fatalf("usage lost: %+v", rec[0])
	}
}
```

Run: `go test ./convert ./proxy ./router ./admin -count=1 && task lint && gofmt -l convert proxy router admin`
Expected: PASS, lint `0 issues`, gofmt prints no files.

- [ ] **Step 5: Commit**

```bash
git add convert/ proxy/ router/ && git commit -m "feat(convert): translate non-streamed chat answers for anthropic clients"
```

---

### Task 8: The stream emitter — `chat` chunks → Anthropic SSE

**Files:**
- Create: `convert/stream.go`
- Modify: `router/convert.go` (arm the emitter in `newResponseTranslator`)
- Test: `convert/stream_test.go`

**Interfaces:**
- Consumes: chat SSE chunks — with `[DONE]`, with `reasoning_content`, with `tool_calls` deltas, with `usage` on the final chunk.
- Produces:
  - `type Stream interface{ WriteFrame(frame []byte) [][]byte; FinishFrames() [][]byte; Notes() []string }`
  - `func NewStream(p Plan) (Stream, bool)` — `(nil, false)` for identity and for every unimplemented pair
  - `func NewChatToMessages() Stream`

- [ ] **Step 1: Write the failing tests**

Create `convert/stream_test.go`:

```go
package convert

import (
	"strings"
	"testing"
)

func emit(s Stream, chunks ...string) string {
	var b strings.Builder
	for _, c := range chunks {
		for _, f := range s.WriteFrame([]byte("data: " + c + "\n\n")) {
			b.Write(f)
		}
	}
	for _, f := range s.FinishFrames() {
		b.Write(f)
	}
	return b.String()
}

func TestChatToMessagesStreamText(t *testing.T) {
	out := emit(NewChatToMessages(),
		`{"id":"c1","model":"claude","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}`,
		`{"id":"c1","model":"claude","choices":[{"index":0,"delta":{"content":"lo"},`+
			`"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":2}}`,
		`[DONE]`)

	for _, want := range []string{
		"event: message_start", `"type":"message_start"`,
		"event: content_block_start", `"index":0`, `"type":"text"`,
		`"text":"Hel"`, `"text":"lo"`,
		"event: content_block_stop", "event: message_delta", `"stop_reason":"end_turn"`,
		"event: message_stop",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("stream lacks %s:\n%s", want, out)
		}
	}
	if strings.Contains(out, "[DONE]") {
		t.Fatalf("[DONE] must not reach an Anthropic client:\n%s", out)
	}
	// Every emitted frame is one complete SSE event: the client's parser splits on
	// blank lines, and a half frame glues itself onto the next (opencode #7692).
	for _, f := range strings.Split(strings.TrimRight(out, "\n"), "\n\n") {
		if !strings.HasPrefix(f, "event: ") || !strings.Contains(f, "\ndata: ") {
			t.Fatalf("incomplete frame emitted: %q", f)
		}
	}
	if !strings.Contains(out, `"output_tokens":2`) {
		t.Fatalf("usage must ride message_delta:\n%s", out)
	}
	// A chat backend learns prompt tokens only at the end, and Anthropic reports
	// them in message_start: message_delta must restate them, or a client that
	// shows cost reads a permanent zero.
	if !strings.Contains(out, `"input_tokens":7`) {
		t.Fatalf("input tokens lost:\n%s", out)
	}
}

func TestChatToMessagesStreamToolCallsAndThinking(t *testing.T) {
	out := emit(NewChatToMessages(),
		`{"choices":[{"index":0,"delta":{"reasoning_content":"hm"}}]}`,
		`{"choices":[{"index":0,"delta":{"content":"ok"}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1",`+
			`"function":{"name":"fetch","arguments":"{\"u\":"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,`+
			`"function":{"arguments":"1}"}}]},"finish_reason":"tool_calls"}]}`)

	for _, want := range []string{`"type":"thinking"`, `"thinking_delta"`, `"type":"tool_use"`,
		`"partial_json":"{\"u\":"`, `"partial_json":"1}"`, `"stop_reason":"tool_use"`} {
		if !strings.Contains(out, want) {
			t.Fatalf("lacks %s:\n%s", want, out)
		}
	}
	// A kind change closes the previous block: indices are distinct and ordered,
	// and every block opened is closed — thinking, text and tool_use each open
	// once, and the finish_reason closes the last one.
	if got := strings.Count(out, `"content_block_start"`); got != 3 {
		t.Fatalf("content_block_start x%d, want 3 (thinking, text, tool_use):\n%s", got, out)
	}
	if got := strings.Count(out, "content_block_stop"); got != 3 {
		t.Fatalf("content_block_stop x%d, want 3:\n%s", got, out)
	}
}

func TestChatToMessagesStreamPassesThroughWhatItCannotTranslate(t *testing.T) {
	s := NewChatToMessages()
	// A frame that is not a Chat Completions chunk is not ours to translate: an
	// upstream-authored error frame, or the Anthropic mid-stream error the proxy
		// injects in the client's dialect. Swallowing it would leave the client a
	// truncated message and no error. Comments (heartbeats) and [DONE] carry
	// nothing an Anthropic client needs, so those are dropped.
	frames := s.WriteFrame([]byte("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\"}}\n\n"))
	if len(frames) != 1 || !strings.Contains(string(frames[0]), "event: error") {
		t.Fatalf("error frame swallowed: %q", frames)
	}
	if got := s.WriteFrame([]byte(": keep-alive\n\n")); len(got) != 0 {
		t.Fatalf("heartbeat forwarded to the translator output: %q", got)
	}
	if got := s.WriteFrame([]byte("data: [DONE]\n\n")); len(got) != 0 {
		t.Fatalf("[DONE] forwarded: %q", got)
	}
	out := emit(s, `{"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":"stop"}]}`)
	if strings.Count(out, "content_block_delta") != 1 {
		t.Fatalf("contentless frames produced deltas:\n%s", out)
	}
	// A stream that never carried message_start must not be closed with an
	// Anthropic envelope out of thin air.
	if got := NewChatToMessages().FinishFrames(); got != nil {
		t.Fatalf("unstarted stream produced frames: %q", got)
	}
}

func TestNewStreamIdentityArmsNothing(t *testing.T) {
	if s, ok := NewStream(Plan{From: Chat, Dialect: Chat}); ok || s != nil {
		t.Fatal("an identity plan must arm no translator (hot path pays nothing)")
	}
	if _, ok := NewStream(Plan{From: Responses, Dialect: Chat}); ok {
		t.Fatal("unimplemented pairs must report no translator")
	}
	s, ok := NewStream(Plan{From: Messages, Dialect: Chat, Path: "/v1/chat/completions"})
	if !ok || s == nil {
		t.Fatal("chat answered, messages asked: must translate")
	}
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `go test ./convert -run 'ChatToMessagesStream|NewStream' -v`
Expected: FAIL — `undefined: NewChatToMessages`, `undefined: NewStream`.

- [ ] **Step 3: Write the implementation**

Create `convert/stream.go`:

```go
package convert

import (
	"bytes"
	"encoding/json"
	"strconv"
)

// Stream converts a streamed answer from the backend's dialect to the client's.
// It is stateful per request on purpose: Anthropic opens and closes numbered
// content blocks, and a Chat Completions chunk carries no block index, so the
// emitter allocates them as content kinds appear.
type Stream interface {
	WriteFrame(frame []byte) [][]byte
	FinishFrames() [][]byte
	Notes() []string
}

// NewStream returns the translator for a plan's response direction (the backend
// answered in p.Dialect, the client asked in p.From), and false when none exists:
// identity, a dialectless request, or a pair this build does not implement. The
// caller arms the relay only on true, so an unconverted request pays nothing.
func NewStream(p Plan) (Stream, bool) {
	if p.Dialect == Chat && p.From == Messages {
		return NewChatToMessages(), true
	}
	return nil, false
}

func NewChatToMessages() Stream { return &chatToMessages{open: -1, inputTok: -1} }

type chatToMessages struct {
	id, model   string
	open        int    // index of the open block; -1 = none
	kind        string // the open block's kind: text | thinking | tool_use
	toolID      string
	buf         []byte // held tail of a frame split across reads
	out         [][]byte
	inputTok    int // -1 until a chunk reports usage
	outputToks  int
	started     bool
}

func (c *chatToMessages) Notes() []string { return nil }

// WriteFrame takes one or more complete upstream SSE frames and returns the
// complete frames to write to the client.
func (c *chatToMessages) WriteFrame(frame []byte) [][]byte {
	c.buf = append(c.buf, frame...)
	out := c.out[:0]
	for {
		i := bytes.Index(c.buf, []byte("\n\n"))
		if i < 0 {
			break
		}
		out = append(out, c.handle(c.buf[:i])...)
		c.buf = c.buf[i+2:]
	}
	c.out = out
	return out
}

func (c *chatToMessages) handle(frame []byte) [][]byte {
	data := bytes.TrimSpace(frame)
	if len(data) == 0 {
		return nil
	}
	// SSE comments are heartbeats; Anthropic's are injected below the translator.
	if bytes.HasPrefix(data, []byte(":")) {
		return nil
	}
	data = bytes.TrimSpace(bytes.TrimPrefix(data, []byte("data:")))
	if bytes.Equal(data, []byte("[DONE]")) {
		return nil
	}
	var chunk struct {
		ID      string          `json:"id"`
		Model   string          `json:"model"`
		Choices []cStreamChoice `json:"choices"`
		Usage   *cUsage         `json:"usage"`
	}
	if json.Unmarshal(data, &chunk) != nil {
		// Not a Chat Completions chunk (an upstream error frame, say): pass it
		// through untouched — it is already in the client's dialect.
		return [][]byte{append(append([]byte{}, frame...), '\n', '\n')}
	}
	var frames [][]byte
	if chunk.ID != "" {
		c.id = chunk.ID
	}
	if chunk.Model != "" {
		c.model = chunk.Model
	}
	if u := chunk.Usage; u != nil {
		c.inputTok = u.PromptTokens - u.PromptDetails.CachedTokens
		c.outputToks = u.CompletionTokens
	}
	for _, ch := range chunk.Choices {
		if u := ch.Usage; u != nil {
			c.inputTok = u.PromptTokens - u.PromptDetails.CachedTokens
			c.outputToks = u.CompletionTokens
		}
		if !c.started {
			frames = append(frames, c.start())
		}
		frames = append(frames, c.text(ch.Delta.Reasoning, "thinking")...)
		frames = append(frames, c.text(ch.Delta.Content, "text")...)
		for _, tc := range ch.Delta.ToolCalls {
			frames = append(frames, c.tool(tc)...)
		}
		if ch.FinishReason != "" {
			frames = append(frames, c.finish(ch.FinishReason)...)
		}
	}
	// A chunk that carried nothing (a role-only delta, a bare usage chunk) opened no
	// block: emit nothing rather than an empty Anthropic event.
	return frames
}

type cStreamChoice struct {
	Delta struct {
		Role      string          `json:"role"`
		Content   string          `json:"content"`
		Reasoning string          `json:"reasoning_content"`
		ToolCalls []deltaToolCall `json:"tool_calls"`
	} `json:"delta"`
	FinishReason string  `json:"finish_reason"`
	Usage        *cUsage `json:"usage"`
}

type deltaToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

func (c *chatToMessages) start() []byte {
	c.started = true
	id := c.id
	if id == "" {
		id = "msg_" + shortHash("stream")
	}
	input := c.inputTok
	if input < 0 {
		input = 0
	}
	return frame("message_start", map[string]any{"type": "message_start", "message": map[string]any{
		"id": id, "type": "message", "role": "assistant", "model": c.model,
		"content": []any{},
		"usage":   map[string]any{"input_tokens": input, "cache_read_input_tokens": 0},
	}})
}

func frame(event string, payload any) []byte {
	b, _ := json.Marshal(payload) // values built here always marshal
	return append(append([]byte("event: "+event+"\ndata: "), b...), '\n', '\n')
}

// text appends a delta to the open block of that kind, opening (and closing the
// previous one) when the content kind changed.
func (c *chatToMessages) text(s, kind string) [][]byte {
	if s == "" {
		return nil
	}
	var out [][]byte
	if c.kind != kind {
		out = append(out, c.close())
		c.openBlock(kind)
		out = append(out, frame("content_block_start", map[string]any{"type": "content_block_start",
			"index": c.open, "content_block": blockStart(kind)}))
	}
	field, deltaType := "text", "text_delta"
	if kind == "thinking" {
		field, deltaType = "thinking", "thinking_delta"
	}
	return append(out, frame("content_block_delta", map[string]any{"type": "content_block_delta",
		"index": c.open, "delta": map[string]any{"type": deltaType, field: s}}))
}

func (c *chatToMessages) tool(tc deltaToolCall) [][]byte {
	var out [][]byte
	if c.kind != "tool_use" || (tc.ID != "" && tc.ID != c.toolID) {
		out = append(out, c.close())
		c.openBlock("tool_use")
		if tc.ID != "" {
			c.toolID = tc.ID
		}
		if c.toolID == "" {
			c.toolID = "toolu_" + strconv.Itoa(c.open)
		}
		out = append(out, frame("content_block_start", map[string]any{"type": "content_block_start",
			"index": c.open, "content_block": map[string]any{"type": "tool_use",
				"id": c.toolID, "name": tc.Function.Name, "input": map[string]any{}}}))
	}
	if tc.Function.Arguments != "" {
		out = append(out, frame("content_block_delta", map[string]any{"type": "content_block_delta",
			"index": c.open, "delta": map[string]any{"type": "input_json_delta",
				"partial_json": tc.Function.Arguments}}))
	}
	return out
}

func (c *chatToMessages) openBlock(kind string) { c.kind, c.open = kind, c.open+1 }

func (c *chatToMessages) close() []byte {
	if c.kind == "" {
		return nil
	}
	c.kind = ""
	return frame("content_block_stop", map[string]any{"type": "content_block_stop", "index": c.open})
}

func blockStart(kind string) map[string]any {
	if kind == "thinking" {
		return map[string]any{"type": "thinking", "thinking": ""}
	}
	return map[string]any{"type": "text", "text": ""}
}

func (c *chatToMessages) finish(fr string) [][]byte {
	return [][]byte{
		c.close(),
		frame("message_delta", map[string]any{"type": "message_delta",
			"delta": map[string]any{"stop_reason": stopReason(fr), "stop_sequence": nil},
			"usage": c.deltaUsage()}),
	}
}

// deltaUsage carries the output count, and the input count when a usage chunk
// supplied one. Anthropic's own streams report input tokens in message_start,
// which a chat backend only knows at the end — so message_delta restates it and a
// client that shows cost never reads a permanent zero.
func (c *chatToMessages) deltaUsage() map[string]any {
	u := map[string]any{"output_tokens": c.outputToks}
	if c.inputTok >= 0 {
		u["input_tokens"] = c.inputTok
	}
	return u
}

// FinishFrames closes a stream that ended without a finish_reason (a backend that
// died mid-generation, a client-driven drain) so an Anthropic SDK never sees an
// unterminated message.
func (c *chatToMessages) FinishFrames() [][]byte {
	if !c.started {
		return nil
	}
	return [][]byte{
		c.close(),
		frame("message_delta", map[string]any{"type": "message_delta",
			"delta": map[string]any{"stop_reason": "end_turn", "stop_sequence": nil},
			"usage": map[string]any{"output_tokens": c.outputToks}}),
		frame("message_stop", map[string]any{"type": "message_stop"}),
	}
}
```

`router/convert.go` — the adapter now arms the emitter. Replace its struct, constructor and three SSE methods with:

```go
type responseTranslator struct {
	plan   convert.Plan
	stream convert.Stream
	notes  []string
}

func newResponseTranslator(p convert.Plan) *responseTranslator {
	t := &responseTranslator{plan: p}
	if s, ok := convert.NewStream(p); ok {
		t.stream = s
	}
	return t
}

func (t *responseTranslator) WriteFrame(frame []byte) [][]byte {
	if t.stream == nil {
		return [][]byte{frame}
	}
	return t.stream.WriteFrame(frame)
}

func (t *responseTranslator) FinishFrames() [][]byte {
	if t.stream == nil {
		return nil
	}
	return t.stream.FinishFrames()
}

func (t *responseTranslator) Notes() []string {
	if t.stream == nil {
		return t.notes
	}
	return append(t.notes, t.stream.Notes()...)
}
```

`TranslateJSON` is untouched.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./convert -count=1 -v && task lint && gofmt -l convert`
Expected: PASS, lint `0 issues`, gofmt prints no files (run `task fmt` if it lists the struct tags above).

- [ ] **Step 5: Commit**

```bash
git add convert/ router/ && git commit -m "feat(convert): stateful chat→anthropic SSE emitter"
```

---

### Task 9: Streaming end to end, mid-stream included

**Files:**
- Test: `router/convert_stream_test.go`
- Modify: `proxy/proxy.go` / `router/router.go` only if a test below proves a gap

**Interfaces:**
- Consumes: the emitter (Task 8), `rh.Translate` arming (Task 7), `translateWriter`'s frame path and `passthrough()` (Task 7), `sendMidStreamError`'s client-dialect flag (`proxy.go:1392`, keyed on `req.URL.Path`).
- Produces: the streaming regressions the spec names, pinned.

- [ ] **Step 1: Write the failing tests**

Create `router/convert_stream_test.go`:

```go
package router

import (
	"bufio"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"llm-api-router/domain"
)

// sseEvents splits a stream into blank-line-separated events as a client parses
// them.
func sseEvents(s string) []string {
	var out []string
	sc := bufio.NewScanner(strings.NewReader(s))
	sc.Split(bufio.ScanLines)
	var cur []string
	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			if len(cur) > 0 {
				out = append(out, strings.Join(cur, "\n"))
				cur = nil
			}
			continue
		}
		cur = append(cur, line)
	}
	if len(cur) > 0 {
		out = append(out, strings.Join(cur, "\n"))
	}
	return out
}

func TestHandleMessagesServedByChatBackendStreaming(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if p := r.URL.Path; p != "/v1/chat/completions" {
			t.Errorf("backend path = %q", p)
		}
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"include_usage":true`) {
			t.Errorf("a chat upstream must get the usage injection: %s", body)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		// Deliberately split one frame across two writes: the relay re-frames per
		// read, and the emitter must never forward a partial frame.
		_, _ = w.Write([]byte("data: {\"id\":\"c1\",\"model\":\"claude\",\"choices\":" +
			"[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"}}]}\n"))
		f.Flush()
		_, _ = w.Write([]byte("\ndata: {\"id\":\"c1\",\"model\":\"claude\",\"choices\":" +
			"[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]," +
			"\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n"))
		f.Flush()
	}))
	defer backend.Close()

	r, store, ms, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: backend.URL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"}, ServerID: "s1",
		TargetModel: "claude", NumRetries: -1, Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","max_tokens":10,"stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	out := w.Body.String()
	if !strings.Contains(out, "event: message_start") || !strings.Contains(out, "event: message_stop") {
		t.Fatalf("no Anthropic envelope:\n%s", out)
	}
	if strings.Contains(out, "data: [DONE]") {
		t.Fatalf("OpenAI sentinel leaked:\n%s", out)
	}
	if !strings.Contains(out, `"text":"Hel"`) || !strings.Contains(out, `"text":"lo"`) {
		t.Fatalf("text deltas lost:\n%s", out)
	}
	if !strings.Contains(out, `"output_tokens":2`) {
		t.Fatalf("usage lost:\n%s", out)
	}
	// One delivery must be exactly one event: clients that treat a delivery as an
	// event glue the tail of one frame onto the next.
	for _, ev := range sseEvents(out) {
		if !strings.HasPrefix(ev, "event: ") {
			t.Fatalf("delivered chunk is not one event: %q", ev)
		}
	}
	rec := ms.Recent()
	if len(rec) != 1 || rec[0].Convert != "messages>chat" {
		t.Fatalf("metric: %+v", rec)
	}
	// metricsWriter sits above the relay, so it read the native usage object.
	if rec[0].PromptTokens != 5 || rec[0].CompletionTokens != 2 {
		t.Fatalf("usage not read from native bytes: %+v", rec[0])
	}
}

// A converted stream that dies mid-generation must tell the client in its own
// dialect: an Anthropic SDK ignores a bare data: frame and would report no error.
func TestHandleMessagesServedByChatBackendMidStreamError(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		_, _ = w.Write([]byte("data: {\"id\":\"c1\",\"model\":\"claude\",\"choices\":" +
			"[{\"index\":0,\"delta\":{\"content\":\"par\"}}]}\n\n"))
		f.Flush()
	}))
	defer backend.Close()

	r, store, _, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: backend.URL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"}, ServerID: "s1",
		TargetModel: "claude", NumRetries: -1, Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","max_tokens":10,"stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	out := w.Body.String()
	if !strings.Contains(out, "event: error") {
		t.Fatalf("mid-stream failure must be reported in the client's dialect:\n%s", out)
	}
	if strings.Contains(out, "event: message_stop") {
		t.Fatalf("after an error frame the emitter must not append a normal close:\n%s", out)
	}
	if !strings.Contains(out, `"text":"par"`) {
		t.Fatalf("the content produced before the failure must survive:\n%s", out)
	}
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `go test ./router -run 'ServedByChatBackendStreaming|MidStreamError' -v`
Expected: `TestHandleMessagesServedByChatBackendStreaming` FAILS with `no Anthropic envelope` — the emitter is not armed on the stream path yet, so `rh.Translate` forwards the upstream frames untouched and the client sees raw OpenAI chunks plus `[DONE]`. `TestHandleMessagesServedByChatBackendMidStreamError` PASSES — Task 7 calls `tw.passthrough()` before `sendMidStreamError`, so the proxy's Anthropic `event: error` frame already reaches the client intact; it is the pin for that, not a bug report.

- [ ] **Step 3: Write the implementation**

The emitter is armed by Task 7's `newResponseTranslator` plus Task 8's `NewStream`, and `proxy.StreamProxy` reaches it through `rh.Translate` for both SSE frames and the JSON body, so no new production code is expected. If the second test fails on the `passthrough()` call, read `StreamProxy`'s mid-stream branch (`proxy.go:1385-1400`) and confirm `tw.passthrough()` runs **before** `sendMidStreamError(ld, ...)` — the error frame enters the chain at `ld` and flows down through `tw`.

- [ ] **Step 4: Run everything**

Run: `go test ./convert ./proxy ./router ./admin ./config -count=1 && task lint && gofmt -l .`
Expected: PASS, lint `0 issues`, gofmt prints no files. Then the full suite once: `task test` (slow — dial timeouts dominate its runtime; compare any failure against the pre-plan baseline with `git stash` before calling it yours, and report it by name either way).

- [ ] **Step 5: Commit**

```bash
git add router/ proxy/ && git commit -m "test(router): convert streamed anthropic clients on chat backends, mid-stream included"
```

---

### Task 10: Fallbacks, jev pools, and refusals that stay clean

**Files:**
- Modify: `router/router.go`, `router/jev.go` (only if a test proves the plumbing is missing)
- Test: `router/convert_test.go`, `router/jev_test.go`

**Interfaces:**
- Consumes: `FallbackEntry.Convert`, `JevCandidate.Convert`, `serverAttempt.convert` (Task 3), `config.NewHealthTracker`, `HealthTracker.GetStatus`.
- Produces: the spec's two named regressions, and proof that a refusal leaves health/rate-limit bookkeeping untouched.

- [ ] **Step 1: Write the failing tests**

Append to `router/convert_test.go` (add `path/filepath`, `time`, `llm-api-router/config`, `llm-api-router/metrics`, `llm-api-router/trace` to its imports):

```go
// The hole the feature exists for: a rule whose primary cannot serve the client's
// dialect and whose fallback can. Before the converter the primary's 404 was
// forwarded as-is (4xx is not an error), the fallback was never tried, and the
// rule looked healthy.
func TestHandleProtocolMismatchConvertsFallback(t *testing.T) {
	var primaryHits int
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		primaryHits++
		http.Error(w, "nope", http.StatusNotFound)
	}))
	defer primary.Close()
	var fbPath string
	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fbPath = r.URL.Path
		_, _ = w.Write([]byte(`{"id":"c","model":"claude","choices":[{"index":0,` +
			`"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer fallback.Close()

	r, store, ms, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "primary", Name: "primary", URL: primary.URL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/responses"}}}); err != nil {
		t.Fatalf("AddServer primary: %v", err)
	}
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: fallback.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI}}); err != nil {
		t.Fatalf("AddServer s1: %v", err)
	}
	// The fallback declares no protocols, so only its own `convert` can plan it
	// onto chat — which is what pins FallbackEntry.Convert being read.
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"}, ServerID: "primary",
		TargetModel: "claude", NumRetries: -1, Enabled: true,
		Fallbacks: []domain.FallbackEntry{{ServerID: "s1", Convert: "messages>chat"}}}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", w.Result().StatusCode, w.Body.String())
	}
	if primaryHits != 0 {
		t.Fatalf("the refusing primary got %d requests, want 0", primaryHits)
	}
	if fbPath != "/v1/chat/completions" {
		t.Fatalf("fallback path = %q", fbPath)
	}
	if h := w.Result().Header.Get("X-Router-Convert"); h != "messages>chat" {
		t.Fatalf("X-Router-Convert = %q", h)
	}
	rec := ms.Recent()
	if len(rec) != 1 || rec[0].ServerID != "s1" || rec[0].Convert != "messages>chat" {
		t.Fatalf("the metric must describe the serving attempt: %+v", rec)
	}
}

// A refusal is not a transport failure: no health mark (the server was never
// contacted), so a rule with one bad protocol declaration does not take a healthy
// backend out of rotation.
func TestHandleRefusalLeavesHealthClean(t *testing.T) {
	var hits int
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { hits++ }))
	defer backend.Close()

	store, err := config.NewStore(filepath.Join(t.TempDir(), "config.json"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	health := config.NewHealthTracker(store, time.Hour)
	r := New(store, metrics.New(100), health, nil, nil, trace.New(100, domain.TraceSettings{}))
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: backend.URL,
		APITypes:  []domain.APIType{domain.APITypeOpenAI},
		Protocols: []domain.ServerProtocol{{Path: "/v1/chat/completions"}}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"r"}, ServerID: "s1",
		TargetModel: "r", NumRetries: -1, Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	health.Start() // marks the existing servers healthy; interval keeps it idle
	defer health.Stop()
	if !health.IsHealthy("s1") {
		t.Fatalf("fixture: s1 starts unhealthy")
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"r","input":"hi"}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusBadRequest || hits != 0 {
		t.Fatalf("status %d, hits %d: %s", w.Result().StatusCode, hits, w.Body.String())
	}
	if !health.IsHealthy("s1") {
		t.Fatal("a refusal must not mark a server unhealthy — it was never contacted")
	}
}
```

Append to `router/jev_test.go` (its imports already cover `net/http`, `net/http/httptest`, `strings`, `testing`, `domain`; add `io` if the file lacks it):

```go
// A jev pool may mix dialects: the candidate's own `convert` plans it, so one pool
// can span an Anthropic-only and an OpenAI-only backend for one Anthropic client.
// The candidate declares no protocols and carries the explicit edge, which is what
// pins JevCandidate.Convert being read by `jevAttempts`.
func TestHandleJevCandidateConverts(t *testing.T) {
	decision := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"answers":{"model":{"type":"choice","choice":"big","confidence":0.9}},` +
			`"usage":{"input_tokens":5,"output_tokens":1}}`))
	}))
	defer decision.Close()
	var gotPath string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		_, _ = w.Write([]byte(`{"id":"c","model":"big","choices":[{"index":0,` +
			`"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer backend.Close()

	r, store, _, _ := newTestRouter(t)
	servers := []*domain.Server{
		{ID: "d", Name: "d", URL: decision.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
		{ID: "b", Name: "b", URL: backend.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
	}
	for _, srv := range servers {
		if err := store.AddServer(srv); err != nil {
			t.Fatalf("AddServer %s: %v", srv.ID, err)
		}
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"}, Enabled: true,
		Router: domain.RouterJev, Jev: &domain.JevRouter{ServerID: "d", DefaultTier: 0,
			Candidates: []domain.JevCandidate{{ServerID: "b", TargetModel: "big", Tier: 0,
				Convert: "messages>chat"}}}}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", w.Result().StatusCode, w.Body.String())
	}
	if gotPath != "/v1/chat/completions" {
		t.Fatalf("jev candidate path = %q, want the converted endpoint", gotPath)
	}
}

// A refusal is per attempt, so a pool may contain a candidate that cannot serve
// the client's dialect at all: it is skipped without being contacted, and the next
// tier answers. (With `convert: off` the same candidate is passed through and
// rides its own 404 to the client — off is today's behavior, not a refusal.)
func TestHandleJevCandidateRefusalFallsToNextTier(t *testing.T) {
	decision := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"answers":{"model":{"type":"choice","choice":"weak","confidence":0.9}},` +
			`"usage":{"input_tokens":5,"output_tokens":1}}`))
	}))
	defer decision.Close()
	var weakHits, strongHits int
	weak := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		weakHits++
		http.Error(w, "nope", http.StatusNotFound)
	}))
	defer weak.Close()
	strong := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		strongHits++
		_, _ = w.Write([]byte(`{"type":"message","role":"assistant","content":[],"model":"strong"}`))
	}))
	defer strong.Close()

	r, store, _, _ := newTestRouter(t)
	for _, srv := range []*domain.Server{
		{ID: "d", Name: "d", URL: decision.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
		{ID: "weak", Name: "weak", URL: weak.URL,
			APITypes:  []domain.APIType{domain.APITypeOpenAI},
			Protocols: []domain.ServerProtocol{{Path: "/v1/responses"}}},
		{ID: "strong", Name: "strong", URL: strong.URL,
			APITypes: []domain.APIType{domain.APITypeAnthropic}},
	} {
		if err := store.AddServer(srv); err != nil {
			t.Fatalf("AddServer %s: %v", srv.ID, err)
		}
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"claude"}, Enabled: true,
		Router: domain.RouterJev, Jev: &domain.JevRouter{ServerID: "d", DefaultTier: 0,
			Candidates: []domain.JevCandidate{
				{ServerID: "weak", TargetModel: "weak", Tier: 0},
				{ServerID: "strong", TargetModel: "strong", Tier: 1},
			}}}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", w.Result().StatusCode, w.Body.String())
	}
	if weakHits != 0 {
		t.Fatalf("the refusing candidate got %d requests, want 0", weakHits)
	}
	if strongHits != 1 {
		t.Fatalf("the next tier must serve the request, strongHits=%d", strongHits)
	}
}
```

Both jev tests drive the candidate's own `Convert` through `jevAttempts` (Task 3's third `serverAttempt` construction site).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `go test ./router -run 'ConvertsFallback|RefusalLeavesHealthClean|JevCandidate' -v`
Expected: FAIL — `TestHandleProtocolMismatchConvertsFallback` and `TestHandleJevCandidateConverts` fail on the conversion assertions (`fallback path = "/v1/messages"`, `jev candidate path = "/v1/messages"`): the per-attempt `convert` value is not read from the fallback entry / jev candidate yet, so nothing plans them onto chat. `TestHandleRefusalLeavesHealthClean` and `TestHandleJevCandidateRefusalFallsToNextTier` PASS — Task 3's refusal branch already skips an unreachable attempt and touches no tracker; those two are the pins.

- [ ] **Step 3: Write the implementation**

Read the refusal branch Task 3 added and confirm it touches none of `r.health`, `r.rateLimit`, `r.quota` (it appends to `attemptErrors` and `break`s/`continue`s before the quota reservation at `router.go:306`). Then confirm the three `serverAttempt` construction sites set the field: `router.go:200` → `rule.Convert`, `router.go:207` → the fallback entry's `Convert`, `router/jev.go:295` → the candidate's `Convert`. Add whichever of those three is missing; the tests are the rest of this task's deliverable.

- [ ] **Step 4: Run everything**

Run: `go test ./router ./convert ./proxy ./admin ./config -count=1 && task lint`
Expected: PASS, lint `0 issues`.

- [ ] **Step 5: Commit**

```bash
git add router/ && git commit -m "test(router): cross-dialect fallbacks, jev candidates, clean refusals"
```

---

### Task 11: GUI pill and the notes tooltip

**Files:**
- Modify: `admin/static/index.html`
- Test: `admin/convert_test.go`

**Interfaces:**
- Consumes: `RequestMetric.Convert` / `.UpstreamEndpoint` / `.ConvertNotes` (serialized by Task 3).
- Produces: a `messages→chat` tag on a Recent Requests row whose tooltip carries the backend endpoint and the dropped-field notes.

- [ ] **Step 1: Write the failing test**

Append to `admin/convert_test.go` (add `import "os"` to its import block):

```go
// The GUI is embedded HTML: the only cheap assertion is that the fields the tag
// renders are read. A dropped field shows up as a silently missing tag.
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./admin -run GUIShowsConversion -v`
Expected: FAIL — `gui never reads "r.convert"`.

- [ ] **Step 3: Write the implementation**

In `admin/static/index.html`, in the Recent Requests row builder (the function rendering `audioNote`, `jevCell` and `apiCell`, around `index.html:1409-1452`), add beside the `klass` tag:

```js
    // Protocol conversion: what the backend was actually asked for, and what the
    // translation had to drop along the way.
    let convertNote = '';
    if (r.convert) {
      const parts = ['backend ' + (r.upstream_endpoint || '')];
      (r.convert_notes || []).forEach(n => parts.push(n));
      convertNote = `<br><span class="tag" style="background:#3a2a1a;color:#ffa657" title="${escAttr(parts.join(' · '))}">${esc(r.convert)}</span>`;
    }
```

and append `${convertNote}` to the template of the cell that renders `apiCell` (the API Type / endpoint column), so a converted request reads `openai`, `/v1/messages`, `messages→chat`.

- [ ] **Step 4: Run the test and smoke-check the binary**

Run: `go test ./admin -count=1 && task build`
Expected: PASS; the binary builds with the embedded GUI.

Smoke-check by hand and report the result: with a server entry declaring only `/v1/chat/completions` and a rule pointing at it, run

```bash
curl -s -X POST localhost:8080/v1/messages -H 'Content-Type: application/json' \
  -d '{"model":"<the incoming model>","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}' | head -c 200
curl -s localhost:8080/admin/api/metrics/recent | head -c 400
```

Expected: an Anthropic message object (`"type":"message"`, `"stop_reason"`), and a recent-request entry carrying `"convert":"messages>chat"` with `"upstream_endpoint":"/v1/chat/completions"`.

- [ ] **Step 5: Commit**

```bash
git add admin/ && git commit -m "feat(admin): conversion tag with dropped-field notes"
```

---

### Task 12: Docs

**Files:**
- Modify: `CLAUDE.md`
- Modify: `AGENTS.md`
- Test: none (prose)

- [ ] **Step 1: Add the CLAUDE.md entry**

In **Critical Behavior Notes**, after the audio-endpoints bullet, add:

```markdown
- **Protocol conversion (`convert`, opt-in per server)**: a rule can route a request to a backend that does not speak the client's dialect. Capability is declared in `Server.Protocols` (`[{path, state}]`); **an undeclared server stays byte-for-byte passthrough, exactly as before the feature** — `api_types` keeps its meaning as the per-protocol URL selector and never triggers conversion. The hub is `chat`: the implemented spoke is `messages → chat` (an Anthropic client on an OpenAI-only backend, JSON and SSE); `responses`, `/v1/completions` and every other family are refused with a 400 naming the field, never guessed. The decision is **per attempt** (a fallback server and a jev candidate carry their own `convert`, and the request header `X-Router-Convert: off` outranks config), so the plan, `X-Router-Convert`, `RequestMetric.UpstreamEndpoint/Convert/ConvertNotes` and the translator all describe the attempt that served. The translator sits at the bottom of the writer chain — after `metricsWriter`, `modelRewriteWriter` and `sseRelay`'s frame buffer, above the heartbeat writers — so usage/native-timing extraction, model rewriting and the trace keep seeing the backend's native bytes, the emitter never sees a partial frame or the stripped usage artifact, and heartbeats stay in the client's dialect. A frame the emitter cannot parse as a chat chunk (an upstream error object) is passed through untouched, and the proxy's own mid-stream error frame bypasses the translator entirely — swallowing either would leave a client with a truncated message and no error. `settings.convert.max_tokens_default` (4096) supplies the output budget Anthropic requires and OpenAI clients omit; `drop_params` turns the default refusal into a reported drop; `thinking` is refused even under `drop_params` (signed thinking blocks can not round-trip). `POST /admin/api/convert/preview` dry-runs the transform for one hypothetical request, at zero cost to the backend. A backend 4xx/5xx body on a converted attempt is translated too (`{"error":{…}}` → `{"type":"error","error":{…}}`), because an SDK that cannot read the error reports a parse failure and loses the reason. Covered by `TestDecide*`, `TestMessagesToChat*`, `TestChatToMessages*`, `TestRequestDispatch`, `TestResponseDispatch`, `TestResponseTranslatesErrorEnvelope`, `TestNewStream*`, `TestHandleMessagesServedByChatBackend*`, `TestHandleProtocolMismatch*`, `TestHandleConvertOff`, `TestHandleJevCandidate*`, `TestConvertPreview*`, `TestServerProtocolsAndRuleConvertSurviveClone`, and in `sdktest/` by `TestAnthropic*ServedByChatBackend`.
```

In the **Packages** table add:

```markdown
| `convert` | `convert/dialect.go`, `convert/plan.go`, `convert/errors.go`, `convert/anthropic.go`, `convert/response.go`, `convert/stream.go` | Protocol dialects, the per-attempt conversion plan, request/response translation, the stateful chat→anthropic stream emitter |
```

In **Key URLs** add:

```markdown
| `/admin/api/convert/preview` | POST | Dry-run a conversion: the plan, the translated body, and what it cost (never contacts a backend) |
```

- [ ] **Step 2: Mirror it in AGENTS.md**

`AGENTS.md` is the condensed duplicate: add the package row, the URL row, and a four-line digest of the bullet (declared capability, undeclared = passthrough; hub is chat, refusal names the field; per-attempt decision and per-attempt observability; translator below the metrics writer, above the heartbeats).

- [ ] **Step 3: Verify the two files agree**

Read both new sections side by side and confirm they say the same thing (the length differs, the claims must not).

`CLAUDE.md` and `AGENTS.md` are gitignored in this repo (`git ls-files` lists neither) — they are local operator docs, not versioned files. There is therefore **nothing to commit** for this task; the verification above is the deliverable. Do not `git add -f` them.

---

### Task 13: SDK wire compliance — the official Anthropic SDK against a chat-only backend

**Files:**
- Modify: `sdktest/harness_test.go` (one helper, after `addRule`)
- Modify: `sdktest/anthropic_test.go` (append two tests + the chat payloads)

**Interfaces:**
- Consumes: the whole converter chain (Tasks 3–9) behind the real router; the `stack` harness (`newStack`, `addRule`, `lastRequest`, `lastMetric`, `accumulateStream`, `jsonBackend`, `sseBackend`); `config.Store.GetServer`/`UpdateServer` to declare the fake backend's protocols.
- Produces: proof that the emitter's frames parse in `anthropic-sdk-go` — `Message.Accumulate` over `Messages.NewStreaming`, with `stream.Err()` as the gate. This is the layer the spec calls the main one: our own Go tests assert strings, this asserts the wire format a real client reads.

`sdktest` is a separate module that builds only where the gitignored `.info/apis/` SDK clones exist — local-only, never CI. Run it from `sdktest/`.

- [ ] **Step 1: Write the failing tests**

Append to `sdktest/anthropic_test.go`:

```go
// A chat-only backend, answering in Chat Completions shape.
const chatCompletionJSON = `{"id":"chatcmpl-1","model":"` + targetModel + `",` +
	`"object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant",` +
	`"content":"Bonjour from fake backend"},"finish_reason":"stop"}],` +
	`"usage":{"prompt_tokens":10,"completion_tokens":5}}`

var chatStreamFrames = []string{
	"data: {\"id\":\"chatcmpl-1\",\"model\":\"" + targetModel +
		"\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0," +
		"\"delta\":{\"role\":\"assistant\",\"content\":\"Bon\"}}]}\n\n",
	"data: {\"id\":\"chatcmpl-1\",\"model\":\"" + targetModel +
		"\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0," +
		"\"delta\":{\"content\":\"jour\"},\"finish_reason\":\"stop\"}]," +
		"\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n",
	"data: [DONE]\n\n",
}

// The phase-2 promise, checked by the official SDK: an Anthropic client, a
// chat-only backend, and the SDK's own JSON decoding accepts the translated
// answer — including the model name rewritten back to the client's.
func TestAnthropicMessagesJSONServedByChatBackend(t *testing.T) {
	st := newStack(t, jsonBackend(http.StatusOK, chatCompletionJSON))
	st.addRule([]string{anthropicModel}, targetModel)
	st.declareProtocols("/v1/chat/completions")

	resp, err := st.anthropic.Messages.New(context.Background(), anthropicParams())
	if err != nil {
		t.Fatalf("Messages.New: %v", err)
	}
	req, body := st.lastRequest()
	if req == nil {
		t.Fatal("fake backend saw no request")
	}
	if req.URL.Path != "/v1/chat/completions" {
		t.Errorf("upstream path = %q, want /v1/chat/completions", req.URL.Path)
	}
	if !strings.Contains(string(body), `"max_tokens":100`) {
		t.Errorf("client max_tokens not carried into the chat request: %s", body)
	}
	if resp.Model != anthropicModel {
		t.Errorf("Model = %q, want %q (response rewrite)", resp.Model, anthropicModel)
	}
	if len(resp.Content) != 1 {
		t.Fatalf("content blocks = %d, want 1: %+v", len(resp.Content), resp.Content)
	}
	if got := resp.Content[0].AsText().Text; got != "Bonjour from fake backend" {
		t.Errorf("content = %q", got)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("stop_reason = %q, want end_turn", resp.StopReason)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 5 {
		t.Errorf("usage = in %d out %d, want 10/5", resp.Usage.InputTokens, resp.Usage.OutputTokens)
	}

	m := st.lastMetric()
	if m.Convert != "messages>chat" || m.UpstreamEndpoint != "/v1/chat/completions" {
		t.Errorf("metric convert = %q upstream = %q, want messages>chat / /v1/chat/completions",
			m.Convert, m.UpstreamEndpoint)
	}
	// Usage is read from the native bytes, above the translator.
	if m.PromptTokens != 10 || m.CompletionTokens != 5 {
		t.Errorf("metric tokens = %d/%d, want 10/5", m.PromptTokens, m.CompletionTokens)
	}
}

// The stream half: every frame the emitter authored must survive
// `Message.Accumulate`, and `stream.Err()` is the gate that says the SDK parsed
// them — a frame with a wrong event name or a missing field fails there.
func TestAnthropicStreamServedByChatBackend(t *testing.T) {
	st := newStack(t, sseBackend(chatStreamFrames...))
	st.addRule([]string{anthropicModel}, targetModel)
	st.declareProtocols("/v1/chat/completions")

	stream := st.anthropic.Messages.NewStreaming(context.Background(), anthropicParams())
	msg := accumulateStream(t, stream)

	if msg.Model != anthropicModel {
		t.Errorf("streamed Model = %q, want %q", msg.Model, anthropicModel)
	}
	if msg.ID != "chatcmpl-1" {
		t.Errorf("streamed ID = %q, want chatcmpl-1", msg.ID)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("content blocks = %d, want 1: %+v", len(msg.Content), msg.Content)
	}
	if got := msg.Content[0].AsText().Text; got != "Bonjour" {
		t.Errorf("streamed content = %q, want Bonjour", got)
	}
	if msg.StopReason != "end_turn" {
		t.Errorf("stop_reason = %q, want end_turn", msg.StopReason)
	}
	if msg.Usage.OutputTokens != 5 {
		t.Errorf("output_tokens = %d, want 5 (from message_delta)", msg.Usage.OutputTokens)
	}
	if msg.Usage.InputTokens != 10 {
		t.Errorf("input_tokens = %d, want 10 (restated in message_delta)", msg.Usage.InputTokens)
	}
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd sdktest && go test ./... -run 'ServedByChatBackend' -v`
Expected: FAIL — `st.declareProtocols undefined` (the helper lands in Step 3).

- [ ] **Step 3: Write the harness helper**

Append to `sdktest/harness_test.go`, after `addRule`:

```go
// declareProtocols declares the endpoints the fake backend serves, which is what
// decides whether the router converts the request (see convert.Decide). The
// default stack declares nothing, so every existing test stays a passthrough.
func (st *stack) declareProtocols(paths ...string) {
	st.t.Helper()
	srv, ok := st.store.GetServer("srv1")
	if !ok {
		st.t.Fatalf("server srv1 not found")
	}
	for _, p := range paths {
		srv.Protocols = append(srv.Protocols, domain.ServerProtocol{Path: p})
	}
	if err := st.store.UpdateServer("srv1", srv); err != nil {
		st.t.Fatalf("UpdateServer: %v", err)
	}
}
```

- [ ] **Step 4: Run the whole module**

Run: `cd sdktest && go test ./...`
Expected: PASS — the two new tests plus every pre-existing SDK test (the declared-protocol stack is opt-in, so passthrough behavior must be untouched).

- [ ] **Step 5: Commit**

```bash
git add sdktest/ && git commit -m "test(sdktest): anthropic SDK served by a chat-only backend, json and stream"
```
