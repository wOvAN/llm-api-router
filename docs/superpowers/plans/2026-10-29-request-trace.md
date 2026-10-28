# Request Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, memory-only full-text trace of proxied requests/responses (bodies, raw SSE frames, stream digest) plus an always-on request classification, switchable live from the admin API and GUI.

**Architecture:** A new `trace` package owns the ring buffer, the live capture window, request redaction, classification and the response digest. The router decides once per request whether the request is traced and builds a `*trace.Capture`; the proxy's existing writer chain (`metricsWriter`, which already sees every response byte) taps chunks into it, so no second `RoundTripper` and no re-read of the response exists. Aggregated per-class summaries reuse `metrics.Store`'s existing summary-key mechanism.

**Tech Stack:** Go 1.27 stdlib only (no new dependencies), `net/http/httptest` for tests, embedded admin GUI (`admin/static/index.html`, vanilla JS).

**Spec:** `docs/superpowers/specs/2026-10-29-request-trace-design.md`

## Global Constraints

- One external dependency allowed in the whole module (`golang.org/x/net/proxy`); everything new is stdlib (Task 1-10 add zero imports outside stdlib + this module's own packages).
- Trace is memory-only: nothing is written to disk, nothing is persisted, a restart never leaves prompt capture switched on. The live window state lives in `trace.Store` and is **never** written to `config.json` (only the startup defaults in `Settings.Trace` are).
- Caps: 512 KiB per request body, 512 KiB per response body, ring of 100 entries (`trace.New(100, …)`), truncation is prefix + `…[truncated N bytes]` + a `truncated` flag.
- Redaction replaces binaries only (`<image 1.2MB sha256:ab12…>` / `<audio …>`); prompt/response text is never cut. `Authorization` is never stored.
- `config.json` is user-managed and gitignored — never write it; `example.config.json` is the committed reference (Task 7 edits that one only).
- `CLAUDE.md` and `AGENTS.md` are duplicates of each other — both are edited in the same task (Task 12).
- Redaction/classification/digest are best-effort: no code path on this route may panic, log-noise is the only failure mode.
- Lint gate: `task lint` must report `0 issues`; every returned error is handled (`errcheck`), including `hash.Hash.Write` (`_, _ = `).

## Review Focus

The design's seams that no single task's happy-path test covers, most likely to bite first:

- **A traced request whose backend died with an SSE-bodied 5xx** (the peeked body in `StreamProxy` never reaches the writer chain): the trace entry must still carry what the backend said, not an empty `response` — the router falls back to `ProxyMetrics.ErrorBody`.
- **A traced binary response** (TTS WAV / raw PCM): the entry must carry the `<audio …>` stub plus the byte count, and the ~4 MB of audio must never be buffered.
- **A window that expires mid-generation**: a request that started inside the window is traced to the end; a request that starts after expiry is not traced at all.
- **A traced multipart audio upload**: the audio bytes must not reach the ring (redaction runs before the capture stores anything), while the `model` field and boundaries stay readable.
- **A retried / fell-back request**: the capture is per-request, not per-attempt, so exactly one entry exists, carrying the response of the attempt that answered plus `retry:<n>` / `fallback` features.

---

### Task 1: `trace` package — entry types, ring buffer, live window

**Files:**
- Create: `trace/store.go`
- Create: `trace/entry.go`
- Test: `trace/store_test.go`

**Interfaces:**
- Consumes: `domain.TraceSettings` (defined in Task 7 — for this task, declare it in `domain/config.go` exactly as Task 7 specifies; Task 7 only adds the `Settings.Trace` field, the example config and the config test).
- Produces:
  - `type Entry struct{ ID, Timestamp, Method, Path, Model, TargetModel, ServerID, Class string; Features []string; StatusCode int; LatencyMs, TTFBMs int64; JevChoice, Request, Response string; Digest Digest; RequestTruncated, ResponseTruncated bool }` (all JSON-tagged, `trace/store.go`)
  - `type Digest struct{ Text string; Events map[string]int; FinishReason string; AudioDeltas int }` (Task 4 fills it; declared here)
  - `type Window struct{ Enabled bool; ExpiresAt *time.Time; Models, Servers []string }`
  - `func New(max int, st domain.TraceSettings) *Store`
  - `func (s *Store) Start(seconds int, models, servers []string)`
  - `func (s *Store) Stop()`
  - `func (s *Store) Window() Window`
  - `func (s *Store) Armed(model, serverID string) bool`
  - `func (s *Store) Add(e Entry)`
  - `func (s *Store) List() []Entry`
  - `func (s *Store) Get(id string) (Entry, bool)`
  - `func (s *Store) Reset()`
  - `func (s *Store) Len() int`
  - `func humanBytes(n int64) string` (internal, Tasks 3-5 format stubs with it)
  - `func truncatePrefix(b []byte, max int) (string, bool)` (internal, shared by Tasks 3 and 5)

- [ ] **Step 1: Declare `domain.TraceSettings`**

Append to `domain/config.go`:

```go
// TraceSettings are the startup defaults for the on-demand request trace.
// They arm the window at process start only; the live admin toggle is kept in
// trace.Store and never persisted, so a restart can not leave prompt capture
// switched on. Seconds = 0 means "until switched off manually".
type TraceSettings struct {
	Enabled bool     `json:"enabled,omitempty"`
	Seconds int      `json:"seconds,omitempty"`
	Models  []string `json:"models,omitempty"`
	Servers []string `json:"servers,omitempty"`
}
```

- [ ] **Step 2: Write the failing tests**

Create `trace/store_test.go`:

```go
package trace

import (
	"testing"
	"time"

	"llm-api-router/domain"
)

func TestStoreRingOverwritesOldest(t *testing.T) {
	s := New(3, domain.TraceSettings{})
	for i := range 4 {
		s.Add(Entry{ID: string(rune('a' + i))})
	}
	if s.Len() != 3 {
		t.Fatalf("len = %d, want 3", s.Len())
	}
	list := s.List()
	if list[0].ID != "b" || list[2].ID != "d" {
		t.Errorf("ring order = %q..%q, want b..d (chronological, oldest overwritten)", list[0].ID, list[2].ID)
	}
	if _, ok := s.Get("a"); ok {
		t.Error("overwritten entry still readable")
	}
	if e, ok := s.Get("c"); !ok || e.ID != "c" {
		t.Error("Get by id must find a live entry")
	}
}

func TestStoreStartArmsWindowAndFilters(t *testing.T) {
	s := New(10, domain.TraceSettings{})
	if s.Armed("gpt", "s1") {
		t.Fatal("fresh store must not be armed")
	}
	s.Start(0, []string{"gpt"}, []string{"s1"})
	if !s.Armed("gpt", "s1") {
		t.Error("matching model+server must be armed")
	}
	if s.Armed("other", "s1") {
		t.Error("model filter must exclude")
	}
	if s.Armed("gpt", "s2") {
		t.Error("server filter must exclude")
	}
	s.Start(0, nil, nil)
	if !s.Armed("anything", "anything") {
		t.Error("empty filters must match everything")
	}
	s.Stop()
	if s.Armed("anything", "anything") {
		t.Error("Stop must disarm")
	}
}

func TestStoreWindowExpiresLazily(t *testing.T) {
	now := time.Now()
	s := New(10, domain.TraceSettings{})
	s.now = func() time.Time { return now }
	s.Start(60, nil, nil)
	w := s.Window()
	if !w.Enabled || w.ExpiresAt == nil || !w.ExpiresAt.Equal(now.Add(time.Minute)) {
		t.Fatalf("window = %+v, want enabled, expiring in 60s", w)
	}
	now = now.Add(61 * time.Second)
	if s.Armed("m", "s") {
		t.Error("expired window must not arm")
	}
	if w := s.Window(); w.Enabled {
		t.Error("Window() must report the lazy expiry")
	}
}

func TestStoreNewArmsFromConfigDefaults(t *testing.T) {
	s := New(10, domain.TraceSettings{Enabled: true, Seconds: 300, Models: []string{"m"}})
	if !s.Armed("m", "any") {
		t.Error("config defaults must arm the window at startup")
	}
	if s.Armed("other", "any") {
		t.Error("config default filter must apply")
	}
}

func TestStoreResetKeepsWindow(t *testing.T) {
	s := New(10, domain.TraceSettings{})
	s.Start(0, nil, nil)
	s.Add(Entry{ID: "a"})
	s.Reset()
	if s.Len() != 0 {
		t.Fatal("reset must empty the ring")
	}
	if !s.Armed("m", "s") {
		t.Error("reset must not touch the live window")
	}
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `go test ./trace/`
Expected: compile failure — `undefined: New`, `undefined: Entry`, `s.now undefined`.

- [ ] **Step 4: Write `trace/entry.go`**

```go
// Package trace is the router's on-demand wire capture: the live window that
// decides what gets traced, the redacted request/response bodies, the request
// classification and the stream digest. It is deliberately memory-only — the
// ring dies with the process, so enabling prompt capture is never sticky.
package trace

// Digest is the derived view of a response body: the completion text (stream
// deltas reassembled), a histogram of the SSE event names seen, and the finish
// reason. Binary responses carry none of it (see Capture.Entry).
type Digest struct {
	Text         string         `json:"text,omitempty"`
	Events       map[string]int `json:"events,omitempty"`
	FinishReason string         `json:"finish_reason,omitempty"`
	// AudioDeltas counts speech.audio.delta frames (audio.cpp TTS streams):
	// their payload is base64 PCM, so it is counted, never stored.
	AudioDeltas int `json:"audio_deltas,omitempty"`
}

func humanBytes(n int64) string {
	switch {
	case n >= 1<<20:
		return itoa(n>>10>>10) + "MB"
	case n >= 1<<10:
		return itoa(n>>10) + "KB"
	default:
		return itoa(n) + "B"
	}
}

func itoa(n int64) string {
	if n == 0 {
		return "0"
	}
	var b [24]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

// truncatePrefix keeps a prefix and reports that it had to, so a trace entry
// says what it lost instead of silently shortening.
func truncatePrefix(b []byte, max int) (string, bool) {
	if len(b) <= max {
		return string(b), false
	}
	return string(b[:max]) + "…[truncated " + itoa(int64(len(b)-max)) + " bytes]", true
}
```

- [ ] **Step 5: Write `trace/store.go`**

```go
package trace

import (
	"sync"
	"time"

	"llm-api-router/domain"
)

// Entry is one traced request: the bodies as they went on the wire (binaries
// replaced by stubs) plus the derived views of them.
type Entry struct {
	ID          string    `json:"id"`
	Timestamp   time.Time `json:"timestamp"`
	Method      string    `json:"method"`
	Path        string    `json:"path"`
	Model       string    `json:"model"`
	TargetModel string    `json:"target_model,omitempty"`
	ServerID    string    `json:"server_id,omitempty"`
	Class       string    `json:"class,omitempty"`
	Features    []string  `json:"features,omitempty"`
	StatusCode  int       `json:"status_code"`
	LatencyMs   int64     `json:"latency_ms"`
	TTFBMs      int64     `json:"ttfb_ms"`
	JevChoice   string    `json:"jev_choice,omitempty"`

	Request  string `json:"request,omitempty"`
	Response string `json:"response,omitempty"`
	Digest   Digest `json:"digest"`

	RequestTruncated  bool `json:"request_truncated,omitempty"`
	ResponseTruncated bool `json:"response_truncated,omitempty"`
}

// Window is the live capture state: what is being traced right now.
type Window struct {
	Enabled   bool       `json:"enabled"`
	ExpiresAt *time.Time `json:"expires_at,omitempty"`
	Models    []string   `json:"models"`
	Servers   []string   `json:"servers"`
}

// Store is the trace ring buffer plus the live window. The ring overwrites by
// index like metrics.Store's recent requests; the window expires lazily, on the
// next read, so an idle router has no timer.
type Store struct {
	mu      sync.Mutex
	entries []Entry
	ringIdx int
	count   int
	max     int

	enabled   bool
	expiresAt time.Time
	models    []string
	servers   []string

	// now is the clock seam: tests advance it instead of sleeping.
	now func() time.Time
}

// New creates an empty ring and arms the window from the config defaults.
func New(max int, st domain.TraceSettings) *Store {
	s := &Store{
		entries: make([]Entry, max),
		max:     max,
		models:  st.Models,
		servers: st.Servers,
		now:     time.Now,
	}
	if st.Enabled {
		s.enabled = true
		if st.Seconds > 0 {
			s.expiresAt = s.now().Add(time.Duration(st.Seconds) * time.Second)
		}
	}
	return s
}

// Start arms the window for the next `seconds`, 0 = until Stop.
func (s *Store) Start(seconds int, models, servers []string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.enabled = true
	s.models, s.servers = models, servers
	if seconds > 0 {
		s.expiresAt = s.now().Add(time.Duration(seconds) * time.Second)
	} else {
		s.expiresAt = time.Time{}
	}
}

// Stop disarms the window; existing entries stay readable until Reset.
func (s *Store) Stop() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.enabled = false
	s.expiresAt = time.Time{}
}

// Window reports the live state, expiring a finished window as it reports it.
func (s *Store) Window() Window {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.expireLocked()
	w := Window{Enabled: s.enabled, Models: s.models, Servers: s.servers}
	if !s.expiresAt.IsZero() {
		at := s.expiresAt
		w.ExpiresAt = &at
	}
	return w
}

// Armed reports whether a request for `model` on `serverID` is traced. An empty
// filter matches everything.
func (s *Store) Armed(model, serverID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.expireLocked() {
		return false
	}
	return matches(s.models, model) && matches(s.servers, serverID)
}

// expireLocked disarms a finished window; it reports whether tracing is off.
// Callers hold s.mu.
func (s *Store) expireLocked() bool {
	if !s.enabled {
		return true
	}
	if !s.expiresAt.IsZero() && !s.now().Before(s.expiresAt) {
		s.enabled = false
		s.expiresAt = time.Time{}
		return true
	}
	return false
}

func matches(filter []string, v string) bool {
	if len(filter) == 0 {
		return true
	}
	for _, f := range filter {
		if f == v {
			return true
		}
	}
	return false
}

// Add writes an entry into the ring.
func (s *Store) Add(e Entry) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries[s.ringIdx] = e
	s.ringIdx = (s.ringIdx + 1) % s.max
	if s.count < s.max {
		s.count++
	}
}

// List returns the ring in chronological order.
func (s *Store) List() []Entry {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Entry, s.count)
	if s.count == 0 {
		return out
	}
	if s.count < s.max {
		copy(out, s.entries[:s.count])
	} else {
		n := copy(out, s.entries[s.ringIdx:])
		copy(out[n:], s.entries[:s.ringIdx])
	}
	return out
}

// Get returns one entry by ID (newest wins on an id collision).
func (s *Store) Get(id string) (Entry, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := s.count - 1; i >= 0; i-- {
		if s.entries[i].ID == id {
			return s.entries[i], true
		}
	}
	return Entry{}, false
}

// Len reports how many entries the ring holds.
func (s *Store) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.count
}

// Reset clears the ring; the live window is untouched (clearing the ring is not
// switching collection off).
func (s *Store) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries = make([]Entry, s.max)
	s.ringIdx = 0
	s.count = 0
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `go test ./trace/ && go build ./...`
Expected: `ok  	llm-api-router/trace`

- [ ] **Step 7: Commit**

```bash
git add trace/ domain/config.go
git commit -m "feat(trace): trace ring buffer and live capture window"
```

---

### Task 2: Request classification

**Files:**
- Create: `trace/classify.go`
- Test: `trace/classify_test.go`
- Modify: `router/router.go` (`apiEndpointFromPath` becomes a one-line delegation)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `func EndpointLabel(path string) string` — the endpoint vocabulary (values: `messages`, `chat`, `responses`, `embeddings`, `speech`, `transcriptions`, `voices`, `tts`, `completions`, else last `/v1/` segment, else `unknown`)
  - `func Classify(path, contentType string, body []byte) (class string, features []string)` — `class` is `EndpointLabel(path)`; `features` is an ordered subset of `stream`, `tools:<n>`, `tool_result`, `images:<n>`, `system:<chars>`, `reasoning`, `audio-upload`, `json:<bytes>`

- [ ] **Step 1: Write the failing tests**

Create `trace/classify_test.go`:

```go
package trace

import (
	"strings"
	"testing"
)

func containsFeature(t *testing.T, f []string, want string) {
	t.Helper()
	for _, v := range f {
		if v == want {
			return
		}
	}
	t.Errorf("features %q missing %q", f, want)
}

func TestClassifyClasses(t *testing.T) {
	cases := map[string]string{
		"/v1/chat/completions":      "chat",
		"/v1/messages":              "messages",
		"/v1/responses":             "responses",
		"/v1/embeddings":            "embeddings",
		"/v1/completions":           "completions",
		"/v1/audio/speech":          "speech",
		"/v1/audio/transcriptions":  "transcriptions",
		"/v1/audio/voices":          "voices",
		"/tts":                      "tts",
		"/v1/systemone":             "systemone",
	}
	for path, want := range cases {
		if got, _ := Classify(path, "application/json", nil); got != want {
			t.Errorf("%s = %q, want %q", path, got, want)
		}
	}
}

func TestClassifyChatFeatures(t *testing.T) {
	body := []byte(`{"model":"gpt","stream":true,"reasoning":{"effort":"low"},` +
		`"system":"you are helpful","tools":[{"type":"function"},{"type":"function"}]}`)
	class, f := Classify("/v1/chat/completions", "application/json", body)
	if class != "chat" {
		t.Fatalf("class = %q, want chat", class)
	}
	containsFeature(t, f, "stream")
	containsFeature(t, f, "tools:2")
	containsFeature(t, f, "system:15")
	containsFeature(t, f, "reasoning")
	containsFeature(t, f, "json:"+itoa(int64(len(body))))
}

func TestClassifyAgentTurnFeatures(t *testing.T) {
	body := []byte(`{"model":"opus","max_tokens":100,"messages":[{"role":"user","content":[` +
		`{"type":"tool_result","content":"ok"},` +
		`{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}]}]}`)
	_, f := Classify("/v1/messages", "application/json", body)
	containsFeature(t, f, "tool_result")
	containsFeature(t, f, "images:1")
}

func TestClassifyAudioUpload(t *testing.T) {
	_, f := Classify("/v1/audio/transcriptions", "multipart/form-data; boundary=x", []byte("--x\r\n"))
	containsFeature(t, f, "audio-upload")
	for _, v := range f {
		if strings.HasPrefix(v, "json:") {
			t.Errorf("multipart body must not be labeled json, got %q", v)
		}
	}
}

func TestClassifyEmptyBody(t *testing.T) {
	class, f := Classify("/v1/audio/voices", "", nil)
	if class != "voices" || len(f) != 0 {
		t.Errorf("GET voice list = %q/%q, want voices with no features", class, f)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./trace/ -run TestClassify`
Expected: compile failure — `undefined: Classify`.

- [ ] **Step 3: Write `trace/classify.go`**

```go
package trace

import (
	"bytes"
	"strings"
)

// EndpointLabel names the API endpoint of a request path. It is the single
// source of the vocabulary the metrics, the trace and the GUI all speak.
func EndpointLabel(path string) string {
	switch {
	case strings.Contains(path, "/messages"):
		return "messages"
	case strings.Contains(path, "/chat/completions"):
		return "chat"
	case strings.Contains(path, "/responses"):
		return "responses"
	case strings.Contains(path, "/embeddings"):
		return "embeddings"
	case strings.Contains(path, "/audio/speech"):
		return "speech"
	case strings.Contains(path, "/audio/transcriptions"):
		return "transcriptions"
	case strings.Contains(path, "/audio/voices"):
		return "voices"
	case strings.HasSuffix(path, "/tts"):
		return "tts"
	case strings.Contains(path, "/completions"):
		return "completions"
	}
	if _, after, ok := strings.Cut(path, "/v1/"); ok {
		if seg, _, _ := strings.Cut(after, "/"); seg != "" {
			return seg
		}
	}
	return "unknown"
}

// Classify labels a request and lists the secondary signs a byte scan can see.
// The body is scanned, never parsed: the same discipline as the router's model
// extraction and the multipart rewriter (an agent turn is megabytes, and the
// features are heuristics for a human reading the trace, not a parser).
func Classify(path, contentType string, body []byte) (string, []string) {
	class := EndpointLabel(path)
	var f []string
	if len(body) == 0 {
		return class, f
	}
	if jsonKeyIsTrue(body, "stream") {
		f = append(f, "stream")
	}
	// ponytail: tool declarations counted by their dialect's marker — OpenAI
	// sends {"type":"function"}, Anthropic sends input_schema. A count, not a
	// parse: good enough to spot "the client sent 40 tools".
	if n := bytes.Count(body, []byte(`"input_schema"`)) + bytes.Count(body, []byte(`"type":"function"`)); n > 0 {
		f = append(f, "tools:"+itoa(int64(n)))
	}
	if bytes.Contains(body, []byte(`"tool_result"`)) {
		f = append(f, "tool_result")
	}
	// The three shapes an image request takes on the wire: a data URL (both
	// dialects), OpenAI's image_url field, Anthropic's base64 source (which
	// names media_type instead of a URL).
	if n := max(bytes.Count(body, []byte("data:image/")),
		bytes.Count(body, []byte(`"image_url"`)),
		bytes.Count(body, []byte(`"media_type"`))); n > 0 {
		f = append(f, "images:"+itoa(int64(n)))
	}
	if start, end, ok := topLevelValueSpan(body, "system"); ok {
		f = append(f, "system:"+itoa(int64(end-start)))
	}
	if bytes.Contains(body, []byte(`"reasoning`)) || bytes.Contains(body, []byte(`"thinking"`)) {
		f = append(f, "reasoning")
	}
	if strings.HasPrefix(contentType, "multipart/") {
		f = append(f, "audio-upload")
	} else if strings.HasPrefix(contentType, "application/json") || contentType == "" {
		f = append(f, "json:"+itoa(int64(len(body))))
	}
	return class, f
}

// jsonKeyIsTrue reports whether the top-level key's value is the literal true.
func jsonKeyIsTrue(body []byte, key string) bool {
	start, end, ok := topLevelValueSpan(body, key)
	return ok && bytes.Equal(bytes.TrimSpace(body[start:end]), []byte("true"))
}

// topLevelValueSpan locates the value of a top-level JSON key, scanning bytes:
// it walks quoted strings (so a `"system"` inside a prompt is not a key) and
// container depth, then spans the value. For a string the span excludes the
// quotes, so `system:<chars>` is the text length.
func topLevelValueSpan(body []byte, key string) (start, end int, ok bool) {
	needle := []byte(`"` + key + `"`)
	depth := 0
	for i := 0; i < len(body); {
		switch body[i] {
		case '{', '[':
			depth++
			i++
		case '}', ']':
			depth--
			i++
		case '"':
			isKey := depth == 1 && bytes.HasPrefix(body[i:], needle)
			j := i + len(needle)
			if isKey && j < len(body) && isNameByte(body[j]) {
				isKey = false // a longer key: "system_prompt" is not "system"
			}
			if !isKey {
				i = skipJSONString(body, i)
				continue
			}
			i = j
			for i < len(body) && isSpaceByte(body[i]) {
				i++
			}
			if i >= len(body) || body[i] != ':' {
				i++
				continue
			}
			i++
			for i < len(body) && isSpaceByte(body[i]) {
				i++
			}
			return i, valueEnd(body, i), true
		default:
			i++
		}
	}
	return 0, 0, false
}

// valueEnd spans a JSON value from start: a string to its closing quote
// (exclusive), a container to the brace that closes it, anything else to the
// next comma or closing brace.
func valueEnd(body []byte, start int) int {
	if start >= len(body) {
		return start
	}
	switch body[start] {
	case '"':
		return skipJSONString(body, start) - 1
	case '{', '[':
		depth := 0
		for i := start; i < len(body); i++ {
			switch body[i] {
			case '{', '[':
				depth++
			case '}', ']':
				depth--
				if depth == 0 {
					return i + 1
				}
			case '"':
				i = skipJSONString(body, i) - 1
			}
		}
		return len(body)
	default:
		end := start
		for end < len(body) && body[end] != ',' && body[end] != '}' && body[end] != ']' {
			end++
		}
		return end
	}
}

func skipJSONString(body []byte, i int) int {
	i++
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

func isNameByte(b byte) bool {
	return b == '_' || b == '-' || (b >= '0' && b <= '9') || (b|0x20 >= 'a' && b|0x20 <= 'z')
}

func isSpaceByte(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./trace/ -run TestClassify -v`
Expected: PASS for all five tests.

- [ ] **Step 5: Delegate the router's labeler**

In `router/router.go`, replace the body of `apiEndpointFromPath` (keep the doc comment, extend it) so the vocabulary has one owner:

```go
// apiEndpointFromPath extracts the specific API endpoint from the request path.
// The vocabulary lives in trace.EndpointLabel, so metrics, the trace and the
// GUI describe a request with the same words.
func apiEndpointFromPath(path string) string {
	return trace.EndpointLabel(path)
}
```

Add `llm-api-router/trace` to the router's import block.

This delegator is temporary: Task 9 makes the router carry the class in `reqMeta`, after which `apiEndpointFromPath` has no caller and is deleted there (its test's cases are already covered by `TestClassifyClasses`).

- [ ] **Step 6: Run the router's existing label test**

Run: `go test ./router/ -run TestAPIEndpointFromPath -v`
Expected: PASS (unchanged expectations — the labels are identical).

- [ ] **Step 7: Commit**

```bash
git add trace/classify.go trace/classify_test.go router/router.go
git commit -m "feat(trace): byte-scan request classification"
```

---

### Task 3: Redaction (binaries → stubs)

**Files:**
- Create: `trace/redact.go`
- Test: `trace/redact_test.go`

**Interfaces:**
- Consumes: `humanBytes`, `truncatePrefix`, `itoa` (`trace/entry.go`).
- Produces:
  - `const MaxRequestBytes = 512 << 10`
  - `const MaxResponseBytes = 512 << 10`
  - `func RedactRequest(body []byte, contentType string) (text string, truncated bool)`

- [ ] **Step 1: Write the failing tests**

Create `trace/redact_test.go`:

```go
package trace

import (
	"bytes"
	"strings"
	"testing"
)

func TestRedactRequestDataURLAndAnthropicSource(t *testing.T) {
	body := []byte(`{"model":"opus","messages":[{"role":"user","content":[` +
		`{"type":"image_url","image_url":{"url":"data:image/png;base64,` + strings.Repeat("A", 4000) + `"}},` +
		`{"type":"text","text":"what is in the picture?"}` +
		`]}]}`)
	got, trunc := RedactRequest(body, "application/json")
	if trunc {
		t.Error("small body must not be truncated")
	}
	if strings.Contains(got, strings.Repeat("A", 100)) {
		t.Error("base64 payload must be replaced by a stub")
	}
	if !strings.Contains(got, "<image ") || !strings.Contains(got, "sha256:") {
		t.Errorf("stub missing: %s", got)
	}
	if !strings.Contains(got, "what is in the picture?") {
		t.Error("prompt text must survive untouched")
	}
}

func TestRedactRequestRawBase64Source(t *testing.T) {
	body := []byte(`{"model":"sonnet","source":{"type":"base64","media_type":"image/png","data":"` +
		strings.Repeat("QUJD", 300) + `"}}`)
	got, _ := RedactRequest(body, "application/json")
	if strings.Contains(got, strings.Repeat("QUJD", 4)) {
		t.Error("raw base64 source.data must be replaced by a stub")
	}
	if !strings.Contains(got, "<image ") {
		t.Errorf("stub missing: %s", got)
	}
}

func TestRedactRequestTextUntouched(t *testing.T) {
	body := []byte(`{"model":"gpt","messages":[{"role":"user","content":"hello"}]}`)
	got, _ := RedactRequest(body, "application/json")
	if got != string(body) {
		t.Errorf("text-only request changed:\n%s", got)
	}
}

func TestRedactRequestTruncatesOversizedBody(t *testing.T) {
	body := bytes.Repeat([]byte("x"), MaxRequestBytes+4096)
	got, trunc := RedactRequest(body, "text/plain")
	if !trunc {
		t.Error("oversized body must report truncation")
	}
	if !strings.HasSuffix(got, "…[truncated 4096 bytes]") {
		t.Errorf("truncation marker missing: …%s", got[len(got)-40:])
	}
	if len(got) <= MaxRequestBytes {
		t.Errorf("len = %d, want the prefix plus the marker", len(got))
	}
}

func TestRedactRequestMultipartAudio(t *testing.T) {
	audio := append([]byte("RIFF"), bytes.Repeat([]byte{0x01}, 4000)...)
	var b bytes.Buffer
	mw := multipart.NewWriter(&b)
	_ = mw.WriteField("model", "whisper")
	part, err := mw.CreateFormFile("file", "clip.wav")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(audio); err != nil {
		t.Fatal(err)
	}
	mw.Close()

	got, _ := RedactRequest(b.Bytes(), mw.FormDataContentType())
	if !bytes.Contains(b.Bytes(), audio) {
		t.Fatal("test fixture lost the audio")
	}
	if strings.Contains(got, "\x01\x01\x01") {
		t.Error("audio bytes must be replaced by a stub")
	}
	if !strings.Contains(got, "<audio ") || !strings.Contains(got, "sha256:") {
		t.Errorf("stub missing: %s", got)
	}
	if !strings.Contains(got, `name="model"`) || !strings.Contains(got, "whisper") {
		t.Error("the model field and part headers must stay readable")
	}
}
```

Add the fixture helper at the bottom of the same file:

Add `mime/multipart` to the test file's imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./trace/ -run TestRedact`
Expected: compile failure — `undefined: RedactRequest`.

- [ ] **Step 3: Write `trace/redact.go`**

```go
package trace

import (
	"bytes"
	"encoding/json"
	"strings"
)

// MaxRequestBytes and MaxResponseBytes bound what one trace entry keeps per
// body. 512 KiB is generous for a prompt (a 100k-token context is a megabyte of
// text) and small enough that a full ring is tens of megabytes.
const (
	MaxRequestBytes  = 512 << 10
	MaxResponseBytes = 512 << 10
)

// RedactRequest returns the request body as it may live in the trace: prompt
// text untouched, every inlined binary replaced by `<kind SIZE sha256:xxxx…>`.
// The bool reports a body truncated past MaxRequestBytes.
//
// The hash runs over the payload as stored — for a data URL / base64 field that
// is the base64 text, which identifies the payload as well as decoding it, and
// keeps a decoder off the request path.
func RedactRequest(body []byte, contentType string) (string, bool) {
	var out string
	switch {
	case len(body) == 0:
		return "", false
	case strings.HasPrefix(contentType, "multipart/"):
		out = redactMultipart(body)
	default:
		out = redactJSON(body)
	}
	return truncatePrefix([]byte(out), MaxRequestBytes)
}

// redactJSON rewrites the binary leaves of a JSON body. A body that is not JSON
// (plain text, a truncated upload) is returned as-is: redaction is best-effort
// and must never eat the request it documents.
func redactJSON(body []byte) string {
	var v any
	if err := json.Unmarshal(body, &v); err != nil {
		return string(body)
	}
	if !walkRedact(v) {
		return string(body)
	}
	out, err := json.Marshal(v)
	if err != nil {
		return string(body)
	}
	return string(out)
}

func walkRedact(v any) bool {
	changed := false
	switch t := v.(type) {
	case map[string]any:
		for k, val := range t {
			if s, ok := val.(string); ok {
				if kind, size, raw, isBin := binaryString(k, s); isBin {
					t[k] = stub(kind, size, raw)
					changed = true
					continue
				}
			}
			if walkRedact(val) {
				changed = true
			}
		}
	case []any:
		for _, val := range t {
			if walkRedact(val) {
				changed = true
			}
		}
	}
	return changed
}

// binaryString recognises an inlined binary in a JSON string: a data URL (under
// any key), or a long base64 run under an image/audio-flavoured key — the
// Anthropic `source.data` shape, which carries raw base64 with no data: prefix.
func binaryString(key, s string) (kind string, size int64, raw []byte, ok bool) {
	if i := strings.Index(s, ";base64,"); strings.HasPrefix(s, "data:") && i > 0 {
		payload := s[i+len(";base64,"):]
		return coarseKind(s[len("data:"):i]), decodedLen(len(payload)), []byte(payload), true
	}
	kind = kindForKey(key)
	if kind == "" || len(s) < 512 {
		return "", 0, nil, false
	}
	for i := range 64 {
		if i >= len(s) {
			break
		}
		if !isBase64Byte(s[i]) {
			return "", 0, nil, false
		}
	}
	return kind, decodedLen(len(s)), []byte(s), true
}

// kindForKey names a stub after the JSON field the binary came from.
func kindForKey(key string) string {
	switch key {
	case "audio", "audio_data", "input_audio":
		return "audio"
	case "data", "image", "image_data":
		return "image"
	case "pdf_data":
		return "document"
	default:
		return ""
	}
}

// coarseKind maps a media type to a stub kind ("image/png" → image,
// "audio/wav" → audio): the kind a human scans for, not the exact subtype.
func coarseKind(mediaType string) string {
	base, _, _ := strings.Cut(strings.ToLower(strings.TrimSpace(mediaType)), ";")
	top, sub, hasSub := strings.Cut(base, "/")
	switch top {
	case "image", "audio", "video", "text":
		return top
	}
	switch {
	case !hasSub:
		return "binary"
	case sub == "wav" || sub == "mp3" || sub == "ogg" || sub == "opus" || sub == "flac" || sub == "pcm" || sub == "mpeg":
		return "audio"
	default:
		return sub
	}
}

func decodedLen(n int) int64 { return int64(n) * 3 / 4 }

func isBase64Byte(b byte) bool {
	return (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') || (b >= '0' && b <= '9') ||
		b == '+' || b == '/' || b == '='
}

// redactMultipart keeps the part headers and the `model` field and replaces
// every file part's payload with a stub.
//
// ponytail: naive boundary split, the discipline of proxy/multipart.go — the
// form is never parsed, so megabytes of audio are copied zero times.
func redactMultipart(body []byte) string {
	nl := bytes.IndexByte(body, '\n')
	if nl < 0 {
		return string(body)
	}
	delim := string(bytes.TrimRight(body[:nl], "\r\n"))
	var out bytes.Buffer
	for _, part := range bytes.Split(body[nl+1:], []byte("\n"+delim)) {
		out.WriteString(delim)
		head, payload, ok := splitPart(part)
		if !ok {
			out.Write(part)
			continue
		}
		out.Write(head)
		kind := partKind(head)
		if kind == "" {
			out.Write(payload)
			continue
		}
		data := bytes.TrimRight(payload, "\r\n")
		out.WriteString("\r\n\r\n")
		out.WriteString(stub(kind, int64(len(data)), data))
		out.WriteString("\r\n")
	}
	return out.String()
}

// splitPart splits a multipart part into its header block (through the blank
// line, separators included) and its payload.
func splitPart(part []byte) (head, payload []byte, ok bool) {
	for _, sep := range [][]byte{[]byte("\r\n\r\n"), []byte("\n\n")} {
		if i := bytes.Index(part, sep); i >= 0 {
			return part[:i+len(sep)], part[i+len(sep):], true
		}
	}
	return nil, nil, false
}

// partKind names a file part after its Content-Type (falling back to the
// presence of a filename), and returns "" for a plain form field like `model`.
func partKind(head []byte) string {
	lower := bytes.ToLower(head)
	if i := bytes.Index(lower, []byte("content-type:")); i >= 0 {
		value := string(head[i+len("content-type:"):])
		value, _, _ = strings.Cut(value, "\n")
		if kind := coarseKind(strings.TrimSpace(value)); kind != "binary" {
			return kind
		}
	}
	if bytes.Contains(lower, []byte("filename=")) {
		return "file"
	}
	return ""
}
```

Add the shared renderer at the bottom of `trace/entry.go`:

```go
// stub renders a redaction placeholder: the kind, the size, and a short hash
// that identifies the payload across requests.
func stub(kind string, size int64, raw []byte) string {
	sum := sha256.Sum256(raw)
	if len(raw) < sha256.Size {
		sum = sha256.Sum256(append(append([]byte{}, raw...), make([]byte, sha256.Size-len(raw))...))
	}
	return fmt.Sprintf("<%s %s sha256:%x…>", kind, humanBytes(size), sum[:4])
}
```

with `crypto/sha256` and `fmt` added to `trace/entry.go`'s imports.

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./trace/ -run TestRedact -v`
Expected: PASS for all five tests. Fix the implementation, not the expectations, unless a stub shape is genuinely wrong.

- [ ] **Step 5: Commit**

```bash
git add trace/redact.go trace/redact_test.go trace/entry.go
git commit -m "feat(trace): redact inlined binaries to size+hash stubs"
```

---

### Task 4: Response digest

**Files:**
- Create: `trace/digest.go`
- Test: `trace/digest_test.go`

**Interfaces:**
- Consumes: nothing from earlier tasks (declares the `Digest` fields from Task 1 — move the type declaration out of `trace/store.go`… it stays in `trace/entry.go`; this task only fills it).
- Produces: `func DigestResponse(raw []byte, contentType string) Digest` — the spec names this `trace.Digest`; the package already declares a *type* of that name, so the function takes the longer one

- [ ] **Step 1: Write the failing tests**

Create `trace/digest_test.go`:

```go
package trace

import "testing"

func TestDigestOpenAIStream(t *testing.T) {
	raw := []byte("data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"lo\",\"reasoning_content\":\"thinking\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n")
	d := DigestResponse(raw, "text/event-stream")
	if d.Text != "Hello thinking" {
		t.Errorf("text = %q, want the deltas joined (content then reasoning)", d.Text)
	}
	if d.FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want stop", d.FinishReason)
	}
	if d.Events["data"] != 3 {
		t.Errorf("events = %v, want data:3 (the [DONE] sentinel is not a frame)", d.Events)
	}
}

func TestDigestAnthropicStream(t *testing.T) {
	raw := []byte("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hm\"}}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"},\"usage\":{\"output_tokens\":2}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	d := DigestResponse(raw, "text/event-stream")
	if d.Text != "Himhm" {
		t.Errorf("text = %q, want body and thinking joined (no separator, as for OpenAI)", d.Text)
	}
	if d.FinishReason != "max_tokens" {
		t.Errorf("finish_reason = %q, want max_tokens", d.FinishReason)
	}
	if d.Events["content_block_delta"] != 2 || d.Events["message_stop"] != 1 {
		t.Errorf("events = %v, want content_block_delta:2 message_stop:1", d.Events)
	}
}

func TestDigestResponsesStream(t *testing.T) {
	raw := []byte("event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"Once\"}\n\n" +
		"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\" upon\"}\n\n" +
		"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n")
	d := DigestResponse(raw, "text/event-stream")
	if d.Text != "Once upon" {
		t.Errorf("text = %q, want \"Once upon\"", d.Text)
	}
	if d.FinishReason != "completed" {
		t.Errorf("finish_reason = %q, want completed", d.FinishReason)
	}
}

func TestDigestAudioStream(t *testing.T) {
	raw := []byte("event: speech.audio.delta\ndata: {\"type\":\"speech.audio.delta\",\"delta\":\"AAAA\"}\n\n" +
		"event: transcript.text.delta\ndata: {\"type\":\"transcript.text.delta\",\"delta\":\"hi\"}\n\n" +
		"event: transcript.text.done\ndata: {\"type\":\"transcript.text.done\"}\n\n")
	d := DigestResponse(raw, "text/event-stream")
	if d.Text != "hi" {
		t.Errorf("text = %q, want the transcript only", d.Text)
	}
	if d.AudioDeltas != 1 {
		t.Errorf("audio_deltas = %d, want 1 (base64 PCM counted, never stored)", d.AudioDeltas)
	}
}

func TestDigestJSON(t *testing.T) {
	d := DigestResponse([]byte(`{"choices":[{"message":{"content":"Hi there"},"finish_reason":"stop"}]}`), "application/json")
	if d.Text != "Hi there" || d.FinishReason != "stop" {
		t.Errorf("digest = %+v, want the message content and finish_reason", d)
	}
	if d.Events != nil {
		t.Errorf("events = %v, want none for a JSON response", d.Events)
	}

	d = DigestResponse([]byte(`{"role":"assistant","stop_reason":"end_turn","content":[{"type":"text","text":"A "},{"type":"text","text":"B"}]}`), "application/json")
	if d.Text != "A B" || d.FinishReason != "end_turn" {
		t.Errorf("digest = %+v, want Anthropic content blocks joined", d)
	}
}

func TestDigestBinary(t *testing.T) {
	d := DigestResponse([]byte{0x52, 0x49, 0x46, 0x46}, "audio/wav")
	if d.Text != "" || d.Events != nil {
		t.Errorf("digest = %+v, want nothing parsed out of a binary body", d)
	}
}

func TestDigestUnparseableFramesIgnored(t *testing.T) {
	d := DigestResponse([]byte("data: not json\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n"), "text/event-stream")
	if d.Text != "ok" {
		t.Errorf("text = %q, want the frame that did parse", d.Text)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./trace/ -run TestDigest`
Expected: compile failure — `undefined: DigestResponse`.

- [ ] **Step 3: Write `trace/digest.go`**

```go
package trace

import (
	"bytes"
	"encoding/json"
	"strings"
)

// DigestResponse reads a response body once, when the entry is written to the ring.
// SSE bodies are stored as the raw frames the backend sent, so the deltas are
// reassembled and the event names counted here (OpenAI chat, OpenAI Responses,
// Anthropic messages, audio.cpp's audio SSE) — the compact view the GUI shows
// before a human expands the raw stream.
func DigestResponse(raw []byte, contentType string) Digest {
	ct := strings.ToLower(contentType)
	switch {
	case strings.Contains(ct, "text/event-stream"):
		return digestStream(raw)
	case strings.Contains(ct, "json"):
		return digestJSONResponse(raw)
	default:
		// TTS WAV, raw PCM: nothing to parse, the capture renders the stub.
		return Digest{}
	}
}

// digestStream reassembles deltas and counts event names on one pass. A frame
// that fails to parse is counted and skipped — the digest is a view of the
// stream, never a reason to lose the entry.
func digestStream(raw []byte) Digest {
	d := Digest{}
	d.Events = map[string]int{}
	var text strings.Builder
	var event string

	forEachLine(raw, func(line []byte) {
		if bytes.HasPrefix(line, []byte("event:")) {
			event = string(bytes.TrimSpace(line[len("event:"):]))
			d.Events[event]++
			return
		}
		if !bytes.HasPrefix(line, []byte("data:")) {
			return
		}
		payload := bytes.TrimSpace(line[len("data:"):])
		if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
			return
		}
		var obj map[string]any
		if err := json.Unmarshal(payload, &obj); err != nil {
			d.Events["unparsed"]++
			return
		}
		name := event
		if t, _ := obj["type"].(string); t != "" {
			name = t
		}
		if name == "" {
			name = "data"
		}
		if event == "" || name != event {
			d.Events[name]++
		}

		switch name {
		case "response.output_text.delta", "transcript.text.delta":
			if s, _ := obj["delta"].(string); s != "" {
				text.WriteString(s)
			}
		case "speech.audio.delta":
			d.AudioDeltas++
		case "content_block_delta":
			if delta, _ := obj["delta"].(map[string]any); delta != nil {
				writeStrings(&text, delta["text"], delta["thinking"])
			}
		case "message_delta", "response.completed":
			// Only a real answer overwrites: the frames after message_delta
			// carry no reason and must not erase it.
			if fr := firstFinishReason(obj); fr != "" {
				d.FinishReason = fr
			}
		}
		if _, ok := obj["choices"]; ok {
			openAIChunkText(&text, &d, obj)
		}
		event = ""
	})

	d.Text = text.String()
	if len(d.Events) == 0 {
		d.Events = nil
	}
	return d
}

// openAIChunkText pulls text and the finish reason out of OpenAI's
// `choices[]` shape (chat completions and legacy completions).
func openAIChunkText(text *strings.Builder, d *Digest, obj map[string]any) {
	choices, _ := obj["choices"].([]any)
	if len(choices) == 0 {
		return
	}
	c, _ := choices[0].(map[string]any)
	if c == nil {
		return
	}
	if delta, _ := c["delta"].(map[string]any); delta != nil {
		writeStrings(text, delta["content"], delta["reasoning_content"])
	}
	if msg, _ := c["message"].(map[string]any); msg != nil {
		writeStrings(text, msg["content"])
	}
	if fr, _ := c["finish_reason"].(string); fr != "" {
		d.FinishReason = fr
	}
}

func firstFinishReason(obj map[string]any) string {
	if inner, _ := obj["delta"].(map[string]any); inner != nil {
		if s, _ := inner["stop_reason"].(string); s != "" {
			return s
		}
	}
	if inner, _ := obj["response"].(map[string]any); inner != nil {
		if s, _ := inner["status"].(string); s != "" {
			return s
		}
	}
	return ""
}

func writeStrings(b *strings.Builder, vals ...any) {
	for _, v := range vals {
		switch t := v.(type) {
		case string:
			b.WriteString(t)
		case []any: // Responses' output_text is an array of strings
			for _, item := range t {
				if s, ok := item.(string); ok {
					b.WriteString(s)
				}
			}
		}
	}
}

// digestJSONResponse reads a non-streaming response: the completion text
// (OpenAI `choices[].message.content`, Anthropic `content[]` blocks, Responses
// `output_text`) and its finish reason.
func digestJSONResponse(raw []byte) Digest {
	var d Digest
	var obj map[string]any
	if err := json.Unmarshal(raw, &obj); err != nil {
		return d
	}
	var text strings.Builder
	if _, ok := obj["choices"]; ok {
		openAIChunkText(&text, &d, obj)
	}
	writeStrings(&text, obj["output_text"])
	switch blocks := obj["content"].(type) {
	case string:
		text.WriteString(blocks)
	case []any:
		for _, blk := range blocks {
			m, _ := blk.(map[string]any)
			if m == nil {
				continue
			}
			writeStrings(&text, m["text"], m["thinking"])
		}
	}
	d.Text = text.String()
	if fr := firstFinishReason(obj); fr != "" {
		d.FinishReason = fr
	}
	if s, _ := obj["finish_reason"].(string); s != "" {
		d.FinishReason = s
	}
	if s, _ := obj["stop_reason"].(string); s != "" {
		d.FinishReason = s
	}
	return d
}

func forEachLine(body []byte, fn func(line []byte)) {
	for offset := 0; offset < len(body); {
		nl := bytes.IndexByte(body[offset:], '\n')
		var line []byte
		if nl == -1 {
			line = body[offset:]
			offset = len(body)
		} else {
			line = body[offset : offset+nl]
			offset += nl + 1
		}
		line = bytes.TrimSpace(line)
		if len(line) > 0 {
			fn(line)
		}
	}
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./trace/ -run TestDigest -v`
Expected: PASS for all seven tests. Deltas are concatenated with no separator (the same rule for every dialect, so the Anthropic case expects `"Himhm"`).

- [ ] **Step 5: Commit**

```bash
git add trace/digest.go trace/digest_test.go
git commit -m "feat(trace): response digest — stream delta reassembly and event histogram"
```

---

### Task 5: Per-request capture

**Files:**
- Create: `trace/capture.go`
- Test: `trace/capture_test.go`

**Interfaces:**
- Consumes: `RedactRequest`, `Digest`, `Entry`, `truncatePrefix`, `stub`, `humanBytes`, `MaxRequestBytes`, `MaxResponseBytes`.
- Produces:
  - `func NewCapture(method, path, model, contentType string, body []byte) *Capture` — stores the redacted request, classifies it, starts a 4-hex-byte random ID and a running SHA-256 of the response
  - `func (c *Capture) ID() string`
  - `func (c *Capture) ResponseContentType(ct string)`
  - `func (c *Capture) WriteResponse(p []byte)`
  - `func (c *Capture) HasResponse() bool`
  - `func (c *Capture) Entry() Entry`
  - `func isBinaryType(contentType string) bool` (internal: neither JSON nor SSE)

- [ ] **Step 1: Write the failing tests**

Create `trace/capture_test.go`:

```go
package trace

import (
	"strings"
	"testing"
)

func TestCaptureStoresRedactedRequestAndRawResponse(t *testing.T) {
	c := NewCapture("POST", "/v1/chat/completions", "gpt", "application/json",
		[]byte(`{"model":"gpt","messages":[{"role":"user","content":"hi"}],"stream":true}`))
	if c.ID() == "" {
		t.Fatal("capture must carry a trace id")
	}
	c.ResponseContentType("text/event-stream")
	c.WriteResponse([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"He\"}}]}\n\n"))
	c.WriteResponse([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"y\"},\"finish_reason\":\"stop\"}]}\n\n"))

	e := c.Entry()
	if e.Request == "" || !strings.Contains(e.Request, "hi") {
		t.Errorf("request = %q, want the prompt text", e.Request)
	}
	if !strings.Contains(e.Response, "data: {") {
		t.Errorf("response = %q, want the frames as they arrived", e.Response)
	}
	if e.Digest.Text != "Hey" || e.Digest.FinishReason != "stop" {
		t.Errorf("digest = %+v, want the reassembled text and finish_reason", e.Digest)
	}
	if len(e.Features) == 0 {
		t.Error("features from the request body must be carried")
	}
}

func TestCaptureBinaryResponseIsAStub(t *testing.T) {
	c := NewCapture("POST", "/v1/audio/speech", "tts", "application/json", []byte(`{"model":"tts","input":"hi"}`))
	c.ResponseContentType("audio/wav")
	c.WriteResponse([]byte("RIFF"))
	c.WriteResponse([]byte(strings.Repeat("\x00", 4096)))

	e := c.Entry()
	if !strings.HasPrefix(e.Response, "<audio ") || !strings.Contains(e.Response, "sha256:") {
		t.Errorf("response = %q, want an audio stub with size and hash", e.Response)
	}
	if strings.Contains(e.Response, "\x00") {
		t.Error("audio bytes must not be buffered")
	}
	if e.Digest.Text != "" {
		t.Error("a binary response has nothing to digest")
	}
	if !c.HasResponse() {
		t.Error("HasResponse must report the bytes seen")
	}
}

func TestCaptureCapsResponse(t *testing.T) {
	c := NewCapture("POST", "/v1/chat/completions", "gpt", "application/json", nil)
	c.ResponseContentType("application/json")
	chunk := strings.Repeat("x", 128<<10)
	for range (MaxResponseBytes/(128<<10) + 2) {
		c.WriteResponse([]byte(chunk))
	}
	e := c.Entry()
	if !e.ResponseTruncated {
		t.Error("response over the cap must report truncation")
	}
	if !strings.HasSuffix(e.Response, "bytes]") {
		t.Errorf("response tail = %q, want the truncation marker", e.Response[len(e.Response)-20:])
	}
}

func TestCaptureWithoutResponseIsEmpty(t *testing.T) {
	c := NewCapture("POST", "/v1/chat/completions", "gpt", "application/json", nil)
	if c.HasResponse() {
		t.Error("no chunks means no response")
	}
	if e := c.Entry(); e.Response != "" {
		t.Errorf("response = %q, want empty so the router can fill the error body", e.Response)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./trace/ -run TestCapture`
Expected: compile failure — `undefined: NewCapture`.

- [ ] **Step 3: Write `trace/capture.go`**

```go
package trace

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"hash"
	"strings"
	"sync"
	"time"
)

// Capture collects one traced request. The router builds it at request start
// (the decision is made once, so a request that starts inside the window is
// traced to the end even if the window expires mid-generation) and the proxy's
// writer chain streams the response into it — no second read of the response,
// nothing deep-copied.
type Capture struct {
	mu    sync.Mutex
	id    string
	ts    time.Time
	method string
	path  string
	model string

	request   string
	reqTrunc  bool
	class     string
	features  []string

	resp      bytes.Buffer
	respTrunc bool
	respBytes int64
	respType  string
	sha       hash.Hash
}

// NewCapture starts a capture of a request: the body is redacted here, at the
// one moment the whole request body is in hand.
func NewCapture(method, path, model, contentType string, body []byte) *Capture {
	var id [4]byte
	if _, err := rand.Read(id[:]); err != nil {
		// crypto/rand failing is a broken platform, not a request error; a
		// constant id is better than losing the trace.
		id = [4]byte{0, 0, 0, 0}
	}
	req, trunc := RedactRequest(body, contentType)
	class, features := Classify(path, contentType, body)
	return &Capture{
		id:       hex.EncodeToString(id[:]),
		ts:       time.Now(),
		method:   method,
		path:     path,
		model:    model,
		request:  req,
		reqTrunc: trunc,
		class:    class,
		features: features,
		sha:      sha256.New(),
	}
}

// ID is the trace id, surfaced to the client as X-Router-Trace-Id.
func (c *Capture) ID() string { return c.id }

// ResponseContentType records the response's content type before its first
// chunk (the proxy calls it at WriteHeader): JSON and SSE bodies are stored,
// everything else (a TTS WAV, raw PCM) becomes a stub.
func (c *Capture) ResponseContentType(ct string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.respType = ct
}

// WriteResponse records one response chunk.
func (c *Capture) WriteResponse(p []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.respBytes += int64(len(p))
	// The hash runs over the whole body so a stub identifies the payload even
	// when the stored prefix does not contain it.
	if _, err := c.sha.Write(p); err != nil {
		return
	}
	if isBinaryType(c.respType) {
		return // the stub carries size + hash; the bytes are useless here
	}
	if c.resp.Len() < MaxResponseBytes {
		c.resp.Write(p)
		return
	}
	c.respTrunc = true
}

// HasResponse reports whether anything arrived from the backend.
func (c *Capture) HasResponse() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.respBytes > 0
}

// Entry finalizes the capture, computing the digest once from what was stored.
func (c *Capture) Entry() Entry {
	c.mu.Lock()
	defer c.mu.Unlock()
	e := Entry{
		ID:                c.id,
		Timestamp:         c.ts,
		Method:            c.method,
		Path:              c.path,
		Model:             c.model,
		Class:             c.class,
		Features:          c.features,
		Request:           c.request,
		ResponseTruncated: c.respTrunc,
		RequestTruncated:  c.reqTrunc,
	}
	if isBinaryType(c.respType) {
		e.Response = stub(coarseKind(c.respType), c.respBytes, c.sha.Sum(nil))
		return e
	}
	e.Response, e.ResponseTruncated = truncatePrefix(c.resp.Bytes(), MaxResponseBytes)
	e.Digest = DigestResponse(c.resp.Bytes(), c.respType)
	return e
}

// isBinaryType reports a response body that is neither JSON nor SSE.
func isBinaryType(contentType string) bool {
	ct := strings.ToLower(contentType)
	return !strings.Contains(ct, "json") && !strings.Contains(ct, "text/event-stream")
}
```

The response hash covers every byte the backend sent, so a stub identifies a
payload whose stored prefix does not contain it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./trace/ -v`
Expected: PASS — every test in the package, Tasks 1-5 together.

- [ ] **Step 5: Commit**

```bash
git add trace/capture.go trace/capture_test.go
git commit -m "feat(trace): per-request capture with binary stubs and response cap"
```

---

### Task 6: Classification reaches metrics

**Files:**
- Modify: `domain/metric.go` (`RequestMetric`)
- Modify: `metrics/metrics.go` (`Add`)
- Test: `metrics/metrics_test.go`

**Interfaces:**
- Consumes: nothing (the class label is a plain string; Task 9 fills it).
- Produces:
  - `domain.RequestMetric.Class string` (`json:"class,omitempty"`), `domain.RequestMetric.Features []string` (`json:"features,omitempty"`)
  - a fourth summary key family: `class:<label>` alongside `model` / `server:<id>` / `target:<server>:<model>`

- [ ] **Step 1: Write the failing tests**

Append to `metrics/metrics_test.go`:

```go
func TestAddSummariesByRequestClass(t *testing.T) {
	s := New(100)
	s.Add(domain.RequestMetric{Model: "gpt", ServerID: "s1", StatusCode: 200, Class: "chat",
		PromptTokens: 100, CachedTokens: 50, CacheCreationTokens: 50})
	s.Add(domain.RequestMetric{Model: "gpt", ServerID: "s1", StatusCode: 200, Class: "chat",
		PromptTokens: 100, CachedTokens: 0})
	s.Add(domain.RequestMetric{Model: "gpt", ServerID: "s1", StatusCode: 200, Class: "messages",
		PromptTokens: 10, CachedTokens: -1})

	summaries := s.Summaries()
	chat, ok := summaries["class:chat"]
	if !ok {
		t.Fatalf("summaries %v missing class:chat", keys(summaries))
	}
	if chat.TotalRequests != 2 {
		t.Errorf("requests = %d, want 2", chat.TotalRequests)
	}
	// Cache hit is read off the group's token sums by the GUI, so the sums must
	// be complete: 50 cached out of 200+50+50 prompt-side tokens.
	if chat.TotalCachedTok != 50 || chat.TotalPromptTok != 200 || chat.TotalCacheCreationTok != 50 {
		t.Errorf("token sums = cached %d prompt %d new %d, want 50/200/50",
			chat.TotalCachedTok, chat.TotalPromptTok, chat.TotalCacheCreationTok)
	}
	if _, ok := summaries["class:messages"]; !ok {
		t.Error("every class must get its own summary")
	}
}
```

Reuse the existing key-listing helper if one exists; if not, add at the bottom of the file:

```go
func keys(m map[string]domain.Summary) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./metrics/ -run 'TestAddSummariesByRequestClass|TestPrometheusSeparatesClassGroup'`
Expected: compile failure — `unknown field Class in struct literal`.

- [ ] **Step 3: Add the fields**

In `domain/metric.go`, inside `RequestMetric`, after `APIEndpoint`:

```go
	// Request shape, classified by heuristic on every request (0 tokens):
	// the endpoint class plus the signs a byte scan saw (stream, tools:<n>,
	// images:<n>, …). Aggregated into `class:<label>` summaries.
	Class    string   `json:"class,omitempty"`
	Features []string `json:"features,omitempty"`
```

- [ ] **Step 4: Add the summary key**

In `metrics/metrics.go`, in `Add`, after the `target:` line:

```go
	// Requests of the same shape aggregate together (chat vs speech vs …), so
	// the GUI's table machinery and the cache-hit maths work per class too.
	if m.Class != "" {
		s.updateSummary("class:"+m.Class, m)
	}
```

`metrics/prometheus.go` needs no change: its exposition format carries a single
`group="<key>"` label, and `target:` summary keys already surface that way, so
`group="class:chat"` is the established shape, not a leak.

- [ ] **Step 5: Run the full metrics suite**

Run: `go test ./metrics/`
Expected: `ok  	llm-api-router/metrics`

- [ ] **Step 6: Commit**

```bash
git add domain/metric.go metrics/
git commit -m "feat(metrics): aggregate summaries by request class"
```

---

### Task 7: `Settings.Trace` startup defaults

**Files:**
- Modify: `domain/config.go` (`Settings`)
- Modify: `example.config.json`
- Test: `config/store_test.go`

**Interfaces:**
- Consumes: `domain.TraceSettings` (Task 1).
- Produces: `domain.Settings.Trace TraceSettings` (`json:"trace,omitempty"`), read via the existing `config.Store.GetSettings()`.

- [ ] **Step 1: Write the failing test**

Append to `config/store_test.go`:

```go
func TestSettingsTraceRoundTrips(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	src := `{"servers":{},"profiles":[{"id":"default","name":"default","rules":[]}],"settings":{"trace":{"enabled":true,"seconds":120,"models":["gpt"],"servers":["s1"]}}}`
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	store, err := NewStore(path)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	tr := store.GetSettings().Trace
	if !tr.Enabled || tr.Seconds != 120 || len(tr.Models) != 1 || tr.Models[0] != "gpt" || len(tr.Servers) != 1 {
		t.Fatalf("trace settings = %+v, want the file's values", tr)
	}

	if err := store.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}
	reloaded, err := NewStore(path)
	if err != nil {
		t.Fatalf("re-NewStore: %v", err)
	}
	if got := reloaded.GetSettings().Trace; !got.Enabled || got.Seconds != 120 {
		t.Errorf("after Save trace settings = %+v, want them persisted", got)
	}
}
```

Add `os` and `path/filepath` to the test file's imports if absent.

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./config/ -run TestSettingsTraceRoundTrips`
Expected: compile failure — `store.GetSettings().Trace undefined`.

- [ ] **Step 3: Wire the field**

In `domain/config.go`, inside `Settings`:

```go
	// Trace are the startup defaults for the on-demand request trace
	// (trace.Store holds the live window; it is never persisted).
	Trace TraceSettings `json:"trace,omitempty"`
```

- [ ] **Step 4: Document it in the reference config**

In `example.config.json`, the `settings` block:

```json
  "settings": {
    "always_include_stream_usage": false,
    "trace": {
      "enabled": false,
      "seconds": 300,
      "models": [],
      "servers": []
    }
  }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `go test ./config/ && python3 -I -c "import json;json.load(open('example.config.json'))"`
Expected: `ok  	llm-api-router/config`, and the JSON check prints nothing (valid).

- [ ] **Step 6: Commit**

```bash
git add domain/config.go example.config.json config/store_test.go
git commit -m "feat(config): trace startup defaults in settings"
```

---

### Task 8: Proxy tap

**Files:**
- Modify: `proxy/proxy.go` (`TraceTap`, `RouterHeaders.Trace`, `metricsWriter`, `StreamProxy`)
- Test: `proxy/proxy_test.go`

**Interfaces:**
- Consumes: `*trace.Capture` satisfies the interface structurally: `WriteResponse([]byte)`, `ResponseContentType(string)` (Task 5).
- Produces:
  - `type TraceTap interface { WriteResponse(p []byte); ResponseContentType(ct string) }`
  - `RouterHeaders.Trace TraceTap` — the per-request capture (nil = tracing off; not a client-visible header, like `RouterHeaders.Proxy`)
  - every response byte that reaches `metricsWriter.Write` is also written to the tap, and the response content type is announced at `WriteHeader`

- [ ] **Step 1: Write the failing tests**

Append to `proxy/proxy_test.go`:

```go
func TestStreamProxyTapsTraceCapture(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"))
	}))
	defer backend.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true}`))
	req.Header.Set("Content-Type", "application/json")
	req.Body = io.NopCloser(req.Body)
	w := httptest.NewRecorder()

	tr := trace.NewCapture(http.MethodPost, "/v1/chat/completions", "gpt-4", "application/json",
		[]byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true}`))
	pm, err := StreamProxy(req.Context(), backend.URL, "key", req, w, "gpt-4", "gpt-4",
		&RouterHeaders{ServerID: "s1", Trace: tr}, false, nil)
	if err != nil {
		t.Fatalf("StreamProxy: %v", err)
	}
	if pm.ResponseSize == 0 {
		t.Fatal("proxy recorded no response bytes")
	}
	e := tr.Entry()
	if !strings.Contains(e.Request, "hi") {
		t.Errorf("captured request = %q, want the prompt text", e.Request)
	}
	if !strings.Contains(e.Response, "data: {") {
		t.Errorf("captured response = %q, want the raw SSE frames", e.Response)
	}
	if e.Digest.Text != "hi" {
		t.Errorf("digest = %+v, want the reassembled stream text", e.Digest)
	}
}

func TestStreamProxyTapsBinaryResponseAsStub(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "audio/wav")
		_, _ = w.Write(bytes.Repeat([]byte{0x10}, 2048))
	}))
	defer backend.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/audio/speech", strings.NewReader(`{"model":"tts"}`))
	req.Body = io.NopCloser(req.Body)
	w := httptest.NewRecorder()
	tr := trace.NewCapture(http.MethodPost, "/v1/audio/speech", "tts", "application/json", []byte(`{"model":"tts"}`))

	if _, err := StreamProxy(req.Context(), backend.URL, "key", req, w, "tts", "tts",
		&RouterHeaders{ServerID: "s1", Trace: tr}, false, nil); err != nil {
		t.Fatalf("StreamProxy: %v", err)
	}
	e := tr.Entry()
	if !strings.HasPrefix(e.Response, "<audio ") {
		t.Errorf("captured response = %q, want an audio stub", e.Response)
	}
	if strings.Contains(e.Response, "\x10") {
		t.Error("audio bytes must not reach the trace")
	}
}

func TestStreamProxyWithoutTraceIsUnchanged(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"hi"}}]}`))
	}))
	defer backend.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	req.Body = io.NopCloser(req.Body)
	w := httptest.NewRecorder()
	pm, err := StreamProxy(req.Context(), backend.URL, "key", req, w, "gpt-4", "gpt-4", nil, false, nil)
	if err != nil {
		t.Fatalf("StreamProxy: %v", err)
	}
	if pm.ResponseSize == 0 || pm.StatusCode != http.StatusOK {
		t.Errorf("metrics = %+v, want a served request", pm)
	}
}
```

Add `llm-api-router/trace` to the test file's imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./proxy/ -run 'TestStreamProxyTaps'`
Expected: compile failure — `unknown field Trace in struct literal`.

- [ ] **Step 3: Declare the tap**

In `proxy/proxy.go`, next to `RouterHeaders`:

```go
// TraceTap is the slice of a request trace capture StreamProxy writes into: the
// response content type at header time, every response chunk after. The proxy
// deliberately knows nothing about the trace package (no import, no cycle) —
// the router hands it the capture, and nil means tracing is off so the relay
// hot path stays a single nil check.
type TraceTap interface {
	WriteResponse(p []byte)
	ResponseContentType(ct string)
}
```

and in `RouterHeaders`, after `Proxy`:

```go
	// Trace is the per-request trace capture (nil = tracing off). Not a
	// client-visible header, like Proxy.
	Trace TraceTap
```

- [ ] **Step 4: Hook the writer**

In `proxy/proxy.go`, in `metricsWriter`, add the field:

```go
	trace TraceTap // per-request trace capture (nil = off)
```

In `metricsWriter.WriteHeader`, after the `contentEncoding` line:

```go
	if m.trace != nil {
		m.trace.ResponseContentType(m.contentType)
	}
```

In `metricsWriter.Write`, immediately after `m.bodyBuffer.Write(data)`:

```go
	if m.trace != nil {
		m.trace.WriteResponse(data)
	}
```

In `StreamProxy`, where the router-mode writer is configured:

```go
	if rh != nil {
		SetRouterHeaders(rw, rh)
		mw.injectStatus = true
		mw.trace = rh.Trace
	}
```

- [ ] **Step 5: Run the proxy suite**

Run: `go test ./proxy/`
Expected: `ok  	llm-api-router/proxy` — the tap changes nothing for the 30 existing call sites (nil tap).

- [ ] **Step 6: Commit**

```bash
git add proxy/proxy.go proxy/proxy_test.go
git commit -m "feat(proxy): tap response chunks into the request trace capture"
```

---

### Task 9: Router wiring

**Files:**
- Modify: `router/router.go` (`Router.trace`, `New`, `Handle`, `record`, `reqMeta`)
- Test: `router/router_test.go`

**Interfaces:**
- Consumes: `trace.New(max, settings)`, `Store.Armed`, `Store.Add`, `Store.List`, `NewCapture`, `Capture.Entry`, `Capture.HasResponse`, `Capture.ID`, `proxy.RouterHeaders.Trace`, `domain.RequestMetric.Class/Features`.
- Produces:
  - `func New(store *config.Store, m *metrics.Store, health *config.HealthTracker, rateLimit *config.RateLimiter, quota *config.QuotaTracker, tr *trace.Store) *Router`
  - `RequestMetric.Class`/`.Features` filled on every recorded request; `features` gains `fallback` and `retry:<n>`
  - `X-Router-Trace-Id` on every traced response
  - one `trace.Entry` per traced request in `trace.Store`

- [ ] **Step 1: Write the failing tests**

First update the helper and the four positional call sites (`router_test.go:32,1026,1687,1769`, `router/jev_test.go:961`, `sdktest/harness_test.go:97`) to pass a trace store:

```go
func newTestRouter(t *testing.T) (*Router, *config.Store, *metrics.Store, *trace.Store) {
	t.Helper()
	store, err := config.NewStore("/nonexistent/path.json")
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	ms := metrics.New(100)
	ts := trace.New(100, domain.TraceSettings{})
	return New(store, ms, nil, nil, nil, ts), store, ms, ts
}
```

and append the new tests:

```go
func TestHandleTraceDisabledRecordsNothing(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}]}`))
	}))
	defer backend.Close()

	r, store, ms, ts := newTestRouter(t)
	addTestRoute(t, store, "gpt", backend.URL)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if ts.Len() != 0 {
		t.Fatalf("trace entries = %d, want 0 with the window off", ts.Len())
	}
	if w.Header().Get("X-Router-Trace-Id") != "" {
		t.Error("a request outside the window must not advertise a trace id")
	}
	if ms.Recent()[0].Class != "chat" {
		t.Errorf("metric class = %q, want chat (classification is always on)", ms.Recent()[0].Class)
	}
}

func TestHandleTraceCapturesRequestAndResponse(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"He\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"y\"},\"finish_reason\":\"stop\"}]}\n\n"))
	}))
	defer backend.Close()

	r, store, _, ts := newTestRouter(t)
	addTestRoute(t, store, "gpt", backend.URL)
	ts.Start(0, nil, nil)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt","stream":true,"messages":[{"role":"user","content":"secret prompt"}]}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer super-secret")
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Header().Get("X-Router-Trace-Id") == "" {
		t.Fatal("traced response must carry X-Router-Trace-Id")
	}
	entries := ts.List()
	if len(entries) != 1 {
		t.Fatalf("entries = %d, want 1", len(entries))
	}
	e := entries[0]
	if !strings.Contains(e.Request, "secret prompt") {
		t.Errorf("request = %q, want the prompt text", e.Request)
	}
	if strings.Contains(e.Request, "super-secret") {
		t.Error("the Authorization header is not part of the trace")
	}
	if !strings.Contains(e.Response, "data: {") {
		t.Errorf("response = %q, want the raw frames", e.Response)
	}
	if e.Digest.Text != "Hey" || e.Digest.FinishReason != "stop" {
		t.Errorf("digest = %+v, want the reassembled stream", e.Digest)
	}
	if e.StatusCode != http.StatusOK || e.ServerID == "" || e.Class != "chat" {
		t.Errorf("entry = %+v, want status/server/class filled", e)
	}
	if !slices.Contains(e.Features, "stream") {
		t.Errorf("features = %v, want stream", e.Features)
	}
}

func TestHandleTraceFiltersByModelAndServer(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{}`))
	}))
	defer backend.Close()

	r, store, _, ts := newTestRouter(t)
	addTestRoute(t, store, "gpt", backend.URL)
	srv, _ := store.GetServer("s1")

	post := func() {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt"}`))
		req.Header.Set("Content-Type", "application/json")
		r.Handle(httptest.NewRecorder(), req)
	}
	ts.Start(0, []string{"other"}, nil)
	post()
	if ts.Len() != 0 {
		t.Fatal("a model filter must exclude other models")
	}
	ts.Start(0, []string{"gpt"}, []string{"other-server"})
	post()
	if ts.Len() != 0 {
		t.Fatal("a server filter must exclude other servers")
	}
	ts.Start(0, []string{"gpt"}, []string{srv.ID})
	post()
	if ts.Len() != 1 {
		t.Fatalf("entries = %d, want 1 when both filters match", ts.Len())
	}
}

func TestHandleTraceOneEntryPerRequestWithFallback(t *testing.T) {
	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer fallback.Close()

	r, store, _, ts := newTestRouter(t)
	addTestRouteFallback(t, store, "gpt", "http://localhost:1", fallback.URL)
	ts.Start(0, nil, nil)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt"}`))
	req.Header.Set("Content-Type", "application/json")
	r.Handle(httptest.NewRecorder(), req)

	entries := ts.List()
	if len(entries) != 1 {
		t.Fatalf("entries = %d, want one per request (the capture is per-request, not per-attempt)", len(entries))
	}
	if !slices.Contains(entries[0].Features, "fallback") {
		t.Errorf("features = %v, want the fallback marker", entries[0].Features)
	}
}

func TestHandleTraceRecordsErroredAttemptBody(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "bad request body from backend", http.StatusBadRequest)
	}))
	defer backend.Close()

	r, store, _, ts := newTestRouter(t)
	addTestRoute(t, store, "gpt", backend.URL)
	ts.Start(0, nil, nil)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt"}`))
	req.Header.Set("Content-Type", "application/json")
	r.Handle(httptest.NewRecorder(), req)

	e := ts.List()[0]
	if !strings.Contains(e.Response, "bad request body from backend") {
		t.Errorf("response = %q, want the backend's 4xx body verbatim", e.Response)
	}
	if e.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", e.StatusCode)
	}
}
```

Add `slices` and `llm-api-router/trace` to the test imports, and the two fixtures (the file builds routes inline today; these tests need six of them):

```go
// addTestRoute wires one incoming model to one backend, with no retries: a
// failing backend must fall through (or answer) without a 100ms-scale backoff.
func addTestRoute(t *testing.T, store *config.Store, incoming, url string) {
	t.Helper()
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: url,
		APITypes: []domain.APIType{domain.APITypeOpenAI}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{incoming},
		ServerID: "s1", TargetModel: incoming, NumRetries: -1, Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
}

func addTestRouteFallback(t *testing.T, store *config.Store, incoming, primaryURL, fallbackURL string) {
	t.Helper()
	if err := store.AddServer(&domain.Server{ID: "primary", Name: "primary", URL: primaryURL,
		APITypes: []domain.APIType{domain.APITypeOpenAI}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddServer(&domain.Server{ID: "s1", Name: "s1", URL: fallbackURL,
		APITypes: []domain.APIType{domain.APITypeOpenAI}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{incoming},
		ServerID: "primary", TargetModel: incoming, NumRetries: -1, Enabled: true,
		Fallbacks: []domain.FallbackEntry{{ServerID: "s1"}}}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./router/ -run TestHandleTrace`
Expected: compile failure — `too many arguments in call to New` / `ts undefined`.

- [ ] **Step 3: Wire the router**

In `router/router.go`:

Struct + constructor:

```go
type Router struct {
	store     *config.Store
	metrics   *metrics.Store
	health    *config.HealthTracker
	rateLimit *config.RateLimiter
	quota     *config.QuotaTracker
	caps      *config.CapabilitiesCache
	trace     *trace.Store
	…
}

// New creates a new Router.
func New(store *config.Store, m *metrics.Store, health *config.HealthTracker, rateLimit *config.RateLimiter,
	quota *config.QuotaTracker, tr *trace.Store) *Router {
	return &Router{store: store, metrics: m, health: health, rateLimit: rateLimit, quota: quota, trace: tr,
		caps: config.NewCapabilitiesCache(), jevCache: map[string]jevCacheEntry{}, jevSticky: map[string]jevStickyEntry{}}
}
```

Request-shape carrier, next to `serverAttempt`:

```go
// reqMeta is what a request *is*, computed once at request start: the
// classification (always, 0 tokens) and the trace capture (only inside the
// window). It rides into record so the metric and the trace entry agree.
type reqMeta struct {
	class     string
	features  []string
	capture   *trace.Capture
	retries   int
}
```

Two steps, because the classification is needed on every exit path (including the security-gate block, which has no attempts yet) while the capture needs to know which server the request will hit:

1. Immediately after `body` is read (before the rule lookup), compute the shape:

```go
	// Classification is always on and costs a byte scan: metrics, the trace and
	// the GUI then describe a request with the same words.
	meta := &reqMeta{}
	meta.class, meta.features = trace.Classify(req.URL.Path, req.Header.Get("Content-Type"), body)
```

2. Immediately after the `if len(attempts) == 0 { … }` guard (so the traced server is the one the router is about to try — for a jev rule, the decision's choice), arm the capture:

```go
	// On-demand trace: the window is consulted once, here, so a request that
	// starts inside it is traced to the end even if the window expires
	// mid-generation. The capture holds the ORIGINAL body (redacted) — before
	// the stream-usage injection and before per-attempt model rewrites.
	if r.trace != nil && r.trace.Armed(model, attempts[0].server.ID) {
		meta.capture = trace.NewCapture(req.Method, req.URL.Path, model, req.Header.Get("Content-Type"), body)
		w.Header().Set("X-Router-Trace-Id", meta.capture.ID())
	}
```

The capture already classified the same body, so its entry and the metric agree by construction (both come from the one `Classify` call: `NewCapture` classifies, and `reqMeta` carries the labels for the non-traced path).

Then, in `record`, use the carrier and drop the duplicated labeler call:

```go
		APIEndpoint:           meta.class,
```

(replacing `APIEndpoint: apiEndpointFromPath(req.URL.Path)` — the label is now computed once, at request start, on the body), and at the end of `record`, before `r.metrics.Add(m)`:

```go
	m.Class, m.Features = meta.class, meta.features
	if wasFallback {
		m.Features = append(m.Features, "fallback")
	}
	if meta.retries > 0 {
		m.Features = append(m.Features, "retry:"+strconv.Itoa(meta.retries))
	}

	if meta.capture != nil {
		e := meta.capture.Entry()
		e.StatusCode = statusCode
		e.LatencyMs = m.LatencyMs
		e.TTFBMs = m.TTFBMs
		e.ServerID = serverID
		e.TargetModel = targetModel
		e.Class = m.Class
		e.Features = m.Features
		if jev != nil {
			e.JevChoice = jev.Choice
		}
		// A 5xx body that the proxy peeked and returned as an error (the
		// stream-frame 5xx path) never reached the writer chain, so the capture
		// saw nothing: fall back to what the proxy captured.
		if e.Response == "" && pm.ErrorBody != "" {
			e.Response = pm.ErrorBody
		}
		r.trace.Add(e)
	}
```

Change `record`'s signature to take the carrier, and pass `meta` at all six call sites:

```go
func (r *Router) record(req *http.Request, requestStart time.Time, model, targetModel, serverID string, pm proxy.ProxyMetrics, statusCode int, errorBody string, wasFallback bool, apiType domain.APIType, jev *jevOutcome, meta *reqMeta) {
```

In the retry loop, right after `rh.Retries = retry`, record the retry count for the feature list:

```go
			meta.retries = retry
```

and set the tap on the per-attempt headers, next to `Proxy`:

```go
		if meta.capture != nil {
			rh.Trace = meta.capture
		}
```

Every `r.record(...)` call site (the security-gate block, success, client disconnect, mid-stream, and the all-backends-failed tail) passes `meta` as the last argument. The blocked request is not traced (nothing was forwarded upstream) and carries class/features only.

`apiEndpointFromPath` now has no caller: delete it (Task 2's delegator) together with `TestAPIEndpointFromPath` in `router/router_test.go` — `TestClassifyClasses` in `trace/classify_test.go` asserts the same labels at their new home.

- [ ] **Step 4: Run the router suite**

Run: `go build ./... && go test ./router/ ./trace/ ./proxy/`
Expected: `ok` for all three. The jev router suite is part of `./router/` — every existing test must pass unchanged apart from the extra constructor argument.

- [ ] **Step 5: Commit**

```bash
git add router/
git commit -m "feat(router): classify every request and capture traced ones"
```

---

### Task 10: Admin API

**Files:**
- Modify: `admin/handler.go` (`Handler.trace`, `NewHandler`, `ServeHTTP`, four handlers)
- Modify: `main.go` (construct and inject the trace store)
- Test: `admin/handler_test.go`

**Interfaces:**
- Consumes: `trace.Store.Window/Start/Stop/List/Get/Reset/Len`, `trace.Entry`, `trace.Digest`, `trace.New`.
- Produces:
  - `GET /admin/api/trace` → `{enabled, expires_at, models, servers, entries:[Entry]}` — the compact list (bodies and `digest.text` stripped, `digest.events` kept)
  - `GET /admin/api/trace?id=<id>` → one full `Entry`, 404 when unknown
  - `POST /admin/api/trace` body `{enabled, seconds, models[], servers[]}` → the new window state
  - `POST /admin/api/trace/reset` → `{"status":"reset"}`
  - `func NewHandler(store *config.Store, m *metrics.Store, health *config.HealthTracker, tr *trace.Store) *Handler`

- [ ] **Step 1: Write the failing tests**

Update `newTestHandler` in `admin/handler_test.go`:

```go
func newTestHandler(t *testing.T) (*Handler, *config.Store, *trace.Store) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	store, err := config.NewStore(path)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	ms := metrics.New(100)
	ts := trace.New(100, domain.TraceSettings{})
	return NewHandler(store, ms, nil, ts), store, ts
}
```

and fix every existing call site to `h, _, _ := newTestHandler(t)` (and `h, store := …` variants to three values). Append:

```go
func TestTraceStartAndStop(t *testing.T) {
	h, _, ts := newTestHandler(t)

	req := httptest.NewRequest(http.MethodPost, "/admin/api/trace",
		strings.NewReader(`{"enabled":true,"seconds":0,"models":["gpt"],"servers":["s1"]}`))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("start = %d, want 200 (%s)", w.Code, w.Body.String())
	}
	if !ts.Armed("gpt", "s1") {
		t.Fatal("the window must be armed after POST")
	}

	req = httptest.NewRequest(http.MethodGet, "/admin/api/trace", nil)
	w = httptest.NewRecorder()
	h.ServeHTTP(w, req)
	var state struct {
		Enabled   bool       `json:"enabled"`
		ExpiresAt *time.Time `json:"expires_at"`
		Models    []string   `json:"models"`
	}
	if err := json.NewDecoder(w.Body).Decode(&state); err != nil {
		t.Fatalf("decode state: %v", err)
	}
	if !state.Enabled || state.ExpiresAt != nil {
		t.Errorf("state = %+v, want enabled with no expiry (seconds=0)", state)
	}
	if len(state.Models) != 1 || state.Models[0] != "gpt" {
		t.Errorf("models = %v, want [gpt]", state.Models)
	}

	req = httptest.NewRequest(http.MethodPost, "/admin/api/trace", strings.NewReader(`{"enabled":false}`))
	w = httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if ts.Armed("gpt", "s1") {
		t.Error("enabled=false must disarm the window")
	}
}

func TestTraceListIsCompactAndGetByIDIsFull(t *testing.T) {
	h, _, ts := newTestHandler(t)
	ts.Add(trace.Entry{ID: "abc123", Model: "gpt", Class: "chat",
		Request: `{"model":"gpt"}`, Response: "data: {}\n\n",
		Digest:  trace.Digest{Text: "hello", Events: map[string]int{"data": 1}}})

	w := httptest.NewRecorder()
	h.ServeHTTP(httptest.NewRequest(http.MethodGet, "/admin/api/trace", nil), w)
	var list struct {
		Entries []trace.Entry `json:"entries"`
	}
	if err := json.NewDecoder(w.Body).Decode(&list); err != nil {
		t.Fatalf("decode list: %v", err)
	}
	if len(list.Entries) != 1 {
		t.Fatalf("entries = %d, want 1", len(list.Entries))
	}
	if list.Entries[0].Request != "" || list.Entries[0].Response != "" || list.Entries[0].Digest.Text != "" {
		t.Error("the list must stay compact: bodies and the digest text are fetched by id")
	}
	if list.Entries[0].Digest.Events["data"] != 1 {
		t.Error("the event histogram is cheap and must stay in the list")
	}

	w = httptest.NewRecorder()
	h.ServeHTTP(httptest.NewRequest(http.MethodGet, "/admin/api/trace?id=abc123", nil), w)
	var full trace.Entry
	if err := json.NewDecoder(w.Body).Decode(&full); err != nil {
		t.Fatalf("decode entry: %v", err)
	}
	if full.Request == "" || full.Response == "" || full.Digest.Text != "hello" {
		t.Errorf("full entry = %+v, want the bodies and the digest", full)
	}

	w = httptest.NewRecorder()
	h.ServeHTTP(httptest.NewRequest(http.MethodGet, "/admin/api/trace?id=nope", nil), w)
	if w.Code != http.StatusNotFound {
		t.Errorf("unknown id = %d, want 404", w.Code)
	}
}

func TestTraceResetKeepsWindow(t *testing.T) {
	h, _, ts := newTestHandler(t)
	ts.Start(0, nil, nil)
	ts.Add(trace.Entry{ID: "a"})

	w := httptest.NewRecorder()
	h.ServeHTTP(httptest.NewRequest(http.MethodPost, "/admin/api/trace/reset", nil), w)
	if w.Code != http.StatusOK {
		t.Fatalf("reset = %d, want 200", w.Code)
	}
	if ts.Len() != 0 {
		t.Error("reset must clear the ring")
	}
	if !ts.Armed("m", "s") {
		t.Error("clearing the ring must not switch collection off")
	}
}
```

Add `time` and `llm-api-router/trace` to the test file's imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./admin/ -run TestTrace`
Expected: compile failure — `too many arguments in call to NewHandler`.

- [ ] **Step 3: Wire the handler**

In `admin/handler.go`: import `llm-api-router/trace`; struct + constructor:

```go
type Handler struct {
	store   *config.Store
	metrics *metrics.Store
	health  *config.HealthTracker
	trace   *trace.Store
}

// NewHandler creates a new admin handler.
func NewHandler(store *config.Store, m *metrics.Store, health *config.HealthTracker, tr *trace.Store) *Handler {
	return &Handler{store: store, metrics: m, health: health, trace: tr}
}
```

routes, after the `/metrics/reset` case:

```go
	case path == "/trace" && req.Method == http.MethodGet:
		h.getTrace(w, req)
	case path == "/trace/reset" && req.Method == http.MethodPost:
		h.resetTrace(w, req)
	case path == "/trace" && req.Method == http.MethodPost:
		h.setTrace(w, req)
```

handlers at the end of the file:

```go
// --- Trace ---

// getTrace returns the live window plus the ring. The list is compact: bodies
// and the reassembled response text are the expensive fields, so a polling
// client pays for them per entry (?id=), not per poll.
func (h *Handler) getTrace(w http.ResponseWriter, req *http.Request) {
	if id := req.URL.Query().Get("id"); id != "" {
		entry, ok := h.trace.Get(id)
		if !ok {
			writeError(w, http.StatusNotFound, "trace entry not found")
			return
		}
		writeJSON(w, http.StatusOK, entry)
		return
	}

	entries := h.trace.List()
	for i := range entries {
		entries[i].Request = ""
		entries[i].Response = ""
		entries[i].Digest.Text = ""
	}
	win := h.trace.Window()
	writeJSON(w, http.StatusOK, map[string]any{
		"enabled":    win.Enabled,
		"expires_at": win.ExpiresAt,
		"models":     win.Models,
		"servers":    win.Servers,
		"entries":    entries,
	})
}

// setTrace arms or disarms the capture window. The state lives in the trace
// store only — a live toggle is an action, not a setting — so it is never
// written to config.json and a restart never restores it.
func (h *Handler) setTrace(w http.ResponseWriter, req *http.Request) {
	var body struct {
		Enabled bool     `json:"enabled"`
		Seconds int      `json:"seconds"`
		Models  []string `json:"models"`
		Servers []string `json:"servers"`
	}
	if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	if body.Enabled {
		h.trace.Start(body.Seconds, body.Models, body.Servers)
	} else {
		h.trace.Stop()
	}
	writeJSON(w, http.StatusOK, h.trace.Window())
}

// resetTrace clears the ring; the live window keeps running (clearing the ring
// is not switching collection off, and it must not touch metrics' active count).
func (h *Handler) resetTrace(w http.ResponseWriter, req *http.Request) {
	h.trace.Reset()
	writeJSON(w, http.StatusOK, map[string]string{"status": "reset"})
}
```

- [ ] **Step 4: Run the admin suite**

Run: `go test ./admin/`
Expected: `ok  	llm-api-router/admin`

- [ ] **Step 5: Wire `main.go`**

In `main.go`, after `metricsStore := metrics.New(100)`:

```go
	// Request trace: the ring lives in memory only, and the window starts armed
	// only if settings.trace says so.
	traceStore := trace.New(100, store.GetSettings().Trace)
```

and pass it through: `router.New(store, metricsStore, healthTracker, rateLimiter, quotaTracker, traceStore)` and `admin.NewHandler(store, metricsStore, healthTracker, traceStore)`. Add `llm-api-router/trace` to the imports.

- [ ] **Step 6: Build and run everything**

Run: `go build -o /tmp/llm-api-router . && go test ./...`
Expected: build succeeds; every package `ok`.

- [ ] **Step 7: Commit**

```bash
git add admin/handler.go admin/handler_test.go main.go
git commit -m "feat(admin): request trace endpoints (window toggle, ring list, entry body)"
```

---

### Task 11: Admin GUI — Trace panel, class pill, cache-hit column

**Files:**
- Modify: `admin/static/index.html` (tabs, a new panel, metrics headers, JS)

**Interfaces:**
- Consumes: `GET /admin/api/trace`, `GET /admin/api/trace?id=`, `POST /admin/api/trace`, `POST /admin/api/trace/reset` (Task 10); `RequestMetric.class`/`.features` and the `class:*` summaries (Task 6, Task 9).
- Produces: the operator UI — a Trace tab whose label carries the live window countdown, an expandable entry table, a class pill in Recent Requests, a fourth summary table, and a `Cache hit` column.

- [ ] **Step 1: Add the tab and the panel**

In the `.tabs` div:

```html
  <div class="tab" id="trace-tab" onclick="switchTab('trace')">Trace</div>
```

After the Metrics panel's closing `</div>` (before `<!-- Server Modal -->`), and a fourth summary table inside the Metrics panel (after the `Summary by Target Model / Server` table, before `Recent Requests`) — copy that table's `<thead>` row, change every `data-table="sum-target"` to `data-table="sum-class"`, its first header to `Class`, and its `<tbody>` id to `class-metrics-body`:

```html
  <h2 style="margin-top:20px;">Summary by Request Class</h2>
  <table>
    <thead><tr><th class="sortable" data-table="sum-class" data-col="name">Class</th><th class="sortable" data-table="sum-class" data-col="total_requests">Requests</th><th class="sortable" data-table="sum-class" data-col="success_count">OK</th><th class="sortable" data-table="sum-class" data-col="error_count">Errors</th><th class="sortable" data-table="sum-class" data-col="fallback_count">Fallbacks</th><th class="sortable" data-table="sum-class" data-col="avg_latency_ms">Avg Latency</th><th class="sortable" data-table="sum-class" data-col="min_max">Min/Max</th><th class="sortable" data-table="sum-class" data-col="avg_prefill_tok_per_sec">Prefill tok/s</th><th class="sortable" data-table="sum-class" data-col="avg_decode_tok_per_sec">Decode tok/s</th><th class="sortable" data-table="sum-class" data-col="total_prompt_tokens">P Tok</th><th class="sortable" data-table="sum-class" data-col="total_completion_tokens">C Tok</th><th class="sortable" data-table="sum-class" data-col="total_cached_tokens">Cached</th><th class="sortable" data-table="sum-class" data-col="total_reasoning_tokens">Reasoning</th><th class="sortable" data-table="sum-class" data-col="total_cache_creation_tokens">Cache New</th><th class="sortable" data-table="sum-class" data-col="total_bytes">Bytes</th></tr></thead>
    <tbody id="class-metrics-body"></tbody>
  </table>
```

Then the Trace panel:

```html
<!-- Trace Panel -->
<div id="panel-trace" class="panel">
  <div class="btn-row">
    <label style="display:flex;align-items:center;gap:6px;font-size:0.85rem;">Models
      <input id="trace-models" type="text" placeholder="empty = all" style="width:170px;padding:4px 8px;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;border-radius:4px;font-size:0.8rem;">
    </label>
    <label style="display:flex;align-items:center;gap:6px;font-size:0.85rem;">Servers
      <input id="trace-servers" type="text" placeholder="empty = all" style="width:130px;padding:4px 8px;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;border-radius:4px;font-size:0.8rem;">
    </label>
    <label style="display:flex;align-items:center;gap:6px;font-size:0.85rem;">Seconds
      <input id="trace-seconds" type="number" min="0" step="30" value="300" title="0 = until switched off" style="width:70px;padding:4px 8px;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;border-radius:4px;font-size:0.8rem;">
    </label>
    <button class="primary" onclick="startTrace()">Capture</button>
    <button onclick="stopTrace()">Stop</button>
    <button class="danger" onclick="clearTrace()">Clear</button>
    <button onclick="loadTrace()">↻ Refresh</button>
    <span id="trace-state" style="font-size:0.8rem;color:#484f58;margin-left:auto;"></span>
  </div>
  <table>
    <thead><tr><th>Time</th><th>Class</th><th>Model</th><th>Target</th><th>Server</th><th>Status</th><th>Latency</th><th>TTFB</th><th>Events</th><th>Features</th><th>Trunc</th><th>Jev</th><th>ID</th></tr></thead>
    <tbody id="trace-body"></tbody>
  </table>
  <p style="font-size:0.75rem;color:#484f58;">Bodies live in memory only (512KB per request/response, last 100 entries) and die with the process. Base64 images and audio uploads are stored as size+hash stubs; prompt text is stored verbatim. Click a row for the request, the reassembled response and the raw frames.</p>
</div>
```

- [ ] **Step 2: Add the trace JS**

Insert before the closing `</script>`:

```js
// --- Trace ---
let traceInterval = null;
let traceWin = null;

// traceState renders the live window: in the panel, and as a countdown on the
// tab label so the operator sees "capture is on" from any other tab.
function traceState() {
  const on = !!traceWin && traceWin.enabled;
  const state = document.getElementById('trace-state');
  const tab = document.getElementById('trace-tab');
  if (!on) {
    if (state) state.textContent = 'off — nothing is captured';
    if (tab) tab.textContent = 'Trace';
    return;
  }
  const scope = [];
  if ((traceWin.models || []).length) scope.push('models: ' + traceWin.models.join(','));
  if ((traceWin.servers || []).length) scope.push('servers: ' + traceWin.servers.join(','));
  let left = 'until stopped';
  if (traceWin.expires_at) {
    const s = Math.max(0, Math.round((new Date(traceWin.expires_at).getTime() - Date.now()) / 1000));
    left = Math.floor(s / 60) + 'm' + String(s % 60).padStart(2, '0') + 's left';
  }
  if (state) state.textContent = 'capturing, ' + left + (scope.length ? ' · ' + scope.join(' · ') : ' · everything');
  if (tab) tab.textContent = left === 'until stopped' ? 'Trace ●' : 'Trace ● ' + left;
}

async function loadTrace() {
  const res = await fetch(API + '/trace');
  if (!res.ok) return;
  traceWin = await res.json();
  traceState();

  const body = document.getElementById('trace-body');
  const openIds = new Set([...body.querySelectorAll('.trace-detail[data-open="1"]')].map(d => d.dataset.id));
  const entries = (traceWin.entries || []).slice().reverse();
  if (!entries.length) {
    body.innerHTML = '<tr><td colspan="13" class="empty">Nothing traced yet</td></tr>';
    return;
  }
  body.innerHTML = entries.map(e => {
    const time = new Date(e.timestamp).toLocaleTimeString([], { hour12: false });
    const statusColor = e.status_code >= 200 && e.status_code < 300 ? '#3fb950' : '#f85149';
    const events = Object.entries((e.digest && e.digest.events) || {}).map(([k, v]) => k + ':' + v).join(' ');
    const feats = (e.features || []).join(' ');
    const trunc = [e.request_truncated ? 'req' : '', e.response_truncated ? 'resp' : ''].filter(Boolean).join('+');
    return `<tr style="cursor:pointer;" onclick="toggleTraceEntry(this,'${escAttr(e.id)}')">
      <td>${time}</td>
      <td><span class="tag" style="background:#1a3a3a;color:#39c5cf">${esc(e.class || '-')}</span></td>
      <td>${esc(e.model)}</td>
      <td>${esc(e.target_model || '-')}</td>
      <td>${esc(e.server_id || '-')}</td>
      <td style="color:${statusColor}">${e.status_code}</td>
      <td>${e.latency_ms}ms</td>
      <td>${e.ttfb_ms}ms</td>
      <td style="font-size:0.75rem;color:#8b949e" title="${escAttr(events)}">${esc(events || '-')}</td>
      <td style="font-size:0.75rem;color:#8b949e" title="${escAttr(feats)}">${esc(feats || '-')}</td>
      <td>${esc(trunc || '-')}</td>
      <td>${esc(e.jev_choice || '-')}</td>
      <td style="font-family:monospace;font-size:0.8rem;color:#8b949e;">${esc(e.id)}</td>
    </tr>
    <tr class="trace-detail" data-id="${escAttr(e.id)}" style="display:none;"><td colspan="13" class="empty">Loading…</td></tr>`;
  }).join('');
  for (const id of openIds) {
    const detail = body.querySelector('.trace-detail[data-id="' + CSS.escape(id) + '"]');
    if (detail) toggleTraceEntry(detail.previousElementSibling, id);
  }
}

// toggleTraceEntry expands (fetching the full entry once) or collapses a row:
// the compact list carries no bodies, so the body arrives on first expansion.
async function toggleTraceEntry(row, id) {
  const detail = row.nextElementSibling;
  if (!detail || !detail.classList.contains('trace-detail')) return;
  if (detail.dataset.open === '1') {
    detail.style.display = 'none';
    detail.dataset.open = '0';
    return;
  }
  detail.style.display = '';
  detail.dataset.open = '1';
  if (detail.dataset.loaded === '1') return;
  const cell = detail.cells[0];
  try {
    const res = await fetch(API + '/trace?id=' + encodeURIComponent(id));
    if (!res.ok) { cell.textContent = 'Trace entry not found (overwritten by the ring?)'; return; }
    const e = await res.json();
    detail.dataset.loaded = '1';
    let req = e.request || '(empty)';
    try { req = JSON.stringify(JSON.parse(req), null, 2); } catch {}
    const d = e.digest || {};
    const events = Object.entries(d.events || {}).map(([k, v]) =>
      `<span class="tag" style="background:#1a3a3a;color:#39c5cf">${esc(k)}:${v}</span>`).join('');
    const pre = 'white-space:pre-wrap;word-break:break-all;max-height:260px;overflow:auto;font-size:0.75rem;background:#0d1117;padding:8px;border-radius:4px;margin:4px 0 12px;';
    const title = 'font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:#58a6ff;';
    cell.style.textAlign = 'left';
    cell.innerHTML =
      `<div style="${title}">Request${e.request_truncated ? ' (truncated)' : ''}</div><pre style="${pre}">${esc(req)}</pre>` +
      `<div style="${title}">Response${e.response_truncated ? ' (truncated)' : ''} · finish ${esc(d.finish_reason || 'n/a')}</div>` +
      `<div style="margin:4px 0;">${events}</div>` +
      `<pre style="${pre}">${esc(d.text || '(no text — binary, audio, or a non-text response)')}</pre>` +
      `<details><summary style="cursor:pointer;color:#8b949e;font-size:0.8rem;">Raw frames (${(e.response || '').length} bytes stored)</summary><pre style="${pre}">${esc(e.response || '(empty)')}</pre></details>`;
  } catch (err) {
    cell.textContent = 'failed to load trace entry: ' + err;
  }
}

async function startTrace() {
  const list = (id) => document.getElementById(id).value.split(',').map(s => s.trim()).filter(Boolean);
  const res = await fetch(API + '/trace', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      enabled: true,
      seconds: parseInt(document.getElementById('trace-seconds').value, 10) || 0,
      models: list('trace-models'),
      servers: list('trace-servers'),
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    alert('Trace start failed: ' + (err.error || res.statusText));
    return;
  }
  await loadTrace();
}

async function stopTrace() {
  await fetch(API + '/trace', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled: false }),
  });
  await loadTrace();
}

async function clearTrace() {
  if (!confirm('Clear traced requests?')) return;
  await fetch(API + '/trace/reset', { method: 'POST' });
  await loadTrace();
}
```

- [ ] **Step 3: Auto-refresh wiring**

In `switchTab`, alongside the existing interval resets:

```js
  if (traceInterval) { clearInterval(traceInterval); traceInterval = null; }
```

and after the `servers` branch:

```js
  if (name === 'trace') {
    loadTrace();
    traceInterval = setInterval(loadTrace, 3000);
  }
```

- [ ] **Step 4: The fourth summary table**

In `loadMetrics`, next to the other three `renderSummaries` calls:

```js
  renderSummaries('class-metrics-body', 'sum-class', entries.filter(([key]) => key.startsWith('class:')),
    (key) => key.slice(6));
```

In the sortable-header click handler, the `table.startsWith('sum-')` branch already re-renders every summary table, so nothing changes there; in `restoreSortState`'s `tables` map add `'sum-class': summaryCols`.

- [ ] **Step 5: Cache hit by token sum**

In each of the four summary `<thead>` rows (including the new class table), immediately after the `Cached` header, add the sortable column with that table's own `data-table` value:

```html
<th class="sortable" data-table="sum-server" data-col="cache_hit">Cache hit</th>
```

Next to `summaryRow`, add the helper (an average of per-request ratios would let one small request outvote a large one, so the ratio is taken from the group's token sums):

```js
function cacheHitRatio(s) {
  const denom = (s.total_prompt_tokens || 0) + (s.total_cached_tokens || 0) + (s.total_cache_creation_tokens || 0);
  return denom > 0 ? 100 * (s.total_cached_tokens || 0) / denom : 0;
}
```

In `summaryRow`, compute and render it (an explicit label beats a dash: `0 cached` is a miss, and no usage at all is not a zero):

```js
  const cacheHitCell = (s.total_prompt_tokens || s.total_cached_tokens || s.total_cache_creation_tokens)
    ? (cacheHitRatio(s) > 0 ? cacheHitRatio(s).toFixed(0) + '%' : 'cache miss')
    : 'usage not reported';
```

and add `<td>${cacheHitCell}</td>` right after the `cachedTok` cell. Bump the empty-row colspan in `renderSummaries` from 15 to 16.

In `sortData`, next to the `min_max` branch:

```js
    else if (col === 'cache_hit') { va = cacheHitRatio(a); vb = cacheHitRatio(b); }
```

and add `'cache_hit'` to `summaryCols` in `restoreSortState`.

- [ ] **Step 6: Class pill and “why is there no number” in Recent Requests**

In `loadMetrics`'s recent-request row, add next to the other tags:

```js
    const klass = r.class
      ? `<br><span class="tag" style="background:#1a3a3a;color:#39c5cf" title="${escAttr((r.features || []).join(', '))}">${esc(r.class)}</span>`
      : '';
```

replace `<td>${esc(r.model)}</td>` with `<td>${esc(r.model)}${klass}</td>`, and replace the `cached` cell value:

```js
    const cached = (r.prompt_tokens || r.completion_tokens) ? (r.cached_tokens >= 0 ? r.cached_tokens : '-') : 'not reported';
```

The Recent Requests header row's `Cached` label stays as-is; the empty-row colspan (20) is unchanged.

- [ ] **Step 7: Build, run, check the panel**

Run:

```bash
task build
gofmt -l .
./llm-api-router &
sleep 1
curl -s -X POST localhost:8080/admin/api/trace -d '{"enabled":true,"seconds":60}'
curl -s localhost:8080/admin/api/trace | head -c 200
curl -s localhost:8080/admin/ | grep -c 'id="panel-trace"'
kill %1
```

Expected: the POST answers with `enabled: true` and an `expires_at` about a minute out, the GET echoes the window, and the `grep -c` prints `1`. No entries appear with no backend configured — the traced-entry path is pinned by the Go tests.

- [ ] **Step 8: Commit**

```bash
git add admin/static/index.html
git commit -m "feat(admin): trace panel, request-class summaries, token-sum cache-hit column"
```

---

### Task 12: Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: the shipped behaviour of Tasks 1-11.
- Produces: the next session's map of the feature (both files are the standing brief for engineers working here).

- [ ] **Step 1: CLAUDE.md**

Add two bullets to **Critical Behavior Notes** (normal prose — the file is documentation, not chat):

```markdown
- **On-demand request trace (`trace`)**: a second, opt-in plane of observability next to the aggregates — full request/response bodies for a bounded window, for debugging a client against a backend. Armed at startup by `Settings.Trace` (`settings.trace: {enabled, seconds, models, servers}`) and switched live by `POST /admin/api/trace`; the live state lives in `trace.Store` and is **never** persisted, so a restart can not leave prompt capture on. `seconds: 0` = until switched off. The decision is made once in `router.Handle` (window active AND the models filter empty-or-matches AND the first attempt's server matches the servers filter), so a request started inside the window is traced to the end even if the window expires mid-generation. Capture taps the existing writer chain (`metricsWriter` already sees every response byte), armed by `RouterHeaders.Trace` — no second RoundTripper, no re-read of the response, and a nil tap is one branch on the hot path. SSE frames are stored raw (what arrived, in what pieces) and `trace.Digest` reassembles them for display: OpenAI chat `choices[].delta.content`/`reasoning_content`, Responses `response.output_text.delta`, Anthropic `content_block_delta` (text/thinking) with `message_delta`'s `stop_reason`, audio.cpp `transcript.text.delta` (`speech.audio.delta` is counted, never stored) — plus an event-name histogram and the finish reason. Binaries become `<wav 4.3MB sha256:ab12…>` stubs (`audio/wav`, raw PCM), and base64 inside a request body is redacted the same way (`trace.RedactRequest`: data URLs, Anthropic `source.data`, multipart file parts — byte scan, the form is never parsed). Prompt text is NOT redacted (a deliberate user decision); `Authorization` is never part of the trace. Caps: 512KB per body (prefix + `…[truncated N bytes]` + flag), ring of 100 (`trace.New(100, …)` in `main.go`). A 5xx whose SSE body the proxy peeked and returned (`StreamErrorResponse`) never reaches the writer chain, so `router.record` fills the entry from `ProxyMetrics.ErrorBody`. `X-Router-Trace-Id` is written only when the entry was taken. Covered by `TestClassify*`, `TestRedactRequest*`, `TestDigest*`, `TestStore*`, `TestCapture*`, `TestStreamProxyTaps*`, `TestHandleTrace*`, `TestTrace*`.
- **Request classification (always on, 0 tokens)**: `trace.Classify(path, contentType, body)` labels every request (`chat`/`messages`/`responses`/`embeddings`/`completions`/`speech`/`transcriptions`/`voices`/`tts`/… — the vocabulary is `trace.EndpointLabel`, which `router.apiEndpointFromPath` delegates to) and lists the signs a byte scan sees: `stream`, `tools:<n>`, `tool_result`, `images:<n>`, `system:<chars>`, `reasoning`, `audio-upload`, `json:<bytes>`, plus `fallback`/`retry:<n>` from the router. It rides into `RequestMetric.Class`/`.Features` and aggregates into a fourth summary key family `class:<label>`, rendered as "Summary by Request Class" and as a pill in Recent Requests. `class:*` keys reach `/metrics` as `group="class:chat"`, the same shape `target:*` keys already use.
```

Add to **Key URLs**:

```markdown
| `/admin/api/trace` | GET/POST | Trace window + compact entry list / arm or disarm the window (`{enabled, seconds, models[], servers[]}`) |
| `/admin/api/trace?id=` | GET | One traced request: redacted `request`, raw `response`, `digest{text,events,finish_reason}`, truncation flags |
| `/admin/api/trace/reset` | POST | Clear the trace ring (the live window keeps running) |
```

Add to the **Packages** table:

```markdown
| `trace` | `trace/store.go`, `trace/capture.go`, `trace/classify.go`, `trace/redact.go`, `trace/digest.go`, `trace/entry.go` | Request trace: ring buffer + live capture window, per-request capture (the proxy's write tap), binary redaction, request classification, stream digest |
```

Add a bullet to the **Admin GUI** section:

```markdown
- Trace tab: `Capture`/`Stop`/`Clear` with model+server filters and a countdown in the tab label, so the operator sees that capture is on from any tab. The compact entry table expands per row into **Request** (pretty JSON), **Response** (reassembled text + event histogram + finish reason) and **Raw** (the stored frames) — fetched by `?id=` on first expansion, because the list deliberately carries no bodies. Metrics has a fourth table (Summary by Request Class), a `Cache hit` column computed from each group's token sums (explicit `cache miss`, `usage not reported` when no usage), and a class pill on each Recent Requests row (features in the tooltip)
```

- [ ] **Step 2: AGENTS.md**

Mirror the same notes in the file's condensed voice (the wording differs from CLAUDE.md by design — match its density, keep every identifier exact):

```markdown
- **Request trace (`trace`)**: opt-in full-text capture of request/response bodies for a bounded window, for debugging. Startup defaults in `Settings.Trace` (`settings.trace`), switched live by `POST /admin/api/trace {enabled,seconds,models,servers}`; `GET /admin/api/trace[?id=]`, `POST /admin/api/trace/reset`. Live window state lives in `trace.Store` and is never persisted (a restart can not leave prompt capture on). Decided once at request start; capture taps `metricsWriter` via `RouterHeaders.Trace` (nil = hot path untouched, no second RoundTripper); SSE stored raw and `trace.Digest` reassembles deltas + counts event names; binaries and in-body base64 → `<kind SIZE sha256:…>` stubs (`trace.RedactRequest`, byte scan), prompt text untouched, `Authorization` never stored; 512KB per body, ring of 100. `X-Router-Trace-Id` only when traced.
- **Request classification**: `trace.Classify` labels every request (`chat`/`messages`/`responses`/`speech`/… — vocabulary in `trace.EndpointLabel`, which `router.apiEndpointFromPath` delegates to) plus scanned features (`stream`, `tools:<n>`, `tool_result`, `images:<n>`, `system:<chars>`, `reasoning`, `audio-upload`, `json:<bytes>`, and `fallback`/`retry:<n>` from the router) into `RequestMetric.Class`/`.Features` and a fourth summary key family `class:<label>`.
```

Add `trace` to AGENTS.md's package table in the same shape as the other packages, and the three `/admin/api/trace*` routes to its endpoint list (if the condensed file carries one; otherwise the two bullets above are enough).

- [ ] **Step 3: Verify and commit**

Run: `grep -c "admin/api/trace" CLAUDE.md AGENTS.md`
Expected: a non-zero count for both files.

```bash
git add CLAUDE.md AGENTS.md
git commit -m "docs: request trace and request classification"
```

---

## Final Check

```bash
task fmt
task lint                        # expect: 0 issues
task test                        # expect: every package ok
cd sdktest && go test ./...      # expect: ok (router.New gained the trace-store argument)
```

Trace ships off, so Tasks 8-11 must leave every existing test untouched apart from constructor arguments: a nil tap and an unarmed window are the paths the whole existing suite walks.
