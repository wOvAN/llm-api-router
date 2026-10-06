package router

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"

	"llm-api-router/config"
	"llm-api-router/domain"
	"llm-api-router/metrics"
)

// jevPool is the pool used by the policy tests: four tiers, cheapest first.
func jevPool() []domain.JevCandidate {
	return []domain.JevCandidate{
		{ServerID: "s0", TargetModel: "haiku", Tier: 0},
		{ServerID: "s1", TargetModel: "sonnet", Tier: 1},
		{ServerID: "s2", TargetModel: "opus", Tier: 2},
		{ServerID: "s3", TargetModel: "fable", Tier: 3},
	}
}

// jevDecision serves a canned System One answer and counts the calls.
func jevDecision(t *testing.T, answer string, calls *int) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*calls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(answer))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func jevRule(decisionURL string, minConf float64, defaultTier int) *domain.RoutingRule {
	return &domain.RoutingRule{
		IncomingModels: []string{"auto"},
		Router:         domain.RouterJev,
		Enabled:        true,
		Jev: &domain.JevRouter{
			ServerID:      "decision",
			Model:         "laya",
			Candidates:    jevPool(),
			MinConfidence: minConf,
			DefaultTier:   defaultTier,
		},
	}
}

func TestJevAttempts(t *testing.T) {
	const answer = `{"model":"laya","answers":{"model":{"type":"choice","choice":"%s","confidence":%g}},` +
		`"usage":{"input_tokens":12,"output_tokens":1}}`

	tests := []struct {
		name        string
		choice      string
		confidence  float64
		defaultTier int
		headerOff   bool
		wantOrder   []string
		wantReason  string
		wantCall    bool
	}{
		{
			name: "choice honored, rest of pool steps up first", choice: "opus", confidence: 0.9, defaultTier: 1,
			wantOrder: []string{"s2", "s3", "s1", "s0"}, wantReason: "jev", wantCall: true,
		},
		{
			name: "cheapest choice keeps the whole pool behind it", choice: "haiku", confidence: 0.9, defaultTier: 1,
			wantOrder: []string{"s0", "s1", "s2", "s3"}, wantReason: "jev", wantCall: true,
		},
		{
			name: "low confidence blocks a downgrade", choice: "haiku", confidence: 0.1, defaultTier: 1,
			wantOrder: []string{"s1", "s2", "s3", "s0"}, wantReason: "low-confidence-no-downgrade", wantCall: true,
		},
		{
			name: "low confidence caps an upgrade", choice: "fable", confidence: 0.1, defaultTier: 1,
			wantOrder: []string{"s2", "s3", "s1", "s0"}, wantReason: "low-confidence-capped", wantCall: true,
		},
		{
			name: "low confidence keeps a one-tier upgrade", choice: "opus", confidence: 0.1, defaultTier: 1,
			wantOrder: []string{"s2", "s3", "s1", "s0"}, wantReason: "jev", wantCall: true,
		},
		{
			name: "unknown choice falls to the default tier", choice: "gpt-4", confidence: 0.9, defaultTier: 2,
			wantOrder: []string{"s2", "s3", "s1", "s0"}, wantReason: "unknown-choice", wantCall: true,
		},
		{
			name: "default tier missing uses the nearest above", choice: "gpt-4", confidence: 0.9, defaultTier: 9,
			wantOrder: []string{"s3", "s2", "s1", "s0"}, wantReason: "unknown-choice", wantCall: true,
		},
		{
			name: "header bypasses the decision", headerOff: true, defaultTier: 1,
			wantOrder: []string{"s1", "s2", "s3", "s0"}, wantReason: "off", wantCall: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, store, _ := jevTestRouter(t)
			calls := 0
			decision := jevDecision(t, fmt.Sprintf(answer, tt.choice, tt.confidence), &calls)
			addJevServers(t, store, decision.URL)

			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
				strings.NewReader(`{"model":"auto","messages":[{"role":"user","content":"hi"}]}`))
			if tt.headerOff {
				req.Header.Set("X-Router-Jev", "off")
			}
			attempts, out := r.jevAttempts(req, jevRule(decision.URL, 0.3, tt.defaultTier),
				[]byte(`{"model":"auto","messages":[{"role":"user","content":"hi"}]}`), "auto")

			var order []string
			for _, a := range attempts {
				order = append(order, a.server.ID)
			}
			if strings.Join(order, ",") != strings.Join(tt.wantOrder, ",") {
				t.Errorf("attempts = %v, want %v", order, tt.wantOrder)
			}
			if out.Reason != tt.wantReason {
				t.Errorf("reason = %q, want %q", out.Reason, tt.wantReason)
			}
			if calls != boolInt(tt.wantCall) {
				t.Errorf("decision calls = %d, want %d", calls, boolInt(tt.wantCall))
			}
		})
	}
}

func TestJevAttemptsDecisionUnavailable(t *testing.T) {
	r, store, _ := jevTestRouter(t)
	calls := 0
	decision := jevDecision(t, `{}`, &calls)
	addJevServers(t, store, decision.URL)
	decision.Close() // endpoint dead → no usable answer

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"auto","messages":[{"role":"user","content":"hello"}]}`))
	attempts, out := r.jevAttempts(req, jevRule(decision.URL, 0.3, 1), []byte(`{"model":"auto","messages":[{"role":"user","content":"hello"}]}`), "auto")

	if len(attempts) != 4 || attempts[0].server.ID != "s1" {
		t.Errorf("attempts = %v, want the pool with the default tier first", attemptIDs(attempts))
	}
	if out.Reason != "unavailable" {
		t.Errorf("reason = %q, want unavailable", out.Reason)
	}
}

func TestJevAttemptsEmptyPool(t *testing.T) {
	r, _, _ := jevTestRouter(t)
	rule := jevRule("http://127.0.0.1:1", 0.3, 0)
	for i := range rule.Jev.Candidates {
		rule.Jev.Candidates[i].Enabled = new(false)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"auto","messages":[{"role":"user","content":"hello"}]}`))
	attempts, out := r.jevAttempts(req, rule, []byte(`{"model":"auto","messages":[{"role":"user","content":"hello"}]}`), "auto")
	if len(attempts) != 0 || out.Reason != "no-pool" {
		t.Errorf("attempts = %v reason = %q, want empty pool", attemptIDs(attempts), out.Reason)
	}
}

func TestHandleJev(t *testing.T) {
	var decisionBody map[string]any
	decision := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &decisionBody)
		_, _ = w.Write([]byte(`{"model":"laya","answers":{"model":{"type":"choice","choice":"sonnet","confidence":0.8}},` +
			`"usage":{"input_tokens":12,"output_tokens":1}}`))
	}))
	defer decision.Close()

	var hit string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"sonnet","choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer backend.Close()

	r, store, ms := jevTestRouter(t)
	_ = store.AddServer(&domain.Server{ID: "decision", Name: "decision", URL: decision.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI}})
	_ = store.AddServer(&domain.Server{ID: "cheap", Name: "cheap", URL: backend.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI}})
	_ = store.AddServer(&domain.Server{ID: "strong", Name: "strong", URL: backend.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI}})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"auto"},
		Router:         domain.RouterJev,
		Enabled:        true,
		Jev: &domain.JevRouter{
			ServerID:    "decision",
			Model:       "laya",
			DefaultTier: 1,
			Candidates: []domain.JevCandidate{
				{ServerID: "cheap", TargetModel: "haiku", Tier: 0},
				{ServerID: "strong", TargetModel: "sonnet", Tier: 1},
			},
		},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"auto","messages":[{"role":"user","content":"<system-reminder>noise</system-reminder>refactor the parser"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("status = %d: %s", w.Result().StatusCode, w.Body.String())
	}
	if hit == "" {
		t.Fatal("backend never reached")
	}
	// The chosen candidate's target model is exposed, not rewritten back.
	if body := w.Body.String(); !strings.Contains(body, `"model":"sonnet"`) {
		t.Errorf("response model not the chosen target: %s", body)
	}
	res := w.Result()
	if got := res.Header.Get("X-Router-Jev-Choice"); got != "sonnet" {
		t.Errorf("X-Router-Jev-Choice = %q", got)
	}
	if got := res.Header.Get("X-Router-Jev-Reason"); got != "jev" {
		t.Errorf("X-Router-Jev-Reason = %q", got)
	}

	// The decision model saw the user turn (reminders stripped) and the pool.
	if decisionBody["model"] != "laya" {
		t.Errorf("decision body model = %v", decisionBody["model"])
	}
	state, _ := decisionBody["state"].(map[string]any)
	if state["request"] != "refactor the parser" {
		t.Errorf("decision state.request = %v", state["request"])
	}
	env, _ := state["environment"].(map[string]any)
	if got, _ := env["available_models"].([]any); len(got) != 2 || got[0] != "haiku" {
		t.Errorf("decision pool = %v", env["available_models"])
	}

	recent := ms.Recent()
	if len(recent) != 1 {
		t.Fatalf("recorded %d requests, want 1", len(recent))
	}
	m := recent[0]
	if m.JevChoice != "sonnet" || m.JevReason != "jev" || m.JevModel != "laya" || m.JevTokens != 13 {
		t.Errorf("metric jev fields = %+v", m)
	}
	if m.JevLatencyMs < 0 || m.ServerID != "strong" || m.TargetModel != "sonnet" {
		t.Errorf("metric routing = %+v", m)
	}
}

func TestJevState(t *testing.T) {
	body := []byte(`{"model":"auto","messages":[` +
		`{"role":"system","content":"you are a bot"},` +
		`{"role":"user","content":[{"type":"text","text":"first"},{"type":"text","text":"turn"}]},` +
		`{"role":"user","content":"<system-reminder>injected</system-reminder>second turn"}]}`)
	text, tokens := jevRequestText(body)
	state := jevState(text, tokens, "auto", []string{"haiku", "opus"})

	if got := state["request"]; got != "second turn" {
		t.Errorf("request = %v, want the last user turn", got)
	}
	session, _ := state["session"].(map[string]any)
	if session["current_model"] != "auto" {
		t.Errorf("current_model = %v", session["current_model"])
	}
	if tokens <= 0 {
		t.Errorf("context_tokens = %v, want a positive estimate", tokens)
	}
	env, _ := state["environment"].(map[string]any)
	if got, _ := env["available_models"].([]string); len(got) != 2 {
		t.Errorf("available_models = %v", env["available_models"])
	}

	// Responses API: "input" is used when there are no messages.
	text, tokens = jevRequestText([]byte(`{"model":"auto","input":"summarise this"}`))
	state = jevState(text, tokens, "auto", nil)
	if got := state["request"]; got != "summarise this" {
		t.Errorf("responses request = %v", got)
	}

	// Truncation never splits a UTF-8 rune.
	text, _ = jevRequestText([]byte(`{"messages":[{"role":"user","content":"éééé"}]}`))
	if text = jevTruncate(text, 5); !utf8.ValidString(text) {
		t.Errorf("truncated request is not valid UTF-8: %q", text)
	}

	// A tool_result-only last turn (agent loop) is an auxiliary call: no text,
	// never the raw messages JSON.
	text, tokens = jevRequestText([]byte(`{"model":"auto","messages":[` +
		`{"role":"user","content":"real question"},` +
		`{"role":"assistant","content":"working"},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"` + strings.Repeat("x", 4000) + `"}]}]}`))
	if text != "" {
		t.Errorf("tool_result-only turn request = %q..., want empty", text[:min(60, len(text))])
	}
	if tokens <= 0 {
		t.Errorf("context_tokens = %v, want a positive estimate", tokens)
	}
}

func TestJevAttemptsSecurity(t *testing.T) {
	tests := []struct {
		name       string
		noul       string
		wantReason string
		wantCall   int
	}{
		{name: "harm intent blocks", noul: "0.97", wantReason: "blocked", wantCall: 1},
		{name: "benign request passes", noul: "0.02", wantReason: "jev", wantCall: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, store, _ := jevTestRouter(t)
			calls := 0
			decision := jevDecision(t, `{"answers":{"model":{"type":"choice","choice":"sonnet","confidence":0.9},`+
				`"security":{"type":"noul","noul":`+tt.noul+`}}}`, &calls)
			addJevServers(t, store, decision.URL)
			rule := jevRule(decision.URL, 0.3, 1)
			rule.Jev.Security = true

			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"auto","messages":[{"role":"user","content":"hello"}]}`))
			attempts, out := r.jevAttempts(req, rule, []byte(`{"model":"auto","messages":[{"role":"user","content":"hello"}]}`), "auto")
			if out.Reason != tt.wantReason {
				t.Errorf("reason = %q, want %q", out.Reason, tt.wantReason)
			}
			if tt.wantReason == "blocked" && attempts != nil {
				t.Errorf("blocked decision produced attempts: %v", attemptIDs(attempts))
			}
			if tt.wantReason == "jev" && len(attempts) != 4 {
				t.Errorf("attempts = %v, want the full pool", attemptIDs(attempts))
			}
		})
	}
}

func TestHandleJevSecurityBlock(t *testing.T) {
	decision := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"answers":{"model":{"type":"choice","choice":"sonnet","confidence":0.9},` +
			`"security":{"type":"noul","noul":0.98}}}`))
	}))
	defer decision.Close()
	hit := false
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { hit = true }))
	defer backend.Close()

	r, store, ms := jevTestRouter(t)
	_ = store.AddServer(&domain.Server{ID: "decision", Name: "decision", URL: decision.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI}})
	_ = store.AddServer(&domain.Server{ID: "strong", Name: "strong", URL: backend.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI}})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"auto"}, Router: domain.RouterJev, Enabled: true,
		Jev: &domain.JevRouter{ServerID: "decision", Security: true, DefaultTier: 0,
			Candidates: []domain.JevCandidate{{ServerID: "strong", TargetModel: "sonnet", Tier: 0}}},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"auto","messages":[{"role":"user","content":"hello"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Result().StatusCode)
	}
	if hit {
		t.Error("blocked request reached the backend")
	}
	recent := ms.Recent()
	if len(recent) != 1 || recent[0].JevReason != "blocked" || recent[0].StatusCode != 400 {
		t.Errorf("metric = %+v, want one blocked 400", recent)
	}
}

func TestJevAttemptsScores(t *testing.T) {
	r, store, _ := jevTestRouter(t)
	calls := 0
	decision := jevDecision(t, `{"answers":{"model":{"type":"choice","choice":"opus","confidence":0.9},`+
		`"task_complexity":{"type":"score","score":5},"reasoning_required":{"type":"score","score":7},`+
		`"tool_complexity":{"type":"score","score":2}}}`, &calls)
	addJevServers(t, store, decision.URL)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"auto","messages":[{"role":"user","content":"hello"}]}`))
	_, out := r.jevAttempts(req, jevRule(decision.URL, 0.3, 1), []byte(`{"model":"auto","messages":[{"role":"user","content":"hello"}]}`), "auto")
	if out.Scores != "task 5, reasoning 7, tools 2" {
		t.Errorf("scores = %q", out.Scores)
	}
}

func TestJevAttemptsCache(t *testing.T) {
	r, store, _ := jevTestRouter(t)
	calls := 0
	decision := jevDecision(t, `{"model":"laya","answers":{"model":{"type":"choice","choice":"opus","confidence":0.9}},`+
		`"usage":{"input_tokens":12,"output_tokens":1}}`, &calls)
	addJevServers(t, store, decision.URL)
	rule := jevRule(decision.URL, 0.3, 1)
	rule.Jev.CacheTTL = 60

	body := []byte(`{"model":"auto","messages":[{"role":"user","content":"same text"}]}`)
	for i := 0; i < 2; i++ {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(body)))
		attempts, out := r.jevAttempts(req, rule, body, "auto")
		if len(attempts) != 4 || attempts[0].server.ID != "s2" {
			t.Fatalf("run %d: attempts = %v", i, attemptIDs(attempts))
		}
		if i == 1 && out.Reason != "cache" {
			t.Errorf("run %d: reason = %q, want cache", i, out.Reason)
		}
	}
	if calls != 1 {
		t.Errorf("decision calls = %d, want 1 (cache hit)", calls)
	}

	// A different request text is a different decision.
	other := []byte(`{"model":"auto","messages":[{"role":"user","content":"other text"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(other)))
	r.jevAttempts(req, rule, other, "auto")
	if calls != 2 {
		t.Errorf("decision calls = %d, want 2 (new text misses the cache)", calls)
	}
}

func TestJevAttemptsContextNoDowngrade(t *testing.T) {
	r, store, _ := jevTestRouter(t)
	calls := 0
	decision := jevDecision(t, `{"answers":{"model":{"type":"choice","choice":"haiku","confidence":0.9}}}`, &calls)
	addJevServers(t, store, decision.URL)

	// ~22k estimated context tokens (> the 20k cache-rebuild bound) blocks the
	// downgrade from the default tier.
	big := strings.Repeat("x", 90000)
	body := []byte(`{"model":"auto","messages":[{"role":"user","content":"` + big + `"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(body)))
	attempts, out := r.jevAttempts(req, jevRule(decision.URL, 0.3, 1), body, "auto")
	if out.Reason != "context-no-downgrade" {
		t.Errorf("reason = %q, want context-no-downgrade", out.Reason)
	}
	if len(attempts) == 0 || attempts[0].server.ID != "s1" {
		t.Errorf("attempts = %v, want the default tier first", attemptIDs(attempts))
	}
}

func TestJevAttemptsImages(t *testing.T) {
	var gotImages []any
	decision := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(b, &body)
		gotImages, _ = body["images"].([]any)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"answers":{"model":{"type":"choice","choice":"opus","confidence":0.9}}}`))
	}))
	t.Cleanup(decision.Close)
	r, store, _ := jevTestRouter(t)
	addJevServers(t, store, decision.URL)
	rule := jevRule(decision.URL, 0.3, 1)
	rule.Jev.Images = true

	// OpenAI image_url (data URL kept, http URL skipped) + Anthropic base64 source.
	body := []byte(`{"model":"auto","messages":[{"role":"user","content":[` +
		`{"type":"text","text":"what is in these images?"},` +
		`{"type":"image_url","image_url":{"url":"data:image/png;base64,AAA"}},` +
		`{"type":"image_url","image_url":{"url":"https://x/y.png"}},` +
		`{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"BBB"}}` +
		`]}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(body)))
	if _, out := r.jevAttempts(req, rule, body, "auto"); out.Reason != "jev" {
		t.Fatalf("reason = %q, want jev", out.Reason)
	}
	want := []string{"data:image/png;base64,AAA", "data:image/jpeg;base64,BBB"}
	if len(gotImages) != len(want) {
		t.Fatalf("decision images = %v, want %v", gotImages, want)
	}
	for i, v := range want {
		if gotImages[i] != v {
			t.Errorf("image %d = %v, want %v", i, gotImages[i], v)
		}
	}

	// Off by default: the same body sends no images.
	gotImages = nil
	rule.Jev.Images = false
	req = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(body)))
	r.jevAttempts(req, rule, body, "auto")
	if len(gotImages) != 0 {
		t.Errorf("images = %v with Images off, want none", gotImages)
	}
}

func TestJevRequestImagesCap(t *testing.T) {
	var parts []string
	for i := 0; i < 10; i++ {
		parts = append(parts, fmt.Sprintf(`{"type":"image_url","image_url":{"url":"data:image/png;base64,%d"}}`, i))
	}
	body := []byte(`{"model":"auto","messages":[{"role":"user","content":[` + strings.Join(parts, ",") + `]}]}`)
	imgs := jevRequestImages(body)
	if len(imgs) != jevMaxImages {
		t.Errorf("images = %d, want the cap %d", len(imgs), jevMaxImages)
	}
}

func TestJevAttemptsCountTokensSkipsDecision(t *testing.T) {
	r, store, _ := jevTestRouter(t)
	calls := 0
	decision := jevDecision(t, `{"answers":{"model":{"type":"choice","choice":"opus","confidence":0.9}}}`, &calls)
	addJevServers(t, store, decision.URL)
	body := []byte(`{"model":"auto","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", strings.NewReader(string(body)))
	attempts, out := r.jevAttempts(req, jevRule(decision.URL, 0.3, 1), body, "auto")
	if out.Reason != "off" || calls != 0 {
		t.Errorf("reason = %q calls = %d, want off with no decision call", out.Reason, calls)
	}
	if len(attempts) == 0 || attempts[0].server.ID != "s1" {
		t.Errorf("attempts = %v, want the default tier first", attemptIDs(attempts))
	}

	// A tool_result-only last turn (agent loop) is an auxiliary call too.
	body = []byte(`{"model":"auto","messages":[{"role":"user","content":[{"type":"tool_result","content":"output"}]}]}`)
	req = httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(string(body)))
	_, out = r.jevAttempts(req, jevRule(decision.URL, 0.3, 1), body, "auto")
	if out.Reason != "off" || calls != 0 {
		t.Errorf("tool_result turn: reason = %q calls = %d, want off with no decision call", out.Reason, calls)
	}
}

func TestListModelsJevPoolContext(t *testing.T) {
	r, store, _ := jevTestRouter(t)
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"auto"}, Router: domain.RouterJev, Enabled: true,
		Jev: &domain.JevRouter{ServerID: "decision", Candidates: []domain.JevCandidate{
			{ServerID: "a", TargetModel: "big", ContextWindow: 128000},
			{ServerID: "b", TargetModel: "small", ContextWindow: 32000},
			{ServerID: "c", TargetModel: "unknown"},
		}},
	})
	w := httptest.NewRecorder()
	r.listModels(w, httptest.NewRequest(http.MethodGet, "/v1/models", nil))

	var resp struct {
		Data []struct {
			ID            string `json:"id"`
			ContextWindow int    `json:"context_window"`
		} `json:"data"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v (%s)", err, w.Body.String())
	}
	if len(resp.Data) != 1 || resp.Data[0].ContextWindow != 32000 {
		t.Errorf("models = %+v, want context_window = pool minimum 32000", resp.Data)
	}
}

func TestJevCandidateIDs(t *testing.T) {
	pool := []domain.JevCandidate{
		{ServerID: "a", TargetModel: "sonnet"},
		{ServerID: "b", TargetModel: "sonnet"},
		{ServerID: "c"},
	}
	got := jevCandidateIDs(pool)
	want := []string{"sonnet", "sonnet@b", "c"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("ids = %v, want %v", got, want)
	}
}

// jevTestRouter is newTestRouter with a writable config file, so AddServer
// persists instead of failing on the shared /nonexistent path.
func jevTestRouter(t *testing.T) (*Router, *config.Store, *metrics.Store) {
	t.Helper()
	store, err := config.NewStore(filepath.Join(t.TempDir(), "config.json"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	ms := metrics.New(100)
	return New(store, ms, nil, nil, nil), store, ms
}

func addJevServers(t *testing.T, store *config.Store, decisionURL string) {
	t.Helper()
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	t.Cleanup(backend.Close)
	for _, id := range []string{"s0", "s1", "s2", "s3"} {
		if err := store.AddServer(&domain.Server{ID: id, Name: id, URL: backend.URL,
			APITypes: []domain.APIType{domain.APITypeOpenAI}}); err != nil {
			t.Fatalf("AddServer: %v", err)
		}
	}
	if err := store.AddServer(&domain.Server{ID: "decision", Name: "decision", URL: decisionURL,
		APITypes: []domain.APIType{domain.APITypeOpenAI}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
}

func attemptIDs(attempts []serverAttempt) []string {
	ids := make([]string, len(attempts))
	for i, a := range attempts {
		ids[i] = a.server.ID
	}
	return ids
}

func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
