package router

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"llm-api-router/config"
	"llm-api-router/domain"
	"llm-api-router/metrics"
	"llm-api-router/proxy"
	"llm-api-router/trace"
)

func newTestRouter(t *testing.T) (*Router, *config.Store, *metrics.Store, *trace.Store) {
	t.Helper()
	// A writable config path (not "/nonexistent"): the trace fixtures add
	// servers and rules through the store, which persists on every write.
	store, err := config.NewStore(filepath.Join(t.TempDir(), "config.json"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	ms := metrics.New(100)
	ts := trace.New(100, domain.TraceSettings{})
	return New(store, ms, nil, nil, nil, ts), store, ms, ts
}

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

func TestOrderedFallbacks(t *testing.T) {
	rule := func(fbs ...domain.FallbackEntry) *domain.RoutingRule {
		return &domain.RoutingRule{Fallbacks: fbs}
	}
	ids := func(fbs []domain.FallbackEntry) []string {
		out := make([]string, 0, len(fbs))
		for _, f := range fbs {
			out = append(out, f.ServerID)
		}
		return out
	}
	eq := func(got, want []string) bool {
		if len(got) != len(want) {
			return false
		}
		for i := range got {
			if got[i] != want[i] {
				return false
			}
		}
		return true
	}

	tests := []struct {
		name string
		rule *domain.RoutingRule
		want []string
	}{
		{"all disabled", rule(
			domain.FallbackEntry{ServerID: "a", Enabled: new(false)},
			domain.FallbackEntry{ServerID: "b", Enabled: new(false)},
		), []string{}},
		{"priority ascending", rule(
			domain.FallbackEntry{ServerID: "a", Priority: 2},
			domain.FallbackEntry{ServerID: "b", Priority: 1},
			domain.FallbackEntry{ServerID: "c", Priority: 3},
		), []string{"b", "a", "c"}},
		{"unset sorts last, stable", rule(
			domain.FallbackEntry{ServerID: "a"},
			domain.FallbackEntry{ServerID: "b", Priority: 1},
			domain.FallbackEntry{ServerID: "c"},
		), []string{"b", "a", "c"}},
		{"disabled filtered out", rule(
			domain.FallbackEntry{ServerID: "a", Enabled: new(false)},
			domain.FallbackEntry{ServerID: "b", Priority: 1},
			domain.FallbackEntry{ServerID: "c"},
		), []string{"b", "c"}},
		{"stable among equal priority", rule(
			domain.FallbackEntry{ServerID: "a", Priority: 1},
			domain.FallbackEntry{ServerID: "b", Priority: 1},
			domain.FallbackEntry{ServerID: "c", Priority: 1},
		), []string{"a", "b", "c"}},
		{"mixed set and unset", rule(
			domain.FallbackEntry{ServerID: "a"},
			domain.FallbackEntry{ServerID: "b", Priority: 2},
			domain.FallbackEntry{ServerID: "c", Priority: 1},
			domain.FallbackEntry{ServerID: "d"},
		), []string{"c", "b", "a", "d"}},
		{"explicit enabled true kept", rule(
			domain.FallbackEntry{ServerID: "a", Enabled: new(true), Priority: 2},
			domain.FallbackEntry{ServerID: "b", Priority: 1},
		), []string{"b", "a"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ids(orderedFallbacks(tt.rule))
			if !eq(got, tt.want) {
				t.Errorf("orderedFallbacks() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAPITypeFromPath(t *testing.T) {
	tests := []struct {
		path string
		want domain.APIType
	}{
		{"/v1/chat/completions", domain.APITypeOpenAI},
		{"/v1/responses", domain.APITypeOpenAI},
		{"/v1/completions", domain.APITypeOpenAI},
		{"/v1/messages", domain.APITypeAnthropic},
		{"/v1/messages/stream", domain.APITypeAnthropic},
		{"/v1/embeddings", domain.APITypeOpenAI},
	}
	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			got := apiTypeFromPath(tt.path)
			if got != tt.want {
				t.Errorf("apiTypeFromPath(%q) = %q, want %q", tt.path, got, tt.want)
			}
		})
	}
}

func TestExtractModel(t *testing.T) {
	tests := []struct {
		name    string
		body    []byte
		want    string
		wantErr bool
	}{
		{
			name:    "valid model",
			body:    []byte(`{"model":"gpt-4","messages":[]}`),
			want:    "gpt-4",
			wantErr: false,
		},
		{
			name:    "model with whitespace",
			body:    []byte(`{"model":"  gpt-4  ","messages":[]}`),
			want:    "gpt-4",
			wantErr: false,
		},
		{
			name:    "missing model field",
			body:    []byte(`{"messages":[]}`),
			wantErr: true,
		},
		{
			name:    "model is not a string",
			body:    []byte(`{"model":123}`),
			wantErr: true,
		},
		{
			name:    "invalid JSON",
			body:    []byte(`not json`),
			wantErr: true,
		},
		{
			name: "model inside message content is ignored",
			body: []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"compare \"model\":\"gemini\" vs claude"}]}`),
			want: "gpt-4",
		},
		{
			name:    "nested model not used when top-level missing",
			body:    []byte(`{"metadata":{"model":"internal-1"},"messages":[]}`),
			wantErr: true,
		},
		{
			name: "escaped quote in model value",
			body: []byte(`{"model":"unsloth/Qwen3.6-27B-MTP-GGUF:BF16","messages":[]}`),
			want: "unsloth/Qwen3.6-27B-MTP-GGUF:BF16",
		},
		{
			name:    "empty object",
			body:    []byte(`{}`),
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := extractModel(tt.body)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestListModels(t *testing.T) {
	t.Run("returns enabled models", func(t *testing.T) {
		r, store, _, _ := newTestRouter(t)
		_ = store.AddRule(&domain.RoutingRule{
			IncomingModels: []string{"gpt-4", "gpt-4-turbo"},
			ServerID:       "s1",
			Enabled:        true,
		})
		_ = store.AddRule(&domain.RoutingRule{
			IncomingModels: []string{"claude-3"},
			ServerID:       "s2",
			Enabled:        true,
		})
		_ = store.AddRule(&domain.RoutingRule{
			IncomingModels: []string{"disabled-model"},
			ServerID:       "s3",
			Enabled:        false,
		})

		req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
		w := httptest.NewRecorder()
		r.Handle(w, req)

		resp := w.Result()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("got status %d, want 200", resp.StatusCode)
		}

		var result struct {
			Object string           `json:"object"`
			Data   []map[string]any `json:"data"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			t.Fatalf("decode response: %v", err)
		}

		if result.Object != "list" {
			t.Errorf("object = %q, want %q", result.Object, "list")
		}
		if len(result.Data) != 3 {
			t.Fatalf("got %d models, want 3", len(result.Data))
		}
	})

	t.Run("no models returns empty list", func(t *testing.T) {
		r, _, _, _ := newTestRouter(t)
		req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
		w := httptest.NewRecorder()
		r.Handle(w, req)

		var result struct {
			Data []map[string]any `json:"data"`
		}
		_ = json.NewDecoder(w.Result().Body).Decode(&result)
		if len(result.Data) != 0 {
			t.Errorf("got %d models, want 0", len(result.Data))
		}
	})

	t.Run("exposes context_window", func(t *testing.T) {
		r, store, _, _ := newTestRouter(t)
		_ = store.AddRule(&domain.RoutingRule{
			IncomingModels: []string{"opus"},
			ServerID:       "s1",
			Enabled:        true,
			ContextWindow:  200000,
		})
		_ = store.AddRule(&domain.RoutingRule{
			IncomingModels: []string{"unknown-window"},
			ServerID:       "s2",
			Enabled:        true,
			ContextWindow:  0, // unknown → omitted
		})

		req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
		w := httptest.NewRecorder()
		r.Handle(w, req)

		var result struct {
			Data []map[string]any `json:"data"`
		}
		_ = json.NewDecoder(w.Result().Body).Decode(&result)

		byID := map[string]map[string]any{}
		for _, m := range result.Data {
			byID[m["id"].(string)] = m
		}

		if got, ok := byID["opus"]["context_window"]; !ok || got != float64(200000) {
			t.Errorf("opus context_window = %v (present: %v), want 200000", got, ok)
		}
		// LiteLLM behavior: absent (not null) when malformed/unknown.
		if _, present := byID["unknown-window"]["context_window"]; present {
			t.Error("unknown-window should not have context_window")
		}
	})
}

func TestHandleMethodNotAllowed(t *testing.T) {
	r, _, _, _ := newTestRouter(t)

	methods := []string{http.MethodPut, http.MethodDelete, http.MethodPatch, http.MethodOptions}
	for _, method := range methods {
		t.Run(method, func(t *testing.T) {
			req := httptest.NewRequest(method, "/v1/chat/completions", nil)
			w := httptest.NewRecorder()
			r.Handle(w, req)

			if w.Result().StatusCode != http.StatusMethodNotAllowed {
				t.Errorf("got status %d, want 405", w.Result().StatusCode)
			}
		})
	}
}

func TestHandleInvalidBody(t *testing.T) {
	r, _, _, _ := newTestRouter(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`not json`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusBadRequest {
		t.Errorf("got status %d, want 400", w.Result().StatusCode)
	}
}

func TestHandleMissingModel(t *testing.T) {
	r, _, _, _ := newTestRouter(t)
	body := strings.NewReader(`{"messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusBadRequest {
		t.Errorf("got status %d, want 400", w.Result().StatusCode)
	}
}

// TestHandleSystemone verifies that a llama.cpp /v1/systemone request (no
// "model" field in the body) is routed by endpoint name via a rule whose
// incoming model is "systemone", and that the response model is rewritten back.
func TestHandleSystemone(t *testing.T) {
	var receivedBody string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		receivedBody = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"laya","answers":{"angry":{"type":"noul","noul":0.63}},"usage":{"input_tokens":10,"output_tokens":0}}`))
	}))
	defer backend.Close()

	r, store, ms, _ := newTestRouter(t)
	_ = store.AddServer(&domain.Server{
		ID:       "s1",
		Name:     "decision-server",
		URL:      backend.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"systemone"},
		TargetModel:    "laya",
		ServerID:       "s1",
		Enabled:        true,
	})

	body := strings.NewReader(`{"state":"charged twice","questions":{"angry":{"type":"noul","instructions":"angry?"}}}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/systemone", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("got status %d, want 200: %s", w.Result().StatusCode, w.Body.String())
	}

	// The upstream body keeps state/questions (the injected "model" field is
	// ignored by llama.cpp's systemone handler).
	var sent map[string]any
	if err := json.Unmarshal([]byte(receivedBody), &sent); err != nil {
		t.Fatalf("backend received invalid body %q: %v", receivedBody, err)
	}
	if sent["state"] != "charged twice" {
		t.Errorf("state not forwarded: %v", sent)
	}
	if _, ok := sent["questions"]; !ok {
		t.Errorf("questions not forwarded: %v", sent)
	}

	// Response model rewritten back to the incoming name.
	resp := w.Body.String()
	if !strings.Contains(resp, `"model":"systemone"`) {
		t.Errorf("response model not rewritten to systemone: %s", resp)
	}
	if !strings.Contains(resp, `"noul":0.63`) {
		t.Errorf("answers missing from response: %s", resp)
	}

	// Usage (input_tokens/output_tokens) captured in metrics.
	recent := ms.Recent()
	if len(recent) != 1 {
		t.Fatalf("expected 1 recorded request, got %d", len(recent))
	}
	if recent[0].PromptTokens != 10 {
		t.Errorf("prompt tokens = %v, want 10", recent[0].PromptTokens)
	}
}

// Audio uploads (OpenAI/llama.cpp /v1/audio/transcriptions) arrive as
// multipart/form-data with the model as a form field, not as JSON.
func TestHandleAudioTranscriptionsMultipart(t *testing.T) {
	var receivedBody, receivedType string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		receivedBody = string(b)
		receivedType = r.Header.Get("Content-Type")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"text":"privet","timing":{"wall_ms":900,"audio_duration_ms":3200,"rtf":0.28}}`))
	}))
	defer backend.Close()

	r, store, ms, _ := newTestRouter(t)
	_ = store.AddServer(&domain.Server{
		ID: "s1", Name: "audio", URL: backend.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"whisper"},
		TargetModel:    "whisper-large",
		ServerID:       "s1",
		Enabled:        true,
	})

	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	if err := mw.WriteField("model", "whisper"); err != nil {
		t.Fatalf("WriteField: %v", err)
	}
	if err := mw.WriteField("language", "ru"); err != nil {
		t.Fatalf("WriteField: %v", err)
	}
	fw, err := mw.CreateFormFile("file", "a.wav")
	if err != nil {
		t.Fatalf("CreateFormFile: %v", err)
	}
	if _, err := fw.Write([]byte("RIFFxxxxWAVEfmt ")); err != nil {
		t.Fatalf("write audio: %v", err)
	}
	if err := mw.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if res := w.Result(); res.StatusCode != http.StatusOK {
		t.Fatalf("got status %d, want 200: %s", res.StatusCode, w.Body.String())
	}
	if !strings.HasPrefix(receivedType, "multipart/") {
		t.Errorf("upstream content type = %q, want multipart", receivedType)
	}
	// The model field follows the rule's target model; the upload and the
	// sibling fields are forwarded untouched.
	if got, ok := proxy.ExtractModelFromMultipart([]byte(receivedBody)); !ok || got != "whisper-large" {
		t.Errorf("upstream model field = %q,%v; want whisper-large", got, ok)
	}
	for _, want := range []string{"RIFFxxxxWAVEfmt ", "language"} {
		if !strings.Contains(receivedBody, want) {
			t.Errorf("upload lost %q", want)
		}
	}
	if !strings.Contains(w.Body.String(), `"text":"privet"`) {
		t.Errorf("response not forwarded: %s", w.Body.String())
	}

	recent := ms.Recent()
	if len(recent) != 1 {
		t.Fatalf("expected 1 recorded request, got %d", len(recent))
	}
	if recent[0].APIEndpoint != "transcriptions" {
		t.Errorf("endpoint = %q, want transcriptions", recent[0].APIEndpoint)
	}
	if recent[0].AudioDurationMs != 3200 || recent[0].AudioRTF != 0.28 {
		t.Errorf("audio timing = %dms/%v, want 3200/0.28", recent[0].AudioDurationMs, recent[0].AudioRTF)
	}
}

// llama.cpp's TTS route lives outside /v1 and answers with raw bytes: the audio
// must reach the client intact, and audio.cpp reports its metering in
// X-AudioCPP-* headers when there is no JSON body to parse.
func TestHandleTTSBinaryPassthrough(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(b), `"model":"voice-1"`) {
			t.Errorf("upstream body = %s", b)
		}
		w.Header().Set("Content-Type", "audio/wav")
		w.Header().Set("X-AudioCPP-Audio-Duration-Ms", "4500")
		w.Header().Set("X-AudioCPP-RTF", "0.4")
		_, _ = w.Write([]byte("RIFFxxxxWAVEfmt "))
	}))
	defer backend.Close()

	r, store, ms, _ := newTestRouter(t)
	_ = store.AddServer(&domain.Server{
		ID: "s1", Name: "audio", URL: backend.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"tts"},
		TargetModel:    "voice-1",
		ServerID:       "s1",
		Enabled:        true,
	})

	req := httptest.NewRequest(http.MethodPost, "/tts", strings.NewReader(`{"model":"tts","input":"hello"}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Body.String() != "RIFFxxxxWAVEfmt " {
		t.Fatalf("audio not forwarded intact: %q", w.Body.String())
	}
	recent := ms.Recent()
	if len(recent) != 1 {
		t.Fatalf("expected 1 recorded request, got %d", len(recent))
	}
	if recent[0].APIEndpoint != "tts" {
		t.Errorf("endpoint = %q, want tts", recent[0].APIEndpoint)
	}
	if recent[0].AudioDurationMs != 4500 || recent[0].AudioRTF != 0.4 {
		t.Errorf("audio timing = %dms/%v, want 4500/0.4", recent[0].AudioDurationMs, recent[0].AudioRTF)
	}
}

// audio.cpp's voice picker is a GET with the model in the query string.
func TestHandleAudioVoicesGet(t *testing.T) {
	var receivedQuery string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedQuery = r.URL.RawQuery
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"voices":["alba","cosette"]}`))
	}))
	defer backend.Close()

	r, store, ms, _ := newTestRouter(t)
	_ = store.AddServer(&domain.Server{
		ID: "s1", Name: "audio", URL: backend.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"pocket-tts"},
		ServerID:       "s1",
		Enabled:        true,
	})

	req := httptest.NewRequest(http.MethodGet, "/v1/audio/voices?model=pocket-tts", nil)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if res := w.Result(); res.StatusCode != http.StatusOK {
		t.Fatalf("got status %d, want 200: %s", res.StatusCode, w.Body.String())
	}
	if !strings.Contains(receivedQuery, "model=pocket-tts") {
		t.Errorf("query not forwarded: %q", receivedQuery)
	}
	if !strings.Contains(w.Body.String(), "cosette") {
		t.Errorf("response not forwarded: %s", w.Body.String())
	}
	if recent := ms.Recent(); len(recent) != 1 || recent[0].APIEndpoint != "voices" {
		t.Errorf("recent = %+v, want one voices request", recent)
	}
}

func TestHandleNoRoutingRule(t *testing.T) {
	r, _, _, _ := newTestRouter(t)
	body := strings.NewReader(`{"model":"unknown-model","messages":[]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusNotFound {
		t.Errorf("got status %d, want 404", w.Result().StatusCode)
	}
}

func TestHandleServerNotFound(t *testing.T) {
	r, store, _, _ := newTestRouter(t)
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		ServerID:       "nonexistent-server",
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"gpt-4","messages":[]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusInternalServerError {
		t.Errorf("got status %d, want 500", w.Result().StatusCode)
	}
}

func TestHandleAllBackendsFail(t *testing.T) {
	r, store, _, _ := newTestRouter(t)

	_ = store.AddServer(&domain.Server{
		ID:       "s1",
		Name:     "test-server",
		URL:      "http://localhost:1",
		APIKey:   "test-key",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "s1",
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	// With no rule retries the default became 10; opt out explicitly to keep
	// this test fast (failure is instant anyway).
	req.Header.Set("X-Router-Retries", "0")
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusBadGateway {
		t.Errorf("got status %d, want 502", w.Result().StatusCode)
	}
}

// TestHandleInjectsStreamUsage verifies the router injects
// stream_options.include_usage for streaming OpenAI chat completions and
// strips the resulting usage chunk from the client stream while recording it
// in metrics.
func TestHandleInjectsStreamUsage(t *testing.T) {
	var receivedBody string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		buf := make([]byte, 4096)
		n, _ := r.Body.Read(buf)
		receivedBody = string(buf[:n])

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4}}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer backend.Close()

	r, store, ms, _ := newTestRouter(t)
	_ = store.AddServer(&domain.Server{
		ID:       "s1",
		Name:     "test-server",
		URL:      backend.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "s1",
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"gpt-4","stream":true,"messages":[]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	var sent map[string]any
	if err := json.Unmarshal([]byte(receivedBody), &sent); err != nil {
		t.Fatalf("backend received invalid body %q: %v", receivedBody, err)
	}
	so, ok := sent["stream_options"].(map[string]any)
	if !ok {
		t.Fatalf("stream_options not injected into upstream request: %v", sent)
	}
	if inc, _ := so["include_usage"].(bool); !inc {
		t.Error("include_usage should be true upstream")
	}

	if strings.Contains(w.Body.String(), "usage") {
		t.Errorf("client stream should not contain usage chunk:\n%s", w.Body.String())
	}

	sum := ms.Summaries()
	if len(sum) == 0 {
		t.Fatal("no metrics recorded")
	}
	found := false
	for _, s := range sum {
		if s.TotalRequests > 0 {
			found = true
			if s.TotalPromptTok != 3 || s.TotalCompleteTok != 1 {
				t.Errorf("usage should be captured despite stripping, got prompt=%d completion=%d", s.TotalPromptTok, s.TotalCompleteTok)
			}
		}
	}
	if !found {
		t.Errorf("no metric summary for gpt-4: %+v", sum)
	}
}

// TestHandleResponsesAPI verifies /v1/responses (OpenAI Responses API) requests
// are routed as OpenAI API type: model rewritten to the target upstream, then
// rewritten back to the client's original model, with usage extracted from the
// Responses API usage format (top-level "usage", input_tokens/output_tokens).
func TestHandleResponsesAPI(t *testing.T) {
	var gotPath, receivedBody string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		buf, _ := io.ReadAll(r.Body)
		receivedBody = string(buf)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","model":"gpt-4o-real","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hi"}]}],"usage":{"input_tokens":11,"output_tokens":3,"total_tokens":14,"input_tokens_details":{"cached_tokens":5}}}`))
	}))
	defer backend.Close()

	r, store, ms, _ := newTestRouter(t)
	_ = store.AddServer(&domain.Server{
		ID:       "s1",
		Name:     "test-server",
		URL:      backend.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4o-real",
		ServerID:       "s1",
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"gpt-4","input":"hi"}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Result().StatusCode)
	}
	if gotPath != "/v1/responses" {
		t.Errorf("backend path = %q, want /v1/responses", gotPath)
	}
	var sent map[string]any
	if err := json.Unmarshal([]byte(receivedBody), &sent); err != nil {
		t.Fatalf("backend received invalid body %q: %v", receivedBody, err)
	}
	if m, _ := sent["model"].(string); m != "gpt-4o-real" {
		t.Errorf("upstream model = %q, want gpt-4o-real", m)
	}

	// Client response: model rewritten back to the original name.
	var resp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("client response invalid: %v", err)
	}
	if m, _ := resp["model"].(string); m != "gpt-4" {
		t.Errorf("response model = %q, want gpt-4", m)
	}

	recent := ms.Recent()
	if len(recent) != 1 {
		t.Fatalf("expected 1 metric, got %d", len(recent))
	}
	m := recent[0]
	if m.APIEndpoint != "responses" {
		t.Errorf("APIEndpoint = %q, want responses", m.APIEndpoint)
	}
	if m.APIType != domain.APITypeOpenAI {
		t.Errorf("APIType = %q, want openai", m.APIType)
	}
	if m.PromptTokens != 11 || m.CompletionTokens != 3 || m.TotalTokens != 14 {
		t.Errorf("tokens = prompt %d completion %d total %d, want 11/3/14", m.PromptTokens, m.CompletionTokens, m.TotalTokens)
	}
	if m.CachedTokens != 5 {
		t.Errorf("CachedTokens = %d, want 5", m.CachedTokens)
	}
}

// TestHandleResponsesAPIStreaming verifies streaming /v1/responses: SSE events
// are relayed to the client, the model is rewritten inside streamed response
// objects, and usage is extracted from the response.completed event
// (the "response.usage" path).
func TestHandleResponsesAPIStreaming(t *testing.T) {
	var receivedBody string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		buf, _ := io.ReadAll(r.Body)
		receivedBody = string(buf)

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		_, _ = w.Write([]byte("data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"model\":\"gpt-4o-real\"}}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-4o-real\",\"usage\":{\"input_tokens\":11,\"output_tokens\":3,\"total_tokens\":14,\"input_tokens_details\":{\"cached_tokens\":5}}}}\n\n"))
	}))
	defer backend.Close()

	r, store, ms, _ := newTestRouter(t)
	_ = store.AddServer(&domain.Server{
		ID:       "s1",
		Name:     "test-server",
		URL:      backend.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4o-real",
		ServerID:       "s1",
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"gpt-4","input":"hi"}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	var sent map[string]any
	if err := json.Unmarshal([]byte(receivedBody), &sent); err != nil {
		t.Fatalf("backend received invalid body %q: %v", receivedBody, err)
	}
	if m, _ := sent["model"].(string); m != "gpt-4o-real" {
		t.Errorf("upstream model = %q, want gpt-4o-real", m)
	}

	client := w.Body.String()
	if !strings.Contains(client, `"type":"response.output_text.delta"`) {
		t.Errorf("client stream missing delta event:\n%s", client)
	}
	if strings.Contains(client, "gpt-4o-real") {
		t.Errorf("client stream still contains target model:\n%s", client)
	}
	if !strings.Contains(client, `"model":"gpt-4"`) {
		t.Errorf("client stream missing rewritten model:\n%s", client)
	}

	recent := ms.Recent()
	if len(recent) != 1 {
		t.Fatalf("expected 1 metric, got %d", len(recent))
	}
	m := recent[0]
	if m.APIEndpoint != "responses" {
		t.Errorf("APIEndpoint = %q, want responses", m.APIEndpoint)
	}
	if m.PromptTokens != 11 || m.CompletionTokens != 3 {
		t.Errorf("tokens = prompt %d completion %d, want 11/3", m.PromptTokens, m.CompletionTokens)
	}
	if m.CachedTokens != 5 {
		t.Errorf("CachedTokens = %d, want 5", m.CachedTokens)
	}
}

// syncRecorder is a goroutine-safe ResponseRecorder: the heartbeat timer
// writes from its own goroutine while the relay loop writes from another.
type syncRecorder struct {
	mu     sync.Mutex
	header http.Header
	status int
	body   strings.Builder
}

func newSyncRecorder() *syncRecorder {
	return &syncRecorder{header: make(http.Header)}
}

func (s *syncRecorder) Header() http.Header { return s.header }
func (s *syncRecorder) WriteHeader(code int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.status = code
}
func (s *syncRecorder) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.body.Write(p)
}
func (s *syncRecorder) Flush() {}
func (s *syncRecorder) content() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.body.String()
}

// TestHandleAnthropicKeepAlive verifies the router injects Anthropic ping
// heartbeats ("event: ping", the official protocol frame) into /v1/messages
// streams while the backend stays silent, so clients do not time out while the
// backend is thinking.
func TestHandleAnthropicKeepAlive(t *testing.T) {
	old := proxy.KeepAliveIdle
	proxy.KeepAliveIdle = 30 * time.Millisecond
	defer func() { proxy.KeepAliveIdle = old }()

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		_, _ = w.Write([]byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-3-opus\"}}\n\n"))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		time.Sleep(100 * time.Millisecond) // backend pause longer than the heartbeat idle
		_, _ = w.Write([]byte("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n"))
	}))
	defer backend.Close()

	r, store, _, _ := newTestRouter(t)
	_ = store.AddServer(&domain.Server{
		ID:       "s1",
		Name:     "test-server",
		URL:      backend.URL,
		APITypes: []domain.APIType{domain.APITypeAnthropic},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"claude-3"},
		TargetModel:    "claude-3-opus",
		ServerID:       "s1",
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"claude-3","max_tokens":100}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", body)
	w := newSyncRecorder()
	r.Handle(w, req)

	got := w.content()
	start := strings.Index(got, "message_start")
	ping := strings.Index(got, "event: ping")
	delta := strings.Index(got, "content_block_delta")
	if start < 0 || delta < 0 {
		t.Fatalf("stream incomplete:\n%s", got)
	}
	if ping < 0 {
		t.Fatalf("client stream missing Anthropic ping heartbeat:\n%s", got)
	}
	if start >= ping || ping >= delta {
		t.Errorf("expected ping between the two events:\n%s", got)
	}
}

func TestHandleAnthropicPath(t *testing.T) {
	r, store, _, _ := newTestRouter(t)

	_ = store.AddServer(&domain.Server{
		ID:       "s1",
		Name:     "test-server",
		URL:      "http://localhost:1",
		APIKey:   "test-key",
		APITypes: []domain.APIType{domain.APITypeAnthropic},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"claude-3"},
		TargetModel:    "claude-3-opus",
		ServerID:       "s1",
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"claude-3","max_tokens":100}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusBadGateway {
		t.Errorf("got status %d, want 502", w.Result().StatusCode)
	}
}

func TestHandleClientDisconnectNoFallback(t *testing.T) {
	// When client disconnects (context cancelled), router should:
	// 1. Stop immediately — no fallback attempts
	// 2. NOT mark server unhealthy
	// 3. NOT record metrics (request was aborted, not completed)
	_, store, ms, _ := newTestRouter(t)
	health := config.NewHealthTracker(store, 0) // no auto-check

	_ = store.AddServer(&domain.Server{
		ID:       "primary",
		Name:     "Primary",
		URL:      "http://localhost:1", // unreachable — would trigger fallback
		APIKey:   "test-key",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddServer(&domain.Server{
		ID:       "fallback",
		Name:     "Fallback",
		URL:      "http://localhost:2",
		APIKey:   "test-key",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "primary",
		Fallbacks:      []domain.FallbackEntry{{ServerID: "fallback"}},
		Enabled:        true,
	})

	// Create router with health tracker
	r := New(store, ms, health, nil, nil, nil)

	// Simulate client disconnect by cancelling context
	ctx, cancel := context.WithCancel(context.Background())
	body := strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	req = req.WithContext(ctx)

	// Cancel context before request to simulate immediate disconnect
	cancel()

	w := httptest.NewRecorder()
	r.Handle(w, req)

	// Should NOT have recorded metrics (client gone)
	recent := ms.Recent()
	if len(recent) != 0 {
		t.Errorf("expected 0 metrics after client disconnect, got %d", len(recent))
	}
}

// flakyWriter delivers writes to an embedded recorder until the cancel func is
// called on the second write — simulating a client that dies mid-stream.
type flakyWriter struct {
	header http.Header
	body   strings.Builder
	writes int
	cancel context.CancelFunc
}

func (f *flakyWriter) Header() http.Header {
	if f.header == nil {
		f.header = make(http.Header)
	}
	return f.header
}
func (f *flakyWriter) WriteHeader(int) {}
func (f *flakyWriter) Flush()          {}
func (f *flakyWriter) Write(p []byte) (int, error) {
	f.writes++
	if f.writes == 2 {
		f.cancel()
		return 0, errors.New("client disconnected")
	}
	return f.body.Write(p)
}

// TestHandleClientDisconnectRecordsMetrics verifies a client that goes away
// mid-stream leaves a 499 metric with the tokens the backend produced —
// the generation ran and cost tokens, so it must not vanish from metrics.
func TestHandleClientDisconnectRecordsMetrics(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		f := w.(http.Flusher)
		_, _ = w.Write([]byte("data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n"))
		f.Flush()
		time.Sleep(100 * time.Millisecond) // relay must see two separate writes
		_, _ = w.Write([]byte("data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2,\"total_tokens\":9}}\n\n"))
		f.Flush()
	}))
	defer backend.Close()

	r, store, ms, _ := newTestRouter(t)
	_ = store.AddServer(&domain.Server{
		ID:       "s1",
		Name:     "test-server",
		URL:      backend.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "s1",
		Enabled:        true,
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`))
	req = req.WithContext(ctx)

	r.Handle(&flakyWriter{cancel: cancel}, req)

	recent := ms.Recent()
	if len(recent) != 1 {
		t.Fatalf("expected 1 metric after mid-stream disconnect, got %d", len(recent))
	}
	m := recent[0]
	if m.StatusCode != 499 {
		t.Errorf("StatusCode = %d, want 499", m.StatusCode)
	}
	if m.PromptTokens != 7 || m.CompletionTokens != 2 {
		t.Errorf("tokens = prompt %d completion %d, want 7/2", m.PromptTokens, m.CompletionTokens)
	}
}

func TestMetricsAreRecorded(t *testing.T) {
	r, store, ms, _ := newTestRouter(t)

	_ = store.AddServer(&domain.Server{
		ID:       "s1",
		Name:     "test-server",
		URL:      "http://localhost:1",
		APIKey:   "test-key",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "s1",
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	recent := ms.Recent()
	if len(recent) != 1 {
		t.Fatalf("expected 1 metric, got %d", len(recent))
	}
	if recent[0].Model != "gpt-4" {
		t.Errorf("metric model = %q, want %q", recent[0].Model, "gpt-4")
	}
	if recent[0].StatusCode != http.StatusBadGateway {
		t.Errorf("metric status = %d, want %d", recent[0].StatusCode, http.StatusBadGateway)
	}
}

func TestFallbackPreservesActualModel(t *testing.T) {
	// When fallback occurs, the response should contain the actual model
	// used by the fallback server, not the original client model.
	// Primary server fails, fallback succeeds.

	// Create fallback server that returns a known model name
	fallbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"fallback-model","choices":[{"finish_reason":"stop"}]}`)) //nolint:errcheck
	}))
	defer fallbackServer.Close()

	r, store, _, _ := newTestRouter(t)

	_ = store.AddServer(&domain.Server{
		ID:       "primary",
		Name:     "Primary",
		URL:      "http://localhost:1", // unreachable
		APIKey:   "test-key",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddServer(&domain.Server{
		ID:       "fallback",
		Name:     "Fallback",
		URL:      fallbackServer.URL,
		APIKey:   "test-key",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"opus"},
		TargetModel:    "gpt-4",
		ServerID:       "primary",
		Fallbacks:      []domain.FallbackEntry{{ServerID: "fallback", TargetModel: "fallback-model"}},
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"opus","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("got status %d, want 200", w.Result().StatusCode)
	}

	respBody, _ := io.ReadAll(w.Result().Body)
	var resp map[string]any
	_ = json.Unmarshal(respBody, &resp)

	// Response should contain the actual fallback model, not "opus"
	model, ok := resp["model"].(string)
	if !ok {
		t.Fatalf("no 'model' field in response: %s", respBody)
	}
	if model != "fallback-model" {
		t.Errorf("response model = %q, want %q (actual fallback model)", model, "fallback-model")
	}
}

func TestPrimaryAttemptRewritesModel(t *testing.T) {
	// When primary succeeds, response should contain the original client model
	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`)) //nolint:errcheck
	}))
	defer primaryServer.Close()

	r, store, _, _ := newTestRouter(t)

	_ = store.AddServer(&domain.Server{
		ID:       "primary",
		Name:     "Primary",
		URL:      primaryServer.URL,
		APIKey:   "test-key",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"opus"},
		TargetModel:    "gpt-4",
		ServerID:       "primary",
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"opus","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("got status %d, want 200", w.Result().StatusCode)
	}

	respBody, _ := io.ReadAll(w.Result().Body)
	var resp map[string]any
	_ = json.Unmarshal(respBody, &resp)

	model, ok := resp["model"].(string)
	if !ok {
		t.Fatalf("no 'model' field in response: %s", respBody)
	}
	// Primary attempt: response should contain original client model
	if model != "opus" {
		t.Errorf("response model = %q, want %q (original client model)", model, "opus")
	}
}

func TestNoRewriteWhenModelsMatch(t *testing.T) {
	// When targetModel == originalModel, no rewriting occurs
	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`)) //nolint:errcheck
	}))
	defer primaryServer.Close()

	r, store, _, _ := newTestRouter(t)

	_ = store.AddServer(&domain.Server{
		ID:       "primary",
		Name:     "Primary",
		URL:      primaryServer.URL,
		APIKey:   "test-key",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4", // Same as incoming
		ServerID:       "primary",
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("got status %d, want 200", w.Result().StatusCode)
	}

	respBody, _ := io.ReadAll(w.Result().Body)
	var resp map[string]any
	_ = json.Unmarshal(respBody, &resp)

	model, ok := resp["model"].(string)
	if !ok {
		t.Fatalf("no 'model' field in response: %s", respBody)
	}
	// Models match, so response should contain gpt-4
	if model != "gpt-4" {
		t.Errorf("response model = %q, want %q", model, "gpt-4")
	}
}

func TestFallbackLogsActualModel(t *testing.T) {
	// Verify that when fallback occurs, the log message includes the actual model
	// This is a structural test to ensure wasFallback is set correctly in metrics
	r, store, ms, _ := newTestRouter(t)

	// Create fallback server that succeeds
	fallbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"haiku","choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":10}}`)) //nolint:errcheck
	}))
	defer fallbackServer.Close()

	_ = store.AddServer(&domain.Server{
		ID:       "primary",
		Name:     "Primary",
		URL:      "http://localhost:1", // unreachable
		APIKey:   "test-key",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddServer(&domain.Server{
		ID:       "fallback",
		Name:     "Fallback",
		URL:      fallbackServer.URL,
		APIKey:   "test-key",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"opus"},
		TargetModel:    "gpt-4",
		ServerID:       "primary",
		Fallbacks:      []domain.FallbackEntry{{ServerID: "fallback", TargetModel: "haiku"}},
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"opus","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Fatalf("got status %d, want 200", w.Result().StatusCode)
	}

	recent := ms.Recent()
	if len(recent) != 1 {
		t.Fatalf("expected 1 metric, got %d", len(recent))
	}
	m := recent[0]
	if !m.WasFallback {
		t.Error("WasFallback should be true")
	}
	if m.TargetModel != "haiku" {
		t.Errorf("TargetModel = %q, want %q", m.TargetModel, "haiku")
	}
	if m.ServerID != "fallback" {
		t.Errorf("ServerID = %q, want %q", m.ServerID, "fallback")
	}
	// Model in metric should be the original client model
	if m.Model != "opus" {
		t.Errorf("Model = %q, want %q", m.Model, "opus")
	}
	_ = fmt.Sprintf // suppress unused import
}

func TestMidStreamErrorNoFallback(t *testing.T) {
	// When a mid-stream error occurs, the router should NOT try fallback
	// because headers are already sent to the client. Instead, it should
	// record whatever metrics it has and return.

	faultyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n")) //nolint:errcheck
		// Force disconnect
		hj, ok := w.(http.Hijacker)
		if ok {
			conn, _, err := hj.Hijack()
			if err == nil {
				conn.Close() //nolint:errcheck
			}
		}
	}))
	defer faultyServer.Close()

	// Fallback server that should NOT be called
	fallbackCalled := false
	fallbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fallbackCalled = true
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"backup","choices":[]}`)) //nolint:errcheck
	}))
	defer fallbackServer.Close()

	r, store, _, _ := newTestRouter(t)

	_ = store.AddServer(&domain.Server{
		ID:       "faulty",
		Name:     "Faulty",
		URL:      faultyServer.URL,
		APIKey:   "key1",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddServer(&domain.Server{
		ID:       "backup",
		Name:     "Backup",
		URL:      fallbackServer.URL,
		APIKey:   "key2",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"test-model"},
		TargetModel:    "gpt-4",
		ServerID:       "faulty",
		Fallbacks: []domain.FallbackEntry{
			{ServerID: "backup", TargetModel: "backup-model"},
		},
		Enabled: true,
	})

	body := strings.NewReader(`{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	// The response should contain the partial data from the faulty server
	// (or at least not be a 502, since mid-stream error sends data to client)
	respBody, _ := io.ReadAll(w.Result().Body)

	// Fallback server should NOT have been called
	if fallbackCalled {
		t.Error("fallback server should NOT have been called on mid-stream error")
	}

	// Response should either contain partial data or an error event
	// (not a 502 "all backends failed" error)
	if len(respBody) > 0 && strings.Contains(string(respBody), "all backends failed") {
		t.Errorf("should not return 'all backends failed' for mid-stream error, got: %s", respBody)
	}
}

func TestRetriesSameServerBeforeFallback(t *testing.T) {
	// A server that fails twice (connection closed pre-response) then succeeds
	// should be retried per NumRetries and never hit the fallback.
	var calls atomic.Int32
	flakyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) <= 2 {
			hj, ok := w.(http.Hijacker)
			if ok {
				conn, _, err := hj.Hijack()
				if err == nil {
					conn.Close() //nolint:errcheck
					return
				}
			}
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`)) //nolint:errcheck
	}))
	defer flakyServer.Close()

	fallbackCalled := false
	fallbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fallbackCalled = true
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"backup-model","choices":[{"finish_reason":"stop"}]}`)) //nolint:errcheck
	}))
	defer fallbackServer.Close()

	r, store, _, _ := newTestRouter(t)

	_ = store.AddServer(&domain.Server{
		ID:       "flaky",
		Name:     "Flaky",
		URL:      flakyServer.URL,
		APIKey:   "key1",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddServer(&domain.Server{
		ID:       "backup",
		Name:     "Backup",
		URL:      fallbackServer.URL,
		APIKey:   "key2",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "flaky",
		NumRetries:     2,
		Fallbacks: []domain.FallbackEntry{
			{ServerID: "backup", TargetModel: "backup-model"},
		},
		Enabled: true,
	})

	body := strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if got := w.Result().StatusCode; got != http.StatusOK {
		t.Fatalf("got status %d, want 200", got)
	}
	if calls.Load() != 3 {
		t.Errorf("primary server hit %d times, want 3 (1 + 2 retries)", calls.Load())
	}
	if fallbackCalled {
		t.Error("fallback server should NOT have been called when retries recovered")
	}
}

func TestRetriesExhaustedThenFallsBack(t *testing.T) {
	// A server that always fails should be retried NumRetries times and then
	// fall back to the next server.
	var calls atomic.Int32

	deadServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		hj, ok := w.(http.Hijacker)
		if ok {
			conn, _, err := hj.Hijack()
			if err == nil {
				conn.Close() //nolint:errcheck
				return
			}
		}
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer deadServer.Close()

	fallbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"backup-model","choices":[{"finish_reason":"stop"}]}`)) //nolint:errcheck
	}))
	defer fallbackServer.Close()

	r, store, _, _ := newTestRouter(t)

	_ = store.AddServer(&domain.Server{
		ID:       "dead",
		Name:     "Dead",
		URL:      deadServer.URL,
		APIKey:   "key1",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddServer(&domain.Server{
		ID:       "backup",
		Name:     "Backup",
		URL:      fallbackServer.URL,
		APIKey:   "key2",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "dead",
		NumRetries:     2,
		Fallbacks: []domain.FallbackEntry{
			{ServerID: "backup", TargetModel: "backup-model"},
		},
		Enabled: true,
	})

	body := strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if got := w.Result().StatusCode; got != http.StatusOK {
		t.Fatalf("got status %d, want 200 (via fallback)", got)
	}
	if calls.Load() != 3 {
		t.Errorf("dead server hit %d times, want 3 (1 + 2 retries)", calls.Load())
	}
}

func TestRetriesHeaderOverridesRule(t *testing.T) {
	// X-Router-Retries header overrides rule's num_retries: rule says 0 but
	// header says 1, so the flaky server gets retried and recovers.
	var calls atomic.Int32

	flakyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			hj, ok := w.(http.Hijacker)
			if ok {
				conn, _, err := hj.Hijack()
				if err == nil {
					conn.Close() //nolint:errcheck
					return
				}
			}
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`)) //nolint:errcheck
	}))
	defer flakyServer.Close()

	fallbackCalled := false
	fallbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fallbackCalled = true
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"backup-model","choices":[{"finish_reason":"stop"}]}`)) //nolint:errcheck
	}))
	defer fallbackServer.Close()

	r, store, _, _ := newTestRouter(t)

	_ = store.AddServer(&domain.Server{
		ID:       "flaky",
		Name:     "Flaky",
		URL:      flakyServer.URL,
		APIKey:   "key1",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddServer(&domain.Server{
		ID:       "backup",
		Name:     "Backup",
		URL:      fallbackServer.URL,
		APIKey:   "key2",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "flaky",
		NumRetries:     0,
		Fallbacks: []domain.FallbackEntry{
			{ServerID: "backup", TargetModel: "backup-model"},
		},
		Enabled: true,
	})

	body := strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	req.Header.Set("X-Router-Retries", "1")
	w := httptest.NewRecorder()
	r.Handle(w, req)

	if got := w.Result().StatusCode; got != http.StatusOK {
		t.Fatalf("got status %d, want 200", got)
	}
	if calls.Load() != 2 {
		t.Errorf("flaky server hit %d times, want 2 (1 + 1 retry from header)", calls.Load())
	}
	if fallbackCalled {
		t.Error("fallback server should NOT have been called when header retry recovered")
	}
}

func TestRateLimit429ImmediateCooldownSkipsServer(t *testing.T) {
	// A 429 response from the primary puts it into immediate cooldown; the
	// next request should skip it and go to the fallback.
	var primaryCalls atomic.Int32
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		primaryCalls.Add(1)
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`{"error":{"message":"rate limited","type":"rate_limit_error"}}`)) //nolint:errcheck
	}))
	defer primary.Close()

	var fallbackCalls atomic.Int32
	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fallbackCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"backup-model","choices":[{"finish_reason":"stop"}]}`)) //nolint:errcheck
	}))
	defer fallback.Close()

	r, store, _, _ := newTestRouter(t)

	// Build a router with a real rate limiter (newTestRouter passes nil).
	rl := config.NewRateLimiter(5, 60*time.Second, 5*time.Minute)
	r = New(store, r.metrics, nil, rl, nil, nil)

	_ = store.AddServer(&domain.Server{
		ID:       "primary",
		Name:     "Primary",
		URL:      primary.URL,
		APIKey:   "key1",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddServer(&domain.Server{
		ID:       "fallback",
		Name:     "Fallback",
		URL:      fallback.URL,
		APIKey:   "key2",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "primary",
		Fallbacks: []domain.FallbackEntry{
			{ServerID: "fallback", TargetModel: "backup-model"},
		},
		Enabled: true,
	})

	doRequest := func() *httptest.ResponseRecorder {
		body := strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
		w := httptest.NewRecorder()
		r.Handle(w, req)
		return w
	}

	// First request: 429 forwarded to client, primary enters immediate cooldown
	w1 := doRequest()
	if got := w1.Result().StatusCode; got != http.StatusTooManyRequests {
		t.Fatalf("first request status = %d, want 429", got)
	}

	// Second request: primary in cooldown → fallback serves it
	w2 := doRequest()
	if got := w2.Result().StatusCode; got != http.StatusOK {
		t.Fatalf("second request status = %d, want 200 (via fallback)", got)
	}
	if primaryCalls.Load() != 1 {
		t.Errorf("primary hit %d times, want 1 (skipped after cooldown)", primaryCalls.Load())
	}
	if fallbackCalls.Load() != 1 {
		t.Errorf("fallback hit %d times, want 1", fallbackCalls.Load())
	}
}

func TestQuotaTPMBlocksRoutesToFallback(t *testing.T) {
	// Primary exhausts its TPM quota on the first request; the second request
	// must be routed to the fallback.
	var primaryCalls atomic.Int32
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		primaryCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"gpt-4","usage":{"prompt_tokens":1000,"completion_tokens":0,"total_tokens":1000},"choices":[{"finish_reason":"stop"}]}`)) //nolint:errcheck
	}))
	defer primary.Close()

	var fallbackCalls atomic.Int32
	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fallbackCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"backup-model","choices":[{"finish_reason":"stop"}]}`)) //nolint:errcheck
	}))
	defer fallback.Close()

	r, store, _, _ := newTestRouter(t)

	// Build a router with a quota tracker (newTestRouter passes nil).
	q := config.NewQuotaTracker()
	q.SetLimitOverride(func(id string) (tpm, rpm int64) {
		if id == "primary" {
			return 1000, 0
		}
		return 0, 0
	})
	r = New(store, r.metrics, nil, nil, q, nil)

	_ = store.AddServer(&domain.Server{
		ID:       "primary",
		Name:     "Primary",
		URL:      primary.URL,
		APIKey:   "key1",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddServer(&domain.Server{
		ID:       "fallback",
		Name:     "Fallback",
		URL:      fallback.URL,
		APIKey:   "key2",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "primary",
		Fallbacks: []domain.FallbackEntry{
			{ServerID: "fallback", TargetModel: "backup-model"},
		},
		Enabled: true,
	})

	doRequest := func() *httptest.ResponseRecorder {
		body := strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
		w := httptest.NewRecorder()
		r.Handle(w, req)
		return w
	}

	// First request: primary serves it, uses 1000 tokens (at TPM boundary)
	w1 := doRequest()
	if got := w1.Result().StatusCode; got != http.StatusOK {
		t.Fatalf("first request status = %d, want 200", got)
	}

	// Second request: primary blocked by quota → fallback serves it
	w2 := doRequest()
	if got := w2.Result().StatusCode; got != http.StatusOK {
		t.Fatalf("second request status = %d, want 200 (via fallback)", got)
	}
	if primaryCalls.Load() != 1 {
		t.Errorf("primary hit %d times, want 1 (blocked by TPM quota)", primaryCalls.Load())
	}
	if fallbackCalls.Load() != 1 {
		t.Errorf("fallback hit %d times, want 1", fallbackCalls.Load())
	}
}

func TestRouterObservabilityHeaders(t *testing.T) {
	// Primary fails (transport) → fallback succeeds: response must carry
	// X-Router-Attempts=2/2, X-Router-Retries=0, X-Router-Fallback-Errors=[...].
	var fallbackCalled bool
	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fallbackCalled = true
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"backup-model","choices":[{"finish_reason":"stop"}]}`)) //nolint:errcheck
	}))
	defer fallback.Close()

	r, store, _, _ := newTestRouter(t)

	_ = store.AddServer(&domain.Server{
		ID:       "primary",
		Name:     "Primary",
		URL:      "http://localhost:1", // unreachable
		APIKey:   "key1",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddServer(&domain.Server{
		ID:       "fallback",
		Name:     "Fallback",
		URL:      fallback.URL,
		APIKey:   "key2",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"opus"},
		TargetModel:    "gpt-4",
		ServerID:       "primary",
		NumRetries:     1,
		Fallbacks: []domain.FallbackEntry{
			{ServerID: "fallback", TargetModel: "backup-model"},
		},
		Enabled: true,
	})

	body := strings.NewReader(`{"model":"opus","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("got status %d, want 200", resp.StatusCode)
	}
	if !fallbackCalled {
		t.Fatal("fallback should have been called")
	}

	if got := resp.Header.Get("X-Router-Attempts"); got != "2/2" {
		t.Errorf("X-Router-Attempts = %q, want 2/2", got)
	}
	if got := resp.Header.Get("X-Router-Retries"); got != "0" {
		t.Errorf("X-Router-Retries = %q, want 0", got)
	}
	if got := resp.Header.Get("X-Router-Fallback-Errors"); got == "" {
		t.Error("X-Router-Fallback-Errors should list the primary's error")
	}
	if got := resp.Header.Get("X-Router-Server"); got != "fallback" {
		t.Errorf("X-Router-Server = %q, want fallback", got)
	}
}

func TestRouterObservabilityHeadersRetriesRecovered(t *testing.T) {
	// Server fails once then succeeds: response must carry X-Router-Retries=1
	// and X-Router-Attempts=1/1 (no fallback).
	var calls atomic.Int32
	flaky := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			hj, ok := w.(http.Hijacker)
			if ok {
				conn, _, err := hj.Hijack()
				if err == nil {
					conn.Close() //nolint:errcheck
					return
				}
			}
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`)) //nolint:errcheck
	}))
	defer flaky.Close()

	r, store, _, _ := newTestRouter(t)

	_ = store.AddServer(&domain.Server{
		ID:       "flaky",
		Name:     "Flaky",
		URL:      flaky.URL,
		APIKey:   "key1",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "flaky",
		NumRetries:     2,
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("got status %d, want 200", resp.StatusCode)
	}
	if got := resp.Header.Get("X-Router-Attempts"); got != "1/1" {
		t.Errorf("X-Router-Attempts = %q, want 1/1", got)
	}
	if got := resp.Header.Get("X-Router-Retries"); got != "1" {
		t.Errorf("X-Router-Retries = %q, want 1", got)
	}
	if got := resp.Header.Get("X-Router-Fallback-Errors"); got != "" {
		t.Errorf("X-Router-Fallback-Errors should be empty, got %q", got)
	}
}

func TestHandleRetriesStreamErrorBody(t *testing.T) {
	// Regression: a backend that died mid-generation answers 5xx with a body
	// of buffered SSE frames. The router must retry it like a transport error
	// instead of forwarding the unparseable response to the client.
	var calls atomic.Int32
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(500)
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`))
	}))
	defer backend.Close()

	r, store, _, _ := newTestRouter(t)
	_ = store.AddServer(&domain.Server{
		ID:       "s1",
		Name:     "test-server",
		URL:      backend.URL,
		APIKey:   "key",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "s1",
		NumRetries:     1,
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("got status %d, want 200: %s", resp.StatusCode, w.Body.String())
	}
	if calls.Load() != 2 {
		t.Errorf("backend hit %d times, want 2 (1 + 1 retry)", calls.Load())
	}
	if got := resp.Header.Get("X-Router-Retries"); got != "1" {
		t.Errorf("X-Router-Retries = %q, want 1", got)
	}
}

func TestHandleFallbackOnStreamErrorBody(t *testing.T) {
	// 5xx with SSE body from the primary must trigger fallback to the next
	// server instead of being forwarded to the client.
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(500)
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"dead\"}}]}\n\n")) //nolint:errcheck
	}))
	defer primary.Close()

	backup := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"backup-model","choices":[{"finish_reason":"stop"}]}`)) //nolint:errcheck
	}))
	defer backup.Close()

	r, store, _, _ := newTestRouter(t)
	_ = store.AddServer(&domain.Server{
		ID:       "dead",
		Name:     "Dead",
		URL:      primary.URL,
		APIKey:   "key1",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddServer(&domain.Server{
		ID:       "backup",
		Name:     "Backup",
		URL:      backup.URL,
		APIKey:   "key2",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "dead",
		Fallbacks: []domain.FallbackEntry{
			{ServerID: "backup", TargetModel: "backup-model"},
		},
		Enabled: true,
	})

	body := strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	// Rule has no num_retries (now defaulted to 10); opt out to keep this
	// test fast — fallback must trigger on the first failed attempt.
	req.Header.Set("X-Router-Retries", "0")
	w := httptest.NewRecorder()
	r.Handle(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("got status %d, want 200 (via fallback): %s", resp.StatusCode, w.Body.String())
	}
	if got := resp.Header.Get("X-Router-Server"); got != "backup" {
		t.Errorf("X-Router-Server = %q, want backup", got)
	}
	if got := resp.Header.Get("X-Router-Fallback-Errors"); got == "" {
		t.Error("X-Router-Fallback-Errors should list the failed attempt")
	}
	if !strings.Contains(w.Body.String(), "backup-model") {
		t.Errorf("response model should not be rewritten on fallback, got %q", w.Body.String())
	}
}

func TestHandleDefaultNumRetries(t *testing.T) {
	// Rules without num_retries get defaultNumRetries: the router keeps
	// retrying a flaky backend up to the default instead of failing at once.
	var calls atomic.Int32
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) <= 2 {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(500)
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"die\"}}]}\n\n")) //nolint:errcheck
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`)) //nolint:errcheck
	}))
	defer backend.Close()

	r, store, _, _ := newTestRouter(t)
	_ = store.AddServer(&domain.Server{
		ID:       "s1",
		Name:     "test-server",
		URL:      backend.URL,
		APIKey:   "key",
		APITypes: []domain.APIType{domain.APITypeOpenAI},
	})
	_ = store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "s1",
		Enabled:        true,
	})

	body := strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	w := httptest.NewRecorder()
	r.Handle(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("got status %d, want 200 after default retries: %s", resp.StatusCode, w.Body.String())
	}
	if calls.Load() != 3 {
		t.Errorf("backend hit %d times, want 3 (2 failures + success via defaults)", calls.Load())
	}
	if got := resp.Header.Get("X-Router-Retries"); got != "2" {
		t.Errorf("X-Router-Retries = %q, want 2", got)
	}
}

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
