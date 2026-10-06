package proxy

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestAskJev(t *testing.T) {
	var gotPath, gotAuth, gotAPIKey, gotContentType string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		gotAPIKey = r.Header.Get("x-api-key")
		gotContentType = r.Header.Get("Content-Type")
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"laya","answers":{` +
			`"model":{"type":"choice","choice":"haiku","confidence":0.62,"probabilities":{"haiku":0.62,"opus":0.38}},` +
			`"risk":{"type":"noul","noul":0.04}},` +
			`"usage":{"input_tokens":40,"output_tokens":3}}`))
	}))
	defer srv.Close()

	questions := map[string]JevQuestion{
		"model": {Type: "choice", Instructions: "pick one", Criteria: map[string]any{"haiku": "cheap"}},
	}
	resp, err := AskJev(context.Background(), srv.URL+"/v1", "secret", "laya",
		map[string]any{"request": "hi"}, questions, "", time.Second)
	if err != nil {
		t.Fatalf("AskJev: %v", err)
	}

	if gotPath != "/v1/systemone" {
		t.Errorf("path = %q, want /v1/systemone", gotPath)
	}
	if gotAuth != "Bearer secret" || gotAPIKey != "secret" {
		t.Errorf("auth headers = %q / %q", gotAuth, gotAPIKey)
	}
	if gotContentType != "application/json" {
		t.Errorf("content-type = %q", gotContentType)
	}
	if gotBody["model"] != "laya" {
		t.Errorf("body model = %v, want laya", gotBody["model"])
	}
	if _, ok := gotBody["state"]; !ok {
		t.Errorf("body has no state: %v", gotBody)
	}
	qs, _ := gotBody["questions"].(map[string]any)
	q, _ := qs["model"].(map[string]any)
	if q == nil || q["type"] != "choice" || q["instructions"] != "pick one" {
		t.Errorf("questions.model = %v", qs)
	}
	if _, ok := q["criteria"].(map[string]any); !ok {
		t.Errorf("questions.model.criteria = %v", q["criteria"])
	}

	if resp.Model != "laya" {
		t.Errorf("response model = %q", resp.Model)
	}
	ans, ok := resp.Answer("model")
	if !ok {
		t.Fatalf("no answer for %q: %+v", "model", resp.Answers)
	}
	if ans.Choice != "haiku" || ans.Confidence != 0.62 || ans.Probabilities["opus"] != 0.38 {
		t.Errorf("choice answer = %+v", ans)
	}
	noul, ok := resp.Answer("risk")
	if !ok || !noul.HasNoul || noul.Noul != 0.04 {
		t.Errorf("noul answer = %+v", noul)
	}
	if resp.InputTokens != 40 || resp.OutputTokens != 3 {
		t.Errorf("usage = %d/%d, want 40/3", resp.InputTokens, resp.OutputTokens)
	}
}

// A Choice answer without an explicit confidence reports the winning option's
// probability instead.
func TestAskJevConfidenceFromProbabilities(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"answers":{"model":{"type":"choice","choice":"opus","probabilities":{"haiku":0.2,"opus":0.5}}}}`))
	}))
	defer srv.Close()

	resp, err := AskJev(context.Background(), srv.URL, "", "", nil,
		map[string]JevQuestion{"model": {Type: "choice"}}, "", time.Second)
	if err != nil {
		t.Fatalf("AskJev: %v", err)
	}
	ans, _ := resp.Answer("model")
	if ans.Confidence != 0.5 {
		t.Errorf("confidence = %v, want 0.5", ans.Confidence)
	}
}

func TestAskJevEndpointError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "no such model", http.StatusNotFound)
	}))
	defer srv.Close()

	_, err := AskJev(context.Background(), srv.URL, "", "", nil,
		map[string]JevQuestion{"model": {Type: "choice"}}, "", time.Second)
	if err == nil || !strings.Contains(err.Error(), "404") {
		t.Fatalf("err = %v, want a 404 endpoint error", err)
	}
}
