package config

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"llm-api-router/domain"
)

func TestCapabilitiesCacheVision(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"data": [
				{"id": "haiku", "architecture": {"input_modalities": ["text", "image"]}},
				{"id": "sonnet", "architecture": {"input_modalities": ["text"]}},
				{"id": "mystery"}
			]
		}`))
	}))
	t.Cleanup(srv.Close)

	c := NewCapabilitiesCache()
	server := &domain.Server{ID: "s1", Name: "s1", URL: srv.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}}

	if v, ok := c.Vision(context.Background(), server, "haiku"); !ok || !v {
		t.Errorf("haiku = %v/%v, want known vision", v, ok)
	}
	if v, ok := c.Vision(context.Background(), server, "sonnet"); !ok || v {
		t.Errorf("sonnet = %v/%v, want known blind", v, ok)
	}
	if v, ok := c.Vision(context.Background(), server, "mystery"); ok || v {
		t.Errorf("mystery = %v/%v, want unknown", v, ok)
	}
}

func TestCapabilitiesCacheVLLModalities(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"data": [
				{"id": "vision", "supported_modalities": ["text", "image"]},
				{"id": "text", "supported_modalities": ["text"]}
			]
		}`))
	}))
	t.Cleanup(srv.Close)

	c := NewCapabilitiesCache()
	server := &domain.Server{ID: "s1", Name: "s1", URL: srv.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}}

	if v, ok := c.Vision(context.Background(), server, "vision"); !ok || !v {
		t.Errorf("vision = %v/%v, want known vision", v, ok)
	}
	if v, ok := c.Vision(context.Background(), server, "text"); !ok || v {
		t.Errorf("text = %v/%v, want known blind", v, ok)
	}
}

func TestCapabilitiesCacheNegativeCachesFailure(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)

	c := NewCapabilitiesCache()
	server := &domain.Server{ID: "s1", Name: "s1", URL: srv.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}}

	if v, ok := c.Vision(context.Background(), server, "model"); ok || v {
		t.Fatalf("first probe = %v/%v, want unknown", v, ok)
	}
	if v, ok := c.Vision(context.Background(), server, "model"); ok || v {
		t.Fatalf("second probe = %v/%v, want unknown from cache", v, ok)
	}
	if calls != 1 {
		t.Errorf("probe calls = %d, want 1 (negative cache)", calls)
	}
}
