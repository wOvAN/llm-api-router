package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// TestModelRewriteSurvivesChunkBoundary guards against a model name that
// straddles an upstream read boundary being left un-rewritten: the client must
// always see the original model name, never the backend's, regardless of how
// the stream is chunked.
func TestModelRewriteSurvivesChunkBoundary(t *testing.T) {
	frame := "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}],\"created\":1787509560,\"id\":\"chatcmpl-X\",\"model\":\"unsloth/Qwen3.8-27B-GGUF:BF16\",\"object\":\"chat.completion.chunk\"}\n\n"
	oldModel := "unsloth/Qwen3.8-27B-GGUF:BF16"
	newModel := "opus"

	run := func(chunks ...string) string {
		rec := httptest.NewRecorder()
		relay := newSSERelay(rec, false)
		mw := newMetricsWriter(relay, time.Now())
		mrw := newModelRewriteWriter(mw, oldModel, newModel)
		for _, c := range chunks {
			_, _ = mrw.Write([]byte(c))
		}
		mrw.Finish()
		relay.finish()
		return rec.Body.String()
	}

	want := run(frame)
	if !strings.Contains(want, `"model":"opus"`) || strings.Contains(want, oldModel) {
		t.Fatalf("whole-frame result wrong:\n%s", want)
	}

	idx := strings.Index(frame, oldModel)
	for off := 1; off < len(oldModel); off++ {
		got := run(frame[:idx+off], frame[idx+off:])
		if got != want {
			t.Fatalf("split inside model name (offset %d) changed the result:\n got: %s\nwant: %s", off, got, want)
		}
	}
}

// TestModelRewriteAnyValueSurvivesChunkBoundary covers the oldModel==newModel
// path (backend returns its own name) across a chunk boundary.
func TestModelRewriteAnyValueSurvivesChunkBoundary(t *testing.T) {
	frame := "data: {\"model\":\"ornith-ai/Ornith-1.5-35B-A3B-GGUF:Q4_K_M\",\"id\":\"m1\"}\n\n"
	backendModel := "ornith-ai/Ornith-1.5-35B-A3B-GGUF:Q4_K_M"
	clientModel := "haiku"

	run := func(chunks ...string) string {
		rec := httptest.NewRecorder()
		relay := newSSERelay(rec, false)
		mw := newMetricsWriter(relay, time.Now())
		mrw := newModelRewriteWriter(mw, clientModel, clientModel)
		for _, c := range chunks {
			_, _ = mrw.Write([]byte(c))
		}
		mrw.Finish()
		relay.finish()
		return rec.Body.String()
	}

	want := run(frame)
	if !strings.Contains(want, `"model":"haiku"`) || strings.Contains(want, backendModel) {
		t.Fatalf("whole-frame result wrong:\n%s", want)
	}

	idx := strings.Index(frame, backendModel)
	for off := 1; off < len(backendModel); off++ {
		got := run(frame[:idx+off], frame[idx+off:])
		if got != want {
			t.Fatalf("split inside model name (offset %d) changed the result:\n got: %s\nwant: %s", off, got, want)
		}
	}
}

// TestStreamProxyRewritesModelSplitAcrossReads is the end-to-end guard: a real
// StreamProxy with a backend that delivers a frame split inside the model name
// (as a slow backend + TCP segmentation would). The client must see the
// original model name, never the backend's.
func TestStreamProxyRewritesModelSplitAcrossReads(t *testing.T) {
	frame := "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}],\"id\":\"c1\",\"model\":\"unsloth/Qwen3.8-27B-GGUF:BF16\",\"object\":\"chat.completion.chunk\"}\n\ndata: [DONE]\n\n"
	oldModel := "unsloth/Qwen3.8-27B-GGUF:BF16"
	idx := strings.Index(frame, oldModel)

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		_, _ = w.Write([]byte(frame[:idx+5]))
		w.(http.Flusher).Flush()
		_, _ = w.Write([]byte(frame[idx+5:]))
	}))
	defer backend.Close()

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"opus"}`))
	_, err := StreamProxy(req.Context(), backend.URL, "key", req, rec, oldModel, "opus", nil, false, nil)
	if err != nil {
		t.Fatalf("StreamProxy: %v", err)
	}
	out := rec.Body.String()
	if strings.Contains(out, oldModel) {
		t.Errorf("client saw raw backend model name (rewrite missed on split):\n%s", out)
	}
	if !strings.Contains(out, `"model":"opus"`) {
		t.Errorf("client did not see the rewritten model:\n%s", out)
	}
}
