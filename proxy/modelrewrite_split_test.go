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

// TestModelRewriteLogTruncationDoesNotCorruptChunk reproduces a production
// regression: a coalesced upstream read (>256 bytes) that contains the
// substring "model" (here as a delta value) but no "model":"<oldModel>" pair
// triggers the debug-log path, which calls truncateBytes(data, 256). The old
// truncateBytes did append(b[:256], ...), writing "..." into the live buffer
// past offset 256 — clobbering an item_id the client then failed to parse.
// The forwarded bytes must be byte-identical to the upstream bytes.
func TestModelRewriteLogTruncationDoesNotCorruptChunk(t *testing.T) {
	oldModel := "unsloth/Qwen3.8-27B-GGUF:Q4_K_XL"
	newModel := "sonnet"

	itemID := "rs_e9YT5vRymLtc3QokD3rKbzUX1dN1cq5S"
	frame := func(delta string) string {
		return "event: response.reasoning_text.delta\n" +
			"data: {\"type\":\"response.reasoning_text.delta\",\"delta\":\"" + delta +
			"\",\"item_id\":\"" + itemID + "\"}\n\n"
	}
	chunk := frame("model") + frame("Rew")
	if len(chunk) <= 256 {
		t.Fatalf("test chunk must exceed 256 bytes to hit the truncate path, got %d", len(chunk))
	}
	// The second frame's item_id must land at/after offset 256 so a mutating
	// truncate would visibly corrupt it.
	if idx := strings.Index(chunk[256:], itemID); idx < 0 {
		t.Fatalf("item_id is not within chunk[256:]; test premise broken")
	}

	rec := httptest.NewRecorder()
	relay := newSSERelay(rec, false)
	mw := newMetricsWriter(relay, time.Now())
	mrw := newModelRewriteWriter(mw, oldModel, newModel)
	if _, err := mrw.Write([]byte(chunk)); err != nil {
		t.Fatalf("Write: %v", err)
	}
	mrw.Finish()
	relay.finish()
	got := rec.Body.String()
	if got != chunk {
		t.Fatalf("forwarded bytes differ from upstream (log truncation mutated the buffer):\n got: %s\nwant: %s", got, chunk)
	}
}

// TestTruncateBytesDoesNotMutateInput is the tight guard on the helper itself:
// the returned prefix must never overwrite the tail of its input slice.
func TestTruncateBytesDoesNotMutateInput(t *testing.T) {
	in := make([]byte, 300)
	for i := range in {
		in[i] = 'x'
	}
	out := truncateBytes(in, 256)
	if len(out) != 259 || string(out[:256]) != strings.Repeat("x", 256) || string(out[256:]) != "..." {
		t.Fatalf("unexpected output: len=%d tail=%q", len(out), out[254:])
	}
	for i := 256; i < len(in); i++ {
		if in[i] != 'x' {
			t.Fatalf("truncateBytes mutated its input at index %d: %q", i, in[i])
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
