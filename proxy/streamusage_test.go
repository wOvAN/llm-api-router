package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestEnsureStreamUsageInjectsIncludeUsage(t *testing.T) {
	body := []byte(`{"model":"gpt-4","stream":true}`)
	rewritten, strip := EnsureStreamUsage(body, false)
	if !strip {
		t.Fatal("expected strip=true when injecting for a client that did not request usage")
	}

	var obj map[string]interface{}
	if err := json.Unmarshal(rewritten, &obj); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	so, ok := obj["stream_options"].(map[string]interface{})
	if !ok {
		t.Fatal("stream_options not injected")
	}
	if inc, _ := so["include_usage"].(bool); !inc {
		t.Error("include_usage should be true")
	}
}

func TestEnsureStreamUsagePreservesExistingStreamOptions(t *testing.T) {
	body := []byte(`{"model":"gpt-4","stream":true,"stream_options":{"foo":"bar"}}`)
	rewritten, _ := EnsureStreamUsage(body, false)

	var obj map[string]interface{}
	_ = json.Unmarshal(rewritten, &obj)
	so := obj["stream_options"].(map[string]interface{})
	if so["foo"] != "bar" {
		t.Errorf("existing stream_options.foo lost: %v", so)
	}
	if inc, _ := so["include_usage"].(bool); !inc {
		t.Error("include_usage should be injected alongside existing options")
	}
}

func TestEnsureStreamUsage_ClientAlreadyRequested(t *testing.T) {
	body := []byte(`{"model":"gpt-4","stream":true,"stream_options":{"include_usage":true}}`)
	rewritten, strip := EnsureStreamUsage(body, false)
	if strip {
		t.Error("should not strip when the client requested include_usage itself")
	}
	if string(rewritten) != string(body) {
		t.Errorf("body should be unchanged, got: %s", rewritten)
	}
}

func TestEnsureStreamUsage_NonStreamingUntouched(t *testing.T) {
	body := []byte(`{"model":"gpt-4","stream":false}`)
	rewritten, strip := EnsureStreamUsage(body, false)
	if strip {
		t.Error("non-streaming request must not enable strip")
	}
	if string(rewritten) != string(body) {
		t.Errorf("non-streaming body should be unchanged, got: %s", rewritten)
	}
}

func TestEnsureStreamUsage_AlwaysIncludeForwards(t *testing.T) {
	body := []byte(`{"model":"gpt-4","stream":true}`)
	rewritten, strip := EnsureStreamUsage(body, true)
	if strip {
		t.Error("always_include_stream_usage=true should forward the usage chunk, not strip")
	}
	var obj map[string]interface{}
	_ = json.Unmarshal(rewritten, &obj)
	so := obj["stream_options"].(map[string]interface{})
	if inc, _ := so["include_usage"].(bool); !inc {
		t.Error("include_usage should still be injected upstream when always-include is set")
	}
}

func TestEnsureStreamUsage_MalformedBodyUntouched(t *testing.T) {
	body := []byte(`not json`)
	rewritten, strip := EnsureStreamUsage(body, false)
	if strip {
		t.Error("malformed body should not enable strip")
	}
	if string(rewritten) != string(body) {
		t.Error("malformed body must be returned unchanged")
	}
}

func TestUsageStripperRemovesInjectedArtifact(t *testing.T) {
	rec := httptest.NewRecorder()
	st := newUsageStripper(rec)

	stream := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"hello"}}]}`,
		`data: {"choices":[{"delta":{"content":" world"}}]}`,
		`data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
		`data: [DONE]`,
	}, "\n\n") + "\n\n"

	if _, err := st.Write([]byte(stream)); err != nil {
		t.Fatalf("write: %v", err)
	}
	st.Flush()

	out := rec.Body.String()
	if strings.Contains(out, "usage") {
		t.Errorf("injected usage artifact not stripped:\n%s", out)
	}
	if !strings.Contains(out, "hello") || !strings.Contains(out, " world") {
		t.Errorf("real content missing:\n%s", out)
	}
	if !strings.Contains(out, "[DONE]") {
		t.Errorf("[DONE] frame should be forwarded:\n%s", out)
	}
}

func TestUsageStripperSplitFrames(t *testing.T) {
	rec := httptest.NewRecorder()
	st := newUsageStripper(rec)

	// Usage artifact split across two Write calls mid-frame.
	prefix := `data: {"choices":[{"delta":{"content":"ok"}}]}` + "\n\ndata: {\"choices\":[],\"us"
	mid := `age":{"prompt_tokens":1}}`
	suffix := "\n\n"

	if _, err := st.Write([]byte(prefix)); err != nil {
		t.Fatal(err)
	}
	if _, err := st.Write([]byte(mid)); err != nil {
		t.Fatal(err)
	}
	if _, err := st.Write([]byte(suffix)); err != nil {
		t.Fatal(err)
	}
	st.Flush()

	out := rec.Body.String()
	if strings.Contains(out, "usage") {
		t.Errorf("usage artifact split across writes not stripped:\n%s", out)
	}
	if !strings.Contains(out, `"ok"`) {
		t.Errorf("content before artifact missing:\n%s", out)
	}
}

func TestUsageStripperDroppedCounting(t *testing.T) {
	rec := httptest.NewRecorder()
	st := newUsageStripper(rec)
	in := []string{`data: {"choices":[{"delta":{"content":"x"}}]}`, `data: {"choices":[],"usage":{}}`}
	payload := strings.Join(in, "\n\n") + "\n\n"

	n, err := st.Write([]byte(payload))
	if err != nil {
		t.Fatalf("write: %v", err)
	}
	if n != len(payload) {
		t.Errorf("Write should report full length even when dropping frames, got %d want %d", n, len(payload))
	}
}

// TestStreamProxyStripUsageIntegration verifies end-to-end that a streamed
// response with an injected usage artifact reaches the client without the usage
// chunk, while ProxyMetrics still reports usage.
func TestStreamProxyStripUsageIntegration(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		body := strings.Join([]string{
			`data: {"choices":[{"delta":{"content":"hi"}}]}`,
			`data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`,
			`data: [DONE]`,
		}, "\n\n") + "\n\n"
		_, _ = w.Write([]byte(body))
	}))

	defer backend.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	w := httptest.NewRecorder()

	pm, err := StreamProxy(req.Context(), backend.URL, "key", req, w, "gpt-4", "gpt-4", nil, true)
	if err != nil {
		t.Fatalf("StreamProxy error: %v", err)
	}
	if pm == nil {
		t.Fatal("pm is nil")
	}

	if strings.Contains(w.Body.String(), "usage") {
		t.Errorf("client stream contains usage artifact:\n%s", w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "hi") {
		t.Errorf("client stream missing content:\n%s", w.Body.String())
	}

	if pm.PromptTokens != 3 || pm.CompletionTokens != 1 {
		t.Errorf("metrics should capture usage despite stripping, got: %+v", pm)
	}
}

func TestEnsureStreamUsageRouterIntegration(t *testing.T) {
	// The router chain: EnsureStreamUsage must be called before RewriteModelInBody
	// so the stream_options survive the per-attempt model rewrite. Exercise at
	// the proxy level: rewrite a body that already has injected stream_options.
	body, _ := EnsureStreamUsage([]byte(`{"model":"gpt-4","stream":true}`), false)
	rewritten, err := RewriteModelInBody(body, "sonnet")
	if err != nil {
		t.Fatalf("rewrite after inject: %v", err)
	}
	var obj map[string]interface{}
	_ = json.Unmarshal(rewritten, &obj)
	so, ok := obj["stream_options"].(map[string]interface{})
	if !ok {
		t.Fatal("stream_options lost during model rewrite")
	}
	if inc, _ := so["include_usage"].(bool); !inc {
		t.Error("include_usage lost during model rewrite")
	}
	if obj["model"] != "sonnet" {
		t.Errorf("model not rewritten, got %v", obj["model"])
	}
}

var _ = context.Background

// TestUsageStripperMidStreamFlushKeepsFramesIntact guards the regression fixed
// alongside per-chunk flushing: a mid-stream Flush must NOT push a partial data
// line to the client. Fragmenting a frame across writes is exactly what makes
// clients glue the tail of one JSON onto the start of the next.
func TestUsageStripperMidStreamFlushKeepsFramesIntact(t *testing.T) {
	rec := httptest.NewRecorder()
	st := newUsageStripper(rec)

	// One data line split across two writes, flushed in between (as the proxy
	// copy loop now does per chunk).
	part1 := `data: {"choices":[{"delta":{"content":"hello`
	part2 := ` world"}}]}` + "\n\n"

	if _, err := st.Write([]byte(part1)); err != nil {
		t.Fatal(err)
	}
	st.Flush()
	if got := rec.Body.String(); got != "" {
		t.Fatalf("mid-stream flush leaked a partial frame: %q", got)
	}
	if _, err := st.Write([]byte(part2)); err != nil {
		t.Fatal(err)
	}
	st.Flush()
	want := `data: {"choices":[{"delta":{"content":"hello world"}}]}` + "\n\n"
	if got := rec.Body.String(); got != want {
		t.Errorf("client got fragmented frame:\ngot:  %q\nwant: %q", got, want)
	}
}

// bufWriter mimics net/http's ~4KB response buffering: writes accumulate until
// Flush pushes them out. With it we can observe mid-stream delivery, which a
// plain httptest.ResponseRecorder cannot (its writes are immediate).
type bufWriter struct {
	mu      sync.Mutex
	buf     bytes.Buffer
	flushed bytes.Buffer
}

func (l *bufWriter) Header() http.Header { return make(http.Header) }
func (l *bufWriter) WriteHeader(int)     {}
func (l *bufWriter) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.buf.Write(p)
	return len(p), nil
}
func (l *bufWriter) Flush() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.flushed.Write(l.buf.Bytes())
	l.buf.Reset()
}
func (l *bufWriter) content() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.flushed.String() + l.buf.String()
}

// TestStreamProxyFlushesPerChunk verifies the router delivers each SSE frame as
// it is produced. The backend writes frame 1, flushes, and blocks until the
// client has actually received it before sending frame 2. If the router did not
// flush per chunk, the client never sees frame 1 until the stream ends, the
// backend times out, and sawSignal never fires.
func TestStreamProxyFlushesPerChunk(t *testing.T) {
	received := make(chan struct{})
	sawSignal := make(chan struct{}) // backend observed the client receiving frame 1
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`data: {"choices":[{"delta":{"content":"first"}}]}` + "\n\n"))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		select {
		case <-received:
			close(sawSignal)
		case <-time.After(2 * time.Second):
			return // client never got frame 1 mid-stream — router is not flushing
		}
		_, _ = w.Write([]byte(`data: [DONE]` + "\n\n"))
	}))
	defer backend.Close()

	client := &bufWriter{}
	clientW := &flushNotifyingWriter{ResponseWriter: client, onFlush: func() {
		if strings.Contains(client.content(), "first") {
			select {
			case <-received:
			default:
				close(received)
			}
		}
	}}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
	pm, err := StreamProxy(req.Context(), backend.URL, "key", req, clientW, "gpt-4", "gpt-4", nil, true)
	if err != nil {
		t.Fatalf("StreamProxy: %v", err)
	}
	if pm == nil {
		t.Fatal("pm is nil")
	}

	select {
	case <-sawSignal:
	default:
		t.Error("first frame was not flushed to the client before the backend proceeded")
	}
	if !strings.Contains(client.content(), "first") || !strings.Contains(client.content(), "[DONE]") {
		t.Errorf("client stream incomplete:\n%s", client.content())
	}
}

// flushNotifyingWriter wraps a writer and calls onFlush every time Flush is
// invoked, so tests can react to delivery without peeking at socket internals.
type flushNotifyingWriter struct {
	http.ResponseWriter
	onFlush func()
}

func (f *flushNotifyingWriter) Flush() {
	f.onFlush()
	if flusher, ok := f.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

// flushCaptureWriter records the bytes delivered between flush calls, so tests
// can assert delivery granularity (one SSE frame per delivery).
type flushCaptureWriter struct {
	deliveries [][]byte
	cur        []byte
}

func (l *flushCaptureWriter) Header() http.Header { return make(http.Header) }
func (l *flushCaptureWriter) WriteHeader(int)     {}
func (l *flushCaptureWriter) Write(p []byte) (int, error) {
	l.cur = append(l.cur, p...)
	return len(p), nil
}
func (l *flushCaptureWriter) Flush() {
	if len(l.cur) > 0 {
		l.deliveries = append(l.deliveries, l.cur)
		l.cur = nil
	}
}

// TestStreamProxyFlushesPerCoalescedRead guards the opencode #7692-style case
// the per-read flush cannot fix on its own: the backend writes several SSE
// frames in a single write (llama.cpp batches finish+usage; TCP merges token
// chunks), so they arrive as one read and previously went to the client as one
// write. Clients that treat one delivery as one SSE event then glue the tail of
// one JSON onto the start of the next. The router must re-frame: every
// complete frame delivered as its own write+flush, in both the stripping and
// non-stripping paths.
func TestStreamProxyFlushesPerCoalescedRead(t *testing.T) {
	stream := strings.Join([]string{
		`data: {"choices":[{"finish_reason":null,"index":0,"delta":{"reasoning_content":"` + "`" + `"}}],"created":1786000887,"id":"chatcmpl-1","model":"haiku"}`,
		`data: {"choices":[],"created":1786000887,"id":"chatcmpl-1","model":"haiku","usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`,
		`data: [DONE]`,
	}, "\n\n") + "\n\n"

	for _, tc := range []struct {
		name          string
		strip         bool
		wantDeliver   int
		wantUsageSeen bool
	}{
		{name: "strip", strip: true, wantDeliver: 2, wantUsageSeen: false},
		{name: "no-strip", strip: false, wantDeliver: 3, wantUsageSeen: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(200)
				_, _ = w.Write([]byte(stream)) // all frames in one write
			}))
			defer backend.Close()

			client := &flushCaptureWriter{}
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4"}`))
			pm, err := StreamProxy(req.Context(), backend.URL, "key", req, client, "gpt-4", "gpt-4", nil, tc.strip)
			if err != nil {
				t.Fatalf("StreamProxy: %v", err)
			}
			if pm == nil {
				t.Fatal("pm is nil")
			}

			if got, want := len(client.deliveries), tc.wantDeliver; got != want {
				t.Fatalf("expected %d separate deliveries, got %d: %q", want, got, client.deliveries)
			}
			for i, d := range client.deliveries {
				if got := strings.Count(string(d), "data:"); got != 1 {
					t.Errorf("delivery %d contains %d frames, want exactly 1: %q", i, got, d)
				}
			}
			joined := strings.Join(func() []string {
				out := make([]string, len(client.deliveries))
				for i, d := range client.deliveries {
					out[i] = string(d)
				}
				return out
			}(), "")
			if got := strings.Contains(joined, "usage"); got != tc.wantUsageSeen {
				t.Errorf("usage seen by client = %v, want %v", got, tc.wantUsageSeen)
			}
		})
	}
}

// TestSSEFrameWriterDeliversEachFrameSeparately guards the multi-frame
// coalescing case: when the upstream sends several SSE frames in one read
// (llama.cpp batches finish+usage, TCP merges token chunks), each frame must
// still reach the client as its own write+flush. Clients that treat one
// delivery as one SSE event mis-split frames that arrive in a single lump.
func TestSSEFrameWriterDeliversEachFrameSeparately(t *testing.T) {
	rec := &flushCaptureWriter{}
	fw := newSSEFrameWriter(rec)

	stream := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"one"}}]}`,
		`data: {"choices":[{"delta":{"content":"two"}}]}`,
		`data: [DONE]`,
	}, "\n\n") + "\n\n"

	if _, err := fw.Write([]byte(stream)); err != nil {
		t.Fatal(err)
	}

	if got, want := len(rec.deliveries), 3; got != want {
		t.Fatalf("expected %d separate deliveries, got %d: %q", want, got, rec.deliveries)
	}
	for i, d := range rec.deliveries {
		if got := strings.Count(string(d), "data:"); got != 1 {
			t.Errorf("delivery %d contains %d frames, want exactly 1: %q", i, got, d)
		}
	}
}

// TestSSEFrameWriterHoldsPartialFrames verifies a data line split across
// writes is not forwarded until the frame is complete, and mid-stream flushes
// do not leak fragments to the client.
func TestSSEFrameWriterHoldsPartialFrames(t *testing.T) {
	rec := &flushCaptureWriter{}
	fw := newSSEFrameWriter(rec)

	if _, err := fw.Write([]byte(`data: {"choices":[{"delta":{"content":"hel`)); err != nil {
		t.Fatal(err)
	}
	fw.Flush()
	if got := len(rec.deliveries); got != 0 {
		t.Fatalf("partial frame leaked to client: %q", rec.deliveries)
	}

	if _, err := fw.Write([]byte(`lo"}}]}` + "\n\n")); err != nil {
		t.Fatal(err)
	}
	if got, want := len(rec.deliveries), 1; got != want {
		t.Fatalf("expected %d delivery after frame completed, got %d: %q", want, got, rec.deliveries)
	}
	if !bytes.Contains(rec.deliveries[0], []byte("hello")) {
		t.Errorf("frame content wrong: %q", rec.deliveries[0])
	}
}

// TestSSEFrameWriterFinishDeliversTrailingBytes verifies a trailing frame the
// backend closed without a blank-line terminator is delivered at stream end.
func TestSSEFrameWriterFinishDeliversTrailingBytes(t *testing.T) {
	rec := &flushCaptureWriter{}
	fw := newSSEFrameWriter(rec)

	if _, err := fw.Write([]byte(`data: {"choices":[{"delta":{"content":"tail"}}]}`)); err != nil {
		t.Fatal(err)
	}
	if got := len(rec.deliveries); got != 0 {
		t.Fatalf("unterminated frame delivered mid-stream: %q", rec.deliveries)
	}
	fw.finish()
	if got := len(rec.deliveries); got != 1 {
		t.Fatalf("finish should deliver the trailing frame, got %d deliveries: %q", got, rec.deliveries)
	}
	if !bytes.Contains(rec.deliveries[0], []byte("tail")) {
		t.Errorf("trailing frame content wrong: %q", rec.deliveries[0])
	}
}
