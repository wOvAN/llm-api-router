package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"llm-api-router/pkg/log"
)

// ResponseHeaderTimeout bounds how long StreamProxy waits for the upstream to
// send response headers after the request is fully written. A backend that
// accepts the connection but never responds (wedged/zombie worker) otherwise
// hangs the request indefinitely — and because the upstream request is detached
// from client cancellation (see StreamProxy), a client with no deadline would
// hang forever, holding a goroutine and a pooled connection. 0 disables the
// bound (net/http default: wait forever). It only covers the header wait, not
// the streamed body, so long generations are unaffected.
var ResponseHeaderTimeout = 5 * time.Minute

// upstreamTransport is a shared HTTP transport with connection pooling.
// Reused across all proxied requests to enable TCP connection reuse,
// TLS session resumption, and keep-alive. Eliminates ~50-200ms per-request
// overhead from new TCP+TLS handshakes.
var upstreamTransport = &http.Transport{
	MaxIdleConns:          200,
	MaxIdleConnsPerHost:   50,
	IdleConnTimeout:       90 * time.Second,
	ResponseHeaderTimeout: ResponseHeaderTimeout,
	DialContext: (&net.Dialer{
		Timeout:   10 * time.Second,
		KeepAlive: 30 * time.Second,
	}).DialContext,
	TLSHandshakeTimeout: 10 * time.Second,
}

// upstreamClient is the shared HTTP client for all upstream requests.
var upstreamClient = &http.Client{Transport: upstreamTransport}

// KeepAliveIdle is the upstream-silence threshold after which StreamProxy
// injects a heartbeat frame into an SSE client stream. Long inference pauses
// (backend thinking between tokens, batched serving) otherwise trip client-side
// read timeouts ("waiting for api responses", "api timeout"). Exported so
// tests (and operators) can tune it.
var KeepAliveIdle = 15 * time.Second

// UpstreamDrainTimeout caps how long a handler keeps draining the upstream
// response body after its client has disconnected. Without the drain, the
// connection closes abruptly and a reverse-proxy backend (llama.cpp router
// mode) interprets it as client cancellation, aborting the generation that the
// client retries seconds later (wasted work, cold-start restart). 0 disables
// draining (legacy behavior: close upstream immediately).
var UpstreamDrainTimeout = 10 * time.Minute

// WaitSignalIdle is the silence threshold after which StreamProxy injects a
// "still waiting" SSE event into an SSE client stream. Long backend think
// times (slow models, large prompts) otherwise trip client-side read timeouts
// ("Request timed out", "api timeout") even though the proxy is working. The
// signal is a custom `event: router_wait` frame that Anthropic/OpenAI SDKs
// ignore (they only parse known event types), but it keeps the TCP read side
// of the client socket alive so the SDK does not give up. Exported so tests
// (and operators) can tune it.
var WaitSignalIdle = 30 * time.Second

// silenceWriter injects a signal frame into an SSE stream when the upstream
// stays silent for idle. Two instances sit in the writer chain with different
// jobs: the bottom-most one (directly on the client writer, so it can never be
// glued onto a partial frame) sends protocol heartbeats after KeepAliveIdle of
// silence; the one above the frame buffer sends "still waiting" signals after
// WaitSignalIdle. Both use the same protocol-appropriate frame — the standard
// Anthropic `event: ping` (recognized by SDKs to reset internal timeouts) or
// an OpenAI SSE comment line, which every SSE parser ignores but keeps TCP
// alive. Never send an event-bearing frame on an OpenAI stream — strict
// clients (opencode) validate every event as a chat chunk and reject unknown
// payloads. Both are only armed for text/event-stream responses (JSON bodies
// must not be polluted).
//
// Start/Stop/Write/Flush are mutex-guarded because the signal timer fires from
// its own goroutine while the relay loop writes from another. A fresh
// AfterFunc timer is created instead of Reset so re-arming from inside the
// callback (signalTick) can never deadlock on Reset's wait.
type silenceWriter struct {
	http.ResponseWriter
	mu        sync.Mutex
	idle      time.Duration
	signal    []byte
	timer     *time.Timer
	started   bool
	lastWrite time.Time
}

func newSilenceWriter(w http.ResponseWriter, idle time.Duration, signal []byte) *silenceWriter {
	return &silenceWriter{ResponseWriter: w, idle: idle, signal: signal}
}

// Start arms the signal timer. Must be called once the response is known to be
// an SSE stream; before Start no signals are sent.
func (s *silenceWriter) Start() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.started {
		return
	}
	s.started = true
	s.lastWrite = time.Now()
	s.armLocked()
}

// Stop disarms the timer; safe to call multiple times.
func (s *silenceWriter) Stop() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.started = false
	if s.timer != nil {
		s.timer.Stop()
	}
}

func (s *silenceWriter) Write(data []byte) (int, error) {
	s.mu.Lock()
	s.lastWrite = time.Now()
	if s.started {
		s.armLocked()
	}
	n, err := s.ResponseWriter.Write(data)
	s.mu.Unlock()
	return n, err
}

func (s *silenceWriter) Flush() {
	s.mu.Lock()
	if s.started {
		s.armLocked()
	}
	s.mu.Unlock()
	if f, ok := s.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// armLocked (re)arms the timer to fire after idle of silence. Called with mu held.
func (s *silenceWriter) armLocked() {
	if s.timer != nil {
		s.timer.Stop()
	}
	s.timer = time.AfterFunc(s.idle, s.signalTick)
}

// signalTick runs on the timer goroutine: inject a signal, flush, and arm the
// next idle window. Skips when a real data write arrived just before the timer
// fired (re-arms instead of sending on top of fresh data).
func (s *silenceWriter) signalTick() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.started {
		return
	}
	if time.Since(s.lastWrite) < s.idle {
		s.armLocked()
		return
	}
	if _, err := s.ResponseWriter.Write(s.signal); err != nil {
		log.Warnf("silenceWriter: signal write failed: %v", err)
		s.started = false
		return
	}
	if f, ok := s.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
	s.armLocked()
}

// bufPool reuses bytes.Buffer instances across requests to reduce GC pressure.
var bufPool = sync.Pool{
	New: func() any {
		return &bytes.Buffer{}
	},
}

// ProxyMetrics holds performance data for a proxied request.
type ProxyMetrics struct {
	StatusCode       int
	ErrorBody        string
	TTFBMs           int64
	ResponseSize     int64
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	CachedTokens     int
	PromptMs         float64
	PredictedMs      float64
	PromptPerSec     float64
	TokensPerSec     float64
	BackendModel     string // model name returned by the backend (before rewriting)
}

// RouterHeaders are custom response headers injected by the router
// to expose routing and performance metadata to the client.
// Set these before calling StreamProxy.
type RouterHeaders struct {
	ServerID   string // Sets X-Router-Server
	ServerName string // Sets X-Router-Server-Name
	// Attempts sets X-Router-Attempts (e.g. "2/3" — current attempt / total).
	Attempts string
	// Retries sets X-Router-Retries: how many retries on this server happened
	// before the request succeeded (0 = first try).
	Retries int
	// FallbackErrors sets X-Router-Fallback-Errors: JSON list of errors from
	// earlier failed attempts.
	FallbackErrors []string
	// KeepAliveIdle, when non-zero, is the server-configured SSE keep-alive
	// heartbeat interval. A per-request X-Router-KeepAlive header still
	// overrides it; zero falls back to the global proxy.KeepAliveIdle.
	KeepAliveIdle time.Duration
}

// SetRouterHeaders sets eager headers (server info) on the given ResponseWriter.
// These are safe to set before the proxy runs since they don't depend on response data.
func SetRouterHeaders(w http.ResponseWriter, h *RouterHeaders) {
	if h == nil {
		return
	}
	w.Header().Set("X-Router-Server", h.ServerID)
	w.Header().Set("X-Router-Server-Name", h.ServerName)
	if h.Attempts != "" {
		w.Header().Set("X-Router-Attempts", h.Attempts)
	}
	// Always write retries (even 0): the response writer's header map persists
	// across retry/fallback attempts, so a previous attempt's value must be
	// overwritten for the attempt that actually served the request.
	w.Header().Set("X-Router-Retries", strconv.Itoa(h.Retries))
	if len(h.FallbackErrors) > 0 {
		if b, err := json.Marshal(h.FallbackErrors); err == nil {
			w.Header().Set("X-Router-Fallback-Errors", string(b))
		} else {
			log.Errorf("failed to marshal fallback errors: %v", err)
		}
	}
}

// metricsWriter wraps an http.ResponseWriter to track TTFB and response size.
type metricsWriter struct {
	http.ResponseWriter
	statusCode      int
	firstWrite      time.Time
	startTime       time.Time
	responseSize    int64
	bodyBuffer      *bytes.Buffer
	contentType     string
	contentEncoding string
	injectStatus    bool // set X-Router-Status at WriteHeader (router mode)
}

func newMetricsWriter(w http.ResponseWriter, start time.Time) *metricsWriter {
	buf := bufPool.Get().(*bytes.Buffer)
	buf.Reset()
	return &metricsWriter{
		ResponseWriter: w,
		startTime:      start,
		statusCode:     http.StatusOK,
		bodyBuffer:     buf,
	}
}

func (m *metricsWriter) WriteHeader(code int) {
	m.statusCode = code
	m.contentType = m.ResponseWriter.Header().Get("Content-Type")
	m.contentEncoding = m.ResponseWriter.Header().Get("Content-Encoding")
	if m.injectStatus {
		m.Header().Set("X-Router-Status", strconv.Itoa(code))
	}
	m.ResponseWriter.WriteHeader(code)
}

func (m *metricsWriter) Flush() {
	if flusher, ok := m.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func (m *metricsWriter) Write(data []byte) (int, error) {
	if m.firstWrite.IsZero() {
		m.firstWrite = time.Now()
	}
	if m.bodyBuffer.Len()+len(data) > 256*1024 {
		keep := 256*1024 - len(data)
		if keep > 0 {
			m.bodyBuffer.Next(keep)
		} else {
			m.bodyBuffer.Reset()
		}
	}
	m.bodyBuffer.Write(data)
	n, err := m.ResponseWriter.Write(data)
	m.responseSize += int64(n)
	return n, err
}

func (m *metricsWriter) metrics() ProxyMetrics {
	var ttfb int64
	if !m.firstWrite.IsZero() {
		ttfb = m.firstWrite.Sub(m.startTime).Milliseconds()
	}

	isStream := strings.Contains(m.contentType, "text/event-stream")
	pm := extractUsageFromResponse(m.bodyBuffer.Bytes(), m.contentEncoding, isStream)
	pm.StatusCode = m.statusCode
	if m.statusCode >= 400 {
		body := m.bodyBuffer.Bytes()
		if len(body) > 4096 {
			body = body[:4096]
		}
		pm.ErrorBody = string(body)
	}
	pm.TTFBMs = ttfb
	pm.ResponseSize = m.responseSize

	return pm
}

// RewriteModelInBody parses the JSON body, replaces the "model" field, and returns the new body.
// Uses json.Marshal for correctness — byte-level replacement can miss edge cases
// with escaped characters in model names (e.g., GGUF names with / that may be
// escaped as \/ by json.Marshal but not matched by byte-level search).
func RewriteModelInBody(body []byte, newModel string) ([]byte, error) {
	var obj map[string]interface{}
	if err := json.Unmarshal(body, &obj); err != nil {
		return nil, fmt.Errorf("unmarshal body: %w", err)
	}

	obj["model"] = newModel

	out, err := json.Marshal(obj)
	if err != nil {
		return nil, fmt.Errorf("marshal body: %w", err)
	}

	return out, nil
}

// DeclaredOutputBudget returns the request's declared output token budget: the
// larger of max_tokens and max_completion_tokens (both spellings can arrive
// together; the provider honours whichever it supports, so the larger keeps the
// TPM reservation an upper bound). 0 when neither is set. Used for TPM
// pre-call reservation.
func DeclaredOutputBudget(body []byte) int64 {
	var s struct {
		MaxTokens           *int64 `json:"max_tokens"`
		MaxCompletionTokens *int64 `json:"max_completion_tokens"`
	}
	if err := json.Unmarshal(body, &s); err != nil {
		return 0
	}
	var max int64
	if s.MaxTokens != nil && *s.MaxTokens > max {
		max = *s.MaxTokens
	}
	if s.MaxCompletionTokens != nil && *s.MaxCompletionTokens > max {
		max = *s.MaxCompletionTokens
	}
	return max
}

// modelRewriteWriter wraps a metricsWriter to replace the target model name
// with the original model name in JSON responses (both streaming and non-streaming).
// It wraps the metricsWriter so that metrics (TTFB, size, buffer) are captured
// for the actual data sent to the client.
type modelRewriteWriter struct {
	http.ResponseWriter
	oldModel      string
	newModel      string
	capturedModel string // backend model name captured before rewriting
}

func newModelRewriteWriter(mw *metricsWriter, oldModel, newModel string) *modelRewriteWriter {
	return &modelRewriteWriter{
		ResponseWriter: mw,
		oldModel:       oldModel,
		newModel:       newModel,
	}
}

func (r *modelRewriteWriter) Write(data []byte) (int, error) {
	rewritten, captured := rewriteModelInResponse(data, r.oldModel, r.newModel)
	if r.capturedModel == "" && captured != "" {
		r.capturedModel = captured
	}
	return r.ResponseWriter.Write(rewritten)
}

func (r *modelRewriteWriter) Flush() {
	if flusher, ok := r.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

// rewriteModelInResponse replaces model names in JSON response data.
// When oldModel != newModel: replaces "model":"oldModel" with "model":"newModel".
// When oldModel == newModel: replaces ANY "model":"X" with "model":"newModel"
// (the backend may return its actual model name even when we sent a different one).
// Returns the rewritten data and the captured backend model name (if any replacement was made).
func rewriteModelInResponse(data []byte, oldModel, newModel string) ([]byte, string) {
	if oldModel == "" || newModel == "" {
		return data, ""
	}

	if oldModel == newModel {
		// Backend may return its real model name (e.g., "unsloth/Qwen3...")
		// even when we sent "opus". Replace any model value with the client's model.
		result, captured := replaceAnyModelValue(data, newModel)
		return result, captured
	}

	result, n := replaceJSONModelValue(data, oldModel, newModel)
	if n == 0 {
		if bytes.Contains(data, []byte(`"model"`)) {
			log.Debugf("modelRewrite: found 'model' key but no match for %q in chunk: %s", oldModel, truncateBytes(data, 256))
		}
		return data, ""
	}
	log.Debugf("modelRewrite: replaced %d occurrence(s) of %q → %q", n, oldModel, newModel)
	return result, ""
}

// replaceAnyModelValue replaces any "model":"X" with "model":"newModel".
// Used when oldModel == newModel but the backend returns its actual model name.
// Returns the rewritten data and the first captured backend model name.
func replaceAnyModelValue(data []byte, newModel string) ([]byte, string) {
	key := []byte(`"model"`)
	var result []byte
	var captured string
	start := 0
	escaped := escapeJSONString(newModel)
	replacements := 0

	for {
		keyIdx := bytes.Index(data[start:], key)
		if keyIdx < 0 {
			break
		}
		absKeyIdx := start + keyIdx
		endOfKey := absKeyIdx + len(key)

		// Skip if it's a longer key like "model_id"
		if endOfKey < len(data) && IsJSONNameChar(data[endOfKey]) {
			start = endOfKey
			continue
		}

		// Find colon
		colonIdx := bytes.Index(data[endOfKey:], []byte{':'})
		if colonIdx < 0 {
			break
		}
		valueStart := endOfKey + colonIdx + 1

		// Skip whitespace
		for valueStart < len(data) && isWhitespace(data[valueStart]) {
			valueStart++
		}
		if valueStart >= len(data) || data[valueStart] != '"' {
			start = valueStart
			continue
		}

		// Find end of string value
		strStart := valueStart + 1
		strEnd := FindJSONStringEnd(data, strStart)
		if strEnd < 0 {
			start = valueStart
			continue
		}
		// strEnd points to the closing quote
		currentValue := data[strStart:strEnd]

		// Skip if already the target model
		if bytes.Equal(currentValue, escaped) {
			start = strEnd + 1
			continue
		}

		// Capture the first backend model name
		if captured == "" {
			captured = string(currentValue)
		}

		// Replace
		if result == nil {
			result = make([]byte, 0, len(data))
		}
		result = append(result, data[start:valueStart]...)
		result = append(result, '"')
		result = append(result, escaped...)
		result = append(result, '"')
		start = strEnd + 1
		replacements++
	}

	if result == nil {
		return data, ""
	}
	if replacements > 0 {
		log.Debugf("modelRewrite(any): replaced %d model value(s) → %q", replacements, newModel)
	}
	return append(result, data[start:]...), captured
}

// FindJSONStringEnd finds the closing quote of a JSON string starting at pos.
// Returns the index of the closing quote, or -1 if not found.
func FindJSONStringEnd(data []byte, pos int) int {
	for pos < len(data) {
		if data[pos] == '\\' {
			pos += 2 // skip escaped character
			continue
		}
		if data[pos] == '"' {
			return pos
		}
		pos++
	}
	return -1
}

func truncateBytes(b []byte, max int) []byte {
	if len(b) <= max {
		return b
	}
	return append(b[:max], byte('.')<<0, byte('.')<<0, byte('.')<<0)
}

// replaceJSONModelValue finds and replaces "model" values matching oldModel.
func replaceJSONModelValue(data []byte, oldModel, newModel string) ([]byte, int) {
	key := []byte(`"model"`)
	replacements := 0
	var result []byte
	start := 0

	for {
		keyIdx := bytes.Index(data[start:], key)
		if keyIdx < 0 {
			break
		}
		absKeyIdx := start + keyIdx
		endOfKey := absKeyIdx + len(key)

		// Verify it's the exact key
		if endOfKey < len(data) && IsJSONNameChar(data[endOfKey]) {
			start = endOfKey
			continue
		}

		// Find colon
		colonIdx := bytes.Index(data[endOfKey:], []byte{':'})
		if colonIdx < 0 {
			break
		}
		valueStart := endOfKey + colonIdx + 1

		// Skip whitespace after colon
		for valueStart < len(data) && isWhitespace(data[valueStart]) {
			valueStart++
		}
		if valueStart >= len(data) || data[valueStart] != '"' {
			start = valueStart
			continue
		}

		// Check if value matches oldModel
		escOld := escapeJSONString(oldModel)
		pattern := append([]byte{'"'}, escOld...)
		pattern = append(pattern, '"')

		if valueStart+len(pattern) > len(data) {
			break
		}
		if !bytes.Equal(data[valueStart:valueStart+len(pattern)], pattern) {
			start = valueStart + 1
			continue
		}

		// Replace
		if result == nil {
			result = make([]byte, 0, len(data))
		}
		result = append(result, data[start:valueStart]...)
		result = append(result, '"')
		result = append(result, escapeJSONString(newModel)...)
		result = append(result, '"')
		start = valueStart + len(pattern)
		replacements++
	}

	if replacements == 0 {
		return data, 0
	}
	return append(result, data[start:]...), replacements
}

// escapeJSONString escapes a string for use in JSON.
func escapeJSONString(s string) []byte {
	b := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch c {
		case '"':
			b = append(b, '\\', '"')
		case '\\':
			b = append(b, '\\', '\\')
		case '\n':
			b = append(b, '\\', 'n')
		case '\r':
			b = append(b, '\\', 'r')
		case '\t':
			b = append(b, '\\', 't')
		case '\b':
			b = append(b, '\\', 'b')
		case '\f':
			b = append(b, '\\', 'f')
		default:
			if c < 0x20 {
				b = append(b, '\\', 'u', '0', '0',
					hex(c>>4), hex(c&0xF))
			} else {
				b = append(b, c)
			}
		}
	}
	return b
}

// hex returns the hex digit for a nibble (0-15).
func hex(n byte) byte {
	if n < 10 {
		return '0' + n
	}
	return 'a' + n - 10
}

// isJSONNameChar reports whether b is a valid JSON object key character.
func IsJSONNameChar(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9') || b == '_' || b == '-'
}

// isWhitespace reports whether b is a JSON whitespace character.
func isWhitespace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}

// loopDetector wraps an http.ResponseWriter to detect when the backend
// gets stuck in an infinite loop, repeatedly sending the same content.
// Based on LiteLLM's REPEATED_STREAMING_CHUNK_LIMIT pattern.
//
// When a loop is detected, the detector cancels the request context
// (which cancels the upstream request) and sends an error frame to the client.
const loopDetectionWindow = 20 // number of recent SSE contents to track

type loopDetector struct {
	http.ResponseWriter
	ctx context.Context
	// Ring buffer of recent SSE text contents
	recent        []string
	index         int
	count         int    // number of contents seen
	detected      bool   // loop already detected
	written       bool   // WriteHeader already called
	capturedModel string // backend model name extracted from raw response
}

func newLoopDetector(w http.ResponseWriter, ctx context.Context) *loopDetector {
	return &loopDetector{
		ResponseWriter: w,
		ctx:            ctx,
		recent:         make([]string, loopDetectionWindow),
	}
}

func (d *loopDetector) Write(data []byte) (int, error) {
	if d.detected {
		return 0, fmt.Errorf("stream loop detected: backend is repeating content")
	}

	// Capture backend model name from raw response (before rewriting)
	if d.capturedModel == "" {
		if m := extractModelFromData(data); m != "" {
			d.capturedModel = m
		}
	}

	// Extract text content from SSE events
	contents := extractSSEContents(data)
	for _, content := range contents {
		if content == "" {
			continue
		}
		// Skip non-content events (role, tool calls, etc.)
		if len(content) > 1024 {
			content = content[:1024]
		}
		d.recent[d.index%loopDetectionWindow] = content
		d.index++
		d.count++

		if d.count >= loopDetectionWindow {
			if d.allRecentIdentical() {
				d.detected = true
				// Cancel upstream request
				if err := d.sendLoopError(); err != nil {
					return 0, err
				}
				return 0, fmt.Errorf("stream loop detected: backend is repeating content")
			}
		}
	}

	return d.ResponseWriter.Write(data)
}

// Flush forwards the flush down the writer chain so SSE events reach the
// client as they are produced. Without this, net/http buffers response writes
// in ~4KB lumps and the client receives short responses as one blob at the end
// — a delivery pattern that triggers client-side chunk-concatenation bugs
// (e.g. opencode #7692).
func (d *loopDetector) Flush() {
	if flusher, ok := d.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func (d *loopDetector) allRecentIdentical() bool {
	first := d.recent[0]
	if first == "" {
		return false
	}
	for i := 1; i < loopDetectionWindow; i++ {
		if d.recent[i] != first {
			return false
		}
	}
	return true
}

func (d *loopDetector) sendLoopError() error {
	// Write through the underlying writer, not d.Write: the detected guard must
	// not swallow the error frame we are sending to the client.
	errJSON := `{"choices":[{"finish_reason":"error","delta":{"content":"Generation stopped: model appears to be stuck in a loop."}}]}`
	event := []byte("data: " + errJSON + "\n\n")

	if !d.written {
		d.written = true
		h := d.Header()
		h.Set("Content-Type", "text/event-stream")
		h.Set("Cache-Control", "no-cache")
		d.WriteHeader(http.StatusInternalServerError)
	}

	_, err := d.ResponseWriter.Write(event)
	if flusher, ok := d.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
	return err
}

// extractSSEContents extracts text content from SSE data events.
// It handles both OpenAI (delta.content) and Anthropic (delta.text) formats.
func extractSSEContents(data []byte) []string {
	var contents []string
	dataStr := string(data)

	// Split by "data:" prefix (with optional leading whitespace after colon)
	for _, line := range strings.Split(dataStr, "data:") {
		line = strings.TrimPrefix(line, " ")
		if len(line) < 2 {
			continue
		}
		// Take only up to the newline
		if nl := strings.IndexByte(line, '\n'); nl >= 0 {
			line = line[:nl]
		}
		if line == "[DONE]" {
			continue
		}
		if len(line) < 2 {
			continue
		}

		// Extract content from JSON; fragments that are not JSON objects
		// (event: lines, partial frames) are skipped, not logged
		if !strings.HasPrefix(line, "{") {
			continue
		}
		var obj map[string]interface{}
		if err := json.Unmarshal([]byte(line), &obj); err != nil {
			log.Debugf("extractSSEContents: failed to parse SSE data line: %v", err)
			continue
		}

		// OpenAI format: choices[0].delta.content
		if choices, ok := obj["choices"].([]interface{}); ok && len(choices) > 0 {
			if choice, ok := choices[0].(map[string]interface{}); ok {
				if delta, ok := choice["delta"].(map[string]interface{}); ok {
					if content, ok := delta["content"].(string); ok && content != "" {
						contents = append(contents, content)
					}
				}
			}
		}
		// Anthropic format: delta.text
		if delta, ok := obj["delta"].(map[string]interface{}); ok {
			if text, ok := delta["text"].(string); ok && text != "" {
				contents = append(contents, text)
			}
		}
	}

	return contents
}

// extractModelFromData extracts the "model" field from raw JSON or SSE data.
// It handles both streaming (data: {...}) and non-streaming ({...}) formats.
// Returns the first model name found, or empty string if not found.
func extractModelFromData(data []byte) string {
	var model string
	forEachDataLine(data, func(payload []byte) bool {
		if m := extractModelFromJSON(payload); m != "" {
			model = m
			return true
		}
		return false
	})
	if model != "" {
		return model
	}
	// Try as plain JSON (non-streaming or data doesn't come in SSE format)
	return extractModelFromJSON(data)
}

// extractModelFromJSON extracts the "model" field from a JSON object.
func extractModelFromJSON(data []byte) string {
	var obj map[string]interface{}
	if err := json.Unmarshal(data, &obj); err != nil {
		return ""
	}
	if m, ok := obj["model"].(string); ok && m != "" {
		return m
	}
	return ""
}

// MidStreamError is returned when the backend fails after headers have already
// been sent to the client. Unlike pre-response errors, mid-stream errors cannot
// trigger fallback because the client already received a partial response.
type MidStreamError struct {
	Err     error
	Written int64 // bytes written to client before the error
}

func (e *MidStreamError) Error() string {
	return fmt.Sprintf("mid-stream error after %d bytes: %v", e.Written, e.Err)
}

func (e *MidStreamError) Unwrap() error {
	return e.Err
}

// StreamErrorResponse is returned when the backend answers a 5xx whose body is
// already-accumulated SSE stream frames (a backend that died mid-generation
// after buffering output). Nothing was written to the client, so the router can
// retry or fall back instead of forwarding an unparseable response.
type StreamErrorResponse struct {
	StatusCode int
	Body       string
}

func (e *StreamErrorResponse) Error() string {
	body := e.Body
	if len(body) > 512 {
		body = body[:512] + "..."
	}
	return fmt.Sprintf("backend returned HTTP %d with SSE-form error body: %q", e.StatusCode, body)
}

// looksLikeStreamFrameBody reports whether body reads as SSE text frames
// (starts with a "data:", "event:", or comment line).
func looksLikeStreamFrameBody(body []byte) bool {
	trimmed := bytes.TrimLeft(body, " \t\r\n")
	return len(trimmed) > 0 &&
		(bytes.HasPrefix(trimmed, []byte("data:")) ||
			bytes.HasPrefix(trimmed, []byte("event:")) ||
			bytes.HasPrefix(trimmed, []byte(": ")))
}

// StreamProxy forwards the request and streams the response directly to the writer.
// Returns an error on network failure (before any headers are written to w).
// HTTP 5xx responses are NOT treated as errors — they are forwarded to the client.
// Exception: a 5xx whose body is already SSE stream frames is returned as
// *StreamErrorResponse without writing anything to the client, so the router
// can retry/fall back (see docs above).
// If originalModel is non-empty and differs from targetModel, the "model" field
// in the response is rewritten from targetModel back to originalModel so the
// client sees its own model name.
// If rh is non-nil, X-Router-* headers are injected into the response.
// If strip is true, injected stream-usage artifacts (empty-choice SSE
// frames appended by upstreams that received stream_options.include_usage) are
// filtered from the client stream while still being captured in metrics.
// If ping is non-nil, SSE heartbeats are injected into text/event-stream
// responses when the upstream stays silent for KeepAliveIdle (client read
// timeouts during long backend pauses). ping carries the protocol-appropriate
// heartbeat frame (Anthropic "event: ping" or an OpenAI SSE comment).
//
// Mid-stream errors (backend disconnects after headers are sent) are detected
// and returned as *MidStreamError. The client receives an error event appended
// to the stream, but fallback is not possible because headers are already sent.
func StreamProxy(ctx context.Context, targetURL string, apiKey string, req *http.Request, w http.ResponseWriter, targetModel, originalModel string, rh *RouterHeaders, strip bool, ping []byte) (*ProxyMetrics, error) {
	rawURL := targetURL
	if !strings.Contains(rawURL, "://") {
		rawURL = "https://" + rawURL
	}

	target, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("parse target URL: %w", err)
	}

	targetBase := strings.TrimRight(target.Path, "/")
	targetPath := targetBase + req.URL.Path
	// URL dedup: a server URL that already ends with /v1 plus a request path
	// that starts with /v1 would otherwise produce /v1/v1/... — drop the
	// redundant prefix so the two merge into a single /v1.
	if strings.HasSuffix(targetBase, "/v1") && strings.HasPrefix(req.URL.Path, "/v1") {
		targetPath = targetBase + req.URL.Path[len("/v1"):]
	}

	start := time.Now()
	baseW := w
	var ka *silenceWriter
	// Per-request keepalive override, resolved as: X-Router-KeepAlive header
	// (seconds, clamped [1,300], "0" disables) > server-configured
	// rh.KeepAliveIdle > global KeepAliveIdle.
	kaIdle := KeepAliveIdle
	if rh != nil && rh.KeepAliveIdle > 0 {
		kaIdle = rh.KeepAliveIdle
	}
	if h := req.Header.Get("X-Router-KeepAlive"); h != "" {
		if n, err := strconv.Atoi(h); err == nil {
			switch {
			case n == 0:
				kaIdle = 0
			case n < 1:
				kaIdle = time.Second
			case n > 300:
				kaIdle = 300 * time.Second
			default:
				kaIdle = time.Duration(n) * time.Second
			}
		}
	}
	if ping != nil && kaIdle > 0 {
		ka = newSilenceWriter(w, kaIdle, ping)
		defer ka.Stop()
		baseW = ka
	}
	// The wait-signal writer sits between the upstream response and the
	// client-facing writers: it monitors silence and injects "still waiting"
	// frames when the upstream stays quiet for WaitSignalIdle. This keeps
	// client SDK read loops alive during long backend think times, preventing
	// "Request timed out" retries that waste work.
	ws := newSilenceWriter(baseW, WaitSignalIdle, ping)
	defer ws.Stop()
	clientW := newSSERelay(ws, strip)
	mw := newMetricsWriter(clientW, start)
	defer bufPool.Put(mw.bodyBuffer) //nolint:errcheck
	mrw := newModelRewriteWriter(mw, targetModel, originalModel)
	var rw http.ResponseWriter = mrw

	// Inject X-Router-* headers: eager ones (Server/Server-Name) now,
	// X-Router-Status at WriteHeader time via metricsWriter.
	if rh != nil {
		SetRouterHeaders(rw, rh)
		mw.injectStatus = true
	}

	// Wrap with loop detector to catch backends that get stuck repeating content.
	ld := newLoopDetector(rw, ctx)

	// Direct the request
	//
	// The upstream request does NOT inherit the client context's cancellation:
	// as soon as the client context is canceled mid-stream, net/http closes the
	// upstream connection — which a reverse-proxy backend (llama.cpp router
	// mode) reads as "client canceled" and aborts the generation the client is
	// about to retry from scratch. Instead the response is drained to
	// completion (see drainUpstreamBody), so the backend finishes naturally.
	// Deadline-based timeouts from the original context are preserved.
	detachedCtx := context.WithoutCancel(req.Context())
	if dl, ok := req.Context().Deadline(); ok {
		var cancel context.CancelFunc
		detachedCtx, cancel = context.WithDeadline(detachedCtx, dl)
		defer cancel()
	}
	proxyReq := req.Clone(detachedCtx)
	proxyReq.URL.Scheme = target.Scheme
	proxyReq.URL.Host = target.Host
	proxyReq.URL.Path = targetPath
	proxyReq.URL.RawQuery = req.URL.RawQuery
	proxyReq.Header = proxyReq.Header.Clone()
	if apiKey != "" {
		// Set BOTH authentication conventions: OpenAI-compatible upstreams
		// expect `Authorization: Bearer`, Anthropic-compatible ones (and
		// llama.cpp's /v1/messages) expect `x-api-key`. Forcing x-api-key
		// also overwrites the client's own key — Claude Code authenticates
		// with a personal x-api-key that must not reach (or be checked
		// against) the configured upstream. llama.cpp validates
		// `Authorization` first and falls back to `X-Api-Key`, so setting
		// both satisfies either check.
		proxyReq.Header.Set("x-api-key", apiKey)
		proxyReq.Header.Set("Authorization", "Bearer "+apiKey)
	}
	proxyReq.Header.Del("Host")
	proxyReq.GetBody = nil
	proxyReq.RequestURI = "" // Must be empty for client requests (Clone preserves the server-side value)

	// Execute the upstream request (shared client with connection pooling)
	upstreamResp, err := upstreamClient.Do(proxyReq)
	if err != nil {
		// Pre-response error — no headers written to client yet.
		// Fallback is possible.
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, fmt.Errorf("upstream request failed: %w", err)
	}
	defer func() {
		if err := upstreamResp.Body.Close(); err != nil {
			log.Errorf("failed to close upstream response body: %v", err)
		}
	}()

	// A dying backend may answer 5xx with a body of already-accumulated SSE
	// frames (its buffered stream). No SSE client can parse such a response —
	// forwarding it changes a crash into "stream idle timeout" noise. Nothing
	// has been written to the client yet, so treat it as a retryable failure.
	// `pre` carries the peeked bytes into the normal relay below when the body
	// turns out to be a real error payload (5xx forwarding stays unchanged).
	var pre []byte
	if upstreamResp.StatusCode >= 500 {
		const peekLimit = 32 * 1024
		pre, _ = io.ReadAll(io.LimitReader(upstreamResp.Body, peekLimit+1))
		if looksLikeStreamFrameBody(pre) {
			m := mw.metrics()
			m.StatusCode = upstreamResp.StatusCode
			return &m, &StreamErrorResponse{StatusCode: upstreamResp.StatusCode, Body: string(pre)}
		}
	}

	// Copy response headers
	headers := ld.Header()
	for k, vv := range upstreamResp.Header {
		headers[k] = vv
	}
	// The response body may be rewritten (model names change the byte length),
	// so drop length/framing headers and let Go compute correct chunked framing.
	headers.Del("Transfer-Encoding")
	headers.Del("Content-Length")

	// Arm heartbeats and wait signals for SSE streams: while the backend thinks
	// (or pauses between tokens) the client would otherwise see silence long
	// enough to trip its read timeout. Heartbeats use protocol-appropriate
	// frames (Anthropic "event: ping" or OpenAI comment); wait signals use a
	// custom `router_wait` event that SDKs ignore but that keeps the TCP read
	// side alive.
	if ka != nil && strings.Contains(upstreamResp.Header.Get("Content-Type"), "text/event-stream") {
		ka.Start()
	}
	if ws != nil && strings.Contains(upstreamResp.Header.Get("Content-Type"), "text/event-stream") {
		ws.Start()
	}

	// Write status code (triggers TTFB and X-Router-Status injection)
	ld.WriteHeader(upstreamResp.StatusCode)

	// Stream the response body, chunk by chunk, to detect mid-stream errors.
	// `pending` holds bytes already read while inspecting a 5xx response above.
	buf := make([]byte, 32*1024)
	firstChunk := true
	pending := pre
	for {
		chunk := pending
		pending = nil
		var readErr error
		if chunk == nil {
			var n int
			n, readErr = upstreamResp.Body.Read(buf)
			if n > 0 {
				chunk = buf[:n]
			}
		}
		if chunk != nil {
			// Log first chunk for debugging model rewrite
			if firstChunk {
				firstChunk = false
				debug := chunk
				if len(debug) > 512 {
					debug = debug[:512]
				}
				log.Debugf("upstream first chunk: %s", debug)
			}
			_, writeErr := ld.Write(chunk)
			if writeErr != nil {
				// Loop detected: the error event was already appended to the stream
				// and headers are sent. Classify as mid-stream so the router does
				// not attempt fallback (a second response would corrupt the stream).
				if ld.detected {
					m := mw.metrics()
					m.BackendModel = mrw.capturedModel
					return &m, &MidStreamError{Err: fmt.Errorf("stream loop detected: backend is repeating content"), Written: m.ResponseSize}
				}
				// Error writing to client (e.g., client disconnect)
				if ctx.Err() != nil {
					// The frontend is gone, but the upstream must not be told
					// "client canceled": a reverse-proxy backend (llama.cpp
					// router mode) aborts the generation on connection close,
					// wasting the run and forcing the client to regenerate from
					// scratch on its next attempt. Drain the remaining response
					// instead so the backend finishes the generation naturally.
					if UpstreamDrainTimeout > 0 {
						drainUpstreamBody(upstreamResp.Body, UpstreamDrainTimeout)
					}
					m := mw.metrics()
					m.BackendModel = mrw.capturedModel
					return &m, ctx.Err()
				}
				m := mw.metrics()
				m.BackendModel = mrw.capturedModel
				return &m, fmt.Errorf("write to client: %w", writeErr)
			}
			// Flush after every chunk so the client receives each SSE event as it
			// is produced (proper relay behavior), instead of net/http's 4KB-buffered
			// lumps that make clients mis-split concatenated frames.
			ld.Flush()
		}

		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			// Mid-stream error from upstream
			if ctx.Err() != nil {
				m := mw.metrics()
				m.BackendModel = mrw.capturedModel
				return &m, ctx.Err()
			}
			// Send error event to the client
			sendMidStreamError(ld, readErr)
			m := mw.metrics()
			m.BackendModel = mrw.capturedModel
			return &m, &MidStreamError{Err: readErr, Written: m.ResponseSize}
		}
	}

	// Check if loop was detected (proxy may have returned after context cancellation)
	if ld.detected {
		m := mw.metrics()
		m.BackendModel = mrw.capturedModel
		return &m, fmt.Errorf("stream loop detected: backend is repeating content")
	}

	// Deliver any bytes the relay is still holding (e.g. a trailing frame the
	// backend closed without a blank-line terminator). Regular flush no longer
	// pushes pending bytes mid-stream (it must not fragment data lines), so the
	// relay gets an explicit finish at stream end.
	clientW.finish()

	m := mw.metrics()
	m.BackendModel = mrw.capturedModel
	return &m, nil
}

// sendMidStreamError appends an error event to the ongoing stream to inform
// the client that the backend disconnected mid-stream.
func sendMidStreamError(w http.ResponseWriter, err error) {
	// Real newlines terminate the SSE event; escapeJSONString keeps the message
	// valid JSON (quotes, backslashes, control chars).
	errEvent := fmt.Sprintf(
		"data: {\"choices\":[{\"finish_reason\":\"error\",\"delta\":{\"content\":\"[error: %s]\"}}]}\n\n",
		escapeJSONString(err.Error()),
	)
	_, writeErr := w.Write([]byte(errEvent))
	if writeErr != nil {
		log.Errorf("failed to send mid-stream error event to client: %v", writeErr)
	}
	// Flush immediately
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
}

// drainUpstreamBody reads the upstream response body to completion, discarding
// the bytes. Used when the client connection is already gone: the upstream TCP
// connection stays open so a reverse-proxy backend finishes the generation
// instead of seeing a closed socket (i.e. cancel) mid-stream. Bounded by
// timeout so a wedged backend cannot pin the handler goroutine forever.
func drainUpstreamBody(body io.ReadCloser, timeout time.Duration) {
	deadline := time.After(timeout)
	buf := make([]byte, 32*1024)
	for {
		select {
		case <-deadline:
			return
		default:
		}
		if _, err := body.Read(buf); err != nil {
			return
		}
	}
}
