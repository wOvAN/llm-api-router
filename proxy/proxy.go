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

// upstreamTransport is a shared HTTP transport with connection pooling.
// Reused across all proxied requests to enable TCP connection reuse,
// TLS session resumption, and keep-alive. Eliminates ~50-200ms per-request
// overhead from new TCP+TLS handshakes.
var upstreamTransport = &http.Transport{
	MaxIdleConns:        200,
	MaxIdleConnsPerHost: 50,
	IdleConnTimeout:     90 * time.Second,
	DialContext: (&net.Dialer{
		Timeout:   10 * time.Second,
		KeepAlive: 30 * time.Second,
	}).DialContext,
	TLSHandshakeTimeout: 10 * time.Second,
}

// upstreamClient is the shared HTTP client for all upstream requests.
var upstreamClient = &http.Client{Transport: upstreamTransport}

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

// headerInjector wraps an http.ResponseWriter to inject X-Router-* headers
// at WriteHeader time. TTFB is measured at first Write() time (first content
// byte) via metricsWriter, so it's not available here.
//
// Headers set eagerly (via SetRouterHeaders): X-Router-Server, X-Router-Server-Name.
// Headers set at WriteHeader time: X-Router-Status.
// Latency and token headers are NOT set in streaming responses — they are
// available in /admin/api/metrics after the fact.
type headerInjector struct {
	http.ResponseWriter
	statusCode int
	written    bool
}

func newHeaderInjector(w http.ResponseWriter) *headerInjector {
	return &headerInjector{ResponseWriter: w}
}

func (h *headerInjector) WriteHeader(code int) {
	h.statusCode = code
	h.written = true

	// Inject headers BEFORE forwarding to client.
	// TTFB is measured at first Write() time (first content byte), not here.
	// At WriteHeader time, we only know the status code.
	headers := h.Header()
	headers.Set("X-Router-Status", strconv.Itoa(code))

	h.ResponseWriter.WriteHeader(code)
}

func (h *headerInjector) Flush() {
	if flusher, ok := h.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
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
		if bytesIndex(data, []byte(`"model"`)) >= 0 {
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
		keyIdx := bytesIndex(data[start:], key)
		if keyIdx < 0 {
			break
		}
		absKeyIdx := start + keyIdx
		endOfKey := absKeyIdx + len(key)

		// Skip if it's a longer key like "model_id"
		if endOfKey < len(data) && isJSONNameChar(data[endOfKey]) {
			start = endOfKey
			continue
		}

		// Find colon
		colonIdx := bytesIndex(data[endOfKey:], []byte{':'})
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
		strEnd := findJSONStringEnd(data, strStart)
		if strEnd < 0 {
			start = valueStart
			continue
		}
		// strEnd points to the closing quote
		currentValue := data[strStart:strEnd]

		// Skip if already the target model
		if bytesEqual(currentValue, escaped) {
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

// findJSONStringEnd finds the closing quote of a JSON string starting at pos.
// Returns the index of the closing quote, or -1 if not found.
func findJSONStringEnd(data []byte, pos int) int {
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
		keyIdx := bytesIndex(data[start:], key)
		if keyIdx < 0 {
			break
		}
		absKeyIdx := start + keyIdx
		endOfKey := absKeyIdx + len(key)

		// Verify it's the exact key
		if endOfKey < len(data) && isJSONNameChar(data[endOfKey]) {
			start = endOfKey
			continue
		}

		// Find colon
		colonIdx := bytesIndex(data[endOfKey:], []byte{':'})
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
		if !bytesEqual(data[valueStart:valueStart+len(pattern)], pattern) {
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

func bytesIndex(data, sub []byte) int {
	for i := 0; i <= len(data)-len(sub); i++ {
		if bytesEqual(data[i:i+len(sub)], sub) {
			return i
		}
	}
	return -1
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// isJSONNameChar reports whether b is a valid JSON object key character.
func isJSONNameChar(b byte) bool {
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

		// Extract content from JSON
		var obj map[string]interface{}
		if err := json.Unmarshal([]byte(line), &obj); err != nil {
			log.Debugf("loopDetector: failed to parse SSE data line: %v", err)
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
	// Try to extract from SSE data: lines
	prefix := []byte("data:")
	for offset := 0; offset < len(data); {
		nl := bytes.IndexByte(data[offset:], '\n')
		var line []byte
		if nl == -1 {
			line = data[offset:]
			offset = len(data)
		} else {
			line = data[offset : offset+nl]
			offset += nl + 1
		}
		line = bytes.TrimSpace(line)
		if len(line) == 0 || !bytes.HasPrefix(line, prefix) {
			continue
		}
		jsonData := bytes.TrimSpace(line[len(prefix):])
		if len(jsonData) == 0 {
			continue
		}
		if m := extractModelFromJSON(jsonData); m != "" {
			return m
		}
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

// StreamProxy forwards the request and streams the response directly to the writer.
// Returns an error on network failure (before any headers are written to w).
// HTTP 5xx responses are NOT treated as errors — they are forwarded to the client.
// If originalModel is non-empty and differs from targetModel, the "model" field
// in the response is rewritten from targetModel back to originalModel so the
// client sees its own model name.
// If rh is non-nil, X-Router-* headers are injected into the response.
// If strip is true, injected stream-usage artifacts (empty-choice SSE
// frames appended by upstreams that received stream_options.include_usage) are
// filtered from the client stream while still being captured in metrics.
//
// Mid-stream errors (backend disconnects after headers are sent) are detected
// and returned as *MidStreamError. The client receives an error event appended
// to the stream, but fallback is not possible because headers are already sent.
func StreamProxy(ctx context.Context, targetURL string, apiKey string, req *http.Request, w http.ResponseWriter, targetModel, originalModel string, rh *RouterHeaders, strip bool) (*ProxyMetrics, error) {
	rawURL := targetURL
	if !strings.Contains(rawURL, "://") {
		rawURL = "https://" + rawURL
	}

	target, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("parse target URL: %w", err)
	}

	targetPath := strings.TrimRight(target.Path, "/") + req.URL.Path
	if strings.HasSuffix(target.Path, "/v1") && strings.HasPrefix(req.URL.Path, "/v1") {
		targetPath = strings.TrimRight(target.Path, "/") + req.URL.Path[1:]
	}

	start := time.Now()
	clientW := w
	if strip {
		clientW = newUsageStripper(w)
	} else {
		clientW = newSSEFrameWriter(w)
	}
	mw := newMetricsWriter(clientW, start)
	defer bufPool.Put(mw.bodyBuffer) //nolint:errcheck
	mrw := newModelRewriteWriter(mw, targetModel, originalModel)
	var rw http.ResponseWriter = mrw

	// Wrap with headerInjector to inject X-Router-* headers at WriteHeader time.
	if rh != nil {
		SetRouterHeaders(rw, rh) // Set ServerID and ServerName eagerly
		rw = newHeaderInjector(rw)
	}

	// Wrap with loop detector to catch backends that get stuck repeating content.
	ld := newLoopDetector(rw, ctx)

	// Direct the request
	proxyReq := req.Clone(ctx)
	proxyReq.URL.Scheme = target.Scheme
	proxyReq.URL.Host = target.Host
	proxyReq.URL.Path = targetPath
	proxyReq.URL.RawQuery = req.URL.RawQuery
	proxyReq.Header = proxyReq.Header.Clone()
	if apiKey != "" {
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

	// Copy response headers
	headers := ld.Header()
	for k, vv := range upstreamResp.Header {
		headers[k] = vv
	}
	// The response body may be rewritten (model names change the byte length),
	// so drop length/framing headers and let Go compute correct chunked framing.
	headers.Del("Transfer-Encoding")
	headers.Del("Content-Length")

	// Write status code (this triggers TTFB / headerInjector)
	ld.WriteHeader(upstreamResp.StatusCode)

	// Stream the response body, chunk by chunk, to detect mid-stream errors
	buf := make([]byte, 32*1024)
	firstChunk := true
	for {
		n, readErr := upstreamResp.Body.Read(buf)
		if n > 0 {
			// Log first chunk for debugging model rewrite
			if firstChunk {
				firstChunk = false
				chunk := buf[:n]
				if len(chunk) > 512 {
					chunk = chunk[:512]
				}
				log.Debugf("upstream first chunk: %s", chunk)
			}
			_, writeErr := ld.Write(buf[:n])
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

	// Deliver any bytes the usage stripper is still holding (e.g. a trailing
	// frame the backend closed without a blank-line terminator). Regular flush
	// no longer pushes pending bytes mid-stream (it must not fragment data
	// lines), so the stripper gets an explicit finish at stream end.
	if s, ok := clientW.(*usageStripper); ok {
		s.finish()
	} else if f, ok := clientW.(*sseFrameWriter); ok {
		f.finish()
	} else if flusher, ok := clientW.(http.Flusher); ok {
		flusher.Flush()
	}

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
