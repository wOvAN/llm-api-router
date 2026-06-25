package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
	"strings"
	"time"
)

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
}

// RouterHeaders are custom response headers injected by the router
// to expose routing and performance metadata to the client.
// Set these before calling StreamProxy.
type RouterHeaders struct {
	ServerID   string // Sets X-Router-Server
	ServerName string // Sets X-Router-Server-Name
}

// SetRouterHeaders sets eager headers (server info) on the given ResponseWriter.
// These are safe to set before the proxy runs since they don't depend on response data.
func SetRouterHeaders(w http.ResponseWriter, h *RouterHeaders) {
	if h == nil {
		return
	}
	w.Header().Set("X-Router-Server", h.ServerID)
	w.Header().Set("X-Router-Server-Name", h.ServerName)
}

// headerInjector wraps an http.ResponseWriter to inject X-Router-* headers
// at WriteHeader time. At that moment, TTFB and status code are known,
// but latency and token counts are not (streaming may still be in progress).
//
// Headers set at WriteHeader time: X-Router-TTFB-Ms, X-Router-Status.
// Headers set eagerly (via SetRouterHeaders): X-Router-Server, X-Router-Server-Name.
// Latency and token headers are NOT set in streaming responses — they are
// available in /admin/api/metrics after the fact.
type headerInjector struct {
	http.ResponseWriter
	mw      *metricsWriter
	statusCode int
	written  bool
}

func newHeaderInjector(w http.ResponseWriter, mw *metricsWriter) *headerInjector {
	return &headerInjector{ResponseWriter: w, mw: mw}
}

func (h *headerInjector) WriteHeader(code int) {
	h.statusCode = code
	h.written = true

	// Set firstWrite on metricsWriter so TTFB is captured.
	// metricsWriter.WriteHeader will be called below (via chain) and will
	// see firstWrite is already set, so it won't overwrite.
	if h.mw.firstWrite.IsZero() {
		h.mw.firstWrite = time.Now()
	}

	// Inject headers BEFORE forwarding to client
	headers := h.ResponseWriter.Header()
	ttfb := h.mw.firstWrite.Sub(h.mw.startTime).Milliseconds()
	headers.Set("X-Router-TTFB-Ms", strconv.FormatInt(ttfb, 10))
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
	return &metricsWriter{
		ResponseWriter: w,
		startTime:      start,
		statusCode:     http.StatusOK,
		bodyBuffer:     &bytes.Buffer{},
	}
}

func (m *metricsWriter) WriteHeader(code int) {
	if m.firstWrite.IsZero() {
		m.firstWrite = time.Now()
	}
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
		if keep < 0 {
			keep = 0
		}
		remaining := m.bodyBuffer.Bytes()[keep:]
		m.bodyBuffer.Reset()
		m.bodyBuffer.Write(remaining)
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

// ProxyResponse holds the result of a proxied request.
type ProxyResponse struct {
	StatusCode int
	Header     http.Header
	Body       io.ReadCloser
}

// Proxy forwards the request to the target server (non-streaming, captured).
func Proxy(ctx context.Context, targetURL string, apiKey string, req *http.Request) (*ProxyResponse, error) {
	rawURL := targetURL
	if !strings.Contains(rawURL, "://") {
		rawURL = "https://" + rawURL
	}

	target, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("parse target URL: %w", err)
	}

	targetPath := target.Path + req.URL.Path
	if strings.HasSuffix(target.Path, "/v1") && strings.HasPrefix(req.URL.Path, "/v1") {
		targetPath = target.Path + req.URL.Path[1:]
	}

	proxy := &httputil.ReverseProxy{
		Director: func(r *http.Request) {
			r.URL.Scheme = target.Scheme
			r.URL.Host = target.Host
			r.URL.Path = targetPath
			r.URL.RawQuery = req.URL.RawQuery

			if apiKey != "" {
				r.Header.Set("Authorization", "Bearer "+apiKey)
			}

			r.Header.Del("Host")
		},
		ModifyResponse: func(r *http.Response) error {
			if r.Header != nil {
				r.Header.Del("Transfer-Encoding")
			}
			return nil
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
		},
	}

	recorder := &responseRecorder{}
	proxyReq := req.Clone(ctx)
	proxyReq.GetBody = nil

	proxy.ServeHTTP(recorder, proxyReq)

	body, err := io.ReadAll(recorder.body)
	if err != nil {
		return nil, fmt.Errorf("read proxy response: %w", err)
	}

	return &ProxyResponse{
		StatusCode: recorder.code,
		Header:     recorder.header,
		Body:       io.NopCloser(bytes.NewReader(body)),
	}, nil
}

// responseRecorder captures the HTTP response from the reverse proxy.
type responseRecorder struct {
	code   int
	header http.Header
	body   *bytes.Buffer
}

func (r *responseRecorder) Header() http.Header {
	return r.header
}

func (r *responseRecorder) WriteHeader(code int) {
	r.code = code
}

func (r *responseRecorder) Write(data []byte) (int, error) {
	return r.body.Write(data)
}

// modelRewriteWriter wraps a metricsWriter to replace the target model name
// with the original model name in JSON responses (both streaming and non-streaming).
// It wraps the metricsWriter so that metrics (TTFB, size, buffer) are captured
// for the actual data sent to the client.
type modelRewriteWriter struct {
	http.ResponseWriter
	oldModel string
	newModel string
}

func newModelRewriteWriter(mw *metricsWriter, oldModel, newModel string) *modelRewriteWriter {
	return &modelRewriteWriter{
		ResponseWriter: mw,
		oldModel:       oldModel,
		newModel:       newModel,
	}
}

func (r *modelRewriteWriter) Write(data []byte) (int, error) {
	rewritten := rewriteModelInResponse(data, r.oldModel, r.newModel)
	return r.ResponseWriter.Write(rewritten)
}

func (r *modelRewriteWriter) Flush() {
	if flusher, ok := r.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

// rewriteModelInResponse replaces "model":"oldModel" with "model":"newModel" in JSON data.
// It handles flexible whitespace around the colon and both single/multi-line formatting.
// For streaming responses (SSE), it rewrites each data: event fragment.
func rewriteModelInResponse(data []byte, oldModel, newModel string) []byte {
	if oldModel == "" || newModel == "" || oldModel == newModel {
		return data
	}

	result, n := replaceJSONModelValue(data, oldModel, newModel)
	if n == 0 {
		return data
	}
	return result
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
		default:
			b = append(b, c)
		}
	}
	return b
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

// StreamProxy forwards the request and streams the response directly to the writer.
// Returns an error on network failure (before any headers are written to w).
// HTTP 5xx responses are NOT treated as errors — they are forwarded to the client.
// If originalModel is non-empty and differs from targetModel, the "model" field
// in the response is rewritten from targetModel back to originalModel so the
// client sees its own model name.
// If rh is non-nil, X-Router-* headers are injected into the response.
func StreamProxy(ctx context.Context, targetURL string, apiKey string, req *http.Request, w http.ResponseWriter, targetModel, originalModel string, rh *RouterHeaders) (*ProxyMetrics, error) {
	rawURL := targetURL
	if !strings.Contains(rawURL, "://") {
		rawURL = "https://" + rawURL
	}

	target, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("parse target URL: %w", err)
	}

	targetPath := target.Path + req.URL.Path
	if strings.HasSuffix(target.Path, "/v1") && strings.HasPrefix(req.URL.Path, "/v1") {
		targetPath = target.Path + req.URL.Path[1:]
	}

	start := time.Now()
	mw := newMetricsWriter(w, start)
	var rw http.ResponseWriter = newModelRewriteWriter(mw, targetModel, originalModel)

	// Wrap with headerInjector to inject X-Router-* headers at WriteHeader time.
	if rh != nil {
		SetRouterHeaders(rw, rh) // Set ServerID and ServerName eagerly
		rw = newHeaderInjector(rw, mw)
	}

	// Capture network errors that ReverseProxy swallows.
	// Note: HTTP 5xx from the backend are NOT caught here — they are forwarded to the client.
	var proxyErr error
	proxy := &httputil.ReverseProxy{
		Director: func(r *http.Request) {
			r.URL.Scheme = target.Scheme
			r.URL.Host = target.Host
			r.URL.Path = targetPath
			r.URL.RawQuery = req.URL.RawQuery

			if apiKey != "" {
				r.Header.Set("Authorization", "Bearer "+apiKey)
			}

			r.Header.Del("Host")
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			proxyErr = err
		},
	}

	proxyReq := req.Clone(ctx)
	proxy.ServeHTTP(rw, proxyReq)

	if proxyErr != nil {
		return nil, proxyErr
	}

	m := mw.metrics()
	return &m, nil
}
