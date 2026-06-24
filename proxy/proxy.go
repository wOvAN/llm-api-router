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

// StreamProxy forwards the request and streams the response directly to the writer.
// Returns an error on network failure (before any headers are written to w).
// HTTP 5xx responses are NOT treated as errors — they are forwarded to the client.
func StreamProxy(ctx context.Context, targetURL string, apiKey string, req *http.Request, w http.ResponseWriter) (*ProxyMetrics, error) {
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
	proxy.ServeHTTP(mw, proxyReq)

	if proxyErr != nil {
		return nil, proxyErr
	}

	m := mw.metrics()
	return &m, nil
}
