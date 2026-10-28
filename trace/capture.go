package trace

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"hash"
	"strings"
	"sync"
	"time"
)

// Capture collects one traced request. The router builds it at request start
// (the decision is made once, so a request that starts inside the window is
// traced to the end even if the window expires mid-generation) and the proxy's
// writer chain streams the response into it — no second read of the response,
// nothing deep-copied.
type Capture struct {
	mu     sync.Mutex
	id     string
	ts     time.Time
	method string
	path   string
	model  string

	request  string
	reqTrunc bool
	class    string
	features []string

	resp      bytes.Buffer
	respBytes int64
	respType  string
	sha       hash.Hash
}

// NewCapture starts a capture of a request: the body is redacted here, at the
// one moment the whole request body is in hand.
func NewCapture(method, path, model, contentType string, body []byte) *Capture {
	var id [4]byte
	if _, err := rand.Read(id[:]); err != nil {
		// crypto/rand failing is a broken platform, not a request error; a
		// constant id is better than losing the trace.
		id = [4]byte{0, 0, 0, 0}
	}
	req, trunc := RedactRequest(body, contentType)
	class, features := Classify(path, contentType, body)
	return &Capture{
		id:       hex.EncodeToString(id[:]),
		ts:       time.Now(),
		method:   method,
		path:     path,
		model:    model,
		request:  req,
		reqTrunc: trunc,
		class:    class,
		features: features,
		sha:      sha256.New(),
	}
}

// ID is the trace id, surfaced to the client as X-Router-Trace-Id.
func (c *Capture) ID() string { return c.id }

// ResponseContentType records the response's content type before its first
// chunk (the proxy calls it at WriteHeader): JSON and SSE bodies are stored,
// everything else (a TTS WAV, raw PCM) becomes a stub.
func (c *Capture) ResponseContentType(ct string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.respType = ct
}

// WriteResponse records one response chunk.
func (c *Capture) WriteResponse(p []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.respBytes += int64(len(p))
	// The hash runs over the whole body so a stub identifies the payload even
	// when the stored prefix does not contain it.
	if _, err := c.sha.Write(p); err != nil {
		return
	}
	if isBinaryType(c.respType) {
		return // the stub carries size + hash; the bytes are useless here
	}
	// Store at most MaxResponseBytes; what did not fit is counted, not kept.
	if room := MaxResponseBytes - c.resp.Len(); room > 0 {
		if len(p) > room {
			p = p[:room]
		}
		c.resp.Write(p)
	}
}

// HasResponse reports whether anything arrived from the backend.
func (c *Capture) HasResponse() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.respBytes > 0
}

// Entry finalizes the capture, computing the digest once from what was stored.
func (c *Capture) Entry() Entry {
	c.mu.Lock()
	defer c.mu.Unlock()
	e := Entry{
		ID:               c.id,
		Timestamp:        c.ts,
		Method:           c.method,
		Path:             c.path,
		Model:            c.model,
		Class:            c.class,
		Features:         c.features,
		Request:          c.request,
		RequestTruncated: c.reqTrunc,
	}
	if c.respBytes == 0 {
		return e // nothing arrived: the router fills Response from the error body
	}
	if isBinaryType(c.respType) {
		e.Response = stub(coarseKind(c.respType), c.respBytes, c.sha.Sum(nil))
		return e
	}
	// The digest reads the stored prefix; a capped body loses whatever the
	// backend sent past it (the tail of a stream is where usage/timings ride, so
	// the cap is generous rather than exact).
	e.Digest = DigestResponse(c.resp.Bytes(), c.respType)
	e.Response = c.resp.String()
	if dropped := c.respBytes - int64(c.resp.Len()); dropped > 0 {
		e.Response += truncationMarker(dropped)
		e.ResponseTruncated = true
	}
	return e
}

// isBinaryType reports a response body that is neither JSON, SSE, nor plain
// text. The text family matters: a backend's 4xx body arrives as text/plain
// (net/http's http.Error) and an error page is the single most useful thing in
// a trace, so it is stored verbatim, not stubbed.
func isBinaryType(contentType string) bool {
	ct := strings.ToLower(contentType)
	return !strings.Contains(ct, "json") && !strings.Contains(ct, "text/event-stream") &&
		!strings.Contains(ct, "text/")
}
