package proxy

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
)

// newRecordingProxy returns an httptest server that answers every request
// itself. A plain HTTP proxy receives the absolute-form request target from
// Go's client, so the recorded URL proves the request was routed through it.
func newRecordingProxy(t *testing.T, seen *string, hits *atomic.Int64) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		*seen = r.URL.String()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"cmpl-1","model":"m"}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestTransportForDirectAndCached(t *testing.T) {
	tr, err := TransportFor("")
	if err != nil {
		t.Fatalf("TransportFor(\"\"): %v", err)
	}
	if tr != http.RoundTripper(upstreamTransport) {
		t.Error("empty proxy must return the shared direct transport")
	}

	first, err := TransportFor("http://proxy.test:3128")
	if err != nil {
		t.Fatalf("TransportFor: %v", err)
	}
	second, err := TransportFor("http://proxy.test:3128")
	if err != nil {
		t.Fatalf("TransportFor: %v", err)
	}
	if first != second {
		t.Error("same proxy URL must reuse one transport (connection pooling)")
	}

	schemeless, err := TransportFor("proxy.test:3128")
	if err != nil {
		t.Fatalf("TransportFor(schemeless): %v", err)
	}
	if schemeless != first {
		t.Error("scheme-less proxy URL should normalize to http:// and share the transport")
	}

	if _, err := TransportFor("://bad proxy"); err == nil {
		t.Error("expected error for invalid proxy URL")
	}
}

func TestStreamProxyUsesPerServerProxy(t *testing.T) {
	var seen string
	var hits atomic.Int64
	p := newRecordingProxy(t, &seen, &hits)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"m"}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	pm, err := StreamProxy(context.Background(), "http://origin.invalid/v1", "key", req, w,
		"m", "m", &RouterHeaders{ServerID: "s1", Proxy: p.URL}, false, nil)
	if err != nil {
		t.Fatalf("StreamProxy: %v", err)
	}
	if hits.Load() != 1 {
		t.Fatalf("proxy hits = %d, want 1 (request bypassed the proxy)", hits.Load())
	}
	if seen != "http://origin.invalid/v1/chat/completions" {
		t.Errorf("proxy saw %q, want the absolute origin URL", seen)
	}
	if pm.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200", pm.StatusCode)
	}
}

func TestStreamProxyInvalidProxyURLFailsBeforeHeaders(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"m"}`))
	w := httptest.NewRecorder()

	_, err := StreamProxy(context.Background(), "http://origin.invalid/v1", "", req, w,
		"m", "m", &RouterHeaders{Proxy: "://bad proxy"}, false, nil)
	if err == nil {
		t.Fatal("expected error for invalid proxy URL")
	}
	if w.Body.Len() != 0 {
		t.Errorf("client must see no bytes on pre-response failure, got %q", w.Body.String())
	}
}

// newSocks5Server returns a minimal no-auth SOCKS5 server that tunnels
// CONNECT requests to their target. Enough to prove the transport really
// routes through SOCKS, not just accepts the scheme.
func newSocks5Server(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	t.Cleanup(func() { _ = l.Close() })
	go func() {
		for {
			c, err := l.Accept()
			if err != nil {
				return
			}
			go func(c net.Conn) {
				defer func() { _ = c.Close() }()
				readFull := func(b []byte) bool {
					_, err := io.ReadFull(c, b)
					return err == nil
				}
				greet := make([]byte, 2)
				if !readFull(greet) || greet[0] != 0x05 {
					return
				}
				if greet[1] > 0 {
					methods := make([]byte, greet[1])
					if !readFull(methods) {
						return
					}
				}
				if _, err := c.Write([]byte{0x05, 0x00}); err != nil {
					return
				}
				hdr := make([]byte, 4)
				if !readFull(hdr) {
					return
				}
				var host string
				var portB []byte
				switch hdr[3] {
				case 0x01: // IPv4: 4-byte IP + 2-byte port in one read
					b := make([]byte, 6)
					if !readFull(b) {
						return
					}
					host = net.IP(b[:4]).String()
					portB = b[4:6]
				case 0x03: // domain: 1-byte length + domain + 2-byte port
					lenB := make([]byte, 1)
					if !readFull(lenB) {
						return
					}
					hostB := make([]byte, lenB[0])
					if !readFull(hostB) {
						return
					}
					host = string(hostB)
					portB = make([]byte, 2)
					if !readFull(portB) {
						return
					}
				case 0x04: // IPv6: 16-byte IP + 2-byte port
					b := make([]byte, 16)
					if !readFull(b) {
						return
					}
					host = net.IP(b).String()
					portB = make([]byte, 2)
					if !readFull(portB) {
						return
					}
				default:
					return
				}
				if _, err := c.Write([]byte{0x05, 0x00, 0x00, 0x01, 0, 0, 0, 0, 0, 0}); err != nil {
					return
				}
				target, err := net.Dial("tcp", net.JoinHostPort(host, strconv.Itoa(int(portB[0])<<8|int(portB[1]))))
				if err != nil {
					return
				}
				defer func() { _ = target.Close() }()
				go func() { _, _ = io.Copy(target, c) }()
				_, _ = io.Copy(c, target)
			}(c)
		}
	}()
	return l.Addr().String()
}

func TestStreamProxyUsesSocks5Proxy(t *testing.T) {
	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"cmpl-1","model":"m"}`))
	}))
	defer origin.Close()

	socksAddr := newSocks5Server(t)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"m"}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	pm, err := StreamProxy(context.Background(), origin.URL, "", req, w,
		"m", "m", &RouterHeaders{ServerID: "s1", Proxy: "socks5://" + socksAddr}, false, nil)
	if err != nil {
		t.Fatalf("StreamProxy via socks5: %v", err)
	}
	if pm.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200", pm.StatusCode)
	}
}

func TestTransportForRejectsUnknownScheme(t *testing.T) {
	if _, err := TransportFor("ftp://proxy.test:2121"); err == nil {
		t.Error("expected error for unsupported proxy scheme")
	}
}
