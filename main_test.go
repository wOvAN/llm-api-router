package main

import (
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"llm-api-router/config"
	"llm-api-router/domain"
)

// The admin GUI is one embedded file, so a GUI test is a text assertion on that
// source. These three pin the protocol-conversion surface: an operator must be
// able to declare which endpoints a backend natively serves (an empty declaration
// is exactly what keeps a server byte-for-byte passthrough, so the control, its
// reset on add and its restore on edit are the feature), a form save must not
// erase the fields it does not render, and a converted request must show its edge
// and its notes in Recent Requests.

func guiSource(t *testing.T) string {
	t.Helper()
	b, err := staticFS.ReadFile("admin/static/index.html")
	if err != nil {
		t.Fatalf("read embedded GUI: %v", err)
	}
	return string(b)
}

// funcBody returns the text of a top-level function, from its signature to the
// closing brace at column 0 — enough to assert on what a function does without
// pinning its whole formatting.
func funcBody(t *testing.T, src, signature string) string {
	t.Helper()
	i := strings.Index(src, signature)
	if i < 0 {
		t.Fatalf("%s not in the GUI source", signature)
	}
	end := strings.Index(src[i:], "\n}")
	if end < 0 {
		t.Fatalf("%s has no closing brace", signature)
	}
	return src[i : i+end]
}

func TestGUIServerFormEditsProtocols(t *testing.T) {
	src := guiSource(t)

	for _, id := range []string{"server-proto-chat", "server-proto-messages",
		"server-proto-responses", "server-proto-completions"} {
		if !strings.Contains(src, `id="`+id+`"`) {
			t.Errorf("the server form has no %s input — an operator can not declare the endpoints a backend serves", id)
		}
	}
	// Empty means "undeclared = passthrough"; a control whose empty state is a
	// behavior change has to say so, or an operator clears it by accident.
	if i := strings.Index(src, "server-proto-chat"); i < 0 ||
		!strings.Contains(src[max(0, i-400):i], "passthrough") {
		t.Error("the endpoints control does not state that an empty declaration means passthrough")
	}

	save := funcBody(t, src, "function getServerProtocols()")
	for _, want := range []string{"/v1/chat/completions", "/v1/messages", "/v1/responses", "/v1/completions"} {
		if !strings.Contains(save, want) {
			t.Errorf("getServerProtocols does not offer %s", want)
		}
	}
	// `state` has no control (nothing reads it at runtime yet), so a save must
	// carry the config-declared flag over per path — a rebuild from the checkboxes
	// alone would silently erase it.
	if !strings.Contains(save, "state:") || !strings.Contains(save, "editingServer") {
		t.Error("getServerProtocols rebuilds protocols from the checkboxes and drops the `state` flag")
	}

	if !strings.Contains(funcBody(t, src, "async function saveServer()"), "protocols: getServerProtocols()") {
		t.Error("saveServer never sends protocols, so a declaration can not be created from the GUI")
	}
	for _, fn := range []string{"function openServerModal(", "function editServer("} {
		body := funcBody(t, src, fn)
		if !strings.Contains(body, "setServerProtocols(") {
			t.Errorf("%s does not set the endpoint checkboxes (add keeps the previous server's marks, edit shows none)", fn)
		}
	}
}

func TestGUIServerFormPreservesUnmanagedFields(t *testing.T) {
	body := funcBody(t, guiSource(t), "async function saveServer()")
	if !strings.Contains(body, "Object.assign({}, editingServer,") {
		t.Error("saveServer rebuilds the server from its inputs, erasing every field the form does not render")
	}
}

func TestGUIShowsConversion(t *testing.T) {
	src := guiSource(t)
	i := strings.Index(src, "r.convert_notes")
	if i < 0 {
		t.Fatal("a converted request reports nothing in Recent Requests")
	}
	window := src[max(0, i-600) : i+300]
	for _, want := range []string{"r.convert", "convert_notes"} {
		if !strings.Contains(window, want) {
			t.Errorf("the conversion pill shows neither %q", want)
		}
	}
	if !strings.Contains(window, "escAttr(") {
		t.Error("the conversion notes go into a title attribute unescaped")
	}
}

// Recent Requests rows are correlated with the client's own logs by request id,
// and the row is built by concatenation — an unescaped value lands in a title
// attribute verbatim.
func TestGUIShowsRequestID(t *testing.T) {
	src := guiSource(t)
	i := strings.Index(src, "request id:")
	if i < 0 {
		t.Fatal("a recent request carries no request id, so a client cannot quote one")
	}
	if !strings.Contains(src[i:i+120], "esc(r.request_id") {
		t.Errorf("the request id is not rendered/escaped: %s", src[i:i+120])
	}
}

// A server in its rate-limit cooldown is skipped by the router while the health
// probe still calls it healthy — without a third state the GUI shows a healthy
// server that answers nothing, and the reason (a 401 key problem, a 429 hint)
// never reaches the operator.
func TestGUIShowsCooldown(t *testing.T) {
	src := guiSource(t)
	i := strings.Index(src, "cooldown_seconds")
	if i < 0 {
		t.Fatal("the servers table never reports a rate-limit cooldown")
	}
	window := src[i-400 : min(i+400, len(src))]
	for _, want := range []string{"cooldown_reason", "esc("} {
		if !strings.Contains(window, want) {
			t.Errorf("the cooldown tag lacks %s", want)
		}
	}
}

// The admin API serves every backend's api_key and rewrites the routing config,
// and /metrics quotes the traffic it saw. Both are gated behind a bearer token:
// ADMIN_TOKEN unset = open (an existing deployment keeps working untouched),
// ADMIN_TOKEN set = every data-bearing route needs it. The GUI shell stays open
// so it can ask for the token.
func TestTokenGuardRequiresBearer(t *testing.T) {
	ok := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })

	guarded := tokenGuard("s3cret", ok)
	for _, tc := range []struct {
		name   string
		header string
		want   int
	}{
		{"absent", "", http.StatusUnauthorized},
		{"wrong scheme", "Basic s3cret", http.StatusUnauthorized},
		{"wrong token", "Bearer nope", http.StatusUnauthorized},
		{"right token", "Bearer s3cret", http.StatusOK},
	} {
		req := httptest.NewRequest(http.MethodGet, "/admin/api/servers", nil)
		if tc.header != "" {
			req.Header.Set("Authorization", tc.header)
		}
		w := httptest.NewRecorder()
		guarded.ServeHTTP(w, req)
		if got := w.Result().StatusCode; got != tc.want {
			t.Errorf("%s: status %d, want %d", tc.name, got, tc.want)
		}
	}

	// Unset token = feature off, and off must mean off: no header, no 401.
	w := httptest.NewRecorder()
	tokenGuard("", ok).ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if got := w.Result().StatusCode; got != http.StatusOK {
		t.Fatalf("empty ADMIN_TOKEN must leave the API open, got %d", got)
	}
}

// A replica under a load balancer must be pulled out of the pool the moment it
// starts draining, so the balancer stops feeding it new generations while the
// drain finishes the ones it accepted.
func TestReadinessReportsDrain(t *testing.T) {
	var rs readyState
	w := httptest.NewRecorder()
	rs.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if got := w.Result().StatusCode; got != http.StatusOK {
		t.Fatalf("ready: status %d, want 200", got)
	}

	rs.SetDraining(true)
	w = httptest.NewRecorder()
	rs.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if got := w.Result().StatusCode; got != http.StatusServiceUnavailable {
		t.Fatalf("draining: status %d, want 503 (the balancer must pull the replica)", got)
	}
}

// The router's own view of its backends — who is healthy, who is cooling down
// and for how long — lives in memory and is visible in the GUI. A Prometheus
// scrape has to see it too, or the alerting layer is blind to exactly the state
// that decides where traffic goes.
func TestServerStateMetricsRendersHealth(t *testing.T) {
	store, err := config.NewStore(filepath.Join(t.TempDir(), "config.json"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	for _, id := range []string{"s1", "s2"} {
		if err := store.AddServer(&domain.Server{ID: id, Name: id, URL: "http://x"}); err != nil {
			t.Fatalf("AddServer: %v", err)
		}
	}
	health := config.NewHealthTracker(store, time.Hour, nil)
	health.Start()
	defer health.Stop()
	health.MarkUnhealthy("s2")

	rl := config.NewRateLimiter(5, time.Minute, 5*time.Minute)
	rl.RecordFailure("s1", http.StatusTooManyRequests, 30*time.Second)

	out := serverStateMetrics(health, rl, store.GetConfig().Servers)
	for _, want := range []string{
		`llm_router_server_healthy{server="s1"} 1`,
		`llm_router_server_healthy{server="s2"} 0`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in:\n%s", want, out)
		}
	}
	if !strings.Contains(out, `llm_router_server_cooldown_seconds{server="s1"} 30`) {
		t.Errorf("cooldown not reported (want ~30s for s1):\n%s", out)
	}
	if strings.Contains(out, `llm_router_server_cooldown_seconds{server="s2"}`) {
		t.Errorf("a server that is not cooling down must not report a cooldown:\n%s", out)
	}
}

// A rule can balance across a pool of interchangeable backends, and the GUI is
// the only place an operator can express it: the dialog needs the mode select,
// the per-member weight it renumbers, and the table a place to show the pool —
// a rule that balances is a rule whose "fallbacks" are traffic targets, not a
// rescue chain, and nothing in the row says that unless the tag does.
func TestGUIEditsBalanceAndWeight(t *testing.T) {
	src := guiSource(t)
	for _, want := range []string{
		`id="rule-balance"`,
		`value="round-robin"`,
		`className = 'fb-weight'`,
		`fb.weight`,
		`r.balance === 'round-robin'`,
		`data.balance`,
	} {
		if !strings.Contains(src, want) {
			t.Errorf("the rule dialog is missing %s", want)
		}
	}
}

// `/admin/api/version` matches the mux as its own (exact) pattern, so its guard
// is a separate call site from the /admin/api/ prefix — one mux.Handle line that
// forgets tokenGuard turns the version endpoint into a hole in an ADMIN_TOKEN
// deployment. The pin is on the wiring itself, because a unit test of
// tokenGuard's behavior cannot see which routes the server actually guards.
func TestAllAdminRoutesAreGuarded(t *testing.T) {
	src, err := os.ReadFile("main.go")
	if err != nil {
		t.Fatalf("read main.go: %v", err)
	}
	routed := 0
	for _, line := range strings.Split(string(src), "\n") {
		trimmed := strings.TrimSpace(line)
		if !strings.HasPrefix(trimmed, `mux.Handle("/admin/api`) {
			continue
		}
		routed++
		if !strings.Contains(line, "tokenGuard(adminToken,") {
			t.Errorf("admin route without a token guard: %s", trimmed)
		}
	}
	if routed < 2 {
		t.Fatalf("found %d admin routes in main.go — the file no longer wires them the way this test reads it", routed)
	}
}

// The drain bound has to fit the traffic: 10s is right for an API that answers
// in milliseconds and cuts a five-minute generation in half. /readyz already
// pulls the replica out of the pool the moment the signal lands, so the drain is
// free to be generous — it just has to be the operator's call.
func TestDrainTimeoutConfigurable(t *testing.T) {
	for _, tc := range []struct {
		env  string
		want time.Duration
	}{
		{"", 10 * time.Second},
		{"5m", 5 * time.Minute},
		{"0s", 0},
		{"nonsense", 10 * time.Second},
	} {
		t.Setenv("DRAIN_TIMEOUT", tc.env)
		if got := drainTimeout(); got != tc.want {
			t.Errorf("DRAIN_TIMEOUT=%q → %v, want %v", tc.env, got, tc.want)
		}
	}
}

// The client-facing edge is where a browser client gets permission to read the
// routing headers and where a runaway agent loop gets a 429 — so both middlewares
// have to be on the path a request actually takes, and off the path the admin
// console takes (a limiter there is how an operator locks themselves out of the
// console during the incident they are logging into it to fix).
func TestRoutedEndpointsCarryClientMiddlewares(t *testing.T) {
	src, err := os.ReadFile("main.go")
	if err != nil {
		t.Fatalf("read main.go: %v", err)
	}
	source := string(src)
	for _, route := range []string{`mux.Handle("/v1/", apiHandler)`, `mux.Handle("/tts", apiHandler)`, `mux.Handle("/audio/", apiHandler)`} {
		if !strings.Contains(source, route) {
			t.Errorf("main.go no longer routes %s through the wrapped handler", route)
		}
	}
	for _, mw := range []string{"config.NewInboundLimiter(settings.Inbound).Middleware(apiHandler)",
		"config.NewCORS(settings.CORS).Middleware(apiHandler)"} {
		if !strings.Contains(source, mw) {
			t.Errorf("main.go lost a client-edge middleware: %s", mw)
		}
	}
	if strings.Contains(source, `tokenGuard(adminToken, apiHandler)`) {
		t.Error("the admin console must not pass through the client rate limiter")
	}
}

// Averages hide the tail, and the tail is what pages. The summary buckets the
// Prometheus scrape already uses are what the GUI's percentile column reads.
func TestGUIShowsLatencyPercentiles(t *testing.T) {
	src, err := fs.ReadFile(staticFS, "admin/static/index.html")
	if err != nil {
		t.Fatalf("read embedded GUI: %v", err)
	}
	source := string(src)
	for _, want := range []string{
		"function percentile(buckets, q)",
		"latency_ms_buckets",
		"data-col=\"p95\"",
		"${pcell}",
	} {
		if !strings.Contains(source, want) {
			t.Errorf("the GUI lost %q", want)
		}
	}
	// The bounds are a Go list; the GUI copy has to say where it came from.
	if !strings.Contains(source, "domain.LatencyBucketsMs") {
		t.Error("the GUI's bucket bounds lost their pointer to the Go list they mirror")
	}
}
