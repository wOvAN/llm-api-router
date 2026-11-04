package main

import (
	"context"
	"crypto/subtle"
	"embed"
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"os/signal"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"llm-api-router/admin"
	"llm-api-router/config"
	"llm-api-router/domain"
	"llm-api-router/metrics"
	"llm-api-router/pkg/log"
	"llm-api-router/router"
	"llm-api-router/trace"
)

//go:embed admin/static/*
var staticFS embed.FS

// readyState is the load-balancer-facing readiness flag. A replica under a load
// balancer has to leave the pool the moment it starts draining — otherwise the
// balancer keeps feeding it new generations that its 10s shutdown will cut short.
type readyState struct{ draining atomic.Bool }

func (rs *readyState) SetDraining(v bool) { rs.draining.Store(v) }

func (rs *readyState) ServeHTTP(w http.ResponseWriter, _ *http.Request) {
	if rs.draining.Load() {
		http.Error(w, "draining", http.StatusServiceUnavailable)
		return
	}
	_, _ = w.Write([]byte("ready"))
}

// serverStateMetrics renders the router's live view of its backends — health and
// rate-limit cooldown — as Prometheus gauges. The GUI shows both; a scrape has to
// see them too, or alerting is blind to the exact state that moves traffic.
func serverStateMetrics(health *config.HealthTracker, rl *config.RateLimiter,
	servers map[string]*domain.Server) string {
	var sb strings.Builder
	sb.WriteString("# HELP llm_router_server_healthy Server marked healthy by the health tracker (1) or not (0).\n")
	sb.WriteString("# TYPE llm_router_server_healthy gauge\n")
	ids := make([]string, 0, len(servers))
	for id := range servers {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	for _, id := range ids {
		healthy := 0
		if health.IsHealthy(id) {
			healthy = 1
		}
		sb.WriteString("llm_router_server_healthy{server=\"" + id + "\"} " + strconv.Itoa(healthy) + "\n")
	}
	sb.WriteString("# HELP llm_router_server_cooldown_seconds Remaining rate-limit cooldown per server.\n")
	sb.WriteString("# TYPE llm_router_server_cooldown_seconds gauge\n")
	for _, id := range ids {
		if d := rl.CooldownRemaining(id); d > 0 {
			sb.WriteString("llm_router_server_cooldown_seconds{server=\"" + id + "\"} " +
				strconv.Itoa(int(d.Round(time.Second).Seconds())) + "\n")
		}
	}
	return sb.String()
}

// tokenGuard gates a handler behind a bearer token. An empty token means open,
// which is the default: an existing deployment changes nothing until ADMIN_TOKEN
// is set. The comparison is constant-time because the token is a secret.
func tokenGuard(token string, next http.Handler) http.Handler {
	if token == "" {
		return next
	}
	const prefix = "Bearer "
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		got := req.Header.Get("Authorization")
		if !strings.HasPrefix(got, prefix) ||
			subtle.ConstantTimeCompare([]byte(strings.TrimPrefix(got, prefix)), []byte(token)) != 1 {
			w.Header().Set("WWW-Authenticate", `Bearer realm="llm-api-router admin"`)
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, req)
	})
}

func main() {
	log.InitFromEnv()

	showVersion := flag.Bool("version", false, "print version and exit")
	flag.Parse()
	if *showVersion {
		fmt.Println(Version)
		return
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	configFile := os.Getenv("CONFIG_FILE")
	if configFile == "" {
		configFile = "config.json"
	}

	// Admin API and /metrics carry every backend's api_key and the routing
	// config, so they can be gated behind a bearer token. Unset = open, which is
	// the historical behavior; a Prometheus scraper sets `authorization:` in its
	// scrape config, and the GUI asks for the token on its first 401.
	adminToken := os.Getenv("ADMIN_TOKEN")

	// Readiness: the load balancer pulls the replica out of the pool the moment
	// it starts draining, so a shutdown never cuts a generation short mid-stream.
	var ready readyState

	store, err := config.NewStore(configFile)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	if _, err := os.Stat(configFile); os.IsNotExist(err) {
		if err := store.Save(); err != nil {
			log.Fatalf("Failed to create default config: %v", err)
		}
		log.Infof("Created default config file: %s", configFile)
	}

	// Two ways a config silently stops saying what it means, both reported here
	// where the operator is still looking at the file: a key nobody reads (a
	// typo, a key from a newer router) and a rule naming a server that is not
	// defined (a rename, a delete by hand). Warnings, not a refusal — a router
	// configured for yesterday must still boot.
	if data, err := os.ReadFile(configFile); err == nil {
		for _, key := range config.UnknownFields(data) {
			log.Warnf("Config: unknown key %q — nothing reads it (a typo? it is ignored)", key)
		}
	}
	for _, ref := range store.AuditServerRefs() {
		log.Warnf("Config: %s — requests on it will fail until the server exists", ref)
	}

	metricsStore := metrics.New(100)

	capsCache := config.NewCapabilitiesCache()
	healthTracker := config.NewHealthTracker(store, 30*time.Second, capsCache)
	healthTracker.Start()

	// Rate limiter: skip server after 5 failures within 60s, cooldown for 5min.
	// 429/401/403 cooldown immediately; per-server cooldown_time (seconds)
	// overrides the global duration.
	rateLimiter := config.NewRateLimiter(5, 60*time.Second, 5*time.Minute)
	rateLimiter.SetCooldownOverride(func(id string) time.Duration {
		if srv, ok := store.GetServer(id); ok && srv.CooldownTime > 0 {
			return time.Duration(srv.CooldownTime) * time.Second
		}
		return 0
	})

	// Quota tracker: per-server TPM/RPM limits from config (lazy lookup).
	quotaTracker := config.NewQuotaTracker()
	quotaTracker.SetLimitOverride(func(id string) (tpm, rpm int64) {
		if srv, ok := store.GetServer(id); ok {
			return srv.TPMLimit, srv.RPMLimit
		}
		return 0, 0
	})

	// On-demand request trace: a memory-only ring plus the live capture window,
	// armed from the config defaults at startup (the admin toggle is never
	// persisted, so a restart can not leave prompt capture switched on).
	traceStore := trace.New(100, store.GetSettings().Trace)

	apiRouter := router.New(store, metricsStore, healthTracker, rateLimiter, quotaTracker, traceStore, capsCache)
	// The client-facing edge, wrapped outside-in: CORS (a browser client reads the
	// answer, then needs permission to read the routing headers) and the inbound
	// rate limit (one runaway agent loop otherwise occupies the whole box).
	// Mounted on the routed endpoints only: the admin GUI and /metrics keep their
	// own story, and a limiter there is how an operator locks themselves out of
	// the console during an incident. Both are nil (pass-through) by default.
	settings := store.GetSettings()
	apiHandler := newClientEdge(store, http.HandlerFunc(apiRouter.Handle))

	adminHandler := admin.NewHandler(store, metricsStore, healthTracker, traceStore)
	// The GUI reports which servers are in their rate-limit cooldown, and why.
	adminHandler.SetRateLimiter(rateLimiter)
	// A reload changes the settings the client edge is built from, so the edge is
	// rebuilt with them.
	adminHandler.SetOnReload(apiHandler.Rebuild)

	adminStatic, err := fs.Sub(staticFS, "admin/static")
	if err != nil {
		log.Fatalf("Failed to load admin static files: %v", err)
	}

	if settings.Inbound.RPS > 0 {
		scope := "shared"
		if settings.Inbound.PerIP {
			scope = "per client address"
		}
		log.Infof("Inbound rate limit: %.4g rps (burst %d, %s)", settings.Inbound.RPS,
			max(settings.Inbound.Burst, 1), scope)
	}
	if len(settings.CORS.Origins) == 0 && settings.CORS.Enabled {
		log.Warnf("CORS is enabled for ANY origin — set settings.cors.origins to name the ones you mean")
	}

	mux := http.NewServeMux()

	mux.Handle("/readyz", &ready)
	mux.Handle("/v1/", apiHandler)
	// llama.cpp keeps its TTS route (and the STT alias) outside /v1 — route them
	// like any other endpoint so audio traffic isn't 404'd at the mux.
	mux.Handle("/tts", apiHandler)
	mux.Handle("/audio/", apiHandler)
	mux.Handle("/admin/api/", tokenGuard(adminToken, adminHandler))

	// Build version (exact pattern beats the /admin/api/ catch-all above).
	mux.Handle("/admin/api/version", tokenGuard(adminToken, http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]string{"version": Version}); err != nil {
			log.Errorf("failed to encode version response: %v", err)
		}
	})))

	// Prometheus text-format metrics endpoint (zero deps).
	mux.Handle("/metrics", tokenGuard(adminToken, http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
		out := metricsStore.PrometheusMetrics() +
			serverStateMetrics(healthTracker, rateLimiter, store.GetConfig().Servers)
		if _, err := w.Write([]byte(out)); err != nil {
			log.Errorf("[%s] failed to write metrics response: %v", req.URL.Path, err)
		}
	})))

	mux.HandleFunc("/admin/", func(w http.ResponseWriter, req *http.Request) {
		path := strings.TrimPrefix(req.URL.Path, "/admin/")

		if path == "" {
			http.ServeFileFS(w, req, adminStatic, "index.html")
			return
		}

		f, err := adminStatic.Open(path)
		if err != nil {
			http.ServeFileFS(w, req, adminStatic, "index.html")
			return
		}
		if err := f.Close(); err != nil {
			log.Errorf("[%s] failed to close admin static file: %v", req.URL.Path, err)
		}

		http.ServeFileFS(w, req, adminStatic, path)
	})

	addr := ":" + port

	srv := &http.Server{
		Addr: addr,
		// Bound the header read: without it a client that opens a connection and
		// dribbles bytes (Slowloris) holds a connection slot forever. The body and
		// the response stay unbounded on purpose — a long generation is normal
		// here, and the upstream side has its own bounds
		// (proxy.ResponseHeaderTimeout, the drain cap).
		ReadHeaderTimeout: 10 * time.Second,
		Handler:           mux,
	}

	// Graceful shutdown on SIGINT/SIGTERM
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-stop
		log.Infof("Shutting down...")
		ready.SetDraining(true)
		healthTracker.Stop()
		ctx, cancel := context.WithTimeout(context.Background(), drainTimeout())
		defer cancel()
		if err := srv.Shutdown(ctx); err != nil {
			log.Errorf("Server shutdown error: %v", err)
		}
	}()

	log.Infof("LLM API Router %s starting on %s", Version, addr)
	log.Infof("Admin GUI: http://localhost%s/admin", addr)
	log.Infof("API routes: http://localhost%s/v1/*", addr)
	log.Infof("Shutdown drain: %s (DRAIN_TIMEOUT)", drainTimeout())
	if adminToken == "" {
		log.Warnf("ADMIN_TOKEN is unset — the admin API (backend keys, routing config) and /metrics are unauthenticated")
	}

	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Server failed: %v", err)
	}
}

// drainTimeout bounds the graceful-shutdown drain. 10s is the right answer for
// an API that answers in milliseconds and a footgun for one that generates for
// five minutes: /readyz pulls the replica out of the pool the moment the signal
// lands, so nothing new arrives during the drain and the only thing the bound
// protects is a wedged connection. A deployment that streams long generations
// wants minutes, not seconds.
// clientEdge is the client-facing middleware chain, rebuilt on a config reload.
// A middleware is built once from its settings, so mounting the built chain
// freezes settings.inbound and settings.cors at their startup values: raising the
// rate limit in config.json and hitting reload — the GUI's natural apply — would
// edit a file nothing reads any more. settings.convert is read per request; these
// two are the only chain assembled at startup, so the chain itself is what has to
// be swappable.
type clientEdge struct {
	store *config.Store
	inner http.Handler
	next  atomic.Pointer[http.Handler]
}

func newClientEdge(store *config.Store, inner http.Handler) *clientEdge {
	e := &clientEdge{store: store, inner: inner}
	e.Rebuild()
	return e
}

// Rebuild assembles the chain from the settings the store holds right now.
func (e *clientEdge) Rebuild() {
	s := e.store.GetSettings()
	h := e.inner
	h = config.NewInboundLimiter(s.Inbound).Middleware(h)
	h = config.NewCORS(s.CORS).Middleware(h)
	e.next.Store(&h)
}

func (e *clientEdge) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if h := e.next.Load(); h != nil {
		(*h).ServeHTTP(w, req)
	}
}

func drainTimeout() time.Duration {
	v := os.Getenv("DRAIN_TIMEOUT")
	if v == "" {
		return 10 * time.Second
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		log.Warnf("DRAIN_TIMEOUT=%q is not a duration (try 30s, 5m) — using 10s", v)
		return 10 * time.Second
	}
	return d
}
