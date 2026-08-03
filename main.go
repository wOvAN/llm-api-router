package main

import (
	"context"
	"embed"
	"io/fs"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"llm-api-router/admin"
	"llm-api-router/config"
	"llm-api-router/metrics"
	"llm-api-router/pkg/log"
	"llm-api-router/router"
)

//go:embed admin/static/*
var staticFS embed.FS

func main() {
	log.InitFromEnv()

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	configFile := os.Getenv("CONFIG_FILE")
	if configFile == "" {
		configFile = "config.json"
	}

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

	metricsStore := metrics.New(100)

	healthTracker := config.NewHealthTracker(store, 30*time.Second)
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

	apiRouter := router.New(store, metricsStore, healthTracker, rateLimiter, quotaTracker)
	adminHandler := admin.NewHandler(store, metricsStore, healthTracker)

	adminStatic, _ := fs.Sub(staticFS, "admin/static")

	mux := http.NewServeMux()

	mux.HandleFunc("/v1/", apiRouter.Handle)
	mux.HandleFunc("/admin/api/", adminHandler.ServeHTTP)

	// Prometheus text-format metrics endpoint (zero deps).
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
		_, _ = w.Write([]byte(metricsStore.PrometheusMetrics()))
	})

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
		_ = f.Close()

		http.ServeFileFS(w, req, adminStatic, path)
	})

	addr := ":" + port

	srv := &http.Server{
		Addr:    addr,
		Handler: mux,
	}

	// Graceful shutdown on SIGINT/SIGTERM
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-stop
		log.Infof("Shutting down...")
		healthTracker.Stop()
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := srv.Shutdown(ctx); err != nil {
			log.Errorf("Server shutdown error: %v", err)
		}
	}()

	log.Infof("LLM API Router starting on %s", addr)
	log.Infof("Admin GUI: http://localhost%s/admin", addr)
	log.Infof("API routes: http://localhost%s/v1/*", addr)

	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Server failed: %v", err)
	}
}
