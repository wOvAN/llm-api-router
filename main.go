package main

import (
	"embed"
	"io/fs"
	"net/http"
	"os"
	"strings"
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
	rateLimiter := config.NewRateLimiter(5, 60*time.Second, 5*time.Minute)

	apiRouter := router.New(store, metricsStore, healthTracker, rateLimiter)
	adminHandler := admin.NewHandler(store, metricsStore, healthTracker)

	adminStatic, _ := fs.Sub(staticFS, "admin/static")

	mux := http.NewServeMux()

	mux.HandleFunc("/v1/", apiRouter.Handle)
	mux.HandleFunc("/admin/api/", adminHandler.ServeHTTP)

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
	log.Infof("LLM API Router starting on %s", addr)
	log.Infof("Admin GUI: http://localhost%s/admin", addr)
	log.Infof("API routes: http://localhost%s/v1/*", addr)

	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
