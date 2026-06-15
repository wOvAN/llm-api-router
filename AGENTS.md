# AGENTS.md — LLM API Router

## Build & run

```bash
go build -o llm-api-router .
PORT=9090 CONFIG_FILE=myconfig.json ./llm-api-router
```

Go 1.26, **zero external deps** (stdlib only).

## Commands

```bash
go build -o llm-api-router .     # build binary
go test ./...                     # run all tests
go test -v ./...                  # verbose
go test -v ./router               # single package
go fmt ./...                      # format
golangci-lint run ./...           # lint (config: .golangci.yml)
```

Or via `Taskfile.yml` (install `go-task`): `task`, `task build`, `task test`, `task testv`, `task fmt`, `task lint`, `task clean`.

## CI pipeline (`.github/workflows/ci.yml`)

On push/PR to `main`: `go fmt` → `golangci-lint` (v2.9, action v7) → `go build` → `go test -v ./...` → Docker build. Both jobs (ci, docker) run in parallel.

## Architecture

| Package | File(s) | Role |
|---------|---------|------|
| `main` | `main.go` | Entrypoint: `//go:embed admin/static/*`, wire handlers, `http.ListenAndServe` |
| `domain` | `domain/*.go` | Models: `Config`, `Server`, `RoutingRule`, `RequestMetric`, `Summary`, `FallbackEntry` |
| `config` | `config/store.go` | Thread-safe store with JSON persistence & auto-migration of legacy fields |
| `router` | `router/router.go` | `/v1/*` handler: extract model → rule lookup → rewrite body → proxy chain w/ fallbacks → metrics |
| `proxy` | `proxy/proxy.go`, `proxy/usage.go` | `StreamProxy` (SSE streaming relays to client) + `RewriteModelInBody` + usage extraction from response |
| `metrics` | `metrics/metrics.go` | Ring buffer (last **100** requests), aggregated summaries by model & server |
| `admin` | `admin/handler.go` | REST CRUD at `/admin/api/`, inline routing, saves to config file on every write |

Admin GUI embedded at build time. Modify `admin/static/index.html` and rebuild.

## URL routing

- `POST /v1/chat/completions` → OpenAI protocol
- `POST /v1/messages` → Anthropic protocol
- `GET /v1/models` → lists enabled incoming models
- `/admin/api/*` → admin CRUD, see README for full table
- `POST /admin/api/config/reload` → re-read config.json from disk
- `POST /admin/api/metrics/reset` → clear metrics

Protocol determined by path: `strings.Contains(path, "/messages")` → Anthropic, else OpenAI.

## Config quirks

- Servers have `openai_url` / `anthropic_url` optional overrides; falls back to base `url`
- Base URL paths **deduplicated**: if server URL ends with `/v1` and request path starts with `/v1`, they merge
- Fallback `target_model` empty → inherits rule's primary `TargetModel`
- Rules matched by first match on `IncomingModels`; disabled rules silently skipped
- Legacy fields auto-migrated on load: `api_type` (string) → `api_types` (array), `FallbackServerIDs` → `Fallbacks`
- `config.json` is `.gitignore`d (contains real API keys) — **do not overwrite**

## Request flow

- Model extracted from `"model"` field in JSON body (not URL/headers) via `router.go:extractModel`
- `metricsWriter` captures TTFB on first `WriteHeader`, buffers last **256KB** of response body for usage parsing
- Usage extracted from final SSE `data:` event (streaming) or top-level `"usage"` object (non-streaming); supports OpenAI + Anthropic + llama-server `"timings"` formats
- Metrics: `PrefillTimeMs = TTFB`, `DecodeTimeMs = Latency - TTFB`; native llama-server timings override wall-clock when present

## Docker

```bash
docker compose up -d           # port 8888 → 8080
docker compose build --no-cache
```

`config.json` must exist as a **file** before volume mount (Docker creates a directory if path doesn't exist). Image: golang:1.26-alpine → alpine:3.21, ~15MB.

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | HTTP listen port |
| `CONFIG_FILE` | `config.json` | Path to config file |
