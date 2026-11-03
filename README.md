# LLM API Router

[![CI](https://github.com/wOvAN/llm-api-router/actions/workflows/ci.yml/badge.svg)](https://github.com/wOvAN/llm-api-router/actions/workflows/ci.yml)

A Go-based reverse proxy that routes LLM inference API requests between multiple backend servers, supporting both OpenAI and Anthropic API protocols. **One external dependency** (`golang.org/x/net`, for SOCKS proxy support); everything else is stdlib.

## Features

- **Multi-backend routing** — configure multiple inference backends with different API protocols
- **Model name remapping** — map incoming model names to different backend model names (e.g., `opus` → `haiku`)
- **Fallback servers** — automatic failover when the primary is unavailable **or answers 429/5xx** (a backend that is throttled or dying serves nothing, so the next one is tried), with per-fallback model overrides
- **Wildcard rules** — one rule can own a model family (`qwen3-*`); the client's own model name then rides to the backend untouched
- **Pool balancing** — `balance: "round-robin"` shares traffic across a pool of interchangeable backends, weighted per member
- **Web GUI** — configure servers and routing rules at runtime via a browser (embedded, rebuilt with `go build`)
- **Streaming support** — SSE streaming responses are relayed without buffering
- **Metrics & analytics** — ring buffer of last 100 requests with aggregated summaries by model and server (TTFB, decode time, token throughput)
- **Usage extraction** — parses token usage from OpenAI, Anthropic, and llama-server response formats (streaming and non-streaming)

## Quick Start

```bash
# Build and run
go build -o llm-api-router .
./llm-api-router

# Or with Docker Compose
docker compose up -d   # port 8888 → 8080
```

The server starts on port 8080 (configurable via `PORT`):
- **API routes**: `http://localhost:8080/v1/*`
- **Admin GUI**: `http://localhost:8080/admin`

## Architecture

| Layer | Role |
|-------|------|
| `main.go` | Entrypoint, wiring, HTTP server |
| `domain/` | Models (Config, Server, RoutingRule, Metrics) |
| `config/` | Thread-safe config store with JSON persistence |
| `router/` | Model extraction → rule matching → body rewrite → proxy chain |
| `proxy/` | Reverse proxy (SSE streaming) + token usage extraction |
| `metrics/` | Ring buffer (last 100), aggregated summaries |
| `admin/` | REST CRUD API + embedded SPA GUI |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | HTTP listen port |
| `CONFIG_FILE` | `config.json` | Path to config file |
| `ADMIN_TOKEN` | *(unset)* | Bearer token required by `/admin/api/*` and `/metrics` (unset = both open; the GUI asks on its first 401) |

## Config File

The config is stored in `config.json` (path configurable via `CONFIG_FILE`). You can also manage it through the web GUI at `/admin`.

```json
{
  "servers": {
    "primary": {
      "id": "primary",
      "name": "Primary OpenAI",
      "url": "api.openai.com/v1",
      "api_key": "sk-...",
      "api_types": ["openai"]
    },
    "fallback": {
      "id": "fallback",
      "name": "Fallback Anthropic",
      "url": "api.anthropic.com/v1",
      "api_key": "sk-ant-...",
      "api_types": ["anthropic"]
    }
  },
  "profiles": [
    {
      "id": "default",
      "name": "default",
      "rules": [
        {
          "incoming_models": ["opus", "claude-3-opus"],
          "target_model": "claude-3-5-haiku-latest",
          "server_id": "primary",
          "enabled": true
        },
        {
          "incoming_models": ["sonnet"],
          "target_model": "claude-3-5-haiku-latest",
          "server_id": "primary",
          "fallbacks": [
            {"server_id": "fallback", "target_model": "claude-3-haiku"}
          ],
          "enabled": true
        }
      ]
    }
  ],
  "active_profile_id": "default"
}
```

### Key config quirks

- Rules live in named **profiles**; `active_profile_id` selects which profile routes traffic. Only the active profile's rules are used (and listed by `/v1/models`). Any profile's rules can be edited independently of the active one — GUI profile selector, or `/admin/api/rules?profile_id=`
- Legacy top-level `rules` arrays auto-migrate to a single `default` profile on load (the file is rewritten in the new shape on the next save)
- `api_types` accepts `"openai"` and/or `"anthropic"` per server; legacy `api_type` (string) is auto-migrated to array on load
- Each server can specify `openai_url` / `anthropic_url` overrides; falls back to base `url`
- Each server can route through a proxy: `proxy` (empty = direct connection) with a `proxy_enabled` toggle (default off; `true` enables the proxy, otherwise the proxy URL is ignored). Supported schemes: `http`, `https`, `socks5`, `socks5h`, `socks4`, `socks4a`
- Base URL paths are deduplicated (e.g. `api.openai.com/v1` + `/v1/chat/completions` → `api.openai.com/v1/chat/completions`)
- Rules are matched by an **exact** name first, then by the **most specific wildcard** (`path.Match` globs: `qwen3-*`, `claude-?`); disabled rules (`"enabled": false`) are silently skipped
- A rule matched by a pattern has no single target: an **empty `target_model` is pass-through** — the client's model name is sent to the backend as-is (and `target_model` empty on an exact rule does the same)
- `balance: "round-robin"` turns the rule's `server_id` + enabled `fallbacks` into one pool sharing traffic; `fallbacks[].weight` (default 1) is that member's share, ignored by a failover-only rule
- Fallback `target_model` empty → inherits the rule's primary `TargetModel`
- `context_window` (on a rule, a fallback, or a jev candidate) declares the context size that member can fit; a request whose estimated prompt (`bytes/4`) exceeds it is routed past that member to one that declares room (`0`/absent = unguarded)

## API Usage

### OpenAI Protocol

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "opus",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Anthropic Protocol

```bash
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sonnet",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### List available models

```bash
curl http://localhost:8080/v1/models
```

## Admin API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/admin/api/servers` | List servers |
| POST | `/admin/api/servers` | Add server |
| PUT | `/admin/api/servers/:id` | Update server |
| DELETE | `/admin/api/servers/:id` | Delete server |
| GET | `/admin/api/servers/:id/models` | Fetch model list from upstream |
| POST | `/admin/api/servers/test` | Test connectivity to a server |
| GET | `/admin/api/rules` | List routing rules (`?profile_id=` to target a specific profile; default = active) |
| POST | `/admin/api/rules` | Add routing rule (`?profile_id=` to target a specific profile; default = active) |
| PUT | `/admin/api/rules/:idx` | Update routing rule by index (`?profile_id=` to target a specific profile; default = active) |
| DELETE | `/admin/api/rules/:idx` | Delete routing rule by index (`?profile_id=` to target a specific profile; default = active) |
| GET | `/admin/api/profiles` | List profiles + active profile id |
| POST | `/admin/api/profiles` | Create profile (optionally copy the active profile's rules) |
| PUT | `/admin/api/profiles/:id` | Rename profile |
| DELETE | `/admin/api/profiles/:id` | Delete profile (the last one is protected) |
| POST | `/admin/api/profiles/:id/activate` | Set the active profile |
| GET | `/admin/api/config` | Get full config |
| POST | `/admin/api/config/reload` | Re-read config from disk |
| GET | `/admin/api/metrics` | Get aggregated summaries |
| GET | `/admin/api/metrics/recent` | Get recent request log |
| POST | `/admin/api/metrics/reset` | Clear all metrics |

## Metrics

Metrics are kept in a ring buffer (last 100 requests) and aggregated by model and server:
- Request count, success/error/fallback counts
- Latency (avg, min, max), TTFB
- Token counts (prompt, completion, cached)
- Token throughput (prefill and decode tokens/sec)
- Native llama-server timings override wall-clock when present
- `GET /metrics` (Prometheus text format): the same figures per model / server / target model / request class, plus status-class counters, latency and TTFB **histograms** (so p95 is computable), and `llm_router_server_healthy` / `llm_router_server_cooldown_seconds` gauges
- `GET /readyz` answers `ready`, or `503 draining` once shutdown started — a load balancer pulls a replica out of the pool before its drain cuts a generation short

## Docker

```bash
# Build and run with compose (port 8888 → 8080)
docker compose up -d

# Rebuild without cache
docker compose build --no-cache
```

`config.json` must exist as a **file** before the volume mount (Docker creates a directory if the path doesn't exist). Image: golang:1.27-alpine → alpine:3.21, ~15MB final.
