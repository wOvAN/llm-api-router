# LLM API Router

[![CI](https://github.com/wOvAN/llm-api-router/actions/workflows/ci.yml/badge.svg)](https://github.com/wOvAN/llm-api-router/actions/workflows/ci.yml)

A Go-based reverse proxy that routes LLM inference API requests between multiple backend servers, supporting both OpenAI and Anthropic API protocols. **Zero external dependencies** (stdlib only).

## Features

- **Multi-backend routing** — configure multiple inference backends with different API protocols
- **Model name remapping** — map incoming model names to different backend model names (e.g., `opus` → `haiku`)
- **Fallback servers** — automatic failover to backup servers when the primary is unavailable, with per-fallback model overrides
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
```

### Key config quirks

- `api_types` accepts `"openai"` and/or `"anthropic"` per server; legacy `api_type` (string) is auto-migrated to array on load
- Each server can specify `openai_url` / `anthropic_url` overrides; falls back to base `url`
- Base URL paths are deduplicated (e.g. `api.openai.com/v1` + `/v1/chat/completions` → `api.openai.com/v1/chat/completions`)
- Rules are matched by first match on `incoming_models`; disabled rules (`"enabled": false`) are silently skipped
- Fallback `target_model` empty → inherits the rule's primary `TargetModel`

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
| GET | `/admin/api/rules` | List routing rules |
| POST | `/admin/api/rules` | Add routing rule |
| PUT | `/admin/api/rules/:idx` | Update routing rule (by index) |
| DELETE | `/admin/api/rules/:idx` | Delete routing rule (by index) |
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

## Docker

```bash
# Build and run with compose (port 8888 → 8080)
docker compose up -d

# Rebuild without cache
docker compose build --no-cache
```

`config.json` must exist as a **file** before the volume mount (Docker creates a directory if the path doesn't exist). Image: golang:1.26-alpine → alpine:3.21, ~15MB final.
