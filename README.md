# LLM API Router

A Go-based reverse proxy that routes LLM inference API requests between multiple backend servers, supporting both OpenAI and Anthropic API protocols.

## Features

- **Multi-backend routing** — configure multiple inference backends with different API protocols
- **Model name remapping** — map incoming model names to different backend model names (e.g., `opus` → `haiku`)
- **Fallback servers** — automatic failover to backup servers when the primary is unavailable
- **Web GUI** — configure servers and routing rules at runtime via a browser
- **Streaming support** — SSE streaming responses are relayed without buffering

## Quick Start

```bash
# Build and run
go build -o llm-api-router .
./llm-api-router

# Or with Docker
docker build -t llm-api-router .
docker run -p 8080:8080 llm-api-router
```

The server starts on port 8080:
- **API routes**: `http://localhost:8080/v1/*`
- **Admin GUI**: `http://localhost:8080/admin`

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
      "api_type": "openai"
    },
    "fallback": {
      "id": "fallback",
      "name": "Fallback Anthropic",
      "url": "api.anthropic.com/v1",
      "api_key": "sk-ant-...",
      "api_type": "anthropic"
    }
  },
  "rules": [
    {
      "incoming_model": "opus",
      "target_model": "haiku",
      "server_id": "primary",
      "fallback_server_ids": ["fallback"]
    },
    {
      "incoming_model": "sonnet",
      "target_model": "haiku",
      "server_id": "primary",
      "fallback_server_ids": ["fallback"]
    }
  ]
}
```

## API Routes

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
    "model": "opus",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## Admin API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/admin/api/servers` | List servers |
| POST | `/admin/api/servers` | Add server |
| PUT | `/admin/api/servers/:id` | Update server |
| DELETE | `/admin/api/servers/:id` | Delete server |
| GET | `/admin/api/rules` | List routing rules |
| POST | `/admin/api/rules` | Add routing rule |
| PUT | `/admin/api/rules/:idx` | Update routing rule |
| DELETE | `/admin/api/rules/:idx` | Delete routing rule |
| GET | `/admin/api/config` | Get full config |
