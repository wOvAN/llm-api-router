# sdktest

Wire-protocol compliance tests for the router. The **official** OpenAI
([openai-go](https://github.com/openai/openai-go)) and Anthropic
([anthropic-sdk-go](https://github.com/anthropics/anthropic-sdk-go)) Go SDKs
act as real clients of the proxied responses; a fake OpenAI/Anthropic backend
(llama.cpp/vLLM-shaped payloads, including native `timings` and the llama.cpp
Anthropic stream without `message_start`) sits behind the router.

What it validates, through the full request/response pipeline:

- request model rewrite and upstream auth header injection (both
  `Authorization` and `x-api-key`)
- response model rewrite back to the client's model name, in JSON and inside
  stream frames
- OpenAI: JSON + SSE streaming, `stream_options.include_usage` injection and
  its client-side stripping, tool-call streaming, `/v1/models`
- Anthropic: JSON + SSE streaming (canonical and llama.cpp shapes), `ping`
  tolerance, 4xx error passthrough
- proxy-side metrics capture of token usage and llama.cpp native `timings`

## Separate module, on purpose

The router's zero-external-dependency promise covers the main module only.
This directory is its own Go module so the SDK imports never touch the main
`go.mod`/`go.sum`. `go build ./...` / `go test ./...` from the repo root never
enter it, and CI is unaffected.

The SDK sources come from the gitignored `.info/` reference clones (see
`go.mod` replace directives), so **this module builds only where `.info/` is
populated** — it is a local development tool, not part of CI.

## Running

```bash
cd sdktest
go test ./...        # or: go test -v -run TestOpenAI
```

If `go mod tidy` is needed after updating the `.info/` clones, run it inside
this directory.
