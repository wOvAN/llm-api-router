// Package convert translates requests, responses and SSE streams between the API
// dialects the router proxies (OpenAI Chat Completions, OpenAI Responses,
// Anthropic Messages). The hub is chat: every other dialect is defined as a spoke
// around it, so N dialects cost 2N transforms, not N².
package convert

import "strings"

// Dialect names an API dialect by the endpoint that carries it.
type Dialect string

const (
	Chat      Dialect = "chat"
	Messages  Dialect = "messages"
	Responses Dialect = "responses"
	Legacy    Dialect = "completions"
)

// DialectForPath names the dialect a request path belongs to. The empty string
// means the path carries no dialect this package can translate: audio uploads,
// embeddings, llama.cpp's /v1/systemone, and /v1/messages/count_tokens (a token
// counter, not a messages call — it has no Chat Completions counterpart, so it
// must never be planned onto one). Those are never conversion subjects.
//
// Order matters: /v1/chat/completions contains /completions.
func DialectForPath(path string) Dialect {
	switch {
	case strings.Contains(path, "/count_tokens"):
		return ""
	case strings.Contains(path, "/messages"):
		return Messages
	case strings.Contains(path, "/chat/completions"):
		return Chat
	case strings.Contains(path, "/responses"):
		return Responses
	case strings.Contains(path, "/completions"):
		return Legacy
	}
	return ""
}

// PathFor returns the canonical endpoint path carrying a dialect ("" for the
// dialectless and legacy families, which are never converted).
func PathFor(d Dialect) string {
	switch d {
	case Chat:
		return "/v1/chat/completions"
	case Messages:
		return "/v1/messages"
	case Responses:
		return "/v1/responses"
	}
	return ""
}
