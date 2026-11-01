package convert

import (
	"errors"
	"strings"
	"testing"

	"llm-api-router/domain"
)

func protocols(spec ...string) []domain.ServerProtocol {
	out := make([]domain.ServerProtocol, 0, len(spec))
	for _, s := range spec {
		p, state, _ := strings.Cut(s, "=")
		out = append(out, domain.ServerProtocol{Path: p, State: state == "state"})
	}
	return out
}

func TestDecidePassthrough(t *testing.T) {
	// A server declaring nothing: today's behavior, no conversion.
	p, err := Decide(Messages, "/v1/messages", nil, "auto")
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if p.Dialect != Messages || p.Path != "/v1/messages" {
		t.Fatalf("got %+v, want passthrough to /v1/messages", p)
	}
	// A server declaring the client's own endpoint: passthrough.
	p, err = Decide(Messages, "/v1/messages", protocols("/v1/messages", "/v1/chat/completions"), "auto")
	if err != nil || p.Dialect != Messages {
		t.Fatalf("got %+v, %v, want passthrough", p, err)
	}
	// Legacy on an undeclared backend is passthrough too: the refusal is for a
	// backend that declares what it serves, not for a config that says nothing.
	p, err = Decide(Legacy, "/v1/completions", nil, "auto")
	if err != nil || p.Dialect != Legacy || p.Path != "/v1/completions" {
		t.Fatalf("got %+v, %v, want passthrough", p, err)
	}
}

func TestDecideConvertsToDeclaredDialect(t *testing.T) {
	// A backend that only speaks chat: an Anthropic client is planned onto chat.
	p, err := Decide(Messages, "/v1/messages", protocols("/v1/chat/completions"), "auto")
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if p.Dialect != Chat || p.Path != "/v1/chat/completions" || p.From != Messages {
		t.Fatalf("got %+v, want messages→chat at /v1/chat/completions", p)
	}
}

func TestDecideResponsesClient(t *testing.T) {
	// A backend that only speaks chat: a Responses client is planned onto chat.
	p, err := Decide(Responses, "/v1/responses", protocols("/v1/chat/completions"), "auto")
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if p.Dialect != Chat || p.From != Responses || p.Edge() != "responses>chat" {
		t.Fatalf("got %+v, want responses→chat", p)
	}
	if p.Path != "/v1/chat/completions" {
		t.Fatalf("the plan must carry the declared path, got %q", p.Path)
	}
	// A backend that declares the Responses API keeps the client's own endpoint.
	p, err = Decide(Responses, "/v1/responses", protocols("/v1/responses"), "auto")
	if err != nil || p.Dialect != Responses || p.Path != "/v1/responses" {
		t.Fatalf("got %+v, %v, want passthrough", p, err)
	}
	// Spokes never touch each other: a Responses client is not planned onto a
	// Messages backend (that edge costs two transforms and buys nothing).
	if _, err := Decide(Responses, "/v1/responses", protocols("/v1/messages"), "auto"); err == nil {
		t.Fatal("messages↔responses must never be planned directly")
	}
	// A declared `state` backend is not a license to continue a response: the
	// router stores nothing, and the request-level refusal is the spoke's job.
	if p, err := Decide(Responses, "/v1/responses", protocols("/v1/responses=state"), "auto"); err != nil || p.Dialect != Responses {
		t.Fatalf("got %+v, %v, want a declared stateful responses backend to stay passthrough", p, err)
	}
}

func TestDecideRefusesWhenNoEdgeExists(t *testing.T) {
	// Anthropic on a responses-only backend: the hub is chat, so nothing connects
	// them — refused, never guessed.
	_, err := Decide(Messages, "/v1/messages", protocols("/v1/responses"), "auto")
	var ce *Error
	if !errors.As(err, &ce) || ce.Code != CodeNoDialect {
		t.Fatalf("got %v, want *convert.Error{%s}", err, CodeNoDialect)
	}
}

func TestDecideNeverConvertsLegacy(t *testing.T) {
	if _, err := Decide(Legacy, "/v1/completions", protocols("/v1/chat/completions"), "auto"); err == nil {
		t.Fatal("legacy completions must never be converted (the chat template lives in the backend)")
	}
}

func TestDecideForcedEdgeAndOff(t *testing.T) {
	p, err := Decide(Messages, "/v1/messages", nil, "messages>chat")
	if err != nil || p.Dialect != Chat || !p.Forced {
		t.Fatalf("got %+v, %v, want forced messages→chat", p, err)
	}
	p, err = Decide(Messages, "/v1/messages", protocols("/v1/chat/completions"), "off")
	if err != nil || p.Dialect != Messages {
		t.Fatalf("got %+v, %v, want passthrough with convert off", p, err)
	}
	p, err = Decide(Responses, "/v1/responses", nil, "responses>chat")
	if err != nil || p.Dialect != Chat || !p.Forced {
		t.Fatalf("got %+v, %v, want forced responses→chat", p, err)
	}
	if _, err := Decide(Messages, "/v1/messages", nil, "chat>responses"); err == nil {
		t.Fatal("an unknown forced edge must be an error")
	}
}

func TestDecideDialectlessPathsArePassthrough(t *testing.T) {
	// Audio, embeddings, systemone: no dialect, so never a conversion subject.
	p, err := Decide("", "/v1/audio/speech", protocols("/v1/chat/completions"), "auto")
	if err != nil || p.Dialect != "" || p.Path != "/v1/audio/speech" {
		t.Fatalf("got %+v, %v, want untouched passthrough", p, err)
	}
	// count_tokens is a token counter, not a messages call: it has no chat
	// counterpart, so a chat-only backend must get the client's own path (without
	// it, Claude Code's token count is answered with a completion).
	p, err = Decide(Messages, "/v1/messages/count_tokens", protocols("/v1/chat/completions"), "auto")
	if err != nil || p.Dialect != Messages || p.Path != "/v1/messages/count_tokens" {
		t.Fatalf("got %+v, %v, want passthrough of the count_tokens path", p, err)
	}
}

func TestDecideIdentityComparesDialects(t *testing.T) {
	// The root-served llama.cpp form (/messages) is the same endpoint as
	// /v1/messages: declaring it is passthrough, not a refusal naming two paths
	// that mean the same thing.
	p, err := Decide(Messages, "/v1/messages", protocols("/messages"), "auto")
	if err != nil || p.Dialect != Messages || p.Path != "/v1/messages" {
		t.Fatalf("got %+v, %v, want identity with the client's path", p, err)
	}
}
