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

func TestDecideRefusesWhenNoEdgeExists(t *testing.T) {
	// Responses on a chat-only backend is a phase-4 edge: refused, not guessed.
	_, err := Decide(Responses, "/v1/responses", protocols("/v1/chat/completions"), "auto")
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
}
