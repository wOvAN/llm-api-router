package convert

import (
	"fmt"
	"strings"

	"llm-api-router/domain"
)

// Plan is the per-attempt decision: which dialect and path this attempt speaks,
// and which dialect it is translating from.
type Plan struct {
	From    Dialect
	Dialect Dialect
	Path    string
	Forced  bool // `convert` named the edge explicitly
}

// Options are the conversion knobs (see domain.ConvertSettings). The router builds
// them from settings once per request.
type Options struct {
	MaxTokens  int
	DropParams bool
}

// Edge names the conversion this plan performs, e.g. "messages>chat"; "" for an
// identity (or dialectless) plan, which is what the report sites key on.
func (p Plan) Edge() string {
	if p.Dialect == p.From || p.From == "" {
		return ""
	}
	return string(p.From) + ">" + string(p.Dialect)
}

// OptionsFrom builds the conversion knobs from the persisted settings. One
// construction site so the router and the admin preview can never disagree
// about the defaults.
func OptionsFrom(s domain.ConvertSettings) Options {
	return Options{MaxTokens: s.MaxTokens(), DropParams: s.DropParams}
}

// Decide picks the upstream dialect for one attempt.
//
// A backend that declares no protocols keeps today's behavior exactly: the
// client's own path, no translation. Otherwise the client's endpoint is looked up
// in the declaration; a backend that does not serve it is planned onto the one
// dialect the backend does serve, for the edges this build implements
// (messages→chat, responses→chat). Anything else is a refusal, never a guess.
//
// forced is rule/fallback/candidate `convert`: "auto" (or "") decides by the
// declaration, "off" forces passthrough, "messages>chat"/"responses>chat" name the
// edge.
func Decide(client Dialect, path string, protocols []domain.ServerProtocol, forced string) (Plan, error) {
	ident := Plan{From: client, Dialect: client, Path: path}
	if forced == "off" || DialectForPath(path) == "" {
		return ident, nil
	}
	if forced != "" && forced != "auto" {
		from, to, ok := strings.Cut(forced, ">")
		if !ok || Dialect(from) != client {
			return Plan{}, &Error{Code: CodeNoDialect, Msg: fmt.Sprintf(
				"convert %q does not apply to a %s request", forced, client)}
		}
		if (from == string(Messages) || from == string(Responses)) && to == string(Chat) {
			return Plan{From: client, Dialect: Chat, Path: PathFor(Chat), Forced: true}, nil
		}
		return Plan{}, &Error{Code: CodeNoDialect, Msg: fmt.Sprintf(
			"conversion %s is not supported", forced)}
	}
	if len(protocols) == 0 {
		// Nothing declared: today's behavior, the client's own endpoint — checked
		// before the legacy refusal, so an undeclared backend keeps /v1/completions.
		return ident, nil
	}
	if client == Legacy {
		// /v1/completions wants a rendered prompt: the chat template lives in the
		// backend's tokenizer, which the proxy does not have.
		return Plan{}, &Error{Code: CodeNoDialect, Field: "path", Msg: "converting /v1/completions is not supported: the chat template is applied by the backend"}
	}
	// Identity compares dialects, not path strings: an operator who declares the
	// root-served llama.cpp form (/messages, /chat/completions — a form this
	// router's URL join deliberately supports) means the same endpoint.
	for _, p := range protocols {
		if DialectForPath(p.Path) == client {
			return ident, nil
		}
	}
	if client == Messages || client == Responses {
		for _, p := range protocols {
			if DialectForPath(p.Path) == Chat {
				return Plan{From: client, Dialect: Chat, Path: p.Path}, nil
			}
		}
	}
	return Plan{}, &Error{Code: CodeNoDialect, Msg: fmt.Sprintf(
		"backend serves %s, not %s and no conversion exists for it",
		strings.Join(declaredPaths(protocols), ", "), path)}
}

// decodedKeys is the plan's request decoder set — the keys the spoke reads, which
// is the baseline DroppedKeys counts against. An identity plan has no spoke.
func (p Plan) decodedKeys() map[string]bool {
	switch p.From {
	case Messages:
		return DecodedRequestKeys
	case Responses:
		return DecodedResponsesKeys
	}
	return nil
}

func declaredPaths(protocols []domain.ServerProtocol) []string {
	out := make([]string, 0, len(protocols))
	for _, p := range protocols {
		out = append(out, p.Path)
	}
	return out
}

// Request translates a request body for a plan. Identity plans return the body
// untouched. The notes list what a deliberate drop cost.
func Request(p Plan, body []byte, o Options) ([]byte, []string, error) {
	if p.Dialect == p.From || p.From == "" {
		return body, nil, nil
	}
	if p.From == Messages && p.Dialect == Chat {
		return MessagesToChat(body, o)
	}
	if p.From == Responses && p.Dialect == Chat {
		return ResponsesToChat(body, o)
	}
	return nil, nil, &Error{Code: CodeNoDialect, Msg: "conversion " + p.Edge() + " is not supported"}
}
