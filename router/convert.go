package router

import (
	"encoding/json"
	"fmt"
	"llm-api-router/convert"
	"llm-api-router/proxy"
)

// responseTranslator adapts convert's response direction to the shape
// proxy.StreamTranslator expects. Here it translates non-streamed JSON bodies;
// Task 8 gives it the stream emitter.
type responseTranslator struct {
	plan  convert.Plan
	notes []string
}

var _ proxy.StreamTranslator = (*responseTranslator)(nil)

func newResponseTranslator(p convert.Plan) *responseTranslator {
	return &responseTranslator{plan: p}
}

func (t *responseTranslator) TranslateJSON(body []byte) ([]byte, error) {
	b, notes, err := convert.Response(t.plan, body)
	if err != nil {
		return nil, err
	}
	t.notes = append(t.notes, notes...)
	return b, nil
}

// The SSE path is Task 8's: a frame is forwarded untouched, which is also the
// right answer for a pair with no stream emitter.
func (t *responseTranslator) WriteFrame(frame []byte) [][]byte { return [][]byte{frame} }

func (t *responseTranslator) FinishFrames() [][]byte { return nil }

func (t *responseTranslator) Notes() []string { return t.notes }

// convertedRequestKeys are the top-level fields the request spoke decodes — the
// json tags of its own request struct. A spoke marshals a whitelist, so a key it
// never decodes vanishes with nothing to say so: no note, no refusal. The spec
// forbids silent loss, and a refusal would 400 real traffic on every field the
// providers ship next month (leaving `convert: off` as the only escape), so the
// routed path counts them: `keys:<n>dropped`.
//
// The list is the spoke's decoder set, not a guess at what clients send: a key on
// it is a key the spoke handled — mapped straight through (present on the way out,
// so never counted), folded into another key (`system` into `messages`), renamed
// (`stop_sequences` into `stop`), or dropped under a `field=dropped` note. Counting
// it would report the plain system prompt of every Anthropic client as a loss. Keep
// it in sync with the spoke's request struct: a field missing here reads as a loss,
// a field listed here that the spoke stops decoding goes silent again.
var convertedRequestKeys = map[string]bool{
	"model": true, "system": true, "messages": true, "max_tokens": true,
	"stop_sequences": true, "temperature": true, "top_p": true, "top_k": true,
	"stream": true, "tools": true, "tool_choice": true, "output_format": true,
	"thinking": true, "metadata": true,
}

// droppedKeys compares what the client sent against what the backend will see and
// reports the keys that left with no per-field note to explain them. The preview
// path (admin, which calls convert.Request directly) reports no such note: it shows
// the client its own fields, one by one, and has no routed attempt to account for.
func droppedKeys(in, out []byte) []string {
	inKeys, outKeys := keySet(in), keySet(out)
	if inKeys == nil {
		return nil
	}
	n := 0
	for k := range inKeys {
		if _, kept := outKeys[k]; kept || convertedRequestKeys[k] {
			continue
		}
		n++
	}
	if n == 0 {
		return nil
	}
	return []string{fmt.Sprintf("keys:%ddropped", n)}
}

func keySet(body []byte) map[string]json.RawMessage {
	var m map[string]json.RawMessage
	if json.Unmarshal(body, &m) != nil {
		return nil
	}
	return m
}
