package router

import (
	"encoding/json"
	"fmt"
	"llm-api-router/convert"
	"llm-api-router/proxy"
)

// responseTranslator adapts convert's response direction to the shape
// proxy.StreamTranslator expects: the stream emitter for an SSE answer, the JSON
// translator for a buffered one. A pair with no emitter forwards frames untouched.
type responseTranslator struct {
	plan   convert.Plan
	stream convert.Stream
	notes  []string
}

var _ proxy.StreamTranslator = (*responseTranslator)(nil)

func newResponseTranslator(p convert.Plan) *responseTranslator {
	t := &responseTranslator{plan: p}
	if s, ok := convert.NewStream(p); ok {
		t.stream = s
	}
	return t
}

func (t *responseTranslator) TranslateJSON(body []byte) ([]byte, error) {
	b, notes, err := convert.Response(t.plan, body)
	if err != nil {
		return nil, err
	}
	t.notes = append(t.notes, notes...)
	return b, nil
}

func (t *responseTranslator) WriteFrame(frame []byte) [][]byte {
	if t.stream == nil {
		return [][]byte{frame}
	}
	return t.stream.WriteFrame(frame)
}

func (t *responseTranslator) FinishFrames() [][]byte {
	if t.stream == nil {
		return nil
	}
	return t.stream.FinishFrames()
}

func (t *responseTranslator) Notes() []string {
	if t.stream == nil {
		return t.notes
	}
	return append(t.notes, t.stream.Notes()...)
}

// droppedKeys compares what the client sent against what the backend will see and
// reports the keys that left with no per-field note to explain them. A spoke
// marshals a whitelist, so a key it never decodes vanishes with nothing to say so:
// no note, no refusal. The spec forbids silent loss, and a refusal would 400 real
// traffic on every field the providers ship next month (leaving `convert: off` as
// the only escape), so the routed path counts them: `keys:<n>dropped`.
//
// The baseline is the spoke's own decoder set (`convert.DecodedRequestKeys`,
// derived from its request struct, so the two cannot drift): a key on it is a key
// the spoke handled — mapped straight through (present on the way out, so never
// counted), folded into another key (`system` into `messages`), renamed
// (`stop_sequences` into `stop`), or dropped under a `field=dropped` note. Counting
// it would report the plain system prompt of every Anthropic client as a loss, and
// make the operator count a field twice.
//
// The preview path (admin, which calls convert.Request directly) reports no such
// note: it shows the client its own fields, one by one, and has no routed attempt
// to account for.
func droppedKeys(in, out []byte) []string {
	inKeys, outKeys := keySet(in), keySet(out)
	if inKeys == nil {
		return nil
	}
	n := 0
	for k := range inKeys {
		if _, kept := outKeys[k]; kept || convert.DecodedRequestKeys[k] {
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
