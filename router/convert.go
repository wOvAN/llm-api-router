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

// droppedKeys reports the top-level fields a converted request lost without a
// per-field drop. The spoke marshals a whitelist, so a key it never decodes — a
// field Anthropic ships next month, a stray client key — disappears with nothing
// to say so, and the spec forbids silent loss. Counting keys in against keys out
// is the visibility the operator needs (a refusal would 400 real traffic on every
// new Anthropic field, leaving `convert: off` as the only escape). A request
// whose keys all map reports nothing.
func droppedKeys(in, out []byte) []string {
	n := keyCount(in) - keyCount(out)
	if n <= 0 {
		return nil
	}
	return []string{fmt.Sprintf("keys:%ddropped", n)}
}

func keyCount(body []byte) int {
	var m map[string]json.RawMessage
	if json.Unmarshal(body, &m) != nil {
		return 0
	}
	return len(m)
}
