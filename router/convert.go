package router

import (
	"llm-api-router/convert"
	"llm-api-router/proxy"
)

// maxConvertNotesBytes bounds RequestMetric.ConvertNotes (the spec's ≤ 512B): the
// list is persisted with every metric and rendered in the GUI, so a chatty
// translator must not be able to grow it without bound. Whole notes are kept,
// the tail is dropped.
const maxConvertNotesBytes = 512

func capNotes(notes []string) []string {
	n := 0
	for i, s := range notes {
		if n+len(s) > maxConvertNotesBytes {
			return notes[:i]
		}
		n += len(s)
	}
	return notes
}

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
