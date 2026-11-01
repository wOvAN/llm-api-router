package router

import (
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
