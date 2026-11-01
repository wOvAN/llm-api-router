package convert

// Error is a conversion refusal: the request must not reach the backend, because
// speaking the backend's dialect would silently answer a different question than
// the one the client asked.
type Error struct {
	Code  string // CodeNoDialect, CodeField, CodeState
	Field string // the request field at issue ("" when the whole dialect is the issue)
	Msg   string // the client-facing sentence
}

const (
	CodeNoDialect = "no_dialect"
	CodeField     = "unsupported_field"
	CodeState     = "state"
)

func (e *Error) Error() string { return e.Msg }
