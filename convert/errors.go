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
	// CodeState names a request that depends on backend-side conversation state
	// (`previous_response_id`). Declared scaffolding for the capability-probe
	// phase (`settings.convert.probe`, `domain.ServerProtocol.State`): the
	// `responses` spokes that need it are a later phase, so nothing raises it yet.
	CodeState = "state"
)

func (e *Error) Error() string { return e.Msg }
