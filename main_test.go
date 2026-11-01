package main

import (
	"strings"
	"testing"
)

// The admin GUI is one embedded file, so a GUI test is a text assertion on that
// source. These three pin the protocol-conversion surface: an operator must be
// able to declare which endpoints a backend natively serves (an empty declaration
// is exactly what keeps a server byte-for-byte passthrough, so the control, its
// reset on add and its restore on edit are the feature), a form save must not
// erase the fields it does not render, and a converted request must show its edge
// and its notes in Recent Requests.

func guiSource(t *testing.T) string {
	t.Helper()
	b, err := staticFS.ReadFile("admin/static/index.html")
	if err != nil {
		t.Fatalf("read embedded GUI: %v", err)
	}
	return string(b)
}

// funcBody returns the text of a top-level function, from its signature to the
// closing brace at column 0 — enough to assert on what a function does without
// pinning its whole formatting.
func funcBody(t *testing.T, src, signature string) string {
	t.Helper()
	i := strings.Index(src, signature)
	if i < 0 {
		t.Fatalf("%s not in the GUI source", signature)
	}
	end := strings.Index(src[i:], "\n}")
	if end < 0 {
		t.Fatalf("%s has no closing brace", signature)
	}
	return src[i : i+end]
}

func TestGUIServerFormEditsProtocols(t *testing.T) {
	src := guiSource(t)

	for _, id := range []string{"server-proto-chat", "server-proto-messages",
		"server-proto-responses", "server-proto-completions"} {
		if !strings.Contains(src, `id="`+id+`"`) {
			t.Errorf("the server form has no %s input — an operator can not declare the endpoints a backend serves", id)
		}
	}
	// Empty means "undeclared = passthrough"; a control whose empty state is a
	// behavior change has to say so, or an operator clears it by accident.
	if i := strings.Index(src, "server-proto-chat"); i < 0 ||
		!strings.Contains(src[max(0, i-400):i], "passthrough") {
		t.Error("the endpoints control does not state that an empty declaration means passthrough")
	}

	save := funcBody(t, src, "function getServerProtocols()")
	for _, want := range []string{"/v1/chat/completions", "/v1/messages", "/v1/responses", "/v1/completions"} {
		if !strings.Contains(save, want) {
			t.Errorf("getServerProtocols does not offer %s", want)
		}
	}
	// `state` has no control (nothing reads it at runtime yet), so a save must
	// carry the config-declared flag over per path — a rebuild from the checkboxes
	// alone would silently erase it.
	if !strings.Contains(save, "state:") || !strings.Contains(save, "editingServer") {
		t.Error("getServerProtocols rebuilds protocols from the checkboxes and drops the `state` flag")
	}

	if !strings.Contains(funcBody(t, src, "async function saveServer()"), "protocols: getServerProtocols()") {
		t.Error("saveServer never sends protocols, so a declaration can not be created from the GUI")
	}
	for _, fn := range []string{"function openServerModal(", "function editServer("} {
		body := funcBody(t, src, fn)
		if !strings.Contains(body, "setServerProtocols(") {
			t.Errorf("%s does not set the endpoint checkboxes (add keeps the previous server's marks, edit shows none)", fn)
		}
	}
}

func TestGUIServerFormPreservesUnmanagedFields(t *testing.T) {
	body := funcBody(t, guiSource(t), "async function saveServer()")
	if !strings.Contains(body, "Object.assign({}, editingServer,") {
		t.Error("saveServer rebuilds the server from its inputs, erasing every field the form does not render")
	}
}

func TestGUIShowsConversion(t *testing.T) {
	src := guiSource(t)
	i := strings.Index(src, "r.convert_notes")
	if i < 0 {
		t.Fatal("a converted request reports nothing in Recent Requests")
	}
	window := src[max(0, i-600) : i+300]
	for _, want := range []string{"r.convert", "convert_notes"} {
		if !strings.Contains(window, want) {
			t.Errorf("the conversion pill shows neither %q", want)
		}
	}
	if !strings.Contains(window, "escAttr(") {
		t.Error("the conversion notes go into a title attribute unescaped")
	}
}
