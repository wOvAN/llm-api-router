package config

import (
	"bytes"
	"encoding/json"
	"strings"

	"llm-api-router/domain"
)

// unknownFieldScanCap bounds the report loop: each pass strips one key, so a
// config made of nothing but typos still terminates.
const unknownFieldScanCap = 50

// UnknownFields reports the names of the keys in a config document that no
// field of domain.Config reads, at any depth.
//
// A plain json.Unmarshal drops an unread key silently, and a typo in a key is
// invisible in every other way: `colddown_time` parses, lands on the floor, and
// the operator spends the afternoon explaining why cooldowns look ignored. The
// report is a warning, never a refusal — a config written for a newer router
// must still boot it.
func UnknownFields(data []byte) []string {
	var out []string
	for len(out) < unknownFieldScanCap {
		dec := json.NewDecoder(bytes.NewReader(data))
		dec.DisallowUnknownFields()
		var cfg domain.Config
		if err := dec.Decode(&cfg); err == nil {
			return out
		} else if path, ok := unknownFieldPath(err); ok {
			out = append(out, path[len(path)-1])
			if stripped := stripAt(data, path); stripped != nil {
				data = stripped
				continue
			}
		}
		return out // a type mismatch or a truncated file: not a key report
	}
	return out
}

// stripAt decodes the document, deletes the key the path names, and encodes it
// again, so the next strict pass reports the next typo.
func stripAt(data []byte, path []string) []byte {
	var tree any
	if json.Unmarshal(data, &tree) != nil {
		return nil
	}
	deletePath(tree, path)
	out, err := json.Marshal(tree)
	if err != nil {
		return nil
	}
	return out
}

// deletePath removes the leaf key the path names. encoding/json reports the
// chain from the wrapper it decoded through, which starts BELOW the document
// root (the root's own keys are named struct fields, not wrappers), so the path
// is matched wherever it starts and then followed. A map key is a NAME (a
// server id), never a config key, which is why the chain is followed key by key
// instead of deleting every key with the reported name.
func deletePath(node any, path []string) bool {
	if m, ok := node.(map[string]any); ok {
		if child, ok := m[path[0]]; ok {
			if len(path) == 1 {
				delete(m, path[0])
				return true
			}
			if deletePath(child, path[1:]) {
				return true
			}
		}
	}
	switch v := node.(type) {
	case map[string]any:
		for _, child := range v {
			if deletePath(child, path) {
				return true
			}
		}
	case []any:
		for _, child := range v {
			if deletePath(child, path) {
				return true
			}
		}
	}
	return false
}

// unknownFieldPath reads the key path out of the messages encoding/json reports
// through an anonymous-struct wrapper:
//
//	json: unknown field "x"
//	json: unknown field "y" in anonymous struct field "Server" reading from "servers"
//	json: unknown field "patth" in anonymous struct field "Protocols" reading from "protocols" in anonymous struct field "Server" reading from "servers"
//
// The outermost wrapper segment names the wire key it was reached through, so
// unwinding outermost-first yields the document's path, root-first.
func unknownFieldPath(err error) ([]string, bool) {
	const wrapper = " in anonymous struct field "
	const wire = `" reading from "`
	msg := err.Error()
	var path []string
	for {
		i := strings.LastIndex(msg, wrapper)
		if i < 0 {
			break
		}
		tail := msg[i+len(wrapper):]
		j := strings.Index(tail, wire)
		if j < 0 {
			return nil, false
		}
		path = append(path, strings.Trim(tail[j+len(wire):], `"`))
		msg = msg[:i]
	}
	name := strings.Trim(strings.TrimPrefix(msg, "json: unknown field "), `"`)
	if name == "" {
		return nil, false
	}
	return append(path, name), true
}
