package proxy

import (
	"strings"
	"testing"
	"time"
)

func TestWaitSignalWriterSendsSignalAfterIdle(t *testing.T) {
	w := &syncWriter{}
	ws := newWaitSignalWriter(w, 50*time.Millisecond)
	ws.Start()
	defer ws.Stop()

	// Wait longer than idle — signal should fire.
	time.Sleep(100 * time.Millisecond)
	if !strings.Contains(w.content(), "event: router_wait") {
		t.Errorf("expected router_wait signal after idle, got %q", w.content())
	}
}

func TestWaitSignalWriterNoSignalBeforeStart(t *testing.T) {
	w := &syncWriter{}
	ws := newWaitSignalWriter(w, 30*time.Millisecond)
	defer ws.Stop()

	time.Sleep(80 * time.Millisecond)
	if strings.Contains(w.content(), "router_wait") {
		t.Error("router_wait must not fire before Start")
	}
}

func TestWaitSignalWriterWriteResetsIdle(t *testing.T) {
	w := &syncWriter{}
	ws := newWaitSignalWriter(w, 60*time.Millisecond)
	ws.Start()
	defer ws.Stop()

	// Writes every 50ms — below the 60ms idle — must keep pushing the signal back.
	for i := 0; i < 3; i++ {
		if _, err := ws.Write([]byte("data: {\"a\":1}\n\n")); err != nil {
			t.Fatalf("Write: %v", err)
		}
		time.Sleep(50 * time.Millisecond)
	}
	if strings.Contains(w.content(), "router_wait") {
		t.Fatalf("router_wait fired despite steady writes: %q", w.content())
	}
	// After the last write, wait for idle + buffer.
	time.Sleep(120 * time.Millisecond)
	if !strings.Contains(w.content(), "router_wait") {
		t.Errorf("router_wait should fire after the last write: %q", w.content())
	}
}

func TestWaitSignalWriterStopPreventsSignals(t *testing.T) {
	w := &syncWriter{}
	ws := newWaitSignalWriter(w, 30*time.Millisecond)
	ws.Start()

	time.Sleep(80 * time.Millisecond)
	before := w.content()
	if !strings.Contains(before, "router_wait") {
		t.Fatalf("expected router_wait before Stop: %q", before)
	}

	ws.Stop()
	time.Sleep(80 * time.Millisecond)
	after := w.content()
	// Count occurrences — should be the same as before (no new signals).
	beforeCount := strings.Count(before, "router_wait")
	afterCount := strings.Count(after, "router_wait")
	if afterCount != beforeCount {
		t.Errorf("router_wait fired after Stop: before=%d after=%d", beforeCount, afterCount)
	}
}

func TestWaitSignalWriterSignalFormat(t *testing.T) {
	w := &syncWriter{}
	ws := newWaitSignalWriter(w, 30*time.Millisecond)
	ws.Start()
	defer ws.Stop()

	time.Sleep(80 * time.Millisecond)
	content := w.content()
	if !strings.Contains(content, "event: router_wait") {
		t.Errorf("expected 'event: router_wait' in output, got %q", content)
	}
	if !strings.Contains(content, `"status":"waiting"`) {
		t.Errorf("expected 'status: waiting' in output, got %q", content)
	}
}
