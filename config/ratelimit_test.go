package config

import (
	"testing"
	"time"
)

func TestRateLimiterNoSkipWhenHealthy(t *testing.T) {
	rl := NewRateLimiter(3, 30*time.Second, 1*time.Minute)

	if rl.ShouldSkip("srv-1") {
		t.Error("should not skip healthy server")
	}
}

func TestRateLimiterNilSafe(t *testing.T) {
	var rl *RateLimiter

	if rl.ShouldSkip("srv-1") {
		t.Error("nil RateLimiter should not skip")
	}
	rl.RecordSuccess("srv-1") // should not panic
	rl.RecordFailure("srv-1")  // should not panic
}

func TestRateLimiterRecordsSuccess(t *testing.T) {
	rl := NewRateLimiter(3, 30*time.Second, 1*time.Minute)

	// Record some failures
	rl.RecordFailure("srv-1")
	rl.RecordFailure("srv-1")

	if rl.FailureCount("srv-1") != 2 {
		t.Errorf("expected 2 failures, got %d", rl.FailureCount("srv-1"))
	}

	// Record success — should clear failures
	rl.RecordSuccess("srv-1")

	if rl.FailureCount("srv-1") != 0 {
		t.Errorf("expected 0 failures after success, got %d", rl.FailureCount("srv-1"))
	}
	if rl.ShouldSkip("srv-1") {
		t.Error("should not skip after success")
	}
}

func TestRateLimiterTriggersCooldown(t *testing.T) {
	rl := NewRateLimiter(3, 30*time.Second, 1*time.Minute)

	rl.RecordFailure("srv-1")
	rl.RecordFailure("srv-1")

	if rl.ShouldSkip("srv-1") {
		t.Error("should not skip yet (below threshold)")
	}

	triggered := rl.RecordFailure("srv-1")

	if !triggered {
		t.Error("should have triggered cooldown on 3rd failure")
	}
	if !rl.ShouldSkip("srv-1") {
		t.Error("should skip after cooldown triggered")
	}
}

func TestRateLimiterCooldownExpires(t *testing.T) {
	rl := NewRateLimiter(3, 30*time.Second, 50*time.Millisecond)

	// Trigger cooldown
	for i := 0; i < 3; i++ {
		rl.RecordFailure("srv-1")
	}

	if !rl.ShouldSkip("srv-1") {
		t.Error("should be in cooldown")
	}

	// Wait for cooldown to expire
	time.Sleep(100 * time.Millisecond)

	if rl.ShouldSkip("srv-1") {
		t.Error("cooldown should have expired")
	}
}

func TestRateLimiterWindowExpires(t *testing.T) {
	rl := NewRateLimiter(3, 50*time.Millisecond, 1*time.Minute)

	rl.RecordFailure("srv-1")
	rl.RecordFailure("srv-1")

	// Wait for window to expire
	time.Sleep(100 * time.Millisecond)

	// These failures should be in a new window
	rl.RecordFailure("srv-1")
	rl.RecordFailure("srv-1")

	if rl.ShouldSkip("srv-1") {
		t.Error("should not skip — old failures expired, only 2 in new window")
	}
}

func TestRateLimiterCooldownRemaining(t *testing.T) {
	rl := NewRateLimiter(3, 30*time.Second, 1*time.Minute)

	for i := 0; i < 3; i++ {
		rl.RecordFailure("srv-1")
	}

	remaining := rl.CooldownRemaining("srv-1")
	if remaining <= 0 {
		t.Error("should have remaining cooldown")
	}
	if remaining > 1*time.Minute {
		t.Errorf("remaining %v exceeds cooldown duration", remaining)
	}

	// Not in cooldown
	rl2 := NewRateLimiter(3, 30*time.Second, 1*time.Minute)
	if rl2.CooldownRemaining("srv-1") != 0 {
		t.Error("should have 0 remaining cooldown")
	}
}

func TestRateLimiterClearCooldown(t *testing.T) {
	rl := NewRateLimiter(3, 30*time.Second, 1*time.Minute)

	for i := 0; i < 3; i++ {
		rl.RecordFailure("srv-1")
	}

	if !rl.ShouldSkip("srv-1") {
		t.Error("should be in cooldown")
	}

	rl.ClearCooldown("srv-1")

	if rl.ShouldSkip("srv-1") {
		t.Error("should not skip after clearing cooldown")
	}
	if rl.FailureCount("srv-1") != 0 {
		t.Error("failures should be cleared")
	}
}

func TestRateLimiterMultipleServers(t *testing.T) {
	rl := NewRateLimiter(3, 30*time.Second, 1*time.Minute)

	// Server A exceeds threshold
	for i := 0; i < 3; i++ {
		rl.RecordFailure("srv-a")
	}

	// Server B is below threshold
	rl.RecordFailure("srv-b")
	rl.RecordFailure("srv-b")

	if !rl.ShouldSkip("srv-a") {
		t.Error("srv-a should be in cooldown")
	}
	if rl.ShouldSkip("srv-b") {
		t.Error("srv-b should not be in cooldown")
	}
}

func TestRateLimiterRepeatedFailuresExtendCooldown(t *testing.T) {
	rl := NewRateLimiter(3, 30*time.Second, 50*time.Millisecond)

	// Trigger cooldown
	for i := 0; i < 3; i++ {
		rl.RecordFailure("srv-1")
	}

	// Wait for cooldown to expire
	time.Sleep(100 * time.Millisecond)

	// New failures — should trigger cooldown again
	for i := 0; i < 3; i++ {
		rl.RecordFailure("srv-1")
	}

	if !rl.ShouldSkip("srv-1") {
		t.Error("should be in cooldown again after new failures")
	}
}
