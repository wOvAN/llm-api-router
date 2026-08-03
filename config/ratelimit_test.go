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
	rl.RecordSuccess("srv-1")    // should not panic
	rl.RecordFailure("srv-1", 0) // should not panic
}

func TestRateLimiterRecordsSuccess(t *testing.T) {
	rl := NewRateLimiter(3, 30*time.Second, 1*time.Minute)

	// Record some failures
	rl.RecordFailure("srv-1", 0)
	rl.RecordFailure("srv-1", 0)

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

	rl.RecordFailure("srv-1", 0)
	rl.RecordFailure("srv-1", 0)

	if rl.ShouldSkip("srv-1") {
		t.Error("should not skip yet (below threshold)")
	}

	triggered := rl.RecordFailure("srv-1", 0)

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
		rl.RecordFailure("srv-1", 0)
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

	rl.RecordFailure("srv-1", 0)
	rl.RecordFailure("srv-1", 0)

	// Wait for window to expire
	time.Sleep(100 * time.Millisecond)

	// These failures should be in a new window
	rl.RecordFailure("srv-1", 0)
	rl.RecordFailure("srv-1", 0)

	if rl.ShouldSkip("srv-1") {
		t.Error("should not skip — old failures expired, only 2 in new window")
	}
}

func TestRateLimiterCooldownRemaining(t *testing.T) {
	rl := NewRateLimiter(3, 30*time.Second, 1*time.Minute)

	for i := 0; i < 3; i++ {
		rl.RecordFailure("srv-1", 0)
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
		rl.RecordFailure("srv-1", 0)
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
		rl.RecordFailure("srv-a", 0)
	}

	// Server B is below threshold
	rl.RecordFailure("srv-b", 0)
	rl.RecordFailure("srv-b", 0)

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
		rl.RecordFailure("srv-1", 0)
	}

	// Wait for cooldown to expire
	time.Sleep(100 * time.Millisecond)

	// New failures — should trigger cooldown again
	for i := 0; i < 3; i++ {
		rl.RecordFailure("srv-1", 0)
	}

	if !rl.ShouldSkip("srv-1") {
		t.Error("should be in cooldown again after new failures")
	}
}

func TestRateLimiterImmediateCooldownOn429(t *testing.T) {
	rl := NewRateLimiter(3, 30*time.Second, 1*time.Minute)

	if triggered := rl.RecordFailure("srv-1", 429); !triggered {
		t.Error("429 should trigger immediate cooldown on first occurrence")
	}
	if !rl.ShouldSkip("srv-1") {
		t.Error("should skip after single 429")
	}
}

func TestRateLimiterImmediateCooldownOn401And403(t *testing.T) {
	for _, status := range []int{401, 403} {
		rl := NewRateLimiter(3, 30*time.Second, 1*time.Minute)
		if triggered := rl.RecordFailure("srv-1", status); !triggered {
			t.Errorf("status %d should trigger immediate cooldown", status)
		}
		if !rl.ShouldSkip("srv-1") {
			t.Errorf("should skip after single %d", status)
		}
	}
}

func TestRateLimiter5xxUsesServerThreshold(t *testing.T) {
	rl := NewRateLimiter(3, 30*time.Second, 1*time.Minute)

	rl.RecordFailure("srv-1", 500)
	rl.RecordFailure("srv-1", 503)
	if rl.ShouldSkip("srv-1") {
		t.Error("should not skip yet (below 5xx threshold)")
	}

	if triggered := rl.RecordFailure("srv-1", 502); !triggered {
		t.Error("should trigger cooldown on 3rd 5xx failure")
	}
}

func TestRateLimiter4xxUsesLenientThreshold(t *testing.T) {
	rl := NewRateLimiter(3, 30*time.Second, 1*time.Minute)

	// 3 client-side 4xx failures are NOT enough (lenient threshold = 2*maxFails = 6)
	for i := 0; i < 3; i++ {
		rl.RecordFailure("srv-1", 400)
	}
	if rl.ShouldSkip("srv-1") {
		t.Error("should not skip after 3 client 4xx failures")
	}

	// A single 5xx failure mixes in but doesn't reach its threshold either
	rl.RecordFailure("srv-1", 502)
	if rl.ShouldSkip("srv-1") {
		t.Error("should not skip — 5xx count below threshold, 4xx count below lenient threshold")
	}

	// Reach the lenient threshold with 4xx failures
	for i := 0; i < 3; i++ {
		rl.RecordFailure("srv-1", 400)
	}
	if !rl.ShouldSkip("srv-1") {
		t.Error("should skip after 6 client 4xx failures")
	}
}

func TestRateLimiterMixedKindsCountSeparately(t *testing.T) {
	rl := NewRateLimiter(3, 30*time.Second, 1*time.Minute)

	// 2 transport + 2 client-4xx: neither threshold reached independently
	rl.RecordFailure("srv-1", 0)
	rl.RecordFailure("srv-1", 0)
	rl.RecordFailure("srv-1", 400)
	rl.RecordFailure("srv-1", 400)
	if rl.ShouldSkip("srv-1") {
		t.Error("should not skip — kinds are counted independently")
	}
}

func TestRateLimiterPerServerCooldownOverride(t *testing.T) {
	rl := NewRateLimiter(3, 30*time.Second, 1*time.Minute)
	rl.SetCooldownOverride(func(id string) time.Duration {
		if id == "short" {
			return 50 * time.Millisecond
		}
		return 0 // global default
	})

	// Server with override: short cooldown
	rl.RecordFailure("short", 429)
	if !rl.ShouldSkip("short") {
		t.Error("should be in cooldown after 429")
	}
	time.Sleep(100 * time.Millisecond)
	if rl.ShouldSkip("short") {
		t.Error("override cooldown should have expired")
	}

	// Server without override: long cooldown
	rl.RecordFailure("default", 429)
	if !rl.ShouldSkip("default") {
		t.Error("should be in cooldown with default duration")
	}
}
