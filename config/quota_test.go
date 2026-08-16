package config

import (
	"testing"
	"time"
)

func TestQuotaNoLimitsAlwaysAllowed(t *testing.T) {
	q := NewQuotaTracker()
	q.SetLimitOverride(func(id string) (tpm, rpm int64) { return 0, 0 })

	if !q.Allow("srv-1") {
		t.Error("server without limits should always be allowed")
	}
}

func TestQuotaNilSafe(t *testing.T) {
	var q *QuotaTracker
	if !q.Allow("srv-1") {
		t.Error("nil QuotaTracker should allow everything")
	}
	q.RecordRequest("srv-1") // should not panic
	q.RecordTokens("srv-1", 100)
}

func TestQuotaTPMBlocksAtBoundary(t *testing.T) {
	q := NewQuotaTracker()
	q.SetLimitOverride(func(id string) (tpm, rpm int64) { return 1000, 0 })

	if !q.Allow("srv-1") {
		t.Fatal("should be allowed initially")
	}

	q.RecordTokens("srv-1", 600)
	if !q.Allow("srv-1") {
		t.Error("should be allowed below limit")
	}

	q.RecordTokens("srv-1", 400)
	if q.Allow("srv-1") {
		t.Error("should be blocked at token limit boundary")
	}
}

func TestQuotaTPMWindowExpires(t *testing.T) {
	q := NewQuotaTracker()
	q.SetLimitOverride(func(id string) (tpm, rpm int64) { return 1000, 0 })

	q.RecordTokens("srv-1", 1000)
	if q.Allow("srv-1") {
		t.Fatal("should be blocked at limit")
	}

	time.Sleep(70 * time.Millisecond)
	window60s = 50 * time.Millisecond
	defer func() { window60s = 60 * time.Second }()

	if !q.Allow("srv-1") {
		t.Error("should be allowed again after window expires")
	}
}

func TestQuotaRPMBlocksAtLimit(t *testing.T) {
	q := NewQuotaTracker()
	q.SetLimitOverride(func(id string) (tpm, rpm int64) { return 0, 2 })

	q.RecordRequest("srv-1")
	q.RecordRequest("srv-1")
	if q.Allow("srv-1") {
		t.Error("should be blocked at request limit")
	}
}

func TestQuotaUsage(t *testing.T) {
	q := NewQuotaTracker()
	q.SetLimitOverride(func(id string) (tpm, rpm int64) { return 0, 0 })

	q.RecordTokens("srv-1", 250)
	q.RecordTokens("srv-1", 350)
	q.RecordRequest("srv-1")

	tokens, requests := q.Usage("srv-1")
	if tokens != 600 {
		t.Errorf("tokens = %d, want 600", tokens)
	}
	if requests != 1 {
		t.Errorf("requests = %d, want 1", requests)
	}
}

func TestQuotaPerServerLimits(t *testing.T) {
	q := NewQuotaTracker()
	q.SetLimitOverride(func(id string) (tpm, rpm int64) {
		if id == "limited" {
			return 100, 0
		}
		return 0, 0
	})

	q.RecordTokens("limited", 100)
	if q.Allow("limited") {
		t.Error("limited server should be blocked")
	}
	if !q.Allow("unlimited") {
		t.Error("unlimited server should be allowed")
	}
}

func TestQuotaReserveBlocksConcurrent(t *testing.T) {
	q := NewQuotaTracker()
	q.SetLimitOverride(func(id string) (tpm, rpm int64) { return 1000, 0 })

	// Two in-flight requests each reserving 600: the first fits, the second
	// would push the committed total past the limit and is blocked.
	if !q.Reserve("srv-1", 600) {
		t.Fatal("first reservation should be admitted")
	}
	if q.Reserve("srv-1", 600) {
		t.Error("second 600 reservation should be blocked (600+600>1000)")
	}
	// A reservation that exactly fills the remaining budget is admitted.
	if !q.Reserve("srv-1", 400) {
		t.Error("400 should fit exactly under the remaining 400")
	}
	// Fully reserved: nothing more fits.
	if q.Reserve("srv-1", 1) {
		t.Error("should be blocked once the budget is fully reserved")
	}
}

func TestQuotaCompleteReleasesReservation(t *testing.T) {
	q := NewQuotaTracker()
	q.SetLimitOverride(func(id string) (tpm, rpm int64) { return 1000, 0 })

	if !q.Reserve("srv-1", 800) {
		t.Fatal("reservation should be admitted")
	}
	if q.Reserve("srv-1", 300) {
		t.Error("300 should be blocked while 800 is reserved (800+300>1000)")
	}

	// Complete releases the 800 reservation and records 100 actual tokens.
	q.Complete("srv-1", 800, 100)
	// Now used=100, reserved=0: a 900 reservation fits (100+900<=1000).
	if !q.Reserve("srv-1", 900) {
		t.Error("after release, 900 should fit under the remaining 900")
	}
}

func TestQuotaReserveNoBudgetAlwaysAdmitted(t *testing.T) {
	q := NewQuotaTracker()
	q.SetLimitOverride(func(id string) (tpm, rpm int64) { return 100, 0 })

	// No declared output budget (estimate 0) reserves nothing and is always
	// admitted, even at the limit.
	if !q.Reserve("srv-1", 0) {
		t.Error("zero-estimate reservation should be admitted")
	}
	q.RecordTokens("srv-1", 100)
	if !q.Reserve("srv-1", 0) {
		t.Error("zero-estimate reservation should stay admitted at the limit")
	}
}
