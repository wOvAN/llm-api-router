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
