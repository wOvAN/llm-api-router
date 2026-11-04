package config

import (
	"bytes"
	"encoding/json"
	"net/http"
	"sync"
	"time"
)

// Alerter is the operator's out-of-band notice: one edge event (a backend went
// down, came back, entered its cooldown) sent somewhere they are actually
// watching. LiteLLM has the same surface (alerting, alert_to_webhook_url); a
// self-hosted router with a GPU on the desk has no other way to learn that a
// replica died at 03:00, because nobody is looking at the GUI.
type Alerter interface {
	Notify(event string)
}

// webhookAlerter posts each event as JSON to an HTTP endpoint. Fire and forget:
// an alerting path that can block a request path is worse than no alerting, and
// an unreachable webhook must not become an outage on the thing it reports.
type webhookAlerter struct {
	url    string
	client *http.Client

	// mu guards sent, the dedup window: a wedged backend that fails every
	// request must not send one alert per request.
	mu   sync.Mutex
	sent map[string]time.Time
}

// alertDedupWindow bounds how often the same event text may be sent.
const alertDedupWindow = time.Minute

// NewWebhookAlerter returns nil (no alerts) for an empty URL, so the caller can
// install whatever it gets unconditionally.
func NewWebhookAlerter(url string) Alerter {
	if url == "" {
		return nil
	}
	return &webhookAlerter{
		url:    url,
		client: &http.Client{Timeout: 5 * time.Second},
		sent:   make(map[string]time.Time),
	}
}

func (a *webhookAlerter) Notify(event string) {
	if a == nil || event == "" {
		return
	}
	a.mu.Lock()
	if last, ok := a.sent[event]; ok && time.Since(last) < alertDedupWindow {
		a.mu.Unlock()
		return
	}
	a.sent[event] = time.Now()
	a.mu.Unlock()

	body, err := json.Marshal(map[string]string{"event": event, "at": time.Now().UTC().Format(time.RFC3339)})
	if err != nil {
		return
	}
	go func() {
		resp, err := a.client.Post(a.url, "application/json", bytes.NewReader(body))
		if err != nil {
			return
		}
		_ = resp.Body.Close()
	}()
}
