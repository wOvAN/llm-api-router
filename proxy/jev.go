package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// JevQuestion is one System One question. Type is "noul", "choice" or "score";
// Criteria is a list of names for Choice, a legend for Score, nil for Noul.
type JevQuestion struct {
	Type         string `json:"type"`
	Instructions string `json:"instructions,omitempty"`
	Criteria     any    `json:"criteria,omitempty"`
}

// JevAnswer is one parsed answer from the "answers" object.
type JevAnswer struct {
	Type          string
	Choice        string
	Noul          float64 // probability for a Noul answer
	HasNoul       bool
	Score         int
	Confidence    float64
	Probabilities map[string]float64
}

// JevResponse is a parsed System One response.
type JevResponse struct {
	Model        string
	Answers      map[string]JevAnswer
	InputTokens  int
	OutputTokens int
}

// Answer returns the answer for a question name.
func (r *JevResponse) Answer(name string) (JevAnswer, bool) {
	a, ok := r.Answers[name]
	return a, ok
}

// jevMaxResponse bounds the decision response we are willing to buffer.
const jevMaxResponse = 1 << 20

// AskJev posts a System One decision request to baseURL (a server base URL, the
// /v1/systemone path is appended with the same dedup the proxy uses) and parses
// the answers. images (data URLs, llama.cpp accepts max 8) are optional vision
// input for decision models that support it. proxyURL is the decision server's
// own proxy, timeout bounds the whole exchange, ctx cancels it (client disconnect
// included).
func AskJev(ctx context.Context, baseURL, apiKey, model string, state any, images []string, questions map[string]JevQuestion, proxyURL string, timeout time.Duration) (*JevResponse, error) {
	endpoint := strings.TrimRight(baseURL, "/")
	if !strings.HasSuffix(endpoint, "/v1") {
		endpoint += "/v1"
	}
	endpoint += "/systemone"

	payload := map[string]any{
		"state":     state,
		"model":     model,
		"questions": questions,
	}
	if len(images) > 0 {
		payload["images"] = images
	}
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("encode jev request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payloadBytes))
	if err != nil {
		return nil, fmt.Errorf("build jev request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if apiKey != "" {
		req.Header.Set("x-api-key", apiKey)
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	transport, err := TransportFor(proxyURL)
	if err != nil {
		return nil, fmt.Errorf("jev transport: %w", err)
	}
	client := &http.Client{Transport: transport, Timeout: timeout}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("jev request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(io.LimitReader(resp.Body, jevMaxResponse))
	if err != nil {
		return nil, fmt.Errorf("read jev response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("jev endpoint %d: %s", resp.StatusCode, truncateBytes(body, 240))
	}

	var obj map[string]any
	if err := json.Unmarshal(body, &obj); err != nil {
		return nil, fmt.Errorf("parse jev response: %w", err)
	}

	out := &JevResponse{}
	if m, ok := obj["model"].(string); ok {
		out.Model = m
	}
	if u := getField(obj, "usage"); u != nil {
		out.InputTokens = intToFloat64(u["input_tokens"]) + intToFloat64(u["prompt_tokens"])
		out.OutputTokens = intToFloat64(u["output_tokens"]) + intToFloat64(u["completion_tokens"])
	}
	answers := getField(obj, "answers")
	if len(answers) == 0 {
		return nil, fmt.Errorf("jev response has no answers")
	}
	out.Answers = make(map[string]JevAnswer, len(answers))
	for name, v := range answers {
		ans, ok := v.(map[string]any)
		if !ok {
			continue
		}
		a := JevAnswer{Confidence: floatToFloat64(ans["confidence"])}
		a.Type, _ = ans["type"].(string)
		a.Choice, _ = ans["choice"].(string)
		a.Noul, a.HasNoul = floatToFloat64(ans["noul"]), ans["noul"] != nil
		a.Score = intToFloat64(ans["score"])
		if probs, ok := ans["probabilities"].(map[string]any); ok {
			a.Probabilities = make(map[string]float64, len(probs))
			for k, p := range probs {
				a.Probabilities[k] = floatToFloat64(p)
			}
		}
		// A Choice answer without an explicit confidence reports the winning
		// option's probability instead (JevRouter's provider does the same).
		if a.Confidence == 0 && a.Choice != "" {
			a.Confidence = a.Probabilities[a.Choice]
		}
		out.Answers[name] = a
	}
	return out, nil
}
