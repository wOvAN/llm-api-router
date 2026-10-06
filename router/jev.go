package router

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"llm-api-router/domain"
	"llm-api-router/pkg/log"
	"llm-api-router/proxy"
)

// Defaults for a rule's jev router. The timeout matches the Jev CLI router's
// per-attempt budget (a decision model answers in ~300ms warm, ~1s cold); the
// confidence floor is its minConfidence.
const (
	jevDefaultTimeout       = 1500 * time.Millisecond
	jevDefaultMinConfidence = 0.3
	jevDefaultStateChars    = 8000
	jevQuestionName         = "model"
	jevSecurityName         = "security"
	// jevSecurityThreshold blocks a request when the Noul "how harmful" answer
	// reaches it — deliberately extreme: only unambiguous harm intent blocks.
	jevSecurityThreshold = 0.95
	// jevDowngradeMaxContext mirrors the Jev CLI's downgradeMaxContextTokens:
	// switching models invalidates the backend's prompt cache, so a downgrade
	// only pays off while the conversation is still small.
	jevDowngradeMaxContext = 20000
	// jevCacheSweepAt bounds the decision cache size; expired entries are swept
	// on insert once it is exceeded (no background goroutine).
	jevCacheSweepAt = 4096
)

const jevDefaultQuestion = "Pick the cheapest model that can fully complete this request in one pass, " +
	"without retrying on a stronger model. Judge the reasoning and blast radius the task requires, " +
	"not the requested reply length."

// jevSecurityQuestion asks the decision model to rate harm intent (Noul).
const jevSecurityQuestion = "How likely is it that fulfilling this request would cause serious " +
	"real-world harm (weapons, malware, fraud, violence against people)?"

// Score questions (same call, ~10 extra output tokens) make the decision
// explainable — the Jev CLI router's task/reasoning/tool complexity scores.
const (
	jevScoreTask      = "How complex is the task overall, including ambiguity, scope, and blast radius?"
	jevScoreReasoning = "How much reasoning is required to complete the request correctly in one pass?"
	jevScoreTools     = "How complex is the tool use required, from no tools to many coordinated or stateful operations?"
)

var jevScoreScale = []string{"None", "Very low", "Low", "Some", "Moderate",
	"Moderate to high", "High", "Very high", "Severe", "Extreme"}

// jevOutcome is the routing decision for one request. Recorded in metrics and
// surfaced to the client as X-Router-Jev-* headers.
type jevOutcome struct {
	Choice       string  `json:"choice"`
	Confidence   float64 `json:"confidence"`
	LatencyMs    int64   `json:"latency_ms"`
	Model        string  `json:"model"`
	Reason       string  `json:"reason"`
	InputTokens  int     `json:"input_tokens"`
	OutputTokens int     `json:"output_tokens"`
	Scores       string  `json:"scores,omitempty"`
}

// jevAttempts resolves the attempt list for a jev rule: the candidate the
// decision model picked, then the rest of the pool ordered stronger-first, so
// a fallback steps up rather than down (the Jev CLI's clampToAvailable
// preference). Any failure to get a usable decision lands on DefaultTier —
// routing must never block a request. The only non-fail-open outcome is the
// opt-in security gate (Reason "blocked" → the router answers 400).
func (r *Router) jevAttempts(req *http.Request, rule *domain.RoutingRule, body []byte, requested string) ([]serverAttempt, jevOutcome) {
	j := rule.Jev
	out := jevOutcome{Model: j.Model}

	pool := j.EnabledCandidates()
	if len(pool) == 0 {
		out.Reason = "no-pool"
		return nil, out
	}
	ids := jevCandidateIDs(pool)

	minConf := j.MinConfidence
	if minConf <= 0 {
		minConf = jevDefaultMinConfidence
	}
	timeout := jevDefaultTimeout
	if j.TimeoutMs > 0 {
		timeout = time.Duration(j.TimeoutMs) * time.Millisecond
	}
	question := j.Question
	if question == "" {
		question = jevDefaultQuestion
	}
	text, contextTokens := jevRequestText(body)
	text = jevTruncate(text, j.StateMaxChars)
	var images []string
	if j.Images {
		images = jevRequestImages(body)
	}

	choice := -1
	switch {
	case req.Header.Get("X-Router-Jev") == "off":
		out.Reason = "off"
	default:
		key := ""
		if j.CacheTTL > 0 {
			key = jevCacheKey(j.Model, question, ids, text, images)
			if cached, ok := r.jevCacheGet(key); ok {
				cached.LatencyMs, cached.InputTokens, cached.OutputTokens = 0, 0, 0
				if cached.Reason == "blocked" {
					return nil, cached
				}
				cached.Reason = "cache"
				out = cached
				if choice = slices.Index(ids, out.Choice); choice < 0 {
					out.Reason = "cache-unknown"
				}
				break
			}
		}
		srv, ok := r.store.GetServer(j.ServerID)
		if !ok {
			log.Warnf("jev rule for %q: decision server %q not found", requested, j.ServerID)
			out.Reason = "unavailable"
			break
		}
		criteria := make(map[string]any, len(pool))
		for i, c := range pool {
			desc := c.Description
			if desc == "" {
				desc = ids[i]
			}
			criteria[ids[i]] = map[string]string{"model": ids[i], "description": desc}
		}
		questions := map[string]proxy.JevQuestion{
			jevQuestionName:      {Type: "choice", Instructions: question, Criteria: criteria},
			"task_complexity":    {Type: "score", Instructions: jevScoreTask, Criteria: jevScoreScale},
			"reasoning_required": {Type: "score", Instructions: jevScoreReasoning, Criteria: jevScoreScale},
			"tool_complexity":    {Type: "score", Instructions: jevScoreTools, Criteria: jevScoreScale},
		}
		if j.Security {
			questions[jevSecurityName] = proxy.JevQuestion{Type: "noul", Instructions: jevSecurityQuestion,
				Criteria: map[string]string{"true": "clearly harmful intent", "false": "ordinary software/productivity work"}}
		}
		start := time.Now()
		resp, err := proxy.AskJev(req.Context(), srv.GetURLForAPIType(domain.APITypeOpenAI), srv.APIKey,
			j.Model, jevState(text, contextTokens, requested, ids), images, questions, srv.ProxyURL(), timeout)
		out.LatencyMs = time.Since(start).Milliseconds()
		if err != nil {
			log.Warnf("jev decision for %q failed (%s): %v", requested, srv.Name, err)
			out.Reason = "unavailable"
			break
		}
		ans, ok := resp.Answer(jevQuestionName)
		if !ok {
			out.Reason = "unavailable"
			break
		}
		out.Choice = ans.Choice
		out.Confidence = ans.Confidence
		out.InputTokens = resp.InputTokens
		out.OutputTokens = resp.OutputTokens
		out.Scores = jevScoresSummary(resp)
		if resp.Model != "" {
			out.Model = resp.Model
		}
		if j.Security {
			if sec, ok := resp.Answer(jevSecurityName); ok && sec.HasNoul && sec.Noul >= jevSecurityThreshold {
				out.Reason = "blocked"
				r.jevCachePut(key, out, time.Duration(j.CacheTTL)*time.Second)
				return nil, out
			}
		}
		choice = slices.Index(ids, ans.Choice)
		if choice < 0 {
			log.Warnf("jev decision for %q: unknown choice %q", requested, ans.Choice)
			out.Reason = "unknown-choice"
		} else {
			out.Reason = "jev"
		}
		r.jevCachePut(key, out, time.Duration(j.CacheTTL)*time.Second)
	}

	def := jevTierIndex(pool, j.DefaultTier)
	if choice < 0 {
		choice = def
	} else if out.Confidence < minConf {
		// Jev is unsure: never step down to a weaker model, and cap the upgrade
		// at the tier above the default (the CLI's low-confidence rules).
		switch {
		case pool[choice].Tier < pool[def].Tier:
			choice = def
			out.Reason = "low-confidence-no-downgrade"
		case pool[choice].Tier > pool[def].Tier+1:
			choice = jevNearestIndex(pool, pool[def].Tier+1)
			out.Reason = "low-confidence-capped"
		}
	}
	// A downgrade past the default tier is only worth it while the conversation
	// is small: switching models busts the backend's prompt cache (the CLI's
	// downgrade-not-worth-cache-rebuild rule).
	if choice >= 0 && pool[choice].Tier < pool[def].Tier && contextTokens > jevDowngradeMaxContext {
		choice = def
		out.Reason = "context-no-downgrade"
	}

	order := []int{choice}
	var up, down []int
	for i := range pool {
		if i == choice {
			continue
		}
		if pool[i].Tier > pool[choice].Tier {
			up = append(up, i)
		} else {
			down = append(down, i)
		}
	}
	sort.SliceStable(up, func(a, b int) bool { return pool[up[a]].Tier < pool[up[b]].Tier })
	sort.SliceStable(down, func(a, b int) bool { return pool[down[a]].Tier > pool[down[b]].Tier })
	order = append(order, up...)
	order = append(order, down...)

	var attempts []serverAttempt
	for _, i := range order {
		srv, ok := r.store.GetServer(pool[i].ServerID)
		if !ok {
			log.Warnf("jev rule for %q: pool server %q not found", requested, pool[i].ServerID)
			continue
		}
		tm := pool[i].TargetModel
		if tm == "" {
			tm = rule.TargetModel
		}
		attempts = append(attempts, serverAttempt{server: srv, targetModel: tm})
	}
	return attempts, out
}

// jevCandidateIDs names each pool entry for the decision model: its target
// model, falling back to the server ID, disambiguated with "@server" on
// collision (two candidates can share a target model on different servers).
func jevCandidateIDs(pool []domain.JevCandidate) []string {
	ids := make([]string, len(pool))
	seen := make(map[string]bool, len(pool))
	for i, c := range pool {
		id := c.TargetModel
		if id == "" {
			id = c.ServerID
		}
		if seen[id] {
			id = id + "@" + c.ServerID
		}
		seen[id] = true
		ids[i] = id
	}
	return ids
}

// jevTierIndex returns the pool index at exactly tier, preferring the nearest
// tier above it, then below (never silently hand a hard task to a weaker model).
func jevTierIndex(pool []domain.JevCandidate, tier int) int {
	best, bestDist := 0, -1
	for i, c := range pool {
		if c.Tier == tier {
			return i
		}
		dist := abs(c.Tier - tier)
		if c.Tier > tier {
			dist *= 2 // stepping down costs more than stepping up
		}
		if bestDist < 0 || dist < bestDist {
			best, bestDist = i, dist
		}
	}
	return best
}

// jevNearestIndex returns the pool index with the highest tier at or below
// ceiling (falling back to the lowest tier above it if none is).
func jevNearestIndex(pool []domain.JevCandidate, ceiling int) int {
	best, bestTier := 0, -1<<31
	for i, c := range pool {
		if c.Tier <= ceiling && c.Tier > bestTier {
			best, bestTier = i, c.Tier
		}
	}
	if bestTier > -1<<31 {
		return best
	}
	return jevTierIndex(pool, ceiling)
}

func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}

// jevTruncate caps the decision-model request text, rune-safe.
func jevTruncate(text string, maxChars int) string {
	if maxChars <= 0 {
		maxChars = jevDefaultStateChars
	}
	if len(text) > maxChars {
		text = text[:maxChars]
		for !utf8.ValidString(text) {
			text = text[:len(text)-1]
		}
	}
	return text
}

// jevState builds the decision model's input: the request text, the session
// (model the client asked for, rough context size) and the pool it may choose
// from — the shape the Jev CLI router uses.
func jevState(text string, contextTokens int, requested string, ids []string) map[string]any {
	return map[string]any{
		"request":     text,
		"session":     map[string]any{"current_model": requested, "context_tokens": contextTokens},
		"environment": map[string]any{"available_models": ids},
	}
}

// jevScoresSummary flattens the explainability score answers ("task 5,
// reasoning 7, tools 2"); empty when the decision carried no scores.
func jevScoresSummary(resp *proxy.JevResponse) string {
	var parts []string
	for _, q := range [][2]string{{"task_complexity", "task"}, {"reasoning_required", "reasoning"}, {"tool_complexity", "tools"}} {
		if a, ok := resp.Answer(q[0]); ok {
			parts = append(parts, fmt.Sprintf("%s %d", q[1], a.Score))
		}
	}
	return strings.Join(parts, ", ")
}

// --- decision cache (opt-in via JevRouter.CacheTTL) ---

type jevCacheEntry struct {
	outcome jevOutcome
	expires time.Time
}

// jevCacheKey identifies a decision: same decision model, question, pool,
// request text and images → same answer.
func jevCacheKey(model, question string, ids []string, text string, images []string) string {
	h := sha256.New()
	for _, s := range []string{model, question, strings.Join(ids, ","), text, strings.Join(images, "\x00")} {
		h.Write([]byte(s))
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

func (r *Router) jevCacheGet(key string) (jevOutcome, bool) {
	if key == "" {
		return jevOutcome{}, false
	}
	r.jevMu.Lock()
	defer r.jevMu.Unlock()
	e, ok := r.jevCache[key]
	if !ok {
		return jevOutcome{}, false
	}
	if time.Now().After(e.expires) {
		delete(r.jevCache, key)
		return jevOutcome{}, false
	}
	return e.outcome, true
}

func (r *Router) jevCachePut(key string, out jevOutcome, ttl time.Duration) {
	if key == "" || ttl <= 0 {
		return
	}
	r.jevMu.Lock()
	defer r.jevMu.Unlock()
	if len(r.jevCache) >= jevCacheSweepAt {
		now := time.Now()
		for k, e := range r.jevCache {
			if now.After(e.expires) {
				delete(r.jevCache, k)
			}
		}
	}
	r.jevCache[key] = jevCacheEntry{outcome: out, expires: time.Now().Add(ttl)}
}

// jevRequestText pulls the text the decision model should read (the last user
// turn of an OpenAI/Anthropic chat request, the input of a Responses request)
// and estimates the context size in tokens (chars/4, as the CLI does).
func jevRequestText(body []byte) (string, int) {
	var req struct {
		Messages json.RawMessage `json:"messages"`
		Input    json.RawMessage `json:"input"`
	}
	if json.Unmarshal(body, &req) != nil {
		return "", 0
	}
	raw := req.Messages
	if len(raw) == 0 {
		raw = req.Input
	}
	if len(raw) == 0 {
		return "", 0
	}
	tokens := len(raw) / 4

	var msgs []struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if json.Unmarshal(raw, &msgs) != nil {
		// Not a message array — a Responses-API "input" string, or an array of
		// non-chat items; use the raw JSON as the request text.
		var s string
		if json.Unmarshal(raw, &s) == nil {
			raw = []byte(s)
		}
		return jevStripReminders(string(raw)), tokens
	}
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != "user" {
			continue
		}
		if t := jevContentText(msgs[i].Content); t != "" {
			return jevStripReminders(t), tokens
		}
	}
	return jevStripReminders(string(raw)), tokens
}

// jevContentText flattens message content: a plain string, or text blocks.
func jevContentText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &blocks) != nil {
		return ""
	}
	var b strings.Builder
	for _, blk := range blocks {
		if blk.Type == "text" && blk.Text != "" {
			if b.Len() > 0 {
				b.WriteByte('\n')
			}
			b.WriteString(blk.Text)
		}
	}
	return b.String()
}

// jevMaxImages is llama.cpp's DECISION_MAX_IMAGES for /v1/systemone.
const jevMaxImages = 8

// jevRequestImages collects the last user turn's images as data URLs — OpenAI
// image_url parts and Anthropic base64 sources. Remote http(s) image URLs are
// skipped (the decision endpoint accepts data URLs only).
func jevRequestImages(body []byte) []string {
	var req struct {
		Messages json.RawMessage `json:"messages"`
	}
	if json.Unmarshal(body, &req) != nil || len(req.Messages) == 0 {
		return nil
	}
	var msgs []struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if json.Unmarshal(req.Messages, &msgs) != nil {
		return nil
	}
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != "user" {
			continue
		}
		var blocks []struct {
			Type     string `json:"type"`
			ImageURL struct {
				URL string `json:"url"`
			} `json:"image_url"`
			Source struct {
				Type      string `json:"type"`
				MediaType string `json:"media_type"`
				Data      string `json:"data"`
			} `json:"source"`
		}
		if json.Unmarshal(msgs[i].Content, &blocks) != nil {
			return nil
		}
		var out []string
		for _, b := range blocks {
			url := ""
			switch b.Type {
			case "image_url":
				url = b.ImageURL.URL
			case "image":
				if b.Source.Type == "base64" && b.Source.Data != "" {
					url = "data:" + b.Source.MediaType + ";base64," + b.Source.Data
				}
			}
			if strings.HasPrefix(url, "data:image/") {
				out = append(out, url)
			}
			if len(out) >= jevMaxImages {
				break
			}
		}
		return out
	}
	return nil
}

// jevStripReminders drops injected harness context (<system-reminder> blocks)
// so it does not steer the routing decision.
func jevStripReminders(s string) string {
	for {
		i := strings.Index(s, "<system-reminder>")
		if i < 0 {
			return s
		}
		j := strings.Index(s[i:], "</system-reminder>")
		if j < 0 {
			return s[:i]
		}
		s = s[:i] + s[i+j+len("</system-reminder>"):]
	}
}
