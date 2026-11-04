package router

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"llm-api-router/config"
	"llm-api-router/convert"
	"llm-api-router/domain"
	"llm-api-router/metrics"
	"llm-api-router/pkg/log"
	"llm-api-router/proxy"
	"llm-api-router/trace"
)

// Router handles incoming LLM API requests and routes them to configured backends.
type Router struct {
	store     *config.Store
	metrics   *metrics.Store
	health    *config.HealthTracker
	rateLimit *config.RateLimiter
	quota     *config.QuotaTracker
	caps      *config.CapabilitiesCache
	trace     *trace.Store
	rotations sync.Map                  // balanced rule pools: ids -> *atomic.Uint64
	pinMu     sync.Mutex                // guards pins
	pins      map[string]stickyPin      // conversation -> backend that served it
	jevMu     sync.Mutex                // guards jevCache and jevSticky
	jevCache  map[string]jevCacheEntry  // jev decision cache (opt-in per rule)
	jevSticky map[string]jevStickyEntry // last jev decision per conversation
}

// New creates a new Router.
func New(store *config.Store, m *metrics.Store, health *config.HealthTracker, rateLimit *config.RateLimiter,
	quota *config.QuotaTracker, tr *trace.Store, caps *config.CapabilitiesCache) *Router {
	if caps == nil {
		caps = config.NewCapabilitiesCache()
	}
	return &Router{store: store, metrics: m, health: health, rateLimit: rateLimit, quota: quota, trace: tr,
		caps: caps, jevCache: map[string]jevCacheEntry{},
		jevSticky: map[string]jevStickyEntry{}, pins: map[string]stickyPin{}}
}

// reqMeta is what a request *is*, computed once at request start: the
// classification (always, 0 tokens) and the trace capture (only inside the
// window). It rides into record so the metric and the trace entry agree.
type reqMeta struct {
	requestID    string
	class        string
	features     []string
	capture      *trace.Capture
	retries      int
	upstreamPath string
	convert      string
	convertNotes []string
	// translator is the serving attempt's response translator (nil =
	// passthrough), kept so record can merge the notes the response side cost.
	translator proxy.StreamTranslator
}

// serverAttempt is one proxy target: a backend server plus the model to send it.
type serverAttempt struct {
	server      *domain.Server
	targetModel string
	convert     string // the attempt's own `convert` value ("" = "auto")
	// contextWindow is what this backend declares it can fit (tokens); 0 =
	// unknown, which gates nothing.
	contextWindow int
}

// balanceRotation returns a balanced rule's attempt list starting on its next
// pool member, so successive requests land on different backends. Weights deal
// out the starting slot (a replica with weight 3 starts three requests out of
// four), and what follows that start stays the failover chain — a pool member
// that is down costs one rotation, not a request.
//
// The counter is keyed by the pool's server ids in order, so a config reload
// never resets the rotation and two pools never share a counter.
func (r *Router) balanceRotation(attempts []serverAttempt, weights []int, pinnedID string) []serverAttempt {
	ids := make([]string, len(attempts))
	slots := 0
	for i, a := range attempts {
		ids[i] = a.server.ID
		slots += weights[i]
	}
	counter, _ := r.rotations.LoadOrStore(strings.Join(ids, ","), new(atomic.Uint64))

	// A conversation already pinned to a member starts there and spends no
	// rotation: its turns must land on the backend holding its prompt cache, and
	// letting it advance the shared counter would drift every other
	// conversation's share along with it.
	if pinnedID != "" {
		for i, a := range attempts {
			if a.server.ID == pinnedID {
				if i == 0 {
					return attempts
				}
				rotated := make([]serverAttempt, 0, len(attempts))
				rotated = append(rotated, attempts[i:]...)
				return append(rotated, attempts[:i]...)
			}
		}
	}

	slot := int(counter.(*atomic.Uint64).Add(1)-1) % slots

	start := 0
	for i, w := range weights {
		if slot < w {
			start = i
			break
		}
		slot -= w
	}
	if start == 0 {
		return attempts
	}
	rotated := make([]serverAttempt, 0, len(attempts))
	rotated = append(rotated, attempts[start:]...)
	return append(rotated, attempts[:start]...)
}

// orderedFallbacks returns the rule's enabled fallbacks in attempt order:
// sorted by priority ascending (lower number tried first). Fallbacks with
// priority 0 (unset) sort after all prioritized ones, and their original list
// order is preserved among ties (stable sort).
func orderedFallbacks(rule *domain.RoutingRule) []domain.FallbackEntry {
	fbs := make([]domain.FallbackEntry, 0, len(rule.Fallbacks))
	for _, fb := range rule.Fallbacks {
		if fb.IsEnabled() {
			fbs = append(fbs, fb)
		}
	}
	sort.SliceStable(fbs, func(i, j int) bool {
		a, b := fbs[i].Priority, fbs[j].Priority
		if a == b {
			return false
		}
		if a == 0 {
			return false // unset sorts after set
		}
		if b == 0 {
			return true
		}
		return a < b
	})
	return fbs
}

// apiTypeFromPath determines the API type from the request path.
func apiTypeFromPath(path string) domain.APIType {
	if strings.Contains(path, "/messages") {
		return domain.APITypeAnthropic
	}
	return domain.APITypeOpenAI
}

// clientIP extracts the client IP from the request, checking X-Forwarded-For
// and X-Real-IP first (for proxied traffic), then falling back to RemoteAddr.
func clientIP(req *http.Request) string {
	if fwd := req.Header.Get("X-Forwarded-For"); fwd != "" {
		if before, _, ok := strings.Cut(fwd, ","); ok {
			return strings.TrimSpace(before)
		}
		return strings.TrimSpace(fwd)
	}
	if rip := req.Header.Get("X-Real-IP"); rip != "" {
		return strings.TrimSpace(rip)
	}
	if ip, _, err := net.SplitHostPort(req.RemoteAddr); err == nil {
		return ip
	}
	return req.RemoteAddr
}

// Handle processes the incoming request.
func (r *Router) Handle(w http.ResponseWriter, req *http.Request) {
	if req.Method == http.MethodGet && req.URL.Path == "/v1/models" {
		r.listModels(w, req)
		return
	}

	// Audio routes are the one GET family the router proxies: audio.cpp's
	// `GET /v1/audio/voices?model=<id>` (voice pickers in clients like Open
	// WebUI). Their model rides in the query string, so they join the normal
	// routing below.
	if req.Method != http.MethodPost &&
		(req.Method != http.MethodGet || !strings.HasPrefix(req.URL.Path, "/v1/audio/")) {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Cap the request body to protect against memory exhaustion. LLM prompts can
	// be large (long-context requests), so 64MB is generous.
	req.Body = http.MaxBytesReader(w, req.Body, 64<<20)
	body, err := io.ReadAll(req.Body)
	if err != nil {
		log.Errorf("[%s] failed to read request body: %v", req.URL.Path, err)
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return
	}
	defer func() {
		if err := req.Body.Close(); err != nil {
			log.Errorf("[%s] failed to close request body: %v", req.URL.Path, err)
		}
	}()

	// Classification is always on and costs a byte scan: metrics, the trace and
	// the GUI then describe a request with the same words.
	meta := &reqMeta{}
	// Every response carries a request id and the metric carries the same one, so
	// a client can quote a number in a bug report without the trace window being
	// armed. A client that already has an id keeps it: its logs and the router's
	// then agree.
	meta.requestID = strings.TrimSpace(req.Header.Get("X-Request-ID"))
	if meta.requestID == "" {
		meta.requestID = newRequestID()
	}
	w.Header().Set("X-Router-Request-Id", meta.requestID)
	meta.class, meta.features = trace.Classify(req.URL.Path, req.Header.Get("Content-Type"), body)

	// Anthropic clients ask what a prompt costs before sending it (Claude Code
	// fires bursts of these to decide when to compact). The answer needs no
	// generation, and most backends do not serve the endpoint at all — an
	// OpenAI-only backend answers 404, which the client can only log — so the
	// router answers with the same chars/4 estimate the context-window gate and
	// the Jev state use, over the whole body (system prompt, every turn, the
	// tool schemas: the material a real tokenizer would charge for). LiteLLM
	// calls the provider's real count_tokens first; a stdlib-only router has no
	// tokenizer, so the number is an estimate and X-Router-Count-Tokens says so.
	if strings.Contains(req.URL.Path, "/count_tokens") {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Router-Count-Tokens", "estimate")
		_, _ = fmt.Fprintf(w, `{"input_tokens":%d}`, len(body)/4)
		return
	}

	model, err := extractModel(body)
	if err != nil {
		// Endpoints with no JSON "model" field: audio uploads (OpenAI, llama.cpp,
		// audio.cpp) are multipart/form-data with a `model` form field; audio.cpp's
		// GET voice list takes it as a query parameter; llama.cpp's /v1/systemone
		// (TypeSafe decision API) has no model at all and is routed by endpoint
		// name.
		if m, ok := proxy.ExtractModelFromMultipart(body); ok {
			model = m
		} else if m := strings.TrimSpace(req.URL.Query().Get("model")); m != "" && req.Method == http.MethodGet {
			model = m
		} else if req.URL.Path == "/v1/systemone" {
			model = "systemone"
		} else {
			log.Errorf("[%s] invalid request body: %v", req.URL.Path, err)
			http.Error(w, fmt.Sprintf("invalid request body: %v", err), http.StatusBadRequest)
			return
		}
	}

	// Track active requests for metrics display.
	r.metrics.IncrActive()
	defer r.metrics.DecrActive()

	rule, ok := r.store.GetRuleByModel(model)
	if !ok {
		// LiteLLM's model_not_found names the models the proxy accepts, and that
		// is the whole difference between a 404 a caller can act on and one it
		// has to go and read the config for. Bounded: an unrouted model name is
		// the most popular thing to send a router.
		names := r.store.ModelNames()
		if len(names) > 20 {
			names = append(names[:20], "…")
		}
		msg := fmt.Sprintf("no routing rule for model %q", model)
		if len(names) > 0 {
			msg += "; this router routes: " + strings.Join(names, ", ")
		}
		log.Errorf("[%s] %s", req.URL.Path, msg)
		http.Error(w, msg, http.StatusNotFound)
		return
	}

	primaryServer, ok := r.store.GetServer(rule.ServerID)
	if !ok && !rule.IsJev() {
		log.Errorf("[%s] server %q not found for model %q", req.URL.Path, rule.ServerID, model)
		http.Error(w, fmt.Sprintf("server %q not found", rule.ServerID), http.StatusInternalServerError)
		return
	}

	var jev *jevOutcome
	var attempts []serverAttempt
	// pinKey is the conversation identity a balanced rule pins to one backend
	// ("" = the request carries no identity, so nothing is pinned).
	pinKey := ""
	requestStart := time.Now()
	apiType := apiTypeFromPath(req.URL.Path)
	// The client's own dialect (from its path) and the per-request conversion
	// override; heartbeats keep keying on apiType — they are for the client.
	clientDialect := convert.DialectForPath(req.URL.Path)
	headerConvert := req.Header.Get("X-Router-Convert")
	// The override is the client's instruction to US. Unlike X-Router-Retries and
	// X-Router-KeepAlive (which ride upstream today, harmless), a conversion edge
	// named in a header means nothing to a backend — and on a converted attempt the
	// upstream request is a clone of this one, so it is deleted here.
	req.Header.Del("X-Router-Convert")
	if rule.IsJev() {
		// The decision model picks the pool entry; the rest of the pool stays
		// behind it as the fallback chain.
		list, outcome := r.jevAttempts(req, rule, body, model)
		jev = &outcome
		attempts = list
	} else {
		// Static rule: the primary, then its enabled fallbacks in priority order.
		// With `balance: "round-robin"` the same list is a pool, and the request
		// starts on the next member of it.
		attempts = []serverAttempt{{server: primaryServer, targetModel: rule.TargetModel,
			convert: rule.Convert, contextWindow: rule.ContextWindow}}
		weights := []int{1}
		for _, fb := range orderedFallbacks(rule) {
			if srv, ok := r.store.GetServer(fb.ServerID); ok {
				tm := fb.TargetModel
				if tm == "" {
					tm = rule.TargetModel
				}
				window := fb.ContextWindow
				if window == 0 {
					window = rule.ContextWindow
				}
				attempts = append(attempts, serverAttempt{server: srv, targetModel: tm,
					convert: fb.Convert, contextWindow: window})
				weights = append(weights, max(fb.Weight, 1))
			}
		}
		if rule.Balance == domain.BalanceRoundRobin && len(attempts) > 1 {
			// Only a pool has a choice to pin, so only a pool pays the body scan.
			pinKey = conversationKey(req, body)
			attempts = r.balanceRotation(attempts, weights, r.stickyGet(pinKey))
		}
	}
	if jev != nil && jev.Reason == "blocked" {
		// Opt-in Noul security gate: the decision model rated the request as
		// near-certain harm intent. Nothing was forwarded upstream.
		r.record(req, requestStart, model, "", "", proxy.ProxyMetrics{}, http.StatusBadRequest, "blocked by jev security gate", false, apiType, jev, meta)
		log.Warnf("[%s] model=%q — jev security gate blocked the request", req.URL.Path, model)
		http.Error(w, "request blocked by security gate", http.StatusBadRequest)
		return
	}
	if len(attempts) == 0 {
		log.Errorf("[%s] model=%q — rule has no usable backend (jev pool empty or servers missing)", req.URL.Path, model)
		http.Error(w, "routing rule has no usable backend", http.StatusInternalServerError)
		return
	}
	// On-demand trace: the window is consulted once, here, so a request that
	// starts inside it is traced to the end even if the window expires
	// mid-generation. The capture holds the ORIGINAL body (redacted) — before
	// the stream-usage injection and before per-attempt model rewrites.
	if r.trace != nil && r.trace.Armed(model, attempts[0].server.ID) {
		meta.capture = trace.NewCapture(req.Method, req.URL.Path, model, req.Header.Get("Content-Type"), body)
		w.Header().Set("X-Router-Trace-Id", meta.capture.ID())
	}
	if jev != nil {
		// Observability: what the decision model answered, so a client (and the
		// metrics table) can see why a request landed on a given model.
		w.Header().Set("X-Router-Jev-Choice", jev.Choice)
		w.Header().Set("X-Router-Jev-Confidence", strconv.FormatFloat(jev.Confidence, 'g', 3, 64))
		w.Header().Set("X-Router-Jev-Ms", strconv.FormatInt(jev.LatencyMs, 10))
		w.Header().Set("X-Router-Jev-Reason", jev.Reason)
	}

	// SSE heartbeats keep streams alive when the backend pauses between tokens
	// (slow/thinking inference) — clients otherwise trip read timeouts
	// ("waiting for api responses", "api timeout"). Anthropic clients expect the
	// official ping event; OpenAI clients get a bare SSE comment line, which
	// every SSE parser ignores. StreamProxy only emits heartbeats for
	// text/event-stream responses.
	var ping []byte
	if apiType == domain.APITypeAnthropic {
		ping = []byte("event: ping\ndata: {\"type\":\"ping\"}\n\n")
	} else {
		ping = []byte(": keep-alive\n\n")
	}
	var lastErr error
	// The refusal of the last attempt: a backend this attempt can not speak to,
	// which nothing downstream of it can fix.
	var refusal error
	// The last backend answer the proxy buffered (429/5xx, nothing sent to the
	// client). If no attempt serves, this is what the client gets — the
	// backend's own answer, not a proxy-invented status.
	var lastAnswer *bufferedAnswer
	// Errors from failed attempts, surfaced to the client via
	// X-Router-Fallback-Errors (preceding attempts only).
	var attemptErrors []string
	for i, attempt := range attempts {
		srv := attempt.server
		targetModel := attempt.targetModel

		// Skip unhealthy servers (except the last attempt — try it anyway)
		if r.health != nil && !r.health.IsHealthy(srv.ID) && i < len(attempts)-1 {
			log.Warnf("[%s] model=%q — skipping unhealthy server %s (attempt %d/%d)",
				req.URL.Path, model, srv.Name, i+1, len(attempts))
			continue
		}

		// Skip rate-limited servers (except the last attempt — try it anyway)
		if r.rateLimit != nil && r.rateLimit.ShouldSkip(srv.ID) && i < len(attempts)-1 {
			remaining := r.rateLimit.CooldownRemaining(srv.ID)
			log.Warnf("[%s] model=%q — skipping rate-limited server %s (cooldown %v, attempt %d/%d)",
				req.URL.Path, model, srv.Name, remaining.Round(time.Second), i+1, len(attempts))
			continue
		}

		// Skip attempts that declare a window too small for this request (see
		// contextOverflows); the last attempt is tried anyway.
		if i < len(attempts)-1 && contextOverflows(attempts[i].contextWindow, body) {
			log.Warnf("[%s] model=%q — skipping %s: declared context window %d < estimated prompt %d (attempt %d/%d)",
				req.URL.Path, model, srv.Name, attempts[i].contextWindow, len(body)/4, i+1, len(attempts))
			continue
		}

		// Skip quota-exceeded servers (TPM/RPM at limit; except the last attempt)
		if r.quota != nil && !r.quota.Allow(srv.ID) && i < len(attempts)-1 {
			tokens, reqs := r.quota.Usage(srv.ID)
			log.Warnf("[%s] model=%q — skipping quota-exceeded server %s (tpm=%d rpm=%d, attempt %d/%d)",
				req.URL.Path, model, srv.Name, tokens, reqs, i+1, len(attempts))
			continue
		}

		// The dialect decision is per attempt: a fallback server and a jev
		// candidate each declare their own capability, and a per-request header
		// outranks the config.
		forced := attempt.convert
		if headerConvert == "off" {
			forced = "off"
		}
		plan, planErr := convert.Decide(clientDialect, req.URL.Path, srv.Protocols, forced)
		if planErr != nil {
			// A backend this attempt can not speak to is a refusal, not a
			// transport failure: nothing was sent, so no health/rate-limit/quota
			// bookkeeping happens. Try the next attempt; if the last one refuses
			// too, the client gets our 400 naming what the backend serves.
			attemptErrors = append(attemptErrors, planErr.Error())
			if i < len(attempts)-1 {
				log.Warnf("[%s] model=%q — skipping %s: %v", req.URL.Path, model, srv.Name, planErr)
				continue
			}
			refusal = planErr
			break
		}
		// Assigned on every attempt so the recorded metric describes the attempt
		// that served, never an earlier failed one. stripStream is declared here for
		// the same reason: it is a per-attempt decision too (set in the injection
		// block below), and a later attempt must never inherit an earlier one's.
		meta.upstreamPath, meta.convert, meta.convertNotes, meta.translator = plan.Path, plan.Edge(), nil, nil
		stripStream := false

		rewrittenBody, err := rewriteAttemptBody(req, body, targetModel)
		if err != nil {
			log.Errorf("[%s] failed to rewrite model %q -> %q: %v", req.URL.Path, model, targetModel, err)
			http.Error(w, fmt.Sprintf("failed to rewrite model: %v", err), http.StatusInternalServerError)
			return
		}

		// A planned attempt must speak the backend's dialect in the body, not just
		// in the path: the plan moved the endpoint, this is what moves the request.
		// A refusal is the same class as the plan's own refusal — nothing was sent,
		// so no health/rate-limit/quota bookkeeping, the next attempt is tried, and
		// the last refusal is our 400 naming the field.
		if meta.convert != "" {
			converted, notes, convErr := convert.Request(plan, rewrittenBody,
				convert.OptionsFrom(r.store.GetSettings().Convert))
			if convErr != nil {
				attemptErrors = append(attemptErrors, convErr.Error())
				if i < len(attempts)-1 {
					log.Warnf("[%s] model=%q — skipping %s: %v", req.URL.Path, model, srv.Name, convErr)
					continue
				}
				refusal = convErr
				break
			}
			rewrittenBody = converted
			// A top-level field the spoke never decodes vanishes without a
			// per-field drop; the routed path reports the difference (the preview
			// path counts the same keys, from the same producer).
			meta.convertNotes = append(notes, convert.DroppedKeys(plan, body, converted)...)
		}

		// stream_options.include_usage is an OpenAI field: injected into the body in
		// the dialect this attempt speaks, after any conversion. stripStream hides
		// the injected usage chunk from the client (unless configured otherwise) —
		// except on a converted attempt, where the emitter IS the filter and has to
		// SEE the chunk: a chat backend reports its prompt tokens only there, and the
		// Anthropic events must carry them. A frame with no choices makes the emitter
		// emit nothing, so the client still never sees it. The gate reads this
		// attempt's plan, never the request-scoped copy.
		if plan.Dialect == convert.Chat && strings.Contains(plan.Path, "/chat/completions") {
			var injected bool
			rewrittenBody, injected = proxy.EnsureStreamUsage(rewrittenBody,
				r.store.GetSettings().AlwaysIncludeStreamUsage)
			stripStream = injected && plan.Edge() == ""
		}

		// TPM pre-call reservation: hold the declared output budget so
		// concurrent requests can't collectively exceed the server's TPM limit.
		// Released by releaseQuota on every exit path below. The last attempt
		// sends even if the reservation would exceed the limit (better to try
		// than to fail); it simply reserves nothing.
		estimate := proxy.DeclaredOutputBudget(rewrittenBody)
		reservedTokens := int64(0)
		if r.quota != nil && estimate > 0 {
			if r.quota.Reserve(srv.ID, estimate) {
				reservedTokens = estimate
			} else if i < len(attempts)-1 {
				log.Warnf("[%s] model=%q — skipping TPM-reserved server %s (attempt %d/%d)",
					req.URL.Path, model, srv.Name, i+1, len(attempts))
				continue
			}
		}

		// releaseQuota is this attempt's single reservation release: Complete is
		// the only release there is (the tracker keeps no TTL), so a path that
		// forgets it subtracts the request's declared budget from the server's
		// TPM headroom for the rest of the process's life.
		releaseQuota := func(used int64) {
			if r.quota != nil {
				r.quota.Complete(srv.ID, reservedTokens, used)
			}
		}

		wasFallback := i > 0

		log.Infof("[%s] model=%q -> %q on %s (attempt %d/%d)",
			req.URL.Path, model, targetModel, srv.Name, i+1, len(attempts))

		// chat, responses and legacy all ride the server's OpenAI URL; only
		// messages is Anthropic. A dialectless path (audio, count_tokens,
		// systemone) keeps the client's own protocol — the plan never moves it.
		upstreamType := apiType
		if plan.Dialect != "" {
			upstreamType = domain.APITypeOpenAI
			if plan.Dialect == convert.Messages {
				upstreamType = domain.APITypeAnthropic
			}
		}
		serverURL := srv.GetURLForAPIType(upstreamType)
		// On fallback, preserve the actual model used (don't rewrite back to
		// the original) so the client knows which model actually responded.
		responseModel := model
		if wasFallback || jev != nil {
			responseModel = targetModel
		}
		// Log the response rewrite parameters
		if responseModel != targetModel {
			log.Debugf("[%s] response rewrite: %q → %q", req.URL.Path, targetModel, responseModel)
		} else {
			log.Debugf("[%s] response rewrite: any model → %q (backend may return its actual model)", req.URL.Path, responseModel)
		}
		// Per-server keep-alive override (seconds), clamped to the same [1,300]s
		// bounds as the per-request header. 0 = use the global default.
		var srvKeepAlive time.Duration
		if srv.KeepAliveIdle > 0 {
			n := srv.KeepAliveIdle
			if n < 1 {
				n = 1
			} else if n > 300 {
				n = 300
			}
			srvKeepAlive = time.Duration(n) * time.Second
		}
		rh := &proxy.RouterHeaders{
			ServerID:       srv.ID,
			ServerName:     srv.Name,
			Attempts:       fmt.Sprintf("%d/%d", i+1, len(attempts)),
			FallbackErrors: attemptErrors,
			KeepAliveIdle:  srvKeepAlive,
			Proxy:          srv.ProxyURL(),
			UpstreamPath:   plan.Path,
			Convert:        meta.convert,
			ConvertNotes:   meta.convertNotes,
		}
		if meta.capture != nil {
			rh.Trace = meta.capture
		}
		// A converted attempt answers in the backend's dialect, so its response
		// must be translated back. Per attempt (the translator describes the
		// serving one), and only when the plan says anything changes.
		if meta.convert != "" {
			rh.Translate = newResponseTranslator(plan)
			meta.translator = rh.Translate
		}

		// Retry-before-fallback: on transport (pre-response) errors retry the
		// same server up to NumRetries times before moving to the next server.
		// Precedence: X-Router-Retries header > rule's num_retries.
		numRetries := rule.NumRetries
		explicitRetries := rule.NumRetries != 0
		if numRetries == 0 {
			// Rules without num_retries get a generous default: absorb a dying
			// backend (crashed worker, restart) before marking it unhealthy and
			// falling back. Disable per request with X-Router-Retries: 0.
			numRetries = defaultNumRetries
		}
		if n, ok := clientRetries(req.Header.Get("X-Router-Retries")); ok {
			numRetries = n
			explicitRetries = true
		}

		for retry := 0; ; retry++ {
			// The upstream client consumes req.Body on each attempt; rebuild it
			// so a retry sends a fresh body.
			req.Body = io.NopCloser(bytes.NewReader(rewrittenBody))
			req.ContentLength = int64(len(rewrittenBody))
			rh.Retries = retry
			meta.retries = retry

			pm, err := proxy.StreamProxy(req.Context(), serverURL, srv.APIKey, req, w, targetModel, responseModel, rh, stripStream, ping)
			if err == nil {
				// Success — mark healthy. Rate limiter is status-aware: only
				// successful (2xx) responses clear failures/cooldown; 4xx/5xx
				// responses count toward cooldown (429/401/403 immediately).
				if r.health != nil {
					r.health.MarkHealthy(srv.ID)
				}
				if r.rateLimit != nil {
					if pm.StatusCode >= 400 {
						// The backend's Retry-After (429) is the cooldown length
						// when it sent one — a busy server is benched for exactly
						// as long as it asked for, not for the 5min default.
						r.rateLimit.RecordFailure(srv.ID, pm.StatusCode,
							time.Duration(pm.RetryAfterMs)*time.Millisecond)
					} else {
						r.rateLimit.RecordSuccess(srv.ID)
					}
				}
				// Account quota usage (TPM/RPM) for successful responses: count
				// the request (RPM) and release the reservation, recording actual
				// usage (TPM).
				if r.quota != nil {
					r.quota.RecordRequest(srv.ID)
				}
				releaseQuota(int64(pm.TotalTokens))

				if pm.StatusCode >= 400 {
					log.Errorf("[%s] model=%q -> %q on %s returned HTTP %d %s: %s",
						req.URL.Path, model, targetModel, srv.Name, pm.StatusCode, http.StatusText(pm.StatusCode), pm.ErrorBody)
				}

				if pinKey != "" && pm.StatusCode < 400 {
					// The conversation's next turn belongs to the backend that
					// just cached its prefix.
					r.stickyPut(pinKey, srv.ID)
				}
				r.record(req, requestStart, model, targetModel, srv.ID, *pm, pm.StatusCode, pm.ErrorBody, wasFallback, apiType, jev, meta)
				return
			}

			// Client disconnect: stop immediately, no fallback needed. A
			// disconnect after the backend answered is an interrupted response —
			// the generation ran (and cost tokens), so record its telemetry
			// like a mid-stream error, with 499 "client closed request" (nginx
			// convention) to keep it distinct from served requests. A disconnect
			// before any backend response carries nothing to record.
			if req.Context().Err() != nil {
				var used int64
				if pm != nil {
					used = int64(pm.TotalTokens)
				}
				releaseQuota(used)
				log.Warnf("[%s] model=%q — client disconnected during proxy (attempt %d/%d)",
					req.URL.Path, model, i+1, len(attempts))
				if pm != nil && pm.StatusCode > 0 {
					r.record(req, requestStart, model, targetModel, srv.ID, *pm, 499, "client disconnected", wasFallback, apiType, jev, meta)
				}
				return
			}
			// Mid-stream error: headers already sent to client, retry/fallback impossible
			if _, isMidStream := err.(*proxy.MidStreamError); isMidStream {
				if r.health != nil {
					r.health.MarkUnhealthy(srv.ID)
				}
				if r.rateLimit != nil {
					r.rateLimit.RecordFailure(srv.ID, pm.StatusCode,
						time.Duration(pm.RetryAfterMs)*time.Millisecond)
				}
				releaseQuota(int64(pm.TotalTokens))
				log.Errorf("[%s] model=%q — mid-stream error on %s (attempt %d/%d): %v",
					req.URL.Path, model, srv.Name, i+1, len(attempts), err)
				// Record whatever metrics we have
				r.record(req, requestStart, model, targetModel, srv.ID, *pm, pm.StatusCode, pm.ErrorBody, wasFallback, apiType, jev, meta)
				return
			}

			// Transport error (pre-response), or a backend answer the proxy
			// buffered so it could be failed over (429/5xx): retry the same
			// server if the budget for that class of failure allows, otherwise
			// fall through to the next attempt.
			lastErr = err
			var buffered *proxy.UpstreamResponseError
			isBuffered := errors.As(err, &buffered)
			retriesLeft := sameServerRetries(err, numRetries)
			if isBuffered {
				retriesLeft = bufferedAnswerRetries(buffered.StatusCode, numRetries, explicitRetries)
				lastAnswer = &bufferedAnswer{resp: buffered, serverID: srv.ID,
					targetModel: targetModel, wasFallback: wasFallback, translator: meta.translator}
			}
			if retry < retriesLeft {
				log.Warnf("[%s] model=%q — retrying %s (%d/%d): %v",
					req.URL.Path, model, srv.Name, retry+1, numRetries, err)
				select {
				case <-time.After(retryBackoff(retry)):
				case <-req.Context().Done():
					// The client is gone and no answer was served: the
					// reservation is released exactly like every other exit.
					var used int64
					if pm != nil {
						used = int64(pm.TotalTokens)
					}
					releaseQuota(used)
					return
				}
				continue
			}
			// A backend that answered is reachable: the health tracker owns
			// "reachable", and the rate limiter (which is status-aware, and
			// cools a 429 down on its first occurrence) owns "keeps refusing us".
			if r.health != nil && !isBuffered {
				r.health.MarkUnhealthy(srv.ID)
			}
			if r.rateLimit != nil {
				if isBuffered {
					r.rateLimit.RecordFailure(srv.ID, buffered.StatusCode,
						time.Duration(pm.RetryAfterMs)*time.Millisecond)
				} else {
					r.rateLimit.RecordFailure(srv.ID, 0)
				}
			}
			var used int64
			if pm != nil {
				used = int64(pm.TotalTokens)
			}
			releaseQuota(used)
			attemptErrors = append(attemptErrors, err.Error())
			log.Errorf("[%s] fallback from %s: %v", req.URL.Path, srv.Name, err)
			break
		}
	}

	if refusal != nil {
		// Nothing was sent for the refusing attempt, so it served nothing: reset
		// the per-attempt plan fields (an earlier attempt that reached the proxy
		// filled them, and its eager headers sit on the same writer map) and
		// delete the convert headers. The refusal is reported the same way the
		// single-attempt refusal is — no conversion, no upstream endpoint — in
		// the headers and the metric alike.
		meta.upstreamPath, meta.convert, meta.convertNotes, meta.translator = "", "", nil, nil
		w.Header().Del("X-Router-Convert")
		w.Header().Del("X-Router-Convert-Notes")
		r.record(req, requestStart, model, attempts[0].targetModel, attempts[0].server.ID,
			proxy.ProxyMetrics{}, http.StatusBadRequest, refusal.Error(),
			len(attempts) > 1, apiType, jev, meta)
		log.Warnf("[%s] model=%q — refused: %v", req.URL.Path, model, refusal)
		http.Error(w, refusal.Error(), http.StatusBadRequest)
		return
	}

	// Nothing served, and at least one backend answered 429/5xx with a body the
	// proxy held back: the client's answer is that backend's own.
	if lastAnswer != nil {
		r.serveLastAnswer(w, req, requestStart, model, apiType, jev, meta, lastAnswer)
		return
	}

	errorBody := ""
	if lastErr != nil {
		errorBody = lastErr.Error()
	}
	r.record(req, requestStart, model, attempts[0].targetModel, attempts[0].server.ID, proxy.ProxyMetrics{}, http.StatusBadGateway, errorBody, len(attempts) > 1, apiType, jev, meta)

	log.Errorf("[%s] model=%q — all backends failed: %v", req.URL.Path, model, lastErr)
	http.Error(w, fmt.Sprintf("all backends failed: %v", lastErr), http.StatusBadGateway)
}

// serveLastAnswer delivers the last buffered backend answer (429/5xx, no byte of
// it sent to the client) verbatim: the status, the backend's headers and its
// body, restated in the client's dialect when that attempt converted. A
// proxy-invented 502 quoting the backends would bury the one fact the client can
// act on — "your prompt is over the context window" arrives as a 500 body from
// most engines, and a client that retries it elsewhere learns nothing.
func (r *Router) serveLastAnswer(w http.ResponseWriter, req *http.Request, requestStart time.Time,
	model string, apiType domain.APIType, jev *jevOutcome, meta *reqMeta, answer *bufferedAnswer) {
	resp := answer.resp
	body := resp.Body.Bytes()
	if answer.translator != nil {
		if translated, err := answer.translator.TranslateJSON(body); err == nil {
			body = translated
		} else {
			log.Warnf("[%s] model=%q — buffered error body stayed in the backend's dialect: %v",
				req.URL.Path, model, err)
		}
	}
	for k, values := range resp.Header {
		if k == "Content-Length" || k == "Transfer-Encoding" {
			continue // our own framing; the body may have been rewritten
		}
		if strings.HasPrefix(k, "X-Router-") {
			// The namespace describes the hop that wrote it. A chained
			// deployment's inner router stamps its own serving server, and that
			// must not outvote ours (the metric records the outer one).
			continue
		}
		w.Header().Del(k)
		for _, v := range values {
			w.Header().Add(k, v)
		}
	}
	w.Header().Set("X-Router-Server", answer.serverID)
	if srv, ok := r.store.GetServer(answer.serverID); ok {
		w.Header().Set("X-Router-Server-Name", srv.Name)
	}
	w.Header().Set("X-Router-Status", strconv.Itoa(resp.StatusCode))
	w.WriteHeader(resp.StatusCode)
	if _, err := w.Write(body); err != nil {
		log.Errorf("[%s] failed to write the backend's answer: %v", req.URL.Path, err)
	}
	r.record(req, requestStart, model, answer.targetModel, answer.serverID,
		proxy.ProxyMetrics{StatusCode: resp.StatusCode, ResponseSize: int64(len(body))},
		resp.StatusCode, string(body), answer.wasFallback, apiType, jev, meta)
	log.Errorf("[%s] model=%q — every attempt failed; served %s's HTTP %d",
		req.URL.Path, model, answer.serverID, resp.StatusCode)
}

// record stores a request metric; pm fields feed the usage/timing columns.
// jev is the auto-routing decision (nil on static rules).
func (r *Router) record(req *http.Request, requestStart time.Time, model, targetModel, serverID string, pm proxy.ProxyMetrics, statusCode int, errorBody string, wasFallback bool, apiType domain.APIType, jev *jevOutcome, meta *reqMeta) {
	m := domain.RequestMetric{
		Timestamp:             requestStart,
		RequestID:             meta.requestID,
		Model:                 model,
		TargetModel:           targetModel,
		BackendModel:          pm.BackendModel,
		ServerID:              serverID,
		StatusCode:            statusCode,
		ErrorBody:             errorBody,
		LatencyMs:             time.Since(requestStart).Milliseconds(),
		TTFBMs:                pm.TTFBMs,
		ResponseSize:          pm.ResponseSize,
		WasFallback:           wasFallback,
		PromptTokens:          pm.PromptTokens,
		CompletionTokens:      pm.CompletionTokens,
		TotalTokens:           pm.TotalTokens,
		CachedTokens:          pm.CachedTokens,
		ReasoningTokens:       pm.ReasoningTokens,
		CacheCreationTokens:   pm.CacheCreationTokens,
		DraftTokens:           pm.DraftTokens,
		DraftTokensAccepted:   pm.DraftTokensAccepted,
		NativePromptMs:        pm.PromptMs,
		NativePredictedMs:     pm.PredictedMs,
		NativePromptTokPerSec: pm.PromptPerSec,
		NativeDecodeTokPerSec: pm.TokensPerSec,
		NativeQueueMs:         pm.QueueMs,
		AudioDurationMs:       pm.AudioDurationMs,
		AudioRTF:              pm.AudioRTF,
		APIType:               apiType,
		APIEndpoint:           meta.class,
		UpstreamEndpoint:      meta.upstreamPath,
		Convert:               meta.convert,
		ConvertNotes:          meta.convertNotes,
		ClientIP:              clientIP(req),
	}
	m.Class, m.Features = meta.class, meta.features
	// What the response translation dropped joins what the request translation
	// dropped; the metric (and the tooltip it feeds) carries both, bounded.
	if meta.translator != nil {
		m.ConvertNotes = append(m.ConvertNotes, meta.translator.Notes()...)
	}
	m.ConvertNotes = proxy.CapConvertNotes(m.ConvertNotes)
	if wasFallback {
		m.Features = append(m.Features, "fallback")
	}
	if meta.retries > 0 {
		m.Features = append(m.Features, "retry:"+strconv.Itoa(meta.retries))
	}

	if jev != nil {
		m.JevChoice = jev.Choice
		m.JevConfidence = jev.Confidence
		m.JevLatencyMs = jev.LatencyMs
		m.JevModel = jev.Model
		m.JevReason = jev.Reason
		m.JevTokens = jev.InputTokens + jev.OutputTokens
		m.JevScores = jev.Scores
	}
	r.metrics.Add(m)

	if meta.capture != nil {
		e := meta.capture.Entry()
		e.StatusCode = statusCode
		e.LatencyMs = m.LatencyMs
		e.TTFBMs = m.TTFBMs
		e.ServerID = serverID
		e.TargetModel = targetModel
		e.Class = m.Class
		e.Features = m.Features
		if jev != nil {
			e.JevChoice = jev.Choice
		}
		// A body that never reached the writer chain (a stream-frame 5xx the
		// proxy peeked and returned as an error, a transport failure, a client
		// that hung up mid-stream) is recorded from the proxy's capture, so the
		// trace shows what the backend actually said.
		if e.Response == "" && errorBody != "" {
			e.Response = errorBody
		}
		r.trace.Add(e)
	}
}

// defaultNumRetries is the effective num_retries for rules that don't set one:
// a dying backend (crashed llama.cpp worker, restart) usually recovers within
// seconds, so retry before marking the server unhealthy and falling back.
// Override per request via X-Router-Retries ("0" disables retries).
const defaultNumRetries = 10

// maxClientRetries caps what a client may ask for via X-Router-Retries. The
// budget is a request for patience, not a licence to spend days: on the
// accepted-then-closed path every retry costs a dial plus up to 1s of backoff,
// so an unbounded number in the header is a self-inflicted outage per request.
const maxClientRetries = 100

// sameServerRetries caps the retry loop by what the transport actually reported,
// because the generous default (absorb a backend that is restarting) is the wrong
// answer to a backend that is not there:
//
//   - refused / unreachable: nothing is listening, and it will not start
//     listening inside one request. Zero retries — fall over to the next attempt
//     now instead of burning ~8s of backoff in front of a fallback that answers.
//
// Deliberately not capped for a connection accepted then closed (`ECONNRESET`,
// EOF): that is a backend mid-restart, which closes *every* pooled connection in
// turn, so the generous default is what recovers it. Everything else (dial and
// response timeouts, TLS, mid-body truncation) keeps the configured budget.
func sameServerRetries(err error, configured int) int {
	if errors.Is(err, syscall.ECONNREFUSED) || errors.Is(err, syscall.ECONNABORTED) ||
		errors.Is(err, syscall.EHOSTUNREACH) || errors.Is(err, syscall.ENETUNREACH) ||
		errors.Is(err, syscall.ENETDOWN) || errors.Is(err, syscall.EADDRNOTAVAIL) {
		return min(configured, 0)
	}
	return configured
}

// bufferedAnswer is a backend answer the proxy buffered instead of relaying
// (429/5xx — see proxy.UpstreamResponseError), kept with the identity of the
// attempt that produced it so the metric, the fallback marker and the response
// translation all describe the attempt that answered.
type bufferedAnswer struct {
	resp        *proxy.UpstreamResponseError
	serverID    string
	targetModel string
	wasFallback bool
	translator  proxy.StreamTranslator
}

// contextOverflows reports whether a request's estimated prompt can not fit the
// context window a backend declares. The estimate is the same chars/4 the jev
// state uses, over the WHOLE body (bytes/4) — system prompt, every turn, tools,
// the works, which is what the window has to hold. An undeclared window (0)
// gates nothing: an absent declaration is not a declaration of zero, so a config
// that never mentions windows routes exactly as it always did.
func contextOverflows(window int, body []byte) bool {
	return window > 0 && len(body)/4 > window
}

// bufferedAnswerRetries is the same-server retry budget for such an answer://
//   - 429: none. The backend stated in its own response that it is at its limit
//     (and for how long); nothing about that changes while we sleep between
//     tries, so the next attempt answers now instead of in 5s.
//   - 5xx: only what the operator asked for. The generous default exists to ride
//     out a backend that never answered at all; eleven dials into a stable 500
//     are pure latency in front of a fallback that answers.
func bufferedAnswerRetries(status, configured int, explicit bool) int {
	if status == http.StatusTooManyRequests {
		return 0
	}
	if explicit {
		return configured
	}
	return 0
}

// newRequestID is a 16-hex-char correlation id (8 bytes of crypto/rand).
func newRequestID() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return ""
	}
	return hex.EncodeToString(b[:])
}

// retryBackoff returns the delay before retry attempt n (0-based): 100ms, 250ms,
// 500ms, then capped at 1s. Keeps retries fast but avoids hammering a failing
// backend.
// clientRetries parses the per-request retry override: a non-negative integer,
// capped at maxClientRetries. ok = false for anything the header can not mean
// (absent, unparseable, negative), which leaves the rule's own budget in place.
func clientRetries(h string) (int, bool) {
	if h == "" {
		return 0, false
	}
	n, err := strconv.Atoi(h)
	if err != nil || n < 0 {
		return 0, false
	}
	return min(n, maxClientRetries), true
}

func retryBackoff(n int) time.Duration {
	switch {
	case n <= 0:
		return 100 * time.Millisecond
	case n == 1:
		return 250 * time.Millisecond
	case n == 2:
		return 500 * time.Millisecond
	default:
		return time.Second
	}
}

// listModels returns the list of incoming model names (what clients can request).
func (r *Router) listModels(w http.ResponseWriter, req *http.Request) {
	rules := r.store.GetActiveRules()

	models := make([]map[string]any, 0)
	for _, rule := range rules {
		if !rule.Enabled {
			continue
		}
		for _, name := range rule.IncomingModels {
			// A wildcard entry is a matching rule, not a name a client can send,
			// so it is not listed.
			if domain.IsModelPattern(name) {
				continue
			}
			m := map[string]any{
				"id":       name,
				"object":   "model",
				"created":  time.Now().Unix(),
				"owned_by": "router",
				"target":   rule.TargetModel,
				"server":   rule.ServerID,
			}
			// Clients (Claude Code etc.) read context_window to size prompts.
			// Omitted entirely when unknown (malformed/0 → absent, per LiteLLM).
			// A jev rule without its own value reports the pool minimum.
			cw := rule.ContextWindow
			if rule.IsJev() && cw == 0 {
				for _, c := range rule.Jev.EnabledCandidates() {
					if c.ContextWindow > 0 && (cw == 0 || c.ContextWindow < cw) {
						cw = c.ContextWindow
					}
				}
			}
			if cw > 0 {
				m["context_window"] = cw
			}
			// A jev rule has no single target: advertise the pool it picks from.
			if rule.IsJev() {
				m["router"] = domain.RouterJev
				pool := make([]string, 0, len(rule.Jev.Candidates))
				for _, c := range rule.Jev.EnabledCandidates() {
					if c.TargetModel != "" {
						pool = append(pool, c.TargetModel)
					}
				}
				m["models"] = pool
			}
			// Health is a signal, not a filter: the listing stays the config view
			// of what this router accepts (a model whose backends are down is
			// still routed to — the last attempt always tries), so a client that
			// shows a picker can grey the name out instead of being handed a
			// listing that changes under it every 30s.
			if r.health != nil {
				m["healthy"] = r.anyServerHealthy(rule)
			}
			models = append(models, m)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data":   models,
	}); err != nil {
		log.Errorf("[%s] failed to encode /v1/models response: %v", req.URL.Path, err)
	}
}

// anyServerHealthy reports whether at least one server that can serve the rule
// is up: the jev pool for an auto-routing rule, the primary and its enabled
// fallbacks for a static one.
func (r *Router) anyServerHealthy(rule *domain.RoutingRule) bool {
	ids := []string{rule.ServerID}
	if rule.IsJev() {
		// The decision server answers questions; the pool serves the request.
		ids = ids[:0]
		for _, c := range rule.Jev.EnabledCandidates() {
			ids = append(ids, c.ServerID)
		}
	} else {
		for _, f := range rule.Fallbacks {
			if f.IsEnabled() {
				ids = append(ids, f.ServerID)
			}
		}
	}
	for _, id := range ids {
		if r.health.IsHealthy(id) {
			return true
		}
	}
	return false
}

// rewriteAttemptBody points the request body at the attempt's target model.
// JSON bodies are re-marshalled; multipart uploads (audio uploads) keep every
// byte but their `model` form field; GETs have no body at all.
//
// An empty target is a pass-through rule (a wildcard rule owning a whole model
// family, a rule that only moves the server): the client's own model name IS the
// backend's, so the body is left exactly as it arrived.
func rewriteAttemptBody(req *http.Request, body []byte, targetModel string) ([]byte, error) {
	switch {
	case req.Method == http.MethodGet, targetModel == "":
		return body, nil
	case strings.HasPrefix(req.Header.Get("Content-Type"), "multipart/"):
		return proxy.RewriteModelInMultipart(body, targetModel)
	default:
		return proxy.RewriteModelInBody(body, targetModel)
	}
}

// extractModel reads the top-level "model" field from a JSON body using byte-level
// scanning. Tracks object depth and string boundaries so "model" keys nested
// inside other objects or quoted string content (e.g. a prompt discussing model
// names) are not mistaken for the request model. Avoids full JSON unmarshal
// allocation for a single field extraction.
func extractModel(body []byte) (string, error) {
	key := []byte(`"model"`)
	keyLen := len(key)
	depth := 0

	for i := 0; i < len(body); {
		switch body[i] {
		case '{':
			depth++
			i++
			continue
		case '}':
			depth--
			i++
			continue
		case '"':
			if depth == 1 && i+keyLen <= len(body) && bytes.Equal(body[i:i+keyLen], key) {
				endOfKey := i + keyLen
				// Skip longer keys like "model_id" / "model_name".
				if endOfKey == len(body) || !proxy.IsJSONNameChar(body[endOfKey]) {
					model, found, err := parseModelValue(body, endOfKey)
					if err != nil {
						return "", err
					}
					if found {
						return model, nil
					}
				}
			}
			i = skipString(body, i)
			continue
		default:
			i++
		}
	}

	return "", fmt.Errorf("missing 'model' field")
}

// parseModelValue extracts the string value of a "model" key, starting just past
// the key's closing quote. found is false when the value is not a string (the
// caller keeps scanning); an error is returned only for an unterminated value.
func parseModelValue(body []byte, afterKey int) (string, bool, error) {
	colonIdx := bytes.IndexByte(body[afterKey:], ':')
	if colonIdx < 0 {
		return "", false, nil
	}
	valueStart := afterKey + colonIdx + 1
	for valueStart < len(body) && (body[valueStart] == ' ' || body[valueStart] == '\t' || body[valueStart] == '\n' || body[valueStart] == '\r') {
		valueStart++
	}
	if valueStart >= len(body) || body[valueStart] != '"' {
		return "", false, nil
	}
	strStart := valueStart + 1
	strEnd := proxy.FindJSONStringEnd(body, strStart)
	if strEnd < 0 {
		return "", false, fmt.Errorf("invalid model value")
	}
	return strings.TrimSpace(string(body[strStart:strEnd])), true, nil
}

// skipString returns the index just past the closing quote of the JSON string
// starting at i (body[i] == '"'). Handles backslash escapes; returns len(body)
// for an unterminated string.
func skipString(body []byte, i int) int {
	i++ // skip opening quote
	for i < len(body) {
		switch body[i] {
		case '\\':
			i += 2
		case '"':
			return i + 1
		default:
			i++
		}
	}
	return len(body)
}
