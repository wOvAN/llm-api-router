package router

import (
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"llm-api-router/domain"
)

// A rule with `balance: "round-robin"` is a pool of interchangeable backends
// (two GPUs, two replicas of one model): successive requests start on a
// different member instead of hammering the primary until it fails. The rest of
// the list stays the failover chain, so a pool member that is down costs one
// rotation, not a request.
func TestHandleRoundRobinBalancesAcrossReplicas(t *testing.T) {
	var a, b atomic.Int32
	handler := func(counter *atomic.Int32) http.HandlerFunc {
		return func(w http.ResponseWriter, req *http.Request) {
			if answerProbe(w, req) {
				return
			}
			counter.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`))
		}
	}
	first := httptest.NewServer(handler(&a))
	defer first.Close()
	second := httptest.NewServer(handler(&b))
	defer second.Close()

	r, store, ms, _ := newTestRouter(t)
	for _, s := range []*domain.Server{
		{ID: "a", Name: "A", URL: first.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
		{ID: "b", Name: "B", URL: second.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
	} {
		if err := store.AddServer(s); err != nil {
			t.Fatalf("AddServer: %v", err)
		}
	}
	if err := store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "a",
		Fallbacks:      []domain.FallbackEntry{{ServerID: "b"}},
		Balance:        domain.BalanceRoundRobin,
		Enabled:        true,
	}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}

	for i := range 4 {
		// Distinct first turns: four separate conversations, so nothing pins and
		// the rotation is what spreads them (a conversation that repeats itself
		// is a pin, pinned by TestHandleRoundRobinPinsConversationToMember).
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
			strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"turn `+strconv.Itoa(i)+`"}]}`))
		w := httptest.NewRecorder()
		r.Handle(w, req)
		if got := w.Result().StatusCode; got != http.StatusOK {
			t.Fatalf("status = %d, want 200: %s", got, w.Body.String())
		}
	}
	if a.Load() != 2 || b.Load() != 2 {
		t.Errorf("hits = a %d, b %d, want 2 and 2", a.Load(), b.Load())
	}
	// A rotation is not a failover: the request the pool served on its own
	// starting slot must not count as a fallback in the metrics.
	for _, m := range ms.Recent() {
		if m.WasFallback {
			t.Errorf("request %s recorded as a fallback, want a served pool slot", m.ServerID)
		}
	}
}

// Weights are the point of balancing a pool of unequal replicas (a 4-GPU box
// beside a 1-GPU one): the start slot is dealt out in proportion, so traffic
// lands 1:3 without an operator counting anything.
func TestHandleRoundRobinHonoursWeights(t *testing.T) {
	var a, b atomic.Int32
	handler := func(counter *atomic.Int32) http.HandlerFunc {
		return func(w http.ResponseWriter, req *http.Request) {
			if answerProbe(w, req) {
				return
			}
			counter.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`))
		}
	}
	first := httptest.NewServer(handler(&a))
	defer first.Close()
	second := httptest.NewServer(handler(&b))
	defer second.Close()

	r, store, _, _ := newTestRouter(t)
	for _, s := range []*domain.Server{
		{ID: "a", Name: "A", URL: first.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
		{ID: "b", Name: "B", URL: second.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
	} {
		if err := store.AddServer(s); err != nil {
			t.Fatalf("AddServer: %v", err)
		}
	}
	if err := store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "a",
		Fallbacks:      []domain.FallbackEntry{{ServerID: "b", Weight: 3}},
		Balance:        domain.BalanceRoundRobin,
		Enabled:        true,
	}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}

	for i := range 4 {
		// Distinct conversations — the deal-out is the rotation's, not a pin's.
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
			strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"turn `+strconv.Itoa(i)+`"}]}`))
		w := httptest.NewRecorder()
		r.Handle(w, req)
	}
	if a.Load() != 1 || b.Load() != 3 {
		t.Errorf("hits = a %d, b %d, want 1 and 3 (weights 1:3)", a.Load(), b.Load())
	}
}

// A rule without `balance` keeps the historical contract exactly: the primary
// takes every request it can serve and a fallback is a rescue, not a share.
func TestHandleBalanceOffKeepsFailoverOrder(t *testing.T) {
	var a, b atomic.Int32
	handler := func(counter *atomic.Int32) http.HandlerFunc {
		return func(w http.ResponseWriter, req *http.Request) {
			if answerProbe(w, req) {
				return
			}
			counter.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`))
		}
	}
	first := httptest.NewServer(handler(&a))
	defer first.Close()
	second := httptest.NewServer(handler(&b))
	defer second.Close()

	r, store, _, _ := newTestRouter(t)
	for _, s := range []*domain.Server{
		{ID: "a", Name: "A", URL: first.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
		{ID: "b", Name: "B", URL: second.URL, APITypes: []domain.APIType{domain.APITypeOpenAI}},
	} {
		if err := store.AddServer(s); err != nil {
			t.Fatalf("AddServer: %v", err)
		}
	}
	if err := store.AddRule(&domain.RoutingRule{
		IncomingModels: []string{"gpt-4"},
		TargetModel:    "gpt-4",
		ServerID:       "a",
		Fallbacks:      []domain.FallbackEntry{{ServerID: "b"}},
		Enabled:        true,
	}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}

	for range 4 {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
			strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`))
		w := httptest.NewRecorder()
		r.Handle(w, req)
	}
	if a.Load() != 4 || b.Load() != 0 {
		t.Errorf("hits = a %d, b %d, want 4 and 0 (no balancing requested)", a.Load(), b.Load())
	}
}

// Config churn is how a long-lived router grows its rotation map: every pool
// composition the process has ever balanced leaves a counter behind, and a GUI
// operator edits rules on purpose. The sweep keeps what the active profile can
// build right now.
func TestHandleBalanceRotationSweepsDeadPools(t *testing.T) {
	handler := func(w http.ResponseWriter, req *http.Request) {
		if answerProbe(w, req) {
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`))
	}
	servers := map[string]string{}
	for _, name := range []string{"a", "b", "c"} {
		srv := httptest.NewServer(http.HandlerFunc(handler))
		t.Cleanup(srv.Close)
		servers[name] = srv.URL
	}

	r, store, _, _ := newTestRouter(t)
	for id, url := range servers {
		if err := store.AddServer(&domain.Server{ID: id, Name: id, URL: url,
			APITypes: []domain.APIType{domain.APITypeOpenAI}}); err != nil {
			t.Fatalf("AddServer: %v", err)
		}
	}

	original := rotationSweepAt
	t.Cleanup(func() { rotationSweepAt = original })
	rotationSweepAt = 2

	// Three pools in a row: the first two compositions stop existing when their
	// rule is replaced, so their counters are dead weight by the third request.
	pools := [][]string{{"a", "b"}, {"a", "c"}, {"b", "c"}}
	for i, pool := range pools {
		if i > 0 {
			if err := store.DeleteRuleFromProfile("", 0); err != nil {
				t.Fatalf("DeleteRuleFromProfile: %v", err)
			}
		}
		rule := &domain.RoutingRule{IncomingModels: []string{"gpt-4"},
			TargetModel: "gpt-4", ServerID: pool[0], Balance: domain.BalanceRoundRobin,
			Enabled: true}
		for _, id := range pool[1:] {
			rule.Fallbacks = append(rule.Fallbacks, domain.FallbackEntry{ServerID: id})
		}
		if err := store.AddRule(rule); err != nil {
			t.Fatalf("AddRule: %v", err)
		}
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
			strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"turn `+
				strconv.Itoa(i)+`"}]}`))
		w := httptest.NewRecorder()
		r.Handle(w, req)
		if got := w.Result().StatusCode; got != http.StatusOK {
			t.Fatalf("pool %v: status = %d: %s", pool, got, w.Body.String())
		}
	}

	var keys []string
	r.rotations.Range(func(k, _ any) bool { keys = append(keys, k.(string)); return true })
	if len(keys) != 1 || keys[0] != "b,c" {
		t.Errorf("rotation keys = %v, want only the live pool b,c", keys)
	}
}

// A pool whose replicas are not equally fast wants the idle one, not the next
// slot: a five-minute generation occupies its backend for its whole length, so a
// rotation that lands the next request on that replica sends it to the tail of a
// queue. `balance: "least-connections"` orders the pool by requests in flight,
// fewest first (LiteLLM's `least-busy`, olla's `least-connections`).
func TestBalanceLeastConnectionsPrefersIdleMember(t *testing.T) {
	r, _, _, _ := newTestRouter(t)
	a := serverAttempt{server: &domain.Server{ID: "a"}}
	b := serverAttempt{server: &domain.Server{ID: "b"}}

	if got := r.balanceLeast([]serverAttempt{a, b}, nil, ""); got[0].server.ID != "a" {
		t.Errorf("idle pool ordered %v-first, want the declared order kept", got[0].server.ID)
	}
	r.countInflight("a", 2)
	got := r.balanceLeast([]serverAttempt{a, b}, nil, "")
	if got[0].server.ID != "b" {
		t.Errorf("busy pool ordered %v-first, want the idle member b", got[0].server.ID)
	}
	// A conversation pinned to the busy member keeps its prompt cache: the pin
	// outranks the load signal, exactly as it outranks the rotation.
	if got := r.balanceLeast([]serverAttempt{a, b}, nil, "a"); got[0].server.ID != "a" {
		t.Error("the pinned member was ordered behind the idle one")
	}
}

// A pool of replicas that are not the same size (an 80GB box beside a 24GB one,
// so 8 slots beside 2) has no useful "least busy" in raw request counts: the
// big box can hold four times as many generations at the same utilisation. The
// weight the round-robin pool already deals its slots by is that capacity
// number, so least-connections ranks by load per slot.
func TestBalanceLeastConnectionsHonoursWeights(t *testing.T) {
	r, _, _, _ := newTestRouter(t)
	big := serverAttempt{server: &domain.Server{ID: "big"}}
	small := serverAttempt{server: &domain.Server{ID: "small"}}
	r.countInflight("big", 2)   // 2 of 4 slots
	r.countInflight("small", 1) // 1 of 1 slot — full

	if got := r.balanceLeast([]serverAttempt{big, small}, []int{4, 1}, ""); got[0].server.ID != "big" {
		t.Errorf("ordered %v-first, want the member at half its capacity ahead of the one at full",
			got[0].server.ID)
	}
	// With no weights declared, the mode is exactly the count comparison above.
	if got := r.balanceLeast([]serverAttempt{big, small}, []int{1, 1}, ""); got[0].server.ID != "small" {
		t.Errorf("unweighted pool ordered %v-first, want the fewest requests in flight", got[0].server.ID)
	}
}

// The signal above has to be real: a request must occupy its backend's slot for
// exactly the duration of the attempt — one inside the proxied request, zero once
// it is served — or least-connections routes on a constant and a leak quietly
// pins every request to whichever replica was least often busy.
func TestHandleInflightCountsAnAttempt(t *testing.T) {
	var r *Router
	during := make(chan int32, 1)
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if answerProbe(w, req) {
			return
		}
		during <- r.inflightCount("a")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`))
	}))
	defer backend.Close()

	router, store, _, _ := newTestRouter(t)
	r = router
	if err := store.AddServer(&domain.Server{ID: "a", Name: "A", URL: backend.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"gpt-4"},
		TargetModel: "gpt-4", ServerID: "a", Balance: domain.BalanceLeastConnections,
		Enabled: true}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	r.Handle(w, req)
	if got := w.Result().StatusCode; got != http.StatusOK {
		t.Fatalf("status = %d: %s", got, w.Body.String())
	}

	select {
	case got := <-during:
		if got != 1 {
			t.Errorf("in-flight during the attempt = %d, want 1", got)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the backend never saw the request")
	}
	if got := r.inflightCount("a"); got != 0 {
		t.Errorf("in-flight after a served request = %d, want 0 (the slot leaked)", got)
	}
}

// A backend with N slots answers the N+1-th request by queueing it, not by
// refusing it, and queue time is invisible to the caller. `max_parallel_requests`
// lets the operator state the slot count so the router sends the overflow to the
// member that has room instead.
func TestHandleSkipsServerAtParallelCap(t *testing.T) {
	release := make(chan struct{})
	busy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if answerProbe(w, req) {
			return
		}
		<-release
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`))
	}))
	defer busy.Close()
	free := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if answerProbe(w, req) {
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"gpt-4","choices":[{"finish_reason":"stop"}]}`))
	}))
	defer free.Close()

	router, store, _, _ := newTestRouter(t)
	if err := store.AddServer(&domain.Server{ID: "busy", Name: "busy", URL: busy.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI}, MaxParallelRequests: 1}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddServer(&domain.Server{ID: "free", Name: "free", URL: free.URL,
		APITypes: []domain.APIType{domain.APITypeOpenAI}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}
	if err := store.AddRule(&domain.RoutingRule{IncomingModels: []string{"gpt-4"},
		ServerID: "busy", TargetModel: "gpt-4", NumRetries: -1, Enabled: true,
		Fallbacks: []domain.FallbackEntry{{ServerID: "free"}}}); err != nil {
		t.Fatalf("AddRule: %v", err)
	}

	body := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`
	firstDone := make(chan struct{})
	go func() {
		defer close(firstDone)
		w := httptest.NewRecorder()
		router.Handle(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	}()

	deadline := time.Now().Add(5 * time.Second)
	for router.inflightCount("busy") == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if got := router.inflightCount("busy"); got != 1 {
		t.Fatalf("in-flight on busy = %d, want 1 (the first request never reached it)", got)
	}

	w := httptest.NewRecorder()
	router.Handle(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if got := w.Result().Header.Get("X-Router-Server"); got != "free" {
		t.Errorf("server = %q, want free: a server at its parallel cap must be skipped, not queued on",
			w.Result().Header.Get("X-Router-Fallback-Errors"))
	}

	close(release)
	<-firstDone
	if got := router.inflightCount("busy"); got != 0 {
		t.Errorf("in-flight after both finished = %d, want 0 (the slot leaked)", got)
	}
}
