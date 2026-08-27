package proxy

import (
	"math/rand"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// TestModelRewriteChainByteExactAcrossAllSplits is the production-corruption
// guard: the client observed a duplicated tail inside one SSE payload
// (created/id/model written twice, mid-chunk). The writer chain may only ever
// replace model values — no byte may be dropped or duplicated, whatever the
// upstream segmentation. This feeds a multi-frame stream through the real
// chain (modelRewriteWriter → metricsWriter → sseRelay) split at every
// possible byte boundary (plus random multi-splits) and requires the client
// bytes to be exactly the expected rewritten stream.
func TestModelRewriteChainByteExactAcrossAllSplits(t *testing.T) {
	const oldModel = "Jackrong/Qwopus3.6-27B-v2-MTP-GGUF:BF16"
	const newModel = "opus"
	stream := "data: {\"choices\":[{\"finish_reason\":null,\"index\":0,\"delta\":{\"reasoning_content\":\" \\\"\"}}],\"created\":1787779279,\"id\":\"chatcmpl-LpRY5kI0CrdKAa2GAXtLRWcsu9i2BGMQ\",\"model\":\"" + oldModel + "\",\"system_fingerprint\":\"b10639-5e6a37cb1\",\"object\":\"chat.completion.chunk\"}\n\n" +
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}],\"created\":1787779279,\"id\":\"chatcmpl-LpRY5kI0CrdKAa2GAXtLRWcsu9i2BGMQ\",\"model\":\"" + oldModel + "\",\"system_fingerprint\":\"b10639-5e6a37cb1\",\"object\":\"chat.completion.chunk\"}\n\n" +
		"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n" +
		"data: [DONE]\n\n"
	want := strings.ReplaceAll(stream, `"model":"`+oldModel+`"`, `"model":"`+newModel+`"`)

	run := func(old, new string, parts ...string) string {
		rec := httptest.NewRecorder()
		relay := newSSERelay(rec, false)
		mw := newMetricsWriter(relay, time.Now())
		mrw := newModelRewriteWriter(mw, old, new)
		for _, c := range parts {
			if _, err := mrw.Write([]byte(c)); err != nil {
				t.Fatalf("Write: %v", err)
			}
		}
		mrw.Finish()
		relay.finish()
		return rec.Body.String()
	}

	// Whole stream, then every single byte-boundary split.
	if got := run(oldModel, newModel, stream); got != want {
		t.Fatalf("unsplit result wrong:\n got: %s\nwant: %s", got, want)
	}
	for i := 1; i < len(stream); i++ {
		if got := run(oldModel, newModel, stream[:i], stream[i:]); got != want {
			t.Fatalf("split at %d changed the result:\n got: %s\nwant: %s", i, got, want)
		}
	}

	// Random multi-splits (bounded carry across several reads).
	rnd := rand.New(rand.NewSource(1))
	for trial := 0; trial < 200; trial++ {
		var parts []string
		for i := 0; i < len(stream); {
			n := 1 + rnd.Intn(40)
			if i+n > len(stream) {
				n = len(stream) - i
			}
			parts = append(parts, stream[i:i+n])
			i += n
		}
		if got := run(oldModel, newModel, parts...); got != want {
			t.Fatalf("random split %d changed the result:\n got: %s\nwant: %s", trial, got, want)
		}
	}

	// oldModel == newModel path (replaceAnyModelValue) must be byte-exact too.
	const sameModel = "opus"
	for i := 1; i < len(stream); i += 7 {
		if got := run(sameModel, sameModel, stream[:i], stream[i:]); got != want {
			t.Fatalf("any-value mode, split at %d changed the result:\n got: %s\nwant: %s", i, got, want)
		}
	}
}
