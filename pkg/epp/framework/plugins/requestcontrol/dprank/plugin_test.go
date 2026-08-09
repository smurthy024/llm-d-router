package dprank

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func newTestPlugin(t *testing.T, params Parameters) *Plugin {
	t.Helper()
	p, err := New("dp", params)
	require.NoError(t, err)
	return p
}

func f64(v float64) *float64 { return &v }

func completionsReq(model, prompt string) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID:   "req",
		TargetModel: model,
		Headers:     map[string]string{},
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: prompt}},
		},
	}
}

func chatReq(model, text string) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		TargetModel: model,
		Headers:     map[string]string{},
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{
					{Role: "user", Content: fwkrh.Content{Raw: text}},
				},
			},
		},
	}
}

// route runs PreRequest and returns the assigned rank, or -1 if no header was set.
func route(t *testing.T, p *Plugin, req *fwksched.InferenceRequest) int {
	t.Helper()
	_ = p.PreRequest(context.Background(), req, nil)
	v, ok := req.Headers[DPRankHeader]
	if !ok {
		return -1
	}
	var rank int
	_, err := fmt.Sscanf(v, "%d", &rank)
	require.NoError(t, err)
	return rank
}

func longPrompt(seed string, blocks int) string {
	var sb strings.Builder
	for i := 0; i < blocks*defaultBlockSizeChars/len(seed)+1; i++ {
		sb.WriteString(seed)
	}
	return sb.String()
}

// --- factory / config ---------------------------------------------------

func TestFactory_Defaults(t *testing.T) {
	dec := fwkplugin.StrictDecoder(json.RawMessage(`{"dpSize": 8}`))
	pl, err := Factory("dp", dec, nil)
	require.NoError(t, err)

	p, ok := pl.(*Plugin)
	require.True(t, ok)
	assert.Equal(t, fwkplugin.TypedName{Type: DPRankPluginType, Name: "dp"}, p.TypedName())
	assert.Equal(t, 8, p.dpSize)
	assert.Equal(t, defaultBlockSizeChars, p.blockSizeChars)
	assert.Equal(t, defaultMaxPrefixBlocks, p.maxPrefixBlocks)
	assert.InDelta(t, defaultLoadWeight, p.loadWeight, 1e-9)
	assert.Equal(t, 10*time.Second, p.halfLife)
	assert.Equal(t, defaultMaxIndexEntries, p.index.maxEntry)
}

func TestFactory_Errors(t *testing.T) {
	cases := map[string]string{
		"missing dpSize":  `{}`,
		"zero dpSize":     `{"dpSize": 0}`,
		"negative dpSize": `{"dpSize": -3}`,
		"dpSize over 64":  `{"dpSize": 65}`,
		"negative weight": `{"dpSize": 4, "loadWeight": -0.1}`,
		"zero half life":  `{"dpSize": 4, "halfLifeSeconds": 0.0}`,
		"unknown field":   `{"dpSize": 4, "bogus": 1}`,
		"bad json":        `{"dpSize": "four"}`,
	}
	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := Factory("dp", fwkplugin.StrictDecoder(json.RawMessage(raw)), nil)
			assert.Error(t, err)
		})
	}

	// nil decoder means no parameters at all, and dpSize is required.
	_, err := Factory("dp", nil, nil)
	assert.Error(t, err)
}

// --- affinity -----------------------------------------------------------

func TestSamePrefixIsSticky(t *testing.T) {
	p := newTestPlugin(t, Parameters{DPSize: 8})
	prompt := longPrompt("hello-world-prefix-", 4)

	first := route(t, p, completionsReq("m", prompt))
	require.GreaterOrEqual(t, first, 0)

	for i := 0; i < 20; i++ {
		got := route(t, p, completionsReq("m", prompt))
		require.Equal(t, first, got, "iteration %d", i)
	}
}

func TestZeroLoadWeightPureAffinity(t *testing.T) {
	p := newTestPlugin(t, Parameters{DPSize: 8, LoadWeight: f64(0)})
	hot := longPrompt("hot-prefix-", 4)

	first := route(t, p, completionsReq("m", hot))
	for i := 0; i < 50; i++ {
		require.Equal(t, first, route(t, p, completionsReq("m", hot)))
	}
}

func TestDifferentPrefixesSpread(t *testing.T) {
	p := newTestPlugin(t, Parameters{DPSize: 4})

	seen := map[int]int{}
	for i := 0; i < 200; i++ {
		prompt := longPrompt(fmt.Sprintf("distinct-prefix-%d-", i), 2)
		r := route(t, p, completionsReq("m", prompt))
		require.GreaterOrEqual(t, r, 0)
		seen[r]++
	}
	assert.Greater(t, len(seen), 1, "distribution is degenerate: %v", seen)
}

func TestDifferentModelsAreDistinctPrefixes(t *testing.T) {
	p := newTestPlugin(t, Parameters{DPSize: 4})
	prompt := longPrompt("shared-text-", 2)

	// The model name is folded into the first block's hash, so the same text
	// under two models must not share index entries.
	h1 := p.chainHashes("model-a", prompt)
	h2 := p.chainHashes("model-b", prompt)
	require.Len(t, h1, len(h2))
	assert.NotEqual(t, h1[0], h2[0])
}

func TestChainHashIsCumulative(t *testing.T) {
	p := newTestPlugin(t, Parameters{DPSize: 4, BlockSizeChars: 4})

	// Shared 8-char prefix, divergent tail.
	a := p.chainHashes("m", "AAAABBBBCCCC")
	b := p.chainHashes("m", "AAAABBBBDDDD")
	require.Len(t, a, 3)
	require.Len(t, b, 3)
	assert.Equal(t, a[0], b[0])
	assert.Equal(t, a[1], b[1])
	assert.NotEqual(t, a[2], b[2])
}

func TestMaxPrefixBlocksIsRespected(t *testing.T) {
	p := newTestPlugin(t, Parameters{DPSize: 4, BlockSizeChars: 4, MaxPrefixBlocks: 3})
	h := p.chainHashes("m", strings.Repeat("x", 400))
	assert.Len(t, h, 3)
}

// --- load override ------------------------------------------------------

func TestHighLoadWeightSpreadsHotPrefix(t *testing.T) {
	p := newTestPlugin(t, Parameters{DPSize: 4, LoadWeight: f64(5)})
	hot := longPrompt("very-hot-prefix-", 4)

	seen := map[int]int{}
	for i := 0; i < 40; i++ {
		r := route(t, p, completionsReq("m", hot))
		require.GreaterOrEqual(t, r, 0)
		seen[r]++
	}
	assert.Greater(t, len(seen), 1, "load weight failed to override affinity: %v", seen)
}

// --- safety -------------------------------------------------------------

func TestNoPanicAndNoHeaderOnUnusableInput(t *testing.T) {
	p := newTestPlugin(t, Parameters{DPSize: 4})
	ctx := context.Background()

	t.Run("nil request", func(t *testing.T) {
		assert.NotPanics(t, func() { p.PreRequest(ctx, nil, nil) })
	})

	t.Run("nil plugin receiver", func(t *testing.T) {
		var np *Plugin
		assert.NotPanics(t, func() { np.PreRequest(ctx, completionsReq("m", "hi"), nil) })
	})

	t.Run("nil body", func(t *testing.T) {
		req := &fwksched.InferenceRequest{Headers: map[string]string{}}
		assert.NotPanics(t, func() { p.PreRequest(ctx, req, nil) })
		assert.NotContains(t, req.Headers, DPRankHeader)
	})

	t.Run("empty body struct", func(t *testing.T) {
		req := &fwksched.InferenceRequest{Headers: map[string]string{}, Body: &fwkrh.InferenceRequestBody{}}
		assert.NotPanics(t, func() { p.PreRequest(ctx, req, nil) })
		assert.NotContains(t, req.Headers, DPRankHeader)
	})

	t.Run("nil headers map", func(t *testing.T) {
		req := completionsReq("m", "some prompt")
		req.Headers = nil
		assert.NotPanics(t, func() { p.PreRequest(ctx, req, nil) })
		assert.Nil(t, req.Headers)
	})

	t.Run("empty prompt", func(t *testing.T) {
		req := completionsReq("m", "")
		assert.NotPanics(t, func() { p.PreRequest(ctx, req, nil) })
		assert.NotContains(t, req.Headers, DPRankHeader)
	})

	t.Run("empty chat messages", func(t *testing.T) {
		req := &fwksched.InferenceRequest{
			Headers: map[string]string{},
			Body: &fwkrh.InferenceRequestBody{
				ChatCompletions: &fwkrh.ChatCompletionsRequest{},
			},
		}
		assert.NotPanics(t, func() { p.PreRequest(ctx, req, nil) })
		assert.NotContains(t, req.Headers, DPRankHeader)
	})
}

func TestChatAndAnthropicBodiesAreRouted(t *testing.T) {
	p := newTestPlugin(t, Parameters{DPSize: 4})

	assert.GreaterOrEqual(t, route(t, p, chatReq("m", longPrompt("chat-", 2))), 0)

	anthropic := &fwksched.InferenceRequest{
		TargetModel: "m",
		Headers:     map[string]string{},
		Body: &fwkrh.InferenceRequestBody{
			Messages: &fwkrh.MessagesRequest{
				System: fwkrh.AnthropicContent{Raw: "you are helpful"},
				Messages: []fwkrh.AnthropicMessage{
					{Role: "user", Content: fwkrh.AnthropicContent{
						Structured: []fwkrh.AnthropicContentBlock{
							{Type: "text", Text: longPrompt("anthropic-", 2)},
							{Type: "image"},
						},
					}},
				},
			},
		},
	}
	assert.GreaterOrEqual(t, route(t, p, anthropic), 0)
}

func TestConcurrentPreRequest(t *testing.T) {
	p := newTestPlugin(t, Parameters{DPSize: 8, MaxIndexEntries: 1000})
	ctx := context.Background()

	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				prompt := longPrompt(fmt.Sprintf("concurrent-%d-", (g*i)%37), 2)
				req := completionsReq("m", prompt)
				p.PreRequest(ctx, req, nil)
			}
		}(g)
	}
	wg.Wait()

	assert.LessOrEqual(t, p.index.len(), 1000, "LRU exceeded its bound")
}

func TestLRUIsBounded(t *testing.T) {
	p := newTestPlugin(t, Parameters{DPSize: 4, BlockSizeChars: 8, MaxIndexEntries: 50})
	for i := 0; i < 5000; i++ {
		route(t, p, completionsReq("m", fmt.Sprintf("prompt-number-%d-tail", i)))
	}
	assert.LessOrEqual(t, p.index.len(), 50)
	assert.Positive(t, p.index.len())
}

// --- decay --------------------------------------------------------------

func TestDecayHalvesAfterHalfLife(t *testing.T) {
	base := time.Now()
	var d decayedCounter
	halfLife := 10 * time.Second

	d.add(1.0, base, halfLife)
	assert.InDelta(t, 1.0, d.value(base, halfLife), 1e-9)
	assert.InDelta(t, 0.5, d.value(base.Add(halfLife), halfLife), 1e-9)
	// Decay is applied on read, so continuing from t+10s to t+20s halves again.
	assert.InDelta(t, 0.25, d.value(base.Add(2*halfLife), halfLife), 1e-9)
}

func TestPluginLoadDecays(t *testing.T) {
	p := newTestPlugin(t, Parameters{DPSize: 4, HalfLifeSeconds: f64(10)})
	base := time.Now()
	var mu sync.Mutex
	now := base
	p.now = func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		return now
	}

	prompt := longPrompt("decay-test-", 2)
	r := route(t, p, completionsReq("m", prompt))
	require.GreaterOrEqual(t, r, 0)

	before := p.snapshotLoads()[r]
	require.InDelta(t, 1.0, before, 1e-9)

	mu.Lock()
	now = base.Add(10 * time.Second)
	mu.Unlock()

	assert.InDelta(t, 0.5, p.snapshotLoads()[r], 1e-9)
}

// TestColdPrefixesSpreadAtZeroLoadWeight is the regression test for the
// collapse-onto-rank-0 bug, using prefixes that differ from byte zero (so they
// share no block and genuinely tie at affinity 0).
// Original note:
// collapse-onto-rank-0 bug: with loadWeight=0 every cold prefix ties at
// affinity 0, and a lowest-index tie-break sent all of them to rank 0.
// The rendezvous tie-break must spread them instead.
func TestColdPrefixesSpreadAtZeroLoadWeight(t *testing.T) {
	zero := 0.0
	p, err := New("t", Parameters{DPSize: 8, BlockSizeChars: 16, LoadWeight: &zero})
	if err != nil {
		t.Fatal(err)
	}
	seen := map[int]int{}
	for i := 0; i < 200; i++ {
		req := &fwksched.InferenceRequest{
			TargetModel: "m",
			Headers:     map[string]string{},
			Body: &fwkrh.InferenceRequestBody{
				Completions: &fwkrh.CompletionsRequest{
					Prompt: fwkrh.Prompt{Raw: fmt.Sprintf("%08d-wholly-distinct-from-byte-zero-%s", i*7919, strings.Repeat("x", 64))},
				},
			},
		}
		_ = p.PreRequest(context.Background(), req, nil)
		r, err := strconv.Atoi(req.Headers[DPRankHeader])
		if err != nil {
			t.Fatalf("no rank header on request %d", i)
		}
		seen[r]++
	}
	if len(seen) < 8 {
		t.Fatalf("cold prefixes collapsed onto %d rank(s), want all 8: %v", len(seen), seen)
	}
	for r, n := range seen {
		if n < 200/8/3 {
			t.Errorf("rank %d starved: %d of 200 (%v)", r, n, seen)
		}
	}
}

// TestStickyAcrossEvictionOrder verifies a prefix's home does not depend on
// arrival order -- two plugins seeing the same prefix after different traffic
// must agree, which naive load-based placement cannot guarantee.
func TestStickyHomeIsOrderIndependent(t *testing.T) {
	zero := 0.0
	mk := func() *Plugin {
		p, err := New("t", Parameters{DPSize: 8, BlockSizeChars: 16, LoadWeight: &zero})
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	ask := func(p *Plugin, prompt string) string {
		req := &fwksched.InferenceRequest{
			TargetModel: "m", Headers: map[string]string{},
			Body: &fwkrh.InferenceRequestBody{
				Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: prompt}},
			},
		}
		_ = p.PreRequest(context.Background(), req, nil)
		return req.Headers[DPRankHeader]
	}
	target := "the-target-prefix-" + strings.Repeat("y", 64)
	a := mk()
	first := ask(a, target)
	b := mk()
	for i := 0; i < 50; i++ {
		ask(b, fmt.Sprintf("noise-%d-%s", i, strings.Repeat("z", 64)))
	}
	if got := ask(b, target); got != first {
		t.Errorf("home rank depends on arrival order: %s vs %s", first, got)
	}
}

// The EPP decodes plugin parameters with DisallowUnknownFields at factory time
// (phase two), NOT at raw config load. These assert the factory is the layer
// that actually catches a bad ConfigMap.
func TestFactoryRejectsUnknownField(t *testing.T) {
	if _, err := Factory("dp-rank-router", fwkplugin.StrictDecoder(json.RawMessage(`{"dpSize":8,"loadWeightt":1.0}`)), nil); err == nil {
		t.Fatal("unknown parameter accepted; a ConfigMap typo would reach production")
	}
}

func TestFactoryRejectsMissingDPSize(t *testing.T) {
	if _, err := Factory("dp-rank-router", fwkplugin.StrictDecoder(json.RawMessage(`{"loadWeight":1.0}`)), nil); err == nil {
		t.Fatal("dpSize is required but absent was accepted")
	}
}

func TestFactoryAcceptsProductionParams(t *testing.T) {
	if _, err := Factory("dp-rank-router", fwkplugin.StrictDecoder(json.RawMessage(`{"dpSize":8,"loadWeight":1.0}`)), nil); err != nil {
		t.Fatalf("the params we intend to ship were rejected: %v", err)
	}
}
