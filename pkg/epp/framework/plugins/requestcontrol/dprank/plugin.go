// Package dprank provides an out-of-tree EPP PreRequest plugin that assigns a
// data-parallel rank to each inference request based on prefix affinity and
// decayed per-rank load, and emits it as the "x-data-parallel-rank" header.
package dprank

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/cespare/xxhash/v2"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	// DPRankPluginType is the plugin type string used in the EPP plugin configuration.
	DPRankPluginType = "dp-rank-router"

	// DPRankHeader is the request header written with the chosen rank.
	DPRankHeader = "x-data-parallel-rank"

	defaultBlockSizeChars  = 512
	defaultMaxPrefixBlocks = 256
	defaultLoadWeight      = 0.3
	defaultHalfLifeSeconds = 10.0

	// maxDPSize is the hard upper bound on dpSize, imposed by the uint64 bitset
	// used to track which ranks hold a block.
	maxDPSize = 64

	// defaultMaxIndexEntries bounds the LRU so memory cannot grow unbounded.
	defaultMaxIndexEntries = 500_000
)

// Parameters is the user-facing plugin configuration block.
//
// The framework hands factories a strict decoder (DisallowUnknownFields), so
// every accepted key must appear here.
type Parameters struct {
	// DPSize is the number of data-parallel ranks. Required, >= 1, <= 64.
	DPSize int `json:"dpSize"`
	// BlockSizeChars is the prefix block size in characters. Default 512.
	BlockSizeChars int `json:"blockSizeChars,omitempty"`
	// MaxPrefixBlocks caps how many leading blocks are considered. Default 256.
	MaxPrefixBlocks int `json:"maxPrefixBlocks,omitempty"`
	// LoadWeight trades prefix affinity against load balancing. Default 0.3, >= 0.
	LoadWeight *float64 `json:"loadWeight,omitempty"`
	// HalfLifeSeconds is the half-life of the per-rank decayed load counter. Default 10.
	HalfLifeSeconds *float64 `json:"halfLifeSeconds,omitempty"`
	// MaxIndexEntries bounds the block-hash LRU. Default 500000.
	MaxIndexEntries int `json:"maxIndexEntries,omitempty"`
}

// Factory is the plugin factory registered under DPRankPluginType.
func Factory(name string, parameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := Parameters{}
	if parameters == nil {
		return nil, fmt.Errorf("%s: parameters are required (dpSize must be set)", DPRankPluginType)
	}
	if err := parameters.Decode(&params); err != nil {
		return nil, fmt.Errorf("%s: failed to decode parameters: %w", DPRankPluginType, err)
	}
	return New(name, params)
}

// New validates the parameters and constructs the plugin.
func New(name string, params Parameters) (*Plugin, error) {
	if params.DPSize < 1 {
		return nil, fmt.Errorf("%s: dpSize is required and must be >= 1 (got %d)", DPRankPluginType, params.DPSize)
	}
	if params.DPSize > maxDPSize {
		return nil, fmt.Errorf("%s: dpSize must be <= %d (got %d)", DPRankPluginType, maxDPSize, params.DPSize)
	}

	blockSize := params.BlockSizeChars
	if blockSize == 0 {
		blockSize = defaultBlockSizeChars
	}
	if blockSize < 1 {
		return nil, fmt.Errorf("%s: blockSizeChars must be >= 1 (got %d)", DPRankPluginType, blockSize)
	}

	maxBlocks := params.MaxPrefixBlocks
	if maxBlocks == 0 {
		maxBlocks = defaultMaxPrefixBlocks
	}
	if maxBlocks < 1 {
		return nil, fmt.Errorf("%s: maxPrefixBlocks must be >= 1 (got %d)", DPRankPluginType, maxBlocks)
	}

	loadWeight := defaultLoadWeight
	if params.LoadWeight != nil {
		loadWeight = *params.LoadWeight
	}
	if loadWeight < 0 || math.IsNaN(loadWeight) || math.IsInf(loadWeight, 0) {
		return nil, fmt.Errorf("%s: loadWeight must be >= 0 and finite (got %v)", DPRankPluginType, loadWeight)
	}

	halfLife := defaultHalfLifeSeconds
	if params.HalfLifeSeconds != nil {
		halfLife = *params.HalfLifeSeconds
	}
	if halfLife <= 0 || math.IsNaN(halfLife) || math.IsInf(halfLife, 0) {
		return nil, fmt.Errorf("%s: halfLifeSeconds must be > 0 and finite (got %v)", DPRankPluginType, halfLife)
	}

	maxEntries := params.MaxIndexEntries
	if maxEntries == 0 {
		maxEntries = defaultMaxIndexEntries
	}
	if maxEntries < 1 {
		return nil, fmt.Errorf("%s: maxIndexEntries must be >= 1 (got %d)", DPRankPluginType, maxEntries)
	}

	return &Plugin{
		typedName:       fwkplugin.TypedName{Type: DPRankPluginType, Name: name},
		dpSize:          params.DPSize,
		blockSizeChars:  blockSize,
		maxPrefixBlocks: maxBlocks,
		loadWeight:      loadWeight,
		halfLife:        time.Duration(halfLife * float64(time.Second)),
		index:           newBlockIndex(maxEntries),
		load:            make([]decayedCounter, params.DPSize),
		now:             time.Now,
	}, nil
}

// compile-time interface assertions
var (
	_ fwkplugin.Plugin = &Plugin{}
	_ fwkrc.PreRequest = &Plugin{}
)

// Plugin implements requestcontrol.PreRequest.
type Plugin struct {
	typedName       fwkplugin.TypedName
	dpSize          int
	blockSizeChars  int
	maxPrefixBlocks int
	loadWeight      float64
	halfLife        time.Duration

	index *blockIndex

	mu   sync.Mutex
	load []decayedCounter

	// now is injectable for deterministic decay tests.
	now func() time.Time
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *Plugin) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// PreRequest picks a data-parallel rank and writes it to the request headers.
// It always returns nil: failures are expressed by omitting the header, and it
// must never panic (a panic here would kill the ext_proc stream).
func (p *Plugin) PreRequest(_ context.Context, request *fwksched.InferenceRequest, _ *fwksched.SchedulingResult) error {
	if p == nil || request == nil {
		return nil
	}
	// Headers is aliased to the ext_proc request headers. A nil map cannot be
	// written to, so there is nothing useful to do.
	if request.Headers == nil {
		return nil
	}

	prompt := extractPrompt(request.Body)
	if prompt == "" {
		return nil
	}

	hashes := p.chainHashes(request.TargetModel, prompt)
	if len(hashes) == 0 {
		return nil
	}

	rank := p.pick(hashes)
	if rank < 0 || rank >= p.dpSize {
		return nil
	}

	p.record(rank, hashes)
	request.Headers[DPRankHeader] = strconv.Itoa(rank)
	return nil
}

// chainHashes splits the prompt into blocks and returns the cumulative chain
// hash of each block: h_0 = H(model, block_0); h_i = H(h_{i-1}, block_i).
func (p *Plugin) chainHashes(model, prompt string) []uint64 {
	if p.blockSizeChars < 1 || p.maxPrefixBlocks < 1 || prompt == "" {
		return nil
	}

	// Operate on bytes; the exact split point does not need to be rune-aligned,
	// it only needs to be deterministic.
	n := len(prompt)
	nBlocks := (n + p.blockSizeChars - 1) / p.blockSizeChars
	if nBlocks > p.maxPrefixBlocks {
		nBlocks = p.maxPrefixBlocks
	}
	if nBlocks < 1 {
		return nil
	}

	hashes := make([]uint64, 0, nBlocks)
	var prev uint64
	for i := 0; i < nBlocks; i++ {
		start := i * p.blockSizeChars
		if start >= n {
			break
		}
		end := start + p.blockSizeChars
		if end > n {
			end = n
		}

		d := xxhash.New()
		if i == 0 {
			_, _ = d.WriteString(model)
		} else {
			var buf [8]byte
			putUint64(&buf, prev)
			_, _ = d.Write(buf[:])
		}
		_, _ = d.WriteString("\x00")
		_, _ = d.WriteString(prompt[start:end])

		prev = d.Sum64()
		hashes = append(hashes, prev)
	}
	return hashes
}

func putUint64(b *[8]byte, v uint64) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
	b[4] = byte(v >> 32)
	b[5] = byte(v >> 40)
	b[6] = byte(v >> 48)
	b[7] = byte(v >> 56)
}

// pick scores every rank and returns the argmax, breaking ties toward the
// lowest rank index.
func (p *Plugin) pick(hashes []uint64) int {
	total := len(hashes)
	if total == 0 || p.dpSize < 1 {
		return -1
	}

	// masks[i] is the bitset of ranks holding block i.
	masks := p.index.lookup(hashes)

	// Longest contiguous run of leading blocks held by each rank.
	matched := make([]int, p.dpSize)
	alive := ^uint64(0)
	if p.dpSize < 64 {
		alive = (uint64(1) << uint(p.dpSize)) - 1
	}
	for i := 0; i < total && alive != 0; i++ {
		alive &= masks[i]
		for r := 0; r < p.dpSize; r++ {
			if alive&(uint64(1)<<uint(r)) != 0 {
				matched[r] = i + 1
			}
		}
	}

	loads := p.snapshotLoads()
	maxLoad := 0.0
	for _, l := range loads {
		if l > maxLoad {
			maxLoad = l
		}
	}

	bestRank := 0
	bestScore := math.Inf(-1)
	bestWeight := uint64(0)
	for r := 0; r < p.dpSize; r++ {
		affinity := float64(matched[r]) / float64(total)
		normLoad := 0.0
		if maxLoad > 0 {
			normLoad = loads[r] / maxLoad
		}
		score := affinity - p.loadWeight*normLoad

		// Rendezvous (highest-random-weight) tie-break. Without this, a prefix
		// that no rank holds yet scores 0 on every rank, and a naive
		// lowest-index tie-break would send every cold prefix to rank 0 --
		// collapsing the deployment onto a single DP rank. Hashing the prefix
		// with the rank gives each prefix a deterministic, uniformly spread
		// home that is stable across evictions and independent of arrival
		// order, so concurrent first-turns of one conversation agree.
		// Key the home on the LAST chain hash: because hashes are cumulative,
		// it encodes the whole prefix. Keying on hashes[0] would give every
		// request sharing a common system prompt the same home rank, which is
		// most of production traffic.
		weight := rendezvousWeight(hashes[total-1], r)

		if score > bestScore || (score == bestScore && weight > bestWeight) {
			bestScore = score
			bestWeight = weight
			bestRank = r
		}
	}
	return bestRank
}

// rendezvousWeight returns the highest-random-weight score of rank r for the
// given prefix key. The rank with the greatest weight is the prefix's home.
func rendezvousWeight(prefixKey uint64, rank int) uint64 {
	var b [16]byte
	putUint64((*[8]byte)(b[0:8]), prefixKey)
	putUint64((*[8]byte)(b[8:16]), uint64(rank))
	return xxhash.Sum64(b[:])
}

// record marks the chosen rank as holding every block, and charges it one unit
// of decayed load.
func (p *Plugin) record(rank int, hashes []uint64) {
	p.index.set(hashes, rank)

	now := p.nowFn()
	p.mu.Lock()
	defer p.mu.Unlock()
	if rank < 0 || rank >= len(p.load) {
		return
	}
	p.load[rank].add(1.0, now, p.halfLife)
}

func (p *Plugin) snapshotLoads() []float64 {
	now := p.nowFn()
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]float64, len(p.load))
	for i := range p.load {
		out[i] = p.load[i].value(now, p.halfLife)
	}
	return out
}

func (p *Plugin) nowFn() time.Time {
	if p.now != nil {
		return p.now()
	}
	return time.Now()
}

// decayedCounter is an exponentially-decaying counter with a configurable
// half-life. It is not itself synchronized; the Plugin mutex guards it.
type decayedCounter struct {
	val        float64
	lastUpdate time.Time
}

func (d *decayedCounter) decay(now time.Time, halfLife time.Duration) {
	if d.lastUpdate.IsZero() {
		d.lastUpdate = now
		return
	}
	elapsed := now.Sub(d.lastUpdate)
	if elapsed <= 0 || halfLife <= 0 {
		d.lastUpdate = now
		return
	}
	factor := math.Pow(0.5, elapsed.Seconds()/halfLife.Seconds())
	if math.IsNaN(factor) || math.IsInf(factor, 0) {
		factor = 0
	}
	d.val *= factor
	d.lastUpdate = now
}

func (d *decayedCounter) value(now time.Time, halfLife time.Duration) float64 {
	d.decay(now, halfLife)
	return d.val
}

func (d *decayedCounter) add(delta float64, now time.Time, halfLife time.Duration) {
	d.decay(now, halfLife)
	d.val += delta
}

// extractPrompt pulls plain prompt text out of the parsed request body. It
// tolerates every field being nil and returns "" when no text is reachable.
func extractPrompt(body *fwkrh.InferenceRequestBody) string {
	if body == nil {
		return ""
	}

	if c := body.Completions; c != nil {
		if t := c.Prompt.PlainText(); t != "" {
			return t
		}
	}

	if cc := body.ChatCompletions; cc != nil && len(cc.Messages) > 0 {
		var sb strings.Builder
		for i := range cc.Messages {
			msg := &cc.Messages[i]
			sb.WriteString(msg.Role)
			sb.WriteString("\n")
			sb.WriteString(msg.Content.PlainText())
			sb.WriteString("\n")
		}
		if t := sb.String(); strings.TrimSpace(t) != "" {
			return t
		}
	}

	if m := body.Messages; m != nil {
		var sb strings.Builder
		sb.WriteString(anthropicText(m.System))
		for i := range m.Messages {
			msg := &m.Messages[i]
			sb.WriteString(msg.Role)
			sb.WriteString("\n")
			sb.WriteString(anthropicText(msg.Content))
			sb.WriteString("\n")
		}
		if t := sb.String(); strings.TrimSpace(t) != "" {
			return t
		}
	}

	if e := body.Embeddings; e != nil {
		if t := e.Input.PlainText(); t != "" {
			return t
		}
	}

	return ""
}

// anthropicText flattens an AnthropicContent. The router exposes no public
// PlainText helper for this type (only an unexported textLen), so we handle
// both the raw-string and structured-block forms here.
func anthropicText(c fwkrh.AnthropicContent) string {
	if c.Raw != "" {
		return c.Raw
	}
	var sb strings.Builder
	for _, block := range c.Structured {
		if block.Type == "text" {
			sb.WriteString(block.Text)
		}
	}
	return sb.String()
}
