package dprank

import (
	"container/list"
	"sync"
)

// blockIndex is a bounded, mutex-protected LRU mapping a block chain-hash to a
// bitset of the data-parallel ranks known to hold that block.
//
// dpSize is capped at 64, so a uint64 bitset is sufficient.
type blockIndex struct {
	mu       sync.Mutex
	maxEntry int
	entries  map[uint64]*list.Element
	order    *list.List // front = most recently used
}

type indexEntry struct {
	hash uint64
	mask uint64
}

func newBlockIndex(maxEntries int) *blockIndex {
	if maxEntries < 1 {
		maxEntries = 1
	}
	return &blockIndex{
		maxEntry: maxEntries,
		entries:  make(map[uint64]*list.Element),
		order:    list.New(),
	}
}

// lookup returns, for each input hash, the bitset of ranks holding that block.
// Unknown blocks yield 0. The returned slice always has len(hashes) elements.
func (b *blockIndex) lookup(hashes []uint64) []uint64 {
	out := make([]uint64, len(hashes))
	if b == nil || len(hashes) == 0 {
		return out
	}

	b.mu.Lock()
	defer b.mu.Unlock()
	for i, h := range hashes {
		el, ok := b.entries[h]
		if !ok || el == nil {
			continue
		}
		e, ok := el.Value.(*indexEntry)
		if !ok || e == nil {
			continue
		}
		out[i] = e.mask
		b.order.MoveToFront(el)
	}
	return out
}

// set marks rank as holding every given block, evicting least-recently-used
// entries to stay within the configured bound.
func (b *blockIndex) set(hashes []uint64, rank int) {
	if b == nil || len(hashes) == 0 || rank < 0 || rank >= maxDPSize {
		return
	}
	bit := uint64(1) << uint(rank)

	b.mu.Lock()
	defer b.mu.Unlock()
	for _, h := range hashes {
		if el, ok := b.entries[h]; ok && el != nil {
			if e, ok := el.Value.(*indexEntry); ok && e != nil {
				e.mask |= bit
				b.order.MoveToFront(el)
				continue
			}
			// Corrupt element; drop and re-insert below.
			b.order.Remove(el)
			delete(b.entries, h)
		}
		b.entries[h] = b.order.PushFront(&indexEntry{hash: h, mask: bit})
	}
	b.evictLocked()
}

func (b *blockIndex) evictLocked() {
	for b.order.Len() > b.maxEntry {
		el := b.order.Back()
		if el == nil {
			return
		}
		b.order.Remove(el)
		if e, ok := el.Value.(*indexEntry); ok && e != nil {
			delete(b.entries, e.hash)
		}
	}
}

func (b *blockIndex) len() int {
	if b == nil {
		return 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.order.Len()
}
