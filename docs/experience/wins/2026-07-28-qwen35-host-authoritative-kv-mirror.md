# Qwen3.5/3.6 off its second KV allocator — warm TTFT stops scaling with prefix

## Context

`executor/qwen35.rs` ran the device pool's own allocator, a page-id space
unrelated to the engine's host pool. A radix hit was therefore only a *token*
match, so the recurrent sidecar carried the matched prefix's whole
full-attention KV through the host every turn — 0.27 ms/token, linear in prefix
length (`errors/2026-07-28-qwen35-second-kv-allocator-blocks-page-reuse.md`).
`executor/qwen.rs` never had this: host pool as single allocator, page list
lowered with `mirror_slot`, host ids indexing device rows 1:1.

## What Worked

All 23 device-allocator sites became `mirror_host_slot`; the sidecar's
`full_attn_kv` blob is gone — restore mirrors the prefix's own resident pages.

Three seam gaps closed first:

- `KvAllocator::reinstate_slot_page` — inverse of `evict_slot_page`, needed by
  `--kv-recall` prefetch on the host side.
- `alloc_to_len_with_prefix_reclaim` — spec decode writes a whole draft chain
  past the one token the engine pre-allocated, so the executor grows the host
  pool and the engine's post-step append grows-to-target instead of blindly
  appending.
- `promote_slot` forwards the engine's `slot_pages` into the whole-slot swap.

Deleted: `kv_device_gate_active` / `kv_device_fit` for this arm — the device
pool no longer allocates, so the host pool is the fit authority.

## Measurement

Matched A/B, one box, one GPU, one model, sequential. `Qwen3.6-35B-A3B-FP8`,
1×H20, eager, `ttft_scale.py` at c=1 (one long prompt resent warm, so TTFT is
almost all restore cost).

| prefix | base warm TTFT | mirror warm TTFT | Δ |
|--------|---------------:|-----------------:|---|
| 4k  | 0.480 s | 0.161 s | 3.0× |
| 8k  | 0.814 s | 0.147 s | 5.5× |
| 16k | 1.708 s | 0.162 s | 10.6× |
| 33k | 3.020 s | 0.175 s | **17.3×** |

Warm TTFT is now flat across an 8× prefix span — the linear per-token term is
gone. Cold drops too (33k 37.94 → 28.96 s) from the save-side D2H.

Gate: `needle_gate.py 512,4096,16384,32768 3 0.0` — exact=3 DET on both arms.

## Rule

**Two pools with the same page-id type are one address space or they are not —
decide it at the seam, once, not per executor.** The sidecar's KV blob was the
cost of a second allocator, and stayed invisible until someone plotted warm TTFT
against prefix length. When a backend keeps its own copy of state the engine
already owns, the copy is the symptom; look for the ownership split above it.
