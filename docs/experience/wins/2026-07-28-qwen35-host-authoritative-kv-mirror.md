# Qwen3.5/3.6 off its second KV allocator — warm TTFT stops scaling with prefix length

## Context

`executor/qwen35.rs` ran the device `PagedKVPool`'s OWN `alloc_tokens` /
`free_slot`, a second allocator whose page ids were unrelated to the engine's
host pool. A radix prefix hit was therefore only a *token* match: the KV bytes
behind it were unreachable by page identity, so the recurrent sidecar carried
the whole matched prefix's full-attention KV and round-tripped it through the
host at every turn — D2H + serialize on save, tier read + deserialize + H2D on
restore. Measured at 0.27 ms/token, strictly linear in prefix length
(`errors/2026-07-28-qwen35-second-kv-allocator-blocks-page-reuse.md`).

`executor/qwen.rs` never had this: the host pool is its single allocator and
each row's page list is lowered with `PagedKVPool::mirror_slot`, host page ids
indexing device storage rows 1:1.

## What Worked

Converged qwen35 onto that same host-authoritative model. All 23 device-allocator
call sites became `mirror_host_slot`, and the sidecar's `full_attn_kv` blob is
gone — restore now mirrors the radix prefix's own pages, which are already in HBM.

Three gaps the seam had to close first, all small:

- `KvAllocator::reinstate_slot_page` — the inverse of `evict_slot_page`, which
  the `--kv-recall` prefetch path needed on the host side (default no-op impl).
- `alloc_to_len_with_prefix_reclaim` — speculative decode materializes a whole
  draft chain's KV in one forward, past the single token the engine
  pre-allocated. The executor now grows the host pool itself and the engine's
  post-step append of the accepted extras grows-to-target instead of blindly
  appending, so it no-ops rather than double-counting.
- `promote_slot` passes the engine's `slot_pages` through to the whole-slot swap
  restore (the seam already carried them; the qwen35 arm had ignored them).

Deleted with their reason: `kv_device_gate_active` / `kv_device_fit` for this arm
(the device pool no longer allocates, so the host pool is the fit authority —
same as dense Qwen).

## Measurement

Matched A/B, one box, one model, same prompts, sequential on the same GPU.
`Qwen3.6-35B-A3B-FP8` (MoE), 1×H20, eager, c=1, `ttft_scale.py` — one long prompt
resent to a warm server, so TTFT is almost entirely restore cost.

| prefix | baseline warm TTFT | mirror warm TTFT | Δ |
|--------|-------------------|------------------|---|
| 4k  | 0.480 s | 0.161 s | 3.0× |
| 8k  | 0.814 s | 0.147 s | 5.5× |
| 16k | 1.708 s | 0.162 s | 10.6× |
| 33k | 3.020 s | 0.175 s | **17.3×** |

Warm TTFT is now flat in prefix length (0.147–0.175 s across a 8× span) — the
linear per-token term is gone, which is the signature the diagnosis predicted.
Cold TTFT also drops (33k 37.94 → 28.96 s, −24%; 16k 35.95 → 27.81 s, −23%) from
the save-side D2H + serialize disappearing.

Correctness gate: `needle_gate.py 512,4096,16384,32768 3 0.0` — exact=3 DET at
every length on both arms, identical.

## Rule

**Two pools with the same page-id type are one address space or they are not —
decide it once, at the seam, not per executor.** The sidecar's KV blob was not a
feature; it was the cost of a second allocator, and it stayed invisible until
someone measured warm TTFT against prefix length. When a backend carries its own
copy of state the engine already owns, the copy is the symptom — look for the
ownership split above it.
