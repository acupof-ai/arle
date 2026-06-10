# Qwen3-dense CUDA prefix attach: host-authoritative device page mirror

**Date:** 2026-06-10. **Backend:** CUDA, Qwen3-dense BF16 (single GPU).
**Scope:** `cuda-kernels/src/paged_kv.rs` (`mirror_slot`), `infer-cuda/src/executor.rs`
(Qwen submit path), `infer-api/src/loaded.rs` (admission split, separate commit).
**Status: pending-remote** — needs V100 verification (no CUDA GPU locally).

## Context

The rewrite ran TWO independent page allocators for Qwen3-dense: the host
`CudaKvPool` (engine admission + radix prefix cache) and the device
`TokenKVPool` (its own LIFO free list inside the executor). The engine's
prefix attach wrote only the host table; the device pool's
attach/retain/COW machinery (monolith-era) had zero callers in the rewrite.
Consequences, established by source review (2026-06-10 storage-utilization
review):

- Any radix prefix hit (≥16 shared leading tokens — every chat-template
  prompt) produced `PrefillRow.start_pos > 0` and tripped the executor's
  `materialized == start_pos` assert → engine thread exits → every later
  request gets "engine thread closed". Same failure shape as the
  2026-06-08 DSv4/Qwen3Moe crash, which left `Qwen3Dense` prefix reuse
  enabled on the (false) assumption that the paged pool could honor it.
- Host and device page ids silently diverged after multi-slot churn (host
  frees at finish, device freed only on slot reuse), so the device could
  hit "out of pages" while host admission believed capacity existed.

## What Worked

Make the host pool the SINGLE page allocator. The executor stops calling
`alloc_tokens`/`free_slot` on the device pool entirely; per scheduled row it
lowers the engine-built `KvBatchDescriptor` (host page table covering
`[0, append_end)`) into the device pool via the new
`TokenKVPool::mirror_slot(slot, pages, seq_len)` — page ids now index device
storage rows 1:1, the same lowering pattern DSv4 already used for its
adapter. Radix prefix attach then works with no extra machinery: retained
host pages keep their device KV rows (the host never recycles them), so a
fresh slot mirroring those ids reads the published prefix KV directly.

The old device-side materialized asserts are replaced by an executor
watermark (`slot_progress`: occupant epoch + materialized length): same
epoch ⇒ contiguous appends enforced loudly; new epoch may start at
`append_pos > 0` only via attached retained pages. Warmup capture mirrors a
dummy 1-token view onto page 0 instead of driving the device allocator.
DSv4/Qwen3.5 arms untouched (descriptor now built per arm; the Qwen3.5 arm
skips the per-step page-id flattening it never consumed).

## Verification

- Mac typecheck `cargo check -p infer-api --features cuda,no-cuda --lib` PASS;
  `cargo test -p infer-core` 33/33, `cargo test -p infer-cuda` (host) 40/40;
  clippy: 0 new warnings (15 pre-existing in untouched files).
- **pending-remote (V100):** (1) serve Qwen3-dense, two same-prefix ≥16-token
  curls — pre-change the second kills the engine thread, post-change it must
  complete with `start_pos > 0` reuse; (2) needle gate; (3) bench
  `scripts/bench_guidellm.sh` vs latest V100 Qwen3-dense baseline with Δ%.

## Rule

- A page-addressable KV claim is only true when ONE allocator owns page
  identity end-to-end. Two mirrored LIFO allocators with different free
  timing are a divergence engine; lower host page tables into the device
  (descriptor → view), never run a second allocator.
- When a crash fix exempts one model kind ("only X can honor this"), verify
  the exempted path actually executes its happy path before shipping the
  claim — the 2026-06-08 fix verified DSv4/Qwen3Moe but never exercised a
  Qwen3Dense prefix hit.
