# DSv4 + Qwen3.5/3.6-MoE prefix-cache crash fixed (recurrent-KV ≠ page-reuse)

**Date:** 2026-06-08. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Commit:** `702454fe`. **Scope:** `infer-core` scheduler + `infer-api` serve config.

## Context

After re-wiring DSv4 multi-rank serve (`63d814a4`), the **second** HTTP request that
shared any prompt prefix with the first crashed the engine thread:

```
infer-server engine step failed: DSv4 slot seq_len 75 != start_pos 16;
decode requires contiguous appends
```

Root cause: the host **radix prefix cache** (`infer-core` `RadixCache`, keyed by
`KvPool::page_size`) handed the request a cross-request prefix hit
(`start_pos = 16`, the 16 shared leading tokens). But DSv4's per-slot KV is
**recurrent / not page-addressable** — a sliding-window ring (`sw_window_cache`,
slot = `pos % 128`) plus compressor/indexer running state — and its forward
asserts contiguous appends from a reset (`slot.seq_len == start_pos`,
`dsv4.rs:1549`). The cached prefix's KV cannot be re-attached to a fresh slot, so
`start_pos > 0` is unservable and the assert fires.

This is **not DSv4-only**: `Qwen3Moe` (Qwen3.5/3.6) is also a hybrid —
gated-delta **linear-attention** layers (the majority) carry per-slot recurrent
state + a conv ring (`qwen35.rs:6,86`) and assert the same contiguous-append
invariant (`qwen35.rs:723`). Only pure full-attention `Qwen3Dense` has a
page-addressable KV pool that can honor prefix reuse.

## What worked (verified on 8×H20)

`SchedulerConfig::enable_prefix_cache` (default **on** — `Qwen3Dense` keeps
prefix reuse) gates the radix `longest_prefix_match` / `peek_longest_prefix_match`
(empty match ⇒ every request resets at `start_pos == 0`). `cuda_serve_handle`
flips it **off** for `Dsv4 | Qwen3Moe`. Both the rank-0 coordinator and worker
ranks build through that same helper with the same `kind`, so the disable stays
in lockstep across the NCCL group.

> **Correction (`6e2e572e`).** This commit also pinned `chunked_prefill_size =
> max(.., dsv4_max_seq_len())` (single-chunk) on a *hypothesized* multi-rank NCCL
> chunk-boundary desync. That hypothesis was **FALSE** and single-chunk conflicts
> with origin/main's `query_chunk` memory bound (asserts ≤4096 query tok/call) —
> >4096 prompts crashed the engine and 900K OOMs. Reverted to
> `chunked_prefill_size = 4096`; multi-rank chunked prefill works (single 64K = 13
> contiguous chunks, 29.5s, no hang). Same commit lifts the 32768 `max_prompt_tokens`
> cap (it silently returned empty for >32K prompts). See
> [`2026-06-08-dsv4-per-layer-rope-theta-longctx-fix.md`].

**Verified:** a back-to-back needle sweep (50 / 130 / 300 / 596 + a repeat —
prompts sharing the `<｜User｜>…` prefix) ran to completion with **0**
`engine step failed` errors. Before the fix the second request crashed the engine
thread and every subsequent request returned HTTP 400 "engine thread closed".

cargo check (cuda,no-cuda) PASS; `cargo test -p infer-core` 31/31 PASS (prefix
tests unchanged — default stays on).

## Rule

- Cross-request prefix reuse is **only** sound for page-addressable
  full-attention KV. Recurrent / hybrid KV (DSv4 sliding-window+compressor,
  Qwen3.5/3.6 gated-delta linear attention) must disable it — gate by model
  kind at the single engine-builder, never per-serve-call.
- A parallel in-tree refactor generalizes this to a capability
  (`BackendExecutor::reusable_prefix_pages` + `clamp_prefix_to_backend`); once it
  lands, the `matches!(kind, Dsv4 | Qwen3Moe)` name-match here is superseded by
  the executor declaring its own reusable-prefix bound.
- pending-remote: full guidellm sweep deferred until the separate long-context
  DSA prefill-correctness bug (below) lands — output is garbage past ~80 tokens,
  so throughput numbers would be meaningless.
