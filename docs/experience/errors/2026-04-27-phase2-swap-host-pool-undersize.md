# Phase 2 KV swap-out — two layered blockers: undersized host pool + scheduler overlap conflict

## Context

Phase 2 of [`active-kv-swap-out-unification`](../../projects/active-kv-swap-out-unification.md)
landed end-to-end as commit
[`221704e0`](https://github.com/cklxx/agent-infer/commit/221704e0) +
CLI flag commit `2ea2a8a9`: `PreemptionMode::Swap` is plumbed through,
`try_swap_out_victim_for_admission` + `try_resume_swapped_victims` are
wired into `assign_slots` and the tick loop, codex-reviewed clean
across nine iterations.

Bench at the design doc's matched 2026-04-26 SGLang config (Qwen3-4B,
L4, `--num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94
--chunked-prefill-size 4096`, `bench_guidellm.sh --fast` =
profile=concurrent, rate=16, 4096-in / 256-out, 30 s):

| metric | `--preemption-mode swap` | `--preemption-mode recompute` |
|---|---:|---:|
| TTFT p50 (ms) | 4052 | 4065 |
| TTFT p99 (ms) | 20675 | 20675 |
| ITL p50 (ms) | 48.6 | 48.6 |
| out tok/s | 87.99 | 88.01 |
| peak_active | 6 | 6 |
| peak_kv_util | 96.2% | 96.3% |
| swap-out events in server log | **0** | n/a |

A/B between Swap and Recompute is identical to within run-to-run noise
on the same Phase 2 commit. The swap path fired zero times in 30 s of
sustained c=16 / 4096-token-prompt load.

The TTFT improvement vs the design doc's reference baseline
(16357 → 4052 ms) is real but came from unrelated commits in the
2026-04-26..2026-04-27 window (TileLang AOT decode/prefill landings,
KV-tier coordinator simplification, prefix-cache work). Phase 2
contributed zero of it.

## Root Cause

The CUDA scheduler's host-pinned T1 pool is sized for **prefix-cache
demote** (a few sealed prefix blocks per slot at most), not
**whole-slot swap-out**. From `infer/src/scheduler/cuda/core.rs:1003`:

```rust
let host_block_bytes = paged_kv_pool.storage_bytes_for_tokens(PREFIX_CACHE_BLOCK_SIZE);
let host_pool_capacity = host_block_bytes
    .saturating_mul(config.max_slots.saturating_mul(16).max(1))
    .max(64 * 1024 * 1024);
```

For `max_slots=16` and `PREFIX_CACHE_BLOCK_SIZE=16` tokens, this gives
`16 (slots) * 16 (blocks) * 16 (tokens/block) = 4096 tokens` of T1
capacity = ~604 MB at Qwen3-4B BF16 (matches the server log
`host_pool=604.0MB`). One slot's KV at the bench's `max_seq_len=4608`
is **also ~604 MB**. So T1 fits exactly **one** swap-out victim — and
because the canonical workload uses uniform 4096-token prompts, that
single park doesn't free enough budget to admit another full-ISL
candidate.

Concretely, in `try_swap_out_victim_for_admission`:
1. Pick victim, compute pages.
2. `paged_kv_pool.copy_pages_to_host` — succeeds (~600 MB Vec<u8>).
3. `host_pinned_pool.lock()?.reserve(payload.len())?` — when T1 has
   any residual prefix-cache demote pages, this returns `None`, and
   the helper returns `Ok(false)`.

The guard in `assign_slots` then defers the candidate, and on the next
tick `try_resume_swapped_victims` runs first — restoring the parked
victim before admission re-tries — which is why the server log shows
zero `swap-out (Phase::WaitingFetch)` info lines: the swap-out path
**either fails T1 reserve immediately, or its first park gets
restored before pressure can re-fire it**.

The math the design doc assumed ("604 MB sat idle while 11 requests
waited 16 s") was right that prefix-cache demote rarely uses the
604 MB; but **swap-out for full-slot KV needs ~`max_slots × max_seq_len`
tokens of T1**, not `max_slots × 16 × 16` tokens. For c=16 / 4608-len
that's `16 × 4608 ≈ 73 K tokens ≈ 10 GB` of host-pinned memory — 16×
the current sizing.

## Second blocker — scheduler overlap conflict

After bumping the host pool to ~5.4 GB (formula change in `core.rs`:
`max(prefix_demote_estimate, swap_estimate)` where `swap_estimate =
max_slots / 2 × max_seq_len × token_bytes`), instrumentation showed
**5106 swap-out attempts in 30 s**, but **all returned `Ok(false)`**
because the eligibility filter found `pending_gpu=6, eligible=0` on
every call.

`run()` deliberately overlaps the prior tick's GPU readback with this
tick's admission planning (`runtime.rs:1477` — the comment reads "step
keeps decode/prefill readback pending across loop turns so this
iteration's intake/admission work can overlap the previous iteration's
GPU compute"). All 6 `Phase::Decoding` slots in `running_batch` are
marked `slot_has_pending_gpu_work` during admission, and the swap
eligibility filter excludes them — `release_slot_pages_only` calls
`state.reset()` which would corrupt the in-flight `select_tokens_batch`
readback that holds `&mut states`.

Calling `step_decode_readback()` at the top of admission (before the
swap loop) lets swap actually fire — peak_active jumps from 6 to 14,
swap events appear in the log. But the explicit sync **catastrophically
breaks the overlap**:

| metric | overlap intact (no swap) | drain-then-swap |
|---|---:|---:|
| peak_active | 6 | 14 |
| ITL p50 (ms) | 48.6 | 1345 |
| out tok/s | 88.0 | 2.93 |
| swap events | 0 | 8 |

Even draining ONCE per admission swap burst (not per attempt) gives
the same regression — the trade is "admit more concurrently" vs "lose
~50 ms/tick of GPU/CPU overlap". For 4096-token prompts the overlap
matters more than concurrency: 14 vs 6 active is +133% capacity, but
ITL going from 48 ms to 1345 ms is a 28× regression.

## Fix

Both blockers stay open under
[`active-kv-swap-out-unification.md`](../../projects/active-kv-swap-out-unification.md)
as **Phase 2.5**.

**1. Host pool sizing — landed.** `core.rs::host_pool_capacity` now
takes `max(prefix_demote_estimate, swap_estimate)` so the host pool
can fit ~half the slots' worth of swapped KV without bumping a flag.
This is correct independently of the second blocker.

**2. Async memcpy on the swap path.** The synchronous T0↔T1 copies
are the deepest cost. Using `cudaMemcpyAsync` on a dedicated stream
(decoupled from the decode stream) would let swap-out and swap-in
overlap with decode. Pre-req for landing the readback drain.

**3. Reorder admission planning vs readback.** Alternative: move
`assign_slots` to run AFTER `step_decode_readback` in the tick. Loses
the overlap unconditionally, but predictable, and the swap path
becomes correct without per-call sync. The cost may not matter if
admission rarely runs (most ticks just decode).

**4. Iterate victim list on T1 reserve failure.** Codex flagged in the
final review; uniform-size canonical bench doesn't exercise it but
heterogeneous-prompt workloads do.

The Phase 2 code itself (admission + resume + cleanup paths) is
correct and stays — codex-clean across nine review iterations. It's
unable to demonstrate net-positive value until at least (2) lands.

## Rule

**Validate the supporting tier before assuming a feature works at
scale.** A swap-out path is only as useful as its target tier's
capacity. The design doc cited the existing `host_pool=604.0MB` as
"sat idle" — true for prefix-cache demote, fundamentally false for
whole-slot swap. Future tier-shift work must derive its tier-size
budget from the new use case, not infer from the existing tier's
historical idleness.

A/B benches between modes on the SAME commit are the cheapest possible
sanity check that a path actually fires; would have caught this on
the first run if I had run them together rather than just measuring
the swap-mode delta vs an unrelated baseline.

## Cross-references

- Design: [`projects/active-kv-swap-out-unification.md`](../../projects/active-kv-swap-out-unification.md)
- Phase 2 commit: `221704e0` (combined swap-out + swap-in)
- CLI flag commit: `2ea2a8a9` (`--preemption-mode`)
- Bench artefacts:
  - `bench-output/2026-04-27-cuda-l4-phase2-swap/`
  - `bench-output/2026-04-27-cuda-l4-phase2-recompute/`
- Earlier reverted Phase 1 attempt:
  [`errors/2026-04-27-recompute-admission-thrash-on-long-prompts.md`](2026-04-27-recompute-admission-thrash-on-long-prompts.md)
