# DSv4 decode comm-overlap — shared expert on comm_stream behind the MoE all-reduce (correct, token-exact, default-OFF: B=1 gain sub-noise)

**Date:** 2026-06-06. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Status:** **implemented + token-exact ×3, kept DEFAULT-OFF** (opt-in
`ARLE_DSV4_COMM_OVERLAP=1`). Perf at B=1 is **+3.6% within harness noise** (one
of three reps below baseline) → **NOT default-flipped**. This is the #1 decode
*kernel* bucket (comm = `ncclAllReduce` 16.4% + `ncclAllGather` 16.0% = 32.4% of
the clean decode window), so the lever is real; the realized B=1 standalone gain
is small and the license is deferred to the K-token EAGLE verify forward (where
the overlapped shared expert is K× larger). See
[`2026-06-06-dsv4-decode-6ms-remaining-levers.md`](../../plans/2026-06-06-dsv4-decode-6ms-remaining-levers.md)
lever 1.

## What worked (correctness)

- **Fence-orchestrated overlap on the FAST masked decode path** (`dsv4.rs` MoE
  block). The shared expert reads `normed` (not the all-reduce output) so it is
  provably dependency-free of the routed MoE all-reduce:
  1. Compute stream records a fence after `normed`; **comm_stream waits it**
     (no stale read — `feedback_private_stream_needs_stream_wait`).
  2. `dsv4_shared_expert_forward(..., &ctx.comm_stream, ...)` runs the shared
     FFN on comm_stream; its scratch is **allocated on comm_stream** (correct
     stream-ordered-alloc — cross-stream use of a ctx-stream alloc without a
     matched alloc stream would race).
  3. comm_stream records a fence after `shared`.
  4. The MoE `all_reduce_sum` runs on the **compute stream concurrently**.
  5. Before `add_batch` (needs both), the compute stream waits the shared fence.
- **`dsv4_shared_expert_forward` stream-parameterized** (`moe.rs`): added a
  `stream: &Arc<CudaStream>` param threaded to `dsv4_shared_expert` /
  `dsv4_shared_expert_pooled` / the D2D. An `Arc::ptr_eq(stream, &ctx.stream)`
  guard keeps the **default caller (`&ctx.stream`) byte-for-byte identical to
  main** — zero behavior change off the flag. All four prefill/decode call-sites
  pass `&ctx.stream` by default.
- **Decoupled from the pooled path.** The earlier attempt bundled a
  `dsv4.rs use_gpu_router` default-on flip that forced the slow pooled decode
  scratch (28.4 vs 37.6 tok/s, −20%); that was caught + reverted
  ([[reference_dsv4_pooled_decode_slower_than_masked_b1]]). This redo keeps
  `use_gpu_router = env::var_os(...).is_some()` (masked default) and drops the
  `&& use_moe_decode_scratch` coupling + the `ensure!`.

## Perf A/B (same-binary env flip, 64-tok decode, TP=8/EP=8 pod)

| run | decode_tok_s | token-exact |
|---|---|---|
| **off (masked baseline)** | **37.788** | `[344,34837,2907,…]` ✓ |
| on rep 1 | 38.467 | PASS |
| on rep 2 | **37.085** | PASS |
| on rep 3 | 41.892 | PASS |

mean(on) ≈ **39.15 = +3.6%**, but **on rep 2 < off** and the on-spread is ~13%.
Against that run-to-run noise, +3.6% is **not distinguishable from zero** on a
1-off / 3-on design → **no default flip**. The masked baseline is preserved
(37.788, not the 28.4 pooled trap), confirming the `use_gpu_router` revert took.

## Rule

The comm-overlap ceiling at **B=1** is `min(shared_expert_time, moe_AR_time)` —
both are small for one token, so the standalone saving is ~1–2% (consistent with
the +3.6% noisy measurement). The lever is **kept default-off** and correct
because its value **scales with the EAGLE/MTP verify forward**: at K>1 tokens the
shared expert is K× larger, so the overlapped compute that hides the all-reduce
grows. License the default-flip then, with a tight interleaved A/B (≥3 off / 3 on
on the same resident process), not the cold-reload 1-off/3-on harness that can't
resolve a sub-5% effect. For the *AllGather* half of the 32.4% comm bucket (EP
dispatch/combine) this overlap does nothing — that needs the one-shot/fused-EP
all-reduce (lever 1b). [[feedback_b1_decode_gpu_bound_overhead_removal_wash]]
([[feedback_matched_ab_for_small_bench_effects]]).
