# Online-softmax decode-attention kernel — measured null at rollout=64

**Date**: 2026-05-29
**Predecessor**: [`docs/research/2026-05-28-opd-rollout-perf-208s-bottleneck.md`](2026-05-28-opd-rollout-perf-208s-bottleneck.md)

## TL;DR

Committed `causal_sdpa_decode_gqa_cache_online_f32_hd256` (cb9b6d73) —
the online-softmax + warp-reduce drop-in for the legacy two-pass
`causal_sdpa_decode_gqa_cache_f32`. Measured A/B at rollout=64 shows
**bit-identical KL** and **<0.3% wall-clock difference** (within noise).

The kernel is correct but does not provide a measurable speedup at the
shape that matters today. The 0.0099·n² quadratic term in the OPD
rollout perf fit is **memory bandwidth** on the KV cache reads, not
algorithmic. Online softmax saves the serial softmax loop on `tid==0`
in the legacy kernel, but at n=64 that loop is ~1μs/head — invisible
against the ~60s/step student_rollout total.

## A/B data (rollout=64, n_gen=64, sample-prompts.jsonl, RTX 4070 Ti SUPER)

|              | ONLINE (default) | LEGACY (`ARLE_AUTOGRAD_DECODE_ATTN_LEGACY=1`) | Δ |
|--------------|-----------------:|----------------------------------------------:|--:|
| step 1 cold student_rollout | 62.64 s | 62.56 s | +0.13% |
| step 2 warm student_rollout | 61.53 s | 61.32 s | +0.34% |
| step 2 train_kl | 8.168259955710e-5 | 8.168259955710e-5 | **identical** |
| step 2 heldout_kl | 7.917464972707e-5 | 7.917464972707e-5 | **identical** |

Numerical parity is exact (bit-for-bit KL across 4 prompts × 28 layers
× 14 heads × 64 attention calls). Wall-clock difference is sign-flipped
and inside noise on both step 1 and step 2.

Raw: `runs/2026-05-29-online-attn-bench/run.log`.

## Why the algorithmic improvement didn't help

The original analysis in tick 7 said the legacy kernel does "two passes
over visible" — implying 2× K HBM reads. Re-reading
`crates/autograd/src/backend_cuda/kernels/attention.cu:89-176` shows
this is **not** what the legacy kernel does:

1. **One K read pass** (lines 128-147): for each pos in [0, visible),
   load K row + Q-dot reduce + write score to shared mem.
2. **Single-thread softmax** (lines 149-164): tid==0 loops over the
   already-resident-in-shared-memory `scores[visible]` array. No HBM.
3. **One V read pass** (lines 167-175): for each pos, load V row +
   weighted accumulate into output.

Total HBM traffic per kernel call: 1 K read + 1 V read. Same as the
online softmax variant.

The online softmax's actual algorithmic improvement is to **merge
passes 1+2+3 into a single fused loop**, with running max + denom +
numerator in registers. Wins:
- No `scores[visible]` shared memory buffer → smaller per-block
  shared mem footprint → better occupancy at large `visible`
- No `tid==0` single-threaded softmax loop → no serialization at
  large `visible`
- All threads active throughout (vs ~half-idle in the legacy QK dot
  at `head_dim=256`, `BLOCK=256`, no — actually 256/256=1 elem/thread,
  full utilization)

At `visible=64`, the single-threaded softmax loop runs `~64 expf`
calls ≈ 1μs serial work on one thread. The kernel's overall work is
on the order of 30-50μs per call (dominated by HBM K+V loads).
Eliminating 1μs of serial work in a 30μs kernel = ~3% kernel-level
speedup at best; measured at <1% (within noise).

## The real bottleneck — KV HBM bandwidth

The fit `student_rollout(n) = 0.31n + 0.0099n²` has its quadratic term
dominated by HBM traffic for K and V cache reads:

- At step t, the kernel reads `t × head_dim × sizeof(f32) × num_kv_heads`
  bytes for K, plus the same for V.
- Qwen3.5-0.8B: head_dim=256, num_kv_heads=2, → 4 KB per layer per step.
- Per step total: 28 layers × 4 KB × 2 (K+V) = 224 KB at t=1.
- Per step at t=128: 224 × 128 KB = 28 MB.
- Across 128 rollout steps with growing t: ~ 1.8 GB cumulative reads.
- @600 GB/s HBM bandwidth: ~3 ms bandwidth-only. But measured 169 s
  quadratic cost at n=130. There's ~50,000× discrepancy — meaning
  either (a) actual achieved bandwidth is much lower, or (b) per-call
  launch overhead + sync compounds.

Most likely (a): the f32 KV cache + scalar per-thread loads + non-
coalesced layout reduce effective bandwidth to ~10 GB/s, giving
~60 ms per quadratic-contribution-of-1-token-of-context. At 128
average tokens of context: 60 × 64 × 128 ≈ 0.5s/step quadratic
contribution. Still doesn't fully account for the 169s observed, so
there's likely additional launch/sync overhead per step too.

## Path forward — BF16 KV cache (the original BF16 plan)

The committed kernel includes the template parameter `TKV` for a
BF16 variant (the bf16 wrapper was stripped in
[commit, this file] because NVRTC needs explicit `cuda_bf16.h` setup;
defer to follow-up). The proper next step:

1. Add a BF16 KV cache mode for rollout phase
   (`Qwen35LayerKvCache::{k,v}_bf16: Option<TensorId>`).
2. Cast K/V projection output to BF16 once before writing to cache;
   keep F32 only for tape-enabled backward path.
3. New backend method `causal_sdpa_decode_gqa_cache_bf16` that
   widens BF16 → F32 at load time inside the kernel.
4. Numerical parity test: BF16 cache vs F32 cache on a 60-step v8-
   equivalent run; assert |Δacc| ≤ 1pp on MMLU/GSM8K.

Expected speedup: **~2× on the quadratic term**. At rollout=128:
total step 310s → ~225s (45s saved on quadratic). At rollout=256:
1200s → ~720s.

## Rule

- **Bandwidth-bound kernels don't win from algorithmic shape changes
  alone.** Tile-friendly access patterns and dtype reductions are the
  levers. The "online softmax wins X% on attention" intuition from
  long-context prefill doesn't transfer to short-decode single-query
  attention — the legacy kernel already lives in shared mem, and the
  HBM traffic is the same either way.
- **Validate algorithmic optimization claims with measurement before
  shipping a wins entry.** This entry is *not* a wins entry — it's a
  research note documenting that the work was correct but didn't deliver
  measurable speedup at the workload shape that matters.
- **Keep the online kernel committed even though null at rollout=64.**
  At rollout ≥ 1024 the single-threaded softmax loop in the legacy
  kernel starts to bite (~20μs serial per call × 28 layers × 14 heads
  × 1024 steps = ~8 s/step), so the online path will eventually
  matter — just not at the current rollout sizes.

## Rollout=128 follow-up (same session, 2026-05-29)

`runs/2026-05-29-online-attn-bench/rollout128_online.log`:
- step 1 cold student_rollout = 206.52 s (vs v4 baseline 208 s = **−0.7%**)
- step 2 warm student_rollout = 201.71 s (vs v4 baseline ~205 s = **−1.6%**)

The online kernel scales to a measurable ~1-3% advantage as `n_gen`
grows, consistent with the analysis: the serial `tid==0` softmax loop
in the legacy kernel contributes O(visible) per-call serial work, and
that becomes visible at large n. But the magnitude is small (10s of
microseconds per call × tens of thousands of calls).

**BF16 KV cache is the only realistic path to 2× rollout speedup.**
Starting that work in the next commit.

## Pending

- BF16 KV cache implementation as outlined above.
