# Qwen3 / Qwen3.5 prefill gap — single systemic root cause: `CONTIGUOUS_KV_TOKENS=512`

## TL;DR

Both the **TTFT gap** (100–510ms vs sglang at seq_len=4096) and the
**peak-throughput gap** (−46% Qwen3.5 vs sglang) trace to **one
architectural constant**:

```rust
// infer/src/scheduler/cuda/core.rs:21
pub(super) const CONTIGUOUS_KV_TOKENS: usize = 512;
```

This constant serves two coupled purposes:

1. It sizes the **per-slot contiguous K/V working buffer** allocated in
   `KVCache::init_if_needed` — `cache_size = num_kv_heads * max_seq_len
   * head_dim`, `max_seq_len = CONTIGUOUS_KV_TOKENS` (set at
   `scheduler/cuda/core.rs:304`).
2. It **caps every prefill chunk** to ≤ 512 tokens to prevent overflowing
   that buffer (`scheduler/cuda/core.rs:886` —
   `.min(CONTIGUOUS_KV_TOKENS)`).

Because our prefill attention kernel
(`SinglePrefillWithKVCacheDispatched`, HD128 + HD256 variants in
`crates/cuda-kernels/csrc/attention/flashinfer_prefill*.cu`)
consumes a **contiguous** K/V tensor, the contiguous buffer size is a
hard ceiling on chunk size. sglang sidesteps this by calling
`BatchPrefillWithPagedKVCache` — paged prefill needs no contiguous
scratch, so sglang has no equivalent cap and prefills a 4096-tok prompt
in one forward.

## Evidence chain (files + lines)

| Observation | File:Line |
|---|---|
| Constant definition + rationale comment | `infer/src/scheduler/cuda/core.rs:18-21` |
| Contiguous buffer sizing uses the constant | `infer/src/scheduler/cuda/core.rs:304` (`state.set_max_seq_len(CONTIGUOUS_KV_TOKENS)`) |
| Allocation is `num_kv_heads × max_seq_len × head_dim` per layer per dir | `infer/src/model/kv_cache.rs:98,105-106` |
| Chunk-size getter caps with the same constant | `infer/src/scheduler/cuda/core.rs:886` |
| Prefill forward consumes a contiguous K/V scratch | `crates/cuda-kernels/csrc/attention/flashinfer_prefill.cu:92`, `flashinfer_prefill_hd256.cu:66` |
| After each chunk, KV migrates to paged pool | `infer/src/scheduler/cuda/prefill.rs:264` |
| Timing breakdown: 107ms/chunk × 8 chunks = 856ms ≈ 820ms observed | `docs/experience/wins/2026-04-17-qwen35-prefill-timing-breakdown.md` |
| Per-chunk compute is fine (0.193 ms/tok vs sglang 0.150); compounding factor is 8× setup | same |

## Why this one constant explains both humps

### TTFT hump (+100–510ms)
A 4096-token prompt fires **8 sequential prefill forwards** at
`chunk_size = 512`. Each forward repays the full setup cost:
embedding (~1ms), scratch alloc (~1ms), 32-layer forward (~99ms),
LM-head/logits (~5ms), plus the scheduler cycle/migration between
chunks (~5–10ms). 8× this cadence adds up to 80–100ms of overhead that
sglang avoids by running the whole prefill in one forward call. This
matches the 100ms single-request 4096-tok gap measured externally in
`2026-04-17-ttft-scaling-infer-vs-sglang-l4.md`.

### Peak-throughput hump (−46%)
At high concurrency, every in-flight prefill still chunks at 512, so
the 627-kernel launch storm fires **N× more often** than sglang's
single-shot prefill. Under continuous batching, the scheduler spends
most of its time in kernel submission rather than compute, which caps
steady-state throughput. CUDA Graph capture (the P1 plan) targets the
per-chunk launch overhead; chunk elimination targets the number of
chunks.

**Both fixes work on the same bottleneck — kernel-launch cost — at two
different granularities.** Lifting the cap reduces the number of
launch episodes; graph capture reduces the cost of each launch.

### MIXED_PREFILL_CAP=256 regression is consistent with this theory
The 2026-04-17 `-mixed256` experiment regressed −14% throughput because
raising mixed-prefill cap to 256 pushed `max_tokens` past the decode
CUDA-graph capture window (74), so decode fell off the replay path.
The TTFT p99 regression at 0.135 r/s (+31%) came from the longer
single mixed step stalling decode while the prefill chunk ran.
**Neither effect is prefill-chunk-cap-related** — that
experiment moved a different knob. It's not counter-evidence for this
diagnosis.

## The systemic fix — two coherent paths

### Path A (tactical, 1 file + bench): raise `CONTIGUOUS_KV_TOKENS`

Change `CONTIGUOUS_KV_TOKENS = 512` → `2048`. The cap on chunking
becomes 2048; single-forward prefill for any prompt ≤ 2048 tokens.
Agent workloads (user's stated target) are dominantly ≤1024 tokens,
so this covers 100% of the short-prompt regime we already beat sglang
on, plus most mid-length prompts.

**Memory math** (Qwen3-4B, 36 dense layers, num_kv_heads=8, HD128):
- Per slot: `8 × 2048 × 128 × 2(K+V) × 36 × 2bytes = 288 MB`
- 10 slots: `2.88 GB` contiguous
- Delta vs 512: +2.16 GB stolen from paged pool

Qwen3.5-4B has only 8 full-attn layers × HD256 — cost is comparable or
lower. On L4 24 GB with 88% mem-fraction, 2.88 GB is affordable.

At 2048, a 4096-token prompt still chunks at 2 chunks instead of 8 →
saves ~60 ms TTFT on long prompts, fully eliminates chunking on
≤2048-tok prompts (the dominant agent workload). Expected peak-
throughput gain: modest (5–10%) because per-chunk kernel launches
still dominate when the prompt fits in one chunk.

Risk: steals ~2 GB from paged pool → ~10k fewer cached tokens on L4.
For agent workloads that rely on prefix caching this may hurt cold
cache. Must measure.

### Path B (structural, multi-file refactor): switch to paged prefill

Call `BatchPrefillWithPagedKVCacheDispatched` instead of the
single-/contiguous-variant. Eliminates the contiguous buffer entirely,
aligns our prefill with sglang's architecture, removes the
chunk-cap root cause forever. Prefill writes directly to paged pool
(like decode does today via `decode_prep_paged`).

Benefit: no chunking cap at all; capable of multi-request prefill
batching (another sglang advantage we can't currently exploit); closes
the last structural gap with sglang.

Cost: refactor `forward_prefill` + ops/attention.rs paths for HD128
and HD256 to consume paged KV layout; update
`migrate_kv_range_to_paged` call sites (likely obsoleted); regenerate
baselines. 5–8 files touched. Per CLAUDE.md §Approach-first: requires
explicit user approval before starting.

### Recommended sequence

1. **Do A first** (one-line constant + bench). Fast signal on whether
   the diagnosis is correct. If a 2048 cap closes ≥50% of the TTFT
   gap and doesn't starve the paged pool, the theory is validated.
2. **In parallel, keep P1** (piecewise CUDA Graph capture of
   linear-attention groups) as the peak-throughput lever — it's
   orthogonal to A and targets different kernels.
3. **Defer B** until A + P1 have landed and measured. If together they
   close ≥80% of both humps, B may not be worth the refactor risk. If
   not, B is the long-term path.

## What NOT to do

- Don't revive the "allocation hoisting" plan as a standalone perf win
  — timing shows it's 1% of prefill time
  (`2026-04-17-qwen35-prefill-timing-breakdown.md`). It remains valid
  only as a *prerequisite* for full-forward CUDA Graph capture.
- Don't push `MIXED_PREFILL_CAP` — it's not this lever
  (`2026-04-17-bench-guidellm-qwen3-4b-infer-l4-mixed256.md` documents
  the regression).
- Don't plan-cache FlashInfer — sglang itself doesn't
  (`docs/plans/flashinfer-planned-prefill.md` closed this out).

## Rule

When two perf gaps seem independent but their numbers fit the same
equation — **look for one constraint that sets both**. Here, the 8×
TTFT setup overhead AND the 8× kernel-launch storm-per-prompt are
both `4096 / CONTIGUOUS_KV_TOKENS`. One constant drives both.

Before tuning individual scheduler knobs, trace the **buffer sizing**
those knobs feed into. Scheduler constants that cap batch / chunk
shapes almost always do so because a downstream fixed-size allocation
can't grow.
