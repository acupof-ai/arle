# Long-context decode attention on Qwen3.6-27B — where the time goes, and the order to fix it

The first long-agent baseline for ThinkingCap-Qwen3.6-27B-FP8 (1×H20, multi-turn
32k) put decode, not prefill, at the top of the cost. Prefill runs at ~3.9–4.1k
tok/s and is flat across c=1/4/8 — it is saturated and uninteresting. Decode is
77 ms/token at c=1 on a step whose KV read has a 0.52 ms roofline.

This document separates what is measured from what is read off the source from
what is still a hypothesis, and orders the work so each step is licensed by the
one before it.

## What is measured

Arm: no-spec, `bench-agent-32k-8x8.jsonl` (8 sessions × 8 turns, sha
`78c70bda…`), 64 requests/point, max_tokens 214, greedy, GPU 0. Prefix hit rate
0.9585 against the TraceLab 95.7% reference.

| c | TTFT p50 | prefill tok/s | TPOT p50 | TPOT p99 | decode tok/s/req | aggregate decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 12.4 s | 4086 | 77.0 ms | 80.9 ms | 13.0 | 13.0 |
| 4 | 9.1 s | 3951 | 95.2 ms | 8937 ms | 10.5 | 42.0 |
| 8 | 13.5 s | 3949 | 140.5 ms | 9553 ms | 7.1 | 56.9 |

`output_tokens_per_s` from the runner (4.1 / 7.8 / 8.4) is not decode speed — it
divides generated tokens by a wall clock that is mostly prefill on a 32k
workload. Decode speed is `1000 / TPOT`.

TPOT is linear in context, and the slope falls out of the cold/warm split within
each point (cold = turn 0 at ~32.4k prompt tokens, warm = later turns at ~35.0k):

| c | TPOT @ 32.4k | TPOT @ 35.0k | slope | intercept |
|---|---:|---:|---:|---:|
| 1 | 73.1 ms | 78.4 ms | 2.09 ms / 1k ctx | 5.0 ms |
| 4 | 91.1 ms | 96.5 ms | 2.13 ms / 1k ctx | 22.1 ms |
| 8 | 136.6 ms | 143.1 ms | 2.56 ms / 1k ctx | 53.6 ms |

**~73 of the 77 ms at c=1 is context-scaling.** The context-free step is ~5 ms.
Whatever is slow is slow per context token, which is attention over the KV
cache, not the MoE FFN and not the 48 gated-delta-net layers (their state is
context-independent).

TPOT p99 blowing to ~9 s at c≥4 while p50 stays at 95/140 ms is a separate
issue — queueing/preemption tail, not decode speed. It gets its own
investigation, not this one.

## What the source says

Geometry: 64 layers, `full_attention_interval 4` → **16 full-attention layers**,
`num_attention_heads 24`, `num_key_value_heads 4` (GQA ratio 6), `head_dim 256`,
bf16 KV. So **64 KB of KV per context token** and **2.1 GB read per decode step
at 32k** — 0.52 ms at 4 TB/s.

Two structural findings in
`crates/cuda-kernels/csrc/attention/fused_attention.cu`, both read directly off
the kernel, neither inferred:

**1. KV is read `gqa_ratio` times over.** The grid is keyed on the query head
(`:319`), and the KV head is derived from it (`:435`):

```c
// Grid: (num_qheads, NUM_KV_SPLITS, batch_size)
int q_head_idx = blockIdx.x;
int kv_head_idx = q_head_idx / gqa_ratio;   // 24 / 4 = 6
```

Six CTAs each walk the same KV head's entire cache independently. Real traffic
is **6 × 2.1 = 12.9 GB per step**, not 2.1 GB. Against the measured TPOT that is
168 GB/s at c=1 and 735 GB/s at c=8 — 4% and 18% of peak.

**2. The split count is a compile-time constant** (`:325`):

```c
#define NUM_KV_SPLITS 4
```

At c=1 the grid is (24, 4, 1) = **96 CTAs on a 78-SM H20**, each serially walking
8192 context tokens in 64-token tiles. This is the same shape as the DSpark
draft-attention finding earlier the same day
([2026-07-26-dspark-ragged-window-draft-attention](../experience/wins/2026-07-26-dspark-ragged-window-draft-attention.md)):
not launch overhead, too few blocks.

The two interact, and the direction matters: **fixing (1) makes (2) mandatory.**
A GQA-aware grid is `(num_kvheads, splits, batch)` = **16 CTAs** at c=1 with
splits still 4. The traffic fix alone would trade a 6× read reduction for a 6×
occupancy loss.

There is precedent for the tunable form already in the tree: the FA3 decode path
carries `qwen35_fa3_decode_splits`, runtime-settable and clamped to [2, 256]
(`crates/infer-cuda/src/runtime_flags.rs:81`). Only this batched path is pinned.

## What is still a hypothesis

- That occupancy is the binding constraint at c=1. The CTA count and the
  bandwidth gap are consistent with it, and the c=1 → c=8 move (96 → 768 CTAs,
  168 → 735 GB/s) is the right shape, but nothing here measures achieved
  occupancy or memory throughput. `ncu` decides it.
- That the ~735 GB/s at c=8 — where the CTA count is no longer starving — is a
  per-CTA efficiency ceiling rather than a second instance of the same problem.
  Candidates: unvectorized bf16 loads, page-table indirection per tile, the
  256-thread/one-output-dim mapping limiting ILP on the QK dot. Not diagnosed.
- That the 2.09 ms/1k slope is entirely attention. It is the only
  context-dependent term in the step, but it has not been isolated with a phase
  timer.

## Order of work

Each step is gated on the previous one producing a number.

**Step 0 — attribute, no code.** One `ncu` on
`fused_gqa_attention_decode_batched_kernel` at c=1 and c=8: achieved occupancy,
`dram__bytes_read.sum` (does it equal 6× the theoretical minimum?), and warp
stall reasons. Plus a context sweep (4k/8k/16k/32k, same treatment) to confirm
the slope is linear and attention-shaped rather than a threshold effect. This
step decides whether the rest of the plan is aimed at the right target.

**Step 0b — free A/B, no code.** Turn on `--qwen35-fa3-decode` and sweep
`--qwen35-fa3-decode-splits` against the baseline arm. It answers "are splits the
binding constraint" without writing a kernel, and per the adopt-official-first
rule it also answers whether a vendored decode kernel can serve head_dim 256 +
GQA 6 outright, which would delete steps 1–2 rather than implement them.

**Step 1 — GQA-aware CTA mapping.** Grid over `num_kvheads`; each CTA holds the
`gqa_ratio` query vectors for its KV head and computes that many dot products per
loaded K tile. Upper bound: 6× less KV traffic, at every concurrency, independent
of occupancy headroom. Ship with step 2 or the occupancy loss eats it.

**Step 2 — scale splits to the GPU.** Replace the `#define` with a runtime value
chosen so `kv_heads × splits × batch ≈ 2 × SM count` (≈ 39 at c=1 post-step-1,
falling to ~5 at c=8). Mirror the FA3 flag's shape — runtime-settable, clamped —
so the sweep that picks the constant is reproducible.

**Step 3 — FP8 KV cache.** Halves the dominant term again: 64 → 32 KB/token.
`decode_attention_varlen_fp8.cu` and `CudaKvCacheDtype` already exist, so this is
a serve flag plus a needle gate, not new kernel work. Sequenced last because it
trades quality for bandwidth and the first two do not.

## Gates

Steps 1–3 are runtime changes on the decode hot path: each needs a dated
`wins/` entry with a before/after `ncu` (the GPU-kernel gate), and re-measured
TPOT on this same long-agent dataset — not on short prompts, which is the regime
error this whole line of work has already made twice
([block size](../experience/wins/2026-07-26-dspark-block-size-is-a-lever-at-concurrency.md)).
Step 3 additionally needs a needle gate before any default flip.

## Rule

Report prefill and decode separately or the headline number is meaningless: the
same run that reads as "4.1 output tok/s" is a saturated 4086 tok/s prefill plus
a decode running at 4% of memory peak, and only one of those is worth an
engineer's week. And when a decode step is slow, get the KV traffic from the
grid mapping before profiling anything — a `blockIdx` keyed on the query head
under GQA is a `gqa_ratio`× read amplification sitting in plain sight.
