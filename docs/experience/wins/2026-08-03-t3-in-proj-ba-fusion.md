# T3 in_proj_b+a row-fusion: −4.7% decode ITL — CUDA, 2026-08-03

> Status: **Shipped, default path** (`4952f0df5`, #196 T3). c=1 W8A16 decode
> ITL p50 **26.31 → 25.08 ms**; with T1, **26.88 → 25.08 ms (−6.7%)** against
> the pre-#196 baseline. Greedy byte-identical to the unfused binary.

## What shipped

`in_proj_b` and `in_proj_a` (`[Vh, hidden]` each — one scalar per v-head, the
gated-delta-rule's beta/alpha inputs) load as ONE row-fused `[2*Vh, hidden]`
matrix. The step runs one GEMM into a fused `ba` scratch, then a
`split_halves` kernel writes the existing `b_proj`/`a_proj` buffers — so
conv/GDR, spec capture, and the FlashQLA prefill path see byte-identical
inputs and needed no changes. TP loads each head shard as before and fuses on
device.

## Why it was worth 1.2 ms

Each `[48, 5120]` bf16 GEMV moved 0.5 MB and took **~44 µs** — under 10 µs of
that is data. 96 launches/step of near-pure overhead collapse to 48 GEMMs +
48 trivial splits.

| arm | ITL p50 | ITL p99 |
|---|---:|---:|
| pre-#196 unfused | 26.88 | 27.46 |
| T1 (gate+up) | 26.31 | 26.99 |
| **T1+T3** | **25.08** | 25.84 |
| SGLang, same kernel + same weights | 17.07 | 18.67 |

Correctness: greedy 120-tok and 60-tok completions md5-identical to the
unfused binary. Same 32k c=1 protocol as the SGLang matched A/B (H20 GPU 6,
16 requests × 256 tokens, seed 20260416).

## Learnings

**Fusion recovered 1.8 ms of the 9.8 ms gap — the rest is the launch model,
not the projections.** T1+T3 removed 160 launches/step (1156 → ~996) and
bought 1.8 ms, tracking the ~4.9 µs/launch gap measured by nsys almost
exactly. What remains is structural: ~5.7 ms of GPU idle spread across the
still-eager ~1000 launches. No further projection fusion can touch it — T4
(whole-step decode CUDA graph under paged KV) is the only lever left with the
size to close the gap, and it is a phase-level change, not a tranche.
