# FA3 paged decode: one launch per layer, not one per row

## Context

After the KV-mirror landed, the c=8/16 decode step was the remaining wall. ITL
p50 fit `12.7 ms + 5.12·B` (MoE) and `24.6 ms + 5.19·B` (dense) — the per-row
marginal is what makes concurrency expensive, and it was identical on two models
whose KV bytes per token differ 3.2×, so it was not bandwidth.

An nsys capture at c=16 named it: 34,212 FA3 launches over ~218 decode steps =
157 per step against **10** full-attention layers, i.e. one per row. Each took
305 µs (sd 2.8) for 71.7 MB of KV — 17× the 18 µs HBM roofline, because
grid = splits(8) × kv_heads(2) × 1 row = 16 CTAs on 78 SMs. 48.8 ms of a 94.6 ms
step: 52%. MoE expert kernels were the other 40%.

## What Worked

The loop's stated reason was that FA3 zeroes the page stride when `seqused_k` is
set. It does not — only `cu_seqlens_k` drops the K/V batch strides
(`flash_api.cpp:105-108`). `seqused_k` is exactly the per-row KV extent a paged
batch needs.

So the shim takes a batch: q/o packed `[total_q, h, d]` behind `cu_seqlens_q`
(rows may differ in query length, so decode and spec verify collapse into one
call), per-row KV from `seqused_k`, and a rectangular page table strided by
`page_table_batch_stride`. Both pointers are optional — null keeps the contiguous
slot-cache lane on the non-varlen b=1 path.

The trap: varlen makes the launch template run `prepare_varlen_num_blocks` itself
(`flash_fwd_launch_template.h:163`), and that kernel **writes** through
`num_splits_dynamic` / `num_m_blocks` / `varlen_batch_idx` / `num_nheads_in_l2`.
Passing a 1-element semaphore and nulls is an illegal access, not a fallback —
serve died at `flash_fwd_launch_template.h:193` and the needle gate read
exact=0 NONDET at every length. One caller-owned i32 buffer now carves all four
vectors plus the semaphore, sliced as `flash_api.cpp:995-1027` slices its tensor.

`PageMeta` gained the rectangular page table and a device copy of `kv_lens`. The
ragged `kv_indices` stays: the quant/turboquant decode kernels derive their page
count from the `kv_indptr` delta, and rewriting their ABI to serve FA3's calling
convention would make four consumers pay for one.

## Measurement

Matched, same GPU, same binary lineage, `Qwen3.6-35B-A3B-FP8`, 1×H20,
48 req/point. ITL p50 is the steady-state step (tail-free).

| metric | per-row | batched | Δ |
|---|---:|---:|---|
| ITL p50, c=8 | 53.64 ms | **37.21 ms** | 1.44× |
| ITL p50, c=16 | 94.61 ms | **59.51 ms** | 1.59× |
| TPOT, c=8 | 87.90 ms | **42.61 ms** | 2.06× |
| TPOT, c=16 | 105.14 ms | **71.24 ms** | 1.48× |
| decode tok/s, c=16 | 9.5 | **14.0** | 1.47× |

Step model `12.7 + 5.12·B` → `14.9 + 2.79·B`: the **per-row marginal falls 1.83×**
while the intercept is flat, which is the shape a deleted per-row launch should have. c=1 is unchanged (16.03 → 16.04 ms) — one row is one launch either way.

The after-capture confirms the mechanism, not just the wall clock: FA3 launches
**34,212 → 3,049** over the same 35 s window, and steps went 218 → 305. That is
3049/305 = **10.0 launches per step against 10 full-attention layers — exactly
one per layer.** Per launch 305 µs → 1,571 µs while covering 16 rows instead of
1: 3.1× better per row, and 5.5× off the 287 µs batch roofline where it was 17×.
FA3's share of GPU time falls 44.3% → 19.6%, leaving the MoE expert kernels
(`dsv4_fp8_grouped_{down,swiglu}_decode`) at **53.9%** as the next item.

Gate: `needle_gate.py 512,4096,16384,32768 3 0.0` exact=3 DET at every length.

An occupancy-only fix was measured first and is strictly worse:
`--qwen35-fa3-decode-splits` 8→32 buys c=16 TPOT 105.14 → 70.71 ms (1.49×) and
still pays 16 serialized launches and 16 combines. Batching subsumes it.

## Open after this

- **MoE expert decode is now the wall** — `dsv4_fp8_grouped_{down,swiglu}_decode`
  at 53.9% of GPU time, 1,025 µs/layer against an ~89 µs weight-read roofline
  (R=128, ~113 active experts, FP8, 4 TB/s). ncu: DRAM 5-12%, SM 20-25%, L2
  5-9%, occupancy 20-27% against a 25-37.5% theoretical ceiling set by 72/86
  registers, waves/SM 105. Neither bandwidth- nor compute-bound — latency-bound
  with occupancy too low to hide it.
  Three replacements measured and rejected: the batch kernels are 10.8× worse
  at c=16 (`--qwen35-moe-decode-kernel false`); DeepGEMM masked FP8 exists
  upstream and is the sm90 best practice, but its band is O(E) — 256 experts at
  EP=1 means 32,768 padded rows for 128 real ones and a 524k-block pack grid;
  DeepGEMM contiguous pads per group the same way. Both grouped layouts assume
  few experts with many rows each. The remaining path is the hand kernel itself:
  cp.async/TMA pipelining, registers to ≤64, persistent tile scheduler.
- **Dense carries a prefill-blocking tail MoE does not** — 17.3% of inter-token
  gaps exceed 3× p50 and carry 55.4% of decode wall at c=16 (p90 426 vs p50 66.6
  ms), against 6.1%/21.7% for MoE. Nothing regressed: the steady-state step got
  1.62× faster and the spikes did not, so dense TPOT moves 1.35× while its p50
  moves 1.62×. Scheduler-side, not kernel-side.
- **Dense loses 2/128 requests per point** (never finish, not errors). MoE is
  128/128. Unexplained.

## Rule

**A per-row loop around a batched kernel is a claim about the kernel's API —
check the API, not the comment.** The loop had a specific, plausible,
wrong reason written above it, and it survived long enough to cost 52% of the
decode step. Reading `flash_api.cpp` took ten minutes; the comment had been true
of nothing.
