# The anchor window, partitioned exactly — prefill arithmetic is at the floor, the data-prep tail is not, CUDA, 2026-08-09

> Status: **measurement, no runtime change.** Same `nsys` capture as 08-08
> (`70760bc09`, `/host/c16decomp/c16.sqlite`); no new GPU time. It closes the two
> largest open items and replaces the performance model's fitted coefficients
> with an exact partition.

## Context

`docs/perf-qwen36-27b.md` carried a per-token model, `c_prefill = 251.9
µs/token`, that closed only **79%** of the anchor window. Two of its terms were
explicitly unscorable: FA3's efficiency (the persistent grid hides sequence
lengths) and the chunk count, on which two independent readings disagreed
(`silu_mul` said 33, FA3 said 61). The doc named the FA3 denominator "the
highest-value measurement in the document".

## What worked

**1. Partition instead of fit.** Every kernel was assigned to prefill or decode
by whether its start falls inside one of the seven decode windows (defined by
`silu_mul` launches with `gridY ≤ 64`, expanded ±3 ms). The result is a ledger
that sums to the window by construction:

```
wall 29,642 ms | GPU busy 28,676 (idle 966, 3.3%) | kernel 28,601
  = 28,168 prefill  +  433 decode
prefill 90,208 tokens -> c_prefill = 312.3 us/token   (was 251.9, 79% closure)
```

**2. The chunk count was two different quantities.** 33 is full 2048-token
chunks; 61 is *segments*, because a chunk spanning two requests is issued as two
launches. `33 + 8 + 8 + 3 + 3` prefill segments `+ 6` decode passes `= 61`, and
`977 = 16 × 61` full-attention launches. **44 chunks in total** — the doc's
earlier "176", inferred from duration clusters, was wrong.

**3. FA3 scored from a slope, not a denominator.** Joining each FA3 launch to the
`split_qkv` before it gives `Q`; sorting the `Q = 2048` launches by duration
gives a ladder whose step is **732 µs per 2048 tokens of KV depth**, constant to
±0.8% across nine consecutive rungs. One rung is one clean rectangle of work:

```
4 x 2048 x 2048 x 24 heads x 256 head_dim = 1.031e11 FLOP / 732 us
  = 140.8 TFLOPS = 95.1% of the 148 TFLOPS bf16 peak
```

No absolute depth needed. Mean effective KV depth follows as **18.0K**, range
5.9K–30.5K, consistent with a 32K anchor.

**4. Every GEMM resolved by `(K, M)`.** `K` from the `pack_quantize` that feeds
it, `M` from the co-located `silu_mul`, `N` from the consumer's grid
(`split2` at `64 × 256` pins `in_proj`'s N at 16384; `split_qkv` at `56 × 256`
pins `qkv`'s N at 14336, which was previously mis-read as 6144).

## Result

**Prefill arithmetic is finished.**

| term | floor µs/tok | measured | of peak |
|---|---:|---:|---:|
| `in_proj` | 27.2 | 28.6 | 95.1% |
| `gate_up` | 77.1 | 83.8 | 92.0% |
| `qkv` | 7.9 | 8.9 | 88.8% |
| `down_proj` | 38.5 | 44.6 | 86.3% |
| `out_proj` | 13.6 | 15.8 | 86.1% |
| FA3 | — | 51.5 | **95.1%** of bf16 |

GEMM + attention are **74.6% of prefill time with 1797 ms of headroom, 6.1% of
wall**. No kernel lever remains there.

**The data-prep tail is where the headroom is.**

| kernel | ms | GB | TB/s | of 3.5 |
|---|---:|---:|---:|---:|
| `pack_quantize` | 2216 | 593 | 0.27 | **7.6%** |
| `silu_mul` | 589 | 605 | 1.03 | 29.4% |
| `conv1d` | 590 | 222 | 0.38 | 10.7% |
| `split2` | 360 | 289 | 0.80 | 23.0% |
| `rms_norm_gated` | 259 | 160 | 0.62 | 17.7% |
| `gdr_fq_prep` | 207 | 107 | 0.51 | 14.7% |
| `add_native` | 117 | 46 | 0.39 | 11.3% |
| `split_qkv` | 101 | 83 | 0.83 | 23.6% |
| `rms_norm_batched` | 105 | 6 | 0.06 | 1.6% |
| | **4544** | 2111 | | floor **603 ms** |

**3940 ms of headroom — 13.3% of wall, 2.2× the entire arithmetic headroom**, in
kernels that do no model arithmetic at all.

**One mechanism explains the whole column.** `pack_quantize`
(`csrc/gemm/dsv4_deepgemm_ops.cu:89`): one 128-thread block per 128-element
quantization block, **one `uint16_t` per thread**, a shared-memory block
reduction for `amax`, then a second pass re-reading the same input to scale it.
2-byte accesses where H20 needs ≥8, a `__syncthreads` per 128 elements, and
double the necessary traffic. The rest of the tail is written the same way.

Total identified headroom rises from 12% to **22.6% of wall**, and 59% of it is
in the tail.

## The tail's gap is inflation, not stall — `ncu`, same day

A microbench of `pack_quantize` at the traced shapes (rows 2048, cols 5120 /
17408 / 6144), current form against one warp per quantization block with
`ushort4` loads and values held in registers:

```
cols= 5120   cur  88.5 us (0.36 TB/s)   vec  25.1 us (1.27 TB/s)   3.53x   mismatches=0
cols=17408   cur 319.3 us (0.34 TB/s)   vec  86.3 us (1.25 TB/s)   3.70x   mismatches=0
cols= 6144   cur 107.3 us (0.36 TB/s)   vec  29.6 us (1.29 TB/s)   3.62x   mismatches=0
```

| `ncu` metric | current | vectorized |
|---|---:|---:|
| duration | 101.4 µs | 27.6 µs (3.67×) |
| executed instructions | 46.79 M | **11.81 M (3.96× fewer)** |
| SM (compute) throughput | 81.3% | 74.9% |
| **DRAM throughput** | **5.2%** | 19.2% |
| achieved occupancy | 89.8% | 83.7% |
| executed IPC | 3.30 | 3.21 |

**The speedup equals the instruction reduction, and the memory system was never
the constraint.** Both versions issue at ~80% of SM throughput with IPC above
3.2 and occupancy near 90%; DRAM sits at 5.2% while the current kernel runs. The
gap the tail carries is `T_inflation` — instructions the implementation added —
and `T_stall` is empty.

This contradicted my own framing, which had called the tail "inflation and stall
jointly" on the reasoning that 2-byte accesses cannot keep enough bytes in
flight. They can; the kernel simply never asks for enough of them, because it
spends its cycles on address arithmetic, a block reduction, and two
`__syncthreads` per 128 elements. **The stall-class remedies — TMA, async copy,
warp specialization, deeper pipelining — would return nothing here.**

## Rule

**When the denominator is hidden, measure the slope.** FA3's grid is persistent,
so no launch carries its sequence length and the doc had declared its efficiency
unscorable. But cost is linear in KV depth, so the *difference* between two
adjacent rungs is a known rectangle of work with no unknown in it. A rate can be
recovered from a derivative when it cannot be recovered from a ratio.

**Two counts that disagree may be counting different things.** 33 vs 61 was read
as a contradiction that blocked a measurement for a day. It was chunks vs
segments. Before treating a mismatch as an error, check that both sides name the
same unit.

**A kernel at 7.6% of bandwidth is not necessarily bandwidth-bound.** The
inference "slow, and it moves a lot of bytes, so it is starved on memory" was
wrong by an order of magnitude: DRAM was at 5.2% and the SM at 81%. A low
achieved bandwidth is equally consistent with a kernel that is too busy to ask
for memory. Check SM throughput and IPC before assigning a bandwidth-class
remedy — the four gap buckets have disjoint remedies precisely so that a
misassignment is expensive.

**Partition beats fit.** The old model was a sum of point estimates and closed
79%; the missing 21% read as one unexplained effect. Assigning every kernel to a
side instead makes closure exact by construction, and the "residual" turned out
to be three ordinary things (small-`M` GEMM variants, the elementwise tail, and a
`pack_quantize` count low by 40%). A model that cannot fail to close is a model
whose gaps are visible as line items.

Related: [`2026-08-08-anchor-fp8-gemm-is-at-90-percent-of-peak.md`](2026-08-08-anchor-fp8-gemm-is-at-90-percent-of-peak.md),
[`2026-08-08-dspark-draft-attention-slot-batched.md`](2026-08-08-dspark-draft-attention-slot-batched.md).
