# The anchor's biggest line is at ~90% of FP8 peak — the GEMM lever is closed, CUDA, 2026-08-08

> Status: **measurement, no runtime change.** Read off the existing `nsys`
> anchor capture; no new GPU time. It closes the largest open item in the perf
> chain by showing there is nothing there.

## Problem

`sm90_fp8_gemm_1d2d_impl` is **57.7% of all kernel time** on the anchor (16.51 s
of 28.60 s) — the single largest cost in the workload ARLE targets. The perf
chain priced it from §1.1, "199 / 189 TFLOPS, 64–67% of peak — leave alone",
which was measured at **33K cold, single request**, not at the served shape.
With a third of the machine unaccounted at the served shape, "leave alone" was
resting on the wrong measurement.

## Method

The `nsys` c=16 anchor capture (`70760bc09`, `/host/c16decomp/c16.sqlite`). The
launch sequence is exactly periodic, so each GEMM is identified without NVTX by
the kernel it sits between and the dimension of the `pack_quantize` that feeds
it. 14,094 launches, one grid shape — (78, 1, 1), one persistent block per SM,
which is DeepGEMM's design rather than a starved grid — resolve into **four
GEMMs per layer over 176 prefill chunks of 2048 tokens**:

```
pack K=5120   -> GEMM  503.7 us   attention out_proj   (K 6144 -> N 5120)
pack K=5120   -> GEMM 2646.8 us   gate_up              (K 5120 -> N 34816)
                 silu_mul (68, 2048)
pack K=17408  -> GEMM 1409.0 us   down_proj            (K 17408 -> N 5120)
pack K=5120   -> GEMM 1256.7 us   linear-attn in_proj
                 split2 -> conv1d -> fq_fwd
```

`silu_mul` grid (68, 2048) pins it: 68 × 256 = 17408 = `intermediate_size` per
token over a 2048-token chunk, so `gate_up` is the fused 2×17408 projection.
Config confirms `hidden_size` 5120, `intermediate_size` 17408, 64 layers.

Duration clusters corroborate the period: every cluster count is an exact
multiple of 176 (4, 8, 20, 1, 10, 9, 12, 4, 12 → 80 GEMMs per chunk).

## Result

| GEMM | share of FP8 GEMM | TFLOPS | of ~296 FP8 peak |
|---|---:|---:|---:|
| `gate_up` | **33.9%** | 275.9 | **93.2%** |
| `down_proj` | 24.2% | 259.1 | 87.5% |
| attention `out_proj` | 8.7% | 255.8 | 86.4% |
| linear-attn `in_proj` | 21.6% | — | N not resolvable from the trace |

```
gate_up  2 x 2048 x 5120 x 34816 = 7.301e11 FLOP / 2.6468 ms = 275.9 TFLOPS
down     2 x 2048 x 17408 x 5120 = 3.651e11 FLOP / 1.4090 ms = 259.1 TFLOPS
out_proj 2 x 2048 x 6144 x 5120  = 1.289e11 FLOP / 0.5037 ms = 255.8 TFLOPS
```

MLP alone (`gate_up` + `down_proj`) is **69.7% of FP8 GEMM time and ~40% of all
GPU kernel time** on the anchor.

## What it changes

**The largest lever in the project is closed, because it is already at the
floor.** 87–93% of FP8 peak leaves nothing for a kernel to win. The only ways
to reduce this cost are to do less matrix work — prefix-cache hit rate,
sparsity, effective context length — not to do it faster.

§1.1's verdict ("leave alone") was right; its number was not. 64–67% was the
single-request 33K-cold shape; the served c=16 shape runs 20–29 points higher.

Remaining prefill kernel candidates are FA3 prefill (16.3%) and
`pack_quantize` + norms (12.3%), neither decomposed, and together half the size
of the line that is already at the floor.

## Learnings

**A number measured at one shape governing a decision at another — the third
instance in two days, and this one in the direction of overstating a lever.**
The draft attention was understated 7× (4.3% read as the share where it is
30.5%); the FP8 GEMM was understated on efficiency (64–67% read where it runs
93%). Same defect, opposite signs. The provenance table exists because of the
first two; this row is now in it.

**A periodic launch sequence identifies kernels without instrumentation.** No
NVTX, no rebuild, no GPU time: the repeating order plus the `pack_quantize`
input dimensions plus one activation kernel's grid was enough to name four
GEMMs and score three of them against peak. Check the trace on disk before
booking a machine.
