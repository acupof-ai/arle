# NVFP4 GEMV decode — 7.5x on the kernel, CUDA, 2026-08-19

> Status: Shipped

## Goal

Make NVFP4 decode competitive with FP8 on an H20. The two checkpoints are
architecturally identical (hidden 5120, intermediate 17408, 64 layers, 24 heads,
head_dim 256, attn_output_gate, vocab 248320 — every field matches), so the
comparison isolates the quantization format.

## Results

c=1 decode, 1xH20, `--kv-cache-dtype fp8`, no spec, profiling OFF:

| | dense_ffn | forward_hidden | decode |
|---|---:|---:|---:|
| NVFP4 start | 86.19 ms/step | 106.70 ms/step | — |
| NVFP4 bit-manipulation decode (`cb109750e`) | 23.30 | 42.26 | — |
| NVFP4 PRMT decode (`5185ce517`) | 11.46 | 31.21 | **52.3 tok/s** |
| Qwen3.6-27B-FP8 | 9.84 | 29.22 | **57.6 tok/s** |

7.5x on the kernel. NVFP4 remains 9% behind FP8 on decode.

The per-op columns are measured under `ARLE_CUDA_PROFILE=1`, which costs 66-73%
of throughput (a `cudaEventRecord` pair per op, 192 per step). They are valid
for attribution between ops and invalid as throughput. The decode column is
measured with profiling off.

## What each step did

**Constant-memory table -> bit manipulation** (`cb109750e`). The decode was
`__constant__ float LUT[16]` indexed by the nibble: one memory read per weight
at a data-dependent address, serialising across a warp when nibbles differ.
Replaced by assembling the bf16 pattern with shifts and masks. Also removed a
runtime integer division per scale index (group_size is a power of two) and a
redundant scale reload (a 32-weight chunk spans two groups, not four).

**Bit manipulation -> PRMT byte lookups** (`5185ce517`). ncu showed the shift/mask
form pinning the sm_90 ALU pipe at 92.4% against FP8's 59.4%, costing 59.8% of
the kernel. Both bytes of the target bf16 turn out to be 8-entry functions of
`n & 7`, which is exactly PRMT's table size:

    low  byte = {00,00,80,c0,00,40,80,c0}[n & 7]
    high byte = {00,3f,3f,3f,40,40,40,40}[n & 7] | (n & 8 ? 0x80 : 0)

`__byte_perm` takes a runtime selector, so masking a packed word with
0x77777777 turns four nibbles into four table indices in one AND. Eight weights
cost 15 integer instructions — 1.875 per weight against the 11.81 ncu measured.
Indexing the high table by `n & 7` rather than by (sign, exponent) removes the
zero special case: entry 0 is 0x00 and OR-ing the sign unconditionally gives
0x8000 for code 8, which is the negative zero the reference already holds.

Both decodes were verified bit-exact — all 16 codes, all 256 packed bytes, and
for PRMT all 2^32 packed words — before landing. Needle 512/4096 x3 = 6/6 exact
and deterministic at each step.

## Why NVFP4 is still behind FP8

Two reasons, both measured.

**The byte advantage has no channel to convert through.** NVFP4 moves 150.4 MB
of dense-MLP weights per layer against FP8's 267.5 MB, but both compute the same
N*K MACs — FP4 only stores them denser. Fewer bytes becomes speed only when
bandwidth binds, and the FP4 kernel runs at 21.0% of DRAM against FP8's 43.5%.
It is bound by instruction issue, not bytes.

**sm_90 has no FP4 conversion instruction.** FP8 decodes with a hardware
`cvt.e4m3` at 0.65 instructions per weight, issued on the FMA pipe. FP4 must
synthesize the conversion from integer ops; even after PRMT that is ~7.8
instructions per weight. This is a direct consequence of Hopper lacking FP4
tensor cores.

## What was measured and rejected

- **Multi-row GEMV** (one thread accumulating 4 output rows to amortise the x
  loads): 21.3 tok/s against 23.5. ncu had already shown FP4's memory side was
  healthier than FP8's (L1 throughput 35.84% vs 85.71%, global-load stall 1.65
  vs 2.71 cycles/issue), so there was no load pressure to relieve.
- **Cold-L2 latency** as the explanation for a 32% microbench-vs-in-situ gap:
  disproven. Rotating 8 distinct weight buffers to defeat L2 changed nothing
  (100.07 -> 100.06 us), and `lts__t_sector_hit_rate` was already 1-3%. The gap
  was the profiling overhead, not the cache.
- **Unroll x2 / x4, prefetch i+1, cp.async double-buffering**: +3.9%, +4.7%,
  +2.0%, +22.1% — all slower cold. The kernel is against two walls at once
  (issue 84.6%, L1 wavefront 87.1%); relieving one alone does nothing. A variant
  that cut L1 from 87% to 34% did not get faster.
- **ILP with 4 independent accumulators**: a wash, then a 3% loss at 44
  registers when occupancy fell 96% -> 61%. ncu had ruled out the dependency
  chain (FP4 stalls *less* than FP8 on `wait`: 1.06 vs 1.61 cycles/issue).

## What is left

The scalar path is within 12% of its own instruction floor at ~7.8 instructions
per 4-bit weight, so it cannot reach FP8, let alone pass it. One variant (one
warp per row plus span-8 coalesced x) measured -12.4% on gate_up but +6.0% on
down, where N=5120 gives only 1.03 waves and the tail idles; taking it needs a
dispatch on N.

The mechanism with real headroom is the tensor core: one `HMMA.16816` is 2048
MACs. The vendored Marlin template already supports NVFP4 natively —
`kFE2M1f` in `marlin/scalar_type.hpp:306`, `is_8bit_scale` at
`marlin/marlin_template.h:489` (exactly NVFP4's FP8 E4M3 group scale), the
E2M1->bf16x2 dequant at `marlin/dequant.h:391`, and the `s2` global-scale
parameter already threaded through `marlin_w8a16.cu`. Only `kU8B128` is
instantiated today. Whether it accepts group_size 16 (Marlin traditionally wants
128) is the open question.

## Learnings

Diffing two structurally identical kernels against each other found what
profiling the slow one alone did not: the FP8 and FP4 GEMVs share their geometry
and reduction exactly, so every difference in the inner loop was a candidate,
and the ones that mattered were visible by reading them side by side.

Three of my own hypotheses were killed by measurement in this session — load
count, cold-L2 latency, and the dependency chain — each after I had reasoned my
way to it confidently. The ncu counter that settled each one took minutes; the
reasoning that produced each wrong answer took longer.
