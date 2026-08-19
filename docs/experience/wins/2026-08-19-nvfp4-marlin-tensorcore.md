# NVFP4 on the Marlin tensor-core kernel — CUDA, 2026-08-19

> Status: Shipped

## Goal

Get NVFP4 decode past FP8 on an H20. The two checkpoints are architecturally
identical — hidden 5120, intermediate 17408, 64 layers, 24 heads, 4 KV heads,
head_dim 256, attn_output_gate, vocab 248320, every field matches — so the
comparison isolates the quantization format.

## Why the scalar GEMV could not get there

The hand-written FP4 GEMV had been taken from 86.19 to 11.46 ms/step dense_ffn
across two decode rewrites (constant table -> bit manipulation -> PRMT byte
lookups, `cb109750e` and `5185ce517`). At that point it was within 12% of its
own instruction floor at ~7.8 integer instructions per 4-bit weight, and ncu
showed it against two walls at once: instruction issue 84.6%, L1 wavefront
87.1%. Unroll x2/x4, prefetch, and cp.async double-buffering all measured
slower cold. A variant that cut L1 from 87% to 34% did not get faster, because
the issue wall did not move.

The reason FP4 could not simply ride its byte advantage: both formats compute
the same N*K MACs, and the FP4 kernel ran at 21% of DRAM bandwidth against
FP8's 43.5%. Fewer bytes converts to speed only when bandwidth binds. On top of
that, sm_90 has a hardware `cvt.e4m3` for FP8 (0.65 instructions per weight, on
the FMA pipe) and no FP4 conversion instruction at all.

## What was already in the tree

The vendored Marlin template supported NVFP4 and nothing instantiated it:

- `marlin/scalar_type.hpp:306` — `kFE2M1f`
- `marlin/marlin_template.h:489` — `is_8bit_scale = w_type == kFE2M1f`, which is
  exactly NVFP4's FP8 E4M3 group scale
- `marlin/dequant.h:391` — `dequant<nv_bfloat162, kFE2M1f.id()>`
- the `s2` global-scale parameter, already threaded through the W8A16 shim

group_size 16 needed no special handling: in Marlin it is `group_blocks = 1`,
one 16x16 block. The concern that Marlin requires group 128 did not hold.

## Results

c=1 decode, 1xH20, `--kv-cache-dtype fp8`, no spec, **profiling OFF**:

| | decode | dense-MLP bytes/layer |
|---|---:|---:|
| NVFP4 scalar GEMV (`5185ce517`) | 52.3 tok/s | 150.4 MB |
| NVFP4 Marlin (`25a87ad2a`) | **57.9 tok/s** | 150.4 MB |
| Qwen3.6-27B-FP8 | 57.6 tok/s | 267.5 MB |

NVFP4 matches FP8 while reading 56% of the weight bytes. +11% over the scalar
path.

`ARLE_CUDA_PROFILE=1` costs 66-73% of throughput (a `cudaEventRecord` pair per
op, 192 per step), so every figure here is measured with it off.

Correctness: needle 512/4096 x3 = 6/6 exact and deterministic. Marlin
accumulates in a different order than the scalar GEMV and the result holds.

## Quality

Standard MMLU/GSM8K could not run: this pod cannot reach huggingface.co
(`Errno 99 Cannot assign requested address`) and has no cached copy. The
failure is in `datasets`, not in inference — the harness's own client is stdlib
urllib.

Instead, an offline probe of 8 tasks with checkable answers (three GSM8K items,
arithmetic, syllogism, factual recall, Python semantics), identical prompts to
both servers:

| | correct | completion tokens |
|---|---:|---:|
| NVFP4 Marlin | 8/8 | 580 |
| Qwen3.6-27B-FP8 | 8/8 | 2595 |

Same answers on every item.

The 4.5x token difference is a **model** difference, not a format or kernel
one — Qwen3.8 reasons far more briefly than Qwen3.6 on these prompts. It is not
attributable to this work and must not be quoted as a speedup. It does mean the
two servers finish this particular suite in 10.0 s and 45.1 s respectively,
which is a real user-facing gap arising from the checkpoints, not the runtime.

## What changed

`marlin_w8a16.cu` becomes `marlin_gemm.cu` and instantiates both weight types in
one translation unit — `get_marlin_kernel` builds both regardless, so splitting
them would only double the ~5 min nvcc cost. The loader repacks NVFP4 weights
into the Marlin layout at load time. `try_fp4_marlin_gemm_batch` returns
`Ok(false)` for anything unrepacked (unaligned shape, group_size != 16), so the
scalar GEMV remains the fallback rather than a hard failure.

## Marlin only wins below M ~= 1024

Measured against the shipped dequant(FP4->BF16) + cuBLAS prefill path, cold:

| M | gate_up | down |
|---:|---:|---:|
| 1 | -96.1% | -95.8% |
| 256 | -57.1% | -58.3% |
| 1024 | -4.0% | -9.9% |
| 1536 | **+10.5%** | **+4.0%** |
| 2048 | **+21.5%** | **+12.4%** |

`SchedulerConfig::chunked_prefill_size` defaults to 2048
(`infer-core/src/lib.rs:106`), which is exactly where Marlin loses. Routing all
M through Marlin trades a decode win for a 12-21% TTFT regression on the dense
MLP, so `try_fp4_marlin_gemm_batch` gates on `x.seq_len <=
QWEN_FP4_MARLIN_MAX_M` (1024) and prefill keeps the dequant+cuBLAS path.

This also means `repack_for_marlin_fp4` must keep `qweight_u8` / `qscale_fp8` /
`scale_f32` resident rather than consuming them — prefill still reads the packed
nibbles. The Marlin-layout scales live in the tail of the `marlin_packed`
allocation, costing ~1.13x the packed-weight VRAM.

## Why a naive wiring returns all zeros

Marlin's weight dequant runs with `dequant_skip_flop=true`, leaving a `2^-126`
factor, and its FP8 scale dequant leaves `2^-120`. Their product underflows bf16
to exactly zero — measured as `nonzero 0/256` before the fix. Upstream avoids
this by not storing raw E4M3: `nvfp4_marlin_process_scales` re-encodes each
scale to an S0E5M3 field (the high byte of `f16(scale * 2^7) << 1`) so the scale
dequant returns `scale * 2^7`, and folds the remaining `2^119` into the global
scale. Confirmed on-device with a probe kernel (E2M1 1.0 -> 1.175e-38 = 2^-126;
E4M3 1.0 -> 7.523e-37 = 2^-120) and against vLLM's `marlin_utils_fp4.py`.

## Cold microbench, per shape

Harness derived from the existing `fp4mlp/coldbench.cu` with Marlin as one more
variant, so buffers, the 8-copy L2-defeating rotation and the f32 anchor are
identical:

| shape | scalar | Marlin | |
|---|---:|---:|---:|
| gate_up N=34816 K=5120 | 100.02 us | 76.59 us | -23.4% |
| down N=5120 K=17408 | 49.69 us | 40.70 us | -18.1% |
| per-layer dense MLP | 149.71 us | **117.29 us** | **-21.7%** |

Bandwidth 1004 -> 1288 GB/s (25% -> 32% of peak), reproduced across three runs.

Numerics against an f64 reference at m=1: Marlin's `max|err|/rms` is 1.005e-2 at
gate_up against the scalar kernel's 1.382e-2, and 1.055e-2 at down for both.
Mean `out/ref` 0.99999. Marlin is equal or better than the scalar path it
replaces.

## Config search: blocks-per-SM was pinned to 1

`determine_exec_config` walked a tile table and filtered it through
`is_valid_config`, then returned the first survivor as `{1, th_config}`.
blocks_per_sm was never anything but 1, and the commented-out
`m_tiles/n_tiles/k_tiles` lines show upstream left the selection unfinished.

ncu at the decode shapes showed why that costs: issue slots 56%, ALU 52.6% of
peak (Hopper's narrowest pipe, 2 warp-inst/cycle against the FMA pipe's 4, and
where the FP4 unpack lands), achieved occupancy 12.5% capped by `Block Limit
Shared Mem = 1` — Marlin opts into the full 232 KB while the tile needs ~45 KB.
One block cannot fill the issue slots it holds shared memory for.

The search now covers blocks_per_sm too, with the per-block budget
`max_shared_mem / bps - 1024` (the split `marlin_mm` already applies at launch),
ranked by waves = `div_ceil(n_tiles, sms * bps)`, ties to the larger tile. The
winner falls out of the shape; nothing is pinned.

| | decode |
|---|---:|
| NVFP4 Marlin | 57.7 tok/s |
| NVFP4 Marlin + config search | **60.2 tok/s** |
| Qwen3.6-27B-FP8 | 57.6 tok/s |

NVFP4 leads FP8 by 4.5% on identical architectures while reading 56% of the
weight bytes. Needle 512/4096 x3 = 6/6 exact.

> **SCOPE, added 2026-08-19.** This row is c=1 on an 8-token synthetic prompt
> against FP8 with speculation OFF. It does not generalise: at c>=2 the same
> build loses 5x to FP8 (matched A/B, same binary and moment), and the shipped
> FP8 configuration runs DSpark. Cause and fix:
> [errors/2026-08-19-fp8-dequant-arm-shadows-decode.md](../errors/2026-08-19-fp8-dequant-arm-shadows-decode.md).

## Speculative decode: both paths are a large net loss here

> **CORRECTED 2026-08-19.** The attribution below — "the verify forward runs
> depth+1 tokens through the recurrent linear attention on 48 of 64 layers" — is
> wrong. Marlin's per-call cost is flat in M (34816x5120: 68.9 us at both M=1 and
> M=3) and the gated-delta route change accounts for +2.1 ms, 1.5% of the delta.
> The real cause is `try_fp8_dequant_bf16_gemm_batch` firing at `M >= 2`: a spec
> verify submits M=3 (MTP d=2) or M=7 (DSpark block 6), and each one
> re-dequantises all 11.56 G FP8 params in the checkpoint to BF16 — 84.35 ms per
> forward, 84% of the M=3 quantised-GEMM budget. Measured acceptance was 35.1%,
> so acceptance was never the problem; even at 100% MTP would lose 1.65x.
> The same defect costs 5x aggregate throughput at c>=2 and crashed the server at
> 34K. Root cause and fix:
> [`errors/2026-08-19-fp8-dequant-arm-shadows-decode.md`](../errors/2026-08-19-fp8-dequant-arm-shadows-decode.md).
> The numbers in this section stand as measurements of the defective build.

| configuration | decode |
|---|---:|
| no spec | **60.2 tok/s** |
| `--spec-type dspark --mtp-draft-model Qwen3.6-27B-DFlash --dspark-block-size 6` | 16.8 (-72%) |
| `--spec-type mtp --mtp-draft-tokens 2` | 13.9 (-77%) |

MTP had already measured negative before this work (6.2 against 9.3 tok/s,
-33%). It is now *worse* in relative terms, and the reason is instructive: the
kernel work sped up single-token decode 6.5x, but the verify forward runs
depth+1 tokens through the recurrent linear attention (gated-delta scan) on 48
of 64 layers — untouched by any of it. The denominator shrank and the numerator
did not.

That also confirms the shape of what was optimized: dense MLP is now fast enough
that linear attention dominates the forward, which is a different problem from
the one this entry solves.

DSpark loads and runs correctly (5-layer DFlash drafter, block 6, taps
[1,16,31,46,61]) and is less bad than MTP, but the same verify cost applies.
Neither is worth enabling at c=1 on this model.

## What is left

57.9 against a target of 30% over FP8 (74.9 tok/s) leaves 1.29x. The bottleneck
has moved onto the tensor core and the scalar-era analysis no longer applies —
the numbers that justified "the scalar path is exhausted" say nothing about
this one. A fresh ncu pass on the Marlin kernel is the next step, not an
extrapolation from the old one.

## Learnings

The kernel that solved this had been vendored in the tree the whole time, one
`kFE2M1f` instantiation away. Three sessions of hand-optimizing the scalar GEMV
produced a real 7.5x and still could not reach what the existing SOTA kernel
gave immediately. Checking what the vendored code already supports belongs
before the optimization work, not after it.
