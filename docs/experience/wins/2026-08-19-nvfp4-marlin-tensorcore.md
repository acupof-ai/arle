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
