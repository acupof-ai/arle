# Per-channel FP8 on Marlin — CUDA, 2026-08-19

> Status: Landed, **numerics gate pending-remote** (parity harness building on the H20)

## Why

`Qwen3.8-27B-NVFP4` is a mixed checkpoint: 145 of ~200 quantised GEMMs per
forward are per-channel FP8 (all 48 linear-attn in/out_proj, all 16 self-attn
q/k/v/o, MLP on 8 of 64 layers, lm_head), and only 56 layers' MLP is NVFP4. The
earlier Marlin port covered the NVFP4 minority. The FP8 majority stayed on a
scalar batched GEMV while the comparison checkpoint's equivalents go to DeepGEMM.

That is the whole remaining asymmetry. Matched grid, same binary, aggregate tok/s:

| c | NVFP4 | FP8 | |
|---:|---:|---:|---:|
| 1 | 74.2 | 61.6 | +20.5% |
| 2 | 103.3 | 100.2 | +3.0% |
| 4 | 165.3 | 196.6 | −15.9% |
| 8 | 216.8 | 360.6 | −39.9% |
| 16 | 235.9 | 636.7 | −63.0% |

NVFP4's step cost grows 4.5× over 16× concurrency against FP8's 1.45×. The GEMV's
own comment names the ceiling: `TILE == B` means register pressure scales with the
tile, and above B=8 it falls back to a fixed-8 tile with `grid.y = ceil(B/8)`,
re-reading the weight. Marlin is flat in M — 68.9 us/call at both M=1 and M=3 on
34816×5120.

## What was already in the tree

`BIGGROUP_GET_IF(host::kFE4M3fn)` (`gptq_marlin.cuh:416`) is **already
instantiated**, covering `group_blocks ∈ {−1, 8}`; `−1` is Marlin's channelwise
mode (`group_size = -1` at `:614`). `dequant<nv_bfloat162, kFE4M3fn>` exists at
`dequant.h:321`, and `:577` already admits the type. So this adds no template
instantiation and no nvcc cost — only a host entry, a repack, and routing.

## The two things that had to be right

**The 2^120 fold.** `dequant_skip_flop = !is_int_type` (`marlin_template.h:328`)
and `is_int_type` covers only the kU4/kU8 family, so `kFE4M3fn` takes the skip-flop
arm and the kernel never applies its exponent-rebias multiply. Shifting an E4M3
exponent (bias 7) into BF16's field (bias 127) without rebiasing scales every
weight by `2^-120`. Unlike NVFP4 there is no `s2` global-scale channel — only
`kFE2M1f` reads `scale2_ptr` — so the per-channel scale carries the correction.
Folded as `f32::from_bits(0x7B80_0000)`; a channel scale ≥ 255.5 would overflow
BF16 and is rejected with a warn-and-skip before any buffer is built.

**The channelwise scale permutation.** Not the length-64 `scale_perm` the grouped
W8A16 repack implements — channelwise uses the length-32 `scale_perm_single`.

Both are silent failure modes: wrong values at the right speed. The `kFE2M1f` port
returned `nonzero 0/256` for exactly this class of mistake, which no perf
measurement would have caught. Hence the parity harness gates the perf run.

## Review findings on the first draft

A review pass against in-tree evidence caught four defects before any build:

| defect | written | correct |
|---|---|---|
| K alignment | `k % 16` | **`k % 64`** — `min_thread_k = 64`, so a `k ≡ 16 (mod 64)` weight repacks cleanly and throws on every call |
| overflow bound | scale in 2^113..2^117 | **2^110..2^117**, rejection at `scale ≥ 255.5` |
| overflow guard | in-loop `ensure!` | **pre-loop scan**, so an abort cannot leave a half-built buffer |
| constant | decimal literal | **`f32::from_bits(0x7B80_0000)`** |

It also overturned the plan's claim that `QWEN_DEQUANT_GEMM_PREFILL_MIN_M` becomes
redundant: `qwen_fp8_dense_sm_supports_deepgemm` requires `major == 9` exactly
(`quant_linear.rs:257`), so on sm_80/100/120 every 128×128 FP8 weight reaches the
dequant arm and cannot get a Marlin layout (`quant_block_m != 1`). Deleting the
floor would reproduce
[the decode-step dequant bug](../errors/2026-08-19-fp8-dequant-arm-shadows-decode.md)
on a non-Hopper box. Kept.

## Falsifiable prediction

Marlin's per-call cost is flat in M; the GEMV's is not. If the routing engages,
c≥4 ITL falls substantially and c=1 barely moves. If c≥4 does not move,
`cuda.qwen.fp8_marlin_tensorcore` in `/v1/stats` will be 0 and the arm never
engaged — a routing miss, not a kernel limit.

## Pending

Numerics parity, `/v1/stats` engagement check, needle ladder, then the ITL sweep.
The harness is `crates/infer-cuda/examples/marlin_fp8_parity.rs`; it anchors on an
f64 reference rather than lane-vs-lane agreement, and fails when Marlin's error is
many times the GEMV lane's.
