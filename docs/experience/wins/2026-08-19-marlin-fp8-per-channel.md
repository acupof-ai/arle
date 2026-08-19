# Per-channel FP8 on Marlin — CUDA, 2026-08-19

> Status: Shipped. Numerics 31/31, perf measured against a matched FP8 arm.

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

## Numerics: 31/31 PASS

`marlin_fp8_parity` on 1xH20, five shape families (q_proj/in_proj_qkvz 12288x5120,
k_proj/v_proj 1024x5120, o_proj 5120x6144, linear_attn, lm_head 248320x5120) x
M in {1, 2, 4, 16, 64, 256}:

```
[q_proj/in_proj_qkvz m=256 n=12288 k=5120] marlin relL2=1.6602e-3 max/rms=8.7266e-3
    mean(out/ref)=1.000000 | gemv relL2=1.6602e-3 max/rms=8.7266e-3
    mean(out/ref)=1.000001 | ratio=1.00 PASS
```

All three silent failure modes are excluded:

| mode | signature | measured |
|---|---|---|
| 2^120 not folded in | `mean(out/ref) = 0` | **1.000000** |
| wrong scale permutation | `mean ≈ 1` but `max/rms` O(1) | 8e-3, the quantisation floor |
| wrong repack layout | Marlin error >> GEMV error | **ratio = 1.00** |

`ratio = 1.00` was checked for the dead-arm case — the two lanes are not the same
code path. `mean(out/ref)` differs in the last digit between them (0.999988 vs
0.999989), so both kernels ran; the errors coincide because they dequantise the
same E4M3 weights, making the quantisation error common-mode and leaving only
accumulation order. That is the expected result, not a stuck harness.

## Results

1xH20, synthetic prompt, 30 s/point, `--kv-cache-dtype fp8`, no spec, decode graph
on. Both arms on the SAME binary; the FP8 arm was re-measured after this change
rather than reused. Aggregate tok/s.

| c | before | after | delta | FP8 | vs FP8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 74.2 | **82.1** | +10.6% | 61.5 | **+33.4%** |
| 2 | 103.3 | **128.5** | +24.4% | 99.8 | **+28.8%** |
| 4 | 165.3 | **233.8** | +41.5% | 195.9 | **+19.3%** |
| 8 | 216.8 | **367.1** | +69.3% | 358.0 | **+2.5%** |
| 16 | 235.9 | **472.4** | **+100.2%** | 632.5 | −25.3% |

The gain grows monotonically with concurrency, which is the mechanism confirming
itself: Marlin's per-call cost is flat in M, the batched GEMV's is not. ITL at
c=16 goes 67.82 -> 33.87 ms, and NVFP4's step-cost growth over 16x concurrency
falls from **4.5x to 2.8x** against FP8's 1.45x. That structural defect is what
this change was for.

Engagement was checked before the numbers were read — a perf figure without it
measures nothing:

```
cuda.fp4.marlin_tensorcore        184016
cuda.qwen.fp8_marlin_tensorcore   157728   <- the new arm
cuda.qwen.fp8_gemv                 80507   <- 34% still scalar
```

The control arm was checked too: the FP8 server reports
`cuda.qwen.fp8_pack_deepgemm 256` and `cuda.qwen.fp8_gemv 512` with
`fp8_marlin_tensorcore` **absent**, so this change did not reach into the
comparison checkpoint. Its 128x128 blocks fail `quant_block_m == 1` as intended.

## Not the ceiling: 34% of FP8 calls are still scalar

`cuda.qwen.fp8_gemv` still takes 80507 of 238235 FP8 calls. Four load sites never
call any repack — `lm_head`, `linear_attn.out_proj` x48, the TP=1 qkv, and the MTP
fc — the same four flagged in `5499e20a7` when `down_proj`/`o_proj` were fixed for
FP4. 48 linear-attn layers is the right order of magnitude for the residue.
Wiring those is the next step and should land where it is still needed most, at
c>=8.

## Pending

The needle ladder x3 on this configuration, and the four unwired load sites above.
The 32K long-agent row also needs re-measuring — the open prefill crash
(`errors/2026-08-19-blocks-per-sm-search-two-latent-bugs.md`) has not been
reproduced deterministically yet; `scripts/force_partial_restore.py` exists for
that and has not been run.
