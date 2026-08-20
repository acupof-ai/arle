# W4AFP8 MoE path: five silent-correctness bugs shipped, caught by review — CUDA, 2026-08-20

> Status: Fix committed (`7cde1ce84` + shape validation), math gate 3/3 PASS. Long-prefill + concurrent verification pending-remote.

## Context

NVFP4→W4AFP8 load-time conversion shipped 2026-08-18 (`2026-08-18-nvfp4-w4afp8-tp2-serve.md`). The math correctness gate (3 short prompts) passed. Code review on 2026-08-19/20 found five bugs that produce silent wrong outputs — no crash, no NaN, just degraded or zeroed MoE results.

## Bugs

### 1. amax cross-block reduction used per-block shared memory

`w4a8_per_tensor_amax_kernel` wrote the block result to `__shared__ float s_max`, then block 0 stored its own local max as the "global" amax. Shared memory is per-block; block 0 never sees other blocks' maxima.

**Impact**: FP8 activation quant scale too small → all MoE gate/up/down outputs saturate or clip. Visible on any input where block 0 doesn't see the global max.

### 2. Raw amax saved, CUTLASS reads it as dequant factor

After fixing bug 1 to write raw amax, the CUTLASS epilogue reads the scale pointer as `alpha = amax/448` (dequant factor). Raw amax would amplify GEMM output ~448×.

**Impact**: All MoE outputs amplified 448×. Model would produce garbage, but the math gate's short prompts may not surface this if the amplified values happen to land on the correct token.

### 3. SwiGLU kernel capped at 256 blocks

`if (blocks > 256) blocks = 256;` limited output to 65,536 elements. At `i_dim=2048`, that's 32 rows. DSv4 prefill with m=64 tokens × 8 experts/token = 512 routed rows — 94% of SwiGLU output silently zeroed.

**Impact**: Any prefill batch with >32 routed rows produces zeroed SwiGLU output past row 32. Decode (1 token × 8 experts = 8 rows) is unaffected.

### 4. `serve_dsv4_tp4.sh` used defunct `INFER_TP_SIZE`

The env var was removed when TP config moved to CLI flags. The script set it but the executor ignored it, silently serving TP=1 on 4 GPUs.

**Impact**: TP=4 bench results were actually TP=1. No correctness bug, but all TP=4 perf data from that script is invalid.

### 5. NVFP4 K%512 guard + loader source validation

The conversion kernel's scale layout `[K//512, N*4]` requires K%512==0, but validation only checked K%128. The loader also didn't validate source tensor dtype, shape, or byte lengths before GPU upload — the kernel reads N*K/32 scale bytes at fixed offsets, so a mismatch OOBs.

**Impact**: OOB write on checkpoints with K%512!=0 (none currently, but a format boundary hole). Incompatible checkpoints could OOB read.

### 6. Shape rank + cross-expert consistency (uncommitted at time of writing)

`weight.shape[0]`/`[1]` panics on non-2D weights. Cross-expert shape mismatch passes validation but the fused w13 buffer is read with uniform stride — wrong offsets, silent corruption.

## Root Cause

Three patterns:

1. **Shared memory for cross-block reduction** — `__shared__` is per-block; cross-block state needs global memory or a finalize kernel.
2. **Scale semantics mismatch at kernel boundary** — the amax kernel writes raw amax; the CUTLASS epilogue expects amax/448. No single place documents which convention the scale pointer uses.
3. **Defensive cap too low for the actual shape** — 256 blocks was a reasonable guess for a generic MoE, but DSv4's 2048 intermediate × 512 routed rows needs 4096 blocks.

The math gate didn't catch bugs 1–3 because it uses short prompts (few tokens, few routed rows). Bug 3 (SwiGLU cap) only triggers above 32 routed rows — decode with 1 token × 8 experts = 8 rows. Bugs 1–2 may not change the argmax on short prompts.

## Fix

All in `7cde1ce84` (+ shape validation in loader.rs):

1. `atomicMax(reinterpret_cast<int*>(amax_out), __float_as_int(local_max))` — global cross-block reduction.
2. Added `w4a8_amax_finalize_kernel` (single thread, converts raw amax → amax/448) between amax and quantize kernels.
3. Removed the 256-block cap; the kernel already has `if (idx < rows * i_dim)` bounds check.
4. `--tensor-parallel-size 4` CLI flag replaces `INFER_TP_SIZE`.
5. K%512 guard in the kernel; dtype/shape/byte validation in the loader before GPU upload.
6. Rank check (`weight.shape.len() != 2`) + cross-expert w1/w3/w2 shape consistency in the loader.

## Verification

- Math correctness gate: 3/3 PASS (short prompts, same as before — confirms no regression).
- Long-prefill + concurrent decode: **pending-remote** — needs pod run with prefill m≥64 to exercise the SwiGLU path past 32 rows.

## Rule

MoE quantization changes need a long-prefill correctness gate (m≥64 tokens, enough routed rows to exceed any per-block cap), not just short math prompts. Cross-block reductions must use global memory. Scale pointer semantics (raw amax vs dequant factor) must be documented at the kernel boundary.
