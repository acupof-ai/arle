# Qwen3.6-35B post-attn add+RMSNorm → flashinfer FusedAddRMSNorm (U4) — opt-in, validated WASH

**Date:** 2026-06-12 (impl); **2026-06-13** validation.
**Backend:** CUDA, Qwen3.6-35B-A3B, H20, TP=1.
**Verdict (2026-06-13):** lane RUNS (probe fired) + correct (DET, minor
fusion-FP envelope caveat); perf is a **WASH** at every c. Stays opt-in, no
default flip. See the
[validation entry](2026-06-13-qwen36-sgl-kernel-align-validate-bistability.md).

## Context

Third tranche of the Qwen-lane SGLang kernel alignment
([plan](../../plans/2026-06-12-qwen-lane-kernel-alignment-sglang.md), ckl
directive "kernel 全对齐 sglang 不要再自己写了"). Each decoder layer runs two
residual-add + RMSNorm pairs (post-attn, post-MLP). SGLang/flashinfer fuse the
residual add into the following RMSNorm (`FusedAddRMSNorm`), the canonical
vLLM/SGLang residual-stream pattern. U4 adopts that kernel for the **post-attn**
pair (Scope MIN — 40 of 80 add+norm pairs, no cross-layer carry, no
first/last special-case).

## What landed

- **Vendored flashinfer norm kernel** (`crates/cuda-kernels/csrc/vendor/flashinfer/`):
  `norm.cuh` (`FusedAddRMSNormKernel` @ 387) + its 12-file transitive header
  closure (`utils`/`vec_dtypes`/`math`/`activation`/`exception`/`logging` +
  `trtllm/common/*`). Header-only C++17.
- **C shim** (`crates/cuda-kernels/csrc/misc/arle_fused_add_rmsnorm.cu`):
  `arle_fused_add_rmsnorm_offset_bf16_cuda` replicates flashinfer's
  `FusedAddRMSNorm<bf16>` launch config (gcd vec_size, smem, PDL-off) but passes
  **`weight_bias = 1.0f`** — flashinfer's host launcher hardcodes `weight_bias=0`
  whereas ARLE/Qwen3.x apply the Gemma-style `(1+weight)` gain (verified against
  `rms_norm_offset_kernel` / `fused_add_rms_norm_offset_kernel` in `norm.cu`,
  both `(1.0f + weight)`). Calling the launcher directly would be numerically
  wrong, so the shim launches `FusedAddRMSNormKernel` itself with the right bias.
  `enable_pdl=false` (not in a PDL graph); `griddepcontrol` asm is
  `__CUDA_ARCH__>=900`-guarded (H20-safe).
- **build.rs nvcc arm + FFI** (`build.rs`, `src/ffi/norm.rs`): `-std=c++17
  --expt-relaxed-constexpr --expt-extended-lambda -Icsrc/vendor`; the shim's
  `#include "vendor/flashinfer/norm.cuh"` resolves via the always-present
  `-Icsrc`, norm.cuh's internal `"flashinfer/..."` via `-Icsrc/vendor`.
- **Scope MIN wiring** (`crates/infer-cuda/src/qwen35.rs`,
  `crates/infer-cuda/src/ops.rs`): opt-in `ARLE_QWEN35_FUSED_ADDNORM` (OnceLock).
  Flag ON replaces `add_batch(hidden,attn_out)->hidden_mid` +
  `rms_norm_offset(hidden_mid,post_attn)->normed` with one
  `fused_add_rms_norm_offset(input=attn_out, residual=hidden, post_attn)` that
  mutates `hidden`→post-attn sum and `attn_out`→normalized; MoE reads the fused
  `attn_out`; the MLP residual folds straight into `hidden` via the new
  `add_batch_inplace` (aliased `add_cuda(acc,addend,acc)`). Wired at BOTH
  `forward_hidden_staged` (prefill/OPD + captured decode-graph body) and
  `forward_decode_batch`. Layout resolved as **row-major `[seq_len,hidden_dim]`**
  (token = `hidden_dim` contiguous bf16) — confirmed against
  `rms_norm_batched_kernel`'s `x + blockIdx.x*hidden_dim` indexing; the stale
  `[H,S]` doc-comment was overridden. Flag OFF → byte-for-byte the hand
  `add_batch`+`rms_norm_offset` path (no half-state; hand kernels stay baseline).
  Capture-safe: no module load / no per-step alloc.

## Verification (Mac, no nvcc)

- `cargo fmt --check` clean; `cargo check -p infer-api --features cuda,no-cuda
  --lib` green; `cargo check -p agent-infer --features cpu,no-cuda,cli` green;
  `cargo clippy -p infer-cuda --features cuda,no-cuda` 0 warnings. The `.cu` is
  compiled only by nvcc on the pod (no-cuda skips kernel compile) — the
  kernel-signature/include-path correctness was verified by source review:
  `FusedAddRMSNormKernel(input,residual,weight,d,stride_input,stride_residual,
  weight_bias,eps)` matches the shim's `cudaLaunchKernelEx` arg order exactly.

## Pending (one-shot pod pass, #88)

1. Build on 8×H20 with the flashinfer shim compiling (not a stub); confirm via
   the runtime loud-fail probe under `ARLE_QWEN35_FUSED_ADDNORM=1`.
2. Needle gate ×3 DET vs the locked 2026-06-12 envelope (len 2000/8000 exact).
3. Same-binary same-shell A/B `ARLE_QWEN35_FUSED_ADDNORM` OFF vs ON, c=1/2/4/8,
   vs the locked baseline 93.5/152.3/207.5/255.6 tok/s. Δ% per c. The fused
   kernel reads the same residual bytes — at B=1 this is overhead-shaving on a
   bandwidth-bound path, expected wash-to-marginal; the real payoff is one fewer
   launch/full-buffer-write per layer at c≥2 prefill/batched-decode. License-or-
   kill on wall-clock per shape; a losing A/B keeps the lane opt-in with the
   verdict recorded.

## Rule

When adopting an upstream fused norm, verify the gain convention (`weight` vs
`1+weight`) against the kernel you are *replacing*, not the upstream default —
flashinfer's `weight_bias=0` would silently miscompute Qwen3.x's `(1+weight)`
RMSNorm. The fix is a one-float launch-param override in a thin shim, not a
fork of the kernel.
