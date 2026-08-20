# GDR decode fusion: compiled_compute_g_beta for decode + f32 g/beta — Metal, 2026-08-20

> Status: Shipped

## Context

Qwen3.8-27B-MLX-2bit decode on Metal spent ~965 kernel launches per step.
Two avoidable costs per GDR layer (48 layers):

1. **Separate sigmoid + compute_g**: decode (S=1) used `sigmoid(b_raw)` +
   `compiled_compute_g()` = 2 kernels. Prefill (S>1) already used the fused
   `compiled_compute_g_beta()` = 1 kernel. The `S > 1` gate was the only
   reason decode didn't use the fused path.

2. **bf16 cast for g/beta**: `compiled_compute_g_beta()` outputs f32 (because
   `neg_exp_a` is f32). The materialization lines cast g/beta to bf16 via
   `astype(..., bfloat16)` before passing to the `gated_delta_step` Metal
   kernel — 2 extra kernels per layer. The kernel auto-generates its function
   signature from actual array dtypes (`custom_kernel.cpp:74`), so f32 inputs
   produce `device float*` pointers natively. `InT` only governs the output
   cast (`static_cast<InT>(out)`), not input pointer types.

## What Worked

1. Removed the `S > 1` condition on `compiled_compute_g_beta()` — decode now
   uses the fused path (1 kernel instead of 2 per GDR layer).
2. Removed `astype(bfloat16)` for g/beta in the kernel materialization lines.
   f32 g/beta flow directly into the kernel; the auto-generated signature
   handles the dtype.
3. Relaxed the `mlx_tape_replay` dtype gate for g from bf16-only to
   bf16-or-f32. The tape_replay kernel body uses `g_[hv_idx]` in float
   arithmetic and never references `InT` for g — both dtypes work natively.
   Legacy bf16 tapes still pass the gate.

## Result

M4 Pro 48GB, Qwen3.8-27B-MLX-2bit, `--max-running-requests 1`, streaming
50-token decode, 3-run median:

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| Decode speed (excl TTFT) | 19.2 tok/s | 20.6 tok/s | +7.3% |
| TTFT | 0.74s | 0.55s | −26% |
| Kernels eliminated per step | — | 144 | — |

Kernels eliminated: 48 sigmoid + 96 astype = 144 per decode step.

Theoretical ceiling: 27.1 tok/s (273 GB/s ÷ 10.09 GB weight bytes).
ARLE now at 76% of theoretical (was 71%).

## MLX compile attempt — no improvement

Tried compiling the GDR preprocessing (SiLU + QK norm + g/beta) and
postprocessing (rms_norm + SiLU-mul) into single `mlx::core::compile()`
functions. Zero measurable change (20.6 → 20.5 tok/s, within noise).

Cause: MLX compile cannot fuse rms_norm reductions with elementwise ops
into one kernel. The compiled function launches the same number of kernels
as the separate compiled helpers — the reduction is a barrier. Reverted;
the separate `compiled_silu` / `compiled_qk_norm_scale` /
`compiled_compute_g_beta` helpers are already at the minimum kernel count
for this op graph.

## Environment

- Host: M4 Pro 48GB, macOS
- Model: majentik/Qwen3.8-27B-MLX-2bit (64 layers, 48 GDR + 16 full attn)
- Flags: `--backend metal --max-running-requests 1`
- Files: `crates/mlx-sys/src/mlx_qwen35_model.cpp`, `crates/mlx-sys/src/mlx_bridge.cpp`
