# Vendored from SGLang (Apache-2.0):
#   sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py
#   `_moe_sum_reduce_kernel` (lines 1052-1101). The combine epilogue after the
#   fused MoE GEMM2 (down): cache3 [M, top_k, K] -> out [M, K], summing the
#   top_k partial-K rows per token and scaling by `routed_scaling_factor`.
#
# Launch config resolved from `moe_sum_reduce_triton` (the fused_experts_impl
# caller): BLOCK_M = 1, BLOCK_DIM = 2048, NUM_STAGE = 1, num_warps = 16,
# grid = (cdiv(M, 1), cdiv(K, 2048)). For Qwen3.6 K = 2048 => grid_y = 1.
# `routed_scaling_factor` is a compile-time constexpr upstream; we thread the
# Qwen3.6 config value into the AOT signature as a float literal (read from the
# loaded MoeConfig, NOT hardcoded — the build bakes the production value and the
# Rust dispatch asserts the config matches). The per-route topk_weight multiply
# is carried by GEMM2 (fused_moe_kernel MUL_ROUTED_WEIGHT=True), matching
# upstream `fused_experts_impl`; this kernel applies ONLY routed_scaling_factor.
#
# DEVIATIONS (every intentional change vs. the upstream .py file):
#   1. The torch / sglang.srt.* module-level imports are DROPPED (see
#      arle_fused_moe.py DEVIATION 1). Only `triton` + `triton.language`.
#   2. Sibling launch wrappers and unrelated kernels are not vendored here.
# The live sum + routed_scaling_factor math is byte-equal to upstream.

import triton
import triton.language as tl


@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    mask_token = offs_token < token_num
    mask_dim = offs_dim < hidden_dim

    base_ptrs = input_ptr + offs_token[:, None] * input_stride_0 + offs_dim[None, :]

    accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)

    for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
        tile = tl.load(
            base_ptrs + i * input_stride_1,
            mask=mask_token[:, None] & mask_dim[None, :],
            other=0.0,
        )
        accumulator += tile.to(tl.float32)
    accumulator *= routed_scaling_factor

    # -------- Write back --------
    store_ptrs = output_ptr + offs_token[:, None] * output_stride_0 + offs_dim[None, :]
    tl.store(
        store_ptrs,
        accumulator.to(input_ptr.dtype.element_ty),
        mask=mask_token[:, None] & mask_dim[None, :],
    )
