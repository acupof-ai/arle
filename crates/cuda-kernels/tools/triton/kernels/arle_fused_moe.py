# Vendored from SGLang (Apache-2.0):
#   sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py
#   `write_zeros_to_output` (lines 71-89) + `fused_moe_kernel` (lines 323-638),
#   the tensor-core grouped GEMM for the fused MoE FFN. Launch config resolved
#   from `invoke_fused_moe_kernel` (the bf16 dense Qwen3.6-35B-A3B path) and the
#   `fused_experts_impl` caller (the GEMM1 gate||up + GEMM2 down stages).
#
# The kernel BODIES are byte-for-byte upstream — every quant branch
# (use_fp8_w8a8 / use_int8_w8a8 / use_int8_w8a16 / per_channel_quant / block-
# wise scales / swap_ab / TMA descriptors) is KEPT verbatim so the fp8/int8/int4
# specializations are a 1-shape AOT add later (compile with the matching
# constexpr flags True + the scale pointers wired). For NOW we AOT-compile ONLY
# the bf16 specialization: all quant flags False, a_desc/b_desc/bias_ptr/
# add_mask_ptr/a_scale_ptr/b_scale_ptr passed as constexpr `None` (triton DCEs
# every `if … is not None` / `if use_… ` branch at compile time, leaving the
# clean `accumulator += tl.dot(a, b)` inner loop), so the bf16 cubin carries no
# dead quant code.
#
# DEVIATIONS (every intentional change vs. the upstream .py file):
#   1. The torch / sglang.srt.* module-level imports are DROPPED — they pull in
#      torch and the whole SGLang runtime, which the AOT generator's
#      `exec_module` cannot import on a build box. Only `triton` +
#      `triton.language` are imported (the U1 precedent: arle_gdn_rms_norm_gated
#      vendored the kernel body the same way). The kernel SOURCE is untouched.
#   2. Sibling launch wrappers (`invoke_fused_moe_kernel`, the gptq/awq kernel,
#      the activation/sum_reduce kernels) are NOT vendored here — they live in
#      their own files / are launched from Rust. Only the two `@triton.jit`
#      symbols this GEMM needs (`write_zeros_to_output` callee +
#      `fused_moe_kernel`) are present.
# The live grouped-GEMM math is byte-equal to upstream.

import triton
import triton.language as tl


@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    a_desc,
    b_ptr,
    b_desc,
    bias_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    add_mask_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_bias_e,
    stride_bias_n,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    even_Ks: tl.constexpr,
    c_sorted: tl.constexpr,
    filter_expert: tl.constexpr,
    swap_ab: tl.constexpr,
    FUSE_ADD_TO_OUTPUT: tl.constexpr,
    FUSE_SUM_ALL_REDUCE: tl.constexpr,
    ROUTER_TOPK: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.

    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts_i32 = tl.load(expert_ids_ptr + pid_m)
    off_experts = off_experts_i32.to(tl.int64)

    if filter_expert and off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        if not FUSE_ADD_TO_OUTPUT:
            # skip the zero-write to preserve existing values.
            write_zeros_to_output(
                c_ptr,
                stride_cm,
                stride_cn,
                pid_n,
                N,
                offs_token,
                token_mask,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                compute_type,
            )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if a_desc is not None:
        assert use_fp8_w8a8 and group_n > 0 and group_k > 0
        start_offs_m = pid_m * BLOCK_SIZE_M
    else:
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )

    if b_desc is not None:
        start_offs_n = pid_n * BLOCK_SIZE_N
    else:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        )

    if bias_ptr is not None:
        bias = tl.load(
            bias_ptr + off_experts * stride_bias_e + offs_bn[None, :] * stride_bias_n
        )
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            if a_desc is not None:
                a_scale_ptrs = a_scale_ptr + offs_token_id * stride_asm
            else:
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            if BLOCK_SIZE_N > group_n:
                offs_bsn = offs_bn // group_n
            else:
                offs_bsn = pid_n * BLOCK_SIZE_N // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    if swap_ab:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if a_desc is not None:
            a = a_desc.load([start_offs_m, k_start])
        elif even_Ks:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                other=0.0,
            )

        if b_desc is not None:
            b = (
                b_desc.load([off_experts_i32, start_offs_n, k_start])
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
                .T
            )
        elif even_Ks:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)

        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                if swap_ab:
                    a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
                    a_scale, b_scale = b_scale, a_scale
                if BLOCK_SIZE_N > group_n:
                    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
                else:
                    accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale)
            else:
                if use_fp8_w8a8:
                    if swap_ab:
                        a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        if a_desc is None:
            a_ptrs += BLOCK_SIZE_K * stride_ak
        if b_desc is None:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if swap_ab:
        accumulator = tl.trans(accumulator, (1, 0))

    if use_int8_w8a16:
        accumulator *= b_scale
    elif use_fp8_w8a8 or use_int8_w8a8:
        if group_k == 0 or group_n == 0:
            accumulator *= a_scale * b_scale

    if bias_ptr is not None:
        accumulator += bias

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if FUSE_ADD_TO_OUTPUT:
        # Accumulate into existing output with per-token mask.
        offs_token_out = offs_token // ROUTER_TOPK
        add_mask = tl.load(add_mask_ptr + offs_token_out, mask=token_mask, other=False)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & add_mask[:, None] & (offs_cn[None, :] < N)
        existing = tl.load(c_ptrs, mask=c_mask, other=0.0)
        tl.store(c_ptrs, existing + accumulator, mask=c_mask)
    elif FUSE_SUM_ALL_REDUCE:
        offs_token_out = offs_token // ROUTER_TOPK
        c_ptrs = (
            c_ptr + stride_cm * offs_token_out[:, None] + stride_cn * offs_cn[None, :]
        )
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask)
    else:
        if c_sorted:
            c_ptrs = (
                c_ptr
                + stride_cm * offs_token_id[:, None]
                + stride_cn * offs_cn[None, :]
            )
        else:
            c_ptrs = (
                c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
            )
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)
