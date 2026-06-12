# Vendored from SGLang (Apache-2.0):
#   sglang/srt/layers/attention/mamba/causal_conv1d_triton.py
#   `_causal_conv1d_update_kernel` (lines 573-979), wrapper
#   `causal_conv1d_update` (lines 982-1189) for the baked launch config.
# Upstream wrapper bakes (decode call, conv_state_indices set, no spec-decode):
#   BLOCK_N = 256, state_len = width - 1 = 3, NP2_STATELEN = 4,
#   NP2_SEQLEN = 1, KERNEL_WIDTH = 4, SILU_ACTIVATION = True (swish/silu),
#   IS_CONTINUOUS_BATCHING = True, grid = (batch, cdiv(dim, BLOCK_N) = 32),
#   num_warps = triton default (4).
#
# DEVIATIONS (every intentional change; the live decode math is byte-equal):
#   1. `conv_state_indices_ptr` (+ stride_state_indices) is REPLACED by a
#      `conv_state_ptrs` u64 device table, one entry per batch row:
#      `base = tl.load(conv_state_ptrs + idx_seq).to(tl.pointer_type(
#      tl.bfloat16))`. The per-seq stride (stride_conv_state_seq) and the
#      batch-coord indirection collapse to that base (coord is implicitly 0
#      relative to each row's own ring).
#   2. The spec-decode / EAGLE-tree / intermediate-window / circular-buffer
#      machinery is stripped: dropped args cache_seqlens_ptr,
#      num_accept_tokens_ptr, intermediate_conv_window_ptr,
#      intermediate_state_indices_ptr, retrieve_next_token_ptr,
#      retrieve_next_sibling_ptr, retrieve_parent_token_ptr, pad_slot_id and
#      all their strides; constexprs IS_SPEC_DECODING, SAVE_INTERMEDIATE,
#      HAS_EAGLE_TREE_CUSTOM_ATTN_MASK, USE_PAD_SLOT, NP2_SEQLEN,
#      num_cache_lines are baked False/absent. Only the KERNEL_WIDTH==4
#      non-eagle decode branch (upstream lines 899-939) is kept, byte-equal.
#   3. State layout is OURS: `[dim=8192, state_len=3]` bf16, dim-major rows.
#      Their (stride_conv_state_dim, stride_conv_state_tok) map to (3, 1).
#      VERIFIED interoperable with our prefill path
#      (crates/cuda-kernels/csrc/misc/conv1d.cu): both use a shift-window
#      oldest-first ring (col0/col1/col2 = oldest..newest; update shifts left
#      by one and appends the new token), so the triton decode kernel consumes
#      exactly what the hand prefill kernel wrote.
#   4. x / o are OURS `[B, dim]` token-contiguous (seqlen == 1): stride_x_seq =
#      dim, stride_x_dim = 1, stride_x_token = 0 (idx_token is always 0 here).
#      Same for o. All baked as constexprs in build.rs.
# HAS_BIAS = False (no conv bias), KERNEL_WIDTH = 4, SILU_ACTIVATION = True,
# state_len = 3, NP2_STATELEN = 4, seqlen = 1, BLOCK_N = 256 — all baked.

import triton
import triton.language as tl


@triton.jit
def arle_gdn_conv1d_update(
    x_ptr,  # (batch, dim), token-contiguous (seqlen == 1)
    w_ptr,  # (dim, width)
    conv_state_ptrs,  # u64 device table: one [dim, state_len] bf16 ring per row
    o_ptr,  # (batch, dim)
    # Matrix dimensions
    batch,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # ruff: noqa: E501
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    # DEVIATION 1: per-row ring base from the u64 table; the batch-coord
    # indirection collapses to this base.
    conv_state_ptr = tl.load(conv_state_ptrs + idx_seq).to(tl.pointer_type(tl.bfloat16))
    conv_state_batch_coord = 0
    conv_state_token_offset = 0

    # STEP 1: READ init_state data
    conv_states_base = conv_state_ptr + (idx_feats * stride_conv_state_dim)
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens  # [BLOCK_N]
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N]
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)

    # STEP 2: assume state_len > seqlen
    idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

    # The conv_state updates works in a sliding window manner,
    # at each forward pass, the tokens are shift by 1, so we
    # load since idx_tokens + 1.
    conv_state_ptrs_source = (
        conv_state_ptr
        + conv_state_token_offset * stride_conv_state_tok
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + seqlen) * stride_conv_state_tok)[:, None]
    )  # [BLOCK_M, BLOCK_N]
    mask = ((idx_tokens + seqlen) < state_len)[:, None] & (idx_feats < dim)[None, :]
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)  # [BLOCK_N]

    x_ptrs = (
        x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    )  # [BLOCK_M, BLOCK_N]

    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )  # token-index  # token-index  # feature-index
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    conv_state_base = conv_state_ptr + (idx_feats * stride_conv_state_dim)  # [BLOCK_N,]
    conv_state_ptrs_target = (
        conv_state_base + (idx_tokens * stride_conv_state_tok)[:, None]
    )  # [BLOCK_M, BLOCK_N]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats  # noqa: F821
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # STEP 4:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base  # starting of chunk [BLOCK_N]
    mask_x_1d = idx_feats < dim

    # STEP 5: compute each token
    for idx_token in tl.static_range(seqlen):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0

        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (
            idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
            o_ptr
            + (idx_seq) * stride_o_seq
            + idx_token * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)
