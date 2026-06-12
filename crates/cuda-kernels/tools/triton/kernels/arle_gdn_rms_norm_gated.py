# Vendored from SGLang (Apache-2.0):
#   sglang/srt/layers/attention/fla/layernorm_gated.py
#   `_layer_norm_fwd_1pass_kernel` (lines 67-169), with the launch config
#   resolved from `_layer_norm_fwd` (lines 197-270), `calc_rows_per_block`
#   (lines 182-194), and the GDN call site `RMSNormGated(head_v_dim=128,
#   group_size=None, norm_before_gate=True, activation="swish")` in
#   sglang/srt/models/qwen3_5.py:234.
#
# RESHAPE RESOLUTION (the spec's open question):
#   RMSNormGated is constructed with `group_size=None` → `_layer_norm_fwd`
#   sets group_size = N. `rms_norm_gated` reshapes x to (-1, head_v_dim) =
#   (tokens*HV, 128). Hence N = 128, group_size = 128, ngroups = N//group_size
#   = 1, M = tokens*HV. Weight shape is (128,). The group axis (program_id(1))
#   is trivially 0 (ngroups == 1). This matches our existing
#   `rms_norm_gated_cuda` call, which passes num_heads = Vh*tokens and
#   head_dim = 128 over the row-major [tokens, Vh*Vd] buffers, i.e. exactly
#   the (tokens*HV, 128) view. X = our GDR output, Z = our z GEMM output,
#   both [tokens, 4096] bf16 = [tokens*HV, 128] when row-flattened.
#
# Baked launch config (decode M <= 8*32 = 256 rows):
#   N = 128, BLOCK_N = min(MAX_FUSED_SIZE, next_pow2(group_size=128)) = 128,
#   num_warps = min(max(BLOCK_N//256, 1), 8) = 1,
#   ROWS_PER_BLOCK = calc_rows_per_block(...) = MAX_ROWS_PER_BLOCK = 4
#     (calc_rows_per_block returns the constant MAX_ROWS_PER_BLOCK=4 whenever
#      piecewise cuda graph is enabled — SGLang's default — to avoid dynamic
#      guards; AOT can't query the SM count at build time, so we bake that
#      same graph-safe constant 4),
#   grid = (cdiv(M, ROWS_PER_BLOCK), ngroups = 1),
#   HAS_Z = 1, NORM_BEFORE_GATE = 1, IS_RMS_NORM = 1, ACTIVATION = "swish".
#
# DEVIATIONS (every intentional change; the live RMS+gate math is byte-equal):
#   1. Mean (the non-RMS branch, IS_RMS_NORM=0) and B (HAS_BIAS=1) args and
#      their stores are dropped — GDN is always RMS-norm with no bias. The
#      Mean pointer arg and the `if not IS_RMS_NORM` store branch are removed.
#   2. The Rstd store is kept (cheap); Rstd is a preallocated [max_rows = 8*32]
#      f32 scratch on the Rust side.
#   3. W is OUR f32 norm-weight buffer (verified: qwen35.rs passes
#      `attn.norm_weight` as `*const f32`, and ffi/norm.rs declares
#      `rms_norm_gated_cuda(..., weight: *const f32, ...)`). SGLang's W is the
#      checkpoint dtype (bf16); we bake W as *fp32 and keep the in-kernel
#      `.to(tl.float32)` cast (a no-op for f32 input, semantically identical).
# Everything else (the HAS_Z + NORM_BEFORE_GATE swish gate `y *= z*sigmoid(z)`,
# the rstd = rsqrt(mean(x^2)+eps) reduction) is byte-equal to upstream.

import triton
import triton.language as tl


@triton.jit
def arle_gdn_rms_norm_gated(
    X,  # pointer to the input  (bf16, [M, N] row-major)
    Y,  # pointer to the output (bf16, [M, N])
    W,  # pointer to the weights (f32, [N]) — DEVIATION 3
    Z,  # pointer to the gate branch (bf16, [M, N])
    Rstd,  # pointer to the 1/std (f32, [ngroups*M])
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N: tl.constexpr,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Map the program id to the starting row of X and Y it should compute.
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    group = tl.program_id(1)

    # Create 2D tile: [ROWS_PER_BLOCK, BLOCK_N]
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)

    # Compute offsets for 2D tile
    row_offsets = rows[:, None] * stride_x_row
    col_offsets = cols[None, :] + group * N

    # Base pointers
    X_base = X + row_offsets + col_offsets
    Y_base = Y + rows[:, None] * stride_y_row + col_offsets

    # Create mask for valid rows and columns
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask

    # Load input data with 2D tile
    x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)

    if HAS_Z and not NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            x *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            x *= tl.sigmoid(z)

    # Compute variance per row (reduce along axis 1) — IS_RMS_NORM only.
    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / N  # Shape: [ROWS_PER_BLOCK]

    rstd = tl.rsqrt(var + eps)  # Shape: [ROWS_PER_BLOCK]

    # Store rstd for each row
    rstd_offsets = group * M + rows
    rstd_mask = rows < M
    tl.store(Rstd + rstd_offsets, rstd, mask=rstd_mask)

    # Load weights (broadcast across rows)
    w_offsets = cols + group * N
    w_mask = cols < N
    w = tl.load(W + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    # Normalize and apply linear transformation (RMS norm: no mean subtract).
    x_hat = x * rstd[:, None]

    y = x_hat * w[None, :]

    if HAS_Z and NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            y *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            y *= tl.sigmoid(z)

    # Write output
    tl.store(Y_base, y, mask=mask)
