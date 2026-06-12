# Vendored from SGLang (Apache-2.0):
#   sglang/srt/layers/attention/fla/fused_recurrent.py
#   `fused_recurrent_gated_delta_rule_packed_decode_kernel` (lines 186-265),
#   plus the `exp` helper imported at line 11 from
#   `sglang/srt/layers/attention/fla/op.py`.
# Upstream wrapper (lines 268-402) is the reference for the launch config
# baked into build.rs / the Rust call site, not vendored here:
#   BK = next_pow2(K) = 128, BV = min(next_pow2(V), 32) = 32,
#   num_warps = 1, num_stages = 3, grid = (NV = V/BV = 4, B*HV).
#
# DEVIATIONS (every intentional change; kernel math is byte-equal):
#   1. `ssm_state_indices` + `stride_indices_seq` + the separate `h0`/`ht`
#      pool tensors are REPLACED by one `state_ptrs` arg (u64 device table,
#      one entry per batch row). `base = tl.load(state_ptrs + i_n)
#      .to(tl.pointer_type(tl.float32))`; h0 and ht both address `base`
#      (in-place update — safe: each program owns its (i_hv, i_v) tile and
#      reads its full h0 slice before writing the same ht slice).
#   2. State addressing is OURS (k-major `[HV, K, V]`, V contiguous):
#      `base + i_hv*K*V + o_k[None,:]*V + o_v[:,None]`
#      (replaces SGLang's v-major `i_hv*V*K + o_v[:,None]*K + o_k[None,:]`).
#      `b_h` keeps shape [BV, BK]; the reductions over the BK axis (axis 1)
#      are unchanged, so only the load/store offsets change.
#   3. The `state_idx < 0` zero-output branch is dropped (our scheduler never
#      packs invalid slots into a decode batch).
#   4. The `exp` helper is vendored inline (see below) with identical
#      semantics to fla/op.py (plain `tl.exp`, matching the in-kernel
#      `tl.exp` already used for softplus and the A_log gate at lines 252-253).
#   5. `scale` stays a runtime arg; all of stride_mixed_qkv_tok / stride_a_tok
#      / stride_b_tok / H / HV / K / V / BK / BV / SOFTPLUS_THRESHOLD /
#      USE_QK_L2NORM_IN_KERNEL are baked constexprs (see build.rs signature).
# Everything else identical, including the bf16-roundtrip on beta
# (`tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)`) and the 1e-6
# l2norm eps.

import triton
import triton.language as tl


@triton.jit
def exp(x):
    # Vendored from sglang/srt/layers/attention/fla/op.py — the GDN decode
    # kernel imports `exp` from there. fla's op.exp is a thin wrapper over
    # `tl.exp`; we inline it with identical semantics (the surrounding kernel
    # already uses `tl.exp` directly for softplus and the A_log gate).
    return tl.exp(x)


@triton.jit
def arle_gdn_fused_recurrent_decode(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    state_ptrs,  # u64 device table: one [HV, K, V] f32 state base per batch row
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    # DEVIATION 1+3: one u64 state base per row; no ssm_state_indices, no
    # state_idx < 0 early-out.
    base = tl.load(state_ptrs + i_n).to(tl.pointer_type(tl.float32))
    p_o = o + (i_n * HV + i_hv) * V + o_v

    # DEVIATION 2: k-major addressing (our [HV, K, V], V contiguous).
    p_h0 = base + i_hv * V * K + o_k[None, :] * V + o_v[:, None]
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_off = i_h * K + o_k
    k_off = (H * K) + i_h * K + o_k
    v_off = (2 * H * K) + i_hv * V + o_v
    b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

    b_h *= exp(g_val)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    # DEVIATION 2: store back into the same k-major slice (in-place update).
    p_ht = base + i_hv * V * K + o_k[None, :] * V + o_v[:, None]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
