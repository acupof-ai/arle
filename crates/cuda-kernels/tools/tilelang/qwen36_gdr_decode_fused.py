"""Fused GDR-decode core for Qwen3.6-27B GDN layers (Stage S1a).

Fuses the two value-head-parallel steps of arle's gated-delta-rule decode path —
the recurrent state update (arle `gdr_decode_batch.cu`) and the gated RMSNorm on
its output (arle `rms_norm_gated_cuda`) — into ONE TileLang kernel.

Why only these two: both tile by value-head (grid = (num_value_heads, B)), so
they fuse with zero grid-sync and zero cross-block contention. conv1d+SiLU is
channel-parallel over the full fused qkv and is folded in a later stage (S1b),
because GQA head-sharing makes its conv-state ring race under per-head tiling.

Numerics mirror arle exactly (validate tensor-by-tensor):
  - q,k L2-normalized per head; q additionally scaled by rsqrt(key_dim).
  - g = -exp(A_log) * softplus(a_proj + dt_bias);  exp_g = exp(g)
  - beta = sigmoid(b_proj)
  - recurrence per value head, state S[key_dim, val_dim] fp32, in place:
        S       *= exp_g                       # decay
        kv_mem[v] = sum_j S[j,v] * k[j]         # read memory
        delta[v]  = (v_in[v] - kv_mem[v]) * beta
        S[j,v]  += delta[v] * k[j]              # rank-1 update
        out[v]    = sum_j S[j,v] * q[j]         # output
  - gated RMSNorm on out:
        rms = rsqrt(mean(out^2) + eps)
        normed[v] = out[v] * rms * norm_weight[v]
        gated[v]  = normed[v] * silu(z[v])      # silu(x)=x*sigmoid(x)

Layouts match arle's decode buffers:
  qkv_conv : [B, q_dim + k_dim + v_dim]   (post-conv1d fused q|k|v)
             q_dim = k_dim = num_key_heads*key_dim ; v_dim = num_value_heads*val_dim
  z        : [B, num_value_heads*val_dim]  (in_proj_z gate, pre-silu)
  b_proj, a_proj : [B, num_value_heads]
  dt_bias, A_log, norm_weight : shared across batch
  state    : [B, num_value_heads, key_dim, val_dim]  fp32, updated in place
  out      : [B, num_value_heads*val_dim]  bf16
"""

# NOTE: no `from __future__ import annotations` — T.prim_func resolves shape
# annotations via get_type_hints, and stringified ones lose the builder's locals.

import tilelang
import tilelang.language as T



# Qwen3.6-27B GDN geometry (arle config).
QWEN36_27B_NUM_KEY_HEADS = 16
QWEN36_27B_NUM_VALUE_HEADS = 48
QWEN36_27B_KEY_DIM = 128
QWEN36_27B_VAL_DIM = 128
QWEN36_27B_CONV_KERNEL = 4
QWEN36_27B_RMS_EPS = 1e-6


def gdr_decode_gated_norm(
    B: int,
    num_key_heads: int = QWEN36_27B_NUM_KEY_HEADS,
    num_value_heads: int = QWEN36_27B_NUM_VALUE_HEADS,
    key_dim: int = QWEN36_27B_KEY_DIM,
    val_dim: int = QWEN36_27B_VAL_DIM,
    eps: float = QWEN36_27B_RMS_EPS,
    in_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    out_dtype: str = "bfloat16",
):
    """Build the fused GDR-decode + gated-RMSNorm kernel (Stage S1a).

    One block per (value_head, batch). `threads = val_dim` (128): thread `v`
    owns output column v and state column S[:, v]. The k·S and S·q contractions
    over the key dimension are done as serial per-thread loops over `key_dim`
    (128) — matches arle's decode kernel structure; no T.gemm needed at M=1.
    """
    q_dim = num_key_heads * key_dim
    k_dim = num_key_heads * key_dim
    v_dim = num_value_heads * val_dim
    qkv_stride = q_dim + k_dim + v_dim
    scale = float(key_dim) ** -0.5

    @T.prim_func
    def kernel(
        qkv_conv: T.Tensor((B, qkv_stride), in_dtype),
        z: T.Tensor((B, v_dim), in_dtype),
        b_proj: T.Tensor((B, num_value_heads), in_dtype),
        a_proj: T.Tensor((B, num_value_heads), in_dtype),
        dt_bias: T.Tensor((num_value_heads,), in_dtype),
        A_log: T.Tensor((num_value_heads,), accum_dtype),
        norm_weight: T.Tensor((val_dim,), accum_dtype),
        state: T.Tensor((B, num_value_heads, key_dim, val_dim), accum_dtype),
        out: T.Tensor((B, v_dim), out_dtype),
    ):
        with T.Kernel(num_value_heads, B, threads=val_dim) as (vh, bb):
            tv = T.get_thread_binding(0)  # value-dim index this thread owns, 0..val_dim-1
            kh = vh * num_key_heads // num_value_heads  # GQA: key head for this value head

            q_s = T.alloc_shared((key_dim,), accum_dtype)
            k_s = T.alloc_shared((key_dim,), accum_dtype)
            v_s = T.alloc_shared((val_dim,), accum_dtype)
            qn = T.alloc_shared((1,), accum_dtype)  # 1/||q||
            kn = T.alloc_shared((1,), accum_dtype)  # 1/||k||
            exp_g_s = T.alloc_shared((1,), accum_dtype)
            beta_s = T.alloc_shared((1,), accum_dtype)
            out_s = T.alloc_shared((val_dim,), accum_dtype)
            rms_s = T.alloc_shared((1,), accum_dtype)

            # --- load q/k/v for this head into fragments (thread tv handles index tv) ---
            # q,k live in key_dim (==val_dim here=128), v lives in val_dim.
            q_s[tv] = qkv_conv[bb, kh * key_dim + tv]
            k_s[tv] = qkv_conv[bb, q_dim + kh * key_dim + tv]
            v_s[tv] = qkv_conv[bb, q_dim + k_dim + vh * val_dim + tv]
            # All threads must finish writing q_s/k_s before thread 0 reduces them.
            T.tvm_storage_sync("shared")

            # --- L2 norms (single-thread reduction; key_dim==128 small) ---
            if tv == 0:
                acc_q = T.alloc_local((1,), accum_dtype)
                acc_k = T.alloc_local((1,), accum_dtype)
                T.clear(acc_q)
                T.clear(acc_k)
                for j in T.serial(key_dim):
                    acc_q[0] += q_s[j] * q_s[j]
                    acc_k[0] += k_s[j] * k_s[j]
                qn[0] = T.rsqrt(acc_q[0] + 1e-12)
                kn[0] = T.rsqrt(acc_k[0] + 1e-12)
                # g / beta scalars for this (value_head)
                x = a_proj[bb, vh] + dt_bias[vh]
                softplus = T.if_then_else(x > 20.0, x, T.log(1.0 + T.exp(x)))
                g = -T.exp(A_log[vh]) * softplus
                exp_g_s[0] = T.exp(g)
                beta_s[0] = T.sigmoid(b_proj[bb, vh])
            T.tvm_storage_sync("shared")

            # normalize q (extra *scale), k
            q_s[tv] = q_s[tv] * qn[0] * scale
            k_s[tv] = k_s[tv] * kn[0]
            T.tvm_storage_sync("shared")

            exp_g = exp_g_s[0]
            beta = beta_s[0]

            # --- recurrence: thread tv owns state column S[:, tv] ---
            # Pass 1: decay + kv_mem[tv] = sum_j S[j,tv]*k[j]
            kv_mem = T.alloc_local((1,), accum_dtype)
            T.clear(kv_mem)
            for j in T.serial(key_dim):
                sj = state[bb, vh, j, tv] * exp_g
                state[bb, vh, j, tv] = sj
                kv_mem[0] += sj * k_s[j]
            delta = (v_s[tv] - kv_mem[0]) * beta

            # Pass 2: rank-1 update + out[tv] = sum_j S[j,tv]*q[j]
            acc_o = T.alloc_local((1,), accum_dtype)
            T.clear(acc_o)
            for j in T.serial(key_dim):
                sj = state[bb, vh, j, tv] + delta * k_s[j]
                state[bb, vh, j, tv] = sj
                acc_o[0] += sj * q_s[j]
            out_s[tv] = acc_o[0]
            T.tvm_storage_sync("shared")

            # --- gated RMSNorm on out (mean over val_dim, single-thread reduce) ---
            if tv == 0:
                acc_sq = T.alloc_local((1,), accum_dtype)
                T.clear(acc_sq)
                for j in T.serial(val_dim):
                    acc_sq[0] += out_s[j] * out_s[j]
                rms_s[0] = T.rsqrt(acc_sq[0] / val_dim + eps)
            T.tvm_storage_sync("shared")

            normed = out_s[tv] * rms_s[0] * norm_weight[tv]
            gate = z[bb, vh * val_dim + tv]
            silu_g = gate * T.sigmoid(gate)
            out[bb, vh * val_dim + tv] = normed * silu_g

    return kernel


def gdr_decode_conv_gated_norm(
    B: int,
    num_key_heads: int = QWEN36_27B_NUM_KEY_HEADS,
    num_value_heads: int = QWEN36_27B_NUM_VALUE_HEADS,
    key_dim: int = QWEN36_27B_KEY_DIM,
    val_dim: int = QWEN36_27B_VAL_DIM,
    conv_kernel: int = QWEN36_27B_CONV_KERNEL,
    eps: float = QWEN36_27B_RMS_EPS,
    in_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    out_dtype: str = "bfloat16",
):
    """Stage S1b: fold depthwise conv1d+SiLU in front of the S1a fused core.

    Adds the conv1d over this value-head's 384 channels (q[kh], k[kh], v[vh]) so
    the whole conv → gdr recurrent → gated RMSNorm chain is ONE launch (arle runs
    conv1d, gdr_decode, rms_norm_gated as 3 separate kernels).

    GQA race handling: q/k channels belong to key head `kh` shared by 3 value
    heads. Each block recomputes its own q/k conv (cheap, in registers, no
    cross-block dependency), but the conv-STATE ring for q/k is written only by
    the group-representative block (`vh % (num_value_heads // num_key_heads) == 0`)
    to avoid 3 blocks racing the same ring slot; every block writes its own v ring.

    conv1d math (arle conv1d_decode_batch.cu, kernel_size=4):
        sum = s0*w0 + s1*w1 + s2*w2 + x*w3     (s0..s2 = ring, x = new token)
        out = silu(bf16(sum))                   (bf16 round before SiLU for parity)
        ring <- [s1, s2, x]                     (shift left, append)

    Buffers:
      qkv_in    : [B, qkv_stride]  pre-conv fused q|k|v (bf16)
      conv_w    : [qkv_stride, conv_kernel]  depthwise weights (bf16)
      conv_state: [B, qkv_stride, conv_kernel-1]  ring, updated in place (bf16)
    """
    q_dim = num_key_heads * key_dim
    k_dim = num_key_heads * key_dim
    v_dim = num_value_heads * val_dim
    qkv_stride = q_dim + k_dim + v_dim
    scale = float(key_dim) ** -0.5
    sw = conv_kernel - 1  # ring width
    group = num_value_heads // num_key_heads  # GQA group size (3 for 27B)

    @T.prim_func
    def kernel(
        qkv_in: T.Tensor((B, qkv_stride), in_dtype),
        conv_w: T.Tensor((qkv_stride, conv_kernel), in_dtype),
        conv_state: T.Tensor((B, qkv_stride, sw), in_dtype),
        z: T.Tensor((B, v_dim), in_dtype),
        b_proj: T.Tensor((B, num_value_heads), in_dtype),
        a_proj: T.Tensor((B, num_value_heads), in_dtype),
        dt_bias: T.Tensor((num_value_heads,), in_dtype),
        A_log: T.Tensor((num_value_heads,), accum_dtype),
        norm_weight: T.Tensor((val_dim,), accum_dtype),
        state: T.Tensor((B, num_value_heads, key_dim, val_dim), accum_dtype),
        out: T.Tensor((B, v_dim), out_dtype),
    ):
        with T.Kernel(num_value_heads, B, threads=val_dim) as (vh, bb):
            tv = T.get_thread_binding(0)
            kh = vh * num_key_heads // num_value_heads
            is_group_rep = (vh % group) == 0

            q_s = T.alloc_shared((key_dim,), accum_dtype)
            k_s = T.alloc_shared((key_dim,), accum_dtype)
            v_s = T.alloc_shared((val_dim,), accum_dtype)
            qn = T.alloc_shared((1,), accum_dtype)
            kn = T.alloc_shared((1,), accum_dtype)
            exp_g_s = T.alloc_shared((1,), accum_dtype)
            beta_s = T.alloc_shared((1,), accum_dtype)
            out_s = T.alloc_shared((val_dim,), accum_dtype)
            rms_s = T.alloc_shared((1,), accum_dtype)

            # --- conv1d + SiLU on this head's 3 channel groups (q[kh], k[kh], v[vh]) ---
            # channel offsets in the fused qkv row:
            qc = kh * key_dim + tv          # q channel this thread handles
            kc = q_dim + kh * key_dim + tv  # k channel
            vc = q_dim + k_dim + vh * val_dim + tv  # v channel

            # depthwise K=4 conv over ring[c] + new token qkv_in[bb,c], then SiLU.
            # Inlined per channel (q/k/v) — tilelang's builder does not trace
            # nested python helpers, so the three convs are written out.
            cq = T.alloc_local((1,), accum_dtype)
            ck = T.alloc_local((1,), accum_dtype)
            cvv = T.alloc_local((1,), accum_dtype)
            cq[0] = qkv_in[bb, qc] * conv_w[qc, sw]
            ck[0] = qkv_in[bb, kc] * conv_w[kc, sw]
            cvv[0] = qkv_in[bb, vc] * conv_w[vc, sw]
            for t in T.serial(sw):
                cq[0] += conv_state[bb, qc, t] * conv_w[qc, t]
                ck[0] += conv_state[bb, kc, t] * conv_w[kc, t]
                cvv[0] += conv_state[bb, vc, t] * conv_w[vc, t]
            # bf16 round before SiLU for numerical parity with arle prefill.
            sq = cq[0].astype(in_dtype).astype(accum_dtype)
            sk = ck[0].astype(in_dtype).astype(accum_dtype)
            svv = cvv[0].astype(in_dtype).astype(accum_dtype)
            q_s[tv] = sq * T.sigmoid(sq)
            k_s[tv] = sk * T.sigmoid(sk)
            v_s[tv] = svv * T.sigmoid(svv)

            # --- update conv rings (shift-left, append new pre-conv token) ---
            # v ring: every block owns its v channels -> always write.
            # q/k rings: shared across the GQA group -> only the representative writes.
            x_v = qkv_in[bb, vc]
            for t in T.serial(sw - 1):
                conv_state[bb, vc, t] = conv_state[bb, vc, t + 1]
            conv_state[bb, vc, sw - 1] = x_v.astype(in_dtype)
            if is_group_rep:
                x_q = qkv_in[bb, qc]
                x_k = qkv_in[bb, kc]
                for t in T.serial(sw - 1):
                    conv_state[bb, qc, t] = conv_state[bb, qc, t + 1]
                    conv_state[bb, kc, t] = conv_state[bb, kc, t + 1]
                conv_state[bb, qc, sw - 1] = x_q.astype(in_dtype)
                conv_state[bb, kc, sw - 1] = x_k.astype(in_dtype)
            T.tvm_storage_sync("shared")

            # --- from here identical to S1a: L2-norm, g/beta, recurrence, gated RMSNorm ---
            if tv == 0:
                acc_q = T.alloc_local((1,), accum_dtype)
                acc_k = T.alloc_local((1,), accum_dtype)
                T.clear(acc_q)
                T.clear(acc_k)
                for j in T.serial(key_dim):
                    acc_q[0] += q_s[j] * q_s[j]
                    acc_k[0] += k_s[j] * k_s[j]
                qn[0] = T.rsqrt(acc_q[0] + 1e-12)
                kn[0] = T.rsqrt(acc_k[0] + 1e-12)
                x = a_proj[bb, vh] + dt_bias[vh]
                softplus = T.if_then_else(x > 20.0, x, T.log(1.0 + T.exp(x)))
                g = -T.exp(A_log[vh]) * softplus
                exp_g_s[0] = T.exp(g)
                beta_s[0] = T.sigmoid(b_proj[bb, vh])
            T.tvm_storage_sync("shared")

            q_s[tv] = q_s[tv] * qn[0] * scale
            k_s[tv] = k_s[tv] * kn[0]
            T.tvm_storage_sync("shared")

            exp_g = exp_g_s[0]
            beta = beta_s[0]

            kv_mem = T.alloc_local((1,), accum_dtype)
            T.clear(kv_mem)
            for j in T.serial(key_dim):
                sj = state[bb, vh, j, tv] * exp_g
                state[bb, vh, j, tv] = sj
                kv_mem[0] += sj * k_s[j]
            delta = (v_s[tv] - kv_mem[0]) * beta

            acc_o = T.alloc_local((1,), accum_dtype)
            T.clear(acc_o)
            for j in T.serial(key_dim):
                sj = state[bb, vh, j, tv] + delta * k_s[j]
                state[bb, vh, j, tv] = sj
                acc_o[0] += sj * q_s[j]
            out_s[tv] = acc_o[0]
            T.tvm_storage_sync("shared")

            if tv == 0:
                acc_sq = T.alloc_local((1,), accum_dtype)
                T.clear(acc_sq)
                for j in T.serial(val_dim):
                    acc_sq[0] += out_s[j] * out_s[j]
                rms_s[0] = T.rsqrt(acc_sq[0] / val_dim + eps)
            T.tvm_storage_sync("shared")

            normed = out_s[tv] * rms_s[0] * norm_weight[tv]
            gate = z[bb, vh * val_dim + tv]
            silu_g = gate * T.sigmoid(gate)
            out[bb, vh * val_dim + tv] = normed * silu_g

    return kernel


def _reference(qkv, z, b_proj, a_proj, dt_bias, A_log, norm_weight, state,
               num_key_heads, num_value_heads, key_dim, val_dim, eps):
    """fp32 PyTorch reference mirroring arle's gdr_decode + rms_norm_gated."""
    import torch

    B = qkv.shape[0]
    q_dim = num_key_heads * key_dim
    k_dim = q_dim
    v_dim = num_value_heads * val_dim
    state = state.clone()
    out = torch.zeros(B, v_dim, dtype=torch.float32, device=qkv.device)
    qf = qkv.float()
    for b in range(B):
        for vh in range(num_value_heads):
            kh = vh * num_key_heads // num_value_heads
            q = qf[b, kh * key_dim:(kh + 1) * key_dim].clone()
            k = qf[b, q_dim + kh * key_dim: q_dim + (kh + 1) * key_dim].clone()
            v = qf[b, q_dim + k_dim + vh * val_dim: q_dim + k_dim + (vh + 1) * val_dim].clone()
            q = q * torch.rsqrt((q * q).sum() + 1e-12) * (key_dim ** -0.5)
            k = k * torch.rsqrt((k * k).sum() + 1e-12)
            x = a_proj[b, vh].float() + dt_bias[vh].float()
            sp = x if x > 20 else torch.log(1 + torch.exp(x))
            exp_g = torch.exp(-torch.exp(A_log[vh]) * sp)
            beta = torch.sigmoid(b_proj[b, vh].float())
            S = state[b, vh] * exp_g
            kv = (S * k[:, None]).sum(0)
            delta = (v - kv) * beta
            S = S + delta[None, :] * k[:, None]
            o = (S * q[:, None]).sum(0)
            rms = torch.rsqrt((o * o).mean() + eps)
            gate = z[b, vh * val_dim:(vh + 1) * val_dim].float()
            out[b, vh * val_dim:(vh + 1) * val_dim] = o * rms * norm_weight * (gate * torch.sigmoid(gate))
            state[b, vh] = S
    return out, state


def _self_check():
    """Validate the fused kernel against the fp32 reference (arle-style tol)."""
    import torch

    B, NKH, NVH, KD, VD, EPS = 2, QWEN36_27B_NUM_KEY_HEADS, QWEN36_27B_NUM_VALUE_HEADS, \
        QWEN36_27B_KEY_DIM, QWEN36_27B_VAL_DIM, QWEN36_27B_RMS_EPS
    q_dim = NKH * KD
    qkv_stride = 2 * q_dim + NVH * VD
    v_dim = NVH * VD
    dev = "cuda"
    torch.manual_seed(0)
    qkv = torch.randn(B, qkv_stride, dtype=torch.bfloat16, device=dev)
    z = torch.randn(B, v_dim, dtype=torch.bfloat16, device=dev)
    bpr = torch.randn(B, NVH, dtype=torch.bfloat16, device=dev)
    apr = torch.randn(B, NVH, dtype=torch.bfloat16, device=dev)
    dtb = torch.randn(NVH, dtype=torch.bfloat16, device=dev)
    alog = torch.randn(NVH, dtype=torch.float32, device=dev)
    nw = torch.randn(VD, dtype=torch.float32, device=dev)
    state0 = torch.randn(B, NVH, KD, VD, dtype=torch.float32, device=dev)

    out_ref, state_ref = _reference(qkv, z, bpr, apr, dtb, alog, nw, state0, NKH, NVH, KD, VD, EPS)

    kernel = tilelang.compile(gdr_decode_gated_norm(B=B), target="cuda")
    st = state0.clone()
    out = torch.zeros(B, v_dim, dtype=torch.bfloat16, device=dev)
    kernel(qkv, z, bpr, apr, dtb, alog, nw, st, out)
    torch.cuda.synchronize()

    # mixed abs+rel tol (small-denominator-safe): bf16 inputs -> ~1e-2 abs.
    for name, got, ref in (("out", out, out_ref), ("state", st, state_ref)):
        g, r = got.float(), ref.float()
        within = ((g - r).abs() <= 2e-2 + 2e-2 * r.abs()).float().mean().item()
        assert within > 0.999, f"{name}: only {within*100:.2f}% within tol"
    print("qwen36_gdr_decode_fused self-check PASSED (out+state within bf16 tol)")


if __name__ == "__main__":
    _self_check()
