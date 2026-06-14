#!/usr/bin/env python
"""Numpy CPU reference for the dense Qwen3.6-27B (arch `qwen35`) forward.

Loads the on-box Q8_0 GGUF, dequantizes tensors exactly as
`crates/infer-gguf/src/dequant.rs` does, and computes the forward STAGE BY
STAGE so the on-device ARLE forward (`crates/infer-vulkan/src/forward.rs`,
dumped behind `ARLE_VK_DUMP=1`) can be diffed per stage to localize the first
diverging op.

This is a *correctness oracle*, not a perf path: everything is f32 host math,
transcribed line-for-line from the CUDA reference
(`crates/infer-cuda/src/qwen35.rs` + `crates/cuda-kernels/csrc/...`). The one
deliberate convention: all RMSNorms apply the PLAIN GGUF weight
(`x*inv_rms*w`) because the GGUF converter folded the HF `(1+w)` offset into
the stored weight (verified on-box: attn_norm mean ~0.98, output_norm ~1.96).

Usage:
    python tools/qwen35_ref.py            # dump layer 0 + 3 stages + final top5
    python tools/qwen35_ref.py --layers 0 3 --tokens 760 6511 314 9338 369
"""

import argparse
import sys

import numpy as np
from gguf import GGUFReader

MODEL = r"C:\Users\Asus\models\qwen3.6\Qwen3.6-27B-Q8_0.gguf"

# ── dims (verified against the on-box 27B GGUF metadata) ──────────────────────
HIDDEN = 5120
N_LAYERS = 64
N_HEADS = 24
N_KV = 4
HEAD_DIM = 256
ROTARY_DIM = 64           # qwen35.rope.dimension_count (PARTIAL rotary)
ROPE_THETA = 1.0e7
RMS_EPS = 1e-6
FFN_INTER = 17408
VOCAB = 248320

# linear (gated-delta) dims
L_KD = 128                # linear_key_head_dim
L_VD = 128                # linear_value_head_dim
L_NK = 16                 # linear_num_key_heads
L_NV = 48                 # linear_num_value_heads
L_KERNEL = 4
Q_DIM_T = L_NK * L_KD     # 2048
K_DIM_T = L_NK * L_KD     # 2048
V_DIM_T = L_NV * L_VD     # 6144
QKV_DIM = Q_DIM_T + K_DIM_T + V_DIM_T  # 10240


def dequantize(tensor):
    """Dequantize a GGUFReader tensor to a row-major f32 np.array of shape
    `[ne1, ne0]` (i.e. [out, in] for a weight, matching the GEMV contract)."""
    tt = tensor.tensor_type.name
    shape = [int(s) for s in tensor.shape]  # gguf reports [ne0, ne1, ...]
    if tt == "F32":
        return tensor.data.astype(np.float32).reshape(shape[::-1])
    if tt == "F16":
        return tensor.data.astype(np.float32).reshape(shape[::-1])
    if tt == "Q8_0":
        # raw bytes: [..rows.., 34*nblocks]; each block = f16 d + 32 int8.
        raw = tensor.data  # (rows, row_bytes) uint8
        rows, row_bytes = raw.shape
        nb = row_bytes // 34
        blocks = raw.reshape(rows, nb, 34)
        d = blocks[:, :, :2].copy().view(np.float16).astype(np.float32)  # (rows, nb, 1)
        q = blocks[:, :, 2:].view(np.int8).astype(np.float32)            # (rows, nb, 32)
        deq = (q * d).reshape(rows, nb * 32)
        return deq  # [ne1, ne0] = [out, in]
    raise SystemExit(f"unhandled tensor type {tt} for {tensor.name}")


def rms_norm(x, w, eps=RMS_EPS):
    """PLAIN RMSNorm: x * 1/sqrt(mean(x^2)+eps) * w."""
    inv = 1.0 / np.sqrt(np.mean(x * x) + eps)
    return x * inv * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def round_to_bf16(x):
    """Round f32 -> bf16 (round-to-nearest-even), matching forward.rs."""
    b = x.astype(np.float32).view(np.uint32)
    lsb = (b >> 16) & 1
    b = (b + 0x7FFF + lsb) & 0xFFFF0000
    return b.view(np.float32)


def rope_neox(x, pos, half=ROTARY_DIM // 2, theta=ROPE_THETA, rotary_dim=ROTARY_DIM):
    """NeoX RoPE on one head vector [head_dim]; pairs (d, d+half)."""
    out = x.copy()
    d = np.arange(half)
    inv_freq = theta ** (-(2.0 * d) / rotary_dim)
    angle = pos * inv_freq
    c, s = np.cos(angle), np.sin(angle)
    x0 = x[:half]
    x1 = x[half:2 * half]
    out[:half] = x0 * c - x1 * s
    out[half:2 * half] = x0 * s + x1 * c
    return out


class Model:
    def __init__(self, path):
        self.r = GGUFReader(path)
        self.t = {x.name: x for x in self.r.tensors}
        self._cache = {}
        # layer types
        self.layer_types = []
        for i in range(N_LAYERS):
            if f"blk.{i}.ssm_conv1d.weight" in self.t:
                self.layer_types.append("L")
            else:
                self.layer_types.append("F")

    def w(self, name):
        if name not in self._cache:
            self._cache[name] = dequantize(self.t[name])
        return self._cache[name]

    def vec(self, name):
        return self.w(name).flatten()

    def gemv(self, name, x):
        """y[out] = W[out,in] @ x[in]; weight stored [out,in].

        For huge weights (the lm_head / embedding) the fully-materialized f32
        matrix would be multiple GiB, so Q8_0 weights are dequantized and
        contracted in row chunks to keep peak memory bounded.
        """
        tensor = self.t[name]
        if tensor.tensor_type.name == "Q8_0":
            raw = tensor.data  # (rows, row_bytes) uint8
            rows, row_bytes = raw.shape
            nb = row_bytes // 34
            ncols = nb * 32
            assert x.shape[0] == ncols, f"{name}: x {x.shape} vs ncols {ncols}"
            out = np.empty(rows, dtype=np.float32)
            chunk = max(1, (1 << 24) // max(row_bytes, 1))  # ~16M bytes/chunk
            for r0 in range(0, rows, chunk):
                r1 = min(r0 + chunk, rows)
                blk = raw[r0:r1].reshape(r1 - r0, nb, 34)
                d = blk[:, :, :2].copy().view(np.float16).astype(np.float32)
                q = blk[:, :, 2:].view(np.int8).astype(np.float32)
                deq = (q * d).reshape(r1 - r0, ncols)
                out[r0:r1] = deq @ x
            return out
        W = self.w(name)
        return W @ x

    def embed(self, token):
        # Dequant just the one Q8_0 row (the full table is ~4.7 GiB in f32).
        tensor = self.t["token_embd.weight"]
        raw = tensor.data[token]  # (row_bytes,) uint8
        nb = raw.shape[0] // 34
        blk = raw.reshape(nb, 34)
        d = blk[:, :2].copy().view(np.float16).astype(np.float32)
        q = blk[:, 2:].view(np.int8).astype(np.float32)
        return (q * d).reshape(nb * 32)


def dump(tag, v):
    v = np.asarray(v, dtype=np.float32).flatten()
    head = " ".join(f"{x:+.5f}" for x in v[:6])
    print(f"  [{tag:28s}] l2={np.linalg.norm(v):+.5f} first6=[{head}]")


def linear_attention(m, layer, normed, state):
    """Gated-delta linear attention for one token. Returns out_proj result."""
    qkv = m.gemv(f"blk.{layer}.attn_qkv.weight", normed)
    z = m.gemv(f"blk.{layer}.attn_gate.weight", normed)
    a_proj = m.gemv(f"blk.{layer}.ssm_alpha.weight", normed)
    b_proj = m.gemv(f"blk.{layer}.ssm_beta.weight", normed)
    dump("lin.qkv", qkv)
    dump("lin.z", z)
    dump("lin.a_proj", a_proj)
    dump("lin.b_proj", b_proj)

    # depthwise causal conv1d (kernel taps = [ring | x]), SiLU. Single token =>
    # ring is all zeros on first call, so conv = x * conv_w[:, kernel-1].
    conv_w = m.w(f"blk.{layer}.ssm_conv1d.weight").reshape(QKV_DIM, L_KERNEL)
    ring = state["conv_ring"]  # (QKV_DIM, kernel-1)
    sw = L_KERNEL - 1
    taps = np.concatenate([ring, qkv[:, None]], axis=1)  # (QKV_DIM, kernel)
    pre = np.sum(taps * conv_w, axis=1)
    qkv_conv = silu(round_to_bf16(pre))
    dump("lin.qkv_conv", qkv_conv)
    # advance ring
    state["conv_ring"] = np.concatenate([ring[:, 1:], qkv[:, None]], axis=1)

    a_log = m.vec(f"blk.{layer}.ssm_a")          # (nv,)
    dt_bias = m.vec(f"blk.{layer}.ssm_dt.bias")  # (nv,)

    gdr = state["gdr"]  # (nv, kd, vd)
    out = np.zeros(V_DIM_T, dtype=np.float32)
    for vh in range(L_NV):
        # llama.cpp repeats the 16 key heads to 48 value heads via ggml_repeat
        # (delta-net-base.cpp:362 / qwen35.cpp:326-327), which TILES: dst head
        # vh reads src head `vh % num_k_heads`, NOT `vh * nk / nv`. See
        # ggml_compute_forward_repeat_f32 (dst dim1 = i1*ne01 + k1, k1 = src).
        kh = vh % L_NK
        q_head = qkv_conv[kh * L_KD: kh * L_KD + L_KD]
        k_head = qkv_conv[Q_DIM_T + kh * L_KD: Q_DIM_T + kh * L_KD + L_KD]
        v_head = qkv_conv[Q_DIM_T + K_DIM_T + vh * L_VD: Q_DIM_T + K_DIM_T + vh * L_VD + L_VD]

        q_norm = 1.0 / np.sqrt(np.sum(q_head * q_head) + 1e-12)
        k_norm = 1.0 / np.sqrt(np.sum(k_head * k_head) + 1e-12)
        q_scale = q_norm * (1.0 / np.sqrt(L_KD))

        # GGUF ssm_a already stores A = -exp(A_log) (converter pre-applied -exp),
        # so the log-decay is A*softplus(dt), NOT -exp(A)*softplus(dt).
        if vh == 0 and layer == 0:
            print(f"  [ssm_a stats] min={a_log.min():+.4f} max={a_log.max():+.4f} mean={a_log.mean():+.4f}")
        exp_g = np.exp(a_log[vh] * softplus(a_proj[vh] + dt_bias[vh]))
        beta = sigmoid(b_proj[vh])

        S = gdr[vh]  # (kd, vd)
        kj = k_head * k_norm  # (kd,)
        # pass 1: decay + kv_mem
        S *= exp_g
        kv_mem = kj @ S  # (vd,)  sum_j S[j,val]*kj[j]
        delta = (v_head - kv_mem) * beta  # (vd,)
        # pass 2: rank-1 update + output
        S += np.outer(kj, delta)  # S[j,val] += kj[j]*delta[val]
        qj = q_head * q_scale
        out[vh * L_VD: vh * L_VD + L_VD] = qj @ S  # sum_j S[j,val]*qj[j]
        gdr[vh] = S
    dump("lin.gdr_out", out)

    # gated output RMSNorm: per head over val_dim, PLAIN weight, * silu(z).
    ssm_norm = m.vec(f"blk.{layer}.ssm_norm.weight")  # (vd,)
    normed_out = np.zeros(V_DIM_T, dtype=np.float32)
    for vh in range(L_NV):
        x = out[vh * L_VD: vh * L_VD + L_VD]
        gate = z[vh * L_VD: vh * L_VD + L_VD]
        inv = 1.0 / np.sqrt(np.mean(x * x) + RMS_EPS)
        normed_out[vh * L_VD: vh * L_VD + L_VD] = x * inv * ssm_norm * silu(gate)
    dump("lin.gated_norm", normed_out)

    res = m.gemv(f"blk.{layer}.ssm_out.weight", normed_out)
    dump("lin.out_proj", res)
    return res


def full_attention(m, layer, normed, full_state, pos):
    q_full = m.gemv(f"blk.{layer}.attn_q.weight", normed)   # 2*nq*hd
    k_in = m.gemv(f"blk.{layer}.attn_k.weight", normed)     # nkv*hd
    v_in = m.gemv(f"blk.{layer}.attn_v.weight", normed)     # nkv*hd
    dump("full.q_full", q_full)
    dump("full.k_in", k_in)
    dump("full.v_in", v_in)

    q_norm = m.vec(f"blk.{layer}.attn_q_norm.weight")  # (hd,)
    k_norm = m.vec(f"blk.{layer}.attn_k_norm.weight")  # (hd,)

    q = np.zeros(N_HEADS * HEAD_DIM, dtype=np.float32)
    for h in range(N_HEADS):
        src = q_full[h * 2 * HEAD_DIM: h * 2 * HEAD_DIM + HEAD_DIM]
        nh = rms_norm(src, q_norm)
        q[h * HEAD_DIM: h * HEAD_DIM + HEAD_DIM] = rope_neox(nh, pos)
    k_row = np.zeros(N_KV * HEAD_DIM, dtype=np.float32)
    for h in range(N_KV):
        src = k_in[h * HEAD_DIM: h * HEAD_DIM + HEAD_DIM]
        nh = rms_norm(src, k_norm)
        k_row[h * HEAD_DIM: h * HEAD_DIM + HEAD_DIM] = rope_neox(nh, pos)
    dump("full.q_roped", q)
    dump("full.k_roped", k_row)

    full_state["k"].append(k_row)
    full_state["v"].append(v_in.copy())
    K = np.stack(full_state["k"])  # (kv_len, kv_dim)
    V = np.stack(full_state["v"])
    kv_len = K.shape[0]

    scale = 1.0 / np.sqrt(HEAD_DIM)
    group = N_HEADS // N_KV
    attn = np.zeros(N_HEADS * HEAD_DIM, dtype=np.float32)
    for h in range(N_HEADS):
        kv_h = h // group
        qh = q[h * HEAD_DIM: h * HEAD_DIM + HEAD_DIM]
        Kh = K[:, kv_h * HEAD_DIM: kv_h * HEAD_DIM + HEAD_DIM]  # (kv_len, hd)
        Vh = V[:, kv_h * HEAD_DIM: kv_h * HEAD_DIM + HEAD_DIM]
        scores = (Kh @ qh) * scale
        scores -= scores.max()
        w = np.exp(scores)
        w /= w.sum()
        attn[h * HEAD_DIM: h * HEAD_DIM + HEAD_DIM] = w @ Vh
    dump("full.attn_sdpa", attn)

    # per-head sigmoid gate from q_full's gate half
    for h in range(N_HEADS):
        gate = q_full[h * 2 * HEAD_DIM + HEAD_DIM: h * 2 * HEAD_DIM + 2 * HEAD_DIM]
        attn[h * HEAD_DIM: h * HEAD_DIM + HEAD_DIM] *= sigmoid(gate)
    dump("full.attn_gated", attn)

    res = m.gemv(f"blk.{layer}.attn_output.weight", attn)
    dump("full.out_proj", res)
    return res


def run(tokens, dump_layers):
    m = Model(MODEL)
    print(f"layer types: {''.join(m.layer_types)}")
    # per-linear-layer state
    n_lin = m.layer_types.count("L")
    n_full = m.layer_types.count("F")
    # we run the full prefix; only dump for the LAST token of the prompt.
    lin_state = [{"gdr": np.zeros((L_NV, L_KD, L_VD), dtype=np.float32),
                  "conv_ring": np.zeros((QKV_DIM, L_KERNEL - 1), dtype=np.float32)}
                 for _ in range(n_lin)]
    full_state = [{"k": [], "v": []} for _ in range(n_full)]

    last_logits = None
    for pos, token in enumerate(tokens):
        is_last = pos == len(tokens) - 1
        hidden = m.embed(token)
        if is_last:
            print(f"\n=== token {token} at pos {pos} (LAST) ===")
            dump("embedding", hidden)
        li = fi = 0
        for layer, lt in enumerate(m.layer_types):
            do_dump = is_last and layer in dump_layers
            attn_norm = m.vec(f"blk.{layer}.attn_norm.weight")
            normed = rms_norm(hidden, attn_norm)
            if do_dump:
                print(f"\n--- layer {layer} ({lt}) ---")
                dump("input_rmsnorm", normed)
            if lt == "L":
                st = lin_state[li]
                if do_dump:
                    attn_out = linear_attention(m, layer, normed, st)
                else:
                    attn_out = _linear_quiet(m, layer, normed, st)
                li += 1
            else:
                st = full_state[fi]
                if do_dump:
                    attn_out = full_attention(m, layer, normed, st, pos)
                else:
                    attn_out = _full_quiet(m, layer, normed, st, pos)
                fi += 1

            post_sum = hidden + attn_out
            post_norm = m.vec(f"blk.{layer}.post_attention_norm.weight")
            mlp_in = rms_norm(post_sum, post_norm)
            gate = m.gemv(f"blk.{layer}.ffn_gate.weight", mlp_in)
            up = m.gemv(f"blk.{layer}.ffn_up.weight", mlp_in)
            act = silu(gate) * up
            mlp_out = m.gemv(f"blk.{layer}.ffn_down.weight", act)
            hidden = post_sum + mlp_out
            if do_dump:
                dump("attn_residual", post_sum)
                dump("mlp_out", mlp_out)
                dump("layer_out", hidden)

        final_norm = m.vec("output_norm.weight")
        normed = rms_norm(hidden, final_norm)
        logits = m.gemv("output.weight", normed)
        last_logits = logits

    print("\n=== final logits ===")
    dump("logits", last_logits)
    top5 = np.argsort(last_logits)[::-1][:5]
    print("  top5:")
    for i in top5:
        print(f"    token {int(i):>6} -> {last_logits[i]:+.5f}")
    return last_logits


# quiet variants reuse the verbose ones but suppress dumps by monkeypatching
def _linear_quiet(m, layer, normed, st):
    return _no_dump(linear_attention, m, layer, normed, st)


def _full_quiet(m, layer, normed, st, pos):
    return _no_dump(full_attention, m, layer, normed, st, pos)


def _no_dump(fn, *a):
    global dump
    old = dump
    dump = lambda *args, **kw: None  # noqa: E731
    try:
        return fn(*a)
    finally:
        dump = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+",
                    default=[760, 6511, 314, 9338, 369])
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 3])
    args = ap.parse_args()
    print(f"tokens={args.tokens} dump_layers={args.layers}")
    run(args.tokens, set(args.layers))


if __name__ == "__main__":
    main()
