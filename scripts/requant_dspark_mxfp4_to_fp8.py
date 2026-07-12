#!/usr/bin/env python3
"""Requant the DSpark draft's MXFP4 routed experts -> fp8 e4m3 block-128, so each
stage loads as all-fp8 on the existing DSv4 fp8 MoE path. Stage-agnostic: the
tensor match keys on `.ffn.experts.`, so one invocation converts whatever stage(s)
the given shard(s) hold. Pass one or more src dst pairs (mtp.0/1/2 = shard 46/47/48).

Manual safetensors I/O (raw bytes) — the numpy/torch frameworks can't materialize
F8_E8M0, so we interpret every tensor's bytes ourselves. Deps: numpy + ml_dtypes.

MXFP4 in (experts.{i}.w{1,2,3}): weight I8 [rows, K/2] (e2m1, 2 nibbles/byte,
  logical K = 2*stored), scale F8_E8M0 [rows, K/32] (ue8m0, group=32).
fp8 out: weight F8_E4M3 [rows, K], scale F8_E8M0 [rows/128, K/128] (block 128,
  ue8m0 power-of-two — matches the base DSv4 fp8 experts).
Everything else (attn / main_proj / shared_experts / gate / norms) copied through.
"""
import sys, json, struct, numpy as np, ml_dtypes

FP8_MAX = 448.0
assert hasattr(ml_dtypes, "float4_e2m1fn"), "need ml_dtypes with float4_e2m1fn"
_E2M1 = np.arange(16, dtype=np.uint8).view(ml_dtypes.float4_e2m1fn).astype(np.float32)

# safetensors dtype name -> (numpy container dtype for raw view, bytes/elem)
_ST = {"I8": (np.int8, 1), "F8_E8M0": (np.uint8, 1), "F8_E4M3": (np.uint8, 1),
       "BF16": (ml_dtypes.bfloat16, 2), "F16": (np.float16, 2),
       "F32": (np.float32, 4), "I32": (np.int32, 4), "I64": (np.int64, 8)}

def read_st(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        blob = fh.read()  # data section
    hdr.pop("__metadata__", None)
    return hdr, blob

def raw(hdr, blob, name):
    e = hdr[name]
    s, t = e["data_offsets"]
    return e["dtype"], e["shape"], blob[s:t]

def dequant_mxfp4(w_bytes, w_shape, s_bytes, group=32):
    rows, half = w_shape  # I8 [rows, K/2]
    b = np.frombuffer(w_bytes, dtype=np.uint8).reshape(rows, half)
    nib = np.empty((rows, half * 2), dtype=np.uint8)
    nib[:, 0::2] = b & 0x0F           # low nibble = even element
    nib[:, 1::2] = (b >> 4) & 0x0F
    vals = _E2M1[nib]                 # [rows, K]
    K = half * 2
    exp = np.frombuffer(s_bytes, dtype=np.uint8).reshape(rows, K // group).astype(np.int32) - 127
    scale = np.repeat(np.exp2(exp.astype(np.float32)), group, axis=1)
    return vals * scale               # f32 [rows, K]

def quant_fp8_block(w, block=128):
    R, C = w.shape
    assert R % block == 0 and C % block == 0, (R, C)
    wb = w.reshape(R // block, block, C // block, block)
    absmax = np.maximum(np.abs(wb).max(axis=(1, 3)), 1e-12)
    exp = np.ceil(np.log2(absmax / FP8_MAX)).astype(np.int32)   # smallest pow2 s.t. fits
    scale = np.exp2(exp.astype(np.float32))
    w_fp8 = (wb / scale[:, None, :, None]).reshape(R, C).astype(ml_dtypes.float8_e4m3fn)
    s_e8 = (exp + 127).astype(np.uint8)                         # ue8m0 raw byte
    return w_fp8.view(np.uint8).tobytes(), (R, C), s_e8.tobytes(), (R // block, C // block)

def main(src, dst):
    hdr, blob = read_st(src)
    out_hdr, out_blobs, off, n_req = {}, [], 0, 0
    def emit(name, dtype, shape, data):
        nonlocal off
        out_hdr[name] = {"dtype": dtype, "shape": list(shape),
                         "data_offsets": [off, off + len(data)]}
        out_blobs.append(data); off += len(data)

    for name in hdr:
        is_w = (".ffn.experts." in name and name.endswith(".weight")
                and any(f".{w}.weight" in name for w in ("w1", "w2", "w3")))
        if ".ffn.experts." in name and name.endswith(".scale"):
            continue  # emitted with its weight
        if not is_w:
            dt, sh, data = raw(hdr, blob, name)
            emit(name, dt, sh, data); continue
        base = name[:-len(".weight")]
        wdt, wsh, wb = raw(hdr, blob, name)
        _, _, sb = raw(hdr, blob, base + ".scale")
        assert wdt == "I8", (name, wdt)
        w_f32 = dequant_mxfp4(wb, wsh, sb, group=32)
        wq, wq_sh, sq, sq_sh = quant_fp8_block(w_f32, block=128)
        emit(name, "F8_E4M3", wq_sh, wq)
        emit(base + ".scale", "F8_E8M0", sq_sh, sq)
        n_req += 1
        if n_req <= 3:
            req = (np.frombuffer(wq, np.uint8).view(ml_dtypes.float8_e4m3fn)
                   .astype(np.float32).reshape(wq_sh))
            se = (np.frombuffer(sq, np.uint8).astype(np.int32) - 127).reshape(sq_sh)
            req = req * np.repeat(np.repeat(np.exp2(se.astype(np.float32)), 128, 0), 128, 1)
            fro = np.linalg.norm(req - w_f32) / (np.linalg.norm(w_f32) + 1e-9)
            print(f"  [check] {base}: {wsh}->{list(wq_sh)} frob-rel-err {fro:.4f}", flush=True)
            assert fro < 0.10, f"requant Frobenius error too high: {fro}"

    print(f"requantized {n_req} expert matrices (expect 768)", flush=True)
    assert n_req == 768, f"expected 768, got {n_req}"
    hdr_bytes = json.dumps(out_hdr, separators=(",", ":")).encode()
    pad = (-len(hdr_bytes)) % 8
    hdr_bytes += b" " * pad
    with open(dst, "wb") as fh:
        fh.write(struct.pack("<Q", len(hdr_bytes)))
        fh.write(hdr_bytes)
        for b in out_blobs:
            fh.write(b)
    print(f"wrote {dst} ({(off)/1e9:.2f} GB data)", flush=True)

if __name__ == "__main__":
    args = sys.argv[1:]
    assert args and len(args) % 2 == 0, "usage: script.py src1 dst1 [src2 dst2 ...]"
    for src, dst in zip(args[0::2], args[1::2]):
        print(f"=== {src} -> {dst} ===", flush=True)
        main(src, dst)
