#!/usr/bin/env python3
"""Make a W8A16 checkpoint whose lm_head IS quantised and NOT tied.

`scripts/quantize.py` skips `lm_head.weight` on purpose (W8A16_SKIP_ENDINGS),
so the path `834a87aed` added has never had an input. This builds one from the
existing W8A16 model: untie, then quantise the tied embedding into an lm_head
using the same per-row per-128-group symmetric INT8 the quantiser uses.
"""
import json, os, shutil, struct, sys
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "/data00/qwen35-08b-w8a16"
DST = sys.argv[2] if len(sys.argv) > 2 else "/data00/qwen35-08b-w8a16-lmhead"
GS = 128

def read(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        blob = f.read()
    return hdr, blob

def bf16_to_f32(raw):
    u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    return (u16 << 16).view(np.float32) if False else np.frombuffer(
        (u16 << 16).astype(np.uint32).tobytes(), dtype=np.float32)

def f32_to_bf16(a):
    u32 = a.astype(np.float32).view(np.uint32)
    # round-to-nearest-even
    r = ((u32 >> 16) & 1) + 0x7FFF
    return (((u32 + r) >> 16).astype(np.uint16)).tobytes()

os.makedirs(DST, exist_ok=True)
files = sorted(f for f in os.listdir(SRC) if f.endswith(".safetensors"))
assert len(files) == 1, files
hdr, blob = read(os.path.join(SRC, files[0]))

emb = hdr["model.language_model.embed_tokens.weight"]
assert emb["dtype"] == "BF16", emb["dtype"]
rows, cols = emb["shape"]
a, b = emb["data_offsets"]
w = bf16_to_f32(blob[a:b]).reshape(rows, cols)
assert cols % GS == 0, cols

# per-row, per-128-group symmetric INT8 — the quantiser's `per_group_int8`
v = w.reshape(rows, cols // GS, GS)
scale = np.abs(v).max(axis=2) / 127.0
scale[scale == 0] = 1.0
q = np.rint(v / scale[:, :, None]).clip(-127, 127).astype(np.int8).reshape(rows, cols)

err = np.abs(q.reshape(rows, cols // GS, GS) * scale[:, :, None] - v).max()
print(f"lm_head [{rows}, {cols}] -> I8 + BF16 scale [{rows}, {cols // GS}]  max abs err {err:.4g}")

out, new_hdr, off = bytearray(), {}, 0
def put(name, arr_bytes, dtype, shape):
    global off
    new_hdr[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + len(arr_bytes)]}
    out.extend(arr_bytes); off += len(arr_bytes)

for name, meta in hdr.items():
    if name == "__metadata__":
        continue
    s, e = meta["data_offsets"]
    put(name, blob[s:e], meta["dtype"], meta["shape"])
put("lm_head.weight", q.tobytes(), "I8", [rows, cols])
put("lm_head.weight_scale", f32_to_bf16(scale), "BF16", [rows, cols // GS])

hdr_bytes = json.dumps(new_hdr, separators=(",", ":")).encode()
pad = (-len(hdr_bytes)) % 8
hdr_bytes += b" " * pad
with open(os.path.join(DST, files[0]), "wb") as f:
    f.write(struct.pack("<Q", len(hdr_bytes))); f.write(hdr_bytes); f.write(bytes(out))

for extra in os.listdir(SRC):
    if not extra.endswith(".safetensors"):
        shutil.copy2(os.path.join(SRC, extra), os.path.join(DST, extra))
cfg_path = os.path.join(DST, "config.json")
cfg = json.load(open(cfg_path))
cfg["tie_word_embeddings"] = False
json.dump(cfg, open(cfg_path, "w"), indent=1)
print(f"wrote {DST}  tie_word_embeddings=False  lm_head quantised at group {GS}")
