"""ARLE W8A16 checkpoint -> GPTQ v1 (8-bit, gs=128, sym) for SGLang gptq_marlin.

Mechanical repack, no re-quantization: the int8 values ARLE serves are packed
verbatim (uint8 = int8+128, kU8B128 semantics), so both A/B arms run identical
quantized weights. Language-model projections -> qweight/qzeros/scales/g_idx;
visual/mtp quantized tensors -> dequantized bf16 (unused in the text bench);
in_proj_b/a stay bf16 (excluded via GPTQModel `dynamic` negative match:
N=48 per shard < marlin's 64-alignment). qzeros = 0x7F7F7F7F (v1 stores zp-1).
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/host/nvme0/models/iso-tc-huihui-w8a16")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "/host/nvme0/models/iso-tc-huihui-gptq8")
GS = 128
SHARD_BYTES = 5 * 2**30

idx = json.load(open(SRC / "model.safetensors.index.json"))
wmap = idx["weight_map"]
names = list(wmap)
scale_bases = {n[: -len(".weight_scale")] for n in names if n.endswith(".weight_scale")}

DST.mkdir(parents=True, exist_ok=True)
for f in SRC.iterdir():
    if f.suffix != ".safetensors" and f.name != "model.safetensors.index.json" and f.is_file():
        shutil.copy2(f, DST / f.name)

cfg = json.load(open(SRC / "config.json"))
cfg["quantization_config"] = {
    "quant_method": "gptq",
    "bits": 8,
    "group_size": GS,
    "sym": True,
    "desc_act": False,
    "static_groups": False,
    "true_sequential": True,
    "lm_head": False,
    "checkpoint_format": "gptq",
    "dynamic": {
        "-:.*in_proj_ba.*": {},
        "-:.*visual.*": {},
        "-:.*mtp.*": {},
    },
}
json.dump(cfg, open(DST / "config.json", "w"), indent=2)

handles = {}


def get(name):
    shard = wmap[name]
    if shard not in handles:
        handles[shard] = safe_open(SRC / shard, framework="pt", device="cpu")
    return handles[shard].get_tensor(name)


def pack_gptq(w_int8, scale_bf16):
    n, k = w_int8.shape
    assert k % GS == 0 and k % 4 == 0, (n, k)
    q = (w_int8.to(torch.int32) + 128).t().contiguous()  # [K, N] in [0,255]
    q = q.view(k // 4, 4, n)
    qweight = q[:, 0] | (q[:, 1] << 8) | (q[:, 2] << 16) | (q[:, 3] << 24)
    scales = scale_bf16.t().contiguous().to(torch.float16)  # [K/gs, N]
    assert n % 4 == 0, n
    qzeros = torch.full((k // GS, n // 4), 0x7F7F7F7F, dtype=torch.int32)
    g_idx = (torch.arange(k, dtype=torch.int32) // GS).contiguous()
    return qweight.contiguous(), qzeros, scales, g_idx


out_map, shard_bufs, shard_sizes = {}, {}, {}
cur_id, cur_bytes = 0, 0


def emit(name, t):
    global cur_id, cur_bytes
    nbytes = t.numel() * t.element_size()
    if cur_bytes and cur_bytes + nbytes > SHARD_BYTES:
        cur_id += 1
        cur_bytes = 0
    shard = f"model-{cur_id:05d}.safetensors"
    shard_bufs.setdefault(shard, {})[name] = t
    out_map[name] = shard
    cur_bytes += nbytes


def flush(final=False):
    active = f"model-{cur_id:05d}.safetensors"
    for shard, bufs in list(shard_bufs.items()):
        if not final and shard == active:
            continue  # still receiving tensors
        save_file(bufs, DST / shard)
        shard_sizes[shard] = sum(t.numel() * t.element_size() for t in bufs.values())
        del shard_bufs[shard]


done = 0
for name in sorted(names):
    if name.endswith(".weight_scale"):
        continue
    base = name[: -len(".weight")] if name.endswith(".weight") else None
    if base in scale_bases:
        w = get(name)
        s = get(base + ".weight_scale")
        if base.startswith("model.language_model."):
            qweight, qzeros, scales, g_idx = pack_gptq(w, s)
            emit(base + ".qweight", qweight)
            emit(base + ".qzeros", qzeros)
            emit(base + ".scales", scales)
            emit(base + ".g_idx", g_idx)
        else:  # visual / mtp: dequant back to bf16
            deq = w.to(torch.bfloat16) * s.repeat_interleave(GS, dim=1)
            emit(name, deq.contiguous())
    else:
        emit(name, get(name).contiguous())
    done += 1
    if done % 200 == 0:
        print(f"{done} tensors", flush=True)
        flush()
flush(final=True)

json.dump(
    {"metadata": {"total_size": sum(shard_sizes.values())}, "weight_map": out_map},
    open(DST / "model.safetensors.index.json", "w"),
)
print(f"DONE tensors={len(out_map)} shards={len(shard_sizes)} bytes={sum(shard_sizes.values())}")
