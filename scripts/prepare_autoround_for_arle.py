"""Prepare AutoRound checkpoint for ARLE — dequant ALL non-expert weights to BF16.
Keep experts in GPTQ format (ARLE handles natively).
"""
import torch, os, glob, json, shutil
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/home/chenkailun.c/models/Qwen3.6-35B-A3B-AutoRound-W4A16"
DST = "/home/chenkailun.c/models/Qwen3.6-35B-A3B-AutoRound-W4A16-arle"
GS = 128

os.makedirs(DST, exist_ok=True)

for f in os.listdir(SRC):
    if not f.endswith(".safetensors"):
        src_path = os.path.join(SRC, f)
        dst_path = os.path.join(DST, f)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)

def dequant_gptq(qweight, scales, qzeros, group_size):
    k_packed, n = qweight.shape
    k = k_packed * 8
    groups = scales.shape[0]
    qw = qweight.long()
    unpacked = torch.zeros(k, n, dtype=torch.long, device=qweight.device)
    for i in range(8):
        unpacked[i::8] = (qw >> (i * 4)) & 0xF
    qz = qzeros.long()
    zeros_unpacked = torch.zeros(groups, n, dtype=torch.long, device=qzeros.device)
    for i in range(8):
        zeros_unpacked[:, i::8] = (qz >> (i * 4)) & 0xF
    zeros = zeros_unpacked + 1
    result = torch.zeros(k, n, dtype=scales.dtype, device=qweight.device)
    for g in range(groups):
        k_start = g * group_size
        k_end = min((g + 1) * group_size, k)
        result[k_start:k_end] = (unpacked[k_start:k_end].float() - zeros[g].float().unsqueeze(0)) * scales[g].float().unsqueeze(0)
    return result.to(torch.bfloat16)

files = sorted(glob.glob(f"{SRC}/model-*.safetensors"))
new_index = {}

for fp in files:
    fname = os.path.basename(fp)
    print(f"Processing {fname}...", flush=True)

    tensors = {}
    with safe_open(fp, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    new_tensors = {}
    for k, v in tensors.items():
        nk = k
        if nk.startswith("model.") and not nk.startswith("model.language_model."):
            nk = "model.language_model." + nk[len("model."):]

        # Dequantize ALL non-expert GPTQ weights to BF16
        if k.endswith(".qweight") and "experts" not in k:
            sk = k.replace(".qweight", ".scales")
            zk = k.replace(".qweight", ".qzeros")
            if sk in tensors and zk in tensors:
                qw = tensors[k]
                s = tensors[sk]
                qz = tensors[zk]
                w = dequant_gptq(qw, s, qz, GS).t().contiguous()  # [k,n]=[in,out] -> [out,in]
                new_key = nk.replace(".qweight", ".weight")
                new_tensors[new_key] = w
                continue

        # Skip non-expert scales/qzeros (replaced by BF16 weight)
        if any(x in k for x in [".scales", ".qzeros", ".g_idx"]) and "experts" not in k:
            continue

        new_tensors[nk] = v

    dst_path = os.path.join(DST, fname)
    save_file(new_tensors, dst_path)
    print(f"  Saved: {fname} ({len(new_tensors)} tensors)", flush=True)

    for k in new_tensors.keys():
        new_index[k] = fname

# Add quantization_config to config.json
config_path = os.path.join(DST, "config.json")
with open(config_path) as f:
    cfg = json.load(f)
cfg["quantization_config"] = {
    "quant_method": "autoround",
    "format": "gptq",
    "group_size": GS,
    "bits": 4,
    "desc_act": False,
    "sym": True,
}
with open(config_path, "w") as f:
    json.dump(cfg, f, indent=2)
print("Updated config.json with quantization_config", flush=True)

index_path = os.path.join(DST, "model.safetensors.index.json")
with open(index_path, "w") as f:
    json.dump({
        "metadata": {"total_size": 0},
        "weight_map": new_index
    }, f, indent=2)
print(f"Done! ARLE-ready checkpoint saved to {DST}", flush=True)
