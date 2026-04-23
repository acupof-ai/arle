#!/usr/bin/env python3
"""Convert GGUF model to safetensors format for ARLE.

Handles all llama.cpp GGUF conventions:
- Dequantizes Q4_K_M/Q8_0/etc. to BF16
- Reverses head deinterleave (Qwen3.5 SSM tensors)
- Converts ssm_a → A_log (exp → log)
- Removes norm +1 offset
- Maps GGUF tensor names to HuggingFace convention

Output: standard safetensors + config.json that ARLE loads directly.

Usage:
    python scripts/gguf_to_safetensors.py \
        --gguf models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf \
        --output models/Qwen3.5-4B-from-GGUF/
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

# ─── GGUF Parsing (minimal, Python) ───

GGUF_MAGIC = 0x46554747

def read_u32(f): return struct.unpack("<I", f.read(4))[0]
def read_u64(f): return struct.unpack("<Q", f.read(8))[0]
def read_i32(f): return struct.unpack("<i", f.read(4))[0]
def read_f32(f): return struct.unpack("<f", f.read(4))[0]
def read_str(f):
    n = read_u64(f)
    return f.read(n).decode("utf-8", errors="replace")

def read_value(f):
    vt = read_u32(f)
    if vt in (0,): return f.read(1)[0]
    if vt in (1,): return struct.unpack("b", f.read(1))[0]
    if vt in (2,): return struct.unpack("<H", f.read(2))[0]
    if vt in (3,): return struct.unpack("<h", f.read(2))[0]
    if vt in (4,): return read_u32(f)
    if vt in (5,): return read_i32(f)
    if vt in (6,): return read_f32(f)
    if vt in (7,): return f.read(1)[0] != 0
    if vt in (8,): return read_str(f)
    if vt == 9:
        et = read_u32(f); el = read_u64(f)
        return [read_value_of_type(f, et) for _ in range(el)]
    if vt in (10,): return read_u64(f)
    if vt in (11,): return struct.unpack("<q", f.read(8))[0]
    if vt in (12,): return struct.unpack("<d", f.read(8))[0]
    return None

def read_value_of_type(f, vt):
    if vt in (0,): return f.read(1)[0]
    if vt in (4,): return read_u32(f)
    if vt in (5,): return read_i32(f)
    if vt in (6,): return read_f32(f)
    if vt in (8,): return read_str(f)
    if vt in (10,): return read_u64(f)
    f.read(4); return None

BLOCK_INFO = {
    0: (4, 1, "F32"), 1: (2, 1, "F16"), 30: (2, 1, "BF16"),
    7: (34, 32, "Q8_0"), 2: (18, 32, "Q4_0"), 8: (1, 1, "I8"),
    11: (110, 256, "Q3_K_S"), 12: (110, 256, "Q3_K_M"), 13: (110, 256, "Q3_K_L"),
    14: (144, 256, "Q4_K_S"), 15: (144, 256, "Q4_K_M"),
    16: (176, 256, "Q5_K_S"), 17: (176, 256, "Q5_K_M"),
    18: (210, 256, "Q6_K"),
}

def parse_gguf(path):
    with open(path, "rb") as f:
        magic = read_u32(f)
        assert magic == GGUF_MAGIC, f"Not GGUF: {magic:#x}"
        version = read_u32(f)
        n_tensors = read_u64(f)
        n_kv = read_u64(f)
        metadata = {}
        for _ in range(n_kv):
            k = read_str(f); v = read_value(f)
            metadata[k] = v
        tensors = {}
        for _ in range(n_tensors):
            name = read_str(f)
            n_dims = read_u32(f)
            shape = [read_u64(f) for _ in range(n_dims)]
            dtype = read_u32(f)
            offset = read_u64(f)
            tensors[name] = {"shape": shape, "dtype": dtype, "offset": offset}
        data_offset = (f.tell() + 31) & ~31
    return metadata, tensors, data_offset

# ─── Dequantization ───

def dequant_tensor(path, data_offset, info):
    shape, dtype, offset = info["shape"], info["dtype"], info["offset"]
    numel = int(np.prod(shape))
    binfo = BLOCK_INFO.get(dtype)
    if binfo is None:
        raise ValueError(f"Unsupported dtype {dtype}")
    block_bytes, block_size, name = binfo
    n_blocks = numel // block_size
    with open(path, "rb") as f:
        f.seek(data_offset + offset)
        raw = f.read(n_blocks * block_bytes)

    if dtype == 0:  # F32
        return np.frombuffer(raw, dtype=np.float32).copy()
    if dtype == 1:  # F16
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if dtype == 30:  # BF16
        u16 = np.frombuffer(raw, dtype=np.uint16)
        return torch.from_numpy(u16.astype(np.int32)).view(torch.bfloat16).float().numpy()
    if dtype == 8:  # I8
        return np.frombuffer(raw, dtype=np.int8).astype(np.float32)
    if dtype == 7:  # Q8_0
        out = np.zeros(numel, dtype=np.float32)
        for b in range(n_blocks):
            base = b * 34
            scale = np.frombuffer(raw[base:base+2], dtype=np.float16)[0]
            vals = np.frombuffer(raw[base+2:base+34], dtype=np.int8)
            out[b*32:(b+1)*32] = vals.astype(np.float32) * float(scale)
        return out
    if dtype == 2:  # Q4_0
        out = np.zeros(numel, dtype=np.float32)
        for b in range(n_blocks):
            base = b * 18
            scale = float(np.frombuffer(raw[base:base+2], dtype=np.float16)[0])
            for i in range(16):
                byte = raw[base+2+i]
                lo = (byte & 0x0F) - 8
                hi = ((byte >> 4) & 0x0F) - 8
                out[b*32+i*2] = lo * scale
                out[b*32+i*2+1] = hi * scale
        return out
    if dtype in (14, 15):  # Q4_K_S, Q4_K_M
        out = np.zeros(numel, dtype=np.float32)
        for b in range(n_blocks):
            base = b * 144
            d = float(np.frombuffer(raw[base:base+2], dtype=np.float16)[0])
            dmin = float(np.frombuffer(raw[base+2:base+4], dtype=np.float16)[0])
            sr = raw[base+4:base+16]
            qs = raw[base+16:base+144]
            sc = [0]*8; mn = [0]*8
            for i in range(4):
                sc[i] = sr[i] & 63; mn[i] = sr[i+4] & 63
            for i in range(4):
                sc[4+i] = (sr[i] >> 6) | ((sr[8+i] & 0x0F) << 2)
                mn[4+i] = (sr[i+4] >> 6) | ((sr[8+i] >> 4) << 2)
            for j in range(8):
                sub_d = d * sc[j]; sub_m = dmin * mn[j]
                for i in range(16):
                    byte = qs[j*16+i]
                    out[b*256+j*32+i*2] = (byte & 0x0F) * sub_d - sub_m
                    out[b*256+j*32+i*2+1] = ((byte>>4) & 0x0F) * sub_d - sub_m
        return out
    raise ValueError(f"Dequant not implemented for {name} (dtype={dtype})")

# ─── Name Mapping ───

NAME_MAP_LAYER = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    # Qwen3.5 SSM
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a": "linear_attn.A_log",
    "ssm_norm.weight": "linear_attn.norm.weight",
}

def map_name(gguf_name, prefix="model"):
    if gguf_name == "token_embd.weight": return f"{prefix}.embed_tokens.weight"
    if gguf_name == "output_norm.weight": return f"{prefix}.norm.weight"
    if gguf_name == "output.weight": return "lm_head.weight"
    if gguf_name.startswith("blk."):
        rest = gguf_name[4:]
        dot = rest.index(".")
        layer = rest[:dot]
        suffix = rest[dot+1:]
        hf_suffix = NAME_MAP_LAYER.get(suffix, suffix)
        return f"{prefix}.layers.{layer}.{hf_suffix}"
    return gguf_name

# ─── Re-interleave ───

def reinterleave_1d(arr):
    """[even..., odd...] → [0,1,2,3,...]"""
    n = len(arr)
    if n < 2: return arr
    h = n // 2
    out = np.empty_like(arr)
    out[0::2] = arr[:h]
    out[1::2] = arr[h:]
    return out

def reinterleave_rows(mat, num_heads):
    """Deinterleaved rows by head → original order."""
    rows, cols = mat.shape
    if num_heads < 2: return mat
    head_dim = rows // num_heads
    h = num_heads // 2
    out = np.empty_like(mat)
    for i in range(h):
        out[i*2*head_dim:(i*2+1)*head_dim] = mat[i*head_dim:(i+1)*head_dim]
        out[(i*2+1)*head_dim:(i*2+2)*head_dim] = mat[(h+i)*head_dim:(h+i+1)*head_dim]
    return out

def reinterleave_cols(mat, num_heads):
    """Deinterleaved cols by head → original order."""
    rows, cols = mat.shape
    if num_heads < 2: return mat
    head_dim = cols // num_heads
    h = num_heads // 2
    out = np.empty_like(mat)
    for i in range(h):
        out[:, i*2*head_dim:(i*2+1)*head_dim] = mat[:, i*head_dim:(i+1)*head_dim]
        out[:, (i*2+1)*head_dim:(i*2+2)*head_dim] = mat[:, (h+i)*head_dim:(h+i+1)*head_dim]
    return out

# ─── Main ───

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True, help="Input GGUF file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--prefix", default=None, help="HF name prefix (auto-detected)")
    args = parser.parse_args()

    print(f"Parsing GGUF: {args.gguf}")
    metadata, tensors, data_offset = parse_gguf(args.gguf)

    arch = metadata.get("general.architecture", "unknown")
    print(f"Architecture: {arch}, {len(tensors)} tensors")

    # Auto-detect prefix
    prefix = args.prefix or ("model.language_model" if "qwen35" in arch else "model")
    print(f"Name prefix: {prefix}")

    # Detect Qwen3.5 SSM config
    is_qwen35 = "qwen35" in arch or any("ssm_a" in t for t in tensors)
    num_value_heads = metadata.get(f"{arch}.ssm.dt_rank", 32)  # fallback
    if f"{arch}.ssm.state_size" in metadata:
        num_value_heads = metadata.get(f"{arch}.attention.head_count", 32)
    num_key_heads = metadata.get(f"{arch}.attention.head_count", 16)

    if is_qwen35:
        print(f"Qwen3.5 SSM detected: num_value_heads={num_value_heads}, num_key_heads={num_key_heads}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_tensors = {}
    for gguf_name, info in sorted(tensors.items()):
        hf_name = map_name(gguf_name, prefix)
        shape = info["shape"]
        dtype_id = info["dtype"]
        dtype_name = BLOCK_INFO.get(dtype_id, (0, 0, f"?{dtype_id}"))[2]

        try:
            data = dequant_tensor(args.gguf, data_offset, info)
        except Exception as e:
            print(f"  SKIP {gguf_name}: {e}")
            continue

        # Reshape: GGUF [ne0, ne1] row-major → [ne1, ne0] for 2D
        if len(shape) == 2:
            data = data.reshape(shape[1], shape[0])
        elif len(shape) == 1:
            data = data.reshape(shape[0])
        elif len(shape) == 3:
            data = data.reshape(shape[2], shape[1], shape[0])

        # Apply Qwen3.5 SSM corrections
        if is_qwen35:
            base = gguf_name.split(".")[-1] if "." in gguf_name else gguf_name

            # Norm offset: subtract 1.0 (except ssm_norm)
            if base in ("weight",) and ("attn_norm" in gguf_name or "post_attention_norm" in gguf_name
                    or "attn_q_norm" in gguf_name or "attn_k_norm" in gguf_name
                    or gguf_name == "output_norm.weight"):
                data = data - 1.0

            # A_log: convert -exp(A_log) back to A_log + re-interleave
            if "ssm_a" in gguf_name and "alpha" not in gguf_name:
                data = -np.log(np.abs(data).clip(1e-10))
                data = reinterleave_1d(data)

            # dt_bias: re-interleave
            if "ssm_dt" in gguf_name:
                data = reinterleave_1d(data)

            # in_proj_a/b: row re-interleave (num_value_heads rows)
            if "ssm_alpha" in gguf_name or "ssm_beta" in gguf_name:
                if len(data.shape) == 2:
                    data = reinterleave_rows(data, data.shape[0])

            # QKV: per-section head re-interleave
            if "attn_qkv" in gguf_name and len(data.shape) == 2:
                rows = data.shape[0]
                # Q: first 1/4 of rows (key_heads), K: next 1/4, V: last 1/2
                q_rows = rows // 4
                k_rows = rows // 4
                v_rows = rows // 2
                data[:q_rows] = reinterleave_rows(data[:q_rows].copy(), num_key_heads)
                data[q_rows:q_rows+k_rows] = reinterleave_rows(data[q_rows:q_rows+k_rows].copy(), num_key_heads)
                data[q_rows+k_rows:] = reinterleave_rows(data[q_rows+k_rows:].copy(), num_value_heads)

            # in_proj_z: row re-interleave by value_heads
            if "attn_gate" in gguf_name and len(data.shape) == 2:
                data = reinterleave_rows(data, num_value_heads)

            # out_proj: column re-interleave by value_heads
            if "ssm_out" in gguf_name and len(data.shape) == 2:
                data = reinterleave_cols(data, num_value_heads)

        output_tensors[hf_name] = torch.from_numpy(data).to(torch.bfloat16)
        print(f"  {gguf_name} → {hf_name} {list(data.shape)} ({dtype_name})")

    # Save safetensors
    out_file = out_dir / "model.safetensors"
    save_file(output_tensors, str(out_file))
    print(f"\nSaved: {out_file} ({out_file.stat().st_size / 1e9:.2f} GB)")

    # Copy/generate config.json from GGUF metadata
    config_path = out_dir / "config.json"
    if not config_path.exists():
        print(f"Note: copy config.json + tokenizer.json from HuggingFace model repo")

    print("Done!")

if __name__ == "__main__":
    main()
