#!/usr/bin/env python3
"""Convert GPTQ safetensors to ARLE W4A16 format.

GPTQ format:
  *.qweight  -> int32, 4-bit packed (8 weights per int32, low-nibble-first)
  *.scales   -> float16, per-group scales [K//group_size, N]
  *.qzeros   -> int32, zero points (symmetric = all 8s)

ARLE W4A16 format:
  *.weight         -> uint8, 4-bit packed (2 weights per byte, same bytes)
  *.weight_scale   -> BF16, per-group scales [K//group_size, N]
  (no zero-point tensor; hardcoded to 8)

Usage:
  python3 convert_gptq_to_w4a16.py <hf_repo> <output_dir> [--proxy <proxy>]

Example:
  python3 convert_gptq_to_w4a16.py Qwen/Qwen3.5-27B-GPTQ-Int4 /data/models/qwen35-27b-w4a16
"""
import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import safetensors
from safetensors.torch import load_file, save_file
import torch


def download_file(url, dest_path, proxy=None, chunk_size=8 * 1024 * 1024):
    """Download a file with progress."""
    import urllib.request

    if proxy:
        os.environ["HTTPS_PROXY"] = proxy
        os.environ["HTTP_PROXY"] = proxy

    dest_path = Path(dest_path)
    if dest_path.exists():
        print(f"  [skip] {dest_path.name} already exists")
        return

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    req = urllib.request.Request(url)
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=300)
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp_path, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    mb = downloaded / (1024 * 1024)
                    print(f"\r  downloading {dest_path.name}: {pct}% ({mb:.1f} MB)", end="", flush=True)
        tmp_path.rename(dest_path)
        dt = time.time() - t0
        mb = downloaded / (1024 * 1024)
        print(f"\r  downloaded {dest_path.name}: {mb:.1f} MB in {dt:.1f}s ({mb/dt:.1f} MB/s)")
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}") from e


def convert_shard(input_path, output_path):
    """Convert one GPTQ safetensors shard to W4A16 format."""
    print(f"  converting {Path(input_path).name}...")
    t0 = time.time()

    tensors = load_file(input_path)
    new_tensors = {}
    dropped = 0

    for key, tensor in tensors.items():
        if key.endswith(".qweight"):
            new_key = key[: -len(".qweight")] + ".weight"
            # int32 [K//8, N] -> uint8 [K//2, N] (reinterpret bytes)
            t = tensor.contiguous()
            raw = t.view(torch.uint8)  # [K//8, N*4]
            raw = raw.reshape(t.shape[0] * 4, t.shape[1])  # [K//2, N]
            new_tensors[new_key] = raw

        elif key.endswith(".scales"):
            new_key = key[: -len(".scales")] + ".weight_scale"
            # float16 -> BF16 (via float32)
            bf16 = tensor.to(torch.float32).to(torch.bfloat16)
            new_tensors[new_key] = bf16

        elif key.endswith(".qzeros"):
            # Verify symmetric (all 8s), then drop
            if not torch.all(tensor == 8):
                # Print warning but still drop — ARLE hardcodes zp=8
                unique = tensor.unique()
                print(f"    WARNING: {key} has non-8 qzeros: {unique[:5]}")
            dropped += 1
            continue

        else:
            # Keep as-is (BF16 attention, norms, embed, lm_head, etc.)
            new_tensors[key] = tensor

    # Save to temp then rename (input may be mmap'd by load_file)
    tmp_out = Path(output_path).with_suffix(".converted.tmp")
    save_file(new_tensors, str(tmp_out))
    tmp_out.rename(output_path)
    dt = time.time() - t0
    print(f"    saved {Path(output_path).name}: {len(new_tensors)} tensors, dropped {dropped} qzeros ({dt:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Convert GPTQ safetensors to ARLE W4A16 format")
    parser.add_argument("repo", help="HF repo ID, e.g. Qwen/Qwen3.5-27B-GPTQ-Int4")
    parser.add_argument("output_dir", help="Output directory for converted model")
    parser.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY", ""), help="HTTP proxy")
    parser.add_argument("--keep-original", action="store_true", help="Keep original GPTQ shards after conversion")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"https://huggingface.co/{args.repo}/resolve/main"

    # 1. Download + parse config.json
    print(f"Fetching config.json for {args.repo}...")
    config_path = out_dir / "config.json.orig"
    download_file(f"{base_url}/config.json", config_path, args.proxy)
    with open(config_path) as f:
        config = json.load(f)

    qc = config.get("quantization_config", {})
    assert qc.get("quant_method") == "gptq", f"Not GPTQ: {qc.get('quant_method')}"
    assert qc.get("bits") == 4, f"Not 4-bit: {qc.get('bits')}"
    assert qc.get("sym") is True, f"Not symmetric: {qc.get('sym')}"
    assert qc.get("desc_act") is False, f"Has act-order (not supported): {qc.get('desc_act')}"
    group_size = qc.get("group_size", 128)
    print(f"  GPTQ 4-bit, group_size={group_size}, sym=True, desc_act=False")

    # 2. Download + parse index.json
    print("Fetching model.safetensors.index.json...")
    index_path = out_dir / "model.safetensors.index.json.orig"
    download_file(f"{base_url}/model.safetensors.index.json", index_path, args.proxy)
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    print(f"  {len(weight_map)} tensors in {len(shards)} shards")

    # 3. Build new weight map (rename .qweight -> .weight, .scales -> .weight_scale, drop .qzeros)
    new_weight_map = {}
    for key, shard in weight_map.items():
        if key.endswith(".qzeros"):
            continue
        if key.endswith(".qweight"):
            new_key = key[: -len(".qweight")] + ".weight"
        elif key.endswith(".scales"):
            new_key = key[: -len(".scales")] + ".weight_scale"
        else:
            new_key = key
        new_shard = shard  # keep same shard name
        new_weight_map[new_key] = new_shard

    # 4. Convert each shard
    for shard in shards:
        # Download original shard
        shard_url = f"{base_url}/{shard}"
        shard_path = out_dir / shard
        download_file(shard_url, shard_path, args.proxy)

        # Convert
        out_shard = out_dir / shard  # overwrite in place
        convert_shard(str(shard_path), str(out_shard))

        # Remove original if not keeping
        if not args.keep_original:
            # The converted shard overwrites the original, so nothing to remove
            pass

    # 5. Write new index.json
    new_index = {
        "metadata": index.get("metadata", {}),
        "weight_map": new_weight_map,
    }
    new_index_path = out_dir / "model.safetensors.index.json"
    with open(new_index_path, "w") as f:
        json.dump(new_index, f, indent=2)
    print(f"Wrote {new_index_path.name}: {len(new_weight_map)} tensors")

    # 6. Write new config.json with W4A16 quantization_config
    # Determine modules_to_not_convert from GPTQ dynamic field
    dynamic = qc.get("dynamic", {})
    not_convert = []
    for key in dynamic:
        if key.startswith("-:"):
            # Regex pattern — convert to module name hint
            pattern = key[2:]
            not_convert.append(pattern)
        else:
            not_convert.append(key)

    # Also add standard non-quantized modules
    for m in ["lm_head", "model.embed_tokens", "model.norm", "mtp"]:
        if m not in not_convert:
            not_convert.append(m)

    new_config = dict(config)
    new_config["quantization_config"] = {
        "quant_method": "w4a16",
        "bits": 4,
        "group_size": group_size,
        "sym": True,
        "desc_act": False,
        "modules_to_not_convert": not_convert,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(new_config, f, indent=2)
    print(f"Wrote config.json with W4A16 quantization_config")

    # 7. Download tokenizer files
    print("Downloading tokenizer files...")
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "chat_template.jinja",
        "generation_config.json",
    ]
    for tf in tokenizer_files:
        try:
            download_file(f"{base_url}/{tf}", out_dir / tf, args.proxy)
        except Exception:
            pass  # optional files

    # Cleanup original index
    if not args.keep_original:
        for p in [config_path, index_path]:
            if p.exists():
                p.unlink()

    print(f"\nDone! Converted model saved to {out_dir}")
    print(f"  Run with: arle serve --model-path {out_dir} --backend cuda --port 8000")


if __name__ == "__main__":
    main()
