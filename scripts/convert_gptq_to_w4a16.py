#!/usr/bin/env python3
"""Convert GPTQ v1 safetensors to ARLE W4A16."""

import argparse
import ctypes
import errno
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

GPTQ_VALUES_PER_WORD = 8
W4_VALUES_PER_BYTE = 2
INDEX_NAME = "model.safetensors.index.json"
SIDECAR_SUFFIXES = (".qweight", ".qzeros", ".scales", ".g_idx")


def convert_gptq_tensors(
    qweight,
    qzeros,
    scales,
    g_idx=None,
    *,
    group_size,
    tensor_name="qweight",
    channel_chunk_size=256,
):
    """Convert one symmetric GPTQ v1 tensor to row-major W4A16."""
    if qweight.dtype != torch.int32 or qweight.ndim != 2:
        raise ValueError(
            f"{tensor_name}: qweight must be rank-2 int32, got "
            f"{qweight.dtype} {tuple(qweight.shape)}"
        )
    if qzeros.dtype != torch.int32 or qzeros.ndim != 2:
        raise ValueError(
            f"{tensor_name}: qzeros must be rank-2 int32, got "
            f"{qzeros.dtype} {tuple(qzeros.shape)}"
        )
    if scales.ndim != 2:
        raise ValueError(
            f"{tensor_name}: scales must be rank 2, got {tuple(scales.shape)}"
        )
    if group_size <= 0:
        raise ValueError(
            f"{tensor_name}: group_size must be positive, got {group_size}"
        )
    if channel_chunk_size <= 0:
        raise ValueError(f"{tensor_name}: channel_chunk_size must be positive")

    packed_k, output_channels = qweight.shape
    k = packed_k * GPTQ_VALUES_PER_WORD
    if k % group_size != 0 or k % W4_VALUES_PER_BYTE != 0:
        raise ValueError(
            f"{tensor_name}: K={k} must be divisible by group_size={group_size} and 2"
        )

    num_groups = k // group_size
    expected_scales = (num_groups, output_channels)
    if output_channels % GPTQ_VALUES_PER_WORD != 0:
        raise ValueError(
            f"{tensor_name}: output channels {output_channels} must be divisible by 8"
        )
    expected_qzeros = (num_groups, output_channels // GPTQ_VALUES_PER_WORD)
    if tuple(scales.shape) != expected_scales:
        raise ValueError(
            f"{tensor_name}: scales shape {tuple(scales.shape)} != {expected_scales}"
        )
    if tuple(qzeros.shape) != expected_qzeros:
        raise ValueError(
            f"{tensor_name}: qzeros shape {tuple(qzeros.shape)} != {expected_qzeros}"
        )

    expected_g_idx = (
        torch.arange(k, device=qweight.device, dtype=torch.long) // group_size
    )
    has_desc_act = False
    if g_idx is not None:
        if (
            g_idx.ndim != 1
            or g_idx.numel() != k
            or g_idx.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(f"{tensor_name}: g_idx must be int32/int64 [{k}]")
        g_idx_long = g_idx.to(device=qweight.device, dtype=torch.long)
        if not torch.equal(g_idx_long, expected_g_idx):
            has_desc_act = True

    # Unpack qweight to signed int4 (stored/permuted column order).
    signed = torch.empty((output_channels, k), dtype=torch.int8, device=qweight.device)
    for start in range(0, output_channels, channel_chunk_size):
        end = min(start + channel_chunk_size, output_channels)
        channels = torch.arange(start, end, device=qweight.device, dtype=torch.long)
        zero_words = qzeros[:, channels // GPTQ_VALUES_PER_WORD]
        zero_shifts = (channels % GPTQ_VALUES_PER_WORD) * 4
        zeros = (((zero_words >> zero_shifts.unsqueeze(0)) & 0xF) + 1).to(torch.int8).T
        if not torch.all(zeros == 8):
            values = torch.unique(zeros).tolist()
            raise ValueError(
                f"{tensor_name}: symmetric GPTQ requires zero point 8, got {values[:8]}"
            )

        qweight_chunk = qweight[:, start:end]
        for nibble in range(GPTQ_VALUES_PER_WORD):
            unsigned = ((qweight_chunk >> (nibble * 4)) & 0xF).to(torch.int8).T
            groups = expected_g_idx[nibble::GPTQ_VALUES_PER_WORD]
            signed[start:end, nibble::GPTQ_VALUES_PER_WORD] = unsigned - zeros[:, groups]

    minimum = int(signed.min().item())
    maximum = int(signed.max().item())
    if minimum < -8 or maximum > 7:
        raise ValueError(
            f"{tensor_name}: signed codes out of range [-8, 7]: "
            f"min={minimum}, max={maximum}"
        )

    if has_desc_act:
        # desc_act: input channels were permuted by activation magnitude before
        # quantization. g_idx[i] gives the stored group of original channel i.
        # Dequantize in stored order, then unpermute to original channel order,
        # then re-quantize with canonical groups so the ARLE W4A16 kernel
        # (which assumes canonical groups) gets correct per-channel scales.
        stored_groups = expected_g_idx  # group of each stored column
        scales_f = scales.to(torch.float32)  # [num_groups, output_channels]
        dequant = signed.to(torch.float32) * scales_f[stored_groups].T  # [N, K]
        # Unpermute columns: original channel i lives at stored position
        # argsort(g_idx)[i]. Stable sort keeps original order within a group,
        # which is the best reconstruction possible from g_idx alone.
        inv_perm = torch.argsort(g_idx_long, stable=True)
        dequant = dequant[:, inv_perm]
        # Re-quantize with canonical groups (symmetric, zero-point 8).
        w = dequant.reshape(output_channels, num_groups, group_size)
        absmax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
        new_scales = (absmax / 7.0).squeeze(-1)  # [N, num_groups]
        w_q = torch.clamp(torch.round(w / absmax * 7.0), -8, 7).to(torch.int8)
        signed = w_q.reshape(output_channels, k)
        scales_out = new_scales.to(torch.bfloat16).contiguous()
    else:
        scales_out = scales.T.contiguous().to(torch.bfloat16)

    unsigned = (signed + 8).to(torch.uint8)
    packed = unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)

    return packed, scales_out


def _read_json(path):
    with path.open() as handle:
        return json.load(handle)


def _root_file(root, name):
    path = (root / name).resolve()
    if path.parent != root:
        raise ValueError(f"path escapes model directory: {name!r}")
    return path


def _rename_noreplace(src, dst):
    src = os.fsencode(src)
    dst = os.fsencode(dst)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        result = rename(-100, src, -100, dst, 1)
    elif sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = rename(src, dst, 4)
    else:
        raise OSError(
            errno.ENOTSUP, f"atomic no-replace rename is unsupported on {sys.platform}"
        )
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), os.fsdecode(dst))


def _source_quantization_config(input_dir, config):
    inline = config.get("quantization_config")
    if inline:
        return inline
    path = _root_file(input_dir, "quantize_config.json")
    if path.exists():
        return _read_json(path)
    raise ValueError("GPTQ quantization_config is missing")


def _validate_gptq_v1(config):
    supported = (
        config.get("quant_method", "gptq").lower() == "gptq"
        and config.get("bits") == 4
        and isinstance(config.get("group_size"), int)
        and config["group_size"] > 0
        and config.get("sym") is True
    )
    version = config.get("gptq_version", config.get("version"))
    checkpoint_format = str(
        config.get("checkpoint_format", config.get("format", ""))
    ).lower()
    explicit_v2 = version in (2, "2", "v2") or checkpoint_format in ("gptq_v2", "v2")
    if not supported or explicit_v2:
        raise ValueError(
            "Only GPTQ v1 4-bit symmetric quantization is supported; "
            f"got {config}"
        )


def _validate_shard_name(name):
    if (
        not isinstance(name, str)
        or not name
        or "/" in name
        or "\\" in name
        or Path(name).name != name
        or Path(name).suffix != ".safetensors"
    ):
        raise ValueError(f"invalid safetensors shard name: {name!r}")


def _inspect_index(input_dir, index):
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("index weight_map must be a non-empty object")
    for key, shard in weight_map.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"invalid tensor name in index: {key!r}")
        _validate_shard_name(shard)

    source_shards = sorted(
        source
        for source in input_dir.iterdir()
        if source.is_file() and source.suffix == ".safetensors"
    )
    for source in source_shards:
        _validate_shard_name(source.name)
        if source.resolve() != _root_file(input_dir, source.name):
            raise ValueError(f"shard escapes model directory: {source.name!r}")

    declared_shards = set(weight_map.values())
    actual_shards = {source.name for source in source_shards}
    if declared_shards != actual_shards:
        raise ValueError(
            "index shard set mismatch: "
            f"missing={sorted(declared_shards - actual_shards)}, "
            f"unindexed={sorted(actual_shards - declared_shards)}"
        )

    actual_map = {}
    for source in source_shards:
        with safe_open(source, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in actual_map:
                    raise ValueError(
                        f"tensor {key} appears in both {actual_map[key]} and {source.name}"
                    )
                actual_map[key] = source.name

    declared_keys = set(weight_map)
    actual_keys = set(actual_map)
    wrong_shards = sorted(
        key for key in declared_keys & actual_keys if weight_map[key] != actual_map[key]
    )
    if declared_keys != actual_keys or wrong_shards:
        raise ValueError(
            "index tensor map mismatch: "
            f"missing={sorted(declared_keys - actual_keys)}, "
            f"unindexed={sorted(actual_keys - declared_keys)}, "
            f"wrong_shard={wrong_shards}"
        )
    return weight_map


def _gptq_groups(weight_map):
    prefixes = sorted(
        key[: -len(".qweight")] for key in weight_map if key.endswith(".qweight")
    )
    if not prefixes:
        raise ValueError("no GPTQ .qweight tensors found")

    groups = {}
    consumed = set()
    for prefix in prefixes:
        keys = {suffix: prefix + suffix for suffix in SIDECAR_SUFFIXES}
        for suffix in (".qweight", ".qzeros", ".scales"):
            if keys[suffix] not in weight_map:
                raise ValueError(f"{prefix}.qweight: missing {keys[suffix]}")
        groups[prefix] = keys
        consumed.update(
            keys[suffix] for suffix in SIDECAR_SUFFIXES if keys[suffix] in weight_map
        )

    generated = [
        name
        for prefix in prefixes
        for name in (prefix + ".weight", prefix + ".weight_scale")
    ]
    duplicates = sorted(name for name, count in Counter(generated).items() if count > 1)
    conflicts = sorted(set(generated) & (set(weight_map) - consumed))
    if duplicates or conflicts:
        raise ValueError(
            f"generated tensor name collision: duplicates={duplicates}, existing={conflicts}"
        )
    return groups, consumed


def _load_tensor(input_dir, weight_map, key):
    shard = weight_map[key]
    with safe_open(
        _root_file(input_dir, shard), framework="pt", device="cpu"
    ) as handle:
        return handle.get_tensor(key)


def _copy_sources(input_dir):
    excluded = {INDEX_NAME, "config.json", "quantize_config.json"}
    sources = []
    for source in input_dir.iterdir():
        if (
            not source.is_file()
            or source.suffix == ".safetensors"
            or source.name in excluded
        ):
            continue
        if source.resolve().parent != input_dir:
            raise ValueError(f"model file escapes input directory: {source.name!r}")
        sources.append(source)
    return sources


def convert_model_dir(input_dir, output_dir):
    """Convert one indexed local model directory without modifying its source."""
    input_dir = Path(input_dir).resolve()
    output_path = Path(output_dir)
    output_dir = output_path.parent.resolve() / output_path.name
    if not input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {input_dir}")
    if os.path.lexists(output_dir):
        raise ValueError(f"output directory already exists: {output_dir}")
    if not output_dir.parent.is_dir():
        raise ValueError(f"output parent directory does not exist: {output_dir.parent}")
    if output_dir == input_dir or output_dir.is_relative_to(input_dir):
        raise ValueError("output directory must not equal or be inside input directory")
    if input_dir.is_relative_to(output_dir):
        raise ValueError("input directory must not be inside output directory")

    config_path = _root_file(input_dir, "config.json")
    index_path = _root_file(input_dir, INDEX_NAME)
    if not config_path.is_file() or not index_path.is_file():
        raise ValueError(f"input must contain config.json and {INDEX_NAME}")

    config = _read_json(config_path)
    quantization_config = _source_quantization_config(input_dir, config)
    _validate_gptq_v1(quantization_config)
    group_size = quantization_config["group_size"]
    index = _read_json(index_path)
    weight_map = _inspect_index(input_dir, index)
    groups, consumed = _gptq_groups(weight_map)
    copy_sources = _copy_sources(input_dir)

    output_weight_map = {}
    total_size = 0
    shards = sorted(set(weight_map.values()))
    by_target_shard = {}
    for prefix, keys in groups.items():
        by_target_shard.setdefault(weight_map[keys[".qweight"]], []).append(
            (prefix, keys)
        )

    prefix = f".{output_dir.name}.staging-"
    with tempfile.TemporaryDirectory(prefix=prefix, dir=output_dir.parent) as temporary:
        staging_dir = Path(temporary)
        for source in copy_sources:
            shutil.copy2(source, staging_dir / source.name)

        for shard in shards:
            output_tensors = {}
            with safe_open(
                _root_file(input_dir, shard), framework="pt", device="cpu"
            ) as handle:
                for key in handle.keys():
                    if key not in consumed:
                        output_tensors[key] = handle.get_tensor(key)

            for tensor_prefix, keys in by_target_shard.get(shard, []):
                packed, scales = convert_gptq_tensors(
                    _load_tensor(input_dir, weight_map, keys[".qweight"]),
                    _load_tensor(input_dir, weight_map, keys[".qzeros"]),
                    _load_tensor(input_dir, weight_map, keys[".scales"]),
                    _load_tensor(input_dir, weight_map, keys[".g_idx"])
                    if keys[".g_idx"] in weight_map
                    else None,
                    group_size=group_size,
                    tensor_name=keys[".qweight"],
                )
                output_tensors[tensor_prefix + ".weight"] = packed
                output_tensors[tensor_prefix + ".weight_scale"] = scales

            if not output_tensors:
                continue
            target = _root_file(staging_dir.resolve(), shard)
            save_file(output_tensors, target)
            for key, tensor in output_tensors.items():
                output_weight_map[key] = shard
                total_size += tensor.numel() * tensor.element_size()

        output_config = dict(config)
        output_config["quantization_config"] = {
            "quant_method": "w4a16",
            "bits": 4,
            "group_size": group_size,
            "sym": True,
            "desc_act": False,
        }
        (staging_dir / "config.json").write_text(
            json.dumps(output_config, indent=2) + "\n"
        )
        metadata = dict(index.get("metadata", {}))
        metadata["total_size"] = total_size
        output_index = {"metadata": metadata, "weight_map": output_weight_map}
        (staging_dir / INDEX_NAME).write_text(json.dumps(output_index, indent=2) + "\n")

        actual_map = {}
        for shard_path in sorted(staging_dir.glob("*.safetensors")):
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in actual_map:
                        raise ValueError(
                            f"output tensor {key} appears in both "
                            f"{actual_map[key]} and {shard_path.name}"
                        )
                    actual_map[key] = shard_path.name
        if actual_map != output_weight_map:
            raise ValueError("staging tensor map does not match output index")

        try:
            _rename_noreplace(staging_dir, output_dir)
        except FileExistsError as error:
            raise ValueError(
                f"output directory already exists: {output_dir}"
            ) from error

    return output_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    convert_model_dir(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
