#!/usr/bin/env python3
"""Convert a dense Qwen3.6 checkpoint to ARLE-readable NVFP4 side tensors.

This is an offline checkpoint-prep tool for Colab/remote validation. It is not
used by the runtime hot path.

Output ABI:
  <base>.weight_packed        uint8, two FP4 E2M1 values per byte, low nibble first
  <base>.weight_scale         float8_e4m3fn, [rows, cols / 16]
  <base>.weight_global_scale  float32, [1], inverse-global RedHat/unsloth style
  <base>.input_global_scale   float32, [1], metadata only in ARLE v1

Dense stacked MoE expert tensors are split to per-expert packed tensors:
  <mlp>.experts.gate_up_proj -> <mlp>.experts.<i>.{gate,up}_proj.*
  <mlp>.experts.down_proj    -> <mlp>.experts.<i>.down_proj.*
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


GROUP_SIZE = 16
FP4_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
F8_E4M3_MAX = 448.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, help="dense Qwen3.6 model directory")
    parser.add_argument("--dst", type=Path, help="output NVFP4 model directory")
    parser.add_argument(
        "--max-global-scale",
        type=float,
        default=65536.0,
        help="cap for per-tensor FP32 global scale; larger improves small-scale precision",
    )
    parser.add_argument(
        "--keep-stacked-experts",
        action="store_true",
        help="also copy original dense stacked expert tensors; off by default to save space",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print conversion decisions without writing safetensors",
    )
    parser.add_argument(
        "--limit-shards",
        type=int,
        default=0,
        help="debug: convert at most N source shards (0 = all)",
    )
    parser.add_argument("--self-test", action="store_true", help="run a small round-trip test and exit")
    return parser.parse_args()


def require_float8() -> None:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("PyTorch with torch.float8_e4m3fn is required")


def copy_metadata(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if not path.is_file() or path.suffix == ".safetensors":
            continue
        if path.name == "model.safetensors.index.json":
            continue
        shutil.copy2(path, dst / path.name)


def source_shards(src: Path) -> list[str]:
    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            index = json.load(f)
        return sorted(set(index["weight_map"].values()))
    shards = sorted(path.name for path in src.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards found under {src}")
    return shards


def tensor_base(weight_name: str) -> str:
    if weight_name.endswith(".weight"):
        return weight_name[: -len(".weight")]
    raise ValueError(f"expected .weight tensor name, got {weight_name}")


def is_dense_2d(tensor: torch.Tensor) -> bool:
    return tensor.ndim == 2 and tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)


def dequant_fp8_block_scaled(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """FP8 E4M3 weight + per-block inverse scales -> float32 dense."""
    rows, cols = weight.shape
    srows, scols = scale_inv.shape
    block_r = -(-rows // srows)
    block_c = -(-cols // scols)
    scales = scale_inv.to(torch.float32)
    scales = scales.repeat_interleave(block_r, dim=0)[:rows]
    scales = scales.repeat_interleave(block_c, dim=1)[:, :cols]
    return weight.to(torch.float32) * scales


def dequant_shard_fp8(tensors: dict[str, torch.Tensor]) -> None:
    """Dequantize FP8 block-scaled weights that will be converted to NVFP4.
    Tensors outside the conversion policy keep their FP8 weight + scale_inv
    (the runtime loads those through its FP8 path)."""
    for name in list(tensors):
        if not name.endswith(".weight") or tensors[name].dtype != torch.float8_e4m3fn:
            continue
        scale_name = name + "_scale_inv"
        if scale_name not in tensors:
            continue
        dense = dequant_fp8_block_scaled(tensors[name], tensors[scale_name])
        if should_quantize_2d_weight(name, dense):
            tensors[name] = dense
            del tensors[scale_name]


def should_quantize_2d_weight(name: str, tensor: torch.Tensor) -> bool:
    if not name.endswith(".weight") or not is_dense_2d(tensor):
        return False
    if "embed_tokens.weight" in name or name == "lm_head.weight":
        return False
    if name.endswith(".norm.weight") or "layernorm.weight" in name:
        return False
    if ".linear_attn." in name:
        return False
    if name.endswith(".mlp.gate.weight") or name.endswith(".shared_expert_gate.weight"):
        return False
    if ".self_attn." in name and any(
        name.endswith(f".{proj}_proj.weight") for proj in ("q", "k", "v", "o")
    ):
        return True
    if any(name.endswith(f".mlp.{proj}_proj.weight") for proj in ("gate", "up", "down")):
        return True
    if any(
        name.endswith(f".mlp.shared_expert.{proj}_proj.weight")
        for proj in ("gate", "up", "down")
    ):
        return True
    if ".mlp.experts." in name and any(
        name.endswith(f".{proj}_proj.weight") for proj in ("gate", "up", "down")
    ):
        return True
    return False


def stacked_gate_up_prefix(name: str) -> str | None:
    for suffix in (".experts.gate_up_proj", ".experts.gate_up_proj.weight"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def stacked_down_prefix(name: str) -> str | None:
    for suffix in (".experts.down_proj", ".experts.down_proj.weight"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def fp4_pack(weight: torch.Tensor, *, max_global_scale: float) -> dict[str, torch.Tensor]:
    weight = weight.detach().to(torch.float32).cpu().contiguous()
    if weight.ndim != 2:
        raise ValueError(f"NVFP4 expects a rank-2 matrix, got {tuple(weight.shape)}")
    rows, cols = weight.shape
    if cols % GROUP_SIZE != 0:
        raise ValueError(f"NVFP4 K dimension must be divisible by {GROUP_SIZE}, got {cols}")
    if cols % 2 != 0:
        raise ValueError(f"NVFP4 packed pairs require an even K dimension, got {cols}")

    groups = weight.reshape(rows, cols // GROUP_SIZE, GROUP_SIZE)
    desired_scale = groups.abs().amax(dim=2) / 6.0
    nonzero = desired_scale[desired_scale > 0]
    max_desired = float(nonzero.max().item()) if nonzero.numel() else 1.0
    global_scale = min(max_global_scale, max(1.0, 0.95 * F8_E4M3_MAX / max_desired))

    raw_group_scale = desired_scale * global_scale
    raw_group_scale = torch.where(
        desired_scale > 0,
        raw_group_scale,
        torch.ones_like(raw_group_scale),
    )
    weight_scale = raw_group_scale.to(torch.float8_e4m3fn).contiguous()
    effective_scale = (weight_scale.to(torch.float32) / global_scale).clamp_min(1.0e-12)

    ratio = groups / effective_scale.unsqueeze(-1)
    values = FP4_E2M1_VALUES.to(ratio.device)
    nearest = (ratio.abs().unsqueeze(-1) - values).abs().argmin(dim=-1).to(torch.uint8)
    sign = (ratio < 0).to(torch.uint8) << 3
    nibbles = (nearest | sign).reshape(rows, cols)
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).contiguous()

    return {
        "weight_packed": packed,
        "weight_scale": weight_scale,
        "weight_global_scale": torch.tensor([global_scale], dtype=torch.float32),
        "input_global_scale": torch.tensor([1.0], dtype=torch.float32),
    }


def add_packed(out: dict[str, torch.Tensor], base: str, weight: torch.Tensor, max_global_scale: float) -> None:
    packed = fp4_pack(weight, max_global_scale=max_global_scale)
    for suffix, tensor in packed.items():
        out[f"{base}.{suffix}"] = tensor


def convert_stacked_gate_up(
    out: dict[str, torch.Tensor],
    name: str,
    tensor: torch.Tensor,
    max_global_scale: float,
) -> int:
    mlp_prefix = stacked_gate_up_prefix(name)
    if mlp_prefix is None:
        return 0
    if tensor.ndim != 3:
        raise ValueError(f"{name}: expected [experts, 2*intermediate, hidden], got {tuple(tensor.shape)}")
    experts, fused_rows, _hidden = tensor.shape
    if fused_rows % 2 != 0:
        raise ValueError(f"{name}: fused gate/up rows must be even, got {fused_rows}")
    intermediate = fused_rows // 2
    for expert_idx in range(experts):
        expert = tensor[expert_idx]
        add_packed(
            out,
            f"{mlp_prefix}.experts.{expert_idx}.gate_proj",
            expert[:intermediate, :],
            max_global_scale,
        )
        add_packed(
            out,
            f"{mlp_prefix}.experts.{expert_idx}.up_proj",
            expert[intermediate:, :],
            max_global_scale,
        )
    return experts * 2


def convert_stacked_down(
    out: dict[str, torch.Tensor],
    name: str,
    tensor: torch.Tensor,
    max_global_scale: float,
) -> int:
    mlp_prefix = stacked_down_prefix(name)
    if mlp_prefix is None:
        return 0
    if tensor.ndim != 3:
        raise ValueError(f"{name}: expected [experts, hidden, intermediate], got {tuple(tensor.shape)}")
    experts = tensor.shape[0]
    for expert_idx in range(experts):
        add_packed(
            out,
            f"{mlp_prefix}.experts.{expert_idx}.down_proj",
            tensor[expert_idx],
            max_global_scale,
        )
    return experts


def write_config(src: Path, dst: Path) -> None:
    config_path = dst / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    config["quantization_config"] = {
        "quant_method": "nvfp4",
        "fmt": "nvfp4",
        "format": "nvfp4",
        "activation_scheme": "static",
        "group_size": GROUP_SIZE,
        "modules_to_not_convert": [],
        "arle_notes": "Generated by scripts/qwen36_dense_to_nvfp4.py; linear_attn/router/norm tensors stay dense.",
    }
    with config_path.open("w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def convert(args: argparse.Namespace) -> None:
    require_float8()
    if args.src is None or args.dst is None:
        raise SystemExit("--src and --dst are required unless --self-test is used")
    src = args.src
    dst = args.dst
    if not src.is_dir():
        raise FileNotFoundError(src)
    if dst.exists() and any(dst.iterdir()) and not args.dry_run:
        raise FileExistsError(f"{dst} exists and is not empty")
    if not args.dry_run:
        copy_metadata(src, dst)

    shards = source_shards(src)
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    width = max(5, len(str(len(shards))))
    weight_map: dict[str, str] = {}
    converted = 0
    copied = 0

    for shard_idx, shard_name in enumerate(shards, start=1):
        tensors = load_file(src / shard_name, device="cpu")
        dequant_shard_fp8(tensors)
        out: dict[str, torch.Tensor] = {}
        out_name = f"model-{shard_idx:0{width}d}-of-{len(shards):0{width}d}.safetensors"
        for name, tensor in tensors.items():
            gate_up = stacked_gate_up_prefix(name)
            down = stacked_down_prefix(name)
            if gate_up is not None:
                converted += convert_stacked_gate_up(out, name, tensor, args.max_global_scale)
                if args.keep_stacked_experts:
                    out[name] = tensor
                continue
            if down is not None:
                converted += convert_stacked_down(out, name, tensor, args.max_global_scale)
                if args.keep_stacked_experts:
                    out[name] = tensor
                continue
            if should_quantize_2d_weight(name, tensor):
                add_packed(out, tensor_base(name), tensor, args.max_global_scale)
                converted += 1
                continue
            out[name] = tensor
            copied += 1

        for out_tensor_name in out:
            weight_map[out_tensor_name] = out_name
        print(
            f"{shard_name}: wrote {len(out)} tensors to {out_name} "
            f"(converted so far={converted}, copied so far={copied})"
        )
        if not args.dry_run and out:
            save_file(out, dst / out_name)

    if not args.dry_run:
        total_size = sum(path.stat().st_size for path in dst.glob("*.safetensors"))
        with (dst / "model.safetensors.index.json").open("w") as f:
            json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2, sort_keys=True)
            f.write("\n")
        write_config(src, dst)
    print(f"converted {converted} matrices; copied {copied} tensors")


def self_test() -> None:
    require_float8()
    weight = torch.randn(8, 32, dtype=torch.float32) * 0.03
    packed = fp4_pack(weight, max_global_scale=65536.0)
    assert packed["weight_packed"].shape == (8, 16)
    assert packed["weight_scale"].shape == (8, 2)
    assert packed["weight_scale"].dtype == torch.float8_e4m3fn
    assert packed["weight_global_scale"].shape == (1,)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "toy.safetensors"
        save_file({f"w.{name}": tensor for name, tensor in packed.items()}, path)
        loaded = load_file(path, device="cpu")
        assert loaded["w.weight_scale"].dtype == torch.float8_e4m3fn
        assert loaded["w.weight_packed"].dtype == torch.uint8
    print("self-test ok")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    convert(args)


if __name__ == "__main__":
    main()
