"""Qwen3 W4A8 Marlin packer for ARLE.

Thin wrapper around scripts/quantize.py W4A8MarlinQuantizer. Keeps `pack_w4a8`
and `get_perms` importable (used by convert_gptq_w4a16_to_w4a8_marlin.py,
verify_gptq_w4a8_repack_quality.py, diag_w4a8_pack_roundtrip.py).

Output: {base}.marlin_w4a8_qweight (int32) + .marlin_w4a8_s_channel (float32)
+ .marlin_w4a8_s_group (float16).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quantize import (
    W4A8_GROUP,
    W4A8MarlinQuantizer,
    CheckpointIO,
    _get_perms as get_perms,
    is_w4a8_quantizable as is_quantized_linear,
    pack_w4a8,
)

__all__ = ["pack_w4a8", "get_perms", "is_quantized_linear", "W4A8_GROUP"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    args = parser.parse_args()
    CheckpointIO(W4A8MarlinQuantizer()).run(args.src, args.dst)


if __name__ == "__main__":
    main()
