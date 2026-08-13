"""BF16 HF checkpoint -> W8A16 (per-group signed INT8 weights, BF16 scales).

Thin wrapper around scripts/quantize.py W8A16Quantizer. Keeps `per_group_int8`
importable.

Format: {base}.weight (INT8 [rows,cols]) + {base}.weight_scale (BF16 [rows, cols/gs]).
Dequant: w ~= int8 * scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quantize import (
    INT8_MAX,
    W8A16_GROUP,
    W8A16Quantizer,
    CheckpointIO,
    per_group_int8,
)

__all__ = ["per_group_int8", "W8A16_GROUP", "INT8_MAX"]


def run(bf16_dir: Path, ref_dir: Path | None, out_dir: Path, group_size: int) -> None:
    CheckpointIO(W8A16Quantizer(group_size=group_size)).run(bf16_dir, out_dir, ref_dir=ref_dir)


def _selfcheck() -> None:
    W8A16Quantizer().selfcheck()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bf16", help="source BF16 HF checkpoint dir")
    ap.add_argument("--ref", help="reference quantized checkpoint defining quant scope; "
                                  "omit to quantize all linear weights")
    ap.add_argument("--out", help="output W8A16 checkpoint dir")
    ap.add_argument("--group-size", type=int, default=W8A16_GROUP)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck()
        return
    if not (args.bf16 and args.out):
        ap.error("--bf16 and --out required (or --selfcheck)")
    run(Path(args.bf16), Path(args.ref) if args.ref else None, Path(args.out), args.group_size)


if __name__ == "__main__":
    main()
