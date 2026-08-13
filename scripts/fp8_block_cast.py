"""BF16 HF checkpoint -> DeepSeek-style FP8 block-scaled checkpoint.

Thin wrapper around scripts/quantize.py FP8BlockCastQuantizer. Keeps
`per_block_cast_to_fp8` and `dequant` importable (w8a16_quant.py selfcheck
uses them).

Algorithm: 128x128 block, sf = blockwise_amax / 448.0.
Stored: {base}.weight (E4M3) + {base}.weight_scale_inv = sf (BF16).
Dequant: w ~= fp8 * sf.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quantize import (
    FP8_BLOCK,
    FP8BlockCastQuantizer,
    CheckpointIO,
    fp8_dequant as dequant,
    per_block_cast_to_fp8,
)

__all__ = ["per_block_cast_to_fp8", "dequant", "FP8_BLOCK"]


def run(bf16_dir: Path, ref_dir: Path, out_dir: Path) -> None:
    CheckpointIO(FP8BlockCastQuantizer()).run(bf16_dir, out_dir, ref_dir=ref_dir, config_src=ref_dir)


def _selfcheck() -> None:
    FP8BlockCastQuantizer().selfcheck()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bf16", help="merged BF16 HF checkpoint dir")
    ap.add_argument("--ref-fp8", help="official FP8 checkpoint (defines quant scope + config)")
    ap.add_argument("--out", help="output FP8 checkpoint dir")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck()
        return
    if not (args.bf16 and args.ref_fp8 and args.out):
        ap.error("--bf16, --ref-fp8, --out required (or --selfcheck)")
    run(Path(args.bf16), Path(args.ref_fp8), Path(args.out))


if __name__ == "__main__":
    main()
