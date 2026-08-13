"""TurboQuant weight quantization: Hadamard rotation + Lloyd-Max 2-4 bit.

Thin wrapper around scripts/quantize.py TurboQuantQuantizer.

Output: {base}.tq_packed (uint8) + .tq_scales (float16) + .tq_signs (int8).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quantize import TurboQuantQuantizer, CheckpointIO


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboQuant weight quantization")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()
    CheckpointIO(
        TurboQuantQuantizer(bits=args.bits, group_size=args.group_size)
    ).run(Path(args.model_path), Path(args.output_path))


if __name__ == "__main__":
    main()
