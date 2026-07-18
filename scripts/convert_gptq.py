#!/usr/bin/env python3
"""Compatibility wrapper for convert_gptq_to_w4a16.py."""

import argparse
from pathlib import Path

from convert_gptq_to_w4a16 import convert_model_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    convert_model_dir(
        args.input_dir, args.output or Path(f"{args.input_dir}-converted")
    )


if __name__ == "__main__":
    main()
