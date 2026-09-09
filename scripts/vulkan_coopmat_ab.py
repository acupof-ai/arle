#!/usr/bin/env python3
"""A/B the cooperative-matrix prefill GEMM against the `mul_mmq` fallback.

Boots `arle serve --backend vulkan` twice against the same GGUF at the same
chunk width, once per GEMM route, and reports prefill tok/s plus the generated
text for each:

    coopmat : `mul_mm` COOPMAT  (`B` staged as f16, `A` dequantized to f16
              shared memory, multiplied on the device's matrix cores)
    mmq     : `mul_mmq`         (`B` staged as q8_1_x4, integer dot product)

The route is a whole-model decision in `prefill.rs::coopmat_route`, flipped
here with `ARLE_VK_DISABLE_COOPMAT` — the same knob shape as llama.cpp's
`GGML_VK_DISABLE_COOPMAT`, so the two runtimes can be compared lane-for-lane.

The text comparison is a semantic gate, not a checksum: the two kernels
accumulate in different orders and at different precisions (f16 multiply vs
exact 8-bit integer dot), so late divergence in a long generation is expected
drift. Divergence in the first few tokens is a bug.

Usage:
    python scripts/vulkan_coopmat_ab.py --model-path <gguf> [--width 256]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vulkan_prefill_sweep import run_width  # noqa: E402

ROUTES = [
    ("coopmat", {}, "_cm"),
    ("mmq", {"ARLE_VK_DISABLE_COOPMAT": "1"}, "_mmq"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--binary", default="target/release/arle.exe")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--prompt-reps", type=int, default=16)
    ap.add_argument("--port", type=int, default=8043)
    ap.add_argument("--load-timeout", type=float, default=900.0)
    args = ap.parse_args()

    rows = []
    for name, env, tag in ROUTES:
        row = run_width(args, args.width, extra_env=env, tag=tag)
        row["route"] = name
        rows.append(row)
        print(
            f"  route={name:<8} chunks={row['chunks']:<3} tokens={row['tokens']:<5} "
            f"prefill={row['prefill_s']:6.2f}s ({row['prefill_tok_s']:7.1f} tok/s)  "
            f"ttft={row['ttft_s']:6.2f}s",
            flush=True,
        )

    print(f"\n== prefill tok/s by GEMM route (chunk width {args.width}) ==")
    base = next(r["prefill_tok_s"] for r in rows if r["route"] == "mmq")
    for row in rows:
        print(
            f"{row['route']:>8} : {row['prefill_tok_s']:8.1f} tok/s "
            f"({row['prefill_tok_s'] / base:5.2f}x vs mmq)"
        )

    print("\n== generated text (semantic gate) ==")
    for row in rows:
        print(f"{row['route']:>8} : {row['text']!r}")
    if rows[0]["text"] == rows[1]["text"]:
        print("routes agree token-for-token")
    else:
        print("routes DIVERGE — check whether it is late drift or a first-token bug")
    return 0


if __name__ == "__main__":
    sys.exit(main())
