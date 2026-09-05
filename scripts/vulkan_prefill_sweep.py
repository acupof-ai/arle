#!/usr/bin/env python3
"""Sweep the batched-prefill chunk width and report prefill tok/s per width.

Two knobs have to move together: the planner's `--chunked-prefill-size` decides
how many tokens reach `BackendExecutor::forward_tokens` in one row, and
`ARLE_VULKAN_PREFILL_CHUNK` sizes the device arena that row is processed in.
Either one left at 64 pins the effective width to 64 — and 64 is exactly the
boundary where `MmqSpec::choose` stops picking its LARGE tile, so the width is
worth measuring rather than assuming.

Reads the `vulkan batched prefill: N tok @ P in T s (R tok/s)` lines the runtime
logs, so the number reported is prefill proper, not TTFT (which also carries
tokenize + sampling + the first decode step).

Usage:
    python scripts/vulkan_prefill_sweep.py --model-path <gguf> --widths 64 128 256
"""

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vulkan_prefill_parity import PROMPT_UNIT, complete, wait_ready  # noqa: E402

CHUNK_RE = re.compile(
    r"vulkan batched prefill: (\d+) tok @ (\d+) in ([\d.]+)s \(([\d.]+) tok/s"
)


def run_width(args, width: int, extra_env: dict | None = None, tag: str = "") -> dict:
    env = dict(os.environ)
    env["ARLE_VULKAN_BATCHED_PREFILL"] = "1"
    env["ARLE_VULKAN_PREFILL_CHUNK"] = str(width)
    env.update(extra_env or {})
    log_path = f"prefill_sweep_{width}{tag}.log"
    cmd = [
        os.path.abspath(args.binary),
        "serve",
        "--backend",
        "vulkan",
        "--model-path",
        args.model_path,
        "--port",
        str(args.port),
        "--chunked-prefill-size",
        str(width),
    ]
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_ready(args.port, args.load_timeout, proc)
            complete(args.port, "hi", 1)  # warm the page cache, untimed
            prompt = PROMPT_UNIT * args.prompt_reps + "Summarize the note."
            text, ttft, _ = complete(args.port, prompt, args.tokens)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()

    # Last request's chunks only: skip the warmup's.
    chunks = [
        (int(n), int(p), float(s), float(r))
        for n, p, s, r in CHUNK_RE.findall(open(log_path, encoding="utf-8").read())
    ]
    measured = []
    for n, p, s, r in chunks:
        if p == 0:
            measured = []  # a fresh sequence started; drop the warmup's chunks
        measured.append((n, p, s, r))
    tokens = sum(n for n, _, _, _ in measured)
    secs = sum(s for _, _, s, _ in measured)
    return {
        "width": width,
        "chunks": len(measured),
        "tokens": tokens,
        "prefill_s": secs,
        "prefill_tok_s": tokens / secs if secs else float("nan"),
        "ttft_s": ttft,
        "text": text,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--binary", default="target/release/arle.exe")
    ap.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--prompt-reps", type=int, default=16)
    ap.add_argument("--port", type=int, default=8041)
    ap.add_argument("--load-timeout", type=float, default=900.0)
    args = ap.parse_args()

    rows = []
    for width in args.widths:
        row = run_width(args, width)
        rows.append(row)
        print(
            f"  width={row['width']:<4} chunks={row['chunks']:<3} "
            f"tokens={row['tokens']:<5} prefill={row['prefill_s']:6.2f}s "
            f"({row['prefill_tok_s']:6.1f} tok/s)  ttft={row['ttft_s']:6.2f}s",
            flush=True,
        )

    print("\n== prefill tok/s by chunk width ==")
    base = rows[0]["prefill_tok_s"]
    for row in rows:
        print(
            f"{row['width']:>4} tok/chunk : {row['prefill_tok_s']:7.1f} tok/s "
            f"({row['prefill_tok_s'] / base:5.2f}x vs width {rows[0]['width']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
