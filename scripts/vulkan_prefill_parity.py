#!/usr/bin/env python3
"""A/B the Vulkan batched prefill against the per-token prefill loop.

Boots `arle serve --backend vulkan` twice against the same GGUF — once with
`ARLE_VULKAN_BATCHED_PREFILL=1` (the `mul_mmq` chunk path) and once with `=0`
(the GEMV-per-token loop) — sends the same greedy prompts to both, and reports
the generated text plus TTFT side by side.

TTFT is the metric that matters here: it IS prefill latency. The two paths are
not bit-identical (a chunked GEMM accumulates in a different order than a chain
of GEMVs), so the text comparison is a semantic gate, not a checksum — a
divergence deep into a long generation is expected drift, a divergence in the
first few tokens is a bug.

Usage:
    python scripts/vulkan_prefill_parity.py --model-path <gguf> [--tokens 256]
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

PROMPT_UNIT = (
    "The following is a technical note about memory bandwidth on unified-memory "
    "accelerators, written for an audience of systems engineers. "
)


def wait_ready(port: int, timeout_s: float, proc: subprocess.Popen) -> None:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(1.0)
    raise RuntimeError(f"server not ready on port {port} after {timeout_s}s")


def complete(port: int, prompt: str, max_tokens: int) -> tuple[str, float, float]:
    """Return (text, ttft_s, total_s) for a streamed greedy completion."""
    body = json.dumps(
        {
            "model": "local",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    ttft = None
    out = []
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            # Qwen3.8 is a thinking model: with a small `max_tokens` every token
            # lands in `reasoning_content` and `content` stays empty, which would
            # read as "generated nothing". Count both.
            piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if piece:
                if ttft is None:
                    ttft = time.time() - start
                out.append(piece)
    return "".join(out), (ttft if ttft is not None else float("nan")), time.time() - start


def run_side(args, batched: bool, port: int) -> list[dict]:
    env = dict(os.environ)
    env["ARLE_VULKAN_BATCHED_PREFILL"] = "1" if batched else "0"
    cmd = [
        os.path.abspath(args.binary),
        "serve",
        "--backend",
        "vulkan",
        "--model-path",
        args.model_path,
        "--port",
        str(port),
    ]
    cmd += args.extra
    log = open(f"prefill_ab_{'batched' if batched else 'serial'}.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    results = []
    try:
        wait_ready(port, args.load_timeout, proc)
        # Warmup. `/v1/models` answers as soon as the server binds, but the
        # first real request still faults ~16 GiB of weights in from disk, which
        # buries the prefill signal under page-in time on whichever side runs
        # first. Pay it once, untimed.
        complete(port, "hi", 1)
        for reps in args.prompt_reps:
            prompt = PROMPT_UNIT * reps + "Summarize the note in one sentence."
            text, ttft, total = complete(port, prompt, args.tokens)
            results.append(
                {"reps": reps, "text": text, "ttft_s": ttft, "total_s": total}
            )
            print(
                f"  [{'batched' if batched else 'serial'}] reps={reps:<3} "
                f"ttft={ttft:7.2f}s total={total:7.2f}s",
                flush=True,
            )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument(
        "--extra", nargs="*", default=[], help="extra flags forwarded to `arle serve`"
    )
    ap.add_argument("--binary", default="target/release/arle.exe")
    ap.add_argument("--tokens", type=int, default=32, help="max completion tokens")
    ap.add_argument(
        "--prompt-reps",
        type=int,
        nargs="+",
        default=[1, 8, 24],
        help="repetitions of the ~30-token prompt unit (prompt length knob)",
    )
    ap.add_argument("--port", type=int, default=8031)
    ap.add_argument("--load-timeout", type=float, default=900.0)
    ap.add_argument("--serial", action="store_true", help="also run the serial side")
    args = ap.parse_args()

    print("== batched (ARLE_VULKAN_BATCHED_PREFILL=1) ==", flush=True)
    batched = run_side(args, True, args.port)
    if not args.serial:
        print(json.dumps({"batched": batched}, indent=2, ensure_ascii=False))
        return 0

    print("== serial (ARLE_VULKAN_BATCHED_PREFILL=0) ==", flush=True)
    serial = run_side(args, False, args.port + 1)

    print("\n== comparison ==")
    ok = True
    for b, s in zip(batched, serial):
        same = b["text"] == s["text"]
        speedup = s["ttft_s"] / b["ttft_s"] if b["ttft_s"] else float("nan")
        print(
            f"reps={b['reps']:<3} ttft {s['ttft_s']:7.2f}s -> {b['ttft_s']:7.2f}s "
            f"({speedup:5.2f}x)  text_identical={same}"
        )
        if not same:
            ok = False
            print(f"    serial : {s['text'][:200]!r}")
            print(f"    batched: {b['text'][:200]!r}")
    print(json.dumps({"batched": batched, "serial": serial}, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
