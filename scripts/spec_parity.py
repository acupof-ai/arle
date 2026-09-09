#!/usr/bin/env python3
"""Speculative-decoding parity gate.

Runs N prompts greedy against two Metal serves of the same model — one
plain, one with --draft-model — and diffs the generated token ids. Spec
decode is correctness-preserving by construction (the verify step owns
every emitted token), so the gate is ZERO token-id mismatches. Prints the
draft arm's accept rate and decode tok/s for both arms. Accept rate comes
from the /v1/stats spec_decode counters when wired (CUDA); on Metal those
counters are not fed, so it is derived as generated_tokens/decode_steps − 1
(mean accepted drafts per step).

    python3 scripts/spec_parity.py
    python3 scripts/spec_parity.py --model models/Qwen3.5-0.8B-MLX-4bit \\
        --draft-model r3lax/Qwen3.5-0.8B-DSpark

Negative control — proves the gate can go red. Runs the baseline arm at
temperature 0.3 so the two arms diverge, then reports the mismatches and
exits 1 (a zero-mismatch negative control means the gate is dead). The
draft arm stays greedy: Metal DFlash refuses non-greedy sampling, so the
stochastic arm has to be the baseline.

    python3 scripts/spec_parity.py --negative-control
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arle_stats import spec_decode  # noqa: E402

PROMPTS = [
    "Explain how a transformer model works in one paragraph.",
    "Write a Python function that computes the Fibonacci sequence.",
    "What is the difference between TCP and UDP? Give examples.",
    "Describe the CAP theorem and its implications for distributed systems.",
    "How does gradient descent work in machine learning?",
    "Write a Rust function that reverses a linked list.",
    "Explain bloom filters and a common use case.",
    "How does consistent hashing work in distributed caches?",
    "What are the differences between SQL and NoSQL databases?",
    "Explain the concept of context window in LLMs.",
    "Describe how speculative decoding works.",
    "What is the role of a tokenizer in NLP?",
]

DEFAULT_MODEL = "models/Qwen3.5-0.8B-MLX-4bit"
DEFAULT_DRAFT = "r3lax/Qwen3.5-0.8B-DSpark"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def spawn_server(binary: str, model: str, port: int, draft: str | None,
                 memory_budget_gb: float | None, log: Path) -> subprocess.Popen:
    cmd = [
        binary, "serve", "--backend", "metal",
        "--model-path", model, "--port", str(port),
    ]
    if draft:
        cmd += ["--draft-model", draft]
    if memory_budget_gb:
        cmd += ["--memory-budget-bytes", str(int(memory_budget_gb * 1024**3))]
    return subprocess.Popen(
        cmd,
        stdout=open(log, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def wait_ready(port: int, timeout_s: float = 300.0) -> None:
    import httpx

    deadline = time.time() + timeout_s
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5.0) as client:
        while time.time() < deadline:
            try:
                r = client.get("/v1/models")
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1.0)
    sys.exit(f"server on port {port} did not become ready in {timeout_s:.0f}s")


def stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    proc.wait(timeout=10)


def parity_ids(port: int, prompt: str, max_tokens: int, temperature: float) -> list[int]:
    """Non-streaming completion with return_token_ids — the parity arm."""
    import httpx

    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=300.0) as client:
        r = client.post(
            "/v1/completions",
            json={
                "model": "default",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "ignore_eos": True,
                "return_token_ids": True,
            },
        )
        r.raise_for_status()
        ids = r.json()["choices"][0].get("token_ids")
        if not ids:
            sys.exit(f"port {port}: completion returned no token_ids")
        return ids


def decode_tok_s(port: int, prompt: str, max_tokens: int) -> float:
    """Streaming completion; decode tok/s = (n-1) / (t_last - t_first).

    Robust to spec decode's bursty deltas (several accepted tokens per
    event), where per-event ITL would understate the rate.
    """
    import httpx

    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=300.0) as client:
        first = last = None
        n_tokens = 0
        with client.stream(
            "POST",
            "/v1/completions",
            json={
                "model": "default",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("usage"):
                    n_tokens = chunk["usage"]["completion_tokens"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                text = choices[0].get("text")
                if text:
                    now = time.time()
                    if first is None:
                        first = now
                    last = now
        if first is None or last is None or last <= first or n_tokens < 2:
            return 0.0
        return (n_tokens - 1) / (last - first)


def stats_counters(port: int) -> dict:
    """Spec counters plus throughput, for accept-rate derivation.

    The /v1/stats spec_decode counters are wired on CUDA only; on Metal the
    DFlash path feeds no counters, so accept rate is derived from
    generated_tokens / decode_forward_steps (tokens per step minus the one
    token every step emits = mean accepted drafts per step).
    """
    import httpx

    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5.0) as client:
        body = client.get("/v1/stats").json()
    thru = (body.get("throughput") or {})
    return {
        "spec": spec_decode(body),
        "generated_tokens": thru.get("generated_tokens", 0),
        "decode_steps": thru.get("decode_forward_steps", 0),
    }


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Speculative-decoding parity gate")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT)
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--binary", default="target/release/arle")
    parser.add_argument("--negative-control", action="store_true",
                        help="run the baseline arm at temperature 0.3; expect mismatches")
    parser.add_argument("--memory-budget-gb", type=float, default=None,
                        help="pass --memory-budget-bytes to both serves (constrained-box "
                             "override of the anti-swap guard; default uses the guard)")
    parser.add_argument("--keep-servers", action="store_true",
                        help="leave servers running (debug)")
    parser.add_argument("--output", default=None, help="JSON output file")
    args = parser.parse_args()

    prompts = PROMPTS[: args.prompts]
    if not Path(args.binary).exists():
        sys.exit(f"binary not found: {args.binary} (build with "
                 "`cargo build --release --no-default-features --features metal,no-cuda`)")

    port_a, port_b = free_port(), free_port()
    log_dir = Path("bench-output")
    log_dir.mkdir(exist_ok=True)
    log_a = log_dir / "spec_parity_baseline.log"
    log_b = log_dir / "spec_parity_draft.log"
    proc_a = spawn_server(args.binary, args.model, port_a, None, args.memory_budget_gb, log_a)
    proc_b = None
    try:
        print(f"waiting for baseline (port {port_a})...")
        wait_ready(port_a)
        # Sequential startup: two concurrent Metal model loads stall the MLX
        # bridge (the draft arm hangs mid-build). The baseline is cheap to wait for.
        proc_b = spawn_server(args.binary, args.model, port_b, args.draft_model, args.memory_budget_gb, log_b)
        print(f"waiting for draft (port {port_b})...")
        wait_ready(port_b)
    except BaseException:
        stop_server(proc_a)
        if proc_b is not None:
            stop_server(proc_b)
        raise
    try:
        stats_before = stats_counters(port_b)
        # The draft arm is always greedy (Metal DFlash refuses anything else);
        # the negative control makes the BASELINE stochastic so the arms diverge.
        baseline_temp = 0.3 if args.negative_control else 0.0

        # Parity pass: non-streaming, token ids.
        mismatches = 0
        for i, prompt in enumerate(prompts):
            ids_a = parity_ids(port_a, prompt, args.max_tokens, baseline_temp)
            ids_b = parity_ids(port_b, prompt, args.max_tokens, 0.0)
            if ids_a != ids_b:
                first_div = next(
                    (j for j, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y),
                    min(len(ids_a), len(ids_b)),
                )
                mismatches += 1
                print(f"  prompt {i}: MISMATCH at token {first_div} "
                      f"(len {len(ids_a)} vs {len(ids_b)})")
            else:
                print(f"  prompt {i}: {len(ids_a)} tokens, identical")

        # Timing pass: streaming, decode tok/s.
        rates_a, rates_b = [], []
        for prompt in prompts:
            rates_a.append(decode_tok_s(port_a, prompt, args.max_tokens))
            rates_b.append(decode_tok_s(port_b, prompt, args.max_tokens))

        stats_after = stats_counters(port_b)
        drafted = stats_after["spec"].get("drafted", 0) - stats_before["spec"].get("drafted", 0)
        accepted = stats_after["spec"].get("accepted", 0) - stats_before["spec"].get("accepted", 0)
        if drafted > 0:
            accept_rate = accepted / drafted
            accept_source = "counters"
        else:
            # Metal: spec_decode counters are not wired; derive from step counts.
            gen = stats_after["generated_tokens"] - stats_before["generated_tokens"]
            steps = stats_after["decode_steps"] - stats_before["decode_steps"]
            accept_rate = max(0.0, gen / steps - 1.0) if steps > 0 else 0.0
            accept_source = "tokens_per_step"
    finally:
        if not args.keep_servers:
            stop_server(proc_a)
            if proc_b is not None:
                stop_server(proc_b)

    result = {
        "model": args.model,
        "draft_model": args.draft_model,
        "prompts": len(prompts),
        "max_tokens": args.max_tokens,
        "negative_control": args.negative_control,
        "mismatches": mismatches,
        "accept_rate": accept_rate,
        "accept_rate_source": accept_source,
        "accepted": accepted,
        "drafted": drafted,
        "decode_tok_s_baseline": median(rates_a),
        "decode_tok_s_draft": median(rates_b),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))

    if args.negative_control:
        if mismatches == 0:
            sys.exit("NEGATIVE CONTROL FAILED: arms did not diverge; the gate is dead")
        print(f"negative control confirmed: gate detects {mismatches} mismatches")
        sys.exit(1)
    if mismatches:
        sys.exit(f"PARITY FAILED: {mismatches} prompt(s) differ")
    print("PARITY PASS: zero token-id mismatches")


if __name__ == "__main__":
    main()
