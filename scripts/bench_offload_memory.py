"""Rigorous KV offload memory experiment.

Experiment design:
- Independent variable: max_gpu_kv (256, 512, 1024, unlimited)
- Dependent variables: RSS (CPU memory), generation time, correctness
- Control: same prompt, same model, same max_tokens, greedy decoding
- Methodology:
  1. Record baseline RSS before inference
  2. Generate N tokens with offload enabled
  3. Sample RSS at fixed intervals during generation
  4. Record peak RSS and final RSS
  5. Compare output across conditions (correctness check)

Expected results:
- Lower max_gpu_kv → more offload → higher RSS → slower generation
- Output should be IDENTICAL across all conditions (greedy decoding)
"""

import subprocess
import time
import os
import sys
import threading
import json

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-8B"
MAX_TOKENS = 4096
PROMPT = "Write a detailed history of computing from Babbage to modern AI."

# Conditions to test
CONDITIONS = [
    {"label": "unlimited (no offload)", "max_gpu_kv": None},
    {"label": "1024 tok GPU", "max_gpu_kv": 1024},
    {"label": "512 tok GPU", "max_gpu_kv": 512},
    {"label": "256 tok GPU", "max_gpu_kv": 256},
]


def monitor_rss(pid: int, interval: float, results: list, stop_event: threading.Event):
    """Sample RSS of a process at fixed intervals."""
    while not stop_event.is_set():
        try:
            rss_kb = int(open(f"/proc/{pid}/statm").read().split()[1]) * 4  # pages→KB
            results.append({"time_s": time.perf_counter(), "rss_mb": rss_kb / 1024})
        except (FileNotFoundError, ProcessLookupError):
            break
        stop_event.wait(interval)


def run_condition(condition: dict) -> dict:
    """Run one experimental condition."""
    label = condition["label"]
    max_gpu_kv = condition["max_gpu_kv"]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib64-nvidia:/usr/local/cuda/lib64"

    cmd = [
        "./target/release/agent-infer",
        "--model-path", MODEL_PATH,
        "--max-tokens", str(MAX_TOKENS),
        "--no-cuda-graph",
    ]
    if max_gpu_kv is not None:
        cmd += ["--max-gpu-kv", str(max_gpu_kv)]

    # Start process
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )

    # Wait for model to load
    time.sleep(8)

    # Start RSS monitor
    rss_samples = []
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=monitor_rss,
        args=(proc.pid, 0.5, rss_samples, stop_event),
        daemon=True,
    )
    monitor.start()

    # Send prompt
    wall_start = time.perf_counter()
    try:
        proc.stdin.write(PROMPT + "\nquit\n")
        proc.stdin.flush()
        proc.stdin.close()
    except BrokenPipeError:
        pass

    # Wait for completion
    stdout = proc.stdout.read()
    proc.wait(timeout=300)
    wall_time = time.perf_counter() - wall_start

    stop_event.set()
    monitor.join(timeout=2)

    # Extract metrics from output
    import re
    offload_count = len(re.findall(r"offload:", stdout))
    prefetch_count = len(re.findall(r"prefetch:", stdout))
    generated_match = re.search(r"Generated (\d+) chars", stdout)
    generated_chars = int(generated_match.group(1)) if generated_match else 0

    # Extract first 200 chars of model output for correctness comparison
    output_lines = []
    for line in stdout.split("\n"):
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
        if clean and not clean.startswith("2026-") and not clean.startswith("===") and clean.strip():
            output_lines.append(clean.strip())
    model_output = " ".join(output_lines[-10:])[:300]

    # RSS analysis
    if rss_samples:
        t0 = rss_samples[0]["time_s"]
        for s in rss_samples:
            s["time_s"] = round(s["time_s"] - t0, 1)

        # Baseline: first sample (before inference starts)
        baseline_rss = rss_samples[0]["rss_mb"]
        peak_rss = max(s["rss_mb"] for s in rss_samples)
        final_rss = rss_samples[-1]["rss_mb"]
        delta_rss = peak_rss - baseline_rss
    else:
        baseline_rss = peak_rss = final_rss = delta_rss = 0

    return {
        "label": label,
        "max_gpu_kv": max_gpu_kv,
        "wall_time_s": wall_time,
        "generated_chars": generated_chars,
        "offloads": offload_count,
        "prefetches": prefetch_count,
        "baseline_rss_mb": round(baseline_rss),
        "peak_rss_mb": round(peak_rss),
        "delta_rss_mb": round(delta_rss),
        "rss_samples": len(rss_samples),
        "output_preview": model_output[:150],
    }


def main():
    print("=" * 70)
    print("KV Offload Memory Experiment")
    print("=" * 70)
    print(f"Model:      {MODEL_PATH}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Prompt:     {PROMPT[:60]}...")
    print(f"Conditions: {len(CONDITIONS)}")
    print(f"Control:    greedy decoding (temperature=0)")
    print()

    results = []
    for i, cond in enumerate(CONDITIONS):
        print(f"[{i+1}/{len(CONDITIONS)}] {cond['label']}...")
        result = run_condition(cond)
        results.append(result)
        print(f"  Time: {result['wall_time_s']:.1f}s | "
              f"RSS: {result['baseline_rss_mb']}→{result['peak_rss_mb']}MB "
              f"(Δ{result['delta_rss_mb']}MB) | "
              f"Offloads: {result['offloads']} | "
              f"Chars: {result['generated_chars']}")
        time.sleep(2)  # Cool down between runs

    # Summary table
    print()
    print("=" * 70)
    print(f"{'Condition':<25} {'Time':>7} {'Base RSS':>9} {'Peak RSS':>9} {'ΔRSS':>7} {'Offloads':>9}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<25} {r['wall_time_s']:>6.1f}s {r['baseline_rss_mb']:>8}MB {r['peak_rss_mb']:>8}MB {r['delta_rss_mb']:>6}MB {r['offloads']:>9}")
    print("=" * 70)

    # Correctness check: compare output previews
    print()
    print("Correctness check (output preview):")
    for r in results:
        print(f"  [{r['label'][:20]}] {r['output_preview'][:80]}...")

    # Check if unlimited and 256 produce same first 100 chars
    if len(results) >= 2:
        ref = results[0]["output_preview"][:100]
        for r in results[1:]:
            match = r["output_preview"][:100] == ref
            print(f"  {r['label']} vs unlimited: {'MATCH' if match else 'DIFFER'}")


if __name__ == "__main__":
    main()
