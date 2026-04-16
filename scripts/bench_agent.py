"""Benchmark agent-infer Rust binary with long agent prompts.

Sends multi-turn agent conversations via stdin to the Rust binary,
measures TTFT, throughput, KV cache hit rates from logs.
"""

import subprocess
import time
import json
import re
import sys

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-4B"

# Long agent prompt sequences - each is a multi-turn conversation
BENCHMARK_PROMPTS = [
    # 1. Multi-step math (3 tool calls expected)
    "Calculate the sum of squares from 1 to 100 using python, then find its square root, and tell me if the result is prime",

    # 2. System analysis (shell + python)
    "Check the system's CPU info and memory usage with shell commands, then use python to parse and summarize the results",

    # 3. Code generation and execution
    "Write a python program that generates the first 50 prime numbers, then calculate their average and standard deviation",

    # 4. File operations + analysis
    "List all .py files in the current directory with shell, count the total lines of code, and find the largest file",

    # 5. Complex calculation
    "Use python to calculate: the 100th Fibonacci number, then verify it using Binet's formula, and compute the percentage error",
]


def run_single_prompt(prompt: str, max_tokens: int = 4096) -> dict:
    """Run a single prompt through the Rust agent binary."""
    cmd = [
        "./target/release/agent-infer",
        "--model-path", MODEL_PATH,
        "--max-tokens", str(max_tokens),
        "--max-turns", "5",
    ]

    env = {
        "LD_LIBRARY_PATH": "/usr/lib64-nvidia:/usr/local/cuda/lib64",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/root",
    }

    start = time.perf_counter()

    proc = subprocess.run(
        cmd,
        input=f"{prompt}\nquit\n",
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )

    total_time = time.perf_counter() - start
    output = proc.stdout + proc.stderr

    # Parse metrics from logs
    kv_hits = re.findall(r"KV prefix cache HIT: reusing (\d+)/(\d+) tokens \(saving ([\d.]+)%", output)
    kv_misses = len(re.findall(r"KV prefix cache MISS", output))
    kv_partials = len(re.findall(r"KV prefix cache PARTIAL", output))

    turns = re.findall(r"Agent turn (\d+)/\d+: prompt length = (\d+) chars", output)
    generated = re.findall(r"Generated (\d+) chars, finish_reason=(\w+)", output)
    tool_calls = re.findall(r"\[tool: (\w+)\]", output)

    # Extract final response
    final_lines = []
    for line in output.split("\n"):
        if line.startswith("\x1b[1;34m"):  # Blue text = final response
            clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
            final_lines.append(clean)

    return {
        "prompt": prompt[:80],
        "total_time_s": total_time,
        "turns": len(turns),
        "tool_calls": tool_calls,
        "kv_hits": [(int(a), int(b), float(c)) for a, b, c in kv_hits],
        "kv_misses": kv_misses,
        "kv_partials": kv_partials,
        "generated_chars": [int(g[0]) for g in generated],
        "response": "\n".join(final_lines)[:200],
        "exit_code": proc.returncode,
    }


def main():
    print("=" * 70)
    print("agent-infer Benchmark (Pure Rust, No Python Glue)")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Prompts: {len(BENCHMARK_PROMPTS)}")
    print()

    results = []
    for i, prompt in enumerate(BENCHMARK_PROMPTS):
        print(f"[{i+1}/{len(BENCHMARK_PROMPTS)}] {prompt[:60]}...")
        result = run_single_prompt(prompt)
        results.append(result)

        # Print per-prompt stats
        total_kv_saved = sum(h[2] for h in result["kv_hits"]) / max(len(result["kv_hits"]), 1)
        print(f"  Time: {result['total_time_s']:.1f}s | "
              f"Turns: {result['turns']} | "
              f"Tools: {result['tool_calls']} | "
              f"KV hits: {len(result['kv_hits'])} (avg {total_kv_saved:.1f}% saved)")
        if result["response"]:
            print(f"  Response: {result['response'][:100]}...")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_time = sum(r["total_time_s"] for r in results)
    total_turns = sum(r["turns"] for r in results)
    total_tools = sum(len(r["tool_calls"]) for r in results)
    total_kv_hits = sum(len(r["kv_hits"]) for r in results)
    total_kv_misses = sum(r["kv_misses"] for r in results)
    all_kv_savings = [h[2] for r in results for h in r["kv_hits"]]
    avg_kv_saving = sum(all_kv_savings) / max(len(all_kv_savings), 1)

    print(f"Total time:        {total_time:.1f}s")
    print(f"Total turns:       {total_turns}")
    print(f"Total tool calls:  {total_tools}")
    print(f"KV cache hits:     {total_kv_hits}")
    print(f"KV cache misses:   {total_kv_misses}")
    print(f"Avg KV savings:    {avg_kv_saving:.1f}%")
    print(f"Avg time/prompt:   {total_time/len(results):.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
