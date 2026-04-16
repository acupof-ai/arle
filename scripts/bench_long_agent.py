#!/usr/bin/env python3
"""Long-sequence agent inference benchmark.

Simulates multi-turn agent conversations via the HTTP /v1/chat/completions API.
Each "agent turn" sends the growing context (system + user + assistant history + tool results)
back to the model, measuring how performance degrades with context length.

Metrics:
  - TTFT per turn (time to first token)
  - ITL (inter-token latency)
  - Throughput (tok/s) per turn and aggregate
  - Context growth rate
  - KV cache utilization
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass

try:
    import httpx
except ImportError:
    sys.exit("pip install httpx")


SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools. "
    "When you need to use a tool, output a JSON tool call. "
    "Think step by step."
)

# Multi-turn agent scenarios — each is a list of user messages for one conversation
AGENT_SCENARIOS = [
    {
        "name": "math-chain",
        "description": "Multi-step math with tool calls",
        "turns": [
            "Calculate 17 * 23 using python",
            "Now take that result and compute its square root, rounded to 2 decimal places",
            "Is the rounded result a prime number? Check with python",
            "Sum up all three results you computed",
        ],
    },
    {
        "name": "sysinfo",
        "description": "System inspection with growing context",
        "turns": [
            "What operating system is this? Use shell: uname -a",
            "How much memory does this system have? Use shell: free -h",
            "What CPU is installed? Use shell: cat /proc/cpuinfo | head -20",
            "How much disk space is available? Use shell: df -h /",
            "Summarize all the system information you gathered",
        ],
    },
    {
        "name": "code-gen",
        "description": "Iterative code generation",
        "turns": [
            "Write a python function to check if a number is prime",
            "Now test it with the first 20 numbers and print the primes",
            "Modify the function to also return the smallest prime factor for non-primes",
            "Run the modified function on numbers 1 to 30 and display results as a table",
        ],
    },
    {
        "name": "data-pipeline",
        "description": "Data processing pipeline",
        "turns": [
            "Use python to generate a list of 50 random integers between 1 and 1000",
            "Sort them and find the median, mean, and standard deviation",
            "Filter out numbers above the mean and compute stats on the remaining",
            "Create a frequency histogram (text-based) of the original data with 10 bins",
        ],
    },
    {
        "name": "long-context",
        "description": "Deliberate context growth",
        "turns": [
            "Use python to print the fibonacci sequence up to the 40th number, one per line",
            "Now compute the ratio of consecutive fibonacci numbers and print all 39 ratios",
            "Which ratio is closest to the golden ratio (1.618034)? Print the index and value",
            "Compute the sum of all fibonacci numbers you generated, and verify with the formula F(n+2)-1",
            "Write a summary of all results: the sequence, ratios, golden ratio match, and sum verification",
        ],
    },
]


@dataclass
class TurnResult:
    turn: int
    ttft_ms: float
    total_ms: float
    prompt_tokens: int
    completion_tokens: int
    itl_ms: float  # average inter-token latency


@dataclass
class ScenarioResult:
    name: str
    turns: list  # list[TurnResult]
    total_ms: float


async def run_scenario(client: httpx.AsyncClient, url: str, scenario: dict, max_tokens: int) -> ScenarioResult:
    """Run a multi-turn agent scenario, accumulating context."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    turn_results = []
    scenario_start = time.time()

    for turn_idx, user_msg in enumerate(scenario["turns"]):
        messages.append({"role": "user", "content": user_msg})

        payload = {
            "model": "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }

        ttft = None
        tokens = []
        token_times = []
        prompt_tokens = 0
        completion_tokens = 0
        full_content = ""

        turn_start = time.time()
        try:
            async with client.stream("POST", f"{url}/v1/chat/completions", json=payload, timeout=120) as resp:
                if resp.status_code != 200:
                    # Non-streaming error
                    body = await resp.aread()
                    print(f"    Turn {turn_idx+1}: HTTP {resp.status_code} — {body[:200]}")
                    turn_results.append(TurnResult(
                        turn=turn_idx + 1, ttft_ms=0, total_ms=0,
                        prompt_tokens=0, completion_tokens=0, itl_ms=0
                    ))
                    messages.append({"role": "assistant", "content": "(error)"})
                    continue

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Extract delta
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            now = time.time()
                            if ttft is None:
                                ttft = (now - turn_start) * 1000
                            tokens.append(content)
                            token_times.append(now)
                            full_content += content

                    usage = chunk.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)

        except Exception as e:
            print(f"    Turn {turn_idx+1}: Error — {e}")
            turn_results.append(TurnResult(
                turn=turn_idx + 1, ttft_ms=0, total_ms=0,
                prompt_tokens=0, completion_tokens=0, itl_ms=0
            ))
            messages.append({"role": "assistant", "content": "(error)"})
            continue

        total_ms = (time.time() - turn_start) * 1000

        # Compute ITL
        if len(token_times) > 1:
            deltas = [token_times[i] - token_times[i-1] for i in range(1, len(token_times))]
            itl_ms = (sum(deltas) / len(deltas)) * 1000
        else:
            itl_ms = 0

        tr = TurnResult(
            turn=turn_idx + 1,
            ttft_ms=ttft or 0,
            total_ms=total_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens or len(tokens),
            itl_ms=itl_ms,
        )
        turn_results.append(tr)

        # Add assistant response to context for next turn
        messages.append({"role": "assistant", "content": full_content or "(empty)"})

    total = (time.time() - scenario_start) * 1000
    return ScenarioResult(name=scenario["name"], turns=turn_results, total_ms=total)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--scenarios", type=int, default=0, help="0=all")
    args = parser.parse_args()

    scenarios = AGENT_SCENARIOS[:args.scenarios] if args.scenarios > 0 else AGENT_SCENARIOS

    print("=" * 78)
    print("Long-Sequence Agent Inference Benchmark")
    print(f"  Server:      {args.url}")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  Scenarios:   {len(scenarios)}")
    print("=" * 78)

    async with httpx.AsyncClient() as client:
        # Verify server
        try:
            r = await client.get(f"{args.url}/v1/stats", timeout=5)
            print(f"  Server:      OK ({r.status_code})")
        except Exception as e:
            print(f"  Server:      UNREACHABLE ({e})")
            return

        print()

        all_results = []
        for si, scenario in enumerate(scenarios):
            if si > 0:
                # Wait for all scheduler slots to drain between scenarios
                for _ in range(30):
                    try:
                        sr = await client.get(f"{args.url}/v1/stats", timeout=2)
                        if "active=0" in sr.text:
                            break
                    except Exception:
                        pass
                    await asyncio.sleep(1)
            print(f"--- {scenario['name']}: {scenario['description']} ({len(scenario['turns'])} turns) ---")

            result = await run_scenario(client, args.url, scenario, args.max_tokens)
            all_results.append(result)

            # Print per-turn stats
            for tr in result.turns:
                tps = (tr.completion_tokens / (tr.total_ms / 1000)) if tr.total_ms > 0 else 0
                print(
                    f"  T{tr.turn}  "
                    f"TTFT={tr.ttft_ms:7.0f}ms  "
                    f"ITL={tr.itl_ms:5.1f}ms  "
                    f"prompt={tr.prompt_tokens:5d}  "
                    f"gen={tr.completion_tokens:4d}  "
                    f"tok/s={tps:5.1f}  "
                    f"total={tr.total_ms/1000:5.1f}s"
                )
            print(f"  Scenario total: {result.total_ms/1000:.1f}s")
            print()

    # Aggregate summary
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    all_turns = [tr for r in all_results for tr in r.turns if tr.total_ms > 0]
    if not all_turns:
        print("  No successful turns.")
        return

    ttfts = [tr.ttft_ms for tr in all_turns if tr.ttft_ms > 0]
    itls = [tr.itl_ms for tr in all_turns if tr.itl_ms > 0]
    total_prompt = sum(tr.prompt_tokens for tr in all_turns)
    total_gen = sum(tr.completion_tokens for tr in all_turns)
    total_time = sum(r.total_ms for r in all_results) / 1000

    print(f"  Scenarios:         {len(all_results)}")
    print(f"  Total turns:       {len(all_turns)}")
    print(f"  Total prompt tok:  {total_prompt:,}")
    print(f"  Total gen tok:     {total_gen:,}")
    print(f"  Wall time:         {total_time:.1f}s")
    print(f"  Throughput:        {total_gen / total_time:.1f} tok/s")
    print()

    if ttfts:
        ttfts.sort()
        print(f"  TTFT  p50={ttfts[len(ttfts)//2]:.0f}ms  p90={ttfts[int(len(ttfts)*0.9)]:.0f}ms  max={ttfts[-1]:.0f}ms")
    if itls:
        itls.sort()
        print(f"  ITL   p50={itls[len(itls)//2]:.1f}ms  p90={itls[int(len(itls)*0.9)]:.1f}ms  max={itls[-1]:.1f}ms")

    # Context growth analysis
    print()
    print("  Context growth (prompt tokens per turn):")
    for r in all_results:
        ctx_sizes = [f"{tr.prompt_tokens}" for tr in r.turns if tr.prompt_tokens > 0]
        if ctx_sizes:
            print(f"    {r.name:20s}  {' → '.join(ctx_sizes)}")

    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
