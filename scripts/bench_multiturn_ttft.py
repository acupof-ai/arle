#!/usr/bin/env python3
"""Per-turn TTFT of a coding-agent-shaped conversation against any OpenAI-compatible server.

Every turn re-sends the whole history (system prompt + prior turns + a new tool
result), the way Claude Code / opencode do. The request sequence is fully
deterministic (synthetic content, fixed assistant replies), so two servers see
byte-identical prompts and the only variable is how much prefill each one
re-does per turn.

    python3 scripts/bench_multiturn_ttft.py --url http://localhost:8000 --label arle
    python3 scripts/bench_multiturn_ttft.py --url http://localhost:8080 --label mlx-lm
"""

import argparse
import json
import random
import sys
import time
import urllib.request

WORDS = (
    "config parser module returns handle buffer stream token schema index "
    "route cache worker queue branch commit merge lint test fixture socket "
    "thread mutex latency budget replica shard offset header payload cursor"
).split()


def filler(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def served_model(url: str) -> str:
    try:
        with urllib.request.urlopen(f"{url}/v1/models", timeout=5) as r:
            return json.load(r)["data"][0]["id"]
    except Exception:
        return "default"


def one_turn(url: str, model: str, messages: list, max_tokens: int) -> tuple[float, int]:
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    prompt_tokens = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens") or prompt_tokens
            choices = chunk.get("choices") or []
            if ttft is None and choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    ttft = time.perf_counter() - t0
    return (ttft or (time.perf_counter() - t0)) * 1000, prompt_tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--label", default="")
    ap.add_argument("--model", default=None)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--system-words", type=int, default=4500, help="~6K tokens: a coding-agent system prompt")
    ap.add_argument("--tool-words", type=int, default=300, help="~400 tokens of tool output appended per turn")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--json", default=None, help="write per-turn rows here")
    ap.add_argument("--warmup", action="store_true", help="one throwaway request first (model load, JIT)")
    args = ap.parse_args()

    model = args.model or served_model(args.url)
    if args.warmup:
        one_turn(args.url, model, [{"role": "user", "content": "hi"}], 1)
    messages = [{"role": "system", "content":
                 "You are a coding agent. Repository notes follow.\n" + filler(args.system_words, 0)}]
    rows = []
    print(f"{args.label or args.url}  model={model}")
    print(f"{'turn':>4} {'prompt_tok':>10} {'ttft_ms':>9}")
    for t in range(1, args.turns + 1):
        messages.append({"role": "user", "content":
                         f"Turn {t}: inspect the tool output and continue.\n<tool_result>\n"
                         + filler(args.tool_words, t) + "\n</tool_result>"})
        ttft_ms, ptok = one_turn(args.url, model, messages, args.max_tokens)
        # Fixed reply keeps the next prompt identical across servers.
        messages.append({"role": "assistant", "content": f"Turn {t} done. Next I will read the following file."})
        rows.append({"turn": t, "prompt_tokens": ptok, "ttft_ms": round(ttft_ms, 1)})
        print(f"{t:>4} {ptok:>10} {ttft_ms:>9.0f}", flush=True)

    later = sorted(r["ttft_ms"] for r in rows[1:]) or [rows[0]["ttft_ms"]]
    summary = {
        "label": args.label, "url": args.url, "model": model, "turns": args.turns,
        "turn1_ttft_ms": rows[0]["ttft_ms"],
        "turnN_ttft_ms": rows[-1]["ttft_ms"],
        "turn2plus_median_ttft_ms": later[len(later) // 2],
        "final_prompt_tokens": rows[-1]["prompt_tokens"],
    }
    print(json.dumps(summary))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"summary": summary, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
