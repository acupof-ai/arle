#!/usr/bin/env python3
"""Decode-rate probe: long-context prompt, stream N tokens, report TTFT and
steady decode tok/s. Stdlib only (pod has no httpx).

Usage: decode_rate_probe.py --target-tokens N --max-tokens M [--port P] [--runs R]
"""
import argparse, json, os, sys, time, urllib.request

TOPICS = [
    "The river flowed gently past the old stone bridge.",
    "Mountains rose sharply against the pale morning sky.",
    "She opened the wooden door and stepped into the hall.",
    "The market was full of fruit, spices, and fresh bread.",
    "A long train crossed the wide green valley at dawn.",
    "Children played near the fountain in the city square.",
    "The library held thousands of dusty leather books.",
    "Rain fell softly on the roof throughout the night.",
]


def build_prompt(target, seed):
    n = max(1, target // 16)
    sents = ["Note %d: %s" % (i + 1, TOPICS[(i + seed) % len(TOPICS)]) for i in range(n)]
    return "Run %d.\n" % seed + " ".join(sents) + "\n\nContinue."


def one_run(base, prompt, max_tokens):
    body = {"model": "x", "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0, "stream": True}
    req = urllib.request.Request(base + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    n_tok = 0
    last_t = None
    prompt_tokens = None
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = d.get("choices") or [{}]
            choice = choices[0]
            text = choice.get("text", "")
            if text:
                if ttft is None:
                    ttft = time.time() - t0
                n_tok += 1
                last_t = time.time()
            if d.get("usage"):
                prompt_tokens = d["usage"].get("prompt_tokens")
    total = time.time() - t0
    return ttft, total, n_tok, prompt_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=131072)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "18189")))
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    base = "http://127.0.0.1:" + str(args.port)
    for r in range(args.runs):
        seed = args.seed + r
        prompt = build_prompt(args.target_tokens, seed)
        ttft, total, n_tok, pt = one_run(base, prompt, args.max_tokens)
        decode_s = (total - (ttft or 0))
        rate = (n_tok - 1) / decode_s if n_tok > 1 and decode_s > 0 else -1.0
        print("run=%d target=%d prompt_tokens=%s gen=%d ttft_s=%.3f total_s=%.3f decode_tok_s=%.2f"
              % (r, args.target_tokens, pt, n_tok, ttft or -1.0, total, rate), flush=True)


if __name__ == "__main__":
    main()
