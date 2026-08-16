#!/usr/bin/env python3
"""Cold-prefill TTFT probe for the ARLE OpenAI-compatible server.

Builds a deterministic filler prompt targeting ~N tokens, sends it with
stream=True and max_tokens=1, and reports time-to-first-token (== prefill
time for a cold KV) plus the server-reported prompt_tokens.

Usage: ttft_probe.py [--target-tokens N] [--port P] [--runs R] [--seed S]
"""
import argparse, json, os, sys, time, urllib.request

BASE = "http://127.0.0.1:"
NEEDLE = "738291"
PRE = "Important: the secret access code is " + NEEDLE + ". Keep it in mind.\n\n"
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
CUE = "\n\nRecall the secret access code stated earlier. The secret access code is"


def build_prompt(target, seed):
    # ~16 tokens/sentence (matches needle_gate.py heuristic).
    n = max(1, target // 16)
    sents = ["Note %d: %s" % (i + 1, TOPICS[(i + seed) % len(TOPICS)]) for i in range(n)]
    # Unique per-seed prefix kills prefix-cache reuse across runs (cold each run).
    return "Run %d.\n" % seed + " ".join(sents) + CUE


def one_run(base, prompt, seed):
    body = {"model": "x", "prompt": prompt, "max_tokens": 1,
            "temperature": 0, "stream": True}
    req = urllib.request.Request(base + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    text = ""
    prompt_tokens = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if ttft is None:
                ttft = time.time() - t0
            choice = d.get("choices", [{}])[0]
            text += choice.get("text", "")
            if d.get("usage"):
                prompt_tokens = d["usage"].get("prompt_tokens")
    total = time.time() - t0
    return ttft, total, prompt_tokens, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=131072)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "18189")))
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    base = BASE + str(args.port)
    for r in range(args.runs):
        seed = args.seed + r
        prompt = build_prompt(args.target_tokens, seed)
        ttft, total, pt, text = one_run(base, prompt, seed)
        print("run=%d target=%d prompt_tokens=%s ttft_s=%.3f total_s=%.3f out=%r"
              % (r, args.target_tokens, pt, ttft if ttft else -1.0, total, text[:40]),
              flush=True)


if __name__ == "__main__":
    main()
