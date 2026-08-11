#!/usr/bin/env python3
"""Find GSM8K train problems the baseline model gets wrong.

Uses the chat endpoint with a step-by-step instruction, matching the eval
format in arle_capability_eval.py. Concurrent requests for throughput.
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from arle_capability_eval import ArleClient, _gsm8k_extract_answer, _gsm8k_gold_answer


def extract_question(text: str) -> str:
    m = re.match(r"Q:\s*(.+?)\s*\nA:", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    client = ArleClient(args.base_url, args.model, timeout=600)

    with open(args.input) as f:
        problems = [json.loads(line) for line in f if line.strip()]

    if args.limit > 0:
        problems = problems[: args.limit]

    print(f"[find-wrong] {len(problems)} problems, concurrency={args.concurrency}", flush=True)

    correct = 0
    wrong = []
    invalid = 0
    t0 = time.time()

    def work(idx_p):
        idx, p = idx_p
        question = extract_question(p["text"])
        gold = _gsm8k_gold_answer(p["completion"])
        instruction = (
            f"{question}\n\n"
            "Solve step by step. End your response with the final numeric "
            "answer on its own, preceded by ####.\n<think>"
        )
        try:
            resp = client.chat([{"role": "user", "content": instruction}], args.max_tokens)
        except Exception as e:
            return idx, p, None, gold, f"request_error:{e}"
        pred = _gsm8k_extract_answer(resp)
        if pred is None:
            return idx, p, None, gold, "extract_fail"
        if pred == gold:
            return idx, p, pred, gold, "correct"
        return idx, p, pred, gold, "wrong"

    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(work, (i, p)): i for i, p in enumerate(problems)}
        for fut in as_completed(futures):
            idx, p, pred, gold, status = fut.result()
            done += 1
            if status == "correct":
                correct += 1
            elif status == "wrong":
                wrong.append({**p, "predicted": pred, "gold": gold, "status": status})
            else:
                invalid += 1
                entry = {**p, "predicted": None, "gold": gold, "status": status}
                if status == "extract_fail":
                    entry["response"] = ""
                wrong.append(entry)

            if done % 50 == 0 or done == len(problems):
                acc = correct / max(1, done - invalid)
                print(
                    f"[find-wrong] {done}/{len(problems)} "
                    f"acc={acc:.3f} correct={correct} wrong={len(wrong)} invalid={invalid}",
                    flush=True,
                )

    with open(args.output, "w") as f:
        for w in wrong:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    total = len(problems)
    print(
        f"[find-wrong] done: {correct}/{total} correct, "
        f"{len(wrong)} wrong, {invalid} invalid in {elapsed:.0f}s",
        flush=True,
    )
    print(f"[find-wrong] wrong problems saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
