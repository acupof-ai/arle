#!/usr/bin/env python3
"""Training data for the DSpark draft.

Two stages, following deepseek-ai/DeepSpec `scripts/data/` minus its third:
we do NOT precompute a target cache. At our context length the cache runs to
tens of terabytes against a few GPU-hours to recompute, and the pod's /host is
0.2 GB/s — `spec_train::trainer::Target` recomputes instead.

    # 1. prompts
    python3 scripts/spec_train_data.py split \
        --out data/spec-train/train.jsonl --eval-out data/spec-train/eval.jsonl

    # 2. regenerate the assistant turns through OUR target, so the draft
    #    learns the distribution it will actually have to predict
    arle serve --model <target> --port 8000 &
    python3 scripts/spec_train_data.py regen \
        --base-url http://127.0.0.1:8000 --model <target> \
        --in data/spec-train/train.jsonl --out data/spec-train/train_regen.jsonl

`regen` is resumable: rerun it and it picks up after the ids already written.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

ROLES = {"human": "user", "gpt": "assistant", "chatgpt": "assistant",
         "bing": "assistant", "bard": "assistant"}


def normalize(row: dict, idx: int) -> dict:
    turns = [{"role": ROLES[m["from"]], "content": m["value"]}
             for m in row["conversations"] if m["from"] in ROLES]
    return {"id": idx, "conversations": turns}


def usable(conv: dict) -> bool:
    turns = conv["conversations"]
    return (bool(turns) and turns[0]["role"] == "user"
            and all(isinstance(m["content"], str) and m["content"] for m in turns))


def cmd_split(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split="train")
    if args.sample_size:
        ds = ds.select(range(min(args.sample_size, len(ds))))
    split = ds.train_test_split(test_size=args.test_size, seed=args.seed)

    for name, rows, path in (("train", split["train"], args.out),
                             ("eval", split["test"], args.eval_out)):
        path.parent.mkdir(parents=True, exist_ok=True)
        kept = 0
        with path.open("w", encoding="utf-8") as fh:
            for i, row in enumerate(rows):
                conv = normalize(row, i)
                if not usable(conv):
                    continue
                if name == "eval":
                    conv = {"id": conv["id"], "turns": [m["content"] for m in
                                                        conv["conversations"]
                                                        if m["role"] == "user"]}
                fh.write(json.dumps(conv, ensure_ascii=False) + "\n")
                kept += 1
        print(f"{name}: {kept} rows -> {path}")


def complete(base_url: str, model: str, messages: list, args: argparse.Namespace) -> str:
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def regen_one(conv: dict, args: argparse.Namespace) -> dict | None:
    """Replay the user turns, take every assistant turn from our target."""
    history: list[dict] = []
    out: list[dict] = []
    for msg in conv["conversations"]:
        if msg["role"] == "user":
            history.append(msg)
            out.append(msg)
            continue
        content = complete(args.base_url, args.model, history, args)
        if not content:
            return None
        reply = {"role": "assistant", "content": content}
        history.append(reply)
        out.append(reply)
    return {"id": conv["id"], "conversations": out}


def cmd_regen(args: argparse.Namespace) -> None:
    convs = [json.loads(line) for line in
             args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    done: set[int] = set()
    if args.out.exists():
        done = {json.loads(line)["id"] for line in
                args.out.read_text(encoding="utf-8").splitlines() if line.strip()}
        print(f"resuming past {len(done)} rows")
    todo = [c for c in convs if c["id"] not in done]
    if not todo:
        print("nothing to do")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock = Lock()
    failed = 0
    with args.out.open("a", encoding="utf-8") as fh:
        def work(conv: dict) -> bool:
            try:
                row = regen_one(conv, args)
            except (urllib.error.URLError, OSError, KeyError, TimeoutError) as exc:
                print(f"id {conv['id']}: {exc}", file=sys.stderr)
                return False
            if row is None:
                return False
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            return True

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for i, ok in enumerate(pool.map(work, todo), start=1):
                failed += not ok
                if i % 100 == 0:
                    print(f"{i}/{len(todo)} ({failed} failed)", flush=True)
    print(f"done: {len(todo) - failed} written, {failed} failed -> {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("split", help="download prompts and split train/eval")
    s.add_argument("--dataset", default="mlabonne/open-perfectblend")
    s.add_argument("--sample-size", type=int)
    s.add_argument("--test-size", type=float, default=0.05)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", type=Path, default=Path("data/spec-train/train.jsonl"))
    s.add_argument("--eval-out", type=Path, default=Path("data/spec-train/eval.jsonl"))
    s.set_defaults(fn=cmd_split)

    r = sub.add_parser("regen", help="regenerate assistant turns through our target")
    r.add_argument("--base-url", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--in", dest="input", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--concurrency", type=int, default=32)
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument("--top-p", type=float, default=0.8)
    r.add_argument("--max-tokens", type=int, default=4096)
    r.add_argument("--timeout", type=float, default=600.0)
    r.set_defaults(fn=cmd_regen)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
