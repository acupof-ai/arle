#!/usr/bin/env python3
"""Convert Terminal-Bench `terminus` agent-logs into verl token records for OPD.

Each episode's `agent-logs/episode-N/debug.json` (a LiteLLM call dump) carries the
request messages (`input`/`messages`) and the model's `original_response`. We turn
each episode into one masked-CE `(prompt, completion)` pair — completion = the
assistant action, mask=1 on all completion tokens — using the model's own chat
template so the tokenization matches serving. Feed the output to
`arle train agent-opd --replay-records`.

Usage:
  python3 terminus_to_records.py <run_dir> <tokenizer_dir> <out.jsonl> <task_id>...

`run_dir` is a `tb run --output-path` run directory; task_ids select which tasks'
trajectories to distill (typically the PASSING ones, execution-filtered upstream).
Needs `transformers` for `apply_chat_template`.
"""
import glob
import json
import os
import sys

from transformers import AutoTokenizer


def resp_content(orig):
    """Pull the assistant text out of a LiteLLM `original_response` (str or dict)."""
    if isinstance(orig, str):
        try:
            orig = json.loads(orig)
        except json.JSONDecodeError:
            return orig
    if isinstance(orig, dict):
        return (orig.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
    return ""


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    run_dir, tok_dir, out = sys.argv[1:4]
    tasks = sys.argv[4:]
    tok = AutoTokenizer.from_pretrained(tok_dir)

    recs = []
    for t in tasks:
        pattern = f"{run_dir}/**/{t}*/agent-logs/episode-*/debug.json"
        for dbg in sorted(glob.glob(pattern, recursive=True)):
            d = json.load(open(dbg))
            msgs = d.get("input") or d.get("messages") or []
            comp = resp_content(d.get("original_response"))
            if not msgs or not comp.strip():
                continue
            prompt_txt = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            pids = tok(prompt_txt, add_special_tokens=False)["input_ids"]
            rids = tok(comp, add_special_tokens=False)["input_ids"]
            recs.append(
                {
                    "label": f"{t}:{os.path.basename(os.path.dirname(dbg))}",
                    "prompt_ids": pids,
                    "response_ids": rids,
                    "response_mask": [1] * len(rids),
                    "masked_tokens": len(rids),
                    "total_tokens": len(pids) + len(rids),
                }
            )

    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(recs)} records from {len(tasks)} task(s) -> {out}")
    for r in recs[:8]:
        print(f"  {r['label']} tok={r['total_tokens']} masked={r['masked_tokens']}")


if __name__ == "__main__":
    main()
