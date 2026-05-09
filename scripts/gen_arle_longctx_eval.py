#!/usr/bin/env python3
"""Generate long-context eval data from ARLE codebase for Phase 3b PPL eval.

Concatenates ARLE docs (README + ROADMAP + docs/index + support-matrix +
architecture + codebase-map + selected plans/research) → tokenizes via
Qwen3 tokenizer → splits into 40k / 64k / 80k token examples → writes
tokenized JSONL per `eval_lm.rs::TokenizedJsonlRecord` schema.

100% offline (no HF Hub download), workaround for #34 blocker.

Usage:
  ./scripts/gen_arle_longctx_eval.py \\
      --tokenizer infer/models/Qwen3-4B/tokenizer.json \\
      --out-dir bench-output/eval-longctx/ \\
      --target-tokens 40960 64000 81920

Per docs/plans/2026-05-10-rope-yarn-phase3b-ppl-eval-plan.md §2.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def gather_arle_text(repo_root: Path) -> str:
    """Concatenate ARLE markdown docs to produce a long English text corpus."""
    sources = []
    # Top-level
    for name in ["README.md", "ROADMAP.md", "CHANGELOG.md", "CLAUDE.md", "CONTRIBUTING.md"]:
        p = repo_root / name
        if p.exists():
            sources.append((name, p.read_text(errors="replace")))

    # docs/ root files
    docs = repo_root / "docs"
    if docs.exists():
        for p in sorted(docs.glob("*.md")):
            sources.append((f"docs/{p.name}", p.read_text(errors="replace")))

        # Plans + research + projects (Markdown only)
        for sub in ["plans", "research", "projects", "experience/wins", "experience/errors"]:
            d = docs / sub
            if not d.exists():
                continue
            for p in sorted(d.glob("*.md")):
                sources.append((f"docs/{sub}/{p.name}", p.read_text(errors="replace")))

    chunks = []
    for name, text in sources:
        chunks.append(f"\n\n=== {name} ===\n\n{text}")
    return "".join(chunks)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", type=Path, default=Path("infer/models/Qwen3-4B/tokenizer.json"))
    p.add_argument("--out-dir", type=Path, default=Path("bench-output/eval-longctx"))
    p.add_argument("--target-tokens", type=int, nargs="+", default=[40960, 64000, 81920], help="Per-example token counts")
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--repeat-if-short", action="store_true", default=True, help="Repeat corpus if too short for target tokens")

    args = p.parse_args()

    if not args.tokenizer.exists():
        raise SystemExit(f"tokenizer not found: {args.tokenizer}")

    # Lazy import — keeps the script importable even if transformers missing.
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(args.tokenizer))
    except Exception:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(args.tokenizer.parent))

    # Gather + tokenize
    corpus = gather_arle_text(args.repo_root)
    print(f"[gen] corpus chars: {len(corpus):,}", file=sys.stderr)

    if hasattr(tok, "encode") and hasattr(tok.encode("hi"), "ids"):
        # tokenizers Tokenizer (returns Encoding)
        ids = tok.encode(corpus).ids
    else:
        # transformers AutoTokenizer
        ids = tok.encode(corpus, add_special_tokens=False)
    print(f"[gen] corpus tokens: {len(ids):,}", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for n in args.target_tokens:
        if len(ids) < n:
            if args.repeat_if_short:
                print(f"[gen] target {n} > corpus {len(ids)}, repeating to fill", file=sys.stderr)
                # Tile
                k = (n + len(ids) - 1) // len(ids)
                example_ids = (ids * k)[:n]
            else:
                print(f"[gen] target {n} > corpus {len(ids)}, skipping", file=sys.stderr)
                continue
        else:
            # Just slice the head
            example_ids = ids[:n]

        out = args.out_dir / f"eval-longctx-{n}.tokenized.jsonl"
        record = {"input_ids": example_ids}
        # Single-example file (eval iterates over JSONL lines). For multi-example
        # variation, slice from offsets (e.g. ids[:n], ids[n:2n], ...) — for now
        # just one example per file is enough for PPL trend analysis.
        with out.open("w") as f:
            json.dump(record, f)
            f.write("\n")
        print(f"[gen] wrote {out} ({len(example_ids):,} tokens, 1 example)")

    print()
    print("Run eval with:")
    print(f"  ./target/release/arle train eval --model <model> --data {args.out_dir}/eval-longctx-N.tokenized.jsonl --seq-len N")


if __name__ == "__main__":
    main()
