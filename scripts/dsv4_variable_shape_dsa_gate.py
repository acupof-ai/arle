#!/usr/bin/env python3
"""Variable-shape DSv4 DSA correctness gate.

Runs a list of coherent, real-text prompts through one loaded `dsv4_parity`
engine session per variant:

  1. legacy csa_select multiple times (`ARLE_DSV4_DSA_INDEXER=0`) to establish
     the same-config non-determinism floor;
  2. official DSA once (`ARLE_DSV4_DSA_INDEXER=1`).

The prompts vary in length and are processed sequentially in the same executor
session, exercising slot reset / scratch reuse across request shapes. The gate is
relative correctness: official must diverge from legacy no earlier than legacy
diverges from itself under the same config. Timing is reported as a sanity read,
not as a formal A/B license.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer


DEFAULT_LENGTHS = [64, 256, 512, 1024, 2048, 4096]
DEFAULT_ANSWERS = ["saffron", "cobalt", "marigold", "topaz", "willow", "ember"]
CASE_RE = re.compile(
    r"case_result index=(?P<index>\d+) prompt_len=(?P<prompt_len>\d+) "
    r"clean_tokens=\[(?P<tokens>[^\]]*)\]"
)
TIMING_RE = re.compile(
    r"\[dsv4-parity timing rank=0\] case=(?P<index>\d+) "
    r"prompt_tokens=(?P<prompt_tokens>\d+) max_new=(?P<max_new>\d+) "
    r"load_ms=(?P<load_ms>[-+0-9.]+) prefill_ms=(?P<prefill_ms>[-+0-9.]+) "
    r"decode_steps=(?P<decode_steps>\d+) decode_ms=(?P<decode_ms>[-+0-9.]+) "
    r"decode_tok_s=(?P<decode_tok_s>[-+0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/data01/models/DeepSeek-V4-Flash")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--bin", default="target/release/examples/dsv4_parity")
    parser.add_argument("--max-new", type=int, default=8)
    parser.add_argument("--lengths", default=",".join(map(str, DEFAULT_LENGTHS)))
    parser.add_argument(
        "--legacy-runs",
        type=int,
        default=3,
        help="same-config legacy repetitions used to estimate the non-determinism floor",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="default: /tmp/dsv4-variable-dsa-gate-<unix>",
    )
    parser.add_argument(
        "--parse-existing",
        action="store_true",
        help="read existing legacy1/legacy2/official logs under --output-dir; do not run GPUs",
    )
    return parser.parse_args()


def encode(tok: Tokenizer, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False).ids


def decode(tok: Tokenizer, ids: list[int]) -> str:
    return tok.decode(ids, skip_special_tokens=False)


def load_real_text(repo_root: Path) -> str:
    candidates = [
        repo_root / "docs" / "architecture.md",
        repo_root / "docs" / "plans" / "2026-06-06-dsv4-pd-systematic-analysis.md",
        repo_root / "docs" / "index.md",
        repo_root / "README.md",
        repo_root / "AGENTS.md",
    ]
    chunks: list[str] = []
    for path in candidates:
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    if not chunks:
        raise SystemExit(f"no real-text prompt sources found under {repo_root}")
    text = "\n\n".join(chunks)
    # Use project docs as the source material, but strip markdown/table/code
    # structure. Raw markdown makes the greedy base model continue with bullets,
    # backticks, and table punctuation instead of answering the final needle.
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    cleaned_lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("#", "|", "```")):
            continue
        if set(line) <= {"-", " ", "|", ":"}:
            continue
        line = re.sub(r"`([^`]*)`", r"\1", line)
        line = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", line)
        line = re.sub(r"[*_>#|{}\\[\\]()/\\\\]", " ", line)
        line = re.sub(r"https?://\\S+", " ", line)
        line = " ".join(line.split())
        if len(line) >= 40:
            cleaned_lines.append(line)
    if not cleaned_lines:
        raise SystemExit("real-text prompt source sanitized to empty text")
    return " ".join(cleaned_lines)


def build_prompt(
    tok: Tokenizer,
    real_text: str,
    target_len: int,
    answer: str,
) -> tuple[str, list[int]]:
    suffix = (
        f"\n\nFinal verification sentence. The answer word is {answer}.\n"
        "The answer word is"
    )
    suffix_ids = encode(tok, suffix)
    if len(suffix_ids) + 8 > target_len:
        raise ValueError(f"target length {target_len} too small for suffix")

    budget = target_len - len(suffix_ids)
    text_ids = encode(tok, real_text)
    if len(text_ids) < budget:
        repeats = (budget // max(1, len(text_ids))) + 2
        text_ids = encode(tok, (real_text + " ") * repeats)
    prefix_ids = text_ids[:budget]
    prompt_ids = prefix_ids + suffix_ids
    prompt = decode(tok, prompt_ids)
    reparsed = encode(tok, prompt)
    if len(reparsed) != len(prompt_ids):
        # Tokenizers can normalize whitespace on decode. Keep the original IDs
        # for the harness; the text is only for reporting.
        prompt = decode(tok, prompt_ids)
    return prompt, prompt_ids


def write_prompt_list(path: Path, prompts: list[list[int]]) -> None:
    path.write_text(
        "\n".join(",".join(map(str, ids)) for ids in prompts) + "\n",
        encoding="utf-8",
    )


def parse_tokens(raw: str) -> list[int]:
    raw = raw.strip()
    if not raw:
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_variant_log(name: str, log_path: Path) -> list[dict]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rows = []
    timing = {}
    for match in TIMING_RE.finditer(text):
        decode_steps = int(match.group("decode_steps"))
        decode_ms = float(match.group("decode_ms"))
        timing[int(match.group("index"))] = {
            "prefill_ms": float(match.group("prefill_ms")),
            "decode_steps": decode_steps,
            "decode_ms": decode_ms,
            "decode_tok_s": float(match.group("decode_tok_s")),
            "decode_ms_per_token": decode_ms / decode_steps if decode_steps else None,
        }
    for match in CASE_RE.finditer(text):
        index = int(match.group("index"))
        rows.append(
            {
                "index": index,
                "prompt_len": int(match.group("prompt_len")),
                "tokens": parse_tokens(match.group("tokens")),
                **timing.get(index, {}),
            }
        )
    if not rows:
        raise RuntimeError(f"{name} produced no case_result lines; log={log_path}")
    rows.sort(key=lambda row: row["index"])
    return rows


def run_variant(
    repo_root: Path,
    env_base: dict[str, str],
    name: str,
    official: bool,
    log_path: Path,
) -> list[dict]:
    env = os.environ.copy()
    env.update(env_base)
    env["ARLE_DSV4_DSA_INDEXER"] = "1" if official else "0"
    env.pop("ARLE_DSV4_STAGE_PROFILE", None)
    env.pop("ARLE_DSV4_LINEAR_PROFILE", None)
    env.pop("ARLE_DSV4_FLASHMLA_DECODE", None)

    proc = subprocess.run(
        ["bash", "scripts/dsv4_multigpu_parity.sh"],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with exit {proc.returncode}; log={log_path}")
    return parse_variant_log(name, log_path)


def coherent(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    head = stripped[:6]
    return not (len(head) >= 5 and len(set(head)) == 1)


def first_divergence_index(left: list[int], right: list[int]) -> int | None:
    for idx, (lhs, rhs) in enumerate(zip(left, right)):
        if lhs != rhs:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def floor_value(divergence: int | None, token_count: int) -> int:
    return token_count if divergence is None else divergence


def display_divergence(divergence: int | None) -> str:
    return "none" if divergence is None else str(divergence)


def legacy_floor_divergence(legacy_rows: list[dict], case_idx: int) -> int | None:
    floor: int | None = None
    for left_idx in range(len(legacy_rows)):
        for right_idx in range(left_idx + 1, len(legacy_rows)):
            div = first_divergence_index(
                legacy_rows[left_idx][case_idx]["tokens"],
                legacy_rows[right_idx][case_idx]["tokens"],
            )
            if div is None:
                continue
            if floor is None or div < floor:
                floor = div
    return floor


def main() -> int:
    args = parse_args()
    if args.legacy_runs < 2:
        raise SystemExit("--legacy-runs must be >= 2")
    repo_root = Path(args.repo_root).resolve()
    model_path = Path(args.model_path)
    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    lengths = [int(v) for v in args.lengths.split(",") if v.strip()]
    answers = DEFAULT_ANSWERS[:]
    if len(lengths) > len(answers):
        raise SystemExit(f"only {len(answers)} built-in answers for {len(lengths)} lengths")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"/tmp/dsv4-variable-dsa-gate-{int(time.time())}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    real_text = load_real_text(repo_root)
    prompts: list[list[int]] = []
    cases = []
    for idx, (target_len, answer) in enumerate(zip(lengths, answers)):
        prompt_text, prompt_ids = build_prompt(tokenizer, real_text, target_len, answer)
        prompts.append(prompt_ids)
        cases.append(
            {
                "index": idx,
                "target_len": target_len,
                "prompt_len": len(prompt_ids),
                "answer": answer,
                "prompt_tail": prompt_text[-320:],
            }
        )

    prompt_file = output_dir / "prompt_ids.list"
    write_prompt_list(prompt_file, prompts)

    env_base = {
        "INFER_DSV4_MODEL_PATH": str(model_path),
        "DSV4_PARITY_BIN": str((repo_root / args.bin).resolve()),
        "INFER_DSV4_PROMPT_IDS_LIST_FILE": str(prompt_file),
        "INFER_DSV4_MAX_NEW": str(args.max_new),
        "CUDA_HOME": os.environ.get("CUDA_HOME", "/usr/local/cuda"),
        "CUDARC_CUDA_VERSION": os.environ.get("CUDARC_CUDA_VERSION", "12090"),
        "ARLE_DEEPEP_DIR": os.environ.get(
            "ARLE_DEEPEP_DIR", str((repo_root / "../DeepEP").resolve())
        ),
    }

    if args.parse_existing:
        legacy_runs = [
            parse_variant_log(f"legacy{idx}", output_dir / f"legacy{idx}.log")
            for idx in range(1, args.legacy_runs + 1)
        ]
        official = parse_variant_log("official", output_dir / "official.log")
    else:
        legacy_runs = [
            run_variant(
                repo_root,
                env_base,
                f"legacy{idx}",
                False,
                output_dir / f"legacy{idx}.log",
            )
            for idx in range(1, args.legacy_runs + 1)
        ]
        official = run_variant(repo_root, env_base, "official", True, output_dir / "official.log")

    by_name = {
        **{f"legacy{idx}": rows for idx, rows in enumerate(legacy_runs, start=1)},
        "official": official,
    }
    report_rows = []
    all_pass = True
    for case in cases:
        idx = case["index"]
        l1 = legacy_runs[0][idx]["tokens"]
        off = official[idx]["tokens"]
        legacy_tokens = [run[idx]["tokens"] for run in legacy_runs]
        legacy_texts = [decode(tokenizer, run[idx]["tokens"]) for run in legacy_runs]
        off_text = decode(tokenizer, off)
        answer = case["answer"].lower()
        legacy_hit = any(answer in text.lower() for text in legacy_texts)
        official_hit = answer in off_text.lower()
        legacy_divergence = legacy_floor_divergence(legacy_runs, idx)
        official_divergence = first_divergence_index(off, l1)
        same_twice = legacy_divergence is None
        exact_to_legacy = any(off == tokens for tokens in legacy_tokens)
        within_floor = floor_value(official_divergence, len(off)) >= floor_value(
            legacy_divergence, len(l1)
        )
        l1_ms = legacy_runs[0][idx].get("decode_ms_per_token")
        off_ms = official[idx].get("decode_ms_per_token")
        decode_ms_delta = (
            None if l1_ms is None or off_ms is None else off_ms - l1_ms
        )
        prefill_ms_delta = (
            None
            if legacy_runs[0][idx].get("prefill_ms") is None
            or official[idx].get("prefill_ms") is None
            else official[idx]["prefill_ms"] - legacy_runs[0][idx]["prefill_ms"]
        )
        passed = coherent(off_text) and within_floor
        all_pass = all_pass and passed
        report_rows.append(
            {
                **case,
                "legacy_floor_divergence_idx": legacy_divergence,
                "official_divergence_idx": official_divergence,
                "legacy_floor_divergence": display_divergence(legacy_divergence),
                "official_divergence": display_divergence(official_divergence),
                "legacy_same_twice": same_twice,
                "legacy_hit": legacy_hit,
                "official_hit": official_hit,
                "official_exact_to_any_legacy": exact_to_legacy,
                "within_floor": within_floor,
                "pass": passed,
                "legacy1_prefill_ms": legacy_runs[0][idx].get("prefill_ms"),
                "legacy2_prefill_ms": legacy_runs[1][idx].get("prefill_ms"),
                "official_prefill_ms": official[idx].get("prefill_ms"),
                "prefill_ms_delta_official_minus_legacy1": prefill_ms_delta,
                "legacy1_decode_ms_per_token": l1_ms,
                "legacy2_decode_ms_per_token": legacy_runs[1][idx].get("decode_ms_per_token"),
                "official_decode_ms_per_token": off_ms,
                "decode_ms_per_token_delta_official_minus_legacy1": decode_ms_delta,
                "legacy_tokens": legacy_tokens,
                "official_tokens": off,
                "legacy_texts": legacy_texts,
                "official_text": off_text,
            }
        )

    payload = {
        "model_path": str(model_path),
        "max_new": args.max_new,
        "legacy_runs": args.legacy_runs,
        "output_dir": str(output_dir),
        "variants": {name: len(rows) for name, rows in by_name.items()},
        "rows": report_rows,
        "all_pass": all_pass,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    print(f"DSV4_VARIABLE_DSA_GATE output_dir={output_dir}")
    print(
        "idx target prompt legacy_floor_idx official_div_idx within_floor "
        "legacy_prefill_ms official_prefill_ms legacy_decode_ms_tok official_decode_ms_tok "
        "delta_prefill_ms delta_decode_ms_tok pass"
    )
    for row in report_rows:
        print(
            "{index} {target_len} {prompt_len} {legacy_floor_divergence} "
            "{official_divergence} {within_floor} "
            "{legacy1_prefill_ms} {official_prefill_ms} "
            "{legacy1_decode_ms_per_token} {official_decode_ms_per_token} "
            "{prefill_ms_delta_official_minus_legacy1} "
            "{decode_ms_per_token_delta_official_minus_legacy1} {pass}".format(**row)
        )
        for legacy_idx, text in enumerate(row["legacy_texts"], start=1):
            print(f"  legacy{legacy_idx}: {text!r}")
        print(f"  official: {row['official_text']!r}")
    return 0 if all_pass else 2


if __name__ == "__main__":
    sys.exit(main())
