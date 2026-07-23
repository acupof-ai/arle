#!/usr/bin/env python3
"""Comfort-band corpus filter — the "27B-profile → filter" of task #3.

Selects the subset of an agent-OPD corpus that is trainable for dapo/GRPO on a
single 96 GB GPU: tasks the student solves at an INTERMEDIATE rate (within-group
reward variance → non-zero advantage) AND whose rollouts fit under the masked-CE
writeback cap (`--max-update-seq`, default 23K; the 27B LoRA backward OOMs ~30K).

Why this exists (docs/experience/errors/2026-07-22-agent-opd-dapo-null-gradient-
trajectory-vram-wall.md): `hard` gives variance but ~30K-token trajectories that
the writeback SKIPS → null gradient; `medium` gives short trajectories the 27B
always passes → zero variance. Neither difficulty knob alone lands in the band;
the band must be measured per-task, not guessed. Quantized KV does NOT help — the
rollout KV pool is released before the writeback, so its dtype is irrelevant.

Pipeline (single 96 GB box, e.g. one H20):
  1. stage    python3 scripts/gen_agent_opd_tasks.py --out C --difficulty hard --self-check
  2. profile  arle train agent-opd --dataset C/tasks_train.jsonl --staged-root C/staged \\
                --samples-per-prompt 8 --rollout-temperature 1.0 --rounds 1 --eval-every 0 \\
                --task-selection false --metrics-out C/profile.jsonl ...
  3. filter   python3 scripts/comfort_band.py --metrics C/profile.jsonl --corpus C --out C-band
  4. rerun    arle train agent-opd --dataset C-band/tasks_train.jsonl --staged-root C-band/staged ...

The filter reads the per-group rows the training loop already emits to
`metrics.jsonl` (`kind=group`: task_id, rewards, prompt_tokens, completion_tokens),
so it needs no runtime change and no GPU — only a completed profile run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
from pathlib import Path


class TaskProfile:
    """Per-task aggregate over every profiled group (rewards + trajectory tokens)."""

    __slots__ = ("rewards", "prompt_tokens", "completion_tokens")

    def __init__(self) -> None:
        self.rewards: list[float] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def n(self) -> int:
        return len(self.rewards)

    def pass_rate(self, threshold: float) -> float:
        return sum(r >= threshold for r in self.rewards) / max(self.n, 1)

    def reward_std(self) -> float:
        return statistics.pstdev(self.rewards) if self.n > 1 else 0.0

    def avg_traj_tokens(self) -> float:
        # Per-sample trajectory length = (prompt + completion) / n. The writeback
        # cap is per-trajectory; avg is the proxy (metrics dumps the group sum).
        return (self.prompt_tokens + self.completion_tokens) / max(self.n, 1)


def load_profiles(metrics_path: Path) -> dict[str, TaskProfile]:
    """Aggregate `kind=group` rows from a training/profile metrics.jsonl by task."""
    profiles: dict[str, TaskProfile] = {}
    with metrics_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("kind") != "group":
                continue
            p = profiles.setdefault(row["task_id"], TaskProfile())
            p.rewards.extend(float(r) for r in row.get("rewards", []))
            p.prompt_tokens += int(row.get("prompt_tokens", 0))
            p.completion_tokens += int(row.get("completion_tokens", 0))
    return profiles


def classify(p: TaskProfile, args: argparse.Namespace) -> str:
    """'keep' or the reason it falls outside the comfort band."""
    if p.n < args.min_samples:
        return f"too-few-samples({p.n})"
    if p.avg_traj_tokens() > args.max_seq:
        return f"too-long({p.avg_traj_tokens():.0f}>{args.max_seq})"
    rate = p.pass_rate(args.pass_threshold)
    if rate < args.pass_lo:
        return f"too-hard(pass={rate:.2f})"
    if rate > args.pass_hi:
        return f"too-easy(pass={rate:.2f})"
    return "keep"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def write_corpus(corpus: Path, out: Path, keep_ids: set[str]) -> tuple[int, int]:
    """Emit the filtered train split + its staged trees; copy eval split as-is."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "staged").mkdir(exist_ok=True)

    train = read_jsonl(corpus / "tasks_train.jsonl")
    kept = [r for r in train if r["instance_id"] in keep_ids]
    (out / "tasks_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))

    for r in kept:
        src = corpus / "staged" / r["instance_id"]
        if src.is_dir():
            shutil.copytree(src, out / "staged" / r["instance_id"], dirs_exist_ok=True)

    # Eval split is the held-out generalization gate — copy unchanged (+ its staged).
    eval_src = corpus / "tasks_eval.jsonl"
    n_eval = 0
    if eval_src.is_file():
        eval_rows = read_jsonl(eval_src)
        n_eval = len(eval_rows)
        shutil.copy2(eval_src, out / "tasks_eval.jsonl")
        for r in eval_rows:
            src = corpus / "staged" / r["instance_id"]
            if src.is_dir():
                shutil.copytree(src, out / "staged" / r["instance_id"], dirs_exist_ok=True)
    return len(kept), n_eval


def run(args: argparse.Namespace) -> int:
    profiles = load_profiles(args.metrics)
    if not profiles:
        print(f"error: no kind=group rows in {args.metrics}", file=sys.stderr)
        return 1

    verdicts = {tid: classify(p, args) for tid, p in profiles.items()}
    keep_ids = {tid for tid, v in verdicts.items() if v == "keep"}

    # Min-tests granularity floor: dense reward = passed/|fail_to_pass|, so at
    # |fail_to_pass| < min_tests it can only reach {0,1} → binary, and the group
    # collapses to zero-variance far more often. The count lives in the corpus
    # jsonl, not the profile rows classify sees, so filter it here.
    if args.min_tests > 1:
        low_gran = {r["instance_id"] for r in read_jsonl(args.corpus / "tasks_train.jsonl")
                    if len(r.get("fail_to_pass") or []) < args.min_tests}
        dropped = keep_ids & low_gran
        keep_ids -= low_gran
        if dropped:
            print(f"[comfort-band] dropped {len(dropped)} low-granularity tasks "
                  f"(|fail_to_pass| < {args.min_tests}: dense reward = binary)")

    print(f"[comfort-band] profiled {len(profiles)} tasks → kept {len(keep_ids)} "
          f"(pass∈[{args.pass_lo},{args.pass_hi}], avg_traj≤{args.max_seq}, tests≥{args.min_tests})")
    for tid, p in sorted(profiles.items()):
        mark = "KEEP" if verdicts[tid] == "keep" else f"drop:{verdicts[tid]}"
        print(f"  {tid:32s} pass={p.pass_rate(args.pass_threshold):.2f} "
              f"std={p.reward_std():.2f} avg_traj={p.avg_traj_tokens():6.0f} n={p.n}  {mark}")

    if not keep_ids:
        print("[comfort-band] WARNING: empty band — no task is both intermediate-difficulty "
              "AND short enough. Need harder short-context bugs, or raise --max-seq only if the "
              "writeback can take it (it usually can't; see the errors entry).", file=sys.stderr)
        return 2

    n_train, n_eval = write_corpus(args.corpus, args.out, keep_ids)
    print(f"[comfort-band] wrote {args.out}: {n_train} train (band) + {n_eval} eval (held-out, unchanged)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", type=Path, help="profile run's metrics.jsonl (kind=group rows)")
    ap.add_argument("--corpus", type=Path, help="staged corpus dir (tasks_train.jsonl + staged/)")
    ap.add_argument("--out", type=Path, help="output dir for the filtered comfort-band corpus")
    ap.add_argument("--pass-lo", type=float, default=0.2, help="min pass rate (below = too hard)")
    ap.add_argument("--pass-hi", type=float, default=0.8, help="max pass rate (above = too easy)")
    ap.add_argument("--pass-threshold", type=float, default=1.0, help="reward counted as a pass")
    ap.add_argument("--max-seq", type=int, default=22000,
                    help="max avg trajectory tokens (< the writeback --max-update-seq, default 23K)")
    ap.add_argument("--min-samples", type=int, default=4, help="min profiled samples for a verdict")
    ap.add_argument("--min-tests", type=int, default=2,
                    help="min |fail_to_pass| tests/task; below this dense reward degenerates to binary")
    ap.add_argument("--self-check", action="store_true", help="run the built-in correctness gate and exit")
    return ap


def self_check() -> int:
    """Synthetic profile + corpus → the filter must keep only the comfort-band task."""
    ap = build_parser()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        corpus, out = td / "c", td / "band"
        (corpus / "staged").mkdir(parents=True)
        # instance_id -> fail_to_pass tests. onetest is band-difficulty + short but
        # has only 1 test, so the granularity floor must drop it despite passing the
        # pass/length filters (dense reward would be binary there).
        rows = {
            "arle__easy": ["t::1", "t::2"],
            "arle__band": ["t::1", "t::2"],
            "arle__hard": ["t::1", "t::2"],
            "arle__long": ["t::1", "t::2"],
            "arle__onetest": ["t::1"],
        }
        ids = list(rows)
        (corpus / "tasks_train.jsonl").write_text(
            "".join(json.dumps({"instance_id": i, "problem_statement": "p", "fail_to_pass": rows[i]}) + "\n"
                    for i in ids))
        (corpus / "tasks_eval.jsonl").write_text(json.dumps({"instance_id": "arle__ev"}) + "\n")
        for i in ids + ["arle__ev"]:
            (corpus / "staged" / i).mkdir()
            (corpus / "staged" / i / "mod.py").write_text("x = 1\n")
        # samples/task: easy=all pass, band=mixed short, hard=all fail, long=mixed but 30K,
        # onetest=band-mixed but 1 test → dropped by the granularity floor.
        metrics = [
            {"kind": "group", "task_id": "arle__easy", "rewards": [1, 1, 1, 1], "prompt_tokens": 4000, "completion_tokens": 400},
            {"kind": "group", "task_id": "arle__band", "rewards": [1, 1, 0, 0.5], "prompt_tokens": 8000, "completion_tokens": 800},
            {"kind": "group", "task_id": "arle__hard", "rewards": [0, 0, 0, 0], "prompt_tokens": 6000, "completion_tokens": 600},
            {"kind": "group", "task_id": "arle__long", "rewards": [1, 1, 0, 0], "prompt_tokens": 118000, "completion_tokens": 4000},
            {"kind": "group", "task_id": "arle__onetest", "rewards": [1, 1, 0, 0.5], "prompt_tokens": 8000, "completion_tokens": 800},
            {"kind": "other"},  # ignored
        ]
        mpath = td / "profile.jsonl"
        mpath.write_text("".join(json.dumps(m) + "\n" for m in metrics))
        args = ap.parse_args(["--metrics", str(mpath), "--corpus", str(corpus), "--out", str(out)])
        rc = run(args)
        assert rc == 0, f"run failed rc={rc}"
        kept = {r["instance_id"] for r in read_jsonl(out / "tasks_train.jsonl")}
        assert kept == {"arle__band"}, f"expected only arle__band, got {kept}"
        assert (out / "staged" / "arle__band" / "mod.py").is_file(), "band staged tree not copied"
        assert (out / "tasks_eval.jsonl").is_file(), "eval split not copied"
        assert {r["instance_id"] for r in read_jsonl(out / "tasks_eval.jsonl")} == {"arle__ev"}
        assert (out / "staged" / "arle__ev").is_dir(), "eval staged tree not copied"
    print("self-check OK: keeps comfort-band, drops easy/hard/long/onetest(1-test), copies eval unchanged")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.self_check:
        return self_check()
    for req in ("metrics", "corpus", "out"):
        if getattr(args, req) is None:
            print(f"error: --{req} is required (or pass --self-check)", file=sys.stderr)
            return 2
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
