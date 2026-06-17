#!/usr/bin/env python3
"""OPD/QAT capability-curve driver — thin orchestrator over the existing evals.

Given a manifest of checkpoints, this script, **for each checkpoint**:

  1. ensures an OpenAI-v1 serve is reachable — either an existing `base_url`
     you pass through, or a `model_path` it launches `arle serve` for and
     tears down afterwards;
  2. runs `scripts/arle_capability_eval.py` (mmlu, gsm8k) by subprocess
     (optionally multi-seed → seed_<N>/ subdirs);
  3. optionally runs `scripts/arle_swe_pro_eval.py` prepare + generate by
     subprocess (the official Docker `evaluate` stays out of band; see
     docs/opd-capability-curve.md);
  4. collects the per-task metric into `curve.json` (label → metric + seeds);
  5. prints a Δ-vs-baseline table, reusing `scripts/analyze_multi_seed.py`
     for binomial / Wilson CI where seeds are present.

It reimplements no eval logic. MMLU/GSM8K scoring lives in
arle_capability_eval.py; SWE-bench Pro lives in arle_swe_pro_eval.py; CI math
lives in analyze_multi_seed.py. This file is glue: serve lifecycle + manifest
fan-out + a Δ table.

It works against the BASE model today (the baseline reference point). Adapter /
checkpoint points slot in unchanged once the serving blocker in
docs/opd-capability-curve.md is fixed (Option B: each OPD `step_NNNNNN/` becomes
a `model_path` curve point with no driver change).

------------------------------------------------------------------------------
Manifest

A JSON file: a list of checkpoint specs. Each spec is one of:

  - {"label": "...", "base_url": "http://host:port", "model_id": "..."}
      an already-running serve. The driver does not launch/teardown it.
  - {"label": "...", "model_path": "/path/to/dir", "model_id": "..."}
      a model dir the driver serves via `arle serve` then tears down.
  - {"label": "...", "model_path": "...", "adapter_path": "..."}
      reserved for Option A (serve-side adapter load). Today the driver passes
      `--adapter-path` through to `arle serve` IF you also pass
      --serve-supports-adapter; otherwise it errors clearly (the flag does not
      exist on `arle serve` yet — see docs/opd-capability-curve.md).

`model_id` is optional; defaults to the dir basename (matches
infer-api's model_id_from_path) or the label.

The first manifest entry is the baseline by default (override --baseline-label).

------------------------------------------------------------------------------
Examples

  # base-model baseline against an ALREADY-RUNNING serve (cheapest; no launch)
  arle serve --backend cuda --model-path /models/Qwen3-4B --port 8123 &
  cat > /tmp/curve.json <<'JSON'
  [{"label": "base", "base_url": "http://localhost:8123", "model_id": "Qwen3-4B"}]
  JSON
  python scripts/opd_capability_curve.py \
    --manifest /tmp/curve.json \
    --tasks mmlu,gsm8k --n-samples 200 --seeds 0,1,2 \
    --output bench-output/curve-2026xxxx

  # baseline as a single inline checkpoint, driver launches+tears down the serve
  python scripts/opd_capability_curve.py \
    --checkpoint label=base,model_path=/models/Qwen3-4B \
    --tasks mmlu,gsm8k --n-samples 100 --seed 0 \
    --serve-backend cuda \
    --output bench-output/curve-base

  # curve with checkpoints (needs the serving blocker fixed — Option B)
  cat > /tmp/curve.json <<'JSON'
  [
    {"label": "base",       "model_path": "/models/Qwen3-4B"},
    {"label": "opd@2k",     "model_path": "runs/opd/step_002000"},
    {"label": "opd@10k",    "model_path": "runs/opd/step_010000"}
  ]
  JSON
  python scripts/opd_capability_curve.py --manifest /tmp/curve.json \
    --tasks mmlu,gsm8k --n-samples 500 --seeds 0,1,2,3,4 \
    --serve-backend cuda --output bench-output/curve-opd

  # dry-run: print exactly what would happen, launch nothing, eval nothing
  python scripts/opd_capability_curve.py --manifest /tmp/curve.json \
    --tasks mmlu,gsm8k,swe_pro --n-samples 200 --seed 0 --dry-run \
    --output bench-output/curve-dry
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
CAP_EVAL = SCRIPTS_DIR / "arle_capability_eval.py"
SWE_EVAL = SCRIPTS_DIR / "arle_swe_pro_eval.py"
ANALYZE = SCRIPTS_DIR / "analyze_multi_seed.py"

CAP_TASKS = {"mmlu", "gsm8k"}
SWE_TASK = "swe_pro"


# ───────────────────────── checkpoint spec ──────────────────────────


@dataclass
class Checkpoint:
    label: str
    base_url: str | None = None
    model_path: str | None = None
    adapter_path: str | None = None
    model_id: str | None = None

    def resolved_model_id(self) -> str:
        if self.model_id:
            return self.model_id
        if self.model_path:
            return Path(self.model_path).name
        return self.label

    def validate(self) -> None:
        if not self.label:
            raise ValueError("checkpoint missing 'label'")
        if not self.base_url and not self.model_path:
            raise ValueError(
                f"checkpoint {self.label!r} needs either 'base_url' (existing serve) "
                "or 'model_path' (driver launches arle serve)"
            )
        if self.base_url and self.model_path:
            raise ValueError(
                f"checkpoint {self.label!r} has both 'base_url' and 'model_path'; "
                "pick one (base_url = reuse a running serve; model_path = launch one)"
            )


def _parse_inline_checkpoint(raw: str) -> Checkpoint:
    """Parse `label=base,model_path=/x,base_url=...,adapter_path=...,model_id=...`."""
    fields: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"inline checkpoint field {part!r} must be key=value")
        key, value = part.split("=", 1)
        fields[key.strip()] = value.strip()
    allowed = {"label", "base_url", "model_path", "adapter_path", "model_id"}
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"inline checkpoint unknown keys {sorted(unknown)}; allowed {sorted(allowed)}")
    cp = Checkpoint(**fields)  # type: ignore[arg-type]
    cp.validate()
    return cp


def load_checkpoints(args: argparse.Namespace) -> list[Checkpoint]:
    checkpoints: list[Checkpoint] = []
    if args.manifest:
        raw = json.loads(Path(args.manifest).read_text())
        if not isinstance(raw, list):
            raise ValueError("manifest must be a JSON list of checkpoint objects")
        for entry in raw:
            cp = Checkpoint(
                label=entry.get("label", ""),
                base_url=entry.get("base_url"),
                model_path=entry.get("model_path"),
                adapter_path=entry.get("adapter_path"),
                model_id=entry.get("model_id"),
            )
            cp.validate()
            checkpoints.append(cp)
    for raw in args.checkpoint or []:
        checkpoints.append(_parse_inline_checkpoint(raw))
    if not checkpoints:
        raise ValueError("no checkpoints; pass --manifest FILE or one or more --checkpoint k=v,...")
    labels = [c.label for c in checkpoints]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate checkpoint labels: {labels}")
    return checkpoints


# ───────────────────────── serve lifecycle ──────────────────────────


@dataclass
class ServeSession:
    base_url: str
    proc: subprocess.Popen | None = None
    log_path: Path | None = None

    def teardown(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/v1/models"


def serve_is_ready(base_url: str, timeout: float = 5.0) -> bool:
    """GET /v1/models — a 200 from the in-process single-model serve is ready."""
    try:
        req = urllib.request.Request(_models_url(base_url), method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


def launch_serve(cp: Checkpoint, args: argparse.Namespace, output_dir: Path) -> ServeSession:
    """Launch `arle serve` for a model_path checkpoint; poll /v1/models until ready."""
    arle = args.arle_bin or shutil.which("arle") or str(REPO_ROOT / "target" / "release" / "arle")
    port = args.serve_port
    base_url = f"http://{args.serve_host}:{port}"
    cmd = [
        arle, "serve",
        "--backend", args.serve_backend,
        "--model-path", cp.model_path,  # type: ignore[list-item]
        "--port", str(port),
        "--bind", args.serve_host,
    ]
    # Size the serve token ceiling so it never silently caps the eval generation
    # below --gsm8k-max-tokens (the reasoning gen budget). Auto = prompt
    # headroom (1024) + gen budget.
    serve_total = args.serve_max_total_tokens or (1024 + args.gsm8k_max_tokens)
    cmd += ["--max-total-tokens", str(serve_total), "--max-prompt-tokens", str(serve_total)]
    if cp.adapter_path:
        if not args.serve_supports_adapter:
            raise RuntimeError(
                f"checkpoint {cp.label!r} sets adapter_path but `arle serve` has no "
                "--adapter-path flag yet (serving blocker, Option A — see "
                "docs/opd-capability-curve.md). Either merge the adapter into a full "
                "checkpoint (Option B) and use model_path, or pass "
                "--serve-supports-adapter once the flag lands."
            )
        cmd += ["--adapter-path", cp.adapter_path]
    cmd += args.serve_extra_arg

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "serve.log"
    log_fh = log_path.open("w", encoding="utf-8")
    print(f"[curve] launch: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, cwd=REPO_ROOT)

    deadline = time.time() + args.serve_ready_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"arle serve for {cp.label!r} exited early (code {proc.returncode}); see {log_path}"
            )
        if serve_is_ready(base_url):
            print(f"[curve] serve ready for {cp.label!r} at {base_url}", flush=True)
            return ServeSession(base_url=base_url, proc=proc, log_path=log_path)
        time.sleep(2.0)
    proc.terminate()
    raise RuntimeError(
        f"arle serve for {cp.label!r} not ready within {args.serve_ready_timeout}s; see {log_path}"
    )


# ───────────────────────── eval subprocess calls ──────────────────────────


def _run(cmd: list[str], *, dry_run: bool, cwd: Path | None = None) -> int:
    printable = " ".join(str(c) for c in cmd)
    print(f"[curve] $ {printable}", flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=cwd)
    return completed.returncode


def run_capability_eval(
    cp: Checkpoint, base_url: str, args: argparse.Namespace, cap_tasks: list[str], cp_out: Path
) -> None:
    cap_dir = cp_out / "capability"
    cmd = [
        sys.executable, str(CAP_EVAL),
        "--backend", "arle",
        "--base-url", base_url,
        "--model-id", cp.resolved_model_id(),
        "--tasks", ",".join(cap_tasks),
        "--n-samples", str(args.n_samples),
        "--output", str(cap_dir),
    ]
    if args.seeds:
        cmd += ["--seeds", args.seeds]
    elif args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.gsm8k_shots is not None:
        cmd += ["--gsm8k-shots", str(args.gsm8k_shots)]
    cmd += ["--gsm8k-max-tokens", str(args.gsm8k_max_tokens)]
    rc = _run(cmd, dry_run=args.dry_run)
    if rc != 0 and not args.dry_run:
        raise RuntimeError(f"capability eval failed for {cp.label!r} (exit {rc})")


def run_swe_pro(cp: Checkpoint, base_url: str, args: argparse.Namespace, cp_out: Path) -> None:
    swe_dir = cp_out / "swe_pro"
    prepare = [
        sys.executable, str(SWE_EVAL), "prepare",
        "--output", str(swe_dir),
        "--limit", str(args.swe_limit),
    ]
    if args.seed is not None:
        prepare += ["--seed", str(args.seed)]
    rc = _run(prepare, dry_run=args.dry_run)
    if rc != 0 and not args.dry_run:
        raise RuntimeError(f"swe_pro prepare failed for {cp.label!r} (exit {rc})")
    generate = [
        sys.executable, str(SWE_EVAL), "generate",
        "--output", str(swe_dir),
        "--base-url", base_url,
        "--model-id", cp.resolved_model_id(),
        "--limit", str(args.swe_limit),
    ]
    if args.seed is not None:
        generate += ["--seed", str(args.seed)]
    rc = _run(generate, dry_run=args.dry_run)
    if rc != 0 and not args.dry_run:
        raise RuntimeError(f"swe_pro generate failed for {cp.label!r} (exit {rc})")
    print(
        "[curve] swe_pro: patches generated; the official Docker `evaluate` phase is "
        "out of band (needs docker/modal + prebuilt images). Run:\n"
        f"    python {SWE_EVAL} evaluate --output {swe_dir} "
        "--eval-repo /private/tmp/SWE-bench_Pro-os --use-local-docker --docker-platform linux/amd64",
        flush=True,
    )


# ───────────────────────── metric collection ──────────────────────────


def _read_cap_metrics(cap_dir: Path, cap_tasks: list[str]) -> dict[str, Any]:
    """Pull accuracy (+ per-seed when multi-seed) from the eval's summary.json.

    Single-seed: summary.json has tasks[task].accuracy.
    Multi-seed:  summary.json has per_seed[seed][task].accuracy; per-seed
                 summaries live in seed_<N>/summary.json (used by analyze_multi_seed).
    """
    summary_path = cap_dir / "summary.json"
    out: dict[str, Any] = {"summary_path": str(summary_path), "tasks": {}}
    if not summary_path.exists():
        out["status"] = "missing"
        return out
    summary = json.loads(summary_path.read_text())
    out["status"] = "ok"
    if summary.get("mode") == "multi_seed":
        out["mode"] = "multi_seed"
        out["seeds"] = summary.get("seeds")
        for task in cap_tasks:
            per_seed = {
                seed: blob.get(task, {}).get("accuracy")
                for seed, blob in summary.get("per_seed", {}).items()
            }
            accs = [v for v in per_seed.values() if v is not None]
            out["tasks"][task] = {
                "per_seed": per_seed,
                "mean_accuracy": (sum(accs) / len(accs)) if accs else None,
                "n_seeds": len(accs),
            }
    else:
        out["mode"] = "single"
        for task in cap_tasks:
            report = summary.get("tasks", {}).get(task, {})
            out["tasks"][task] = {
                "accuracy": report.get("accuracy"),
                "n_correct": report.get("n_correct"),
                "n_scored": report.get("n_scored"),
                "n_invalid": report.get("n_invalid"),
                "status": report.get("status"),
            }
    return out


def _metric_value(cap_metrics: dict[str, Any], task: str) -> float | None:
    blob = cap_metrics.get("tasks", {}).get(task)
    if not blob:
        return None
    if cap_metrics.get("mode") == "multi_seed":
        return blob.get("mean_accuracy")
    return blob.get("accuracy")


# ───────────────────────── delta table ──────────────────────────


def print_delta_table(curve: dict[str, Any], baseline_label: str, cap_tasks: list[str]) -> None:
    points = curve["points"]
    base = next((p for p in points if p["label"] == baseline_label), None)
    print("\n========== capability curve (Δ vs baseline) ==========", flush=True)
    print(f"baseline = {baseline_label}\n")
    header = ["label"] + [f"{t}" for t in cap_tasks] + [f"Δ{t}(pp)" for t in cap_tasks]
    print("  " + "  ".join(f"{h:>14}" for h in header))
    for p in points:
        row = [p["label"]]
        for t in cap_tasks:
            v = _metric_value(p["capability"], t)
            row.append(f"{v:.4f}" if v is not None else "—")
        for t in cap_tasks:
            v = _metric_value(p["capability"], t)
            bv = _metric_value(base["capability"], t) if base else None
            if v is None or bv is None:
                row.append("—")
            else:
                row.append(f"{100 * (v - bv):+.2f}")
        print("  " + "  ".join(f"{c:>14}" for c in row))


def maybe_run_analyze(curve_out: Path, points: list[dict[str, Any]], baseline_label: str, cap_tasks: list[str], dry_run: bool) -> None:
    """For multi-seed runs, call analyze_multi_seed.py per checkpoint vs the baseline."""
    base = next((p for p in points if p["label"] == baseline_label), None)
    if base is None:
        return
    base_cap = Path(base["capability"]["summary_path"]).parent
    if not list(base_cap.glob("seed_*")) and not dry_run:
        return  # single-seed; analyze_multi_seed needs seed_<N>/ subdirs
    for p in points:
        if p["label"] == baseline_label:
            continue
        cap_dir = Path(p["capability"]["summary_path"]).parent
        for task in cap_tasks:
            cmd = [
                sys.executable, str(ANALYZE),
                str(cap_dir),
                "--paired-vs", str(base_cap),
                "--task", task,
            ]
            print(f"\n[curve] analyze {p['label']} vs {baseline_label} ({task}):", flush=True)
            _run(cmd, dry_run=dry_run)


# ───────────────────────── main ──────────────────────────


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = parser.add_argument_group("checkpoint source")
    src.add_argument("--manifest", type=Path, default=None,
                     help="JSON list of checkpoint specs (see module docstring)")
    src.add_argument("--checkpoint", action="append", default=None,
                     help="inline checkpoint: label=base,model_path=/x[,base_url=,adapter_path=,model_id=] (repeatable)")
    parser.add_argument("--baseline-label", default=None,
                        help="which checkpoint is the Δ baseline (default: first)")

    ev = parser.add_argument_group("eval selection")
    ev.add_argument("--tasks", default="mmlu,gsm8k",
                    help="comma list of: mmlu, gsm8k, swe_pro")
    ev.add_argument("--n-samples", type=int, default=200, help="samples per MMLU/GSM8K task")
    ev.add_argument("--seed", type=int, default=None, help="single seed (mutually exclusive with --seeds)")
    ev.add_argument("--seeds", default=None, help="comma seed list → multi-seed seed_<N>/ + CI")
    ev.add_argument("--gsm8k-shots", type=int, default=None, help="few-shot count for GSM8K")
    ev.add_argument("--gsm8k-max-tokens", type=int, default=2048,
                    help="GSM8K generation budget (reasoning). 2048 covers base-model CoT; "
                         ">=10K for thinking-mode models. Old 256 truncated chains before ####.")
    ev.add_argument("--swe-limit", type=int, default=3, help="SWE-bench Pro instance count when swe_pro is selected")

    sv = parser.add_argument_group("serve lifecycle (model_path checkpoints only)")
    sv.add_argument("--serve-backend", default="cuda", help="arle serve --backend value")
    sv.add_argument("--serve-host", default="127.0.0.1")
    sv.add_argument("--serve-port", type=int, default=8123)
    sv.add_argument("--serve-ready-timeout", type=float, default=600.0,
                    help="seconds to wait for /v1/models to come up")
    sv.add_argument("--serve-max-total-tokens", type=int, default=0,
                    help="arle serve --max-total-tokens ceiling. 0 = auto-size to "
                         "(prompt headroom 1024 + --gsm8k-max-tokens) so the serve never "
                         "silently caps the eval generation below --gsm8k-max-tokens.")
    sv.add_argument("--serve-extra-arg", action="append", default=[],
                    help="extra arg passed verbatim to arle serve (repeatable)")
    sv.add_argument("--serve-supports-adapter", action="store_true",
                    help="assert `arle serve` has --adapter-path (Option A); off by default")
    sv.add_argument("--arle-bin", default=None, help="path to the arle binary (default: PATH / target/release)")

    parser.add_argument("--output", type=Path, required=True, help="curve output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="print every action, launch no serve, run no eval")
    args = parser.parse_args(argv)

    if args.seed is not None and args.seeds:
        print("--seed and --seeds are mutually exclusive", file=sys.stderr)
        return 2

    requested = [t.strip() for t in args.tasks.split(",") if t.strip()]
    cap_tasks = [t for t in requested if t in CAP_TASKS]
    want_swe = SWE_TASK in requested
    unknown = [t for t in requested if t not in CAP_TASKS and t != SWE_TASK]
    if unknown:
        print(f"unknown tasks {unknown}; supported: {sorted(CAP_TASKS) + [SWE_TASK]}", file=sys.stderr)
        return 2
    if not cap_tasks and not want_swe:
        print("no tasks selected", file=sys.stderr)
        return 2

    try:
        checkpoints = load_checkpoints(args)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"checkpoint manifest error: {exc}", file=sys.stderr)
        return 2

    baseline_label = args.baseline_label or checkpoints[0].label
    if baseline_label not in {c.label for c in checkpoints}:
        print(f"--baseline-label {baseline_label!r} not among manifest labels", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    curve: dict[str, Any] = {
        "schema_version": "opd-capability-curve-v1",
        "baseline_label": baseline_label,
        "tasks": requested,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "seeds": args.seeds,
        "dry_run": args.dry_run,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "points": [],
    }

    for cp in checkpoints:
        print(f"\n========== checkpoint: {cp.label} ==========", flush=True)
        cp_out = args.output / cp.label
        cp_out.mkdir(parents=True, exist_ok=True)
        session: ServeSession | None = None
        try:
            if cp.base_url:
                base_url = cp.base_url
                if not args.dry_run and not serve_is_ready(base_url):
                    raise RuntimeError(
                        f"checkpoint {cp.label!r} base_url {base_url} not reachable "
                        "(GET /v1/models failed); start the serve first"
                    )
                print(f"[curve] using existing serve {base_url}", flush=True)
            else:
                if args.dry_run:
                    base_url = f"http://{args.serve_host}:{args.serve_port}"
                    arle = args.arle_bin or shutil.which("arle") or str(REPO_ROOT / "target/release/arle")
                    print(
                        f"[curve] would launch: {arle} serve --backend {args.serve_backend} "
                        f"--model-path {cp.model_path} --port {args.serve_port}",
                        flush=True,
                    )
                else:
                    session = launch_serve(cp, args, cp_out)
                    base_url = session.base_url

            if cap_tasks:
                run_capability_eval(cp, base_url, args, cap_tasks, cp_out)
            if want_swe:
                run_swe_pro(cp, base_url, args, cp_out)
        except (RuntimeError, OSError) as exc:
            print(f"[curve] error on checkpoint {cp.label!r}: {exc}", file=sys.stderr)
            return 1
        finally:
            if session is not None:
                print(f"[curve] teardown serve for {cp.label!r}", flush=True)
                session.teardown()

        cap_metrics = (
            _read_cap_metrics(cp_out / "capability", cap_tasks)
            if cap_tasks and not args.dry_run
            else {"status": "dry_run" if args.dry_run else "skipped",
                  "summary_path": str(cp_out / "capability" / "summary.json"),
                  "tasks": {}}
        )
        curve["points"].append({
            "label": cp.label,
            "model_id": cp.resolved_model_id(),
            "model_path": cp.model_path,
            "base_url": cp.base_url,
            "adapter_path": cp.adapter_path,
            "capability": cap_metrics,
            "swe_pro": {"ran_generate": want_swe, "dir": str(cp_out / "swe_pro")} if want_swe else None,
        })

    curve["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    curve_path = args.output / "curve.json"
    curve_path.write_text(json.dumps(curve, indent=2))
    print(f"\n[curve] wrote {curve_path}", flush=True)

    if cap_tasks:
        print_delta_table(curve, baseline_label, cap_tasks)
        if args.seeds:
            maybe_run_analyze(args.output, curve["points"], baseline_label, cap_tasks, args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
