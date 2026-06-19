#!/usr/bin/env python3
"""OPD/QAT capability-curve driver — thin orchestrator over the existing evals.

Given a manifest of checkpoints, this script, **for each checkpoint**:

  1. ensures an OpenAI-v1 serve is reachable — either an existing `base_url`
     you pass through, or a `model_path` it launches `arle serve` for and
     tears down afterwards;
  2. runs `scripts/arle_capability_eval.py` (mmlu, gsm8k, math500) by subprocess
     (optionally multi-seed → seed_<N>/ subdirs);
  3. optionally runs `scripts/arle_swe_pro_eval.py` prepare + generate by
     subprocess (the official Docker `evaluate` stays out of band; see
     docs/opd-capability-curve.md);
  4. collects the per-task metric into `curve.json` (label → metric + seeds);
  5. prints a Δ-vs-baseline table, reusing `scripts/analyze_multi_seed.py`
     for binomial / Wilson CI where seeds are present.

It reimplements no eval logic. MMLU/GSM8K/MATH-500 scoring lives in
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
    --tasks mmlu,gsm8k,math500 --n-samples 100 --seed 0 \
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
    --tasks math500 --n-samples 500 --math-max-tokens 4096 \
    --serve-backend cuda --output bench-output/curve-opd

  # dry-run: print exactly what would happen, launch nothing, eval nothing
  python scripts/opd_capability_curve.py --manifest /tmp/curve.json \
    --tasks mmlu,gsm8k,swe_pro --n-samples 200 --seed 0 --dry-run \
    --output bench-output/curve-dry
"""

from __future__ import annotations

import argparse
import html
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

CAP_TASKS = {"mmlu", "gsm8k", "math500"}
SWE_TASK = "swe_pro"


# ───────────────────────── checkpoint spec ──────────────────────────


@dataclass
class Checkpoint:
    label: str
    base_url: str | None = None
    model_path: str | None = None
    adapter_path: str | None = None
    model_id: str | None = None
    arm: str | None = None
    training_seed: int | None = None
    delta_vs_arm: str | None = None

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
    allowed = {
        "label",
        "base_url",
        "model_path",
        "adapter_path",
        "model_id",
        "arm",
        "training_seed",
        "delta_vs_arm",
    }
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"inline checkpoint unknown keys {sorted(unknown)}; allowed {sorted(allowed)}")
    kwargs: dict[str, object] = dict(fields)
    seed_raw = fields.get("training_seed")
    if seed_raw is not None:
        try:
            kwargs["training_seed"] = int(seed_raw)
        except ValueError as exc:
            raise ValueError(
                f"inline checkpoint training_seed must be an integer, got {seed_raw!r}"
            ) from exc
    cp = Checkpoint(**kwargs)  # type: ignore[arg-type]
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
                arm=entry.get("arm"),
                training_seed=entry.get("training_seed"),
                delta_vs_arm=entry.get("delta_vs_arm"),
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
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        else:
            self.proc.wait(timeout=0)


def _models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


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
    # below the selected reasoning gen budget. Auto = prompt
    # headroom (1024) + gen budget.
    serve_total = args.serve_max_total_tokens or (1024 + max(args.gsm8k_max_tokens, args.math_max_tokens))
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
    log_fh.close()
    session = ServeSession(base_url=base_url, proc=proc, log_path=log_path)

    deadline = time.time() + args.serve_ready_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            session.teardown()
            raise RuntimeError(
                f"arle serve for {cp.label!r} exited early (code {proc.returncode}); see {log_path}"
            )
        if serve_is_ready(base_url):
            print(f"[curve] serve ready for {cp.label!r} at {base_url}", flush=True)
            return session
        time.sleep(2.0)
    session.teardown()
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
    cmd += ["--math-max-tokens", str(args.math_max_tokens)]
    cmd += ["--request-timeout", str(args.request_timeout)]
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
            per_seed_reports = summary.get("per_seed", {})
            per_seed = {seed: blob.get(task, {}).get("accuracy") for seed, blob in per_seed_reports.items()}
            per_seed_n_scored = {
                seed: blob.get(task, {}).get("n_scored") for seed, blob in per_seed_reports.items()
            }
            accs = [v for v in per_seed.values() if v is not None]
            scored = [v for v in per_seed_n_scored.values() if v is not None]
            out["tasks"][task] = {
                "per_seed": per_seed,
                "per_seed_n_scored": per_seed_n_scored,
                "mean_accuracy": (sum(accs) / len(accs)) if accs else None,
                "n_seeds": len(accs),
                "n_scored": min(scored) if scored else None,
            }
    else:
        out["mode"] = "single"
        for task in cap_tasks:
            report = summary.get("tasks", {}).get(task, {})
            out["tasks"][task] = {
                "accuracy": report.get("accuracy"),
                "ci95": report.get("ci95"),
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


def write_curve_svg(curve: dict[str, Any], task: str, out_path: Path) -> bool:
    points = [
        p for p in curve["points"]
        if _metric_value(p["capability"], task) is not None
    ]
    if not points:
        return False

    width, height = 760, 420
    left, right, top, bottom = 70, 170, 34, 58
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_have_seed = all(p.get("training_seed") is not None for p in points)
    xs_raw = [p["training_seed"] if all_have_seed else i for i, p in enumerate(points)]
    ys_raw = [100.0 * _metric_value(p["capability"], task) for p in points]
    x_min, x_max = min(xs_raw), max(xs_raw)
    y_min, y_max = min(ys_raw), max(ys_raw)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    y_pad = max(1.0, (y_max - y_min) * 0.2)
    y_min = max(0.0, y_min - y_pad)
    y_max = min(100.0, y_max + y_pad)

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    grouped: dict[str, list[tuple[float, float, str]]] = {}
    for idx, (p, x, y) in enumerate(zip(points, xs_raw, ys_raw)):
        name = p.get("arm") or p["label"]
        grouped.setdefault(name, []).append((float(x), y, p["label"]))

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="24" font-family="sans-serif" font-size="16" font-weight="700">OPD capability curve: {html.escape(task)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<text x="{left + plot_w / 2}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="12">{"training seed" if all_have_seed else "checkpoint"}</text>',
        f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" text-anchor="middle" font-family="sans-serif" font-size="12">accuracy (%)</text>',
    ]
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yv = y_min + frac * (y_max - y_min)
        y = sy(yv)
        elems.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        elems.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{yv:.1f}</text>')
    for i, (name, vals) in enumerate(sorted(grouped.items())):
        vals.sort()
        color = colors[i % len(colors)]
        path = " ".join(("M" if j == 0 else "L") + f"{sx(x):.1f},{sy(y):.1f}" for j, (x, y, _) in enumerate(vals))
        elems.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y, label in vals:
            elems.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"><title>{html.escape(label)} {y:.2f}%</title></circle>')
        lx = left + plot_w + 18
        ly = top + 20 + i * 22
        elems.append(f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 18}" y2="{ly - 4}" stroke="{color}" stroke-width="2"/>')
        elems.append(f'<text x="{lx + 24}" y="{ly}" font-family="sans-serif" font-size="12">{html.escape(name)}</text>')
    elems.append("</svg>")
    out_path.write_text("\n".join(elems) + "\n")
    return True


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


def _mean_sigma(values: list[float]) -> tuple[float, float]:
    mu = sum(values) / len(values)
    if len(values) < 2:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return mu, var ** 0.5


def print_training_seed_table(curve: dict[str, Any], cap_tasks: list[str]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for point in curve["points"]:
        arm = point.get("arm")
        if arm is None or point.get("training_seed") is None:
            continue
        grouped.setdefault(arm, []).append(point)
    if not grouped:
        return

    print("\n========== across training seeds ==========", flush=True)
    for task in cap_tasks:
        print(f"\n{task}", flush=True)
        print(f"  {'arm':>28}  {'n':>3}  {'mean':>8}  {'sigma':>8}  {'delta_vs':>24}  {'Δmean(pp)':>10}  {'Δsigma':>8}")
        for arm, points in sorted(grouped.items()):
            by_seed = {
                p["training_seed"]: _metric_value(p["capability"], task)
                for p in points
                if _metric_value(p["capability"], task) is not None
            }
            vals = list(by_seed.values())
            if not vals:
                continue
            mu, sig = _mean_sigma(vals)
            delta_vs = next((p.get("delta_vs_arm") for p in points if p.get("delta_vs_arm")), None)
            delta_mean = delta_sigma = None
            if delta_vs and delta_vs in grouped:
                control = {
                    p["training_seed"]: _metric_value(p["capability"], task)
                    for p in grouped[delta_vs]
                    if _metric_value(p["capability"], task) is not None
                }
                deltas = [by_seed[s] - control[s] for s in sorted(by_seed.keys() & control.keys())]
                if deltas:
                    delta_mean, delta_sigma = _mean_sigma(deltas)
            print(
                f"  {arm:>28}  {len(vals):>3}  {mu:>8.4f}  {sig:>8.4f}  "
                f"{(delta_vs or '—'):>24}  "
                f"{(f'{100 * delta_mean:+.2f}' if delta_mean is not None else '—'):>10}  "
                f"{(f'{100 * delta_sigma:.2f}' if delta_sigma is not None else '—'):>8}",
                flush=True,
            )


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
                    help="comma list of: mmlu, gsm8k, math500, swe_pro")
    ev.add_argument("--n-samples", type=int, default=200, help="samples per selected capability task")
    ev.add_argument("--seed", type=int, default=None, help="single seed (mutually exclusive with --seeds)")
    ev.add_argument("--seeds", default=None, help="comma seed list → multi-seed seed_<N>/ + CI")
    ev.add_argument("--gsm8k-shots", type=int, default=None, help="few-shot count for GSM8K")
    ev.add_argument("--gsm8k-max-tokens", type=int, default=2048,
                    help="GSM8K generation budget (reasoning). 2048 covers base-model CoT; "
                         ">=10K for thinking-mode models. Old 256 truncated chains before ####.")
    ev.add_argument("--math-max-tokens", type=int, default=4096,
                    help="MATH-500 generation budget (reasoning). OPD A/B uses >=4096.")
    ev.add_argument("--request-timeout", type=float, default=1800.0,
                    help="per-sample ARLE HTTP timeout for capability eval subprocesses")
    ev.add_argument("--swe-limit", type=int, default=3, help="SWE-bench Pro instance count when swe_pro is selected")

    sv = parser.add_argument_group("serve lifecycle (model_path checkpoints only)")
    sv.add_argument("--serve-backend", default="cuda", help="arle serve --backend value")
    sv.add_argument("--serve-host", default="127.0.0.1")
    sv.add_argument("--serve-port", type=int, default=8123)
    sv.add_argument("--serve-ready-timeout", type=float, default=600.0,
                    help="seconds to wait for /v1/models to come up")
    sv.add_argument("--serve-max-total-tokens", type=int, default=0,
                    help="arle serve --max-total-tokens ceiling. 0 = auto-size to "
                         "(prompt headroom 1024 + max reasoning max-tokens) so the serve never "
                         "silently caps the eval generation below the selected task budget.")
    sv.add_argument("--serve-extra-arg", action="append", default=[],
                    help="extra arg passed verbatim to arle serve (repeatable)")
    sv.add_argument("--serve-supports-adapter", action="store_true",
                    help="assert `arle serve` has --adapter-path (Option A); off by default")
    sv.add_argument("--arle-bin", default=None, help="path to the arle binary (default: PATH / target/release)")

    parser.add_argument("--output", type=Path, required=True, help="curve output directory")
    parser.add_argument("--skip-existing", action="store_true",
                        help="reuse <checkpoint>/capability/summary.json when present")
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
        "requested_n_samples": args.n_samples,
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
        cap_summary = cp_out / "capability" / "summary.json"
        if cap_tasks and args.skip_existing and cap_summary.exists():
            print(f"[curve] skip existing capability summary for {cp.label!r}: {cap_summary}", flush=True)
        else:
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
            "arm": cp.arm,
            "training_seed": cp.training_seed,
            "delta_vs_arm": cp.delta_vs_arm,
            "capability": cap_metrics,
            "actual_n_scored": {
                task: cap_metrics.get("tasks", {}).get(task, {}).get("n_scored")
                for task in cap_tasks
            },
            "swe_pro": {"ran_generate": want_swe, "dir": str(cp_out / "swe_pro")} if want_swe else None,
        })

    curve["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if cap_tasks:
        svg_path = args.output / "curve.svg"
        if write_curve_svg(curve, cap_tasks[0], svg_path):
            curve["curve_svg"] = str(svg_path)
            print(f"[curve] wrote {svg_path}", flush=True)
    curve_path = args.output / "curve.json"
    curve_path.write_text(json.dumps(curve, indent=2))
    print(f"\n[curve] wrote {curve_path}", flush=True)

    if cap_tasks:
        print_delta_table(curve, baseline_label, cap_tasks)
        print_training_seed_table(curve, cap_tasks)
        if args.seeds:
            maybe_run_analyze(args.output, curve["points"], baseline_label, cap_tasks, args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
