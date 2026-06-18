#!/usr/bin/env bash
# OPD capability A/B eval launcher.
# Reuses scripts/opd_capability_curve.py + scripts/arle_capability_eval.py.
#
# Default is DRY_RUN=1 so local prep can print the exact serve/eval commands
# without touching H20. Set DRY_RUN=0 when the checkpoints are ready.

set -euo pipefail

ROOT="${ARLE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"

OUTPUT="${OUTPUT:-$ROOT/bench-output/opd-ab-math500-curve-$(date +%Y%m%d-%H%M%S)}"
MANIFEST="${MANIFEST:-$OUTPUT/manifest.json}"
DRY_RUN="${DRY_RUN:-1}"
DRY_RUN_ALLOW_UNREADY="${DRY_RUN_ALLOW_UNREADY:-1}"
ALLOW_MISSING="${ALLOW_MISSING:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-50}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-4}"

TRAINING_SEEDS="${TRAINING_SEEDS:-0,1,2}"
FIRST_TRAINING_SEED="${TRAINING_SEEDS%%,*}"
BASELINE_LABEL="${BASELINE_LABEL:-baseline-forward-kl-greedy-s${FIRST_TRAINING_SEED}}"
N_SAMPLES="${N_SAMPLES:-500}"
TASKS="${TASKS:-math500}"
MATH_MAX_TOKENS="${MATH_MAX_TOKENS:-4096}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}"
SERVE_BACKEND="${SERVE_BACKEND:-cuda}"
SERVE_PORT="${SERVE_PORT:-8123}"
SERVE_READY_TIMEOUT="${SERVE_READY_TIMEOUT:-900}"
ARLE_BIN="${ARLE_BIN:-$ROOT/target/release/arle}"

if [[ ! "$EVAL_CUDA_VISIBLE_DEVICES" =~ ^[4-7](,[4-7])*$ ]]; then
  echo "[opd-ab-eval] EVAL_CUDA_VISIBLE_DEVICES must only contain GPUs 4-7, got: $EVAL_CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

mkdir -p "$(dirname "$MANIFEST")"
python3 - <<PY
import json
import os
from pathlib import Path

seeds = [s.strip() for s in "$TRAINING_SEEDS".split(",") if s.strip()]
if len(seeds) < 3:
    raise SystemExit("TRAINING_SEEDS must list at least 3 OPD training seeds")
step = int("$CHECKPOINT_STEP")
step_dir = f"step_{step:06d}"
allow_missing = "$ALLOW_MISSING" == "1"
dry_run = "$DRY_RUN" == "1"
allow_unready = dry_run and "$DRY_RUN_ALLOW_UNREADY" == "1"
cells = [
    ("baseline", "baseline-forward-kl-greedy", "BASELINE_MODEL_PATH", None),
    ("dense-forward-kl-control", "dense-forward-kl-control", "DENSE_FORWARD_KL_MODEL_PATH", "baseline"),
    ("reverse-kl", "reverse-kl-greedy", "REVERSE_KL_MODEL_PATH", "baseline"),
    ("beta-jsd-0.3", "beta-jsd-0.3-greedy", "BETA_JSD_03_MODEL_PATH", "dense-forward-kl-control"),
    ("beta-jsd-0.5", "beta-jsd-0.5-greedy", "BETA_JSD_05_MODEL_PATH", "dense-forward-kl-control"),
    ("stochastic-rollout", "forward-kl-stochastic-t0.9", "STOCHASTIC_MODEL_PATH", "baseline"),
]
manifest = []
missing = []
for seed in seeds:
    for arm, label, env_prefix, delta_vs_arm in cells:
        path_env = f"{env_prefix}_S{seed}"
        run_env = f"{env_prefix.replace('MODEL_PATH', 'RUN_ROOT')}_S{seed}"
        model_path = os.environ.get(path_env)
        if not model_path and os.environ.get(run_env):
            model_path = str(Path(os.environ[run_env]) / "checkpoints" / step_dir)
        if not model_path:
            message = f"set {path_env} or {run_env}"
            if allow_missing:
                missing.append(message)
                continue
            raise SystemExit(message)
        if not Path(model_path).exists() and not allow_unready:
            message = f"missing checkpoint {model_path} ({path_env} or {run_env})"
            if allow_missing:
                missing.append(message)
                continue
            raise SystemExit(message)
        if not Path(model_path).exists():
            missing.append(f"planned unready checkpoint {model_path} ({path_env} or {run_env})")
        entry = {
            "label": f"{label}-s{seed}",
            "model_path": model_path,
            "arm": arm,
            "training_seed": int(seed),
        }
        if delta_vs_arm:
            entry["delta_vs_arm"] = delta_vs_arm
        manifest.append(entry)
if not manifest:
    raise SystemExit("no ready checkpoints; manifest would be empty")
Path("$MANIFEST").write_text(json.dumps(manifest, indent=2) + "\n")
print("$MANIFEST")
if missing:
    print("[opd-ab-eval] skipped missing checkpoints:")
    for item in missing:
        print(f"  - {item}")
PY

args=(
  --manifest "$MANIFEST"
  --baseline-label "$BASELINE_LABEL"
  --tasks "$TASKS"
  --n-samples "$N_SAMPLES"
  --math-max-tokens "$MATH_MAX_TOKENS"
  --request-timeout "$REQUEST_TIMEOUT"
  --serve-backend "$SERVE_BACKEND"
  --serve-port "$SERVE_PORT"
  --serve-ready-timeout "$SERVE_READY_TIMEOUT"
  --arle-bin "$ARLE_BIN"
  --output "$OUTPUT"
)

if [[ "$SKIP_EXISTING" == "1" ]]; then
  args+=(--skip-existing)
fi

if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi

printf '[opd-ab-eval] command: CUDA_VISIBLE_DEVICES=%q python3 scripts/opd_capability_curve.py' "$EVAL_CUDA_VISIBLE_DEVICES"
printf ' %q' "${args[@]}"
printf '\n'

CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES" \
python3 scripts/opd_capability_curve.py "${args[@]}"
