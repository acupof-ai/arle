# `arle train opd` HF-dir wiring — 2026-05-24

## Goal

- Ship the OPD CLI main path that turns `arle train opd --student-model <dir>`
  from a pending stub into an end-to-end local training entrypoint.

## Context

- Before this change, the CLI path was useful for `--smoke`, but a real
  Hugging Face directory returned a pending-loader error; users had to fall
  back to examples to exercise real Qwen3.5 OPD.
- The 2026-05-24 A1 brief scoped this as the commit-ready OPD CLI axis and
  explicitly left other dirty M-state files alone.
- The same brief came from the docs gap feedback that many optimizations and
  user-facing paths were not being written down.

## What Worked

- `crates/cli/src/train_cli.rs` now routes non-smoke OPD runs through
  `run_opd_from_dirs()`.
- The new CLI path stays self-contained: it directly calls
  `train::qwen35_loader::load_qwen35_from_hf_dir` for teacher/student loading
  and `train::opd::opd_step` for each training step.
- No `infer/src/kv_tier/`, `infer/src/scheduler/`, CUDA scheduler, or GPU run
  state was touched.

## Verification

```bash
NVCC_CCBIN=g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
cargo check -p train --features cuda --no-default-features

NVCC_CCBIN=g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
cargo check -p cli

NVCC_CCBIN=g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
cargo clippy -p train --features cuda --no-default-features --no-deps -- -D warnings

NVCC_CCBIN=g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
cargo clippy -p cli --no-deps -- -D warnings
```

- All commands exited 0.
- The CUDA train checks rebuilt TileLang/CUDA artifacts but did not start
  training, serving, or GPU workloads.
- Existing `infer` warnings still appear while compiling dependencies; the
  target `train` and `cli` packages were clippy-clean under `--no-deps`.

## Bench Status

- Exempt. This is pure CLI wiring for an existing Qwen3.5 loader plus OPD step;
  it does not change numerical output, scheduler behavior, kernels, or serving
  runtime parameters.

## Rule

- Any CLI user path change must land with an experience entry in the same
  tranche; do not commit user-facing wiring without the corresponding docs
  breadcrumb.

