# Plan — OPD training on the Metal backend (Apple Silicon)

> Status: Active — 2026-07-06 · Driver: ckl (from the OPD review: "P2 给 OPD
> 训练加 Metal 后端"). Scoped as a plan only; no code this session — needs
> Apple-Silicon iteration, not the H20 pod.

## Context / problem

`build_opd_store` (`crates/cli/src/train_cli.rs`) resolves the OPD training math
backend to **CUDA or CPU only** — there is no Metal arm. But Metal is ARLE's
canonical *serving* backend (Qwen3.6-35B-A3B-4bit), and the user's primary dev
machine is Apple Silicon. Consequence: every meaningful OPD run is
`pending-remote` (8×H20 pod); the Mac can only exercise the tiny-config CPU
path. This blocks local OPD iteration entirely.

## What already exists (verified)

- `autograd::backend_metal::MetalBackend` implements the `Backend` trait
  (`crates/autograd/src/backend_metal.rs:65`) — the training forward/backward/
  optimizer math **already has a Metal implementation**. `build_opd_store` just
  never selects it.
- The `train` crate builds under `--features metal` (autograd/metal forwarded).

## The real gap (decomposed)

1. **Backend selection** — add a Metal arm to `build_opd_store` +
   `OpdBackendArg::Metal`. Low effort. But by itself it only gets the *math* on
   Metal, not a runnable end-to-end OPD step.
2. **Rollout path** — `InferStudent` (fast infer-engine rollout) is
   **CUDA-only** (`crates/train/src/infer_student.rs`, `#[cfg(feature="cuda")]`).
   On Metal the step must fall back to `student_rollout_only` (the train-crate
   hand-written decode, opd.rs). That path exists and is backend-neutral, so a
   Metal OPD step is reachable — but at the O(n²) decode cost the CUDA path
   moved away from (the 4.99× P4 win doesn't apply on Metal until an
   infer-metal rollout bridge exists).
3. **Teacher forward** — `InProcessTeacher` (same-store `Qwen35Model.forward`)
   is backend-neutral and works on Metal. `InferTeacher` (LoadedInferenceEngine
   device-ptr bridge) is CUDA-only → Metal uses in-process teacher only.
4. **MoE / hybrid kernels** — Qwen3.6-35B-A3B is MoE; confirm the autograd Metal
   backend covers the ops the student LoRA backward needs (grouped GEMM, gather).
   Likely the first real wall; probe with the tiny-config on Metal first.

## Proposed sequencing

- **S1** `OpdBackendArg::Metal` + `build_opd_store` Metal arm; wire `--backend
  metal`. Gate the CUDA-only `InferStudent`/`InferTeacher` cleanly so Metal
  routes to `student_rollout_only` + `InProcessTeacher`.
- **S2** Tiny-config (`INFER_TEST_MODEL_PATH` small model) OPD smoke on Metal —
  the first end-to-end `arle train opd --backend metal --smoke`-scale run. This
  is the mid-band gate: does a Metal OPD step complete at all?
- **S3** Real Qwen3.5-family student OPD step on Metal; measure step wall-clock
  vs the CUDA path (expect much slower rollout — that's the known cost).
- **S4** (stretch) infer-metal rollout bridge so Metal gets a fast rollout too.

## Verification

`cargo test -p cli --features metal,no-cuda`; `arle train opd --backend metal
--smoke`; then a tiny-config real-weights step on Apple Silicon. Bench entry per
§Benchmarks once S3 produces a number.

## Risks

- MoE backward op coverage on the autograd Metal backend (S1/S2 may surface a
  `todo!`/unimplemented op) — probe early with the tiny config.
- Metal rollout at O(n²) may make real-model OPD steps impractically slow until
  S4; S3 measures whether it's usable for local iteration at all.
