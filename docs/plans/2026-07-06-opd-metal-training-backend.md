# Plan — OPD training on the Metal backend

> Status: Active — 2026-07-06 · Driver: ckl (OPD review P2). Plan only; needs
> Apple-Silicon iteration, not the pod.

**Verdict:** reachable but rollout-slow. `MetalBackend` already implements the
autograd `Backend` trait — the training math runs on Metal today; the gap is
that the fast rollout path is CUDA-only.

## Problem
`build_opd_store` (`train_cli.rs`) resolves CUDA or CPU only — no Metal arm. So
every real OPD run is `pending-remote`; the Mac can only run the tiny-config CPU
path.

## Gaps
1. **Backend selection** — add `OpdBackendArg::Metal` + a Metal arm to
   `build_opd_store`. Low effort; gets the math on Metal, not a runnable step.
2. **Rollout** — `InferStudent` is CUDA-only; Metal falls back to
   `student_rollout_only` (backend-neutral but O(n²), loses the P4 4.99× win).
3. **Teacher** — `InProcessTeacher` works on Metal; `InferTeacher` is CUDA-only.
4. **MoE ops** — confirm the autograd Metal backend covers grouped GEMM / gather
   for the Qwen3.6-35B-A3B student backward. Likely the first wall.

## Sequencing
- **S1** `--backend metal` wired; CUDA-only `InferStudent`/`InferTeacher` gated
  off → `student_rollout_only` + `InProcessTeacher`.
- **S2** tiny-config OPD smoke on Metal — does a step complete?
- **S3** real Qwen3.5-family step; measure wall-clock (expect slow rollout).
- **S4** (stretch) infer-metal rollout bridge.

## Verify
`cargo test -p cli --features metal,no-cuda`; `arle train opd --backend metal
--smoke`; then a tiny real-weights step on Apple Silicon.

## Risk
MoE backward op coverage (probe early); O(n²) Metal rollout may make real steps
too slow until S4.
