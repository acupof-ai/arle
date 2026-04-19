# Metal GDR Xcode-capture hook — bench (pending-remote)

**Status**: pending-remote
**Scope**: `crates/mlx-sys/src/mlx_metal_capture.mm`,
`crates/mlx-sys/src/mlx_qwen35_model.cpp::qwen35_compiled_step_session`,
`crates/mlx-sys/build.rs`.

## Context

Env-gated `MTLCaptureManager` hook around Qwen3.5 decode step for
`gated_delta_step` attribution (see
`docs/plans/metal-gdr-kernel-xcode-capture.md` §Step 2b). Default-off via
`INFER_CAPTURE_STEP` — hot path cost when unset is one relaxed atomic load
(`g_step_counter.fetch_add` is gated behind an early return when
`cfg.target_step < 0`, so the unset path is a single atomic load of
`target_step` which is cached after first call; on subsequent calls the
config-static-local fast-path is a plain branch).

## Why pending

Change is debug-only instrumentation: the capture code path is dead under
production env (unset `INFER_CAPTURE_STEP`). The hot-path delta is bounded
above by one relaxed atomic load per step, which is below `bench_guidellm`'s
per-run stderr (~0.3%). A regression bench would require:

- CUDA-less Mac with Qwen3.5-4B-MLX-4bit checkout (model weights not local
  in this session; ckl's setup has them on the M4 Max)

**Remote owner**: ckl, on the M4 Max where the Xcode capture will actually
run. The same session that executes the capture should also run
`scripts/bench_guidellm.sh qwen35-capture-regression-check` with env vars
UNSET against the most recent Qwen3.5 Metal baseline
(`docs/experience/wins/2026-04-19-metal-qwen35-final-state.md`) and confirm
Δ decode throughput within ±1% (thermal band; see
`memory/feedback_matched_ab_for_small_bench_effects.md`).

## Acceptance (remote)

- Build clean: `cargo build --release --no-default-features --features metal,no-cuda --bin metal_bench -p infer` (verified locally, this session).
- Default-off regression bench: ≤1% decode tok/s delta vs latest Qwen3.5 Metal baseline.
- Non-empty `.gputrace` written when `MTL_CAPTURE_ENABLED=1 INFER_CAPTURE_STEP=5` is set.

## Local verification (this session)

- Release build passes for `metal_bench` on Mac with `metal,no-cuda`.
- `metal_bench --help` unchanged.
- `metal_bench --model /nonexistent/path` fails cleanly before the FFI
  step_session call (so the capture hook's static-local init does not run
  on cold startup path).

Cross-linked from `docs/plans/metal-gdr-kernel-xcode-capture.md`.
