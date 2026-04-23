# CUDA decode SGLang alignment pending remote verification

## Goal

Validate the 2026-04-23 CUDA decode alignment refactor that collapses the
scheduler to one batch per tick, moves decode metadata planning into the
model boundary, and restores a canonical mixed-batch lowering instead of the
old scheduler-owned dual-launch shape.

## Hypothesis

Qwen3 should preserve or improve steady-state decode throughput because mixed
ticks no longer split into separate scheduler launches, while Qwen3.5 should
at minimum preserve correctness and batch-shape stability under the unified
entry contract before any deeper kernel tuning.

## Params

- Label: `cuda-decode-sglang-alignment`
- Planned commands:
  - `scripts/bench_guidellm.sh cuda-decode-sglang-alignment-qwen3`
  - `scripts/bench_guidellm.sh cuda-decode-sglang-alignment-qwen35`
- Planned models:
  - `Qwen/Qwen3-4B`
  - `Qwen/Qwen3.5-4B`
- Feature set: `cargo build --release`

## Env

- Local code change landed on 2026-04-23 in a macOS workspace without a CUDA
  bench host
- Remote GuideLLM sweep is required for the canonical TTFT / ITL / tok-s
  comparison and mixed-tick trace validation

## Results

- Status: `pending-remote`
- Local verification completed:
  - `cargo check -p infer --release --no-default-features --features cuda,no-cuda`
- Local verification attempted but blocked by existing workspace issues:
  - `cargo test -p infer --release --no-default-features --features cuda,no-cuda retract_prefers_ -- --nocapture`
    currently trips an existing `infer/tests/greedy_consistency.rs` struct
    initializer drift (`IncomingRequest` missing `prompt_tokens` /
    `trace_context`) and then fails to link CUDA test targets on this macOS
    host
  - `cargo clippy -p infer --lib --release --no-default-features --features cuda,no-cuda -- -D warnings`
    is still blocked by pre-existing repo-wide `-D warnings` failures outside
    this refactor; touched-file lint regressions from this change were cleaned
    before recording this stub

## Problems

- No remote `guidellm` sweep or CUDA trace has run yet for the new batch
  contract
- Final perf judgment still needs remote confirmation for both `Qwen3` and
  `Qwen3.5`

## Learnings

- Mixed decode correctness depends on sharing one canonical decode-token
  allocation path between decode-only and mixed ticks; otherwise metadata
  positions silently drift even when the scheduler shape looks correct
