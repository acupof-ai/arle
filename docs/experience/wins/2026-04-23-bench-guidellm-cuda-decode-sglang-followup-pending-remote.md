# CUDA decode SGLang follow-up cleanup pending remote verification

## Goal

Validate the 2026-04-23 follow-up cleanup that restores an explicit
decode+prefill split path for models without a true mixed lowering and
unifies decode-metadata sizing across scheduler budgeting and decode-context
allocation.

## Hypothesis

Qwen3 should keep the real mixed path while avoiding mixed-metadata realloc
churn, and Qwen3.5 should recover the prior decode+prefill scheduler shape
without falsely advertising mixed-batch support.

## Params

- Label: `cuda-decode-sglang-followup`
- Planned commands:
  - `scripts/bench_guidellm.sh cuda-decode-sglang-followup-qwen3`
  - `scripts/bench_guidellm.sh cuda-decode-sglang-followup-qwen35`
- Planned models:
  - `Qwen/Qwen3-4B`
  - `Qwen/Qwen3.5-4B`
- Feature set: `cargo build --release`

## Env

- Local code change landed on 2026-04-23 in a macOS workspace without a CUDA
  bench host
- Follow-up to:
  [2026-04-23-bench-guidellm-cuda-decode-sglang-alignment-pending-remote.md](2026-04-23-bench-guidellm-cuda-decode-sglang-alignment-pending-remote.md)
- Remote GuideLLM sweep is required for the canonical TTFT / ITL / tok-s
  comparison and decode+prefill scheduling trace validation

## Results

- Status: `pending-remote`
- Local verification completed:
  - `cargo check -p infer --release --no-default-features --features cuda,no-cuda`

## Problems

- No remote `guidellm` sweep or CUDA trace has run yet for the follow-up
  cleanup
- The scheduler/path-shape comparison against SGLang still needs remote
  confirmation under both `Qwen3` and `Qwen3.5`

## Learnings

- Mixed-batch capability needs to stay an honest model contract: if a model
  only has separate decode and prefill launches, the scheduler should say so
  explicitly instead of routing it through a fake mixed path
