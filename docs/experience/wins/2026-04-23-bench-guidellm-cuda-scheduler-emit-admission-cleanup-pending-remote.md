# CUDA scheduler emit/admission cleanup pending remote verification

## Goal

Validate the 2026-04-23 CUDA scheduler cleanup that:

- moves waiting-request normalization to scheduler ingress,
- removes the global emit-gate stall from the step loop, and
- unifies admission/decode headroom around one clipped request estimate.

## Hypothesis

The scheduler should stay closer to current SGLang behavior under mixed
decode/prefill pressure because emit-gated requests no longer stall unrelated
slots, waiting-queue pressure is computed from real prompt lengths, and page
reservation is less pessimistic for long-tail `max_tokens`.

## Params

- Label: `cuda-scheduler-emit-admission-cleanup`
- Planned commands:
  - `scripts/bench_guidellm.sh cuda-scheduler-emit-admission-cleanup-qwen3`
  - `scripts/bench_guidellm.sh cuda-scheduler-emit-admission-cleanup-qwen35`
- Planned models:
  - `Qwen/Qwen3-4B`
  - `Qwen/Qwen3.5-4B`
- Feature set: `cargo build --release`

## Env

- Local change date: `2026-04-23`
- Local machine: macOS workspace without a CUDA bench host
- Follow-up to:
  - [2026-04-23-bench-guidellm-cuda-decode-sglang-alignment-pending-remote.md](2026-04-23-bench-guidellm-cuda-decode-sglang-alignment-pending-remote.md)
  - [2026-04-23-bench-guidellm-cuda-decode-sglang-followup-pending-remote.md](2026-04-23-bench-guidellm-cuda-decode-sglang-followup-pending-remote.md)

## Results

- Status: `pending-remote`
- Local verification completed:
  - `cargo check -p infer --release --no-default-features --features cuda,no-cuda`
- Local verification attempted but blocked by host limitations:
  - `cargo test -p infer --lib --release --no-default-features --features cuda,no-cuda ...`
    still fails to link CUDA symbols on this macOS host (`/usr/local/cuda/lib64/stubs`
    unavailable, unresolved CUDA kernels at link time)

## Problems

- No remote `guidellm` sweep or CUDA trace has run yet for the scheduler
  cleanup, so TTFT / ITL / tok-s impact is still pending.
- This workspace cannot validate CUDA-linked unit tests locally.

## Learnings

- Treating emit gating as a per-request runnable-state gate is cleaner than a
  whole-loop barrier and matches SGLang's asynchronous output handling better.
- If the waiting queue stores raw prompts, prefix-cache pressure accounting is
  necessarily wrong for any request that has not been tokenized yet.
- One shared request-size estimate is easier to reason about than mixing
  full-tail reservation in one path and clipped estimates in another.
