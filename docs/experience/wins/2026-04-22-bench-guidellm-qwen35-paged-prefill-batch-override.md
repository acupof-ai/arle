# Qwen3.5 paged-prefill batch override — guidellm sweep, cuda, 2026-04-22

## Goal

- Regression-check the `qwen35` runtime change that replaces the trait-default
  per-request paged-prefill fallback with a model-owned
  `forward_prefill_batch_with_pool()` override.

## Hypothesis

- High-concurrency TTFT should improve modestly because `qwen35` no longer
  re-enters the generic per-request fallback path once paged-prefill admission
  has already grouped requests into one scheduler tick.

## Command

```bash
scripts/bench_guidellm.sh cuda \
  --model Qwen/Qwen3.5-4B \
  --processor models/Qwen3.5-4B
```

Status: `pending-remote` (CUDA bench host required).

## Environment

- **Backend:** `cuda`
- **Model:** `Qwen/Qwen3.5-4B`
- **Hardware:** pending remote CUDA host
- **Commit:** pending local commit for this micro-tranche
- **Feature set:** `cargo build --release`
- **Non-default flags / env vars:** none
- **Server launch:** pending remote validation

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh cuda`

## Results — sweep headline table

Pending remote run.

## Problems

- Local environment is not a CUDA bench host, so no in-repo guidellm sweep was
  run before commit.

## Learnings

- `prefill_uses_paged_pool() == true` is not enough by itself; the model also
  needs a model-owned batch override so the scheduler does not fall back to the
  generic per-request paged-prefill loop under load.

## Δ vs baseline

- **Baseline:** first run for this micro-tranche

## Artefacts

- Pending remote run

## Notes

- What changed in the code since baseline: `Qwen35Model` now owns
  `forward_prefill_batch_with_pool()`, batch-allocates paged KV upfront, and
  prepares all per-slot paged-prefill buffers before replaying the canonical
  `prefill_forward_paged()` path
- Suspected cause of any regression: n/a
- Follow-ups: remote CUDA sweep to measure whether the structural override is
  enough or whether `qwen35` still needs a true packed/varlen multi-request
  prefill kernel path
