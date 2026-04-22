# Qwen3/Qwen3.5 model-side unification follow-up — guidellm sweep, cuda, 2026-04-22

## Goal

- Record the pending remote CUDA sweep for the model-side cleanup that unified
  scheduler/admission behavior across `Qwen3` and `Qwen3.5`, while making the
  remaining `Qwen3.5` packed-prefill boundary explicit.

## Hypothesis

- The shipped changes should remove the old `Qwen3.5` contiguous-prefill
  fallback and reduce scheduler-side divergence from `Qwen3`.
- This entry does **not** claim that `Qwen3.5` already matches `Qwen3`'s packed
  batched paged-prefill implementation shape; that remains the main model-side
  boundary to validate and, if needed, improve next.

## Command

```bash
scripts/bench_guidellm.sh cuda-qwen35-model-unification \
  --target http://<remote-host>:8000 \
  --model Qwen/Qwen3.5-4B \
  --processor models/Qwen3.5-4B \
  --trace-interval-ms 1000
```

Invoked via: `scripts/bench_guidellm.sh cuda-qwen35-model-unification [--target URL] [--model NAME] [--processor PATH] [--trace-interval-ms N]`

## Environment

- **Backend:** `cuda`
- **Model:** `Qwen/Qwen3.5-4B`
- **Hardware:** `pending-remote`
- **Commit:** `pending-remote`
- **Feature set:** `cargo build --release --no-default-features --features cuda`
- **Non-default flags / env vars:** `pending-remote`
- **Server launch:** `pending-remote`

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

Pending remote CUDA run.

## Problems

- Not run locally: this tranche needs a remote CUDA machine to make any
  throughput/TTFT claim.

## Learnings

- The codebase should now be described as:
  - scheduler/admission/page sizing largely unified across `Qwen3` and `Qwen3.5`
  - `Qwen3.5` shipped on paged prefill, with its own batch override
  - `Qwen3.5` **not yet** on the same packed varlen batched paged-prefill shape
    as `Qwen3`

## Δ vs baseline

- **Baseline:** use the most recent CUDA/Qwen3.5 guidellm snapshot on the
  remote bench host
- Delta table: `pending-remote`

## Artefacts

- Raw: `pending-remote`
- CSV: `pending-remote`
- HTML: `pending-remote`
- Service trace (before): `pending-remote`
- Service trace (during): `pending-remote`
- Service trace (after): `pending-remote`
- Service trace (summary): `pending-remote`

## Notes

- What changed in the code since baseline:
  - `Qwen3.5` now ships on paged prefill and no longer uses the old
    contiguous-prefill default path
  - scheduler-side mixed path and admission logic are unified with the current
    canonical CUDA scheduler flow
  - this benchmark is specifically meant to validate the model-side cleanup and
    remaining boundary, not to over-claim full `Qwen3`/`Qwen3.5` implementation
    identity
- Suspected cause of any regression: if the remote run still lags, the first
  model-local suspect is the remaining gap between `Qwen3`'s packed batched
  paged-prefill implementation and `Qwen3.5`'s current per-request replay over
  a batched allocation
- Follow-ups:
  - capture the remote CUDA sweep
  - decide whether `Qwen3.5` needs a real packed batched paged-prefill path to
    close the remaining gap
