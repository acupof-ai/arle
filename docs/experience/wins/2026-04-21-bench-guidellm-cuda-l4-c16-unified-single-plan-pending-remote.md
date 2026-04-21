# CUDA c16 unified single-plan scheduler — guidellm sweep, pending-remote, 2026-04-21

## Goal

- Record the required post-change benchmark stub for the CUDA scheduler rewrite that deletes the old two-stage `mixed + extra serial prefill` execution path.

## Hypothesis

- A clean `4096/256` c16 sweep should improve high-concurrency queue draining by running exactly one scheduler-planned GPU batch per tick, closer to sglang's `get_next_batch_to_run() -> run_batch()` semantics.

## Command

```bash
scripts/bench_guidellm.sh cuda-l4-c16-unified-single-plan
```

Invoked via: `scripts/bench_guidellm.sh cuda-l4-c16-unified-single-plan`

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** NVIDIA L4
- **Commit:** pending-remote
- **Feature set:** `cargo test -p infer --release --lib scheduler::cuda`
- **Non-default flags / env vars:** `ZIG=/tmp/zig-local/zig-x86_64-linux-0.16.0/zig` for local verification only
- **Server launch:** `scripts/start_infer.sh <model> <port>`

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh cuda-l4-c16-unified-single-plan`

## Results — sweep headline table

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| pending-remote | pending-remote | pending-remote | pending-remote | pending-remote | pending-remote | pending-remote |

## Problems

- This commit was pushed before the canonical clean sweep completed; only local compile + focused test verification ran.
- Local `cargo check/test` required a temporary Zig binary because the host image did not have `zig` on `PATH`.

## Learnings

- The important semantic correction is structural, not a token-budget tweak: one tick now plans one batch path instead of launching mixed and then appending more serial prefills on the same tick.

## Δ vs baseline

- **Baseline:** [2026-04-21-bench-guidellm-cuda-l4-c16-delete-path-full-clean.md](./2026-04-21-bench-guidellm-cuda-l4-c16-delete-path-full-clean.md)
- **Delta table:** pending-remote

## Artefacts

- Raw: `pending-remote`
- CSV: `pending-remote`
- HTML: `pending-remote`

## Notes

- What changed in the code since baseline: deleted the old two-stage scheduler tick and replaced it with a single planned path (`Idle` / `DecodeOnly` / `Mixed` / `PrefillOnly`), while keeping request-length and retract semantics aligned with the prior sglang work.
- Suspected cause of any regression: n/a until remote run lands
- Follow-ups: run a fresh `cuda-l4-c16-unified-single-plan` sweep and write a non-stub wins entry with deltas against the best prior c16 baseline
