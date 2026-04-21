# KV tier live readmission tranche — guidellm sweep, cuda, 2026-04-21

## Goal

- Regression-check the local CUDA live-readmission tranche (`ReadmissionPlan + WaitingFetch + FetchCompleted + promote_fetched_prefix`) against the latest tiered-KV CUDA baseline.

## Hypothesis

- Local staged readmission should preserve the existing CUDA serving baseline within noise while removing the old spill-only/document-only gap in the scheduler/runtime contract.

## Command

```bash
scripts/bench_guidellm.sh kv-tier-live-readmission-tranche
```

Invoked via: `scripts/bench_guidellm.sh kv-tier-live-readmission-tranche`

## Environment

- **Status:** pending-remote
- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** pending remote CUDA host
- **Commit:** local live-readmission tranche (final commit hash recorded once pushed; remote run still pending)
- **Feature set:** `cargo build --release --no-default-features --features cuda`
- **Non-default flags / env vars:** pending remote
- **Server launch:** pending remote

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh kv-tier-live-readmission-tranche`

## Results — sweep headline table

Pending remote run.

## Problems

- CUDA/guidellm validation was not available on this machine.

## Learnings

- The local verification bar for this tranche is structural only: `cargo clippy`, `cargo check` on `no-cuda`, `cuda,no-cuda`, `metal,no-cuda`, plus targeted `kv_tier / prefix_cache / scheduler` tests. Runtime performance still needs the remote CUDA sweep.

## Δ vs baseline

- **Baseline:** [2026-04-21-bench-guidellm-kv-tier-unified-multilayer-simplify.md](2026-04-21-bench-guidellm-kv-tier-unified-multilayer-simplify.md)
- Delta table pending remote run.

## Artefacts

- Raw: pending remote run
- CSV: pending remote run
- HTML: pending remote run

## Notes

- What changed in the code since baseline: local staged readmission is now live on the CUDA lane via `ReadmissionPlan`, coordinator `FetchQueue`, `Phase::WaitingFetch`, and `promote_fetched_prefix`; docs/AGENTS were updated to match the shipped runtime.
- Suspected cause of any regression: extra host-pinned allocation/copy work on staged-prefix hits, plus the new fetch dedupe and queue bookkeeping.
- Follow-ups: run `scripts/bench_guidellm.sh kv-tier-live-readmission-tranche` on the remote CUDA validation host and diff against the listed baseline.
