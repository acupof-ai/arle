# Qwen3.6 DFlash Support — guidellm sweep, metal-qwen36-dflash, 2026-04-22

## Goal

- Record the required benchmark stub for the `Qwen3.6-35B-A3B` + DFlash Metal runtime support landing; local validation in this change is smoke-only, so the full guidellm sweep is pending remote execution.

## Hypothesis

- The runtime should now load `mlx-community/Qwen3.6-35B-A3B-4bit` together with `z-lab/Qwen3.6-35B-A3B-DFlash` and complete a minimal generation locally; throughput/latency impact still needs a canonical sweep.

## Command

```bash
scripts/bench_guidellm.sh metal-qwen36-dflash \
  --target http://localhost:8000 \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --processor /path/to/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46
```

Invoked via: `pending-remote`

## Environment

- **Backend:** metal
- **Model:** `mlx-community/Qwen3.6-35B-A3B-4bit` + `z-lab/Qwen3.6-35B-A3B-DFlash`
- **Hardware:** Apple Silicon local dev box; full guidellm sweep pending remote / dedicated bench run
- **Commit:** pending-local-commit
- **Feature set:** `cargo build --release --no-default-features --features metal,no-cuda`
- **Non-default flags / env vars:** `--dflash-draft-model z-lab/Qwen3.6-35B-A3B-DFlash`
- **Server launch:** pending-remote

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh metal-qwen36-dflash`

## Results — sweep headline table

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| pending-remote | pending-remote | pending-remote | pending-remote | pending-remote | pending-remote | pending-remote |

## Problems

- Full guidellm sweep not run in this turn; only local compile/test/smoke validation was executed.

## Learnings

- `z-lab/Qwen3.6-35B-A3B-DFlash` is compatible with the target once DFlash validation compares Q/KV projection widths instead of requiring identical head-count bucketing.

## Δ vs baseline

- **Baseline:** first run for this pair in this repo

## Artefacts

- Raw: `pending-remote`
- CSV: `pending-remote`
- HTML: `pending-remote`

## Notes

- What changed in the code since baseline: added Qwen3.6 single-request Metal DFlash path, widened draft/target compatibility checks to accept rebucketed heads with matching projection widths, and added default draft discovery for Qwen3.6 A3B.
- Suspected cause of any regression: n/a
- Follow-ups: run the canonical remote guidellm sweep and replace every `pending-remote` field with real data
