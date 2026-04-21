# Train Control Proxy — guidellm sweep, cpu, 2026-04-21

> Pending-remote stub for the infer HTTP runtime change that adds the optional
> `/v1/train/*` control-plane proxy. No local guidellm run was performed in this
> patch because the change does not touch the token-generation hot path.

## Goal

- Regressions check for the optional infer-side `/v1/train/*` proxy without
  changing token-generation throughput or latency.

## Hypothesis

- The proxy only affects control-plane routes and should leave `/v1/completions`
  and `/v1/chat/completions` sweep metrics unchanged.

## Command

```bash
scripts/bench_guidellm.sh cpu \
  --target http://localhost:8000 \
  --model Qwen/Qwen3.5-4B \
  --processor models/Qwen3.5-4B
```

Invoked via: `scripts/bench_guidellm.sh cpu [--target URL] [--model NAME] [--processor PATH]`

## Environment

- **Backend:** cpu
- **Model:** Qwen/Qwen3.5-4B
- **Hardware:** pending remote
- **Commit:** `pending-remote`
- **Feature set:** `cargo build --release --no-default-features --features cpu,no-cuda`
- **Non-default flags / env vars:** `--train-control-url http://127.0.0.1:19101`
- **Server launch:** pending remote

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json,csv,html`
- Wrapper: `scripts/bench_guidellm.sh cpu`

## Results — sweep headline table

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| pending remote | pending remote | pending remote | pending remote | pending remote | pending remote | pending remote |

## Problems

- Local verification covered proxy correctness only: `/v1/train/status`,
  `/v1/train/events`, `/v1/train/save`, `/v1/train/stop`, plus a live
  `/v1/completions` request while the proxy was configured.
- Canonical guidellm sweep still needs to run on the remote benchmark machine.

## Learnings

- The proxy can be validated functionally without changing model weights or the
  generation loop; throughput validation still belongs in the normal guidellm
  sweep.

## Δ vs baseline

- **Baseline:** first run after adding the train-control proxy
- Delta table: pending first remote run

## Artefacts

- Raw: pending remote
- CSV: pending remote
- HTML: pending remote

## Notes

- What changed in the code since baseline: added optional infer-side proxy for
  `/v1/train/*` to bridge the train control plane through the serving HTTP API
- Suspected cause of any regression: n/a
- Follow-ups: run the canonical guidellm sweep remotely and update this stub
