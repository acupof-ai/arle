# Metal DFlash continuous batching contract guard — guidellm pending, 2026-05-02

> Status: `pending-remote`. This tranche tightens the Metal Qwen3.5 DFlash
> continuous-batching contract and adds local unit coverage. It does not run a
> canonical DFlash server bench in this workspace because the required
> Qwen3.5+DFlash model pair and matching Metal bench setup are not local.

## Goal

- Regression guard: prove mixed DFlash-ready and stale continuous-batching rows
  preserve scheduler order, commit only ready-row tokens, and scalar-fallback
  stale rows before running the matched guidellm sweep.

## Hypothesis

- The refactor should be behavior-preserving for throughput because it only
  extracts existing selection/dispatch rules into checked helpers; the matched
  DFlash sweep should show no TTFT/ITL regression versus the latest Metal
  Qwen3.5 DFlash baseline.

## Command

```bash
cargo test -p infer --release --no-default-features --features metal,no-cuda dflash_ready_selection -- --nocapture
cargo test -p infer --release --no-default-features --features metal,no-cuda dflash_row_dispatch_plan -- --nocapture
cargo clippy -p infer --no-default-features --features metal,no-cuda --lib -- -D warnings
```

Canonical guidellm command, pending:

```bash
scripts/bench_guidellm.sh metal-qwen35-dflash-continuous-batching \
  --model Qwen/Qwen3.5 \
  --processor models/Qwen3.5
```

Invoked via: local contract tests only; canonical guidellm pending matched
Metal DFlash bench host.

## Environment

- **Backend:** metal
- **Model:** Qwen3.5 with DFlash draft model
- **Hardware:** pending matched Apple Silicon host
- **Commit:** pending commit for P2-1 DFlash continuous-batching contract guard
- **Feature set:** `cargo test --release --no-default-features --features metal,no-cuda`
- **Non-default flags / env vars:** none for local contract tests
- **Server launch:** pending canonical guidellm run

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

Canonical guidellm sweep: pending.

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---:|---:|---:|---:|---:|---:|
| pending | pending | pending | pending | pending | pending | pending |

## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| peak active | pending |
| peak waiting | pending |
| peak prefill_queue | pending |
| peak kv_util | pending |
| `prefix_hit_rate` | pending |
| `prefix_skip_rate` | pending |
| `kv_fetch_q` | pending |
| `kv_fetch_waiters` | pending |
| `kv_store_q` | pending |
| `kv_store` | pending |
| `kv_bp` | pending |
| `tier_recall` | n/a |
| `tier_src` | n/a |
| `tier_promoted` | n/a |
| `tier_fallback` | n/a |

## Results — request accounting

| metric | value |
|---|---:|
| completed input tokens | pending |
| incomplete input tokens | pending |
| completed output tokens | pending |
| incomplete output tokens | pending |

## Problems

- Canonical throughput/latency numbers are pending until a matched Metal
  Qwen3.5+DFlash server bench runs.
- Local verification covers the scheduler-order and stale-row fallback
  contracts, not the full MLX/GPU speculative decode path.

## Learnings

- Mixed DFlash continuous batches need an explicit dispatch plan: ready-row
  token vectors must match ready indices exactly, and stale rows must preserve
  original scheduler order when falling back to scalar decode.

## Delta vs baseline

- **Baseline:** latest Metal Qwen3.5 DFlash guidellm snapshot, pending matched
  host selection.
- **Delta table:** pending canonical guidellm run.

| metric | baseline | now | delta |
|---|---:|---:|---:|
| TTFT p50 @ synchronous | pending | pending | pending |
| ITL p50 @ saturation | pending | pending | pending |
| out tok/s @ saturation | pending | pending | pending |

## Artefacts

- Raw: pending
- CSV: pending
- HTML: pending
- Service trace: pending

## Notes

- What changed in the code since baseline: P2-1 extracted DFlash ready-row
  subset selection and ready/stale dispatch planning into checked helpers with
  unit tests.
- Suspected cause of any regression: a mismatch between ready indices and
  sampled-token ordering, or fallback rows not preserving scheduler order.
- Follow-ups: replace this pending entry with a completed guidellm snapshot
  after the matched Qwen3.5+DFlash Metal server run.
