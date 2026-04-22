# Scheduler async emit worker — guidellm sweep, cuda, 2026-04-22

## Goal

- Regression-check the scheduler change that moves stopless decode/prefill
  detokenize + `CompletionStreamDelta` emission off the scheduler thread and
  onto a dedicated emit worker.

## Hypothesis

- High-concurrency throughput and TTFT tail should improve modestly because the
  scheduler no longer spends its hot-path CPU budget decoding UTF-8 suffixes
  and writing streaming deltas for requests that do not need stop-sensitive
  control-flow gating.

## Command

```bash
scripts/bench_guidellm.sh cuda \
  --model Qwen/Qwen3-4B \
  --processor models/Qwen3-4B
```

Status: `pending-remote` (CUDA bench host required).

## Environment

- **Backend:** `cuda`
- **Model:** `Qwen/Qwen3-4B`
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

- The clean split is not “all emit async” vs “all emit sync”. The useful
  architectural seam is stop-sensitive vs stopless:
  stop-sensitive text still gates decode control flow and stays on the
  scheduler thread; pure streaming output can move entirely to a dedicated
  emit worker.

## Δ vs baseline

- **Baseline:** first run for this micro-tranche

## Artefacts

- Pending remote run

## Notes

- What changed in the code since baseline: added a dedicated CUDA scheduler
  emit worker; stopless decode and prefill-first-token streaming now enqueue
  async append/finish events instead of doing detokenize + `delta_tx.send()`
  directly on the scheduler thread
- Suspected cause of any regression: n/a
- Follow-ups: re-check whether stop-sensitive requests still dominate CPU time
  under realistic stop-list usage before adding a second-stage stop worker
