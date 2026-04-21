# guidellm sweep cuda-l4-c16-sglang-align-single-admission — guidellm sweep, cuda-l4-c16-sglang-align-single-admission, 2026-04-21

> Template for canonical guidellm bench wins. Copy this file when
> `scripts/bench_guidellm.sh` runs, fill the placeholders, commit. Never
> edit an existing wins entry — always create a new dated one and diff
> against the prior. `scripts/bench_throughput.py` is legacy only.
> Canonical params are locked in
> [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md) §3.

## Goal

- Regression-check the first delete-style CUDA scheduler rewrite that moves admission behind a single waiting-queue budgeted path.

## Hypothesis

- Normalizing requests once and materializing them only after the prefill budget admits them should make CUDA semantics closer to sglang, but a too-conservative reservation model could still hurt throughput.

## Command

```bash
scripts/bench_guidellm.sh <backend-label> \
  [--target http://localhost:8000] \
  [--model Qwen/Qwen3-4B] \
  [--processor models/Qwen3-4B]
```

Invoked via: `scripts/bench_guidellm.sh <backend-label> [--target URL] [--model NAME] [--processor PATH]`

Actual invocation:

```bash
PATH=/root/.local/bin:$PATH ./scripts/bench_guidellm.sh \
  cuda-l4-c16-sglang-align-single-admission \
  --model Qwen/Qwen3-4B \
  --processor /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
```

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** NVIDIA L4, 23034 MiB VRAM, driver 580.82.07
- **Commit:** 6fe9097
- **Feature set:** `cargo build -p infer --release --bin infer`
- **Non-default flags / env vars:** `GUIDELLM__MP_CONTEXT_TYPE` present; server flags `--num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 --chunked-prefill-size 4096 --max-prefill-tokens 16384 --enable-mixed-chunk true --kv-cache-dtype auto`
- **Server launch:** `/tmp/agent-infer-bench-target/release/infer --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c --port 8000 --num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 --chunked-prefill-size 4096 --max-prefill-tokens 16384 --enable-mixed-chunk true --kv-cache-dtype auto`

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json,csv,html`
- Wrapper: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 761.5 | 769.1 | 36.23 | 36.36 | 25.91 | 0.083 |
| throughput | 17389.2 | 40782.7 | 103.6 | 139.51 | 69.34 | 0.267 |
| 0.10625r/s | 851 | 870 | 39.59 | 39.71 | 26.81 | 0.1 |
| 0.12916666666666665r/s | 845.6 | 861.6 | 40.74 | 40.83 | 31.47 | 0.117 |
| 0.15208333333333335r/s | 851.3 | 866 | 41.58 | 43.9 | 36.07 | 0.133 |
| 0.175r/s | 862.5 | 879.7 | 45.19 | 47.77 | 40.13 | 0.15 |
| 0.19791666666666669r/s | 867.3 | 937.3 | 46.18 | 48.47 | 44.6 | 0.167 |
| 0.22083333333333333r/s | 848.7 | 880.7 | 46.81 | 49.08 | 49.65 | 0.183 |
| 0.24375000000000002r/s | 872 | 884.6 | 52.32 | 53.01 | 52.29 | 0.2 |
| 0.26666666666666666r/s | 860.7 | 885.1 | 51.55 | 53.88 | 57.14 | 0.217 |


## Problems

- Throughput regressed versus the `headroom-retract` baseline even though the semantics were cleaner: `out tok/s @ throughput` dropped to `69.34` and the highest constant-load leg only reached `57.14`.
- The rewrite still left steady-state far from a real c16 working set. `TTFT p50 @ throughput` rose to `17.4s`, which means the queue was still backing up badly under the sweep profile.

## Learnings

- Moving admission behind a single waiting-queue budget is the right structural simplification, but reserving the whole prompt on first admission is too conservative for chunked prefill.
- Sglang's `PrefillAdder` budgets the extend chunk that will actually run this round, not the full prompt tail. Matching that granularity matters more than simply moving admission later in the control flow.

## Δ vs baseline

- **Baseline:** [2026-04-21-bench-guidellm-cuda-l4-c16-sglang-align-headroom-retract.md](./2026-04-21-bench-guidellm-cuda-l4-c16-sglang-align-headroom-retract.md)
- **Delta table** (only when a prior snapshot exists — else "first run"):

| metric | baseline | now | Δ% |
|---|---|---|---|
| TTFT p99 @ synchronous | 768.2 ms | 769.1 ms | +0.1% |
| out tok/s @ throughput leg | 73.21 | 69.34 | -5.3% |
| out tok/s @ highest constant leg | 71.30 | 57.14 | -19.9% |

## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-21-cuda-l4-c16-sglang-align-single-admission/benchmarks.json`
- CSV:  `/content/workspace/agent-infer/bench-output/2026-04-21-cuda-l4-c16-sglang-align-single-admission/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-04-21-cuda-l4-c16-sglang-align-single-admission/benchmarks.html`

## Notes

- What changed in the code since baseline: CUDA scheduler switched to normalized `QueuedRequest` intake, preserved request priority across requeue, and moved waiting-queue materialization behind step-time prefill admission.
- Suspected cause of any regression: the first reservation model charged waiting requests against the full prompt tail instead of the chunk that would actually run in the current step.
- Follow-ups: narrow the prefill reservation to chunk-sized pool growth and re-run the same c16 sweep before changing anything else.
