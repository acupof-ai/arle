# guidellm sweep cuda-l4-c16-sglang-align-headroom-retract — guidellm sweep, cuda-l4-c16-sglang-align-headroom-retract, 2026-04-21

> Template for canonical guidellm bench wins. Copy this file when
> `scripts/bench_guidellm.sh` runs, fill the placeholders, commit. Never
> edit an existing wins entry — always create a new dated one and diff
> against the prior. `scripts/bench_throughput.py` is legacy only.
> Canonical params are locked in
> [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md) §3.

## Goal

- Regression-check a scheduler refactor that aligns CUDA admission closer to sglang by reserving decode headroom during prefill admission and retracting decode requests under KV pressure instead of finishing them.

## Hypothesis

- Removing decode-time `TokenKVPool: out of pages` failures should materially improve high-concurrency throughput and stabilize the c16 sweep, even if the fixed 8.9 GB BF16 KV pool still cannot sustain a true 16-way working set of `4097 + decode` tokens.

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
  cuda-l4-c16-sglang-align-headroom-retract \
  --model Qwen/Qwen3-4B \
  --processor /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
```

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** NVIDIA L4, 23034 MiB VRAM, driver 580.82.07
- **Commit:** 435225d
- **Feature set:** `cargo build -p infer --release --bin infer`
- **Non-default flags / env vars:** `GUIDELLM__MP_CONTEXT_TYPE` present; server flags `--num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 --chunked-prefill-size 4096 --max-prefill-tokens 16384`
- **Server launch:** `/tmp/agent-infer-target-bench/release/infer --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c --port 8000 --num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 --chunked-prefill-size 4096 --max-prefill-tokens 16384`

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
| sync | 0 | 768.2 | 0 | 35.61 | 101.42 | 0.433 |
| throughput | 21504.1 | 38186.9 | 59.98 | 69.01 | 73.21 | 0.25 |
| 0.41041666666666665r/s | 1218.2 | 13585.1 | 57.81 | 65.82 | 71.3 | 0.267 |
| 0.3875r/s | 3500.7 | 11311.8 | 55.13 | 58.63 | 67.65 | 0.25 |
| 0.36458333333333337r/s | 2638.5 | 9903.8 | 54.87 | 58.76 | 67.02 | 0.25 |
| 0.3416666666666667r/s | 1648.6 | 8381 | 54.31 | 58.69 | 66.22 | 0.25 |
| 0.31875r/s | 886.9 | 5770.1 | 57.09 | 57.92 | 65.47 | 0.25 |
| 0.29583333333333334r/s | 884.2 | 2688.5 | 56.46 | 56.95 | 61.18 | 0.233 |
| 0.2729166666666667r/s | 871.7 | 883.7 | 52.62 | 52.79 | 58.58 | 0.217 |
| 0.25r/s | 870.9 | 893.1 | 51.21 | 51.51 | 53.76 | 0.2 |


## Problems

- The benchmark no longer reproduced `TokenKVPool: out of pages`, but the sweep still does not sustain a true c16 mean working set. Mean request concurrency peaks at `8.04` on the throughput leg and `5.02` on the highest constant leg.
- `TTFT p50 @ synchronous` came back as `0 ms` in this run, while `TTFT p95` stayed at `765.8 ms`. Treat the p50 as a guidellm measurement artifact for this snapshot; the p95 and server logs are the trustworthy sync-TTFT signals here.
- The benchmark JSON records many `cancelled_requests` at phase boundaries. These are bench-side cancellations after `max_seconds` expiry, not server-side scheduler errors.

## Learnings

- The sglang-aligned change that mattered was not “skip large prefills and pack smaller ones”; it was budget semantics. Reserving decode headroom during prefill admission and retracting decode victims under KV pressure removes the decode-OOM failure mode cleanly.
- With this L4 configuration the BF16 paged pool is `60064` tokens. A full `16 * 4097` prompt working set already exceeds that before decode growth, so “fill 16 slots with 4097-token prompts” is physically incompatible with the current KV budget. Further c16 alignment now depends on more KV capacity, KV compression, or a different admission target, not on prefix-cache eviction tweaks.

## Δ vs baseline

- **Baseline:** [2026-04-21-bench-guidellm-cuda-l4-c16-prefill-pack.md](./2026-04-21-bench-guidellm-cuda-l4-c16-prefill-pack.md)
- **Delta table** (only when a prior snapshot exists — else "first run"):

| metric | baseline | now | Δ% |
|---|---|---|---|
| TTFT p95 @ synchronous | 772.7 ms | 765.8 ms | -0.9% |
| out tok/s @ throughput leg | 46.33 | 73.21 | +58.0% |
| req/s actual @ throughput leg | 0.55 | 0.25 | -54.5% |
| mean concurrency @ throughput leg | 20.19 | 8.04 | -60.2% |
| out tok/s @ highest constant leg | 37.61 | 71.30 | +89.6% |

## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-21-cuda-l4-c16-sglang-align-headroom-retract/benchmarks.json`
- CSV:  `/content/workspace/agent-infer/bench-output/2026-04-21-cuda-l4-c16-sglang-align-headroom-retract/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-04-21-cuda-l4-c16-sglang-align-headroom-retract/benchmarks.html`

## Notes

- What changed in the code since baseline: CUDA scheduler refactor to `waiting + prefill_queue + running_batch`, prefill admission now reserves decode headroom, and decode OOM handling retracts/requeues victims instead of finishing requests on pool exhaustion.
- Suspected cause of any regression: lower measured throughput-leg request rate is a by-product of stopping decode OOM crashes and enforcing a physically valid working set under the current KV budget; the scheduler now admits less impossible work.
- Follow-ups: quantify whether `enable_mixed_chunk` improves this c16 profile, and decide whether the next alignment step is KV-capacity work (quantized KV / larger budget) or more sglang-like mixed prefill+decode overlap.
