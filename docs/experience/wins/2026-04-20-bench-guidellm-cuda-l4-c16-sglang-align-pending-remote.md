# SGLang-Style Prefill Budgets — guidellm sweep, cuda-l4-c16-sglang-align, 2026-04-20

## Goal

- Regression-check the CUDA scheduler refactor that replaces decode-active prefill throttles with SGLang-style token/request budgets, then compare against the existing SGLang Qwen3-4B baseline.

## Hypothesis

- Launch-time prefill budgeting should remove empty-output failures and align c16 throughput behavior more closely with SGLang without regressing `--cuda-graph=false` correctness.

## Command

```bash
scripts/bench_guidellm.sh cuda-l4-c16-sglang-align \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

Invoked via: `scripts/bench_guidellm.sh <backend-label> [--target URL] [--model NAME] [--processor PATH]`

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** NVIDIA L4, CUDA 12.x
- **Commit:** pending-remote
- **Feature set:** `cargo build --release`
- **Non-default flags / env vars:** `--num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 --chunked-prefill-size 4096 --max-prefill-tokens 16384`
- **Server launch:** `./target/release/infer --model-path models/Qwen3-4B --port 8000 --num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 --chunked-prefill-size 4096 --max-prefill-tokens 16384`

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json,csv,html`
- Wrapper: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| pending-remote | pending-remote | pending-remote | pending-remote | pending-remote | pending-remote | pending-remote |

## Problems

- Bench deferred to the remote sweep step after `c4`: `scripts/bench_ab.sh sglang-qwen3-4b-sweep infer-qwen3-4b-post-sglang-align-sweep`.

## Learnings

- Pending remote data collection.

## Δ vs baseline

- **Baseline:** `2026-04-17` SGLang Qwen3-4B c16 sweep (`out tok/s = 115.83` at the referenced saturation point).
- Delta table pending remote rerun.

## Artefacts

- Raw: `bench-output/<date>-cuda-l4-c16-sglang-align/benchmarks.json`
- CSV:  `bench-output/<date>-cuda-l4-c16-sglang-align/benchmarks.csv`
- HTML: `bench-output/<date>-cuda-l4-c16-sglang-align/benchmarks.html`

## Notes

- What changed in the code since baseline: scheduler config surface now exposes `chunked_prefill_size`, `max_prefill_tokens`, `prefill_max_requests`, and `enable_mixed_chunk`; launch-time prefill budgeting replaces the old decode-active choke.
- Suspected cause of any regression: pending remote rerun.
- Follow-ups: update this entry with the post-`c4` sweep outputs and Δ table.
