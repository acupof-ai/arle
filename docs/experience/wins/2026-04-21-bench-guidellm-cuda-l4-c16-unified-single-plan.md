# guidellm sweep cuda-l4-c16-unified-single-plan — guidellm sweep, cuda-l4-c16-unified-single-plan, 2026-04-21

> Template for canonical guidellm bench wins. Copy this file when
> `scripts/bench_guidellm.sh` runs, fill the placeholders, commit. Never
> edit an existing wins entry — always create a new dated one and diff
> against the prior. `scripts/bench_throughput.py` is legacy only.
> Canonical params are locked in
> [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md) §3.

## Goal

- Regression-check the deletion-style CUDA scheduler rewrite that collapses one tick into a single planned GPU path instead of `mixed + extra serial prefill`.

## Hypothesis

- Removing the second in-tick prefill phase should slightly improve queue draining and simplify scheduler semantics, but may still remain below the earlier `headroom-retract` high-water mark if long-prefill serialization is the real bottleneck.

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
GUIDELLM__MP_CONTEXT_TYPE=forkserver \
guidellm benchmark run \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-4B \
  --processor Qwen/Qwen3-4B \
  --profile sweep \
  --data prompt_tokens=4096,output_tokens=256 \
  --max-seconds 60 \
  --random-seed 20260416 \
  --output-dir /content/workspace/agent-infer/bench-output/2026-04-21-cuda-l4-c16-unified-single-plan \
  --backend openai_http \
  --backend-kwargs '{"validate_backend": "/v1/models", "request_format": "/v1/completions"}' \
  --disable-console-interactive \
  --outputs json \
  --outputs csv \
  --outputs html
```

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** NVIDIA L4, 23034 MiB VRAM, driver 580.82.07
- **Commit:** `2031b87`
- **Feature set:** `cargo build -p infer --release --bin infer`
- **Non-default flags / env vars:** `GUIDELLM__MP_CONTEXT_TYPE=forkserver`; server flags `--num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 --chunked-prefill-size 4096 --max-prefill-tokens 16384 --enable-mixed-chunk true`
- **Server launch:** `target/release/infer --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c --port 8000 --num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 --chunked-prefill-size 4096 --max-prefill-tokens 16384 --enable-mixed-chunk true`

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 754.1 | 767.2 | 36.24 | 36.32 | 25.92 | 0.083 |
| throughput | 15212.9 | 38579 | 103.41 | 137.07 | 70.26 | 0.267 |
| 0.10625r/s | 864.3 | 884.4 | 39.63 | 39.66 | 26.81 | 0.1 |
| 0.12916666666666665r/s | 840 | 869.1 | 40.54 | 40.68 | 31.47 | 0.117 |
| 0.15208333333333335r/s | 848.3 | 874.7 | 41.3 | 43.63 | 36.13 | 0.133 |
| 0.175r/s | 869.4 | 1435.3 | 44.68 | 47.13 | 40.21 | 0.15 |
| 0.19791666666666669r/s | 856.7 | 897.8 | 45.72 | 48.15 | 44.66 | 0.167 |
| 0.22083333333333333r/s | 864 | 892.6 | 46.84 | 49.05 | 49.63 | 0.183 |
| 0.24375000000000002r/s | 870 | 901.7 | 52.32 | 53.1 | 52.28 | 0.2 |
| 0.26666666666666666r/s | 867.7 | 889.5 | 51.54 | 53.8 | 57.11 | 0.217 |


## Problems

- The single-plan rewrite did not recover the earlier best c16 steady-state. It improved only marginally over `delete-path-full-clean`, and the highest constant-load leg stayed flat at `57.11 tok/s`.
- The run still lagged the stronger `headroom-retract` baseline: throughput-leg output throughput remained `70.26 tok/s` versus `73.21 tok/s`, and the highest constant-load leg remained far lower at `57.11` versus `71.30 tok/s`.
- The throughput leg remained queue-backed rather than settling into a healthy c16 decode-heavy state: `TTFT p50` rose to `15.2 s` and `TTFT p99` reached `38.6 s`.

## Learnings

- Deleting the extra in-tick serial prefill phase is the right semantic simplification. It removes a non-sglang execution shape without hurting sync latency and slightly improves throughput over the immediately prior delete-path baseline.
- That structural cleanup alone is not enough to restore best-known long-context throughput. The remaining loss is deeper than “two-stage tick planning”; long prefill windows still dominate the stream and keep the running decode batch narrower than the earlier high-water mark.

## Δ vs baseline

- **Baseline:** [2026-04-21-bench-guidellm-cuda-l4-c16-delete-path-full-clean.md](./2026-04-21-bench-guidellm-cuda-l4-c16-delete-path-full-clean.md)
- **Delta table** (only when a prior snapshot exists — else "first run"):

| metric | baseline | now | Δ% |
|---|---|---|---|
| TTFT p99 @ synchronous | 771.0 ms | 767.2 ms | -0.5% |
| out tok/s @ throughput leg | 69.78 | 70.26 | +0.7% |
| out tok/s @ highest constant leg | 57.12 | 57.11 | -0.0% |

## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-21-cuda-l4-c16-unified-single-plan/benchmarks.json`
- CSV:  `/content/workspace/agent-infer/bench-output/2026-04-21-cuda-l4-c16-unified-single-plan/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-04-21-cuda-l4-c16-unified-single-plan/benchmarks.html`

## Notes

- What changed in the code since baseline: deleted the old two-stage scheduler tick and replaced it with one planned GPU path per step (`Idle` / `DecodeOnly` / `Mixed` / `PrefillOnly`), while keeping the earlier decode-headroom and retract semantics.
- Suspected cause of any regression: the old two-stage path was not the dominant throughput bottleneck. The remaining gap versus `headroom-retract` is still driven by long-prefill serialization and insufficient steady-state decode overlap.
- Follow-ups: compare the later `kv-tier-stage-path-removal` benchmark against this snapshot on fresh `main`, and keep pushing toward a simpler single-path scheduler plus a more effective long-context overlap story.
