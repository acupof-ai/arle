# guidellm sweep qwen3-4b infer-vs-sglang c1-c16 — guidellm sweep, qwen3-4b-infer-vs-sglang-c1-c16, 2026-04-22

> Template for canonical guidellm bench wins. Copy this file when
> `scripts/bench_guidellm.sh` runs, fill the placeholders, commit. Never
> edit an existing wins entry — always create a new dated one and diff
> against the prior. `scripts/bench_throughput.py` is legacy only.
> Canonical params are locked in
> [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md) §3.

## Goal

- Characterize current default CUDA `infer` against `sglang 0.5.10.post1` on the same L4 with fixed concurrency legs `1,2,4,8,16` under the canonical `4096-in / 256-out` guidellm sweep.

## Hypothesis

- `infer` should stay close to `sglang` at `c1-c2`, but the current admission/prefill path is still likely to collapse throughput and TTFT once backlog builds above `c4`.

## Command

```bash
scripts/bench_guidellm.sh <backend-label> \
  [--target http://localhost:8000] \
  [--model Qwen/Qwen3-4B] \
  [--processor models/Qwen3-4B]
```

Invoked via: `scripts/bench_guidellm.sh <backend-label> [--target URL] [--model NAME] [--processor PATH]`

Actual invocation (`infer`, full `c1,c2,c4,c8,c16` run):

```bash
GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh \
  infer-qwen3-4b-l4-c1-c16-serial \
  --target http://127.0.0.1:8000 \
  --model Qwen/Qwen3-4B \
  --processor /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --concurrencies 1,2,4,8,16 \
  --max-seconds 60 \
  --warmup 5
```

Actual invocation (`infer`, isolated `c4` rerun to replace invalid zero-latency sample from the full run):

```bash
GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh \
  infer-qwen3-4b-l4-conc4-rerun \
  --target http://127.0.0.1:8000 \
  --model Qwen/Qwen3-4B \
  --processor /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --concurrencies 4 \
  --max-seconds 60 \
  --warmup 5
```

Actual invocation (`sglang`, full `c1,c2,c4,c8,c16` run):

```bash
GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh \
  sglang-qwen3-4b-l4-c1-c16-serial \
  --target http://127.0.0.1:8000 \
  --model Qwen/Qwen3-4B \
  --processor /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --concurrencies 1,2,4,8,16 \
  --max-seconds 60 \
  --warmup 5
```

## Environment

- **Backend:** cuda (`infer` default scheduler vs `sglang 0.5.10.post1`)
- **Model:** `Qwen/Qwen3-4B`
- **Hardware:** `NVIDIA L4`, `23034 MiB` VRAM, driver `580.82.07`, CUDA `12.8`
- **Commit:** `f98ca92`
- **Feature set:** `cargo build -p infer --release --bin infer`
- **Non-default flags / env vars:** `GUIDELLM__MP_CONTEXT_TYPE=forkserver`; `infer` flags `--num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 --chunked-prefill-size 4096 --max-prefill-tokens 16384 --enable-mixed-chunk true`; `sglang` flags `--served-model-name Qwen/Qwen3-4B --mem-fraction-static 0.94 --dtype bfloat16 --max-running-requests 16 --context-length 4608 --disable-cuda-graph-padding --disable-piecewise-cuda-graph`
- **Server launch (`infer`):** `/tmp/agent-infer-target/release/infer --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c --port 8000 --num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 --chunked-prefill-size 4096 --max-prefill-tokens 16384 --enable-mixed-chunk true`
- **Server launch (`sglang`):** `python3 -m sglang.launch_server --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c --served-model-name Qwen/Qwen3-4B --port 8000 --mem-fraction-static 0.94 --dtype bfloat16 --max-running-requests 16 --context-length 4608 --disable-cuda-graph-padding --disable-piecewise-cuda-graph`

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json,csv,html`
- Wrapper: `scripts/bench_guidellm.sh <backend-label>`

## Results — paired concurrency table

`infer` `c4` values below come from the isolated rerun because the full `c1,c2,c4,c8,c16` pass emitted invalid `TTFT p50=0` and `ITL p50=0` at `c4` despite successful requests.

| conc | infer out tok/s | sglang out tok/s | Δ out tok/s | infer TTFT p50 (ms) | sglang TTFT p50 (ms) | Δ TTFT | infer ITL p50 (ms) | sglang ITL p50 (ms) | Δ ITL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26.59 | 26.46 | +0.5% | 739.9 | 739.0 | +0.1% | 35.28 | 35.48 | -0.5% |
| 2 | 41.59 | 45.81 | -9.2% | 1485.0 | 1409.4 | +5.4% | 38.82 | 41.22 | -5.8% |
| 4 | 36.70 | 74.05 | -50.4% | 14556.7 | 2079.7 | +600.0% | 44.24 | 48.84 | -9.4% |
| 8 | 57.71 | 107.79 | -46.5% | 15403.7 | 2951.5 | +421.9% | 47.32 | 61.60 | -23.2% |
| 16 | 45.08 | 137.07 | -67.1% | 15405.9 | 5803.0 | +165.5% | 44.61 | 93.99 | -52.5% |

## Results — raw `infer` table

| conc | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual | successful | incomplete | errored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c1 | 739.9 | 752.9 | 35.28 | 35.30 | 26.59 | 0.109 | 7 | 0 | 0 |
| c2 | 1485.0 | 2057.8 | 38.82 | 41.95 | 41.59 | 0.182 | 11 | 1 | 0 |
| c4 | 14556.7 | 16458.7 | 44.24 | 44.69 | 36.70 | 0.145 | 9 | 3 | 0 |
| c8 | 15403.7 | 26635.3 | 47.32 | 57.45 | 57.71 | 0.255 | 15 | 7 | 0 |
| c16 | 15405.9 | 53559.4 | 44.61 | 49.75 | 45.08 | 0.182 | 11 | 15 | 0 |

## Results — raw `sglang` table

| conc | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual | successful | incomplete | errored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c1 | 739.0 | 9259.2 | 35.48 | 35.51 | 26.46 | 0.091 | 6 | 0 | 0 |
| c2 | 1409.4 | 1471.0 | 41.22 | 41.32 | 45.81 | 0.182 | 12 | 0 | 0 |
| c4 | 2079.7 | 2885.1 | 48.84 | 51.94 | 74.05 | 0.291 | 20 | 0 | 0 |
| c8 | 2951.5 | 5764.5 | 61.60 | 72.72 | 107.79 | 0.436 | 24 | 8 | 0 |
| c16 | 5803.0 | 11437.6 | 93.99 | 116.32 | 137.07 | 0.291 | 32 | 0 | 0 |

## Problems

- The full `infer` sweep emitted invalid zero-valued `TTFT p50` and `ITL p50` at `c4`; an isolated `c4` rerun was required to recover usable numbers.
- `infer` stopped scaling once backlog built above `c2`: incomplete requests climbed to `3/12` at `c4`, `7/22` at `c8`, and `15/26` at `c16`, while throughput topped out at only `57.71 out tok/s`.
- `sglang` initially OOMed during piecewise CUDA graph capture at `--mem-fraction-static 0.94 --context-length 4608 --max-running-requests 16`; the working comparison run required `--disable-piecewise-cuda-graph`.
- `scripts/bench_guidellm.sh` polls `/v1/stats`; `sglang` returns `404` on that endpoint, so the wrapper could not produce a comparable service trace for `sglang` even though the main GuideLLM latency and throughput metrics were valid.

## Learnings

- Current `infer` still matches `sglang` only at `c1`: throughput is effectively identical and TTFT is within measurement noise.
- Lower `infer` ITL at `c4+` is not a throughput win. The scheduler is serving fewer active requests, so per-token decode spacing looks better while end-to-end throughput and TTFT collapse.
- The dominant gap is still control-plane admission/prefill pacing, not single-token decode cost. At `c16`, `sglang` reaches `137.07 out tok/s` with `TTFT p50 5803.0 ms`; `infer` stalls at `45.08 out tok/s` with `TTFT p50 15405.9 ms`.

## Δ vs baseline

- **Baseline:** first paired `c1-c16` serial comparison on the current default `c16` profile.
- **Delta table:** first run.

## Artefacts

- `infer` raw: `/content/workspace/agent-infer/bench-output/2026-04-22-infer-qwen3-4b-l4-c1-c16-serial/benchmarks.json`
- `infer` CSV: `/content/workspace/agent-infer/bench-output/2026-04-22-infer-qwen3-4b-l4-c1-c16-serial/benchmarks.csv`
- `infer` HTML: `/content/workspace/agent-infer/bench-output/2026-04-22-infer-qwen3-4b-l4-c1-c16-serial/benchmarks.html`
- `infer` service trace summary: `/content/workspace/agent-infer/bench-output/2026-04-22-infer-qwen3-4b-l4-c1-c16-serial/service_stats_trace_summary.md`
- `infer c4` rerun raw: `/content/workspace/agent-infer/bench-output/2026-04-22-infer-qwen3-4b-l4-conc4-rerun/benchmarks.json`
- `infer c4` rerun CSV: `/content/workspace/agent-infer/bench-output/2026-04-22-infer-qwen3-4b-l4-conc4-rerun/benchmarks.csv`
- `infer c4` rerun HTML: `/content/workspace/agent-infer/bench-output/2026-04-22-infer-qwen3-4b-l4-conc4-rerun/benchmarks.html`
- `sglang` raw: `/content/workspace/agent-infer/bench-output/2026-04-22-sglang-qwen3-4b-l4-c1-c16-serial/benchmarks.json`
- `sglang` CSV: `/content/workspace/agent-infer/bench-output/2026-04-22-sglang-qwen3-4b-l4-c1-c16-serial/benchmarks.csv`
- `sglang` HTML: `/content/workspace/agent-infer/bench-output/2026-04-22-sglang-qwen3-4b-l4-c1-c16-serial/benchmarks.html`
- `sglang` service trace summary: `/content/workspace/agent-infer/bench-output/2026-04-22-sglang-qwen3-4b-l4-c1-c16-serial/service_stats_trace_summary.md`

## Notes

- What changed in the code since baseline: none; this was a bench-only characterization run on `f98ca92`.
- Suspected cause of any regression: `infer` admission/reservation and mixed/prefill progress are still too conservative under backlog, so the server never sustains a true `c16` working set.
- Follow-ups: instrument per-tick admitted prefill tokens, active running count, and incomplete-request causes; then rerun the same `c1,c2,c4,c8,c16` table before changing the benchmark profile.
