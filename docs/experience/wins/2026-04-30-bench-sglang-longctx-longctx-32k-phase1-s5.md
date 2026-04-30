# SGLang longctx baseline — longctx-32k-phase1-s5, 2026-04-30

## Goal

- Baseline: capture the pinned SGLang longctx-32k reference row for ARLE
  Phase 1 S4.

## Hypothesis

- SGLang at the project pin provides the reproducible competitor baseline for
  prompt=32768, output=256, c=1,4.

## Command

```bash
scripts/bench_sglang_longctx.sh longctx-32k-phase1-s5
```

## Environment

- **Backend:** SGLang
- **Model:** Qwen/Qwen3-4B
- **Weights:** `/content/workspace/agent-infer/infer/models/Qwen3-4B`
- **Target:** `http://localhost:30000`
- **Commit:** `214c35b03184c354acf1f86f99746799e1c9b3a9`
- **Expected pin:** `214c35b03184c354acf1f86f99746799e1c9b3a9`
- **Launch:** `python3 -m sglang.launch_server --model-path /content/workspace/agent-infer/infer/models/Qwen3-4B --kv-cache-dtype fp8_e4m3 --max-running-requests 16 --mem-fraction-static 0.85 --disable-radix-cache --max-total-tokens 140000`

## Canonical params

- `--profile concurrent`
- `--data prompt_tokens=32768,prompt_tokens_stdev=1,prompt_tokens_min=32768,prompt_tokens_max=32768,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256`
- `--rate 1,4`
- `--max-seconds 300`
- `--random-seed 20260416`
- Secondary c=1 run: `1` (`360s`)

## Results

| rate | out tok/s | total tok/s | req/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| conc1 | 11.67 | 1504.96 | 0.04 | 11865.94 | 12089.73 | 43.09 | 43.17 |
| conc4 | 16.27 | 2098.28 | 0.05 | 24182.25 | 47241.24 | 119.43 | 211.27 |
| conc1 | 11.57 | 1492.23 | 0.04 | 11862.86 | 11987.23 | 43.10 | 43.16 |

## Problems

- A 5-second SGLang smoke window was too short on a fresh process; the validated
  smoke used `--smoke-seconds 60` before this canonical run.
- The process logs warn that local C++ extensions were skipped because this
  pinned SGLang install selected `torch 2.9.1+cu128` while the extension import
  path expects `torch >= 2.11.0`.
- c=4 improved throughput over c=1 but raised latency sharply:
  TTFT p50 `24182.25 ms` and ITL p99 `211.27 ms`.

## Learnings

- The pinned SGLang row is stable enough for Phase 1 comparison: primary c=1
  `11.67 out tok/s`, secondary c=1 `11.57 out tok/s` (about `0.9%` drift).
- On the local L4, SGLang's c=4 throughput is `16.27 out tok/s`, so ARLE's
  previously captured `9.96 out tok/s` is the immediate Phase 1 gap to close.

## Delta vs baseline

- First pinned SGLang longctx-32k baseline for this mission slice.

## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-30-sglang-longctx-32k-phase1-s5/`
- Primary: `/content/workspace/agent-infer/bench-output/2026-04-30-sglang-longctx-32k-phase1-s5/guidellm-primary/benchmarks.json`
- Secondary c=1: `/content/workspace/agent-infer/bench-output/2026-04-30-sglang-longctx-32k-phase1-s5/guidellm-c1-secondary/benchmarks.json`
- Headline: `/content/workspace/agent-infer/bench-output/2026-04-30-sglang-longctx-32k-phase1-s5/headline_table.md`
- Server log: `/content/workspace/agent-infer/bench-output/2026-04-30-sglang-longctx-32k-phase1-s5/sglang_server.log`
