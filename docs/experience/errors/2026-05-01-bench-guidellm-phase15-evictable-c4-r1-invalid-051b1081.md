# guidellm invalid result phase15-evictable-c4-r1 — 2026-05-01

This entry records the first Phase 1.5 evictable-prefix-budget benchmark run.
GuideLLM completed the 300s c=4 longctx window and wrote raw artifacts, but the
wrapper rejected the result set because successful requests reported non-zero
output tokens while TTFT p50 and ITL p50 were both `0.0`. Treat this as an
invalid benchmark sample, not as a wins entry.

## Goal

- Validate whether `051b1081` removes the c=4 longctx deadlock mode by counting
  cascade-evictable GPU prefix-cache pages in the admission budget.

## Hypothesis

- Including GPU-resident evictable prefix-cache pages in effective free capacity
  should stop the random c=4 zero-throughput mode seen in the `0464fb3e`
  baseline and raise stable c=4 throughput toward the SGLang mission target
  artifact `docs/experience/wins/2026-04-30-bench-sglang-longctx-longctx-32k-phase1-s5.md`.

## Command

```bash
WORKLOAD=longctx-32k LONGCTX_CONCURRENCIES=4 LONGCTX_MAX_SECONDS=300 \
  scripts/bench_guidellm.sh phase15-evictable-c4-r1 \
  --target http://127.0.0.1:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

Invoked via: `WORKLOAD=longctx-32k LONGCTX_CONCURRENCIES=4 LONGCTX_MAX_SECONDS=300 scripts/bench_guidellm.sh phase15-evictable-c4-r1 --target http://127.0.0.1:8000 --model Qwen3-4B --processor infer/models/Qwen3-4B`

## Environment

- **Backend:** CUDA
- **Model:** Qwen3-4B
- **Hardware:** NVIDIA L4, 23034 MiB VRAM, driver 580.82.07, CUDA 12.8 toolchain
- **Commit:** `051b1081`
- **Feature set:** `cargo build -p infer --release --features cuda`
- **Non-default flags / env vars:** `CUDA_HOME=/usr/local/cuda`,
  `TORCH_CUDA_ARCH_LIST=8.9`, `INFER_TRITON_PYTHON=/usr/bin/python3`,
  `INFER_TILELANG_PYTHON=/usr/bin/python3`,
  `CARGO_TARGET_DIR=/tmp/arle-target`,
  `ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig`
- **Server launch:** `/tmp/arle-target/release/infer --model-path infer/models/Qwen3-4B --port 8000 --kv-cache-dtype fp8 --num-slots 16 --max-seq-len 131072 --mem-fraction-static 0.85 --max-num-batched-tokens 16384 --max-prefill-tokens 16384 --schedule-policy fcfs`
- **KV pool:** `136976` max tokens, `8561` pages, `11.0 GB`, FP8E4M3

## Canonical params (resolved by wrapper)

- `--profile concurrent`
- `--data prompt_tokens=32768,prompt_tokens_stdev=1,prompt_tokens_min=32768,prompt_tokens_max=32768,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256`
- `--max-seconds 300`
- `--random-seed 20260416`
- `--rate 4`
- `--outputs json --outputs csv --outputs html`
- Workload: `longctx-32k`
- Wrapper: `scripts/bench_guidellm.sh <backend-label> --workload longctx-32k`

## Results — GuideLLM headline table

GuideLLM emitted a table but the wrapper rejected it as invalid:

| metric | value |
|---|---:|
| completed input tokens | 1,048,580 |
| completed output tokens | 8,192 |
| median concurrency | 4.0 |
| request/s mean | 0.1 |
| output tok/s mean | 29.8 |
| total tok/s mean | 3843.7 |
| request latency p50 | 30.7s |
| request latency p95 | 112.3s |
| TTFT p50 | 0.0 ms invalid |
| ITL p50 | 0.0 ms invalid |

Wrapper validation failure:

```text
guidellm validation failed:
  - conc4: TTFT p50 was 0.0 despite successful requests with non-zero output tokens
  - conc4: ITL p50 was 0.0 despite successful requests averaging more than one output token
error: guidellm wrote benchmark files, but the result set is invalid
```

## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| trace samples | 344 |
| peak active | 4 |
| peak waiting | 0 |
| peak running_batch | 4 |
| peak prefill_queue | 1 |
| peak kv_util | 100.0% |
| `plan_label.idle` | 6 |
| `plan_label.decode` | 262 |
| `plan_label.prefill` | 115 |
| `plan_label.split` | 0 |
| `plan_label.mixed` | 2 |
| `prefix_hit_rate` | peak 0.0%, q75 0.0% |
| `prefix_skip_rate` | peak 0.0% |
| `kv_fetch_waiters` | 0 |
| final observed active | 0 |
| final observed scheduled | 0 |
| final observed tokens_out | 1032 |
| final observed active_ttft_p50 | 60000.0 ms |

## Results — request accounting

| metric | value |
|---|---:|
| GuideLLM created requests | 44 |
| GuideLLM successful requests | 32 |
| GuideLLM errored requests | 0 |
| GuideLLM cancelled requests | 12 |
| service requests counter delta | 36 |
| service tokens_out delta | 1024 |

## Problems

- The patch did remove the obvious first-batch zero-throughput hang shape:
  service-side active returned to `0`, and GuideLLM recorded 32 successful
  requests within the 300s window.
- The result set is still invalid because GuideLLM saw p50 TTFT and p50 ITL as
  `0.0` despite completed non-zero outputs. It cannot be used for the required
  three-run mean or the entrance gate.
- `kv_util` still hit `100.0%`; the run relied on prefix-cache pressure fallback
  and dropped GPU blocks during admission. Prefix hit rate stayed `0.0%`, so
  this run does not validate the agent-loop/prefix-hit workload yet.

## Learnings

- Evictable-prefix admission appears to make progress rather than deadlocking,
  but the measurement path is not valid enough to claim a throughput win.
- The next run should start from a fresh server and check whether the TTFT/ITL
  zeroing is reproducible or a wrapper/stream accounting artifact. If it
  repeats, debug `/v1/completions` streaming timing before treating c=4
  throughput numbers as publishable.

## Delta vs baseline

- **Baseline:** `docs/experience/wins/2026-04-30-bench-guidellm-longctx-32k-phase1-baseline-anchor-0464fb3e.md`
- **Mission target:** `docs/experience/wins/2026-04-30-bench-sglang-longctx-longctx-32k-phase1-s5.md`

| metric | baseline | now | Δ% |
|---|---:|---:|---:|
| c4 valid output tok/s | 2.99 mean | invalid | n/a |
| c4 nonzero completion mode | 2/3 baseline runs | 1/1 invalid run completed | qualitative improvement |
| c4 deadlock/zero mode | 1/3 baseline runs | 0/1 observed | qualitative improvement |
| peak kv_util | 99.0% baseline peak | 100.0% | +1.0 pp |
| valid entrance-gate sample | yes for r1/r2, no for r3 | no | regression vs measurement gate |

## Artefacts

- Raw: `bench-output/2026-05-01-phase15-evictable-c4-r1/benchmarks.json`
- CSV: `bench-output/2026-05-01-phase15-evictable-c4-r1/benchmarks.csv`
- HTML: `bench-output/2026-05-01-phase15-evictable-c4-r1/benchmarks.html`
- Command: `bench-output/2026-05-01-phase15-evictable-c4-r1/command.txt`
- Log: `bench-output/2026-05-01-phase15-evictable-c4-r1/guidellm.log`
- Service trace (before): `bench-output/2026-05-01-phase15-evictable-c4-r1/service_stats_before.txt`
- Service trace (during): `bench-output/2026-05-01-phase15-evictable-c4-r1/service_stats_trace.jsonl`
- Service trace (after): `bench-output/2026-05-01-phase15-evictable-c4-r1/service_stats_after.txt`
- Service trace (summary): `bench-output/2026-05-01-phase15-evictable-c4-r1/service_stats_trace_summary.md`

## Notes

- Code delta since baseline: `051b1081` adds cascade-aware radix evictable block
  enumeration, counts only GPU-resident `block_to_pages` capacity in
  `effective_pool_free_pages`, uses that in admission, and aligns mixed decode
  retraction with the same effective-free budget.
- Follow-up: rerun from a fresh server process. If GuideLLM invalid timing
  repeats while service completes requests, debug timing emission before
  continuing the required three-run mean.
