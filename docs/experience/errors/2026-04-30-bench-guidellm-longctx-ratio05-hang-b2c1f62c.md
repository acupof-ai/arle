# guidellm hang longctx-32k-phase1-patched-ratio05-b2c1f62c — 2026-04-30

This entry records a failed Phase 1.5 parameter-scan point. The GuideLLM
process did not complete after the 300s c=4 window because ARLE remained with
four active requests and no forward progress, so the run was terminated and is
not a valid throughput sample.

## Goal

- Scan `decode_headroom_ratio=0.05` for the patched Phase 1.5 admission policy
  using the required longctx-32k c=4 point.

## Hypothesis

- Reducing final-chunk decode reservation from 10% to 5% might admit more
  concurrent long-prefill work while leaving enough live decode headroom to
  avoid the near-full KV failure edge.

## Command

```bash
WORKLOAD=longctx-32k LONGCTX_CONCURRENCIES=4 scripts/bench_guidellm.sh \
  longctx-32k-phase1-patched-ratio05-b2c1f62c \
  --target http://127.0.0.1:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

Invoked via: `WORKLOAD=longctx-32k LONGCTX_CONCURRENCIES=4 scripts/bench_guidellm.sh longctx-32k-phase1-patched-ratio05-b2c1f62c --target http://127.0.0.1:8000 --model Qwen3-4B --processor infer/models/Qwen3-4B`

## Environment

- **Backend:** CUDA
- **Model:** Qwen3-4B
- **Hardware:** NVIDIA L4, 23034 MiB VRAM, driver 580.82.07, nvcc
  `/usr/local/cuda/bin/nvcc` reports 12.8
- **Commit:** `b2c1f62c` repository HEAD; runtime code is unchanged from
  `a7cfcb1e`
- **Feature set:** `cargo build -p infer --release --features cuda`
- **Non-default flags / env vars:** `CUDA_HOME=/usr/local/cuda`,
  `TORCH_CUDA_ARCH_LIST=8.9`, `INFER_TRITON_PYTHON=/usr/bin/python3`,
  `INFER_TILELANG_PYTHON=/usr/bin/python3`,
  `CARGO_TARGET_DIR=/tmp/arle-target`,
  `ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig`
- **Server launch:** `/tmp/arle-target/release/infer --model-path infer/models/Qwen3-4B --port 8000 --kv-cache-dtype fp8 --num-slots 16 --max-seq-len 131072 --mem-fraction-static 0.85 --max-num-batched-tokens 16384 --max-prefill-tokens 16384 --schedule-policy fcfs --decode-headroom-ratio 0.05`
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

## Results — sweep headline table

No valid GuideLLM headline table was produced. The run was terminated after the
service remained stuck beyond the 300s benchmark window; GuideLLM exited with
status `143`.

## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| trace samples | 505 |
| peak active | 4 |
| peak waiting | 1 |
| peak running_batch | 4 |
| peak prefill_queue | 2 |
| peak kv_util | 97.0% |
| `plan_label.idle` | 1 |
| `plan_label.decode` | 261 |
| `plan_label.prefill` | 34 |
| `plan_label.split` | 0 |
| `plan_label.mixed` | 22 |
| `prefix_hit_rate` | peak 0.0%, q75 0.0% |
| `prefix_skip_rate` | peak 0.0% |
| `kv_fetch_q` | 0/16 |
| `kv_fetch_waiters` | 0 |
| `kv_store_q` | 0/16 |
| final observed active | 4 |
| final observed scheduled | 4 |
| final observed decode_rows | 3 |
| final observed prefill_rows | 1 |
| final observed tokens_out | 1032 |
| final observed active_ttft_p50 | 60000.0 ms |

## Results — request accounting

GuideLLM did not emit `benchmarks.json`, so completed/incomplete request
accounting is unavailable. Service-side counters showed `tokens_out=1032` and
`requests=5` at termination, but the active set was still non-empty; these are
not treated as successful benchmark completions.

## Problems

- `decode_headroom_ratio=0.05` entered a hang state: after the benchmark window,
  the service still reported `active=4`, `scheduled=4`, `decode_rows=3`, and
  `prefill_rows=1`.
- Output stopped at `tokens_out=1032`; the last trace samples were unchanged
  while `kv_util` sat at `97.0%`.
- GuideLLM could not naturally finish and was terminated with status `143`.
- This point is worse than the baseline anchor because it produced no valid
  c=4 throughput sample.

## Learnings

- Lowering the ratio to 5% does not solve the near-full KV edge. It can leave
  the scheduler in a mixed decode+prefill state with active requests that do
  not drain.
- The next scan points should use fresh server processes and preserve hang
  traces as first-class failures, not as missing data.

## Δ vs baseline

- **Baseline:** `2026-04-30-bench-guidellm-longctx-32k-phase1-baseline-anchor-0464fb3e.md`
- **Delta table:** patched ratio 5% vs pre-patch three-run anchor mean.

| metric | baseline | now | Δ% |
|---|---|---|---|
| c4 output tok/s | 3.02 | n/a: hang | n/a |
| c4 valid throughput sample | 2/3 baseline runs | 0/1 | regression |
| peak kv_util | 99.0% baseline peak | 97.0% | -2.0 pp |
| `mixed` plan count | present | 22 | qualitative only |
| `split` plan count | 0 | 0 | unchanged |

## Artefacts

- Command: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-patched-ratio05-b2c1f62c/command.txt`
- Log: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-patched-ratio05-b2c1f62c/guidellm.log`
- Service trace (before): `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-patched-ratio05-b2c1f62c/service_stats_before.txt`
- Service trace (during): `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-patched-ratio05-b2c1f62c/service_stats_trace.jsonl`
- Service trace (after): `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-patched-ratio05-b2c1f62c/service_stats_after.txt`
- Service trace (summary): `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-patched-ratio05-b2c1f62c/service_stats_trace_summary.md`

## Notes

- What changed in the code since baseline: `a7cfcb1e` adds
  `decode_headroom_ratio`, charges ratio-scaled decode headroom for live
  decode rows, and skips future decode-tail reservation on truncated prefill
  chunks. `b2c1f62c` only records the 10% regression doc.
- Suspected cause of regression vs SGLang: ratio-only reservation still lacks
  free-plus-evictable admission and adaptive pressure feedback; at 5%, the
  scheduler admits enough work to reach a mixed decode+prefill drain failure.
- Follow-ups: continue the requested scan at 15% and 20% from fresh server
  processes, then summarize the parameter sweep before any further patch.
