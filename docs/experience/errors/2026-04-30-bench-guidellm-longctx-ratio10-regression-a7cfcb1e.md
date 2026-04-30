# guidellm regression longctx-32k-phase1-patched-ratio10-a7cfcb1e — 2026-04-30

This entry records a failed Phase 1.5 bench point. It uses the canonical
GuideLLM result sections from the wins template, but is intentionally stored
under `docs/experience/errors/` because the run regressed c=4 throughput.

## Goal

- First patched Phase 1.5 longctx-32k run for the SGLang-style
  `decode_headroom_ratio=0.10` admission policy at commit `a7cfcb1e`.

## Hypothesis

- Charging decode headroom only on final prefill chunks, and at 10% of clipped
  remaining decode tokens, should improve c=4 progress against the pre-patch
  anchor while preserving c=1 parity.

## Command

```bash
WORKLOAD=longctx-32k scripts/bench_guidellm.sh \
  longctx-32k-phase1-patched-ratio10-a7cfcb1e \
  --target http://127.0.0.1:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

Invoked via: `WORKLOAD=longctx-32k scripts/bench_guidellm.sh longctx-32k-phase1-patched-ratio10-a7cfcb1e --target http://127.0.0.1:8000 --model Qwen3-4B --processor infer/models/Qwen3-4B`

## Environment

- **Backend:** CUDA
- **Model:** Qwen3-4B
- **Hardware:** NVIDIA L4, 23034 MiB VRAM, driver 580.82.07, nvcc
  `/usr/local/cuda/bin/nvcc` reports 12.8
- **Commit:** a7cfcb1e
- **Feature set:** `cargo build -p infer --release --features cuda`
- **Non-default flags / env vars:** `CUDA_HOME=/usr/local/cuda`,
  `TORCH_CUDA_ARCH_LIST=8.9`, `INFER_TRITON_PYTHON=/usr/bin/python3`,
  `INFER_TILELANG_PYTHON=/usr/bin/python3`,
  `CARGO_TARGET_DIR=/tmp/arle-target`,
  `ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig`
- **Server launch:** `/tmp/arle-target/release/infer --model-path infer/models/Qwen3-4B --port 8000 --kv-cache-dtype fp8 --num-slots 16 --max-seq-len 131072 --mem-fraction-static 0.85 --max-num-batched-tokens 16384 --max-prefill-tokens 16384 --schedule-policy fcfs --decode-headroom-ratio 0.10`

## Canonical params (resolved by wrapper)

- `--profile concurrent`
- `--data prompt_tokens=32768,prompt_tokens_stdev=1,prompt_tokens_min=32768,prompt_tokens_max=32768,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256`
- `--max-seconds 300`
- `--random-seed 20260416`
- `--rate 1,4`
- `--outputs json --outputs csv --outputs html`
- Workload: `longctx-32k`
- Wrapper: `scripts/bench_guidellm.sh <backend-label> --workload longctx-32k`

## Results — sweep headline table

| rate | TTFT mean | TTFT std | TTFT p50 | TTFT p99 | TPOT mean | ITL mean | ITL std | ITL p50 | ITL p95 | ITL p99 | ITL max | E2E mean | E2E p99 | conc p50 | out tok/s | total tok/s | in tok/s | total in | total out | req/s actual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conc1 | 12518.7 | 53.7 | 12519 | 12596 | 104.54 | 55.86 | 0.09 | 55.86 | 56.09 | 56.09 | 56.09 | 26.76 | 26.86 | 1 | 9.98 | 1288.01 | 1346 | 360459 | 2816 | 0.037 |
| conc4 | 31036.8 | 0 | 31036.8 | 31036.8 | 2024.12 | 1910.34 | 0 | 1910.34 | 1910.34 | 1910.34 | 1910.34 | 518.18 | 518.18 | 1 | 0.53 | 67.79 | 0 | 32769 | 256 | 0 |

## Service Trace Peaks

- Poll interval: `1000ms`
- Samples: `843` (ok: `843`, failed: `0`)
- Peak waiting: `2`
- Peak active: `4`
- Peak running_batch: `3`
- Peak prefill_queue: `1`
- Plan labels: `idle=5708338`, `decode=3044`, `prefill=222`, `split=0`, `mixed=25`
- Peak kv_util: `99.0%`
- Prefix hit rate: peak `0.0%`, q75 `0.0%`
- Prefix skip rate peak: `0.0%`
- Peak mem: `n/a` (delta vs before: `n/a`)
- Server ttft_p99 peak: `n/a`
- KV fetch queue samples >0: `0/0`
- KV fetch waiter samples >0: `0/843`
- KV store queue samples >0: `0/0`
- Tier wait peaks: fetch `n/a`, store `n/a`

## Service Trace Distribution


| metric | q25 | q50 | q75 | q99 | peak |
|---|---:|---:|---:|---:|---:|
| waiting | 0 | 0 | 1 | 1 | 2 |
| kv_util | 76.1% | 86.5% | 95.7% | 98.8% | 99.0% |


## Service Token Counters


| metric | q25 | q50 | q75 | q99 | peak |
|---|---:|---:|---:|---:|---:|
| decode_tokens | 0 | 2 | 3 | 3 | 3 |
| prefill_tokens | 0 | 2048 | 2048 | 6144 | 6144 |
| tokens_out | 1800 | 2826 | 2826 | 2826 | 3082 |


## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| peak active | 4 |
| peak waiting | 2 |
| peak prefill_queue | 1 |
| peak kv_util | 99.0% |
| `plan_label.idle` | 5708338 |
| `plan_label.decode` | 3044 |
| `plan_label.prefill` | 222 |
| `plan_label.split` | 0 |
| `plan_label.mixed` | 25 |
| `prefix_hit_rate` | peak 0.0%, q75 0.0% |
| `prefix_skip_rate` | peak 0.0% |
| `kv_fetch_q` | 0/16 |
| `kv_fetch_waiters` | 0 |
| `kv_store_q` | 0/16 |
| `kv_store` | sub:0, done:0, fail:0, rej:0 |
| `kv_bp` | fetch:0, store:0 |
| `tier_recall` | n/a |
| `tier_src` | n/a |
| `tier_promoted` | n/a |
| `tier_fallback` | n/a |

## Results — request accounting

| metric | value |
|---|---:|
| completed input tokens | 393228 |
| incomplete input tokens | 131072 |
| completed output tokens | 3072 |
| incomplete output tokens | 501 |

## Problems

- The 10% ratio point is a regression vs the pre-patch three-run anchor:
  c=4 output throughput fell to `0.53` tok/s vs anchor mean `3.02`.
- c=4 completed only one request and left `501` incomplete c=4 output tokens.
- Peak KV util still reached `99.0%`; the patch did not escape the near-full
  KV edge.
- `mixed=25` and `split=0` show the same qualitative overlap shape as the
  baseline, but with worse c=4 throughput.

## Learnings

- The SGLang-shaped reservation change alone is not sufficient at
  `decode_headroom_ratio=0.10`.
- The next scan values should still run as requested, but this point should be
  treated as failed unless a later ratio clears both throughput and completion
  gates.

## Δ vs baseline

- **Baseline:** `2026-04-30-bench-guidellm-longctx-32k-phase1-baseline-anchor-0464fb3e.md`
- **Delta table:** patched ratio 10% vs pre-patch three-run anchor mean.

| metric | baseline | now | Δ% |
|---|---|---|---|
| c1 output tok/s | 9.77 | 9.98 | +2.1% |
| c1 TTFT p50 | 12496.3 ms | 12519.0 ms | +0.2% |
| c4 output tok/s | 3.02 | 0.53 | -82.5% |
| c4 successful requests | 1.33 | 1 | -24.8% |
| c4 completed output tokens | 341.3 | 256 | -25.0% |
| c4 incomplete output tokens | 186.0 | 501 | +169.4% |
| c4 vs SGLang 16.27 tok/s | 0.186x | 0.033x | -82.3% |

## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-patched-ratio10-a7cfcb1e/benchmarks.json`
- CSV:  `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-patched-ratio10-a7cfcb1e/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-patched-ratio10-a7cfcb1e/benchmarks.html`
- Service trace (before): `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-patched-ratio10-a7cfcb1e/service_stats_before.txt`
- Service trace (during): `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-patched-ratio10-a7cfcb1e/service_stats_trace.jsonl`
- Service trace (after):  `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-patched-ratio10-a7cfcb1e/service_stats_after.txt`
- Service trace (summary): `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-patched-ratio10-a7cfcb1e/service_stats_trace_summary.md`

## Notes

- What changed in the code since baseline: `a7cfcb1e` adds
  `decode_headroom_ratio`, charges ratio-scaled decode headroom for live
  decode rows, and skips future decode-tail reservation on truncated prefill
  chunks.
- Suspected cause of regression vs SGLang: ratio-only admission still uses
  free pages rather than `free + evictable`, lacks adaptive feedback from
  decode retraction, and still reaches 99% KV util under c=4.
- Follow-ups: run the remaining ratio scan points `{5%, 15%, 20%}` from fresh
  server processes; if all fail, move to eviction/adaptive-ratio design before
  mixed-mode acceptance.

## Service Trace

- Poll interval: `1000ms`
- Before: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-patched-ratio10-a7cfcb1e/service_stats_before.txt`
- During: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-patched-ratio10-a7cfcb1e/service_stats_trace.jsonl`
- After: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-patched-ratio10-a7cfcb1e/service_stats_after.txt`
- Summary: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-patched-ratio10-a7cfcb1e/service_stats_trace_summary.md`
