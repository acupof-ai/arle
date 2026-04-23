# Qwen3-4B L4 c16 regression after direct T1 and deferred prefetch

## Goal

- Confirm that the direct T1 staged-readmission fast path and deferred slower-tier prefetch stay neutral on the canonical `Qwen3-4B` / L4 `c16` lane.

## Hypothesis

- Canonical random long prompts should remain effectively tier-cold, so throughput should stay within noise versus the previous tier-readmission baseline and tier counters should remain idle.

## Command

Server:

```bash
./target/release/infer \
  --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --port 8065 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384 \
  --trace-output-path bench-output/infer-qwen3-4b-l4-c16-42ce889-tier-prefetch-server/traces
```

Benchmark:

```bash
GUIDELLM__MP_CONTEXT_TYPE=forkserver \
scripts/bench_guidellm.sh qwen3-4b-l4-c16-42ce889-tier-prefetch \
  --target http://127.0.0.1:8065 \
  --model 1cfa9a7208912126459214e8b04321603b3df60c \
  --processor /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --concurrencies 16 \
  --max-seconds 60 \
  --warmup 5 \
  --trace-interval-ms 200
```

## Environment

- **Backend:** CUDA
- **Model:** `Qwen3-4B`
- **Hardware:** NVIDIA L4
- **Commit base:** `42ce889` plus local direct-T1 / deferred-prefetch scheduler changes in this tranche
- **Feature set:** `cargo build --release -p infer --bin infer`
- **Non-default flags / env vars:** `GUIDELLM__MP_CONTEXT_TYPE=forkserver`; server flags `--num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 --chunked-prefill-size 4096 --max-prefill-tokens 16384`
- **Server launch:** direct `infer` invocation above

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile concurrent`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---:|---:|---:|---:|---:|---:|
| `conc16` | `7331.3` | `29844.0` | `59.93` | `59.96` | `118.32` | `0.364` |

Service-side trace summary:

| metric | value |
|---|---:|
| peak active | `10` |
| peak waiting | `6` |
| peak running_batch | `10` |
| peak prefill_queue | `6` |
| peak kv_util | `98.2%` |
| `prefix_hit_rate` | `0.0%` |
| `prefix_skip_rate` | `0.0%` |
| `kv_fetch_q` | `0/16` |
| `kv_store` | `sub:0,done:0,fail:0,rej:0` |

GuideLLM request accounting during the measured window:

| metric | value |
|---|---:|
| completed input tokens | `114716` |
| incomplete input tokens | `32768` |
| completed output tokens | `7140` |
| incomplete output tokens | `510` |

## Problems

- The canonical lane stayed tier-cold: no fetch queue activity, no store queue activity, and no prompt-token skip. This run does not exercise the new tier behavior by design.
- The benchmark ended with `8` incomplete requests (`32768` input tokens, `510` output tokens) because the `60s` wall-clock limit cut the concurrent stream while requests were still in flight.

## Learnings

- The direct T1 and deferred-prefetch scheduler changes are throughput-neutral on the canonical `c16` lane because that workload does not reuse prefixes.
- Tier changes should be judged with dedicated reuse-heavy traces; canonical long-random `c16` remains a scheduler/prefill benchmark, not a tier-readmission benchmark.

## Δ vs baseline

- **Baseline:** `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-l4-c16-tier-readmission-b70c03b.md`

| metric | baseline | now | Δ% |
|---|---:|---:|---:|
| TTFT p50 @ `conc16` | `7779.0` | `7331.3` | `-5.8%` |
| ITL p50 @ `conc16` | `59.90` | `59.93` | `+0.1%` |
| out tok/s @ `conc16` | `119.14` | `118.32` | `-0.7%` |

## Artefacts

- Raw: `bench-output/2026-04-23-qwen3-4b-l4-c16-42ce889-tier-prefetch-run2/benchmarks.json`
- CSV: `bench-output/2026-04-23-qwen3-4b-l4-c16-42ce889-tier-prefetch-run2/benchmarks.csv`
- HTML: `bench-output/2026-04-23-qwen3-4b-l4-c16-42ce889-tier-prefetch-run2/benchmarks.html`
- Service trace (before): `bench-output/2026-04-23-qwen3-4b-l4-c16-42ce889-tier-prefetch-run2/service_stats_before.txt`
- Service trace (during): `bench-output/2026-04-23-qwen3-4b-l4-c16-42ce889-tier-prefetch-run2/service_stats_trace.jsonl`
- Service trace (after): `bench-output/2026-04-23-qwen3-4b-l4-c16-42ce889-tier-prefetch-run2/service_stats_after.txt`
- Service trace (summary): `bench-output/2026-04-23-qwen3-4b-l4-c16-42ce889-tier-prefetch-run2/service_stats_trace_summary.md`

## Notes

- What changed in the code since baseline: host-only staged prefixes can now complete directly from T1, and deferred staged slower-tier fetches can prefetch into T1 before admission.
- Suspected cause of any regression: `n/a`; the measured throughput delta is within noise and the lane remained tier-cold.
- Follow-ups: keep using dedicated tier traces for reuse-heavy validation, and keep throughput work focused on the scheduler/prefill hot path.

