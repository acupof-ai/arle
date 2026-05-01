# P2.3 verifier bit-identity eager guard - guidellm c4, CUDA, 2026-05-01

## Goal

- Fix the P2.3 greedy verifier correctness gap before P2.5 sparse/spec-speed work, and re-check longctx-32k c=4 with a completion-capable envelope.

## Hypothesis

- The observed `spec_decode_correctness` divergence is not MagicDec verifier logic; it is CUDA Graph replay/capture sensitivity in the canary spec path. Forcing speculative verifier decode to run eagerly for one step should restore bit-identity while preserving graph decode for the normal path.
- The previous 60s longctx c=4 smoke produced `0` successful requests because the first c=4 32k batch needs more than 60s, not because speculative decode itself deadlocked.

## Command

```bash
ZIG=$PWD/.toolchains/zig/zig-x86_64-linux-0.16.0/zig \
  CUDA_HOME=/usr/local/cuda \
  TORCH_CUDA_ARCH_LIST=8.9 \
  INFER_TRITON_PYTHON=/usr/bin/python3 \
  INFER_TILELANG_PYTHON=/usr/bin/python3 \
  cargo build --release -p infer --features cuda

./target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8000 \
  --kv-cache-dtype fp8 \
  --num-slots 16 \
  --max-seq-len 131072 \
  --mem-fraction-static 0.85 \
  --max-num-batched-tokens 16384 \
  --max-prefill-tokens 16384 \
  --schedule-policy fcfs \
  --spec-enabled \
  --spec-draft-k 5 \
  --spec-acceptance-threshold 0.3 \
  --spec-draft-model self

WORKLOAD=longctx-32k LONGCTX_CONCURRENCIES=4 LONGCTX_MAX_SECONDS=300 \
  scripts/bench_guidellm.sh p23-verifier-eager-spec-c4-300s \
  --target http://127.0.0.1:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

Control reran the same 300s workload with the same server flags except all `--spec-*` flags removed.

## Environment

- **Backend:** CUDA
- **Model:** Qwen3-4B
- **Hardware:** NVIDIA L4, 23034 MiB, driver 580.82.07
- **Commit:** `5eddaab8+dirty`
- **Feature set:** `cargo build --release -p infer --features cuda`
- **Non-default flags / env vars:** `kv-cache-dtype=fp8`, `num-slots=16`, `max-seq-len=131072`, `mem-fraction-static=0.85`, `max_num_batched_tokens=16384`, `max_prefill_tokens=16384`, `schedule_policy=fcfs`, `spec_enabled=true`, `spec_draft_k=5`, `spec_acceptance_threshold=0.3`, `spec_draft_model=self`

## Results - c4 headline table

GuideLLM wrote complete artifacts but returned invalid because TTFT/ITL p50 were reported as `0.0` despite successful requests. Treat the throughput row below as successful-only accounting, not a canonical GuideLLM pass.

| run | completed req | incomplete req | completed output tokens | wall seconds | effective out tok/s | GuideLLM out tok/s mean | acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|
| spec eager verifier | 28 | 12 | 7168 | 300 | 23.89 | 27.7 | 100.0% |
| no-spec control | 28 | 12 | 7168 | 300 | 23.89 | 28.4 | n/a |

## Results - service-side KV / scheduler metrics

| metric | spec eager verifier | no-spec control |
|---|---:|---:|
| peak active | 4 | 4 |
| peak waiting | 0 | 0 |
| peak prefill_queue | 2 | 2 |
| peak kv_util | 100.0% | 100.0% |
| plan labels | `idle=5,decode=262,prefill=105,mixed=2` | `idle=6,decode=262,prefill=104,mixed=2` |
| spec counters | `draft=1025,verified=1025,accepted=1025` | `draft=0,verified=0,accepted=0` |
| prefix hit rate | 0.0% | 0.0% |
| kv_fetch_waiters | 0/327 | 0/323 |

## Results - request accounting

| metric | spec eager verifier | no-spec control |
|---|---:|---:|
| completed input tokens | 917508 | 917508 |
| incomplete input tokens | 0 in run summary | 0 in run summary |
| completed output tokens | 7168 | 7168 |
| incomplete output tokens | 0 in run summary | 0 in run summary |

## Problems

- This patch fixes bit-identity but does not produce a speculative throughput win. The scheduler P2.3 path is still a single-token target canary, not real K-token self-spec, so `accept_rate=100%` does not imply speedup.
- The spec path is slightly slower than no-spec by GuideLLM mean out tok/s (`27.7` vs `28.4`) because the verifier path forces eager decode for correctness.
- GuideLLM validation still flags TTFT/ITL p50 as `0.0`; successful-only accounting is required for this c4 longctx envelope until the metric exporter/request finalizer issue is isolated.

## Learnings

- `spec_decode_correctness` must compare request-level spec on/off inside one model/scheduler and print first token divergence. The original test compared separate model loads and hid CUDA Graph sensitivity.
- P2.3 correctness requires an eager verifier guard before graph-safe verifier replay is designed.
- `longctx-32k c=4 max_seconds=60` is not a valid smoke envelope for this path; 300s completes 28 requests.

## Delta vs baseline

- **Baseline:** Phase 1 close baseline `26.169 tok/s` mean c4.

| metric | baseline | spec eager verifier | Delta |
|---|---:|---:|---:|
| successful-only effective out tok/s | 26.169 | 23.89 | -8.7% |
| GuideLLM mean out tok/s | 26.169 | 27.7 | +5.9% |
| acceptance rate | n/a | 100.0% | n/a |

## Artefacts

- Spec raw: `bench-output/2026-05-01-p23-verifier-eager-spec-c4-300s/benchmarks.json`
- Spec CSV: `bench-output/2026-05-01-p23-verifier-eager-spec-c4-300s/benchmarks.csv`
- Spec HTML: `bench-output/2026-05-01-p23-verifier-eager-spec-c4-300s/benchmarks.html`
- Spec service trace: `bench-output/2026-05-01-p23-verifier-eager-spec-c4-300s/service_stats_trace_summary.md`
- No-spec raw: `bench-output/2026-05-01-p23-verifier-eager-nospec-c4-300s-control/benchmarks.json`
- No-spec service trace: `bench-output/2026-05-01-p23-verifier-eager-nospec-c4-300s-control/service_stats_trace_summary.md`

## Notes

- Code change since baseline: speculative verifier decode sets a one-shot eager override on the decode context only when every row in the decode batch is eligible for spec decode; normal non-spec decode and mixed opt-out batches continue to use CUDA Graph replay.
- Follow-up: P2.5 must implement real K-token verifier batching before expecting speedup. Do not use this canary's 100% acceptance as a performance signal.
