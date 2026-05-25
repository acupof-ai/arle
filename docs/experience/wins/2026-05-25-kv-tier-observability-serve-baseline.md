# T4b KV-Tier Observability SERVE Baseline

## Context

T4a added KV-tier observability fields without running SERVE because P5 was
using the GPU. T4b runs the >=4k-token SERVE workload after P5 completed, before
any `PrefetchPolicy::Timeout` or policy change.

Related:

- `docs/projects/2026-05-24-opd-mainline-task-backlog.md` T4.
- `docs/experience/wins/2026-05-25-kv-tier-observability-code-patch.md`.
- `docs/projects/tiered-kv-runtime-flow.md`.

## Commands

Build:

```bash
NVCC_CCBIN=g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
cargo build --release -p infer --features cuda --no-default-features
```

Baseline server:

```bash
RUST_LOG=info target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8131 \
  --num-slots 1 \
  --max-seq-len 8192 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 4096 \
  --max-num-batched-tokens 4096 \
  --t1-host-pinned-min-prompt-tokens 4096 \
  --t1-host-pinned-capacity-mb 64 \
  --disk-store-root bench-output/2026-05-25-t4b-kv-tier-observability-baseline/disk-store \
  --trace-output-path bench-output/2026-05-25-t4b-kv-tier-observability-baseline/traces \
  --trace-level basic
```

GuideLLM:

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
./scripts/bench_guidellm.sh t4b-kv-tier-observability-baseline \
  --target http://127.0.0.1:8131 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B \
  --concurrencies 1 \
  --max-seconds 60 \
  --warmup 5 \
  --trace-interval-ms 1000
```

Controlled observability replay used the same binary/model with
`--mem-fraction-static 0.58`, `--t1-host-pinned-capacity-mb 256`, and
session-aware 4096-token `/v1/completions` requests. That replay is a metric
validation control, not a throughput baseline.

## Environment

- Backend: CUDA.
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB.
- Model: `infer/models/Qwen3-4B`.
- Feature set: `--release -p infer --features cuda --no-default-features`.
- KV layout: contiguous BF16, paged pool FP8E4M3.
- Baseline server: 1 slot, 8192 max seq, 4096 prefill chunk, T2 disk tier on.
- Current code commit before this docs entry: `4d0b106`.

## SERVE Baseline

Raw artifacts:
`bench-output/2026-05-25-t4b-kv-tier-observability-baseline-run2/`.

| metric | value |
|---|---:|
| completed requests | 13 |
| input tokens / request | 4097 |
| output tokens / request | 256 |
| TTFT p50 / p99 | 504.1 ms / 510.8 ms |
| ITL p50 / p99 | 15.42 ms / 15.42 ms |
| TPOT mean | 17.33 ms |
| E2E mean / p99 | 4.44 s / 4.44 s |
| output tok/s | 58.2 |
| total tok/s | 989.59 |
| actual req/s | 0.218 |
| peak active / waiting | 1 / 0 |
| peak KV util | 81.1% |
| prefix hit rate | 0.0% |

GuideLLM does not send `session_id`, so it correctly exercised long prefill
serving latency but did not mark blocks `host_swap_eligible`. T1/T2 counters
stayed zero in this baseline:

| metric | value |
|---|---:|
| `demote_to_host_bytes_total` | 0 |
| `store_bytes_total` | 0 |
| `readmission_fetch_wait_us_p50/p99` | null / null |
| `fetch_queue_saturated_fallback_total` | 0 |
| `recompute_advised_fallback_total` | 0 |

## Controlled KV Replay

Raw artifacts:

- `bench-output/2026-05-25-t4b-kv-tier-observability-pressure/`
- `bench-output/2026-05-25-t4b-kv-tier-observability-readmission/`
- `bench-output/2026-05-25-t4b-kv-tier-observability-readmission-4k/`

The final 4k replay used one session-owned 4096-token prompt, then two
concurrent different 4096-token session prompts to create waiting admission
pressure, then replayed the first session prompt.

| metric | value |
|---|---:|
| TokenKVPool size | 594 pages |
| T1 host pool | 256 MiB |
| requests | 4 |
| T0->T1 demote bytes | 214,106,112 |
| T0->T1 demote histogram count / sum | 176 / 669,589 us |
| readmission fetch-wait histogram count / sum | 1 / 58,492 us |
| readmission p50 / p99 bucket | 100,000 us / 100,000 us |
| engine T0 hit rate after replay | 0.25 |
| prefix hit rate after replay | 0.25 |
| host pool high / low pressure ticks | 0 / 269,140 |

The request trace confirms the staged readmission path:

```text
Request 3 -> slot 0 (prompt=4096 tokens, staged_prefix=4096)
Request 3: staged sealed prefix ready, promoted 4096/4096 tokens into T0
Request 3: staged prefix ready in 58.5ms src=h:176/d:0/r:0 waiters=1
Request 3: paged prefix ATTACH 4095/4096 tokens
```

## Metric Verdict

| Metric family | Runtime verdict |
|---|---|
| Per-tier hit rate | Validated; `T0` rose to `0.25` after staged replay. |
| T0->T1 demote latency + bytes | Validated; 176 demote observations, 214,106,112 bytes. |
| Staged-readmission fetch wait | Validated; one T1 readmission, 58.492 ms raw wait. |
| Queue-saturated fallback | Exposed, not hit; queue never saturated. |
| Recompute-advised fallback | Exposed, not hit; this workload had no recompute-advised lookup. |
| Host pool pressure ticks | Exposed; low-pressure ticks incremented, high-pressure stayed zero. |
| T1->T2/T3 store latency + bytes | Exposed, not hit; session-owned T1 blocks remain protected by session refs, so idle drain did not submit a disk store despite `--disk-store-root`. |

## Problems

- A normal GuideLLM run is not enough to test T1 because GuideLLM does not send
  `session_id`; long prompts alone do not mark blocks `host_swap_eligible`.
- A 64 MiB T1 pool control demoted only 55,959,552 bytes, then pressure fallback
  dropped GPU blocks and destroyed the session readmission opportunity. This is
  why the final readmission control used a 256 MiB T1 pool.
- T1->T2 store did not fire in any session-preserving control. This is a real
  baseline finding: the live path protects session-owned host blocks from store
  drain. Do not claim T2 store behavior from this bench.

## Rule

For KV-tier benchmarks, separate serving throughput from tier-state validation:
GuideLLM supplies the long-prompt latency baseline, while session-aware HTTP
replay is required to license T1 demotion and staged readmission counters.
