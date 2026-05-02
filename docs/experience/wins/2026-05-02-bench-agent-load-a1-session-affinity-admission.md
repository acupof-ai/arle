# A1 Session-Affinity Admission - agent-load benchmark, W3, arle-cuda-l4, 2026-05-02

> Status: local L4 W3 entrance run complete. Workload contract:
> [`docs/plans/2026-05-02-agent-load-bench-spec.md`](../../plans/2026-05-02-agent-load-bench-spec.md).

## Goal

- Optimization: prefer same-session W3 warm turns during CUDA admission when
  RadixCache confirms the matched prefix still belongs to the same `session_id`.

## Hypothesis

- Same-priority warm-session requests should bypass cold head-of-line requests
  only when RadixCache block metadata proves a same-session warm prefix, keeping
  W3 warm TTFT p99 bounded without consuming A2 telemetry as an admission input.

## Command

Server:

```bash
ln -sfn Qwen3-4B models/default
CUDA_HOME=/usr/local/cuda \
CARGO_HOME=/tmp/cargo-home-local \
PEGAINFER_CUDA_SM=89 \
LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64 \
ZIG=/root/.local/lib/python3.12/site-packages/ziglang/zig \
INFER_TILELANG_PYTHON=/usr/bin/python3 \
./target/release/infer --model-path models/default --port 8000 \
  --num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 \
  --max-num-batched-tokens 16384 --max-prefill-tokens 16384 \
  --schedule-policy fcfs
```

Client:

```bash
python3 scripts/bench_agent_trace.py \
  --workload agent-w3-short-multiturn \
  --server http://localhost:8000 \
  --label a1-session-affinity-w3 \
  --out bench-output/2026-05-02-agent-load-a1-session-affinity-w3/results.json \
  --trace-out bench-output/2026-05-02-agent-load-a1-session-affinity-w3/trace.jsonl
```

Local verification:

```bash
cargo fmt --check
ZIG=/root/.local/lib/python3.12/site-packages/ziglang/zig \
  cargo check -p infer --no-default-features --features no-cuda
ZIG=/root/.local/lib/python3.12/site-packages/ziglang/zig \
  cargo check -p infer --no-default-features --features cuda,no-cuda
ZIG=/root/.local/lib/python3.12/site-packages/ziglang/zig \
  cargo test --release -p infer --no-default-features --features no-cuda
ZIG=/root/.local/lib/python3.12/site-packages/ziglang/zig \
  cargo check -p infer --no-default-features --features cuda,no-cuda --tests
```

## Environment

- **Workload:** `agent-w3-short-multiturn`
- **Backend / engine:** arle-cuda
- **Model:** Qwen3-4B Instruct (`models/default -> models/Qwen3-4B`)
- **Tokenizer / processor:** `infer/models/Qwen3-4B`
- **Hardware:** NVIDIA L4 24GB, driver 580.82.07, CUDA 13.0, SM 89
- **Commit:** `b1716819`
- **Feature set:** release CUDA binary, `--no-default-features --features cuda`
- **KV dtype / cache mode:** auto FP8E4M3 paged KV, RadixCache on
- **Session / prefix flags:** HTTP `session_id`; A1 reads RadixCache block metadata directly
- **Python tools:** `tilelang 0.1.9`

## Workload Params

| field | value |
|---|---|
| seed | `20260502` |
| global concurrency | `16` |
| sessions | `64` warm + `64` cold distractor |
| scored turns | `256` warm + `64` cold |
| prompt shape | 1024-token base + 64-token warm tails |
| max output tokens | `64` |
| warm/cold mix | `80%` warm / `20%` cold |
| tool output tokens | n/a |
| run cap | full trace completion |

## Results - Headline

| metric | value |
|---|---:|
| successful scored turns | 320 / 320 |
| incomplete scored turns | 0 |
| successful output tok/s | 154.7 |
| TTFT p50 (ms) | 241.7 |
| TTFT p99 (ms) | 4514.0 |
| ITL p50 (ms) | 50.5 |
| ITL p99 (ms) | 53.9 |
| E2E p50 (ms) | 4128.0 |
| E2E p99 (ms) | 12106.5 |

## Results - W3 Warm/Cold

| metric | warm | cold |
|---|---:|---:|
| scored turns | 256 | 64 |
| TTFT p50 (ms) | 233.6 | 638.9 |
| TTFT p99 (ms) | 887.2 | 6740.2 |
| output tok/s (sum-wall) | 14.38 | 9.11 |

## Results - W4 Resume

| metric | value |
|---|---:|
| resume TTFT p50 (ms) | n/a |
| resume TTFT p99 (ms) | n/a |
| resume E2E p50 (ms) | n/a |
| resume E2E p99 (ms) | n/a |
| cold 8k TTFT p99 (ms) | n/a |
| matched prefix tokens | n/a |
| avoided-prefill ratio | n/a |

## Results - Service-Side Cache / Scheduler

| metric | value |
|---|---:|
| peak active | 16 |
| peak waiting | 0 in final snapshot |
| peak prefill_queue | final 0 |
| peak kv_util | final 69.2% |
| `prefix_hit_rate` | 95.8% |
| `prefix_skip_rate` | 57.9% |
| `session_affinity_hit` | 368 |
| `session_affinity_miss` | 16 |
| `matched_prefix_tokens` | final request 1392 |
| `resume_prefill_tokens` | final request 113 |
| `tool_resume_count` | n/a |
| `tool_resume_prefill_tokens` | n/a |
| `kv_fetch_q` | `0/16` |
| `kv_fetch_waiters` | `0` |
| `kv_store_q` | `0/16` |
| `kv_store` | `sub:0,done:0,fail:0,rej:0` |
| `kv_bp` | `fetch:0,store:0` |
| `tier_recall` | n/a |
| `tier_src` | n/a |
| `tier_promoted` | n/a |
| `tier_fallback` | n/a |

## Four-Engine Comparison

| engine | commit/tag | output tok/s | TTFT p99 (ms) | E2E p99 (ms) | cache report | raw artefact |
|---|---|---:|---:|---:|---|---|
| ARLE | `b1716819` | 154.7 | 4514.0 | 12106.5 | A1 + A2 fields | `bench-output/2026-05-02-agent-load-a1-session-affinity-w3/results.json` |
| SGLang | pending | pending | pending | pending | pending | pending |
| vLLM | pending | pending | pending | pending | pending | pending |
| TensorRT-LLM | pending | pending | pending | pending | pending | pending |
| Mooncake | pending | pending | pending | pending | pending | pending |

Mission margin:

```text
W3 competitor margin is not claimed in A1. A1 entrance signal is local ARLE
warm TTFT p99 = 887.2 ms with 0 failed scored turns.
```

## Problems

- `scripts/bench_agent_trace.py` hard-codes request model `"default"`. To keep
  A1 scoped to `infer/src/scheduler/`, the local run served the same Qwen3-4B
  weights through `models/default -> Qwen3-4B` instead of editing the harness.
- The A2 `session_affinity_hit/miss` counters are outcome telemetry
  ("session-tagged prefix hit/miss"), not the A1 routing signal. A1 consumes
  RadixCache block metadata directly.
- The attempted `cargo test -p infer --no-default-features --features cuda,no-cuda
  session_affinity --lib` link failed because `no-cuda` skips CUDA C objects.
  The required CUDA type-check commands passed.

## Learnings

- A same-priority local promotion is enough to expose warm-session preference
  without changing explicit request priority semantics.
- Reading RadixCache block metadata avoids coupling admission to stats names
  that may remain outcome-oriented.

## Delta Vs Baseline

- **Baseline:** first local W3 run after the remote-CUDA-box correction.

| metric | baseline | now | delta |
|---|---:|---:|---:|
| output tok/s | n/a | 154.7 | n/a |
| warm TTFT p99 | n/a | 887.2 ms | n/a |
| cold TTFT p99 | n/a | 6740.2 ms | n/a |

## Artefacts

- Raw turns / client summary: `bench-output/2026-05-02-agent-load-a1-session-affinity-w3/results.json`
- Generated trace: `bench-output/2026-05-02-agent-load-a1-session-affinity-w3/trace.jsonl`
- Server launch log: `bench-output/server-logs/2026-05-02T07-26-42-port8000-a1-w3-default.log`

## Notes

- What changed in the code since baseline: CUDA scheduler admission now records
  a candidate hint when matched radix prefix blocks carry the same `session_id`,
  then selects that same-priority candidate before a cold head.
- Suspected cause of any regression: cold p99 can rise if warm bursts dominate
  normal-priority admission; this run keeps cold completion successful.
- Follow-ups: run pinned SGLang/vLLM/TensorRT-LLM/Mooncake W3 baselines before
  claiming world-first margin.
