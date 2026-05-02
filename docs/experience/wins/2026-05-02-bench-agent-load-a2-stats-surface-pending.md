# A2 Stats Surface - agent-load benchmark, W3/W4 service stats, arle-local, 2026-05-02

> Status: `pending-remote`. Runtime stats fields landed locally; canonical
> GuideLLM regression needs a live ARLE server on a GPU bench host.
> Workload contract:
> [`docs/plans/2026-05-02-agent-load-bench-spec.md`](../../plans/2026-05-02-agent-load-bench-spec.md).

## Goal

- Regression/pending-remote: expose the A2 server-side stats needed by W3/W4
  clients without changing A1 admission, A3 tool injection, or A5 baselines.

## Hypothesis

- `/metrics` and `/v1/stats?format=json` should expose prefix reuse,
  session-affinity, matched-prefix, and resume-prefill counters with no
  measurable throughput change; the local no-cuda tests should pass.

## Command

Server:

```bash
pending-remote: no local GPU ARLE server was running at http://localhost:8000
```

Client:

```bash
PATH="$HOME/.local/bin:$PATH" scripts/bench_guidellm.sh a2-stats-surface-regression \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

Result:

```text
error: server not running at http://localhost:8000 - start it with scripts/start_infer.sh first
```

Local verification:

```bash
ZIG=/root/.local/lib/python3.12/site-packages/ziglang/zig \
  cargo test --release -p infer --no-default-features --features no-cuda
ZIG=/root/.local/lib/python3.12/site-packages/ziglang/zig \
  cargo clippy --release -p infer --no-default-features --features no-cuda -- -D warnings
ZIG=/root/.local/lib/python3.12/site-packages/ziglang/zig \
  cargo check --release -p infer --no-default-features --features metal,no-cuda
```

## Environment

- **Workload:** W3/W4 service stats surface; no trace replay locally
- **Backend / engine:** arle-local no-cuda verification; ARLE CUDA bench pending
- **Model:** Qwen/Qwen3-4B target shape; no model served locally
- **Tokenizer / processor:** `infer/models/Qwen3-4B` for the pending command
- **Hardware:** local CPU workspace; GPU not used
- **Commit:** this A2 stats-surface commit
- **Feature set:** `cargo test --release -p infer --no-default-features --features no-cuda`
- **KV dtype / cache mode:** pending-remote
- **Session / prefix flags:** existing `session_id` prefix-cache metadata only;
  no A1 admission policy change
- **Non-default flags / env vars:** `ZIG=/root/.local/lib/python3.12/site-packages/ziglang/zig`
  for local `kv-native-sys` builds

## Workload Params

| field | value |
|---|---|
| seed | `20260502` |
| global concurrency | pending-remote |
| sessions | pending-remote |
| scored turns | pending-remote |
| prompt shape | pending-remote |
| max output tokens | pending-remote |
| warm/cold mix | pending-remote |
| tool output tokens | pending-remote |
| run cap | pending-remote |

## Results - Headline

| metric | value |
|---|---:|
| successful scored turns | pending-remote |
| incomplete scored turns | pending-remote |
| successful output tok/s | pending-remote |
| TTFT p50 (ms) | pending-remote |
| TTFT p99 (ms) | pending-remote |
| ITL p50 (ms) | pending-remote |
| ITL p99 (ms) | pending-remote |
| E2E p50 (ms) | pending-remote |
| E2E p99 (ms) | pending-remote |

## Results - W3 Warm/Cold

| metric | warm | cold |
|---|---:|---:|
| scored turns | pending-remote | pending-remote |
| TTFT p50 (ms) | pending-remote | pending-remote |
| TTFT p99 (ms) | pending-remote | pending-remote |
| output tok/s | pending-remote | pending-remote |

## Results - W4 Resume

| metric | value |
|---|---:|
| resume TTFT p50 (ms) | pending-remote |
| resume TTFT p99 (ms) | pending-remote |
| resume E2E p50 (ms) | pending-remote |
| resume E2E p99 (ms) | pending-remote |
| cold 8k TTFT p99 (ms) | pending-remote |
| matched prefix tokens | `/v1/stats.matched_prefix_tokens` when replayed |
| avoided-prefill ratio | `1 - resume_prefill_tokens / cold_prefill_tokens` when replayed |

## Results - Service-Side Cache / Scheduler

| metric | value |
|---|---:|
| peak active | pending-remote |
| peak waiting | pending-remote |
| peak prefill_queue | pending-remote |
| peak kv_util | pending-remote |
| `prefix_hit_rate` | exposed |
| `prefix_skip_rate` | exposed |
| `session_affinity_hit` | exposed |
| `session_affinity_miss` | exposed |
| `matched_prefix_tokens` | exposed |
| `resume_prefill_tokens` | exposed |
| `tool_resume_count` | n/a |
| `tool_resume_prefill_tokens` | `resume_prefill_tokens` |
| `kv_fetch_q` | pending-remote |
| `kv_fetch_waiters` | pending-remote |
| `kv_store_q` | pending-remote |
| `kv_store` | pending-remote |
| `kv_bp` | pending-remote |
| `tier_recall` | pending-remote |
| `tier_src` | pending-remote |
| `tier_promoted` | pending-remote |
| `tier_fallback` | pending-remote |

## Four-Engine Comparison

| engine | commit/tag | output tok/s | TTFT p99 (ms) | E2E p99 (ms) | cache report | raw artefact |
|---|---|---:|---:|---:|---|---|
| ARLE | pending-remote | pending | pending | pending | A2 fields exposed | pending |
| SGLang | pending-remote | pending | pending | pending | pending | pending |
| vLLM | pending-remote | pending | pending | pending | pending | pending |
| TensorRT-LLM | pending-remote | pending | pending | pending | pending | pending |
| Mooncake | pending-remote | pending | pending | pending | pending | pending |

Mission margin:

```text
pending-remote; no performance claim from this stats-surface tranche
```

## Problems

- First local `scripts/bench_guidellm.sh` attempt exited 2 because `guidellm`
  was not on `PATH`; `pip install -e .[bench]` installed `guidellm 0.6.0`.
- Second local attempt exited 2 because no ARLE server was running at
  `http://localhost:8000`.
- `cargo check --features metal,no-cuda` exited 101 on this Linux host while
  building `mlx-sys` because CMake could not find `LAPACK_INCLUDE_DIRS`; rerun
  Metal typecheck on the Apple Silicon bench host.

## Learnings

- The W3/W4 client can fill cache validity rows from `/v1/stats?format=json`
  without inventing A1/A3 placeholder behavior.

## Delta Vs Baseline

- **Baseline:** pending latest ARLE W3/W4 replay on the selected GPU host.

| metric | baseline | now | delta |
|---|---:|---:|---:|
| output tok/s | pending | pending-remote | pending |
| TTFT p99 | pending | pending-remote | pending |
| E2E p99 | pending | pending-remote | pending |

## Artefacts

- Raw turns: pending-remote
- Client summary: pending-remote
- Server launch: pending-remote
- Engine metadata: pending-remote
- Service trace before: pending-remote
- Service trace during: pending-remote
- Service trace after: pending-remote
- Service trace summary: pending-remote

## Notes

- What changed in the code since baseline: added request/session cache stats
  rendering and scheduler prefix accounting for W3/W4 service-side rows.
- Suspected cause of any regression: none expected; the scheduler still follows
  the existing prefix path and only records counters after the decision.
- Follow-ups: run the pending GuideLLM regression on a live ARLE bench host.
