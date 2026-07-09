# W4 Harness Tool Resume Generation - agent-load benchmark, agent-w4-tool-resume, arle-local, 2026-05-02

> Workload contract:
> [`docs/plans/2026-05-02-agent-load-bench-spec.md`](../../plans/2026-05-02-agent-load-bench-spec.md).

## Goal

- Regression/pending-remote: verify the W4 harness generates the canonical
  tool-call resume trace shape before ARLE/SGLang dry-run benchmarking.

## Hypothesis

- `scripts/bench_agent_trace.py --workload agent-w4-tool-resume` should
  generate 128 sessions, one unscored 8k warmup request per session, and one
  scored tool-resume request per session with a 256-token injected tool output.

## Command

Server:

```bash
pending-remote: no local ARLE/SGLang server was launched for this harness-only tranche
```

Client:

```bash
python3 scripts/bench_agent_trace.py \
  --workload agent-w4-tool-resume \
  --generate-only \
  --trace-out /tmp/agent-w4-tool-resume.jsonl
```

Required guidellm regression attempt:

```bash
scripts/bench_guidellm.sh agent-load-w4-harness-regression \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

Result: blocked locally before server check because `guidellm` is not on
`PATH`; remote runner must install `pip install -e .[bench]` and rerun.

## Environment

- **Workload:** agent-w4-tool-resume
- **Backend / engine:** arle-local harness
- **Model:** Qwen/Qwen3-4B target shape; no model served locally
- **Tokenizer / processor:** `infer/models/Qwen3-4B` for the pending guidellm command
- **Hardware:** local workspace; GPU not used
- **Commit:** pending at entry creation; committed with the W4 harness tranche
- **Feature set:** Python harness only
- **KV dtype / cache mode:** pending-remote
- **Session / prefix flags:** every generated request carries `session_id`;
  resume turns are tagged `resume` and use `request_after=true`
- **Non-default flags / env vars:** none

## Workload Params

| field | value |
|---|---|
| seed | `20260502` |
| global concurrency | `8` |
| sessions | `128` |
| scored turns | `128` resume turns |
| prompt shape | base `8192 +/- 64`, tool output `256 +/- 16` tokenish words |
| max output tokens | warmup `64`, resume `256` |
| warm/cold mix | n/a |
| tool output tokens | `256 +/- 16` |
| run cap | full trace completion when replayed |

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

Generation validation:

| metric | value |
|---|---:|
| generated sessions | `128` |
| warmup turns | `128` |
| scored resume turns | `128` |
| warmup max_tokens | `64` |
| resume max_tokens | `256` |

## Results - W3 Warm/Cold

| metric | warm | cold |
|---|---:|---:|
| scored turns | n/a | n/a |
| TTFT p50 (ms) | n/a | n/a |
| TTFT p99 (ms) | n/a | n/a |
| output tok/s | n/a | n/a |

## Results - W4 Resume

| metric | value |
|---|---:|
| resume TTFT p50 (ms) | pending-remote |
| resume TTFT p99 (ms) | pending-remote |
| resume E2E p50 (ms) | pending-remote |
| resume E2E p99 (ms) | pending-remote |
| cold 8k TTFT p99 (ms) | pending-remote |
| matched prefix tokens | pending-remote |
| avoided-prefill ratio | pending-remote |

## Results - Service-Side Cache / Scheduler

| metric | value |
|---|---:|
| peak active | pending-remote |
| peak waiting | pending-remote |
| peak prefill_queue | pending-remote |
| peak kv_util | pending-remote |
| `prefix_hit_rate` | pending-remote |
| `prefix_skip_rate` | pending-remote |
| `session_affinity_hit` | pending-remote |
| `session_affinity_miss` | pending-remote |
| `tool_resume_count` | pending-remote |
| `tool_resume_prefill_tokens` | pending-remote |
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
| ARLE | pending-remote | pending | pending | pending | pending | pending |
| SGLang | pending-remote | pending | pending | pending | pending | pending |
| vLLM | pending-remote | pending | pending | pending | pending | pending |
| TensorRT-LLM | pending-remote | pending | pending | pending | pending | pending |
| Mooncake | pending-remote | pending | pending | pending | pending | pending |

Mission margin:

```text
pending-remote; no mission claim from this harness tranche
```

## Problems

- Local `scripts/bench_guidellm.sh` regression attempt exited 2 because
  `guidellm` is not installed on `PATH`.
- This tranche validates trace generation only; ARLE and SGLang dry-run
  artefacts are still required before A4 W4 is complete.

## Learnings

- The existing trace replayer can model tool resume without server-side
  scaffolding by marking a `tool` message with `request_after=true`; the
  service still decides whether it supports that transcript shape.

## Delta Vs Baseline

- **Baseline:** first W4 harness trace-generation entry.

| metric | baseline | now | delta |
|---|---:|---:|---:|
| output tok/s | n/a | pending-remote | n/a |
| TTFT p99 | n/a | pending-remote | n/a |
| E2E p99 | n/a | pending-remote | n/a |

## Artefacts

- Raw turns: pending-remote
- Client summary: pending-remote
- Server launch: pending-remote
- Engine metadata: pending-remote
- Service trace before: pending-remote
- Service trace during: pending-remote
- Service trace after: pending-remote
- Service trace summary: pending-remote
- Local generation sample: `/tmp/agent-w4-tool-resume.jsonl`

## Notes

- What changed in the code since baseline: added W4 trace generation and
  turn-level `request_after` / `max_tokens` replay metadata to
  `scripts/bench_agent_trace.py`.
- Suspected cause of any regression: n/a until ARLE/SGLang replay runs.
- Follow-ups: run ARLE and SGLang W4 dry runs with `guidellm` installed and a
  live server.
