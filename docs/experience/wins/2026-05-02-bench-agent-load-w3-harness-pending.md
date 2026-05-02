# W3 Harness Trace Generation - agent-load benchmark, agent-w3-short-multiturn, arle-local, 2026-05-02

> Workload contract:
> [`docs/plans/2026-05-02-agent-load-bench-spec.md`](../../plans/2026-05-02-agent-load-bench-spec.md).

## Goal

- Regression/pending-remote: verify the W3 harness generates the canonical
  short-prompt multi-turn trace shape before ARLE/SGLang dry-run benchmarking.

## Hypothesis

- `scripts/bench_agent_trace.py --workload agent-w3-short-multiturn` should
  generate 64 warm sessions, 64 cold sessions, 256 scored warm turns, 64 scored
  cold turns, and an exact 80% scored warm mix without touching runtime code.

## Command

Server:

```bash
pending-remote: no local ARLE/SGLang server was launched for this harness-only tranche
```

Client:

```bash
python3 scripts/bench_agent_trace.py \
  --workload agent-w3-short-multiturn \
  --generate-only \
  --trace-out /tmp/agent-w3-short-multiturn.jsonl
```

Required guidellm regression attempt:

```bash
scripts/bench_guidellm.sh agent-load-w3-harness-regression \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

Result: blocked locally before server check because `guidellm` is not on
`PATH`; remote runner must install `pip install -e .[bench]` and rerun.

## Environment

- **Workload:** agent-w3-short-multiturn
- **Backend / engine:** arle-local harness
- **Model:** Qwen/Qwen3-4B target shape; no model served locally
- **Tokenizer / processor:** `infer/models/Qwen3-4B` for the pending guidellm command
- **Hardware:** local workspace; GPU not used
- **Commit:** pending at entry creation; committed with the W3 harness tranche
- **Feature set:** Python harness only
- **KV dtype / cache mode:** pending-remote
- **Session / prefix flags:** every generated request carries `session_id`;
  generated turns are tagged `warmup`, `warm`, or `cold`
- **Non-default flags / env vars:** none

## Workload Params

| field | value |
|---|---|
| seed | `20260502` |
| global concurrency | `16` |
| sessions | `128` (`64` warm + `64` cold) |
| scored turns | `320` (`256` warm + `64` cold) |
| prompt shape | base `1024 +/- 32`, appended warm user tail `64 +/- 8` tokenish words |
| max output tokens | `64` |
| warm/cold mix | `256 / 320 = 80%` scored warm |
| tool output tokens | n/a |
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
| warmup turns | `64` |
| scored warm turns | `256` |
| scored cold turns | `64` |
| scored warm ratio | `0.800` |

## Results - W3 Warm/Cold

| metric | warm | cold |
|---|---:|---:|
| scored turns | `256` | `64` |
| TTFT p50 (ms) | pending-remote | pending-remote |
| TTFT p99 (ms) | pending-remote | pending-remote |
| output tok/s | pending-remote | pending-remote |

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
| peak active | pending-remote |
| peak waiting | pending-remote |
| peak prefill_queue | pending-remote |
| peak kv_util | pending-remote |
| `prefix_hit_rate` | pending-remote |
| `prefix_skip_rate` | pending-remote |
| `session_affinity_hit` | pending-remote |
| `session_affinity_miss` | pending-remote |
| `tool_resume_count` | n/a |
| `tool_resume_prefill_tokens` | n/a |
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
  artefacts are still required before A4 W3 is complete.

## Learnings

- The W3 trace can be generated without runtime scaffolding: one warmup turn
  per warm session plus four scored same-session turns gives the required
  80/20 scored warm/cold mix.

## Delta Vs Baseline

- **Baseline:** first W3 harness trace-generation entry.

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
- Local generation sample: `/tmp/agent-w3-short-multiturn.jsonl`

## Notes

- What changed in the code since baseline: added W3 trace generation to
  `scripts/bench_agent_trace.py`.
- Suspected cause of any regression: n/a until ARLE/SGLang replay runs.
- Follow-ups: run ARLE and SGLang W3 dry runs with `guidellm` installed and a
  live server.
