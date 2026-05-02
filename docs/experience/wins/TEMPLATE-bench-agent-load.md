# <SHORT TITLE> - agent-load benchmark, <WORKLOAD>, <BACKEND-LABEL>, <YYYY-MM-DD>

> Template for W3/W4 agent-load benchmark entries. Copy this file for each run,
> fill every placeholder, and never overwrite an existing dated entry. The
> workload contract is
> [`docs/plans/2026-05-02-agent-load-bench-spec.md`](../../plans/2026-05-02-agent-load-bench-spec.md).

## Goal

- <one sentence describing the benchmark goal and goal type>

## Hypothesis

- <expected outcome before the run>

## Command

Server:

```bash
<exact server launch command>
```

Client:

```bash
<exact W3/W4 benchmark command>
```

## Environment

- **Workload:** <agent-w3-short-multiturn | agent-w4-tool-resume>
- **Backend / engine:** <arle-cuda | arle-metal | sglang | vllm | trtllm | mooncake>
- **Model:** <model id and local path>
- **Tokenizer / processor:** <path or HF id>
- **Hardware:** <GPU/SoC model, VRAM, CUDA/Metal version>
- **Commit:** <short sha, clean tree>
- **Feature set:** `cargo build --release <features>` or competitor build/version
- **KV dtype / cache mode:** <bf16 | fp8 | ...>
- **Session / prefix flags:** <flags or "unavailable">
- **Non-default flags / env vars:** <list or "none">

## Workload Params

| field | value |
|---|---|
| seed | `20260502` |
| global concurrency | <16 for W3, 8 for W4> |
| sessions | <count> |
| scored turns | <count> |
| prompt shape | <token counts> |
| max output tokens | <64 for W3, 256 for W4> |
| warm/cold mix | <W3 only, or n/a> |
| tool output tokens | <W4 only, or n/a> |
| run cap | <seconds or full trace completion> |

## Results - Headline

| metric | value |
|---|---:|
| successful scored turns | ... |
| incomplete scored turns | ... |
| successful output tok/s | ... |
| TTFT p50 (ms) | ... |
| TTFT p99 (ms) | ... |
| ITL p50 (ms) | ... |
| ITL p99 (ms) | ... |
| E2E p50 (ms) | ... |
| E2E p99 (ms) | ... |

## Results - W3 Warm/Cold

| metric | warm | cold |
|---|---:|---:|
| scored turns | ... | ... |
| TTFT p50 (ms) | ... | ... |
| TTFT p99 (ms) | ... | ... |
| output tok/s | ... | ... |

## Results - W4 Resume

| metric | value |
|---|---:|
| resume TTFT p50 (ms) | ... |
| resume TTFT p99 (ms) | ... |
| resume E2E p50 (ms) | ... |
| resume E2E p99 (ms) | ... |
| cold 8k TTFT p99 (ms) | ... |
| matched prefix tokens | ... |
| avoided-prefill ratio | ... |

## Results - Service-Side Cache / Scheduler

| metric | value |
|---|---:|
| peak active | ... |
| peak waiting | ... |
| peak prefill_queue | ... |
| peak kv_util | ... |
| `prefix_hit_rate` | ... |
| `prefix_skip_rate` | ... |
| `session_affinity_hit` | ... |
| `session_affinity_miss` | ... |
| `tool_resume_count` | ... |
| `tool_resume_prefill_tokens` | ... |
| `kv_fetch_q` | ... |
| `kv_fetch_waiters` | ... |
| `kv_store_q` | ... |
| `kv_store` | ... |
| `kv_bp` | ... |
| `tier_recall` | ... / n/a |
| `tier_src` | ... / n/a |
| `tier_promoted` | ... / n/a |
| `tier_fallback` | ... / n/a |

## Four-Engine Comparison

| engine | commit/tag | output tok/s | TTFT p99 (ms) | E2E p99 (ms) | cache report | raw artefact |
|---|---|---:|---:|---:|---|---|
| ARLE | ... | ... | ... | ... | ... | ... |
| SGLang | ... | ... | ... | ... | ... | ... |
| vLLM | ... | ... | ... | ... | ... | ... |
| TensorRT-LLM | ... | ... | ... | ... | ... | ... |
| Mooncake | ... | ... | ... | ... | ... | ... |

Mission margin:

```text
<W3_margin or W4_margin formula with filled values>
```

## Problems

- <anything that degraded, crashed, or deviated from the watch-list>

## Learnings

- <generalizable rule or tuning takeaway>

## Delta Vs Baseline

- **Baseline:** <link to prior dated entry, or "first run">

| metric | baseline | now | delta |
|---|---:|---:|---:|
| output tok/s | ... | ... | ... |
| TTFT p99 | ... | ... | ... |
| E2E p99 | ... | ... | ... |

## Artefacts

- Raw turns: `bench-output/<date>-agent-load-<label>/client_turns.jsonl`
- Client summary: `bench-output/<date>-agent-load-<label>/client_summary.json`
- Server launch: `bench-output/<date>-agent-load-<label>/server_launch.txt`
- Engine metadata: `bench-output/<date>-agent-load-<label>/engine_metadata.json`
- Service trace before: `bench-output/<date>-agent-load-<label>/service_stats_before.txt`
- Service trace during: `bench-output/<date>-agent-load-<label>/service_stats_trace.jsonl`
- Service trace after: `bench-output/<date>-agent-load-<label>/service_stats_after.txt`
- Service trace summary: `bench-output/<date>-agent-load-<label>/service_stats_trace_summary.md`

## Notes

- What changed in the code since baseline: <commits or "none">
- Suspected cause of any regression: <text or "n/a">
- Follow-ups: <issues/plans to open or "none">
