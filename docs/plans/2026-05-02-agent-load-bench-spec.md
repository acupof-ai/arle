# Agent-Load W3/W4 Benchmark Spec

**Status:** Active - A0 contract opened 2026-05-02
**Owner:** ckl
**Drives:** [`projects/2026-05-02-agent-load-mission-expansion.md`](../projects/2026-05-02-agent-load-mission-expansion.md) slice A0
**Template:** [`experience/wins/TEMPLATE-bench-agent-load.md`](../experience/wins/TEMPLATE-bench-agent-load.md)

This document defines the first reproducible benchmark contract for the
agent-load world-#1 expansion. It is a measurement contract, not a performance
claim. Code and tuning work after this point must preserve the workload shapes
below unless a deliberate benchmark-spec commit changes them.

## 1. Goal

Define the W3 and W4 workload panel so ARLE, SGLang, vLLM, TensorRT-LLM, and
Mooncake can be compared on agent-serving behavior without hiding behind a
plain single-turn throughput number.

Required properties:

- session identity is explicit on every request;
- warm-turn and cold-turn latency are reported separately;
- tool-output resume is measured as its own phase;
- prefix-cache reuse is a validity signal, not an anecdote;
- every run records pinned engine commit, launch flags, tokenizer path, raw
  client output, and service stats artefacts.

Non-goals for A0:

- adding the final benchmark driver;
- adding scheduler counters;
- producing world-#1 numbers;
- choosing an optimization path.

## 2. Shared Protocol

All W3/W4 runs use OpenAI-compatible HTTP.

| Field | Requirement |
| --- | --- |
| Endpoint | `POST /v1/chat/completions` unless an engine requires the nearest OpenAI-compatible equivalent. |
| Streaming | `stream=true`; TTFT is measured at the first content delta. |
| Sampling | `temperature=0`, deterministic seed when the engine exposes one. |
| Tokenizer | Same model tokenizer/processor path for all engines in a hardware cell. |
| Session | Every request carries `session_id`; unsupported engines must document the fallback and are not allowed to claim session-affinity wins. |
| Concurrency | Fixed in-flight request count, not `guidellm --profile sweep`. |
| Per-session ordering | Requests inside one session are sequential; concurrency is across sessions. |
| Stats capture | `/v1/stats` or nearest metrics endpoint before, during, after; missing fields are recorded as `unavailable`, never guessed. |
| Tree state | Warmup and measured phases are explicit; no hidden manual cache priming. |
| Dirty tree | Invalid. Run commit must be clean and recorded. |

The eventual driver may reuse `scripts/bench_agent_trace.py`, extend it, or add
a new script. The output must still fill the template linked above and write raw
JSON under `bench-output/YYYY-MM-DD-agent-load-<label>/`.

## 3. W3 - Short-Prompt Multi-Turn Agent Loop

W3 stresses high-turn-count chat agents where the same logical session returns
with a short appended user message and expects resident KV reuse.

### 3.1 Canonical Shape

| Parameter | Value |
| --- | --- |
| Workload id | `agent-w3-short-multiturn` |
| Seed | `20260502` |
| Hardware entrance | H1 L4 first; H2 after four-engine H1 baseline exists. |
| Model entrance | Qwen3-4B family first. |
| Global concurrency | `16` in-flight requests |
| Warm sessions | `64` logical sessions |
| Cold distractor sessions | `64` one-shot sessions |
| Scored warm turns | `4` per warm session, `256` total |
| Scored cold turns | `64` total |
| Reuse mix | `80%` warm same-session turns, `20%` cold distractors |
| Base prompt | `1024 +/- 32` tokenizer tokens on turn 0 |
| Appended user tail | `64 +/- 8` tokenizer tokens per warm turn |
| Max output | `64` tokens per request |
| Run duration | Full trace completion; if capped, cap must be `>= 180s` and incomplete turns must be reported. |

Warm-turn prompts must extend the exact serialized prefix used by that session's
previous turn. A request only counts as W3-warm when at least 80% of its prompt
tokens are shared with the previous prompt for the same `session_id`.

### 3.2 Required Metrics

| Metric | Definition |
| --- | --- |
| successful output tok/s | Output tokens from successful scored turns divided by measured wall time. |
| warm TTFT p50/p99 | TTFT over scored warm turns only. |
| cold TTFT p50/p99 | TTFT over scored cold distractor turns only. |
| ITL p50/p99 | Inter-token latency over all successful scored turns. |
| warm/cold fairness | `cold_ttft_p99 / warm_ttft_p99`; values above `4.0` require a Problems note. |
| prefix hit/skip rate | Server-reported hit and skipped-prefill token ratios when exposed. |
| session-affinity hit/miss | Server-reported same-session routing result when exposed. |
| incomplete turns | Count and reason. |

### 3.3 Entrance And Mission Gates

W3-H1 entrance is green only when both hold:

- ARLE successful output tok/s is `>= 1.00x` the best pinned competitor.
- ARLE warm TTFT p99 is no worse than `1.05x` the best pinned competitor.

W3-H1 mission is green only when:

```text
W3_margin = min(
  ARLE_successful_output_tok_s / best_competitor_successful_output_tok_s,
  best_competitor_warm_ttft_p99_ms / ARLE_warm_ttft_p99_ms
)

W3_margin >= 1.30
```

Prefix-cache and session-affinity counters are validity gates. If an engine
does not expose them, the wins entry must say so and include whatever per-request
cached-token report the engine can return.

## 4. W4 - Tool-Call Resume

W4 stresses the real agent loop break point: a model turn pauses for a tool,
the application injects tool output, and the same session resumes generation
without paying a full 8k re-prefill when resident KV is available.

### 4.1 Canonical Shape

| Parameter | Value |
| --- | --- |
| Workload id | `agent-w4-tool-resume` |
| Seed | `20260502` |
| Hardware entrance | H1 L4 first; H2 after four-engine H1 baseline exists. |
| Model entrance | Qwen3-4B family first. |
| Global concurrency | `8` in-flight resume requests |
| Sessions | `128` logical sessions |
| Warmup phase | One 8k prompt request per session, `max_tokens=64`, not scored except failures. |
| Scored phase | One resume request per session after injected tool output. |
| Base prompt | `8192 +/- 64` tokenizer tokens before the tool break. |
| Tool output injection | `256 +/- 16` tokenizer tokens appended as a `tool` or nearest supported role. |
| Max output | `256` tokens for the resume request. |
| Run duration | Full trace completion; if capped, cap must be `>= 300s` and incomplete turns must be reported. |

The scored resume request must preserve the same `session_id` as warmup and
must serialize the prior conversation plus tool output through the same chat
template used by the engine. Engines with first-class append/session APIs may
use them only if the raw request log proves equivalence to the OpenAI transcript.

### 4.2 Required Metrics

| Metric | Definition |
| --- | --- |
| resume output tok/s | Output tokens from successful scored resume turns divided by scored wall time. |
| resume TTFT p50/p99 | TTFT over scored resume turns. |
| resume E2E p50/p99 | Wall-clock request latency over scored resume turns. |
| cold 8k TTFT p50/p99 | Control run without resident session KV, same prompt and output cap. |
| avoided-prefill ratio | `1 - (resume_prefill_tokens / cold_prefill_tokens)` when exposed. |
| matched-prefix tokens | Reused tokens on resume when exposed. |
| tool injection errors | Rejected role, malformed tool payload, or transcript mismatch count. |
| incomplete resumes | Count and reason. |

### 4.3 Entrance And Mission Gates

W4-H1 entrance is green only when ARLE proves tool resume avoids full re-prefill
for resident sessions. Either signal is acceptable:

- `matched_prefix_tokens / expected_reused_tokens >= 0.90`; or
- `resume_ttft_p99 <= 0.50 * cold_8k_ttft_p99` on the same engine and hardware.

W4-H1 mission is green only when:

```text
W4_margin = min(
  ARLE_resume_output_tok_s / best_competitor_resume_output_tok_s,
  best_competitor_resume_ttft_p99_ms / ARLE_resume_ttft_p99_ms,
  best_competitor_resume_e2e_p99_ms / ARLE_resume_e2e_p99_ms
)

W4_margin >= 1.30
```

## 5. Baseline Panel

Every competitor row must record commit/tag, launch command, tokenizer path,
model path, KV dtype, memory fraction, request limit, prefix-cache/session
settings, and raw artefact path.

| Engine | Baseline requirements |
| --- | --- |
| SGLang | Use the mission-pinned commit unless A5 deliberately updates it. Do not pass `--disable-radix-cache`. If supported by the pinned commit, enable cached-token reporting with `--enable-cache-report`. For structured tool-call variants, record the chosen `--tool-call-parser`. Verify all flags with `python -m sglang.launch_server --help` at the pinned commit; current docs list `--disable-radix-cache`, `--enable-cache-report`, and `--tool-call-parser` as server arguments. |
| vLLM | Pin tag/commit, automatic prefix caching flags, scheduler knobs, KV dtype, and request limit. If no session-affinity equivalent exists, record the gap and still run content-prefix reuse. |
| TensorRT-LLM | Pin tag/commit, inflight batching settings, KV cache precision, and any prompt-cache or session-cache knobs. |
| Mooncake | Pin commit, disaggregated/prefix-cache configuration, storage/transport backend, and cache warmup policy. |

No `latest` container, floating `main`, or undocumented default counts toward
the mission table.

Reference docs for SGLang flag discovery:

- <https://docs.sglang.ai/backend/server_arguments.html>
- <https://sgl-project.github.io/advanced_features/server_arguments.html>

## 6. Report Contract

Each W3/W4 run produces a wins or errors entry using the agent-load template.
The report must include:

- Goal type: `baseline`, `regression`, `optimization`, or `diagnosis`.
- Hypothesis written before the run.
- Exact client command and server launch command.
- Clean commit SHA and feature set.
- Hardware, driver/runtime, model path, tokenizer path.
- Raw per-turn JSON.
- Service stats before, during, after.
- Headline table for the required metrics.
- Four-engine comparison table when a mission claim is being made.
- Delta table versus the previous ARLE run on the same workload/hardware.

Raw artefact layout:

```text
bench-output/YYYY-MM-DD-agent-load-<label>/
├── client_turns.jsonl
├── client_summary.json
├── service_stats_before.txt
├── service_stats_trace.jsonl
├── service_stats_after.txt
├── service_stats_trace_summary.md
├── server_launch.txt
└── engine_metadata.json
```

## 7. Implementation Acceptance For A4

The future W3/W4 harness is complete only when all hold:

- `--workload agent-w3-short-multiturn` generates the exact W3 trace shape.
- `--workload agent-w4-tool-resume` generates the exact W4 trace shape.
- `--engine-label` and `--engine-kind` land in every raw artefact.
- ARLE dry run and SGLang dry run both produce template-complete entries.
- Missing server-side counters are represented as `unavailable`, not inferred.
- The driver exits non-zero when successful scored turns are below `99%`.

## 8. Follow-On Work

This spec unblocks:

- A1 session-affinity admission, using W3 warm-turn p99 and session-affinity
  hit/miss as the acceptance signal.
- A2 stats surface, using the required metric list above.
- A3 tool-call injection path, using W4 avoided-prefill ratio and resume TTFT.
- A4 harness implementation, using the exact trace shapes above.
- A5 four-engine baseline panel, using the launch-evidence table above.
