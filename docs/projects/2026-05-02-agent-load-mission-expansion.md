# Agent-Load Mission Expansion — World #1 by >=30%

**Status:** Active expansion — opened 2026-05-02  
**Owner:** ckl  
**Relationship:** Extends `2026-04-30-longctx-32k-128k-leadership.md`; does not replace W1/W2 longctx mission.

## 1 · Mission

ARLE must become world #1 for agent-serving inference, not only long-context
throughput.

```
On agent-loop workloads, ARLE throughput and tail latency must beat the best
open-source serving engine by >=1.30x across SGLang, vLLM, TensorRT-LLM, and
Mooncake.
```

Agent-loop workloads mean repeated multi-turn requests with high prefix reuse,
session affinity pressure, and tool-call interruptions that inject fresh tokens
between model turns.

The success formula mirrors the longctx mission:

```
success(W, H) := ARLE.score(W, H) >= 1.30 x max(SGLang, vLLM, TRT-LLM, Mooncake).score(W, H)
```

For agent workloads, score is a composite panel, not only output tok/s:

- throughput: successful output tokens per second
- TTFT p99 for warm prefix-hit turns
- tool-call resume latency after injected tool output
- prefix-cache hit and skip rate

No benchmark without commit pin, launch flags, tokenizer path, session profile,
and raw artifacts counts toward the mission.

## 2 · Workload Panel

The existing longctx panel remains:

| Workload | Shape | Status |
| --- | --- | --- |
| W1 · max-throughput | 32k prompt / 256 output / c=4 | Phase 1 L4 closed at 1.609x SGLang |
| W2 · long-decode | 32k prompt / 2048 output / c=4 | Phase 2.B MagicDec sparse-KV in progress |

The agent-load expansion adds:

| Workload | Shape | Primary metric | Why it matters |
| --- | --- | --- | --- |
| W3 · agent-loop short-prompt multi-turn | 1k prompt / 64 output / c=16 / 80% prefix reuse | warm-turn TTFT p99 + successful tok/s | Chat-agent loops are dominated by repeated short turns and prefix reuse, not one cold prefill. |
| W4 · agent-loop tool-call resume | 8k prompt / 256 output / c=8 / inject 256 tool-output tokens mid-session | resume TTFT p99 + end-to-end turn latency | Tool use breaks the normal prefill/decode rhythm and tests whether the scheduler can append injected context without losing KV reuse. |

Baseline panel:

| Engine | Required evidence |
| --- | --- |
| SGLang | commit pin, prefix-cache/session flags, FP8 KV settings where available |
| vLLM | tag/commit pin, automatic prefix caching flags, scheduler knobs |
| TensorRT-LLM | tag/commit pin, inflight batching flags, KV cache precision |
| Mooncake | commit pin, disaggregated/prefix-cache configuration |

## 3 · Current ARLE Fit

| Capability | Status | Notes |
| --- | --- | --- |
| Radix prefix cache | Ready foundation | `RadixCache` is production-connected for CUDA prefix reuse and is stronger than a linear last-prompt cache. |
| FP8 KV pool | Ready foundation | Phase 1 close depended on FP8 capacity; W3/W4 need the same capacity to keep many active sessions resident. |
| Session fields in scheduler policy surface | Partially ready | `scheduler/policy.rs` has session-aware signal plumbing, but routing is not fully wired into CUDA admission. |
| HTTP `session_id` flow | Partially ready | Request types carry session identity in parts of the stack; CUDA admission still needs deterministic session affinity. |
| Tool-call mid-injection | Missing | No canonical scheduler path yet for appending tool-output tokens into a live session while preserving prior KV. |
| Session affinity routing | Missing | Admission should prefer slots and radix subtrees with the same session, bounded by fairness and tail-latency SLOs. |
| Agent benchmark harness | Partial | `scripts/bench_agent_trace.py` exists, but W3/W4 need canonical GuideLLM-compatible entries and raw trace artifacts. |

## 4 · Gap Matrix

| Gap | Current state | Required behavior |
| --- | --- | --- |
| Session affinity routing | Prefix reuse works, but admission does not consistently route same-session turns to the most useful resident KV. | Route by `session_id` and prefix hit depth, with anti-starvation guard for cold requests. |
| Warm-turn TTFT accounting | Existing metrics report general TTFT and prefix counters. | Add per-session warm/cold tags and p50/p99 warm-turn TTFT in bench entries. |
| Tool-output token injection | Tool output is just part of the next prompt. | Add an explicit mid-session append path: committed model tokens + injected tool tokens + resumed decode. |
| Prefix-hit benchmark profile | Existing longctx runs focus on cold 32k prompts. | Canonical W3/W4 workloads with deterministic reuse rate and session mix. |
| Competitor apples-to-apples | SGLang longctx baseline exists; agent panel not pinned. | Four-engine baseline with matching prefix-cache/session settings and artifact names. |
| Tail-latency fairness | Scheduler optimizes throughput and KV pressure. | Add TTFT-sensitive priority for warm turns without starving long prompts. |

## 5 · Engineering Slices

| Slice | Name | Scope | Exit |
| --- | --- | --- | --- |
| A0 | Agent mission benchmark spec | Define W3/W4 prompts, reuse model, run duration, output accounting, and baseline flags. | `docs/plans/2026-05-02-agent-load-bench-spec.md` plus template wins entry. |
| A1 | Session affinity admission | Wire `session_id` into CUDA admission scoring and slot/radix preference. | Same-session warm turns choose resident prefix when available; fairness test covers cold-request starvation. |
| A2 | Agent stats surface | Add warm/cold prefix-hit counters, session-affinity hit/miss, and tool-resume latency stats. | `/v1/stats` and Prometheus expose W3/W4 watch-list counters. |
| A3 | Tool-call injection path | Represent tool-output injection as a scheduler-visible append/resume transition. | 8k + 256 injected tokens resumes without full prompt re-prefill when KV is resident. |
| A4 | W3/W4 harness | Extend bench tooling to generate multi-session traces with controlled prefix reuse and tool injection. | One ARLE dry run and one SGLang baseline run produce comparable raw artifacts. |
| A5 | Four-engine baseline panel | Pin SGLang/vLLM/TRT-LLM/Mooncake for W3/W4. | Baseline table has commits, flags, hardware, and raw links. |
| A6 | Optimization loop | Tune cache admission, active KV retention, and decode overlap for agent load. | ARLE >=1.30x best competitor on W3 and W4 for H1, then repeat on H2. |

## 6 · Scheduling With Longctx Mission

This expansion can run in parallel with W1/W2.

- W1/W2 stress long prompts, KV capacity at the pool edge, and long-decode speculative paths.
- W3/W4 stress short prompts, high turn count, prefix-cache residency, session stickiness, and tool resume latency.

The engineering paths overlap in `RadixCache`, active KV retention, and scheduler
admission, but they do not require the same benchmark envelope. Work should stay
parallel unless a change touches shared hot-path admission logic; then W1/W2
longctx regression and W3/W4 agent regression both become required.

## 7 · Acceptance Gates

| Gate | Requirement |
| --- | --- |
| W3-H1 entrance | ARLE >=1.00x best competitor on c=16 short-turn throughput and warm-turn TTFT p99 within 5%. |
| W3-H1 mission | ARLE >=1.30x best competitor composite score. |
| W4-H1 entrance | Tool resume path avoids full re-prefill for resident KV sessions. |
| W4-H1 mission | ARLE >=1.30x best competitor composite score. |
| H2 repeat | W3/W4 mission gates pass on H100/H20-class hardware with pinned engine baselines. |

Until W3 and W4 have four-engine baselines, claims must say "agent-load
candidate" or "agent-load subset", not "agent-load world #1".

## 8 · Immediate Next Step

Start A0 after the P2.B.7 rerun and FP8 audit:

1. Write the W3/W4 benchmark spec and artifact template.
2. Pin SGLang agent-loop baseline flags first because its prefix-cache behavior is the closest near-term competitor.
3. Add ARLE dry-run instrumentation only after the spec names the counters that decide pass/fail.

