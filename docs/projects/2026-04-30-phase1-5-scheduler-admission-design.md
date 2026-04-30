# Phase 1.5 Scheduler Admission Design

## Status

- Input research: `docs/research/2026-04-30-sglang-admission-policy.md`
- Gap matrix: `docs/projects/2026-04-30-arle-vs-sglang-admission.md`
- Pre-patch anchor: `docs/experience/wins/2026-04-30-bench-guidellm-longctx-32k-phase1-baseline-anchor-0464fb3e.md`
- Runtime patch status: not started in this document. This design replaces the
  reverted static-headroom patch.

## Design Decision

The reverted pool-percent headroom assumption was wrong. SGLang does not keep a
fixed percentage of the KV pool empty. It charges admission against:

```text
prefill_extend_tokens + page_size + remaining_decode_tokens * new_token_ratio
```

and charges `remaining_decode_tokens * new_token_ratio` only for final chunks,
not for truncated mid-prefill chunks. ARLE should follow that shape rather than
reintroducing a static pool-percent reserve.

For Phase 1.5, implement the smallest production CUDA change that can be
benchmarked rigorously:

- Replace all-or-nothing `remaining_decode_reservation_tokens` in prefill
  reservation with a configurable ratio of clipped remaining decode tokens.
- Charge that decode tail only when the current prefill chunk completes the
  request's prefill phase.
- Reserve live `running_batch` rows, not only rows currently runnable for
  decode, when building the prefill budget.
- Keep current decode retraction victim policy unchanged.
- Do not add a new queue, policy engine, or prefix eviction strategy in this
  patch.

## Files To Touch

- `infer/src/scheduler/types.rs`
  - Add `decode_headroom_ratio: f64` to `SchedulerConfig`.
  - Default to `0.10` for the first patched benchmark tranche.
  - Validate `0.0 <= decode_headroom_ratio <= 1.0`.
- `infer/src/scheduler/cuda/execution.rs`
  - Add ratio-aware decode reservation helper.
  - Change `PrefillReservation` to know whether the selected chunk is final.
  - Reserve final-chunk decode tail only; truncated chunks reserve prefill
    growth only.
  - Build prefill budget from live `running_batch` rows, while still using
    runnable decode rows for per-step token budget.
- `infer/src/scheduler/cuda/core/construction.rs`
  - Log the resolved `decode_headroom_ratio` with the scheduler envelope.
- `infer/src/main.rs`
  - Add CLI override `--decode-headroom-ratio` for the required scan:
    `{0.05, 0.10, 0.15, 0.20}`.
- Tests in `infer/src/scheduler/cuda/execution.rs` and
  `infer/src/scheduler/types.rs`.

## Request Classification

Phase 1.5 should classify requests using fields already available at admission
and prefill planning time. No token-content heuristics.

| class | classifier | primary objective | scheduler action |
|---|---|---|---|
| long-prompt 32k+ | `prompt_tokens.len() >= 32768` or remaining prefill chunk above `long_prefill_token_threshold` | throughput without decode starvation | chunk by existing caps; truncated chunks do not reserve decode tail |
| short-prompt <2k | `prompt_tokens.len() < 2048` and not prefix-cache staged | TTFT | keep existing short prompt bypass; do not let long final-tail reservation block a small chunk that fits |
| mixed-mode dynamic | live decode rows plus at least one prefill candidate | throughput plus tail stability | reserve live decode rows with ratio, then admit prefills in queue order under token/page budget |
| agent-loop prefix-hit | prefix hit/staged prefix length is nonzero or `session_id` repeats | TTFT and tail | no new policy in Phase 1.5; validate with prefix-hit bench before promoting |

The implementation does not need a new enum yet. The classes are derived from
prompt length, prefill progress, decode-active state, and prefix-hit telemetry
for docs/bench interpretation. If later patches need class-specific ordering,
they should add a single scoring field to the existing candidate score rather
than a parallel queue.

## Priority By Phase And Dimension

| phase | TTFT-sensitive | throughput-sensitive | latency-tail-sensitive |
|---|---|---|---|
| cold-start prefill | Short prompts keep existing bypass; long prompts enter chunked prefill. | Long prompts can consume large token budget when no decode is active. | Admission must fit page budget plus final-tail reserve. |
| mid-prefill chunk | Continue queued chunks in queue order; short chunks that fit should not be skipped because a long truncated chunk would over-reserve decode tail. | Truncated chunks charge prefill growth only, matching SGLang. | Page budget still prevents allocator overcommit. |
| decode-only | No change: decode launch and retraction own this phase. | Keep largest fitting decode batch. | Retraction still removes least-progressed, tie-breaking toward longer prompts. |
| decode+prefill overlap | Decode rows consume token budget and ratio-based page headroom. | Mixed launch remains enabled when model supports it. | Live rows, including temporarily non-runnable rows, reserve page headroom. |

## Avoiding Long/Short Mutual Blocking

- Long prompts no longer repeatedly reserve their full future decode tail on
  every truncated chunk. That lowers false page-budget rejection for overlap.
- Short prompts keep the existing `short_prompt_bypass_tokens` fast path and
  can fit under the same page budget because long truncated chunks stop
  consuming unrelated decode-tail reservation.
- Long prompts still make progress because the candidate selector preserves
  queue order among schedulable candidates; it skips unschedulable candidates
  instead of stopping at the first miss.
- If short-prompt bursts still starve long chunks in mixed-mode benches, that
  becomes Phase 1.6 scoring work, not part of this patch.

## Headroom Scan Interpretation

The requested scan values `{5%, 10%, 15%, 20%}` map to
`decode_headroom_ratio`, not a static percent of KV pool capacity:

```text
reserved_decode_tail = ceil(clipped_remaining_decode_tokens * ratio)
```

The 10% default is a deliberately conservative first point. It is lower than
ARLE's current 100% clipped-tail reserve and should expose whether the
near-full-KV c=4 collapse is caused by repeated over-reservation of final
decode tail during mid-prefill chunks.

## Bench Matrix

Every row needs its own wins entry and commit.

| row | workload | required result |
|---|---|---|
| long anchor patched | longctx-32k c=1+c=4, three runs | compare mean/stddev to baseline anchor |
| headroom 5% | longctx-32k c=4 | single run vs baseline anchor |
| headroom 10% | longctx-32k c=4 | single run vs baseline anchor |
| headroom 15% | longctx-32k c=4 | single run vs baseline anchor |
| headroom 20% | longctx-32k c=4 | single run vs baseline anchor |
| mixed long+short | 2 long 32k streams plus short <2k streams | report long TPS, short TTFT if harness can separate |
| agent-loop prefix-hit | repeated short turns with shared session/prefix | prefix hit rate, TTFT, p99 |
| decode pressure | decode-dominant short workload | no decode TPS regression |

## Rejection Criteria

Reject the patch if any of these hold:

- c=4 mean remains below `14` output tok/s after selecting the best ratio.
- Any patched longctx-32k run has zero successful c=4 requests.
- TTFT p50/p99 improves throughput only by hiding incomplete requests.
- Mixed long+short bench shows short TTFT regression without long throughput
  improvement.
- Agent-loop prefix-hit bench loses prefix hits or worsens p99 due to over-
  aggressive prefix eviction.

## Open Risks

- ARLE's `PageBudget` uses free pages only. This design does not yet include
  SGLang's `free + evictable_prefix_tokens` budget. That is a separate patch
  because eviction safety depends on active-slot page ownership.
- There is no adaptive `new_token_ratio` feedback loop yet. The CLI ratio scan
  is a controlled substitute for Phase 1.5; if a ratio wins, Phase 1.6 should
  make it adaptive after decode retraction and stable decode windows.
- GuideLLM's current longctx wrapper does not separate long and short request
  classes. Mixed-mode coverage may need a small harness extension before the
  final acceptance scan.
