# ARLE vs SGLang Admission Policy Gap Matrix

## Context

This document compares ARLE `main` after reverting `a008b5be` to SGLang
`214c35b03184c354acf1f86f99746799e1c9b3a9`.

ARLE files:

- `infer/src/scheduler/cuda/execution.rs`
- `infer/src/scheduler/cuda/budget.rs`
- `infer/src/scheduler/cuda/decode.rs`
- `infer/src/scheduler/cuda/runtime/scheduler_loop.rs`

SGLang summary:

- See `docs/research/2026-04-30-sglang-admission-policy.md`.

## Current ARLE Mechanics

### Waiting Admission

`assign_slots` admits waiting requests into active slots using `PageBudget`.
For every active unfinished request it reserves:

```text
estimated_request_target_tokens(prompt_tokens, max_tokens) = prompt_tokens + 1
```

Then each candidate is admitted if the same target fits. The budget starts from
`pool_free_pages()` and current slot seq lens. It does not include a full
future max-new reservation for newly admitted waiting requests.

### Prefill Planning

`PrefillBudget::from_scheduler` builds decode slots from
`slot_is_runnable_decode`, then:

- token budget = `max_num_batched_tokens - runnable_decode_count`, capped by
  `max_prefill_tokens` and request caps.
- page budget = current `pool_free_pages()`.
- running decode reservation = for each runnable decode row, reserve
  `min(max_tokens - generated_tokens, 4096)`.

`prefill_reservation` computes:

```text
prefill_tokens = min(remaining_prefill_tokens, chunk_cap)
page_growth = prefill_tokens + min(max_tokens - generated_tokens, 4096)
```

So every prefill candidate, including a truncated mid-prefill chunk, reserves
its full clipped future decode tail.

### Decode Retraction

Before decode launch, `retract_decode_to_fit` checks whether one token per
decode row plus any mixed-prefill extra pages fits. If not, it preempts and
requeues rows until it fits or only one row remains. The victim score is:

```text
(generated_tokens, Reverse(prompt_tokens))
```

The minimum score is selected, so ARLE also retracts least-progressed rows and
prefers retracting longer prompts among equal progress.

## Policy Gaps

| area | SGLang | ARLE current | gap |
|---|---|---|---|
| Running decode reservation | Reserves all requests in `running_batch` through `PrefillAdder.running_batch.reqs`. | Reserves only `slot_is_runnable_decode` rows during prefill planning. | Emit-gated or otherwise non-runnable live decode rows may not hold admission headroom. Needs measurement before patch. |
| Reservation magnitude | `remaining_max_new * new_token_ratio`; adaptive after retraction and decays after stable decode. | `min(remaining_max_new, 4096)` fixed, no adaptive ratio. | ARLE can be too conservative when many final chunks reserve full tails, or too optimistic if live-but-non-runnable rows are omitted. |
| Mid-prefill chunk reservation | Truncated chunk charges `extend_input_len + page_size`, but `max_new=0`; final chunk charges max-new. | Every chunk charges `prefill_tokens + clipped_remaining_decode`. | Long prompts can be rejected/stalled by repeated full-tail reservation before they are decode-ready. |
| Page alignment overhead | Adds `page_size` per request in `total_tokens` and `_update_prefill_budget`. | Waiting admission uses `prompt + 1`; prefill reservation has no explicit page overhead beyond page rounding in `PageBudget`. | Near-full edge may over-admit by one page unless `PageBudget` COW/tail accounting already covers it. Needs allocator-level verification. |
| Evictable prefix budget | Uses `available_size + tree_cache.evictable_size()` in prefill budgets. | `PageBudget::from_scheduler` starts from `pool_free_pages()`. Prefix cache pressure is handled separately by `evict_prefix_cache_if_pressured` in assign loop and allocation retry paths. | Prefill candidate selection may reject before considering evictable cache capacity. |
| Decode feedback | Retraction computes new `new_token_ratio`; stable decode decays ratio. | Retraction requeues but does not feed an adaptive admission ratio. | ARLE has no closed-loop admission controller for KV edge pressure. |
| Existing chunk priority | `chunked_req` is resumed before new waiting requests. | ARLE queues prefilling slots in `prefill_queue` and scores by queue rank. | Likely similar for FCFS, but no explicit single `chunked_req` leak-prevention invariant. |
| Mixed budget | `num_mixed_decode_tokens=running_bs` subtracts running decode rows from input/chunk budgets. | `StepTokenBudget::for_prefill` subtracts runnable decode count. | Same omission risk for non-runnable live decode rows. |

## Request Mode × Phase × Dimension Gap Matrix

Legend:

- OK: no obvious policy gap from code reading.
- Risk: plausible degradation; requires targeted bench/trace.
- Gap: clear semantic difference from SGLang.

| request mode | phase | TTFT-sensitive | throughput-sensitive | latency-tail-sensitive |
|---|---|---|---|---|
| long-prompt 32k+ | cold-start prefill | Risk: waiting admission is prompt+1 and can admit near pool edge. | Gap: prefill budget sees free pages, not `free + evictable`. | Risk: no adaptive ratio after pressure. |
| long-prompt 32k+ | mid-prefill chunk | Gap: each chunk reserves clipped decode tail; SGLang charges tail only when not truncated. | Gap: repeated full-tail reservation can under-admit overlap. | Risk: chunk progress can stall near 99% KV util. |
| long-prompt 32k+ | decode-only | OK: decode retraction exists and victim policy is close. | OK: decode batch shrinks to fit. | Risk: no adaptive admission feedback after retraction. |
| long-prompt 32k+ | decode+prefill overlap | Gap: only runnable decode rows reserve budget; SGLang reserves running batch. | Gap: fixed 4096 tail vs adaptive ratio. | Risk: mixed steps can alternate with idle when live rows are gated. |
| short-prompt <2k | cold-start prefill | OK/Risk: prompt+1 favors short TTFT, but long prompts may consume slots first under FCFS. | OK: small prompts fit easily. | Risk: no request-class priority to rescue short prompts behind long prefill. |
| short-prompt <2k | mid-prefill chunk | OK: usually not chunked. | OK. | OK. |
| short-prompt <2k | decode-only | OK: decode retraction preserves progress. | OK. | Risk: short rows can still wait behind long prompt chunks if queued later. |
| short-prompt <2k | decode+prefill overlap | Risk: short-prompt bypass exists only for small prefill candidates on non-mixed path. | OK/Risk: mixed can run if budget admits. | Risk: no explicit TTFT class priority. |
| mixed-mode dynamic | cold-start prefill | Gap: no scheduler-level mode classification beyond FCFS/priority. | Risk: long prompts can dominate prefill queue. | Risk: no latency-tail aware admission. |
| mixed-mode dynamic | mid-prefill chunk | Gap: no SGLang-style truncated-vs-final decode-tail distinction. | Gap: long chunks can block overlap via full-tail reservation. | Risk: queue-rank only scoring. |
| mixed-mode dynamic | decode-only | OK. | OK. | Risk: retraction does not change future admission aggressiveness. |
| mixed-mode dynamic | decode+prefill overlap | Gap: no adaptive ratio and possible non-runnable decode omission. | Gap: fixed tail reserve. | Risk: idle spin under near-full KV edge. |
| agent-loop prefix-hit | cold-start prefill | Risk: prefix hit reduces materialization, but admission budget may not count evictable cache the same way as SGLang. | Risk: prefix cache pressure handled outside prefill candidate selection. | Gap: no SLRU-style hit-count eviction; ARLE policy must be checked separately. |
| agent-loop prefix-hit | mid-prefill chunk | Risk: partial-page/cache ownership differs; needs trace under repeated session hits. | Risk: repeated chunks may reserve decode tail. | Risk: cache hit survival under long prompts not established. |
| agent-loop prefix-hit | decode-only | OK. | OK. | Risk: no feedback loop from retraction to admission. |
| agent-loop prefix-hit | decode+prefill overlap | Risk: short loop turn can be delayed by long chunk FCFS. | Risk: prefix-hit capacity not unified with prefill budget. | Risk: no mode-aware tail protection. |

## Phase 1.5 Patch Design Requirements

No patch should be accepted until baseline and scans are complete. A candidate
design must answer these questions explicitly:

- Classification: how does the scheduler classify each request as long-prompt,
  short-prompt, mixed-mode, or agent-loop/prefix-hit without expensive or
  unstable runtime heuristics?
- Priority dimension: when TTFT, throughput, and tail-latency goals conflict,
  which class wins under each phase?
- Chunk accounting: does ARLE distinguish truncated mid-prefill chunks from
  final prefill chunks when charging future decode growth?
- Decode headroom: if adopting SGLang's algorithm, where does ARLE store and
  update a `new_token_ratio` equivalent after decode retraction and stable
  decode?
- Prefix eviction: should prefill planning use `free + evictable` or trigger a
  bounded eviction before rejecting candidates?
- Fairness: how do short prompts avoid being starved behind long-prompt chunks,
  and how do long prompts still make progress under short-prompt bursts?

## Bench Matrix Required Before Patch Acceptance

Baseline first, patch second. Every row gets its own wins entry.

### Long-Prompt Anchor

- Pre-patch `0464fb3e` behavior: S5 `32k in / 256 out`, `c=1,4`, three full
  canonical runs, report mean and standard deviation.
- Patched behavior: same three-run matrix.

### Headroom Parameter Scan

Only after a concrete design exists:

- `c=4`, `32k in / 256 out`, headroom `{5%, 10%, 15%, 20%}`, one run each.
- Each wins entry must include the exact commit, env, params, and delta against
  the three-run pre-patch anchor.

### Mixed-Mode Coverage

Minimum additional rows:

- Long + short mix: concurrent 2 long 32k streams plus short prompt streams
  `<2k`, with TTFT and output TPS reported separately if the harness supports it.
- Agent-loop prefix-hit: repeated short prompts sharing a session/prefix-cache
  key, with prefix hit rate, TTFT, and p99 latency.
- Decode-only pressure: prefilled or short-prompt workload where decode rows
  dominate, to ensure long-prompt headroom changes do not regress decode TPS.

## Immediate Status

- `a008b5be` headroom patch was reverted by `b850ca9e` before any valid
  benchmark was recorded.
- The interrupted `bench-output/2026-04-30-longctx-32k-phase1-p11-headroom`
  run is invalid and must not be used as evidence.
