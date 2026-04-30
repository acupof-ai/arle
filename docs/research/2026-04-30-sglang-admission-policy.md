# SGLang Chunked-Prefill Admission and Eviction Policy

## Context

- Source checkout: `/tmp/sglang-arle-214c35b03184c354acf1f86f99746799e1c9b3a9`
- Primary files:
  - `python/sglang/srt/managers/scheduler.py`
  - `python/sglang/srt/managers/schedule_policy.py`
  - `python/sglang/srt/managers/schedule_batch.py`
  - `python/sglang/srt/mem_cache/radix_cache.py`
  - `python/sglang/srt/mem_cache/evict_policy.py`
- Scope: normal decode worker with chunked prefill and radix prefix cache. DLLM,
  SWA, disaggregation, HiCache, LoRA, and priority preemption are noted only
  where they change the main admission decision.

## Admission Trigger Conditions

SGLang enters `_get_new_batch_prefill_raw` when there is a waiting request or
an unfinished `chunked_req`. It returns no prefill batch when:

- `running_batch.batch_is_full` is set and there is no `chunked_req`.
- `waiting_queue` is empty and there is no `chunked_req`.
- `get_num_allocatable_reqs(running_bs) <= 0`, there is no `chunked_req`, and
  priority preemption is disabled. `get_num_allocatable_reqs` is
  `pp_max_micro_batch_size - running_bs`, additionally capped by request-token
  pool availability under pipeline parallelism.

Before iterating waiting requests, SGLang creates one `PrefillAdder` with the
current `running_batch`, `new_token_ratio`, `max_prefill_tokens`,
`chunked_prefill_size`, and `num_mixed_decode_tokens = running_bs` when mixed
chunked prefill is enabled. This makes decode rows consume input/chunk budget
for the mixed batch.

For each waiting request:

- The scheduler checks LoRA compatibility and optional HiCache prefetch state.
- It marks `running_batch.batch_is_full` when the adder already has as many
  runnable prefill requests as `get_num_allocatable_reqs(running_bs)`.
- If the batch is full, priority preemption may try to create space; otherwise
  iteration stops.
- `req.init_next_round_input(tree_cache)` computes prefix hit state and
  `extend_input_len`.
- `PrefillAdder.add_one_req` returns:
  - `CONTINUE`: request admitted and more candidates may be considered.
  - `NO_TOKEN`: token/KV budget exhausted; mark `batch_is_full` and stop.
  - `OTHER`: non-token cap hit such as input/chunk/request budget; stop.

If a `chunked_req` already exists, SGLang always attempts to continue it before
new waiting requests. That path is special-cased to avoid leaking memory for an
unfinished chunked request.

## Budget and Headroom Formula

`PrefillAdder` keeps two token-offset budgets:

- `rem_total_tokens = available_kv_tokens + evictable_prefix_tokens - rem_total_token_offset`
- `cur_rem_tokens = available_kv_tokens + evictable_prefix_tokens - cur_rem_token_offset`

For each request already in `running_batch`, SGLang reserves:

```text
min(max_new_tokens - generated_tokens, CLIP_MAX_NEW_TOKENS) * new_token_ratio
```

This is decode growth headroom for running requests. It is not a fixed percent
of the whole pool. It is a per-request remaining-output estimate scaled by the
adaptive `new_token_ratio`.

For each newly admitted full prefill request, SGLang first checks:

```text
total_tokens = extend_input_len + max_new + page_size
total_tokens < rem_total_tokens
```

where `max_new = min(max_new_tokens - output_len, CLIP_MAX_NEW_TOKENS)`.
The `page_size` term is an explicit allocator-alignment cushion.

When the request is admitted, `_update_prefill_budget` deducts:

```text
extend_input_len = ceil_to_page(extend_input_len)
rem_total_token_offset += extend_input_len + max_new_tokens + page_size
cur_rem_token_offset   += extend_input_len + page_size
rem_input_tokens       -= extend_input_len
rem_chunk_tokens       -= extend_input_len  # when chunked prefill is enabled
```

For truncated chunked prefill, the truncated chunk deducts `max_new_tokens = 0`.
For the last chunk or non-chunked prefill, it deducts `max_new_tokens`.

`budget_state` stops adding requests when total or current remaining tokens are
non-positive, when input budget is exhausted, or when chunk budget is exhausted.

## Decode OOM and Adaptive Ratio

After a forward pass, `update_running_batch` calls `batch.check_decode_mem()`.
That computes `new_tokens_required_next_decode`, evicts enough prefix-cache
tokens, then checks whether the allocator has enough free KV tokens for the
next decode step.

If decode memory is insufficient, `batch.retract_decode` preempts decode
requests until the remaining selected decode batch fits. The victim order for
non-speculative decoding sorts by:

```text
(len(output_ids), -len(origin_input_ids)), reverse=True
```

and pops from the end, so the actual victim is the least-progressed request,
with longer prompts preferred for retraction among equal progress. At least one
request is retained when possible. If even one request cannot decode, SGLang
aborts that final request.

After retraction, SGLang updates `new_token_ratio` to:

```text
(total_decoded_tokens + SGLANG_RETRACT_DECODE_STEPS * remaining_batch_size)
/
(total_max_new_tokens + 1)
```

bounded to `<= 1.0`. On successful decode checks, the scheduler decays
`new_token_ratio` toward its configured minimum. This makes admission less
optimistic after real decode pressure, then relaxes it when the system remains
stable.

## Prefix Cache Eviction

SGLang's radix cache tracks two sizes:

- `evictable_size_`: cached tokens not protected by lock refs.
- `protected_size_`: cached tokens protected by live request lock refs.

`inc_lock_ref(node)` walks from a matched node to the root. When a node's
`lock_ref` transitions from `0` to `1`, tokens move from evictable to
protected. `dec_lock_ref(node)` does the inverse when `lock_ref` transitions
from `1` to `0`.

Eviction is leaf-based:

1. Build a heap from `evictable_leaves`.
2. Rank leaves by the configured eviction strategy.
3. Pop leaves until at least `num_tokens` are evicted or no leaf remains.
4. Free the leaf KV indices and delete the leaf.
5. If the parent becomes an unlocked leaf, push it into the heap.

Supported strategies:

- `lru`: smallest `last_access_time` first.
- `lfu`: smallest `(hit_count, last_access_time)` first.
- `fifo`: smallest `creation_time` first.
- `mru`: newest access first by negative `last_access_time`.
- `filo`: newest creation first by negative `creation_time`.
- `priority`: smallest `(priority, last_access_time)` first.
- `slru`: probationary nodes (`hit_count < 2`) before protected nodes, then LRU.

Prefix matching updates `last_access_time`. Insert updates `last_access_time`,
propagates request priority by `max`, and increments hit count unless the
insert is from chunked prefill, avoiding self-hit inflation across chunks.

## Request Mode × Phase × Dimension Matrix

Legend:

- TTFT: first-token sensitivity.
- TPS: throughput sensitivity.
- Tail: p95/p99 latency sensitivity.

| request mode | phase | TTFT decision | TPS decision | Tail decision |
|---|---|---|---|---|
| long-prompt 32k+ | cold-start prefill | Admit only if `extend_input_len + max_new + page_size < rem_total_tokens`; chunk if larger than chunk budget. | Chunked prefill caps per-step work by `rem_chunk_tokens` and `max_prefill_tokens`; mixed chunk subtracts running decode tokens from input/chunk budget. | Prefix cache eviction is allowed before budget checks through `available + evictable`; live locked prefixes are protected. |
| long-prompt 32k+ | mid-prefill chunk | Existing `chunked_req` is attempted before new queue entries to avoid leaks and continue progress. | Truncated chunks deduct no future decode max-new until final chunk; this avoids over-reserving decode tail for partial chunks. | If chunk cannot fit, `budget_state` returns `NO_TOKEN`/`OTHER`; next decode safety still runs via `check_decode_mem`. |
| long-prompt 32k+ | decode-only | Decode memory is checked before decode; prefix cache can be evicted for next-token pages. | Batch stays as large as memory permits. | If memory still fails, retract least-progressed decode rows, preserving at least one row when possible. |
| long-prompt 32k+ | decode+prefill overlap | `num_mixed_decode_tokens=running_bs` reduces prefill input/chunk capacity. | Running decode rows reserve adaptive future decode headroom via `new_token_ratio`. | Retraction updates `new_token_ratio`, feeding back into future admission. |
| short-prompt <2k | cold-start prefill | Small `extend_input_len` usually passes total/current budget; request-count caps dominate. | Multiple short requests can be admitted until request/token caps. | Prefix cache hit nodes are locked, so hot short-loop prefixes are protected while active. |
| short-prompt <2k | mid-prefill chunk | Usually not chunked unless configured chunk size is tiny. | `rem_input_tokens` and request caps gate batching. | `page_size` overhead avoids near-full allocator overcommit. |
| short-prompt <2k | decode-only | Decode memory check runs the same; small prompts are less likely to be retracted for equal output progress because victim tie-break prefers retracting longer prompts. | Decode batching is bounded by memory and microbatch size. | Retraction protects tail by shrinking batch rather than failing all rows. |
| short-prompt <2k | decode+prefill overlap | Mixed chunk can overlap short prefill with decode if model path supports it. | Short prefill consumes little chunk budget; decode headroom remains adaptive. | Grammar/LoRA/HiCache readiness can skip individual requests without blocking all queue entries. |
| mixed-mode dynamic | cold-start prefill | Policy priority ordering runs before admission; incompatible LoRA/prefetch-pending rows can be skipped. | `PrefillAdder` admits in policy order until token/request/chunk budget stops. | `batch_is_full` prevents over-admission until decode/retraction frees room. |
| mixed-mode dynamic | mid-prefill chunk | Existing `chunked_req` has priority over new requests. | Truncated chunks avoid charging future max-new until completion. | Prevents memory leak but can reduce fairness if one long chunk monopolizes chunk continuation. |
| mixed-mode dynamic | decode-only | `check_decode_mem` is global across selected decode rows. | Decode rows are retained while fit; retracted rows return to queue. | Victim choice favors preserving rows with more generated tokens. |
| mixed-mode dynamic | decode+prefill overlap | Mixed chunk subtracts decode row count from prefill token budget. | Running rows reserve future decode with `new_token_ratio`; new final chunks reserve max-new. | `new_token_ratio` adapts after real pressure instead of fixed static headroom. |
| agent-loop prefix-hit | cold-start prefill | `match_prefix` page-aligns and locks matched nodes; cache hit reduces `extend_input_len`. | Prefix reuse lowers prefill cost, allowing more short loop turns. | Locked hit nodes become protected and cannot be evicted mid-request. |
| agent-loop prefix-hit | mid-prefill chunk | Chunked inserts do not inflate hit count, so one request does not make its own chunks artificially hot. | Unfinished chunks are cached and rematched to reuse prior chunks. | `cache_protected_len` prevents duplicate frees across partial pages. |
| agent-loop prefix-hit | decode-only | Same decode memory path; prefix cache does not directly affect one-token decode except by freeing evictable pages. | Repeated loop turns benefit from cached prompt prefix. | LRU/LFU/SLRU policy determines whether agent-loop prefixes survive unrelated long prompts. |
| agent-loop prefix-hit | decode+prefill overlap | Prefix-hit short turns can be mixed with decode if budgets allow. | Admission considers available plus evictable cache; active locks protect in-flight loop contexts. | Evictable cold leaves can be dropped before active/hot locked prefixes. |

## Immediate ARLE Implications to Validate

- SGLang does not use a static pool-percent headroom formula. It uses
  per-running-request `remaining_max_new * new_token_ratio`, plus page overhead,
  plus final-chunk max-new reservation.
- SGLang distinguishes truncated mid-prefill chunks from final chunks: truncated
  chunks do not reserve full decode max-new.
- SGLang admission includes evictable prefix-cache tokens in the budget. A
  direct comparison to ARLE must check whether ARLE's prefill budget sees
  evictable cache capacity before rejecting.
- SGLang's memory safety is two-stage: optimistic/adaptive admission first,
  then decode-time `check_decode_mem` + retraction if reality exceeds estimate.
