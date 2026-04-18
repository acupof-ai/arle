# Paged-prefill follow-ups — post plan-hoist structural fix

**Date:** 2026-04-18
**Status:** Active backlog
**Supersedes:** `docs/plans/paged-prefill-lifecycle-audit-2026-04-18.md`
(the audit closed out — root cause was per-layer plan call, fixed in
`5208530`)

## Context

Commit `5208530` shipped the structural fix: FlashInfer's `PrefillPlan`
is now called once per forward via the new `ops::PagedPrefillForward`
handle instead of 36× per layer. Qwen3 paged prefill passes the
guidellm sweep cleanly (zero server errors) and cuts sync TTFT by
~12%. Two follow-up items surfaced during the bench:

## 1. Throughput-mode regression (-21% peak tok/s)

**Symptom:** paged path peak throughput drops from 97.43 tok/s (contig
Apr-18) to 77.05 tok/s in guidellm's `throughput` profile. All
steady-rate cells (0.13–0.33 r/s) stay flat within noise. ITL is
flat. So the regression is admission/tail specific, not per-request
kernel compute.

**Hypothesis A — pool pressure timing.** In the contig path, pool
pages are allocated only AT MIGRATION (after the final prefill
chunk). In the paged path, pages are allocated PRE-CHUNK
(`alloc_pool_tokens_with_retry(slot, chunk_len)` in
`scheduler/cuda/prefill.rs:247`). Under throughput mode with 10
concurrent 4096-token prefills, paged allocates 2560+ pages upfront
while contig defers that. If the pool is under pressure, the
earlier alloc triggers eviction earlier, and the
`evict_prefix_cache_for_allocation → retry` loop chews scheduler
CPU time that the decode-launch path would otherwise use.

**Hypothesis B — indirection cost.** Paged writes K/V via
`page_table[pos / page_size]` indirection (uncoalesced) vs. contig's
sequential slot-local writes. Under throughput mode where L2 cache
has no headroom, the extra cache misses may compound across 10
concurrent prefill forwards.

**Next action:** measure. Add per-step timing breakdown
(`scheduler::cuda::execution.rs` already emits `step breakdown:
decode=X emit=Y new=Z prefill=W`) under `RUST_LOG=info` for one
throughput sweep iteration, compare to the contig baseline. If
`prefill` times are similar → Hypothesis A (admission). If `prefill`
times differ materially → Hypothesis B (kernel indirection cost).
Pick fix based on which dominates.

## 2. `prefix_cache.insert` warnings under bench

**Symptom:** `WARN prefix_cache.insert: expected 4096 tokens, got 0`
(occasional) or `got 4080` (common) fires on every bench request.

**Interpretation sketch:** every bench request tokenises with a
shared BOS/chat-special prefix. Radix cache's first-token child
already exists from a prior request → walk enters it → subsequent
tokens diverge → insert bails partway. `got 4080` = 255 of 256
blocks inserted (one block's worth lost at the divergence point).
`got 0` = even the root's first-token child conflicted (rare; the
sequence of events that produces this isn't yet understood).

**Is this new with paged?** Unclear from `git log`. The warning site
`scheduler/cuda/core.rs:631` predates Phase 3a. Worth running the
Apr-17 contig commit's server under the same bench to confirm.

**Next action:** read `infer/src/prefix_cache.rs::insert_with_fingerprints`
(starts at line 507) end-to-end. Work out under which conditions it
returns `tokens.len() - block_size` vs `0`. If the `got 0` case is a
real bug (e.g. concurrent insert collision), that's the priority. If
it's just "walked into an existing child that diverges one block in",
the warning is noise and should be downgraded to `debug!` or
reworded.

## 3. Qwen3.5 re-enable

Qwen3.5 stays at `prefill_uses_paged_pool() = false`. The plan-hoist
structural fix applies to HD256 too — when re-enabled,
`PagedPrefillForward::new_hd256` is wired correctly. The blocker
is a separate scheduler issue: Qwen3.5's `supports_partial_prefix =
false` makes the scheduler return MISS on radix hits but leaves pool
seq_len advanced from the retained prefix. A reused slot then has
`page_indices` inconsistent with `seq_len` on the recompute path.

**Next action:** trace `runtime.rs::assign_slots` → how `radix_hit`
interacts with `supports_partial_prefix=false`. Likely fix: when
admitting a new request to a slot whose pool carries retained pages
the model can't reuse, explicitly `free_slot(slot)` before calling
`alloc_pool_tokens_with_retry`. Requires a test harness because the
crash only reproduces under concrete sweep load.

## 4. Audit §(c) eviction race — formal closure

Codex's static analysis ruled out mid-request eviction of active-slot
pages (commit lifecycle is clean). The audit doc
`paged-prefill-lifecycle-audit-2026-04-18.md` listed this as
unverified; the root cause turned out to be elsewhere
(per-layer plan race). §(c) is closed but no test locks in the
invariant. A scheduler-level property test would be cheap insurance
against a future change making it untrue.

**Next action:** low priority. Note the invariant in
`infer/src/scheduler/cuda/AGENTS.md` and leave tests as backlog.

## Ordering

1 blocks any end-to-end throughput claim. 2 is likely a nop/log-
wording fix. 3 is the bigger structural work. 4 is hygiene.

Prefer 1 next. Do 2 as a 30-min side audit while the bench is
running. 3 when Qwen3.5 paged-prefill becomes the priority of the
day. 4 any time.
