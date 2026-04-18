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

## 2. `prefix_cache.insert` warnings under bench ✅ resolved 2026-04-19

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

**Findings (2026-04-19, code-read only — no bench rerun):**

Both warnings fire on the **same code path**: the partial-edge-split
branch at `prefix_cache.rs:558-598`. When a walk descends into an
existing child edge and `match_len < child_tokens.len()`, the code:

1. Creates a shared intermediate node holding `child_tokens[..match_len]`.
2. Rewires the original child as the shared node's child (suffix).
3. **`break`s without advancing `block_idx`** — see comment at
   `prefix_cache.rs:597` "Don't advance block_idx — the shared node
   has < block_size tokens."

So `block_idx * block_size` is whatever was accumulated on full-edge
matches before the split. Translation:

- `got 4080` (block_size=16) = 255 whole-block matches, split fires on
  block 256. Shared BOS/chat-template + 255 common blocks, then the
  256th block diverges.
- `got 0` = split fires on block 1. Root already has a child for
  `tokens[0]` (shared first token, e.g. BOS), but the existing edge's
  bytes diverge from our tokens somewhere in positions 1..block_size.
  The shared intermediate ends up with <block_size tokens, no block_id,
  and the break leaves our request's blocks **unregistered**.

So neither case is a concurrent-insert bug. Both are "existing subtree
diverged mid-edge → split left our blocks un-registered."

**Is this a correctness bug?** No corruption — the caller at
`scheduler/cuda/core.rs:639` early-`return`s, so the slot's blocks
simply aren't pinned via `retain_pages`. They're freed when the slot
is freed. But it IS missed sharing.

**Why the fix is non-trivial** (not just "fall through to else"):

The radix `insert` contract ties each `blocks[i]` to
`tokens[i*block_size .. (i+1)*block_size]`. A partial split at
`match_len` leaves `pos = match_len < block_size`. Naively falling
through to the `else` branch would make its first edge cover
`tokens[match_len .. match_len + block_size]` — but that window is
not a caller-provided block; the caller's block 0 was
`tokens[0..block_size]`. Assigning `blocks[0]` to a misaligned
window would register a wrong token→block_id mapping and poison
future `BlockFingerprint` lookups.

**Recommendation:**
- **Don't** just downgrade to `debug!` — the warning flags genuine
  missed cache sharing, not harmless noise.
- **Don't** land a drive-by "continue after split" fix — block-id
  alignment makes it non-local.
- The minimum safe fix: after a mid-block partial split, advance
  `pos` to the next block boundary (`block_size.ceil(match_len)`),
  set `node_idx = shared_idx`, jump `block_idx` accordingly (which
  also means **skipping** the partial first block's block_id; that
  block won't be shared), then continue the outer loop so the
  remaining aligned blocks land as a new subtree rooted at the
  shared intermediate. Needs:
  - Unit tests covering both scenarios (`got 0` = split at token 1,
    `got 4080` = split at block 256) and verifying (a) `inserted ==
    floor_block(publishable_prompt.len())` not 0, (b) both the
    existing suffix and the new subtree are reachable from the
    shared intermediate, (c) no block_id in `block_index` points
    at a node whose token span doesn't match the underlying pool
    block.
  - Pure prefix_cache change, no GPU. Safe to attempt without a
    CUDA box, but needs CUDA E2E re-run before merge to confirm no
    regression in prefill KV reuse.

**Landed 2026-04-19** (commits `a497289` + `265caa3` + `b912990`, pushed):

- `prefix_cache.rs:598` — replaced the unconditional `break` after
  edge split with: if the caller still has an aligned block's worth
  of tokens (`pos + block_size <= tokens.len()` and `block_idx <
  blocks.len()`), place `blocks[block_idx]` as a new child node of
  `shared_idx` covering `tokens[pos+match_len..pos+block_size]`, then
  continue the outer loop. Path-from-root to the new node is
  `match_len + (block_size - match_len) = block_size`, so block-id
  alignment holds.
- Six new tests in `prefix_cache::tests`:
  `split_on_first_block_inserts_remaining_blocks_as_sibling`,
  `split_on_later_block_inserts_divergent_block_under_shared`,
  `split_with_short_tail_still_registers_aligned_block`,
  `reinsert_after_split_does_not_reuse_first_block_id` (Codex round-1
  regression — short-edge block-bearing node confused full-match walk
  advancement; fix in `265caa3` keys advancement on `block_id.is_some()`
  rather than edge length),
  `split_mid_block_via_shared_intermediate_registers_tail_block`,
  `else_branch_mid_block_via_shared_intermediate_registers_tail_block`
  (Codex round-2 regressions — walks through non-block-bearing shared
  intermediates leave `pos` mid-block while `block_idx` stays at the
  current block; fix in `b912990` computes boundaries from
  `(block_idx+1)*block_size` in both the split and else branches).
- Two pre-existing tests (`evict_prunes_orphan_tombstones_into_free_list`,
  `insert_reuses_reclaimed_tombstone_slots`) updated — they were
  silently asserting the old drop-the-divergent-block behavior. New
  assertions use the now-correct tree shape with two sibling branches
  under the shared tombstone and 3-node tombstone cascade on evict(2).
- 50/50 prefix_cache tests + 296/296 infer lib tests pass under
  `--no-default-features --features no-cuda`. CI clippy surface
  (`cd infer && cargo clippy --no-default-features --features
  no-cuda --lib -- -D warnings`) clean. `codex review --commit b912990`
  clean (third review pass — round-1 and round-2 caught real bugs,
  round-3 found none).
- CUDA E2E re-run deferred to user — this is a prefix-sharing
  correctness fix, so worst-case regression is the pre-fix state
  (missed sharing), not corruption.

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

**Findings (2026-04-19, code-read only — CUDA repro still pending):**

Traced the lifecycle end-to-end. The plan's original symptom
description ("MISS on radix hits but leaves pool seq_len advanced")
doesn't match what the code actually does. The real picture is more
specific; refining the hypothesis below so the next GPU session has
a sharper target.

**What the MISS downgrade actually covers** (`prefill.rs:37-44`):

```rust
let prefix_len = if raw_prefix_len > 0
    && raw_prefix_len < cached_prompt_len
    && !state.supports_partial_prefix()
{ 0 } else { raw_prefix_len };
```

The `!supports_partial_prefix` downgrade is **conditional on
`raw < cached`** — it only triggers when the radix matched *less*
than the slot has materialized (e.g. the slot advanced past the
publish via decode). It does **not** trigger when `raw == cached <
prompt_len` — the same-slot-reuse shape that `best_reusable_slot_for_radix_hit`
produces by construction (it accepts when `cached_prompt_len >=
reusable_prefix_len`, and under a clean cleanup → publish → reuse
cycle, equality is the common case).

**Pool state is correct across all three MISS / reuse branches.**
`cleanup() → free_slot()` clears `page_indices[S] = []` and
`seq_lens[S] = 0`. The subsequent `step_new` either (a) skips pool
alloc (pool_prefix_len=0), or (b) allocates fresh pages and migrates
contig → paged from offset 0 or offset L as appropriate. So the
"inconsistent page_indices vs seq_len" framing is incorrect at the
pool layer — pool state is consistent. Which means the
recompute-path bug, if present, is at the **model state** layer,
not the pool.

**The actual hole: hybrid recurrent state under `raw == cached < prompt_len`.**

When `raw == cached == L < prompt_len` (radix matched exactly what the
slot materialized, prompt has more tokens beyond), execution falls
to `prefill.rs:99`:

```rust
} else if prefix_len > 0 && prefix_len == cached_prompt_len {
    state.truncate_to(prefix_len)?;       // Qwen35: zeroes recurrent_state
    state.restore_prefix_snapshot()?;      // restore recurrent_state from snapshot
    // ...
}
```

`Qwen35State::truncate_to` (`model/qwen35/forward.rs:53-60`)
**unconditionally zeroes** the recurrent state, with the comment
"The scheduler should avoid partial prefix hits for hybrid models"
— but the scheduler's guard at line 37-44 doesn't cover this
`raw == cached` path. The recovery depends on
`restore_prefix_snapshot()` returning `Ok(true)` with a valid
snapshot. Two failure modes:

1. **`Ok(false)` (no snapshot)**: line 122-125 logs "prefix HIT" and
   silently continues with recurrent state at zeros. Chunked prefill
   then advances recurrent state on `tokens[L..prompt_len]` starting
   from zeros, not from the correct state-at-L. Garbage output or
   downstream kernel OOB if intermediate state shapes depend on the
   accumulated time axis.
2. **Stale snapshot**: if R1's snapshot was taken, then something
   else overwrote the recurrent state before R2's restore, the
   snapshot could be inconsistent with `cached_prompt_len`. Harder
   to reach but possible.

Under the happy path (R1 completed prefill cleanly, snapshot saved
at line 327, nothing clobbered it between R1 cleanup and R2
assign), the restore succeeds and this branch behaves correctly.
The crash under sweep load probably has a step where one of the
two failure modes fires — hence "only reproduces under concrete
sweep load."

**Refined fix hypothesis** (supersedes "free_slot before alloc"):

Extend the MISS downgrade condition at `prefill.rs:37-44` from
`raw < cached` to `raw <= cached && prefix_len < prompt_len`:

```rust
let prefix_len = if raw_prefix_len > 0
    && raw_prefix_len < prompt_len
    && !state.supports_partial_prefix()
{ 0 } else { raw_prefix_len };
```

Rationale: for hybrid models, the only safe same-slot reuse is
`raw_prefix_len == prompt_len` (full match → exact branch at line
51, which already handles hybrid via `state.reset()` + full
re-prefill). Any partial case — including `raw == cached < prompt_len`
— risks the recurrent-state-from-snapshot path. Downgrading to MISS
avoids the truncate_to + restore dance entirely; the pool path
remains correct (alloc 0, later alloc prompt_len, migrate [0..prompt_len]).

**Acceptance test plan:**

- Add a unit test in `prefill::tests` that constructs the condition
  vector `(raw, cached, prompt)` with `!supports_partial_prefix` and
  asserts `prefix_len == 0` for every `(raw, cached, prompt)` where
  `raw > 0 && raw <= cached && prompt > raw`. (Pure logic, no GPU.)
- Re-enable `prefill_uses_paged_pool() = true` for Qwen3.5 in
  `model/qwen35/forward.rs:211` and run the guidellm sweep. Crash
  disappears → hypothesis confirmed. Crash persists → look at
  contig→paged migration path for partial same-slot reuse.

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

**Landed 2026-04-19 (partial):** invariant recorded as
`infer/src/scheduler/AGENTS.md` item 9 (paths live at `scheduler/` not
`scheduler/cuda/` — updated to the actual location). Property test
still a backlog item.

## Ordering

1 blocks any end-to-end throughput claim. 2 is likely a nop/log-
wording fix. 3 is the bigger structural work. 4 is hygiene.

Prefer 1 next. Do 2 as a 30-min side audit while the bench is
running. 3 when Qwen3.5 paged-prefill becomes the priority of the
day. 4 any time.
