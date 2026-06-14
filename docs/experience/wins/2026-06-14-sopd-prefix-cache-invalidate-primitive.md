# SOPD #92 block-1 — `Engine::invalidate_prefix_cache` primitive (prefix-cache epoch drop)

**Date**: 2026-06-14 · **Issue**: [#92](https://github.com/cklxx/arle/issues/92) (SOPD Phase-0 prerequisite) · **Status**: block-1 (primitive + unit tests) **and** block-2 (serve-path wiring, Option A) both landed · **Bench**: `pending-remote` — see below

## Context

SOPD's live LoRA re-merge (`remerge_student_lora`) mutates resident `q_proj`/`v_proj`
weights. The `RadixCache` (`infer-core/src/radix.rs`) is keyed by token blocks only —
**no weight-epoch key**. After a re-merge, every cached block's `V = v_proj(x)` is
stale; serving a post-update request from that KV is a silent correctness bug (plan
risk R1 HIGH).

**Grounding correction (this session)**: the #91 *training* rollout path
(`InferStudent::decode_next_token` → `forward_token_logits` → `run_on_executor` →
raw executor forward) **bypasses the Engine scheduler and RadixCache entirely** — each
step re-forwards the full sequence (`positions = 0..len`, stateless) under the current
adapter. So #92 is **not** a #91 prerequisite (the earlier survey-level claim was
wrong). #92 is the correctness primitive for the **production 训推一体 serve path**
(`Engine.complete()` + RadixCache + live re-merge), needed at the serving-integration
phase, not the keystone probe.

## What Worked

`Engine::invalidate_prefix_cache()` in `infer-core/src/prefix.rs` (device-neutral,
locally unit-tested on Mac, no CUDA):

- **Drops, never demotes.** Unlike `evict_prefix_cache_for_pages` (which demotes
  victims to the host tier, keeping them promotable), invalidation drops everything —
  demoting would only relocate stale-epoch KV to host, where a later prefix match could
  promote it back. Buffer-level disposition, each proven:
  - resident pages → `radix.evict_lru(cached_page_count)` (re-scans the evictable
    frontier per step → one call drains the whole idle trie) → `kv.release_pages` +
    `executor.release_prefix_pages`.
  - demoted host-tier blocks → `radix.lru_demoted_key` + `drop_demoted` loop → severed.
  - severed tier keys → `drain_dropped_tier_keys` → `executor.drop_kv_tier_entries`
    (no leak — radix doc-comment invariant).
- **Precondition (proven in test)**: pages pinned by an in-flight request
  (`ref_count > 0`) are **skipped** (never freed under a live reader). The OPD inline
  loop calls this between rollouts on a quiesced engine → every page idle → full drop.
  Concurrent serve + live update needs per-request epoch tagging (out of Phase-0 scope).
- **Tests** (`infer-core/src/lib.rs`): `invalidate_prefix_cache_drops_all_idle_cached_pages`
  (2 cached pages → 0, all pages back in pool) and
  `invalidate_prefix_cache_keeps_pinned_drops_idle` (idle dropped, pinned survives, the
  in-flight request still completes — no use-after-free). 57/57 infer-core green, clippy clean.

### block-2 — atomic serve-path wiring (Option A, ckl-selected 2026-06-14)

The control seam was the constraint: `drain_control` (`infer-server/execution.rs:287`)
held `&mut Engine<E,K>` but applied each control closure to `engine.executor_mut()`
(`&mut E`), and `invalidate_prefix_cache` is `&mut self` on `Engine` — so a control
closure literally could not reach it. **Option A** (ckl's pick over a `BackendExecutor`
dirty-flag) promotes the seam to engine level:

- `ControlMessage<E>` → `ControlMessage<E, K> = Box<dyn FnOnce(&mut Engine<E, K>) + Send>`
  (`execution.rs:86`); `drain_control` now calls `closure(engine)` (`:287`). The K
  type-param threads through `engine_loop`/`engine_loop_with_tick_broadcaster`
  (`:108`/`:135`), `drain_control` (`:281`), the `ServeHandle.control_tx` field
  (`lib.rs:75`), and both channel constructions (`:205`/`:261`) — purely mechanical.
- New `ServeHandle::run_on_engine<R,F>` (`F: FnOnce(&mut Engine<E,K>) -> R`); the old
  `run_on_executor` becomes a one-line wrapper `run_on_engine(|e| f(e.executor_mut()))`,
  so its 4 existing callers (raw-logits forward, weight offload/reload, kv-tier-disk)
  are byte-for-byte unchanged.
- `remerge_student_lora` (`infer-api/serve_engine.rs:133`) now runs **one** engine-level
  closure: `engine.executor_mut().remerge_student_lora(update)?; engine.invalidate_prefix_cache();`
  — the weight change and the cache drop happen in the same control message, so no
  scheduler step can interleave and serve a post-merge request from pre-merge KV.

Why Option A over the dirty-flag: invalidation stays in infer-core (Engine owns the
radix), the device-neutral `BackendExecutor` seam trait gains **no** method (a
control-plane concept never leaks across the 5 backends), and atomicity is local and
explicit at the single call site rather than split executor-sets-flag / loop-reacts.
Verify: `infer-server` 28/28 tests green + clippy `-D warnings` clean; `infer-api`
CUDA path typechecks (Mac `cuda,no-cuda` lane). (The repo's 5 pre-existing
`infer-cuda` clippy-1.95 lints in `moe.rs`/`dsv4.rs` are unrelated — not touched here.)

## Bench (`pending-remote`)

block-2's only production consumer is `remerge_student_lora`, which fires during OPD
adapter updates — **not** the serving steady state. The serving hot path
(`admit_waiting`/`step`/decode/forward) and `run_on_executor`'s 4 callers are
**byte-for-byte unchanged** (the wrapper has identical closure semantics), so serve perf
is unaffected. The real bench lands when the #91 keystone + the production concurrent
serve+update path run on the H20 pod — a needle-retrieval gate (prefix-cache ON == OFF
after a re-merge, proving no stale-KV contamination) plus a `scripts/bench_guidellm.sh`
serve sweep to confirm the invalidation cost is off the steady-state path. Cross-link:
[SOPD plan](../../plans/2026-06-14-self-training-lora-opd-sopd.md).

## Rule

A token-keyed cache shared across weight epochs is silently stale after any in-place
weight mutation; the invalidation must **drop** (not demote/evict-to-tier), or stale KV
re-enters via promotion. Verify the actual hot path (which forward call the rollout
uses) before declaring a cache a prerequisite — the SOPD rollout never touches the radix.
