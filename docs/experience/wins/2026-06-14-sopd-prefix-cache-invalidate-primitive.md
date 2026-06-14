# SOPD #92 block-1 — `Engine::invalidate_prefix_cache` primitive (prefix-cache epoch drop)

**Date**: 2026-06-14 · **Issue**: [#92](https://github.com/cklxx/arle/issues/92) (SOPD Phase-0 prerequisite) · **Status**: block-1 landed (primitive + unit tests); block-2 (serve-path wiring) pending · **Bench**: `pending-remote` — see below

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

## Bench (`pending-remote`)

Dormant primitive: the only caller today is the unit tests; the serving hot path
(`admit_waiting`/`step`/decode/forward) is **byte-for-byte unchanged**, so serve perf is
unaffected. The real bench lands when block-2 (control-seam wiring so a live re-merge
triggers invalidation) + the #91 keystone run on the H20 pod — a needle-retrieval gate
(prefix-cache ON == OFF after a re-merge, proving no stale-KV contamination) plus a
`scripts/bench_guidellm.sh` serve sweep to confirm the invalidation cost is off the
steady-state path. Cross-link: [SOPD plan](../../plans/2026-06-14-self-training-lora-opd-sopd.md).

## Rule

A token-keyed cache shared across weight epochs is silently stale after any in-place
weight mutation; the invalidation must **drop** (not demote/evict-to-tier), or stale KV
re-enters via promotion. Verify the actual hot path (which forward call the rollout
uses) before declaring a cache a prerequisite — the SOPD rollout never touches the radix.
