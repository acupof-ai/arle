# Qwen KV pool sizing follows measured VRAM, not a constant floor (#178)

> Status: Code landed (`5c2931cd3`) and **runtime-VERIFIED 2026-07-25** on the
> 32 GB V100 at default flags — the box that surfaced the defect.

## Context

The #168 V100 re-bench could not boot Qwen3.6-27B-W4A16 on a 32 GB card at
HEAD: 21.9 GB weights left <2 GB, and the first dspark prefill's >1.8 GB alloc
OOM'd the engine thread. The 2026-07-21 V100 runs on the same box predated the
defect (a 16384-token / 1.1 GB pool worked). `--kv-cache-dtype fp8` is
`CUDA_ERROR_NOT_SUPPORTED` on sm_70 and int8 is unsupported on qwen35 pools, so
no flag escaped it.

## What Worked

Attribution: both Qwen pool-sizing sites took
`profiled_pages.max(requested_pages)`, and `requested_pages` comes from
`EngineLoadConfig::total_pages` — default 8192 (131072 tokens, 8.6 GB BF16) and
**not user-facing** since `--num-slots`/`--total-pages` were removed. So a
constant floor overrode the measured-VRAM profile and booked HBM the card does
not have; the failure then surfaced one allocation later, at first prefill,
instead of at boot.

Fix (one line each, `executor/qwen35.rs` + `executor/qwen.rs`): the profile IS
the sizing. `requested_pages` survives only as the fallback for a failed
free-VRAM probe. `profile_kv_pool_tokens` already floors at 4096 tokens, so the
pool still cannot collapse to zero, and big boxes are unchanged (profiled ≫ 8192
pages there — the floor never bound).

DSv4 does not share the defect: `kv_budget_plan` solves the pool from measured
free VRAM with no constant floor and fails **closed** at startup ("Lower
--max-total-tokens or free VRAM") rather than over-booking. It never receives
`config.total_pages` at all.

Gates: cuda-lane clippy clean. A box-local 8192→1024-page patch had already
served fine on the V100, which is the same direction as this fix.

## Runtime verify (2026-07-25, `eacd62ce7`, 1×V100-32GB sm_70)

Qwen3.6-27B-W4A16, default flags (no `--max-total-tokens`, no
`--mem-fraction-static`), binary symbol-checked (`pages (advisory)` ×2):

```
free 14092MB / total 32501MB, recurrent reservation 12625MB (86 slots × 146MB),
free_after_recurrent 1466MB, cell 65536B/tok -> max_total_tokens 4096
(256 pages); requested 8192 pages (advisory) -> sizing 256 pages
```

**256 pages, 32× below the old 8192 floor.** Boot ~100 s, peak 19.4/32.5 GB,
zero OOM across 6 requests including one 3709-token prefill — the exact
allocation class that used to kill the engine thread.

Both directions confirmed on the same binary: BF16 Qwen3.5-0.8B on the same box
profiled `sizing 120449 pages` (1.93M tokens), i.e. the removed floor does not
cap a big-headroom card.

Discriminator run, worth keeping: the 27B W4A16 generations are token salad
(#179, sm_70 W4A16 defect), but the 0.8B BF16 arm on the same binary answers
coherently — so the garbage is not the pool fix and not a runtime-wide break.
Attributing it took one control run, not an argument.

## Rule

- A constant that is no longer user-facing must not outrank a measured
  quantity. `max(profiled, requested)` reads as "honor the user's ask" only
  while the ask is real; once the flag is gone it is just a floor that lies.
- Over-booking a pool moves the failure one allocation downstream — the boot
  succeeds and an unrelated-looking prefill OOMs. Sizing must fail where it is
  decided (DSv4's fail-closed `ensure!` is the pattern).

## Follow-up exposed (#182)

The verify line reads `86 slots × 146MB recurrent = 12625MB` of 14092MB free,
leaving 1466MB → 4096 KV tokens for 86 slots (**47 tokens each**).
`kv_budget_num_slots` solves for slots alone (`SlotBudget::from_free(free, 0.9,
0, per_slot)` — fixed term zero), so it maximizes slot count against all of free
VRAM and the pool profiles the remainder. DSv4's `kv_budget_plan` already solves
`(num_slots, pool_tokens)` jointly; Qwen3.6 should converge on that. Pre-existing
— before this fix the over-booked pool hid it by OOMing instead.
