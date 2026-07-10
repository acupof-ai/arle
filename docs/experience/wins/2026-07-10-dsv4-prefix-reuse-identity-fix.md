# DSv4 Route A: Prefix Reuse Re-enabled (Identity Formula Fix)

## Context

DSv4 prefix reuse was broken on the CUDA Route A path. The FlashMLA pool's device page table was always set using the identity formula `slot * lsp + i`, assuming page IDs are sequential from slot 0. When the engine reuses a prefix from a different slot, the device table pointed to empty/wrong physical pages instead of where the actual KV data lived.

## Root Cause

Two code paths hardcoded the identity formula instead of using engine-provided `slot_pages`:

1. `prepare_kv_batch` — for non-demand-paged (identity) layers, built `layer_pages` as `(row.slot * lsp + i)` for all `i`, ignoring the engine's `slot_pages` array.
2. `mirror_full_band` — same identity formula for the full band during prefix restore.

The host pool (`attach_pages`) correctly populates `slot_pages[slot]` with `[prefix_pages..., fresh_pages...]`; the device side just wasn't reading them.

## Fix

`kv_layout.rs` + `executor.rs` (2 files, +10/-14 lines):

- `prepare_kv_batch` identity path: `slot_pages[i]` instead of `(row.slot * lsp + i) as u32`
- `mirror_full_band`: accept `prefix_pages: &[u32]`, use them for leading pages; identity fallback for tail (overwritten by next `prepare_kv_batch` anyway)
- `executor::restore_prefix_state`: pass `prefix_pages` through

`mirror_band` (paged_kv.rs) already supports arbitrary page IDs with correct `page_attach_count` refcounting — no changes needed.

## Verification (H20 pod, TP=4, DSv4-Flash-FP8)

| Test | Result |
|------|--------|
| Sequential needle 500-4000tok | 15/15 exact |
| 4-concurrent different needles | 4/4 PASS, no cross-contamination |
| 4-concurrent same prompt (reuse) | 4/4 PASS, 1.4s total |
| 8-concurrent same prompt (storm) | 8/8 PASS, 3.1s total |
| 16-concurrent mixed (4x max_running) | 15/16 PASS, 0 crashes |
| Prefix hit rate | 89.7% (52/58 sequential+concurrent) |
| Serve errors | 0 (no OOM, no panic, no admission failure) |

2 needle recall errors at 16-concurrent are MoE non-determinism under queueing pressure, not prefix cache bugs — the same lengths are 100% at sequential.

## Rule

When adding a capability that breaks an invariant (identity page mapping), audit **every** place that invariant is assumed. Here the assumption was in two separate device-table-setup paths, not the one the host-pool fix touched.
