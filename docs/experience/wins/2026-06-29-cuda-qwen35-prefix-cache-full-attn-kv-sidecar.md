# #85 Qwen3.5/3.6 Prefix Cache — Full-Attn KV Sidecar

## Context

Qwen3.5/3.6 hybrid models interleave linear-attention (recurrent gdr/conv state) and
full-attention (paged KV, `full_attn_kv: PagedKVPool`) layers. Prefix reuse requires
both to be snapshotted at publish time and restored at reuse time.

Prior state: `reusable_prefix_blocks` returned 0 for hybrid models (prefix reuse
disabled). The recurrent sidecar (`Qwen35RecurrentSnapshot`) existed but was incomplete:
full-attention KV pages live in a separate device-side `PagedKVPool` with its own page-ID
namespace, disconnected from the host radix cache pool.

## What Worked

**Two-phase sidecar protocol:**

**Capture** (`capture_recurrent_sidecar`, after prefill completes):
1. `slot_state.snapshot_recurrent(ctx)` — D2H copy of gdr (f32) + conv (bf16) state
2. `full_attn_kv.page_indices(slot)` → `pool.copy_pages_to_host(ctx, pages)` — D2H
   serialize all full-attention KV pages for the slot into `Vec<u8>`
3. Attach `full_attn_kv: Some(data)` to `Qwen35RecurrentSnapshot`; insert to sidecar

**Restore** (`restore_recurrent_sidecar`, before tail prefill on prefix hit):
1. `acquire_recurrent` — allocate device recurrent buffers
2. `restore_recurrent_from_snapshot` — H2D restore gdr + conv (stream-ordered + sync)
3. `full_attn_kv.free_slot(slot)` — release prior occupant's device pool tracking
4. `full_attn_kv.alloc_tokens(slot, matched_len)` — allocate fresh device pages
5. `pool.copy_pages_from_host(ctx, new_pages, kv_data)` — H2D restore KV content

`reusable_prefix_blocks` re-enabled: `pages_only_reusable_prefix_blocks(blocks, |_| false)`.

## Bug Found After Initial Landing — seq_len Not Reset on Restore

`restore_recurrent_sidecar` called `acquire_recurrent` without first releasing the prior
occupant's recurrent block. Since prefix-reuse prefill has `start_pos != 0`, the
`release_recurrent + acquire_recurrent` pair inside `submit_prefill_row` (guarded by
`start_pos == 0`) was never reached. `acquire_recurrent`'s early return
(`!gdr_states.is_empty() → return Ok(())`) silently kept the old block with `seq_len`
from the completed request (e.g. 593). At the first decode step, `submit_decode_row`'s
invariant `slot.seq_len == kv_seq_len` failed (`593 != 329`).

Fix (commit `1b0f0459`): call `release_recurrent` before `acquire_recurrent` in
`restore_recurrent_sidecar`, then `set_seq_len(matched_len)` after restoring the snapshot.

## Verification

**Initial smoke (Qwen3-4B, commit `a9748208`, c=1):**

| | Cold (req 1) | Warm (req 2) |
|---|---|---|
| prompt tokens | 50 | 60 (same 50-tok prefix + 10 new) |
| published_pages | 3 | — |
| prefix hits | 0 | **1** |
| crash / assertion | none | **none** |

**Production validation (Qwen3.5-122B-A10B TP=4 GPU 0,2,5,6, commit `1b0f0459`, 512 in / 256 out):**

| concurrency | throughput (tok/s) | TTFT p50 |
|---|---|---|
| c=1 | 40.3 | 4.9s |
| c=2 | **53.0** | 9.7s |
| c=4 | 52.0 | 19.8s |
| c=8 | 48.1 | 41.6s |

No `Qwen3.5 materialized state len N != DecodeRow.kv_seq_len M` at any concurrency.
Peak 53 tok/s at c=2 (4 slots, 122B-MoE, TP=4).

## Key Design Points

- `full_attn_kv` (device-side `PagedKVPool`) has its own page-ID namespace, separate from
  the host radix cache pool. Host page IDs cannot be passed to `attach_pages` on the
  device pool — always use `alloc_tokens` + `copy_pages_from_host` at restore time.
- Device pool sync (`free_slot` + `alloc_tokens`) runs unconditionally in
  `restore_recurrent_sidecar` so `prefill_row_paged_default`'s `seq_len == start_pos`
  invariant always holds, even on sidecar cache-miss (eviction beyond `RECURRENT_SIDECAR_CAP`).
- `RECURRENT_SIDECAR_CAP = 32`. Each entry: ~49 MiB recurrent + ~2.9 MiB/page full-attn KV.
- Capture is skipped (returns `Ok(())` without inserting) if D2H fails, preventing
  incomplete sidecar entries that would leave the device pool out of sync at restore time.

## Commits

- `52e2fdb4` — recurrent sidecar skeleton (snapshot_recurrent / hash_prefix_tokens)
- `36cd91d5` — revert prefix reuse (full-attn KV not yet cached)
- `ed32d3df` — full-attn KV sidecar: D2H capture + H2D restore + re-enable prefix reuse
- `a9748208` — fix `*budget` deref in `set_kv_recall` (unrelated, same push)
- `1b0f0459` — fix seq_len mismatch: release_recurrent + set_seq_len(matched_len) in restore

## Rule

Hybrid models with multiple KV state types (recurrent + paged) require each type
snapshotted separately. The device KV pool namespace is always independent of the host
radix pool — never bridge them with page IDs; always re-allocate device pages and H2D copy.
