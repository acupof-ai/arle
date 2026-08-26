# Metal KV disk tier: content-keyed, survives restart — metal, 2026-08-26

> Status: Confirmed (local M4 Pro 48 GB)

## Context

`--kv-disk` on Metal wrote every prefix page to an *ephemeral* namespace
(`KvTierStore::set_disk`, wiped on drop) addressed by a per-process logical id.
A restarted server therefore re-prefilled everything: a 12.6k-token prompt cost
15.4 s of TTFT again, even though its KV had just been written to disk.

## What worked

Address the disk tier by **content** instead of process-local ids, and make the
namespace durable:

- `infer_seam::prefix_block_content_key(parent, block)` — FNV-1a chaining the
  parent block's key with the block's tokens, masked to the store's 56 key bits.
  Each radix node carries its key (`RadixCache::content_key_of_page`), and
  `try_demote_pages` uses it instead of the engine's `next_tier_key` counter
  (deleted; whole-slot park keeps its own counter).
- `Engine::adopt_cold_tier_blocks` probes the tier past the in-memory match and
  registers each hit as a demoted node (`RadixCache::adopt_demoted_block`), so
  the existing promote path restores it.
- Metal attaches with `KvTierStore::load` under an epoch tag derived from the
  model path plus KV dtype, so a different model never serves these pages. Page
  payloads and the prefix restore snapshot (conv states) are keyed by content;
  `MetalPageStore` no longer keeps `tier_to_logical` / `disk_prefixes` indexes.
- The store is a fixed-size file that evicts its own coldest record
  (`DiskTier::lru`), so `--kv-disk` never grows past its budget. The Metal
  default is a flat 1 GiB (`infer_metal::default_t2_budget_bytes`), clamped by
  free disk; CUDA keeps the disk-fraction default.
- Two write-path fixes the feature needs to work at all: a full writer queue now
  blocks instead of dropping the write (only 41 of 787 pages landed before), and
  the sidecar syncs the manifest after publishing (`KvTierStore::sync_manifest`,
  which does not stop the writer) because a killed process never runs `Drop`.

Measured, LFM2.5-8B-A1B-MLX-4bit, one 12.6k-token prompt, default `--kv-disk`:

| run | TTFT |
|---|---|
| cold (no store) | 15.4 s |
| same process, second request | 0.14 s |
| **after restart** | **0.47 s** |

790 pages restored from the manifest, all 787 blocks adopted by content key.
Generation is identical across no-disk / cold / restarted runs. Needle gate
(`NEEDLE_MAX_TOKENS=400`, 512–8000, ×3) exact 3/3 before and after restart.
Fixed-space behavior checked at a 96 MiB cap: the file stays exactly 96 MiB and
the store evicts; a prefix that cannot fit the store is skipped rather than
evicting its own head while writing its tail.

Cost: publishing a 12.6k prompt writes ~160 MB, which the blocking queue makes
synchronous — first-request TTFT 14.3 s (no disk) vs 15.4 s.

## Rule

- A cache that is meant to survive a restart must be addressed by content, not
  by any id minted at runtime. A process-local key is invisible to the next
  process even when its bytes are still on disk.
- A durable store is only recoverable up to its last manifest write. Anything
  that persists solely in `Drop` is lost on `kill`; sync at the natural
  low-frequency checkpoint instead (here: prefix publish).
- `try_send` on a bounded writer queue silently truncates a bulk write. When the
  whole batch must land for the feature to work, block.
