# Metal MetalPageStore: prompt-only publish + stale prefix-key pruning

## Context

`MetalPageStore` (`crates/infer-metal/src/executor.rs`) holds device K/V page
blocks (`pages: HashMap<u32, MetalPageBlock>`) and GDR recurrent-state
snapshots keyed by page-id prefix vectors (`prefixes`). `publish_slot` was
called from three sites; code review established two defects:

1. **Decode-time publishing was unreachable dead weight.** Engine-core's
   radix cache only publishes PROMPT pages
   (`crates/infer-core/src/prefix.rs:114`: `publishable_tokens =
   request.prompt_len().min(self.kv.seq_len(slot))`), so pages/snapshots
   covering generated tokens can never be offered for attach. Yet every
   decode step re-sliced ALL full pages (O(full_pages × 2×num_full_layers)
   lazy MLX array creations per token) and every 16th token inserted a GDR
   snapshot pinning a full recurrent+conv state copy per linear-attention
   layer into a map that is never evicted — unbounded device-memory growth
   over long generations, all unreachable.
2. **Stale prefix snapshot aliasing.** Host page ids are recycled LIFO after
   radix eviction; `prefixes` keys were never removed. A new occupant's
   publish overwrote `pages` entries but stale `prefixes` keys containing the
   reused ids survived. A later radix match colliding with a stale key made
   `materialize_slot_from_prefix` serve the NEW occupant's K/V pages with the
   OLD prompt's GDR snapshot — silently corrupted linear-attention output.

## What Worked

- Removed the `publish_slot` calls from `submit_decode`
  (`crates/infer-metal/src/executor.rs:499` pre-change) and
  `commit_pending_then_prequeue` (`crates/infer-metal/src/executor.rs:566`
  pre-change). The `submit_prefill` call stays, with a comment citing the
  `infer-core` prefix.rs `publishable_tokens` clamp as the reason publish is
  prefill-only.
- In `publish_slot`, when `pages.insert` returns `Some(_)` (page id
  overwritten → recycled or republished), `prefixes.retain` drops every key
  containing that page id EXCEPT exact prefixes of the live occupant's own
  page-id list (`page_ids[..k]`), which the boundary snapshot insert keeps
  coherent. Linear scan is fine: `prefixes` stays small once decode-time
  publishing is gone.
- Unit tests exercising `MetalPageStore` directly with hand-built
  `MetalSlotState::from_arrays` + `MetalKvPool` (page_size=4, no model load):
  `page_reuse_prunes_stale_prefix_snapshot` (LIFO reuse prunes the first
  occupant's key, the second occupant's boundary snapshot survives with ITS
  GDR values, `reusable_prefix_pages` no longer claims the stale prefix) and
  `republish_same_slot_keeps_own_prefix_snapshots` (own earlier boundary
  snapshots survive a republish).
- Verification: `cargo test -p infer-metal --release --no-default-features
  --features metal` → 10 passed; `cargo test -p cli --release
  --no-default-features --features metal,no-cuda` → 125 passed;
  `cargo clippy -p infer-metal --release --no-default-features --features
  metal --tests -- -D warnings` clean.

**Bench pending.** The mandatory matched A/B on the canonical Metal model has
NOT run yet. The lead session runs `./scripts/bench_guidellm.sh
metal-pagestore-prompt-only --model mlx-community/Qwen3.6-35B-A3B-4bit` vs
the latest Metal baseline after integration; this entry gets the Δ% row then.

## Rule

A device-side cache keyed by recyclable host ids must invalidate on id reuse,
and publish work must be reachable by a consumer — check who can ever attach
(here: radix publishes prompt pages only, `prefix.rs publishable_tokens`)
before paying per-token publish cost.
