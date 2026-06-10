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

**Bench (2026-06-10, M-series local, Qwen3.6-35B-A3B-4bit).** Canonical
`bench_guidellm.sh` is BLOCKED against the rewrite Metal server: its streaming
probe gets HTTP 400 `stream=true is deferred in R5 tranche 2` — environmental,
pre-existing, unrelated to this diff. Fallback: direct same-shape timing
(4096-token prompt, 1200-token greedy completion, c=1), same shell, two
interleaved before/after pairs (before = parent-commit binary `0f80fdd6`):

| run | before tok/s | after tok/s |
|-----|-------------|------------|
| pair 1 (cold) | 31.97 | 51.70 |
| pair 2 (warm) | 56.28 | 56.46 |

Verdict: **perf wash**. Pair 1's +62% is cold-load/thermal confounding (the
same before binary jumps 32→56 between runs); the warm matched pair is ±0.3%.
The licensed claims are correctness (stale-alias prune, unit-tested) and
bounded growth (decode-time publishes deleted by construction — the snapshot
map can no longer grow during decode), NOT throughput. `ps` RSS could not
isolate the GDR-snapshot footprint (dominated by 19 GB model page-touching;
±0 between before/after) — the memory delta stays code-derived, not measured.

## Rule

A device-side cache keyed by recyclable host ids must invalidate on id reuse,
and publish work must be reachable by a consumer — check who can ever attach
(here: radix publishes prompt pages only, `prefix.rs publishable_tokens`)
before paying per-token publish cost.
