# Metal prefix restore survives past the first turn

Date: 2026-09-02 · Runtime: `8540af887` + this fix · Metal, M4 Pro 48 GB

## Context

A coding agent re-sends the whole conversation every turn. On Metal, the
prefix cache restored the shared prefix on turn 2 and then licensed 0 blocks on
every later turn, so turns 3+ re-prefilled the whole prompt. Two defects in
`crates/infer-metal/src/kv_ssd.rs`, found with a 12-turn agent-shaped
conversation (`scripts/bench_multiturn_ttft.py`, new) and `infer_metal=debug`:

1. `publish_slot` minted a new logical id for every page whose owner changed.
   A restored slot republishes the radix's shared pages under a new
   `slot_epoch`, so all of them counted as "recycled to a new occupant" and the
   alias-hazard prune deleted every earlier boundary snapshot
   (`prefixes 5->0` in the log).
2. Radix dedup keeps the ORIGINAL page for a block the slot recomputed, so the
   boundary snapshots a restored slot did leave were keyed to its own page
   chain, which the radix never hands out again. The seam contract already
   names this case (`save_prefix_sidecar` receives `slot_pages` for exactly
   this repair); the Metal impl ignored the argument.

## What Worked

- `MetalSlotState.restored_len` (set by `materialize_slot_from_prefix`, explicit
  in `from_arrays`): pages below it keep their logical id on republish.
- `save_prefix_sidecar` re-keys each boundary snapshot from the slot chain onto
  the radix-canonical chain when the two diverge
  (`alias_snapshots_to_canonical_chain`).
- Two store tests: `restore_republish_keeps_prior_boundary_snapshots`,
  `sidecar_aliases_snapshots_onto_radix_canonical_chain`. `cargo test -p
  infer-metal --release --no-default-features --features metal`: 14 passed.

Qwen3.5-0.8B-MLX-4bit, `arle serve --backend metal`, default flags, 12 turns
(4.8K-token system prompt, +~350 tokens of tool output per turn, 8.6K tokens
at turn 12), greedy, `max_tokens` 32, TTFT to the first streamed delta:

| Arm | Turn 1 | Turns 2–12 median | Turn 12 | Licensed blocks |
|---|---:|---:|---:|---|
| Before fix (8 turns measured) | 1.61 s | 2.01 s | 2.27 s (turn 8) | 300 on turn 2, then 0 |
| After fix | 1.95 s | **180 ms** | 202 ms | tracks the raw match every turn |
| mlx-lm 0.31.2 `mlx_lm.server --prompt-cache-size 4`, same weights | 1.26 s | 249 ms | 248 ms | n/a |

Turn-1 variance across arle runs on this box: 1.15–1.95 s (same prompt; the
machine carried 17–20 GB of macOS swap from other processes throughout).

Correctness: greedy output of turns 3 and 6 in the restored chain equals the
cold single-prompt output byte for byte. Needle ladder 115/300/1000/2000/4000/
8000 ×3 on the fixed binary: 18/18 exact, every length deterministic; runs 2–3
of each length attached a restored prefix (17 attaches).

Qwen3.6-35B-A3B-4bit on the same box: the resource guard rejected startup
(available 21.9 GiB vs 23 GiB required) and with the guard relaxed
(`ARLE_METAL_AVAILABLE_RESERVE_MB=512 ARLE_METAL_RUNTIME_HEADROOM_MB=768
--allow-swap --memory-budget-bytes 22GiB --mem-fraction-static 0.97`) the
restore path engaged on all 11 later turns but decode ran at 16 tok/s against
the 85 tok/s baseline, so those TTFTs (1.5–1.9 s per turn) measure swap, not
the runtime. Pending a clean box.

Snapshots: `benchmarks/snapshots/2026-09-02-metal-multiturn-ttft-{arle,mlx-lm}-0.8b.json`.

## Rule

- A prefix-restore test must run at least three turns. Turn 2 exercises restore;
  only turn 3 exercises republish-after-restore, which is where both defects
  lived.
- When the seam passes a repair argument (`slot_pages`), grep every backend impl
  for `_slot_pages`: an ignored underscore parameter is a contract half
  implemented.
