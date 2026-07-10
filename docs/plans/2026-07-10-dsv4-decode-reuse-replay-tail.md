# DSv4 decode-region reuse — replay-tail carry rebuild (graph-compatible)

> Status: Active

**Verdict**: Replace the eager-only per-decode-step boundary capture with a
graph-compatible design: capture positional CONTENT at natural sync points
(prefill chunk ends + request finish), and rebuild the small transient CARRY
(SW ring + compressor overlap) by REPLAYING the last ≤128 tokens at restore.
Decode hot loop / CUDA graph runs untouched. Zero extra storage.

## Why the redesign (not a patch)

The current mechanism captures boundary sections (ring + overlap) during decode
via `publish_completed_prefix_pages` — a per-step D2H + `sync()`. Two fatal
flaws:
1. **Graph-incompatible** (ckl 2026-07-10 hard constraint): a per-step D2H/sync
   cannot live inside a captured CUDA graph. Eager-only by construction; dies
   the moment single-GPU graph decode is on. See
   [[feedback_decode_features_must_be_graph_compatible]].
2. **Flaky + storage-heavy**: boundary sections only capture when a tick ends
   exactly on a 128-boundary (missed under MTP overshoot), and the ring is
   ~150KB/layer/boundary (≈2× pool footprint if captured everywhere). W1
   floored decode reuse at the prompt boundary (384) despite the content being
   present — this whole class disappears with the redesign, so the 384
   attribution is moot (we delete the mechanism, not patch it).

## Design

**Carry = f(last ≤128 tokens).** The SW ring is the raw KV of the last
`sliding_window` tokens; the compressor overlap is the last completed block's
raw rows. Both are recomputable by re-forwarding the tail — no need to store
the transient instant.

- **Capture (positional, at sync points, graph-irrelevant):**
  - Prefill chunk ends (already 128-aligned) — unchanged, covers prompt-prefix
    sharing (Lane A / W2).
  - **NEW: request finish** — one D2H of the slot's compressed CONTENT
    (`compressed` staging + DSA key-cache rows) for the generated region,
    BEFORE `free_slot_pages`. Runs at `finish_slot` (already a sync point).
  - **DELETE**: boundary sections (`overlap_kv/score`, `idx_overlap_*`,
    `ring`) from `Dsv4LayerPageState`; the per-step decode boundary capture;
    the `boundary` flag; the `meta.boundary` gate in `reusable_prefix_blocks`.
- **Restore (rebuild carry, one-off, outside the decode graph):**
  1. Restore compressed content into the slot (existing copy path) up to the
     matched frontier page.
  2. **Replay-tail**: run a mini-prefill over the last `min(sliding_window,
     matched_len)` tokens (token ids from the radix chain), attending to the
     restored content, to regenerate the SW ring + compressor overlap in the
     slot's live registers.
  3. Continue decode normally.
- **`reusable_prefix_blocks`**: reuse every page whose CONTENT is present (drop
  the boundary gate) — carry is rebuilt, not required to be captured. Still
  fail-closed on demoted keys.

## Steps (file:line)

1. `attention/prefix_state.rs`: delete boundary sections from
   `Dsv4LayerPageState` (overlap_kv/score, idx_overlap_kv/score, ring) + their
   codec push/read; drop `boundary` from `Dsv4PrefixPageEntry` + `page_meta`.
2. `executor.rs:2378` `publish_completed_prefix_pages`: drop the `boundary`
   computation + boundary capture; keep positional content capture. Add a
   finish-time full-content capture (new fn, called from the engine's finish
   path before `free_slot_pages`).
3. `executor.rs:2508` `reusable_prefix_blocks`: drop `meta.boundary &&
   page_end % align` gate → count every present-content page (still break on
   missing meta / demoted).
4. `executor.rs:2544` `restore_prefix_state` + `dsv4.rs:1222`: after restoring
   content, invoke the replay-tail mini-prefill (reuse the existing prefill
   forward over the last ≤128 token ids) to rebuild ring + overlap; drop the
   ring/overlap restore-from-entry path.
5. `dsv4_attention.cu`: **no kernel change** — replay-tail uses the existing
   prefill forward. (Confirms graph-compat: decode kernels untouched.)
6. `infer-core` finish path (`lib.rs:960` `finish_slot`): call the new
   finish-content-capture before `free_slot_pages`.

## Verification (pod)

- Correctness: `needle_gate.py` x3 same-config repeats vs baseline envelope
  (MoE non-det floor), TP=4 DSv4-Flash-FP8.
- **W1 multi-turn (the target)**: P + R + follow-up; assert turn-2 matched
  length reaches `floor((pt1+|R|)/page)·page` and `crosses_into_R=true`, needle
  exact. Bake into `scripts/prefix_reuse_gate.py`.
- **Graph lane**: single-GPU (or graph-forced) decode + a reuse turn — assert
  the decode graph captures (no eager fallback) AND reuse still works. This is
  the regression guard the old design could never pass.
- Perf: replay-tail restore vs cold prefill (expect the W3 ~8× restore win
  extended to the generated region); ≤128-token replay is bounded regardless
  of sequence length.
- Bench entry per §Benchmarks; wins/ on pass.

## Non-goals

- No cross-request reuse of an IN-FLIGHT (still-decoding) request — finish-time
  capture covers completed-turn reuse (the real multi-turn case).
- No per-step decode capture of any kind (the graph-incompat trap).
