# DSv4 decode-region reuse — finish write-through (working area / settling area)

> Status: Active — supersedes the killed replay-tail plan
> (`2026-07-10-dsv4-decode-reuse-replay-tail.md`).

**Model (ckl 2026-07-10)**: working area (live HBM slot) stores NO carry — it's
live in registers. On settle (request finish), write the full state THROUGH to
the settling area (L2/L3 = the existing content-keyed `Dsv4PrefixStatePool`).
Restore promotes it and prefills only the new suffix. Graph-safe (capture at the
finish sync point, decode hot loop untouched) and correct (real carry captured
at P1, never rebuilt — dodges the killed replay-tail's stale-`prev_overlap` bug).

## Why this is correct where replay-tail was not
Replay-tail rebuilt the carry and read a stale `prev_overlap` at the replay
boundary. Here we CAPTURE the real carry (`prev_overlap` + ring + pending) at
the exact finish position P1, where it is live. No rebuild, no stale read.

## Reuse, don't build
- **Settling store**: `Dsv4PrefixStatePool` already IS L2 (host) + L3 (mmap),
  content-keyed by radix page id. Extend its entry; don't add a slot_tier.
- **Machinery reference**: Qwen3.5 `demote_slot`/`promote_slot`
  (`executor/qwen35.rs:474/525`) — capture-full-image → `sync()` →
  `insert_chunked`; read → `from_bytes` → restore. DSv4 mirrors the shape but
  writes into the prefix pool, not a separate slot_tier.

## Steps (file:line)
1. **Decode stops storing** — `executor/dsv4.rs` `publish_completed_prefix_pages`
   (decode call site, forward_decode_batch): make the decode-lane call a NO-OP
   (the D2H+sync per tick is the graph-incompat trap). Prefill-boundary content
   capture stays (prefill isn't graphed). This alone makes decode graph-safe.
2. **Finish writes through** — new `capture_finish_frontier(slot, tokens)` on
   the DSv4 executor: one D2H of (a) generated-region CONTENT for every
   generated page, (b) the frontier CARRY at P1 = `prev_overlap_kv/score`,
   `idx_overlap_*`, `ring`, `pending_kv/score` (all live in registers at
   finish), into the pool keyed by the radix chain. Ends in one `sync()`.
   Wire from `infer-core` `finish_slot` (`lib.rs:960`) BEFORE `free_slot_pages`,
   via a new default-noop seam method (`infer-seam`, device-neutral).
3. **Entry** — `attention/prefix_state.rs`: the frontier carry sections
   (overlap/ring/**+pending**) attach to the FRONTIER page's entry, captured
   once at finish (not per-boundary). Drop the per-boundary `boundary` flag
   semantics; a page is a valid restore frontier iff it carries the finish
   carry. Content pages need only content.
4. **`reusable_prefix_blocks`** (`executor/dsv4.rs`): reuse = content-present
   prefix up to the last page that carries the finish frontier carry (break on
   missing content / demoted key). Drop the `page_end % align` boundary gate.
5. **Restore at P1** — `restore_prefix_state` (`executor/dsv4.rs` +
   `dsv4.rs:1222`): restore content pages + the frontier carry, set the slot to
   the EXACT P1 (including the partial last block via captured `pending`), drop
   the `matched_len % 128` alignment `ensure!` for finish-captured frontiers.
   The next turn prefills only [P1, end].
6. **Opt-in flag** first (`--dsv4-decode-reuse` / env), default OFF; baseline
   byte-identical. Flip default only after the pod perf+correctness license.

## Hardest sub-problem (flag, don't guess)
Restore-at-P1 with a non-64-block-aligned frontier: the radix matches at 64-block
granularity (frontier B = floor(P1/64)·64 full blocks), but the pool entry
extends to P1 (the partial block [B, P1) lives in `pending`). Restore must set
the slot to P1 (content [0,B) from pages + [B,P1) from pending + carry at P1),
and report the restored length as P1 so the engine prefills from P1, not B.

## Verification (pod, opt-in flag ON)
- Correctness: `needle_gate.py` x3 same-config vs baseline envelope, TP=4.
- **W1 multi-turn**: P + R + follow-up → turn-2 restores to P1 (the whole prior
  turn), prefills only the follow-up; needle exact; assert reuse length ≈ P1
  (not the prompt floor). Bake into `scripts/prefix_reuse_gate.py`.
- **Graph lane**: single-GPU (graph on) + a reuse turn → decode graph captures
  (no eager fallback) AND reuse works — the regression guard the old design
  could never pass.
- Perf: restore vs cold-prefill of the reused span; storage = one frontier
  carry per finished turn (small), tier-budgeted.
- Bench entry per §Benchmarks; wins/ on pass.

## ROI note
v1 is correct today; this is opt-in and reuses the pool + Qwen3.5 pattern (not
greenfield). Reland decision: implement behind the flag, pod-verify, flip
default only if the long-re-sent-turn workload shows the win.
