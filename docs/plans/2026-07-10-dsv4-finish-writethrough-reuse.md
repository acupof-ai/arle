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

## Decompose finding (2026-07-10) — this is a cross-crate SEAM project, not a flag flip

The whole feature is jointly gated on a restore-at-P1 seam change; the pieces
CANNOT land separately (a flag whose ON-path can't reuse = speculative
interface shaping). Land as ONE unit behind the flag:

1. **`infer-seam`**: `restore_prefix_sidecar -> Result<usize>` = **EXTRA tokens
   restored BEYOND `matched_len`** (0 = restored exactly the match). NOT
   `matched_len` — echoing the input is a tautology and unsafe. Default +
   Qwen3.5 + Qwen dense + Metal return **0** (byte-identical; a backend that
   forgets → 0 → conservative-correct). DSv4 returns `P1 − B` (< 64).
2. **`infer-core` `attach_prefix_to_request`** (`prefix.rs:129-160`):
   `let extra = restore(...)?; let restored_len = matched_len + extra;`
   if `extra > 0`, `alloc(slot, extra)` into the slot-owned top-up band
   (partial page never radix-published → no retain/release) +
   `prefill_start_pos = restored_len.min(prompt_len)`. `extra == 0` path is
   exactly today's.
3. **`infer-core` `finish_slot`** (`lib.rs:979-985`): `capture_finish_frontier`
   between `publish_prefix_blocks` and `free_slot_pages` (new default-noop seam).
4. **DSv4** (`executor/dsv4.rs`, `attention/prefix_state.rs`): capture partial
   `[B,P1)` content + carry + NEW `pending` section; relax `reusable_prefix_blocks`
   (drop `% 128` gate, `:666`) and `restore_prefix_state` (drop `% 128` ensure
   `:704`, restore partial page + carry, set seq_len=P1, return P1). All behind
   `dsv4_decode_reuse_enabled()`; OFF = byte-identical.

**Correctness spine (enumerate-every-buffer, §0.1 — the discipline whose breach
killed replay-tail)**: before coding step 4, write the per-buffer disposition
table for the partial region `[B,P1)`: `staging` rows `[B/ratio, P1/ratio)`,
DSA rows, `pending_kv/score` (holds the trailing sub-`ratio` block that
`compressed.data` does NOT yet have), `prev_overlap`/`idx_overlap` (the 21
`ratio<16` layers, `attention.rs:7760`), FP8 band + SW-ring rebuild counters
(`restore_prefix_counters` currently ASSUMES `matched_len % ratio == 0` →
`prefix_state.rs:714`; `pending` fills the gap). Each buffer: exact range +
disposition, proven, not guessed.

## Per-buffer disposition at restore-to-P1 (the correctness spine — implement verbatim)

`B = floor(P1/64)·64` (radix frontier); partial region `[B, P1)` ≤ 63 tokens.
Per (slot, layer), for the FRONTIER entry restore. `ratio = compress_ratio`,
`ir = index_ratio` (1 for SparseIndexed else ratio). Decision: **prove each,
don't skip** — the row that killed replay-tail is `pending`.

| Buffer | State/range at P1 | Capture (finish) | Restore |
|---|---|---|---|
| `compressed.data` (staging) | completed blocks, rows `[0, P1/ratio)` | per-page content, incl. partial page rows `[B/ratio, P1/ratio)` | memcpy per page; `seq_len = P1/ratio` |
| **`pending_kv/score`** | the incomplete tail block, `P1 % ratio` tokens × `width` | **NEW section — clone_dtoh `[0, (P1%ratio)·width)`** | **memcpy_htod; the next forward derives `pending_len = P1 % ratio`** (today `restore_prefix_counters` ASSUMES this is empty — the fix) |
| `prev_overlap_kv/score` | block `P1/ratio − 1` raw rows | live at finish (existing section) | memcpy (needed by the 21 `ratio<16` layers, `attention.rs:7760`) |
| `idx_overlap_kv/score` | indexer, same | live at finish | memcpy |
| `ring` (`sw_window_cache`) | raw bf16 `[P1−128, P1)` | live at finish | memcpy (SlidingWindow layers need ONLY this) |
| `dsa_data/scale` | rows `[0, P1/ir)` | per-page content, incl. partial page | memcpy per page; `packed_rows = P1/ir` |
| `compressed.seq_len` / `indexer.seq_len` | `P1/ratio` / `P1/ir` | — | `restore_prefix_counters` (formula already floors; just feed P1, and STOP zeroing pending) |
| `fp8_kv_comp_packed_rows` / `fp8_kv_sw_bootstrapped` | `0` / `false` | — | unchanged (first decode rebuilds band from staging + ring) |
| host KvPool seq_len / `prefill_start_pos` | `P1` (not B) | — | infer-core: `alloc(slot, P1−B)` into the slot-owned partial page; cursor = P1 |
| device band cursor | `P1` | — | `mirror_full_band` already accepts an arbitrary cursor |

Ratio-0 (SlidingWindow) layers: only `ring`; no compressor/overlap/pending/dsa.
`restore_prefix_counters` (`prefix_state.rs:717`) loses its `matched_len % ratio
== 0` precondition — it must set `pending_len = P1 % ratio` and leave `pending`
restored, not zeroed.

## ROI note
v1 is correct today; this is a real cross-crate project (~5 files, delicate DSv4
partial-region core, pod-gated incl. a graph lane). ckl 2026-07-10: **build
now**. The win is concentrated in long-re-sent-turn (agentic) workloads.
