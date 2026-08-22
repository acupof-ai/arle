# --kv-recall repaired: 0/48 → 48/48 mid-context under TP=2 CP=2 — 2026-08-18

## Context

`--kv-recall` on ThinkingCap-Qwen3.6-27B-FP8 lost 100% of mid-context content
under TP=2 CP=2, aborted the process without CP, and emitted degenerate tokens
above the prefill chunk size. Five defects, found and fixed in one session; the
survey that found them is
[`errors/2026-08-18-kv-recall-loses-mid-context-under-cp.md`](../errors/2026-08-18-kv-recall-loses-mid-context-under-cp.md).

Gate: `scripts/needle_concurrent.py <port> 16 16000 3 50` — 16 concurrent rows,
a distinct needle per row, row-unique filler, needle at 50% depth. Depth 50 is
the load-bearing part: a front needle lands in the pinned sink window and passes
on a completely broken configuration.

## Result

| Arm | Before | After |
|---|---|---|
| TP=2 CP=2, depth 50 | 0/48 | **48/48** |
| TP=2 CP=2, depth 50, L1÷13 + L2 1 GiB/rank + L3 | — | **48/48** |
| TP=1 CP=1, depth 50 | abort, then 11/48 | **48/48** |
| TP=2 CP=2, depth 0 (sink) | 48/48 | 48/48 |
| TP=2 CP=2, serial ladder ×3, 115→8000 (`RAW=1`) | — | 27/27 exact, deterministic |

Treatment arm proven live: **19 local L1 pages held during decode for a
34,926-token context**, against 1092 for the full context. The 544-token working
set is 17 pages under CP=2, so recall is still restricting — the needle passes
because the selector picks the right block, not because recall degenerated to
all-resident.

Tiers under CP, measured in the L1÷13 arm: L2 1 GiB/rank, L3 spilling 951 MB per
rank into per-rank stores
(`arle-kv-recall-st-<epoch>-format-1-world-4-rank-{0,1,2,3}-page-524288`),
1.24 GB of tier writes on the rank-0 counter.

## What worked

Five fixes, each gated before the next was written.

**1. FA3 sized from the full context** (`1bf969aa9`). `for_recall_decode` passed
`total_len` as `seqlen_k` without CP while handing FA3 a page table holding only
the working set — the combine kernel indexed splits with no pages behind them
and aborted in `flash_fwd_combine_launch_template.h:52`. The CP branch already
derived the length from the table, so both branches collapse to one.

**2. The recall cycle ran on every prefill chunk** (`26014ff0e`). Evicting at an
earlier chunk's tail left `EVICTED_PAGE` sentinels in a page table the next
chunk's prefill must still attend through. Any prompt above the chunk size
(clamped to 4096) returned token 0. Non-final chunks now do a plain paged
prefill.

**3. The block representation carried no ranking signal** (`099d764cd`). K is
cached post-RoPE, so a mean over `l_bs` consecutive positions rotates each key by
a different angle and the high-frequency channels cancel. Replaced with a
per-channel `[min | max]` envelope scored as `Σ_d max(q_d·lo_d, q_d·hi_d)` — an
upper bound on the block's true max `q·k`, which is what makes a top-k selection
admissible. 11/48 → 34/48 at TP=1.

**4. Recall state survived a prefix-hit slot reuse** (`588bac752`). The
new-occupant reset lives under `row.start_pos == 0`, which a prefix hit never
reaches. The prior session's envelopes stayed at the same block indices and
`update_block_reps` only grows past `len()`, so they were never recomputed.
34/48 → 48/48 at TP=1.

**5. The scoring query was never captured under CP** (`8e98f5cfe`). Recall with
`attn_cp>1` is forced onto the ring prefill branch, which had no `layer0_query`
parameter, so the scorer received an empty vec, every block scored `0.0`, and
`plan_recall`'s index tie-break kept middle blocks `0..top_k` for every request.
The capture now lives in `ring_prefill_full_attention`, reading `q_ring`
(head-major, unlike the dense path's row-major `q_prepped`), broadcast from the
tail-owning cp rank. Selection is reconciled across ranks by widening the key
envelopes over `attn_cp` (min/max) and summing scores over `attn_tp`, both
host-side over an all-gather sized from the new `infer_core::recall_block_count`
— `block_reps.len()` is not rank-invariant and would hang the collective.
0/48 → 48/48 at TP=2 CP=2.

## Simplification pass

A behaviour-preserving cleanup followed, re-gated to the same numbers (TP=2
CP=2: 48/48 depth-50, 16/16 depth-0, 27/27 serial ladder; TP=1: 48/48;
residency 18 pages). Net −66 lines. The load-bearing parts:

- Block envelopes became two flat `f32` buffers instead of `Vec<Vec<f32>>`, so
  the cross-shard reduction hands the collective a slice with no marshalling —
  `widen_envelopes` collapsed to two resizes and two calls, and `resize`'s fill
  value *is* the min/max identity. ~1500 fewer allocations and ~12 MB less host
  copying per 16K prefill.
- `broadcast_f32_over` deleted: exactly one cp rank captures a query and the
  rest pad to zeros, so a sum over `attn_cp` lands that rank's vector exactly.
  That also deleted a second derivation of which rank owns the prompt tail,
  which did not model the capture's own `rows > 1` condition — the two could
  have disagreed and rooted the broadcast on a rank that captured nothing.
- The query captures now copy only the rows they average instead of the whole
  `q` buffer: 131 MB → 262 KB on the ring path at rows=8000, 268 MB → 262 KB on
  the dense path at 16K.
- `plan_recall` calls `recall_block_count` rather than restating its arithmetic.
  That count sizes a cross-rank collective, so a drift between the two copies
  would hang the group.

## Deferred cleanups (2026-08-19, `e5c20c13c`)

Re-gated on pod at every arm above, all unchanged — including residency at 19
local pages, so none of it turned recall into a no-op.

- The block representation moved to `infer_core::recall` (`fold_key`,
  `score_block`), and **Metal stopped running the mean scorer**. Both backends
  now rank blocks identically. A unit test pins the property: on keys that
  cancel channel-wise, the envelope still bounds the true `max q·k` while the
  mean scores exactly zero — the failure that cost this whole investigation.
- `RecallConfig::VALIDATED` replaces two hand-copied constant sets;
  `is_page_aligned` moves the region invariant onto the config and off the
  per-prefill-row path.
- `PrefillRow::end_pos` / `is_final_chunk` replace fifteen open-coded
  derivations across five crates. The predicate existed as both `==` and `>=`.
- `loader::local_kv_extent` is now the single place a sharded page table's KV
  extent is computed; `refresh_sharded_decode` had a hand-rolled `owns_page`
  against `ShardSpec`'s claim to be the only ownership predicate.
- `reduce_f32_over` runs an in-place device all-reduce instead of gathering
  `world_size` copies to the host: ~18 MB of host copying per call removed.
- `update_block_reps` sorts its readbacks by physical page and coalesces runs —
  ~1000 latency-bound 32 KB copies per 16K prefill collapse to a handful.

## Perf (2026-08-19) — the repair is correct and the policy still does not pay

The gates above are all correctness. The wall-clock measurement that was owed:
TP=2 CP=2, Qwen3.6-27B, same binary, back-to-back on the same four GPUs. TTFT is
the `max_tokens=1` wall; decode tok/s is `(N-1)/(wall_N - wall_1)`, which
cancels prefill.

| ctx | TTFT on/off (s) | decode tok/s on/off |
| --- | --- | --- |
| 1,787 | 0.62 / 0.56 | 94.6 / 113.5 |
| 7,412 | 2.50 / 2.36 | 144.7 / 150.6 |
| 14,913 | 5.07 / 4.76 | 107.6 / 148.2 |
| 30,913 | 10.59 / 10.07 | 112.8 / 145.3 |

**Uniformly slower.** TTFT +5–11% is expected — the scoring cycle is pure added
work at the prefill tail. Decode −4…−27% is not what the design predicts: the
page table shrinks from ~2000 pages to 19, so attention should get cheaper.
Re-running the ON arm at `mem_fraction_static 0.25` (L1 4112 pages instead of
52885) did not recover it either — decode 86.4 / 122.1 / 117.8 / 123.4 tok/s at
the same four lengths.

Read plainly: shrinking the attended set buys nothing while HBM is not the
binding constraint, and the cost is paid regardless. The remaining decode gap is
un-attributed — the recall decode path is a separate route from the default one,
and no profile was taken.

**Consequence.** `--kv-recall`'s working-set restriction is no longer the
direction; L2/L3 are wanted as a *lossless* capacity extension instead (full
attention, only residency moves). The five correctness fixes stand on their own
— they are what makes the tiering plumbing trustworthy — and the plumbing is
what the new target reuses.

## Rule

Order the fixes by what unblocks measurement, not by what looks most important.
The CP wiring (5) was the headline defect and the last thing written: (2) had to
land before any long prompt could be measured at all, and (3) and (4) had to land
before a CP gate could distinguish "the query arrived" from "the selector works".
Wiring the query up first would have moved CP from 0/48 to 11/48 and read as a
failed fix.

Prove the arm is still engaged after a correctness fix. A recall fix that
silently degenerates to all-resident passes every needle gate. The residency
number (19 pages against 1092) is what separates "selects correctly" from
"stopped selecting".

## Notes

- The gate's default `/v1/chat/completions` route returns empty completions on
  this model at every length, including lengths where recall is a no-op
  (115, 241). Pre-existing and unrelated to recall; use `RAW=1`.
- Tier read counters stay at 0 on single-shot runs by construction: under 2D the
  planner emits one prefill row per prompt, and prefetch only has work when an
  earlier row already evicted. The read path is first reachable on a second
  prefill row for the same slot (multi-turn continuation) and is still ungated.
- Not covered here: `top_k`/budget is still hardcoded at
  `recall.rs:default_recall_config`, selection is still one scalar per block from
  layer 0 only and frozen at prefill. Those are quality ceilings, not defects,
  and the gate now measures them.
