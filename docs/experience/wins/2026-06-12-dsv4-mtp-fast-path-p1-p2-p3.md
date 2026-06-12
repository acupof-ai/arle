# DSv4 MTP fast path P1+P2+P3 — tree spec decode beats no-spec; MoE expert-read scaling is the verify floor

Commits: `a88483c9` (P1 batched tree verify), `bac82d87` (P3 level-batched
draft + device top-k), `ccc7bd7e` (P2 commit fold, opt-in), `c0217f74`
(batched verify extraction), `9c86ac47` (step-phase probe). Pod: 8×H20 TP=8,
EOS-honoring 256-token essay workload, same binary per comparison. Plan:
[2026-06-12-dsv4-mtp-tree-fast-path](../../plans/2026-06-12-dsv4-mtp-tree-fast-path.md).

## Ladder (each stage needle-gated 3k/6k ×3 @0.5 — ALL exact ×6)

| stage | tok/s | A | step | note |
|---|---|---|---|---|
| no-spec baseline | 32.52 | 1.0 | 30.7 ms | |
| tree d2k2, per-row verify (pre-P1) | 11.49 | 2.43 | 164 ms | ring-replay lane |
| + P1 batched verify + P3 batched draft | 20.96 | 2.38 | 113 ms | +41% |
| + P2 commit fold | **33.64** | 2.29 | 68 ms | **first config over no-spec (+3.4%)** |
| + batched extraction | 33.36 | 2.29 | 69 ms | wash — the 10 ms estimate was a mis-attribution |
| d3k2 fold (15-node tree) | 28.84 | 2.34 | 81 ms | depth 3 KILL: q₂=0.41 craters, +0.05 A for +13 ms |

## Step-phase profile (sync-bounded, d2k2 fold, ×12 steps)

`capture=1.8  draft=4.2  verify=55.8  commit=4.9 ms` (Σ=67.6 ✓ matches)

- P3 and P2 hit their budgets exactly (draft 4.2 vs ~5 planned; commit 4.9
  vs ~4).
- **The verify forward is 55.8 ms, not the ~32 ms one-forward floor the plan
  assumed.** Root cause: **MoE expert-read scaling** — a 7-row chunk
  activates up to 7×8 distinct routed experts per layer vs 8 for one decode
  row, so routed weight reads grow ~linearly in rows. "A forward costs the
  same for 1 token or a whole tree" holds for dense weights only; on a
  fine-grained MoE the verify chunk costs ~1.8× a decode forward. This also
  explains why batching the verify attention bought only 14 ms (70 → 56):
  the per-row lane's cost was never launch-dominated either.

## Corrected theory

- d2k2 architectural ceiling on this model ≈ draft 4 + verify ~50 (MoE
  floor) + commit 5 + capture 2 + scheduling 2 ≈ 63 ms → **~36 tok/s; the
  shipped 33.6 is within ~7% of it.**
- The remaining levers are structural, not engineering overhead:
  1. **Tree pruning** (EAGLE-2 style budget trees): fewer rows → fewer
     distinct experts. e.g. dropping level-2 under the rank-2 child: −2 rows
     (~−6 ms) for ~−0.1 A — likely +5-8%.
  2. **The q-curve** (training axis): an OPD-distilled depth-robust head is
     the only way past A≈2.4 — kernels are done paying.
  3. MoE-side: grouped-GEMM expert batching efficiency at m≈7 (nsys first).

## Rule

- **On a fine-grained MoE, spec-decode verify cost scales with DISTINCT
  activated experts, not just weight bytes** — budget `verify ≈ (1 +
  α·(rows−1)) × decode_forward` with measured α (~0.13/row here), never
  "one forward is one forward".
- **Sync-bounded phase probes beat component estimates**: two plausible
  decompositions (extraction ~10 ms, verify ~32 ms) were both wrong; one
  20-line `ARLE_DSV4_MTP_STEP_PROFILE` probe settled it in a single serve
  cycle.
- Depth-3 trees re-confirmed dead with the nextn-1 head (q₂ ≤ 0.41) — width
  beats depth at every measured operating point.

## State

- Default runtime behavior unchanged (topk=1 chain; tree + fold are env
  opt-ins: `ARLE_DSV4_MTP_TOPK=2 ARLE_DSV4_MTP_UNCLAMP=1 MTP_DEPTH=2
  ARLE_DSV4_MTP_COMMIT_FOLD=1`).
- Default-flip license deferred: +3.4% is real but thin; flip decision after
  tree pruning lands (target ≥ +10%) or with the OPD head.
