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

## UPDATE — width deleted; chain-fold depth sweep (7d316523, same binary)

ckl's minimal-scheme verdict landed: a wide candidate without its row ≡ the
bonus (free), and candidate rows cost more in MoE expert reads than their
continuation returns → the complete tree was DELETED (`7d316523`, −708
lines). Chain + fold sweep on the EOS-clean essay workload:

| chain+fold depth | tok/s | A | TPOT |
|---|---|---|---|
| **d2** | **38.04 (+17.0% vs no-spec 32.52)** | 1.98 | **26.3 ms** |
| d3 | 36.35 | 2.17 | 27.5 ms |
| d4 | 30.64 | 2.11 | 32.6 ms |

d2 is the data-confirmed sweet spot (depth ≥3: +A never pays its draft level
+ verify row). Needle exact ×6 (d2), ×3 (d3); d4 speed-only.

## Base-regression hunt (open): today 33.8 vs the 6-11 record 44.04

Same serve script, same harness, co-tenant-free (the 8×51.5 GB "[Not Found]"
nvidia-smi PIDs were OUR OWN ranks seen from another PID namespace), CPU
idle, RUST_LOG=warn no-op, clean reboot no-op. Trace census vs era docs:
kernel counts IDENTICAL (the +172 pack_quantize / −215 scalar-gemv shift is
the licensed 6-07 DeepGEMM lever), checkable durations identical (mhc 8.46
vs 8.51 µs, FlashMLA 15.2 µs). The delta sits in NCCL wait (rank-arrival
skew: ranks 2/4 arrive last, others spin) + launch dust. The decisive
control — rebuild exactly `d7be8c9b` and bench in today's session — was
started but yielded to the sweep; it is the ONE remaining run to split
"binary regression" from "era session context (e.g. DeepGEMM JIT cache
state)". Until then, all within-session deltas (everything above) stand;
cross-day absolutes don't (the standing rule, re-proven).

## Rules (appended)

- **A stale background waiter is a co-tenant**: a leftover engine-ready
  probe loop fired its own bench against the next serve and halved B=1
  throughput (38 → 18) while doubling step counts. Kill prior clients
  before measuring; treat step-count-vs-token mismatch as the tell.
- **nvidia-smi `[Not Found]` compute PIDs on a pod are often your own ranks
  in another PID namespace** — verify by stopping your serve before
  declaring a co-tenant.

## State

- Default runtime behavior unchanged (chain depth-1, no fold). Best known
  config (opt-in): `ARLE_DSV4_MTP_COMMIT_FOLD=1 ARLE_DSV4_MTP_UNCLAMP=1
  MTP_DEPTH=2` → 38.04 tok/s = 26.3 ms TPOT, the best recorded number on
  this lane's clean-workload harness.
- Default-flip license: +17% clears the bar on this shape; owes one more
  binding shape (long-input generation) per the multi-shape rule, plus the
  d7be8c9b control to close the base question.
- Pod left at `7d316523` serving the best config (ckl's in-flight main
  temporarily gates `--spec-type` off CUDA).
