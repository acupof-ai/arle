# DSv4 MTP tree verify via SW-ring node schedule (topk≥2, one frozen forward)

Commit: `b6e8d9d7`. Pod: 8×H20 TP=8, DeepSeek-V4-Flash, allreduce MoE +
DeepGEMM, FlashMLA decode ON, batched verify ON.

## Context

Spec-decode's only lever is **A = accepted tokens per verify forward** (the
forward is weight-read-bound; see
[2026-06-11-dsv4-mtp-frozen-kv-p1-longctx-fix](2026-06-11-dsv4-mtp-frozen-kv-p1-longctx-fix.md)).
Raising A needs a topk≥2 draft tree verified in ONE forward. The blocker was
the tree mask: how do tree rows attend to exactly their own branch?

## What Worked

**No new kernel.** Two observations collapse the problem:

1. **The compressed/DSA side is tree-invariant under the frozen verify.**
   P1-A pins the CSA selector to the committed compressed keys, so every tree
   row sees the identical committed set — topology-free.
2. **Only the SW/FP8 rings are position-keyed** (`pos % sliding_window`), so
   the only conflict is same-depth siblings sharing one ring slot.

So the tree mask reduces to a host-precomputed **row schedule** over the
existing per-row verify loop (BFS row order):

- **BFS order** keeps every speculative ring write at a position whose
  displaced committed slot (`pos - sliding_window`) is already outside every
  later row's window — chains rely on the same invariant implicitly.
- **Park/replay contested slots**: after a row that later branches chain from,
  D2D its just-written SW slot + FP8 data/scale into a per-node scratch
  (`spec_nodes`, `MAX_SPEC_TREE_NODES`=64 slots/layer); before a row whose
  ancestor's slot a sibling overwrote, replay it. Exact fix-up set from a
  per-depth slot-owner simulation (`DraftTree::verify_schedule`), identical
  across layers.
- **The draft loop needs the same fix-ups on ONE layer** (the MTP frozen
  target layer — `mtp_forward` writes its ring per expansion), driven by the
  same owner walk.
- **Chain = degenerate case**: topk=1 produces strictly increasing positions
  and zero fix-ups — the validated per-token verify byte-identical, no gate
  flag needed.
- **Commit needs no tree-specific restore**: every branch's writes land in
  `start_pos..=start_pos+depth`; the re-forward overwrites the accepted
  prefix and `restore_spec_ring_tail` (at the CAPTURED depth) replays the
  rest.

Schedule simulation unit-tested (chain → zero fix-ups; full topk=2 tree →
exact restores/saves; ragged tree → repeated positions still require the
batched path: `is_chain` gates the per-row fallback).

## Results

GATE (needle, depth 0.5, same binary b6e8d9d7, same day):

| config | 3000 | 6000 | verdict |
|--------|------|------|---------|
| no-spec (control) | exact 5/6, partial 1/6 (`738292`) | — | base-path nondet floor at this shape |
| chain depth=1 | exact 4/6, partial 2/6 (`738292`, same signature) | exact 3/3 | PASS — matches the no-spec floor |
| tree topk=2 depth=2 | **exact 3/3** | **exact 3/3** | **PASS** — tree verify correct with full SW-ring wrap |

The `738292` digit flip appears in the NO-SPEC lane too (same wrong token), so
it is the base path's nondeterminism floor on this prompt shape, not spec
machinery. (The 2026-06-11 "exact ×12" logs on the pod are needle depth 0.00 —
a different, easier envelope; depth 0.5 is the binding one going forward.)

PERF (ab_decode.py, 5×128 tok, p50 of warm, same binary same shell, two env
flips between configs):

| config | tok/s p50 | A (committed/step) | step cost | Δ vs no-spec |
|--------|-----------|--------------------|-----------|--------------|
| no-spec | 33.07 | 1.0 | ~30 ms | baseline |
| chain depth=1 | 16.68 | 1.36 (Δaccept 170/470 steps) | ~82 ms | -50% |
| tree topk=2 depth=2 | 11.49 | **1.81** (Δaccept 285/350 steps) | ~158 ms | **-65% KILL** |

(Chain measured 20.9 in the 2026-06-11 session — today's same-binary 16.68
re-confirms cross-day baselines are ghosts; only this triplet is comparable.
A on this fixed ab_decode prompt is workload-specific: chain hit 1.60 on the
needle-mixed window, 1.36 here; the tree's 1.81 is on the same prompt as the
chain's 1.36, so the A gain is real and understated if anything.)

**Verdict: tree default KILLED on wall-clock; the A axis itself is LICENSED.**
The tree raises A exactly as designed (1.36 → 1.81 on the same workload, the
first measured tree acceptance gain), but the verify forward is NOT
~constant-cost in rows: the per-row attention loop (7 rows × 61 layers of
decode-attention launches + 2 D2D row copies) plus 3 `mtp_forward` expansions
(each a full-vocab lm_head GEMV + 129k-logit D2H host top-k) roughly double
the chain's step. Differencing tree−chain (158−82 ms over +5 verify rows
+2 mtp calls) puts a verify row at ≈10 ms and an mtp expansion at ≈10 ms —
launch volume, not weight reads. The "weight-read-bound forward" premise holds
for the batched point-wise/MoE half only; attention is the serialized
exception.

## Next lever (the real unlock)

The accept machinery is now correct and banked; the cost side needs the verify
attention BATCHED across rows:

1. **Multi-query mid-sequence decode attention**: all tree rows' Q in one
   launch per layer with per-row KV index sets — exactly the vendored FlashMLA
   sparse forward shape (`s_q` queries, per-query `indices`/`topk_length`).
   Committed compressed top-k comes from the CSA selector as today; the SW
   window rows + in-tree ancestors become per-row indices instead of ring
   replay. Target: verify ≈ 1× decode forward → tree ≈ A × no-spec.
2. **Level-batched draft**: expand a whole tree level in ONE `mtp_forward`
   batch (siblings are independent rows) + **device-side top-k** (kill the
   per-expansion 129k-logit D2H). Draft cost → ~1 small forward per LEVEL.

Break-even math: beating no-spec needs `step_ms < A × 30.2`. At A=1.81 the
budget is ~55 ms (today: 158). Batched verify + level-batched draft land
~95 ms (draft 2 levels ≈ 25 + verify ≈ 32 + re-forward ≈ 35) — still over
budget, so a NET win additionally needs the commit re-forward folded away
(commit from the verify rows' own KV instead of re-forwarding the accepted
prefix) and/or deeper trees once verify is row-cheap (depth 3–4, topk 2–3 →
A ≈ 2.2–2.6 → budget 66–79 ms). Each rung is now measurable in isolation;
none is licensed until its own same-binary A/B.

## Rule

- **Before reaching for a masked-attention kernel, enumerate which KV
  surfaces actually see the speculative topology.** Frozen-KV verify pins the
  compressed path; only the position-keyed rings conflict, and ring conflicts
  are restorable with the snapshot primitives already in the tree.
- **Order is a correctness tool**: BFS row order makes the
  displaced-committed-slot hazard structurally unreachable; DFS would read
  destroyed window positions.
