# DSv4 batched MTP decode — +81% @c=12, breaks per-row MTP's throughput plateau (fold commit)

## Context
Production DSv4 serves `--spec-type mtp --mtp-draft-tokens 2` at steady high
concurrency, but MTP **disabled the batched lane** → the executor looped per-row
`spec_step` (one full draft+verify+commit pipeline per slot, sequentially). The
理想态 (ckl): **batched MTP** — batch the draft+verify across N slots, combining MTP's
~2× acceptance with batched amortization (what SGLang's `frozen_kv_mtp_worker` does).
[plan](../../plans/dsv4-batched-mtp-decode.md). Gated OFF `ARLE_DSV4_BATCHED_MTP`.

## Result (pod 8×H20 TP=8/EP=8, allreduce, same binary, `--num-slots 16`)

| arm @ c=12 offered | agg decode tok/s | sustained active | per-req tok/s |
|-----|---|---|---|
| per-row MTP (control) | 42.18 | ~7 | 6.03 |
| **batched fold MTP** | **76.50** | **12** | 6.38 |

**+81% aggregate.** Decode-read coherence 4/4 both arms (France→Paris/Eiffel,
Italy→Rome/Colosseum, Canada→Ottawa, Egypt→Cairo/Giza) — identical answers, MTP
acceptance preserved (~0.75 accept/step depth=2, same as per-row).

### The mechanism — per-row MTP PLATEAUS, batched SCALES (the load-bearing finding)
Per-row MTP aggregate is FLAT across offered concurrency — it cannot use more slots:

| offered c | per-row MTP agg | batched fold agg |
|---|---|---|
| 4 | 42.37 | — |
| 8 | 41.71 (avg act 6) | — |
| 12 | 42.18 (avg act 7) | **76.50 (avg act 12)** |

Per-row loops `spec_step` per slot → the decode wave grows linearly with c → throughput
ceilings ~42 tok/s regardless of offered load. Batched fold runs ONE amortized wave →
aggregate scales with c. This is why batching wins: it breaks the sequential plateau.

## What it took — the iteration arc (every step measured, §0 wall-clock)
1. **Stage 1 (sub-mode 2 ring-replay attn): −44% @c=4** — un-batched the attention
   per-row MTP already tree-batches (12 FlashMLA/layer vs 4)
   ([errors](../errors/2026-06-15-dsv4-batched-mtp-stage1-submode2-regression.md)).
2. **tree-attn per slot (sub-mode 1): −44%→−32%** — attention now matches per-row, but
   still losing.
3. **re-forward commit can't win: 30.75 vs 41.71 @c=8** — batched saves c verify-forwards
   but ADDS c re-forward commits ≈ same cost. Per-row is fast *because* of fold.
4. **batched FOLD commit (the win): 76.50** — `forward_decode_batch_verify` persists each
   slot's per-layer attn-normed chain rows into the OWNING slot's `spec_normed` (codex P2
   per-slot scatter, no cross-slot aliasing); `spec_step_batched` commits via
   `commit_accepted_fold` (the cheap path, no re-forward). The 30.75→76.50 swing is fold.
5. **cap root-caused:** default `num_slots=4` (`infer-api/src/loaded.rs:69`) — the serve
   must pass `--num-slots` to reach c≥8; without it the c=8 SLO is unreachable.

## Design (gated OFF until a default-flip license)
`spec_step_batched` (executor/spec_decode.rs): per-slot capture+draft (looped) → ONE
`forward_decode_batch_verify` (dsv4.rs: tree-attn per slot + grouped MoE over all M rows
+ per-slot `spec_normed` persist) → per-slot `commit_accepted_fold`. Allreduce transport;
default OFF → main per-row path byte-identical.

## Residual (next, not blocking the win)
The per-slot DRAFT (`mtp_forward_level` looped, depth-sequential) and per-slot
capture/restore stay un-batched — batching the draft (N rows × depth levels) is the next
amortization. The verify (the dominant phase) is batched; draft is the remaining serial
tail.

## Rule
- **A batched lane must batch the axis that ceilings throughput, AND match the commit the
  reference uses.** Per-row MTP plateaus on sequential per-slot processing AND wins its
  per-step race via cheap fold; batched must batch the wave (verify+MoE) AND fold-commit
  — re-forward commit (chosen to dodge the spec_normed scatter) erased the entire win.
- **Plateau vs scale is the right frame for a concurrency lever** — compare aggregate
  ACROSS offered c, not at one point. Per-row's flat 42-tok/s across c=4/8/12 is the
  tell; a single-c number would have hidden it ([[feedback_measure_batching_before_ceiling]]).
- **decode-read coherence at c≥4 gated every iteration** — the fold commit is a NEW path;
  4/4 coherent confirmed it before the perf number was trusted (faster-nonsense guard).
- **Verify the SLO concurrency is reachable first** ([[feedback_verify_slo_lane_runs_before_optimizing]]):
  default num_slots=4 capped decode at c=4; the win only appears once `--num-slots` lifts it.
