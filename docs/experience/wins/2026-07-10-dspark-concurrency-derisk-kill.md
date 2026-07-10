# DSpark concurrency scheduler (P3 Algorithm 1) — KILLED (decode-phase re-measured); OPD concurrent rollout uses plain-batched

## Context

The "DSpark doesn't batch" claim (`5bc1b3e8f`) was corrected against the paper:
DSpark's Hardware-Aware Prefix Scheduler (Algorithm 1) is *designed* to hold
throughput at concurrency. Before building it (decompose: ≥1-week batched
ragged-verify substrate, terminates in frozen attention files), one spike
measured whether there's a prize. 8×H20 GPU1, Qwen3.6-27B-FP8 + DFlash draft,
ctx=2048 gen=256 temp=0.7, per-row dspark vs plain, C∈{1,2,4,8}.

## ⚠ First spike was wall-diluted — superseded by a clean decode-phase re-measure

The first spike measured request-WALL tok/s (gen ÷ total_wall) at ctx=2048
gen=256 — the 2048-token prefill diluted decode ~3×, reporting a bogus
"dspark 40 tok/s at C=1" that contradicted our own licensed 87–175 decode band.
ckl caught it ("c1 dspark 都能做到 120 t/s"). Arithmetic reconciles: decode
~87 → wall 256/(2.1s decode + ~3.5s prefill) ≈ 40. The **relative** shape held
(util-corroborated) but the **magnitude was wrong**. Re-measured decode-phase
(ITL = (N−1)/(t_last−t_first), prefill excluded; cross-validated by an
independent prefill-subtracted method, 77.7 vs 79.8 agree; C streams as separate
processes; short ~180-tok prompt, gen 512; one frozen binary copy).

### Clean decode-phase aggregate tok/s

**greedy (temp 0):**
| C | dspark agg | plain agg | winner | dspark util | plain util |
|---|---|---|---|---|---|
| 1 | **87.6** | 45.4 | dspark 1.93× | 90% | 96% |
| 2 | **75.0** | 70.1 | dspark 1.07× | 93% | 100% |
| 4 | 73.5 | **109.3** | plain 1.49× | 93% | 91% |
| 8 | 72.7 | **252.0** | plain **3.47×** | 94% | 79% |

**temp 0.7 (OPD-realistic):** C=1 dspark 79.2 vs 45.4 (1.81×); C=2 plain 64.4>59.4;
C=8 plain 197.7 vs 58.2 = **3.40×**.

- **C=1 sanity gate PASSED first:** dspark 87.6 = 1.93× plain, in the licensed
  band — proving the harness clean before trusting the sweep.
- **dspark aggregate dead flat** (88→75→74→73); TTFT explodes 0.5s→48s —
  requests QUEUE, not batch. util pegs 90-94% = per-row saturation.
- **plain aggregate SCALES**; util DROPS 96→79% — batching amortizes weight
  reads, converting concurrency into throughput.
- **Crossover (true): greedy C=2→4, sampled C=2.** OPD best-of-N is C≈4-8,
  squarely in plain's win region.
- **Corrected deciding number: at C=8, plain 252 vs dspark 73 = 3.47× greedy**
  (the wall-diluted spike under-reported this as 2.37×). Verdict unchanged,
  stronger.
- The drafter is weak: accept ≈18%, so the single-stream spec win is only +24%
  at C=1 and gone by C=2. The ≥1-week batched-verify+scheduler substrate's
  marginal prize *over free plain-batched* is at best +24% — and only if 18%
  acceptance survives batching.

## flashmla ragged-Q (buildability, for the record)

`arle_flashmla_sm90_sparse_decode_fwd` (ffi/misc.rs:601) is a dense uniform
(b×s_q) grid with s_q strides but **no cu_seqlens_q** — uniform padded-chain
batched verify is buildable-under-freeze; ragged per-slot lengths need a frozen
ABI edit. Caveat: that's the DSv4 MLA path; Qwen dspark verify routes through
the FA3 nonpaged-prefill shim (qwen35.rs:132), which must be re-checked
separately — not settled here.

## Verdict — KILL the scheduler, take the free win

- **KILL** the batched-verify substrate + Algorithm 1 as motivated: marginal
  gain over free plain-batched is +24% at 18% accept — not worth ≥1 week +
  frozen-file risk.
- **Free win, zero code:** OPD *concurrent* rollout (best-of-N groups) should run
  `--spec-type none` (plain-batched) — reclaims the 2.37× today.
- **DSpark stays the win only for genuinely serial c=1 decode** (single-sample
  rollout, latency-bound interactive) — its licensed 2× holds there.
- **Correct next lever is the DRAFTER, not the scheduler:** batched spec only
  becomes worth building after the DFlash drafter's acceptance is lifted
  18%→>50% (P3: train our own DSpark heads on rollout dumps). Order was inverted;
  train the head first, revisit batching after.

## Rule

- **Decode-phase tok/s, never request-wall tok/s, for a spec-decode A/B.**
  gen÷total_wall folds prefill into the number; at gen=256/ctx=2048 it diluted
  decode 3× and produced a self-contradicting "40 tok/s" that a glance at our
  own licensed 87-175 band should have caught. ALWAYS gate a new spec harness
  with a C=1 sanity check against the known decode band before trusting a sweep.
- A "does not batch" measurement on a weak drafter conflates two things: the
  *architecture* (per-row vs batched verify) and the *drafter quality* (accept
  rate). Separate them — a batched-verify substrate is worthless while accept is
  10-17% because plain-batched already captures the concurrency win for free.
- Decompose-then-spike before a multi-file hot-path build: this ≥1-week,
  frozen-file-blocked item was killed by one afternoon of measurement on the
  existing path. The prize must be quantified against the *free alternative*
  (plain-batched), not against B=1 spec.
