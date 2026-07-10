# DSpark concurrency scheduler (P3 Algorithm 1) — KILLED by a one-afternoon spike; OPD concurrent rollout should use plain-batched

## Context

The "DSpark doesn't batch" claim (`5bc1b3e8f`) was corrected against the paper:
DSpark's Hardware-Aware Prefix Scheduler (Algorithm 1) is *designed* to hold
throughput at concurrency. Before building it (decompose: ≥1-week batched
ragged-verify substrate, terminates in frozen attention files), one spike
measured whether there's a prize. 8×H20 GPU1, Qwen3.6-27B-FP8 + DFlash draft,
ctx=2048 gen=256 temp=0.7, per-row dspark vs plain, C∈{1,2,4,8}.

## What the spike found

| C | dspark agg tok/s | plain agg tok/s | dspark util | plain util |
|---|---|---|---|---|
| 1 | 40.2 | 32.4 | 82% | 86% |
| 2 | 39.8 | 40.8 | 84% | 68% |
| 4 | 40.1 | 53.6 | 88% | 66% |
| 8 | 39.5 | **93.6** | 88% | 56% |

- **dspark aggregate is dead flat (~40 tok/s); plain scales 2.9× to C=8.** Per-row
  spec verify saturates the weight-read path at C=1 and throws away the batch
  dimension — concurrency just divides a fixed pie (p50 40→22→10→5). Plain packs
  all rows into one forward; util *drops* 86→56% as batching amortizes weight
  reads, converting concurrency into throughput.
- **The reframe that matters: OPD rollout is NOT B=1.** best-of-N /
  samples-per-prompt submit the sample group concurrently to the continuous-
  batching engine (`infer_student.rs:195` — "decode all N concurrently, batching
  amortizes the weight reads"). Realistic OPD concurrency ≈ C=8, exactly where
  plain-batched beats dspark **2.37×** (93.6 vs 39.5).
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

- A "does not batch" measurement on a weak drafter conflates two things: the
  *architecture* (per-row vs batched verify) and the *drafter quality* (accept
  rate). Separate them — a batched-verify substrate is worthless while accept is
  18% because plain-batched already captures the concurrency win for free.
- Decompose-then-spike before a multi-file hot-path build: this ≥1-week,
  frozen-file-blocked item was killed by one afternoon of measurement on the
  existing path. The prize must be quantified against the *free alternative*
  (plain-batched), not against B=1 spec.
