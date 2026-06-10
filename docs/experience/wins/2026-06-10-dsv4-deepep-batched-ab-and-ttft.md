# deepep_ll batched lane runs: −5~7% vs allreduce @ c≤8 intranode; TTFT 30/178/548 ms; LL prefill cap found

**Date:** 2026-06-10. **Backend:** CUDA DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Commit:** `3504afb2` (batched decode over the DeepEP transport). Same binary
both arms, env-flip `ARLE_DSV4_MOE_BACKEND` + `INFER_DSV4_BATCHED_DECODE=1`,
serial. Harness: `scripts/dsv4_c_sweep.py` (96 tok/req), `ttft_probe.py`
(max_tokens=1 wall, warmup dropped).

## Goal

The A/B the deepep_ll plan always needed: deepep-vs-allreduce at the BATCHED
lane, now that the lane exists (`23b69249`) and the −55% silu artifact is gone
(`b5f00399`).

## Results — batched c-sweep (aggregate tok/s, burst / 1 s-stagger)

| c | allreduce batched | deepep_ll batched | Δ |
|---|---|---|---|
| 2 | 51.88 / 44.14 | 48.48 / 42.26 | −6.6% / −4.3% |
| 4 | 50.56 / 55.02 | 48.56 / 51.00 | −4.0% / −7.3% |
| 8 | 58.09 / 58.70 | 54.18 / 55.83 | −6.7% / −4.9% |

- deepep arm: all lanes complete, server alive, **24/24 expected-substring
  hits** (cleanest correctness of any sweep today).
- Verdict at this shape: **deepep_ll loses −5~7% intranode at c≤8** — the
  cumulative arc is 2.2× handicap (silu grid) → −11% (B=1, fixed) → −5~7%
  (batched). On a single NVLink node the ring all-reduce is simply hard to
  beat; DeepEP LL's structural win is multi-node EP (where the all-reduce
  would cross IB anyway) and larger N. **allreduce stays the intranode
  default**; deepep_ll is now a fair, runnable opt-in rather than a
  2.2×-handicapped one.

## TTFT (allreduce-batched arm, B=1, p50 of 3 post-warmup, max_tokens=1)

| prompt tokens | TTFT p50 |
|---|---|
| 5 | **30 ms** |
| 645 | **178 ms** |
| 2565 | **548 ms** |

Composition: ~25 ms floor (one decode step + HTTP) + prefill at ~0.2 ms/token
(≈ 4.8–5.2k tok/s single-request prefill at these sizes). Cold first request
after load: 746 ms (JIT warm).

## Found: deepep_ll prefill cap (pre-existing, now loud)

The TTFT probe's 2565-token prompt KILLED the deepep arm: prefill chunks also
route through the LL path, and `owned tokens 320 exceed
num_max_dispatch_tokens_per_rank 256` — the LL buffer caps prefill at
256/rank × 8 = **2048 tokens per chunk**. Never hit before because nothing
ever prefilled >2048 through deepep_ll. Notable: **all 8 ranks failed
symmetrically at the same tick (#1192) with the same ensure** — lockstep
turned what would have been a desync hang into one attributable, synchronized
error. Fix direction (SGLang posture): transport per shape — LL for decode +
small prefill, grouped+all-reduce for large prefill chunks. Until then
deepep_ll serves prompts ≤2048 tokens only.

## Learnings

- A transport verdict needs the lane AND clean kernels: every prior deepep
  number was confounded (no lane, then a mask-oblivious grid). The honest
  intranode answer is a single-digit loss, not a 2.2× one.
- max_tokens=1 probes double as prefill-path fuzzers — one TTFT sweep found a
  cap three days of decode benches never touched.
- Lockstep failure semantics are a feature: symmetric ensure-death beats
  asymmetric NCCL spin for every future executor-side bound.

## Refs

- Lane: [`wins/2026-06-10-dsv4-lockstep-admission-c-sweep-lane-exists.md`](2026-06-10-dsv4-lockstep-admission-c-sweep-lane-exists.md)
- Silu fix: [`wins/2026-06-10-dsv4-deepep-ll-silu-grid-fix-b1-recovery.md`](2026-06-10-dsv4-deepep-ll-silu-grid-fix-b1-recovery.md)
- Pod logs: `c_sweep_ab_{allreduce,deepep}.log`, `ttft_allreduce.log` (build dir)
