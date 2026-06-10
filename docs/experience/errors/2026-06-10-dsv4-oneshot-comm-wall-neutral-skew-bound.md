# One-shot comm is wall-neutral in production (9th wall-neutral lever) — DSv4 decode is SKEW-bound, not protocol-bound

**Date:** 2026-06-10. **Backend:** CUDA DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Commits:** `0c120e84` (integration) + `a0fd3a12` (vote fix) + default flip back to NCCL.

## Context

The comm bench licensed a 3.05× per-op win (car one-shot 6.2 µs vs NCCL 19 µs
@14 KB, [wins entry](../wins/2026-06-10-dsv4-comm-bench-oneshot-licensed.md)).
T2 wired it into `TpRuntime` (default-on, collective auto-degrade, 512 KB
registered scratch, staging copy-in) and ran the same-harness matched A/B.

## Evidence (same binary, env-flip `--comm-backend`, serial arms)

- **Path verified** (per the path-probe rule): nsys shows
  `sglang::cross_device_reduce_1stage` ×176128 (2/layer·tok·rank) and
  `cross_device_allgather_1stage` ×88032 on the decode chain; NCCL kernels gone.
- **Per-op improvement REAL in production**: AR med 51.9→25.8 µs, Q-AG med
  100.5→51.1 µs (kernel durations, production trace).
- **Wall UNMOVED**:

| metric | nccl arm | oneshot arm | Δ |
|---|---|---|---|
| B=1 p50 tok/s | 38.49 | 38.72 | +0.6% (noise) |
| c=2 burst/stagger agg | 53.60 / 45.36 | 53.09 / 46.70 | −1% / +3% |
| c=4 burst/stagger | 52.53 / 53.83 | 53.39 / 55.06 | +1.6% / +2.3% |
| c=8 burst/stagger | 59.04 / 60.86 | 59.58 / 59.85 | +0.9% / −1.7% |
| TTFT 5/645/2565 tok | 30/178/548 ms (prior) | 29/178/549 ms | flat |

Ranges overlap everywhere → wash per the matched-A/B rule. (An earlier +2.5%
vs the MORNING baseline was day-drift — cross-day comparisons lied again.)

## Root cause — the complete account

The production AR med (51.9 µs) was never protocol: the quiet-machine bench
measured NCCL's own exec at ~19 µs (b2b), so ~33 µs of the production med is
**rank-arrival skew** absorbed in the collective. One-shot kernels absorb the
same skew in their spin-barrier (their production med: 25.8 µs vs 6.2 µs
quiet). Net protocol saving ≈ 13 µs/op × 3 ops/layer, minus ~8 µs/op staging
(copy-in + extra launch — activations are transient pool allocs, so staging is
structural without buffer registration/graphs) ⇒ ~0.5 ms/step theoretical,
inside the run-to-run noise floor. **The wall is Σ_layers max-over-ranks
(per-rank leg jitter) — a skew-bound chain. Faster collectives cannot shorten
it; only fewer serial steps (MTP) or shorter per-rank legs (fusion) can.**

## Disposition

- `--comm-backend` default flipped back to **nccl** (wash doesn't ship as
  default). The one-shot stack (vendored kernels, collective-safe boot with
  ok-votes + self-test, bench) stays as a maintained opt-in: it is the
  foundation for (a) multi-node EP, where ring AR costs multiply, and (b) the
  T3 fused AR+rmsnorm (a DIFFERENT mechanism — fusion shortens the per-rank
  leg and removes a dependency boundary, attacking the skew itself).
- Bonus validated en route: the first live boot tripped a real bug
  (`all_gather_bytes` 4-byte alignment) and **all 8 ranks degraded loudly and
  collectively to NCCL with the serve healthy** — the auto-degrade design
  works under real failure, not just in theory.

## Rule

- **A licensed per-op win is still hypothesis at the wall.** 3× isolated, path
  verified on-chain, per-op med halved in production — and the wall did not
  move. License-or-kill must run THROUGH the wall A/B, always.
- **Decode B=1 verdict after 9 kills: the chain is skew-bound.** Remaining
  levers, in order: MTP (÷accept ≈ 1.85 — the only multiplier), fused
  AR+norm / per-rank leg shortening, multi-node where protocol costs rescale.
- Production collective med = exec + peer-wait. Always difference against the
  quiet-machine bench number before attributing it to protocol.

## Refs

- License bench: [`wins/2026-06-10-dsv4-comm-bench-oneshot-licensed.md`](../wins/2026-06-10-dsv4-comm-bench-oneshot-licensed.md)
- Plan (T2 closed): [`plans/2026-06-10-dsv4-oneshot-allreduce-isolated-bench.md`](../../plans/2026-06-10-dsv4-oneshot-allreduce-isolated-bench.md)
- Pod logs: `ab_{oneshot,ncclcomm}.log`, `c_sweep_{oneshot,ncclcomm}.log`, `nsys_b1_allreduce.nsys-rep` (evening capture)
