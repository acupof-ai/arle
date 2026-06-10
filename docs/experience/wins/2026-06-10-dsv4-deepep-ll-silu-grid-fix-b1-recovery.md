# deepep_ll −55% B=1 root cause was OUR masked-silu grid, not NVSHMEM — expected_m fix recovers +93%

**Date:** 2026-06-10. **Backend:** CUDA DSv4-Flash FP8 TP=8/EP=8, 8×H20, CUDA 12.9.
**Commit:** `b5f00399`. **Commissioned by:** ckl — "找出为什么性能会差这么多".

## Goal

Quantify where deepep_ll's −55% B=1 regression (17.97 vs allreduce 39.88 tok/s,
[error entry](../errors/2026-06-10-dsv4-deepep-ll-b1-regression-no-batch-lane.md))
actually goes. That entry's *hypothesis* was "NVSHMEM LL round-trip with 7 ranks
idle — pure overhead by construction". Per §0 that was never licensed by
measurement.

## Hypothesis → Measurement

nsys on the serve (8-rank process tree, `-t cuda,nvtx --delay=170 --duration=60`),
B=1, 2×128-token greedy requests per arm, **same binary, env-flip
`ARLE_DSV4_MOE_BACKEND`** (`nsys_b1_{deepep,allreduce}.nsys-rep`, build dir).

### deepep arm — kernel time (window = 256 tokens × 8 ranks × 43 MoE layers)

| kernel | % GPU | avg/call | calls | note |
|---|---|---|---|---|
| `dsv4_deepgemm_silu_mul_masked_quant_kernel` | **52.9%** | **631.5 µs** | 88 064 | min≈max≈631 µs → shape-static, mask-oblivious |
| `dsv4_mhc_params_kernel` | 5.9% | 35.4 µs | 176 128 | same both arms |
| `ncclAllReduce` (RING_LL) | 5.6% | med 25.0 µs | 176 128 | |
| `ncclAllGather` (RING_LL) | 5.0% | med 35.2 µs | 88 032 | |
| `internode_ll::dispatch` | 4.4% | 52.7 µs | 88 064 | NVSHMEM LL |
| `internode_ll::combine` | 3.1% | 36.8 µs | 88 064 | NVSHMEM LL |
| masked grouped DeepGEMM w13/w2 | ~3% | 21.9 / 13.9 µs | 88 064 ea | masks handled internally, cheap |

**631 µs × 43 layers = 27.1 ms/token of empty-block drain** — the grid covered
the full padded band `[E_local=48, m_padded=2048(=8×256), k_blocks=16]` ≈ 1.57 M
blocks/call, nearly all early-exiting on `masked_m` (valid rows at B=1: ≤ 8 of
98 304 expert-rows). That alone is the bulk of the ~30 ms/token wall gap.

### allreduce arm (control capture)

| kernel | % GPU | med/call | calls/layer·token |
|---|---|---|---|
| `ncclAllReduce` | 22.6% | 51.9 µs | 2 |
| `ncclAllGather` | 18.5% | 100.5 µs | 1 |

Per-token median comm: allreduce arm ≈ 8.8 ms vs deepep arm ≈ 3.7 ms (+2.3 ms
LL dispatch+combine + ~1 ms extras) — **the LL transport itself is per-token
CHEAPER than the ring collectives it replaces**; the hypothesized NVSHMEM
round-trip cost is real but secondary. (NCCL kernel durations absorb peer-wait;
medians used.)

## Fix (`b5f00399`)

`dsv4_deepgemm_silu_mul_masked_quant_cuda` takes `expected_m` — host-known
per-expert valid-row bound = the step's global token count (DeepEP LL packs
recv rows from row 0; per-expert recv ≤ global tokens). Grid token-dim becomes
`min(expected_m, m_padded)` (B=1: 2048 → 1, grid 1.57 M → 768 blocks); memory
strides keep the padded band; device-side `__trap()` if any `masked_m` exceeds
the bound (loud, never silent). Call site passes `tokens.len()` (full-step
count, pre-slice).

## Results (same harness `dsv4_ab_bench.py`, same binary, serial arms)

| arm | B=1 p50 tok/s | seq agg tok/s | Δ vs pre-fix | Δ vs same-day allreduce control |
|---|---|---|---|---|
| deepep_ll pre-fix (`d8b675e1`) | 17.97 | — | — | −54.9% (vs that day's 39.88) |
| **deepep_ll + silu fix** | **34.62** | 33.90 | **+92.6%** | **−11.2%** |
| allreduce control (same binary) | 38.98 | 39.52 | — | (−2.3% vs the 39.88 entry baseline — day-noise; silu kernel not on this path) |

Output coherent greedy (" Paris.\n…" stable across 5 reqs, byte-stable rates
34.5–34.7).

## Problems / residual

- Residual −11% vs allreduce at B=1 ≈ the genuine LL overhead (dispatch
  52.7 µs + combine 36.8 µs + extra hop structure per layer ≈ +2–4 ms/token,
  partially offset by cheaper collectives). This is the part that batching
  amortizes — the original "deepep wins at the batched lane" premise now has a
  fair starting point instead of a 2.2× handicap.
- deepep_ll stays opt-in: the batched serving lane is still blocked by the
  cross-rank lockstep deadlock
  ([error entry](../errors/2026-06-10-dsv4-c2-layer2-lockstep-admission-deadlock.md),
  [plan](../../plans/2026-06-10-dsv4-multirank-lockstep-admission.md)).

## Learnings

- **"By construction slower" is a hypothesis, not a measurement.** The error
  entry's NVSHMEM-idle-ranks story survived one full A/B cycle unchallenged;
  one nsys kernel-sum table killed it in minutes: the regression was a
  mask-oblivious launch grid in OUR kernel (`0b70190b`), and the LL transport
  is actually cheaper per token than ring collectives at B=1.
- **min==max==avg kernel duration is the mask-oblivious signature.** A masked
  kernel whose duration is shape-static does worst-case work regardless of
  occupancy — check `StdDev` in `cuda_gpu_kern_sum` before blaming comm.
- Empty-block drain is not free: 1.57 M early-exit blocks ≈ 631 µs on H20
  (~0.4 ns/block). Grid-by-bound beats in-kernel early-exit when waste is 2000×.
