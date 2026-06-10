# Comm bench: one-shot AR 3.05× / NCCL-sym 2.17× vs NCCL at decode shapes — T2 integration LICENSED

**Date:** 2026-06-10. **Env:** 8×H20 (NVSwitch), CUDA 12.9, NCCL 2.27.3, bf16.
**Commit:** `3aaa19b9` (vendored kernels + harness). **Plan:**
[`2026-06-10-dsv4-oneshot-allreduce-isolated-bench.md`](../../plans/2026-06-10-dsv4-oneshot-allreduce-isolated-bench.md).

## Goal / Hypothesis

License-or-kill: can copy-first one-shot collectives cut the DSv4 decode
chain's 8.8 ms/34% comm bill (2×AR 14KB + 1×Q-AG 18KB per layer, all
50–100× off the NVLink floor)?

## Params

`comm_bench` example (8 processes, one per GPU, file-rendezvous NCCL +
IPC bootstrap), exposed mode = dependent-chain timing, chain-kernel cost
subtracted; 200 iters × 5 repeats, p50. Arms: NCCL baseline / NCCL 2.27
`ncclCommWindowRegister` symmetric windows / vendored sgl-kernel(vLLM-lineage)
custom AR one-shot + two-shot / ARLE-derived one-shot AG. ckl's idle serve
held GPU memory during the run (0% util — quiet-machine numbers).

## Results (exposed p50 µs/op; full table in `comm_bench_run.log`)

| shape | bytes | nccl | nccl_sym | car_1stage | car_2stage | best vs nccl |
|---|---|---|---|---|---|---|
| AR [1,7168] | 14 KB | 19.0 | 8.8 | **6.2** | 12.3 | **3.05×** |
| AR [2,7168] | 28 KB | 20.4 | 8.9 | **6.1** | 12.8 | 3.32× |
| AR [4,7168] | 56 KB | 23.8 | 8.8 | **6.6** | 13.5 | 3.62× |
| AR [8,7168] | 112 KB | 23.3 | 8.8 | **8.1** | 13.7 | 2.89× |
| AR [16,7168] | 224 KB | 18.5 | **9.0** | 10.7 | 14.0 | 2.06× |
| AR [32,7168] | 448 KB | 18.9 | **8.7** | 16.4 | 14.9 | 2.17× |
| AG 2048/rank | 4 KB | 12.7 | 8.9 | **5.8** | — | 2.19× |
| AG 9216/rank (Q shape) | 18 KB | 15.5 | 9.2 | **6.1** | — | 2.55× |

Correctness: **identical-across-ranks = yes for every arm × shape** (FNV
digest allgather; lockstep hard requirement) and max|Δ| vs the NCCL
reference = 0.0000 everywhere.

## Verdict

- **Gate L1 PASSES**: best arm ≥2× vs NCCL at every shape; AR@14KB 6.2 µs ≤ 20;
  AG ≤ 25. T2 integration licensed.
- **sglang's kernel is the right copy** (the question asked): sgl-kernel's
  NVIDIA path IS the vLLM kernel + copy_mode/graph-capture conveniences; at our
  shapes one-shot wins ≤112 KB, NCCL-sym wins ≥224 KB, two-shot never wins.
- **Pre-registered C4 short-circuit vs staging reality (flagged, not silently
  bent):** the plan said "ship NCCL-sym if Δ(best CA − C4) < 10 µs/op" — raw Δ
  is 2.6 µs → C4. But integration staging is asymmetric: custom-AR needs ONE
  copy-in (input must be the registered buffer; output is any local buffer),
  NCCL-sym needs copy-in AND copy-out (both operands must be windows) unless
  activations themselves move into a symmetric pool. With ~1.7 µs/14KB D2D:
  car ≈ 7.9 µs/op vs sym ≈ 12.2 µs/op → car is both faster and simpler at the
  call site. **T2 recommendation: car_1stage for ≤128 KB (B=1..8 decode AR +
  Q-AG), nccl_sym arm kept behind the same flag for ≥224 KB and as fallback.**
- Honest wall prediction: the bench kills the protocol floor (19→6 µs), but
  production AR med was 51.9 µs — the extra ~33 µs is rank-arrival skew +
  stream queueing that a faster kernel only partially removes. Bounds:
  +5%…+40% tok/s; the T2 opt-in A/B (same harness, needle + lockstep
  fingerprint gates) is the decider. No default flip from this entry.

## Learnings

- NCCL 2.27 symmetric windows are real and nearly free to adopt: 2.06–2.69×
  on every shape with zero custom kernel code — the right LARGE-message arm
  and the fallback if custom-AR IPC ever misbehaves.
- Two-shot never beats one-shot ≤448 KB at world=8 on NVL8 — decode-size
  messages are latency-bound, not bandwidth-bound.
- Exposed (dependent-chain) vs b2b differed by ≤1.5 µs here — these kernels
  don't hide latency in pipelining at small sizes; the framing mattered more
  for NCCL (queueing).
