# DSv4 prefill projections → DeepGEMM: clean per-stage A/B (wq_b/wo −94%; prefill_ms was noise)

## Context

After wiring prefill wq_b + wo to DeepGEMM (`ARLE_DSV4_PREFILL_PROJ_DEEPGEMM`), the
first A/B used `prefill_ms` — which is **load-tail + DeepGEMM-JIT noise** (the same
37-tok needle ranged 1814–7996 ms across runs). Re-ran with the rank-0
`ARLE_DSV4_LINEAR_PROFILE` per-stage kernel time (excludes load/JIT), the correct
evidence per `feedback_measured_floor_is_not_physical_floor` / the NVTX-sync framing
traps.

## What worked — clean per-stage time (8×H20 TP=8, M=1024 prefill, same binary)

| stage | DG=0 scalar fp8_gemv | DG=1 DeepGEMM | Δ |
|---|---:|---:|---:|
| wq_b | 138.3 ms | 8.4 ms | **−94% (16.4×)** |
| wo_a | 132.0 ms | 8.6 ms | **−93%** |
| wo_b | 137.7 ms | 6.3 ms | **−95%** |
| indexer_wq_b | 135.1 ms | 135.1 ms | unchanged (still scalar) |

Correctness: the needle output is byte-identical DG=0 vs DG=1 (`[223,30793,929,16,
19018,436,7681,16]`). The scalar `dsv4_fp8_gemv_batch` is a decode (M=1) GEMV; at
M=1024 it is ~16× slower than tensor-core DeepGEMM — this IS the P/D-nsys "62% of
prefill projection" bottleneck.

## Rule

- **Never license a prefill perf change on `prefill_ms`** — it's dominated by model
  load-tail + DeepGEMM JIT-compile-on-first-use, which swing multiple seconds
  run-to-run. Use the rank-0 `ARLE_DSV4_LINEAR_PROFILE` per-stage kernel time (synced,
  so absolute is inflated but the OFF-vs-ON delta of the same stage is the true kernel
  difference). Same class as the stage-profile per-stage-sync trap.
- **indexer_wq_b (135 ms, now 67% of post-wq_b/wo linear) is the #1 remaining
  projection** — wire it (+ weights_proj) through DeepGEMM next (needs an indexer
  DeepGEMM cache in the loader + the scratch threaded into `csa_select`).
