# Qwen3.5 Medusa Snapshot Ring Step 0 Killed

## Context

`docs/plans/M_medusa-phase1b-qwen35-v2-snapshot-ring-redesign.md`
proposed a Step 0 prototype for Medusa verifier rollback on Qwen3.5
hybrid recurrent state. The test measures K+1 recurrent snapshots at
K=5 (`k_plus_1=6`) on the local Qwen3.5-4B config before committing to
the full snapshot-ring substrate. Per review, the final benchmark
preallocates the ring slots before timing, so the measured window covers
only D2D state copies into the ring plus one restore.

## Result

Command:

```bash
CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 TORCH_CUDA_ARCH_LIST=8.9 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
cargo test --release -p infer --features cuda --lib \
  qwen35_recurrent_snapshot_ring_bench_k6 -- --ignored --nocapture --test-threads=1
```

Three runs:

| Run | Total K+1 snapshots + restore | Per snapshot | Estimated ring memory |
|---:|---:|---:|---:|
| 1 | 45.556 ms | 7.593 ms | 294.8 MiB |
| 2 | 44.922 ms | 7.487 ms | 294.8 MiB |
| 3 | 45.648 ms | 7.608 ms | 294.8 MiB |

Mean total: 45.375 ms. Sample sigma: ~0.40 ms (~0.9%).

## Root Cause

The prototype preallocates six recurrent snapshot slots, then measures six
state-copy overwrites plus one restore. The result is still far above the
`<2 ms` license target and the `>5 ms` kill threshold, so the bottleneck is
not ring allocation. The D2D copy volume for Qwen3.5 recurrent state is itself
too expensive for K=5 verifier rollback.

## Fix

Do not proceed with the full §3 snapshot-ring substrate as written.
Pivot to the brief's alternatives:

- Option 2: shadow recurrent state, if verifier can avoid K+1 full-state
  clones per step.
- Option 3: defer Qwen3.5 Medusa until a lower-copy rollback mechanism exists.

## Rule

Prototype rollback mechanisms with the same copy/allocation shape that the
design intends to use. Formula estimates around GPU memcpy bandwidth are not
enough when the real path copies many independent recurrent-state slabs.
