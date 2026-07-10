# Qwen3.5 Medusa Snapshot Ring Graph License

## Context

This revisits
[`2026-05-10-medusa-qwen35-snapshot-ring-step0-killed.md`](../errors/2026-05-10-medusa-qwen35-snapshot-ring-step0-killed.md).
The earlier Step 0 measured the preallocated Qwen3.5 recurrent snapshot ring
at ~45 ms for `K+1=6`, above the `>5 ms` kill threshold. The revisit tested
Option 1.5c: capture the snapshot and restore copy sequences into CUDA Graphs,
then time graph replay only.

## What Worked

- Added `RecurrentSnapshotRing` in `infer/src/model/qwen35/recurrent_state.rs`
  as a Step 0-only prototype:
  - preallocates `Vec<RecurrentSnapshot>` slots,
  - captures one snapshot graph per slot,
  - captures one restore graph per slot,
  - replays `6` snapshot graphs plus one restore graph in the measured window.
- Kept the existing memcpy-only ignored bench for A/B comparison.
- The graph replay steady-state total is below the `<2 ms` license threshold.

## Results

Command:

```bash
CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 TORCH_CUDA_ARCH_LIST=8.9 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
cargo test --release -p infer --features cuda --lib \
  qwen35_recurrent_snapshot_ring_bench_k6_graph -- --ignored --nocapture --test-threads=1
```

Graph-only run after capture:

| Run shape | Samples (ms) | Mean | Sigma | Decision |
|---|---:|---:|---:|---|
| Cargo test graph-only | 1.472, 1.389, 1.440 | 1.434 ms | 0.034 ms | LICENSE |

Direct test-binary replay showed one cold-process outlier followed by stable
steady-state:

| Run | Mean total | Sigma | Notes |
|---:|---:|---:|---|
| 1 | 42.251 ms | 0.359 ms | cold process / first graph path |
| 2 | 1.408 ms | 0.012 ms | steady-state |
| 3 | 1.406 ms | 0.013 ms | steady-state |
| 4 | 1.409 ms | 0.015 ms | steady-state |
| 5 | 1.407 ms | 0.010 ms | steady-state |

Memcpy-only A/B from the same test binary:

| Path | Cold total | Warm total | Notes |
|---|---:|---:|---|
| direct `memcpy_dtod` loop | 42.299 ms | 1.559 ms | existing bench still passes |
| CUDA Graph replay | 42.343 ms | 1.401 ms | graph is ~10% faster once warm |

## License

LICENSE for proceeding to the §3 verifier-integration design, with a constraint:
snapshot graph capture and first replay must be done outside the verifier hot
path. The steady-state replay path is ~1.4 ms at `K+1=6`, below the `<2 ms`
threshold. The 42 ms cold path is not acceptable inside the first verifier step.

## Rule

The earlier KILL was correct for cold one-shot timing but wrong as a steady-state
verdict. For CUDA Graph micro-benches, separate cold capture/first-launch cost
from warmed replay cost, and report both. Graph replay is licensed here, but the
observed win is ~10% over warmed direct D2D copies, not the predicted ~95x.
