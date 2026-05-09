# #37 Path B Device Metadata Prefill Graph — Functional Gate

## Goal

- Make Qwen3 paged prefill graph replay reuse graph-stable pointers while still refreshing per-request metadata.
- Scope is functional only. Throughput license remains owned by `./scripts/post_p24_commit_pipeline.sh` / `docs/experience/wins/TEMPLATE-2026-05-10-bench-37-w4hybrid-prefill-graph-throughput.md`.

## Hypothesis

- Path A missed because the graph key included per-request fields such as `start_pos` and page layout details, causing capture churn.
- Moving request-varying scalar metadata into device buffers and refreshing contents before replay should preserve correctness while allowing repeated shapes to hit the cache.

## Command

```bash
cargo fmt --all
git diff --check

env CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
  INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  TORCH_CUDA_ARCH_LIST=8.9 \
  cargo check --release -p infer --features cuda

env CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
  INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  TORCH_CUDA_ARCH_LIST=8.9 \
  cargo clippy --release -p infer --features cuda --lib -- -D warnings

env INFER_PREFILL_GRAPH=1 INFER_HYBRID_W4A8_PREFILL=1 \
  CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
  INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  TORCH_CUDA_ARCH_LIST=8.9 \
  cargo test --release -p infer --features cuda --test e2e \
    test_e2e_generation -- --test-threads=1

env CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
  INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  TORCH_CUDA_ARCH_LIST=8.9 \
  cargo test --release -p infer --features cuda --test greedy_consistency \
    test_greedy_solo_vs_concurrent -- --test-threads=1
```

## Environment

- **Backend:** CUDA
- **Model:** `infer/models/Qwen3-4B`
- **Hardware:** RTX 4070 Ti SUPER 16 GiB, sm_89
- **CUDA:** `/opt/cuda`, CUDA 13.2 toolchain, `NVCC_CCBIN=/usr/bin/g++-14`
- **Feature set:** `-p infer --features cuda --release`
- **Non-default flags / env vars:** `INFER_PREFILL_GRAPH=1`, `INFER_HYBRID_W4A8_PREFILL=1` for graph-on smoke

## Results

| Gate | Result |
|---|---|
| `cargo fmt --all` | pass |
| `git diff --check` | pass |
| `cargo check --release -p infer --features cuda` | pass |
| `cargo clippy --release -p infer --features cuda --lib -- -D warnings` | pass |
| `e2e::test_e2e_generation` with `INFER_PREFILL_GRAPH=1` | pass |
| `greedy_consistency::test_greedy_solo_vs_concurrent` | pass |

Graph evidence from e2e smoke:

```text
Qwen3 prefill graph capture key: tokens=4 batch=1 pages=1 prefix_rows=0 marlin_scratch=false
Qwen3 prefill graph capture key: tokens=3 batch=1 pages=1 prefix_rows=0 marlin_scratch=false
Qwen3 prefill graph capture key: tokens=8 batch=1 pages=1 prefix_rows=0 marlin_scratch=false
Qwen3 prefill graph capture key: tokens=1 batch=1 pages=1 prefix_rows=0 marlin_scratch=false
```

The repeated e2e prompts reused the cached keys instead of recapturing every request.

## Implementation

- Changed the HD128 paged prefill prep CUDA ABI so `start_pos` and per-row page-table offset are read from device pointers.
- Added graph-lifetime metadata buffers for start positions, page-table offsets, and sequence lengths, refreshed before capture/replay.
- Refreshed the captured `PagedPrefillForward` metadata buffers (`qo_indptr`, `kv_indptr`, `kv_last_page_len`) before replay because `kv_last_page_len` depends on the updated start position.
- Narrowed the prefill graph key by removing request-varying `start_positions` and `num_pages` while keeping `seq_lens`, `total_tokens`, `page_indices_len`, `prefix_token_rows_len`, `batch_size`, and `page_size` as launch-topology guards.
- Replaced the one-entry graph resource with an eight-key LRU cache so alternating stable shapes do not evict each other.

## Problems

- Full `greedy_consistency` still includes the pre-existing W4A8-vs-BF16 token-diff accuracy gate. That is tracked separately in `docs/experience/errors/2026-05-08-w4a8-quantize-broken-100pct-token-diff.md`; the B=1 vs B=3 greedy consistency gate passes here.
- This entry is not the throughput license. #37 Phase 2 still needs matched-control 4k/c=4 graph-off vs graph-on N=3 bench.

## Learnings

- CUDA graph keys should describe allocation sizes and launch topology, not request-varying scalar metadata. Scalar request state belongs in device buffers whose contents refresh before replay.
- Removing `seq_lens` from the key would be unsafe without a masked/capacity launch rewrite, because sequence lengths still influence the captured kernel launch geometry.
- Single-entry graph caches can hide successful replay in alternating-shape tests; even a small LRU cache is enough to separate key-churn bugs from legitimate shape diversity.

## Delta vs Baseline

- **Baseline:** `docs/experience/wins/2026-05-10-bench-p24-w4a8-prefill-graph-hoist.md`

| metric | baseline | now | delta |
|---|---:|---:|---:|
| Qwen3 prefill graph key includes `start_positions` | yes | no | request scalar removed |
| Qwen3 prefill graph key includes `num_pages` | yes | no | page offset device-refreshed |
| Qwen3 prefill graph cache size | 1 key | 8 keys | alternating shape reuse unblocked |
| 4k/c=4 TTFT | deferred to #37 | pending Phase 2 | n/a |

## Artefacts

- Throughput template: `docs/experience/wins/TEMPLATE-2026-05-10-bench-37-w4hybrid-prefill-graph-throughput.md`
- Pipeline runner: `scripts/post_p24_commit_pipeline.sh`
