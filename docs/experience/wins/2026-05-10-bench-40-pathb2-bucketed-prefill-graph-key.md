# #40 Path B.2 Bucketed Prefill Graph Key — Functional Gate

## Goal

- Convert #37 Path B's production 4k cache-miss finding into a finite graph key space by bucketing allocation-size dimensions.
- Scope is functional only. The 4k/c=4 throughput license remains the post-commit `./scripts/post_p24_commit_pipeline.sh full` run.

## Hypothesis

- Path B v1 missed in production because `page_indices_len` and `prefix_token_rows_len` encoded per-request allocation sizes into the CUDA graph key.
- Rounding those dimensions into fixed buckets should reduce 4k production capture keys from one per request to a small reusable set while preserving graph-safe launch capacities.

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

Smoke evidence now shows page-count bucketing:

```text
Qwen3 prefill graph capture key: tokens=4 batch=1 pages=64 prefix_rows=0 marlin_scratch=false
Qwen3 prefill graph capture key: tokens=3 batch=1 pages=64 prefix_rows=0 marlin_scratch=false
Qwen3 prefill graph capture key: tokens=8 batch=1 pages=64 prefix_rows=0 marlin_scratch=false
Qwen3 prefill graph capture key: tokens=1 batch=1 pages=64 prefix_rows=0 marlin_scratch=false
```

## Implementation

- Rounded `page_indices_len` to 64-entry buckets and `prefix_token_rows_len` to 128-row buckets in `Qwen3PrefillGraphKey`.
- Padded graph upload buffers with zeros to the bucket capacity so replay never consumes stale rows from a prior larger request.
- Used bucket capacity for captured TileLang `total_pages` and prefix-refill `prefix_token_count`, not the first request's exact lengths.
- Left scalar device metadata refresh, `kv_last_page_len` refresh, and the 8-key LRU from #37 Path B unchanged.

## Problems

- This is not the throughput license. #40 still needs the matched-control 4k/c=4 graph-off vs graph-on N=3 bench to determine whether bucketing reaches the expected reuse rate.
- Full `greedy_consistency` still includes the pre-existing W4A8-vs-BF16 token-diff accuracy gate tracked outside this change.

## Learnings

- Bucketed graph keys must also bucket the captured scalar launch parameters. A cache hit with stale scalar `total_pages` or `prefix_token_count` is still a semantic miss.
- Padding graph input buffers is simpler and safer than relying on stale tail contents when a bucketed replay processes capacity rather than exact request length.

## Delta vs Baseline

- **Baseline error:** `docs/experience/errors/2026-05-10-37-pathB-bench-tier4-kill-cache-miss-at-4k.md`

| metric | Path B v1 | Path B.2 |
|---|---:|---:|
| `page_indices_len` key | exact | 64-entry bucket |
| `prefix_token_rows_len` key | exact | 128-row bucket |
| `total_pages` graph scalar | exact first capture | bucket capacity |
| `prefix_token_count` graph scalar | exact first capture | bucket capacity |
| 4k/c=4 TTFT | Tier 4 KILL | pending post-commit bench |

## Artefacts

- Throughput template: `docs/experience/wins/TEMPLATE-2026-05-10-bench-37-w4hybrid-prefill-graph-throughput.md`
- Pipeline runner: `scripts/post_p24_commit_pipeline.sh`
