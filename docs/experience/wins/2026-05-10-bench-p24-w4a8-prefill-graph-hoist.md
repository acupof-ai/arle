# #24 W4A8 Prefill Graph Hoist — Phase 1 Functional Gate

## Goal

- Make Qwen3 paged prefill CUDA-graph-capable for W4 / W4-hybrid weights by hoisting Marlin prefill scratch out of per-call linear dispatch.
- Scope is functional only. Throughput license for multi-key graph reuse remains deferred to #37.

## Hypothesis

- W4-hybrid prefill graph capture is blocked by allocation/capture safety, not by scheduler policy.
- Preallocating Marlin scratch at prefill-context lifetime and marking W4 packed weights graph-safe should let `INFER_PREFILL_GRAPH=1` enter Qwen3 paged prefill graph capture without changing default behavior.

## Command

```bash
env CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
  INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  TORCH_CUDA_ARCH_LIST=8.9 \
  cargo check --release -p infer --features cuda

env CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
  INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  TORCH_CUDA_ARCH_LIST=8.9 \
  cargo clippy --release -p infer --features cuda -- -D warnings

env CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
  INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  TORCH_CUDA_ARCH_LIST=8.9 \
  cargo test --release -p infer --features cuda --test e2e -- --test-threads=1

env CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
  INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  TORCH_CUDA_ARCH_LIST=8.9 \
  cargo test --release -p infer --features cuda --test greedy_consistency \
  test_greedy_solo_vs_concurrent -- --test-threads=1 --nocapture

env CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
  INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  TORCH_CUDA_ARCH_LIST=8.9 \
  cargo test --release -p infer --features cuda --test greedy_consistency \
  test_greedy_w4a8_marlin_optional -- --test-threads=1 --nocapture
```

Functional smoke:

```bash
INFER_PREFILL_GRAPH=1 INFER_HYBRID_W4A8_PREFILL=1 RUST_LOG=info \
  CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 \
  ./target/release/infer \
    --model-path infer/models/Qwen3-4B-W4-hybrid-zpfix \
    --port 8000 --num-slots 4 --max-seq-len 5120

curl -sS --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-4B-W4-hybrid-zpfix","prompt":"Tell me a short story about a painter","max_tokens":8,"temperature":0}'
```

## Environment

- **Backend:** CUDA
- **Model:** `infer/models/Qwen3-4B-W4-hybrid-zpfix`
- **Hardware:** RTX 4070 Ti SUPER 16 GiB, sm_89
- **CUDA:** `/opt/cuda`, CUDA 13.2 toolchain, `NVCC_CCBIN=/usr/bin/g++-14`
- **Feature set:** `-p infer --features cuda --release`
- **Non-default flags / env vars:** `INFER_PREFILL_GRAPH=1`, `INFER_HYBRID_W4A8_PREFILL=1`
- **Server launch:** direct `target/release/infer`, port 8000

## Results

| Gate | Result |
|---|---|
| `cargo check --release -p infer --features cuda` | pass |
| `cargo clippy --release -p infer --features cuda -- -D warnings` | pass |
| `git diff --check` | pass |
| `cargo test --release -p infer --features cuda --test e2e -- --test-threads=1` | pass, 2/2 |
| `greedy_consistency::test_greedy_solo_vs_concurrent` | pass |
| `greedy_consistency::test_greedy_w4a8_marlin_optional` | pass |
| W4-hybrid prefill graph smoke | pass, HTTP 200 |

Smoke response:

```json
{"text":" who is trying to paint a portrait of"}
```

Graph evidence:

```text
Qwen3 prefill graph capture key: tokens=8 batch=1 pages=1 prefix_rows=0 marlin_scratch=true
```

## Implementation

- Added Qwen3 paged prefill graph resources keyed by exact layout: token count, page size, page index length, prefix rows, batch size, sequence lengths, start positions, and page count.
- Added prefill-lifetime Marlin scratch using the existing decode scratch arena type, but with a prefill-specific config. Hybrid decode uses W4A16, while hybrid prefill uses W4A8, so decode scratch config is insufficient.
- Allowed graph-safe batched weights for dense BF16, W4A16 Marlin, W4A8 Marlin, and W4-hybrid with `INFER_HYBRID_W4A8_PREFILL=1`.
- Split prefill scratch detection by actual prefill path: pure W4A16 allocates W4 scratch, W4A8/hybrid prefill allocates W4A8 scratch, and hybrid is not graph-eligible unless the W4A8 prefill env gate is enabled.
- Kept logits readout eager and left default behavior unchanged unless `INFER_PREFILL_GRAPH=1`.

## Problems

- Full `greedy_consistency` still includes the pre-existing W4A8-vs-BF16 token-diff accuracy gate. That is tracked separately in `docs/experience/errors/2026-05-08-w4a8-quantize-broken-100pct-token-diff.md` and is not introduced by this graph hoist.
- This entry is not a throughput license. #37 owns multi-key graph cache and matched-control 4k/c=4 TTFT measurement.

## Learnings

- Hybrid W4 graph capture needs phase-specific scratch accounting: decode scratch cannot be reused as the source of truth for prefill scratch because the hybrid dispatch policy intentionally uses different weight paths by phase.
- Graph keys must include page layout and start-position fields for prefill. Token-count-only keys risk replaying a captured graph against stale paged-KV metadata.

## Delta vs Baseline

- **Baseline:** `docs/experience/wins/2026-05-09-bench-sglang-reverify-post-p1.0-p1.2.md`

| metric | baseline | now | delta |
|---|---:|---:|---:|
| W4-hybrid `INFER_PREFILL_GRAPH=1` server smoke | not functional | HTTP 200 | unblocked |
| Qwen3 prefill graph key with Marlin scratch | absent | present | unblocked |
| 4k/c=4 TTFT | 1639 ms | deferred to #37 | n/a |

## Artefacts

- Server log: `/tmp/infer-p24-smoke.log`
- Smoke response: `/tmp/infer-p24-smoke.response`
- Raw benchmark output: n/a, Phase 1 functional gate only
