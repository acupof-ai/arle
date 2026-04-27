# infer

`infer` is the ARLE runtime and serving crate. It owns the scheduler, HTTP
surface, backend loading, model runtime contracts, and the non-CLI binaries
used for serving and benchmarking.

This README is intentionally crate-local. Public product facts should not be
duplicated here.

## Canonical Docs

- [../README.md](../README.md) — public overview, install, CLI, architecture
- [../README.zh-CN.md](../README.zh-CN.md) — Chinese public entry point
- [../docs/http-api.md](../docs/http-api.md) — HTTP contract and streaming behavior
- [../docs/support-matrix.md](../docs/support-matrix.md) — backend/model/quant support
- [../docs/environment.md](../docs/environment.md) — environment variables
- [../CONTRIBUTING.md](../CONTRIBUTING.md) — contributor workflow and validation
- [../docs/codebase-map.md](../docs/codebase-map.md) — where to start reading the tree

## What Lives In This Crate

- `infer/src/server_engine.rs` — `InferenceEngine` contract and runtime loader
- `infer/src/http_server/` — OpenAI-compatible routes, SSE, metrics, ops endpoints
- `infer/src/scheduler/` — continuous batching, prefix reuse, slot lifecycle
- `infer/src/backend/` — CUDA, Metal, and CPU smoke backends behind feature gates
- `infer/src/model/` — model forward traits and model-family implementations
- `infer/src/ops/` — backend-dispatched operator surface
- `infer/test_data/` — JSON baselines for numerical regression tests

## Build And Run

### CUDA (Linux + NVIDIA)

```bash
export CUDA_HOME=/usr/local/cuda
export INFER_TRITON_PYTHON=.venv/bin/python
cargo build -p infer --release
./target/release/infer --model-path /path/to/Qwen3-4B --port 8000
```

### Metal (Apple Silicon)

```bash
cargo build -p infer --release --no-default-features --features metal,no-cuda --bin metal_serve
./target/release/metal_serve --model-path mlx-community/Qwen3-0.6B-4bit
```

Current Metal GGUF performance floor:

```bash
cargo build -p infer --release --no-default-features --features metal,no-cuda --bin metal_bench
./target/release/metal_bench \
  --model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt-tokens 512 --generation-tokens 1024 --ignore-eos --json
```

### CPU smoke path

```bash
cargo run -p infer --release --no-default-features --features cpu,no-cuda --bin cpu_serve -- \
  --model-path Qwen/Qwen3-0.6B
```

Use `--release` for every path. Debug GPU builds are not a meaningful dev loop.

## Verification

```bash
# CPU-only library checks
cargo check -p infer --no-default-features --features no-cuda --lib
cargo test -p infer --release --no-default-features --features no-cuda --lib
cargo clippy -p infer --no-default-features --features no-cuda --lib -- -D warnings

# Backend-specific checks
cargo check -p infer --no-default-features --features metal,no-cuda --lib
INFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e
cargo test --release --test e2e_qwen35
```

If you change numerical output, regenerate the JSON baselines under
[`infer/test_data/`](test_data/).

## Notes

- Dated benchmark snapshots live under
  [../docs/experience/wins/](../docs/experience/wins/).
- Active CUDA benchmark closure work lives in
  [../docs/plans/2026-04-23-cuda-decode-sglang-alignment.md](../docs/plans/2026-04-23-cuda-decode-sglang-alignment.md);
  the current Metal GGUF floor is
  [../docs/experience/wins/2026-04-27-bench-metal-qwen35-0p8b-gguf-q5-q8-q6qmv.md](../docs/experience/wins/2026-04-27-bench-metal-qwen35-0p8b-gguf-q5-q8-q6qmv.md).
- Public API claims belong in [../docs/http-api.md](../docs/http-api.md), not
  in this crate README.
