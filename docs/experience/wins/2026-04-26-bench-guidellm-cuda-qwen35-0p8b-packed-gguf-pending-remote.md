# CUDA Qwen3.5 0.8B Packed GGUF — pending remote

## Goal

- Type: regression.
- Record the required CUDA benchmark stub for the packed GGUF weight changes
  that touched `crates/cuda-kernels/csrc/` and CUDA inference loaders.

## Hypothesis

- CUDA Qwen3.5-0.8B GGUF Q4_K_M should keep embeddings and linear weights
  packed for Q8_0/Q3_K/Q4_K/Q5_K/Q6_K and avoid BF16 load-time materialization
  except where V-column reorder still requires the dense fallback.

## Command

Planned remote run:

```bash
cargo build --release
scripts/bench_guidellm.sh cuda-qwen35-0p8b-packed-gguf \
  --model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  --processor models/Qwen3.5-0.8B
```

Local validation performed on Mac:

```bash
cargo check -p infer --no-default-features --features cuda,no-cuda
```

## Environment

- Backend: CUDA continuous batching
- Model: Qwen3.5-0.8B GGUF Q4_K_M
- Hardware: pending remote NVIDIA host
- CUDA version: pending remote
- Commit: `ea6b3aa` + dirty working tree for this change
- Feature set: `cargo build --release`
- Non-default flags / env vars: none planned
- Server launch: pending remote

## Results

Status: `pending-remote`.

Local check:

| Check | Result |
|---|---|
| `cargo check -p infer --no-default-features --features cuda,no-cuda` | passed |

## Problems

- No local NVIDIA GPU or nvcc on this Mac, so CUDA kernels were typechecked
  through the `cuda,no-cuda` feature path but not compiled by nvcc or measured.
- New CUDA Q5_K and Q8_0 embedding kernels need remote parity coverage against
  GGUF host dequantization.

## Learnings

- The CUDA loader must not return a packed `DeviceMatrix` unless the consuming
  op supports that `WeightFormat`; Q8_0 embedding needed explicit decode
  kernels once embeddings stopped forcing BF16.

## Delta vs baseline

- Baseline: pending remote.

| Metric | baseline | now | Delta |
|---|---:|---:|---:|
| TTFT p50 | pending | pending | pending |
| out tok/s | pending | pending | pending |
| peak VRAM | pending | pending | pending |

## Artefacts

- Raw: pending remote
- CSV: pending remote
- HTML: pending remote
- Service trace: pending remote

## Notes

- Code change since baseline: CUDA packed Q5_K GEMV/dequant, Q8_0/Q3_K/Q4_K/Q5_K/Q6_K embedding dispatch, Qwen3.5 GGUF packed loader cleanup.
- Suspected cause of any regression: packed embedding/kernel launch overhead for small batches.
- Follow-ups: run on the CUDA bench host and replace this stub with a dated completed entry.
