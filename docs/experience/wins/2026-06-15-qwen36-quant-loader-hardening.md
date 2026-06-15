# Qwen3.6 Quant Loader Hardening

## Context

Follow-up hardening after `68a331a0` wired Qwen3.6 resident quant loader and MoE
dispatch.

## What Worked

- Pure-BF16 checkpoints now skip quant-header probing when `config.json` has no
  `quantization_config`, avoiding unnecessary safetensors header deserialization
  on the default path.
- Routed quant MoE now validates dispatch signatures at load time: every local
  expert within each projection must match rows, cols, quant scale shape, block
  shape, and group size. Gate/up signatures must also match because the paired
  dispatch derives metadata from the first gate expert.
- FP4 global pointer-table construction is guarded by `routed_quant` for symmetry
  with the scale table path.

## Verification

```bash
cargo fmt -p infer-cuda -- --check
CUDARC_CUDA_VERSION=12060 cargo check -p infer-api --release \
  --no-default-features --features cuda,no-cuda --lib
cargo test -p infer-cuda --release --no-default-features --features no-cuda --lib
CUDARC_CUDA_VERSION=12060 cargo clippy -p infer-api --release \
  --no-default-features --features cuda,no-cuda --lib -- -D warnings
```

Results: all passed; `infer-cuda` no-cuda lib tests remain 85/85.

## Rule

Quant dispatch metadata must be validated where weights are loaded. Do not let a
later kernel launch inherit shape assumptions from expert 0 without proving every
expert in that dispatch group has the same signature.
