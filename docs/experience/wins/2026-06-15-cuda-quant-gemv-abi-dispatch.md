# CUDA Quant GEMV ABI Dispatch

## Context

P4/P5 of `docs/plans/2026-06-15-cuda-quant-subsystem-plan.md` needed resident
Qwen FP8/NVFP4 decode-linear kernels without dense BF16 materialization, while
proving the existing DSv4 E8M0 block-scaled path did not regress.

This entry is a correctness/build gate for kernel ABI + operator dispatch. It is
not a Qwen3.6 serve or throughput claim; P6/P7 loader/MoE wiring is still required
before `scripts/bench_guidellm.sh` can run against the target checkpoints.

## What Worked

- Added ABI-generic GEMV launchers in the existing `quantized_gemv.cu`:
  `gemv_fp8_block_scaled{,_batch}_cuda` for FP8 E4M3 + direct f32 block/per-shard
  scales, and `gemv_fp4_e2m1_group{,_batch}_cuda` for packed E2M1 + E4M3 group
  scales + f32 global scale.
- Kept the existing `dsv4_fp{8,4}_*` C symbols and E8M0 scale path intact; Qwen
  ABI entrypoints are separate symbols in the same source file.
- Split resident quant linear dispatch into `crates/infer-cuda/src/ops/quant_linear.rs`;
  `ops.rs` now only shape-checks and branches before touching the dense dummy
  `DeviceMatrix::data`.
- Split CUDA GEMV reference tests out of `ffi/gemm.rs` into
  `ffi/gemm_tests.rs`.

## Evidence

Local Mac gates:

```bash
cargo fmt --all -- --check
git diff --check
CUDARC_CUDA_VERSION=12060 cargo check -p infer-api --release \
  --no-default-features --features cuda,no-cuda --lib
cargo test -p infer-cuda --release --no-default-features --features no-cuda --lib
```

Result: pass; `infer-cuda` no-cuda suite ran 85 tests.

H20 CUDA reference tests:

```bash
CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=12060 \
cargo test --offline -p cuda-kernels --release --features cuda \
  gemv_matches_reference -- --nocapture
```

Result:

```text
test ffi::gemm::tests::fp8_block_scaled_gemv_matches_reference ... ok
test ffi::gemm::tests::fp4_group_gemv_matches_reference ... ok
```

DSv4 post-P4 regression gate:

```bash
DSV4_MODEL_ROOT=<remote DSv4 model root> \
INFER_DSV4_MODEL_PATH="$DSV4_MODEL_ROOT" \
ARLE_DSV4_MODEL_PATH="$DSV4_MODEL_ROOT" \
INFER_DSV4_BATCH_DECODE_VALIDATE=2,4 \
INFER_DSV4_MAX_NEW=8 \
scripts/dsv4_multigpu_parity.sh
```

Result:

```text
batch_decode_reference clean_tokens=[11111, 603, 671, 6102, 294, 8760, 344, 11111]
batch_decode_reference_rerun ... ref_self_parity=true ref_self_first_div=None
batch_decode_validate batch=2 byte_parity=true
batch_decode_validate batch=4 byte_parity=true
```

The post-P4 tokens match the pre-P4 baseline captured in
`2026-06-15-dsv4-pre-p4-quantized-gemv-baseline.md`.

## Rule

Resident quant decode kernels should add ABI-named entrypoints and dispatch before
`DeviceMatrix::data` dereference. DSv4 E8M0 symbols stay regression-gated by
byte-parity whenever `quantized_gemv.cu` changes.
