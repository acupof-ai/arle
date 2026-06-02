# DSv4 spec verifier and SGLang-contract gate

Date: 2026-06-02

## Context

The DSv4 CUDA path was being compared against a SGLang DSv4-Flash TP8 + EAGLE
target, but ARLE did not implement a DSv4 target verifier for speculative
decode and could silently fall back to slower non-comparable paths.

## What Worked

- Added a DSv4 `forward_spec_verify_batch` implementation that snapshots the
  model-owned incremental compressor/indexer state, runs target verification,
  and replays only the accepted verifier inputs on commit.
- Kept verifier decode eager for the local verifier context so normal decode
  graph cache state is not contaminated by verifier rows.
- Made `sglang` profile fail closed with an explicit missing-path reason when
  the checkpoint declares `num_nextn_predict_layers=1` but ARLE CUDA does not
  load or execute the internal `mtp.0` / EAGLE draft path.
- Kept the shared FlashMLA FP8 KV pool prefill path from touching decode-only
  sub-ranges before `bind_fp8_kv_pool_view`.

## Verification

Local:

```text
cargo fmt --check
git diff --check
cargo check -p infer --no-default-features --features no-cuda
CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda
```

Remote pod `/data01/build/arle`, commit
`8aef4cca83a7d9ca607ba3c920cbd0ff24ea0fcb` plus this working-tree diff:

```text
RUSTUP_TOOLCHAIN=stable CUDARC_CUDA_VERSION=12080 CUDA_HOME=/usr/local/cuda \
TORCH_CUDA_ARCH_LIST=9.0 ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1 \
ARLE_DEEPGEMM_ROOT=/data01/build/arle/crates/cuda-kernels/vendor/deepgemm \
ARLE_DEEPGEMM_LIBRARY_ROOT=/data01/build/arle/crates/cuda-kernels/vendor/deepgemm/deep_gemm \
ARLE_DEEPGEMM_CUTLASS_INCLUDE=/data01/build/arle/crates/cuda-kernels/vendor/flashmla/csrc/cutlass/include \
ARLE_DEEPEP_DIR=/data01/build/arle/../DeepEP \
bash scripts/dsv4_fast_build.sh
```

Result: prebuilt CUDA artifacts were used; build completed in 16.63 s and did
not harvest older `OUT_DIR` artifacts.

`sglang` profile contract test failed closed as intended:

```text
DeepSeek V4 profile `sglang-best-practice` requested, but this binary is not on the SGLang best-practice path:
 - CUDA graph decode must be full_decode ...
 - DSv4 distributed decode still needs token-owned DP/EP request routing before graph capture
 - DSv4 checkpoint declares num_nextn_predict_layers=1, but ARLE CUDA does not load or execute the internal mtp.0/EAGLE draft path yet
```

Debug fallback correctness smoke:

```text
prompt="The capital of France is"
max_new_tokens=32
decode_ms=1582.7
generated_text=" Paris.\nThe capital of France is Paris.\nThe capital of France is Paris.\n..."
```

This is about 49.5 ms/token on the fallback path. It is not comparable with the
SGLang TP8 + EAGLE + CUDA graph target.

## Rule

Do not treat `--spec-enabled`, piecewise CUDA graph, or debug-fallback token
output as evidence for the DSv4 SGLang target. The comparable path requires
full decode graph capture, token-owned DP/EP routing, and the internal
`mtp.0`/EAGLE draft path from the DSv4 checkpoint.
