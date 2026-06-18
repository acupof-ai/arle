# Qwen3.6 loader zero-copy borrow path pending remote A/B

## Goal

Remove the extra host `Vec<u8>` copies in the CUDA safetensors loader before H2D
upload. The target is the Qwen3.6-35B-A3B startup path where the stacked BF16
loader profile showed cold shard I/O plus `tensor.owned_copy` dominating load
time.

## Hypothesis

Borrow tensor byte ranges from the mmap-backed shard cache for full tensors, and
allocate only when TP sharding has to materialize a sliced byte buffer. This
should reduce loader RSS and startup wall-clock without changing device bytes.

## Params

- Code path: `crates/infer-cuda/src/loader.rs`
- Model target: Qwen3.6-35B-A3B BF16 first, FP8 follows the same borrowed raw
  tensor path for quant views.
- Verification status: local compile/unit gates complete; remote `.62` load A/B
  pending.

## Results

Local:

```text
rustfmt --edition 2024 --check crates/infer-cuda/src/loader.rs
PASS

CUDARC_CUDA_VERSION=12090 cargo test -p infer-cuda --release \
  --no-default-features --features cuda,no-cuda shards_ --lib -- --nocapture
PASS (3 tests, including the two loader zero-copy tests)

CUDARC_CUDA_VERSION=12090 cargo check -p infer-api --release \
  --no-default-features --features cuda,no-cuda --lib
PASS

CUDARC_CUDA_VERSION=12090 cargo clippy -p infer-api --release \
  --no-default-features --features cuda,no-cuda --lib -- -D warnings
PASS
```

Remote `.62` A/B is pending. The attempted clean verification build was blocked
by pod toolchain/dependency hygiene first, then by the known TileLang AOT
pipeline-stage failure when not using the project prebuilt-kernel path. No
runtime load-speed claim is made in this entry.

## Problems

- Package-wide local `rustfmt --check` is still blocked by unrelated dirty
  `crates/cli/src/train_cli.rs`.
- The remote `.62` build must use `scripts/dsv4_fast_build.sh` plus a valid
  `ARLE_CUDA_KERNELS_PREBUILT_DIR`; raw `cargo build --features cuda` triggers
  TileLang AOT and is not the right verification route on that pod.

## Learnings

For safetensors loader optimization, keep the invariant simple: full-shard
loads borrow mmap bytes, partial TP shards own sliced bytes. Unit-test that
ownership boundary directly so future loader refactors do not accidentally
reintroduce full-tensor host copies.
