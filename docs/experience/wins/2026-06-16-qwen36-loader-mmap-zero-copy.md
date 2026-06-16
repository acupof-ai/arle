# Qwen3.6 loader mmap + BF16 zero-copy upload

## Goal

Optimization / diagnosis follow-up for #101: remove the measured 35B-A3B BF16
loader host-copy wall by uploading BF16 tensors directly from safetensors shard
bytes instead of materializing per-tensor `Vec<u8>` copies.

## Hypothesis

The 35B-A3B BF16 load profile showed `loader.shard_read` at 37.8s and
`loader.tensor.owned_copy` at 39.8s inside a 47.7s total load. Replacing
`fs::read` shard loads with mmap and routing direct BF16 uploads through
`SharedTensor` should eliminate the large `tensor.owned_copy` records and reduce
cold-load wall once an uncontended GPU is available for the 35B rerun.

## Command

Local compile gates:

```bash
CUDARC_CUDA_VERSION=12090 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
cargo fmt --check -p infer-cuda
CUDARC_CUDA_VERSION=12090 cargo clippy -p infer-api --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
```

Remote CUDA build and smoke:

```bash
CUDARC_CUDA_VERSION=12090 CUDA_HOME=/usr/local/cuda cargo build --release --features cuda --bin arle
CUDA_VISIBLE_DEVICES=0 RUST_LOG=info ARLE_CUDA_STARTUP_PROFILE=1 \
  target/release/arle serve --backend cuda --model-path /data01/models/Qwen3.5-0.8B --port 18121
```

## Environment

- Local commit under test: dirty working tree with only `crates/infer-cuda/src/loader.rs`
  changed by this tranche; unrelated `crates/cli/src/train_cli.rs` remained dirty
  and untouched.
- Remote build host: H20 / CUDA 12.9, `CUDARC_CUDA_VERSION=12090`.
- Remote source: `/data01/arle-clean` clean source tree with only this
  `loader.rs` patch applied.

## Results

Before baseline from the #101 follow-up profile:

| Phase | Count | Total ms | Avg ms |
|---|---:|---:|---:|
| `qwen35.total` | 1 | 47,691.6 | 47,691.6 |
| `loader.shard_read` | 27 | 37,807.7 | 1,400.3 |
| `loader.tensor.owned_copy` | 613 | 39,782.9 | 64.9 |
| `loader.shard_deserialize` | 26 | 2.2 | 0.1 |

Patch behavior:

- shard cache now mmaps shard files (`loader.shard_mmap`) instead of
  `fs::read` into heap `Vec<u8>`;
- BF16 direct upload APIs (`load_vec`, `load_matrix`, `load_conv1d_vec`, and
  BF16 sharded variants) borrow `SharedTensor` bytes directly;
- stacked MoE was already on the borrow path, and remains unchanged.

Remote smoke on Qwen3.5-0.8B:

| Check | Result |
|---|---:|
| CUDA release build | PASS |
| HTTP serve bind | PASS |
| Generation smoke | PASS |
| `loader.tensor.owned_copy` count | 54 |
| `loader.tensor.owned_copy` total bytes | 10,944 |
| owned copies >1 MiB | 0 |
| `loader.shard_mmap` count / total | 1 / 0.0 ms |
| `loader.shard_deserialize` count / total | 1 / 0.6 ms |

The remaining owned copies in the smoke are tiny F32/BF16 normalization vectors
from dtype-conversion paths, not large BF16 matrix uploads.

## Problems

The required 35B-A3B BF16 after-run could not be completed in this session:
all 8 H20s reported 47,015 MiB already used, `nvidia-smi` listed no process in
this container, `lsof /dev/nvidia*` did not expose an owner, and
`nvidia-smi --gpu-reset` failed with "In use by another client". That is an
external GPU-client confounder, so a 35B load delta taken now would not be
SOLID.

## Δ vs baseline

| Metric | Before | After |
|---|---:|---:|
| 35B-A3B BF16 cold load wall | 47.7s | pending uncontended GPU |
| Large `tensor.owned_copy` uploads | present, 39.8s total phase | eliminated on direct BF16 path; small smoke has 0 copies >1 MiB |
| Shard backing | `fs::read` heap copy | mmap-backed `SharedTensor` |

## Learnings

The real #101 fix is copy hygiene on the hot BF16 upload path, not per-expert
CPU parse. After the mmap/borrow patch, the remaining decisive gate is a clean
35B-A3B BF16 cold-load A/B on an uncontended GPU; any run with hidden 47 GiB
contexts is an invalid baseline.
