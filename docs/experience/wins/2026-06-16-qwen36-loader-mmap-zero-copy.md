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
cold-load wall. Wall-clock is the ground-truth delta because startup phase
timers are nested/overlapping.

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

35B cold-load after run:

```bash
sync
echo 3 > /proc/sys/vm/drop_caches
CUDA_VISIBLE_DEVICES=0 RUST_LOG=info ARLE_CUDA_STARTUP_PROFILE=1 \
  target/release/arle serve --backend cuda --model-path /data01/models/Qwen3.6-35B-A3B --port 18123
```

## Environment

- Code commit under test: `8038994f` (`perf(cuda): mmap qwen bf16 loader tensors`).
  Unrelated `crates/cli/src/train_cli.rs` remained dirty and untouched.
- Remote build/run host: H20 / CUDA 12.9, `CUDARC_CUDA_VERSION=12090`.
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

35B-A3B BF16 after run with page cache dropped and GPUs empty:

| Phase | Count | Total ms | Avg ms |
|---|---:|---:|---:|
| `qwen35.total` | 1 | 39,480.7 | 39,480.7 |
| `loader.shard_mmap` | 27 | 1,574.6 | 58.3 |
| `loader.tensor.owned_copy` | 90 | 0.0 | 0.0 |
| `loader.shard_deserialize` | 26 | 5.7 | 0.2 |
| `loader.moe.stacked_routed_load` | 40 | 35,471.4 | 886.8 |

Owned-copy byte audit after the patch:

| Metric | Result |
|---|---:|
| `loader.tensor.owned_copy` total bytes | 11,520 |
| owned copies >1 MiB | 0 |
| max owned-copy tensor | 256 bytes |

Warm page-cache reference before the cold rerun: `qwen35.total=12,789.4 ms`,
`loader.shard_mmap=980.7 ms`, and `owned_copy_big_count=0`.

## Problems

The first 35B after attempt was delayed because all 8 H20s initially reported
47,015 MiB used by an external GPU client invisible from this container
(`nvidia-smi` listed no process, `lsof /dev/nvidia*` exposed no owner, and
`nvidia-smi --gpu-reset` failed with "In use by another client"). The final
after run above was taken only after the GPUs returned to 0 MiB used and
`drop_caches` succeeded.

## Δ vs baseline

| Metric | Before | After |
|---|---:|---:|
| 35B-A3B BF16 cold load wall (`qwen35.total`) | 47.69s | 39.48s (-17.2%) |
| Large `tensor.owned_copy` uploads | present, 39.8s total phase | eliminated; 0 copies >1 MiB, 11.5 KiB total |
| Shard open/read phase | `loader.shard_read` 37.81s | `loader.shard_mmap` 1.57s |
| Shard backing | `fs::read` heap copy | mmap-backed `SharedTensor` |

## Learnings

The real #101 fix is copy hygiene on the hot BF16 upload path, not per-expert
CPU parse. mmap + direct BF16 borrow removes the large host-copy phase and cuts
the measured cold 35B-A3B BF16 load wall by 17.2%; further load work should
target the now-dominant H2D/page-fault path inside stacked expert upload, not
safetensors metadata parse.
