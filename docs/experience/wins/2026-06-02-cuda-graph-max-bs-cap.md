# CUDA Graph warmup now has an operator batch cap

## Context

Goal: continue CUDA Graph integration without falsely enabling DSv4 graph
replay before the DSv4 decode body is graph-safe.

The previous scheduler warmup always pre-captured graph-capable decode batch
sizes up to `min(num_slots, 256)`. That worked as an internal default, but it
did not expose the SGLang-style `--cuda-graph-max-bs` control that DSv4 and
other CUDA models need for matched experiments and production startup tuning.

## What Worked

- Added `SchedulerConfig::cuda_graph_max_bs` with default `256` and validation.
- Added `infer --cuda-graph-max-bs <N>` and forwarded `arle serve
  --cuda-graph-max-bs <N>` to the CUDA backend only.
- Changed CUDA warmup to pre-capture up to `min(num_slots,
  cuda_graph_max_bs)`.
- Added the cap to the CUDA bootstrap scheduling envelope log so every run
  records the graph warmup contract.
- Kept DSv4 fail-closed for CUDA Graph decode. The gate now names the real
  remaining blockers: trace syncs, TP/EP collectives, host launch metadata,
  and host-updated attention/compressor cache counters.

## Verification

Local:

- `cargo fmt --check`
- `git diff --check`
- `CARGO_TARGET_DIR=/tmp/arle-local-cli-test cargo test -p cli --no-default-features --features no-cuda cuda_serve_forwards_cuda_graph_max_bs`

Remote pod:

- Worktree: `/tmp/arle-cuda-graph-worktree-20260602062613`
- Base: `origin/main` at `53ac63a3267195781fdd5784ce1943cb06fb3dbe`
- Patch: current local CUDA Graph cap diff applied cleanly
- `git diff --check`
- `RUSTC=/root/.rustup/toolchains/1.95.0-x86_64-unknown-linux-gnu/bin/rustc CARGO_TARGET_DIR=/tmp/arle-cuda-graph-target /root/.rustup/toolchains/1.95.0-x86_64-unknown-linux-gnu/bin/cargo test -p cli --no-default-features --features no-cuda cuda_serve_forwards_cuda_graph_max_bs --offline`
- `RUSTC=/root/.rustup/toolchains/1.95.0-x86_64-unknown-linux-gnu/bin/rustc CUDARC_CUDA_VERSION=12080 CARGO_TARGET_DIR=/tmp/arle-cuda-graph-target /root/.rustup/toolchains/1.95.0-x86_64-unknown-linux-gnu/bin/cargo check -p infer --no-default-features --features cuda,no-cuda --offline`
- `RUSTC=/root/.rustup/toolchains/1.95.0-x86_64-unknown-linux-gnu/bin/rustc CARGO_TARGET_DIR=/tmp/arle-cuda-graph-target-nocuda /root/.rustup/toolchains/1.95.0-x86_64-unknown-linux-gnu/bin/cargo check -p infer --no-default-features --features no-cuda --offline`

Remote `cargo fmt --check` could not run through the repo toolchain because the
pod has Rust 1.95 installed but not the `rustfmt` component; rustup attempted
to download the component and timed out against `static.rust-lang.org`. The
blocked fmt probe was killed. This is a pod toolchain issue, not a compile
failure.

## Pending Remote

No DSv4 TPOT win is claimed from this change. Full DSv4 graph replay still
needs the next implementation tranche:

- move start-position and attention/compressor metadata to graph-replay-safe
  device inputs or graph-updated kernel params;
- finish preallocating DSv4 per-step scratch that is still allocated in the
  decode body;
- isolate or explicitly disable graph capture around TP/EP collective paths;
- run matched graph-off / graph-on correctness and TPOT A/B on the DSv4 pod.

## Rule

Expose graph controls as generic runtime configuration first. Do not flip a
model's `supports_cuda_graph_decode` answer until replay correctness is proven
for that model's host metadata, scratch allocation, and collective contract.
