# Multi-GPU F0 — Remote Verification Guide

Owner: ckl
Date: 2026-04-28
Companion to: [`2026-04-28-single-node-multi-gpu.md`](./2026-04-28-single-node-multi-gpu.md)

This is what to run on a CUDA host to verify the F0 foundation landed
correctly. All Mac-runnable parts already passed locally before push;
this guide covers the parts only a CUDA + NCCL machine can exercise.

## What landed in F0

Foundation only — no caller wires anything into the running serving path
yet. F1 (TP dense) is the first phase that actually uses these.

1. **M0 — explicit device ordinal**
   - `crates/cuda-kernels/src/tensor.rs`: `DeviceContext::on_device(ordinal: u32)`,
     `DeviceContext::new()` reads `INFER_CUDA_DEVICE` env (default 0),
     `sm_count()` queries `self.ctx.cu_device()` instead of literal 0,
     `pub ordinal` field added.
   - `docs/environment.md`: `INFER_CUDA_DEVICE` documented.
2. **Multi-axis rank-layout math** (Mac-tested)
   - `infer/src/tensor_parallel.rs`: `MultiAxisConfig`, `RankCoord`, 8 group
     builders (TP/PP/Attn-TP/Attn-DP/Attn-CP/MoE-TP/MoE-EP/MoE-DP). Ports
     of SGLang `parallel_state.py:1721–1900` and `dp_attention.py:240–271`.
3. **TCP rendezvous protocol** (Mac-tested)
   - `infer/src/distributed/init_method.rs`: `RendezvousServer` /
     `RendezvousClient` for distributing a 128-byte unique_id across N
     ranks. Drop-in for the NCCL `ncclGetUniqueId` payload that F1 wires.
4. **NCCL FFI declarations + `CollectiveBackend` trait**
   - `crates/cuda-kernels/src/ffi/nccl.rs`: extern "C" declarations for
     init/destroy + AllReduce/AllGather/ReduceScatter/Broadcast/Send/Recv/
     GroupStart/End. Gated `#[cfg(feature = "nccl")]`, which now implies
     `cuda` (NCCL needs CUDA).
   - `crates/cuda-kernels/src/collective.rs`: `CollectiveBackend` trait +
     `NcclBackend` impl skeleton (real NCCL calls; needs libnccl at link
     time).
5. **Spec `Shard` annotations**
   - `crates/qwen3-spec/src/lib.rs` and `crates/qwen35-spec/src/lib.rs`:
     per-tensor `Shard` enum (Replicated / Column / Row / MergedColumn /
     QkvFused / VocabParallel) covering every layer + global tensor name.

## Verification on a CUDA host

### Step 1 — Build verification

```bash
# 1.1 Existing CUDA build still works (regression check).
CUDA_HOME=/usr/local/cuda cargo build --release --features cuda

# 1.2 New: NCCL FFI links. Requires libnccl on the system. Sets the F1
#     stage. Should succeed on any host with CUDA + NCCL installed
#     (RHEL/Ubuntu typically: `apt install libnccl2 libnccl-dev`).
CUDA_HOME=/usr/local/cuda cargo build --release --features cuda,nccl
```

**Expected:** both build cleanly. Build 1.2 will fail to link if libnccl
is missing — install it first (any NCCL ≥ 2.18 should match the FFI
signature surface; F1 will pin a min version when the linker is wired).

### Step 2 — Unit test verification

```bash
# 2.1 Multi-axis rank math (24 tests; SGLang docstring scenarios reproduced).
cargo test -p infer --release tensor_parallel

# 2.2 TCP rendezvous (5 tests, world_size 2 and 4 with localhost loopback).
cargo test -p infer --release distributed::init_method

# 2.3 Collective trait + DType / ReduceOp size sanity.
cargo test -p cuda-kernels --release --features cuda collective

# 2.4 NCCL FFI struct/enum sizes (only runs with --features nccl).
cargo test -p cuda-kernels --release --features cuda,nccl ffi::nccl

# 2.5 Spec Shard annotations (32 tests across both crates).
cargo test -p qwen3-spec -p qwen35-spec --release
```

**Expected:** all green. Fixtures are deterministic; no flakes expected.

### Step 3 — Behavior verification (M0 explicit device ordinal)

This is the only F0 change with runtime impact. Goal: confirm zero
behavior change in the default single-GPU path, and confirm
`INFER_CUDA_DEVICE=N` actually pins to GPU N.

```bash
# 3.1 Baseline — env var unset, behaves like prior code (ordinal 0).
unset INFER_CUDA_DEVICE
nvidia-smi  # note GPU 0 baseline memory

# Pick any existing smoke / bench script and run it.
scripts/bench_guidellm.sh m0-baseline

# In another terminal during the run:
nvidia-smi  # confirm GPU 0 is the one busy
```

```bash
# 3.2 INFER_CUDA_DEVICE=0 — explicit, must produce identical numbers.
INFER_CUDA_DEVICE=0 scripts/bench_guidellm.sh m0-explicit-0
```

```bash
# 3.3 INFER_CUDA_DEVICE=1 — pins to GPU 1 (run only on multi-GPU host).
INFER_CUDA_DEVICE=1 scripts/bench_guidellm.sh m0-pin-1

# In another terminal:
nvidia-smi  # confirm GPU 1 is the one busy, GPU 0 is idle
```

**Acceptance** (matches the F0 row in plan §5):

- 3.1 vs 3.2: tok/s and TTFT within ±2% (random run-to-run noise OK).
- 3.3 binds to GPU 1 (visible in `nvidia-smi` during the run).
- All 3 runs produce identical greedy-decode tokens (compare any saved
  output JSON if the bench script writes them; if not, run a short
  prompt manually and diff outputs).

### Step 4 — What's NOT yet verifiable (waits for F1)

These need the F1 layer to land before they can be exercised end-to-end:

- Actual NCCL collective execution. The FFI is declared but no caller
  invokes it yet; F1's `LayerCommunicator::all_reduce_post_attention`
  wires the first one.
- Per-rank thread spawning. F0 has the `RendezvousServer/Client` and the
  `CollectiveBackend` trait; F1's `TpModelWorker` is the first
  consumer.
- Sharded weight loading using `Shard` annotations. F1's loader rewrite
  is the first consumer.

## If anything fails

- **`cargo build --features cuda,nccl` link error**: post the linker
  error; likely libnccl missing or wrong version. Don't proceed to F1
  until this is clean — F1 will need the same link path.
- **`cargo test ... tensor_parallel` failure**: every test name carries a
  SGLang docstring line number it ports; diff against
  `python/sglang/srt/distributed/parallel_state.py:NNNN` at SGLang
  commit `1a55646dcdf06f77441506be5c74afb045341636`.
- **`INFER_CUDA_DEVICE=1` doesn't bind to GPU 1**: check
  `crates/cuda-kernels/src/tensor.rs::parse_device_ordinal_from_env` —
  parsing helper is unit-tested but only on Mac via
  `#[cfg(test)]`; on the CUDA host it runs through `DeviceContext::new()`.
  Confirm the env var made it through (`echo $INFER_CUDA_DEVICE` before
  the run).
- **Greedy parity drifts**: M0 must be a no-op for behavior. If parity
  drifts, the change is not a no-op — that's a regression to triage
  before F1.

## Then what

Once F0 verification passes, post the bench numbers as a comment on the
companion wins entry
`docs/experience/wins/2026-04-28-multi-gpu-f0-foundation.md` (currently
marked `pending-remote`). That unblocks the F1 brief.
