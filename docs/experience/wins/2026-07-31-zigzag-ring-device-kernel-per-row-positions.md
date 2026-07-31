# Zigzag ring device kernel: per-row positions ride the ring — 2026-07-31

> Status: Kernel + backend + ops landed; Mac CUDA typecheck + host ring tests
> green. Pod 2026-08-01: 256K LIVENESS proven (see below); parity still FAILs
> (5.2% rel_err) — a device-ring correctness bug, isolation test in flight.
>
> **Pod liveness (2026-08-01, HEAD e739a1105, GPUs 1,3):** `nd_parallel_parity`
> `ARLE_ND_SEQ=131072` cp=2 (local shard 65536) **completed a full optimizer
> step** — single-card ref (`DONE loss=3.232068` over 131068 targets) then BOTH
> CP ranks forward+CE+backward+optimizer. Peak host RSS **4.37 GB**, min
> MemAvailable **1.57 TB**, GPU ≤20.5 GB. This is the exact stage that host-OOM'd
> at 343 GB before; the fix was `head_dim` 2→128 in the parity model so the
> single-card ref uses the bf16 chunked-prefill kernel (attention.rs:168) instead
> of the f32 composed `causal_sdpa` that materialized `[heads,seq,seq]`. The ring
> itself never materializes full-seq (peak O(seq/N·block)). #59/#66 closed.
>
> **Open (parity):** rel_err 5.2% (seq=16) / 2.2% (seq=131072), deterministic
> (bit-identical rerun), RUN_EXIT=1 = assertion only (no crash). Loss asymmetric
> across zigzag ranks (rank1 contiguous shard fine, rank0 non-contiguous off) →
> device ring diverges from its verified host math. The `head_dim=2` 0.39% was a
> genuine f32-vs-bf16 confound (ref off-envelope); at 128 both paths are bf16 so
> 5.2% is a real kernel/transport bug. Bisected by a single-GPU kernel-vs-host
> isolation test (`device_ring_two_blocks_matches_host_reference_gqa_hd128`).

## Context

CP zigzag `SeqShard` (front+back chunk pair) landed on the host, and
`cp_causal_sdpa` threaded per-row positions through the CPU path — but the CUDA
device ring masked causally by a scalar `q_abs`/`k_abs` base+offset (`k_abs + c >
q_pos`), which assumes each rank's rows are ONE contiguous block. So the device
ring rejected `positions.is_some()` with a loud pending-remote error: on the pod,
CP with zigzag couldn't run at all (and even contiguous device-CP was foreclosed).

## What worked — positions as data, riding the existing f32 ring

Unified the kernel to mask by **per-row absolute position**: q row r attends k col
c iff `k_pos[c] <= q_pos[r]`. Contiguous becomes the special case (`pos =
base..base+n`); zigzag's two non-contiguous chunks mask correctly. `break` →
`continue` in both fwd + bwd loops, since zigzag columns aren't monotonic.

The transport insight: positions are small integers, exact in f32 for seq < 2²⁴
(16M » 256K), so they ride the **existing** `ring_send_recv_kv` (f32) alongside
k/v — no new NCCL primitive. Each rank uploads only its OWN positions; the block
arriving at ring step j carries the true positions of whichever rank owns it
(contiguous or zigzag), so **no rank computes another's layout** — zero
equal-shard assumption. Backward re-uploads each saved block's `k_pos` Vec.

Touch points (all one mechanism):
- `ring_block_attention.cu`: kernels take `const float* q_pos, k_pos` (drop
  `int q_abs, k_abs`); mask `(int)k_pos[c] > (int)q_pos[row]` → continue.
- `ffi/attention.rs` + `backend.rs` `RingBlockDims` (drop scalar bases) +
  `ring_block_fwd_merge`/`ring_block_bwd` trait+cuda impl: add `q_pos`/`k_pos`
  device-handle params.
- `ring_attention.rs`: q_pos from `positions` (or contiguous default); k_pos
  starts local, rings with k/v (`ring_rotate_positions` reads it back for the tape
  ctx); the `positions.is_some()` bail is gone.

## Verification

- **Local (Mac):** `cargo build -p autograd --features no-cuda` (host ring path)
  and `CUDARC_CUDA_VERSION=12080 cargo check -p autograd --features cuda,no-cuda`
  (FFI/backend/ops signatures agree) both green; clippy clean. The 3 host ring
  tests (`ring_matches_full_softmax`, ragged/nonaligned/future,
  `cp_causal_sdpa_world1`) stay green — the CPU reference already masked per-row,
  so this aligns the device path to it.
- **Pod (the gate) — pending-remote:** build `cuda,nccl`, run
  `nd_parallel_parity` `ARLE_ND_SEQ=131072` (cp=2, local 65536, zigzag): completes
  a full optimizer step (no `slice_bwd` OOM — ring never materializes full_seq)
  AND CP loss-sum matches single-card within REL_TOL 1e-3. The `.cu` needs nvcc;
  the device kernel is unverifiable on the Mac.

## Rule

When a device kernel masks by a scalar base+offset, it silently assumes contiguous
layout — a zigzag/packed shard needs per-element positions. Pass positions as
data, and if they must move with a ringed buffer, pick a dtype that rides the
transport you already have (f32 is exact for integer positions < 2²⁴) rather than
adding a parallel i32 channel. Each rank declares only its own positions; never
reconstruct a peer's layout from `rank*size` — that's the equal-shard trap.
