# Zigzag ring device kernel: per-row positions ride the ring — 2026-07-31

> Status: Kernel + backend + ops landed; Mac CUDA typecheck + host ring tests
> green. Pod 2026-08-01: 256K LIVENESS proven. The ring MATH is correct, but the
> shipping wiring was NOT — see the correction below.
>
> **Pod liveness (2026-08-01, HEAD e739a1105, GPUs 1,3):** `nd_parallel_parity`
> `ARLE_ND_SEQ=131072` cp=2 (local shard 65536) **completed a full optimizer
> step** — single-card ref (`DONE loss=3.232068` over 131068 targets) then BOTH
> CP ranks forward+CE+backward+optimizer. Peak host RSS **4.37 GB**, min
> MemAvailable **1.57 TB**, GPU ≤20.5 GB. This is the exact stage that host-OOM'd
> at 343 GB before; the fix was `head_dim` 2→128 in the parity model so the
> single-card ref uses the bf16 chunked-prefill kernel (attention.rs:168) instead
> of the f32 composed `causal_sdpa` that materialized `[heads,seq,seq]`. The ring
> itself never materializes full-seq (peak O(seq/N·block)). #59/#66 closed —
> **liveness only** (a step completes without OOM). Liveness is NOT correctness:
> the shipping CP forward ran plain attention on each shard (below).
>
> **CORRECTION (2026-08-01, HEAD 3d9bc3717): the "5.2% parity FAIL" WAS a real
> bug — a cp split-brain, now fixed.** `masked_writeback_step` sharded the
> sequence by its `cp` ARGUMENT, but the forward decided whether to ring by
> reading a DIFFERENT source, the model field `self.cp` — set only by `set_cp`,
> whose sole non-test caller was the `cp_hidden_parity` diagnostic. No cli/
> production path called it, so `self.cp` was always `single()`: every CP shard
> ran plain attention on its own rows, the ring never fired. The f32-anchored
> gate caught it (loss_cp 3.2425 vs f32 3.0729, 5.5% = 3700× the single-card
> bf16 floor 1.5e-5). The 3-stage device bisection below all PASSED because each
> stage explicitly wired cp (the diagnostic calls `set_cp`) — it verified the
> ring MATH, never the shipping FORWARD, which bypassed it. Fix: thread `cp`
> through `forward_hidden_states`/`forward_batch_hidden_indices`, delete the
> `self.cp` field + `set_cp` (`3d9bc3717`). Pod re-verify (seq=16 FAIL→PASS,
> 256K rings) pending-remote. See
> `errors/2026-08-01-cp-parity-fail-was-bf16-gate-miscalibration.md`. #67 REOPENED
> then fixed under #69.
>
> **Ring math verified (2026-08-01) — with cp explicitly wired:** a 3-stage
> device bisection (each one pod run) proved the ring itself is correct: (1)
> kernel `ring_block_fwd_merge` vs host `ring_forward_tile` PASS at fp32 eps; (2)
> 2-rank NCCL transport vs full-seq causal SDPA PASS at 3e-8 incl the
> non-contiguous zigzag shard; (3) CP hidden (built with `set_cp`) tracks CPU-f32
> as well as single-card (device CE on the reassembled hidden matched host to
> 8e-8). These stages exercised the ring; the shipping forward did not — that
> gap was the bug.


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
