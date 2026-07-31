# CP zigzag SeqShard + per-row-position ring — CPU-gated — 2026-07-31

> Status: Core landed + CPU-gated (zigzag partition exact; cp==1 byte-identical
> contiguous). Device per-row-position ring kernel is pending-remote NCCL.

## Context

Contiguous CP shards are correct but imbalanced: under a causal mask the rank
owning the tail attends ~N× the keys the head rank does, so the ring stalls on the
slowest rank every step. Megatron-Core (`get_batch_on_this_cp_rank`) fixes this by
splitting the sequence into `2N` chunks and giving rank r the pair `{r, 2N−1−r}` —
one front, one back — so every rank carries the same causal work. Our `SeqShard`
was a single contiguous range, so it couldn't express that pair.

## What worked — chunk-list SeqShard + ring masks by absolute position

`SeqShard` is now an ordered chunk list (`context_parallel.rs`):
- `CpContext::shard(seq)` = zigzag: chunks `r` and `2N−1−r` of a `2N`-way split
  (front+back). `seq % (2N) == 0` required (callers pad up).
- `local_rows()` = the gather index into the global sequence (chunk order).
- `local_of(pos)` = its inverse (global position → local row), which `opd.rs`
  rebases loss targets through.
- `DpContext::batch_shard` stays `SeqShard::contiguous` — batch items are
  independent, no causal imbalance to balance. DP must not zigzag.

The two chunks are non-contiguous, so the ring can no longer assume
`q_abs = cp_rank*s`. The whole ring path now masks by **per-row absolute
position** end-to-end (the user's explicit choice over a device-side remap):
- `ring_forward_tile`/`ring_backward_tile` take `q_pos: &[usize]` and each block's
  `k_pos: &[usize]`, masking `k_pos[c] > q_pos[r]` (skip-future, since columns are
  no longer monotonic under zigzag).
- `cp_causal_sdpa` gained `positions: Option<&[usize]>` — `None` = contiguous
  `cp_rank*s+r` (byte-identical legacy), `Some` = the zigzag map.
- `RingAttentionCtx` carries `q_pos: Vec<usize>` + per-block `k_pos: Vec<usize>`
  (was scalar `q_abs`/`k_abs`) so backward replays the same mask.
- `qwen35.rs` derives `positions = cp.shard(seq_len*cp.size).local_rows()` at the
  attention call — the same mesh-derived view `opd.rs` shards RoPE by, not a new
  threaded param. At `cp.size==1` this is `0..seq_len`, byte-identical to `None`.

## Verification (local, CPU)

```
cargo test  -p autograd -p train --no-default-features --features no-cuda
cargo clippy -p autograd -p train --no-default-features --features no-cuda
CUDARC_CUDA_VERSION=12080 cargo check -p train -p infer-api --release \
    --no-default-features --features cuda,no-cuda --lib
```

- `zigzag_covers_sequence_disjointly` / `zigzag_pairs_front_and_back`: every
  position covered exactly once; rank owns seq/size rows as a front+back pair.
- `local_targets_partition_by_owner` / `shard_union_reconstructs_single_card_targets`:
  the union of every shard's rebased targets == the single-card set (no lost, dup,
  or misrebased pair) — a wrong split silently corrupts `inv_n` and every gradient.
- `ring_matches_full_softmax_forward_and_backward` /
  `ring_ragged_blocks_nonaligned_qabs_and_future_block`: the per-row-position ring
  still matches full causal softmax fwd+bwd (the mask refactor preserved the math).
- `cp_causal_sdpa_world1_matches_causal_sdpa_recompute`: world==1 taped grad still
  bit-close to `causal_sdpa_recompute` (positions=None path).
- Full autograd+train no-cuda suite green; clippy clean; Mac CUDA typecheck passes.

## Pending-remote

The device ring kernel is still scalar-position (`q_abs`/`k_abs`). `cp_causal_sdpa`'s
CUDA path errors loudly on `positions.is_some()` — a zigzag shard on GPU needs a
per-row-position kernel (pod-only), so it defers explicitly rather than silently
mis-attending. Pod gate: >65535 local-seq CP parity + a zigzag-vs-contiguous
load-balance c-sweep (multiproc timing change ⇒ TP=N c8/c16, not just an N=2 loss
check).

## Rule

Zigzag makes the shard non-contiguous, so any code that assumed `pos = start + row`
breaks silently. Mask by absolute position through the whole chain (forward, tape
ctx, backward) — a scalar base is the bug. Derive the position map from the mesh at
the point of use (`cp.shard(...).local_rows()`), the same view the rest of the
forward shards by; don't thread a new param. When the device path can't yet honor
the general case, make it a loud error, never a silent contiguous fallback.
