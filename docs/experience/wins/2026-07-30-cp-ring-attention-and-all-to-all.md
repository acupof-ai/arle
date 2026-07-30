# CP ring attention + all-to-all-to-head transport — CPU-gated, device pending-remote — 2026-07-30

> Status: Core landed + CPU-gated (world==1 byte-identical, multi-block ring
> merge/bwd match full-softmax on CPU). Device kernels compile pod-only (nvcc);
> multi-rank NCCL transport + >65535 parity are pending-remote.

## Goal

Replace the option-B all-gather CP full-attention path (materializes the full
`[b,kv_heads,full_seq,hd]`, OOMs in `slice_bwd` at local seq > 65535) with ring
attention that never materializes the full sequence, and add the `all_to_all`
collective the linear-attention CP path needs. Calibrated against Megatron-Core
(`megatron/core`): ring on the local shard, and all-to-all-to-head for the
Markovian linear-attn recurrence (not the serial carry-ring we first planned).

## What worked

- **Ring flash-2 device kernels** (`crates/cuda-kernels/csrc/attention/ring_block_attention.cu`):
  fwd-merge (online-softmax across ring blocks, absolute causal mask, per-row LSE
  out), finalize (`out=O/L`, `lse=M+ln(L)`), bwd (replay `P=exp(S−lse)`,
  atomicAdd grad_k/grad_v). The large dim launches as `grid.x` (up to 2³¹−1), so
  local seq > 65535 clears the gridDim.y 65535 cap — the exact boundary
  option-B's chunked-SDPA desynced on.
- **`cp_causal_sdpa`** (`ops/ring_attention.rs`): world==1 attends the whole
  local shard as one KV block (host `ring_forward_tile`); cuda world>1 rings k/v
  `cp_size` times through the device kernel. GQA resolved per-block in the kernel
  (k/v ship at kv-head width).
- **`all_to_all` differentiable op** (`ops/collective.rs`): self-adjoint with
  scatter/gather axes swapped; world==1 is identity (out_shape == in_shape); cuda
  world>1 is the loud pending-remote boundary (no single NCCL primitive; layout
  shuffle is pod-only). `BackwardOp::AllToAll` + `SavedContext::AllToAllCtx`.
- Wired into `qwen35.rs` CP full-attn branch, deleting the option-B
  `all_gather_seq`+slice+`repeat_kv` gather (no parallel old+new path).

## Verification (local, CPU)

```
cargo test -p autograd -p train --no-default-features --features no-cuda
```

- `all_to_all_single_rank_is_identity`: fwd value/shape + bwd grad identity at N=1.
- `cp_causal_sdpa` world==1 taped grad == `causal_sdpa_recompute`; multi-block
  ring merge+bwd == full-softmax (ragged / non-aligned / future-only blocks).
- Full autograd+train no-cuda suite green (39 test binaries, 0 failures).

## Pending-remote (the gating work)

- Multi-rank ring transport (`ring_send_recv_kv` NCCL send/recv) and the
  `all_to_all` NCCL seq↔head shuffle — need ≥2 GPU + NCCL.
- The one binding pod gate: **local seq > 65535** (cp=2 global 131072 → local
  65536). BEFORE = OOM at `slice_bwd`; AFTER = a full optimizer step completes
  AND CP loss-sum matches single-card within REL_TOL. Then the cp=4 seq=262144
  ladder rung. (Zigzag load-balancing and the linear-attn all-to-all-to-head
  wiring are follow-on batches; this batch is ring + the `all_to_all` primitive.)

## Rule

Ring never materializes full_seq (peak O(seq/N·hd)); the >65535 fix is launching
the large dim as `grid.x`, not a memory story (the option-B ladder already fit
256K on memory — `2026-07-30-cp-ladder-option-b-fits-256k.md`). A CP collective
with no single NCCL primitive (all-to-all) gets a real world==1 identity + a loud
world>1 pending-remote error — never a silent wrong-shape identity.
