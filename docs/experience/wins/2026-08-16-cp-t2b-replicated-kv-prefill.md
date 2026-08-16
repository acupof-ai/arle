# CP T2.b: replicated-KV CP prefill in the qwen35 CUDA executor

Status: pending-remote (needs the H20 pod; see gates below).

## Context

T2 of `docs/plans/2026-08-16-cp-ideal-state.md`: shard prefill compute across
the attn_cp group while KV stays replicated. All CP code lives in `infer-cuda`;
plans, engine-core, and the seam are unchanged.

- Slicing decision + per-chunk geometry: `executor/qwen35.rs`
  `prefill_row_paged_default` (chunk >= cp x 256 rows, DSpark off).
- Slice page table: `loader.rs` `PageMeta::for_slot_slice` (explicit kv_len,
  pool coverage checked as `>=`; `for_rows`' exact ensure unchanged).
- Per-layer KV all-gather + remote page writes: `qwen35_attention.rs`
  `cp_share_chunk_kv`.
- GDN state relay (recv prev / advance / send next / broadcast last):
  `qwen35_attention.rs` `linear_attention` CP arm.
- Attention weights shard over the mesh attn_tp axis and replicate across cp:
  `qwen35_load.rs` (`attn_cfg`); attention reduces route over the attn_tp
  sub-comm under cp (`TpRuntime::attn_all_reduce_sum`).
- Design delta vs the plan: the residual stream stays full-chunk on every rank
  (attention/GDN interiors are sliced, outputs row-gathered per layer). The
  FFN/MoE weights shard over the full tp world, so per-cp-group row slices
  would make the global FFN all-reduce shape- and semantics-incoherent;
  full-chunk FFN input restores it and makes lm_head/sampling rank-identical
  (no token broadcast needed).

## Gates (pending-remote)

- needle ladder x3 at cp=2 (tp=8, attn_cp=2) vs the cp=1 envelope.
- 128K prefill wall-clock, target >= 1.6x at cp=2 on the FA3 path.
- cp=1 A/B: byte-identical behavior (all CP branches key off attn_cp_size>1).

## Rule

CP compute-sharding must not touch the FFN row set: axes that shard over the
full tp world need identical rows on every rank, so slice only the
attn_tp-sharded interiors and gather rows before the next full-world op.
