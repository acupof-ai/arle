# CP T2.b: replicated-KV CP prefill in the qwen35 CUDA executor

Status: shipped 2026-08-16 (pod H20, ThinkingCap-Qwen3.6-27B-FP8, BF16 KV).

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

## What worked

The lockstep invariant held: every rank builds the identical plan, every
rank's `PagedKVPool` covers the whole prefix, and CP shards compute only.
The GDN state chain relays (gdr f32 state + conv tail window) across cp ranks
in chunk order, with the last slice broadcasting its terminal state so every
rank converges.

The gate battery surfaced one real kernel bug, fixed before acceptance:
a missing `__syncthreads()` in the GDR prefill recurrent kernel's smem
publication — latent at cp=1, fired once NCCL interleaving perturbed the warp
schedule. Full post-mortem:
`docs/experience/errors/2026-08-16-gdr-prefill-smem-race.md`.

## Results (H20 pod, 27B FP8, BF16 KV, needle `738291`)

| Gate | Result |
|---|---|
| needle ladder, world=4 (attn_tp=2 × attn_cp=2) | 12/12 exact |
| needle ladder, world=2 (attn_tp=1, attn_cp=2, global-comm alias) | 4/4 exact |
| cp=1 control | exact, unchanged |
| differential self-check (CP arm vs full-chunk reference, 48 layers) | bit-zero after the barrier fix |
| 128K cold-prefill TTFT, cp=1 → cp=2 | 54.14s → 30.93s = **1.75×** (target ≥1.6×) |

## Limits

- cp=2 decode rate regressed at world=2 / 128K: 60 → 43 tok/s vs cp=1.
  Decode under CP is T3's scope (graph-captured merge over sharded KV); the
  regression is recorded here, not diagnosed.
- CP engages only when `cp>1 && dspark off && len >= cp*256`; short prompts
  stay on the cp=1 path byte-identically.
- attn_dp>1 is rejected at load under CP (the attn_tp sharding would
  double-count attention); guard, not support.

## Rule

CP compute-sharding must not touch the FFN row set: axes that shard over the
full tp world need identical rows on every rank, so slice only the
attn_tp-sharded interiors and gather rows before the next full-world op.
