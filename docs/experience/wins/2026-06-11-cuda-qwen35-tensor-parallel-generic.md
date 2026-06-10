# Qwen3.5/3.6 CUDA tensor parallelism — TP as a model-generic abstraction

**Date:** 2026-06-11. **Backend:** CUDA, Qwen3.6-35B-A3B (target TP=2).
**Scope:** `infer-cuda` (qwen35/loader/moe/model/executor/shard_slice),
`cli` serve multiproc gate, `infer-api` kind classifier export.
**Status: pending-remote** — TP=2 runtime verification needs 2 free H20s
(pod currently fully occupied by a resident 8-rank DSv4 serve).

## Context

TP existed only as per-model wiring: dense Qwen3 consumed `TpRuntime` +
`shard_slice` + `infer-topo`, DSv4 rolled its own, and Qwen3.5/3.6 MoE was
hardwired single-GPU (zero `TpRuntime` references) — so the 67 GB 35B-A3B
checkpoint could not fit a 97 GB H20 next to its eager per-slot KV arenas,
and every new model would re-invent TP. `qwen35-spec` already shipped
per-tensor `Shard` contracts that nothing consumed.

## What Worked

Qwen3.5/3.6 now consumes the same shared machinery as dense
(`loader::build_tp_runtime`, now `pub(crate)`, with rank device-bind before
`ncclCommInitRank` mirroring the proven DSv4 flow):

- Full-attn: k/v column-sharded on kv-head boundaries; o_proj row-sharded +
  all-reduce; GATED q_proj sharded per-head as interleaved
  `[query(HD); gate(HD)]` row blocks (kernel evidence:
  `prefill_attention_hd256.cu:53/:151`) via a new
  `shard_slice::shard_head_blocks_column_parallel` (CPU-unit-tested).
- Linear-attn: fused in_proj_qkv `[q(Kh·Kd)|k(Kh·Kd)|v(Vh·Vd)]` block-sharded
  preserving the GQA `k_head = v_head·Kh/Vh` pairing
  (`gated_delta_rule.cu:46/:67-70`); conv1d rows share the SAME block slicer;
  z/b/a/dt_bias/A_log per-v-head; out_proj row + all-reduce; GDR/conv slot
  state sized to local heads; `norm_weight` proven `[Vd]` broadcast →
  replicated (`norm.cu:974-1019`, stale `[V*Vh]` doc fixed).
- MoE: EP via `ExpertSplit::new(experts, world, rank)` — each rank loads only
  its expert range, non-local routes scatter zero (`dsv4_route.cu:1365`
  sentinel), shared expert column/row-sharded, ONE all-reduce covers
  routed+shared partials before the residual add. Router replicated; router
  inputs are post-all-reduce ⇒ identical routing on every rank.
- Logits: replicated lm_head over post-all-reduce hidden ⇒ identical logits ⇒
  position-seeded sampling commits the same token on every rank.
- OPD surfaces (teacher logits, LoRA re-merge, weight offload) bail under
  `tp.is_collective()` — multi-rank OPD out of scope.
- Multi-rank serve spawn was DSv4-kind-gated (`serve.rs:95` `is_dsv4_model`);
  now `infer_api::cuda_model_takes_multiproc_serve` (Dsv4 | Qwen3Moe), so the
  existing coordinator/worker lockstep + NCCL-id mint serves Qwen3.5 TP for
  free.

Divisibility (kv_heads=2 ⇒ TP ∈ {1,2} for 35B-A3B) enforced with loud
ensures; TP=1 byte-identical by construction (all sharding behind
`is_single()`; `all_reduce_sum` no-op on `TpComm::Single`).

## Verification

- Host: `cargo check` (infer-cuda + infer-api, cuda,no-cuda) clean;
  `cargo test -p infer-cuda` 43/43 (4 new shard tests: fused-qkv block shard
  TP∈{2,4}, gated q/gate row pairing, single-GPU identity, indivisible-heads
  rejection); `infer-core` 33/33; clippy zero new warnings.
- **Pod TP=2 run (2026-06-11, GPUs 0,1, built at `bddf174c` + stacked-expert
  loader `b729f8e2`): mechanics LICENSED, numerics BLOCKED.**
  - LICENSED: 67 GB stacked-expert load with per-rank EP ranges (~85 s),
    multiproc spawn via the lifted gate, NCCL init, rank-1 lockstep driver,
    greedy outputs byte-identical across ranks and across same-config ×3,
    decode 49.9 tok/s (TP=2) vs 36.4 (TP=1) on the same shape (+37%, single
    run, contention-free box).
  - BLOCKED: output is degenerate — but **TP=1 with the same binary +
    checkpoint is equally degenerate**, so the defect is in the base rewrite
    Qwen35 forward / stacked loader, NOT the TP sharding (which takes zero
    branches at TP=1). Tracked in
    [`errors/2026-06-11-qwen35-cuda-rewrite-35b-degenerate-output.md`](../errors/2026-06-11-qwen35-cuda-rewrite-35b-degenerate-output.md).
  - The c=2 admission A/B (original E3) additionally needs the single-row
    executor limit lifted (a 2-row decode plan kills the engine thread by
    design today).

## Rule

- TP belongs to the shared substrate (TpRuntime + shard_slice + spec Shard
  contracts); a model arm only declares per-tensor layouts. Spec contracts
  that nothing consumes are bugs waiting — wire them or delete them.
- Sharding a fused/gated tensor needs kernel-level layout evidence (which row
  does the kernel read for head h?) before slicing — a contiguous slice
  across an interleaved block boundary is silent corruption.
