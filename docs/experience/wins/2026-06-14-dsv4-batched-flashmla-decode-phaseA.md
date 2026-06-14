# DSv4 batched FlashMLA decode — Phase A scaffolding (gated, byte-identical default)

## Context
The #1 concurrency lever ([plan](../../plans/dsv4-batched-flashmla-decode.md)):
the batched decode lane's attention is a per-row `for r in 0..n` loop
(`dsv4.rs:1872`), 58% of the c=8 step and 7.27× ∝c (nsys
[step-profile](2026-06-14-dsv4-config-mechanism-classification.md)). Batching it →
~2× aggregate decode tok/s (130.9→65.3 ms/step at c=8). This entry lands **Phase A**
(the precondition infra) per §0.1 land-and-isolate-on-a-clean-baseline.

## What landed (Phase A, gated `ARLE_DSV4_FLASHMLA_DECODE_BATCHED=1`, default OFF)
Model-wide batched scratch `Dsv4FlashMlaDecodeBatchScratch` (all N-sized buffers +
`block_table`/`slot_block_offsets`), allocated once in `Dsv4KvAdapter::new` under
the existing FlashMLA-alloc gate (`max_batch = num_slots`). Per-forward, for the
n>1 non-CSA lane: build `block_table[N]` (`flashmla_slot_first_block` per row) →
**one batched `build_indices(b=N)`** (the orphaned
`dsv4_flashmla_decode_build_indices_batched_raw`) → **per-forward `sched_meta(b=N)`**
(the cached-meta-is-b=1-only pitfall fix). The per-row attention kernel call is
**KEPT** in Phase A — so flag-ON still produces the per-row result; Phase A only
exercises the batched meta-build path. Phase B (`dsv4.rs` `// PHASE B:` marker)
swaps the per-row loop for `gather → sparse_decode_fwd_batched(b=N) → scatter`
(that kernel call is written + stride-verified, `#[allow(dead_code)]` until wired).

## Buffer enumeration (§0.1 — every mutated batched buffer + precondition)
`indices[max_batch×max_topk_unified]` (build_indices, pool-absolute) ·
`topk_length[max_batch]` · `start_pos[max_batch]` + `slot_block_offsets[max_batch]`
(H2D per forward) · `lse_accum[(num_sm_parts+max_batch)×h_q]` +
`o_accum[(num_sm_parts+max_batch)×h_q×head_dim]` — **the split dim folds b via
num_splits per `arle_flashmla_decode_shim.cu:202`, NOT a `b×accum_rows` axis**
(corrected mid-impl; the 5 accum strides stay single-row) · `sched_meta` recomputed
per forward · `num_splits[max_batch+1]` · `q_batched`/`out_batched`/`tp_gathered_q`
(Phase B gather/scatter, sized but unused).

## Verify
- **Mac CUDARC typecheck** (`infer-api`, `cuda,no-cuda`, on main): clean. infer-cuda
  + examples clean; no new dead-code warnings beyond the marked Phase-B function.
- **Default byte-identical**: flag OFF → the batched scratch is allocated but never
  touched; the n>1 default per-row path is unchanged; N=1 never reaches
  `forward_decode_batch_stream_impl`. B=1 42.0 ms/step + needle hold by construction.
- **Pod — PENDING-REMOTE**: (1) flag-ON Phase-A batched-meta path runs clean
  (no IMA, SW/HCA) at c≥2; (2) Phase B wired → needle ×3 (B=1 + c=8 self-consistency)
  + **c=8 aggregate decode tok/s RISES** (the acceptance bar from the long-ctx
  campaign). License-to-default-flip is Tier-2 (pod) per the understand-until-simple
  gate; not flipped here.

## Rule
- **Land the precondition infra gated + byte-identical-default first, verify on a
  clean baseline, then wire the behavior swap** (§0.1). Phase A is opt-in scaffolding
  with zero default-path change; the ~2× is Phase B + a pod perf license.
- The FlashMLA accum buffers are `[num_sm_parts+b, …]` (b folded via num_splits),
  not `[b×accum_rows, …]` — check the shim doc before sizing batched split-KV accums.
