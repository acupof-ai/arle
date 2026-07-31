# TP decoupled: TpContext to train::tensor_parallel — 2026-07-31

> Status: Deletion-refactor, byte-identical (rename + move, no math change).
> Model-agnostic TP core now mirrors CpContext/DpContext. Inference smoke green.

## Context

Of the 5 N-D parallel axes, CP was just made model-agnostic (`cp_causal_sdpa`,
`linear_attention_core_cp` in autograd; qwen35 calls one line). An audit for "is
each axis abstracted so the next model reuses it?" found TP was NOT: the TP core
lived inside `qwen35.rs` as `Qwen35TensorParallelConfig` + `divide_tp` +
`maybe_tp_all_reduce` — a misnomer, since rank/size/all-reduce are pure mesh
logic with nothing Qwen-specific. A second gated-delta / Mamba-hybrid model would
have re-implemented them.

## What worked — split by what actually reads the model config

First-principles cut: which TP logic touches `Qwen35Config`, and which is pure
mesh arithmetic?

- **Model-agnostic → new `train::tensor_parallel`** (`TpContext`): `rank`,
  `world_size`, `single`/`new`/`from_coord`/`is_enabled`, the divisibility rule
  `divide(value) -> Option<usize>`, and the collective wrapper
  `maybe_all_reduce`. Mirror of `CpContext`/`DpContext` — same mesh
  (`infer_topo::MultiAxisConfig`), a different sharded axis (weights, not seq).
- **Model-specific → stays in `qwen35.rs`** as a `Qwen35TpDims` trait impl on
  `TpContext`: `local_attention_heads`/`local_intermediate_size`/… and
  `full_attn_q_dim`/… — all read `Qwen35Config` fields (`num_attention_heads`,
  `head_dim`, `full_attn_gated`). These are "apply the generic rule to this
  model's dimensions"; forcing them up would leak `Qwen35Config` into the generic
  layer — wrong altitude. So the correct home is the model file, on a trait.

Because the model dims stay on a trait impl of `TpContext`, all 26 `tp.local_*(cfg)`
call sites are unchanged — the refactor is a rename (`Qwen35TensorParallelConfig`
→ `TpContext`) + a move, not a rewrite. `divide` returns `Option` so the model
maps `None` to its own `Qwen35Error::InvalidConfig` (exact original error
preserved); `maybe_all_reduce` returns `autograd::Result`, coerced at the 4 tail
call sites via `Ok(..?)`.

## Verification (local)

- `cargo test -p train --no-default-features --features no-cuda`: 197 pass (was
  194; +3 new `tensor_parallel` unit tests — single/divide/from_coord). The
  existing TP parity examples (`moe_tp_parity`, `a2_qwen35_tp_lora_fd`,
  `nd_parallel_parity`) are the byte-identity regression — updated to import
  `TpContext` from the new module.
- `cargo clippy -p train --no-default-features --features no-cuda --tests`: clean.
- Mac CUDA typecheck (`cuda,no-cuda`): `train` + `infer-api` green — backend
  isolation intact.
- **Inference smoke (the "推理能用吗" check):** rebuilt `arle` for Metal, served
  `models/Qwen3.5-0.8B-MLX-4bit`, `/v1/completions` on "The capital of France is"
  → " Paris, and the capital of the United States is Washington, D.C." — correct,
  coherent, no repetition, token counts sane. The refactor is train-only so
  infer-* is untouched; verified empirically, not assumed. (The 35B canonical MoE
  hit the Metal memory guard at 20.4 GiB free < 38 GiB required — the guard doing
  its job, not a regression; the small model exercises the same generate path.)

## Rule

"Abstract into the generic layer" cuts along config dependence, not along "looks
parallel." A TP context's rank/size/all-reduce is pure mesh math → generic
module, one per axis (CP/DP/TP mirror each other). Per-model shard *dimensions*
read the model config → stay in the model file, as a trait impl on the generic
context. A struct named `Qwen35*` that holds only `{rank, world_size}` is a
misnomer to fix, not a boundary to respect.
