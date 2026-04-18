# Metal DFlash Qwen3.5 verify ran as 16 × seq_len=1 calls instead of one seq_len=16 forward

## Status

**Resolved 2026-04-17** (this session).  Fix: new
`qwen35_compiled_verify_block` C++ FFI + `CppQwen35Model::verify_block`
Rust wrapper + single-forward rewrite of
`qwen35_dflash_speculative_block`.  Commits `981fa81` and `82a3dc9` on
`claude/research-optimization-solutions-E9KOq`.

## Context

User tweet (@bstnxbt, arXiv:2602.06036) prompted a review of our Metal
DFlash implementation.  On-device numbers from ckl's Apple Silicon:

- Plain Qwen3.5-4B decode: **74.8 tok/s** (single request).
- Qwen3.5-4B with DFlash speculative decode, `block_size = 16`: **15.2
  tok/s** — a ~5× regression, not the expected ≥1.8× speedup.

First hypothesis (wrong): low acceptance rate.  ckl corrected us:
acceptance rate is fine, the "low" number was a measurement-unit
artefact.  The real question was: *how is verify batched?*

## Root cause

`qwen35_dflash_speculative_block` (`infer/src/backend/metal/dflash.rs`)
ran the 16-token draft block through the target model as **sixteen
sequential `cpp_model.step(token, cache_pos, kv, gdr)` calls at
`seq_len = 1`** — despite the C++ tape kernel
(`gated_delta_tape_kernel`, `crates/mlx-sys/src/mlx_qwen35_model.cpp`)
supporting `T > 1` per launch since day one.

Cost per step call that should have been amortised over the block:

- `set_packed_kv` on every full-attention layer (16 full-attn layers
  for Qwen3.5-4B).
- MLX graph schedule + eval boundary.
- GDR tape drain FFI (`qwen35_read_and_clear_gdr_tapes`) +
  captured-hidden drain (`qwen35_get_captured_hidden`).
- One-token sample + `.item()` host-sync for posterior matching.

Multiply by 16 → verify cost dominated the block and wiped out the
speculative win.  The Qwen3 full-attention DFlash path already did the
right thing: one `qwen3_forward_with_hidden_states(block_tokens, …)`
call at `seq_len = block_size`.  The Qwen3.5 hybrid path was never
brought over.

## Fix

Two-layer refactor, both landed:

1. **C++ FFI** (`qwen35_compiled_verify_block`): thin wrapper around the
   existing prefill forward path that forces
   `current_last_logits_only = false` so all `block_size` positions
   emit logits, and honours the thread-local tape / capture flags to
   emit one tape per GDR layer of shape `[1, block_size, Hv, Dv]` +
   one hidden capture per capture layer of shape
   `[1, block_size, hidden_size]` in a single forward.  Mirrors the
   prefill call shape — no new MLX graph.

2. **Rust verify rewrite** (`qwen35_dflash_speculative_block`): replaced
   the 16-iteration loop + per-step accumulators with one
   `cpp_model.verify_block(…)` call + a single tape drain + single
   hidden drain.  Partial-accept rollback now slices each tape tensor
   along axis 1 to the accepted prefix (`slice_prefix_axis1`) and
   issues one `mlx_tape_replay(…, steps=accepted_inputs)` per GDR
   layer — same math, one launch instead of `accepted_inputs`.

Extracted three helpers (`drain_captured_hidden`,
`qwen35_rollback_to_accepted`, `qwen35_build_updated_target_hidden`) so
the main function reads linearly: draft → snapshot → verify → match →
(rollback if partial) → advance cache → build hidden → return.

## Rule

- **Before optimising acceptance rate on a speculative-decode path,
  audit the verify-batching shape first.**  A `seq_len=1` verify loop
  renders any acceptance rate irrelevant — fixed cost per position
  drowns the win.
- When a hybrid model (full-attn + linear-attn) forks from its
  full-attention counterpart's verify path, make sure the batched-verify
  shape is preserved across the fork.  The tape kernel supports `T > 1`;
  the FFI caller just has to use it.
- Prefer exposing the batched C++ primitive as a first-class FFI
  (`verify_block`) rather than hoping callers stitch together
  `step(seq_len=1)` N times — the latter looks the same on paper but
  pays N× the decode-pipeline tax.
