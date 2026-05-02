---
name: Qwen3 varlen decode follow-up
description: Qwen3 pure-Rust decode_qwen3_batch still requires same-length; the RoPE fix (rope_dynamic) landed but varlen left-pad + mask needs per-layer loop restructuring
type: project
---

Qwen3 batched decode (`decode_qwen3_batch` in `request_state.rs:984`)
and its gate (`MetalRequestState::decode_batch` at line 703-709) still
enforce same-length. The correctness fix (array-offset `rope_dynamic`
at lines 1052-1063) landed in `a6d4525`, so all batch rows now get
correct positional encoding. But actual varlen (mixed `cache_len`
within one batch step) is NOT enabled because the per-layer loop does
explicit `slice_update` + `slice` on per-row K/V caches and then
`concatenate_axis(&batch_k, 0)` — all rows must produce the same
seq-dim width.

**Why:** Adapting requires left-padding each per-row KV gather output
to `batch_cache_len + 1` before concat (using `left_pad_kv_cache_row`
with `valid_len = row_cache_len + 1`), building a varlen mask via
`build_varlen_decode_mask`, and calling
`scaled_dot_product_attention_masked` instead of the string-mode path.
The MetalKVPool gather branch needs the same treatment. ~100 lines of
careful per-layer loop edits.

**How to apply:** follow the Qwen3.5 packed-batch pattern — remove the
same-length gate, compute `batch_cache_len + left_padding`, left-pad,
mask, per-row rope_dynamic offsets (already in place). Start with the
non-pool path (simpler); the pool path is a stretch goal.
