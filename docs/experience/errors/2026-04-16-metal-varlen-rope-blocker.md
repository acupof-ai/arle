# Metal variable-length decode: shared-`cache_pos` RoPE is wrong for left-padded rows

## Context

2026-04-16. Phase 1 of the Metal continuous batching work ported the
mlx-lm `BatchKVCache` pattern to `Qwen35PackedDecodeBatch`: left-pad the
short rows, build an additive causal mask zeroing columns
`[0, left_padding[b])`, send the mask through a new `attn_mask` parameter
on the `qwen35_compiled_step_batch_packed` FFI.

The **masking** half of that plan is correct — MLX `fast::scaled_dot_product_attention`
broadcasts the `[B, 1, 1, key_len]` mask across heads and the row's
padded prefix contributes no attention. Unit test
`backend::metal::mlx::tests::build_varlen_decode_mask_marks_left_padding`
covers the mask shape and values.

The scaffolding (`batch_cache_len` + `left_padding` fields on the packed
batch, `left_pad_kv_cache_row`, `strip_left_padding_from_packed_row`,
`admit_rows`, `admit_row_indices` in the runtime, the mask builder, the
C++ bridge mask parameter) all landed. But the **actual relaxation** in
`try_build_qwen35_packed_decode_batch` — accepting rows with mismatched
`cache_len` — had to be reverted before merge. This note explains why
and what Phase 2 needs to solve.

## Root cause

`crates/mlx-sys/src/mlx_qwen35_model.cpp` `Qwen35CompiledModel::full_attn_step`
applies RoPE with a **single shared scalar offset**:

```cpp
int cache_pos = m->current_cache_pos;  // = batch_cache_len
q = fast::rope(q, rotary_dim, false, rope_theta, 1.0f, cache_pos);
k = fast::rope(k, rotary_dim, false, rope_theta, 1.0f, cache_pos);
```

With same-length batches (`left_padding == [0, 0, …]`) the shared
`cache_pos` equals every row's own `cache_len`, so the rotation is
correct. With variable-length batches, a row whose `cache_len =
batch_cache_len - left_pad` should be rotated at **position
`batch_cache_len - left_pad`**, not at `batch_cache_len`. Rotating the
new Q and K for that row at the wrong position scrambles the positional
encoding — attention then produces garbage output for the row.

This isn't a mask issue and the mask can't compensate. The mask zeros
out which cached K/V positions a query attends to; it cannot re-rotate
the query itself.

## Fix (Phase 2)

MLX has a second `fast::rope` overload that takes `const array& offset`
instead of `int offset` (`mlx/include/mlx/fast.h:36`). The mlx-lm
`BatchKVCache` uses it: `self.offset = mx.array([-l for l in
left_padding])` — a rank-1 `int32` vector of length `B`. The per-layer
attention block then calls
`queries = self.rope(queries, offset=cache.offset)` on a `[B, n_heads,
L, head_dim]` query tensor, and MLX applies the per-row offset to each
batch element.

Porting this requires three things:

1. **Pin the offset-array shape convention.** Codex added
   `crates/mlx-sys/src/mlx_bridge.cpp::mlx_fast_rope_dynamic` and a
   Rust wrapper `super::mlx::rope_dynamic(x, dims, trad, base, scale,
   off: &MlxArray) -> MlxArray`. An initial equivalence test
   `rope_dynamic_matches_rope_for_same_length_batch` (input `[2, 3, 1,
   4]`, offsets `&[5, 5]` as `&[2]`) produced a divergence
   `0 != 16.828888` on MLX 0.31.1. The test was deleted and the
   investigation deferred — before trusting `rope_dynamic` in the hot
   path we need a small, isolated repro that confirms the exact shape
   MLX expects (suspected `[B]` but the failure mode suggests otherwise),
   and a known-correct reference value from a Python `mx.fast.rope`
   call.

2. **Plumb per-row offsets through the bridge.** Add an optional
   `rope_offsets: *mut mlx_array` parameter to
   `qwen35_compiled_step_batch_packed` alongside `attn_mask`, stash it
   into a new `current_rope_offsets: mlx::core::array` field on
   `Qwen35CompiledModel`, and branch `full_attn_step` to call
   `fast::rope(q, ..., current_rope_offsets)` when
   `current_has_rope_offsets` is set. Clear on success and exception
   paths, same as the mask.

3. **Re-enable the varlen admission path.** Remove the same-length
   check in `try_build_qwen35_packed_decode_batch` (see
   `infer/src/backend/metal/request_state.rs:1195-1221`) and let
   `left_padding[i] = batch_cache_len - state.driver.cache_len` take
   effect. The packed KV construction already uses
   `left_pad_kv_cache_row`; the mask path is already wired; only the
   guard needs to lift.

Also verify the Qwen3 pure-Rust `decode_qwen3_batch` path
(`request_state.rs:734`) — it too hardcodes a shared `cache_len` when
calling `super::mlx::rope(&q, ..., cache_len)`. Same fix shape (per-row
rope_dynamic), different call site.

## Rule

**If a compiled-graph step rotates Q/K with a single scalar position,
every row in that step must share that position.** For variable-length
continuous batching, the positional encoding is per-row state, not
batch-global state. The mask only changes which keys a query attends
to; it cannot retroactively re-RoPE the query itself.

A varlen plan that addresses the attention mask without also
addressing per-row RoPE will produce correct-looking shapes and wrong
numerical output — the kind of bug that only shows up under load and
under concurrent mixed-length traffic, not in single-request smoke
tests. Catch this by always walking the compiled attention path
line-by-line for any position-dependent op.
