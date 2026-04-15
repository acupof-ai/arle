# Metal continuous batching — varlen scaffolding (Phase 1)

## Context

2026-04-16. The Metal backend has had a live scheduler runtime since
`M0.2b` (2026-04-15) with chunked prefill, decode-priority interleave,
and same-length Qwen3 + Qwen3.5 batched decode. The remaining exit
blocker for the `M0.2` throughput milestone was **variable-length
decode batching** — until now, two requests could only share a decode
step if their `cache_len` and `kv_capacity` already matched, which
forces most live traffic back onto per-request decode.

The standard Apple-Silicon answer for varlen batching (surveyed across
mlx-lm, mlx_parallm, and waybarrios/vllm-mlx) is **left-padding with a
padding-aware additive causal mask** — the mlx-lm `BatchKVCache`
pattern. No PagedAttention, no custom Metal kernel, no new FFI
primitives beyond an optional mask parameter on the existing compiled
step. Phase 1 lands the scaffolding for that pattern and the
infrastructure pieces that variable length will need.

## What worked

1. **C++ bridge: optional additive mask on the compiled step.**
   `Qwen35CompiledModel` grew two fields
   (`current_attn_mask: mlx::core::array`, `current_has_attn_mask: bool`);
   `full_attn_step` branches — when the flag is set, it calls
   `fast::scaled_dot_product_attention(q, k_full, v_full, scale, "", current_attn_mask)`;
   otherwise it keeps the legacy `"causal"` / `""` string-mode path.
   Both `qwen35_compiled_step_batch` and `..._packed` take a new
   nullable `attn_mask: mlx_array*` parameter and reset the model's
   mask fields on both success and exception exit paths.
   (`crates/mlx-sys/src/mlx_qwen35_model.cpp:345`,
   `crates/mlx-sys/src/lib.rs:373`, `infer/src/backend/metal/qwen35.rs:396`.)

2. **Rust-side mask builder.**
   `super::mlx::build_varlen_decode_mask(left_padding, batch_cache_len)`
   produces a `[B, 1, 1, batch_cache_len + 1]` additive f32 mask where
   columns `[0, left_padding[b])` are `-inf` and the rest are `0.0`.
   MLX `fast::scaled_dot_product_attention` broadcasts the mask across
   the `n_heads` and `n_query` dims. Unit-tested in
   `build_varlen_decode_mask_marks_left_padding`.
   (`infer/src/backend/metal/mlx.rs:495`.)

3. **`Qwen35PackedDecodeBatch` refactor.** The cached packed batch now
   stores `batch_cache_len: i32` (shared column cursor) and
   `left_padding: Vec<i32>` (per-row pad). `retain_rows` slices the
   pad vector alongside the KV rows; a new `admit_rows` method appends
   new rows into an existing packed batch without a full rebuild (with
   the same prefix-preserving-grow invariant the runtime already uses
   for shrink). Helpers `round_up_kv_capacity`,
   `left_pad_kv_cache_row`, `strip_left_padding_from_packed_row`
   handle the per-row → packed and packed → per-row copies.
   `sync_qwen35_packed_decode_batch` strips the left pad when writing
   rows back out. `decode_qwen35_packed_batch` builds the mask (only
   when at least one row is padded) and threads it through the bridge.
   (`infer/src/backend/metal/request_state.rs:323-555`.)

4. **Runtime admit branch (same-length-gated).**
   `execute_qwen35_packed_decode_batch` now has a third branch between
   `retain_rows` (shrink) and `invalidate_qwen35_decode_batch_cache`
   (full rebuild). New helper `admit_row_indices` is a symmetric
   prefix-preserving grow detector; when it fires AND every new row's
   `qwen35_decode_cursor()` is **exactly equal to**
   `cached.batch.batch_cache_len()`, the runtime calls
   `cached.batch.admit_rows(...)` to grow in place. The `==` check is
   the Phase 1 safety gate — a new row with `cache_len < batch_cursor`
   would force `left_padding > 0` and hit the shared-RoPE correctness
   bug below. Phase 2 loosens the check to `<=` once per-row RoPE is
   wired through.
   (`infer/src/backend/metal/runtime.rs:1068-1110`,
   `runtime.rs:1196-1218`.)

## Status

**Same-length enforcement kept in place.** The `try_build_qwen35_packed_decode_batch`
admission check still rejects rows with mismatched `cache_len` /
`kv_capacity`. Variable-length throughput does not improve yet — the
plumbing is in place but the mask bridge is never exercised with a
nonzero `left_padding` in production.

**Why.** Qwen3.5's `full_attn_step` rotates Q/K via
`fast::rope(q, rotary_dim, false, rope_theta, 1.0f, cache_pos)` with a
single scalar `cache_pos = batch_cache_len`. For rows with
`left_pad > 0` the real logical position is
`batch_cache_len - left_pad`, so the shared scalar applies the wrong
positional encoding and attention output is garbage. The mask cannot
compensate — it only changes which cached K/V columns a query attends
to, it cannot re-rotate the query itself. See
[`docs/experience/errors/2026-04-16-metal-varlen-rope-blocker.md`](../errors/2026-04-16-metal-varlen-rope-blocker.md)
for the full analysis and the Phase 2 plan (per-row RoPE via MLX's
`fast::rope(..., array offset)` overload).

## Verification

```
cargo check -p infer --no-default-features --features metal,no-cuda   # green
cargo test  --release -p infer --no-default-features --features metal,no-cuda --lib
  # 308 passed, 0 failed, 7 ignored
cargo clippy -p infer --no-default-features --features metal,no-cuda -- -D warnings
  # 26 pre-existing errors on HEAD; diff is clippy-neutral (no new warnings)
```

End-to-end `e2e_qwen35` / serving bench was not run in this session —
no local model weights. Same-length numerical behavior is unchanged by
construction: the mask-materialization path takes a fast-path when all
`left_padding[i] == 0`, so `decode_qwen35_packed_batch` passes `None`
through to the compiled step and `full_attn_step` takes the legacy
`"causal"` / `""` branch verbatim.

## Rule

**Scaffolding ≠ shipping.** For a correctness-critical optimization
like varlen batching, land the infrastructure first and the
*enablement* switch last. The infrastructure can be merged and
reviewed while the enablement is gated off by a single explicit check;
merging an "almost right" version that produces wrong output under
concurrent load is strictly worse than shipping scaffolding that is
inert until Phase 2 flips the switch.

Corollary for reviewers: when auditing a varlen/masking change, walk
every position-dependent op in the compiled forward pass, not just
the attention op. Masks handle *which keys to attend to*, RoPE
handles *which position the query represents* — these are orthogonal
and both must be fixed for correctness.

## Phase 2 follow-ups

1. Pin down the MLX `fast::rope(..., array offset)` shape convention
   (Codex's `rope_dynamic_matches_rope_for_same_length_batch` test
   failed with `0 != 16.828888`; the equivalent Python call and shape
   need to be confirmed before the Rust wrapper can be trusted).
2. Thread `rope_offsets: *mut mlx_array` through
   `qwen35_compiled_step_batch_packed` alongside `attn_mask`. Store
   into a new `current_rope_offsets` field; branch `full_attn_step`.
3. Lift the same-length admission check in
   `try_build_qwen35_packed_decode_batch`.
4. Run `scripts/bench_guidellm.sh metal-cb-varlen` against the
   Qwen3.5-4B-MLX-4bit model and compare against the `M0.2` baseline
   in
   [`docs/plans/2026-04-15-metal-backend-acceptance-plan.md`](../../plans/2026-04-15-metal-backend-acceptance-plan.md).
5. Apply the same pattern to the Qwen3 pure-Rust `decode_qwen3_batch`
   path (`infer/src/backend/metal/request_state.rs:734`) — it also
   hardcodes a shared `cache_len` in its RoPE call.
