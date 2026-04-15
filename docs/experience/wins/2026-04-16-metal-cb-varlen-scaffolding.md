# Metal continuous batching — variable-length decode enabled + production correctness fix

## Context

2026-04-16. Metal had a live scheduler runtime since `M0.2b`
(2026-04-15) with chunked prefill, decode-priority interleave, and
same-length Qwen3 + Qwen3.5 batched decode. The remaining exit blocker
for the `M0.2` throughput milestone was **variable-length decode
batching** — without it most live traffic fell back to per-request
decode.

This wave landed variable-length continuous batching using the mlx-lm
`BatchKVCache` pattern (left-padding + additive causal mask) AND, as a
direct consequence of investigating the path, discovered a
**production correctness bug that had been silent since M0.2c/d**:
MLX 0.31.1's `fast::rope(..., int offset)` on a `[B, H, S=1, D]`
tensor with `B > 1` silently zeros out batch rows > 0. Every batched
Qwen3 / Qwen3.5 decode step since 2026-04-15 was producing
position-agnostic (and therefore wrong) output for rows beyond the
first. Production throughput benchmarks measured the right shape and
timing; the tokens were just wrong for rows > 0.

The fix for the bug and the enabler for varlen are the same primitive:
use MLX's array-offset `fast::rope` overload (`fast::rope(..., const
array& offset)`), which works correctly for both `B == 1` and `B > 1`,
and carries per-row logical positions needed for varlen anyway.

## What worked

1. **Array-offset RoPE on the C++ side.** `Qwen35CompiledModel` grew
   `current_rope_offsets: mlx::core::array` +
   `current_has_rope_offsets: bool`. `full_attn_step` branches —
   when set, `q = fast::rope(q, rotary_dim, false, rope_theta, 1.0f,
   current_rope_offsets)` (same for K); when not set, legacy scalar
   path (safe only for `B == 1`, i.e. prefill and single-request
   decode). Both `qwen35_compiled_step_batch` and `..._packed` gained
   a nullable `rope_offsets: mlx_array*` parameter and reset the model
   fields on both success and exception exit paths.
   (`crates/mlx-sys/src/mlx_qwen35_model.cpp:347`,
   `crates/mlx-sys/src/lib.rs:367`+`381`,
   `infer/src/backend/metal/qwen35.rs:399`+`454`.)

2. **Per-row offsets always built for batched decode.** In
   `decode_qwen35_packed_batch` (`request_state.rs:1396`) the Rust side
   builds an `int32[B]` vector
   `[batch_cache_len - left_padding[i] for i in 0..B]` and passes it
   through the bridge unconditionally. For same-length batches every
   entry equals `batch_cache_len` — still an array, still correct,
   still works around the scalar bug. For varlen the values diverge.
   `decode_qwen35_batch` (non-packed, `request_state.rs:1569`) does
   the same with a single shared `cache_len`. `decode_qwen3_batch`
   (`request_state.rs:1043`) switches from `super::mlx::rope(..., cache_len)`
   to `super::mlx::rope_dynamic(..., &rope_offsets)` for the same
   reason.

3. **Left-padding + additive mask for varlen.**
   `Qwen35PackedDecodeBatch` carries `batch_cache_len` + `left_padding`;
   `try_build_qwen35_packed_decode_batch` now computes
   `batch_cache_len = max(cache_lens)`,
   `left_padding[i] = batch_cache_len - cache_len[i]`,
   `target_kv_capacity = round_up_kv_capacity(batch_cache_len + 1)`,
   grows every row's KV to `target_kv_capacity` via
   `ensure_capacity`, and left-pads per-row KV via
   `left_pad_kv_cache_row` before `concatenate_axis(.., 0)`.
   `build_varlen_decode_mask` (`mlx.rs`) produces the
   `[B, 1, 1, key_len]` additive `-inf` mask at columns `[0,
   left_padding[b])`; only materialized when at least one row is
   actually padded. `sync_qwen35_packed_decode_batch` strips the left
   pad via `strip_left_padding_from_packed_row` when writing rows back
   into per-request caches. `retain_rows` slices
   `left_padding` alongside the KV rows.

4. **Runtime admit path restored to `<=`.**
   `execute_qwen35_packed_decode_batch` has three cache-update
   branches: `retain_rows` (shrink), `admit_rows` (prefix-preserving
   grow), and full `invalidate_qwen35_decode_batch_cache`. The admit
   pre-check is `cache_len <= batch_cursor` — a newly prefilled row
   with shorter prompt gets left-padded into the active batch in-place
   via `admit_rows`, which computes per-row `left_pad` and
   concatenates the pad-adjusted new row onto the packed KV. No
   expensive rebuild of the existing batch.
   (`runtime.rs:1068-1118`.)

5. **Rope-workaround pinned by unit test.** `mlx.rs:773` defines
   `rope_dynamic_works_on_b_gt_1_s_eq_1_and_matches_per_row_reference`:
   it constructs a `[B=2, H=3, S=1, D=4]` input, calls array-rope with
   `[5, 3]`, and verifies row 0 equals scalar-rope of the row-0 slice
   at offset 5 AND row 1 equals scalar-rope of the row-1 slice at
   offset 3 (B=1 scalar path works correctly; the bug is only on
   B>1). This is a regression gate — if a future MLX upgrade
   changes the array-offset semantics, this test catches it before the
   model sees garbage.

## How the "blocker" turned into a "fix"

The Phase 1 design Claude drafted (and drafted a detailed error entry
about) assumed the RoPE fix was hard because the first attempt at a
same-length equivalence test (array-rope with offsets `[5, 5]` vs
scalar-rope with offset `5` on B=2 S=1) showed `0 != 16.83` at index
16. Claude concluded the array-offset path was broken on MLX 0.31.1
and the fix was Phase 2 work.

Codex's code review caught the real story: **the `0` was the scalar
path's bug — it had been dropping batch row 1 all along.** The
`16.83` from the array path was the actual correct value. The test
was comparing a broken reference against a correct output, and
Claude misread the failure direction.

A 20-line Python probe against `mx.fast.rope` on the installed
MLX 0.31.1 confirmed it: B=1 scalar and B=1 array agree; B=2 array
matches per-row B=1 references; B=2 scalar zeros row 1. The bug is in
scalar-rope + `[B>1, H, S=1, D]` specifically (prefill `S > 1` and
offset=0 B=1 are fine). The fix is the array overload. The same fix
both unblocks varlen and repairs existing same-length batched decode.

## Verification

```
cargo check  -p infer --release --no-default-features --features metal,no-cuda   # green
cargo test   --release -p infer --no-default-features --features metal,no-cuda --lib
  # 309 passed, 0 failed, 7 ignored
cargo clippy -p infer --no-default-features --features metal,no-cuda -- -D warnings
  # 26 pre-existing errors on HEAD; diff is clippy-neutral (no new warnings)
```

End-to-end `e2e_qwen35` and the guidellm HTTP sweep were not run in
this session — no local model weights and no running `metal_serve`.
The new `rope_dynamic_works_on_b_gt_1_s_eq_1_and_matches_per_row_reference`
test is the strongest correctness signal that actually exercises MLX's
`fast::rope` path end-to-end. Once a Metal host with Qwen3.5-4B
weights is available, the expected wins are:

- **Correctness restored** for batch rows > 0 — visible as
  quality improvement on any same-length or varlen Qwen3/Qwen3.5
  serving benchmark where `concurrency > 1`.
- **Variable-length throughput** — Qwen3.5 `512/256 C=4` should
  materially improve past the `66 tok/s` `M0.2b` baseline reported in
  `docs/plans/2026-04-15-metal-backend-acceptance-plan.md`. The
  mlx-lm/mlx_parallm community numbers suggest ~2-3× headroom on
  similar shapes.

## Rule

**When a test comparing an alleged-bug path against a reference-path
fails, confirm which side is wrong before blaming the feature.** The
first `rope_dynamic` test produced `0 != 16.83` and Claude concluded
the feature was broken. The correct read was "one side is returning
zero for non-zero input — which side is that?" A two-minute Python
probe would have caught the MLX scalar-rope bug on day one and
collapsed the entire Phase 1 / Phase 2 split into a single commit.

**Corollary: silent numerical bugs on Apple Silicon.** MLX's lazy
evaluation and per-kernel dispatch means a miscompiled codepath can
return zeros or uniform values without any error surface. A
throughput benchmark that just measures `tokens_per_second` will
happily report a "win" against a path that generates wrong tokens.
Correctness regression tests must be bound to reference values, not
just "didn't crash."
