# Metal variable-length decode RoPE — resolved same session, documented so the miss is visible

## Status

**Resolved 2026-04-16, same session as the initial write-up.** This
note is retained as a retrospective — the "blocker" framing was wrong,
and the reason it was wrong is worth remembering.

## Context

2026-04-16. Phase 1 of Metal continuous batching landed the mlx-lm
`BatchKVCache` pattern for Qwen3.5 packed decode (left-padding +
additive causal mask). Initial write-up of this file concluded:

> Fixing per-row RoPE requires landing per-row `fast::rope(..., array
> offset)`. A first test (`rope_dynamic_matches_rope_for_same_length_batch`)
> comparing the array-offset path against scalar-offset on a same-length
> batch produced `0 != 16.828888` — the array-offset shape convention
> was unclear. Deferred to Phase 2.

That conclusion was wrong. Both the test failure AND the conclusion.

## Root cause of the miss

The failing test pitted `rope(x, scalar_offset=5)` against
`rope_dynamic(x, array_offset=[5, 5])` on a `[2, 3, 1, 4]` input and
expected element-wise equivalence. The first mismatched element had
`lhs = 0, rhs = 16.828888`. Claude read that as "array-offset is
broken" and deferred.

Codex's code review caught it: look at which side is zero and which
side is 16.83. A 20-line Python probe would have answered it:

```python
x = mx.arange(24, dtype=mx.float32).reshape(2, 3, 1, 4)
scalar = fast.rope(x, 4, traditional=False, base=10000.0, scale=1.0, offset=5)
array  = fast.rope(x, 4, traditional=False, base=10000.0, scale=1.0, offset=mx.array([5, 5], dtype=mx.int32))
print(scalar.flatten().tolist()[12:24])  # all zeros
print(array.flatten().tolist()[12:24])   # correct rotated values
```

**MLX 0.31.1's `fast::rope(..., int offset)` silently zeros out batch
rows > 0 on `[B, H, S=1, D]` input.** Prefill (`S > 1`) and `B == 1`
decode are both fine. The bug fires exactly on our compiled
batched-decode shape. Repros on:

- `[2, 3, 1, 4]` — toy shape
- `[2, 16, 1, 128]` — Qwen3.5 production shape

The test was comparing a broken reference against a correct output.
The array-offset path was right all along.

## Impact in the tree

- **Production same-length Qwen3/Qwen3.5 batched decode has been
  silently wrong for rows > 0 since M0.2c/d landed** (2026-04-15).
  Batched attention for any row beyond the first was computed with
  zero-rotated Q and K — i.e. position-agnostic. Generated tokens for
  those rows were not NaN or obviously broken, just positionally
  confused.
- Throughput benchmarks (`guidellm`, `bench_throughput_sweep.py`)
  measured tokens/sec, not correctness, so the regression was
  invisible to the serving dashboards.
- The initial Phase 1 patch preserved the bug (and added a new
  scaffold to work around the wrong conclusion about it).

## Fix

Always route batched-decode RoPE through the array-offset overload:

```cpp
// crates/mlx-sys/src/mlx_qwen35_model.cpp (full_attn_step)
if (current_has_rope_offsets) {
    q = fast::rope(q, rotary_dim, false, rope_theta, 1.0f, current_rope_offsets);
    k = fast::rope(k, rotary_dim, false, rope_theta, 1.0f, current_rope_offsets);
} else {
    // scalar path — only safe for B == 1 (prefill, single-request decode)
    q = fast::rope(q, rotary_dim, false, rope_theta, 1.0f, cache_pos);
    k = fast::rope(k, rotary_dim, false, rope_theta, 1.0f, cache_pos);
}
```

On the Rust side, `decode_qwen35_packed_batch`,
`decode_qwen35_batch`, and `decode_qwen3_batch` always build an
`int32[B]` per-row offsets array (values differ when varlen, are
uniform when same-length, still go through the array path either way
to stay correct for `B > 1`).

The fix is pinned by
`backend::metal::mlx::tests::rope_dynamic_works_on_b_gt_1_s_eq_1_and_matches_per_row_reference`,
which verifies the B=2 array-rope path matches per-row B=1 scalar-rope
references at different offsets.

## Rules

1. **When a test comparing a new path against a reference path fails,
   confirm which side is wrong before assuming the new path is
   broken.** A failing element `lhs = 0, rhs = 16.83` could mean lhs
   is broken (returning zero for non-zero input) or rhs is broken
   (returning garbage). Ask which is the reference and which is the
   novel path, then run the novel path against an INDEPENDENT
   reference — not the one you're already suspecting.

2. **MLX lazy eval + per-kernel dispatch means silent numerical bugs
   are a real failure mode.** A miscompiled code path can return zeros
   or uniform values without any error. Correctness regressions must
   be bound to reference values, not "didn't crash" or "matches
   something nearby." Throughput benchmarks measure throughput, not
   correctness — they cannot catch this class of bug.

3. **For any position-dependent op in a compiled forward pass (RoPE,
   ALiBi, sinusoidal embeddings, etc.), verify the B > 1 case against
   B = 1 per-row references, not just against a scalar-offset call on
   the same B > 1 tensor.** The scalar-offset call may itself be the
   broken codepath.

4. **Delegate the first-opinion pass on a correctness-sensitive diff
   to a separate reviewer.** Claude's own self-review missed the
   scalar-rope bug for the same reason Claude wrote the bug in the
   first place — "rope_dynamic is the new thing, so rope_dynamic is
   the suspect." Codex's review asked "is there a trivial fix Claude
   missed?" specifically to guard against that bias, and the answer
   was yes.
