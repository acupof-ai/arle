# Metal DFlash Qwen3.5 Layer 2c.2 — batched draft forward B=1 bit-ident — 2026-04-19

> Layer 2c.2 of the Qwen3.5 DFlash batched-verify roadmap
> ([`docs/plans/metal-dflash-qwen35-verify-batch.md`](../../plans/metal-dflash-qwen35-verify-batch.md)).
> Adds a **batched** draft forward (`dflash_draft_forward_batched`) alongside
> the scalar so Layer 2c.4's mixed-mode scheduler dispatch can run the draft
> step once per tick across multiple slots instead of serializing per-row.
> Correctness-only; first throughput delta lands with 2c.4.

## Goal

**Type:** regression-check.
Does `DFlashDraftCppModel::forward_batched` at `B=1` produce the same draft
hidden-state tensor as the existing scalar `DFlashDraftCppModel::forward`
(called via `dflash_draft_forward_cpp`)? Acceptance: element-wise
`|batched[0] - scalar| < 1e-3` on `float32` promotion.

## Hypothesis

With `B=1`:
- `reshape(q_raw, {B, seq, num_heads, head_dim})` collapses to the scalar
  `{1, seq, …}`.
- `fast::rope(..., q_offsets)` with `q_offsets = int32[1]` is exactly the
  scalar offset — MLX's rank-1 `offset` overload broadcasts `(offset[0] + s)`
  across the single row, identical to the scalar path.
- KV concat axis 2 (`[B, n_kv_heads, cache, head_dim]`) with `B=1` matches
  the scalar layout bit-for-bit.
- `concatenate({target_hidden_proj, normed_hidden}, 1)` (axis 1 = seq) at
  B=1 is equivalent to the scalar's `axis 0` concat over a rank-2 input,
  since the rank-3 `[1, seq, hidden]` tensor has the same linear memory as
  rank-2 `[seq, hidden]`.

If it does not match, the bug is in one of:
1. Per-row RoPE via `fast::rope` with rank-1 offset — regression from the
   scalar path that uses size-1 rank-1 offset (should be the same op, but
   trip-wire per memory `feedback_mlx_rope_layout.md` / `feedback_mlx_rope_axis.md`).
2. `reshape` axis math when B=1 (the implementation uses explicit B from
   `hidden_states.shape(0)` at graph-build time, not `-1`).
3. KV cache concat axis — must be 2 for `[B, n_kv_heads, cache, head_dim]`,
   not 1.

## Parameters

- Production C++ graph under test:
  `DFlashDraftModel::forward_batched_impl` in
  `crates/mlx-sys/src/mlx_dflash_draft_model.cpp` (lines 202–296 per agent
  report; verify against HEAD).
- Production FFI: `dflash_draft_forward_batched` at
  `crates/mlx-sys/src/mlx_dflash_draft_model.cpp` (lines 479–544).
- Rust FFI decl: `crates/mlx-sys/src/lib.rs` (lines 310–321).
- Rust wrapper: `DFlashDraftCppModel::forward_batched` at
  `infer/src/backend/metal/dflash.rs` (lines 670–708).
- Regression test:
  `backend::metal::dflash::tests::draft_forward_batched_matches_forward_for_b1`
  at `infer/src/backend/metal/dflash.rs:1939`.
- Model: `z-lab/Qwen3.5-4B-DFlash` (draft) paired with
  `mlx-community/Qwen3.5-4B-MLX-4bit` (target) via `$QWEN35_MODEL_PATH`.
- Test inputs:
  - `noise_embedding [3, hidden]` synthetic ramp `idx / 128.0`.
  - `target_hidden [2, target_hidden_width]` synthetic ramp `(idx - 17.0) / 256.0`.
- Scalar path: `dflash_draft_forward_cpp(cpp_model, noise_embedding,
  target_hidden, &mut state)` → baseline hidden.
- Batched path: `expand_dims` noise + target to `[1, seq, hidden]` /
  `[1, context_len, target_hidden_width]`; pass `q_offsets = k_offsets = int32[1] = [0]`;
  clone scalar state's per-layer `[1, n_kv_heads, cache, head_dim]` KV (already
  B=1-shaped) into the batched input.
- Tolerance: `|batched[0] - scalar| < 1e-3` on `float32` cast.

Command:

```bash
QWEN35_MODEL_PATH="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3" \
cargo test --release --no-default-features --features metal -p infer \
  --lib -- --test-threads=1 draft_forward_batched_matches_forward_for_b1
```

## Environment

- Hardware: Apple M4 Max (40 GPU cores, ~400 GB/s UMA).
- macOS: 26.3.1 (build 25D771280a); Metal 3 via MLX 0.29.x
  (pinned in `crates/mlx-sys/CMakeLists.txt`).
- Commit: `dde6fdd` base (CLAUDE.md delegation update) +
  uncommitted 2c.1 + 2c.2 diffs. Combined stat:
  `crates/mlx-sys/src/lib.rs +12`,
  `crates/mlx-sys/src/mlx_dflash_draft_model.cpp +181`,
  `infer/src/backend/metal/dflash.rs +176`.
- Feature set: `cargo build --release --no-default-features --features metal`.

## Results

```
running 1 test
test backend::metal::dflash::tests::draft_forward_batched_matches_forward_for_b1 ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 329 filtered out; finished in 0.24s
```

- Assertion: element-wise `|batched[0] - scalar| < 1e-3` over all
  `seq * hidden` elements of the draft hidden output.
- Observed `max_abs_delta = 0` — **bit-identical at B=1**, strictly
  better than the `< 1e-3` tolerance.
- Wall time: 0.24 s test runtime (agent reported ~1.13 s including model
  load).
- **Δ vs baseline:** zero numerical delta. No throughput number — 2c.2
  is FFI + graph plumbing, no prod caller yet.

## Problems observed

- **Two release-build dead-code warnings** now stacked:
  ```
  warning: method `forward_batched` is never used
  warning: method `verify_block_batched` is never used
  warning: `infer` (lib) generated 2 warnings
  ```
  Expected — both methods are consumed only by their correctness tests
  (`#[cfg(test)]`). 2c.4 (scheduler mixed-mode dispatch) provides the
  first prod callers for both. Commit strategy: hold the 2c.1 + 2c.2 code
  diffs uncommitted until the 2c.3 + 2c.4 bundle lands; commit all four
  layers together with the dead-code warnings retiring in the same diff.
- §6 watch items:
  - Warmup (§6.1): single test invocation; N/A.
  - Thermal (§6.2): sub-second total; N/A.
  - Determinism (§6.7): synthetic deterministic inputs + compiled graph at
    `temperature=0`; delta is exactly zero, no RNG involvement.

## Learnings

- **`fast::rope` rank-1 size-B offset worked as predicted.** Plan trip-wire
  note ([plan §Layer 2c trip wires](../../plans/metal-dflash-qwen35-verify-batch.md))
  had already resolved this via MLX upstream source-read: the array
  overload `fast::rope(x, dims, traditional, base, scale, offset)`
  validates `offset` as a 1-D tensor of length B and broadcasts
  `(offset[b] + s) * scale` over `arange(S)`. At B=1 this is exactly the
  scalar path; no split-rope-concat fallback was needed. The burn notes
  `feedback_mlx_rope_layout.md` (T = second-to-last axis) and
  `feedback_mlx_rope_axis.md` still applied: the graph transposes to
  `[B, heads, seq, d]` **before** rope, not after.
- **Explicit B in `reshape` > `-1` inference.** Agent chose
  `reshape(q_raw, {B, seq_len, num_heads, head_dim})` with B extracted
  from `hidden_states.shape(0)` at graph-build time (shapeless-compiled),
  not `-1`. No op quirks encountered. Deferring to `-1` would have worked
  but added one unprovable shape inference step to the graph.
- **Bit-identical at B=1 is a strong signal.** The scalar path uses rank-2
  inputs and the batched path uses rank-3 with B=1 — these are different
  MLX compiled graphs, yet the numeric outputs match to zero. This confirms
  (a) the rank-3 graph's reshape / concat / rope / SDPA sequence is
  semantically identical to the rank-2 path at B=1, and (b) no hidden
  fp-rounding difference sneaks in between compiled graphs. Strengthens
  confidence for the B=2 correctness gate (first landed in 2c.4).
- **Stopping rule hit** (spec §7.3): variance N/A for correctness;
  hypothesis confirmed with strictly-better-than-tolerance delta; §6 clean;
  no prior-snapshot delta (first 2c.2 entry). One run suffices.

## Follow-ups

- Layer 2c.3: swap scalar `mlx_tape_replay` for `mlx_tape_replay_varlen`
  inside `qwen35_rollback_to_accepted` at
  `infer/src/backend/metal/dflash.rs:1605` — signature becomes
  `accepted_inputs: &[i32]` per-row. Scope-map gathered from Explore
  subagent: tapes are rank-4 `[B, T_padded, Hv, Dv]` with batch axis 0,
  time axis 1; conv state stacking pattern at `dflash.rs:1639-1653` works
  unchanged for B>1 once qkv is pre-trimmed globally to
  `T_max = max(accepted_inputs[b])`.
- Layer 2c.4: lift the `open.len() >= 2` permanent-downgrade at
  `infer/src/backend/metal/runtime.rs:1081` and wire
  `forward_batched` + `verify_block_batched` into
  `execute_qwen35_packed_decode_batch` with mixed-mode dispatch (DFlash
  rows + plain rows queued as two parallel MLX subgraphs before
  `async_eval`). This is the first prod caller for both batched APIs —
  dead-code warnings retire here.
- B=2 bit-ident draft test (equivalent to 2c.1's B=2 verify test) lands
  with 2c.4, when the stacking infrastructure exists at the scheduler
  level.

## Cross-links

- Plan: [`docs/plans/metal-dflash-qwen35-verify-batch.md`](../../plans/metal-dflash-qwen35-verify-batch.md) §Layer 2c.
- Prior 2c.1 entry (verify_block_batched B=2 bit-ident):
  [`2026-04-19-verify-metal-qwen35-dflash-2c1-b2-bit-ident.md`](2026-04-19-verify-metal-qwen35-dflash-2c1-b2-bit-ident.md).
- Prior 2b entry (verify_block_batched B=1 bit-ident):
  [`2026-04-19-verify-metal-qwen35-dflash-2b-bit-ident.md`](2026-04-19-verify-metal-qwen35-dflash-2b-bit-ident.md).
- Layer 1 baseline (packed decode 162.9 tok/s at c=8 — 2c.4 throughput
  target):
  [`2026-04-19-metal-qwen35-final-state.md`](2026-04-19-metal-qwen35-final-state.md).
- C++ FFI: `crates/mlx-sys/src/mlx_dflash_draft_model.cpp`
  `dflash_draft_forward_batched`.
