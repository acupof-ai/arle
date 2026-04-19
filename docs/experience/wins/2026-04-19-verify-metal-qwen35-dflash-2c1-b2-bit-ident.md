# Metal DFlash Qwen3.5 Layer 2c.1 — B=2 bit-ident regression-check — 2026-04-19

> Layer 2c.1 of the Qwen3.5 DFlash batched-verify roadmap
> ([`docs/plans/metal-dflash-qwen35-verify-batch.md`](../../plans/metal-dflash-qwen35-verify-batch.md)).
> Promotes the test-local `verify_block_batched_b1` helper to a production
> `pub(super) fn verify_block_batched` on `CppQwen35Model` and adds the
> first real multi-row correctness test. Prerequisite for 2c.4
> (scheduler mixed-mode dispatch).

## Goal

**Type:** regression-check.
Does `CppQwen35Model::verify_block_batched` at `B=2` produce per-row logits
bit-identical (within `1e-3`) to two independent scalar `verify_block`
calls, when both rows share `prompt_len` (so `attn_mask` can be `None`)?

## Hypothesis

With identical `prompt_len` across rows, the additive `attn_mask` is
null, `cache_pos_arr=[prompt_len, prompt_len]`,
`rope_offsets=[prompt_len, prompt_len]`. Each row's packed slice of the
`[B, n_kv_heads, kv_cap, head_dim]` KV tensor and `[B, Hv, Dv, Dk]` GDR
state is independent of the other's — no cross-row attention, no shared
recurrent state. So batched verify must degenerate to two concurrent
scalar verifies, yielding element-wise `|batched - scalar| < 1e-3` on
`bf16 → f32` promotion.

If it does not match, the bug is in one of:
1. Per-row cache indexing inside the batched compiled graph
   (`qwen35_compiled_verify_block_batched`, `mlx_qwen35_model.cpp:1752`).
2. Per-row RoPE offset handling (row-specific `rope_offsets[b]` not
   plumbed through `mx::fast::rope`).
3. KV / GDR packing: `concatenate_axis(..., 0)` semantic mismatch vs
   the per-row `[1, ...]` layout produced by `MetalRecurrentState::new`
   and the prefill path.

## Parameters

- Production method under test:
  `CppQwen35Model::verify_block_batched`
  (`infer/src/backend/metal/qwen35.rs:725`).
- Regression test:
  `backend::metal::qwen35::tests::verify_block_batched_matches_independent_verify_block_for_b2`
  (`infer/src/backend/metal/qwen35.rs:1881`).
- Also kept: B=1 bit-ident test
  `verify_block_batched_matches_verify_block_for_b1` (Layer 2b gate,
  now calls the promoted prod method, not the test helper).
- Model: `mlx-community/Qwen3.5-4B-MLX-4bit` (via `$QWEN35_MODEL_PATH`).
- Row 0: prompt `[1,2,3,4]`, block tokens `[11,12]`.
- Row 1: prompt `[5,6,7,8]`, block tokens `[13,14]`.
- `block_size=2`, `batch_size=2`, `kv_cap = prompt_len + block_size + 4 = 10`.
- `attn_mask = None` (equal prompt_len → no padding needed).
- `rope_offsets = cache_pos_arr = [4, 4]`.
- Batched KV built by stacking each row's prefill output along axis 0
  (`concatenate_axis(&[row0.clone(), row1.clone()], 0)`); same for GDR +
  conv states.
- Tolerance: element-wise `|batched_row[b] - scalar_row[b]| < 1e-3` on
  `float32` promotion, checked per-row across all `vocab * block_size`
  logits.

Command:

```bash
QWEN35_MODEL_PATH="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3" \
cargo test --release --no-default-features --features metal -p infer \
  --lib -- --test-threads=1 verify_block_batched
```

## Environment

- Hardware: Apple M4 Max (40 GPU cores, ~400 GB/s UMA).
- macOS: 26.3.1 (build 25D771280a); Metal 3 via MLX 0.29.x
  (pinned in `crates/mlx-sys/CMakeLists.txt`).
- Commit: `06e313d` base + uncommitted `qwen35.rs` diff staging 2c.1.
  (Unrelated autograd WIP files in the tree are untouched by this work.)
- Feature set: `cargo build --release --no-default-features --features metal`.

## Results

```
running 2 tests
test backend::metal::qwen35::tests::verify_block_batched_matches_independent_verify_block_for_b2 ... ok
test backend::metal::qwen35::tests::verify_block_batched_matches_verify_block_for_b1 ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 327 filtered out; finished in 0.59s
```

- Assertion: for each row `b ∈ {0,1}`, all `vocab * block_size` logits
  match the corresponding scalar `verify_block` output within `1e-3`.
- Wall time: 0.59 s for both tests combined (model load dominates).
- **Δ vs baseline (Layer 2b B=1):** still zero numerical delta within
  tolerance. Now also verified at B=2 with distinct rows.
- No throughput number — 2c.1 is an API-promotion + correctness step,
  consumed by 2c.2/2c.3/2c.4. First throughput delta lands with 2c.4
  (scheduler lift at `open.len() >= 2`).

## Problems observed

- **Release-build dead-code warning**: `method verify_block_batched is
  never used`. Expected — the only in-tree call sites are the two tests,
  which are gated `#[cfg(test)]`. 2c.2+ (`qwen35_dflash_speculative_block`
  B>1 sibling and the `open>=2` scheduler lift) will supply a prod call
  site and eliminate the warning. **Will not commit this patch until the
  `-D warnings` clean gate is back** — deferred to the 2c.2/2c.4
  integration commit where the prod caller appears in the same diff.
- §6 watch items:
  - Warmup (§6.1): single invocation, not a sweep — N/A.
  - Thermal (§6.2): sub-second total — N/A.
  - Determinism (§6.7): argmax-free comparison on logit tensor;
    compiled graph at `temperature=0` is deterministic.

## Learnings

- **API signature surfaces scheduler needs.** Promoting to
  `pub(super) fn verify_block_batched(&self, tokens, batch_size,
  block_size, cache_pos_arr, packed_kv_caches, packed_gdr_states,
  attn_mask: Option<&MlxArray>, rope_offsets) -> Result<MlxArray>`
  with `attn_mask` as `Option<&MlxArray>` — not a bare `*mut mlx_array`
  — pushes the null-handling down to the one place that needs it
  (`attn_mask.map_or(std::ptr::null_mut(), MlxArray::as_raw)`), and
  keeps the scheduler-side call site symmetric between the
  B=1-no-pad and B>1-with-pad paths.
- **Stacking pattern confirmed.** `concatenate_axis(&[row0, row1], 0)`
  of per-row prefill states matches `MetalRecurrentState::new`'s
  `[B=1, Hv, Dv, Dk]` / `[B=1, conv_kernel-1, qkv_dim]` layout. This is
  what 2c.2 (batched draft forward) and 2c.4 (scheduler packed dispatch)
  will both emit — no per-row padding needed when `prompt_len` matches.
- **Equal-prompt-len is the easy path.** The real hard case is
  `prompt_len` skew across rows, which requires a non-null additive
  mask and (potentially) row-specific `rope_offsets`. That test lives
  with 2c.2 (draft-forward with per-row offsets) and is where a bug
  would most likely hide — the B=2 equal-prompt test is necessary but
  **not sufficient** for the full 2c.4 scheduler lift.
- **Stopping rule hit** (spec §7.3): variance N/A for correctness;
  hypothesis confirmed; §6 watch-list clean; no prior-snapshot delta
  (first 2c.1-specific entry). One run suffices.

## Follow-ups

- Land 2c.1 code + this doc in a single commit once 2c.2 or 2c.4
  supplies a prod caller (to clear the dead-code warning before
  `-D warnings` gate).
- Layer 2c.2: **batched draft forward** (`dflash_draft_forward` over
  `noise_embedding [B, block_size, hidden]`) with B=1 bit-ident
  acceptance gate. Per user directive, this is implemented immediately
  (not deferred).
- Layer 2c.3: swap scalar `mlx_tape_replay` for `mlx_tape_replay_varlen`
  inside `qwen35_rollback_to_accepted`
  (`infer/src/backend/metal/dflash.rs:1565`).
- Layer 2c.4: lift the `open.len() >= 2` permanent-downgrade at
  `infer/src/backend/metal/runtime.rs:1081` and wire
  `verify_block_batched` into `execute_qwen35_packed_decode_batch`
  with the mixed-mode (DFlash-row group + plain-decode group in
  parallel via MLX lazy eval) dispatch path.
- Add a B=2 `prompt_len`-skew test (rows 0/1 with different prompt
  lengths, non-null `attn_mask`) — gate for 2c.4 guidellm sweep.

## Cross-links

- Plan: [`docs/plans/metal-dflash-qwen35-verify-batch.md`](../../plans/metal-dflash-qwen35-verify-batch.md) §Layer 2c.
- Prior Layer 2b entry (B=1 bit-ident):
  [`2026-04-19-verify-metal-qwen35-dflash-2b-bit-ident.md`](2026-04-19-verify-metal-qwen35-dflash-2b-bit-ident.md).
- Layer 1 baseline (packed decode 162.9 tok/s at c=8 — 2c.4 target):
  [`2026-04-19-metal-qwen35-final-state.md`](2026-04-19-metal-qwen35-final-state.md).
- C++ FFI under test: `crates/mlx-sys/src/mlx_qwen35_model.cpp:1752`
  (`qwen35_compiled_verify_block_batched`).
