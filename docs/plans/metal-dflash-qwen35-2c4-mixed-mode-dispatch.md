# Layer 2c.4 Implementation Plan — Metal DFlash Qwen3.5 Mixed-Mode Dispatch

> Companion to [`metal-dflash-qwen35-verify-batch.md`](metal-dflash-qwen35-verify-batch.md)
> §Layer 2c. Designed 2026-04-19 by Plan subagent; all architectural claims
> grounded in file:line refs against HEAD at `49df0ec`.

## Goal

Lift the `open.len() >= 2` permanent DFlash downgrade so DFlash rows
survive into multi-row scheduler ticks, then run DFlash verify and
plain-decode as two lazily-built MLX subgraphs that are co-scheduled via
a single `mlx_async_eval`, wiring in the three already-landed 2c.1/2c.2/2c.3
primitives to retire all three dead-code warnings together.

## Files Touched

| Path | Function / region | Change |
|------|-------------------|--------|
| `infer/src/backend/metal/runtime.rs` | `execute_decode_batch` (:1044–1162), downgrade at :1081–1087, partition at :1089–1098 | Replace hard downgrade with config-gated opt-in (`metal_dflash_concurrency_off: bool`, default `false`). Partition yields three buckets: `dflash_rows`, `plain_rows`, `stale_rows` (rows whose `target_hidden` is not yet captured — warm-up single-row tick). Both DFlash + plain non-empty → `execute_qwen35_mixed_mode_batch(...)`; only DFlash → `execute_qwen35_dflash_packed_batch(...)`; only plain → existing `execute_qwen35_packed_decode_batch`. Stale rows still go through `execute_decode_single`. |
| `infer/src/backend/metal/runtime.rs` | New `execute_qwen35_mixed_mode_batch` + `execute_qwen35_dflash_packed_batch` | Top-level dispatch that borrows `&mut [(RequestId, ActiveMetalRequest)]` slices per bucket, calls `MetalRequestState::try_build_qwen35_dflash_verify_batch` / `try_decode_…` / `sync_…`, and coordinates the two-subgraph `async_eval`. Invalidate plain-decode cache on membership change (rebuild cheap vs race-free). |
| `infer/src/backend/metal/request_state.rs` | New `pub(crate) struct Qwen35DflashVerifyBatch<'a>` near `Qwen35PackedDecodeBatch` (:326–337) | Holds per-tick stacked target KV (`packed_kv_flat: Vec<MlxArray>`), packed GDR state (`packed_gdr_flat`), packed draft KV (2×num_draft_layers flat), per-row `left_padding`, `batch_cache_len`, `accepted_inputs_snapshot` (filled post-verify). |
| `infer/src/backend/metal/request_state.rs` | New `try_build_qwen35_dflash_verify_batch` / `try_decode_qwen35_dflash_verify_batch` / `sync_qwen35_dflash_verify_batch` (mirror the plain-decode trio at :856–928) | Build: stack per-row target `kv_flat` (`[1, n_kv, cache_cap, head_dim]`) with left-pad via `left_pad_kv_cache_row` (:1363) → `[B, n_kv, cache_cap, head_dim]`; stack per-row `gdr_flat` (no pad, recurrent) via `concatenate_axis(..., 0)`; stack draft `k_caches` / `v_caches` after normalizing capacity (`ContiguousKvState::ensure_capacity`, :786); build `cache_pos_arr[B]`, `rope_offsets[B]`, `[B, 1, block_size, key_len]` additive mask. Decode: call `qwen35_dflash_speculative_block_batched` in `dflash.rs`. Sync: per-row unstack via `slice_row` (:1433) + `strip_left_padding_from_packed_row` (:1435). |
| `infer/src/backend/metal/dflash.rs` | New `qwen35_dflash_speculative_block_batched` next to `qwen35_dflash_speculative_block` (:1806–2016) | Batched analogue — sequence enumerated in §Design below. |
| `infer/src/backend/metal/dflash.rs` | Generalize `qwen35_build_updated_target_hidden` (:1778) | Accept `batch_size: i32` + `accepted_inputs: &[i32]` instead of scalar; return `Vec<MlxArray>`. B=1 caller passes `&[accepted as i32]` and takes `[0]`. |
| `infer/src/backend/metal/dflash.rs` | Retire `#[allow(dead_code)]` on `forward_batched`, `qwen35_rollback_to_accepted_varlen`, `verify_block_batched` | All three now have prod callers. |
| `infer/src/backend/metal/mlx.rs` | New `build_varlen_verify_mask(left_padding: &[i32], block_size: i32, batch_cache_len: i32) -> MlxArray` beside `build_varlen_decode_mask` (:521) | Shape `[B, 1, block_size, key_len]`, additive bf16. Cell `-inf` iff `k < left_padding[b]` OR `(k − batch_cache_len) > q` within the block window. |
| `infer/src/backend/metal/AGENTS.md` | Append mixed-mode dispatch invariant | Docs commit. |

## Design

### State stacking pattern

**Target KV caches** stack like plain decode (`request_state.rs:1352–1371`):
per layer `l`, `left_pad_kv_cache_row(cpp_state.kv_flat[l], pad, cache_len, target_kv_capacity)`
then `concatenate_axis(&per_row, 0)`. `pad = batch_cache_len − cpp_state.cache_len`
per row. In mixed-mode, `batch_cache_len = max(DFlash rows, plain rows)` so both
buckets share the same column cursor. DFlash bucket expands capacity to
`batch_cache_len + block_size` pre-stack (analogous to single-row
`ensure_capacity` at `:2925`).

**GDR state** stacks without padding (recurrent, not time-series):
`concatenate_axis(&[row_0.gdr_flat[g], row_1.gdr_flat[g], …], 0)`.
Shape goes `[1, Hv, Dv, Dk] → [B, Hv, Dv, Dk]`. Conv state identical:
`[1, conv_kernel−1, qkv_cols] → [B, ...]`. This is exactly the shape
2c.3's varlen rollback expects (`dflash.rs:1734–1764`).

**Draft KV caches** (`ContiguousKvState`, `dflash.rs:714–784`):
stack after `ensure_capacity(max_required)` across rows so trailing dims
match. No left-padding: each draft forward processes exactly `block_size`
new positions starting at that row's own `state.len`, so
`q_offsets[b] = state[b].len`, `k_offsets[b] = 0` — the 2c.2 per-row
offset FFI handles it.

**Unstacking post-verify**: per-row
`slice(packed, &[b, 0, 0, 0], &[b+1, …], &[1, 1, 1, 1])` (pattern at
`request_state.rs:1433`) into owning `cpp_state.kv_flat[l]` /
`gdr_flat[g]`. Target KVs go through `strip_left_padding_from_packed_row`
(:1435).

**Left-padding + RoPE**: `rope_offsets[b] = cache_len[b]` (DFlash writes
start fresh at each row's own cache end). `cache_pos_arr[b] = cache_len[b]`
(fed to `verify_block_batched` at `qwen35.rs:754` unchanged). Mask columns
`0..left_padding[b]` stay `-inf`.

**`build_varlen_verify_mask`**:
- Shape `[B, 1, block_size, key_len]`, `key_len = batch_cache_len + block_size`.
- Cell `[b, 0, q, k]` = `-inf` iff `k < left_padding[b]` (left-pad) OR
  `k >= batch_cache_len AND (k − batch_cache_len) > q` (causal within block).
- bf16 cast per `build_varlen_decode_mask:547–550`.

### Two-subgraph `async_eval` coordination

**Verified from code**: `mlx_async_eval` at `crates/mlx-sys/src/mlx_bridge.cpp:1067`,
FFI at `lib.rs:715`, Rust wrapper at `mlx.rs:593–599`. Calls MLX
`async_eval(std::vector<array>)` at `vendor/mlx/mlx/transforms.cpp:296–308`,
which invokes `eval_impl(..., async=true)` — same scheduler as `eval`
minus the terminal `.wait()`. Arrays build lazy compute graph by default,
so **building both subgraphs before a single `async_eval(combined_outputs)`
call is sufficient**: `eval_impl` walks the union, schedules per stream,
fans out then joins (`transforms.cpp:280–291` `open_streams` loop).

Tick sequence:

1. Build plain-decode subgraph (returns `logits_plain: MlxArray`, mutates
   `batch.packed_kv_flat`/`packed_gdr_flat`) — lazy, no eval. **Split
   `decode_qwen35_packed_batch` into `_build` (lazy) + `_finalize`
   (sampling + sync + record)**; today's internal `async_eval` at
   `request_state.rs:1558` moves to the `_finalize` phase.
2. Build DFlash verify subgraph (returns `logits_verify`, captured hiddens,
   tapes) — lazy.
3. Concatenate all lazy outputs into `Vec<&MlxArray>`, call
   `mlx::async_eval(&combined)`. MLX schedules union graph.
4. Later `.item_i32()` / `as_slice_i32()` on sampled tokens block only on
   specific outputs.

**MLX parallelizes streams inside one eval** per `transforms.cpp:280–291`.
If Xcode capture shows serial (both buckets on one Metal stream), fallback:
merge DFlash verify + plain decode into one batched verify where plain rows
use `block_size=1` stub draft (the auto-enroll route from 2c plan :116).

### Rollback wire-up

Per-row `accepted_inputs[b]` computed post-verify. B=1 path at
`dflash.rs:1917–1926`:
- Zip `block_tokens[1..]` with `posterior_tokens[..block_size−1]`, count
  leading True.
- `accepted = matched + 1`.

Batched: for row `b`,
- `block_logits_2d[b] = reshape(slice(logits, [b, 0, 0], [b+1, block_size, V]), [block_size, V])`.
- `posterior_tokens[b] = sample_rows(block_logits_2d[b], params[b])`.
- `matched[b] = count_leading_match(block_tokens[b][1..], posterior_tokens[b][..block_size−1])`.
- `accepted_inputs[b] = matched[b] + 1` (i32, matches 2c.3:1679).

Then:
```rust
if accepted_inputs.iter().any(|&k| k < block_size_i32) {
    qwen35_rollback_to_accepted_varlen(
        &mut target_gdr_flat_packed,
        &gdr_snapshot_packed,
        &tapes_packed,
        &accepted_inputs,
    )?;
}
```

Varlen rollback handles identical-`accepted_inputs` efficiently (pre-slice
to `T_padded = max(accepted_inputs)` at `:1690`, single batched
`mlx_tape_replay_varlen` at `:1714`). Conv update loops per row
(`:1734–1764`) — acceptable since tick concurrency ≤ 16.

### Draft cache batching

Same as target cache, minus left-padding. Each draft forward produces
exactly `block_size` new positions starting at row's own `state.len`, so
the 2c.2 per-row offset FFI handles varlen across blocks naturally
(`q_offsets[b] = state[b].len`, `k_offsets[b] = 0`).

Normalize draft capacity before stacking:
`state.ensure_capacity(target_cap)` with `target_cap = max(row.state.len + block_size)`.
Stack, forward, unstack, per-row `state.trim(block_size)` + `apply_window`
— identical to single-row bookkeeping at `dflash.rs:1845–1846` but loop
over `b`. **Do not re-stack post-window-apply** (next tick re-stacks
fresh).

### qwen35_dflash_speculative_block_batched — full sequence

1. Pack `block_tokens[b][0] = current_token[b]; rest = mask_token_id` → `[B, block_size]` int32.
2. Pack `noise_embedding` via stack axis 0 → `[B, block_size, hidden]`; pack `target_hidden` → `[B, hidden]`.
3. Call `DFlashDraftCppModel::forward_batched` (:670–711) with per-row `q_offsets`/`k_offsets`; mask `None` (equal-length blocks within single forward).
4. Unstack hidden via `slice_axis0`; `linear(draft_block_hidden[b], lm_head)` + `sample_rows` → `drafted_suffix[b]`.
5. Snapshot stacked GDR state: `gdr_snapshot = target_gdr_flat.clone()`.
6. Enable tape mode + capture layers on `cpp_model` once.
7. Call `CppQwen35Model::verify_block_batched` (:725–784) with packed tokens, `cache_pos_arr`, packed KV/GDR, additive mask, `rope_offsets`.
8. Unstack logits `[B, block_size, V]`, per-row match → `accepted_inputs: Vec<i32>`.
9. If `any(accepted < block_size)`: `qwen35_rollback_to_accepted_varlen` with packed state + tapes + `&accepted_inputs`.
10. Per-row `updated_target_hidden[b]` from B-dim captured hiddens — generalized `qwen35_build_updated_target_hidden` slices axis 0 per row then axis 1 to `accepted_inputs[b]`.
11. Per-row `target_cache_len[b] += accepted_inputs[b]`.
12. Per-row draft `trim(block_size)` + `apply_window` (:1846).

## Tests

**Test 1 — `dflash_qwen35_verify_batched_matches_two_single_row_runs`** (`dflash.rs`):
- Two synthetic DFlash rows (distinct prompts, distinct blocks). Batched
  `qwen35_dflash_speculative_block_batched` → per-row outputs **bit-identical**
  to two sequential `qwen35_dflash_speculative_block` calls. Predicate:
  `max_abs_delta(batched_row_b.updated_target_hidden, scalar_row_b.updated_target_hidden) == 0.0`
  for `b ∈ {0, 1}`, identical `accepted_tokens[b]` `Vec<u32>`.
- Gate: `QWEN35_MODEL_PATH` env var + `metal_test_guard()` (same as 2c.3
  test at `dflash.rs:2172`).

**Test 2 — `mixed_mode_dispatch_b2_matches_single_row_paths`** (`request_state.rs`
or new integration file):
- One DFlash row + one plain-decode row through `execute_qwen35_mixed_mode_batch`
  → per-row sampled tokens bit-identical to separate single-row runs.
  Predicate: `sampled[dflash_idx] == scalar_dflash` AND
  `sampled[plain_idx] == scalar_plain`.
- Same env + guard.

Both under `cargo test --release --no-default-features --features metal -- --test-threads=1`.

## Risks

- **MLX compile-graph capture with varying B.** Risk: C++ model caches
  compiled graphs keyed on shapes. B=2 then B=3 recompiles every tick.
  Mitigation: verify 2c.2 compile key handling
  (`mlx_qwen35_model.cpp:1752+`). If recompiles observed, round B to
  power of two (2/4/8/16), pad with dummy rows using `mask_token_id`,
  discard output. Budget one diagnosis loop.
- **Attention mask construction.** Risk: off-by-one in causal region when
  `block_size` rows stack on varying `batch_cache_len − left_padding[b]`
  pre-writes. Mitigation: unit test `build_varlen_verify_mask` against
  hand-constructed B=2, block_size=4, cache_lens=[3, 5].
- **Conv state left-padding.** Risk: 2c.3 rollback assumes conv rows align
  with GDR rows per axis 0 — revalidate under B=2 in Test 1.
- **`async_eval` parallel vs serial.** Verified in code that one
  `async_eval` call → `eval_impl(..., true)` at `transforms.cpp:307` fans
  across streams. Residual risk: Metal single-stream serialization.
  Mitigation: Xcode Metal capture post-landing; if serial, fall back to
  stub-draft union-subgraph route.
- **Downgrade fallback mid-tick.** Keep build step fallible (`try_build_…`).
  Packed tensors are refcounted clones — original per-row state intact
  until sync step. On error, `cancel_detached_request` (`runtime.rs:1130`)
  affected rows only; surviving rows untouched (pattern at :1125–1133).
- **Dead-code warning retirement.** Strip all three `#[allow(dead_code)]`
  in same commit; `cargo clippy -- -D warnings` per CLAUDE.md §Verify.
- **Plain-decode cache invalidation.** `CachedQwen35DecodeBatch`
  (`runtime.rs:198`) shapes for plain-only rows; DFlash rows join/leave
  triggers rebuild. Fine for v1 (going 0% → full mixed-mode); revisit if
  profiling shows dominant.
- **`target_hidden` first-tick race.** Row post-prefill without captured
  `target_hidden` is **not DFlash-batchable** — route to `execute_decode_single`
  (stale bucket). Document in partition comment.

## Bench Spec

**Baseline**: `docs/experience/wins/2026-04-19-verify-metal-qwen35-dflash-2b-bit-ident.md` —
Qwen3.5-4B c=8 packed decode 162.9 tok/s (plain, DFlash downgrade active).
Confirm before bench; use most recent matching snapshot if diverged.

**Command** (Apple M4 Max per CLAUDE.md §Build & run):
```bash
cargo build --release --no-default-features --features metal

target/release/metal_serve \
    --model /path/to/Qwen3.5-4B-MLX-4bit \
    --dflash-draft-model z-lab/Qwen3.5-4B-DFlash \
    --max-num-seqs 16 \
    --port 8000 &

scripts/bench_guidellm.sh qwen35-dflash-2c4-mixed-mode
```

Sweep concurrency 1/2/4/8/16. Acceptance:
- **c=1**: ±5% of pre-2c.4 DFlash single-row (no regression — B=1 path unchanged).
- **c=8**: **≥ 2.00× 162.9 tok/s = 325.8 tok/s** (Plan §Layer 2c acceptance).
- **c=16**: ≥ 1.5× c=8 (stretch; documented, not gate).

Auto-iterate per `docs/bench-and-trace-spec.md` §7. If c=8 within noise
(<1.3×) after one run: Xcode Metal capture of mixed-mode tick to confirm
subgraph parallelism. Stop at (a) c=8 ≥ 2× with hypothesis held, or
(b) root cause named in `docs/experience/errors/`.

Snapshot: `docs/experience/wins/2026-04-XX-bench-guidellm-qwen35-dflash-2c4-mixed-mode.md`
using `TEMPLATE-bench-guidellm.md`. Cite 2b bit-ident entry as Δ baseline.

## Commit Strategy

**Code commit** — `feat(metal,qwen35): batched DFlash verify + mixed-mode dispatch (2c.1–2c.4)`:
- 2c.1 (`verify_block_batched`) + 2c.2 (`forward_batched` + mask) + 2c.3
  (`rollback_varlen`) + 2c.4 (mixed-mode dispatch) land together.
- Body cites `docs/plans/metal-dflash-qwen35-verify-batch.md` Layer 2 entries.
- Bench entry stub marked `pending-remote` if bench hasn't run locally.

**Docs commit** — `docs(metal,qwen35): plans + AGENTS for 2c.4 mixed-mode dispatch`:
- This plan doc.
- `infer/src/backend/metal/AGENTS.md` mixed-mode invariant entry.
- Strike 2c in `metal-dflash-qwen35-verify-batch.md`.

Rationale: three stacked dead-code warnings retire only when all callers
land; splitting would leave clippy-dirty intermediate commits
(CLAUDE.md §Verify rejects).

## Out of Scope

- **Cross-slot speculative scheduling (Layer 3).** Still per-slot inner
  loops; 2c.4 packs within tick, not across tick boundaries.
- **Stub-draft plain rows** (auto-enroll alternative). Two-subgraph route
  chosen; revisit as 2c.5 if async_eval doesn't parallelize.
- **CUDA DFlash parallelization.** Metal-only per plan §3 Out of scope.
- **Acceptance-rate tuning.** Temperature/top-k/draft-target fixed.
- **Prefix cache interaction.** Unchanged; 2b handles it.
- **Draft-model arch changes.** Block size, mask token, draft layer count fixed at load.
- **Logits-cache / speculative-tree branching.** Still 1-deep linear draft.

### Critical Files

- `infer/src/backend/metal/runtime.rs`
- `infer/src/backend/metal/request_state.rs`
- `infer/src/backend/metal/dflash.rs`
- `infer/src/backend/metal/qwen35.rs`
- `infer/src/backend/metal/mlx.rs`
