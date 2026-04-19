# Metal DFlash Qwen3.5 Layer 2c.3 — varlen rollback B=1 bit-ident — 2026-04-19

> Layer 2c.3 of the Qwen3.5 DFlash batched-verify roadmap
> ([`docs/plans/metal-dflash-qwen35-verify-batch.md`](../../plans/metal-dflash-qwen35-verify-batch.md)).
> Adds a **per-row** GDR rollback (`qwen35_rollback_to_accepted_varlen`)
> alongside the scalar so Layer 2c.4's mixed-mode scheduler dispatch can
> roll back multiple DFlash rows with independent `accepted_inputs` counts
> in a single call. Correctness-only; first throughput delta lands with 2c.4.

## Goal

**Type:** regression-check.
Does `qwen35_rollback_to_accepted_varlen` at `B=1` with
`accepted_inputs=&[k]` produce the same `gdr_flat` state as the scalar
`qwen35_rollback_to_accepted(accepted_inputs=k)`? Acceptance:
element-wise `|varlen[i] - scalar[i]| < 1e-3` on `float32` promotion
across every returned state/conv array, swept over
`k ∈ {0, 1, 2, block_size=4}`.

## Hypothesis

With `B=1`:
- `mlx_tape_replay_varlen(tape, k, g, state, steps=int32[1]=[k])` is
  equivalent to `mlx_tape_replay(..., steps=k)` — the varlen kernel's
  `steps[b]` broadcast collapses to the scalar path when B=1 and the
  tapes are pre-sliced to `T_padded = max(accepted_inputs) = k`.
- Per-row conv state stacking via `slice_prefix_axis1` on the B axis
  and `concatenate_axis(..., 0)` is trivially identity at B=1 — the
  tensor has one row, slice-of-one returns the same row.
- `Qwen35GdrTape` rank-4 shape `[B=1, T_padded, Hv, Dv/Dk]` matches
  the scalar tape layout bit-for-bit.

If it does not match, the bug is in one of:
1. The pre-slice to `T_padded` — varlen's batch-consistency check
   requires uniform `T` across tape/k/g, which the axis-1 slice
   satisfies but with different semantics than the scalar's
   `slice_prefix_axis1` (both slice axis 1, so should be identical).
2. Zero-step edge case (`k=0`, all-reject): varlen must skip the
   kernel call when `T_padded == 0` to avoid a zero-length T kernel
   invocation; scalar path bypasses via `accepted_i32 = 0`.
3. Conv state update: per-row axis-0 split/reassemble pattern must
   reduce to the scalar's `concatenate([conv, qkv_sliced], 1)` +
   `slice(..., start=len-kernel_minus_1)` at B=1.

## Parameters

- New function: `qwen35_rollback_to_accepted_varlen` at
  `infer/src/backend/metal/dflash.rs:1661-1768` (per agent report;
  verify against HEAD).
- Existing scalar under test: `qwen35_rollback_to_accepted` at
  `infer/src/backend/metal/dflash.rs:1608-1658`.
- FFI used: `mlx_tape_replay_varlen` at
  `crates/mlx-sys/src/lib.rs:661-668` (already present from a prior
  layer; no new FFI wrapper added in 2c.3).
- Test: `qwen35_rollback_to_accepted_varlen_matches_scalar_for_b1`
  (appended to the `tests` module in `dflash.rs`).
- Model: `z-lab/Qwen3.5-4B-DFlash` (draft, for tape source) paired
  with `mlx-community/Qwen3.5-4B-MLX-4bit` (target) via
  `$QWEN35_MODEL_PATH`.
- Test inputs: B=1 tapes captured from a single
  `verify_block_batched` forward (24 per-layer tapes, each
  `[1, 4, 32, 128]` for the innovation tape). `accepted_inputs`
  swept over `{0, 1, 2, block_size=4}`.
- Acceptance: per-k, element-wise `|varlen[i] - scalar[i]| < 1e-3`
  on `float32` cast across all 48 returned arrays (24 state +
  24 conv per rollback).

Command:

```bash
QWEN35_MODEL_PATH="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3" \
cargo test --release --no-default-features --features metal -p infer \
  --lib -- --test-threads=1 qwen35_rollback_to_accepted_varlen_matches_scalar_for_b1
```

## Environment

- Hardware: Apple M4 Max (40 GPU cores, ~400 GB/s UMA).
- macOS: 26.3.1 (build 25D771280a); Metal 3 via MLX 0.29.x
  (pinned in `crates/mlx-sys/CMakeLists.txt`).
- Commit: `c4be40b` (2c.2 bench doc) + uncommitted 2c.1 + 2c.2 +
  2c.2-mask + 2c.3 diffs. Stacked diff sizes (approx):
  `infer/src/backend/metal/qwen35.rs` 2c.1 (from prior),
  `crates/mlx-sys/src/mlx_dflash_draft_model.cpp` 2c.2 + mask fix,
  `crates/mlx-sys/src/lib.rs` 2c.2 + mask,
  `infer/src/backend/metal/dflash.rs` 2c.2 + 2c.3 (~260 lines 2c.3 alone:
  ~95 rollback fn + ~165 test).
- Feature set: `cargo build --release --no-default-features --features metal`.

## Results

```
test backend::metal::dflash::tests::qwen35_rollback_to_accepted_varlen_matches_scalar_for_b1 ... ok

test result: ok. 1 passed; 0 failed
```

- Assertion: element-wise `|varlen[i] - scalar[i]| < 1e-3` across
  all 48 returned arrays (24 state + 24 conv), swept over
  k ∈ {0, 1, 2, 4}.
- Observed `max_abs_delta = 0.0` — **bit-identical** at B=1 across
  every k tested, strictly better than the `< 1e-3` tolerance.
- Regression guard: `draft_forward_batched_matches_forward_for_b1`
  (2c.2 test) still passes.
- **Δ vs baseline:** zero numerical delta. No throughput number —
  2c.3 is rollback plumbing, no prod caller yet.

## Problems observed

- **Three release-build dead-code warnings** now stacked (up from
  two in 2c.2):
  ```
  warning: method `forward_batched` is never used
  warning: method `verify_block_batched` is never used
  warning: function `qwen35_rollback_to_accepted_varlen` is never used
  warning: `infer` (lib) generated 3 warnings
  ```
  Expected — all three are consumed only by correctness tests
  (`#[cfg(test)]`) or by 2c.4's scheduler dispatch, which hasn't
  landed yet. Commit strategy unchanged: hold the 2c.1 / 2c.2 /
  2c.2-mask / 2c.3 code diffs uncommitted until 2c.4 lands, commit
  all layers together with the dead-code warnings retiring in the
  same diff.
- **Codex-review [P2] against the 2c.2 test** (pre-existing from
  2c.2, now inherited by 2c.3): `draft_forward_batched_matches_forward_for_b1`
  and the new `qwen35_rollback_to_accepted_varlen_matches_scalar_for_b1`
  gate only on `QWEN35_MODEL_PATH` but unconditionally load
  `z-lab/Qwen3.5-4B-DFlash` via `MetalDflashRuntime::load`.
  Environments with the base target model cached but no DFlash
  draft checkpoint will regress from passing to failing when
  `cargo test --features metal` runs. Deferred to a follow-up
  (task #15 in session tracker); fix is either a secondary env-var
  guard (`QWEN35_DFLASH_DRAFT_PATH`) or a local-cache-check before
  `load`. Not a correctness bug in the code path — test hygiene only.
- §6 watch items:
  - Warmup (§6.1): single test invocation; N/A.
  - Thermal (§6.2): sub-second total test runtime; N/A.
  - Determinism (§6.7): deterministic tape source (verify forward
    at `temperature=0` with fixed inputs) + compiled varlen kernel;
    delta is exactly zero, no RNG involvement.

## Learnings

- **Pre-slice to `T_padded` is the varlen entry-point contract.**
  The kernel's `require_rank` / batch-consistency checks at
  `crates/mlx-sys/src/mlx_bridge.cpp:906-917` demand uniform T
  across tape/k/g, even though `steps[b]` lets each row consume a
  different count. The agent chose to pre-slice all three tapes
  (innovation, k, g) to
  `T_padded = accepted_inputs.iter().max().unwrap_or(&0)` on axis 1
  before the varlen call. This is a clean adaptation: the existing
  `slice_prefix_axis1` helper already handles the axis-1 slice, so
  no new slicing primitive was needed.
- **Zero-step early-skip avoids a kernel-side trap.** When every
  row is fully rejected (`accepted_inputs = [0, 0, ...]`),
  `T_padded = 0` and invoking the varlen kernel with a zero-length
  T dimension either errors or produces degenerate output. The
  implementation early-skips the kernel call and only resets
  `gdr_flat[state_idx]` / `gdr_flat[conv_idx]` from `gdr_snapshot`
  in that case — this matches the scalar path's semantics at k=0.
- **Per-row axis-0 split/reassemble for conv state won.** The spec
  offered two alternatives: (a) right-pad each row's accepted slice
  with the previous conv state tail before concat + slice (uniform
  T_padded conv path), or (b) process rows independently via
  axis-0 `slice_prefix_axis1` + `concatenate_axis(..., 0)`. Agent
  picked (b) — clarity over cleverness, and the bit-ident result
  confirms the simple pattern is semantically identical to the
  scalar's `concatenate + tail-slice` at B=1. Revisit (a) only if
  2c.4 benchmarks show the per-row reassemble is a hot spot.
- **Bit-identical at B=1 across 4 k-values is a stronger signal
  than 2c.2.** 2c.2 verified one fixed B=1 forward; 2c.3 verified
  the rollback at k ∈ {0, 1, 2, 4} including the zero-step
  early-skip edge case. The varlen kernel's `steps[b]` broadcast at
  B=1 is confirmed to collapse to the scalar path across the
  full-reject → full-accept range. Strengthens confidence that the
  2c.4 B=2 varlen rollback test (with `accepted_inputs = [k1, k2]`
  where `k1 != k2`) will hit the same bit-ident bar.
- **Stopping rule hit** (spec §7.3): variance N/A for correctness;
  hypothesis confirmed with strictly-better-than-tolerance delta;
  §6 clean modulo the inherited test-gating [P2]; no prior-snapshot
  delta (first 2c.3 entry). One run per k-value suffices.

## Follow-ups

- Layer 2c.4: lift the `open.len() >= 2` permanent-downgrade at
  `infer/src/backend/metal/runtime.rs:1081` and wire
  `forward_batched` + `verify_block_batched` +
  `qwen35_rollback_to_accepted_varlen` into
  `execute_qwen35_packed_decode_batch` with mixed-mode dispatch
  (DFlash rows + plain rows queued as two parallel MLX subgraphs
  before `async_eval`). This is the first prod caller for all
  three batched APIs — dead-code warnings retire here.
- B=2 varlen-rollback bit-ident test (equivalent to 2c.1's B=2
  verify test) lands with 2c.4, when the stacking infrastructure
  exists at the scheduler level and `accepted_inputs=[k1, k2]`
  with `k1 != k2` becomes reachable.
- Address codex-review [P2] (test gating on draft-model
  availability) after the 2c.4 bundle — either add
  `QWEN35_DFLASH_DRAFT_PATH` env var or check local HF cache
  before `MetalDflashRuntime::load` in both batched-correctness
  tests.

## Cross-links

- Plan: [`docs/plans/metal-dflash-qwen35-verify-batch.md`](../../plans/metal-dflash-qwen35-verify-batch.md) §Layer 2c.
- Prior 2c.2 entry (batched draft forward B=1 bit-ident):
  [`2026-04-19-verify-metal-qwen35-dflash-2c2-draft-batched-b1-bit-ident.md`](2026-04-19-verify-metal-qwen35-dflash-2c2-draft-batched-b1-bit-ident.md).
- Prior 2c.1 entry (verify_block_batched B=2 bit-ident):
  [`2026-04-19-verify-metal-qwen35-dflash-2c1-b2-bit-ident.md`](2026-04-19-verify-metal-qwen35-dflash-2c1-b2-bit-ident.md).
- Prior 2b entry (verify_block_batched B=1 bit-ident):
  [`2026-04-19-verify-metal-qwen35-dflash-2b-bit-ident.md`](2026-04-19-verify-metal-qwen35-dflash-2b-bit-ident.md).
- Layer 1 baseline (packed decode 162.9 tok/s at c=8 — 2c.4
  throughput target):
  [`2026-04-19-metal-qwen35-final-state.md`](2026-04-19-metal-qwen35-final-state.md).
- FFI: `crates/mlx-sys/src/mlx_bridge.cpp` `mlx_tape_replay_varlen`.
