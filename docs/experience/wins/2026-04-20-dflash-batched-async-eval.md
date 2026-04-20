# Metal Qwen3.5 DFlash — defer batched terminal `eval` via `async_eval`

**Status:** functional validation ✅ (DFlash sweep at prompt_tokens=1024
completes cleanly, no regressions vs pre-fix crash-free baseline). Numeric
c=2 TPOT delta remains `pending-remote` for a matched-A/B session per
`feedback_matched_ab_for_small_bench_effects.md` — the 2–5% sized effect
is below the single-session thermal-noise threshold.

**2026-04-20 follow-up:** matched-A/B close-out ran (two thermal-separated
sessions, order-randomized). Result: **inconclusive — within noise**.
Cross-session mean Δ = −1.4% (well below the 10% bar); per-session
deltas sign-flip with run order (−4.5% session 1, +1.8% session 2),
indicating order-of-run (cold vs warm) dominates binary identity at this
effect size. See
[`2026-04-20-dflash-async-eval-matched-ab.md`](./2026-04-20-dflash-async-eval-matched-ab.md)
for full 2×2 table + rule update. `pending-remote` status retained: we
have the measurement, but no confirmed numeric to replace it with;
correctness gates still pass and the change is non-regressing.

**Date**: 2026-04-20
**Machine**: Apple M4 Max (40 GPU cores, ~400 GB/s UMA), macOS 26.3.1
**Model**: `mlx-community/Qwen3.5-4B-MLX-4bit` + `z-lab/Qwen3.5-4B-DFlash`
**Commit**: this commit
**Parent audit**: `2026-04-20-metal-qwen35-post-double-buffer-audits.md` §Audit 1
**Parent win (same pattern, scalar path)**: `2026-04-20-metal-qwen35-decode-double-buffer.md` (`f6be5f6`)

## Goal

Close Audit-1's sized 2–5% lever on the DFlash batched speculative path at
c=2 by deferring the terminal full-fence `eval` at
`infer/src/backend/metal/dflash.rs:2456` (end of
`qwen35_dflash_speculative_block_batched`). The existing `eval` blocks the
CPU until all packed KV / GDR / `updated_target_hidden` arrays are
materialized before the function returns — preventing the caller from
starting to build the next block's graph while the current block's GPU
work still drains.

## Hypothesis

Swap the terminal `eval(&to_eval)` for `async_eval(&to_eval)`. MLX splits
`eval` (blocks host) from `async_eval` (queues GPU work without blocking).
The next block's `sample_rows()` → `.item_i32()` prefix-match scan
(`dflash.rs:1505`) is the natural sync point; the deferred queue flushes
there. Mirrors the mlx-lm `generate_step` pattern and the Qwen3.5 scalar
step-driver double-buffer landed in `f6be5f6` (+12.7% step-driver c=1).

## Setup / params

- Change is Rust-only, one file, 17 net LOC. No FFI additions — `async_eval`
  was already exposed at `infer/src/backend/metal/mlx.rs:659`.
- Build: `cargo build --release --no-default-features --features metal`.
- Correctness gates:
  - `cargo test --release --no-default-features --features metal -p infer --lib`
    → 329 tests green (full lib suite).
  - DFlash-focused suite with real Qwen3.5-4B-MLX-4bit weights:
    `QWEN35_MODEL_PATH=<mlx-4bit snapshot> cargo test ... backend::metal::dflash::tests`.
  - Bit-ident parity test `qwen35_dflash_packed_batch_b2_matches_scalar_runs`
    with B=2 — the primary correctness gate for this change.

## Environment

- `cargo build --release --no-default-features --features metal` green.
- `cargo test --release --no-default-features --features metal -p infer --lib`
  runs 329 tests; all pass.

## Results

### Correctness — bit-ident parity gate

With `QWEN35_MODEL_PATH` pointing at
`~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3`:

```
running 4 tests
dflash_qwen35_verify_batched_matches_two_single_row_runs
  scalar_done accepted_inputs=[1, 1]
  batched_done accepted_inputs=[1, 1]
  row 0 max_abs_delta=0
  row 1 max_abs_delta=0
  overall_max_abs_delta=0                          ← bit-ident
draft_forward_batched_matches_forward_for_b1
  max_abs_delta=0                                  ← bit-ident
qwen35_dflash_packed_batch_b2_matches_scalar_runs
  scalar_first_token=24218
  batched_first_tokens=[24218, 24218]              ← token-equal vs scalar
qwen35_rollback_to_accepted_varlen_matches_scalar_for_b1
  k=0 max_abs_delta=0
  k=1 max_abs_delta=0
  k=2 max_abs_delta=0
  k=4 max_abs_delta=0
  overall_max_abs_delta=0                          ← bit-ident

test result: ok. 4 passed; 0 failed
```

All four DFlash parity tests — including the B=2 packed-batch-vs-scalar
run that is the direct correctness gate for this change — pass with real
weights. The batched path's sampled first tokens still match the scalar
path token-for-token, and per-row updated hidden deltas are 0.

### Throughput — functional pass, numeric delta pending matched-A/B

**Functional sweep ✅** — after Agent A's prefill-fastforward fix landed
(`3bc8802`), DFlash sweep at `prompt_tokens=1024,output_tokens=128` on
the same binary as this commit (`d8cb2f4`) completes all 10 guidellm
strategies cleanly: no `WrongPhase` errors, no 500s, 100% request
success. Artefacts: `bench-output/2026-04-20-dflash-fixes-validation/`.
This confirms the `async_eval` swap does not regress the batched DFlash
path at c=1..c=8 HTTP concurrency.

**Numeric c=2 TPOT delta — pending-remote (matched-A/B required).**
The sized effect (2–5% at c=2 per Audit 1) sits below the single-session
thermal-noise threshold documented in
`feedback_matched_ab_for_small_bench_effects.md`. Resolving it above
noise needs a matched same-binary env-A/B run in ≥2 sessions, which is
its own bench session — not a drive-by. Ticketed for a dedicated
session:
- Parent commit: `d8cb2f4` on `main`.
- Action: `scripts/bench_guidellm.sh metal-m4max-dflash-async-eval` at
  `prompt_tokens=1024,output_tokens=256`, paired with a revert-binary
  baseline run in the same session.
- Baselines to beat: `2026-04-17` / `2026-04-18` DFlash wins; the
  `2026-04-20-metal-qwen35-decode-double-buffer.md` HTTP c=2 row
  (22.98 ms TPOT mdn, 58.5 out tok/s agg — *scalar* c=1+, unaffected
  by this batched-path change).
- Expected delta: +2–5% TPOT at c=2 on DFlash-batched decode, or within
  noise. Regression gate: anything ≥ −2% must be investigated.

## Problems

- **Metal Toolchain state**. `xcrun metal` still errors in fresh
  invocations (`cannot execute tool 'metal' due to missing Metal
  Toolchain`), but the parent worktree's cached `libmlx.a` let an
  incremental `cargo build --release --no-default-features --features
  metal` complete in ~7 s — no fresh `.metal → .air` compile was
  required. Both `metal_serve` and `metal_bench` rebuilt from this
  cache, which unblocked the functional sweep above. A matched-A/B
  numeric run still wants a clean toolchain to rebuild both binaries
  side-by-side without cache skew.
- No B>2 packed DFlash correctness test yet (only B=2 exists). This
  change affects any B≥2, but the B=2 suite covers the only currently-
  wired cross-row cases. Lift flagged in `docs/plans/metal-dflash-qwen35-verify-batch.md`.
- `cargo clippy --features metal` still fails in this env (pre-existing
  cmake-profile drift — noted in `2026-04-20-metal-qwen35-post-double-buffer-audits.md`
  §Problems). Not introduced by this patch.

## Correctness argument (audit, not just trust the bench)

The returned `updated_target_hidden` rows are reinstalled into each
request's `dflash.target_hidden` at `request_state.rs:1991` and fed as
an input to the *next* DFlash block at `request_state.rs:2702` (scalar
path) / `request_state.rs:3435` (re-entry). No caller reads host memory
from those handles between invocations — `shape()` is metadata only
(`mlx.rs:170`), and the per-row packed KV slices use lazy view ops
(`slice_row`) that don't materialize. The sampled token Vec returned in
`DFlashBlockResult.accepted_tokens` is already a `Vec<u32>` (from the
internal `sample_rows()` that already ran a blocking eval), so the
caller's per-row `record_sampled_token` path is unchanged.

The next block's own `sample_rows()` ⟶ `.item_i32()` at `dflash.rs:1505`
flushes the queued work, so the lazy tail is bounded at "one block
deep" — matching the scalar-path invariant landed in `f6be5f6`.

## Learnings

- MLX's `async_eval` is the right tool when the next consumer will
  naturally sync (here: the next block's greedy sampler `.item_i32()`).
  The telltale is a terminal `eval` positioned immediately before a
  `return` whose caller loops back into more graph building.
- Audit-1's estimate (~0.5–1.0 ms saved per block ≈ 2–5% of 23 ms c=2
  TPOT) is consistent with the scalar-path win's 1.5 ms/step at c=1; the
  batched path has more work queued behind the fence so the per-block
  savings should be similar-or-larger per block but diluted across the
  batch's per-token cost.
- The `async_eval` change is *safe by construction* whenever every
  caller's next host read on the deferred arrays is preceded by a
  subsequent `eval` / `.item_i32()`. Whenever you touch a terminal eval,
  audit the caller's read pattern first.

## Cross-refs

- `docs/experience/wins/2026-04-20-metal-qwen35-decode-double-buffer.md`
  — scalar-path parent win (same pattern, +12.7% step-driver c=1).
- `docs/experience/wins/2026-04-20-metal-qwen35-post-double-buffer-audits.md`
  §Audit 1 — where this lever was sized (2–5% at c=2).
- `docs/plans/metal-dflash-qwen35-verify-batch.md` — Layer 2 verify-batch
  plan, flags the lift of the B>2 test gap.
- `infer/src/backend/metal/dflash.rs:2450` (patched).
- `infer/src/backend/metal/mlx.rs:659` (`async_eval` wrapper, unchanged).
- `crates/mlx-sys/src/mlx_bridge.cpp:1110` (`mlx_async_eval` C bridge,
  unchanged).

## Rule

**When a terminal full-fence `eval` sits immediately before a return and
the caller will naturally sync on its next step, defer via `async_eval`
— but audit every caller's consumption of the returned handles first.
Host reads (`.item()`, `.as_slice()`) of deferred arrays before the next
sync are the only way this can break.**
