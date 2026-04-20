# Train Runtime — Systematic Test Plan v1

Companion to [`train-runtime-architecture-v1.md`](train-runtime-architecture-v1.md).
Written 2026-04-20. Scope: the training runtime primitives that land in Phase 1
(lr_schedule, grad_accum, adamw_state, metrics, grad_clip, checkpoint v2) plus
the Phase 2 `Trainer<O, C, S>` loop and its integration with the current
train binaries (`train_sft`, `pretrain_qwen3`, `train_grpo`,
`train_multi_turn`, `eval_lm`) plus legacy compatibility surfaces that
are being retired.

Current reality: the train-side implementation already includes the
dense/full-attn Qwen3.5-family path, `train_multi_turn` runs on it, and
checkpoints are already HF-style directories. The handwritten
Transformer/TinyLM runtime compatibility path has been deleted, and the
hybrid linear-attn train path has not landed yet. The acceptance
contract below tracks that current path and the post-legacy train test
surface.

This document is **the acceptance contract for Phase 1 + Phase 2**. A change
in the train runtime is not "done" until the matching row here is either
green or explicitly deferred with a reason.

---

## 1. Why test this carefully (real-world context)

Five external failure modes shaped the dimensions below. Pointers, so future
maintainers understand what we are insured against:

| # | Upstream failure | Lesson for us |
|---|------------------|---------------|
| 1 | [MLX `mlx-lm` #2617](https://github.com/ml-explore/mlx/issues/2617): cosine decay started *during* warmup | LR schedule boundary behavior must be asserted at every boundary step, not just interior points. |
| 2 | [HF transformers PR #8624](https://github.com/huggingface/transformers/pull/8624): grad-accum resume computed skip wrongly | `(step, micro_idx)` is one state; resume must land on the optimizer-step boundary and the accumulation counter must start fresh. |
| 3 | [HF transformers #27749](https://github.com/huggingface/transformers/issues/27749): LR didn't anneal after resume | Schedule is re-constructed from persisted params, NOT kept alive across process restart. Step index is the single source of truth. |
| 4 | [HF transformers #35484](https://github.com/huggingface/transformers/issues/35484): cosine LR + grad accum double-stepped | LR advances on optimizer step, NOT on micro-batch. Guard with a property test. |
| 5 | [Candle #1307](https://github.com/huggingface/candle/issues/1307) + [PR #2328](https://github.com/huggingface/candle/pull/2328): NaN in backward on large inputs (activations, ELU) | Numerical stability tests on the loss path; any new op lands with an overflow-edge-case test. |

Additional context absorbed: [Candle #695](https://github.com/huggingface/candle/issues/695)
(standardize optimizer trait — done), [Candle #1383](https://github.com/huggingface/candle/issues/1383)
(perf parity), [Candle #2271](https://github.com/huggingface/candle/issues/2271)
(Metal memory leak in matmul — tape memory growth tests cover the train-side
analogue).

---

## 2. Test dimensions (the matrix)

Every primitive is tested along the dimensions it can fail on. If a row is
"N/A" we say so explicitly rather than leaving it blank.

| Dimension | What we check |
|-----------|---------------|
| **D1. Unit contract** | Each primitive's public API behaves per its docstring at interior values. Single-file tests under `crates/*/tests/`. |
| **D2. Boundary values** | 0, 1, max-step, `usize::MAX` (where meaningful), warmup→decay boundary, `grad_accum_steps == 1`, `total_steps == 0`. |
| **D3. Numerical stability** | NaN / Inf inputs are rejected or pass through as `null` (JsonlSink), not silently become `0.0`. Loss on pathological activations doesn't explode AdamW moments. |
| **D4. State round-trip** | Save → load → step-more produces bit-identical moments (within f32 ULP tolerance) to never-saved baseline after the same total step count. |
| **D5. Concurrency / reentrance** | `Trainer::step` does not leak state between runs. `MetricSink::emit` is `&self` (if ever made so) — today it is `&mut self`, so we only check a single trainer at a time. **N/A at process level** — trainers are single-threaded by design. |
| **D6. Cross-backend parity** | Host-side state (moments, step counter) is identical whether training ran on CPU, Metal, or CUDA. Backend differences live only in forward/backward op results, not optim state. Tests skip gracefully when the backend is unavailable. |
| **D7. Integration end-to-end** | A real binary (`train_sft` on a tiny fixture) runs N steps, saves, the same binary resumes and produces the same metrics for steps N..2N as a continuous run does. |
| **D8. Performance regression** | `docs/bench-and-trace-spec.md` rules apply — every runtime-affecting diff lands with a Δ% row against the prior wins snapshot for the affected backend+model. |
| **D9. Model convergence** | Tiny toy model learns a tiny problem. Loss at step K < loss at step 0 by a margin large enough to detect a broken backward without being flaky. Tolerances chosen for 1% false-positive rate at 95% confidence. |

---

## 3. Per-primitive test inventory

For each primitive: the file where tests live, the dimensions covered, and
the specific failure modes guarded.

### 3.1 `autograd::lr_schedule`

**File:** `crates/autograd/tests/test_lr_schedule.rs` (exists).

| # | Test | Dim | Guards against |
|---|------|-----|----------------|
| LR-1 | `constant_returns_same_lr` | D1 | — |
| LR-2 | `linear_warmup_reaches_full_at_warmup_step` | D2 | off-by-one at boundary |
| LR-3 | `cosine_with_warmup_no_decay_during_warmup` *(new)* | D2, lesson #1 | MLX #2617 |
| LR-4 | `cosine_with_warmup_final_at_total_steps` | D2 | final LR = min_lr, not 0 |
| LR-5 | `parse_lr_schedule_rejects_unknown` *(new)* | D1 | silent fallback |
| LR-6 | `step_past_total_clamps_to_final_lr` *(new)* | D2 | NaN from cosine `cos(π * (step-total)/decay)` |
| LR-7 | `schedule_only_advances_on_optimizer_step` *(integration)* | D2, lesson #4 | HF #35484 |

LR-3 is the MLX-bug guard: loop from step 0 to warmup_steps-1 and assert
`lr(step) == warmup_step * base_lr / warmup_steps`, with no cosine term.
LR-7 lives in the Trainer integration tests (§3.7) because it requires the
loop.

### 3.2 `train::grad_accum`

**File:** `crates/train/tests/test_grad_accum.rs` (exists).

| # | Test | Dim | Guards |
|---|------|-----|--------|
| GA-1 | `accum_steps_1_is_ready_immediately` | D2 | short-circuit path |
| GA-2 | `accum_counts_up_then_triggers` | D1 | — |
| GA-3 | `reset_returns_counter_to_zero` | D1, D4 | — |
| GA-4 | `loss_scale_equals_reciprocal` | D1 | — |
| GA-5 | `state_round_trip_matches_continuous` *(new, integration)* | D4, lesson #2 | HF PR #8624 — resume mid-window is disallowed; we assert that saving only ever happens *between* windows (i.e. `grad_accum_current == 0`). |

GA-5 asserts the invariant that `TrainerStateDoc.grad_accum_current == 0`
for every checkpoint the Trainer writes, AND that loading a doc with
non-zero accum panics with a clear error (defence in depth).

### 3.3 `autograd::adamw_state`

**File:** `crates/autograd/tests/test_adamw_state.rs` (exists).

| # | Test | Dim | Guards |
|---|------|-----|--------|
| AS-1 | `export_then_import_preserves_step` | D4 | — |
| AS-2 | `export_then_import_preserves_moments_bitwise` | D4 | f32 quantization drift |
| AS-3 | `import_with_missing_param_errors` *(new)* | D1 | silent zeros |
| AS-4 | `import_with_shape_mismatch_errors` *(new)* | D1 | — |
| AS-5 | `cross_backend_state_is_host_only` *(new, D6)* | D6 | accidental device-tensor leakage into the state struct |
| AS-6 | `two_optimizers_export_disjoint_states` | D1 | global-state aliasing |

AS-5 builds an `AdamWState`, serialises to bytes, and asserts the byte image
contains no device handles (i.e. the only types in the graph are `Vec<f32>`,
`Vec<usize>`, `u64`, and `String`).

### 3.4 `train::metrics`

**File:** `crates/train/tests/test_metrics.rs` (exists).

| # | Test | Dim | Guards |
|---|------|-----|--------|
| M-1 | `null_sink_is_noop` | D1 | — |
| M-2 | `stdout_sink_writes_line_per_emit` | D1 | — |
| M-3 | `jsonl_sink_buffers_then_flushes_on_drop` | D1, D4 | data loss on panic |
| M-4 | `jsonl_sink_nan_serialises_as_null` | D3 | JSON `NaN` → invalid JSON |
| M-5 | `jsonl_sink_inf_serialises_as_null` | D3 | same |
| M-6 | `multi_sink_fans_out_in_order` | D1 | — |
| M-7 | `open_sink_parses_spec` | D1 | — |
| M-8 | `jsonl_line_is_parseable_by_serde_json` *(new)* | D1 | manual JSON string building drift |

M-8 reads the file back and `serde_json::from_str` each line — a regression
would surface immediately.

### 3.5 `train::grad_clip`

**File:** `crates/train/tests/test_grad_clip.rs` (exists).

| # | Test | Dim | Guards |
|---|------|-----|--------|
| GC-1 | `no_clip_returns_true_norm_unchanged` | D1 | — |
| GC-2 | `global_norm_scales_below_threshold_unchanged` | D2 | threshold boundary |
| GC-3 | `global_norm_scales_above_threshold` | D1 | — |
| GC-4 | `global_norm_zero_max_norm_panics_early` *(new)* | D2 | silent divide-by-zero |
| GC-5 | `norm_computation_matches_legacy_free_fn` *(new)* | D1 | trait vs. free-fn drift |
| GC-6 | `norm_on_empty_params_is_zero` *(new)* | D2 | NaN from 0/0 |

### 3.6 `train::checkpoint` (v2 codec)

**File:** `crates/train/tests/test_checkpoint_v2.rs` (exists).

| # | Test | Dim | Guards |
|---|------|-----|--------|
| CK-1 | `save_then_load_round_trips_trainer_state` | D4 | — |
| CK-2 | `load_rejects_codec_version_mismatch` | D1 | silent truncation |
| CK-3 | `load_rejects_missing_optimizer_file` | D1 | — |
| CK-4 | `save_writes_atomic_on_interrupt` *(new, deferred)* | D1 | partial write |
| CK-5 | `moments_file_readable_by_plain_safetensors` *(new)* | D1, D6 | breaks HF interop |
| CK-6 | `cross_backend_resume_bitwise` *(new, D6)* | D6, lesson #3 | moments drift on reload |
| CK-7 | `load_rejects_schema_mismatch` *(new)* | D1 | cross-optim load clobbers state |

CK-4 is **deferred** because fsync-rename atomicity requires a platform
harness; candle's HF `safetensors` crate already uses atomic rename on POSIX.
Cited deferral in the test body.

### 3.7 `train::trainer` — `Trainer<O, C, S>` loop

**File:** `crates/train/tests/test_trainer_loop.rs` (Wave 2 agent target).

These are the integration tests that tie every primitive together.

| # | Test | Dim | Guards |
|---|------|-----|--------|
| TL-1 | `trainer_runs_total_steps` | D1 | off-by-one on step counter |
| TL-2 | `grad_accum_triggers_step_every_n` | D1 | counter wraparound |
| TL-3 | `lr_schedule_drives_optimizer_lr` | D1, D2, lesson #4 | HF #35484 (cosine + accum double step) |
| TL-4 | `resume_restores_step` | D4 | — |
| TL-5 | `resume_lr_matches_continuous_run` | D4, lesson #3 | HF #27749 |
| TL-6 | `checkpoint_save_writes_directory` | D1 | — |
| TL-7 | `metrics_emit_at_log_every` | D1 | — |
| TL-8 | `nan_loss_does_not_corrupt_moments` *(new, D3, lesson #5)* | D3 | Candle #1307 |
| TL-9 | `trainer_is_architecture_agnostic_smoke` *(new)* | D1 | cfg leakage |
| TL-10 | `two_trainers_do_not_share_state` *(new, D5)* | D5 | hidden globals |

TL-8: inject a `step_fn` that returns a NaN loss on step 5. Assert the
Trainer surfaces the NaN in the metric (as JSON `null`) and that moments
after step 5 are still finite — i.e. the optimizer step must short-circuit
on non-finite gradients OR the caller is responsible. We document which,
and the test pins that contract.

TL-9: build a `Trainer` with a `Module` that returns a deterministic loss
from only `TensorStore` ops — no backend-specific types. Ensures the
generic loop compiles without a specific backend feature flag.

### 3.8 Cross-primitive property tests

**File:** `crates/train/tests/test_train_properties.rs` *(new)*.

Property tests via `proptest` on small generated sequences. Three properties,
each tying two or more primitives together:

| # | Property | Guards |
|---|----------|--------|
| P-1 | For any `(total, warmup, accum)` with `warmup < total`, Trainer ends with `step == total` and never steps past the final LR. | boundary loops |
| P-2 | For any divisible `(total, save_every)`, loading from the final checkpoint and running 0 more steps produces an identical `TrainerStateDoc` to the end of the continuous run. | round-trip |
| P-3 | `(grad_accum_steps, log_every)` do not interact: loss sum over a log window equals the sum of per-optimizer-step scaled losses over the same steps. | window arithmetic |

Property tests run under `cargo test --release` gated on feature
`proptest` (dev-only). Budget: 64 cases each, seed fixed to keep CI
deterministic — `PROPTEST_CASES=64 PROPTEST_SEED=0xfeed`.

### 3.9 Integration end-to-end (tiny SFT)

**File:** `crates/train/tests/test_sft_end_to_end.rs` *(new, §D7)*.

One test: tiny train-side stub on the dense/full-attn Qwen3.5-family path
with 2 layers, hidden=16, vocab=64, trained for 8
optimizer steps with grad_accum=2, log_every=2, save_every=4. Three runs
compared:

1. **Continuous**: 8 steps, no resume.
2. **Split-at-4**: 4 steps, save, fresh process re-loads and trains 4 more.
3. **Split-at-5** (off boundary, must FAIL cleanly): asserting we reject
   attempts to save mid-accumulation-window.

Assertion: continuous.metrics == split_at_4.metrics (byte-identical JSONL
after normalisation), and split_at_5 panics with a clear message.

Runs on CPU backend, skipped on CI if no fixture weights exist. **Metal
and CUDA end-to-end live in existing `tests/test_sft_loop_metal.rs` and
future `tests/test_sft_loop_cuda.rs`** — those wrap real model forward and
compare against JSON baselines under `infer/test_data/`.

### 3.10 Model convergence smoke (D9)

**File:** `crates/train/tests/test_convergence_smoke.rs` *(new)*.

Tiny embedding → linear → cross_entropy on a 4-token fixed vocabulary,
10 optimizer steps, AdamW lr=1e-2. Assertion: `loss[0] - loss[9] > 0.5`
with seed pinned. Passes deterministically on CPU; seed-pin eliminates
flakiness. Purpose: catch *any* broken backward op that the other tests
don't (because every other test uses synthetic loss).

### 3.11 Existing binary regressions

The runtime migration has already landed on the main `Trainer<O, C, S>`
path. The current training binaries (`pretrain_qwen3`, `train_grpo`,
`train_multi_turn`) still compile against the legacy `trainer::*`
re-exports as compatibility aliases; `pretrain_qwen3` is now a generic
Qwen-family entrypoint rather than a Qwen3-only path. This section now
only guards that surface until the aliases are retired. Regression guard: a `#[test]` in
`crates/train/tests/test_legacy_imports.rs` that imports each legacy
symbol by path:

```rust
#[test]
fn legacy_clip_grad_norm_is_exported() {
    let _: fn(&[_], f32, &mut _) = train::trainer::clip_grad_norm;
}
```

Removed once the compatibility aliases are retired.

---

## 4. Non-test guardrails

Tests are not the only line of defence; these are the complementary checks:

- `cargo clippy -p train -p autograd -- -D warnings` in CI — catches
  accidental `unwrap()` in the loop.
- `cargo doc --no-deps --workspace` — docstrings compile (we cite files
  and lessons above; keep them evergreen).
- `codex review --uncommitted` on every Phase 2 commit before push.
- `scripts/bench_guidellm.sh` comparing Trainer-based `train_sft` against
  the pre-Trainer snapshot for step-time regression — one run, labeled
  `trainer-migration-baseline`.

---

## 5. Execution plan

Wave-aligned with the Phase 2 rollout:

| Wave | Tests | Depends on |
|------|-------|------------|
| T-1 | §3.1-§3.6 *(new)* rows | Phase 1 primitives landed |
| T-2 | §3.7 TL-1..TL-7 | Wave 2 Trainer lands |
| T-3 | §3.7 TL-8..TL-10, §3.8, §3.9, §3.10 | Trainer tests green |
| T-4 | §3.11 regression | Phase 2 commit boundary |
| T-5 | retire §3.11 aliases | compatibility cleanup |

Each wave ships as one commit with a matching wins entry citing the
fixtures and the lesson# each new test guards.

---

## 6. Deferred / out-of-scope

| Item | Reason |
|------|--------|
| `fsync`-atomicity checkpoint test | safetensors crate already writes atomically on POSIX; platform harness is overkill for v1. Revisit if we see a partial-write issue in the wild. |
| Multi-process distributed training test | Not in v1 scope; the architecture is single-process. |
| Mixed-precision (bf16/fp16) tests | We are host-f32 for moments by design; no fp16 codepath yet. Re-open when bf16 training lands. |
| CUDA end-to-end test in-tree | Lives in `tests/` gated on `cuda` feature — already a pattern under `infer/`. Adds to the plan when Phase 3 binary migration touches CUDA. |

---

## 7. Deltas from architecture doc

None. This plan is the test-side expansion of
[`train-runtime-architecture-v1.md`](train-runtime-architecture-v1.md) §7
("Acceptance"). If the architecture changes, this plan is updated before
any test file moves.
