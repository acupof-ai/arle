# OPD step retain_ids prune caps per-step TensorStore growth

## Goal

Prevent multi-step `opd_step` runs from accumulating rollout, teacher-forward,
and student-forward temporaries in `TensorStore`. This is a memory-correctness
fix for CPU OPD smoke and future full-model OPD runs; no throughput win is
claimed.

## Hypothesis

`opd_step` cleared the tape after backward but did not prune the store. Because
`Qwen35Model::forward` allocates dense logits and intermediate activations,
live tensor slots should grow roughly linearly across repeated steps unless the
step prunes back to model parameters plus persistent gradients.

## Params

- Backend: CPU `TensorStore`
- Model: embedded tiny Qwen3.5 test config
- Prompt: `[1, 3, 8]`
- Rollout: 2 generated tokens
- Runs: 3 consecutive `opd_step` calls in one store

## Results

Pre-fix regression test failure:

```text
live tensor counts: [1182, 2285, 3388]
```

Post-fix gate:

```text
cargo test -p train --test test_opd_step --release -- --nocapture
test opd_step_prunes_ephemeral_tensors_between_steps ... ok
```

The regression test now asserts step 2 and step 3 reuse the same retained
tensor set as step 1, so forward temporaries no longer survive across steps.

## Verification

- `cargo test -p train --test test_opd_step --release -- --nocapture`
- `cargo test -p train --test test_opd_determinism --release`
- `cargo test -p train --test test_opd_grad_check --release -- --nocapture`
- `cargo test -p train --release`
- `cargo check --workspace`
- `cargo clippy -p train --all-targets --release -- -D warnings`
- `cargo build --workspace --release`

## Problems

This only bounds inter-step growth. It does not reduce the peak memory inside a
single long rollout; intra-rollout pruning is a separate axis because it must
preserve the current rollout token state while the tape is disabled.

## Learnings

For OPD performance work, first pin memory lifetime. Wall-clock benches across
multiple steps are not trustworthy if later steps carry previous-step
activations and logits in the store.
