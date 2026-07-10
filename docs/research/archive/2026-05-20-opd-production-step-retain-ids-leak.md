# 2026-05-20 — Production `arle train opd` loop leaks `TensorStore` between steps

> **Status:** research / findings. Independent of the
> `forward_last_logits` and `lm_head`-transpose perf work — this is a
> production-path memory-correctness bug that surfaced while surveying
> `retain_ids` for that perf research. Codex may already be fixing it
> (active WIP on `crates/cli/src/train_cli.rs` as of 2026-05-20 EOD); if
> not, this is the next-priority handle.

## Finding

`crates/cli/src/train_cli.rs::run_opd` (the `arle train opd` entry point,
roughly lines 240-308) calls `opd_step` in a tight `for step in 1..=args.steps`
loop with **no `retain_ids` or `cleanup_after_backward` between steps**.

```rust
for step in 1..=args.steps {
    let outcome = opd_step(
        &student, &teacher, &prompt_ids,
        step_cfg, &student_params,
        &mut optimizer, &mut store, &mut tape,
    ).with_context(|| format!("opd step {step} failed"))?;
    losses.push(outcome.loss);
    // ... println / json output, no store cleanup
}
```

`opd_step` itself does `tape.entries.clear()` at `crates/train/src/opd.rs:144`
but leaves every ephemeral tensor in the `TensorStore`. Per step at
Qwen3-0.6B this is on the order of 50-150 `TensorId`s, with the largest
being `[1, S, vocab=151_936]` lm_head outputs (~600 MB-class at f32) and
all intermediate attention/MLP buffers.

The `opd.rs` comment at lines 139-146 claims the caller owns the keep set
and *"the training loop in `arle train opd` builds the keep set once and
reuses it"*. It does not. Either the comment describes intended-but-not-
implemented behaviour, or it's stale from a different historical caller.

## Severity

Unbounded growth. A 100-step run with `rollout_len=2, seq_len≈5,
vocab=151_936` accumulates ≈ 100 steps × 150 tensors × ~10 MB average =
**~150 GB** of leaked memory before the kernel kicks in with SIGKILL.
Even a 10-step smoke run leaks ~15 GB on Qwen3-0.6B.

This is also a plausible contributor to the 2026-05-20 cooperative-session
OOM: the A/B bench harness inherited the same "no retain_ids between
calls" pattern, and codex's harness-side fix added retain_ids between
rollout iters — but the production code path still has the same bug at
the per-`opd_step` boundary.

## Fix sketch (codex implements)

Two pieces:

1. **Expose the cos/sin cache `TensorId`s on `Qwen35Model`** so the
   training loop can include them in the keep-set. Currently
   `Qwen35Model::cos_cache` and `sin_cache` are private (they're computed
   eagerly in `Qwen35Model::new` from `rope_cache_len_hint`). Add:

   ```rust
   impl Qwen35Model {
       pub fn cos_cache_id(&self) -> TensorId { self.cos_cache }
       pub fn sin_cache_id(&self) -> TensorId { self.sin_cache }
   }
   ```

2. **Build a keep-set once before the loop, prune after each `opd_step`:**

   ```rust
   use std::collections::HashSet;
   use autograd::TensorId;
   use train::trainer::{cleanup_after_backward, extend_keep_with_params_and_grads};

   let teacher_params = teacher.all_parameter_ids();
   let mut keep_extra: HashSet<TensorId> = HashSet::with_capacity(
       teacher_params.len() * 2 + 4
   );
   extend_keep_with_params_and_grads(&mut keep_extra, teacher_params.iter().copied(), &store);
   keep_extra.insert(student.cos_cache_id());
   keep_extra.insert(student.sin_cache_id());
   keep_extra.insert(teacher.cos_cache_id());
   keep_extra.insert(teacher.sin_cache_id());

   for step in 1..=args.steps {
       let outcome = opd_step(/* ... */)?;
       // ... output
       cleanup_after_backward(&mut store, &mut tape, &student_params, &keep_extra);
   }
   ```

`cleanup_after_backward` (`crates/train/src/trainer.rs:671`) already does
the right thing: `tape.entries.clear()`, `tape.set_enabled(true)`, then
`store.retain_ids(student_params ∪ keep_extra ∪ student-grad-ids)`.

`run_opd_smoke` (`crates/cli/src/train_cli.rs:311`) has the same loop
shape and needs the same fix.

## Why this matters before any perf axis

Per `CLAUDE.md` §0 SOLID — wall-clock framing as ground truth — every
perf bench that runs more than 1 step is contaminated by this leak:
later steps incur store-traversal cost proportional to the leaked
footprint, late steps are slower due to allocator pressure, and the OOM
ceiling is artificially close. **No `lm_head` perf measurement is
trustworthy until this is fixed.**

## Hand-off

- Codex appears to be editing `crates/cli/src/train_cli.rs` (M status as
  of survey time). If that intervention covers this fix, the hand-off is
  automatic — codex will commit + push and this doc captures the
  diagnosis. If codex's WIP is doing something else, this is the next
  patch.
- Either way: a regression test would be valuable — instrument
  `TensorStore::len()` (or live-id count) before and after a 5-step
  `opd_step` loop and assert the count doesn't grow by ≥ N per step.
  Belongs alongside `test_opd_determinism`.

## Cross-links

- Production OPD entry: `crates/cli/src/train_cli.rs:197-308` (`run_opd`)
- Production OPD smoke entry: `crates/cli/src/train_cli.rs:311+`
- OPD step body: `crates/train/src/opd.rs:85-151`
- Cleanup helper: `crates/train/src/trainer.rs:671-682`
- Parent perf work: commit `7aa11d7` (`perf(train): forward_last_logits
  rollout path — pending Qwen3-0.6B verification`) — surfaced this bug
  while debugging cooperative-session OOM during the A/B bench.
