# Plan — Collapse the opd_step overload chain into one entry + config struct

> Status: Active — 2026-07-06 · Driver: ckl (OPD review P6b). Deferred from the
> P6 refactor (which split the internal function body but left the public
> overload chain intact). Plan only — this is a public-API break with no local
> behavior verification (CUDA-gated), so it needs the pod, not a Mac.

## Context / problem

`crates/train/src/opd.rs` exposes a **5-layer overload chain**, each layer
adding one parameter and forwarding to the next:

```
opd_step
  → opd_step_with_teacher_forward           (+ forced_rollout)
  → …_profiled                              (+ profile)
  → …_profiled_gkd                          (+ gkd_lambda)
  → …_profiled_gkd_anchor                   (+ gkd_config, infer_rollout)
```

The naming is telescoping and every call site invokes the longest name
(`…_gkd_anchor`) directly. This is builder-by-positional-parameter — the exact
readability smell the P6 body-split reduced but did not remove.

## Why deferred (not done in the P6 pass)

Collapsing means rewriting **10+ call sites** — CLI (`train_cli.rs:1004/1204/
2916/3127`), examples (`opd_step_cuda_*`), and 6 test sites — most behind
`#[cfg(feature="cuda")]`. On this Mac (no nvcc) the CUDA call sites can't be
compiled, let alone behavior-verified. A public-API break with unverifiable
call-site churn, for a cosmetic payoff, fails the "correct inference ≠ baseline
identity, but don't ship unverified CUDA churn" bar.

## Proposed approach (when on the pod)

- Introduce `OpdStepExtras` with `#[derive(Default)]`, folding `forced_rollout`,
  `profile`, `gkd_config` (already a struct), and the cuda-gated `infer_rollout`
  into named fields.
- One public entry: `opd_step_with_teacher_forward(student, teacher, prompt_ids,
  cfg, student_params, optimizer, store, tape, extras: OpdStepExtras)`.
- Keep `opd_step` (the smoke wrapper) as-is.
- Rewrite all call sites to construct `OpdStepExtras { ..Default::default() }`.
- Delete the three intermediate wrappers.

## Verification (pod-required)

`cargo build -p cli --features cuda` on the pod; `cargo test -p train
--features no-cuda` for the non-cuda arms; a needle/loop regression round on the
pod to confirm behavior identity before/after (byte-identical is confounded by
MoE non-determinism — use the correct-inference gate). Bench-exempt (mechanical
API reshape) but the pod regression round is the ship gate.

## Risk

Low logic risk (pure signature reshape) but high verification cost off-pod —
which is precisely why it's a separate pod-gated task, not folded into the
local P6 body-split.
