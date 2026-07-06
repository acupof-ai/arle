# Plan — Collapse the opd_step overload chain into one entry + config struct

> Status: Active — 2026-07-06 · Driver: ckl (OPD review P6b). Deferred from the
> P6 body-split. Pod-gated (public-API break, no local cuda verification).

**Verdict:** worth doing, but pod-only — collapsing the chain rewrites 10+
CUDA-gated call sites that can't compile on this Mac.

## Problem
`opd.rs` has a 5-layer telescoping chain, each adding one param:
`opd_step_with_teacher_forward → _profiled → _profiled_gkd → _profiled_gkd_anchor`.
Every call site invokes the longest name directly — builder-by-positional-param.

## Why deferred
Collapsing rewrites CLI (`train_cli.rs:1004/1204/2916/3127`), examples, and 6
test sites, mostly `#[cfg(feature="cuda")]`. No nvcc here → can't verify. A public
break with unverifiable churn for a cosmetic payoff fails the bar.

## Approach (on the pod)
- `OpdStepExtras` (`#[derive(Default)]`) folding `forced_rollout`, `profile`,
  `gkd_config`, cuda-gated `infer_rollout`.
- One entry `opd_step_with_teacher_forward(..., extras: OpdStepExtras)`; keep
  `opd_step`; delete the three intermediates; rewrite call sites.

## Verify
`cargo build -p cli --features cuda` on the pod; `cargo test -p train
--features no-cuda`; a needle/loop regression round (byte-identity confounded by
MoE non-determinism — use the correct-inference gate).
