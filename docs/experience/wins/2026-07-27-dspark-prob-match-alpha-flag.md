# DSpark --dspark-prob-match-alpha flag — unblocks the ISO premise sweep, 2026-07-27

> Status: flag wired + typechecks (CUDA/no-CUDA + infer-api); enables the H20 ISO
> premise sweep and the 7b PG-vs-prob-match separation. No perf claim.

## Goal

Make the DSpark objective mix `prob_match_alpha` settable at serve time, so the
plan's mandated ISO premise sweep (`α = 0 / 0.5 / 1`) can actually run.

## Context

The H20 verification of the ISO tranche (HEAD `2e162eff7`) cleared build + all
unit gates but **could not run STAGE 3** — the premise sweep the ISO win entry
and the Phase 5 license both require. Root cause was not a missing head (two are
staged) but a missing knob: `prob_match_alpha` was hardcoded to 0.5 via
`..Default::default()` at `serve.rs:229`, exercised at other values only by a
unit test. DSpark head training exists solely as a serve sidecar
(`--spec-type dspark --dspark-train`), so with no flag, `--prob-match-alpha 0`
(pure acceptance PG, the RLVR regime ISO's premise lives in) was impossible.

## What changed

`--dspark-prob-match-alpha <ALPHA>` threaded CLI → `ServeSpecOptions` → the
sidecar's `DsparkTrainConfig.prob_match_alpha` (defaults to 0.5 when unset, so
existing behavior is unchanged). The step log already prints `pm_alpha=<value>`
next to `iso_drift` (`dspark_train.rs`), so a sweep is now self-describing.

- `args.rs`: the CLI flag.
- `infer-api/src/serve.rs`: `ServeSpecOptions.dspark_prob_match_alpha`.
- `cli/src/serve.rs`: args→options mapping + config construction.

## Results

```text
CUDA/no-CUDA typecheck (cli): passed
infer-api check: passed
fmt + clippy: clean
```

## Problems

The sweep itself is H20 (pending-remote): (1) `α = 0/0.5/1` ISO-off drift sweep —
does the pure-PG arm show near-isospectral drift? (2) if so, the matched ISO A/B.
This flag removes the only blocker the devops run hit; the numbers are the next
remote step.

Also surfaced by the same run (separate tickets, not fixed here):
- 27B rollout ingress re-binds `max_prompt_tokens` to the profiled KV capacity
  (4096), below the ~21K CC agentic prompt → every rollout aborts at ingress
  (`loaded.rs:2311`). Blocks the DAPO mixed-corpus `groups_effective>0` case.
- No staged corpus+model pair produces within-group reward variance (the same
  all-zero-reward wall the GRPO tranche documented), so DAPO refill's live
  effective-fill path stays unexercised; its termination invariant is proven by
  the `all_dead_corpus_terminates_at_launch_cap` unit test and the live allzero
  arm (`launched=4 effective=0`, no step).

## Learnings

**A gate the plan mandates must be runnable, not just implementable.** The ISO
premise sweep was fully specified and the head was staged, but a hardcoded
default made the one axis it varies unreachable from the CLI. Wiring the knob is
cheaper than any amount of A/B design — verify the gate *runs* before optimizing
what it measures.
