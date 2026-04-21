# Bench Stub — KV Tier Runtime Flow Doc And Admission Unification

## Context

This tranche did two runtime-facing things on the CUDA lane:

- documented the live ownership graph and scheduler branch order in
  `docs/projects/tiered-kv-runtime-flow.md`
- centralized prefix admission ordering in
  `scheduler/cuda/runtime.rs::build_prefix_admission_plan()` so direct GPU
  attach, staged readmission, same-slot reuse, and cold-prefill fallback
  follow one canonical decision path

## What Worked

- `no-cuda` release `cargo check` passed after the runtime refactor.
- Release scheduler tests passed with the centralized admission planner.
- The runtime now keeps one explicit parked path (`Phase::WaitingFetch`) and
  the docs point to the exact implementation entrypoints.

## Rule

Status: `pending-remote`

Remote CUDA / guidellm validation is still required because this changes the
live scheduler path under `infer/src/scheduler/cuda/runtime.rs`.
