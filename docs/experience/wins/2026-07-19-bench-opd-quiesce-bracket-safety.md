# OPD quiesce bracket safety — CUDA, 2026-07-19

> Status: pending-remote

## Goal

Prevent OPD group-end resource release or admission resume while an old forward is live or KV re-acquisition failed.

## Hypothesis

Drain executor inflight work during quiesce, abort submissions queued before the transition, and combine KV ensure + resume in one ordered engine-thread control.

## Parameters

Remote gate: agent-OPD group-end on the H20 CUDA lane, including an injected KV ensure failure and a request queued across quiesce.

## Environment

- Local host: Apple Silicon, no CUDA device
- Base: `8c43159f7`
- Backend/model: CUDA Qwen3.5/3.6 OPD student; remote run pending
- Local gates: GPU-free infer-server behavior tests and `cuda,no-cuda` type checks

## Results

| gate | result |
|---|---|
| pre-transition queued submission | aborted, not deferred |
| quiesce with poll pending once | waits until `has_inflight == false`; request finishes `Abort` |
| KV ensure failure | error preserved; engine remains quiesced |
| infer-server release-fast tests | 78 passed |

Raw artifacts: local Cargo output only; no performance JSON/CSV because this is a control-path correctness fix and CUDA hardware is unavailable locally.

## Problems

The remote H20 OPD end-to-end gate was not run by request; W4/pod are out of scope.

## Learnings

Pending remote. Quiescence must cover executor inflight state, and admission resume must be causally downstream of successful KV re-acquisition.
