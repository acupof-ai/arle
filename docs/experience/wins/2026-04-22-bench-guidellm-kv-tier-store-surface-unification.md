# Bench Stub — KV Tier Store Surface Unification

## Context

This tranche removed the legacy `Spill*` coordinator surface and kept one live
write-side path:

- `StoreRequest`
- `StoreTicket`
- `CoordinatorEvent::Store*`

Disk persistence and cluster-shared writes now share that same queue/event
contract, and the CUDA runtime only reacts to `StoreQueued / StoreCompleted /
StoreFailed`.

## What Worked

- Release `no-cuda` verification passed after deleting the redundant
  coordinator surface.
- Coordinator disk-store tests, shared-fs round-trip tests, scheduler tests,
  and clippy all passed with the unified store path.
- Runtime/docs naming now matches implementation: there is one parked fetch
  path and one store queue, not parallel spill/store APIs.

## Rule

Status: `pending-remote`

Remote CUDA / guidellm validation is still required because this changes the
live scheduler/coordinator control surface under `infer/src/kv_tier/*` and
`infer/src/scheduler/cuda/runtime.rs`.
