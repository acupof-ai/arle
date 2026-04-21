# Bench Stub — KV Tier Cluster Backend Unification

## Context

This tranche unified the scheduler/coordinator remote slower-tier surface onto
an explicit `cluster_shared_backend` config and `ClusterSharedBackend` runtime
wrapper, added backend-level existence checks for idempotent remote stores, and
 routed shared-fs remote fetch/store through the same backend contract used by
future NIXL-backed paths.

## What Worked

- Local `no-cuda` verification passed for the new backend surface, coordinator
  remote round-trip, shared-fs backend tests, and `rdma-nixl` compile gate.
- The coordinator now short-circuits repeated remote stores for the same
  fingerprint when the cluster-shared backend already has the payload.

## Rule

Status: `pending-remote`

Remote CUDA / guidellm validation is still required because this changes the
runtime control plane under `infer/src/scheduler/cuda/*` and `infer/src/kv_tier/*`.
