# DSv4 Owner Collectives Readiness PC8

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44 s, TPOT ~4.85 ms, E2E ~7.7 s, output throughput ~196 tok/s.

PC5 attached request token sync and attention-DP/CP subgroups. PC6 booted and
attached NativeDeepEp on all ranks. PC7 carried request-aware token-owned row
validation to the native DeepEP dispatch/combine callsite.

After those fixes, `LayerCommunicator::owner_group_collectives_ready()` was
still hard-coded to `false`. The startup contract therefore kept reporting
`owner-groups-moe-transport-ready`, mixing a solved runtime-collective
readiness problem with unsolved graph capture/replay work.

## Root Cause

The old readiness label bundled two different claims:

- Runtime owner-group collectives are wired and can reach native DeepEP
  dispatch/combine.
- Full-decode CUDA graph capture/replay has made those collectives graph-safe.

The second claim is still false, but the first claim became true after PC5-PC7.
Leaving `owner_group_collectives_ready()` as hard `false` hid the real current
blocker and made future probes less diagnostic.

## Fix

`owner_group_collectives_ready()` now returns
`owner_group_moe_transport_ready()` for non-trivial owner-group layouts.

This requires:

- request token sync NCCL,
- attention-DP/CP subgroup NCCL where non-singleton,
- EP NCCL,
- booted NativeDeepEp transport.

The startup contract still fails closed on the separate blockers:

- full-decode CUDA graph,
- DeepEP/NCCL capture/replay,
- EAGLE graph safety,
- FlashMLA/SWA/C4/C128 metadata replay,
- per-row attention cache/metadata planning.

## Verification

Local:

- `cargo fmt --check`
- `git diff --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check --tests -p infer --no-default-features --features cuda,no-cuda`

Remote DSv4 pod, `/data01/build/arle`, temp patch:

- Build:
  `RUSTUP_TOOLCHAIN=stable CUDARC_CUDA_VERSION=12080 ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1 ARLE_DEEPEP_DIR=/data01/build/DeepEP bash scripts/dsv4_fast_build.sh`
  passed, log `/tmp/dsv4_pc8_collectives_ready_build.log`.
- Probe artifact: `/tmp/dsv4_pc8_collectives_ready_probe_1780454299`.
- Probe status: `STATUS=101`, expected fail-closed startup contract.
- Evidence counts:
  - `native_deepep_booted=8`
  - `native_transport_attached=8`
  - `collectives_ready_layout=8`
  - `moe_transport_layout=0`
  - `moe_transport_blocker=0`
  - `deep_ep_capture_blocker=8`
  - `token_sync_missing=0`
  - `row_ownership_blocker=0`
  - `replicated_guard_blocker=0`

The startup contract advanced from `owner-groups-moe-transport-ready` to
`owner-groups-collectives-ready`. Remaining blockers are now cleanly limited
to full-decode CUDA graph, DeepEP/NCCL capture/replay, EAGLE graph safety, and
FlashMLA/SWA/C4/C128 metadata replay.

## Rule

Readiness labels should name the exact layer that is ready. Do not keep
runtime transport readiness false just because graph replay remains false.
