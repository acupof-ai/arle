# DSv4 Native DeepEP Transport PC6

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44 s, TPOT ~4.85 ms, E2E ~7.7 s, output throughput ~196 tok/s.

PC5 proved request owner token sync and attention-DP subgroup NCCL mapping for
the SGLang TP8 / attention-DP4 / attention-TP2 shape. The startup contract
advanced to `communicator_layout=owner-groups-attn-subgroups-ready`, but the
MoE/native-DeepEP part of the contract was still only an env label.

## Root Cause

`ARLE_DSV4_MOE_BACKEND=native-deepep` made the startup contract print
`moe_backend=native-deepep`, but no code path called `NativeDeepEp::boot` or
`LayerCommunicator::with_native_deepep`. That means the model could claim the
native-DeepEP backend label without proving a DeepEP `Buffer` was created,
IPC handles were exchanged, or the transport was reachable from
`LayerCommunicator`.

## Fix

Under the SGLang best-practice profile, `layer_communicator_from_config` now
boots `NativeDeepEp` from the EP NCCL group when
`ARLE_DSV4_MOE_BACKEND=native-deepep` is set, then attaches it to
`LayerCommunicator`.

Debug fallback still rejects `native-deepep`; the boot is allowed only for the
fail-closed SGLang contract path where token-owned rows and owner-group axes
are already being validated.

`LayerCommunicator::layout_label()` now advances to
`owner-groups-moe-transport-ready` only when request token sync,
non-singleton attention-DP/CP subgroups, EP NCCL, and a booted NativeDeepEp
transport are all present.

The startup contract still fails closed because booted transport is not
DeepEP/NCCL graph capture/replay, EAGLE graph safety, full-decode CUDA graph,
or metadata replay.

## Verification

Local:

- `cargo fmt`
- `cargo fmt --check`
- `git diff --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check --tests -p infer --no-default-features --features cuda,no-cuda`

Remote DSv4 pod, `/data01/build/arle`, temp patch:

- Build:
  `RUSTUP_TOOLCHAIN=stable CUDARC_CUDA_VERSION=12080 ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1 ARLE_DEEPEP_DIR=/data01/build/DeepEP bash scripts/dsv4_fast_build.sh`
  passed, log `/tmp/dsv4_pc6_native_deepep_build.log`.
- Probe artifact: `/tmp/dsv4_pc6_native_deepep_probe_1780453193`.
- Probe status: `STATUS=101`, expected fail-closed startup contract.
- Evidence counts:
  - `native_deepep_booted=8`
  - `native_transport_attached=8`
  - `moe_transport_layout=8`
  - `moe_transport_blocker=8`
  - `deep_ep_capture_blocker=8`
  - `token_sync_missing=0`
  - `row_ownership_blocker=0`

The blocker advanced from missing native DeepEP transport to the next truthful
gap: the native DeepEP transport is booted, but DeepEP/NCCL collective
capture/replay is not wired.

## Rule

Backend labels are not transport evidence. For DSv4 native DeepEP, startup
must prove the `NativeDeepEp` Buffer booted and was attached before any later
MoE/capture result can be attributed to native DeepEP.
