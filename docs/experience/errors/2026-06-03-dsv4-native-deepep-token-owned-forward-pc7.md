# DSv4 Native DeepEP Token-Owned Forward PC7

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44 s, TPOT ~4.85 ms, E2E ~7.7 s, output throughput ~196 tok/s.

PC6 proved the NativeDeepEp transport boots and attaches on all 8 ranks, but
the startup contract still fails closed before decode. Source review found the
next runtime blocker that would fire after the contract is relaxed: the main
decode FFN path still treated native DeepEP as incompatible with the current
route even when request/model rows had already been validated as token-owned
DP/EP shards.

## Root Cause

`forward_decode_batch_with_request` validated `DistributedRequestShard`
metadata, but then called the generic `forward_decode_batch` entry. That lost
the row-ownership proof before the FFN/MoE path.

The FFN native-DeepEP branch then had only a stale replicated-token hard guard:
`native-deepep cannot run on the current DSv4 replicated-token TP/EP route`.
For the SGLang owner-group path, that guard was no longer discriminating real
replicated-token calls from request-aware token-owned calls.

There was also a c=1 trap: `try_decode_batch` returned false for a single row,
forcing the request-aware path into the per-row fallback, which has no
request-ownership metadata. The 256K/1500 comparison can be a single-request
shape, so this would have blocked the target path even after startup was
otherwise fixed.

## Fix

Added an internal `forward_decode_batch_internal` helper with an explicit
`token_owned_rows_validated` bit.

- Plain `forward_decode_batch` passes `false`.
- `forward_decode_batch_with_request` passes `true` only after
  `validate_decode_batch_request_ownership` returns
  `token-owned-dp-ep-shard-validated`.
- `try_decode_batch` now allows `N == 1` to stay on the batch path when native
  DeepEP is requested and token-owned rows were validated.
- The native-DeepEP FFN guard now rejects only calls that reach
  dispatch/combine without the request-aware token-owned validation bit.

EAGLE/MTP remains fail-closed: internal MTP frozen-KV draft still does not
support native DeepEP, and full-decode graph capture/replay is still not
implemented.

## Verification

Local:

- `cargo fmt --check`
- `git diff --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check --tests -p infer --no-default-features --features cuda,no-cuda`

Added pure logic tests for:

- `native_deepep_c1_requires_request_aware_batch_only_when_validated`
- `native_deepep_dispatch_guard_requires_validated_rows`

Attempted local run:
`CUDARC_CUDA_VERSION=12080 cargo test -p infer --lib --no-default-features --features cuda,no-cuda native_deepep -- --nocapture`.
This failed at the final macOS arm64 link step due unresolved CUDA externs in
the broader `cuda,no-cuda` lib-test binary; the two new assertions were not
executed locally. `cargo check --tests` compiles them.

Remote DSv4 pod, `/data01/build/arle`, temp patch:

- Build:
  `RUSTUP_TOOLCHAIN=stable CUDARC_CUDA_VERSION=12080 ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1 ARLE_DEEPEP_DIR=/data01/build/DeepEP bash scripts/dsv4_fast_build.sh`
  passed, log `/tmp/dsv4_pc7_token_owned_forward_build.log`.
- Probe artifact: `/tmp/dsv4_pc7_token_owned_forward_probe_1780454021`.
- Probe status: `STATUS=101`, expected fail-closed startup contract.
- Evidence counts:
  - `native_deepep_booted=8`
  - `native_transport_attached=8`
  - `moe_transport_layout=8`
  - `collectives_ready_layout=0`
  - `moe_transport_blocker=8`
  - `deep_ep_capture_blocker=8`
  - `token_sync_missing=0`
  - `row_ownership_blocker=0`
  - `replicated_guard_blocker=0`

The blocker did not regress to request routing, row ownership, missing
NativeDeepEP, or the direct replicated-token guard. It remains the truthful
next gaps: full-decode CUDA graph, DeepEP/NCCL capture/replay, EAGLE graph
safety, and FlashMLA/SWA/C4/C128 metadata replay.

## Rule

Do not relax DSv4 native DeepEP guards by deleting them. Carry the row-ownership
evidence to the exact dispatch/combine callsite, and keep direct
replicated-token decode blocked.
