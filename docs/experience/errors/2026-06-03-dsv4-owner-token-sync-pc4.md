# DSv4 Owner Token Sync PC4

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44 s, TPOT ~4.85 ms, E2E ~7.7 s, output throughput ~196 tok/s.

PC3 made the startup contract declare SGLang owner-group axes:
`tp={0,1}/2 dp={0..3}/4 cp=0/1 ep={0..7}/8`, but the fail list still
included `DSv4 owner-group token sync NCCL communicator is not attached`.

## Root Cause

The only request-token NCCL communicator previously attached to
`LayerCommunicator` was the global TP group. Under the SGLang shape, the
request owner groups are `[[0, 1], [2, 3], [4, 5], [6, 7]]`, so the scheduler
constructs `DistributedRequestCoordination::new_nccl(shard_rank, 2, nccl)`.
Passing a global TP8 communicator would fail the rank/world check and would be
semantically wrong.

## Fix

For non-global owner-group communicators, derive this rank's owner group from
`build_attn_owner_groups(config.axes)` and attach a dedicated request-token
NCCL group:

- group 0 `[0, 1]` uses `MASTER_PORT + 10`
- group 1 `[2, 3]` uses `MASTER_PORT + 11`
- group 2 `[4, 5]` uses `MASTER_PORT + 12`
- group 3 `[6, 7]` uses `MASTER_PORT + 13`

The layout label advances from `declared-owner-groups-no-collectives` to
`owner-groups-token-sync-ready`. The SGLang startup contract still fails
closed because token sync alone is not attention-DP/CP gather/scatter, MoE
owner-group collectives, DeepEP capture/replay, full-decode graph capture, or
EAGLE graph safety.

## Verification

Local checks:

- `cargo fmt --check`
- `git diff --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check --tests -p infer --no-default-features --features cuda,no-cuda`

Remote build:

- remote checkout: `/data01/build/arle @ e50f5f2f` with
  `/tmp/dsv4_pc4_owner_token_sync.patch` applied
- build log: `/tmp/dsv4_pc4_owner_token_sync_build.log`
- build result: release-fast passed in 18.22 s, `prebuilt fast path used`

Remote startup probe:

- artifact: `/tmp/dsv4_pc4_owner_token_sync_probe_1780451693`
- status: startup fail-closed, `STATUS=101`
- env: `ARLE_DSV4_PERFORMANCE_PROFILE=sglang`,
  `ARLE_TP_SIZE=8`, `ARLE_EP_SIZE=8`, `ARLE_ATTN_DP_SIZE=4`,
  `ARLE_MULTIPROC_SERVE=1`, native DeepEP, DeepGEMM, FP8 KV, FlashMLA
  prefill/decode, shared KV pool
- path contract:
  `workers=8 axes=tp=8 pp=1 ep=8 attn_dp=4 attn_cp=1 attn_tp=2 moe_dp=1 moe_tp=1 world=8`
- request owner groups:
  `groups=[[0, 1], [2, 3], [4, 5], [6, 7]]`
- owner-token sync attachments:
  `group_index=0 ranks=[0, 1] port_offset=10`,
  `group_index=1 ranks=[2, 3] port_offset=11`,
  `group_index=2 ranks=[4, 5] port_offset=12`,
  `group_index=3 ranks=[6, 7] port_offset=13`
- startup contract rows report:
  `communicator_layout=owner-groups-token-sync-ready`,
  `request_ownership=token-owned-dp-ep`,
  `model_row_ownership=token-owned-dp-ep-shard-validated`,
  `request_effective_world_size=8`, and `token_owner_groups=4`
- blocker counts:
  `LayerCommunicator is global-tp-ep-only = 0`,
  `declared-owner-groups-no-collectives = 0`,
  `owner-group token sync NCCL communicator is not attached = 0`,
  `DSv4 model forward must consume token-owned distributed_shard rows = 0`
- after cleanup, `nvidia-smi --query-compute-apps` reported no compute apps

Remaining fail-closed blockers:

- `CUDA graph decode must be full_decode`, still currently `piecewise_decode`
- `DSv4 LayerCommunicator is owner-groups-token-sync-ready...` because full
  attention-DP/CP and MoE owner-group collectives are not wired
- `DSv4 DeepEP/NCCL collective capture/replay contract is not implemented`
- frozen-KV EAGLE draft remains eager-only
- graph-captured FlashMLA/SWA/C4/C128 metadata replay is not implemented
- decode attention cache/metadata core still loops per row

## Rule

Request token-sync NCCL is necessary but not sufficient. Do not compare against
the 256K/1500 hot-cache DSv4-Flash TP8 + EAGLE target until full owner-group
collectives, graph-captured decode, EAGLE, and metadata replay execute in the
same serving run.
