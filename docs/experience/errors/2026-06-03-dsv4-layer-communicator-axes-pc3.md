# DSv4 Layer Communicator Axes PC3

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44 s, TPOT ~4.85 ms, E2E ~7.7 s, output throughput ~196 tok/s.

PC2c proved model decode rows can validate token-owned DP/EP request shards, but
the startup contract still reported
`communicator_layout=global-tp-ep-only communicator_axes=tp=N/8 dp=0/1 cp=0/1 ep=N/8`
under the SGLang axes `tp=8 attn_dp=4 attn_cp=1 attn_tp=2`.

## Root Cause

`DeepseekModel::layer_communicator_from_config` built `LayerCommunicator` from
the runtime-global TP/EP ranks only. That made the contract look like ARLE was
still using one global TP8 attention communicator, even when the request owner
groups were already `[[0, 1], [2, 3], [4, 5], [6, 7]]`.

Attaching the existing global TP NCCL group to an owner-group communicator would
be wrong: owner-group TP is world size 2 for this shape, while the existing TP
NCCL group is world size 8.

## Fix

Declare the communicator's attention axes from `MultiAxisConfig` and
`RankCoord`:

- TP = `attn_tp_rank / attn_tp_size`
- DP = `attn_dp_rank / attn_dp_size`
- CP = `attn_cp_rank / attn_cp_size`
- EP remains the existing expert rank/world

Only attach the global TP NCCL group when the communicator TP axis still
matches global TP. Advanced owner-group axes are labeled
`declared-owner-groups-no-collectives`, and the SGLang startup contract still
fails unless the communicator reaches `owner-groups-collectives-ready`.

This is a contract/diagnostic correction. It does not implement attention-DP/CP
token sync, MoE owner-group collectives, DeepEP capture/replay, full-decode
CUDA graph capture, EAGLE graph safety, or metadata replay.

## Verification

Local checks:

- `cargo fmt --check`
- `git diff --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check --tests -p infer --no-default-features --features cuda,no-cuda`

The attempted narrow test
`cargo test -p infer --no-default-features --features no-cuda axis_summary_and_layout_label_expose_current_contract`
matched zero tests, and `--lib -- --list | rg layer_communicator` also matched
nothing under this feature set. Do not count that as execution evidence.

Remote build:

- remote checkout: `/data01/build/arle @ c8788177` with
  `/tmp/dsv4_pc3_layer_comm.patch` applied
- build log: `/tmp/dsv4_pc3_layer_comm_build.log`
- build result: release-fast passed in 18.58 s, `prebuilt fast path used`

Remote startup probe:

- artifact: `/tmp/dsv4_pc3_layer_comm_probe_1780451174`
- status: startup fail-closed, `STATUS=101`
- env: `ARLE_DSV4_PERFORMANCE_PROFILE=sglang`,
  `ARLE_TP_SIZE=8`, `ARLE_EP_SIZE=8`, `ARLE_ATTN_DP_SIZE=4`,
  `ARLE_MULTIPROC_SERVE=1`, native DeepEP, DeepGEMM, FP8 KV, FlashMLA
  prefill/decode, shared KV pool
- path contract:
  `workers=8 axes=tp=8 pp=1 ep=8 attn_dp=4 attn_cp=1 attn_tp=2 moe_dp=1 moe_tp=1 world=8`
- request owner groups:
  `groups=[[0, 1], [2, 3], [4, 5], [6, 7]]`
- rank 0 communicator declaration:
  `communicator_axes=tp=0/2 dp=0/4 cp=0/1 ep=0/8 global_tp=0/8`
- startup contract rows report:
  `communicator_layout=declared-owner-groups-no-collectives`,
  `request_ownership=token-owned-dp-ep`,
  `model_row_metadata=decode-batch-distributed-shard-visible`,
  `model_row_ownership=token-owned-dp-ep-shard-validated`,
  `request_effective_world_size=8`, and `token_owner_groups=4`
- blocker counts:
  `LayerCommunicator is global-tp-ep-only = 0`,
  `declared-owner-groups-no-collectives = 24`,
  `DSv4 model forward must consume token-owned distributed_shard rows = 0`
- after cleanup, `nvidia-smi --query-compute-apps` reported no compute apps

Remaining fail-closed blockers:

- `CUDA graph decode must be full_decode`, still currently `piecewise_decode`
- `DSv4 LayerCommunicator is declared-owner-groups-no-collectives...`
- `DSv4 owner-group token sync NCCL communicator is not attached`
- `DSv4 DeepEP/NCCL collective capture/replay contract is not implemented`
- frozen-KV EAGLE draft remains eager-only
- graph-captured FlashMLA/SWA/C4/C128 metadata replay is not implemented
- decode attention cache/metadata core still loops per row

## Rule

Axis declaration is not performance readiness. Do not compare against the
256K/1500 hot-cache DSv4-Flash TP8 + EAGLE target until owner-group collectives
and graph-captured decode execute in the same measured serving run.
