# DSv4 Attention-DP Subgroup Mapping PC5

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44 s, TPOT ~4.85 ms, E2E ~7.7 s, output throughput ~196 tok/s.

PC4 attached the request owner-group token sync communicator for the SGLang
shape. The startup contract advanced to
`communicator_layout=owner-groups-token-sync-ready`, but still failed closed
because attention-DP/CP and MoE owner-group collectives were not wired.

## Root Cause

The communicator could synchronize tokens within each request owner group
(`[[0, 1], [2, 3], [4, 5], [6, 7]]`) but had no NCCL subgroup for the
cross-DP attention layout. In the target TP8 / attention-DP4 / attention-TP2
shape, SGLang's attention-DP groups are `[[0, 2, 4, 6], [1, 3, 5, 7]]`.

Without those groups, the startup contract could not distinguish "token sync is
attached" from "the attention axis collectives are at least mapped." Treating
token sync as full owner-group collective readiness would be a false pass.

## Fix

`LayerCommunicator` now stores explicit attention-DP and attention-CP NCCL
subgroups in addition to the request token-sync group.

DeepSeek V4 model boot derives the current rank's attention-DP and
attention-CP groups from the SGLang axis math:

- request owner token-sync groups use `MASTER_PORT + 10 + group_index`
- attention-DP groups use `MASTER_PORT + 20 + group_index`
- attention-CP groups use `MASTER_PORT + 40 + group_index`

For the current target shape, attention-CP is size 1, and attention-DP attaches
two four-rank NCCL groups. The layout label advances to
`owner-groups-attn-subgroups-ready` only when request token sync plus all
non-singleton attention-DP/CP subgroups are present.

The startup contract still fails closed. Attention subgroup construction is not
MoE owner-group collective execution, DeepEP/NCCL capture/replay, full-decode
CUDA graph capture, EAGLE graph safety, or metadata replay.

## Verification

Local checks:

- `cargo fmt --check`
- `git diff --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check --tests -p infer --no-default-features --features cuda,no-cuda`

The attempted narrow unit-test filter
`cargo test -p infer --no-default-features --features no-cuda --lib axis_summary_and_layout_label_expose_current_contract`
matched zero tests and is not execution evidence.

Remote build:

- remote checkout: `/data01/build/arle @ c7072a51` with
  `/tmp/dsv4_pc5_attn_dp.patch` applied
- build log: `/tmp/dsv4_pc5_attn_dp_build.log`
- build result: release-fast passed in 27.15 s, `prebuilt fast path used`
- caveat: the first remote build accidentally picked `CUDARC_CUDA_VERSION=12090`
  and missed the prebuilt manifest; it was stopped before continuing the nvcc
  rebuild, then rerun with `CUDARC_CUDA_VERSION=12080`

Remote startup probe:

- artifact: `/tmp/dsv4_pc5_attn_dp_probe_1780452673`
- status: startup fail-closed, `STATUS=101`
- env: `ARLE_DSV4_PERFORMANCE_PROFILE=sglang`,
  `ARLE_TP_SIZE=8`, `ARLE_EP_SIZE=8`, `ARLE_ATTN_DP_SIZE=4`,
  `ARLE_MULTIPROC_SERVE=1`, native DeepEP, DeepGEMM, FP8 KV, FlashMLA
  prefill/decode, shared KV pool, `--spec-enabled --spec-draft-model eagle`
- attention-DP attachments:
  `group_index=0 ranks=[0, 2, 4, 6] port_offset=20` and
  `group_index=1 ranks=[1, 3, 5, 7] port_offset=21`
- startup contract rows report:
  `communicator_layout=owner-groups-attn-subgroups-ready`,
  `request_ownership=token-owned-dp-ep`,
  `model_row_ownership=token-owned-dp-ep-shard-validated`,
  `request_effective_world_size=8`, and `token_owner_groups=4`
- blocker counts:
  `attention-DP NCCL attached = 8`,
  `communicator_layout=owner-groups-attn-subgroups-ready = 8`,
  `communicator_layout=owner-groups-token-sync-ready = 0`,
  `owner-group token sync NCCL communicator is not attached = 0`,
  `owner-group token sync NCCL is attached but attention-DP/CP = 0`,
  `owner-group token sync and attention-DP/CP subgroups are attached = 8`,
  `DSv4 model forward must consume token-owned distributed_shard rows = 0`,
  `LayerCommunicator is global-tp-ep-only = 0`, and
  `declared-owner-groups-no-collectives = 0`
- after cleanup, `nvidia-smi --query-compute-apps` reported no compute apps

## Rule

SGLang axis mapping must be proven subgroup by subgroup. A request token-sync
group is not an attention-DP collective, and an attention-DP subgroup is still
not a full DSv4-Flash TP8 + EAGLE performance pass.
