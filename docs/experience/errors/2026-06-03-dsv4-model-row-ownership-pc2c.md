# DSv4 Model Row Ownership PC2c

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44 s, TPOT ~4.85 ms, E2E ~7.7 s, output throughput ~196 tok/s.

PC2b allowed the explicit SGLang profile to carry advanced axes as startup
contract input. The remote startup contract then proved request routing was
`token-owned-dp-ep` with owner groups `[[0, 1], [2, 3], [4, 5], [6, 7]]`, but
the model contract still reported `model_row_ownership=replicated-token`.

## Root Cause

DeepSeek forward received `distributed_shard` metadata, but it did not validate
that token-owned decode rows matched the local rank's attention-owner shard.
The startup contract therefore had to keep the row-ownership blocker, even
after request routing became token-owned.

## Fix

Add model-side token-owned row validation:

- derive the local expected request shard from `RankCoord` and SGLang axes:
  `rank = attn_cp_rank * attn_tp_size + attn_tp_rank`,
  `world = attn_tp_size * attn_cp_size`
- validate every token-owned decode row has the expected shard rank/world and
  `emits_visible_output == (shard_rank == 0)`
- reject mixed replicated/token-owned rows in one decode batch
- make the startup contract report
  `model_row_ownership=token-owned-dp-ep-shard-validated` only when the
  scheduler owner-group count matches model axes

This does not implement owner-group communicators, full-decode CUDA graph
capture, EAGLE graph capture, or metadata replay. It removes exactly the
row-ownership blocker from the fail-closed list.

## Verification

Local checks:

- `cargo fmt --check`
- `git diff --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check --tests -p infer --no-default-features --features cuda,no-cuda`

Remote build:

- source: `/data01/build/arle @ 1a9bf3a7` with
  `/tmp/dsv4_pc2c_row_ownership.patch` applied
- build log: `/tmp/dsv4_pc2c_row_ownership_build.log`
- build result: release-fast passed in 21.52 s, `prebuilt fast path used`

Remote startup probe:

- artifact: `/tmp/dsv4_pc2c_row_ownership_probe_1780450298`
- status: startup fail-closed, `STATUS=101`
- env: `ARLE_DSV4_PERFORMANCE_PROFILE=sglang`,
  `ARLE_TP_SIZE=8`, `ARLE_EP_SIZE=8`, `ARLE_ATTN_DP_SIZE=4`,
  `ARLE_MULTIPROC_SERVE=1`, native DeepEP, DeepGEMM, FP8 KV, FlashMLA
  prefill/decode, shared KV pool
- path contract:
  `workers=8 axes=tp=8 pp=1 ep=8 attn_dp=4 attn_cp=1 attn_tp=2 moe_dp=1 moe_tp=1 world=8`
- request owner groups:
  `groups=[[0, 1], [2, 3], [4, 5], [6, 7]]`
- every startup contract row reports:
  `request_ownership=token-owned-dp-ep`,
  `model_row_metadata=decode-batch-distributed-shard-visible`,
  `model_row_ownership=token-owned-dp-ep-shard-validated`,
  `request_effective_world_size=8`, and `token_owner_groups=4`
- row-ownership blocker count:
  `grep -c "DSv4 model forward must consume token-owned distributed_shard rows" = 0`
- after cleanup, `nvidia-smi --query-compute-apps` reported no compute apps

Remaining fail-closed blockers:

- `CUDA graph decode must be full_decode`, still currently `piecewise_decode`
- `DSv4 LayerCommunicator is global-tp-ep-only...`
- `DSv4 DeepEP/NCCL collective capture/replay contract is not implemented`
- frozen-KV EAGLE draft remains eager-only
- graph-captured FlashMLA/SWA/C4/C128 metadata replay is not implemented
- decode attention cache/metadata core still loops per row

## Rule

Do not call token-owned request routing a DSv4-Flash TP8 + EAGLE performance
pass. It is only comparable to the 256K/1500 hot-cache target after owner-group
communicators, full-decode graph capture, EAGLE graph safety, and metadata
replay are all present in the same serving run.
