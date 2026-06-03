# DSv4 SGLang Axes Contract PC2b

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44 s, TPOT ~4.85 ms, E2E ~7.7 s, output throughput ~196 tok/s.

PC2a made per-row `distributed_shard` metadata visible to
`forward_decode_batch_with_request`, but SGLang's richer DP/TP/EP axis layout
was still rejected before request-owner groups could be used as startup
contract input.

## Root Cause

The advanced-axis guard was too early and too broad. It correctly protected the
debug/default execution path from silently accepting a non-comparable SGLang
layout, but it also blocked `ARLE_DSV4_PERFORMANCE_PROFILE=sglang` from proving
the next contract point: request routing must be token-owned DP/EP before model
row execution and communicator wiring can be evaluated.

That meant `attn_dp` owner groups could not be validated in the fail-closed
SGLang lane.

## Fix

Allow advanced axes only as SGLang profile contract input:

- debug/default profiles still reject advanced multi-axis layouts
- SGLang profile may carry axes through config validation
- startup contract can derive request owner groups from
  `ARLE_TP_SIZE=8 ARLE_EP_SIZE=8 ARLE_ATTN_DP_SIZE=4`
- model execution still fails closed because row ownership remains
  `replicated-token` and the communicator layout is still `global-tp-ep-only`

This is not a performance pass. It only advances the fail-closed startup
contract to the next real blocker.

## Verification

Local checks:

- `cargo fmt --check`
- `git diff --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check --tests -p infer --no-default-features --features cuda,no-cuda`

Local limitation:

- `CUDARC_CUDA_VERSION=12080 cargo test -p infer --no-default-features --features cuda,no-cuda advanced_axes -- --nocapture`
  and the narrower `--lib` form both failed at Mac CUDA linking, not at the new
  assertions. The link failed because `/usr/local/cuda/lib64/stubs` and CUDA C
  symbols such as `_gemm_cuda`, `_fused_gqa_attention_decode`, and DSv4 kernel
  symbols are unavailable on this Mac. The CUDA test target still compile-checked
  through `cargo check --tests`.

Remote build:

- source: `/data01/build/arle @ 6d0a22de` with `/tmp/dsv4_pc2b_axes.patch`
  applied
- build log: `/tmp/dsv4_pc2b_axes_build.log`
- first build attempt failed before compilation because rustup tried to sync
  `1.95.0` components and hit a `cargo-fmt` conflict
- rerun used `RUSTUP_TOOLCHAIN=stable` plus the existing CUDA prebuilt artifact
  manifest env
- result: release-fast build passed in 28.68 s, `prebuilt fast path used`

Remote startup probe:

- artifact: `/tmp/dsv4_pc2b_axes_probe_1780449777`
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
  `model_row_ownership=replicated-token`,
  `request_effective_world_size=8`, and `token_owner_groups=4`
- after cleanup, `nvidia-smi --query-compute-apps` reported no compute apps

Remaining fail-closed blockers:

- `CUDA graph decode must be full_decode`, still currently `piecewise_decode`
- `DSv4 model forward must consume token-owned distributed_shard rows...`,
  still `model_row_ownership=replicated-token`
- `DSv4 LayerCommunicator is global-tp-ep-only...`
- `DSv4 DeepEP/NCCL collective capture/replay contract is not implemented`
- frozen-KV EAGLE draft remains eager-only
- graph-captured FlashMLA/SWA/C4/C128 metadata replay is not implemented
- decode attention cache/metadata core still loops per row

## Rule

SGLang axes reachability is not a throughput result. A DSv4-Flash TP8 + EAGLE
number is only comparable to the 256K/1500 hot-cache target after request
routing, model row ownership, owner-group communicators, full-decode graph
capture, and EAGLE graph safety all hold in the same run.
