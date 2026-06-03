# DSv4 Communicator Contract PC1 Guard

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44 s, TPOT ~4.85 ms, E2E ~7.7 s, output throughput ~196 tok/s.

The SGLang path-alignment plan requires runtime-visible topology before native
DeepEP can be compared. `request_ownership=token-owned-dp-ep` and the
multi-axis `RankCoord` log are not enough if the model's actual
`LayerCommunicator` still carries only the global TP/EP collective contract.

## Root Cause

The DeepSeek runtime config already logs the requested multi-axis layout, but
`LayerCommunicator` is still constructed as global TP plus global EP:

`tp={rank}/8 dp=0/1 cp=0/1 ep={rank}/8`

That is a useful debug lane, but it is not proof that attention/MoE owner-group
communicators or token-owned row execution are wired. Without an explicit
communicator contract in the startup log, a run could look closer to the
SGLang path than it really is.

## Fix

Commit `c131aacbfaeed31cf2df664269aa855c4886acb1` adds:

- `LayerCommunicator::axis_summary()`
- `LayerCommunicator::layout_label()`
- DSv4 startup contract fields:
  `communicator_layout` and `communicator_axes`
- an additional SGLang-profile fail-closed item when the communicator is still
  `global-tp-ep-only`

The same commit also repairs CUDA-gated test request initializers so
`IncomingRequest` helpers explicitly use
`DistributedRequestShard::single_rank()`. That keeps the distributed ownership
API visible in test compilation instead of silently skipping stale helpers.

## Verification

Local checks:

- `cargo fmt --check`
- `git diff --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check --tests -p infer --no-default-features --features cuda,no-cuda`

The attempted local `cargo test -p infer --no-default-features --features cuda,no-cuda
axis_summary_and_layout_label_expose_current_contract` is not a valid Mac gate:
after compiling tests it links CUDA symbols and fails on the local
`/usr/local/cuda/lib64/stubs` absence. The useful outcome from that attempt was
finding and fixing the stale CUDA-gated `IncomingRequest` initializers above.

Remote verification:

- remote checkout: `/data01/build/arle @ c131aacb`
- build log: `/tmp/dsv4_comm_contract_c131aacb_build.log`
- build result: release-fast DSv4 prebuilt path passed in 17.13 s
- profile probe artifact: `/tmp/dsv4_profile_probe_serial_1780447731`
- profile: `ARLE_DSV4_PERFORMANCE_PROFILE=sglang`
- MoE: `ARLE_DSV4_MOE_BACKEND=native-deepep`
- expert backend: `ARLE_DSV4_EXPERT_BACKEND=deepgemm`
- TP/EP: 8/8
- result: startup fail-closed, `STATUS=exited EXIT=101`
- all startup logs include `communicator_layout=global-tp-ep-only`
- rank logs include `communicator_axes=tp={rank}/8 dp=0/1 cp=0/1 ep={rank}/8`
- fail-closed list includes both:
  `DSv4 model forward must consume token-owned distributed_shard rows...`
  and
  `DSv4 LayerCommunicator is global-tp-ep-only...`
- after the probe, no `target/release-fast/infer` process remained and
  `nvidia-smi --query-compute-apps` reported no compute apps

## Rule

SGLang-path comparability needs three separate proofs:

1. request routing says which ranks receive a logical request;
2. model row ownership proves which hidden rows each rank forwards;
3. communicator layout proves which attention/MoE owner groups are actually
   wired.

Do not treat any one of those as evidence for the others. Until all three are
true and the matched 256K/1500 hot-cache workload clears TTFT, TPOT, E2E, and
output throughput together, the result remains a contract gap, not a
performance win.
