# DSv4 SGLang Row Ownership PC0 Guard

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44 s, TPOT ~4.85 ms, E2E ~7.7 s, output throughput ~196 tok/s.

The SGLang path-alignment plan requires token-owned request rows before native
DeepEP can be a comparable target path. A prior startup contract log printed
`request_ownership=token-owned-dp-ep`, but that only described the request
routing plan. It did not prove the DeepSeek model forward was consuming
distinct token-owned hidden rows.

## Root Cause

The current DSv4 model route is still replicated-token at model-forward time.
`DistributedRequestShard` can describe request ownership and visible-output
emission, but the DeepSeek forward path does not yet consume it to partition
the logical token rows across ranks before MoE/DeepEP.

That makes `request_ownership=token-owned-dp-ep` insufficient evidence. It is
only a control-plane plan unless the model route also reports token-owned row
ownership.

## Fix

Commit `5fa11e06707eda088aeacf2adc985ae5a45e9c02` adds an explicit startup
contract field:

`model_row_ownership=replicated-token`

For `ARLE_DSV4_PERFORMANCE_PROFILE=sglang`, the profile now fails closed when
that field is not token-owned:

`DSv4 model forward must consume token-owned distributed_shard rows before native DeepEP can be comparable, got model_row_ownership=replicated-token`

This is a truth guard, not a performance fix. It prevents the debug or partial
SGLang route from being counted against the hot-cache 256K/1500 target.

## Verification

Local checks for `5fa11e06` passed:

- `cargo fmt`
- `cargo fmt --check`
- `git diff --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`

Remote build passed with the DSv4 release-fast path:

- `/tmp/dsv4_row_ownership_5fa11e06_build.log`

Remote high-performance profile probe:

- artifact: `/tmp/dsv4_profile_probe_serial_1780446442`
- profile: `ARLE_DSV4_PERFORMANCE_PROFILE=sglang`
- MoE: `ARLE_DSV4_MOE_BACKEND=native-deepep`
- expert backend: `ARLE_DSV4_EXPERT_BACKEND=deepgemm`
- TP/EP: 8/8
- result: startup fail-closed, `STATUS=exited EXIT=101`
- all 8 ranks logged `request_ownership=token-owned-dp-ep model_row_ownership=replicated-token`
- all 8 ranks included the new fail-closed item above
- after the probe, no `target/release-fast/infer` process remained and
  `nvidia-smi --query-compute-apps` reported no compute apps

## Rule

For DSv4-Flash TP8 + EAGLE, SGLang-path comparability requires both request
ownership and model row ownership. Do not treat `request_ownership=token-owned`
as progress toward native DeepEP performance until the model forward consumes
token-owned `distributed_shard` rows and the 256K/1500 hot-cache workload clears
TTFT, TPOT, E2E, and output throughput together.
