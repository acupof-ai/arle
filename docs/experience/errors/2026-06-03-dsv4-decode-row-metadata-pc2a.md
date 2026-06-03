# DSv4 Decode Row Metadata PC2a

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44 s, TPOT ~4.85 ms, E2E ~7.7 s, output throughput ~196 tok/s.

PC1 exposed that request routing and communicator layout were visible at
startup, but the model-facing decode trait still received only `tokens` and
`slot_indices`. `ActiveRequest.distributed_shard` existed inside the scheduler,
yet DeepSeek forward could not inspect per-row request ownership.

## Root Cause

`ModelForward::forward_decode_batch` did not carry per-row distributed request
metadata. That made the startup contract correctly fail closed with:

`DSv4 model forward must consume token-owned distributed_shard rows before native DeepEP can be comparable`

The failure was an API boundary issue, not a DeepEP performance issue.

## Fix

Commit `b4744203` adds a PC2a wrapper:

- `DecodeBatchRequest { tokens, slot_indices, distributed_shards }`
- `ModelForward::forward_decode_batch_with_request(...)`
- scheduler decode launch now passes `ActiveRequest.distributed_shard` rows to
  the model-facing wrapper
- existing models keep default behavior through the old `forward_decode_batch`
- DeepSeek logs `model_row_metadata=decode-batch-distributed-shard-visible`
  while keeping `model_row_ownership=replicated-token`

This does not implement token-owned row execution. It only removes the hidden
API gap so the next tranche can consume the metadata intentionally.

## Verification

Local checks:

- `cargo fmt --check`
- `git diff --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check --tests -p infer --no-default-features --features cuda,no-cuda`

Remote caveat:

- the pod could not fetch GitHub over HTTPS:
  `Failed to connect to github.com port 443`
- SSH remote fetch also failed:
  `git@github.com: Permission denied (publickey)`
- therefore the remote validation used `/data01/build/arle @ c131aacb` with
  `/tmp/dsv4_pc2a_b4744203.patch` applied to the worktree
- this is compile/startup evidence only, not a clean remote git-HEAD pass

Remote build/probe:

- build log: `/tmp/dsv4_pc2a_b4744203_build.log`
- build result: release-fast DSv4 prebuilt path passed in 25.53 s
- probe artifact: `/tmp/dsv4_profile_probe_serial_1780448414`
- profile: `ARLE_DSV4_PERFORMANCE_PROFILE=sglang`
- result: startup fail-closed, `STATUS=exited EXIT=101`
- all startup logs include
  `model_row_metadata=decode-batch-distributed-shard-visible`
- all startup logs still include
  `model_row_ownership=replicated-token`
- fail-closed list still includes:
  - `DSv4 model forward must consume token-owned distributed_shard rows...`
  - `DSv4 LayerCommunicator is global-tp-ep-only...`
  - `CUDA graph decode must be full_decode...`
  - `DSv4 loaded 1 mtp.N layer(s), but frozen-KV EAGLE draft is eager-only...`
  - `DSv4 DeepEP/NCCL collective capture/replay contract is not implemented`
- after the probe, no `target/release-fast/infer` process remained and
  `nvidia-smi --query-compute-apps` reported no compute apps

## Rule

Do not treat metadata visibility as row ownership. The next pass must prove
that DeepSeek consumes token-owned rows and maps them onto explicit owner-group
communicators before any DSv4-Flash TP8 + EAGLE performance number is
comparable to the 256K/1500 hot-cache target.
