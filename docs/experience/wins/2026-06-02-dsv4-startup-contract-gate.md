# DSv4 startup contract gate for SGLang-path claims

## Context

DSv4 optimization was drifting into local operator tuning while the executable
path still differed from SGLang's best-practice path. A launch could silently
serve on the replicated-token debug lane and still look like a performance run.

## What Worked

- Added `ARLE_DSV4_PERFORMANCE_PROFILE=sglang` and `ARLE_DSV4_HIGH_PERF=1` as
  explicit high-performance declarations, with the old
  `ARLE_DSV4_SGLANG_PATH=1` kept as an alias.
- Added a generic `ModelForward::validate_scheduler_contract` hook so models
  can validate scheduler-owned knobs before slot state allocation.
- DSv4 now logs the startup contract: profile, fallback lane, topology, KV pool
  format, CUDA graph support/reason, MoE backend, expert backend, FlashMLA
  gates, shared KV pool, and incremental KV.
- In high-performance profile, DSv4 fails fast if the SGLang best-practice
  contract is not active. The current binary still reports token-owned DP/EP
  sharding and graph-captured DSv4 metadata as missing, so this is a guardrail,
  not a performance claim.
- Follow-up cleanup removed the stale `main.rs`-local
  `ARLE_DSV4_SGLANG_PATH` validator. Startup now logs the unified DSv4
  performance profile and lets the model/scheduler contract own fail-fast
  behavior, so `ARLE_DSV4_PERFORMANCE_PROFILE=sglang`,
  `ARLE_DSV4_HIGH_PERF=1`, and the old alias cannot drift apart.
- PC1 follow-up initializes DSv4 thread-local `ParallelState` during both
  model load and scheduler execution when a distributed DSv4 rank is
  configured. Startup now prints the SGLang-style TP, attention TP/DP/CP, and
  MoE EP/DP/TP rank groups. No fake NCCL subgroup is attached; unsupported
  subgroup execution still fails closed.
- PC2 precondition follow-up makes `DistributedSchedulerGroup` publish its
  request ownership contract. The current multiproc/in-process group logs and
  exposes `replicated-token`; token-owned DP/EP sharding remains a distinct
  missing feature instead of being hidden under a generic distributed name.
- PC2 ownership follow-up removes the hidden replicated-token constructor
  default. Every `DistributedSchedulerGroup` caller now passes an explicit
  ownership mode; the new `token-owned-dp-ep` mode fails closed until real
  request sharding is implemented.
- PC2 shard-contract follow-up adds scheduler-visible
  `DistributedRequestShard` metadata to every `IncomingRequest` and
  `ActiveRequest`. The current in-process and relay fanout now mark each rank
  as `replicated-token rank=N/world_size`; normal single-rank requests mark
  `single_rank()`. This does not implement token-owned DP/EP yet, but it
  removes another hidden assumption that distributed requests are only an
  unlabelled full-rank broadcast.
- PC2 owner-group follow-up starts deleting the wrong request topology instead
  of tuning it. `DistributedSchedulerGroup` now has explicit SGLang-style token
  owner groups for `token-owned-dp-ep`: one owner group is selected per request,
  only ranks in that group receive the request, and ranks in other DP groups do
  not see the logical request. Multiprocess serving remained blocked until the
  relay could address selected owner ranks and return output from a non-rank0
  owner.
- PC2 relay follow-up removes the unsafe accept-order assumption from the
  multiprocess request relay. Worker connections now send an explicit
  rank/world-size hello, the coordinator stores streams by rank, and the relay
  exposes targeted send by global rank. This change proved the control plane
  could address a selected DP-owner rank set without broadcasting to all ranks;
  output return and owner-group communicators remained separate follow-ups.
- PC2 remote-owner output follow-up adds the missing relay return path for
  token-owned control-plane requests. Workers can now send
  `CompletionStreamDelta` envelopes back to the coordinator, the coordinator
  dispatches them by `request_id`, and a guarded token-owned relay route can
  send a single-rank owner request to a remote rank without touching rank0's
  scheduler queue. At that stage, multi-rank owner groups remained a follow-up
  because the SGLang-compatible owner-group NCCL/token-sync subgroup contract
  was not separated from the MoE EP communicator.
- PC2 request-sync cleanup deletes the wrong communicator contract. Scheduler
  request coordination no longer reaches through `ep_nccl()`; models now expose
  an explicit request token-sync NCCL capability, and `SchedulerHandle` carries
  only that request-sync group. This keeps MoE EP transport separate from
  request token broadcast, matching the SGLang distinction between owner groups
  and EP data movement.
- PC2 multi-rank relay follow-up removes the single-rank-only relay half-state.
  A token-owned relay request can now target every rank in the selected owner
  group, mark only group rank 0 as visible-output, send remote follower ranks
  targeted `RequestOwned` envelopes, and submit the local rank only after
  remote envelopes have been sent. NCCL builds fail closed if a multi-rank
  local owner request lacks request token-sync NCCL. If a local submit fails
  after a remote visible-output completion sink was registered, the path
  unregisters that sink before returning `SubmitError`.
- High-performance startup messages now name the remaining structural blockers
  precisely: DSv4 startup still selects the replicated-token lane, owner-group
  NCCL/token-sync subgroup construction from the SGLang axis layout is missing,
  the token-owned relay path is not yet selected by DSv4 startup, batched
  FlashMLA sparse/recent decode is not wired, and DSv4 metadata replay is not
  graph-captured.
- DeepGEMM is now the DSv4 runtime default expert backend, not
  `deepgemm-auto`. Missing or incompatible DeepGEMM now fails before serving
  unless the operator explicitly asks for `ARLE_DSV4_EXPERT_BACKEND=deepgemm-auto`
  or `native` as a debug fallback.
- The replicated-token native DeepEP escape hatch was removed. Unsafe
  `deepep_unsafe`/`unsafe_deepep` aliases are no longer accepted, and
  `ARLE_DSV4_MOE_BACKEND=native-deepep` is reserved for the future token-owned
  DP/EP request path instead of running on the measured-wrong replicated-token
  route.

## Verification

- `cargo fmt --check`
- `git diff --check`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda`
- `cargo check -p infer --no-default-features --features no-cuda`
- Remote DSv4 pod source was synced by bundle because pod-side GitHub HTTPS
  fetch failed. Follow-up source syncs verified clean remote HEADs through
  `5d4e62bf` and then `204db39f`.
- Remote build: `RUSTUP_TOOLCHAIN=stable bash scripts/dsv4_fast_build.sh`
  completed after one rebuild and harvested DSv4 CUDA artifacts; immediate
  repeat used prebuilt artifacts and finished in 4.92 s while skipping nvcc and
  TileLang AOT.
- Follow-up remote build after the required-DeepGEMM default change also used
  the prebuilt CUDA artifacts and finished in 26.42 s at `eb59f1c8`.
- Remote CUDA unit gate with the correct prebuilt env
  `ARLE_CUDA_KERNELS_PREBUILT_DIR=/data01/build/arle/target/dsv4-cuda-kernels-prebuilt`:
  `cargo test -p infer --lib --no-default-features --features cuda,nccl deepgemm_required_is_the_default_expert_backend -- --nocapture`
  passed 1/1 in 20.47 s and printed the prebuilt-artifact skip message.
- PC2 request-ownership follow-up at `5d4e62bf`: remote `dsv4_fast_build.sh`
  used the prebuilt CUDA artifacts, skipped nvcc/TileLang AOT, and finished in
  20.52 s. Remote CUDA unit gate
  `distributed_group_token_owned_mode_fails_closed_until_sharding_exists`
  passed 1/1 with `--features cuda,nccl`.
- Replicated-token DeepEP escape-hatch removal at `204db39f`: remote
  `dsv4_fast_build.sh` used the same prebuilt artifact path, skipped
  nvcc/TileLang AOT, and finished in 17.10 s. Remote CUDA unit gate
  `deepgemm_required_is_the_default_expert_backend` passed 1/1 and now asserts
  `deepep_unsafe` / `unsafe_deepep` are invalid. `strings
  target-pod/release-fast/infer` contains the new
  `native-deepep is reserved for the token-owned DP/EP request path` startup
  message and no old `ARLE_DSV4_NATIVE_DEEPEP_REPLICATED_TOKENS_UNSAFE` symbol.
- PC2 shard-contract local gate: `cargo fmt --check`, `git diff --check`,
  `cargo check -p infer --no-default-features --features no-cuda`,
  `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda`,
  and `cargo test -p infer --no-default-features --features no-cuda request_handle -- --nocapture`
  passed. The request-handle test set now includes
  `distributed_group_marks_replicated_rank_shards`, which proves rank 0 and
  rank 1 carry distinct replicated-token shard metadata.
- PC2 shard-contract remote gate at `c3fa510c`: the pod source was bundle-synced
  cleanly. `RUSTUP_TOOLCHAIN=stable bash scripts/dsv4_fast_build.sh` used the
  DSv4 prebuilt CUDA artifacts, skipped nvcc/TileLang AOT, harvested the
  artifact cache again, and finished in 26.95 s. The remote CUDA/NCCL unit gate
  `cargo test -p infer --lib --no-default-features --features cuda,nccl request_handle -- --nocapture`
  passed 10/10, including both the fail-closed token-owned mode test and the
  replicated-token rank-shard metadata test.
- PC2 owner-group local gate: `cargo fmt --check`, `git diff --check`,
  `cargo test -p infer --no-default-features --features no-cuda request_handle -- --nocapture`,
  `cargo check -p infer --no-default-features --features no-cuda`, and
  `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda`
  passed. The request-handle test set now includes
  `distributed_group_token_owned_routes_to_one_owner_group`, which proves two
  owner groups route two logical requests to separate rank groups without
  enqueueing the requests on non-owner ranks.
- PC2 owner-group remote gate at `27942061`: remote source fast-forwarded
  cleanly from GitHub. `RUSTUP_TOOLCHAIN=stable bash scripts/dsv4_fast_build.sh`
  used the DSv4 prebuilt CUDA artifacts, skipped nvcc/TileLang AOT, harvested
  the cache again, and finished in 21.41 s. The remote CUDA/NCCL unit gate
  `cargo test -p infer --lib --no-default-features --features cuda,nccl request_handle -- --nocapture`
  passed 11/11, including the new token-owned owner-group route test.
- PC2 relay local gate: `cargo fmt --check`, `git diff --check`,
  `cargo test -p infer --no-default-features --features no-cuda multiproc_relay -- --nocapture`,
  `cargo test -p infer --no-default-features --features no-cuda request_handle -- --nocapture`,
  `cargo check -p infer --no-default-features --features no-cuda`, and
  `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda`
  passed. The relay test set now includes
  `coordinator_targeted_send_reaches_only_selected_rank`, which proves a
  selected worker rank receives the envelope while another connected worker
  receives only EOF after coordinator drop.
- PC2 relay remote gate at `91747279`: remote source fast-forwarded cleanly.
  `RUSTUP_TOOLCHAIN=stable bash scripts/dsv4_fast_build.sh` used the DSv4
  prebuilt CUDA artifacts, skipped nvcc/TileLang AOT, harvested the cache
  again, and finished in 21.68 s. Remote CUDA/NCCL unit gates passed:
  `cargo test -p infer --lib --no-default-features --features cuda,nccl multiproc_relay -- --nocapture`
  passed 3/3, and
  `cargo test -p infer --lib --no-default-features --features cuda,nccl request_handle -- --nocapture`
  passed 11/11.
- PC2 remote-owner output local gate: `cargo fmt --check`, `git diff --check`,
  `cargo test -p infer --no-default-features --features no-cuda multiproc_relay -- --nocapture`,
  `cargo test -p infer --no-default-features --features no-cuda request_handle -- --nocapture`,
  `cargo check -p infer --no-default-features --features no-cuda`, and
  `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda`
  passed. The relay test set now includes
  `coordinator_dispatches_worker_completion_to_registered_sink`, and the
  request-handle test set now includes
  `distributed_group_token_owned_relay_routes_remote_owner_output`.
- PC2 remote-owner output remote gate at `d5ac7262`: remote source
  fast-forwarded cleanly from GitHub to the same HEAD. `RUSTUP_TOOLCHAIN=stable
  bash scripts/dsv4_fast_build.sh` used the DSv4 prebuilt CUDA artifacts,
  skipped nvcc/TileLang AOT, harvested the cache again, and finished in
  23.78 s. Remote CUDA/NCCL unit gates passed:
  `cargo test -p infer --lib --no-default-features --features cuda,nccl multiproc_relay -- --nocapture`
  passed 4/4, and
  `cargo test -p infer --lib --no-default-features --features cuda,nccl request_handle -- --nocapture`
  passed 12/12.
- PC2 request-sync cleanup local gate: `cargo fmt --check`, `git diff --check`,
  `cargo test -p infer --no-default-features --features no-cuda multiproc_relay -- --nocapture`,
  `cargo test -p infer --no-default-features --features no-cuda request_handle -- --nocapture`,
  `cargo check -p infer --no-default-features --features no-cuda`, and
  `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda`
  passed. The request-handle test set now includes
  `distributed_group_token_owned_relay_routes_multi_rank_owner_group`, which
  proves the relay can route one token-owned request to local rank 0 and remote
  rank 1 with shard metadata `0/2` and `1/2` while only group rank 0 emits
  visible output. The implementation also unregisters a remote visible-output
  completion sink if later local submit fails. CUDA/no-CUDA typecheck passed
  with pre-existing DSv4 warnings.
- The same test command with the wrong env name
  `ARLE_CUDA_PREBUILT_ARTIFACTS` was stopped after `ps` showed it had fallen
  back to `nvcc`; the valid fast-path env is `ARLE_CUDA_KERNELS_PREBUILT_DIR`.
- Debug/fallback startup contract log:
  `/tmp/dsv4_startup_contract_20260602_095448.log`.
  The log shows `profile=debug-fallback`, `fallback_lane=allowed-debug-only`,
  `kv_pool_format=FP8E4M3`, `cuda_graph_supported=false` with the replicated
  TP/EP graph-safety reason, `moe_backend=allreduce`,
  `expert_backend=deepgemm`, `flashmla_prefill=true`,
  `flashmla_decode=true`, `shared_kv_pool=false`, and
  `incremental_kv=true`.
- High-performance fail-fast log:
  `/tmp/dsv4_highperf_failfast_clean_20260602_095547.log`.
  `ARLE_DSV4_PERFORMANCE_PROFILE=sglang` exited non-zero (`status=101`) before
  serving and named the missing token-owned DP/EP sharding and batched FlashMLA
  contract.

The remote rustup `1.95.0` toolchain currently has a `cargo-fmt` conflict; the
remote validation used `RUSTUP_TOOLCHAIN=stable` (`cargo 1.92.0`) to avoid
triggering rustup component reinstall during the build.

## Rule

High-performance DSv4 launches must fail before serving if they are actually on
the debug/fallback path. Performance numbers are not comparable unless the
startup contract proves the intended topology, KV layout, graph mode, DeepEP,
and DeepGEMM path are active.
