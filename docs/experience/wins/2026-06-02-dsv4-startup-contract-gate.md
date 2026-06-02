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
  fetch failed; remote HEAD verified through the latest source sync as
  `eb59f1c8`.
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
