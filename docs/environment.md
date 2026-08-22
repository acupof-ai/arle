# Environment Variables

This document lists the environment variables used by `ARLE` across runtime,
build, test, and setup workflows.

Current truth is simple: prefer `ARLE_*` for the `arle` front door, keep
`INFER_*` for build/test/runtime plumbing, and treat any remaining
`AGENT_INFER_*` names as compatibility-only.

---

## 0. Policy (2026-04-16, Tier C)

**Env vars are reserved for: build, test model paths, setup, and genuinely
debug/diagnostic runtime overrides.**

**Tuning knobs go on structs**, not env vars. The canonical example is
`SchedulerConfig` in `crates/infer-core/src/lib.rs`: prefix-cache
watermarks (`prefix_cache_high_water`, `prefix_cache_low_water`,
`prefix_cache_retain_hard_cap`), keepalive ticks
(`prefix_cache_keepalive_ticks`, `t1_host_pinned_keepalive_ticks`), and
chunking caps are struct fields with `validate()` guards. Callers that
want to tune them construct a `SchedulerConfig::runtime_defaults(..)`
and assign directly — **there is no `INFER_PREFIX_HIGH_WATER`** or
any other magic env var for runtime tuning. If you want an env-var
escape hatch for a specific tuning knob, justify it as a debug aid and
document the debug-only status here.

**Converted to CLI flags (2026-07-10)** — these env vars no longer exist; the
flag is the single surface (serve flags ride `EngineLoadConfig`, so multiproc
workers see them; train flags apply via `train::apply_runtime_flags`):

| Removed env var | Flag |
| --- | --- |
| `ARLE_QWEN35_DECODE_GRAPH` | `arle serve --qwen35-decode-graph` |
| `ARLE_QWEN35_BATCHED_DECODE` | `arle serve --qwen35-batched-decode` |
| `ARLE_QWEN35_DEEPGEMM` | `arle serve --qwen35-deepgemm` |
| `ARLE_QWEN35_MOE_DECODE_KERNEL` | `arle serve --qwen35-moe-decode-kernel` |
| `ARLE_QWEN35_GPU_ROUTER` | `arle serve --qwen35-gpu-router` |
| `ARLE_QWEN35_FA3` / `_FA3_DECODE_SPLITS` | `arle serve --qwen35-fa3` / `--qwen35-fa3-decode-splits` |
| `ARLE_QWEN35_GDR_CHUNKED` | `arle serve --qwen35-gdr-chunked` |
| `INFER_CUDA_DECODE_GRAPH` | removed with the dense Qwen3 CUDA path; Qwen3.5/3.6 use `--qwen35-decode-graph`, DSv4 `ARLE_DSV4_*` |
| `INFER_DECODE_METADATA_FAST_PAGE16` | `arle serve --decode-metadata-fast-page16` |
| `ARLE_CUDA_MEMPOOL_RETAIN` | `arle serve --cuda-mempool-retain` |
| `ARLE_CUDA_SHARD_CACHE_BYTES` | `arle serve --shard-cache-bytes` |
| `ARLE_NUMA_PIN` | `arle serve --numa-pin` |
| `ARLE_COMM_BACKEND` | `arle serve --comm-backend` (already existed; env transport removed) |
| `ARLE_DSV4_FLASHMLA_DECODE` | `arle serve --dsv4-flashmla-decode` |
| `ARLE_DSV4_DSA_INDEXER_SMS` | `arle serve --dsv4-dsa-indexer-sms` |
| `ARLE_DSV4_MTP_ADAPTIVE` / `ARLE_DSV4_MTP_MIN_ACCEPT` | `arle serve --mtp-adaptive` / `--mtp-min-accept` |
| `ARLE_DSV4_DEEPEP_NUM_SMS` | `arle serve --deepep-num-sms` |
| `ARLE_DSV4_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | `arle serve --deepep-max-dispatch-tokens-per-rank` (SGLANG env still honored when unset) |
| `INFER_METAL_PIPELINE` / `_WARMUP` / `_PAGED_KV_READ` / `_HOST_SAMPLING` | `arle serve --metal-pipeline` / `--metal-warmup` / `--metal-paged-kv-read` / `--metal-host-sampling` |
| `INFER_METAL_NO_SPECULATIVE` / `_DFLASH_DRAFT_MODEL` / `_DFLASH_TOKENS` / `_DFLASH_ACCEPT_TOPK` | `arle serve --no-speculative` / `--draft-model` / `--speculative-tokens` / `--spec-accept-topk` (env transport removed; flags ride `EngineLoadConfig.metal`) |
| `DFLASH_DRAFT_MASK` | removed (rewrite path is mask=none only) |
| `ARLE_DIFFUSION_MAX_DENOISING_STEPS` | `arle serve --diffusion-max-denoising-steps` |
| `ARLE_SUBMIT_CAP` | `arle serve --vulkan-submit-cap` |
| `ARLE_OPD_WRITEBACK_OFFLOAD` | `arle train <opd> --writeback-offload` |
| `ARLE_OPD_ENGINE_OFFLOAD` | `arle train <opd> --engine-offload off\|all\|student\|teacher` |
| `ARLE_OPD_GRADIENT_CHECKPOINTING` | `arle train <opd> --gradient-checkpointing` |
| `ARLE_OPD_CHECKPOINT_OFFLOAD_MIN_BYTES` | `arle train <opd> --checkpoint-offload-min-bytes` |
| `ARLE_OPD_ROLLOUT_RETAIN_INTERVAL` / `_ROLLOUT_PROGRESS_INTERVAL` | `arle train <opd> --rollout-retain-interval` / `--rollout-progress-interval` |
| `ARLE_OPD_MOE_LORA_BWD_EXPERT_TILE` / `_LORA_LINEAR_BWD_TILE_ROWS` | `arle train <opd> --moe-lora-bwd-expert-tile` / `--lora-linear-bwd-tile-rows` |
| `ARLE_OPD_WRITEBACK_FROZEN_PROMPT_KV` | `arle train <opd> --writeback-frozen-prompt-kv` |
| `ARLE_GDR_CHUNKWISE_PREFILL` / `ARLE_LA_BACKWARD_MONO` / `ARLE_AUTOGRAD_DECODE_ATTN_LEGACY` | `arle train <opd> --gdr-chunkwise-prefill` / `--la-backward-mono` / `--autograd-decode-attn-legacy` |

Deferred (read site inside frozen DSv4 files this pass):
`ARLE_DSV4_MOE_TRANSPORT`/`ARLE_DSV4_MOE_BACKEND`.

---

## 1. Naming Rule

- Prefer `ARLE_*` for newly documented user-facing CLI/runtime behavior.
- Treat `AGENT_INFER_*` as legacy compatibility names unless this document
 explicitly calls them out as the current canonical surface.
- Treat `INFER_*` primarily as build, test, or compatibility variables unless
 documented otherwise.
- Treat undocumented variables as internal or experimental.

---

## 1b. Cargo feature lanes (root `arle` package)

Source: root `Cargo.toml` `[features]`. Full notes:
[`onboarding.md`](onboarding.md) §4.

| Target | Command |
| --- | --- |
| Linux + NVIDIA full build | `cargo build --release --features cuda --bin arle` |
| Apple Silicon | `cargo build --release --no-default-features --features metal,no-cuda,cli --bin arle` |
| Mac CUDA typecheck (no GPU) | `CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --no-default-features --features cuda,no-cuda` |
| CPU smoke | `cargo build --release --no-default-features --features cpu,no-cuda,cli --bin arle` |
| Multi-GPU NCCL | `cargo build --release --features cuda,nccl --bin arle` |

`default = ["cli"]` — no backend by default; pick one explicitly.

**Mac typecheck lane coverage.** The `cuda,no-cuda` lane compiles only the
host-side stubs: enabling `no-cuda` removes every block guarded by
`#[cfg(not(feature = "no-cuda"))]`, which is exactly the code that runs on the
pod. A return-expression bug inside such a block (fixed `e4f0a3017`) passed
this lane green and broke the pod build. When a change touches code inside
`cfg(not(feature = "no-cuda"))`, local green is not evidence; the authoritative
gate runs on the pod: `cargo check -p autograd --release --features cuda --lib`
(~5 min).

**Set `CARGO_TARGET_DIR` per lane.** Feature unification makes the lanes
clobber each other's artifacts in a shared `target/`; alternating lanes pays a
full dep rebuild every switch. Pin one dir per lane (disk ~3×, pruned by
`cargo sweep`):

```bash
CARGO_TARGET_DIR=target/lane-cuda CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
CARGO_TARGET_DIR=target/lane-metal cargo test -p cli --release --no-default-features --features metal,no-cuda
CARGO_TARGET_DIR=target/lane-cpu cargo test -p arle --release --no-default-features --features cpu,no-cuda,cli
```

Host-only unit tests (cli / infer-api) may drop `--release`: debug builds are
several times faster, deps are already `debug = false`, and the `--release`
mandate exists for GPU builds only.

---

## 2. User-Facing Runtime Variables

### `ARLE_LOG_DIR`

Rolling log-file directory, default `logs` (relative to CWD). Every `arle`
invocation writes stderr-mirrored logs there too (daily rotation, 256 MiB/file
cap, 14 files retained) — the file sink exists so a hung/killed process still
leaves a log on disk instead of only the terminal/tmux scrollback. Multiproc
workers each get their own file (`arle-rank<N>.log`) so ranks don't clobber
each other. Set to `off`, `none`, or `""` to disable the file sink entirely
(stderr logging is unaffected either way). `RUST_LOG` still controls level.

### `ARLE_MODEL`

Default model path for the top-level CLI when `--model-path` is omitted.
Legacy `AGENT_INFER_MODEL` remains a compatibility fallback, but new docs and
scripts should use `ARLE_MODEL`.

Example:

```bash
export ARLE_MODEL=models/Qwen3.5-4B
./target/release/arle --max-turns 10
```

### `ARLE_KV_MMAP_HUGEPAGE` (debug/probe)

`1` = `madvise(MADV_HUGEPAGE)` on freshly created KV spill mmaps
(`kv-native-sys`). Probe for the write-burst first-touch-fault ceiling;
off by default until an A/B licenses it. Linux only.

### `ARLE_KV_SSD_PATH`

Default root for the opt-in L3 (NVMe) KV spill when `arle serve` gets a bare
`--kv-disk` (no directory value); an explicit `--kv-disk DIR` wins. Unset, the
default is the platform cache dir (`~/.cache/arle/kv-ssd` on Linux).

### Apple Silicon one-command bring-up

The canonical Metal serving entrypoint is `arle serve --backend metal`. The rewrite
removed the separate `metal_serve` binary; `arle serve` loads the model **in-process**
and serves the OpenAI v1 API on `--bind`:`--port` (default `127.0.0.1:8000`).

```bash
arle serve --backend metal --model-path mlx-community/Qwen3.5-0.8B-MLX-4bit
arle serve --backend metal --model-path mlx-community/Qwen3.5-4B-bf16 --port 8012
```

Run `arle serve --help` for the full flag surface.

### Metal runtime memory limits

The rewrite Metal executor auto-pins model weights via `mlx::set_wired_limit`
at construction (model dir size + 1 GiB headroom —
`crates/infer-metal/src/wired_limit.rs`). The monolith-era
`--memory-limit-bytes` / `--cache-limit-bytes` / `--wired-limit-bytes` flags
no longer exist; there is currently no CLI or env override.

### `INFER_MOE_TOP_K`

Override the MoE block's active-expert count below the model's
configured top_k. Optional; clamped to `(0, model_top_k]` so passing
a value larger than the model's default is a no-op. Logs once on
override.

For `mlx-community/Qwen3.6-35B-A3B-4bit` (default top_k=8):
- `INFER_MOE_TOP_K=6` cut c=4 ITL p50 by **−21.4%** (28880 → 22694
 μs) and c=8 by **−9.9%** (41108 → 37044 μs). Quality cost ~3%
 MMLU drop per upstream `vllm-mlx` reports on similar MoE models; not
 validated for Qwen3.6 specifically.

Mirrors `vllm-mlx`'s `--moe-top-k` flag. Use for latency-critical
chat / code workloads; keep the default for evaluation /
quality-sensitive paths. See
[`docs/experience/wins/2026-05-07-bench-qwen36-moe-topk-runtime-knob.md`](experience/wins/2026-05-07-bench-qwen36-moe-topk-runtime-knob.md).

```bash
INFER_MOE_TOP_K=6 ./target/release/arle serve --backend metal \
 --model-path mlx-community/Qwen3.6-35B-A3B-4bit \
 --port 8765 -- --max-running-requests 16
```

### `MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER` (MLX upstream)

Tune MLX's per-command-buffer commit cadence. Defaults vary by Apple
Silicon tier (40/40 on base/pro, 50/50 on Max/Ultra) — see
`mlx/backend/metal/device.cpp:498-522`. **Recommended for any Metal
bench at c≥8**: export `MLX_MAX_OPS_PER_BUFFER=200
MLX_MAX_MB_PER_BUFFER=200`. With Qwen3.6 MoE forward at c≥8, the MLX
defaults force 4–5 implicit `commandBuffer.commit()` per decode step;
boosting them collapses the cliff at c=8→c=10.

```bash
MLX_MAX_OPS_PER_BUFFER=200 \
MLX_MAX_MB_PER_BUFFER=200 \
./target/release/arle serve --backend metal \
 --model-path mlx-community/Qwen3.6-35B-A3B-4bit \
 --port 8765 -- --max-running-requests 16
```

### DiffusionGemma Metal diagnostics

`ARLE_DIFFUSION_CPP_PROFILE` controls the C++/MLX DiffusionGemma per-request
profile line. It is **default-on** for Metal DiffusionGemma so every request
reports the actual split between prompt prefill, denoise work, host scalar sync,
self-conditioning, and final canvas commit. Set it to `0`, `false`, `off`, or
`no` to suppress the line for clean operator logs or pure wall-clock benchmarks.

The max denoise step budget is a CLI flag:
`arle serve --diffusion-max-denoising-steps N` (replaced
`ARLE_DIFFUSION_MAX_DENOISING_STEPS`, 2026-07-10). Lower values can improve
decode speed but are not a quality-preserving default unless separately gated.

```bash
ARLE_DIFFUSION_CPP_PROFILE=0 ./target/release/arle \
 --model-path mlx-community/diffusiongemma-26B-A4B-it-4bit \
 --max-tokens 64 --non-interactive run --prompt "Say hi" --no-tools

./target/release/arle serve --diffusion-max-denoising-steps 4 \
 --backend metal \
 --model-path mlx-community/diffusiongemma-26B-A4B-it-4bit
```

### `AGENT_INFER_GDR_METAL_KERNEL`

Influence Metal GDR kernel path selection.

Status: internal / experimental.

---

## 3. Build and Toolchain Variables

### `CUDA_HOME`

Path to CUDA toolkit.

Typical value:

```bash
export CUDA_HOME=/usr/local/cuda
```

### `CUDA_PATH`

Windows-style alternative to `CUDA_HOME`.

### `INFER_CUDA_DEVICE`

CUDA device ordinal that the default `cuda_kernels::tensor::DeviceContext::new()`
binds to. Single integer, default `0`. Parse failures are a hard error.

Single-GPU runtime path (default): one `DeviceContext::new()` per process,
honours this variable.

Multi-GPU TP path (F1+):
each rank thread bypasses this variable and calls
`DeviceContext::on_device(ordinal)` directly with its assigned ordinal.

```bash
export INFER_CUDA_DEVICE=1 # bind default context to GPU 1
```

### Single-node multi-GPU topology variables (F0.11)

Status: documented contract for the single-node multi-GPU line.
`INFER_CUDA_DEVICE` remains the default single-rank runtime selector. DeepSeek
V4 distributed HTTP serving now consumes `INFER_CUDA_DEVICES` and the TP/EP
axis size overrides below; generic Qwen TP/PP/EP serving remains staged unless
a model path explicitly wires the corresponding collectives.

| Variable | Parsed at startup today | Accepted range / format | Current behavior |
|---|---|---|---|
| `INFER_CUDA_DEVICE` | yes, by `DeviceContext::new()` | one CUDA ordinal, default `0` | Binds the single process to one GPU. Parse failure is a hard error. |
| `INFER_CUDA_DEVICES` | yes, by distributed CUDA worker bootstrap | comma-separated ordinals such as `0,1,2,3`; unique, non-empty | Maps local rank threads to CUDA devices for distributed serving. |
| `INFER_TP_SIZE` | yes for DSv4 / staged for other CUDA models | integer `>= 1`; default `1` | Tensor-parallel axis size. DSv4 also accepts `ARLE_TP_SIZE`; unset DSv4 HTTP runs use the worker world size. |
| `INFER_PP_SIZE` | yes for DSv4 diagnostics / staged for execution | integer `>= 1`; default `1` | Parsed into the DSv4 multi-axis contract. Non-`1` is fail-closed until PP execution is wired. |
| `INFER_EP_SIZE` | yes for DSv4 / staged for other CUDA models | integer `>= 1`; default `1` | Expert-parallel axis size. DSv4 also accepts `ARLE_EP_SIZE`; unset DSv4 HTTP runs use the worker world size. |
| `INFER_ATTN_DP_SIZE` | yes for DSv4 diagnostics / staged for execution | integer `>= 1`; default `1` | Parsed into the DSv4 SGLang-path contract. Non-`1` is fail-closed until attention-DP communicators and token ownership are wired. |
| `INFER_ATTN_CP_SIZE` | yes for DSv4 diagnostics / staged for execution | integer `>= 1`; default `1` | Parsed into the DSv4 SGLang-path contract. Non-`1` is fail-closed until attention-CP communicators are wired. |
| `INFER_MOE_DP_SIZE` | yes for DSv4 diagnostics / staged for execution | integer `>= 1`; default `1` | Parsed into the DSv4 SGLang-path contract. Non-`1` is fail-closed until MoE-DP token ownership is wired. |
| `INFER_NCCL_PORT` | no, reserved F1+ | TCP port `1..=65535` | Future convenience alias for `MASTER_PORT` during single-node rendezvous. |

Current DSv4 parser acceptance rules:

- `INFER_CUDA_DEVICES` length must be at least the local rank count.
- For SGLang-style axes, `world_size = INFER_TP_SIZE * INFER_PP_SIZE`.
 `INFER_ATTN_DP_SIZE * INFER_ATTN_CP_SIZE` must divide `INFER_TP_SIZE`, and
 `INFER_EP_SIZE * INFER_MOE_DP_SIZE` must divide `INFER_TP_SIZE`.
- Current ARLE DSv4 execution also preserves legacy TP-only and EP-only
 overrides where each of `INFER_TP_SIZE` and `INFER_EP_SIZE` is either `1` or
 the CUDA worker count.
- Today's executable DSv4 path accepts only global TP/EP-style layouts for
 execution. Rich SGLang axes are parsed so that explicit path claims can fail
 closed with a clear error instead of silently running the replicated-token
 route.
- Multi-rank values are rejected if CUDA was not built in, NCCL was not enabled
 for a path that needs collectives, or the machine exposes fewer devices than
 requested.
- `INFER_CUDA_DEVICE` and `INFER_CUDA_DEVICES` should not both be used for a
 multi-rank run. `INFER_CUDA_DEVICE` is the single-rank compatibility knob;
 `INFER_CUDA_DEVICES` is the ordered multi-rank map.

Examples of combinations that F1+ bootstrap must reject:

```bash
INFER_TP_SIZE=2 INFER_CUDA_DEVICES=0 # TP=2 but one local device
INFER_TP_SIZE=2 INFER_PP_SIZE=2 INFER_CUDA_DEVICES=0,1
# product world size is 4, but only two local devices are listed

INFER_TP_SIZE=2 INFER_CUDA_DEVICES=0,0 # duplicate device ordinal
INFER_NCCL_PORT=0 # invalid TCP port for rendezvous
```

When the F1+ parser lands, startup logging must print the parsed topology before
model load so bad jobs fail with actionable context. Expected shape:

```text
multi_gpu_config:
 cuda_devices=[0,1]
 tp_size=2 pp_size=1 ep_size=1 attn_dp=1 attn_cp=1 moe_dp=1
 world_size=2 nccl_port=29500
 status=accepted
```

For today's single-rank runtime, the equivalent effective topology is:

```text
multi_gpu_config:
 cuda_devices=[INFER_CUDA_DEVICE or 0]
 tp_size=1 pp_size=1 ep_size=1 attn_dp=1 attn_cp=1
 world_size=1 status=single-rank
```

### DeepSeek V4 distributed CUDA debug variables

Status: experimental DSv4 bring-up controls. These are intentionally documented
as diagnostics and validation gates, not stable tuning API.

| Variable | Values | Default | Current behavior |
|---|---|---|---|
| `ARLE_DSV4_MOE_BACKEND` (alias `ARLE_DSV4_MOE_TRANSPORT`) | `allreduce` (default), `deepep`, `deepep_ll` | `allreduce` | Selects the DSv4 MoE transport (`infer-cuda/src/dsv4.rs::dsv4_use_deepep_transport`). `allreduce` = local routed experts + EP all-reduce (the licensed default). `deepep` / `deepep_ll` = NVSHMEM token-owned DeepEP paths; B=1 deepep_ll is fixed (`b5f00399`) but the batched lane license is open (#61) — not default-worthy yet. |
| `ARLE_DSV4_INCREMENTAL_KV` | `1` / unset | unset | Enables the incremental DSv4 KV state path used by the 8-rank HTTP bring-up. |
| `ARLE_DSV4_OPERATOR_TRACE` | `1` / unset | unset | Enables the same CUDA-synchronizing DSv4 operator aggregate in `request_trace` JSON without emitting every per-layer event log line. The field is `dsv4_operator_trace_process_delta` and is valid for single-inflight profiling only. |
| `ARLE_DSV4_OPERATOR_TRACE_EVENTS` | `1` / unset | unset | With `ARLE_DSV4_OPERATOR_TRACE=1`, also emits the legacy `dsv4_trace layer=... phase=...` event log lines. |
| `ARLE_DSV4_COUNT_EXCHANGE` | `allgather`, `sendrecv` | `allgather` | Selects the tiny per-layer route-count exchange. `sendrecv` keeps the older grouped P2P fallback. |
| `ARLE_DSV4_PADDED_DISPATCH` | `1`, `0`, unset | `1` | Enables the B=1 decode padded dispatch fast path when `ARLE_DSV4_COUNT_EXCHANGE=allgather`. It uses fixed `ep_world * topk` route slots, skips the send-count zero/count kernel, removes the per-layer count AllGather and all-rank count D2H, and pre-sums padded BF16 combine rows to one row per origin peer before the return exchange. Set `0` to force the exact-count fallback. |
| `ARLE_DSV4_FUSE_ATTN_WINDOW_UPDATE` | `1`, `0`, unset | `1` | For B=1 decode, writes the current sliding-window key from the DSv4 SWA/hybrid attention kernel tail and skips the standalone `dsv4_update_window_cache_kernel`. Prefill and multi-token steps still use the separate update kernel. Set `0` to force the older standalone update for A/B diagnosis. |
| `ARLE_DSV4_FUSE_QK_PREP` | `1`, `0`, unset | `1` | Fuses DSv4 Q RMSNorm+RoPE prep and K RoPE prep into one CUDA launch. Set `0` to force the older two-launch `dsv4_prepare_q_kernel` + `dsv4_prepare_k_kernel` path for A/B diagnosis. |
| `ARLE_DSV4_GROUPED_EXPERTS` | `1` / unset | unset | Enables the raw grouped expert GEMV prototype. The current harness caches per-layer local expert weight pointer arrays and launches only indexed active experts, but remains slower than the default scratch-reuse path on B=1 decode until the raw GEMV work is replaced by real grouped GEMM/DeepGEMM. |
| `ARLE_DSV4_PAIR_EXPERT_GEMV` | `1` / unset | unset | Enables the single-expert `w1`/`w3` pair GEMV experiment in the default local expert loop. The 8xH20 Nsight trace shows it is functionally correct but slower on the current B=1 decode shape, so it remains default-off. |
| `ARLE_DSV4_ROUTE_GROUPED_EXPERTS` | `1` / unset | unset | Enables the route-wise grouped local expert experiment for padded B=1 decode. The opt-in path pairs route-local `w1`/`w3` GEMV when DSv4 block-scaled formats match and applies route weights after BF16 `w2` output to preserve baseline rounding. It removes D2H from the filtered decode nsys summary, but remains default-off: 2026-05-26 validation improved short decode only -4.60% and regressed longseq `max_tokens=32` by +1.36%. Use only for diagnostics until replaced by true grouped GEMM/DeepGEMM with DeepEP overlap. |
| `ARLE_DSV4_EXPERT_BACKEND` | `native`, `deepgemm` | `deepgemm` | Selects the DSv4 local expert backend. The runtime default is required DeepGEMM: the native DeepGEMM JIT bridge and resident FP8 expert-weight cache must preflight successfully, or DSv4 fails before serving instead of silently falling back to native grouped GEMV. `native` keeps the current per-expert/raw grouped GEMV paths for controlled diagnosis only. There is no `deepgemm-auto` fallback in the DSv4 fast lane. |
| `ARLE_DSV4_DEEPGEMM_DEVICE_COUNTS` | `1`, `0`, unset | `1` | Enables the padded B=1 DeepGEMM local-expert path that keeps recv-side local expert counts and offsets on device. It uses dense all-local-expert metadata, initializes unused compact route slots to `-1`, and skips them during scatter. Set `0` to force the older host `local_counts` D2H path for A/B diagnosis. |
| `ARLE_DSV4_DEEPGEMM_ZERO_FP8_SCRATCH` | `1` / unset | unset | Forces the pre-2026-06-01 DeepGEMM behavior of clearing FP8 input/activation scratch every expert call. Default unset skips those large FP8 memsets and relies on `masked_m` plus valid-row unpad/scatter; scale buffers are still cleared for TMA-aligned padding safety. |
| `ARLE_DSV4_DEEPGEMM_WEIGHT_CACHE` | `1` / unset | unset | Builds the DSv4 routed-expert FP8 E4M3 + FP32-scale cache at load time without selecting the runtime DeepGEMM backend. On H20/SM90 this is the required conversion boundary for FP4 Flash experts before DeepGEMM masked/contiguous grouped GEMM can replace raw GEMV. It fuses `w1`/`w3` rows into one gate/up cache and builds a separate `w2` cache. |
| `ARLE_CUDA_DISABLE_DEEPGEMM_NATIVE` | `1` / unset | unset | Opt-out for the raw-pointer DeepGEMM C ABI bridge. Native DeepGEMM is default-on when an sm_90 target and vendored DeepGEMM/CUTLASS sources are present. Runtime JIT still needs `${CUDA_HOME}/bin/nvcc`, `cuobjdump`, and a C++20-capable host compiler or a warm `DG_JIT_CACHE_DIR`. |
| `ARLE_DEEPGEMM_ROOT` | path | `crates/cuda-kernels/vendor/deepgemm` | Build-time DeepGEMM source root for the optional native bridge. Use a recursive upstream clone when the vendored third-party submodules are not populated. |
| `ARLE_DEEPGEMM_LIBRARY_ROOT` | path | `${ARLE_DEEPGEMM_ROOT}/deep_gemm` | Runtime DeepGEMM JIT library root consumed by the native bridge. Set this when the runtime source tree differs from the build-time path. |
| `ARLE_DEEPGEMM_CUTLASS_INCLUDE` | path | `${ARLE_DEEPGEMM_ROOT}/third-party/cutlass/include`, falling back to FlashMLA vendor CUTLASS when available | Optional runtime CUTLASS include override for the native NVCC JIT path. The fast-build/toolchain helpers print the effective path, and the native bridge preflight reports whether `cutlass/arch/barrier.h`, `nvcc`, `cuobjdump`, and `deep_gemm/include` are present before any request runs. |
| `ARLE_CUDA_KERNELS_PREBUILT_DIR` | path | unset | Build-time fast path for CUDA kernel artifacts. The directory must contain `libkernels_cuda.a` and `libtilelang_kernels_aot.a`; when set, `crates/cuda-kernels/build.rs` links those archives and skips all `nvcc` and TileLang AOT work. If the directory also contains `arle_deepep_sidecar`, the sidecar path is baked into the binary. Keep the artifact key tied to CUDA toolkit, SM list, feature flags, TileLang version, DeepGEMM root, and the `crates/cuda-kernels` source hash. Produce a pack from a built target tree with `scripts/export_prebuilt_cuda_kernels.sh <dest>`; the consumer key is `arle-cuda-kernels.manifest`, with `manifest.json` kept as human-readable provenance. |
| `ARLE_DEEPEP_SIDECAR_PREBUILT` | path | unset | Build-time fast path for only the ARLE DeepEP sidecar binary. When set, `crates/cuda-kernels/build.rs` bakes this path into `ARLE_DEEPEP_SIDECAR_PATH` and skips sidecar compilation even if `ARLE_DEEPEP_DIR` is set. |
| `ARLE_CUDA_DISABLE_FLASHMLA_DECODE` | `1`, unset | unset | Build-time kill switch for FlashMLA sparse-FP8 decode compilation. Decode FFI symbols are satisfied by stubs while sparse prefill can remain enabled. |
| `ARLE_NVCC_WRAPPER` | command | unset | Optional wrapper for CUDA compilation in `crates/cuda-kernels/build.rs` and `crates/deepep-sys/build.rs`. Typical value: `sccache`, which runs `sccache /usr/local/cuda/bin/nvcc ...`. |
| `ARLE_NVCC_SPLIT_COMPILE` | integer | unset | Optional `nvcc --split-compile=<N>` value for CUDA compilation in `crates/cuda-kernels/build.rs` and `crates/deepep-sys/build.rs`. Use a bounded value such as `8` or `16` on high-core build hosts; unset preserves the current nvcc behavior. |
| `ARLE_NVCC_PARALLEL` | integer | `min(cores, 8)` | Worker count for the bounded parallel nvcc pool over native `.cu` compilation in `crates/cuda-kernels/build.rs`. `1` restores the previous serial loop. Capped at 8 by default because one multi-arch nvcc invocation can take 1-2 GB of RAM. Archive (`ar`) ordering is queue-order, identical to the serial loop. |
| `ARLE_DSV4_COMBINE_DTYPE` | `bf16`, `fp8`, unset | `bf16` | Selects the return-side MoE combine exchange payload. `fp8` is validated as an opt-in experiment but is not faster than the BF16 default on the current 8xH20 trace. |
| `ARLE_DSV4_COMBINE_OVERLAP` | `1`, `0`, unset | unset | Enables the opt-in return-side MoE reduce-scatter overlap experiment. It creates a second EP NCCL communicator on `comm_stream` and returns a routed-output fence so shared expert compute can run before consuming routed output. Real 8xH20 nsys returns exact `406`, but regresses the single-token decode wave from 94.841 ms to 104.359 ms, so the default remains off. |

Canonical DSv4 toolchain helper:

```bash
export ARLE_DSV4_MODEL_PATH=<local-dsv4-model-path>
export ARLE_DEEPGEMM_ROOT=<deepgemm-source-root>
export ARLE_DEEPGEMM_LIBRARY_ROOT=${ARLE_DEEPGEMM_ROOT}/deep_gemm

./scripts/dsv4_toolchain.sh env-check
./scripts/dsv4_toolchain.sh build
./scripts/dsv4_toolchain.sh smoke --max-tokens 32
./scripts/dsv4_toolchain.sh nsys --max-tokens 32
```

The helper validates CUDA/NVCC, NCCL, DeepGEMM/CUTLASS, model path, and
decode token count before running. Build uses `cargo build --release
--features cuda,nccl --bin arle`.
Smoke/nsys default to the DSv4 validation envelope; use
`--max-running-requests` when a smoke needs a tighter active-request cap.
Executor hot-workspace slots are model/VRAM-derived, not a public serve flag.
`max_tokens=1`
must only be used for explicit prefill/TTFT smoke outside this helper; decode
evidence uses `max_tokens>=32`.

### `INFER_TILELANG_PYTHON`

Python interpreter with TileLang installed for build-time AOT kernel generation.

Typical value:

```bash
export INFER_TILELANG_PYTHON=.venv/bin/python
```

### `TORCH_CUDA_ARCH_LIST` (alt: `CMAKE_CUDA_ARCHITECTURES`)

Override the CUDA SM compile targets. Uses the standard PyTorch / vLLM /
SGLang convention. Consumed by
`crates/cuda-kernels/build.rs::detect_sm_targets`. Resolution order:

1. `TORCH_CUDA_ARCH_LIST`
2. `CMAKE_CUDA_ARCHITECTURES`
3. `nvidia-smi --query-gpu=compute_cap`
4. T1 default set `{80, 86, 89, 90}` (no T2 by default)

Accepted formats (any combination per token; separators `;` `,` whitespace):

```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" # PyTorch native
export TORCH_CUDA_ARCH_LIST="8.0 9.0" # space-separated
export TORCH_CUDA_ARCH_LIST="80;90" # packed integer
export TORCH_CUDA_ARCH_LIST="sm_80;sm_90" # nvcc style
export TORCH_CUDA_ARCH_LIST="9.0+PTX" # PyTorch +PTX suffix
export CMAKE_CUDA_ARCHITECTURES="80;86;89;90" # CMake alias
```

**Tier policy**:

- T1 (default): `sm_80 / 86 / 89 / 90` — A100 / A10·3090 / L4·4090 / H100.
- T2 (opt-in): `sm_100 / 120` — B100·B200 / RTX 5090. Must be requested
 explicitly via `TORCH_CUDA_ARCH_LIST`; not auto-included.
- Legacy Volta (opt-in): `sm_70` — V100. Supported as a **separate
 SM-pinned build**: set `TORCH_CUDA_ARCH_LIST="7.0"` (or `"70"`) alone —
 it cannot be mixed with T1/T2 targets, otherwise build errors with
 "sm_70 legacy Volta builds must be SM-pinned". Only BF16 Qwen3.5
 dense-attention + GDR cubins are functional; FP8 KV and DSv4 HD64
 wrappers return `CUDA_ERROR_NOT_SUPPORTED` (build emits a warning).
- T4 (`sm_75`) and older (`sm < 70`) are **rejected** — build panics.

**Difference from PyTorch.** PyTorch is best-effort (warns + skips when
a kernel can't compile for a target SM). ARLE is hard-fail: every target
SM must succeed for every AOT kernel, otherwise build panics with a
suggested `TORCH_CUDA_ARCH_LIST` value that excludes the failing SM.

## 4. Setup Script Variables

These are primarily consumed by `setup.sh`.

### `MODEL_ID`

HuggingFace model ID to download.

Default: `Qwen/Qwen3.5-4B`

### `MODEL_DIR`

Local directory for downloaded model files.

Default: `models/Qwen3.5-4B`

### `SKIP_MODEL`

Skip model download during setup.

### `PYTHON`

Python interpreter used by `setup.sh`.

Default: `python3`

---

## 5. Test and Integration Variables

### `INFER_TEST_MODEL_PATH`

Override model path for infer-side GPU tests.

**Backend defaults**:
- **Metal**: `mlx-community/Qwen3.6-35B-A3B-4bit` (canonical, see
 `AGENTS.md` §"Metal canonical model"). Use `INFER_TEST_MODEL_PATH`
 to opt down to a smaller model for fast iteration on dense-only
 paths.
- **CUDA**: `models/Qwen3.5-4B` (canonical for CUDA bench/test scripts).

Example:

```bash
# CUDA — use a smaller model for a quick e2e test:
INFER_TEST_MODEL_PATH=models/Qwen3.5-4B cargo test --release --test e2e

# Metal — bench the canonical Qwen3.6 35B-A3B MoE:
./target/release/arle serve --backend metal \
 --model-path mlx-community/Qwen3.6-35B-A3B-4bit \
 --port 8765 -- --max-running-requests 16
```

### `INFER_URL`

Base URL for integration-style Python API tests.

### `INFER_MODEL`

Model name expected by integration-style Python API tests.

### `HF_TOKEN`

HuggingFace API token used for private-model downloads in
`crates/infer-util/src/hf_hub.rs`. Unset by default; required for gated
models on the `resolve_model_path` path.

### `HF_HOME`

HuggingFace local cache root override (consumed by `hf_hub.rs`).
Defaults to `$HOME/.cache/huggingface`.

---

## 6. Environment Dependencies

### `LD_LIBRARY_PATH`

Used in some Linux environments and scripts so CUDA shared libraries can be
found.

### `nsjail`

Not an environment variable, but an important Linux dependency for CLI tool
sandboxing.

- Linux prefers `nsjail` when installed.
- macOS falls back to `sandbox-exec`.

---

## 7. Minimal Sets by Scenario

### CLI usage

```bash
export ARLE_MODEL=models/Qwen3.5-4B
```

### CUDA build

```bash
export CUDA_HOME=/usr/local/cuda
export INFER_TILELANG_PYTHON=.venv/bin/python
```

### GPU tests

```bash
export INFER_TEST_MODEL_PATH=models/Qwen3.5-4B
```

### Integration API tests

```bash
export INFER_URL=http://localhost:8000
export INFER_MODEL=Qwen3.5-4B
```

---

## 8. Variables to Treat Carefully

These exist in the repository, but should be treated as less stable unless the
docs promote them more clearly:

- `AGENT_INFER_GDR_METAL_KERNEL`
- `AGENT_INFER_QWEN35_CPP_SEPARATE` — toggle the Rust→C++ separate-proj
 path in `crates/infer-metal/src/qwen35.rs`. Default on; set to `0`
 to force the fused route for A/B comparison
- `AGENT_INFER_QWEN35_CPP_KEEP_PREFILL_INTERMEDIATES` — keep prefill
 intermediate tensors in the Qwen3.5 C++ step model (`mlx_qwen35_model.cpp`)
 for debugging; default off
- `AGENT_INFER_QWEN35_CPP_CLEAR_CACHE` — force MLX cache clears between
 Qwen3.5 C++ steps
- `AGENT_INFER_QWEN35_CPP_PREFILL_LAST_LOGITS_ONLY` — only materialize
 the last token's logits during prefill (default on for the C++ path)
- `AGENT_INFER_QWEN35_CPP_SEPARATE_MLP` — split the MLP evaluation into
 separate up/gate/down passes instead of the fused path
- `AGENT_INFER_QWEN35_CPP_PREFILL_GBETA_HELPER` — toggle the helper-kernel
 g-beta variant during Qwen3.5 prefill
- `AGENT_INFER_QWEN35_CPP_QK_NORM_HELPER` — opt into the helper-kernel
 Q/K norm variant during Qwen3.5 GDR execution; default off because the
 native MLX `fast::rms_norm(...) * scale` lowering is faster on the
 Qwen3.5-0.8B MLX 4bit single-request path
- `AGENT_INFER_METAL_GGUF_NATIVE_Q4` — controls Qwen3.5 Metal GGUF
 load-time conversion for packed K-quant tensors. Default is `off`, keeping
 exact GGUF affine/packed behavior for correctness. Set to `all` / `1` /
 `true` for the lossy MLX native q4 group64 speed path
- `AGENT_INFER_QWEN35_CPP_GDR_TG_Y` /
 `AGENT_INFER_QWEN35_CPP_PREFILL_GDR_TG_Y` /
 `AGENT_INFER_QWEN35_CPP_DECODE_GDR_TG_Y` — Gated Delta Rule tile-Y
 size tuning knobs for the Qwen3.5 C++ recurrent-state path

All `AGENT_INFER_QWEN35_CPP_*` knobs are internal C++ bridge debugging
aids; they are not part of any stable contract and may be renamed or
removed without notice.

If you add, rename, or deprecate an environment variable, update this document
in the same PR.
