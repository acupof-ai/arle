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
`SchedulerConfig` in `infer/src/scheduler/types.rs`: prefix-cache
watermarks (`prefix_cache_high_water`, `prefix_cache_low_water`,
`prefix_cache_retain_hard_cap`), keepalive ticks
(`prefix_cache_keepalive_ticks`, `t1_host_pinned_keepalive_ticks`), and
chunking caps are struct fields with `validate()` guards. Callers that
want to tune them construct a `SchedulerConfig::runtime_defaults(..)`
and assign directly — **there is no `INFER_PREFIX_HIGH_WATER`** or
any other magic env var for runtime tuning. If you want an env-var
escape hatch for a specific tuning knob, justify it as a debug aid and
document the debug-only status here.

---

## 1. Naming Rule

- Prefer `ARLE_*` for newly documented user-facing CLI/runtime behavior.
- Treat `AGENT_INFER_*` as legacy compatibility names unless this document
  explicitly calls them out as the current canonical surface.
- Treat `INFER_*` primarily as build, test, or compatibility variables unless
  documented otherwise.
- Treat undocumented variables as internal or experimental.

---

## 2. User-Facing Runtime Variables

### `ARLE_MODEL`

Default model path for the top-level CLI when `--model-path` is omitted.
Legacy `AGENT_INFER_MODEL` remains a compatibility fallback, but new docs and
scripts should use `ARLE_MODEL`.

Example:

```bash
export ARLE_MODEL=models/Qwen3-4B
./target/release/arle --max-turns 10
```

### `AGENT_INFER_API_KEY`

Default Bearer token for HTTP serving entry points that opt into API auth.

Current use:

- `metal_serve` uses this when `--api-key` is omitted.

Example:

```bash
export AGENT_INFER_API_KEY=dev-secret
./target/release/metal_serve --model-path mlx-community/Qwen3-4B-bf16
```

### Apple Silicon one-command bring-up

The canonical first-time Metal serving entrypoint is
[`scripts/start_metal_serve.sh`](../scripts/start_metal_serve.sh). It hides the
Cargo feature flags, builds `metal_serve`, and starts the server on
`127.0.0.1:8000`.

Defaults:

- model: `ARLE_MODEL` if set, otherwise legacy `AGENT_INFER_MODEL`, otherwise `mlx-community/Qwen3-0.6B-4bit`
- port: `8000`
- bind: `127.0.0.1`

Examples:

```bash
./scripts/start_metal_serve.sh
./scripts/start_metal_serve.sh mlx-community/Qwen3-4B-bf16 8012 -- --warmup 0
```

Extra `metal_serve` flags go after `--`. For example, you can still pass
`--api-key`, `--memory-limit-bytes`, `--cache-limit-bytes`, or
`--wired-limit-bytes` through the wrapper.

### `AGENT_INFER_TEST_MODEL_PATH`

Override model path for selected CLI-side tests.

### `AGENT_INFER_METAL_KV_POOL`

Legacy compatibility fallback for the experimental Metal KV pool path.

Current use:

- `metal_request`
- `metal_bench`
- `metal_serve`

Behavior:

- If neither `--kv-pool` nor `--no-kv-pool` is passed, these entry points use
  `AGENT_INFER_METAL_KV_POOL` as a fallback.
- Prefer the explicit CLI flags over this environment variable.

Status: experimental, fallback-only.

### Metal runtime memory limits

The MLX allocator limits for Metal are currently exposed as CLI flags, not
environment variables:

- `--memory-limit-bytes`
- `--cache-limit-bytes`
- `--wired-limit-bytes`

Current use:

- `metal_request`
- `metal_bench`
- `metal_serve`

These are applied before model load and affect the whole process-local MLX
allocator state.

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

Multi-GPU TP path (F1+, see
[`docs/plans/2026-04-28-single-node-multi-gpu.md`](plans/2026-04-28-single-node-multi-gpu.md)):
each rank thread bypasses this variable and calls
`DeviceContext::on_device(ordinal)` directly with its assigned ordinal.

```bash
export INFER_CUDA_DEVICE=1   # bind default context to GPU 1
```

### Single-node multi-GPU topology variables (F0.11)

Status: documented contract for the single-node multi-GPU line. Only
`INFER_CUDA_DEVICE` is active in the default single-rank runtime today. The
multi-rank variables below are reserved for F1+ bootstrap; until that parser is
wired, they must not be treated as evidence that TP/PP/EP serving is active.

| Variable | Parsed at startup today | Accepted range / format | Current behavior |
|---|---|---|---|
| `INFER_CUDA_DEVICE` | yes, by `DeviceContext::new()` | one CUDA ordinal, default `0` | Binds the single process to one GPU. Parse failure is a hard error. |
| `INFER_CUDA_DEVICES` | no, reserved F1+ | comma-separated ordinals such as `0,1,2,3`; unique, non-empty | Future rank-thread bootstrap will map local ranks to these ordinals. |
| `INFER_TP_SIZE` | no, reserved F1+ | integer `>= 1`; default `1` | Future tensor-parallel world size. `1` means disabled. |
| `INFER_PP_SIZE` | no, reserved F1+ | integer `>= 1`; default `1` | Future pipeline-parallel world size. `1` means disabled. |
| `INFER_EP_SIZE` | no, reserved F1+ | integer `>= 1`; default `1` | Future expert-parallel world size. `1` means disabled. |
| `INFER_ATTN_DP_SIZE` | no, reserved F1+ | integer `>= 1`; default `1` | Future attention data-parallel axis. |
| `INFER_ATTN_CP_SIZE` | no, reserved F1+ | integer `>= 1`; default `1` | Future attention context-parallel axis. |
| `INFER_NCCL_PORT` | no, reserved F1+ | TCP port `1..=65535` | Future convenience alias for `MASTER_PORT` during single-node rendezvous. |

F1+ parser acceptance rules:

- `INFER_CUDA_DEVICES` length must be at least the local rank count.
- `INFER_TP_SIZE * INFER_PP_SIZE * INFER_EP_SIZE` must equal the model-worker
  world size for dense TP/PP/EP bootstrap. Attention DP/CP and MoE axes may add
  further divisibility checks when those phases land.
- Multi-rank values are rejected if CUDA was not built in, NCCL was not enabled
  for a path that needs collectives, or the machine exposes fewer devices than
  requested.
- `INFER_CUDA_DEVICE` and `INFER_CUDA_DEVICES` should not both be used for a
  multi-rank run. `INFER_CUDA_DEVICE` is the single-rank compatibility knob;
  `INFER_CUDA_DEVICES` is the ordered multi-rank map.

Examples of combinations that F1+ bootstrap must reject:

```bash
INFER_TP_SIZE=2 INFER_CUDA_DEVICES=0          # TP=2 but one local device
INFER_TP_SIZE=2 INFER_PP_SIZE=2 INFER_CUDA_DEVICES=0,1
# product world size is 4, but only two local devices are listed

INFER_TP_SIZE=2 INFER_CUDA_DEVICES=0,0        # duplicate device ordinal
INFER_NCCL_PORT=0                             # invalid TCP port for rendezvous
```

When the F1+ parser lands, startup logging must print the parsed topology before
model load so bad jobs fail with actionable context. Expected shape:

```text
multi_gpu_config:
  cuda_devices=[0,1]
  tp_size=2 pp_size=1 ep_size=1 attn_dp=1 attn_cp=1
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
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"          # PyTorch native
export TORCH_CUDA_ARCH_LIST="8.0 9.0"                  # space-separated
export TORCH_CUDA_ARCH_LIST="80;90"                    # packed integer
export TORCH_CUDA_ARCH_LIST="sm_80;sm_90"              # nvcc style
export TORCH_CUDA_ARCH_LIST="9.0+PTX"                  # PyTorch +PTX suffix
export CMAKE_CUDA_ARCHITECTURES="80;86;89;90"          # CMake alias
```

**Tier policy** (see [`plans/sm-coverage.md`](plans/sm-coverage.md)):

- T1 (default): `sm_80 / 86 / 89 / 90` — A100 / A10·3090 / L4·4090 / H100.
- T2 (opt-in):  `sm_100 / 120` — B100·B200 / RTX 5090. Must be requested
  explicitly via `TORCH_CUDA_ARCH_LIST`; not auto-included.
- T3 (rejected): `sm < 80` — V100 / T4 / older. Build panics.

**Difference from PyTorch.** PyTorch is best-effort (warns + skips when
a kernel can't compile for a target SM). ARLE is hard-fail: every target
SM must succeed for every AOT kernel, otherwise build panics with a
suggested `TORCH_CUDA_ARCH_LIST` value that excludes the failing SM.

## 4. Setup Script Variables

These are primarily consumed by `setup.sh`.

### `MODEL_ID`

HuggingFace model ID to download.

Default: `Qwen/Qwen3-8B`

### `MODEL_DIR`

Local directory for downloaded model files.

Default: `models/Qwen3-8B`

### `SKIP_MODEL`

Skip model download during setup.

### `PYTHON`

Python interpreter used by `setup.sh`.

Default: `python3`

---

## 5. Test and Integration Variables

### `INFER_TEST_MODEL_PATH`

Override model path for infer-side GPU tests.

Example:

```bash
INFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e
```

### `INFER_E2E_MODEL_PATH`

Override model path for selected E2E regeneration flows
(`infer/tests/regen_test_data.rs`).

### `INFER_QWEN3_PATH`

Override model path for the Qwen3-4B GGUF smoke test
(`infer/tests/smoke_qwen3_4b_gguf.rs`). Default:
`models/Qwen3-4B-GGUF`.

### `INFER_Q35_PATH`

Override model path for the Qwen3.5 GGUF smoke test
(`infer/tests/smoke_qwen35_gguf.rs`).

### `INFER_QWEN35_4B_GGUF_PATH`

Override model path for the Qwen3.5 4B GGUF ground-truth Q4_K test
(`infer/tests/ground_truth_q4k.rs`).

### `INFER_CARNICE_PATH`

Override model path for Carnice 27B Q4_K / real-tensor-dequant /
dtype-audit tests
(`infer/tests/smoke_carnice_27b_q4k.rs`,
`infer/tests/carnice_real_tensor_dequant.rs`,
`infer/tests/carnice_dtype_audit.rs`,
`infer/tests/carnice_tensor_probe.rs`).

### `INFER_URL`

Base URL for integration-style Python API tests.

### `INFER_MODEL`

Model name expected by integration-style Python API tests.

### `AGENT_INFER_TEST_MODEL_PATH`

CLI-side live-agent integration test model path override
(`tests/cli_agent_live.rs`).

### `HF_TOKEN`

HuggingFace API token used for private-model downloads in
`infer/src/hf_hub.rs`. Unset by default; required for gated
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
export ARLE_MODEL=models/Qwen3-4B
```

### CUDA build

```bash
export CUDA_HOME=/usr/local/cuda
export INFER_TILELANG_PYTHON=.venv/bin/python
```

### GPU tests

```bash
export INFER_TEST_MODEL_PATH=models/Qwen3-4B
```

### Integration API tests

```bash
export INFER_URL=http://localhost:8000
export INFER_MODEL=Qwen3-8B
```

---

## 8. Variables to Treat Carefully

These exist in the repository, but should be treated as less stable unless the
docs promote them more clearly:

- `AGENT_INFER_METAL_KV_POOL`
- `AGENT_INFER_GDR_METAL_KERNEL`
- `INFER_E2E_MODEL_PATH`
- `INFER_ROPE_CACHE_LEN` — override RoPE cache allocation length in `weight_loader.rs`
- `INFER_FORCE_BF16_QUANT` — skip all packed-quant fast paths in
  `weight_loader.rs` and force BF16 tensor load (debug aid for quant-format issues)
- `INFER_DEBUG_DUMP` — enable tensor debug-dump capture in
  `infer/src/model/common.rs` (default off; set to any value to enable)
- `INFER_QWEN3_FP32_RESIDUAL` — force FP32 residual accumulation on
  the Qwen3 prefill path (`infer/src/model/qwen3/prefill.rs`); debug aid
  for numerical-stability investigations
- `AGENT_INFER_QWEN35_CPP_SEPARATE` — toggle the Rust→C++ separate-proj
  path in `infer/src/backend/metal/qwen35.rs`. Default on; set to `0`
  to force the fused route for A/B comparison
- `METAL_NO_CPP` — disable the Metal Qwen3.5 C++ route entirely
  (`infer/src/backend/metal/qwen35.rs:1255`). Default unset (C++
  route enabled). Set to any value to fall back to the Rust reference
  path for debugging
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
