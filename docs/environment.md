# Environment Variables

This document lists the environment variables used by `agent-infer` across
runtime, build, test, and setup workflows.

The repository currently contains both `AGENT_INFER_*` and `INFER_*`
variables. Until naming is fully unified, use this document as the source of
truth.

---

## 0. Policy (2026-04-16, Tier C)

**Env vars are reserved for: build, test model paths, setup, and genuinely
debug/diagnostic runtime overrides.**

**Tuning knobs go on structs**, not env vars. The canonical example is
`SchedulerConfig` in `infer/src/scheduler/types.rs`: prefix-cache
watermarks (`prefix_cache_high_water`, `prefix_cache_low_water`,
`prefix_cache_retain_hard_cap`), keepalive ticks
(`prefix_cache_keepalive_ticks`, `stage_wait_keepalive_ticks`), and
chunking caps are struct fields with `validate()` guards. Callers that
want to tune them construct a `SchedulerConfig::runtime_defaults(..)`
and assign directly — **there is no `INFER_PREFIX_HIGH_WATER`** or
any other magic env var for runtime tuning. If you want an env-var
escape hatch for a specific tuning knob, justify it as a debug aid and
document the debug-only status here.

---

## 1. Naming Rule

- Prefer `AGENT_INFER_*` for user-facing runtime behavior when available.
- Treat `INFER_*` primarily as legacy, build, test, or compatibility
  variables unless documented otherwise.
- Treat undocumented variables as internal or experimental.

---

## 2. User-Facing Runtime Variables

### `AGENT_INFER_MODEL`

Default model path for the top-level CLI when `--model-path` is omitted.

Example:

```bash
export AGENT_INFER_MODEL=models/Qwen3-4B
./target/release/agent-infer --max-turns 10
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

- model: `AGENT_INFER_MODEL` if set, otherwise `mlx-community/Qwen3-0.6B-4bit`
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

### `INFER_TRITON_PYTHON`

Python interpreter with Triton installed for build-time AOT kernel generation.

Typical value:

```bash
export INFER_TRITON_PYTHON=.venv/bin/python
```

### `INFER_CUDA_SM` (alt: `CUDA_SM`)

Override detected CUDA SM targets. Consumed by `crates/cuda-kernels/build.rs`
during nvcc + Triton AOT compile; falls back to `CUDA_SM`, then `nvidia-smi`,
then `sm_80`.

Examples:

```bash
export INFER_CUDA_SM=80
export INFER_CUDA_SM=80,90
```

### `FLASHINFER_INCLUDE_DIR`

Explicit include path override for FlashInfer headers.

Status: advanced build override.

---

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
export AGENT_INFER_MODEL=models/Qwen3-4B
```

### CUDA build

```bash
export CUDA_HOME=/usr/local/cuda
export INFER_TRITON_PYTHON=.venv/bin/python
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
- `FLASHINFER_INCLUDE_DIR`
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
- `AGENT_INFER_QWEN35_CPP_PREFILL_GBETA_HELPER` /
  `AGENT_INFER_QWEN35_CPP_QK_NORM_HELPER` — force the helper-kernel
  variants of g-beta and QK norm during Qwen3.5 prefill
- `AGENT_INFER_QWEN35_CPP_GDR_TG_Y` /
  `AGENT_INFER_QWEN35_CPP_PREFILL_GDR_TG_Y` /
  `AGENT_INFER_QWEN35_CPP_DECODE_GDR_TG_Y` — Gated Delta Rule tile-Y
  size tuning knobs for the Qwen3.5 C++ recurrent-state path

All `AGENT_INFER_QWEN35_CPP_*` knobs are internal C++ bridge debugging
aids; they are not part of any stable contract and may be renamed or
removed without notice.

If you add, rename, or deprecate an environment variable, update this document
in the same PR.
