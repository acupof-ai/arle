# Environment Variables

This document lists the environment variables used by `agent-infer` across
runtime, build, test, and setup workflows.

The repository currently contains both `AGENT_INFER_*` and `PEGAINFER_*`
variables. Until naming is fully unified, use this document as the source of
truth.

---

## 1. Naming Rule

- Prefer `AGENT_INFER_*` for user-facing runtime behavior when available.
- Treat `PEGAINFER_*` primarily as legacy, build, test, or compatibility
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

### `PEGAINFER_TRITON_PYTHON`

Python interpreter with Triton installed for build-time AOT kernel generation.

Typical value:

```bash
export PEGAINFER_TRITON_PYTHON=.venv/bin/python
```

### `PEGAINFER_CUDA_SM` (alt: `CUDA_SM`)

Override detected CUDA SM targets. Consumed by `crates/infer-cuda-kernels/build.rs`
during nvcc + Triton AOT compile; falls back to `CUDA_SM`, then `nvidia-smi`,
then `sm_80`.

Examples:

```bash
export PEGAINFER_CUDA_SM=80
export PEGAINFER_CUDA_SM=80,90
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

### `PEGAINFER_TEST_MODEL_PATH`

Override model path for infer-side GPU tests.

Example:

```bash
PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e
```

### `PEGAINFER_E2E_MODEL_PATH`

Override model path for selected E2E regeneration flows.

### `PEGAINFER_URL`

Base URL for integration-style Python API tests.

### `PEGAINFER_MODEL`

Model name expected by integration-style Python API tests.

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
export PEGAINFER_TRITON_PYTHON=.venv/bin/python
```

### GPU tests

```bash
export PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B
```

### Integration API tests

```bash
export PEGAINFER_URL=http://localhost:8000
export PEGAINFER_MODEL=Qwen3-8B
```

---

## 8. Variables to Treat Carefully

These exist in the repository, but should be treated as less stable unless the
docs promote them more clearly:

- `AGENT_INFER_METAL_KV_POOL`
- `AGENT_INFER_GDR_METAL_KERNEL`
- `PEGAINFER_E2E_MODEL_PATH`
- `FLASHINFER_INCLUDE_DIR`
- `PEGAINFER_ROPE_CACHE_LEN` — override RoPE cache allocation length in `weight_loader.rs`
- `PEGAINFER_FORCE_BF16_QUANT` — skip all packed-quant fast paths in
  `weight_loader.rs` and force BF16 tensor load (debug aid for quant-format issues)
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
