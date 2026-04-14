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

### `AGENT_INFER_TEST_MODEL_PATH`

Override model path for selected CLI-side tests.

### `AGENT_INFER_METAL_KV_POOL`

Enable Metal KV pool path.

Status: experimental.

### `AGENT_INFER_METAL_DFLASH_MODEL`

Enable Metal DFlash and point the backend at the draft model checkpoint.

Accepted values:

- local model directory
- Hugging Face repo id

Example:

```bash
export AGENT_INFER_METAL_DFLASH_MODEL=z-lab/Qwen3-4B-DFlash-b16
./target/release/metal_serve --model-path mlx-community/Qwen3-4B-bf16
```

Status: experimental. Current implementation supports `Qwen3` targets only.

### `AGENT_INFER_METAL_DFLASH_SPECULATIVE_TOKENS`

Optional Metal DFlash block-size override.

Example:

```bash
export AGENT_INFER_METAL_DFLASH_SPECULATIVE_TOKENS=16
```

Recommendation:

- leave this unset unless benchmark data says otherwise

Status: experimental.

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

### `PEGAINFER_CUDA_SM`

Override detected CUDA SM targets.

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
- `AGENT_INFER_METAL_DFLASH_MODEL`
- `AGENT_INFER_METAL_DFLASH_SPECULATIVE_TOKENS`
- `AGENT_INFER_GDR_METAL_KERNEL`
- `PEGAINFER_E2E_MODEL_PATH`
- `FLASHINFER_INCLUDE_DIR`

If you add, rename, or deprecate an environment variable, update this document
in the same PR.
