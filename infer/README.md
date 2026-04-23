# infer

Pure Rust LLM inference engine. No PyTorch, no frameworks — CUDA serving, Metal serial serving, and a development-oriented CPU backend with an OpenAI-compatible API.

## Performance vs SGLang

Measured on **A100-40GB**, BF16, Qwen3-4B, 8 concurrent requests:

| | infer | SGLang | Ratio |
|---|---|---|---|
| Throughput (tok/s) | **811** | 886 | 0.92x |
| Decode ITL (ms) | 10.4 | 8.2 | 1.27x |

## Optimization Journey

Starting point: 128 tok/s at 8 concurrent. Each phase on A100-40GB, Qwen3-4B.

| Phase | Change | tok/s | Delta |
|---|---|---|---|
| Baseline | Single forward pass per request | 128 | — |
| 1 | Token-level KV pool + FlashInfer paged decode | 434 | +239% |
| 2 | Pre-allocate decode buffers (no per-step GPU malloc) | 681 | +57% |
| 3 | FlashInfer plan once per step (not per layer) | 690 | +1% |
| 4 | Pre-allocate embedding + logit buffers | 700 | +1% |
| 5 | CUDA Graph for batched decode | 756 | +8% |
| 6 | Argmax/scatter optimization (batched argmax, skip D2D scatter for greedy) | **811** | +7% |

Remaining gap to SGLang (1.09x): cuBLAS vs SGLang's kernel choices, sync strategy (stream vs event), FusedAddRMSNorm.

## Architecture

```
HTTP → Scheduler (continuous batching) → model.forward(tokens, state)
                                                  │
                              ┌───────────────────┤
                              │                   │
                         Qwen3Model          Qwen35Model
                        (full attn, GQA)   (24 linear + 8 full attn)
                              │                   │
                              └─────────┬─────────┘
                                        │
                     Prefill: chunked (4096 tok), FlashInfer single (HD128) / Triton FA2 (HD256)
                     Decode:  merged QKV+gate-up GEMM + FlashInfer + CUDA Graph
                                        │
                              ops → ffi → CUDA / Triton kernels
```

Key decisions:
- **Zero Python at runtime** — Triton kernels compiled AOT at build time into C wrappers; FlashInfer linked as C library
- **Continuous batching** — decode-priority scheduling; long prefills chunked to 4096 tokens (64 when decode active) to keep decode slots live
- **Token-level KV pool** — SGLang-style `[max_tokens, kv_dim]` layout, page_size=1; FlashInfer handles paged attention
- **CUDA Graph per batch size** — one graph per batch size N, captured on first call, replayed thereafter; ~504 kernel launches eliminated per step
- **BF16 storage, FP32 accumulation** — cuBLAS GEMM for prefill, custom GEMV kernel for decode

## Quick Start

### CUDA (Linux, NVIDIA GPU)

```bash
# One-time: install Triton for AOT kernel compilation (not needed at runtime)
uv venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# Download model
huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B

# Build and run
export CUDA_HOME=/usr/local/cuda
export INFER_TRITON_PYTHON=.venv/bin/python
cargo run --release
```

```bash
# Test
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 32}'
```

> Always use `--release`. Debug builds are extremely slow for GPU code.

### Metal (macOS, Apple Silicon)

Requires Xcode and the Metal Toolchain component:

```bash
xcodebuild -downloadComponent MetalToolchain
```

```bash
# Download model (Qwen3/Qwen3.5 MLX-converted checkpoints)
huggingface-cli download mlx-community/Qwen3-0.6B-4bit --local-dir models/Qwen3-0.6B-4bit

# Build the Metal backend and serial HTTP server
cargo build --release --no-default-features --features metal,no-cuda --lib
```

The Metal backend (`MetalBackend`) implements the same `InferenceBackend` trait as the CUDA path. `metal_serve` is available today, but it still runs as a serial backend runtime rather than the CUDA-style continuous batching scheduler.

### CPU backend (development smoke path)

The CPU backend is intended for local request-path validation on machines without
CUDA or Metal. It exercises the same backend/runtime/HTTP/CLI surfaces, but it
does not claim production-grade CPU inference throughput.

```bash
# Reuse a public Hugging Face repo ID. The CPU backend only downloads runtime
# assets such as config/tokenizer for smoke validation.
cargo run -p agent-infer --bin arle --no-default-features --features cpu,no-cuda,cli -- \
  --model-path Qwen/Qwen3-0.6B --max-turns 1 --max-tokens 64

# Or run the serial HTTP server variant directly.
cargo run -p infer --no-default-features --features cpu,no-cuda --bin cpu_serve -- \
  --model-path Qwen/Qwen3-0.6B
```

<details>
<summary>Environment variables</summary>

| Variable | Description |
|---|---|
| `CUDA_HOME` | CUDA Toolkit path (default: `/usr/local/cuda`) |
| `INFER_TRITON_PYTHON` | Python with Triton for build-time AOT compilation |
| `INFER_CUDA_SM` | GPU SM target override when `nvidia-smi` unavailable (e.g. `120`) |
| `INFER_TEST_MODEL_PATH` | Override test model path (default: `models/Qwen3-4B`) |

</details>

<details>
<summary>Windows (CUDA)</summary>

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"
uv venv .venv --python 3.12
uv pip install "triton-windows<3.7"
$env:INFER_TRITON_PYTHON = ".venv\Scripts\python.exe"
cargo build --release
cargo run --release --bin infer -- --model-path models/Qwen3-4B
```

</details>

## Supported Models

| Model | Architecture | Params | Status |
|---|---|---|---|
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | Full attention (GQA) | 4B | CUDA |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Full attention (GQA) | 8B | CUDA |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Hybrid (24 linear + 8 full attn) | 4B | CUDA |
| Qwen3 / Qwen3.5 MLX-converted checkpoints | Full attention / hybrid | 0.6B–4B | Metal |
| Hugging Face repos with `config.json` and tokenizer assets | Request-path validation only | n/a | CPU backend |

Model type is auto-detected from `config.json`.

## Features

- Continuous batching with decode-priority scheduling
- Chunked prefill (4096-token chunks, 64 when decode active)
- FlashInfer single prefill (HD128) + Triton FA2 (HD256)
- FlashInfer paged decode attention
- Merged QKV + gate-up GEMM (96 fewer kernel launches/step)
- CUDA Graph for batched decode (one graph per batch size)
- Token-level KV pool (SGLang-style)
- CPU KV offload
- top-k / top-p / temperature / min-p / repetition / frequency / presence penalties
- OpenAI `/v1/completions` and `/v1/chat/completions` (SSE streaming)
- Prometheus `/metrics` and `/v1/stats`
- Metal backend for Apple Silicon (serial `metal_serve`, single request at a time)
- Development-oriented CPU backend for non-GPU smoke tests (`cpu_serve`, CPU CLI)

## API

OpenAI-compatible `/v1/completions`:

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | required | Input text |
| `max_tokens` | int | 128 | Max tokens to generate |
| `temperature` | float | 0.0 | Sampling temperature (0 = greedy) |
| `top_k` | int | 50 | Top-k cutoff |
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `stream` | bool | false | SSE streaming |

Also supports `/v1/chat/completions` with `messages` array (ChatML format).

## Benchmark

```bash
# Throughput benchmark (matches SGLang bench_serving.py interface)
python3 scripts/bench_agent.py /path/to/model

# KV cache prefix benchmark
python3 scripts/bench_kv_cache_prefix.py

# Latency benchmark (single request, TTFT + TPOT)
cargo run --release --bin bench_serving
```

## Dev Guide

### Tests

```bash
# Unit tests
cargo test --release

# E2E greedy regression (GPU + model weights required)
INFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e
cargo test --release --test e2e_qwen35
```

E2E tests compare against JSON baselines in `test_data/`. Regenerate baselines after any change affecting numerical output.

### Source layout

```
infer/src/
├── main.rs                  # CLI entry point + HTTP server (axum)
├── server_engine.rs         # Unified InferenceEngine contract + LoadedInferenceEngine
├── http_server.rs + http_server/  # /v1/completions, /v1/chat/completions, /v1/responses, SSE
├── scheduler/               # Continuous batching scheduler (batch, types, policy, cuda/)
├── model.rs + model/        # ModelForward + GenerationState + per-model impls
│   ├── generation_state.rs  # Per-request mutable state
│   ├── qwen3/               # Qwen3: weights, forward, prefill, decode, batch_decode
│   ├── qwen35/              # Qwen3.5: hybrid linear + full attention
│   └── glm4/                # GLM4
├── backend.rs + backend/    # InferenceBackend trait + submodules
│   ├── cuda.rs              # Thin `pub use infer_cuda_kernels::*;` re-export shim
│   ├── cuda/bootstrap.rs    # Model loading, runtime config, scheduler bring-up
│   ├── metal.rs + metal/    # Metal/MLX backend: mlx bridge, gdr, kv_pool,
│   │                          prefix_cache, scheduler, generate, forward
│   ├── cpu.rs               # Development CPU backend (feature `cpu`)
│   └── runtime.rs           # Serial runtime handle for CPU/Metal
├── ops.rs + ops/            # GPU operator dispatch (attention, linear, norm,
│                              recurrent, sampling, elementwise, embedding, kv_ops)
├── kv_tier/                 # Tiered KV cache skeleton (T0 GPU → T1 host → T2 disk)
├── prefix_cache.rs          # Radix-tree prefix cache
├── block_manager.rs         # KV block accounting for the batch scheduler
├── sampler.rs               # Sampling params + CPU sampler
└── tokenizer.rs             # HuggingFace tokenizers wrapper
```

The CUDA C kernels and Triton AOT sources used to live at `infer/csrc/cuda/`
and `infer/tools/triton/`, but were extracted into their own crate on
2026-04-15 (`a4e12f5 refactor(cuda): extract cuda-kernels api`). They
now live at:

```
crates/cuda-kernels/
├── Cargo.toml / build.rs     # nvcc + Triton AOT (lifted from infer/build.rs)
├── csrc/
│   ├── attention/            # flashinfer_*, fused_attention, prefill_attention, decode_prep_paged, ...
│   ├── gemm/                 # gemv, quantized_gemv, marlin_*, turboquant_weight_gemv
│   ├── kv/                   # paged_kv_append, kv_cache_to_paged, kv_quant, scatter_kv
│   ├── quant/                # turboquant, turboquant_fast, dtype_convert
│   └── misc/                 # norm, sampling, pos_enc, conv1d, gdr, fused_mlp, ...
├── tools/triton/             # Triton AOT kernels (compiled at build time)
│   ├── flash_attention_prefill_hd256_kernel.py
│   ├── gated_delta_rule_chunkwise_kernels.py
│   ├── basic_kernels.py / silu_mul_kernel.py
│   └── gen_triton_aot.py     # AOT driver — 13 kernels compiled to C wrappers
└── src/                      # FFI, tensor, paged_kv, flashinfer, graph_pool, prelude
```

`infer/src/backend/cuda.rs` re-exports `infer_cuda_kernels::*` so existing
`crate::backend::cuda::…` call sites keep resolving. See
[`../docs/plans/cuda-kernel-crate-extraction.md`](../docs/plans/cuda-kernel-crate-extraction.md)
for the extraction blueprint (now marked executed) and
[`../docs/architecture.md`](../docs/architecture.md) §"Kernel-Crate Extraction"
for the canonical current shape.

### Triton AOT

Triton kernels are compiled at build time into generated C wrappers. Runtime has no Python dependency. Build triggers when `crates/cuda-kernels/tools/triton/` sources change; output lands in `target/`. See `crates/cuda-kernels/tools/triton/README.md` for setup.

## Roadmap

Phase 0 (foundation) and Phase 2 (quantization) are complete. Active focus:

- **Qwen3.5 SGLang parity** — batched prefill remaining; prefix cache + overlap scheduling already done
- **Llama 3/4** — most requested model architecture
- **DeepSeek-V3 / R1** — requires MLA attention first
- **Tiered KV cache** — hierarchical T0 GPU → T1 host → T2 NVMe → T3 NIXL

Future:
- FlashAttention-3 (SM90 / H100)
- Tensor parallelism (NCCL all-reduce)
- Speculative decoding GPU integration
- Metal HTTP server integration (full server mode on Apple Silicon)

See [ROADMAP.md](../ROADMAP.md) for the full phased plan.

## Acknowledgments

- [PegaInfer](https://github.com/pega-infer/infer) — arle 的 CUDA 推理核心基于 PegaInfer 构建，感谢其高性能 kernel 和模型实现。

## License

MIT
