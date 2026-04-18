# Support Matrix

This document states what `agent-infer` currently supports, what is still
limited, and what validation exists for each area.

If something is not listed as supported here, do not assume it is supported
just because it compiled locally.

State reflected here is based on repository evidence as of 2026-04-17.

---

## 1. Runtime Backends

| Backend | Status | Meaning |
| --- | --- | --- |
| CUDA | Supported | Primary serving path. Main runtime, scheduler, and benchmark focus. |
| Metal | Beta | Usable for local validation and live scheduler-backed serving. Qwen3 ships live prefix reuse with a shared KV pool; Qwen3.5 now ships live prefix reuse via replayed compiled-path snapshots; `scripts/start_metal_serve.sh` is the canonical first-time Apple bring-up path. Metal is still missing full batched-decode parity with CUDA, especially on variable-length Qwen3.5 decode. |
| Metal DFlash | Experimental | Apple Silicon speculative decode path. Qwen3 shipped; Qwen3.5 correctness landed 2026-04-17 but currently regresses vs baseline (acceptance ~28%, serial across requests). Benchmark before use. |
| no-cuda / CPU-only | Development-oriented CPU backend | Build, test, and smoke-validation path for non-GPU logic. Not a production inference target. |

---

## 2. Platform Matrix

| Platform | Backend | Status | Validation |
| --- | --- | --- | --- |
| Linux x86_64 + NVIDIA GPU | CUDA | Supported | Release workflow builds CUDA artifacts; primary target. |
| macOS Apple Silicon | Metal | Beta | CI checks and tests Metal/no-cuda surfaces. |
| Linux/macOS without GPU | no-cuda | Development-oriented CPU backend | Unit tests, compile checks, and CPU backend smoke validation. |

Notes:

- Hosted CI does not provide full CUDA runtime correctness coverage.
- CUDA correctness and performance still require dedicated GPU validation.

---

## 3. Model Family Matrix

| Model family | Status | Notes |
| --- | --- | --- |
| Qwen3 | Supported | Primary supported family. |
| Qwen3.5 | Supported | Supported on normal runtime paths; Metal live runtime now has a narrow same-length decode batch path with packed-batch concurrent decode (2026-04-16 fix). Metal DFlash correctness shipped 2026-04-17 but is currently a perf regression — see §4a. |
| GLM4 | Limited support | Present in project state, but less established than Qwen paths. |
| Llama 3/4 | Planned | Not yet supported. |
| DeepSeek-V3/R1 | Planned | Not yet supported. |
| Mistral / Mixtral / Gemma / Phi | Planned | Not yet supported. |

---

## 4. Quantization Matrix

| Capability | Status | Notes |
| --- | --- | --- |
| FP8 KV cache | Beta | FP8 E4M3 + fused-dequant decode attention; 50% KV memory reduction, benchmarked. |
| INT8 KV cache | Beta | INT8 W8A16 GEMV/GEMM + INT8 KV fused-dequant decode; benchmarked. |
| TurboQuant KV (2–4 bit) | Experimental | Fused decode attention with dequant. Fast-moving optimization area. |
| W8 / W4 / W2 weight quantization | Beta | Native W4 GEMV path + Marlin W4 prefill (5–25× TTFT on long prompts). |
| GPTQ / AWQ (W4A16) | Beta | GEMV + Marlin kernel path; format detection production-ready. |
| GGUF loading | Beta | Supported loader path. Native Q4_K GPU kernel shipped (`q4k_gemv_kernel` + packed fast path in `crates/infer-cuda-kernels/csrc/gemm/quantized_gemv.cu`) — fits Carnice-27B on L4-24GB. |

Backend note:

- The `FP8 KV cache`, `INT8 KV cache`, and `TurboQuant KV` rows above describe
  the shipped project-wide quantized-KV work, which is currently CUDA-backed.
- Metal / MLX does **not** currently ship quantized KV cache. The live Metal
  path stores KV in the model's native dtype today, typically `bf16` / `f16`,
  and it does not expose a `--kv-cache-dtype` surface.
- Metal can still run weight-quantized MLX models; that is separate from
  quantized KV cache support.

---

## 4a. Speculative Decoding Matrix

| Capability | Status | Notes |
| --- | --- | --- |
| Metal DFlash (Qwen3) | Experimental | Apple Silicon speculative decode path. Validated on Qwen3; benchmark before use. |
| Metal DFlash (Qwen3.5) | Experimental — regression | End-to-end correctness landed 2026-04-17 (commits `4db4fe9`, `439293d`). Current single-stream throughput is ~5× slower than baseline (acceptance ~28%, verify_16 on 4-bit target). Concurrent DFlash requests are serial across sessions. Not recommended for production use yet — see `docs/experience/wins/2026-04-17-metal-qwen35-dflash-correctness-bench.md`. |
| CUDA speculative decoding | Not shipped | `infer/src/speculative.rs` is a CPU-only framework; GPU integration pending (see `plans/speculative-decoding-impl.md`). |

---

## 5. Public API Matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `/v1/completions` | Stable | Documented public API. |
| `/v1/chat/completions` | Stable | Documented public API. |
| `/v1/models` | Stable | Loaded-model discovery endpoint. |
| `/v1/responses` | Beta | Non-streaming and SSE forms shipped. Streaming emits `response.created`, `response.output_text.delta`, and terminal `response.completed`; structured outputs are still missing. |
| SSE streaming | Stable at high level | Intended to remain OpenAI-style; edge behavior may improve. |
| `/metrics` | Stable | Prometheus endpoint; Metal now reports live queue / latency / MLX memory gauges. |
| `/v1/stats` | Stable | Human-readable stats endpoint; Metal now reports live queue / latency / MLX memory gauges. |
| Metal runtime memory knobs | Beta | `metal_request`, `metal_bench`, and `metal_serve` expose `--memory-limit-bytes`, `--cache-limit-bytes`, and `--wired-limit-bytes` for MLX allocator control. |
| CLI agent slash commands | Beta | Usable and documented, but not yet treated like the HTTP API for compatibility. |

---

## 6. CI Coverage Matrix

| Area | Coverage |
| --- | --- |
| Rust CPU-only compile/test | Yes |
| Python tests | Yes |
| Metal compile/test | Yes |
| CUDA compile | Partial |
| CUDA runtime correctness | No full hosted CI |
| Performance regression gating | Not yet standardized |

---

## 7. Update Rule

If support changes for a backend, model family, platform, or quantization path,
update all of the following together:

1. `README.md`
2. `ROADMAP.md` if roadmap status changed
3. this file
4. `CHANGELOG.md` when user-visible

Related docs:

- [stability-policy.md](stability-policy.md)
- [compatibility.md](compatibility.md)
