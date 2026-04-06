# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Radix-tree prefix cache for cross-request KV reuse
- Paged KV block manager with copy-on-write sharing
- Token-level KV pool (page_size=1, FlashInfer-compatible)
- GPU-CPU KV offload for contexts beyond VRAM capacity
- CUDA Graph batched decode (per batch size 1-32)
- Continuous batching scheduler with decode-priority and chunked prefill
- OpenAI-compatible API (`/v1/completions`, `/v1/chat/completions`, SSE streaming)
- Prometheus metrics (`/metrics`) and stats endpoint (`/v1/stats`)
- Qwen3 (0.5B-72B) and Qwen3.5-4B model support
- Built-in agent runtime with tool calling (shell, python)
- Benchmark suite (throughput, agent, multi-request)
- macOS Metal backend (experimental)

### Performance
- TTFT 4.6x faster than SGLang v0.5.9 (8.6ms vs 39.3ms at C=1)
- Throughput parity: 0.99x at C=1, 0.92x at C=8
- 100% KV cache hit rate on multi-turn agent benchmarks
