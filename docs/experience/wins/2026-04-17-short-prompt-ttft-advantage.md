# Short-prompt TTFT advantage — infer beats sglang structurally on Qwen3-4B

## Context

Promotion of the structural finding from
[`2026-04-17-ttft-scaling-infer-vs-sglang-l4.md`](2026-04-17-ttft-scaling-infer-vs-sglang-l4.md)
into a standalone wins entry. This is the **agent/chat workload advantage**:
on every measurement from 128 to 1024 input tokens, infer admits and produces
the first token faster than sglang 0.5.10 on the same L4 GPU with the same
Qwen3-4B bf16 weights.

- Hardware: NVIDIA L4 24GB, CUDA 12.8, driver 580.82.07
- Model: Qwen/Qwen3-4B bf16
- Both servers: 10 slots, 5120 max seq, `--mem-fraction-static 0.88`
- Single-request streaming `/v1/completions`, unique prompt per run (seeded
  random token substitution defeats prefix cache), 5 runs per point, median.

## Headline — median first-byte TTFT

| prompt tokens | infer (ms) | sglang (ms) | Δ vs sglang |
|--------------:|-----------:|------------:|:------------|
|  **128** | **63**  | 100 | **-37% faster** |
|  **512** | **102** | 145 | **-30% faster** |
| **1024** | **190** | 210 | **-10% faster** |
|      2048 | 380 | 350 | +8.6% slower |
|      4096 | 797 | 697 | +14.3% slower |

Crossover point: **n ≈ 1500 tokens**. Below that, infer wins. This is the
shape of almost every agent and chat workload — short system prompt + tool
descriptions + one user turn rarely exceed 1000 tokens.

## Why this is structural, not a tuning artefact

### Admission floor (128-token case, ~63ms vs 100ms)

At 128 tokens the GPU prefill work is negligible. The 37ms gap is the
admission stack:

| stage | infer (Rust) | sglang (Python) |
|-------|--------------|-----------------|
| HTTP parse + route | tokio/axum | uvicorn/FastAPI |
| tokenize | `tokenizers` crate | HF transformers |
| scheduler enqueue | in-process channel | torch op dispatch |
| first kernel launch | cudarc direct | torch → Triton/compile |

No Python GIL, no torch dispatch per kernel, no PyTorch eager-mode graph
manufacture. Rust's cost is linker-level; Python's cost is interpreter +
reference counting + dispatch tables. **This gap is not closable by sglang
without rewriting the admission path in a compiled language.**

### Per-token prefill cost (4096-token case)

Per-token prefill cost ratio (infer / sglang):
- 512 tok: **0.87** (infer faster)
- 1024 tok: 1.17 (infer slightly slower)
- 2048 tok: 1.26
- 4096 tok: **1.24**

The ratio settles ~1.24-1.26 from 2048 onward — a constant-factor kernel
efficiency gap, not an algorithmic divergence. We share FlashInfer; sglang's
wrapper is tighter at the batched API level (they use `BatchPrefill*`, we
use `SinglePrefill*`). That's a targetable gap, not a structural one.
Separate plan in `docs/plans/qwen35-single-graph-prefill.md` and the
FlashInfer plan-cache work address the long-prompt regime.

## Where this matters

- **Agent loops**: system prompt + tool list + one user message ~= 500-900
  tokens. TTFT floor on infer ≈ 100ms; sglang ≈ 150ms.
- **Chat assistants**: most turns <1024 tokens of context. infer ≈ 25%
  faster to first byte across the typical range.
- **Tool-use workloads**: multiple short-prompt completions per logical
  turn — the 37ms admission floor compounds linearly.

## What to measure next

This win records single-request TTFT. The short-prompt advantage under
concurrent load (say, 50 r/s of 512-tok prompts) is the real deployment
story and is not captured here. Open item:

- Run a 512-tok `prompt_tokens=512,output_tokens=128` guidellm sweep on
  both engines at matched concurrency and publish a paired wins entry.
- Template: `scripts/bench_guidellm.sh qwen3-4b-short-prompt-<backend>`.

## Rule

**TTFT has two humps: admission floor (CPU-framework cost) and per-token
prefill cost (GPU-kernel cost).** Infer wins the first hump by ~37ms
structurally on Qwen3-4B. Any future benchmark that lumps them into a
single p50/p99 number is misleading — report the crossover prompt length
alongside the number.

## Artefacts

- Probe script: `/tmp/ttft_probe.py`
- Source data: `2026-04-17-ttft-scaling-infer-vs-sglang-l4.md`
- Paired parity run: `2026-04-17-sglang-p99-parity-qwen3-4b.md`
