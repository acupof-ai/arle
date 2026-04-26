# ARLE vs other inference runtimes

A grounded positioning document for "should I use ARLE or X?". Updated
2026-04-26. Linked projects below have their own docs that supersede anything
this page summarizes.

> **TL;DR.** Pick **vLLM** or **SGLang** if you need broad model coverage and
> are happy on Python. Pick **llama.cpp** if you need broad hardware coverage
> on consumer machines and edge devices. Pick **ARLE** if you specifically
> want a Rust runtime where serving, the local agent loop, and the train / RL
> stack share one set of model + scheduler code paths, and you are working on
> Qwen3 / Qwen3.5 today.

## What ARLE is not

ARLE is **not** a drop-in vLLM replacement. As of 2026-04-26 the supported-
model list is short (Qwen3 0.6B–72B, Qwen3.5-4B); see
[support-matrix.md](support-matrix.md). If "support 50 model families on day
one" is on your requirements list, use vLLM.

ARLE is also **not** a research framework — it is a serving / agent runtime.
Researchers wanting tensor-level control closer to PyTorch should look at
candle or directly at PyTorch.

## Positioning grid

| | Language | Models | Multi-turn KV reuse | Train / RL surface | Best fit |
|---|---|---|---|---|---|
| **ARLE** | Pure Rust | Qwen3 / Qwen3.5 | Slot-sticky + radix-backed tiered KV (T0 GPU → T1 host → T2 disk → T3 cluster-shared); CUDA + Metal | Same runtime, in-tree (`arle train pretrain/sft/grpo/multi-turn/eval`) | Rust shops; agent / RL workloads on Qwen3 family that pay a heavy prefill tax per turn |
| **vLLM** | Python (CUDA / ROCm) | Broad (Llama, Qwen, Mistral, DeepSeek, …) | PagedAttention + prefix cache | Separate (vLLM serves; train is your problem) | Production Python serving with a wide model menu |
| **SGLang** | Python | Broad | RadixAttention prefix tree | Structured generation / multi-step prompting strengths | Python serving with structured / agent prompting |
| **mistral.rs** | Pure Rust | Broad (multimodal too) | KV cache + prefix cache | Inference-focused | Rust serving with broad model coverage |
| **candle** | Rust (HF) | Many examples | Library, not a serving runtime | n/a (you wire it up) | Rust ML library for custom serving / research |
| **llama.cpp** | C / C++ | Broad (GGUF) | KV cache | Inference-focused, fine-tune via gguf-tools | Edge / consumer hardware, GGUF-first deployments |

The grid is intentionally sparse — every project does more than one column
shows. Read each project's own docs before committing.

## What ARLE optimizes for that the others don't

1. **Multi-turn agent / RL workloads where prefill dominates latency.** Every
   tool-using agent turn re-processes system prompt + history + tool results;
   ARLE's CUDA scheduler keeps prior-turn KV hot via slot-sticky reuse and a
   radix-backed tiered-KV path so only the new user message prefills. vLLM
   and SGLang both have prefix caching, but ARLE's tiered-KV (T0 → T3)
   spill / promote design is built for the agent loop's working-set pattern,
   not the multi-tenant LLM-as-a-service pattern.

2. **One Rust runtime, no Python control plane.** `infer`, the local `arle`
   agent runtime, and the in-tree train / RL stack all share the same Rust
   model and scheduler code. There is no `engine.py` driving a C++ engine
   and re-implementing model logic on top. For mixed serving + RL workloads
   (rollouts → reward → update) this removes a class of "the trainer's
   model definition drifted from the server's" bugs.

3. **Two backends from one trait.** CUDA (Linux) and Metal (Apple Silicon)
   plug into the same `server_engine::InferenceEngine` contract, so a Rust
   shop can develop on a MacBook and ship on Linux GPUs without rewriting
   model code. CPU is dev-only.

## What ARLE is intentionally not racing

- **Model coverage.** Adding Llama 3 / 4 and DeepSeek V3 / R1 is on the
  [ROADMAP](../ROADMAP.md), not shipped. vLLM / SGLang / mistral.rs / llama.cpp
  all have far broader coverage today.
- **Multi-GPU tensor parallel.** Not in scope as of 2026-04-26. Single-GPU
  serving is the supported path.
- **Quantization breadth.** GPTQ W4 / AWQ W4 / FP8 / INT8 / Q4_K GGUF / MLX
  4-bit are Beta — see
  [support-matrix.md §Quantization](support-matrix.md). Production-critical
  exotic quants live in vLLM / llama.cpp.
- **Python ergonomics.** Embedding ARLE means linking the Rust `infer` crate,
  not `pip install`. There is no `from arle import LLM`-style API.

## When to pick ARLE specifically

- You are building or evaluating an **agent / RL** workload on **Qwen3 or
  Qwen3.5** and want serving + rollout / training to share runtime code.
- You want a **pure-Rust serving binary** (`infer`) that you can embed
  without a Python sidecar.
- You are on **Apple Silicon** and want a runtime where Metal is a
  first-class backend, not a port.
- You care about **multi-turn KV reuse with disk / cluster spillover** and
  the working-set pattern matches your prefill-heavy traffic.

## When to pick something else

- You need a model family ARLE does not list under
  [support-matrix.md §Models](support-matrix.md).
- You need multi-GPU tensor parallel today.
- Your team is Python-only and a Rust dependency would slow you down more
  than ARLE's properties speed you up.
- You are deploying to CPU / edge devices where llama.cpp's GGUF coverage is
  the canonical answer.

If after reading this you are still on the fence, the lowest-cost test is to
pull `ghcr.io/cklxx/arle:latest` and run [`scripts/bench_guidellm.sh`](../scripts/bench_guidellm.sh)
against your own traffic profile, then compare to the same workload on the
runtime you would have picked otherwise.
