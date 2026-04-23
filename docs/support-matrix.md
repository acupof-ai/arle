# Support Matrix

This document states what `agent-infer` currently supports, what is still
limited, and what validation exists for each area.

If something is not listed as supported here, do not assume it is supported
just because it compiled locally.

State reflected here is based on repository evidence as of 2026-04-23.

---

## 1. Runtime Backends

| Backend | Status | Meaning |
| --- | --- | --- |
| CUDA | Supported | Primary serving path. Main runtime, scheduler, and benchmark focus. |
| Metal | Beta | Usable for local validation and live scheduler-backed serving. Qwen3 ships live prefix reuse with a shared KV pool; Qwen3.5 now ships live prefix reuse via replayed compiled-path snapshots; `scripts/start_metal_serve.sh` is the canonical first-time Apple bring-up path. Metal is still missing full batched-decode parity with CUDA, especially on variable-length Qwen3.5 decode. |
| Metal DFlash | Beta | Apple Silicon speculative decode path. Default-on for Qwen3 and Qwen3.5; benchmark before production use. |
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
| Qwen3.5 | Supported | Supported on normal runtime paths; Metal live runtime now has a narrow same-length decode batch path with packed-batch concurrent decode (2026-04-16 fix). Dense Qwen3.5 GGUF now also loads on Metal via Rust GGUF parsing + load-time dequant (validated locally on `Qwen3.5-0.8B-Q4_K_M`, 2026-04-23). Metal DFlash is Beta; see §4a for the current validation note. |
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
| GGUF loading | Beta | Supported loader path. CUDA ships the native packed Q4_K GPU kernel (`q4k_gemv_kernel` + packed fast path in `crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`) — fits Carnice-27B on L4-24GB. Metal now supports dense Qwen3.5 GGUF on Apple Silicon via the shared Rust GGUF parser with load-time BF16 dequant. |

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
| Metal DFlash (Qwen3) | Beta | Apple Silicon speculative decode path. Validated on Qwen3; benchmark before production use. |
| Metal DFlash (Qwen3.5) | Beta | End-to-end correctness landed 2026-04-17 (commits `4db4fe9`, `439293d`); benchmark before production use. |
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
| Train-side `/v1/train/status|events|stop|save` via `pretrain --serve`, `train_sft --serve`, `train_grpo --serve`, `train_multi_turn --serve` | Beta | Current control-plane truth lives in `crates/train`. Shared control-plane wiring is live on all four binaries, and CUDA has now been validated on all four active train surfaces. `train_grpo` and `train_multi_turn` were both exercised on CUDA for live `/v1/train/{status,events,save,stop}` control-plane behavior on 2026-04-21. `infer` can now expose the same surface as an optional proxy via `--train-control-url`, while the train-side server remains the sole authority. |
| Metal runtime memory knobs | Beta | `metal_request`, `metal_bench`, and `metal_serve` expose `--memory-limit-bytes`, `--cache-limit-bytes`, and `--wired-limit-bytes` for MLX allocator control. |
| CLI agent slash commands | Beta | Usable and documented, but not yet treated like the HTTP API for compatibility. |

## 5a. Training Surface Matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `pretrain` | Supported | Canonical scratch-pretrain entrypoint for the current Qwen-family train stack. HF-style checkpoint dirs + `latest` marker + exact optimizer-state resume. CUDA save/eval/resume was validated on L4 on 2026-04-21. Hybrid Qwen3.5 scratch-pretrain is now accepted locally on CPU + Metal via `pretrain --linear-attn-every 2 -> eval_lm` using the same checkpoint layout. |
| `train_sft` | Supported | Qwen3 / Qwen3.5 family dispatch, LoRA-only fine-tune surface, adapter-aware checkpointing and resume. CUDA smoke was validated on Qwen3-0.6B for `train_sft -> eval_lm -> agent-infer -> resume` on 2026-04-21, and Mac-local Metal validation covers the dense/full-attn Qwen3.5 path (`pretrain -> train_sft --backend metal -> eval_lm -> resume`, LoRA rank=8, Apple M4 Pro, 2026-04-21). Hybrid linear-attn Qwen3.5 LoRA fine-tune is also validated on CPU + Metal for tiny synthetic checkpoints on 2026-04-21; CUDA compile surface is checked, but CUDA runtime acceptance for the hybrid path is not yet closed. |
| `train_grpo` | Supported | Single-turn RL surface with exact checkpoint/resume, shared observability/control-plane wiring, and backend selection across `cpu|metal|cuda`. CUDA was validated on 2026-04-21 for `train_grpo -> checkpoint/latest -> eval_lm -> resume` plus live `/v1/train/{status,events,save,stop}` control-plane behavior on the synthetic dense Qwen3.5 path. Hybrid linear-attn Qwen3.5 end-to-end acceptance is now closed locally on CPU + Metal for `train_grpo --linear-attn-every 2` including checkpoint materialization; CUDA compile surface is checked, but CUDA hybrid runtime acceptance is still pending. |
| `train_multi_turn` | Supported | Backend flag supports `cpu|metal|cuda`. CUDA was validated on 2026-04-21 for stepwise GRPO, sequence-level GSPO, exact resume, `/v1/train/{status,events,save,stop}` control-plane endpoints, and checkpoint reload through `eval_lm` on the dense/full-attn path. Hybrid linear-attn Qwen3.5 end-to-end acceptance is now closed locally on CPU + Metal: stepwise-GRPO and sequence-level-GSPO both run against hybrid configs and save checkpoints; CUDA compile surface is checked, but CUDA hybrid runtime acceptance is still pending. |
| `eval_lm` | Supported | Standalone loss / perplexity evaluation for Qwen3 / Qwen3.5 checkpoint dirs on tokenized or chat JSONL. Hybrid linear-attn Qwen3.5 checkpoint evaluation is validated on CPU + Metal on 2026-04-21. |
| Hybrid linear-attn Qwen3.5 LoRA/eval path | Supported | `Qwen35Model` now supports hybrid linear-attn layers for LoRA/frozen-eval use. Acceptance is closed for `train_sft` + `eval_lm` on CPU + Metal using tiny synthetic checkpoints, and the CUDA compile surface is checked. |
| Hybrid linear-attn Qwen3.5 scratch pretrain / RL acceptance | Supported on validated CPU + Metal path | Hybrid scratch pretrain now runs through `pretrain --linear-attn-every > 0`, `eval_lm` reloads the resulting checkpoints, `train_grpo` accepts hybrid configs end-to-end on CPU + Metal, and `train_multi_turn` now has end-to-end hybrid acceptance for both stepwise-GRPO and sequence-level-GSPO across CPU + Metal. CUDA compile surface is checked; CUDA runtime acceptance for the hybrid path remains pending. |
| Infer-side unified `/v1/train/*` bridge | Supported (optional proxy) | `infer` now exposes `/v1/train/status|events|stop|save` when `--train-control-url http://...` is configured, forwarding to the train-side server in `crates/train/src/server.rs` without duplicating trainer logic. Live proxying was validated on 2026-04-21 with `pretrain --serve` behind `cpu_serve --train-control-url`. |

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
