# ARLE Support Matrix

This document is the canonical support-status truth for `ARLE`.

It states what the repository currently supports, what is still limited, and
what validation exists for each area. If something is not listed as supported
here, do not assume it is supported just because it compiled locally.

State reflected here is based on repository evidence as of 2026-04-27.
Project framing lives in [index.md §Current Positioning](index.md#current-positioning).

---

## 1. Runtime Backends

| Backend | Status | Meaning |
| --- | --- | --- |
| CUDA | Supported | Primary serving path. Main runtime, scheduler, and benchmark focus. |
| Metal | Beta | Usable for local validation and live scheduler-backed serving. Qwen3 ships live prefix reuse with a shared KV pool; Qwen3.5 ships live prefix reuse via replayed compiled-path snapshots; `scripts/start_metal_serve.sh` is the canonical first-time Apple bring-up path. Qwen3.5-0.8B GGUF Q4_K_M decode is now measured at 211.7 tok/s on M4 Pro after Q5_K/Q8_0 affine repack and Q6/group16 qmv tile tuning. Metal is still missing full batched-decode parity with CUDA, especially on variable-length Qwen3.5 decode. |
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
| Qwen3.5 | Supported | Supported on normal runtime paths; Metal live runtime has a narrow same-length decode batch path with packed-batch concurrent decode (2026-04-16 fix). Qwen3.5-0.8B GGUF Q4_K_M now runs on Metal through the shared Rust GGUF parser plus MLX affine/tiled quant paths for Q4/Q5/Q6/Q8 hot tensors, validated locally at 211.7 tok/s for 512 prompt / 1024 decode on 2026-04-27. Metal DFlash is Beta; see §4a for the current validation note. |
| Qwen3.6 / Qwen3.5-MoE | Beta (Metal), CUDA stub | Metal loads and runs `mlx-community/Qwen3.6-35B-A3B-4bit` locally. A 2026-04-27 M4 Pro short diagnostic confirmed load/execute behavior, but DFlash performance decisions for this family should use long-context / ultra-long-sequence workloads only. CUDA intentionally returns a GPU-required stub for Qwen3.6 MoE. |
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
| GGUF loading | Beta | Supported loader path. CUDA ships the native packed Q4_K GPU kernel (`q4k_gemv_kernel` + packed fast path in `crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`) — fits Carnice-27B on L4-24GB. Metal supports Qwen3.5 GGUF on Apple Silicon via the shared Rust GGUF parser; Qwen3.5-0.8B Q4_K_M now keeps the decode hot path on MLX affine/tiled quant kernels for the large Q4/Q5/Q6/Q8 tensors instead of falling back to broad load-time BF16 dequant. |

Backend note:

- The `FP8 KV cache`, `INT8 KV cache`, and `TurboQuant KV` rows above describe
  the shipped project-wide quantized-KV work, which is currently CUDA-backed.
- Metal / MLX does **not** currently ship quantized KV cache. The live Metal
  path stores KV in the model's native dtype today, typically `bf16` / `f16`,
  and it does not expose a `--kv-cache-dtype` surface.
- Metal can still run weight-quantized MLX models; that is separate from
  quantized KV cache support.

---

## 4b. Multi-turn KV Reuse / Tiered KV Matrix

The KV-reuse architecture that the README calls out (slot-sticky multi-turn
reuse + radix-backed `T0 GPU → T1 host pinned → T2 NVMe → T3 cluster-shared`).
Code lives in `infer/src/prefix_cache.rs` (radix tree) and
`infer/src/kv_tier/` (tiered-KV plumbing); see
[`docs/codebase-map.md`](codebase-map.md) for the per-file map.

| Capability | Status | Notes |
| --- | --- | --- |
| Slot-sticky multi-turn KV reuse | Supported (CUDA), Beta (Metal) | Prior-turn KV stays in slot for the next turn so only new user tokens prefill. CUDA is the primary path; Metal Qwen3 ships live prefix reuse via shared KV pool, Qwen3.5 via replayed compiled-path snapshots (see §1). |
| Radix-backed prefix cache (T0 GPU) | Supported (CUDA) | Direct GPU-page attach + tail-page CoW on shared prefixes; `RadixNode` carries `hit_count`, `tier_location`, `session_id`, `fingerprint`, `soft_pin_until`, `byte_len`. |
| T1 host-pinned spillover | Beta (CUDA) | Cold blocks demote from GPU to host pinned memory via `HostPinnedPool` (Zig-backed); promote-on-use through `ReadmissionPlan`. |
| T2 NVMe local-disk transport | Beta (CUDA) | Node-local persistence via `kv_tier/transport/disk.rs` on top of `crates/kv-native-sys` (file/block ABI, mmap, WAL). |
| T3 cluster-shared backend | Experimental | Minimal `transport/shared_fs.rs` reference backend ships; **NIXL transport remains stub-only** (`nixl-sys` activates the stub feature, no real link). Treat T3 as scaffolding, not a production tier today. |

---

## 4a. Speculative Decoding Matrix

| Capability | Status | Notes |
| --- | --- | --- |
| Metal DFlash (Qwen3) | Beta | Apple Silicon speculative decode path. Validated on Qwen3; benchmark before production use. |
| Metal DFlash (Qwen3.5) | Beta | End-to-end correctness landed 2026-04-17 (commits `4db4fe9`, `439293d`); benchmark before production use. |
| Metal DFlash (Qwen3.6 / Qwen3.5-MoE) | Beta / diagnostic | Target/draft pairing is wired for `mlx-community/Qwen3.6-35B-A3B-4bit` + `z-lab/Qwen3.6-35B-A3B-DFlash`. Short checks are smoke diagnostics only; future DFlash optimization claims must come from long-context / ultra-long-sequence runs. |
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
| `arle serve` front door | Beta | Launches the matching serving binary (`infer`, `metal_serve`, or `cpu_serve`) from the release artifact or PATH. This is a packaging/DX front door over existing server binaries, not a second HTTP implementation. |
| CLI built-in shell/python tools | Beta | Enabled by default for local trusted agent use. `--no-tools` disables them, and `arle --doctor` reports the detected sandbox backend (`nsjail`, `sandbox-exec`, or `bare`). Do not expose tool-enabled local agent prompts to untrusted users. |

## 5a. Training Surface Matrix

| Surface | Status | Notes |
| --- | --- | --- |
> All training surfaces are reached through `arle train ...` / `arle data ...`. The standalone `pretrain` / `train_sft` / `train_grpo` / `train_multi_turn` / `eval_lm` / `download_dataset` / `convert_dataset` binaries that previously shipped from `crates/train` are no longer produced; their dispatch logic is included in-process from `crates/cli/src/train_cli.rs`.

| `arle train pretrain` | Supported | Canonical scratch-pretrain surface for the current Qwen-family train stack. HF-style checkpoint dirs + `latest` marker + exact optimizer-state resume. CUDA save/eval/resume was validated on L4 on 2026-04-21. Hybrid Qwen3.5 scratch-pretrain is accepted locally on CPU + Metal via `arle train pretrain --linear-attn-every 2 -> arle train eval` using the same checkpoint layout. |
| `arle train sft` | Supported | Qwen3 / Qwen3.5 family dispatch, LoRA-only fine-tune surface, adapter-aware checkpointing and resume. CUDA smoke was validated on Qwen3-0.6B for `train sft -> train eval -> arle -> resume` on 2026-04-21, and Mac-local Metal validation covers the dense/full-attn Qwen3.5 path (`train pretrain -> train sft --backend metal -> train eval -> resume`, LoRA rank=8, Apple M4 Pro, 2026-04-21). Hybrid linear-attn Qwen3.5 LoRA fine-tune is also validated on CPU + Metal for tiny synthetic checkpoints on 2026-04-21; CUDA compile surface is checked, but CUDA runtime acceptance for the hybrid path is not yet closed. |
| `arle train grpo` | Supported | Single-turn RL surface with exact checkpoint/resume, shared observability/control-plane wiring, and backend selection across `cpu|metal|cuda`. CUDA was validated on 2026-04-21 for `train grpo -> checkpoint/latest -> train eval -> resume` plus live `/v1/train/{status,events,save,stop}` control-plane behavior on the synthetic dense Qwen3.5 path. Hybrid linear-attn Qwen3.5 end-to-end acceptance is closed locally on CPU + Metal for `arle train grpo --linear-attn-every 2` including checkpoint materialization; CUDA compile surface is checked, but CUDA hybrid runtime acceptance is still pending. |
| `arle train multi-turn` | Supported | Backend flag supports `cpu|metal|cuda`. CUDA was validated on 2026-04-21 for stepwise GRPO, sequence-level GSPO, exact resume, `/v1/train/{status,events,save,stop}` control-plane endpoints, and checkpoint reload through `arle train eval` on the dense/full-attn path. Hybrid linear-attn Qwen3.5 end-to-end acceptance is closed locally on CPU + Metal: stepwise-GRPO and sequence-level-GSPO both run against hybrid configs and save checkpoints; CUDA compile surface is checked, but CUDA hybrid runtime acceptance is still pending. |
| `arle train eval` | Supported | Loss / perplexity evaluation for Qwen3 / Qwen3.5 checkpoint dirs on tokenized or chat JSONL. Hybrid linear-attn Qwen3.5 checkpoint evaluation is validated on CPU + Metal on 2026-04-21. |
| Hybrid linear-attn Qwen3.5 LoRA/eval path | Supported | `Qwen35Model` supports hybrid linear-attn layers for LoRA/frozen-eval use. Acceptance is closed for `arle train sft` + `arle train eval` on CPU + Metal using tiny synthetic checkpoints, and the CUDA compile surface is checked. |
| Hybrid linear-attn Qwen3.5 scratch pretrain / RL acceptance | Supported on validated CPU + Metal path | Hybrid scratch pretrain runs through `arle train pretrain --linear-attn-every > 0`, `arle train eval` reloads the resulting checkpoints, `arle train grpo` accepts hybrid configs end-to-end on CPU + Metal, and `arle train multi-turn` has end-to-end hybrid acceptance for both stepwise-GRPO and sequence-level-GSPO across CPU + Metal. CUDA compile surface is checked; CUDA runtime acceptance for the hybrid path remains pending. |
| Infer-side unified `/v1/train/*` bridge | Supported (optional proxy) | `infer` exposes `/v1/train/status|events|stop|save` when `--train-control-url http://...` is configured, forwarding to the train-side server in `crates/train/src/server.rs` without duplicating trainer logic. Live proxying was validated on 2026-04-21 with `arle train pretrain --serve` behind `cpu_serve --train-control-url`. |

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
3. `docs/index.md` if the active-doc listing changed
4. this file
5. `CHANGELOG.md` when user-visible

Related docs:

- [stability-policy.md](stability-policy.md)
