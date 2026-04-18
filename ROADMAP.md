# agent-infer Roadmap

**Two parallel main lines as of 2026-04-18**:

1. **Inference engine** (historical main line) — production-grade LLM inference on par with sglang (excluding KV-affinity routing). Phases 0–5 below.
2. **Agent RL 训推一体 · 自进化栈** (new core main line) — Phase 6. Single-node, same-process train+inference, Rust-native autograd written from scratch, LoRA+GRPO, self-evolving agent via multi-verifier reward. See [docs/projects/agent-rl-self-evolving.md](docs/projects/agent-rl-self-evolving.md).

Related docs:

- [docs/projects/agent-rl-self-evolving.md](docs/projects/agent-rl-self-evolving.md) — **Phase 6 project spine** · architecture, scope lock, milestones, risks
- [docs/plans/rust-agent-rl-single-node.md](docs/plans/rust-agent-rl-single-node.md) — **Phase 6 execution plan** · M0–M5 with daily-grain tasks + acceptance gates
- [docs/research/mni-ml-framework-notes.md](docs/research/mni-ml-framework-notes.md) — mni-ml/framework reference analysis (the from-scratch autograd template)
- [docs/projects/xma-future-research.md](docs/projects/xma-future-research.md) — external research note on XMA / accelerated-model-architectures and what it implies for future architecture work

---

## Current State (2026-04-19)

Working: Qwen3/Qwen3.5/GLM4 inference on CUDA + Metal, FlashInfer single prefill (HD128) + Triton FA2 (HD256), FlashInfer batched decode attention, **Tiered KV cache (T0 GPU → T1 host pinned → T2 NVMe → T3 remote NIXL)** with radix-driven prefix reuse and page-level staging (`page_size=16` for BF16), token-level KV pool, continuous batching with chunked prefill (4096 tok), decode-priority scheduling, prefix-aware slot assignment with **recurrent state snapshot/restore for hybrid models**, merged QKV + gate-up GEMM (96 fewer launches/step), CUDA Graph batched decode (per batch size), top-k/p/temp/min-p/penalty sampling, batched sampling, OpenAI `/v1/completions` + `/v1/chat/completions` + `/v1/responses` (non-streaming) + `/v1/models` + SSE, Rust agent binary, Python async agent, Prometheus `/metrics` + `/v1/stats` endpoints, model architecture registry, radix-tree prefix cache (tier-aware `RadixNode` with fingerprint + session affinity), speculative decoding framework (CPU stubs) + Metal DFlash experimental path on Qwen3, tensor parallel config/sharding math (CPU stubs), **weight quantization W2/W4/W8 + Marlin W4 prefill + TurboQuant 3-bit**, **KV quantization FP8/INT8/TurboQuant 2-4 bit + fused-dequant attention**, **native Q4_K GPU kernel + GGUF loader (BF16/F16/Q8_0/Q4_K_M)**, **Metal backend with resumable request state (M0.2a)**, throughput benchmark suite.

**Recent milestones (April 2026)**:
- Qwen3-8B throughput at SGLang parity: C=1 -8%, C=4 +2%, TTFT 2.5x faster
- Qwen3.5-4B scheduler + FlashInfer HD256 batched decode: C=1 123 tok/s, C=4 428 tok/s (+14% over SGLang)
- **Qwen3.5 prefix cache enabled** via recurrent state snapshot/restore (was disabled due to state contamination)
- **TurboQuant complete**: KV cache 3-bit (5x compression), weight 3-bit (fused dequant GEMV), fused decode attention
- **GPTQ/AWQ INT4 production-ready**: format detection, W4A16 GEMV, Marlin W4 prefill (5-25x TTFT speedup)
- **FP8 KV cache**: custom fused FP8 E4M3 decode attention, 50% KV memory reduction
- Merged QKV + gate-up GEMM: 96 fewer kernel launches per decode step
- **Tiered KV Cache M2b + M0.3 + M3a + M3b + M3c locally shipped and remote-accepted on L4 (2026-04-15)**: scheduler selector flip to radix, BF16 `page_size=16`, host-tier skeleton, `lookup_or_stage` contract + page-lifecycle state machine, legacy contiguous CPU KV offload retired. See `docs/projects/tiered-kv-cache.md`.
- **`infer-cuda-kernels` kernel crate extracted** (commit `a4e12f5`): CUDA Rust layer (`paged_kv`, `flashinfer`, `graph_pool`, `tensor`, `ffi`, `kv_quant`, `kv_turboquant`) moved to `crates/infer-cuda-kernels/`; `infer/src/backend/cuda/` keeps only `bootstrap.rs`. One-way dependency `infer → infer-cuda-kernels`.
- **Q4_K native GPU kernel shipped**: `q4k_gemv_kernel` + packed GGUF loader fast path fits Carnice-27B on L4-24GB. See `docs/plans/q4k-native-gpu.md`.
- **Metal M0.2a resumable request state + M0.2b scheduler-backed serving**: Qwen3 + Qwen3.5 request state objects (prefill-in-chunks, one-step decode, deterministic cleanup); `metal_serve` now routes through `SchedulerHandle` via `spawn_metal_scheduler_handle_from_path_with_options_and_metrics` (`infer/src/backend/metal/runtime.rs:580`). M0.2 throughput exit still pending variable-length decode + per-step batch-state rebuild cost work — see [docs/plans/2026-04-15-metal-backend-acceptance-plan.md](docs/plans/2026-04-15-metal-backend-acceptance-plan.md).
- **Metal packed-batch concurrent decode fixed (2026-04-16)**: `extend_kv_cache` batch-dim bug that crashed the scheduler when packed cache rolled past `KV_CACHE_CHUNK` was repaired; varlen additive mask now emitted in bf16 for MLX ≥ 0.32 SDPA.
- **Qwen3.5 DFlash correctness landed (2026-04-17)**: end-to-end DFlash tape + sticky-state reset + bf16 cast of `g`/`k` produces coherent deterministic output matching baseline. Currently a perf regression (~5× slower single-stream vs baseline; acceptance ~28%; serial across concurrent requests). Follow-ups: acceptance investigation and batch-axis packing over 16-token blocks. See [`docs/experience/wins/2026-04-17-metal-qwen35-dflash-correctness-bench.md`](docs/experience/wins/2026-04-17-metal-qwen35-dflash-correctness-bench.md).
- **CPU offload retired**: the legacy contiguous `model/kv_cache.rs` CPU-offload surface (`k_host/v_host`, `OFFLOAD_BLOCK_SIZE=64`, `prefetch/offload` hooks) deleted in M3c (`c3f65f7`). `set_max_gpu_kv` remains as a compatibility no-op warning only.
- **Phase 6 M4 stepwise GRPO + M5 autograd Backend trait (2026-04-18)**: `train_multi_turn` binary runs per-position GRPO with autoregressive rollout and held-out eval harness; `Backend` trait + `CpuBackend` land in `crates/autograd/src/backend.rs`; `MetalBackend` (mlx-sys `mlx_matmul`) in `crates/autograd/src/backend_metal.rs`; `CudaBackend` (cuBLAS SGEMM, Mac-typecheck via `todo!()` stubs, pending remote GPU validation) in `crates/autograd/src/backend_cuda.rs`. CPU vs Metal matmul parity ≤1e-3 across 3 TinyLM shapes; `--backend {cpu,metal,cuda}` flag wired. At TinyLM scale CPU beats Metal ~2× (FFI dominates); at d_model=256/4-layer scale Metal pulls ~1.4× ahead. See [`docs/experience/wins/2026-04-18-bench-train-multi-turn.md`](docs/experience/wins/2026-04-18-bench-train-multi-turn.md).
- **Qwen3.5 Metal concurrent decode closed to ceiling (2026-04-19)**: Iter 7 session-API commit `e97d52a` lands per-step FFI-amortized single-request decode; c=1 HTTP 73.1 tok/s (~13% off the 84.1 tok/s in-process tight-loop ceiling = mlx_lm parity), c=4 HTTP 162.5 tok/s is the concurrent ceiling (c=8 flat at 162.9). StepDriver tight loop 76.5 tok/s; scheduler-tick latency — not FFI — now dominates the HTTP residual gap. Eleven-iteration arc archived in [`docs/experience/wins/2026-04-19-metal-qwen35-final-state.md`](docs/experience/wins/2026-04-19-metal-qwen35-final-state.md).
- **Paged-prefill Phase 3 hoisted + follow-ups (2026-04-18)**: `prefill.rs` chunk loop hoists per-layer constants out of the hot path; guidellm sweep snapshot landed in `docs/experience/wins/2026-04-18-bench-guidellm-paged-phase3a-hoisted.md`. Throughput regression investigation + Qwen3.5 crash repro tracked in [`docs/plans/paged-prefill-followups-2026-04-18.md`](docs/plans/paged-prefill-followups-2026-04-18.md) §1/§3 (CUDA-gated).

Missing: multi-architecture GPU inference (Llama/DeepSeek/Mistral/Gemma/Phi — detection shipped in `model_registry.rs`, per-model forward not wired), MLA attention, tensor parallel communication (NCCL), speculative decoding GPU integration (CPU framework only), FlashAttention-3 (H100), Metal M0.2 throughput exit (scheduler-backed `metal_serve` shipped; pending variable-length decode + per-step batch-state rebuild cost work), Qwen3.5 CUDA batched prefill, scheduler preemption with KV swap.

---

## Implementation Philosophy

**CPU-first**: implement all control-plane logic (schedulers, data structures, APIs, config) without GPU. Mark GPU-only code with `// GPU required` stubs. This lets development and testing proceed without a GPU for the majority of the codebase.

**Incremental validation**: each module has a local-verifiable test path. GPU integration tests run on the A100 machine.

---

## Phase 0 — Foundation (CPU-verifiable, no GPU needed) ✅ Complete

All Phase 0 tasks have been implemented and are in production use.

### 0.1 Chat Completions API ✅
**Files**: `infer/src/http_server/openai_v1.rs`
**Goal**: Add `/v1/chat/completions` endpoint with message → ChatML conversion, streaming SSE, and `finish_reason` handling. Essential for LLM clients (OpenAI SDK, litellm, etc.).

```
ChatCompletionRequest { messages, model, max_tokens, temperature, stream, tools, ... }
ChatCompletionResponse / stream chunks
Tool call delta streaming (OpenAI format)
```

**Local test**: `curl -X POST http://localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"hello"}]}'`

---

### 0.2 Sampler Expansion ✅
**Files**: `infer/src/sampler.rs`, `infer/src/ops/sampling.rs`
**Goal**: Add missing sampling strategies so the engine is compatible with all common LLM configs.

```
repetition_penalty: f32        ← penalize already-generated tokens
frequency_penalty: f32         ← OpenAI-style frequency penalty
presence_penalty: f32          ← OpenAI-style presence penalty
min_p: f32                     ← min-p sampling (popular in 2024–2025)
seed: Option<u64>              ← deterministic sampling
stop_token_ids: Vec<u32>       ← additional stop tokens beyond model EOS
```

**Local test**: unit tests for penalty application on toy logit vectors (no GPU).

---

### 0.3 Radix Tree Prefix Cache ✅
**Files**: `infer/src/prefix_cache.rs`
**Goal**: Content-addressable KV cache reuse across requests. Requests sharing a common prefix (system prompt, few-shot examples) skip prefill for the shared portion.

```
RadixTree<BlockId>
  insert(tokens: &[u32]) → (hit_len, new_blocks)
  lookup(tokens: &[u32]) → (matched_prefix_len, matched_block_ids)
  evict_lru(n_blocks: usize)
  ref_count per node (requests in-flight hold refs)
```

**Integrates with**: `GenerationState::truncate_to()` already exists — radix cache provides the `hit_len` input.
**Local test**: pure data structure tests (no GPU). Fuzz with random token sequences.

---

### 0.4 Paged KV Cache Block Manager ✅
**Files**: `infer/src/block_manager.rs`, `infer/src/model/kv_cache.rs`
**Goal**: Fixed-size block allocator for KV cache. Required for PagedAttention and for radix cache to work across requests without copying.

```
BlockManager {
  block_size: usize         ← tokens per block (e.g. 16)
  num_gpu_blocks: usize
  num_cpu_blocks: usize
  free_gpu_blocks: VecDeque<BlockId>
  free_cpu_blocks: VecDeque<BlockId>
  allocate_gpu(n: usize) → Vec<BlockId>
  free(blocks: &[BlockId])
  swap_out(blocks: &[BlockId])   // GPU → CPU
  swap_in(blocks: &[BlockId])    // CPU → GPU
}
```

**Local test**: allocation/free/swap accounting logic — no GPU needed for the manager itself.

---

### 0.5 Scheduler Improvements ✅
**Files**: `infer/src/scheduler/`
**Goal**: Upgrade scheduler for PagedAttention-era requirements.

- ✅ **Priority queue**: request priority field, FCFS + optional priority override
- ✅ **Waiting queue backpressure**: `max_waiting_requests` config, returns 503 when full
- ✅ **Chunked prefill config**: configurable chunk size (4096 default, 64 when decode active)
- ✅ **Stats endpoint**: `/metrics` Prometheus counters + `/v1/stats` human-readable
- ✅ **Prefix-aware slot assignment**: KV cache reuse across requests
- ✅ **Continuous batching**: decode-priority scheduling, CUDA Graph warmup for batch sizes 1-32
- **Preemption with swap**: when GPU OOM, preempt lowest-priority request, swap its KV blocks to CPU, resume later (not yet implemented)

**Local test**: mock `ModelForward` for scheduler unit tests (no GPU).

---

### 0.6 Model Architecture Registry ✅
**Files**: `infer/src/model_registry.rs`, `infer/src/server_engine.rs`
**Goal**: Replace hard-coded `ModelType` enum with a dynamic registry. Enables adding new architectures without changing the dispatch code.

```rust
pub trait ModelConfig: for<'de> Deserialize<'de> {
    fn model_type() -> &'static str;
}

pub struct ModelRegistry {
    loaders: HashMap<&'static str, Box<dyn ModelLoader>>,
}

impl ModelRegistry {
    pub fn detect_and_load(path: &Path) -> Result<Box<dyn ErasedModelForward>>;
}
```

Register `Qwen3`, `Qwen35`, and stubs for `Llama`, `DeepSeek`, `Mistral`, `Gemma`, `Phi` upfront.

**Local test**: load config.json, assert correct architecture detection.

---

### 0.7 Benchmark Suite ✅
**Files**: `scripts/bench_throughput.py`, `scripts/bench_agent.py`, `scripts/bench_multi_request.py`
**Goal**: Comprehensive benchmark framework measuring all production metrics.

```
Metrics:
  TTFT   — time to first token (prefill latency)
  TPOT   — time per output token (decode latency, aka TBT)
  E2E    — end-to-end request latency
  Throughput — requests/s and tokens/s

Scripts:
  bench_throughput.py --num-prompts 1000 --concurrency 32
  bench_latency.py --input-len 512 --output-len 128 --batch-sizes 1,4,8,16
  bench_serving.rs (Rust binary, lower overhead)
```

ShareGPT dataset loader for realistic prompt distributions.

**Local test**: scripts run against mock HTTP server (no GPU).

---

## Phase 1 — Core GPU Features (GPU required)

### 1.1 PagedAttention Kernel ✅ Shipped
**Files**: `crates/infer-cuda-kernels/csrc/attention/{flashinfer_decode,flashinfer_decode_hd256,decode_prep_paged,decode_prep_paged_hd256,fused_attention}.cu`, `infer/src/ops/attention.rs`
**Goal**: Replace contiguous KV cache with paged blocks. Eliminates memory fragmentation, enables sharing blocks across requests (prefix cache), enables swap.

Shipped via a three-path split (not a single `paged_attention_decode`):
- **BF16 decode**: FlashInfer paged decode via `flashinfer_decode.cu` (HD128) + `flashinfer_decode_hd256.cu` (HD256).
- **INT8/FP8 decode**: custom split-KV kernels with in-register fused dequant in `decode_attention_quantized.cu`.
- **TurboQuant decode**: Q-rotation + centroid gather in `decode_attention_turboquant.cu` — no KV dequantization materialized.
- **Prefill**: FlashInfer `prefill_attention.cu` + `prefill_attention_hd256.cu`.
- **Paged metadata staging**: `flashinfer_metadata.cu` + `decode_prep_paged{,_hd256}.cu` build per-request page-indices on GPU.

Pool bookkeeping lives in `crates/infer-cuda-kernels/src/paged_kv.rs` (`PagedKVPool` + `TokenKVPool`).

---

### 1.2 FlashAttention-3 (prefill)
**Files**: `crates/infer-cuda-kernels/tools/triton/flash_attention_v3_kernel.py`, `infer/src/ops/attention.rs`
**Goal**: Upgrade prefill kernel from FA2 → FA3 for better A100/H100 utilization. FA3 uses warp specialization + async pipeline.

Key improvements over FA2:
- Ping-pong SMEM pipeline (2× producer/consumer overlap)
- FP8 accumulation path for H100
- Better causal mask handling for long sequences

Keep FA2 as fallback for older GPUs (< SM90).

---

### 1.3 Multi-Head Latent Attention (MLA) — DeepSeek
**Files**: `infer/src/ops/attention.rs`, `infer/src/model/deepseek/`
**Goal**: Implement MLA as used in DeepSeek-V2/V3/R1. MLA compresses KV cache to a latent vector (rank << d_model), drastically reducing KV memory.

```
MLA:
  KV compression: C_KV = W_DKV · x   [d_c << d_head * num_kv_heads]
  K decompression: K = W_UK · C_KV   [num_kv_heads, d_head]
  V decompression: V = W_UV · C_KV
  Q compression: C_Q = W_DQ · x (optional)
  RoPE decoupled: only rope-part of K/Q goes through position encoding
```

Cache only `C_KV` (compressed), not `K, V` — huge memory saving.

---

### 1.4 Llama 3/4 Model
**Files**: `infer/src/model/llama/`
**Goal**: Support Meta's Llama 3 (8B, 70B, 405B) and Llama 4 Scout/Maverick.

Llama 3 differences from Qwen3:
- RoPE with `rope_scaling` (YaRN / dynamic)
- Tied embeddings
- Group query attention (8 KV heads for 70B/405B)
- No QK norm
- Different tokenizer (tiktoken-based)

Llama 4 additions:
- Mixture of Experts (MoE) layers
- Interleaved attention + MoE blocks
- 128-expert architecture

---

### 1.5 DeepSeek-V3 / R1 Model
**Files**: `infer/src/model/deepseek/`
**Goal**: Support DeepSeek-V3 (671B MoE) and R1 (671B MoE + chain-of-thought).

Key components:
- MLA (see 1.3)
- DeepSeekMoE: 256 routed experts + 2 shared experts, top-8 routing
- Multi-Token Prediction (MTP) head for speculative decoding
- FP8 mixed precision (E4M3 weights, BF16 compute)

---

### 1.6 Mistral / Mixtral Model
**Files**: `infer/src/model/mistral/`
**Goal**: Support Mistral 7B v0.3, Mixtral 8x7B, Mixtral 8x22B, Mistral Large 2.

- Sliding window attention (Mistral 7B)
- MoE (Mixtral) — top-2 routing, 8 experts
- No bias in linear layers

---

### 1.7 Gemma 2 / 3 Model
**Files**: `infer/src/model/gemma/`
**Goal**: Support Google Gemma 2 (2B, 9B, 27B) and Gemma 3.

Gemma 2 differences:
- Alternating sliding-window + global attention
- Logit soft-capping (tanh gate on logits)
- Pre + post norm (two RMSNorm per layer)
- Group query attention

Gemma 3 additions:
- Much longer context (128K default)
- Interleaved local/global attention with ratio 5:1

---

### 1.8 Phi-4 Model
**Files**: `infer/src/model/phi/`
**Goal**: Support Microsoft Phi-4 (14B).

- Dense transformer, standard GQA
- Custom tokenizer
- Flash attention compatible

---

## Phase 2 — Quantization (GPU required for kernels, CPU for config)

### 2.1 Weight Loader Abstraction
**Files**: `infer/src/weight_loader.rs`
**Goal**: Extend weight loader to detect and dispatch quantized formats.

```rust
pub enum QuantFormat { None, GPTQ, AWQ, FP8, INT8 }

impl WeightLoader {
    pub fn detect_quant_format(model_path: &Path) -> QuantFormat;
    pub fn load_quantized(path: &Path, fmt: QuantFormat) -> Result<QuantizedWeights>;
}
```

**CPU-verifiable**: config detection, metadata parsing, no kernel execution.

---

### 2.2 GPTQ / AWQ INT4 Kernels ✅ Shipped
**Files**: `crates/infer-cuda-kernels/csrc/gemm/{quantized_gemv,marlin_kernel,marlin_repack}.cu`
**Goal**: Fused dequantize-GEMM kernels for INT4 weights. Required for serving quantized open-source models (most popular community models are GPTQ/AWQ quantized).

Shipped:
- **W4A16 GEMV** for decode — `quantized_gemv.cu` handles the single-token dequant-fused path.
- **Marlin W4 prefill** — `marlin_kernel.cu` + `marlin_repack.cu` for A100+ tile-level dequant fused with GEMM; 5–25× TTFT speedup on long prompts.

---

### 2.3 FP8 (E4M3) KV Cache + Attention ✅ Shipped
**Files**: `crates/infer-cuda-kernels/csrc/attention/decode_attention_quantized.cu`, `crates/infer-cuda-kernels/csrc/kv/kv_quant.cu`, `crates/infer-cuda-kernels/src/kv_quant.rs`
**Goal**: FP8 KV cache for long-context agent workloads.

Shipped as an **FP8 KV path**, not an FP8 GEMM path — the latency win for agent workloads comes from halving KV bandwidth, not from FP8 matmul on the weight side.
- FP8 E4M3 KV storage via `migrate_from_contiguous_fp8` and `quantize_scatter_kv_fp8`.
- Custom fused-dequant decode attention with FP32 cast in `decode_attention_quantized.cu`.
- 50 % KV memory reduction vs BF16.

Native FP8 **weight** GEMM (DeepSeek-V3 style) remains deferred — not required until a model that ships FP8 weights is wired.

---

### 2.4 INT8 / W8A16 ✅ Shipped
**Files**: `crates/infer-cuda-kernels/csrc/gemm/quantized_gemv.cu`, `crates/infer-cuda-kernels/csrc/kv/kv_quant.cu`
**Goal**: INT8 weight + INT8 KV cache paths for memory savings.

Shipped:
- **W8A16 GEMV / GEMM** via `quantized_gemv.cu` for INT8-weight decode.
- **INT8 KV** via `kv_quant.cu` with symmetric per-head per-token scaling + fused-dequant decode attention in `decode_attention_quantized.cu`.
SmoothQuant activation migration is not applied today — W8A16 is sufficient for the coverage goal.

---

### 2.5 GGUF Loading ✅
**Files**: `infer/src/gguf.rs`, `infer/src/weight_loader.rs`
**Goal**: Load GGUF format files (llama.cpp ecosystem). Read GGUF header, tensor layout, Q4_K_M / Q8_0 block formats.

Shipped:
- Minimal GGUF parser with tensor directory + per-tensor readers
- BF16 / F16 / Q8_0 fast paths in `load_tensor_2d_gguf`
- **Q4_K_M native GPU path** — `q4k_gemv_kernel` in
  `crates/infer-cuda-kernels/csrc/gemm/quantized_gemv.cu` keeps superblocks
  packed on GPU (no BF16 intermediate), with `DeviceMatrix::from_quantized_q4k`
  consuming verbatim 144-byte Q4_K superblock bytes. Carnice-27B (64L × 5120d)
  fits on L4-24GB. See [docs/plans/q4k-native-gpu.md](docs/plans/q4k-native-gpu.md).
- V-head byte-level row permutation for linear-attention weights
- 3 test suites: `q4k_kernel_correctness.rs`, `ground_truth_q4k.rs`, `smoke_carnice_27b_q4k.rs`

---

## Phase 3 — Tensor Parallelism (GPU required, multi-GPU)

### 3.1 TP Communication Primitives
**Files**: `infer/src/ops/comm.rs` (new)
**Goal**: All-reduce and all-gather over NCCL for tensor parallel.

```rust
pub struct NcclComm { ... }
impl NcclComm {
    pub fn all_reduce_sum(tensor: &mut DeviceVec) -> Result<()>;
    pub fn all_gather(tensor: &DeviceVec, world_size: usize) -> Result<DeviceVec>;
}
```

---

### 3.2 Column / Row Parallel Linear
**Files**: `infer/src/ops/linear.rs`
**Goal**: Sharded linear layers for tensor parallel.

```
ColumnParallelLinear:  Y_i = X · W_i^T / TP_SIZE  (all-reduce after)
RowParallelLinear:     Y = X_i · W_i^T             (scatter input, gather output)
```

---

### 3.3 TP-aware Model Wrappers
**Files**: `infer/src/model/<name>/tp.rs` per model
**Goal**: Wrap each model's attention (split Q/K/V heads) and MLP (split intermediate dim) across TP ranks.

---

### 3.4 Pipeline Parallel (PP)
**Files**: `infer/src/model/pp.rs` (new)
**Goal**: Partition layers across devices. Send hidden states between stages via P2P (cuMemcpyPeer / NCCL send/recv).

---

### 3.5 Expert Parallel (EP) for MoE
**Files**: `infer/src/ops/moe.rs` (new)
**Goal**: Distribute MoE experts across EP ranks. All-to-all routing for expert dispatch/combine.

---

## Phase 4 — Advanced Decoding

### 4.1 Beam Search
**Files**: `infer/src/sampler.rs`
**Goal**: Beam search with configurable beam width. Uses block manager to share KV prefixes across beams.

---

### 4.2 Speculative Decoding
**Files**: `infer/src/speculative.rs` (new)
**Goal**: Draft-verify loop: small draft model generates K candidate tokens, large target model verifies in one forward pass.

```
DraftEngine (small model, e.g. Qwen3-0.5B)
TargetEngine (large model, e.g. Qwen3-32B)
SpeculativeScheduler: draft K tokens, batch-verify, accept/reject
```

For DeepSeek-V3: use MTP heads (built into model) as draft — zero additional model cost.

---

### 4.3 Structured Output (JSON schema / regex)
**Files**: `infer/src/structured_output.rs` (new)
**Goal**: Constrained decoding via logit masking. At each step, mask invalid token IDs per grammar state.

- JSON schema → LALR(1) grammar → token mask table
- Regex → NFA → per-step allowed token set
- Outlines / llguidance integration as optional backend

---

## Phase 5 — Performance Optimization

### 5.1 CUDA Graph for Multi-Request Decode ✅
**Files**: `infer/src/model/cuda_graph.rs`
**Goal**: Extend existing single-request CUDA graph to capture multi-request batched decode. Maintains a pool of graphs per batch size (1–32), captured on first call, replayed thereafter.

---

### 5.2 Overlap Compute and Communication (TP)
**Files**: `infer/src/ops/comm.rs`
**Goal**: Async NCCL all-reduce overlapped with next layer's compute using two CUDA streams.

---

### 5.3 Torch.compile / `torch.export` Integration
**Goal**: For Python-based model paths (if added), use `torch.compile` for additional kernel fusion.

---

### 5.4 Prefill/Decode Disaggregation
**Goal**: Split prefill (compute-bound) and decode (memory-bandwidth-bound) onto separate GPU instances (à la DistServe / Mooncake). Prefill machines send KV cache to decode machines via RDMA.

---

## Dependency Graph

```
Phase 0 (CPU) ✅ COMPLETE
  0.1 Chat API          ✅
  0.2 Sampler           ✅
  0.3 Radix Cache       ✅
  0.4 Block Manager     ✅
  0.5 Scheduler++       ✅ (preemption: recompute mode done, swap mode deferred)
  0.6 Model Registry    ✅
  0.7 Benchmark Suite   ✅

Phase 1 (GPU) — ACTIVE
  1.2 FA3               → depends on existing FA2/FlashInfer
  1.3 MLA               → needed by 1.5
  1.4 Llama             → depends on 0.6 ✅ (registry)
  1.5 DeepSeek          → depends on 0.6 ✅, 1.3
  1.6 Mistral           → depends on 0.6 ✅
  1.7 Gemma             → depends on 0.6 ✅
  1.8 Phi               → depends on 0.6 ✅

Phase 2 (Quantization) ✅ COMPLETE
  2.1 Loader            ✅
  2.2 GPTQ/AWQ          ✅ (W4A16 GEMV + Marlin W4 prefill)
  2.3 FP8               ✅ (FP8 KV cache + custom fused decode)
  2.4 INT8              ✅ (W8A16 GEMV/GEMM + INT8 KV fused-dequant)
  2.5 GGUF              ✅ (BF16/F16/Q8_0 + native Q4_K packed GPU kernel)

Phase 3 (TP/PP/EP)
  3.1 NCCL primitives   → Phase 1 complete
  3.2 Parallel linear   → depends on 3.1
  3.3 TP model wrappers → depends on 3.2
  3.4 PP                → depends on 3.3
  3.5 EP (MoE)          → depends on 3.3, 1.5/1.6 (MoE models)

Phase 4 (Decoding)
  4.1 Beam search       → depends on 0.4 ✅ (block sharing)
  4.2 Speculative       → depends on 1.x (two models loaded)
  4.3 Structured output → depends on 0.2 ✅ (sampler logit hooks)

Phase 5 (Optimization)
  5.1 CUDA graph batch  ✅ (per batch size, 1–32)
  5.2 Compute/comm overlap → depends on 3.1
  5.4 Disaggregation    → depends on all Phase 3
```

---

## Immediate Next Steps

Phase 0 complete. Quantization (Phase 2) complete including GGUF/Q4_K.
Tiered KV Cache M2b+M0.3+M3a+M3b+M3c locally shipped and L4 remote-accepted.
Focus on performance, robustness, and Metal parity:

1. ~~**Qwen3.5 SGLang parity**~~ — ✅ prefix cache fixed, ITL/TTFT ahead at C≤16
2. ~~**2.1–2.5 Quantization**~~ — ✅ GPTQ/AWQ/FP8/INT8/TurboQuant/GGUF+Q4_K all production-ready
3. ~~**Scheduler preemption with KV swap**~~ — ✅ recompute mode done (swap mode deferred)
4. ~~**Overlap scheduling (H2D/D2H with compute)**~~ — ✅ dual-stream + decode-first reordering
5. ~~**Tiered KV Cache M0–M3**~~ — ✅ M2b+M0.3+M3a+M3b+M3c locally + L4 remote-accepted.
   Next: M3b runtime promotion path, M4 disk persistence + session save/load.
6. **Metal M0.2 throughput exit** — `metal_serve` routes through `SchedulerHandle` (M0.2b shipped
   via `spawn_metal_scheduler_handle_from_path_with_options_and_metrics`); M0.2a/c/d also landed
   locally. Throughput exit still gated on variable-length decode + per-step batch-state rebuild
   cost. Observability exit (M0.3/M0.4) tracks separately. See
   [docs/plans/2026-04-15-metal-backend-acceptance-plan.md](docs/plans/2026-04-15-metal-backend-acceptance-plan.md).
7. **Qwen3.5 batched prefill** — prefill multiple requests in one forward pass (CUDA)
8. **4.2 Speculative Decoding GPU integration** — `speculative.rs` CPU framework ✅ done;
   need: DraftEngine, KV rollback in PagedKvPool, SpeculativeScheduler, CUDA Graph 2-phase.
   Research: standard draft model (Qwen3-0.5B) first; EAGLE2 / DFlash-MLX as phase 2.
   Metal DFlash experimental path on Qwen3 is shipped; Qwen3.5 DFlash now correct but a perf regression (see Recent Milestones).
   See [docs/research/speculative-decoding-feasibility.md](docs/research/speculative-decoding-feasibility.md)
9. **1.4 Llama 3/4 Model** — most requested architecture (deferred)
10. **1.5 DeepSeek-V3 / R1** — requires MLA (1.3) first
11. **1.2 FlashAttention-3** — H100 utilization improvement
12. **1.7 Gemma 4 Model** — detection in `model_registry.rs` shipped, per-model forward still
    unwired. See [docs/plans/gemma-gguf-support.md](docs/plans/gemma-gguf-support.md).

### Research Notes (2026-04-14)

- **Speculative decoding**: `speculative.rs` framework production-ready (CPU). EAGLE2/EAGLE3
  offer 2.5–3× speedup on Qwen3 without a separate model (uses target hidden states as draft input).
  See [docs/research/speculative-decoding-feasibility.md](docs/research/speculative-decoding-feasibility.md)

- **KV quantization on Metal**: FP8 KV not feasible (MLX has no FP8 dtype; M3 hardware lacks FP8 cores).
  TurboQuant/PolarQuant INT4 KV is feasible (4.6x compression, 0.995 cosine sim on Qwen3-4B) via
  community MLX implementations. Medium priority — worthwhile at C≥4 long-context.
  See [docs/research/kv-quantization-metal.md](docs/research/kv-quantization-metal.md)

- **DFlash on Metal**: DFlash-MLX (github.com/Aryagm/dflash-mlx) supports Qwen3-4B and Qwen3.5-4B.
  Per-layer KV rollback for hybrid models already solved. mlx-lm built-in `--draft-model` (merged Jan 2025)
  is the easiest 1.5–1.8x speedup path. EAGLE-3 is CUDA-only. Medium priority.
  See [docs/research/dflash-metal-feasibility.md](docs/research/dflash-metal-feasibility.md)

---

## Local Verifiability Checklist

| Task | Verifiable without GPU? | Status |
|------|------------------------|--------|
| 0.1 Chat API | ✅ Mock scheduler | ✅ Done |
| 0.2 Sampler | ✅ Unit tests on f32 arrays | ✅ Done |
| 0.3 Radix Cache | ✅ Pure data structure | ✅ Done |
| 0.4 Block Manager | ✅ Accounting only | ✅ Done |
| 0.5 Scheduler++ | ✅ Mock ModelForward | ✅ Done |
| 0.6 Model Registry | ✅ Config parsing | ✅ Done |
| 0.7 Benchmarks | ✅ Mock HTTP server | ✅ Done |
| 1.x–5.x | ❌ GPU required | 5.1 ✅ Done |
| 6.0 M0 autograd skeleton | ✅ CPU-only tape + add/mul/sum + grad_check | ✅ Done (`ef95994`) |
| 6.1 M1 core op suite | ⚠️ CPU for grad-check oracle; CUDA for cuBLAS | ✅ Done CPU (`318cbf4`); CUDA cuBLAS path pending remote verify |
| 6.2 M2 LoRA graft | ❌ needs Qwen weights + CUDA | ✅ Done — M2a TinyLM CPU (`f183f80`); M2b Qwen3 hot-path LoRA + codex rounds 1–3 (`1e87b4f`…`b5377f6`) |
| 6.3 M3 GRPO closed loop | ❌ needs CUDA, ~1 h | ✅ Done — TinyLM GRPO (`f86a9c7`) + PPO clip (`19b8e02`); 6 h Qwen+CUDA acceptance run remote-pending |
| 6.4 M4 agent self-evolution | ❌ needs CUDA, ~24 h | ✅ Done — M4.1–M4.8 infra shipped (multi-turn, stepwise return, verifiers, curriculum, task gen, /v1/train); 24 h hard-set pass@1 remote-pending |
| 6.5 M5 Metal parity | ⚠️ Mac-local dev path | ✅ Done (`a46fc00`) — Backend trait + Metal matmul; 3-shape parity ≤ 1e-3 |

---

## Phase 6 — Agent RL 训推一体 · 自进化栈（NEW CORE MAIN LINE, 2026-04-18）

**Scope lock** (v3, 2026-04-18): 单机 / CUDA first / LoRA-only / GRPO / 同进程训推 / 训练端从零写。
Project spine: [docs/projects/agent-rl-self-evolving.md](docs/projects/agent-rl-self-evolving.md).
Execution plan: [docs/plans/rust-agent-rl-single-node.md](docs/plans/rust-agent-rl-single-node.md).
Reference analysis: [docs/research/mni-ml-framework-notes.md](docs/research/mni-ml-framework-notes.md).

**Motivation**: agent-infer has been a pure inference engine. Continuing down that road makes us a commodity rollout backend behind Python+Megatron training stacks. Phase 6 is the deliberate move to **训推一体** — a single-node, same-process Rust RL trainer + inference engine that self-evolves via agent tool use and multi-verifier rewards. Rust's structural advantage here (no GIL, `Arc`-shared weights, no NCCL IPC tax) is only realized if we stay single-process; this phase commits to that shape.

**Non-goals (explicit)**: distributed training (DDP/ZeRO/TP/PP), full-parameter training, PPO with critic, Python dependency, **multi-backend abstraction layer on the *inference* side** (the `infer/` backend split stays `#[cfg]`-only per CLAUDE.md §Backend isolation), general-purpose autograd. Each of these is rejected in favor of the LoRA+GRPO+single-process minimum that produces a working self-evolving agent.

The 2026-04-18 `Backend` trait in `crates/autograd/` (M5) is NOT a violation: it is a minimal, per-call upload/compute/download matmul hook narrowly scoped to the autograd crate's training matmul forward, not a device-resident tensor abstraction for inference. See [`docs/plans/rust-agent-rl-single-node.md`](docs/plans/rust-agent-rl-single-node.md) §7 for the M5 scope lock.

### 6.0 Autograd Crate — from scratch (`crates/autograd/`)

**Files**: new workspace crate.
**Reference**: structural (not copied) from [mni-ml/framework](https://github.com/mni-ml/framework) — see [docs/research/mni-ml-framework-notes.md](docs/research/mni-ml-framework-notes.md) for the LOC-level map, borrowed abstractions (TensorId+Store, SavedContext enum, BackwardOp dispatch), and the pruning list.

Core design (locked):
- `TensorId = usize` + `TensorStore: Vec<Option<GpuTensor>>` + free-list slot recycle — no `Arc<Tensor>`, no `Rc<RefCell>`
- `SavedContext` as enum (variants per op), `BackwardOp` as enum, `dispatch_backward` as big match — no `Box<dyn Any>`, no trait objects
- `Tape::backward()`: DFS relevant set + post-order topo + `HashMap<TensorId, TensorId>` grad accumulation
- Optimizer state (Adam m/v) embedded in `GpuTensor` itself — no separate optimizer state map
- Double path `#[cfg(feature = "cuda")]` / `#[cfg(any(feature = "cpu", feature = "webgpu"))]` in every op file; CPU path is the grad-check oracle

**Op set (locked to LoRA+GRPO minimum)**: `add`, `mul`, `mul_scalar`, `matmul` (cuBLAS SGEMM), `sum`, `mean`, `log_softmax`, `gather`, plus `AdamW` step. ~7 ops + optimizer. Conv / pool / full-attention-bwd / embedding-bwd / dropout are all out of scope.

**M0 (3–5 d)**: Tape + add/mul/mul_scalar/sum on CPU. `y = ((a+b)*3).sum(); y.backward()` passes hand-derived assertion.
**M1 (10–14 d)**: matmul (CPU ref + CUDA cuBLAS), log_softmax, gather, mean, Module/Parameter trait, AdamW (CPU + CUDA fused). Grad-check every op (CPU f64 oracle vs CUDA f32 ≤ 1e-3). Toy 2-layer MLP converges on both paths.

---

### 6.1 Train Crate — LoRA + GRPO + rollout loop (`crates/train/`)

**Files**: new workspace crate. Depends on `autograd` + `infer`.

**M2 — LoRA graft (14 d)**:
- `LoRAAdapter { A: Parameter, B: Parameter, rank, alpha }`, forward `B @ (A @ x) * scale`
- Hook into `infer/src/ops/linear.rs`: optional `lora: Option<&LoRAAdapter>` arg; base path unchanged (zero overhead when `None`), LoRA path runs independent cuBLAS small GEMMs and sums
- `Arc<BaseWeights>` frozen-view tensors in autograd (no clone, no tape participation) — hard memory ceiling: Qwen3 4B + LoRA rank=8 ≤ 1.05× base memory
- Double-buffer `Arc<RwLock<LoRAAdapters>>` hot-swap; atomic pointer swap on finish-of-step
- Supervised fine-tune on synthetic (prompt, target) pairs proves the plumbing before RL

**M3 — GRPO closed loop (21 d)**:
- Trajectory struct: `{ prompt_ids, response_ids, step_logprobs, reward }`
- Scheduler emits trajectories on a `tokio::mpsc` channel on request-complete (< 1% inference throughput regression)
- Group advantage: G samples per prompt → `A_i = (r_i - mean) / std`
- GRPO loss with ratio clip + KL to reference policy (ref = LoRA-off base, ref logp frozen at rollout time)
- Alternating rollout↔train main loop (N=16 prompts × G=4 → GRPO K=4 steps → hot-swap → next)
- First verifier: GSM8K-like math exact-match
- Acceptance: 6h stable, held-out pass@1 ≥ base + 15%, not reward-hacking (human spot-check 20 samples → ≥15 coherent)

**M4 — Agent self-evolution MVP (4–6 w)**:
- Multi-turn agent trajectory via `infer-agent` tool loop; step-wise logp on action tokens
- Stepwise reward assignment (γ discount + tool-failure penalty)
- Multi-verifier: math + code (Rust-native pytest-like) + tool-use success rate
- Basic curriculum: difficulty buckets; retire easy tasks when pass@1 > 0.8; introduce harder ones
- Templated self-play task generator (verifier-grounded only — no ungrounded tasks accepted)
- Acceptance: 24h run, base-unsolvable hard-set pass@1 from P0 → P24 ≥ 1.3·P0 and absolute ≥ 0.3

**M5 — Metal parity (parallel to M3/M4)**:
- Same `BackwardOp` tape, Metal op implementations via `mlx-sys` forward + hand-written bwd formulas
- Mac M4 Pro runs Qwen 1.5B LoRA supervised fine-tune, loss shape matches CUDA
- Three-way grad-check (CPU f64 / CUDA f32 / Metal f32) mutual parity ≤ 1e-3

---

### 6.2 Scheduler & Model Integration Points

| Integration | Module | Change |
|---|---|---|
| Trajectory emit | `infer/src/scheduler/` | Add mpsc channel; emit on `RequestOutcome::Finished` |
| LoRA hook | `infer/src/ops/linear.rs` | `Option<&LoRAAdapter>` argument; default `None` is zero-cost |
| Base weights share | `infer/src/backend/cuda/` | Expose `Arc<CudaDevice>` + frozen view tensors |
| Agent callback | `infer-agent/` | Trajectory callback on tool loop steps |
| Train control plane | `infer/src/http_server/train.rs` (new, M4) | `/v1/train/{start,stop,status,checkpoint}` |

---

### 6.3 Acceptance — every milestone must clear

1. `cargo test --workspace --release` — zero regression on existing tests
2. `cargo clippy --workspace -- -D warnings` — clean
3. Grad-check for every new op: CPU f64 oracle vs CUDA f32 rel-err ≤ 1e-3
4. No half-states (`feedback_no_half_states.md`): crate ships whole or not at all
5. Experience entry in `docs/experience/{wins,errors}/YYYY-MM-DD-agent-rl-m<N>-*.md`
6. Progress log row appended in [plan doc §11](docs/plans/rust-agent-rl-single-node.md#11-progress-log)

---

### 6.4 Dependency Graph

```
Phase 6 (Agent RL) — NEW MAIN LINE
  6.0 M0 autograd skeleton          → depends on nothing (CPU only)
  6.1 M1 core op suite              → depends on 6.0
  6.2 M2 LoRA graft                 → depends on 6.1 + infer linear path (already exists)
  6.3 M3 GRPO closed loop           → depends on 6.2 + scheduler trajectory emit
  6.4 M4 agent self-evolution MVP   → depends on 6.3 + infer-agent (already exists)
  6.5 M5 Metal parity               → depends on 6.1 + mlx-sys bridge (already exists);
                                      can run parallel to 6.3/6.4
```

Phase 6 does **not** depend on Phase 1/3/4/5 advances — it's orthogonal to "more inference features". Phase 2 (quantization) already shipped so LoRA on Q4/Q8 base is possible post-M2 if needed.

---
