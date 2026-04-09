# agent-infer Architecture

> Generated 2026-03-31. Read-only analysis of the full Rust + CUDA codebase.

---

## A. High-Level Architecture — HTTP Request to GPU Execution

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           agent-infer REPL (src/)                            │
│   stdin → agent::run_agent() → ServerEngine::complete_stream()               │
│           tools.rs (builtin tools)  chat.rs (ChatML)                         │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │ ServerEngine trait (single-request)
                                 │
┌──────────────────────────────────────────────────────────────────────────────┐
│                        infer HTTP Server (infer/src/main.rs)                 │
│                                                                              │
│   HTTP POST /v1/completions                                                  │
│   HTTP POST /v1/chat/completions ──► axum handler (http_server.rs)          │
│   GET  /metrics (Prometheus)         │                                       │
│   GET  /v1/stats                     │                                       │
│                                      ▼                                       │
│                           AppState { SchedulerHandle, ServerMetrics }        │
│                                      │                                       │
│                           SchedulerHandle::submit(IncomingRequest)           │
│                                      │  mpsc::unbounded_channel             │
└──────────────────────────────────────┼───────────────────────────────────────┘
                                       │
┌──────────────────────────────────────┼───────────────────────────────────────┐
│                    Scheduler<M: ModelForward> (scheduler.rs)                 │
│                           (dedicated blocking thread)                        │
│                                      │                                       │
│   ┌──────────────────────────────────▼──────────────────────────────────┐   │
│   │                        Scheduler::run() loop                        │   │
│   │                                                                     │   │
│   │  1. drain request_rx → waiting: VecDeque<IncomingRequest>          │   │
│   │  2. assign_slots() → active: Vec<ActiveRequest>                    │   │
│   │  3. step():                                                         │   │
│   │     ├─ step_decode_batch()  ← ALL Decoding requests in one pass    │   │
│   │     └─ step_prefill_chunk() or step_new()  ← ONE prefill at a time │   │
│   │  4. cleanup() → send StreamDelta to HTTP handler via mpsc          │   │
│   └──────────────────────────────────┬──────────────────────────────────┘   │
│                                      │                                       │
│   Shared TokenKVPool (paged_kv.rs) — one pool for ALL slots                 │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │  max_total_tokens slots pre-allocated; page_size = 1 (token-level)  │  │
│   │  LIFO free list; per-slot logical→physical index mapping            │  │
│   │  Provides FlashInfer-compatible indptr/indices/last_page_len        │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│   State Pool [N slots] (N = num_slots, default 4)                           │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                      │
│   │ Slot 0   │ │ Slot 1   │ │ Slot 2   │ │ Slot 3   │                      │
│   │ M::State │ │ M::State │ │ M::State │ │ M::State │  ← decode buffers,   │
│   │ (no per- │ │          │ │          │ │          │    CUDA graphs        │
│   │  slot KV)│ │          │ │          │ │          │    (KV in pool)       │
│   └──────────┘ └──────────┘ └──────────┘ └──────────┘                      │
│                                                                               │
│   CUDA Graph Pool — one captured graph per batch size                        │
│   (CudaGraphPool in cuda_graph_pool.rs — active, not stub)                  │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │ ModelForward::forward(tokens, state)
                                       │ BatchDecode path uses FlashInfer
                                       │
┌──────────────────────────────────────┼───────────────────────────────────────┐
│                          Model Layer (model/)                                │
│                                      │                                       │
│         tokens.len() > 1 → Prefill path (batched GEMM + FlashAttention-2)  │
│                              scatter K/V → TokenKVPool via scatter_kv.cu    │
│         batch decode  → GEMM + FlashInfer paged batch decode                │
│         single decode → GEMV + Triton split-KV (single-request path)        │
│                                      │                                       │
│   ┌──────────────────┐   ┌───────────────────────────────────────────────┐  │
│   │   Qwen3Model     │   │             Qwen35Model                        │  │
│   │ 32 full-attn     │   │ 32 layers total:                               │  │
│   │ layers (GQA)     │   │  ├─ 24 linear-attn (gated delta rule)          │  │
│   └──────────────────┘   │  └─  8 full-attn (GQA)                        │  │
│                           │ + RecurrentState (persists across decode steps)│  │
│                           └───────────────────────────────────────────────┘  │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │ ops::* calls
                                       │
┌──────────────────────────────────────┼───────────────────────────────────────┐
│                           ops/ GPU Operations                                │
│                                      │                                       │
│  attention: prefill_attention_batch (FlashAttn-2), FlashInfer batch decode  │
│             fused_attention_decode_into (Triton split-KV, single-req)       │
│  kv_ops:    scatter_write_kv (prefill pool scatter)                         │
│  linear:    gemv (decode), gemm (prefill), fused_mlp_into                   │
│  norm:      rms_norm, fused_add_rms_norm, rms_norm_gated                    │
│  embedding: embedding_batch (prefill), embedding_decode_into                 │
│  recurrent: gated_delta_rule_prefill_chunkwise, conv1d_prefill               │
│  sampling:  argmax, gpu_sample (top-k/p/temperature)                         │
│  element:   silu_mul_batch, add_batch                                        │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │ ffi.rs unsafe extern "C"
                                       │
┌──────────────────────────────────────┼───────────────────────────────────────┐
│                     CUDA / Triton Kernels                                    │
│                                      │                                       │
│  csrc/ (CUDA C, compiled by build.rs via cc crate + nvcc)                   │
│  ├─ common.cuh              shared utilities                                 │
│  ├─ fused_attention.cu      decode attention (split-KV, single-request)     │
│  ├─ prefill_attention.cu / prefill_attention_hd256.cu                        │
│  ├─ decode_prep_paged.cu    per-token QK-norm + RoPE → paged KV write       │
│  ├─ flashinfer_decode.cu    FlashInfer batch decode kernel bindings          │
│  ├─ scatter_kv.cu           scatter prefill K/V into TokenKVPool             │
│  ├─ kv_cache_to_paged.cu    migrate contiguous KV cache → token pool        │
│  ├─ paged_kv_append.cu      append single decode token K/V to pool          │
│  ├─ fused_mlp.cu            SwiGLU MLP fusion                               │
│  ├─ gemv.cu                 GEMV for decode linear projections               │
│  ├─ norm.cu                 RMSNorm + fused variants                         │
│  ├─ pos_enc.cu              RoPE positional encoding                         │
│  ├─ conv1d.cu               1D convolution (Qwen3.5 recurrent)              │
│  ├─ gated_delta_rule.cu     gated delta rule (Qwen3.5 linear attn)          │
│  └─ sampling.cu             top-k / top-p GPU sampling                       │
│                                                                              │
│  tools/triton/ (Triton Python, AOT compiled to .cubin / .h)                 │
│  ├─ flash_attention_prefill_kernel.py   FlashAttention-2 prefill             │
│  ├─ flash_attention_prefill_hd256_kernel.py  hd=256 variant                 │
│  ├─ attention_decode_kernel.py          split-KV decode attention (single)   │
│  ├─ attention_reduce_kernel.py          decode attention reduction           │
│  ├─ silu_mul_kernel.py                  fused SiLU·Mul                      │
│  ├─ basic_kernels.py                    add, embedding variants              │
│  └─ gated_delta_rule_chunkwise_kernels.py  Qwen3.5 recurrent prefill        │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## B. Module Dependency Graph

```
agent-infer (src/)
├── agent.rs         ← uses ServerEngine trait from infer
├── chat.rs          ← uses infer::chat (ChatML formatting)
└── tools.rs         ← pure Rust, no infer dep

infer/src/
├── lib.rs           ← re-exports all public modules
│
├── [CUDA-gated — only compiled with feature = "cuda"]
│   ├── ffi.rs           ← unsafe extern "C" bindings to csrc/ + FlashInfer kernels
│   ├── tensor.rs        ← DeviceContext, DeviceVec, DeviceMatrix (cudarc)
│   ├── weight_loader.rs ← safetensors loading + mmap into GPU
│   ├── paged_kv.rs      ← TokenKVPool (page_size=1), FlashInferMeta, PagedKVPool alias
│   ├── ops.rs           ← re-exports ops submodules
│   │   ├── ops/attention.rs   ← FlashInferWorkspace, FlashInfer batch decode,
│   │   │                         prefill_attention_batch, fused_attention_decode_into
│   │   ├── ops/kv_ops.rs      ← scatter_write_kv (prefill → token pool)
│   │   ├── ops/linear.rs      ← gemv, gemm, fused_mlp
│   │   ├── ops/norm.rs        ← rms_norm variants
│   │   ├── ops/embedding.rs   ← token embedding ops
│   │   ├── ops/elementwise.rs ← silu_mul, add
│   │   ├── ops/recurrent.rs   ← gated delta rule, conv1d
│   │   └── ops/sampling.rs    ← gpu_sample, argmax
│   └── model/
│       ├── model.rs           ← ModelForward + GenerationState traits
│       ├── model/kv_cache.rs  ← KVCache (contiguous buffers + CPU offload)
│       │                         used by single-request (REPL) path
│       ├── model/cuda_graph.rs ← CudaGraphState (capture/replay, per-slot)
│       ├── model/qwen3.rs     ← pub mod re-export
│       │   model/qwen3/
│       │   ├── config.rs      ← JSON config parsing
│       │   ├── weights.rs     ← safetensors loading → Qwen3Model
│       │   ├── decode_buffers.rs ← GPU scratch buffers for single decode step
│       │   ├── prefill.rs     ← prefill forward pass (scatter KV → TokenKVPool)
│       │   ├── decode.rs      ← single-request decode (GEMV path, contiguous KV)
│       │   ├── batch_decode.rs ← batched decode (GEMM + FlashInfer paged KV)
│       │   │                     BatchDecodeBuffers, CUDA Graph per batch size
│       │   └── forward.rs     ← ModelForward impl + Qwen3State
│       └── model/qwen35.rs    ← pub mod re-export
│           model/qwen35/
│           ├── config.rs
│           ├── weights.rs     ← Qwen35Model
│           ├── decode_buffers.rs
│           ├── prefill_buffers.rs
│           ├── single_token_buffers.rs ← decode buffers for Qwen35
│           ├── recurrent_state.rs ← RecurrentState (linear attn state)
│           ├── prefill.rs
│           └── forward.rs     ← ModelForward impl + Qwen35State
│
├── [Always compiled — pure Rust, no GPU dependency]
│   ├── backend.rs         ← InferenceBackend trait (load/generate)
│   ├── metal_backend.rs   ← MetalBackend impl (InferenceBackend)
│   │                          depends on: backend, tokenizer, hf_hub, sampler
│   │                          [feature=metal]: mlx-rs
│   ├── sampler.rs         ← SamplingParams struct (pure Rust)
│   ├── tokenizer.rs       ← Tokenizer wrapper (tokenizers crate)
│   ├── chat.rs            ← ChatML message formatting
│   ├── hf_hub.rs          ← HuggingFace Hub auto-download
│   ├── model_registry.rs  ← ModelArch enum, config.json parsing
│   ├── quant.rs           ← Quantization format detection
│   ├── logging.rs         ← logforth setup
│   ├── metrics.rs         ← ServerMetrics (Prometheus counters)
│   ├── trace_reporter.rs  ← fastrace FileReporter
│   │
│   ├── block_manager.rs   ← BlockManager (paged KV accounting, CPU-only)
│   │                          no deps on GPU modules
│   ├── prefix_cache.rs    ← RadixCache (content-addressable tree, CPU-only)
│   │                          no deps on GPU modules
│   ├── cuda_graph_pool.rs ← GraphPool + batch size padding (CPU accounting)
│   │                          GPU stubs behind #[cfg(feature = "cuda")]
│   ├── speculative.rs     ← SpecConfig, verify_tokens (CPU); DraftModel stub
│   ├── tensor_parallel.rs ← TpConfig, sharding math (CPU); NCCL stubs
│   │
│   ├── server_engine.rs   ← ServerEngine trait, GenericServerEngine<M>
│   │                          depends on: model (cuda), sampler, tokenizer
│   ├── scheduler.rs       ← Scheduler<M>, SchedulerHandle, IncomingRequest
│   │                          manages TokenKVPool shared across all slots
│   │                          depends on: model (cuda), server_engine, tokenizer,
│   │                                      sampler, block_manager, paged_kv
│   └── http_server.rs     ← axum Router, SSE streaming
│       http_server/openai_v1.rs ← OpenAI API request/response types
│                          depends on: scheduler, server_engine, metrics, sampler
```

---

## C. Data Flow Diagram — Single Request Lifecycle

```
Client                 HTTP Server             Scheduler              GPU
  │                       │                      │                     │
  │── POST /v1/chat ──────►│                      │                     │
  │                       │                      │                     │
  │                       │ parse ChatCompletionRequest               │
  │                       │ sampling_params_from_request()            │
  │                       │                      │                     │
  │                       │ mpsc::unbounded_channel() → (tx, rx)     │
  │                       │ IncomingRequest{prompt,params,tx}        │
  │                       │──── handle.submit() ──►│                  │
  │                       │                      │                     │
  │ SSE stream open       │                      │ waiting.push_back()│
  │◄── 200 OK ────────────│                      │                     │
  │                       │                      │                     │
  │                       │         ┌────────────┘                     │
  │                       │         │ assign_slots(): find free slot   │
  │                       │         │ tokenizer.encode(prompt)         │
  │                       │         │ → ActiveRequest{slot=N,          │
  │                       │         │   phase=Phase::New}              │
  │                       │         │                                  │
  │                       │         │ step_new(idx):                   │
  │                       │         │  check prefix cache hit/miss    │
  │                       │         │  phase = Prefilling{tokens, 0}  │
  │                       │         │                                  │
  │                       │         │ step_prefill_chunk():            │
  │                       │         │  alloc tokens in TokenKVPool    │
  │                       │         │  chunk = tokens[0..512]         │
  │                       │         │  model.forward(chunk, state) ───►│
  │                       │         │  scatter K/V → pool indices ────►│◄─ scatter_kv.cu
  │                       │         │  (repeat until all tokens done) │◄─ FlashAttn-2
  │                       │         │  → phase = Decoding              │   (Triton)
  │                       │         │  sample first token              │
  │                       │         │                                  │
  │                       │         │ step_decode_batch():             │
  │                       │         │  build FlashInferMeta for batch │
  │                       │         │  alloc 1 new token per slot     │
  │                       │         │  model.forward_decode_batch()───►│
  │                       │         │  GEMM + FlashInfer paged decode ►│◄─ flashinfer_decode
  │                       │         │  CUDA Graph replay if captured  │     + decode_prep_paged
  │                       │         │  sample tokens                   │
  │                       │         │  emit_delta() → tokenizer.decode │
  │◄── SSE data ──────────│◄─ tx.send(StreamDelta{text_delta}) ──────┘
  │                       │         │
  │                       │         │ ... repeat decode steps ...
  │                       │         │
  │                       │         │ EOS or max_tokens reached:
  │                       │         │  finish() → send_finish(reason)
  │                       │         │  TokenKVPool::free_slot(slot)
  │◄── SSE [DONE] ────────│◄─ tx.send(StreamDelta{finish_reason})
```

**Prefix Cache Reuse** (within Scheduler or GenericServerEngine):
```
New request prompt tokens: [A B C D E F G H]
Cached prompt in slot:     [A B C D X Y]

Common prefix = [A B C D] (len=4)
→ Partial hit: state.truncate_to(4) (keep KV for first 4 tokens)
→ Only prefill [E F G H] (suffix after divergence)
```

---

## D. Dual Backend Architecture

```
                         InferenceBackend trait (backend.rs)
                         ├── load(model_path) → Result<()>
                         ├── generate(prompt, params) → GenerateResult
                         └── name() → &'static str
                                 │
              ┌──────────────────┴────────────────────┐
              │                                        │
    ┌─────────▼──────────┐              ┌─────────────▼─────────────┐
    │   CUDA Backend     │              │   Metal Backend            │
    │  (implicit, via    │              │  metal_backend.rs          │
    │  ModelForward +    │              │  [feature = "metal"]       │
    │  GenericServerEngine│             │                            │
    │  / Scheduler)      │              │  MetalBackend struct:      │
    │                    │              │  - model_dir: PathBuf      │
    │  Feature: "cuda"   │              │  - tokenizer: Tokenizer    │
    │  Deps: cudarc,     │              │  - config: MetalModelConfig│
    │         memmap2    │              │  - weights: MetalWeights   │
    │                    │              │    (mlx-rs Arrays in       │
    │  Supports:         │              │     unified memory)        │
    │  - Multi-request   │              │                            │
    │    Scheduler       │              │  Implements:               │
    │  - Continuous      │              │  - Full Qwen2.5 forward    │
    │    batching        │              │  - RMSNorm, RoPE, GQA attn │
    │  - FlashInfer      │              │  - SwiGLU MLP              │
    │    paged decode    │              │  - Append-based KV cache   │
    │  - CUDA Graphs     │              │  - Single-request only     │
    │    per batch size  │              └────────────────────────────┘
    │  - KV CPU offload  │
    │  - Qwen3 + Qwen35  │
    └────────────────────┘

Build flag selection:
  Linux/GPU: cargo build --release                     → CUDA backend
  macOS/CI:  cargo build --release --no-default-features --features no-cuda
  Apple M*:  cargo build --release --no-default-features --features metal,no-cuda
```

**Code Separation Pattern:**

```rust
// In lib.rs:
#[cfg(feature = "cuda")]          // ← CUDA-only modules
pub mod model;
pub mod ops;
pub mod tensor;
pub mod weight_loader;
pub mod paged_kv;

pub mod metal_backend;            // ← Always compiled
pub mod backend;                  // ← Trait definition, always compiled

// In metal_backend.rs:
#[cfg(feature = "metal")]
use mlx_rs::Array;                // ← mlx-rs only when feature active

pub struct MetalBackend {
    #[cfg(feature = "metal")]
    weights: Option<MetalWeights>, // ← Metal weights only if feature
    #[cfg(not(feature = "metal"))]
    _weights: (),                  // ← Zero-sized placeholder otherwise
}
```

---

## E. Scheduler State Machine

```
                    ┌─────────────────────┐
                    │     WAITING queue   │
                    │  VecDeque<Incoming> │
                    └──────────┬──────────┘
                               │ assign_slots() — free slot found
                               ▼
                    ┌─────────────────────┐
                    │      Phase::New     │ ◄── ActiveRequest created
                    │  (just assigned)    │
                    └──────────┬──────────┘
                               │ step_new() — compute prefix cache
                               │  - Full hit → slice suffix, skip to Prefilling
                               │  - Partial hit → truncate_to(), slice suffix
                               │  - Miss → state.reset(), free old pool tokens
                               ▼
                    ┌─────────────────────┐
                    │  Phase::Prefilling  │
                    │  { tokens, progress }│
                    └──────────┬──────────┘
                               │ step_prefill_chunk() per step:
                               │   alloc chunk tokens in TokenKVPool
                               │   chunk = tokens[progress..progress+512]
                               │   model.forward(chunk, state) → scatter K/V
                               │   progress += 512
                               │
                    ┌──────────┤ progress < total → stay in Prefilling
                    │          │ progress == total → sample first token
                    │          ▼
                    │ ┌─────────────────────┐
                    │ │   Phase::Decoding   │ ← emit first token delta
                    │ └──────────┬──────────┘
                    │            │ step_decode_batch() — EVERY step:
                    │            │   build FlashInferMeta for all Decoding slots
                    │            │   alloc 1 new token per slot in TokenKVPool
                    │            │   model.forward_decode_batch(tokens, states)
                    │            │     → GEMM + FlashInfer paged decode
                    │            │     → CUDA Graph replay if batch_size captured
                    │            │   sample tokens
                    │            │   emit_delta() → StreamDelta via mpsc
                    │            │
                    │ ┌──────────┤ EOS token || max_tokens reached
                    │ │          │ stop sequence found in decoded text
                    │ │          │ client disconnected (tx.is_closed())
                    │ │          ▼
                    │ │ ┌─────────────────────┐
                    │ └─►  Phase::Finished    │ → cleanup(): remove from active
                    │   └──────────┬──────────┘   TokenKVPool::free_slot(slot)
                    │              │ send_finish(FinishReason::Stop|Length)
                    │              ▼
                    │   ┌─────────────────────┐
                    └──►│  Slot returned to   │
                        │  pool (state reused │
                        │  with prefix cache) │
                        └─────────────────────┘

Scheduling priority:
  Each Scheduler::step() does BOTH:
  1. step_decode_batch()  — batch ALL active Decoding requests (priority)
  2. step_prefill_chunk() — exactly ONE prefill chunk (fairness)

  Without the prefill step, decode-priority would completely starve new
  requests. One chunk per step means new requests always make progress.

Preemption (PreemptionMode):
  Recompute (default): evict KV cache, re-prefill on resume (saves GPU mem)
  Swap: KV cache to CPU, swap back on resume (preserves state, costs DRAM)
```

---

## F. KV Cache Management Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      KV Cache System (Two Active Paths)                      │
│                                                                              │
│  Path A: TokenKVPool (paged_kv.rs) — Shared Pool for Batched Decode [ACTIVE]│
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Single shared pool for ALL request slots (Scheduler path)           │  │
│  │  page_size = 1: each token gets its own pool slot index              │  │
│  │                                                                       │  │
│  │  Pool layout per layer:                                              │  │
│  │    k_buffers[layer]: [max_total_tokens * kv_dim] bf16               │  │
│  │    v_buffers[layer]: [max_total_tokens * kv_dim] bf16               │  │
│  │    (NHD layout: offset = idx * kv_dim + h * head_dim + d)           │  │
│  │                                                                       │  │
│  │  Allocation: LIFO free list of token slot indices (Vec<u32>)         │  │
│  │    alloc_tokens(slot, count) → new physical indices                  │  │
│  │    free_slot(slot) → return all indices to free list                 │  │
│  │                                                                       │  │
│  │  FlashInfer metadata (built each decode step):                       │  │
│  │    indptr: [batch_size+1] cumulative token counts                    │  │
│  │    indices: concatenated physical pool indices for batch             │  │
│  │    last_page_len: all 1s (page_size=1)                              │  │
│  │                                                                       │  │
│  │  Pool sizing: budget_bytes / (num_layers * kv_dim * 2 bytes * 2)    │  │
│  │    e.g. Qwen3-4B (28 layers, 8 kv_heads, 128 head_dim):            │  │
│  │    1GB budget ≈ 21k tokens                                           │  │
│  │                                                                       │  │
│  │  Type alias: PagedKVPool = TokenKVPool (backward compat)            │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Path B: KVCache (model/kv_cache.rs) — Contiguous Buffers [Single-Request] │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Per-slot, per-layer contiguous K/V buffers (REPL/ServerEngine path) │  │
│  │  k_cache: Vec<DeviceVec>  [num_layers × (num_kv_heads × max_seq × head_dim)]│
│  │  v_cache: Vec<DeviceVec>                                              │  │
│  │                                                                       │  │
│  │  CPU Offload (spill):                                                 │  │
│  │  max_gpu_seq_len: usize  (default = max_seq_len = 32768)             │  │
│  │  When seq_len > max_gpu_seq_len:                                      │  │
│  │    oldest 64-token blocks → k_host/v_host: Vec<Vec<bf16>> (CPU RAM)  │  │
│  │    offloaded_len: usize   tracks tokens on CPU                       │  │
│  │                                                                       │  │
│  │  Before attention:  ensure_on_gpu() — prefetch CPU → GPU            │  │
│  │  After attention:   offload_to_host() — move prefix GPU → CPU       │  │
│  │                                                                       │  │
│  │  Operations:                                                          │  │
│  │    get_cache_mut(layer) → (&mut DeviceVec, &mut DeviceVec)          │  │
│  │    truncate_to(len)     — prefix reuse, no re-prefill needed         │  │
│  │    reset()              — full clear for new request                 │  │
│  │    offload_if_needed()  — after request ends, reclaim GPU memory     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Layer 2: BlockManager (block_manager.rs) — Paged Block Accounting          │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  CPU-only accounting — no actual CUDA memory copies                  │  │
│  │  Fixed-size blocks; requests hold a block table (logical → physical) │  │
│  │  BlockId(u32), BlockLocation: {Gpu, Cpu}, SwapPlan                   │  │
│  │  Status: CPU accounting complete; GPU swap kernels pending (stub)    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Layer 3: RadixCache (prefix_cache.rs) — Content-Addressable Prefix Tree    │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Maps token sequences → BlockId (cross-request prefix sharing)       │  │
│  │  Node: { tokens, block_id, ref_count, last_access, children }       │  │
│  │  Eviction: LRU leaves with ref_count == 0                            │  │
│  │  Status: CPU data structure complete; GPU-wiring to TokenKVPool pending│ │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Simple Prefix Cache (in Scheduler per-slot):                                │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  cached_prompt: Vec<u32>  — tokens of last request in this slot      │  │
│  │  Linear common-prefix comparison on each new request                  │  │
│  │  Full hit → reuse all KV; Partial hit → truncate_to(prefix_len)     │  │
│  │  This is the ACTIVE per-slot prefix cache                            │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## G. FlashInfer Batch Decode Pipeline

```
Every decode step with B active requests:

1. CPU: build FlashInferMeta from TokenKVPool
   indptr = [0, seq_len[0], seq_len[0]+seq_len[1], ...]
   indices = [pool_idx for all tokens of all requests]
   last_page_len = [1, 1, 1, ...]   (page_size = 1)

2. CPU→GPU: upload indptr, indices, last_page_len

3. FlashInfer plan (flashinfer_decode.cu):
   BatchDecodeWithPagedKVCachePlan(float_workspace, int_workspace,
     plan_info, indptr, last_page_len, B, num_qo_heads, num_kv_heads,
     head_dim, page_size=1, ...)

4. Decode forward pass (per layer):
   a. QKV projection: GEMM [B, hidden] → Q/K/V [B, q_dim/kv_dim]
   b. Prep kernel (decode_prep_paged.cu):
      - QK-norm (in-place on Q and K)
      - RoPE positional encoding
      - Write K/V into TokenKVPool at new token indices
   c. FlashInfer run (flashinfer_decode.cu):
      BatchDecodeWithPagedKVCacheRun(float_workspace, plan_info,
        q, k_pool_ptr, v_pool_ptr, output,
        indptr, indices, last_page_len, ...)

5. CUDA Graph capture (first time for batch_size=B):
   Steps 4a–4c captured as a single graph → replayed for all subsequent
   decode steps with the same batch size.

6. Logits: GEMM [B, hidden] → [B, vocab_size] → sample B tokens
```

---

## H. Module Responsibilities (One-Liner Each)

### infer/src/ — Core Inference Engine

| Module | Responsibility |
|--------|----------------|
| `lib.rs` | Module registration; feature-gates CUDA modules |
| `model.rs` | `ModelForward` + `GenerationState` traits — the core abstraction |
| `paged_kv.rs` | `TokenKVPool` (page_size=1 token pool), `FlashInferMeta`, `PagedKVPool` alias |
| `model/kv_cache.rs` | Contiguous K/V GPU buffers with CPU offload; used by single-request (REPL) path |
| `model/cuda_graph.rs` | CUDA Graph capture on first decode token, replay on subsequent (per-slot) |
| `model/qwen3/config.rs` | Parse Qwen3 `config.json` → typed `Qwen3Config` struct |
| `model/qwen3/weights.rs` | Load safetensors shards into GPU memory → `Qwen3Model` |
| `model/qwen3/decode_buffers.rs` | Pre-allocated GPU scratch buffers for single-request decode step |
| `model/qwen3/prefill.rs` | Prefill forward pass (batched GEMM + FlashAttention-2, scatter K/V → pool) |
| `model/qwen3/decode.rs` | Single-request decode (GEMV path, contiguous KV cache) |
| `model/qwen3/batch_decode.rs` | Batched decode: GEMM + FlashInfer paged KV; `BatchDecodeBuffers` pre-allocated; CUDA Graph per batch_size |
| `model/qwen3/forward.rs` | `ModelForward` impl for `Qwen3Model`; `Qwen3State` definition |
| `model/qwen35/recurrent_state.rs` | Per-request recurrent state for 24 linear-attention layers |
| `model/qwen35/single_token_buffers.rs` | GPU scratch buffers for Qwen3.5 single-token (decode) step |
| `model/qwen35/prefill_buffers.rs` | GPU scratch buffers for Qwen3.5 prefill step |
| `model/qwen35/forward.rs` | `ModelForward` impl for `Qwen35Model`; `Qwen35State` definition |
| `ops.rs` | Re-exports all GPU operation submodules |
| `ops/attention.rs` | `FlashInferWorkspace` (128MB+8MB GPU + pinned CPU bufs); FlashInfer batch decode plan/run; prefill_attention_batch (FlashAttn-2); fused_attention_decode_into (Triton split-KV, single-req) |
| `ops/kv_ops.rs` | `scatter_write_kv` — scatter prefill K/V from contiguous GEMM output to TokenKVPool |
| `ops/linear.rs` | GEMV (decode), GEMM (prefill), fused MLP |
| `ops/norm.rs` | RMSNorm variants including fused add+norm |
| `ops/embedding.rs` | Token embedding lookup (prefill + decode) |
| `ops/elementwise.rs` | SiLU·Mul fused, element-wise add |
| `ops/recurrent.rs` | Gated delta rule (Qwen3.5 linear attn), conv1d |
| `ops/sampling.rs` | GPU top-k/top-p sampling and argmax |
| `tensor.rs` | `DeviceContext` (CUDA context+stream), `DeviceVec`, `DeviceMatrix` |
| `ffi.rs` | Unsafe FFI: all CUDA C kernel entry points + FlashInfer plan/run + paged KV ops |
| `weight_loader.rs` | Memory-mapped safetensors → GPU tensor loading |
| `scheduler.rs` | Multi-request scheduler; manages shared `TokenKVPool`; decode-priority; chunked prefill |
| `server_engine.rs` | `ServerEngine` trait; `GenericServerEngine<M>` (single-request, for REPL/agent) |
| `backend_runtime.rs` | Serial request queue for `StreamingInferenceBackend`; used by Metal HTTP serving today |
| `http_server.rs` | Axum router: OpenAI-compatible `/v1/completions`, `/v1/chat/completions`, SSE |
| `http_server/openai_v1.rs` | OpenAI API request/response JSON types |
| `backend.rs` | `InferenceBackend` trait — backend-agnostic single-request interface |
| `metal_backend.rs` | Apple Silicon Metal backend via mlx-rs; implements `InferenceBackend` and `StreamingInferenceBackend` |
| `sampler.rs` | `SamplingParams` struct (temperature, top-k/p, min-p, penalties, EOS) |
| `tokenizer.rs` | Tokenizer wrapper (HuggingFace `tokenizers` crate) with incremental decoding |
| `chat.rs` | ChatML message formatting and tool definition injection |
| `block_manager.rs` | Paged KV block allocation accounting (CPU-only; GPU wiring pending) |
| `prefix_cache.rs` | Radix tree content-addressable prefix cache (CPU-only; GPU wiring to TokenKVPool pending) |
| `cuda_graph_pool.rs` | Batch-size padding arithmetic + CUDA Graph pool state tracking |
| `speculative.rs` | Speculative decoding framework: `SpecConfig`, `verify_tokens`, `DraftModel` stub |
| `tensor_parallel.rs` | Tensor parallel config + column/row sharding math; NCCL stubs |
| `model_registry.rs` | Architecture detection from `config.json` → `ModelArch` enum (9 architectures) |
| `quant.rs` | Quantization format detection (GPTQ/AWQ/FP8/INT8/GGUF) |
| `metrics.rs` | Prometheus metrics (`ServerMetrics`) |
| `trace_reporter.rs` | fastrace `FileReporter` for request tracing |
| `hf_hub.rs` | HuggingFace Hub auto-download of model weights + tokenizer |
| `logging.rs` | logforth logging setup |

### src/ — Agent Binary

| Module | Responsibility |
|--------|----------------|
| `main.rs` | CLI entry point: detect model type, load engine, run REPL |
| `agent.rs` | Agent loop: generate → parse tool calls → execute tools → feed results back |
| `chat.rs` | ChatML message builder for agent conversations |
| `tools.rs` | Builtin tool definitions (functions the agent can call) |

### CUDA Kernels (infer/csrc/)

| File | Responsibility |
|------|----------------|
| `common.cuh` | Shared CUDA utilities, warp-level primitives |
| `fused_attention.cu` | Decode attention kernel (split-KV, single-request) |
| `decode_prep_paged.cu` | Per-token QK-norm + RoPE + write K/V into TokenKVPool (paged, decode) |
| `flashinfer_decode.cu` | FlashInfer `BatchDecodeWithPagedKVCache` plan + run kernel bindings |
| `scatter_kv.cu` | Scatter prefill K/V from contiguous buffer into TokenKVPool by index |
| `kv_cache_to_paged.cu` | Migrate contiguous KV cache (old format) into TokenKVPool (transition helper) |
| `paged_kv_append.cu` | Append a single decode token's K/V to the paged pool |
| `prefill_attention.cu` | Prefill attention for general head dims |
| `prefill_attention_hd256.cu` | Optimized prefill attention for head_dim=256 |
| `fused_mlp.cu` | Fused gate+up projection → SiLU → down projection |
| `gemv.cu` | GEMV for decode linear projections |
| `norm.cu` | RMSNorm + fused add+norm variants |
| `pos_enc.cu` | RoPE positional encoding |
| `conv1d.cu` | 1D depthwise convolution for Qwen3.5 recurrent layers |
| `gated_delta_rule.cu` | Gated delta rule (Qwen3.5 linear attention mechanism) |
| `sampling.cu` | Top-k, top-p, temperature GPU sampling |

### Triton Kernels (infer/tools/triton/)

| File | Responsibility |
|------|----------------|
| `flash_attention_prefill_kernel.py` | FlashAttention-2 prefill (general head dim) |
| `flash_attention_prefill_hd256_kernel.py` | FlashAttention-2 prefill optimized for head_dim=256 |
| `attention_decode_kernel.py` | Split-KV decode attention (per-head parallelism, single-request) |
| `attention_reduce_kernel.py` | Reduction across split-KV decode results |
| `silu_mul_kernel.py` | Fused SiLU·Mul element-wise kernel |
| `basic_kernels.py` | Add, embedding lookup variants |
| `gated_delta_rule_chunkwise_kernels.py` | Qwen3.5 chunkwise gated delta rule for prefill |
| `gen_triton_aot.py` | AOT compilation driver: runs all Triton kernels through `triton.compile()` |

---

## Key Design Decisions

1. **No PyTorch** — direct cudarc (Rust CUDA bindings) + custom kernels. Eliminates Python GIL, framework overhead, and enables CUDA Graph capture.

2. **Single compute stream** — all GPU work on one CUDA stream. Enables CUDA Graph capture without cross-stream synchronization. `ctx.disable_event_tracking()` called at init.

3. **Weights `&self`, state `&mut State`** — `ModelForward` separates immutable weights from mutable per-request state. Enables N concurrent requests sharing one model copy.

4. **Decode before prefill** — scheduler prioritizes batch-decoding all active requests before any prefill chunk. Prevents decode latency spikes from prefill interference.

5. **Chunked prefill (4096 tokens, 64 when decode active)** — long prompts are prefilled in chunks, allowing decode steps to interleave. Prevents prefill from starving active decoders.

6. **CUDA Graph per batch size** — `BatchDecodeBuffers` captures a CUDA Graph on the first decode pass for each distinct batch_size. Subsequent steps with the same batch_size replay the graph, eliminating CPU-GPU kernel dispatch overhead. Pre-captured at startup for all expected batch sizes.

7. **Token-level KV pool (page_size=1)** — `TokenKVPool` gives every token its own slot index. No page tables, no last_page_len bookkeeping, no partial-page handling. FlashInfer's `page_size=1` API accepts this natively. Simpler than page-granularity PagedAttention.

8. **FlashInfer for batched decode** — FlashInfer's plan/run split allows the plan phase (metadata computation) to be separated from the run phase. Plan is called once per decode step; run is CUDA-Graph-captured per batch size. Achieves near-linear scaling with batch size.

9. **bf16 throughout** — all GPU tensors are `CudaSlice<bf16>`. No FP32 copies needed at inference time.

10. **CPU-complete stubs** — BlockManager, RadixCache, SpecConfig, TpConfig are fully implemented in pure Rust with no GPU dependency. GPU wiring is the only remaining step.

---

## What's Implemented (as of 2026-04-01)

| Component | Status |
|-----------|--------|
| Qwen3 model (GQA) | ✅ |
| Qwen3.5 model (hybrid recurrent+attn) | ✅ |
| Qwen3.5 scheduler + batched decode | ✅ |
| FlashInfer single prefill (HD128) | ✅ |
| FlashAttention-2 (Triton, HD256 prefill) | ✅ |
| FlashInfer batch decode (plan/run, paged KV) | ✅ |
| Merged QKV + gate-up GEMM | ✅ |
| scatter_kv + decode_prep_paged + kv_cache_to_paged kernels | ✅ |
| KV cache with CPU offload (single-request path) | ✅ |
| TokenKVPool — token-level paged KV (page_size=1) | ✅ |
| CUDA Graph per batch size (pre-captured 1–32 at startup) | ✅ |
| Continuous batching scheduler | ✅ |
| Chunked prefill (4096 tok, 64 when decode active) | ✅ |
| Decode-priority scheduling | ✅ |
| Prefix-aware slot assignment | ✅ |
| Request priority + backpressure | ✅ |
| top-k / top-p / temperature / min-p sampling | ✅ |
| Repetition / frequency / presence penalties | ✅ |
| Batched sampling (single sync) | ✅ |
| OpenAI `/v1/completions` API | ✅ |
| OpenAI `/v1/chat/completions` API | ✅ |
| SSE streaming | ✅ |
| Prometheus `/metrics` endpoint | ✅ |
| Stats `/v1/stats` endpoint | ✅ |
| Model architecture registry (9 architectures) | ✅ |
| Quantization format detection (GPTQ/AWQ/FP8/INT8/GGUF) | ✅ (detection only) |
| Radix tree prefix cache (data structure) | ✅ (CPU, not yet wired to TokenKVPool) |
| Paged KV block manager (accounting) | ✅ (CPU, not yet wired to GPU) |
| Speculative decoding framework | ✅ (CPU stubs, GPU pending) |
| Tensor parallel config + sharding math | ✅ (CPU, NCCL stubs) |
| Rust agent binary (tool calling) | ✅ |
| Python agent (async HTTP) | ✅ |
| Metal backend (Apple Silicon, mlx-rs) | ✅ (single-request + serial HTTP runtime) |
| Benchmark suite (throughput, agent, multi-request) | ✅ |
| Llama / DeepSeek / Mistral / Gemma / Phi models | ❌ |
| FlashAttention-3 | ❌ |
| MLA attention (DeepSeek) | ❌ |
| Beam search | ❌ |
| Quantization GPU kernels (GPTQ/AWQ/FP8/INT8) | ❌ |
| NCCL all-reduce / all-gather | ❌ |
| RadixCache GPU-wiring to TokenKVPool | ❌ |
