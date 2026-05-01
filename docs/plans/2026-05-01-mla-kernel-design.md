# MLA kernel design

Date: 2026-05-01
Status: Design only; no implementation.
Scope: DeepSeek-family Multi-head Latent Attention (MLA) CUDA kernel path for
future DS3 work. This document does not change `infer` behavior.

## Inputs Read

Local ARLE surfaces:

- `docs/projects/2026-05-01-deepseek-v4-readiness.md`
- `docs/projects/2026-04-29-scheduler-pipeline-map.md`
- `crates/cuda-kernels/csrc/attention/decode_attention_varlen_fp8.cu`
- `crates/cuda-kernels/csrc/attention/decode_prep_paged.cu`
- `crates/cuda-kernels/src/paged_kv.rs`
- `crates/deepseek-spec/src/lib.rs`

External implementation references checked on 2026-05-01:

- DeepSeek-V2 paper: `https://arxiv.org/abs/2405.04434`
- DeepSeek-V3 report: `https://arxiv.org/abs/2412.19437`
- FlashInfer repo: `https://github.com/flashinfer-ai/flashinfer`
- DeepSeek FlashMLA repo: `https://github.com/deepseek-ai/FlashMLA`
- SGLang repo: `https://github.com/sgl-project/sglang`
- vLLM MLA API/source docs:
  `https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/attention/mla_attention/`

## MLA Formula

Notation follows the DeepSeek V2/V3 MLA contract, using ARLE field names from
`crates/deepseek-spec` where possible:

- `h_t`: layer input at token `t`
- `c_kv_t = W_DKV h_t`: compressed KV latent, dimension `kv_lora_rank`
- `k_nope_t = W_UK c_kv_t`: per-head non-RoPE key component
- `v_t = W_UV c_kv_t`: per-head value component
- `k_rope_t = RoPE(W_KR h_t)`: shared or low-rank RoPE key component,
  dimension `qk_rope_head_dim`
- `q_t = [q_nope_t, q_rope_t]`, where `q_nope_t` comes from `q_a/q_b` or
  `q_proj`, and `q_rope_t` is RoPE-applied
- `k_t = [k_nope_t, k_rope_t]`
- `attn(q_t, k_t, v_t) = softmax((q_t * k_t) / sqrt(qk_nope_head_dim +
  qk_rope_head_dim)) * v_t`

The serving-critical property is the cache payload. A GQA model caches expanded
`K` and `V` per layer. MLA caches compressed `c_kv_t` plus `k_rope_t`, then
either expands or absorbs projection matrices during attention:

1. **Compute-friendly prefill:** materialize enough expanded K/V structure to
   use a FlashAttention-like path when query length is large.
2. **Memory-friendly decode:** keep cache compressed and compute logits against
   latent `c_kv` plus RoPE key. This saves KV memory but changes the dot-product
   kernel.

This is why MLA is not a pure tensor-name variant of Qwen GQA. It changes both
cache layout and attention math.

## ARLE GQA Varlen Interface

Current ARLE long-context path is GQA with paged KV:

| Surface | Current GQA contract | MLA mismatch |
|---|---|---|
| Decode prep | `decode_prep_paged.cu` normalizes Q/K, applies half-split RoPE to Q/K, and writes expanded K/V into paged pool. | MLA decode prep must write `c_kv` and `k_rope`, not expanded K/V. `W_UK/W_UV` are part of attention or absorbed projections. |
| Durable pool | `PagedKVPool` stores per-layer K/V pages. BF16 uses page size 16; quantized pools use FP8/INT8 storage plus optional scales. | MLA pool needs latent page layout: likely `ckv_pool[layer]` with `kv_lora_rank` and `kpe_pool[layer]` with `qk_rope_head_dim`. |
| Varlen FP8 kernel | `decode_attention_varlen_fp8.cu` reads paged K/V, supports variable Q rows, optional causal mask, FP8/INT8 scales, and split-KV merge. | Split scheduling and page table traversal are reusable. Inner loop is not: it assumes expanded `[kv_head, head_dim]` K/V loads and GQA head mapping. |
| Mixed batch | Qwen3 mixed path lowers decode rows and prefill chunks into one varlen attention launch when BF16, and needs FP8/INT8 varlen wiring. | MLA mixed path likely needs separate prefill/decode modes first. A fused mixed MLA path should wait until BF16 correctness and decode performance are proven. |
| FFI shape | Current attention FFI passes Q, K_pool, V_pool, scales, `kv_indptr`, `kv_indices`, `last_page_len`, head counts, `sm_scale`, splits. | MLA FFI must pass Q-nope/Q-rope or packed Q, `ckv_pool`, `kpe_pool`, latent dimensions, optional projection fragments, and projection/absorption mode. |

Design implication: reuse page metadata, split-KV workspace sizing, test style,
and Rust dispatch patterns, but do not mutate `decode_attention_varlen_fp8.cu`
into a dual GQA/MLA mega-kernel.

## External LoC Comparison

Counts below are nonblank LoC from GitHub `main`, fetched on 2026-05-01 with the
GitHub tree API and raw file downloads. They are directional sizing data, not a
vendored dependency plan.

| Engine | MLA-related surface counted | Nonblank LoC | Interpretation |
|---|---|---:|---|
| vLLM | `vllm/v1/attention/backends/mla/*.py` | 4,979 | Multiple backend adapters: FlashMLA, FlashInfer, Cutlass, Triton, ROCm, sparse. |
| vLLM | common MLA/model-layer files: `mla_attention.py`, `mla.py`, `deepseek_v4_attention.py` | 3,692 | Significant model/backend glue beyond kernels. |
| SGLang | `python/sglang/srt/layers/attention/*mla*.py` | 3,222 | Backend selector layer across FlashInfer, FlashMLA, TRT-LLM, Cutlass, ROCm. |
| SGLang | `python/sglang/srt/models/deepseek_common/*.py` | 2,331 | DeepSeek model glue, weight loading, MLA forward mode selection. |
| FlashInfer | MLA kernels/wrappers under `csrc`, `include/flashinfer`, `flashinfer/mla`, and CUTE DSL attention files | 14,080 | Kernel library scale. Includes SM80/SM90/SM100 and FP8 variants. |
| DeepSeek FlashMLA | repo files with MLA/kernel APIs under `csrc`, `flash_mla`, tests, and benchmark | 21,189 | Specialized dense/sparse SM90/SM100 kernel stack. Not a small patch. |

Load-bearing external facts:

- vLLM's MLA docs explicitly split MLA into compute-friendly prefill and
  data-movement-friendly decode modes.
- FlashInfer has native MLA support and a large dedicated kernel family, not a
  small option on its standard paged attention kernel.
- SGLang exposes several MLA backends and keeps DeepSeek model glue separate
  from the attention backend layer.
- DeepSeek FlashMLA is a dedicated dense/sparse kernel repository with separate
  SM90 and SM100 paths.

## ARLE Landing Path

### Recommendation

Create a new MLA kernel family. Reuse ARLE's existing GQA varlen infrastructure
around it.

Do not extend `decode_attention_varlen_fp8.cu` with MLA branches. That file is
currently a Qwen3 GQA quantized varlen kernel whose core assumptions are:

- expanded K/V pages;
- `num_q_heads / num_kv_heads` GQA mapping;
- one `HEAD_DIM` for K and V;
- K/V scale pointers map directly to per-token expanded K/V rows;
- output accumulation over expanded V.

MLA violates all five assumptions. Adding MLA to the same kernel would create a
half-state where one file contains two attention algorithms with different
cache invariants.

### Commit Sequence

1. **DS3.0 MLA state contract.**
   Add `MlaCacheLayout` and shape helpers to `deepseek-spec` or a small
   `infer`-side adapter. Define durable cache tensors:
   `ckv[layer, pages, page_size, kv_lora_rank]` and
   `kpe[layer, pages, page_size, qk_rope_head_dim]`.

2. **DS3.1 CPU/BF16 reference.**
   Add a CPU or simple CUDA-eager reference path for one layer:
   `c_kv + k_rope` cache write, expanded attention output, and comparison
   against an expanded-K/V formulation. This locks formulas before kernel work.

3. **DS3.2 decode prep.**
   Add `mla_decode_prep_paged.cu` to compute Q components and write `c_kv` plus
   `k_rope` into paged MLA cache. Reuse `decode_prep_paged.cu` launch shape and
   page-table conventions where possible, but keep the cache write ABI separate.

4. **DS3.3 BF16 paged decode kernel.**
   Add `decode_attention_mla_paged_bf16.cu` with split-KV style partial and
   merge phases. Reuse the workspace/merge pattern from
   `decode_attention_varlen_fp8.cu`, but implement an MLA inner loop over
   latent `c_kv` plus `k_rope`.

5. **DS3.4 Rust FFI and dispatch.**
   Add FFI declarations in `crates/cuda-kernels/src/ffi/attention.rs`, wrapper
   calls in `crates/cuda-kernels/src`, and a DeepSeek model path that is still
   behind `AttentionVariant::Mla` unsupported guards until numerical tests pass.

6. **DS3.5 Quantized variants.**
   Add FP8 or block-FP8 only after BF16 parity. Block-FP8 should share DS2 scale
   metadata, not current Qwen KV FP8 assumptions.

7. **DS3.6 Mixed/pre-fill optimization.**
   Only after decode correctness and single-mode prefill parity, evaluate a
   mixed MLA path. Avoid fusing mixed decode+prefill before the cache ABI is
   stable.

## First Kernel ABI Sketch

Candidate BF16 decode run ABI:

```text
decode_attention_mla_paged_bf16(
    q_nope_or_absorbed,      // [total_q, num_q_heads, qk_nope_head_dim or kv_lora_rank]
    q_rope,                 // [total_q, num_q_heads, qk_rope_head_dim]
    ckv_pool,               // [max_pages, page_size, kv_lora_rank]
    kpe_pool,               // [max_pages, page_size, qk_rope_head_dim]
    w_uv_or_absorbed,       // optional value up-projection material
    qo_indptr,
    kv_indptr,
    kv_indices,
    last_page_len,
    out,                    // [total_q, num_q_heads, v_head_dim]
    dims,
    sm_scale,
    num_splits
)
```

Open design decision: whether to pass absorbed projections as precomputed
Q-side tensors or perform `W_UK/W_UV` work inside the attention kernel. vLLM's
mode split suggests choosing separately for prefill and decode.

## Validation Gates

- Shape-only tests for DeepSeek-V3 config from `crates/deepseek-spec`.
- Reference equivalence: compressed MLA path vs expanded K/V path for one layer.
- Page math tests for `ckv/kpe` page addressing and `last_page_len`.
- BF16 decode numerical baseline before any FP8/block-FP8 work.
- `cargo test -p deepseek-spec` and CUDA-kernel unit smoke before model wiring.
- No throughput claim until a real DeepSeek checkpoint path runs and records a
  `docs/experience/wins/` or `docs/experience/errors/` entry.
