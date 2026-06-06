# Gemma4 Official/OSS Adoption Spec for ARLE

**Status:** READ-ONLY study / proposal. No code landed. Pixel-level mirror of
SGLang's Gemma4 integration (`/tmp/sglang-full/python/sglang/srt/`), adopting
official/OSS kernels (FlashInfer / FlashAttention-3 / Marlin / CUTLASS / Triton)
rather than hand-rolled ARLE-original ops.

**Owner directive (ckl):** bring Gemma4 in by *pixel-level copying* the
official/SGLang integration + operators (do NOT improvise), quantization
factored in. Adopt official/OSS kernels, not ARLE-original.

> §0 SOLID note: everything below is a **source survey** of SGLang + HF.
> It is *hypothesis-grade* for any perf/correctness claim until a real Gemma4
> checkpoint runs through ARLE with a needle-retrieval + same-config-twice gate.
> The architecture/operator mapping is high-confidence (read directly from the
> SGLang source); the ARLE wiring plan is a proposal whose risky items are
> flagged.

---

## 1. Availability — Gemma4 IS in SGLang (no need to fall back to Gemma3)

SGLang ships a full Gemma4 model family. Files
(`/tmp/sglang-full/python/sglang/srt/models/`):

| File | Role |
|------|------|
| `gemma4_causal.py` (56 KB) | **Text decoder — the canonical reference for ARLE.** `Gemma4ForCausalLM`, `Gemma4TextModel`, `Gemma4DecoderLayer`, `Gemma4Attention`, `Gemma4MoE`, `Gemma4Router`. |
| `gemma4_unified.py` | Unified text+vision+audio wrapper. |
| `gemma4_mm.py`, `gemma4_vision.py`, `gemma4_audio.py` | Multimodal towers (out of scope for ARLE's text runtime). |
| `gemma4_mtp.py` | Eagle/MTP draft assistant (reuses `Gemma4ForCausalLM.load_weights`). |
| `gemma3_causal.py` | **Reused by Gemma4:** `Gemma4MLP = Gemma3MLP`, `Gemma4TextScaledWordEmbedding = Gemma3TextScaledWordEmbedding`. |
| `gemma2.py` | Predecessor — only relevant as the *source* of soft-cap (which Gemma4 drops). |

Supporting layers:
- `layers/gemma4_fused_ops.py` — Gemma4-specific **Triton** fused kernels
  (qkv-rmsnorm, dual-rmsnorm-residual-scalar, fused routing).
- `layers/layernorm.py` — `Gemma4RMSNorm`, `Gemma3RMSNorm`, `GemmaRMSNorm`, `RMSNorm`.
- `layers/activation.py` — `GeluAndMul` (GeGLU).
- `layers/rotary_embedding/` — `get_rope` (per-layer-type theta).
- `layers/radix_attention.py` — backend-agnostic attention module (carries
  `sliding_window_size`, `logit_cap`).
- HF: `transformers.Gemma4TextConfig` is imported directly
  (`gemma4_causal.py:22`), so HF transformers already defines Gemma4.

**Verdict:** Use `gemma4_causal.py` as the authority. There is no architectural
delta to bridge from Gemma3 — Gemma4 is its own arch. (The historical
Gemma3→Gemma4 delta is documented in §2.6 for context, since ARLE supports
neither today.)

---

## 2. Architecture — the exact per-layer forward

Source: `models/gemma4_causal.py` unless noted. Gemma4 is a **sparse-MoE +
dense-MLP hybrid** decoder with several Gemma-3n-derived features (PLE,
KV-sharing). It is materially more complex than Qwen3.5 or DSv4.

### 2.1 Config fields consumed (`Gemma4TextConfig`)

Read off every `config.*` access in `gemma4_causal.py`:

```
hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads,
head_dim, intermediate_size, vocab_size, max_position_embeddings,
rms_norm_eps, attention_bias, tie_word_embeddings,
layer_types                  # per-layer "sliding_attention" | "full_attention"
sliding_window               # SWA window (HF inclusive; SGLang uses sliding_window-1)
rope_parameters              # NESTED per layer-type: {layer_type: {rope_theta, rope_type, partial_rotary_factor}}
hidden_activation            # dense MLP act ("gelu_pytorch_tanh")
# MoE:
num_experts, top_k_experts, moe_intermediate_size, enable_moe_block
# Gemma-3n-derived extras (getattr w/ defaults — may be 0/absent on small SKUs):
swa_num_key_value_heads      # SWA layers may have fewer KV heads than full layers
swa_head_dim                 # SWA layers may use a different head_dim than full layers
num_kv_shared_layers         # last N layers reuse an earlier layer's KV cache
use_double_wide_mlp          # KV-shared layers get 2x intermediate MLP
attention_k_eq_v             # full-attn layers: V weight == K weight (no v_proj)
hidden_size_per_layer_input  # PLE dim (Per-Layer Embeddings); 0 disables
vocab_size_per_layer_input   # PLE embedding table vocab
```

> **Posture note:** the small/common Gemma4 SKU likely has `enable_moe_block=False`,
> `hidden_size_per_layer_input=0`, `num_kv_shared_layers=0`, `attention_k_eq_v=False`.
> ARLE Phase 1 should target that "plain dense Gemma4" config first and gate the
> MoE/PLE/KV-share machinery behind config presence (see §5 phasing). All four
> are read via `getattr(..., default)` so a config that omits them takes the
> simple path.

### 2.2 Embedding (entry)

`Gemma4TextScaledWordEmbedding` (= `Gemma3TextScaledWordEmbedding`,
`gemma3_causal.py:523`): an `nn.Embedding` whose `forward` multiplies by
`embed_scale = hidden_size**0.5` (`gemma4_causal.py:809`). **This √hidden_size
embedding scale is a Gemma signature** — drop it and output is garbage.

PLE path (only if `hidden_size_per_layer_input > 0`, `gemma4_causal.py:875-941`):
a second scaled embedding table (`embed_tokens_per_layer`, scale
`hidden_size_per_layer_input**0.5`) is gathered per token, projected
(`per_layer_model_projection`, scale `hidden_size**-0.5`), RMSNorm'd, and
combined `(proj + ple_embed) * rsqrt(2)`. Result is `[T, num_layers, ple_dim]`,
sliced per layer and fed into each decoder layer's PLE gate.

### 2.3 Attention (`Gemma4Attention`, `gemma4_causal.py:280-512`) — GQA

Per-layer **alternating LOCAL sliding-window / GLOBAL attention** driven by
`config.layer_types[layer_id]`:

- `sliding_attention` → `self.sliding_window = config.sliding_window`, may use
  `swa_num_key_value_heads` / `swa_head_dim`.
- `full_attention` → `sliding_window = None`, uses `head_dim` / `num_key_value_heads`.

Forward (the exact sequence):
1. `qkv = qkv_proj(hidden_states)`; split into q/k/v
   (`q_size = num_heads*head_dim`, `kv_size = num_kv_heads*head_dim`).
2. **Per-head Q/K/V RMSNorm** (`Gemma4RMSNorm` over `head_dim`):
   - `q_norm`, `k_norm`: standard `norm(x) * weight` (`scale_shift=0`, `with_scale=True`).
   - `v_norm`: **`with_scale=False`** → weight is ones → plain `norm(x)` (no learned scale).
   - Fused path: `gemma_qkv_rmsnorm` Triton kernel (`gemma4_fused_ops.py:194`)
     normalizes Q/K/V in-place in one launch when on CUDA/XPU.
3. **RoPE** with **per-layer-type theta**: `rope_parameters[layer_type]["rope_theta"]`
   — local layers and global layers have different bases (`gemma4_causal.py:354-391`).
   `is_neox_style=True`, `partial_rotary_factor` from config (usually 1.0).
4. `attn(q, k, v)` via `RadixAttention` (`gemma4_causal.py:393-405`):
   - **`scaling = 1`** (the literal `1` at line 396) — Gemma4 does NOT apply
     `query_pre_attn_scalar**-0.5`. **This is a Gemma4 vs Gemma2/Gemma3 delta**
     (Gemma2/3 pass `config.query_pre_attn_scalar**-0.5`; Gemma4's
     query scaling is folded into the q_norm weights at training time).
   - **`logit_cap = 0.0`** — Gemma4 has **NO attention logit soft-capping**
     (that was Gemma2-only; dropped in Gemma3 and Gemma4).
   - `sliding_window_size = self.sliding_window` (or `None`).
5. `o_proj(attn_output)`.

KV-sharing (`gemma4_causal.py:362-381`, only if `num_kv_shared_layers>0`): the
last N layers skip writing KV; they re-read an earlier same-type layer's KV
cache via `kv_shared_layer_index`. `attention_k_eq_v` (full-attn layers): no
`v_proj` in the checkpoint — the K weight is duplicated into the V shard at load
time (`gemma4_causal.py:1249-1258, 1323-1324`).

### 2.4 MLP / MoE (`gemma4_causal.py:190-277, 515-565`)

- **Dense MLP** = `Gemma3MLP` (`gemma3_causal.py:77`): `gate_up_proj`
  (`MergedColumnParallelLinear` → 2× intermediate), **GeGLU** via
  `GeluAndMul(approximate="tanh")` (`activation.py:136`, enforces
  `hidden_activation == "gelu_pytorch_tanh"`), then `down_proj`.
- **MoE** (only if `enable_moe_block`, `gemma4_causal.py:190-277`): a *parallel*
  MoE block running **alongside** the dense MLP, not replacing it. Both branches
  are computed and summed (`gemma4_causal.py:680-715`).
  - **Router** (`Gemma4Router`, `:132-187`): RMSNorm (no learned weight) →
    `root_size = hidden_size**-0.5` scaling → learned per-dim `scale` (folded
    into norm weight via `fuse_scale`) → `ReplicatedLinear` to `[T, num_experts]`.
  - **Routing** (`:222-248`): softmax-over-ALL-experts → top-k → renormalize,
    folded into one fused Triton kernel (`gemma4_fused_routing`,
    `gemma4_fused_ops.py:421`). `per_expert_scale` (`[num_experts]`) is gathered
    and multiplied into the routing weights so the fused MoE GEMM computes
    `Σ_e expert_e * w_e * scale_e`.
  - **Experts**: `get_moe_impl_class(quant_config)` (FusedMoE / EP-MoE),
    **`activation="gelu"`** (plain gelu, NOT tanh — note the asymmetry vs the
    dense MLP), `reduce_results=True`.

### 2.5 Norm + residual pattern (`Gemma4DecoderLayer.forward`, `:640-753`)

Four RMSNorms per layer (Gemma's "sandwich" norm), residual ADDED (not the
Gemma2 pre+post-only pattern):

```
residual = x
x = input_layernorm(x)            # standard RMSNorm
x = self_attn(x)
x = post_attention_layernorm(x)
x, residual = pre_feedforward_layernorm(x, residual)   # fused add+norm: residual += x; x = norm(residual)
x = mlp(x)                                              # (+ parallel MoE if enabled)
x = post_feedforward_layernorm(x); x = (x + residual)
x = x * layer_scalar                                    # per-layer learned scalar (buffer, default ones)
# PLE contribution (if has_ple): gate=gelu_tanh(per_layer_input_gate(x)); x += norm(per_layer_projection(gate*ple_input))
```

- `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`,
  `post_feedforward_layernorm` are plain **`RMSNorm`** (`layernorm.py:200`,
  standard `norm*weight`) — **NOT** `GemmaRMSNorm` (`*(1+weight)`). This is a
  Gemma4 delta: Gemma2/3 used `(1+weight)` semantics; Gemma4's decoder-layer
  norms use plain `norm*weight` (the `+1` is baked into the checkpoint weights).
  Q/K/V norms use `Gemma4RMSNorm` (also plain `norm*weight`).
- `layer_scalar` (`gemma4_causal.py:636`): a per-layer learned scalar buffer
  (default 1.0) multiplying the post-FF residual. Honor it.
- Fused kernels (`gemma_rmsnorm_residual_scalar`,
  `gemma_dual_rmsnorm_residual_scalar`, `gemma4_fused_ops.py`) collapse the
  norm+residual+scalar tail into one Triton launch on CUDA.

Final: `norm = RMSNorm` (plain, `:863`); `lm_head` tied to `embed_tokens` when
`tie_word_embeddings` and TP-only (`:1089`). No final-logit soft-cap (Gemma2-only).

### 2.6 Gemma3 → Gemma4 delta (context only; ARLE has neither)

| Aspect | Gemma3 (`gemma3_causal.py`) | Gemma4 (`gemma4_causal.py`) |
|--------|----------------------------|-----------------------------|
| Attention scaling | `query_pre_attn_scalar**-0.5` (`gemma3:158`) | **`1`** (folded into q_norm) |
| Q/K norm | `Gemma3RMSNorm` = `*(1+weight)` | `Gemma4RMSNorm` = `*weight`; adds **v_norm** |
| Decoder norms | `Gemma3RMSNorm` = `*(1+weight)` | plain `RMSNorm` = `*weight` |
| MoE | none (dense only) | optional **parallel MoE** + router |
| PLE | none | optional Per-Layer Embeddings |
| KV-sharing | none | optional `num_kv_shared_layers` |
| `attention_k_eq_v` | none | optional (full-attn V==K) |
| `layer_scalar` | none | per-layer learned residual scalar |
| Soft-cap | none (already dropped) | none |

Both keep: √hidden_size embed scale, alternating local/global, per-layer-type
RoPE theta, sandwich 4-norm layout, GeGLU.

---

## 3. Official/OSS operators that run Gemma4 in SGLang

### 3.1 Attention backend — FlashAttention-3 (default) / FlashInfer

SGLang routes Gemma4's `RadixAttention` to one of:

- **FlashAttention-3** (`layers/attention/flashattention_backend.py`) — the
  default GPU backend. Imports `flash_attn_varlen_func`,
  `flash_attn_with_kvcache` from **`sgl_kernel.flash_attn`** (the FA3 wheel,
  `flashattention_backend.py:33,173-191`). Honors:
  - **sliding window**: `window_size = (layer.sliding_window_size, 0)` for SWA
    layers, `(-1,-1)` for global (`flashattention_backend.py:813-818, 1274-1279`).
  - **soft-cap**: `softcap = layer.logit_cap` passed on every call
    (`:934, 970, 988, 1017, 1160, 1183, 1206, 1332, 1353`). For Gemma4 this is
    `0.0` (no-op), but the plumbing is mandatory for Gemma2-family compat.
  - separate **paged KV pools** for SWA vs global layers
    (`use_sliding_window_kv_pool`, `:143, 695-709`).
- **FlashInfer** (`layers/attention/flashinfer_backend.py`) — alt backend, same
  contract: `sliding_window` via `update_sliding_window` (`:977, 1263`),
  `logits_soft_cap = layer.logit_cap` (`:750, 784-843`), per-window wrapper KV
  pools (`:1011, 1307`).

Neither is in `sgl-kernel/csrc/` — both are **external PyPI wheels** (`flash-attn`
/ FA3, `flashinfer-python`). The only attention C++ in
`/tmp/sglang-scan/sgl-kernel/csrc/attention/` is CUTLASS MLA (DeepSeek) +
`merge_attn_states.cu` — **not** the GQA path Gemma uses.

### 3.2 GEMM / quant path

- Linear layers: `QKVParallelLinear`, `MergedColumnParallelLinear`,
  `RowParallelLinear`, `ReplicatedLinear`, `ParallelLMHead`
  (`layers/linear.py`, `vocab_parallel_embedding.py`). The GEMM kernel is chosen
  by `quant_config` (BF16 cuBLAS / Marlin / CUTLASS-FP8 / etc., see §4).
- MoE GEMM: `FusedMoE` / EP-MoE grouped GEMM
  (`layers/moe/fused_moe_triton/`, `layers/moe/ep_moe/`).

### 3.3 Norm / activation / routing

- RMSNorm: **`sgl_kernel`** `rmsnorm`, `gemma_rmsnorm`, `fused_add_rmsnorm`,
  `gemma_fused_add_rmsnorm` (`layernorm.py:83-88`), optional FlashInfer
  `flashinfer.norm.layernorm` (`:56-77`). HF-semantics JIT path
  `sglang.jit_kernel.rmsnorm_hf` (`:107-130`).
- GeGLU: `sgl_kernel` `gelu_tanh_and_mul` (`activation.py:59-65`).
- Gemma4-specific fused ops: **Triton** in `layers/gemma4_fused_ops.py`
  (`gemma_qkv_rmsnorm`, `gemma_rmsnorm_residual_scalar`,
  `gemma_dual_rmsnorm_residual_scalar`, `gemma4_fused_routing`,
  `gemma_routing_post_topk`). Pure Triton → portable, no C++.

### 3.4 Gemma-specific gotchas the integration MUST honor

1. **√hidden_size embedding scale** (`:809`). Garbage output if dropped.
2. **Alternating local/global** per `layer_types` — each layer is one or the
   other, with **different RoPE theta** and (possibly) different KV-head count /
   head_dim.
3. **SWA window is `config.sliding_window - 1`** (HF inclusive, SGLang exclusive,
   `:71-72`). Off-by-one here silently corrupts long-context attention.
4. **`scaling = 1`** (not `qk_scalar**-0.5`) — Gemma4-specific.
5. **No soft-cap for Gemma4** (`logit_cap=0`), but the backend API still takes a
   `softcap`/`logit_cap` arg — wire it (pass 0) so the same path serves
   Gemma2 later.
6. **v_norm has no learned scale** (`with_scale=False`); q/k norm do.
7. **Decoder norms are plain `norm*weight`**, NOT `*(1+weight)` — opposite of
   the Gemma2/3 RMSNorm convention.
8. **Parallel MoE** runs *alongside* the dense MLP (sum), not instead of it.
9. **MoE act = plain `gelu`; dense MLP act = `gelu_tanh`** — different.
10. **`per_expert_scale` folded into routing weights**; **`layer_scalar`**
    per-layer residual multiplier.

---

## 4. Quantization schemes + official/OSS kernel per scheme

From the Gemma4 `load_weights` (`gemma4_causal.py:1170-1363`) checkpoint-format
handling + SGLang `layers/quantization/`:

| Scheme | Checkpoint families | Official/OSS kernel (SGLang) | ARLE today |
|--------|--------------------|------------------------------|------------|
| **BF16** | base HF release | cuBLAS GEMM + FA3/FlashInfer attn; fused MoE BF16 grouped GEMM | ARLE has BF16 GEMM + own GQA attn |
| **FP8 (W8A8 dynamic)** | `RedHatAI/*-FP8-Dynamic` (compressed-tensors); per-expert keys (`:1264-1292`) | `quantization/fp8.py` + `fp8_kernel.py` (CUTLASS/`sgl_kernel` FP8 GEMM); MoE `w8a8_fp8` | ARLE has DSv4 FP8 grouped GEMM (DeepGEMM) |
| **NVFP4** | `nvidia/Gemma-4-*-NVFP4` (ModelOpt); `weight_scale_2` (`:1264-1292`) | `modelopt_quant.py`, `marlin_utils_fp4.py`, `mxfp4_flashinfer_*` (FlashInfer/CUTLASS FP4) | none |
| **4-bit AWQ** | AWQ checkpoints | `quantization/awq/` + **Marlin** (`marlin_utils.py`) | none |
| **4-bit GPTQ** | GPTQ checkpoints | `quantization/gptq/` + **Marlin** | none |
| **MXFP4** | MXFP4 | `mxfp4_marlin_moe.py`, `mxfp4_flashinfer_*` | none |
| **INT8 (W8A8)** | int8 | `w8a8_int8.py`, `int8_kernel.py` | DSv4-side int8 exists |
| **bitsandbytes** | bnb | `bitsandbytes.py` | none |

Gemma4 load_weights handles **three MoE checkpoint layouts** explicitly
(`:1180-1311`): per-expert compressed-tensors/NVFP4 keys, BF16 fused
`[E,2I,H] gate_up_proj` (chunked to w1/w3), and the dense stacked qkv/gate_up
mapping. Mirror all three or restrict the supported checkpoint set explicitly.

**Recommended ARLE quant order:** BF16 first (kernels exist), then FP8 (reuse
DSv4 FP8 grouped GEMM + per-tensor/per-channel FP8 GEMM), then Marlin-4bit
(AWQ/GPTQ) requires vendoring Marlin. NVFP4 last.

---

## 5. Adoption plan for ARLE

### 5.1 New crate: `crates/gemma-spec` (mirror `qwen35-spec` / `deepseek-spec`)

`crates/gemma-spec/src/lib.rs`:
- `GemmaConfig` (serde from HF `config.json`) — all §2.1 fields, with
  `getattr`-equivalent `Option<...>`/defaults for the Gemma-3n extras
  (`enable_moe_block`, `hidden_size_per_layer_input`, `num_kv_shared_layers`,
  `swa_*`, `attention_k_eq_v`, `use_double_wide_mlp`).
- `LayerType { Sliding, Full }` enum + `layer_types: Vec<LayerType>` parse.
- `rope_parameters` nested map → per-layer-type `rope_theta` resolution helper
  (`rope_theta_for(layer_type)`).
- Reuse the shared `Shard` enum convention from `qwen35-spec` for TP layout, with
  the QKVParallelLinear KV-replication rule applied at runtime.
- `GemmaAttentionTensorNames` / `GemmaMlpTensorNames` / `GemmaMoeTensorNames`
  contracts mirroring the SGLang `load_weights` `stacked_params_mapping`
  (`gemma4_causal.py:1171-1185`).

### 5.2 New model file: `crates/infer-cuda/src/gemma4.rs` (mirror `qwen35.rs`/`dsv4.rs`)

ARLE owns its KV state per slot (no PagedKVPool) on the dense correctness path,
same posture as `qwen35.rs`/`dsv4.rs`. The decoder forward mirrors §2.5 exactly.

Structs (mirroring `dsv4.rs` decomposition):
- `Gemma4Model` (layers, embed, norm, lm_head) implementing the ARLE
  `ModelForward` contract (`infer-cuda/src/model.rs`).
- `Gemma4Attention` (qkv/o proj, q/k/v norm weights, per-layer-type RoPE cache,
  sliding-window vs full dispatch).
- `Gemma4Mlp` (gate_up/down, GeGLU).
- `Gemma4Moe` + `Gemma4Router` (gated behind `enable_moe_block`).
- `Gemma4SlotState` (per-slot K/V caches; separate handling for SWA-windowed vs
  full layers; KV-shared layers point at the donor layer's cache).
- PLE state (gated behind `hidden_size_per_layer_input > 0`).

Loader: extend `crates/infer-cuda/src/loader.rs` with
`from_gemma4_safetensors`, mirroring the three MoE checkpoint layouts only for
the quant schemes ARLE supports in the target phase.

Dispatch: register the Gemma4 architecture string in the executor's model
selection (same place `Dsv4Model` / `Qwen35` are wired in
`infer-cuda/src/lib.rs` / `model.rs` / loader).

### 5.3 Per-operator table — Gemma op → official/OSS kernel → ARLE wiring

| Gemma4 op | SGLang official/OSS kernel | Reference (file:line) | ARLE plan (vendor / reuse / new) |
|-----------|---------------------------|----------------------|----------------------------------|
| Scaled embedding (×√H) | `nn.Embedding` + scalar mul | `gemma3_causal.py:523-539`, `gemma4_causal.py:809` | **Reuse** ARLE `ops::embedding_batch` + scale mul (trivial) |
| GQA prefill attention (local+global, varlen) | **FA3** `flash_attn_varlen_func` | `flashattention_backend.py:33,173-191` | **Reuse ARLE GQA prefill** (`csrc/attention/prefill_attention.cu`, HD128/kv8) for Phase 1; **vendor FlashInfer/FA** for production paged + multi-window perf (see §5.4). Sliding-window mask + per-layer-type theta required. |
| GQA decode attention (paged, sliding+global) | **FA3** `flash_attn_with_kvcache` / FlashInfer | `flashattention_backend.py:34`, `flashinfer_backend.py:750+` | **Reuse ARLE** `csrc/attention/decode_*` for Phase 1; **vendor FlashInfer** for paged SWA pools long-term |
| Attention soft-cap | FA3 `softcap=` / FlashInfer `logits_soft_cap=` | `flashattention_backend.py:934`, `flashinfer_backend.py:750` | **Wire the arg = 0** (Gemma4 no-op; keeps Gemma2 path open) |
| Per-head Q/K/V RMSNorm (fused) | **Triton** `gemma_qkv_rmsnorm` | `gemma4_fused_ops.py:194` | **Reuse** ARLE `csrc/attention/prefill_attention.cu` `prefill_qk_norm_rope` (already fuses QK-norm+RoPE); **add v_norm** (no-scale) + Gemma4 `norm*weight` (not `*(1+w)`) semantics. Or port the Triton kernel. |
| RoPE (per-layer-type theta, neox) | `get_rope` | `gemma4_causal.py:383-391`, `rotary_embedding/` | **Reuse** ARLE RoPE; build **two** cos/sin caches (local θ, global θ), select per layer |
| RMSNorm (decoder, plain `norm*w`) | `sgl_kernel` `rmsnorm` / `fused_add_rmsnorm` | `layernorm.py:83,200` | **Reuse** ARLE RMSNorm; **use plain `norm*weight`** (NOT Gemma `+1`) for Gemma4 decoder norms |
| Fused (rmsnorm+residual)×scalar tail | **Triton** `gemma_rmsnorm_residual_scalar` | `gemma4_fused_ops.py:54` | Phase 1: do as separate ARLE ops; fuse later (perf follow-up, bench-gated) |
| GeGLU dense MLP | `sgl_kernel` `gelu_tanh_and_mul` | `activation.py:136-146` | **New ARLE op** `gelu_tanh_mul` (ARLE has `silu_mul`; add GeGLU-tanh variant) |
| MoE router (norm+rootscale+scale+proj) | `Gemma4Router` + GEMM | `gemma4_causal.py:132-187` | **New** (only if MoE SKU); replicated GEMV + fused-scale norm |
| MoE routing (softmax-topk-renorm + per-expert scale) | **Triton** `gemma4_fused_routing` | `gemma4_fused_ops.py:421` | **Reuse** ARLE MoE TopK + fold `per_expert_scale` into weights (`infer-moe`) |
| MoE expert GEMM (gelu, grouped) | `FusedMoE`/EP-MoE grouped GEMM | `gemma4_causal.py:256-269`, `moe/` | **Reuse** ARLE `moe::moe_forward` grouped GEMM; **set act=gelu** (add gelu variant to the grouped path) |
| Linear (qkv/o/gate_up/down) GEMM | cuBLAS / Marlin / CUTLASS per quant | `layers/linear.py` | **Reuse** ARLE GEMM (BF16); FP8 reuse DSv4 path; Marlin = vendor |
| lm_head + logits | `LogitsProcessor` (no soft-cap) | `gemma4_causal.py:1083,1158` | **Reuse** ARLE `sample_cuda_token` (no final soft-cap) |
| PLE (gate/proj/norm/combine) | `ReplicatedLinear` + gelu-tanh + RMSNorm | `gemma4_causal.py:742-750, 875-941` | **New** (only if PLE SKU); defer to Phase 3 |
| KV-sharing | layer_id remap, skip KV write | `gemma4_causal.py:362-381, 506` | **New** (only if KV-share SKU); defer to Phase 3 |

### 5.4 Vendoring: what ARLE must bring in

ARLE today vendors **FlashMLA** (`crates/cuda-kernels/vendor/flashmla/`),
**DeepGEMM**, **tilekernels** — but Gemma is **GQA, not MLA**, so FlashMLA does
**NOT** apply. ARLE has **no** FlashInfer / FlashAttention vendored.

- **Phase 1 (BF16 dense Gemma4, no FlashInfer):** reuse ARLE's existing CUDA C
  GQA attention kernels (`csrc/attention/prefill_attention.cu`,
  `decode_*`), extending them for the sliding-window mask + per-layer-type theta
  + v_norm. No vendoring needed — fastest path to a correct Gemma4.
- **Phase 2 (production paged + perf):** **vendor FlashInfer** (or the FA3
  kernels SGLang uses via `sgl_kernel.flash_attn`) for paged GQA decode with
  per-window (SWA vs global) KV pools and native sliding-window + softcap.
  This is the official kernel SGLang runs Gemma on, and matches the
  "adopt official/OSS kernels, not ARLE-original" directive. Vendor under
  `crates/cuda-kernels/vendor/flashinfer/` mirroring the flashmla vendoring.
- **Quant:** FP8 reuses DSv4 grouped-GEMM; **Marlin** (AWQ/GPTQ-4bit) and
  **NVFP4/MXFP4** require vendoring CUTLASS/Marlin kernels — defer until a 4-bit
  Gemma4 checkpoint is the target.

### 5.5 Phasing (de-risked, config-gated)

- **Phase 1 — dense BF16 Gemma4** (`enable_moe_block=False`, no PLE/KV-share):
  gemma-spec + gemma4.rs forward + reuse ARLE GQA attn (extend for SWA/theta/v_norm)
  + GeGLU op + load_weights (dense layout). Gate: needle retrieval + same-config-twice.
- **Phase 2 — FP8 + FlashInfer attn** (vendor FlashInfer, wire FP8 GEMM, paged SWA pools).
- **Phase 3 — MoE + PLE + KV-share** (the Gemma-3n-derived machinery), only if
  a checkpoint with those config fields is the target.
- **Phase 4 — Marlin 4-bit / NVFP4** (vendor Marlin/CUTLASS-FP4).

---

## 6. Biggest integration risks (flagged)

1. **Sliding-window correctness (off-by-one + per-window KV).** `sliding_window-1`
   (HF-inclusive vs SGLang-exclusive, `:71`); ARLE must apply the window mask in
   its GQA kernel and key the right KV slice. ARLE's DSv4 SWA ring machinery
   (`infer-cuda/src/attention.rs:558-632`) is a starting point but is MLA/ring
   shaped, not GQA-paged — likely needs a Gemma-specific window mask. **Highest
   correctness risk.**
2. **Alternating local/global with different RoPE theta (and possibly different
   KV-head count / head_dim per layer type).** Wrong-theta-per-layer is exactly
   the DSv4 long-context bug class (`project_dsv4_compressed_attention_longctx_bug`).
   Build separate local/global RoPE caches; verify with needle retrieval to ≥
   sliding_window depth.
3. **`scaling=1` + query-scale folded into q_norm.** Easy to wrongly copy
   Gemma2/3's `qk_scalar**-0.5`. Gemma4 uses `1`; the scale lives in the q_norm
   weights. Wrong scaling = silently degraded (not crash) output.
4. **RMSNorm semantics split.** Decoder norms = plain `norm*weight`; Gemma2/3
   convention is `*(1+weight)`. Q/K/V norms = `Gemma4RMSNorm` (`norm*weight`,
   v_norm no scale). Mixing these up is silent quality loss.
5. **Soft-cap plumbing.** Gemma4 = 0 (no-op), but the attention API must accept
   the arg or the Gemma2 path is blocked later. Low risk if wired now.
6. **MoE asymmetry.** Parallel MoE summed with dense MLP; MoE act=`gelu`, dense
   act=`gelu_tanh`; `per_expert_scale` folded into routing weights. Easy to get
   the activation or the parallel-vs-replace structure wrong.
7. **Vendoring FlashInfer/FA is non-trivial** (build system, sm-tier coverage,
   paged-pool integration with ARLE's slot KV). Phase 1 sidesteps it by reusing
   ARLE GQA kernels; treat FlashInfer as a perf follow-up, bench-gated.
8. **Checkpoint-format sprawl.** Three MoE layouts + per-expert FP8/NVFP4 keys +
   `attention_k_eq_v` K→V duplication + tied embeddings. Restrict the supported
   checkpoint set explicitly per phase rather than mirroring all of
   `load_weights` at once.
9. **§0 evidence gap.** No Gemma4 checkpoint has been run through ARLE; every
   perf/correctness claim is source-survey hypothesis until a real forward +
   needle gate. Canonical Metal target stays Qwen3.6; CUDA Gemma4 bring-up needs
   the 8×H20 / Colab lane.

---

## Appendix — primary source paths

- `/tmp/sglang-full/python/sglang/srt/models/gemma4_causal.py` (canonical)
- `/tmp/sglang-full/python/sglang/srt/models/gemma3_causal.py` (MLP + embedding reuse)
- `/tmp/sglang-full/python/sglang/srt/models/gemma2.py` (soft-cap source; dropped in G4)
- `/tmp/sglang-full/python/sglang/srt/layers/gemma4_fused_ops.py` (Triton fused ops)
- `/tmp/sglang-full/python/sglang/srt/layers/layernorm.py` (Gemma4RMSNorm:818)
- `/tmp/sglang-full/python/sglang/srt/layers/activation.py` (GeluAndMul:136)
- `/tmp/sglang-full/python/sglang/srt/layers/attention/flashattention_backend.py` (FA3 sliding+softcap)
- `/tmp/sglang-full/python/sglang/srt/layers/attention/flashinfer_backend.py` (FlashInfer sliding+softcap)
- `/tmp/sglang-full/python/sglang/srt/layers/quantization/` (fp8/awq/gptq/marlin/modelopt)
- ARLE mirrors: `crates/qwen35-spec/src/lib.rs`, `crates/deepseek-spec/src/v4.rs`,
  `crates/infer-cuda/src/{qwen35,dsv4,attention,moe,loader,model}.rs`,
  `crates/cuda-kernels/csrc/attention/`, `crates/cuda-kernels/vendor/flashmla/`
