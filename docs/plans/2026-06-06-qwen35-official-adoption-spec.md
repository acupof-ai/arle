# Qwen3.5 official/OSS kernel adoption spec — mirror SGLang pixel-level

**Date:** 2026-06-06. **Driver (ckl):** for each model, replace ARLE's hand-rolled
CUDA compute operators with the official/open-source ones (FlashInfer / FlashAttention /
Marlin / DeepGEMM / cutlass / FLA / causal_conv1d) and mirror SGLang's integration
exactly — do NOT hand-roll, do NOT improvise. Quantization factored in.
**Principle:** [[../../memory/feedback_no_closed_door_solutions]].
**Precedent:** [`2026-06-06-dsv4-handrolled-kernel-audit.md`](2026-06-06-dsv4-handrolled-kernel-audit.md)
(DSv4 hand-rolled kernels duplicated vendored FlashMLA/DeepGEMM; fix = wire official + delete hand-roll).

**Operative rule (same as DSv4 audit):** zero self-developed compute operators UNLESS
proven better (same-binary A/B) than the best official/OSS kernel. This is a
**per-operator license-or-kill**, NOT a bulk delete. Each DELETE row is a **hypothesis**
(§0) until the official call-shape is wired + the gate passes (needle retrieval +
same-config-twice non-determinism floor + matched A/B). Glue with no upstream drop-in stays.

> **§0 status discipline.** Everything below is a source-survey mapping (ARLE source +
> SGLang source). That is *hypothesis*, not evidence. No adoption is licensed until the
> wired official kernel passes the correctness gate and a same-binary A/B on the SLO shape.

---

## 0. Model shape (both sides agree — this is the load-bearing fact)

Qwen3.5 / Qwen3.6 is a **HYBRID** model. Per-layer `config.layer_types` interleaves:

- **Linear-attention layers (majority)** — **GatedDeltaNet (GDN)**: input projections
  (qkvz + ba) → depthwise **causal conv1d** front end → **gated-delta-rule** recurrent
  scan (carries per-slot recurrent state + conv ring across prefill/decode) → **gated
  output RMSNorm** (gate = `z`) → out_proj. This is the Qwen3-Next / Mamba-family linear
  attention, NOT softmax attention.
- **Full-attention layers (periodic, every `full_attention_interval`)** — **gated GQA**:
  `qkv_proj` rows = `heads*head_dim*(1+attn_output_gate)` (the extra half is a per-head
  **sigmoid gate** applied to the attention output), q/k RMSNorm, RoPE
  (`partial_rotary_factor`), HD256 / q16 / kv2 on the canonical Qwen3.6-35B-A3B.
- **MLP** — dense MLP (`qwen3_5_text`) or **MoE** (`qwen3_5_moe_text`,
  `Qwen2MoeSparseMoeBlock`: top-k router + grouped experts + optional shared-expert fusion).

**ARLE side** (`crates/infer-cuda/src/qwen35.rs`): `Qwen35Attn::{Full,Linear}` enum
matches exactly — `FullAttn` (q/k/v/o proj + q/k norm, gated q_proj) and `LinearAttn`
(in_proj_qkv/z/b/a + conv1d_weight + dt_bias + a_log + norm_weight + out_proj). MoE via
`crate::moe::moe_forward`. **The clean CUDA Qwen3.5 path is BF16-only** and runs the
*uncached full-prefix recompute* path (owns its KV, no PagedKVPool); quant swap points for
the MoE GEMM are flagged as follow-ups (qwen35.rs header lines 24-26).

**SGLang side** (`models/qwen3_5.py`): `Qwen3_5GatedDeltaNet` (line 118),
`Qwen3_5LinearDecoderLayer` (537), `Qwen3_5AttentionDecoderLayer` (671),
`Qwen3_5ForCausalLM` (985) / `Qwen3_5MoeForCausalLM` (1280).

---

## 1. ARLE Qwen3.5 forward — COMPUTE operator inventory

Source: `qwen35.rs` `forward_tokens` (line 710) / `full_attention` (1117) /
`linear_attention` (1238), `ops.rs`, `moe.rs`. Every device-side compute op:

| # | ARLE op (FFI symbol) | ARLE source | csrc file | hand-rolled? |
|---|---|---|---|---|
| E | `embedding_batched_cuda` | ops.rs:50 | misc/elementwise_basic.cu | hand-rolled (trivial) |
| N1 | `rms_norm_batched_offset_cuda` (1+w offset norm) | qwen35.rs:1412 | misc/norm.cu | hand-rolled |
| N2 | `rms_norm_offset_cuda` (final norm, vec) | qwen35.rs:1439 | misc/norm.cu | hand-rolled |
| N3 | `rms_norm_gated_cuda` (GDN gated output norm) | qwen35.rs:1352 | misc/norm.cu | hand-rolled |
| G1 | `gemm_cuda` (BF16 dense GEMM — all q/k/v/o/in/out/gate/up/down proj + lm_head batch) | ops.rs:146 | gemm/ (cublas-class hand-roll) | hand-rolled |
| G2 | `gemv_cuda` (BF16 lm_head single-token) | ops.rs:178 | gemm/gemv.cu | hand-rolled |
| A1 | `prefill_attention_hd256_prep_cuda` (q/k RMSNorm + RoPE + KV cache write, fused) | qwen35.rs:1163 | attention/prefill_attention_hd256.cu | hand-rolled |
| A2 | `nonpaged_prefill_attention_cuda` (causal softmax attn over contiguous cache; decode = qlen 1) | qwen35.rs:1195 | attention/nonpaged_prefill_attention.cu | hand-rolled |
| A3 | `attention_gate_batch_hd256_cuda` (per-head sigmoid gate on attn output) | qwen35.rs:1219 | attention/prefill_attention_hd256.cu | hand-rolled (glue) |
| L1 | `conv1d_prefill_cuda` (GDN depthwise causal conv1d + ring) | qwen35.rs:1278 | misc/conv1d.cu, conv1d_prefill_batch.cu | hand-rolled |
| L2 | `gated_delta_rule_decode_cuda` / `gated_delta_rule_prefill_recurrent_cuda` (GDN recurrent scan) | qwen35.rs:1307/1323 | misc/gated_delta_rule.cu, gdr_*.cu | hand-rolled |
| M1 | `moe_bf16_grouped_gemm_pair_batch` (MoE gate+up grouped GEMM) | moe.rs:288 | gemm/moe_grouped_gemm.cu | hand-rolled |
| M2 | `moe_bf16_grouped_gemm_batch` (MoE down grouped GEMM) | moe.rs:322 | gemm/moe_grouped_gemm.cu | hand-rolled |
| M3 | `qwen36_route_cuda` / `qwen36_renorm_topk_weights_cuda` / `qwen36_add_shared_expert_gated_cuda` (router topk + renorm + shared-expert) | moe.rs | moe/qwen36_route.cu | hand-rolled (router compute) + glue |
| EL | `add_cuda` (residual), `silu_mul_cuda` (SwiGLU) | ops.rs:209/239 | misc/elementwise_basic.cu, fused_mlp.cu | hand-rolled (trivial) |
| S | `sample_cuda_token` (`argmax_cuda` / `gpu_sample_cuda`) | executor.rs | misc/sampling.cu | hand-rolled |
| R | `precompute_rope` (host-side cos/sin table) | ops.rs:12 | (host) | host glue (keep) |

**ARLE already-vendored** (`crates/cuda-kernels/vendor/`): **FlashMLA**, **DeepGEMM**,
**tilekernels** (TileLang). **None of these is wired into the BF16 Qwen3.5 path above** —
they are DSv4-only today. **Marlin is vendored-as-source** under `csrc/gemm/marlin_*.cu`
(adopted upstream W4A8/W4A16 kernel, FFI `marlin_gemm_cuda` / `gemm_w4a8_marlin_cuda` /
`gemm_w4_fp8_marlin_cuda`) but is NOT wired into the clean Qwen3.5 path (it is reachable
only on the legacy quant-GEMM dispatch by `WeightFormat`).

---

## 2. SGLang Qwen3.5 integration posture — the exact wire-up to mirror

### 2a. Full attention (gated GQA)
`Qwen3_5AttentionDecoderLayer.self_attention` (qwen3_5.py:888) →
`forward_prepare_native` (849):
1. `qkv_proj` = `QKVParallelLinear` (linear.py — **quant-aware**, dispatches FP8/FP4/Marlin
   by `quant_config`); split `[q_size*2, kv_size, kv_size]`, chunk q/gate.
2. q/k norm = `GemmaRMSNorm` (`layernorm.py`); RoPE = `get_rope(...)` →
   `RotaryEmbedding` (`rotary_embedding/` — FlashInfer/sgl_kernel fused apply-rope op).
3. `self.attn = RadixAttention(...)` (radix_attention.py:54) →
   `get_attn_backend().forward(...)` (line 139). **Backend is FlashInfer / FA3 / Triton**
   selected by `--attention-backend` (`attention_registry.py`); on Hopper the default
   full-attn backend for hybrid GDN is FlashInfer or FA3. KV cache is the **paged RadixAttention
   pool** (token_to_kv_pool), NOT a contiguous per-slot recompute.
4. Output gate: `attn_output * torch.sigmoid(gate)` (line 914); `o_proj` =
   `RowParallelLinear` (quant-aware).

### 2b. Linear attention (GatedDeltaNet) — the FLA + causal_conv1d path
`Qwen3_5GatedDeltaNet.forward` (qwen3_5.py:460):
1. `_forward_input_proj` → `in_proj_qkvz` + `in_proj_ba`
   (`MergedColumnParallelLinear`, quant-aware), optional dual-stream overlap.
2. `fused_qkvzba_split_reshape_cat_contiguous` (FLA fused split/reshape, Triton) → mixed_qkv, z, b, a.
3. `core_attn_out = self.attn(forward_batch, mixed_qkv, a, b)` →
   `RadixLinearAttention.forward` (radix_linear_attention.py:78) → `get_attn_backend().forward`
   → **`GDNAttnBackend`** (`attention/linear/gdn_backend.py`). Inside:
   - **`causal_conv1d_update`** (decode) / **`causal_conv1d_fn`** (prefill) — from
     `sgl_kernel.mamba` (CUDA `causal_conv1d`) with a Triton fallback
     (`mamba/causal_conv1d_triton.py`). gdn_backend.py:13-39, 317, 416.
   - **`fused_gdn_gating`** (FLA, `fla/fused_gdn_gating.py`) — computes the gate `g`
     and `beta` from `a`,`b`,`A_log`,`dt_bias`. gdn_backend.py:5.
   - **gated-delta scan** via `GDNKernelDispatcher` (gdn_backend.py:60-150) selecting
     among **Triton** (`kernels/gdn_triton.py`), **CuTe DSL**
     (`kernels/gdn_cutedsl.py`, SM100+), **FlashInfer** (`kernels/gdn_flashinfer.py`).
     The Triton kernel calls **FLA `chunk_gated_delta_rule`** (prefill/extend,
     `fla/chunk.py`) and **FLA `fused_recurrent_gated_delta_rule_packed_decode`** /
     **`fused_sigmoid_gating_delta_rule_update`** (decode,
     `fla/fused_recurrent.py` + `fla/fused_sigmoid_gating_recurrent.py`).
     gdn_triton.py:9-13, 81, 147.
   - MTP/verify: FlashInfer GDN kernel `target_verify` (frozen-state) when supported,
     else Triton (gdn_backend.py:131-139) — matches
     [[../../memory/reference_frozen_kv_mtp_sparse_attention]].
4. `self.norm = RMSNormGated` (FLA `fla/layernorm_gated.py`) — gated output norm (gate = z).
5. `out_proj` = `RowParallelLinear` (quant-aware).

### 2c. MoE
`Qwen2MoeSparseMoeBlock` (qwen2_moe.py:208): `self.topk = TopK(top_k, renormalize)`
(`layers/moe/topk.py` — fused topk-sigmoid/softmax `sgl_kernel/csrc/moe/moe_topk_*`),
`FusedMoE` (`layers/moe/fused_moe_triton/layer.py` — **Triton fused MoE**, quant-aware:
BF16 / FP8 blockwise / FP4 / Marlin-MoE / w4afp8), optional shared-expert fusion.

### 2d. Norm / RoPE / sampling / elementwise
- Norm: `GemmaRMSNorm` / `RMSNormGated` — `layers/layernorm.py` (fused add+rmsnorm, sgl_kernel).
- RoPE: `rotary_embedding/` — FlashInfer/sgl_kernel `apply_rope` (fused, position-indexed).
- Sampling: `layers/sampler.py:28-42` — **FlashInfer** `top_k_top_p_sampling_from_probs` /
  `min_p_sampling_from_probs` + **sgl_kernel** `top_k_renorm_prob` / `top_p_renorm_prob`.
- Elementwise (silu_mul, add): `sgl_kernel/csrc/elementwise/` (fused activation).

---

## 3. Quantization — Qwen3.5 schemes and the official kernel SGLang uses

`quant_config` flows into every `*ParallelLinear` and `FusedMoE`. The Qwen3.5 model
splits a separate `attn_quant_config` / `linear_attn_quant_config` from MoE
`quant_config` (qwen3_5.py:553, 726) so attention can stay higher-precision while MoE
is quantized (the modelopt_fp4 case explicitly does this).

| scheme | SGLang quant module | official/OSS kernel (CUDA) | ARLE `WeightFormat` | ARLE kernel today |
|---|---|---|---|---|
| **BF16** (canonical CUDA Qwen3.5-4B / 30B) | `unquant.py` | cublas / FlashInfer / FA3 + FusedMoE-Triton | `DenseBf16` | **hand-rolled** `gemm_cuda` + `moe_bf16_grouped_gemm_*` |
| **FP8 (w8a8, blockwise E4M3)** | `fp8.py`, `fp8_kernel.py`, `w8a8_fp8.py` | **DeepGEMM** blockwise FP8 GEMM + `fp8_blockwise_moe_kernel` / cutlass FP8 | `Dsv4Fp8BlockScaled` | **vendored DeepGEMM** wired (DSv4 only); NOT on Qwen3.5 path |
| **FP4 / NVFP4 / MXFP4** | `modelopt_quant.py`, `mxfp4*.py`, `marlin_utils_fp4.py` | **FlashInfer/cutlass NVFP4**, `mxfp4_flashinfer_cutlass_moe`, mxfp4-Marlin-MoE | `Dsv4Fp4BlockScaled` | DSv4 FP4 GEMV (hand-rolled), no Qwen3.5 wire |
| **4-bit AWQ** | `awq/` | **Marlin** (awq-marlin) `sgl_kernel/csrc/gemm/awq_kernel.cu` + marlin | `MarlinW4A8` / `W4A16` | **Marlin vendored-as-source** (not on clean Qwen3.5 path) |
| **4-bit GPTQ** | `gptq/` | **Marlin** (gptq-marlin) `csrc/gemm/gptq` + `gptq_marlin_repack` | `W4A16` / `MarlinW4A8` | Marlin repack FFI exists; hand-rolled `w4a16_gemv` fallback |
| **W4A8 (qserve)** | `w4afp8.py`, `qoq.py` | **Marlin W4A8** + `qserve_w4a8_*_gemm.cu` | `MarlinW4A8` | **Marlin W4A8 production** (`gemm_w4a8_marlin_cuda`) — Tier-1 wins |
| **INT8 (w8a8)** | `w8a8_int8.py`, `int8_kernel.py` | cutlass/sgl `int8_gemm_kernel.cu` | `W8A16` | hand-rolled `w8a16_gemv` |
| **GGUF QnK (Metal/CPU origin)** | `gguf.py` | (llama.cpp-class) | `GgufQ3K/Q4K/Q5K/Q6K` | **hand-rolled** `q{3,4,5,6}k_gemv` |
| **TurboQuant KV (ARLE-only)** | — (no SGLang equiv) | — | `TurboQuant` | hand-rolled (KV-quant, KEEP — no drop-in) |

**接入姿势 gotchas (must mirror exactly, else silent garbage):**
- **FP8 blockwise:** weight is row-major FP8 E4M3 with **E8M0 block scales**; activation
  must be **per-token-group quantized** (`per_token_group_quant_8bit`) at the matching
  block_n (qwen3_5.py:304 reads `quant_config.weight_block_size`). DeepGEMM call shape
  (`m_grouped_fp8_gemm_nt_contiguous/masked`) is the DSv4-proven wire (moe.rs:1931).
- **NVFP4:** the Hadamard-rotate-before-quant convention (DSv4 precedent in the audit doc)
  — scale layout + rotation must match the checkpoint's packer.
- **Marlin:** weights must be **repacked into Marlin tile layout** (`gptq_marlin_repack`,
  FFI `gptq_marlin_repack_cuda`) and scales transposed to `[K/group_size, N]`; W4A8 needs
  dynamic INT8 activation quant (`quantize_bf16_rows_to_int8_cuda`). ARLE already has all
  three FFIs — this is the lowest-risk quant adoption for Qwen3.5.
- **MoE quant:** quant applies to the **grouped expert GEMM**, with topk-weight scaling
  and shared-expert fusion preserved. ARLE's `moe_bf16_grouped_gemm_*` are the swap points
  (qwen35.rs header lines 24-26 already names them); DSv4's `deepgemm_grouped_experts*`
  (moe.rs:1814+) is the proven FP8 grouped pattern to copy.
- **Separate attn/MoE quant configs:** mirror qwen3_5.py:553/726 — do not force one quant
  scheme across attention + MoE.

---

## 4. Per-operator adoption table (the deliverable)

Replacement source = official (FlashMLA/DeepGEMM/DSA) OR well-optimized OSS
(FlashInfer / FlashAttention-3 / FLA / causal_conv1d / Marlin / cutlass). Each row is a
**hypothesis** until wired + gated.

| ARLE hand-rolled op | csrc file | official/OSS replacement | vendored in ARLE? | SGLang integration ref | KEEP-glue? |
|---|---|---|---|---|---|
| **A2** softmax attn `nonpaged_prefill_attention` + **A1** prep (q/k norm+RoPE+KV write) | attention/nonpaged_prefill_attention.cu, prefill_attention_hd256.cu | **FlashInfer** paged prefill/decode OR **FlashAttention-3** (Hopper) | ❌ needs vendoring (FlashInfer not in vendor/; FlashMLA is MLA-only, wrong shape for GQA) | radix_attention.py:139 + flashinfer_backend.py / flashattention_backend.py; `attention_registry.py` | A3 gate + KV-write prep are glue (keep, adapt to paged layout) |
| **L1** `conv1d_prefill` / conv1d_decode (GDN depthwise causal conv) | misc/conv1d*.cu | **`causal_conv1d`** (causal-conv1d lib; SGLang `sgl_kernel.mamba`) + Triton fallback | ❌ needs vendoring | gdn_backend.py:13-39, 317, 416 | ring/state plumbing is glue |
| **L2** `gated_delta_rule_*` (GDN recurrent scan, prefill+decode) | misc/gated_delta_rule.cu, gdr_*.cu | **FLA** `chunk_gated_delta_rule` (prefill) + `fused_recurrent_gated_delta_rule_packed_decode` (decode) + `fused_gdn_gating` + `fused_sigmoid_gating_delta_rule_update` | ❌ needs vendoring (flash-linear-attention) | gdn_triton.py:9-13,81,147; fla/chunk.py, fla/fused_recurrent.py | per-slot state carry is glue; **chunkwise WGMMA hung on sm_90 → FLA chunk is the fix, not the recurrent fallback** (errors/2026-05-30) |
| **N3** `rms_norm_gated` (GDN gated output norm) | misc/norm.cu | **FLA `layernorm_gated`** RMSNormGated | ❌ needs vendoring (ships with FLA) | qwen3_5.py:529; fla/layernorm_gated.py | — |
| **M1/M2** `moe_bf16_grouped_gemm_*` (BF16 MoE grouped GEMM) | gemm/moe_grouped_gemm.cu | **FusedMoE-Triton** (BF16) / **DeepGEMM grouped FP8** (when quant) / cutlass-MoE | ✅ DeepGEMM vendored (FP8 path proven in DSv4) | fused_moe_triton/layer.py; moe.rs:1814 DSv4 pattern | EP routing scatter/scan (`dsv4_route`/`qwen36_route` scatter) is glue (C) |
| **M3** `qwen36_route` / `qwen36_renorm_topk_weights` (router topk compute) | moe/qwen36_route.cu | **sgl_kernel** `moe_topk_sigmoid/softmax_kernels` + `moe_align_kernel` | ❌ needs vendoring (sgl-kernel) | qwen2_moe.py:251 TopK; sgl-kernel/csrc/moe/moe_topk_*.cu | scatter/combine/shared-expert-gated-add is glue |
| **G1/G2** BF16 `gemm`/`gemv` (dense proj + lm_head) | gemm/gemv.cu | **cublasLt** (BF16 GEMM) OR FlashInfer fused proj; quant → **Marlin** | ⚠ Marlin vendored-as-source; cublas is system | linear.py `*ParallelLinear`; csrc/gemm/marlin | borderline — BF16 GEMM is a tuned primitive, license-or-kill vs cublasLt |
| **wNa16 GEMV** `w8a16/w4a16/w2a16_gemv*` (uniform-group quant GEMM) | gemm/quantized_gemv*.cu | **Marlin** (W4A16/awq/gptq-marlin) | ✅ Marlin vendored-as-source | linear.py awq/gptq marlin; csrc/gemm/marlin, awq_kernel.cu | — (clear DELETE candidate once Marlin wired on Qwen3.5) |
| **S** `argmax`/`gpu_sample` | misc/sampling.cu | **FlashInfer** `top_k_top_p_sampling_from_probs` + **sgl_kernel** renorm | ❌ needs vendoring | sampler.py:28-42 | low priority (correctness-trivial) |
| **N1/N2** offset RMSNorm | misc/norm.cu | FlashInfer/sgl_kernel `rmsnorm`/`fused_add_rmsnorm` | ❌ needs vendoring | layernorm.py GemmaRMSNorm | low priority (trivial, not a bottleneck) |
| **E / EL** embedding, add, silu_mul | misc/elementwise_basic.cu, fused_mlp.cu | FlashInfer silu_and_mul / sgl elementwise | ❌ | activation.py, elementwise | **KEEP** (trivial, low ROI — same verdict as DSv4 audit §C) |
| **R** RoPE precompute (host cos/sin) | host | match `get_rope` theta/YaRN; apply via FlashInfer rope | — | rotary_embedding/ | KEEP host glue; **wire FlashInfer apply-rope if A1 is replaced** |

### KEEP — genuine ARLE glue (no upstream drop-in)
- Paged-KV layout (`kv/*`), EP routing orchestration (`qwen36_route` scatter/scan/combine,
  `dsv4_route`), TP repack, dtype convert, per-slot recurrent/conv state carry. Same as DSv4 audit §C.
- TurboQuant KV-quant (`quant/turboquant*`) — QuaRot-family, no clean vendored KV-quant drop-in.

---

## 5. Biggest integration-posture risks (接入姿势)

1. **No FlashInfer / FLA / causal_conv1d / sgl-kernel vendored yet.** ARLE's `vendor/`
   has only FlashMLA (MLA — wrong shape for Qwen3.5 GQA) + DeepGEMM (GEMM only) +
   tilekernels. The two heaviest Qwen3.5 compute axes — **GDN (FLA + causal_conv1d)** and
   **full GQA attention (FlashInfer/FA3)** — have **NO vendored equivalent**. This is a
   vendoring task first, a wiring task second. DeepGEMM (FP8 MoE) is the only quant GEMM
   already vendored + proven.
2. **Paged vs contiguous KV mismatch.** SGLang full attention is **RadixAttention paged**;
   the clean ARLE Qwen3.5 path is **uncached contiguous full-prefix recompute** (qwen35.rs:14-18).
   Adopting FlashInfer/FA3 requires moving Qwen3.5 onto the paged KV pool — a scheduler-level
   change, not a kernel swap. License-or-kill on the **SLO prompt length**, not a smoke shape
   ([[../../memory/...slo-prefill-kill]]); contiguous recompute is O(L²) so the paged win grows with L.
3. **GDN state-carry + MTP verify.** FLA chunk/recurrent kernels must preserve ARLE's
   per-slot recurrent state + conv ring semantics across prefill→decode and **freeze state
   on spec-decode verify** ([[../../memory/reference_frozen_kv_mtp_sparse_attention]],
   [[../../memory/feedback_spec_decode_gate_needs_multi_prompt]]). The chunkwise WGMMA path
   hung on sm_90 (errors/2026-05-30) — FLA's chunk kernel is the *intended* fix, but verify
   on Hopper before trusting it.
4. **Quant scale/layout/rotation contract.** FP8 needs E8M0 block scales + per-token-group
   activation quant; NVFP4 needs the Hadamard-rotate-before-quant convention (DSv4 precedent);
   Marlin needs tile-repack + transposed scales. Mismatch = silent garbage, not a crash.
   Smoke-garbage is **config-suspect first** ([[../../memory/...native-deepep-pod-e2e]]).
5. **BF16 dense GEMM may not be a DELETE.** cublasLt is the obvious BF16 replacement, but a
   hand-rolled GEMM can win on small-M decode shapes — license-or-kill per shape
   ([[../../memory/feedback_b1_decode_gpu_bound_overhead_removal_wash]]), don't bulk-delete.
6. **Correctness gate, not byte-identity.** MoE run-to-run non-determinism confounds
   token-exact A/B ([[../../memory/feedback_correct_inference_not_baseline_identity]],
   [[../../memory/reference_dsv4_moe_nondeterminism_confounds_4096_parity]]). Gate adoption on
   needle retrieval + same-config-twice floor + matched A/B, NEVER vs-baseline byte identity.

---

## 6. Sequencing (proposed; each step = wire official → delete hand-roll → gate)

Lowest-risk-highest-confidence first (all hypotheses until gated):

1. **Marlin W4A16/W4A8 on Qwen3.5 dense proj** — Marlin is already vendored-as-source +
   all repack/quant FFIs exist + production Tier-1 wins on the legacy path. Wire it into the
   clean Qwen3.5 GEMM dispatch; delete `wNa16_gemv`. Lowest vendoring cost.
2. **DeepGEMM FP8 MoE grouped GEMM** — DeepGEMM vendored + the DSv4 `deepgemm_grouped_experts*`
   pattern (moe.rs:1814) is proven. Swap `moe_bf16_grouped_gemm_*` for the FP8 path at the
   two flagged call sites (qwen35.rs:24-26). Keep BF16 as default until A/B clears.
3. **FLA GDN (chunk + recurrent + gating) + causal_conv1d** — vendor flash-linear-attention
   + causal-conv1d; replace L1/L2/N3. Highest compute share on the linear-attn-majority model;
   also the sm_90 chunkwise-hang fix. Freeze-state verify for spec-decode.
4. **FlashInfer / FA3 full GQA attention** — vendor FlashInfer; move Qwen3.5 to paged KV;
   replace A1/A2/A3. Largest scheduler-level change; license on SLO prompt length.
5. **FlashInfer/sgl-kernel sampling + norm** — low ROI, do last (or KEEP as glue).

Verify each official call-shape before deleting (DELETE rows are hypotheses until the
wire-up + needle/same-twice/A-B passes). Glue (§4 KEEP) stays.
