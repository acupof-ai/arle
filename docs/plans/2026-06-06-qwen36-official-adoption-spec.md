# Qwen3.6 (35B-A3B MoE) — official/OSS kernel adoption spec (mirror SGLang)

**Date:** 2026-06-06. **Read-only study.** Companion to the DSv4 audit
[`2026-06-06-dsv4-handrolled-kernel-audit.md`](2026-06-06-dsv4-handrolled-kernel-audit.md).

**Driver (ckl):** "用官方的或者开源优化好的替换" — replace ARLE's hand-rolled CUDA
compute operators with official/OSS kernels (FlashInfer / FlashAttention / Marlin /
DeepGEMM / DeepEP / cutlass), mirror SGLang's integration **pixel-level**, do NOT
hand-roll. **Bar:** zero self-developed compute operators UNLESS proven better
(same-binary A/B) than the best OSS. Per-operator license-or-kill, not bulk-delete.
([[../../memory/feedback_no_closed_door_solutions]])

**Quant must be factored in.** The shipped checkpoint is quantized; the expert GEMM
quant kernel is the single highest-value adoption.

---

## 0. CRITICAL — what Qwen3.6-35B-A3B actually is (verified, not assumed)

`config.json` (`mlx-community/Qwen3.6-35B-A3B-4bit`):
- `architectures: ["Qwen3_5MoeForConditionalGeneration"]`, `model_type: qwen3_5_moe`.
- **It is the Qwen3.5 HYBRID family, NOT the standard `qwen3_moe` GQA model.**
- 40 layers = **30 `linear_attention` (gated-delta) + 10 `full_attention`** (`layer_types`).
- Full attn: 16 q heads / 2 kv heads / **head_dim 256**, `attn_output_gate=True` (gated q_proj).
- Linear attn (GatedDeltaNet): `linear_key_head_dim=128`, `linear_value_head_dim=128`,
  `linear_conv_kernel_dim=4` (causal conv1d front-end + gated output RMSNorm).
- MoE: `num_experts=256`, `num_experts_per_tok=8`, `moe_intermediate_size=512`,
  `shared_expert_intermediate_size=512`, `norm_topk_prob=True`, **plain softmax router**
  (no bias, no expert-group), 1 shared expert + scalar `shared_expert_gate` sigmoid.
- hidden 2048, vocab 248320, rope_theta from config (vanilla RoPE, `rope_scaling=None`).
- **Quant: MLX 4-bit affine, group_size 64** (Metal). The router `gate` +
  `shared_expert_gate` are kept at **8-bit**. CUDA-served checkpoints would instead be
  **FP8 block-scaled** or **W4A8/Marlin** (HF safetensors), NOT MLX affine.

**SGLang reference model file = `qwen3_5.py`, NOT `qwen3_moe.py`.** The MoE block is
`Qwen2MoeSparseMoeBlock` (`from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock`,
`qwen3_5.py:78`). The standard `qwen3_moe.py` / `Qwen3MoeSparseMoeBlock` is the dense-GQA
Qwen3-MoE (235B/30B) and does **not** apply here.

**ARLE entry point:** Qwen3.6 shares ARLE's `qwen35.rs` hybrid executor
(`crates/infer-cuda/src/qwen35.rs`, `Qwen35CudaExecutor`,
`from_qwen35_moe_safetensors`). MoE forward = `crates/infer-cuda/src/moe.rs`
`gpu::moe_forward` (single-GPU, **all experts local, no EP/DeepEP** — DeepEP is
DSv4-only). Config bridge: `crates/infer-cuda/src/moe_config.rs`
(`MoeConfig::qwen36`), spec `crates/qwen35-spec/src/lib.rs`.

---

## 1. ARLE Qwen3.6 compute operators — hand-rolled vs vendored

Enumerated from the actual call sites (`qwen35.rs` forward → `moe.rs` →
`cuda-kernels/{attention,moe,gemm}.rs` FFI → `csrc/`).

### 1a. MoE path (`moe.rs::gpu::moe_forward`) — every op HAND-ROLLED

Pipeline (BF16, token-major, all experts local):
```
router gemm (ops::gemm_batch)                                    HAND-ROLLED (generic gemm)
  → D2H logits → infer_moe::route (HOST softmax+top-k+norm)      HOST (Rust, no kernel)
  → flatten_routing (HOST) → route_indices/route_weights         HOST
  → dsv4_count_local_experts  → counts[E]                        HAND-ROLLED (csrc/moe/dsv4_route.cu)
  → dsv4_exclusive_scan_i32   → offsets[E]                       HAND-ROLLED (dsv4_route.cu)
  → dsv4_pack_local_experts_with_slots → packed grouped buffers  HAND-ROLLED (dsv4_route.cu) [GLUE]
  → moe_bf16_grouped_gemm_pair_batch (gate+up)                   HAND-ROLLED (csrc/gemm/moe_grouped_gemm.cu)  ★hottest
  → silu_mul (unclamped SwiGLU)                                  HAND-ROLLED (ops, elementwise)
  → moe_bf16_grouped_gemm_batch (down)                           HAND-ROLLED (moe_grouped_gemm.cu)            ★hottest
  → dsv4_scatter_all_route_slots                                 HAND-ROLLED (dsv4_route.cu) [GLUE]
  → dsv4_combine_route_slot_outputs (sum over topk)              HAND-ROLLED (dsv4_route.cu) [GLUE]
  → shared expert dense SwiGLU (gemm_batch ×3 + silu_mul)        HAND-ROLLED
  → qwen36_add_shared_expert_gated (sigmoid gate + accum)        HAND-ROLLED (csrc/moe/qwen36_route.cu)
  [+ qwen36_renorm_topk_weights when norm_topk_prob — currently HOST-renormed in infer_moe::route]
```
`moe_grouped_gemm.cu` self-documents: "no tensor-core / mma => sm_70-safe ... CUDA-core
warp-reduce ... W4 nibble-decode variant is an explicit follow-up". **It is a
correctness-first BF16 GEMM with no tensor cores and no quant decode** — the prime
adoption target.

### 1b. Full-attention layers (10×, `qwen35.rs::full_attention`) — HAND-ROLLED HD256

```
prefill_attention_hd256_prep_cuda      (fused qk_norm + RoPE + KV-cache write)  HAND-ROLLED (csrc/attention/prefill_attention_hd256.cu)
nonpaged_prefill_attention_cuda        (causal attn over contiguous cache)      HAND-ROLLED (csrc/attention/nonpaged_prefill_attention.cu)
attention_gate_batch_hd256_cuda        (decode attn + per-head sigmoid gate)    HAND-ROLLED (csrc/attention/*hd256*)
```
HD256 + 16q/2kv GQA. Contiguous per-slot KV cache (NOT the TileLang paged HD128/kv8 path
used by dense Qwen3 — that path is `attention.rs::run_tilelang_paged`, a different model).

### 1c. Linear-attention layers (30×, `qwen35.rs::linear_attention`) — HAND-ROLLED GDN

```
conv1d_prefill_cuda                    (causal depthwise conv1d, kernel=4)      HAND-ROLLED (csrc/misc/conv1d*.cu)
gated_delta_rule_decode_cuda /         (gated-delta recurrent state update)     HAND-ROLLED (csrc/misc/gated_delta_rule*.cu)
  gated_delta_rule_prefill_recurrent_cuda                                       (chunkwise WGMMA path exists but disabled — short-seq hang)
rms_norm_gated_cuda                    (gated output RMSNorm, gate=z)           HAND-ROLLED (csrc/misc/norm.cu)
```

### 1d. Shared / dense ops

```
rms_norm_offset / rms_norm_offset_vec  (input/post-attn/final RMSNorm)         HAND-ROLLED (csrc/misc/norm.cu)
ops::gemm_batch                        (router/o_proj/qkv/lm_head/shared GEMMs) HAND-ROLLED (csrc/gemm/gemv.cu + generic)
sample_cuda_token                      (sampling)                              HAND-ROLLED (csrc/misc/sampling.cu)
precompute_rope                        (RoPE table)                            HAND-ROLLED (host)
```

**Summary: the Qwen3.6 path is ~100% hand-rolled** — none of FlashMLA/DeepGEMM/Marlin
(all vendored for DSv4) are on the Qwen3.6 hot path today. The MoE expert GEMM and the
HD256 full attention are the two highest-value targets.

---

## 2. SGLang integration — exact sequence, dtypes, layouts

How SGLang runs Qwen3.6 (`qwen3_5.py` + `qwen2_moe.py` + `layers/`):

### 2a. MoE block (`Qwen2MoeSparseMoeBlock`, `qwen2_moe.py:208-329`)
Construction:
- `self.gate = ReplicatedLinear(hidden, num_experts, bias=False)` (`:279`) — router GEMM,
  kept in higher precision (`quant_config=None`).
- `self.topk = TopK(top_k=num_experts_per_tok, renormalize=norm_topk_prob, layer_id)` (`:251`).
- `self.experts = get_moe_impl_class(quant_config)(...)` (`:257`) →
  **`FusedMoE`** (single-GPU / TP) or **`DeepEPMoE`** (when a2a backend is DeepEP/mori/...),
  `ep_moe/layer.py:269`. `routing_method_type=RenormalizeNaive`, `intermediate=moe_intermediate_size`.
- `self.shared_expert = Qwen2MoeMLP(..., reduce_results=False)` (`:293`) +
  `self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)` (`:320`).

Forward (`qwen2_moe.py:~408-470`):
```
router_logits, _ = self.gate(hidden_states)            # [T, E]
shared_output = self._forward_shared_experts(hidden)   # sigmoid(gate(x)) * shared_mlp(x)  (qwen2_moe.py:43-62)
topk_output  = self.topk(hidden_states, router_logits) # → topk_weights[T,k] fp32, topk_ids[T,k] int32
final = self.experts(hidden_states, topk_output)       # FusedMoE grouped expert GEMM
return final + shared_output                            # (+ TP all_reduce)
```

### 2b. TopK kernel (`layers/moe/topk.py`)
- `TopK.forward_cuda` → `select_experts` → `fused_topk` (`topk.py:668`) for `scoring_func="softmax"`:
  ```python
  topk_softmax(topk_weights, topk_ids, gating_output, renormalize)   # topk.py:698
  ```
- **`topk_softmax` = sgl-kernel CUDA op**, registered `common_extension.cc:171`,
  impl `sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu` (`topk_softmax` :722,
  `topkGatingSoftmax` :341, adapted from vLLM). Fused softmax + top-k + optional renorm
  in one kernel; output **`topk_weights` fp32, `topk_ids` int32, both [T, k]**.

### 2c. Expert grouped GEMM (`FusedMoE` → MoE runner)
The grouped GEMM backend is selected by `get_moe_runner_backend()` + quant_config:
- **DeepGEMM** (`layers/moe/moe_runner/deep_gemm.py`): the canonical block-scaled path.
  - FP8: `deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig` (`:206,286`) /
    `grouped_gemm_nt_f8f8bf16_masked` (`:406,478`).
  - BF16: `grouped_gemm_nt_bf16_contig` (`:323,353`) / `grouped_gemm_nt_bf16_masked` (`:522,551`).
  - Layout: **grouped-contiguous** (sort tokens by expert) or **masked** (fixed M per
    expert, masked_m). UE8M0 block scales, TMA-aligned (`DEEPGEMM_SCALE_UE8M0`,
    `DEEPGEMM_NEED_TMA_ALIGNED_SCALES`). SwiGLU fused into the GEMM epilogue
    (`_varlen_deep_gemm_silu_mul_quant`, `:443`).
- **Triton fused_moe** (`layers/moe/fused_moe_triton/fused_moe.py`): the BF16 / generic
  default when DeepGEMM isn't enabled — `moe_align_block_size` + `invoke_fused_moe_kernel`.
  `moe_align_block_size` = sgl-kernel (`csrc/moe/moe_align_kernel.cu`, `common_extension.cc:165`)
  sorts/pads token-ids into block-aligned per-expert groups for the triton grouped GEMM.
- **cutlass_moe** (`csrc/moe/cutlass_moe/`, `fp8_blockwise_scaled_grouped_mm`
  `common_extension.cc:199`; W4A8 `cutlass_w4a8_moe_mm` `:252`) — FP8/W4A8 grouped GEMM.
- **flashinfer** (trtllm / cutlass / mxfp4) when `get_moe_runner_backend().is_flashinfer_*`.
- The reduce/combine: **`moe_sum_reduce`** sgl-kernel (`csrc/moe/moe_sum_reduce.cu`,
  `common_extension.cc:180`) sums the topk expert outputs per token (× routed_scaling).

### 2d. EP dispatch / combine (only when EP > 1)
`DeepEPMoE` (`ep_moe/layer.py`) wraps **DeepEP** `dispatch` / `combine` (intranode/internode/
low-latency). **For Qwen3.6 at single-GPU / pure-TP this is NOT used** — all experts are
local, mirroring ARLE's all-experts-local `moe.rs::gpu::moe_forward`. EP is the multi-node
follow-up (the DSv4 `deepep.rs` precedent applies if/when Qwen3.6 goes EP).

### 2e. Full attention (`Qwen3_5AttentionDecoderLayer`, `qwen3_5.py:671-840`)
```
qkv = self.qkv_proj(x)                                   # gated: total_heads*(1+attn_output_gate)
q, k = fused_qk_gemma_rmsnorm(q, k, q_norm, k_norm)      # models/utils.py:532  (Triton fused qk-RMSNorm)
q, k = self.rotary_emb(positions, q, k)                  # get_rope() → sgl-kernel rotary  (qwen3_5.py:715)
attn_out = self.attn(q, k, v, forward_batch)            # RadixAttention → FlashInfer / FA3 backend
out, _ = self.o_proj(attn_out * sigmoid(gate))          # attn_output_gate applied to attn output
```
- `self.attn = RadixAttention(num_heads, head_dim=256, scaling, num_kv_heads=2)` (`:755`).
- Backend = **FlashInfer** (`layers/attention/flashinfer_backend.py`, server default) or
  **FlashAttention-3** (`flashattention_backend.py`) — paged KV, GQA, head_dim 256.
- `get_rope()` (`layers/rotary_embedding.py`) → sgl-kernel / FlashInfer rotary kernel.

### 2f. Linear attention (`Qwen3_5GatedDeltaNet`, `qwen3_5.py:118-535`)
```
mixed_qkvz / mixed_ba  via gdn_fused_proj (Triton)      # jit_kernel/triton/gdn_fused_proj
conv1d front-end  → causal_conv1d_fn / causal_conv1d_update   # layers/attention/mamba/causal_conv1d.py (CUDA) + _triton fallback
core_attn_out = self.attn(...)  via RadixLinearAttention → get_attn_backend()  # GDN/Mamba2 backend
  → FLA: chunk_gated_delta_rule (prefill) / fused_recurrent_gated_delta_rule (decode)
self.norm(core_attn_out, z)  = RMSNormGated                  # layers/attention/fla/layernorm_gated.py
out = self.out_proj(...)
```
- FLA = **flash-linear-attention** library (`layers/attention/fla/`: `chunk.py`,
  `fused_recurrent.py`, `fused_gdn_gating.py`, `layernorm_gated.py`, `l2norm.py`, ...).
- conv1d = **causal_conv1d** (the Mamba CUDA kernel, `layers/attention/mamba/causal_conv1d.py`).

### 2g. Norm / sampling
- RMSNorm = `GemmaRMSNorm` (`layers/layernorm.py`) → sgl-kernel / FlashInfer fused
  `rmsnorm` / `fused_add_rmsnorm`.
- Sampling = `layers/sampler.py` → FlashInfer sampling kernels (top-k/top-p).

---

## 3. Quant — Qwen3.6 schemes and the OSS kernel per scheme

| scheme | where served | SGLang grouped-GEMM kernel | scale layout | ARLE status |
|---|---|---|---|---|
| **BF16** (unquantized) | dev / parity | DeepGEMM `grouped_gemm_nt_bf16_{contig,masked}` (or triton fused_moe) | none | hand-rolled `moe_bf16_grouped_gemm_*` ★ |
| **FP8 block-scaled** (E4M3 + UE8M0 blocks) | CUDA prod default | **DeepGEMM** `grouped_gemm_nt_f8f8bf16_{contig,masked}` (`deep_gemm.py:206/406`) | **UE8M0, 128-block, TMA-aligned** (`DEEPGEMM_SCALE_UE8M0`) | DeepGEMM **vendored** (DSv4 already wires `dsv4_deepgemm_m_grouped_fp8_gemm_nt_{masked,contiguous}`) — reuse, don't rewrite |
| **W4A16 / W4A8 (Marlin/GPTQ-AWQ)** | 4-bit weight CUDA | **Marlin** `gptq_marlin` / cutlass `cutlass_w4a8_moe_mm` (`common_extension.cc:252`) | per-group int4 + 8-bit/16-bit scales, group 128 | Marlin **vendored** (in-tree Apache, DSv4 `marlin_w4a8_kernel.cu`/`marlin_w4_fp8_kernel.cu` + repack) — reuse |
| **MLX 4-bit affine, group 64** (router/gate 8-bit) | **Metal only** | n/a (Metal/MLX path) | affine group-64; gate & shared_expert_gate 8-bit | Metal `mlx-sys` path; out of CUDA scope |
| **FP4 / mxfp4** | newer HW | flashinfer mxfp4 moe | mxfp4 block | not in scope |

**Quant gotcha (critical):** the MLX-4bit checkpoint is the *Metal* artifact. A CUDA
Qwen3.6 serve needs an **FP8-block-scaled or Marlin-W4 safetensors** checkpoint to use the
DeepGEMM/Marlin grouped GEMM — you cannot feed MLX affine-int4 to DeepGEMM. The expert GEMM
quant (FP8 block-scaled via DeepGEMM) is the single highest-value adoption: it replaces
both `moe_bf16_grouped_gemm_*` (no tensor cores) AND adds quant in one move, reusing the
**already-vendored, already-DSv4-wired** DeepGEMM bridge.

---

## 4. Per-operator adoption table

`✅`=vendored in-tree, `⚠`=needs vendoring, `[GLUE]`=keep (irreducible orchestration).

| # | hand-rolled ARLE op | file | official/OSS replacement | vendored? | SGLang ref (file:line) | KEEP-glue if no drop-in |
|---|---|---|---|---|---|---|
| **MoE** ||||||| 
| M1 ★ | `moe_bf16_grouped_gemm_pair_batch` (gate+up) | `csrc/gemm/moe_grouped_gemm.cu` | **DeepGEMM** `grouped_gemm_nt_{f8f8bf16,bf16}_{contig,masked}` | ✅ deepgemm | `moe_runner/deep_gemm.py:206,286,323` | — (DSv4 already wires `dsv4_deepgemm_m_grouped_fp8_gemm_nt_*`) |
| M2 ★ | `moe_bf16_grouped_gemm_batch` (down) | `csrc/gemm/moe_grouped_gemm.cu` | same DeepGEMM grouped GEMM (down proj) | ✅ deepgemm | `deep_gemm.py:478,522,551` | — |
| M3 | host `infer_moe::route` + `flatten_routing` softmax/top-k | `infer-moe/route.rs`, `moe.rs` | **sgl-kernel `topk_softmax`** (fused softmax+top-k+renorm, on-device) | ⚠ sgl-kernel | `topk.py:698`, `csrc/moe/moe_topk_softmax_kernels.cu:722` | host route is correctness-anchor; keep as fallback |
| M4 | `qwen36_renorm_topk_weights` | `csrc/moe/qwen36_route.cu` | folded into `topk_softmax(renormalize=True)` | ⚠ sgl-kernel | `topk.py:251` (`renormalize=norm_topk_prob`) | delete once topk_softmax adopted |
| M5 | `dsv4_count/scan/pack/scatter/combine` | `csrc/moe/dsv4_route.cu` | **`moe_align_block_size`** (triton path) / **`moe_sum_reduce`** (combine) | ⚠ sgl-kernel | `csrc/moe/moe_align_kernel.cu`, `moe_sum_reduce.cu` | **[GLUE]** mostly keep (DeepGEMM-grouped path needs ARLE's own pack/offset; combine→`moe_sum_reduce` is a clean swap) |
| M6 | `qwen36_add_shared_expert_gated` | `csrc/moe/qwen36_route.cu` | trivial sigmoid-gate+accum; SGLang does it in Python (`qwen2_moe.py:43`) | n/a | `qwen2_moe.py:43-62` | **[GLUE]** keep (tiny elementwise, no upstream drop-in worth it) |
| M7 | shared-expert dense SwiGLU (3 GEMMs) | `moe.rs` | quantized dense GEMM (Marlin/DeepGEMM) if shared expert is quantized | ✅ | `qwen2_moe.py:293` (`Qwen2MoeMLP`) | dtype-follows the expert GEMM adoption |
| **Full attention (HD256)** ||||||| 
| A1 ★ | `nonpaged_prefill_attention` + `attention_gate_batch_hd256` | `csrc/attention/*hd256*.cu` | **FlashInfer** paged GQA attn (head_dim 256) or **FA3** | ⚠ flashinfer/fa3 | `qwen3_5.py:755` (`RadixAttention`), `flashinfer_backend.py` | needs vendoring; HD256+gated-output is the integration risk |
| A2 | `prefill_attention_hd256_prep` (fused qk_norm+RoPE+KV write) | `csrc/attention/prefill_attention_hd256.cu` | **sgl-kernel `fused_qk_gemma_rmsnorm`** + `get_rope` rotary + FlashInfer KV append | ⚠ | `qwen3_5.py:832` (`fused_qk_gemma_rmsnorm`), `models/utils.py:532` | KV-cache write is paged-layout glue (keep) |
| A3 | output sigmoid gate (attn_output_gate) | `*hd256*.cu` | SGLang does `attn_out * sigmoid(gate)` in Python | n/a | `qwen3_5.py` o_proj path | **[GLUE]** keep (tiny) |
| **Linear attention (GatedDeltaNet)** ||||||| 
| L1 | `gated_delta_rule_{decode,prefill_recurrent}` | `csrc/misc/gated_delta_rule*.cu` | **FLA** `chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule` | ⚠ FLA | `qwen3_5.py:511`→`RadixLinearAttention`→GDN backend; `layers/attention/fla/{chunk,fused_recurrent}.py` | needs vendoring (Triton lib — porting risk) |
| L2 | `conv1d_prefill` | `csrc/misc/conv1d*.cu` | **causal_conv1d** (Mamba CUDA kernel) | ⚠ causal_conv1d | `layers/attention/mamba/causal_conv1d.py` | needs vendoring |
| L3 | `rms_norm_gated` | `csrc/misc/norm.cu` | **FLA** `RMSNormGated` (`layernorm_gated.py`) / FlashInfer gated rmsnorm | ⚠ | `qwen3_5.py:42` (`RMSNormGated`) | low priority |
| **Dense / shared** ||||||| 
| D1 | `ops::gemm_batch` (router/o_proj/qkv/lm_head) | `csrc/gemm/gemv.cu` | cutlass / cuBLAS / Marlin (if quantized) | ✅/⚠ | `linear.py` (ReplicatedLinear/RowParallel) | quantized → Marlin; BF16 dense → cublas/cutlass |
| D2 | `rms_norm_offset` / `fused_add_rms_norm_offset` | `csrc/misc/norm.cu` | FlashInfer `rmsnorm` / `fused_add_rmsnorm` | ⚠ flashinfer | `layers/layernorm.py` (`GemmaRMSNorm`) | low ROI (not a bottleneck) |
| D3 | `sample_cuda_token` | `csrc/misc/sampling.cu` | FlashInfer sampling | ⚠ flashinfer | `layers/sampler.py` | low ROI |
| D4 | `precompute_rope` + RoPE apply | host + `*hd256*.cu` | `get_rope` / FlashInfer rotary | ⚠ | `qwen3_5.py:715` | folded into A2 |

★ = top-value (hot path + duplicates a vendored/OSS kernel).

---

## 5. Integration-posture gotchas (the things that silently break)

1. **MoE topk output format.** sgl-kernel `topk_softmax` writes `topk_weights` **fp32** and
   `topk_ids` **int32**, shape `[T, k]`, **renorm folded in** (`renormalize` arg). ARLE's
   host route produces the same logical content but the device kernel's id dtype/layout and
   the renorm-vs-raw-softmax choice must match `norm_topk_prob=True`. ARLE currently
   renorms host-side; M4's `qwen36_renorm_topk_weights` becomes dead once `topk_softmax`
   owns renorm.

2. **Expert-GEMM layout: grouped-contiguous vs masked.** DeepGEMM needs either (a) tokens
   **sorted contiguous by expert** with per-expert `m_indices`/offsets (`*_contig`), or
   (b) **masked** fixed-M-per-expert with `masked_m` (`*_masked`). ARLE's
   `dsv4_pack_local_experts_with_slots` already produces a grouped-contiguous packing —
   that **glue is reusable** to feed DeepGEMM's contiguous variant; do NOT delete the
   pack/count/scan (M5 stays). The DSv4 FP8 path (`moe.rs::dsv4_gpu`) already does exactly
   this wire-up — copy its pattern.

3. **FP8 block-scale layout = UE8M0, 128-block, TMA-aligned.** DeepGEMM is picky:
   `DEEPGEMM_SCALE_UE8M0` (1×128 block scales in UE8M0) + `get_mn_major_tma_aligned_tensor`.
   Feeding per-tensor or row-major scales silently produces garbage. Quantizer must emit the
   UE8M0 block layout — ARLE's `dsv4_deepgemm_pack_quantize_bf16_to_fp8` already does this
   (reuse).

4. **head_dim 256 attention coverage.** FlashInfer/FA3 must be built with head_dim 256
   support (Qwen3.6 full attn) — this is non-default in many builds. The DSv4 audit flagged
   the Qwen quant-attention track as "needs vendoring" (FlashInfer not yet in-tree). The
   **gated output (`attn_output_gate`)** is Qwen3.5-specific: the upstream attn kernel
   returns plain attn output and SGLang applies `* sigmoid(gate)` outside — keep ARLE's gate
   as post-kernel glue (A3), don't expect it fused in FlashInfer.

5. **Linear-attention is the hardest adopt.** GatedDeltaNet uses FLA (Triton) +
   causal_conv1d (CUDA). Neither is vendored. FLA is a Triton library — adopting it means
   either vendoring the Triton kernels (new build dependency, no precedent in-tree) or
   keeping ARLE's hand-rolled recurrent path. **Recommendation: defer L1-L3** (30 of 40
   layers, but the recurrent kernels are memory-bound and ARLE's already work); prioritize
   M1/M2 (expert GEMM) and A1 (full attn) first.

6. **EP dispatch/combine contract — not needed at single-GPU.** Qwen3.6 ARLE is
   all-experts-local; DeepEP `dispatch`/`combine` (the `feedback_deepep_kernel_api_inverted_naming`
   / `recv_channel_prefix` gotchas) only apply if Qwen3.6 goes multi-node EP. The DSv4
   `deepep.rs` + `dsv4_moe_forward_deepep` is the template if/when it does.

7. **Quant checkpoint mismatch.** The canonical Metal model is MLX-4bit-affine. A CUDA
   DeepGEMM/Marlin adoption requires re-sourcing an **FP8 or Marlin-W4 safetensors**
   checkpoint — the MLX affine weights are not consumable by DeepGEMM/Marlin. Confirm the
   CUDA serving checkpoint before wiring (else you adopt a kernel you can't feed).

8. **Bench gate (per CLAUDE.md §Benchmarks + DSv4 audit).** Each adoption is
   license-or-kill on a **same-binary A/B** with the correctness gate = needle retrieval +
   same-config-twice non-determinism floor (NOT token-exact-vs-BF16 — confounded by MoE
   run-to-run non-determinism, per `feedback_correct_inference_not_baseline_identity` /
   `reference_dsv4_moe_nondeterminism_confounds_4096_parity`). Adopt OSS + delete the
   hand-roll only when OSS ≥ hand-roll with evidence.

---

## 6. Recommended sequence (mirror the DSv4 audit cadence)

1. **M1+M2 (expert GEMM → DeepGEMM)** — highest value, kernel already vendored + DSv4-wired,
   reuses ARLE's pack/offset glue. Adds quant (FP8 block-scaled) for free. Gate: needle +
   same-twice + A/B vs the BF16 hand-roll. *Prereq: FP8 CUDA checkpoint.*
2. **M3+M4 (router topk → sgl-kernel `topk_softmax`)** — removes the D2H host round-trip;
   small, deletes `qwen36_renorm_topk_weights`. *Prereq: vendor sgl-kernel topk.*
3. **A1+A2 (HD256 full attn → FlashInfer/FA3)** — needs vendoring FlashInfer with head_dim
   256; keep gated-output (A3) + KV-prep glue. Re-judge on the official kernel, not the hand-roll.
4. **M5 combine → `moe_sum_reduce`** — clean swap; keep pack/count/scan as glue.
5. **L1-L3 (GatedDeltaNet → FLA + causal_conv1d)** — deferred (Triton vendoring risk; 30
   layers but memory-bound recurrent, hand-roll works).
6. **D1-D4 (dense GEMM / norm / sampling / rope)** — low ROI; adopt opportunistically when
   FlashInfer is already in-tree from step 3.

Each step: verify the official call-shape **before** deleting the hand-roll (DELETE rows are
hypotheses until the wire-up + needle/A-B passes — §0). Glue (M5 pack, M6, A3, KV-prep) stays.
