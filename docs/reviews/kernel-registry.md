# Kernel Registry — live CUDA operator index (`crates/cuda-kernels`)

> **Source of truth = code, not docs.** Rebuilt 2026-06-05 by scanning
> `crates/cuda-kernels/csrc/{attention,gemm,kv,kvcacheio,moe,quant,misc}/`,
> `crates/cuda-kernels/src/ffi/`, and the **live callers in `crates/infer-cuda/src/`**.
> The prior version referenced pre-rewrite files (`batch_decode.rs`,
> `prefill.rs`, `forward.rs`, `linear.rs`) that exist only under
> `.claude/worktrees/` — those are gone from the live tree.
>
> **Live = a real Rust call expression** reaches the FFI symbol, either
> directly from `infer-cuda` (`ffi::sym(...)`) or via a `cuda-kernels/src`
> wrapper fn that `infer-cuda` invokes. Doc-comment mentions and bare `use`
> imports do **not** count. Symbols with zero live caller are listed in
> §Library-present-but-unwired, not as live rows.

## Column contract

| Column | Meaning |
|---|---|
| **family** | logical operator group (= csrc dir, except where misplaced) |
| **csrc file** | `.cu` that defines the `extern "C"` kernel (relative to `csrc/`) |
| **FFI symbol** | the `extern "C"` entry declared in `src/ffi/*.rs` |
| **Rust caller** | live call site as `file::fn` (in `infer-cuda/src` unless noted `[ck]` = `cuda-kernels/src`) |
| **op shape** | M/N/K for GEMM; seq/head/window dims for attention; routing dims for MoE |
| **precision** | bf16 / fp8-e4m3 / fp4-e2m1 / int8 / int4 / mixed |

---

## attention/ — paged attention prep + DSv4 sparse/MLA + quantized decode

| family | csrc file | FFI symbol | Rust caller | op shape | precision |
|---|---|---|---|---|---|
| attention | `attention/prefill_attention_paged_prep.cu` | `prefill_attention_paged_prep_cuda` | `attention.rs::prefill_attention` | paged prefill prep, hd128 | bf16 |
| attention | `attention/decode_prep_paged.cu` | `decode_prep_paged_cuda` | `attention.rs::decode_attention` | paged decode prep, hd128 | bf16 |
| attention | `attention/nonpaged_prefill_attention.cu` | `nonpaged_prefill_attention_cuda` | `qwen35.rs::full_attention` | non-paged prefill, Qwen3.5 full-attn layers | bf16 |
| attention | `attention/prefill_attention_hd256.cu` | `prefill_attention_hd256_prep_cuda` | `qwen35.rs::full_attention` | prefill prep, hd256 | bf16 |
| attention | `attention/prefill_attention_hd256.cu` | `attention_gate_batch_hd256_cuda` | `qwen35.rs::full_attention` | batched attn-gate, hd256 | bf16 |

> Live Qwen3-dense paged attention itself runs on **TileLang AOT** kernels
> (`tilelang_batch_{prefill,decode}_paged_hd128_q{16,32,40,64}_kv8_run_cuda`,
> declared in `src/ffi/attention.rs`, generated from
> `tools/tilelang/batch_{decode,prefill}_paged_hd128*.py`, dispatched in
> `attention.rs::run_tilelang_paged`). Q-tile ∈ {16,32,40,64}, hd128, **kv8 =
> FP8 paged KV**. These are not `.cu` files — listed here for navigation only.

## attention/ — **MISPLACED**: DSv4 attention ops living in `csrc/misc/`

These are core DSv4 MLA attention kernels but sit in `csrc/misc/`. They should
move to `csrc/attention/`.

| family | csrc file | FFI symbol | Rust caller | op shape | precision |
|---|---|---|---|---|---|
| attention (DSv4) | `misc/dsv4_attention.cu` | `dsv4_prepare_qk_cuda` / `…_start_pos_ptr_cuda` | `attention.rs::mla_attention` | Q/K RoPE prep, d_qk 576 (512 NoPE+64 RoPE) | bf16 |
| attention (DSv4) | `misc/dsv4_attention.cu` | `dsv4_swa_attention_cuda` / `…_start_pos_ptr_cuda` | `attention.rs::mla_attention` | sliding-window attn, hd≤1024 | bf16 |
| attention (DSv4) | `misc/dsv4_attention.cu` | `dsv4_hybrid_attention_cuda` / `…_start_pos_ptr_cuda` | `attention.rs::mla_attention` | hybrid SWA+compressed attn | bf16 |
| attention (DSv4) | `misc/dsv4_attention.cu` | `dsv4_compressor_update_cuda` / `…_start_pos_ptr_cuda` | `attention.rs::compressor_forward` | compressor KV update | bf16 |
| attention (DSv4) | `misc/dsv4_attention.cu` | `dsv4_csa_select_cuda` / `…_start_pos_ptr_cuda` | `attention.rs::csa_select` | compressed-attn top-k block select | bf16 |
| attention (DSv4) | `misc/dsv4_mhc.cu` | `dsv4_mhc_params_cuda` | `hc.rs::gen_mhc_params{,_into}` | multi-head-compressor sinkhorn params | bf16/f32 |
| attention (DSv4) | `misc/dsv4_mhc.cu` | `dsv4_mhc_expand_cuda` | `hc.rs::initial_stream_from_embeddings` | MHC stream expand | bf16 |
| attention (DSv4) | `misc/dsv4_mhc.cu` | `dsv4_mhc_pre_cuda` | `hc.rs::hc_pre` | MHC pre-projection | bf16 |
| attention (DSv4) | `misc/dsv4_mhc.cu` | `dsv4_mhc_post_cuda` | `hc.rs::hc_post` | MHC post-projection | bf16 |
| attention (DSv4) | `misc/dsv4_mhc.cu` | `dsv4_mhc_head_pre_cuda` | `hc.rs::head_hidden_from_stream` | MHC per-head hidden | bf16 |

## gemm/ — dense GEMV/GEMM + DSv4 quantized GEMV + DeepGEMM + Marlin

| family | csrc file | FFI symbol | Rust caller | op shape | precision |
|---|---|---|---|---|---|
| gemm | `gemm/gemv.cu` | `gemv_cuda` | `ops.rs::gemv` | (M,K)·(K,)→(M,), seq=1 | bf16 |
| gemm | `gemm/gemv.cu` | `gemm_cuda` | `ops.rs::gemm_batch`, `attention.rs::run_tilelang_paged` | (M,K)·(K,N)→(M,N) | bf16 |
| gemm (DSv4) | `gemm/quantized_gemv.cu` | `dsv4_fp8_gemv_batch_cuda` | `attention.rs::mla_linear{,_vec}` | batched GEMV, N×K block-scaled | fp8-e4m3 |
| gemm (DSv4) | `gemm/quantized_gemv.cu` | `dsv4_fp4_gemv_batch_cuda` | `attention.rs::mla_linear{,_vec}` | batched GEMV, N×K block-scaled | fp4-e2m1 |
| gemm (DSv4) | `gemm/dsv4_deepgemm_ops.cu` | `dsv4_deepgemm_pack_quantize_bf16_to_fp8_cuda` | `attention.rs::run_mla_linear_deepgemm_prefill` `[ck moe.rs]` | bf16→fp8 pack+quantize for DeepGEMM | bf16→fp8-e4m3 |
| gemm (DSv4) | `gemm/deepgemm_native.cu` | `dsv4_deepgemm_fp8_gemm_nt_cuda` | `attention.rs::run_mla_linear_deepgemm_prefill` `[ck moe.rs::dsv4_deepgemm_fp8_gemm_nt]` | NT FP8 GEMM (prefill MLA linear) | fp8-e4m3 |
| gemm (DSv4) | `gemm/deepgemm_bridge_stub.cu` → `deepgemm_native.cu` | `dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked_cuda` | `moe.rs::{deepgemm_grouped_experts,…_pooled,dsv4_shared_expert_pooled}` `[ck moe.rs]` | M-grouped masked FP8 GEMM (MoE experts) | fp8-e4m3 |
| gemm (MoE) | `gemm/moe_grouped_gemm.cu` | `moe_bf16_grouped_gemm_batch_cuda` | `moe.rs::moe_forward` `[ck moe.rs::moe_bf16_grouped_gemm_batch]` | grouped expert GEMM | bf16 |
| gemm (Marlin) | `gemm/marlin_repack.cu` | `gptq_marlin_repack_cuda` | `[ck tensor.rs::repack_for_marlin]` | GPTQ→Marlin weight repack | int4 |
| gemm (Marlin) | `gemm/marlin_int4_fp8_preprocess.cu` | `marlin_int4_fp8_preprocess_without_zp_cuda` | `[ck tensor.rs::from_hybrid_w4_marlin]` | W4 hybrid Marlin preprocess (no zero-point) | int4/fp8 |
| gemm (DSv4 cache) | `gemm/dsv4_fp8_cache.cu` | `dsv4_block_scaled_to_fp8_deepgemm_cuda` | `[ck tensor.rs::dsv4_fill_fp8_deepgemm_weight_cache]` | block-scaled→FP8 DeepGEMM weight cache fill | fp8-e4m3 |

> `deepgemm_native.cu` is default-built when sm_90 plus vendored DeepGEMM/CUTLASS
> sources are present; `deepgemm_bridge_stub.cu` links the same symbols and returns
> `CUDA_ERROR_NOT_SUPPORTED` when native DeepGEMM is unavailable or disabled.

## moe/ — DSv4 / Qwen3.6 expert routing

| family | csrc file | FFI symbol | Rust caller | op shape | precision |
|---|---|---|---|---|---|
| moe (DSv4) | `moe/dsv4_route.cu` | `dsv4_route_cuda` | `moe.rs::{dsv4_route_device,dsv4_moe_forward,dsv4_moe_forward_decode_graph}` `[ck moe.rs::dsv4_route]` | top-k expert routing | bf16 / f32 logits |
| moe (DSv4) | `moe/dsv4_route.cu` | `dsv4_count_local_experts_cuda` | `moe.rs::{moe_forward,dsv4_moe_forward,…_decode_pooled}` `[ck moe.rs]` | per-EP local-expert count | i32 |
| moe (DSv4) | `moe/dsv4_route.cu` | `dsv4_exclusive_scan_i32_cuda` | `[ck moe.rs]` | prefix-sum over expert counts | i32 |
| moe (DSv4) | `moe/dsv4_route.cu` | `dsv4_cast_i32_to_i64_cuda` | `[ck moe.rs]` | index dtype widen | i32→i64 |
| moe (DSv4) | `moe/dsv4_route.cu` | `dsv4_pack_local_experts_with_slots_cuda` | `[ck moe.rs]` | pack tokens→expert slots | i32 |
| moe (DSv4) | `moe/dsv4_route.cu` | `dsv4_scatter_all_route_slots_cuda` | `moe.rs::{moe_forward,dsv4_moe_forward,…_decode_pooled}` `[ck moe.rs]` | scatter route tokens to slots | bf16 |
| moe (DSv4) | `moe/dsv4_route.cu` | `dsv4_combine_route_slot_outputs_cuda` | `moe.rs::{moe_forward,dsv4_moe_forward,…_decode_pooled}` `[ck moe.rs]` | combine slot outputs (weighted) | bf16 |
| moe (DSv4) | `moe/dsv4_route.cu` (`elementwise_basic.cu`) | `dsv4_swiglu_clamped_cuda` / `…_routes_cuda` | `[ck moe.rs::dsv4_swiglu_clamped_batch]` | clamped SwiGLU on routed experts | bf16 |
| moe (DSv4 EP) | `moe/deepseek_mask_indices_by_ep.cu` | `dsv4_mask_indices_by_ep_i32_cuda` / `…_i64_cuda` | `[ck moe.rs::dsv4_mask_indices_by_ep_i32]` | EP-shard index masking | i32 / i64 |
| moe (Qwen3.6) | `moe/qwen36_route.cu` | `qwen36_add_shared_expert_gated_cuda` | `moe.rs::moe_forward` `[ck moe.rs::qwen36_add_shared_expert_gated]` | shared-expert gated add | bf16 |

## misc/ — **MISPLACED core ops**: these are norm / sampling / elementwise / linear-attn, not "misc"

| family | csrc file | FFI symbol | Rust caller | op shape | precision |
|---|---|---|---|---|---|
| norm | `misc/norm.cu` | `rms_norm_cuda` | `ops.rs::rms_norm_vec` | RMSNorm, hidden=n | bf16 |
| norm | `misc/norm.cu` | `rms_norm_batched_cuda` | `ops.rs::rms_norm_batch`, `attention.rs::mla_rms_norm` | batched RMSNorm | bf16 |
| norm | `misc/norm.cu` | `rms_norm_offset_cuda` | `qwen35.rs::rms_norm_offset_vec` | offset RMSNorm | bf16 |
| norm | `misc/norm.cu` | `rms_norm_batched_offset_cuda` | `qwen35.rs::rms_norm_offset` | batched offset RMSNorm | bf16 |
| norm | `misc/norm.cu` | `rms_norm_gated_cuda` | `qwen35.rs::linear_attention` | gated RMSNorm (GDR) | bf16 |
| sampling | `misc/sampling.cu` | `argmax_cuda` | `ops.rs::argmax` | greedy argmax over vocab | bf16/f32 logits |
| elementwise | `misc/elementwise_basic.cu` | `add_cuda` | `ops.rs::add_batch` | elementwise residual add | bf16 |
| elementwise | `misc/elementwise_basic.cu` | `silu_mul_cuda` | `ops.rs::silu_mul` | SwiGLU gate·up | bf16 |
| elementwise | `misc/elementwise_basic.cu` | `embedding_batched_cuda` | `ops.rs::embedding_batch` | token-id gather | bf16 |
| linear-attn (Qwen3.5) | `misc/gated_delta_rule.cu` | `gated_delta_rule_decode_cuda` | `qwen35.rs::linear_attention` | GDR recurrent decode step | bf16 |
| linear-attn (Qwen3.5) | `misc/gated_delta_rule.cu` | `gated_delta_rule_prefill_recurrent_cuda` | `qwen35.rs::linear_attention` | GDR recurrent prefill | bf16 |
| conv (Qwen3.5) | `misc/conv1d.cu` | `conv1d_prefill_cuda` | `qwen35.rs::linear_attention` | short-conv1d over prefill | bf16 |

## kv/ , kvcacheio/ , quant/ — present but **NOT wired in the current rewrite**

`infer-cuda` drives `PagedKVPool` only for page lifecycle (`alloc`, `attach_pages`,
`k_ptr`/`v_ptr`, `free_slot`, `page_indices`, …). KV is written/read by the
**TileLang kv8** kernels. None of the hand-rolled KV-quant / migrate / transfer /
turboquant kernels below have a live caller — they are library code kept for the
quantized-KV path that the rewrite has not re-attached.

---

## Library-present but **UNWIRED** (zero live caller — do NOT treat as live)

Verified by real-call-expression grep (excluding doc comments + `use` imports).
Either dead, test-only, or an as-yet-unattached subsystem in `cuda-kernels/src`.

| group | csrc file(s) | symbols (representative) | status |
|---|---|---|---|
| KV-quant decode-attention | `attention/decode_attention_quantized.cu`, `decode_attention_varlen_fp8.cu` | `decode_attention_{fp8,int8,int4}_per_channel_k_cuda`, `decode_attention_varlen_fp8_cuda` | wrappers in `[ck kv_quant.rs]` have 0 call sites |
| TurboQuant KV | `attention/decode_attention_turboquant.cu`, `quant/turboquant{,_fast}.cu` | `tq_decode_attention_cuda`, `tq_rotate_query_cuda`, `turboquant_*_cuda` | wrappers in `[ck kv_turboquant.rs / turboquant_state.rs]`, 0 live caller |
| KV-quant pack/migrate | `kv/kv_quant.cu`, `kv/kv_cache_to_paged.cu`, `kv/scatter_kv.cu`, `kv/paged_kv_append.cu` | `quantize_paged_kv_*`, `quantize_scatter_kv_fp8_*`, `kv_cache_to_paged_*`, `dequantize_paged_kv_*` | only test-harness / internal-to-`paged_kv.rs` callers |
| KV-tier transport | `kvcacheio/transfer.cu` | `transfer_kv_pages_layer_table_cuda` | `[ck paged_kv.rs::transfer_layer_table_pair]` not reached from `infer-cuda` |
| DSv4 FP8-KV + FlashMLA SM90 | `attention/dsv4_fp8_kv_pack.cu`, `attention/dsv4_flashmla_decode_build_indices.cu`, `misc/arle_flashmla_decode_shim.cu`, `misc/arle_flashmla_shim.cu`, `misc/arle_flashmla_csa_prep.cu`, `attention/arle_flashmla_decode_stubs.cu` | `arle_dsv4_fp8_kv_pack_*`, `arle_dsv4_flashmla_decode_build_indices_*`, `arle_flashmla_sm90_sparse_{prefill,decode}_fwd`, `arle_flashmla_*` | only doc-comment mentions in `dsv4.rs`; live DSv4 decode uses bf16 `dsv4_attention.cu` instead |
| DSv4 output inverse-RoPE | `misc/dsv4_attention.cu` (`arle_dsv4_output_inverse_rope_*`) | `arle_dsv4_output_inverse_rope_{,_start_pos_ptr,_batch_start_pos}_cuda` | FlashMLA-path only; unwired with FlashMLA |
| DSv4 TP attention repack | `misc/dsv4_tp_attention_repack.cu` | `dsv4_tp_q_repack_cuda`, `dsv4_tp_out_slice_cuda` | 0 caller |
| DSv4 misc gemv variants | `gemm/quantized_gemv.cu`, `gemm/quantized_gemv_mma.cu` | `dsv4_{fp8,fp4}_gemv_cuda`, `…_pair_batch_cuda`, `…_grouped_gemv_*`, `…_route_gemv_*`, `dsv4_fp8_gemv_batch_mma_launch` | only `_batch` (non-pair, non-grouped) variants are live |
| Marlin W4/W4A8 GEMM | `gemm/marlin_w4_fp8_kernel.cu`, `marlin_w4a8_kernel.cu`, `marlin_kernel.cu`, `marlin_pf8/` | `gemm_w4_fp8_marlin_cuda`, `gemm_w4a8_marlin_cuda`, `marlin_gemm_cuda` | repack/preprocess live; the GEMM kernels themselves 0 caller |
| GGUF Qk dequant/gemv | (no live `.cu`; ffi decls only) | `q{3,4,5,6}k_*`, `q8_*`, `w{2,4,8}a16_gemv_*` | full GGUF quant family dead in rewrite |
| misc dead | `misc/fused_mlp.cu`, `misc/split_qkv.cu`, `misc/arle_dtype_convert.cu`, `misc/conv1d_{decode,prefill}_batch.cu`, `misc/gdr_{decode,prefill}_batch.cu`, `misc/gdr_prefill_solve.cu`, `misc/elementwise_basic.cu` (`fused_add_rms_norm_*`) | `fused_mlp_cuda`, `split_qkv_cuda`, `arle_bf16_to_f32_cuda`, `conv1d_*_batch_cuda`, `gdr_*_batch_cuda`, `fused_add_rms_norm_*_cuda` | superseded by live `_recurrent` / fused paths |
| attention dead | `attention/decode_prep_paged_hd256.cu`, `attention/prefill_attention_paged_prep.cu` (`…_hd256`), `attention/mla_decode.cu`, `attention/fused_attention.cu`, `attention/prefill_attention.cu` | `decode_prep_paged_hd256_cuda`, `prefill_attention_paged_prep_hd256_cuda`, `mla_decode_paged_bf16_cuda`, `fused_gqa_attention_decode{,_batched}`, `prefill_attention_prep_cuda` | 0 live caller |

---

## Organization verdict

| family dir | verdict |
|---|---|
| `gemm/` | **well-organized** — all live GEMM/GEMV/DeepGEMM/Marlin ops here; carries a long tail of unwired Marlin/quant-gemv variants to prune |
| `moe/` | **well-organized** — DSv4 route + Qwen3.6 route + EP-mask all live and in-family |
| `kv/`, `kvcacheio/`, `quant/` | **dormant** — coherent internally but the whole subsystem is unwired in the rewrite (TileLang kv8 took over) |
| `attention/` | **partly misplaced** — live paged-prep + hd256 ops here, but the live DSv4 MLA attention kernels live in `misc/`; the FlashMLA/FP8-KV `.cu` here are all unwired |
| `misc/` | **MISPLACED dumping ground** — holds **core** ops: `norm.cu`, `sampling.cu`, `elementwise_basic.cu` (add/silu_mul/embedding) and the entire **live DSv4 attention** (`dsv4_attention.cu`, `dsv4_mhc.cu`) + Qwen3.5 linear-attn (`gated_delta_rule.cu`, `conv1d.cu`). Recommend: `norm.cu`→`norm/`, `sampling.cu`→`sampling/`, `elementwise_basic.cu`→`elementwise/`, `dsv4_attention.cu`+`dsv4_mhc.cu`→`attention/`, `gated_delta_rule.cu`+`conv1d.cu`→`recurrent/` (matching the `src/ffi/{norm,sampling,elementwise,recurrent}.rs` split that already exists). |
