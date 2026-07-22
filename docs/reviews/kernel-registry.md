# Kernel Registry — live CUDA operator index (`crates/cuda-kernels`)

> **Source of truth = code, not docs.** The 2026-07-12 reorg snapshot below is
> historical. Current reconciliation: **63** `.cu` under
> `crates/cuda-kernels/csrc/{attention,comm,elementwise,gemm,kv,moe,norm,quant,
> recurrent,sampling}/`, the FFI decls in
> `crates/cuda-kernels/src/ffi/`, and the **live callers in
> `crates/infer-cuda/src/`**. **The reorg** exploded the `misc/` junk drawer
> (0 files now): new `sampling/`·`norm/`·`recurrent/`·`elementwise/` dirs; DSv4
> MLA/DSA/MHC + TP-repack + FlashMLA/FA3 shims → `attention/`; `kvcacheio/`
> merged into `kv/`. Most families align with `src/ffi/`; legacy DSv4 attention
> declarations remain primarily in `src/ffi/misc.rs`. **Dead
> code deleted:** 3 Marlin W4/W4A8 GEMM `.cu` (`marlin_kernel.cu`,
> `marlin_w4_fp8_kernel.cu`, `marlin_w4a8_kernel.cu` + `marlin_dequant.cuh` +
> `marlin_pf8/`), `kv/paged_kv_append.cu`, `kv/scatter_kv.cu`, and their 5
> `src/ffi/{gemm,kv}.rs` extern decls (live `gptq_marlin_repack`/
> `marlin_int4_fp8_preprocess` kept). This rebuild vs the 2026-06-30 snapshot: **added
> the entire DSpark family** (`sampling.cu` chain-rejection/device-filter +
> `nonpaged_prefill_attention.cu`/`prefill_attention_hd256.cu` sliding-window
> ring KV prep), **added the previously-missing dir sections**
> (`comm/`, `quant/`, `deepep_sidecar/`), **promoted to live**
> the FlashMLA/FA3 shim path, the DSA-official family (`dsv4_dsa_official.cu`),
> TP attention repack, hd256 paged decode, and the FP8/FP4 quantized-linear
> gemv path — and **reconciled the count** against the 06-30 deletions
> (`469e24c9b` −2191 LOC dropped `dsv4_grouped_gemm.cu` + `quantized_gemv_mma.cu`;
> `#144` `5f9a0b3ec`/`90de21f0e` merged the FP8 tiled-GEMV pair bodies).
>
> **Live = a real Rust call expression** reaches the FFI symbol, either
> directly from `infer-cuda` (`ffi::sym(...)` / `cuda_moe::wrap(...)`) or via a
> `cuda-kernels/src` wrapper fn that `infer-cuda` invokes. Doc-comment mentions
> and bare `use` imports do **not** count. Symbols with zero live caller are
> listed in §Library-present-but-unwired, not as live rows.

## Column contract

| Column | Meaning |
|---|---|
| **family** | logical operator group (= csrc dir, post 07-12 reorg) |
| **csrc file** | `.cu` that defines the `extern "C"` kernel (relative to `csrc/`) |
| **FFI symbol** | the `extern "C"` entry declared in `src/ffi/*.rs` |
| **Rust caller** | live call site as `file::fn` (in `infer-cuda/src` unless noted `[ck]` = `cuda-kernels/src`) |
| **op shape** | M/N/K for GEMM; seq/head/window dims for attention; routing dims for MoE |
| **precision** | bf16 / fp8-e4m3 / fp4-e2m1 / int8 / int4 / mixed |

## Count reconciliation

- **63 `.cu`** across 10 kernel dirs (`attention` 28, `gemm` 15, `recurrent` 6,
  `kv` 4, `quant` 3, `moe`/`elementwise` 2, `comm`/`sampling`/`norm` 1 each).
  `deepep_sidecar/` is a separate C++ sidecar with **0 `.cu`**. The 2026-07-12
  56-file count is a historical reorg snapshot, not the current inventory.
- **~278 unique `extern "C"` symbols** (> 63 because most `.cu` export several
  launchers, and `deepgemm_native.cu`/`deepgemm_bridge_stub.cu` declare the same
  symbol set twice). Live rows below ≈ 83; the remainder are §unwired.
- **06-30 deletions applied:** `469e24c9b` removed whole files
  `gemm/dsv4_grouped_gemm.cu` (−408) and `gemm/quantized_gemv_mma.cu` (−330) and
  trimmed `quantized_gemv.cu` (−1097) — so the 06-30 unwired rows for
  `quantized_gemv_mma.cu` (`dsv4_fp8_gemv_batch_mma_launch`, `…_pair_batch`,
  `…_grouped_gemv`) are **gone from the tree** and dropped from this table.
  `#144` (`5f9a0b3ec`/`90de21f0e`) collapsed the FP8 tiled-GEMV pair kernels into
  one templated body — the surviving `_batch` GEMV symbols are unchanged callers.

---

## attention/ — paged prep + full-attn + FlashMLA shim + DSpark ring

| family | csrc file | FFI symbol | Rust caller | op shape | precision |
|---|---|---|---|---|---|
| attention | `attention/prefill_attention_paged_prep.cu` | `prefill_attention_paged_prep_cuda` | `attention.rs::prefill_attention` | paged prefill prep, hd128 | bf16 |
| attention | `attention/prefill_attention_paged_prep.cu` | `prefill_attention_paged_prep_hd256_cuda` | `qwen35.rs::full_attention` | paged prefill prep, hd256 | bf16 |
| attention | `attention/decode_prep_paged.cu` | `decode_prep_paged_cuda` | `attention.rs::decode_attention` | paged decode prep, hd128 | bf16 |
| attention | `attention/decode_prep_paged_hd256.cu` | `decode_prep_paged_hd256_cuda` / `attention_gate_paged_hd256_cuda` | `qwen35.rs::full_attention` | paged decode prep + gate, hd256 | bf16 |
| attention | `attention/nonpaged_prefill_attention.cu` | `nonpaged_prefill_attention_cuda` / `…_devpos_cuda` | `qwen35.rs::full_attention`, `qwen35/dspark.rs::dspark_draft_block` | non-paged prefill, Qwen3.5 full-attn + DSpark draft | bf16 |
| attention (DSpark) | `attention/nonpaged_prefill_attention.cu` | `nonpaged_prefill_attention_ring_cuda` | `qwen35/dspark.rs::dspark_draft_block` | draft sliding-window ring attn | bf16 |
| attention | `attention/prefill_attention_hd256.cu` | `prefill_attention_hd256_prep_cuda` / `attention_gate_batch_hd256_cuda` | `qwen35.rs::full_attention` | prefill prep + batched attn-gate, hd256 | bf16 |
| attention (DSpark) | `attention/prefill_attention_hd256.cu` | `prefill_attention_hd256_prep_ring_cuda` | `qwen35/dspark.rs::dspark_append_ctx` | draft-KV cap sliding-window ring prep (`23fa8f3e2`) | bf16 |
| attention | `attention/fused_attention.cu` | `fused_gqa_attention_decode_batched` | `qwen35.rs` (decode) | fused GQA batched decode | bf16 |
| attention (FlashMLA) | `attention/arle_flashmla_decode_shim.cu`, `attention/arle_flashmla_decode_stubs.cu` | `arle_flashmla_sm90_sparse_decode_fwd` | `attention.rs::mla_attention`, `attention/flashmla.rs` | SM90 sparse MLA decode (vendored FlashMLA bridge) | bf16 / fp8-e4m3 kv |
| attention (FlashMLA) | `attention/arle_flashmla_shim.cu` | `arle_flashmla_sm90_sparse_prefill_fwd` | `attention.rs::mla_attention` | SM90 sparse MLA prefill | bf16 |
| attention (FA3) | `attention/arle_fa3_shim.cu`, `arle_fa3_stubs.cu` | `arle_fa3_fwd_hd256_bf16_cuda` | `qwen35.rs::full_attention` | FA3 hd256 full-attn (vendored FA3 bridge) | bf16 |
| attention (FA2 SM70) | `attention/arle_fa2_sm70.cu` | `arle_fa2_sm70_attention_cuda` | `qwen35.rs::full_attention` | SM70 full attention | bf16 |
| attention (DSpark) | `attention/dsv4_dspark_draft_attention.cu` | `dsv4_dspark_draft_attention_cuda` | `dsv4/dspark.rs::forward_stage` | DSv4 draft attention | bf16 |

> Live Qwen3-dense paged attention itself runs on **TileLang AOT** kernels
> (`tilelang_batch_{prefill,decode}_paged_hd128_q{16,32,40,64}_kv8_run_cuda`,
> declared in `src/ffi/attention.rs`, generated from
> `tools/tilelang/batch_{decode,prefill}_paged_hd128*.py`, dispatched in
> `attention.rs::run_tilelang_paged`). Q-tile ∈ {16,32,40,64}, hd128, **kv8 =
> FP8 paged KV**. Not `.cu` files — listed here for navigation only.
>
> The live FlashMLA index/pack helpers are the **vendored** `arle_flashmla_csa_*`
> / `arle_flashmla_hca_build_indice` / `arle_flashmla_chain_verify_build_indice`
> symbols (declared in `src/ffi/attention.rs`, defined in `vendor/`), **not** the
> DSv4-specific `attention/dsv4_flashmla_decode_build_indices.cu` + `dsv4_fp8_kv_pack.cu`
> — those remain §unwired.

## attention/ — DSv4 MLA + DSA + MHC + TP repack

Core DSv4 attention kernels, relocated from `csrc/misc/` into split
`csrc/attention/` translation units. Legacy declarations remain primarily in
`src/ffi/misc.rs`; source and Rust FFI modules are not aligned 1:1.

| family | csrc file | FFI symbol | Rust caller | op shape | precision |
|---|---|---|---|---|---|
| attention (DSv4) | `attention/dsv4_prep.cu` | `dsv4_prepare_qk_cuda` / `…_start_pos_ptr_cuda` / `…_fused_batch_start_pos_cuda` | `attention.rs::mla_attention` | Q/K RoPE prep, d_qk 576 (512 NoPE+64 RoPE) | bf16 |
| attention (DSv4) | `attention/dsv4_swa.cu` | `dsv4_swa_attention_cuda` / `…_start_pos_ptr_cuda` | `attention.rs::mla_attention` | sliding-window attn, hd≤1024 | bf16 |
| attention (DSv4) | `attention/dsv4_hybrid.cu` | `dsv4_hybrid_attention_cuda` / `…_start_pos_ptr_cuda` | `attention.rs::mla_attention` | hybrid SWA+compressed attn | bf16 |
| attention (DSv4) | `attention/dsv4_compressor.cu` | `dsv4_compressor_update_cuda` / `…_start_pos_ptr_cuda` / `…_batched_start_pos_ptr_cuda` | `attention.rs::compressor_forward` | compressor KV update | bf16 |
| attention (DSv4) | `attention/dsv4_swa.cu` | `dsv4_update_window_cache_cuda` / `…_start_pos_ptr_cuda` / `…_batched_ptr_cuda` | `attention.rs` (Route A window cache) | sliding window cache write | bf16 |
| attention (DSv4, Route A) | `attention/dsv4_oproj.cu` | `dsv4_oproj_group_gather_cuda` / `dsv4_oproj_group_scatter_cuda` | `attention.rs::dsv4_oproj_group_{gather,scatter}` | o-proj group gather/scatter over compressor-state pool (`6a78a490d`) | bf16 |
| attention (DSv4) | `attention/dsv4_oproj.cu` | `arle_dsv4_output_inverse_rope_cuda` / `…_start_pos_ptr_cuda` / `…_batched_ptr_cuda` / `…_batch_start_pos_cuda` | `attention.rs::mla_attention` (FlashMLA out path) | inverse-RoPE on MLA output | bf16 |
| attention (DSv4 DSA) | `attention/dsv4_dsa_official.cu` | `dsv4_deepseek_v4_topk_transform_cuda` | `attention.rs` (DSA select) | official DSA top-k transform | bf16 |
| attention (DSv4 DSA) | `attention/dsv4_dsa_official.cu` | `dsv4_dsa_build_select_meta_cuda` | `attention.rs` | DSA block-select metadata | i32 |
| attention (DSv4 DSA) | `attention/dsv4_dsa_official.cu` | `dsv4_dsa_fused_q_indexer_rope_hadamard_quant_cuda` | `attention.rs` | fused Q-indexer RoPE+Hadamard+quant | bf16→fp8 |
| attention (DSv4 DSA) | `attention/dsv4_dsa_official.cu` | `dsv4_dsa_fused_store_index_k_cache_cuda` / `…_batched_cuda` | `attention.rs` | fused index-K cache store | fp8-e4m3 |
| attention (DSv4 DSA) | `attention/dsv4_dsa_official.cu` | `dsv4_dsa_hadamard128_bf16_cuda` / `…_batched_cuda` | `attention.rs` | 128-dim Hadamard rotate | bf16 |
| attention (DSv4 DSA) | `attention/dsv4_dsa_official.cu` | `dsv4_dsa_fill_context_lens_positions_start_pos_cuda` | `attention.rs` | context-len/position fill | i32 |
| attention (DSv4 MHC) | `attention/dsv4_mhc.cu` | `dsv4_mhc_params_cuda` / `dsv4_mhc_params_pre_rms_norm_cuda` | `hc.rs::gen_mhc_params{,_into}` | multi-head-compressor sinkhorn params | bf16/f32 |
| attention (DSv4 MHC) | `attention/dsv4_mhc.cu` | `dsv4_mhc_expand_cuda` | `hc.rs::initial_stream_from_embeddings` | MHC stream expand | bf16 |
| attention (DSv4 MHC) | `attention/dsv4_mhc.cu` | `dsv4_mhc_pre_cuda` / `dsv4_mhc_pre_rms_norm_cuda` | `hc.rs::hc_pre` | MHC pre-projection | bf16 |
| attention (DSv4 MHC) | `attention/dsv4_mhc.cu` | `dsv4_mhc_post_cuda` | `hc.rs::hc_post` | MHC post-projection | bf16 |
| attention (DSv4 MHC) | `attention/dsv4_mhc.cu` | `dsv4_mhc_head_pre_cuda` | `hc.rs::head_hidden_from_stream` | MHC per-head hidden | bf16 |
| attention (DSv4 MTP) | `attention/dsv4_mhc.cu` | `dsv4_mtp_add_eproj_hproj_cuda` | `dsv4.rs` (MTP head) | MTP e-proj + h-proj add | bf16 |
| attention (DSv4 TP) | `attention/dsv4_tp_attention_repack.cu` | `dsv4_tp_q_repack_cuda` / `dsv4_tp_out_slice_cuda` | `attention.rs::mla_attention` (TP=8) | TP Q repack + output slice | bf16 |

## gemm/ — dense GEMV/GEMM + DSv4 quantized GEMV + DeepGEMM + Marlin repack

| family | csrc file | FFI symbol | Rust caller | op shape | precision |
|---|---|---|---|---|---|
| gemm | `gemm/gemv.cu` | `gemv_cuda` | `ops.rs::gemv` `[ck]` | (M,K)·(K,)→(M,), seq=1 | bf16 |
| gemm | `gemm/gemv.cu` | `gemm_cuda` | `ops.rs::gemm_batch`, `attention.rs::run_tilelang_paged` | (M,K)·(K,N)→(M,N) | bf16 |
| gemm (DSv4) | `gemm/quantized_gemv.cu` | `dsv4_fp8_gemv_batch_cuda` | `attention.rs::mla_linear{,_vec}` | batched GEMV, N×K block-scaled | fp8-e4m3 |
| gemm (DSv4) | `gemm/quantized_gemv.cu` | `dsv4_fp4_gemv_batch_cuda` | `attention.rs::mla_linear{,_vec}` | batched GEMV, N×K block-scaled | fp4-e2m1 |
| gemm (DSv4 route) | `gemm/quantized_gemv.cu` | `dsv4_fp8_route_gemv_batch_cuda` / `dsv4_fp4_route_gemv_batch_cuda` | `attention.rs` (routed MLA linear) | route-gathered batched GEMV | fp8/fp4 |
| gemm (quant-linear) | `gemm/quantized_gemv.cu` | `gemv_fp8_block_scaled_cuda` / `…_batch_cuda` | `ops/quant_linear.rs` | block-scaled FP8 GEMV | fp8-e4m3 |
| gemm (quant-linear) | `gemm/quantized_gemv.cu` | `gemv_fp4_e2m1_group_cuda` / `…_batch_cuda` | `ops/quant_linear.rs` | group-scaled FP4 GEMV | fp4-e2m1 |
| gemm (dequant) | `gemm/quantized_gemv.cu` | `dequantize_fp8_block_scaled_to_bf16_cuda` | `qwen35.rs`, `ops/quant_linear.rs` | FP8→bf16 dequant | fp8→bf16 |
| gemm (DSv4) | `gemm/dsv4_deepgemm_ops.cu` | `dsv4_deepgemm_pack_quantize_bf16_to_fp8_cuda` | `attention.rs::run_mla_linear_deepgemm_prefill`, `cuda_moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8` | bf16→fp8 pack+quantize for DeepGEMM | bf16→fp8-e4m3 |
| gemm (DSv4 MoE) | `gemm/dsv4_deepgemm_ops.cu` | `dsv4_deepgemm_silu_mul_masked_quant_cuda` / `dsv4_deepgemm_swiglu_quantize_w13_cuda` / `dsv4_deepgemm_unpad_grouped_bf16_cuda` | `[ck moe.rs]` (grouped-expert DeepGEMM) | masked SwiGLU + w13 quantize + unpad | bf16/fp8-e4m3 |
| gemm (DSv4) | `gemm/deepgemm_native.cu` (↔ `deepgemm_bridge_stub.cu`) | `dsv4_deepgemm_fp8_gemm_nt_cuda` | `cuda_moe::dsv4_deepgemm_fp8_gemm_nt`, `attention.rs` | NT FP8 GEMM (prefill MLA linear) | fp8-e4m3 |
| gemm (DSv4 MoE) | `gemm/deepgemm_native.cu` (↔ stub) | `dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked_cuda` / `…_contiguous_cuda` | `[ck moe.rs]`, `cuda_moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous` | M-grouped FP8 GEMM (MoE experts) | fp8-e4m3 |
| gemm (DSv4) | `gemm/deepgemm_native.cu` (↔ stub) | `dsv4_deepgemm_fp8_paged_mqa_logits_fused_cache_cuda` / `…_metadata_cuda` | `attention.rs`, `cuda_moe::dsv4_deepgemm_paged_mqa_logits_metadata` | paged MQA logits (DSA) | fp8-e4m3 |
| gemm (DSv4) | `gemm/deepgemm_native.cu` (↔ stub) | `dsv4_deepgemm_native_preflight_cuda` | `cuda_moe::dsv4_deepgemm_native_preflight` | native-DeepGEMM availability probe | — |
| gemm (DSv4 MoE decode) | `gemm/dsv4_fp8_decode_moe.cu` | `dsv4_fp8_grouped_swiglu_decode_cuda` / `dsv4_fp8_grouped_down_decode_cuda` | `[ck moe.rs]` (decode-pooled MoE) | grouped SwiGLU + down GEMV, decode | fp8-e4m3 |
| gemm (MoE) | `gemm/moe_grouped_gemm.cu` | `moe_bf16_grouped_gemm_batch_cuda` | `moe.rs::moe_forward` `[ck moe.rs::moe_bf16_grouped_gemm_batch]` | grouped expert GEMM | bf16 |
| gemm (Marlin) | `gemm/marlin_repack.cu` | `gptq_marlin_repack_cuda` | `[ck tensor.rs::repack_for_marlin]` | GPTQ→Marlin weight repack | int4 |
| gemm (Marlin) | `gemm/marlin_int4_fp8_preprocess.cu` | `marlin_int4_fp8_preprocess_without_zp_cuda` | `[ck tensor.rs::from_hybrid_w4_marlin]` | W4 hybrid Marlin preprocess (no zero-point) | int4/fp8 |
| gemm (DSv4 cache) | `gemm/dsv4_fp8_cache.cu` | `dsv4_block_scaled_to_fp8_deepgemm_cuda` | `[ck tensor.rs::dsv4_fill_fp8_deepgemm_weight_cache]` | block-scaled→FP8 DeepGEMM weight cache fill | fp8-e4m3 |
| gemm (Qwen3.6 MoE SM120) | `gemm/fp8_moe_grouped_cutlass_sm120.cu` | `arle_fp8_moe_grouped_gemm_nt_sm120_cuda` | `moe.rs::moe_forward` `[ck moe.rs]` | grouped FP8 expert GEMM | fp8-e4m3 |

> `deepgemm_native.cu` is default-built when sm_90 plus vendored DeepGEMM/CUTLASS
> sources are present; `deepgemm_bridge_stub.cu` links the same symbols and returns
> `CUDA_ERROR_NOT_SUPPORTED` when native DeepGEMM is unavailable or disabled.

## moe/ — DSv4 / Qwen3.6 expert routing

| family | csrc file | FFI symbol | Rust caller | op shape | precision |
|---|---|---|---|---|---|
| moe (DSv4) | `moe/dsv4_route.cu` | `dsv4_route_cuda` | `moe.rs::{dsv4_route_device,dsv4_moe_forward,dsv4_moe_forward_decode_graph}` `[ck moe.rs::dsv4_route]` | top-k expert routing | bf16 / f32 logits |
| moe (DSv4) | `moe/dsv4_route.cu` | `dsv4_count_local_experts_cuda` | `moe.rs::{moe_forward,dsv4_moe_forward,…_decode_pooled}` `[ck moe.rs]` | per-EP local-expert count | i32 |
| moe (DSv4) | `moe/dsv4_route.cu` | `dsv4_exclusive_scan_i32_cuda` / `moe_exclusive_scan_aligned_i32_cuda` | `[ck moe.rs]` | prefix-sum over expert counts | i32 |
| moe (DSv4) | `moe/dsv4_route.cu` | `dsv4_cast_i32_to_i64_cuda` | `[ck moe.rs]` | index dtype widen | i32→i64 |
| moe (DSv4) | `moe/dsv4_route.cu` | `dsv4_pack_local_experts_with_slots_cuda` / `…_and_indices_cuda` | `[ck moe.rs]` | pack tokens→expert slots | i32 |
| moe (DSv4) | `moe/dsv4_route.cu` | `dsv4_scatter_all_route_slots_cuda` | `moe.rs::{moe_forward,dsv4_moe_forward,…_decode_pooled}` `[ck moe.rs]` | scatter route tokens to slots | bf16 |
| moe (DSv4) | `moe/dsv4_route.cu` | `dsv4_combine_route_slot_outputs_cuda` / `dsv4_combine_route_outputs_cuda` | `moe.rs::{moe_forward,dsv4_moe_forward,…_decode_pooled}` `[ck moe.rs]` | combine slot outputs (weighted) | bf16 |
| moe (DSv4) | `moe/dsv4_route.cu` | `dsv4_swiglu_clamped_cuda` / `…_routes_cuda` | `[ck moe.rs::dsv4_swiglu_clamped_batch]` | clamped SwiGLU on routed experts | bf16 |
| moe (DSv4 EP) | `moe/dsv4_route.cu` | `dsv4_scatter_packed_route_slot_cuda` / `dsv4_scale_route_outputs_by_meta_cuda` / `dsv4_sum_padded_route_outputs_by_peer_cuda` | `[ck moe.rs]` (EP dispatch/combine) | packed EP scatter/scale/sum | bf16 |
| moe (Qwen3.6) | `moe/qwen36_route.cu` | `qwen36_add_shared_expert_gated_cuda` | `moe.rs::moe_forward` `[ck moe.rs::qwen36_add_shared_expert_gated]` | shared-expert gated add | bf16 |
| moe (Qwen3.6) | `moe/qwen36_route.cu` | `qwen36_renorm_topk_weights_cuda` | `[ck moe.rs]` | top-k weight renorm | f32 |

> `moe/deepseek_mask_indices_by_ep.cu` from the 06-30 registry no longer exists
> (deleted; the `dsv4_mask_indices_by_ep_*` EP-mask kernels are gone tree-wide, so
> that 06-30 row is dropped). The `[ck moe.rs]` wrappers are reached from
> `infer-cuda/src/moe.rs` (own crate)
> and `qwen35.rs`/`attention.rs` via `use cuda_kernels::moe as cuda_moe`.

## sampling/ · norm/ · elementwise/ · recurrent/ — DSpark sampling, norms, elementwise, linear-attn

(All four dirs split out of the former `misc/` in the 07-12 reorg, aligned to
`src/ffi/{sampling,norm,elementwise,recurrent}.rs`.)

| family | csrc file | FFI symbol | Rust caller | op shape | precision |
|---|---|---|---|---|---|
| sampling | `sampling/sampling.cu` | `argmax_cuda` / `argmax_batch_cuda` | `ops.rs::argmax{,_batch}` | greedy argmax over vocab | bf16/f32 logits |
| sampling (DSpark) | `sampling/sampling.cu` | `dspark_draft_sample_cuda` | `qwen35/dspark.rs::dspark_draft_block` | draft-model filtered multinomial sample (`e22a41637`) | bf16 logits |
| sampling (DSpark) | `sampling/sampling.cu` | `dspark_filter_probs_cuda` | `qwen35/dspark.rs::dspark_accept_commit_sampled` | device-side top-k/top-p prob filter | bf16→f32 |
| sampling (DSpark) | `sampling/sampling.cu` | `dspark_chain_accept_cuda` | `qwen35/dspark.rs::dspark_accept_commit_sampled` | chain-rejection accept + bonus resample | f32 |
| norm | `norm/norm.cu` | `rms_norm_cuda` | `ops.rs::rms_norm_vec` | RMSNorm, hidden=n | bf16 |
| norm | `norm/norm.cu` | `rms_norm_batched_cuda` | `ops.rs::rms_norm_batch`, `attention.rs::mla_rms_norm` | batched RMSNorm | bf16 |
| norm | `norm/norm.cu` | `rms_norm_offset_cuda` / `rms_norm_batched_offset_cuda` | `qwen35.rs::rms_norm_offset{,_vec}` | offset RMSNorm | bf16 |
| norm | `norm/norm.cu` | `rms_norm_gated_cuda` | `qwen35.rs::linear_attention` | gated RMSNorm (GDR) | bf16 |
| elementwise | `elementwise/elementwise_basic.cu` | `add_cuda` / `add_scaled_row_cuda` | `ops.rs::add_batch`, `attention.rs` | elementwise residual/scaled add | bf16 |
| elementwise | `elementwise/elementwise_basic.cu` | `silu_mul_cuda` | `ops.rs::silu_mul` | SwiGLU gate·up | bf16 |
| elementwise | `elementwise/elementwise_basic.cu` | `embedding_batched_cuda` | `ops.rs::embedding_batch` | token-id gather | bf16 |
| dtype | `elementwise/arle_dtype_convert.cu` | `arle_bf16_to_f32_cuda` | `attention.rs` / `ops.rs` | bf16→f32 cast | bf16→f32 |
| linear-attn (Qwen3.5) | `recurrent/gated_delta_rule.cu` | `gated_delta_rule_decode_cuda` | `qwen35.rs::linear_attention` | GDR recurrent decode step | bf16 |
| linear-attn (Qwen3.5) | `recurrent/gated_delta_rule.cu` | `gated_delta_rule_prefill_recurrent_cuda` | `qwen35.rs::linear_attention` | GDR recurrent prefill | bf16 |
| linear-attn (Qwen3.5) | `recurrent/gated_delta_rule.cu` | `gdr_fq_prep_cuda` (+ fixed-point `gdr_fq_{kkt,fwd,cumsum}_cuda`) | `qwen35.rs::linear_attention` | GDR fixed-point-quant solve prep | bf16 |
| linear-attn (Qwen3.5) | `recurrent/gdr_decode_batch.cu` | `gdr_decode_batch_cuda` | `qwen35.rs::linear_attention` | batched GDR decode | bf16 |
| conv (Qwen3.5) | `recurrent/conv1d.cu` | `conv1d_prefill_cuda` | `qwen35.rs::linear_attention` | short-conv1d over prefill | bf16 |
| conv (Qwen3.5) | `recurrent/conv1d_decode_batch.cu` | `conv1d_decode_batch_cuda` | `qwen35.rs::linear_attention` | batched conv1d decode | bf16 |

## comm/ — TP custom all-reduce (LIVE)

| family | csrc file | FFI symbol | Rust caller | op shape | precision |
|---|---|---|---|---|---|
| comm | `comm/custom_all_reduce.cu` | `arle_car_create` / `arle_car_destroy_prod` | `tp.rs` (CAR lifecycle) | NVLink CAR handle create/destroy | — |
| comm | `comm/custom_all_reduce.cu` | `arle_car_allreduce_bf16_into` | `tp.rs` (TP all-reduce) | ring/one-shot all-reduce | bf16 |
| comm | `comm/custom_all_reduce.cu` | `arle_car_allgather_bf16_into` | `tp.rs` (TP all-gather) | all-gather | bf16 |
| comm | `comm/custom_all_reduce.cu` | `arle_car_{alloc,free}_shared` / `arle_car_{open,close}_peer` | `tp.rs` (IPC peer setup) | shared-buffer + peer-handle mgmt | — |

## kv/ (KV-tier transport) — present, UNWIRED from infer-cuda

`kvcacheio/transfer.cu` was merged into `kv/` in the 07-12 reorg.

| family | csrc file | FFI symbol | status |
|---|---|---|---|
| kv-tier | `kv/transfer.cu` | `transfer_kv_pages_layer_table_cuda` | wrapper `[ck paged_kv.rs::transfer_layer_table_pair]` exists but **not reached from `infer-cuda`** — dormant KV-tier transport |

## quant/ — dtype convert + TurboQuant (present, UNWIRED)

| family | csrc file | FFI symbol | status |
|---|---|---|---|
| dtype | `quant/dtype_convert.cu` | `bf16_to_fp16_cuda` / `fp16_to_bf16_cuda` | **0 infer-cuda caller** — dead |
| turboquant | `quant/turboquant.cu` | `turboquant_{quantize,dequantize}_kv_cuda`, `turboquant_generate_rotation`, `turboquant_lloyd_max`, … | **0 live caller** (TurboQuant KV path unattached in rewrite) |
| turboquant | `quant/turboquant_fast.cu` | `turboquant_fast_{quantize,dequantize}_kv_cuda`, `turboquant_generate_signs` | **0 live caller** |

## kv/ — paged-KV quant/pack/migrate

FP8/INT8 KIVI per-channel-K quantization, refill, and fused decode are live via
`infer-cuda/src/attention.rs`; Qwen3.5 also invokes FP8 KV quantization from
`infer-cuda/src/qwen35.rs`. INT4, TurboQuant, per-token-K, migrate, and
incompatible variants remain unwired.

| family | csrc file(s) | FFI symbol (representative) | status |
|---|---|---|---|
| kv migrate | `kv/kv_cache_to_paged.cu` | `kv_cache_to_paged_*_cuda` | 0 infer-cuda caller (`scatter_kv.cu`/`scatter_write_kv_cuda` deleted 07-12) |
| kv append | `kv/paged_kv_metadata.cu` | `paged_kv_append_{new_page,last_token}_indices_cuda` | internal to `[ck tilelang.rs]` only (`paged_kv_append.cu`/`paged_kv_append_cuda` deleted 07-12) |
| kv quant (live FP8/INT8) | `kv/kv_quant.cu` | `quantize_paged_kv_{fp8,int8}_per_channel_cuda`, `dequantize_paged_kv_{fp8,int8}_per_channel_k_to_hnd_cuda` | live via `infer-cuda/src/attention.rs`; FP8 quantize also via `infer-cuda/src/qwen35.rs` |
| kv quant (unwired variants) | `kv/kv_quant.cu` | INT4, per-token-K, scatter, and incompatible quant/dequant variants | 0 live caller |

## deepep_sidecar/ — NVSHMEM DeepEP sidecar (out-of-process, no `.cu`)

| item | file | wiring |
|---|---|---|
| DeepEP sidecar binary | `deepep_sidecar/sidecar_main.cpp` | Standalone process (not FFI-linked into `infer-cuda`). Driven via `deepep-sys` IPC for internode LL dispatch/combine. **No CUDA kernel symbol** in this dir — listed for dir completeness. |

---

## Library-present but **UNWIRED** (zero live caller — do NOT treat as live)

Verified by real-call-expression grep (excluding doc comments + `use` imports).

| group | csrc file(s) | symbols (representative) | status |
|---|---|---|---|
| DSv4 FP8-KV pack + FlashMLA index-build | `attention/dsv4_fp8_kv_pack.cu`, `attention/dsv4_flashmla_decode_build_indices.cu` | `arle_dsv4_fp8_kv_pack{,_strided,_strided_batched}_cuda`, `arle_dsv4_v32_fp8_kv_pack_strided_cuda`, `arle_dsv4_flashmla_decode_build_indices{,_start_pos_ptr,_batched}_cuda` | 0 caller — the live FlashMLA path uses the **vendored** `arle_flashmla_csa_*`/`hca_*` index builders instead |
| FlashMLA CSA prep | `attention/arle_flashmla_csa_prep.cu` | (no `extern "C"` in-file; helper TU) | linked into shim only |
| KV-quant decode-attention tails | `attention/decode_attention_quantized.cu`, `decode_attention_varlen_fp8.cu` | INT4 and varlen/per-token incompatible variants | 0 live caller; FP8/INT8 per-channel-K decode is live via `infer-cuda/src/attention.rs` |
| TurboQuant decode-attention | `attention/decode_attention_turboquant.cu` | `tq_decode_attention_cuda`, `tq_rotate_query_cuda` | wrappers in `[ck kv_turboquant.rs]`, 0 live caller |
| Activation-quant + weight-gemv | `gemm/w4_fp8_activation_quant.cu`, `gemm/w4a8_activation_quant.cu`, `gemm/turboquant_weight_gemv.cu` | `quantize_bf16_rows_to_{fp8_e4m3,int8}_cuda`, `turboquant_weight_{gemv,dequant}_cuda` | 0 caller (their consumer Marlin W4/W4A8 GEMM `.cu` deleted 07-12; TQ-GEMM lane still unwired) |
| DSv4 misc gemv variants | `gemm/quantized_gemv.cu` | `gemv_fp8_wread_probe_cuda`, `moe_fp8_block_scaled_grouped_gemv_pair_batch_cuda`, `moe_fp4_e2m1_grouped_gemv_pair_batch_cuda`, `dequantize_fp8_block_scaled_to_bf16` unpaired variants | probe/pair-grouped variants unused; the `_batch`/`route_batch` variants are live above |
| GGUF Qk dequant/gemv | `gemm/quantized_gemv.cu` | `q{3,4,5,6}k_*`, `q8_*`, `qxk_*`, `w{2,4,8}a16_gemv_*` | full GGUF quant family dead in rewrite |
| DSv4 grouped-GEMM autotune | `gemm/gemv.cu` | `autotune_gemm_cuda`, `autotune_all_cached_gemms_cuda`, `gemm_graphsafe_cuda` | 0 live caller (autotune harness / graph-safe variant unused) |
| dead (relocated from misc/) | `gemm/fused_mlp.cu`, `attention/split_qkv.cu`, `recurrent/gdr_prefill_batch.cu`, `recurrent/gdr_prefill_solve.cu`, `norm/norm.cu` (`fused_add_rms_norm_*`, `cast_*`) | `fused_mlp_cuda`, `split_qkv_cuda`, `silu_mul_fused_cuda`, `gated_delta_rule_prefill_chunk_*_cuda`, `fused_add_rms_norm_*_cuda`, `cast_{bf16_to_f32,f32_to_bf16}_cuda` | superseded by live `_recurrent` / fused paths |
| attention dead | `attention/decode_prep_paged_hd256.cu` (unused arms), `attention/mla_decode.cu`†, `attention/prefill_attention.cu`† | — | †files removed pre-06-30; no live rows |
| MHC bench | `attention/dsv4_mhc.cu` | `dsv4_mhc_params_bench_cuda`, `dsv4_mhc_pre_rms_norm_bench_cuda` | bench-only, 0 runtime caller |
| DSA/DSv4 misc | `attention/dsv4_dsa_official.cu` | `dsv4_deepseek_v4_topk_transform` unused arms | secondary DSA arms unused; core family live above |

---

## Organization verdict

**The 07-12 reorg resolved the `misc/` junk-drawer problem** — `misc/` is deleted
(0 files); every source family now lives in a domain dir. Most align with
`src/ffi/*.rs`; legacy DSv4 attention declarations remain primarily in
`src/ffi/misc.rs`. The prior "recommend: move X→Y" plan below was executed
verbatim. Remaining work is pruning unwired tails, not relocation.

| family dir | verdict |
|---|---|
| `gemm/` | **well-organized** — all live GEMM/GEMV/DeepGEMM/Marlin-repack/quant-linear ops here; carries an unwired tail (TQ-GEMM + activation-quant + GGUF variants) to prune (3 Marlin GEMM `.cu` already deleted 07-12) |
| `moe/` | **well-organized** — DSv4 route + Qwen3.6 route (EP-mask folded into `dsv4_route.cu`) + decode-pooled all live and in-family |
| `comm/` | **well-organized** — single-purpose; the whole CAR family is live via `tp.rs` |
| `attention/` | **well-organized + newly-live** — live paged-prep + hd256 + FA3/FlashMLA shim + DSpark ring **plus** the DSv4 MLA/DSA/MHC + TP-repack kernels relocated here 07-12; the DSv4-specific FP8-KV-pack + FlashMLA-index `.cu` stay unwired (vendored builders won) |
| `sampling/`, `norm/`, `elementwise/`, `recurrent/` | **new, well-organized** — split out of `misc/` 07-12; hold live DSpark sampling / norms / SwiGLU+embedding / Qwen3.5 linear-attn |
| `kv/`, `quant/` | **mixed** — FP8/INT8 KIVI quantize/refill/decode is live; KV-tier transport, INT4/TurboQuant, and incompatible variants remain unwired; `kvcacheio/transfer.cu` merged into `kv/` 07-12 |
| `deepep_sidecar/` | **separate deliverable** — an out-of-process NVSHMEM sidecar binary, not a kernel dir; keep isolated. |
