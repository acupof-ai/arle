use super::{CUresult, CUstream, Half};

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct Sm90MegaMoeWorkspaceLayoutRaw {
    pub num_bytes: u64,
    pub x: u64,
    pub x_sf: u64,
    pub topk_idx: u64,
    pub topk_weights: u64,
    pub l1_acts: u64,
    pub l1_acts_sf: u64,
    pub l1_topk_weights: u64,
    pub l2_acts: u64,
    pub l2_acts_sf: u64,
    pub combine: u64,
    pub num_max_pool_tokens: i32,
    pub num_padded_sf_pool_tokens: i32,
    pub num_max_tokens_per_rank: i32,
}

#[allow(dead_code)]
unsafe extern "C" {
    pub fn dsv4_sm90_mega_moe_workspace_layout_cuda(
        num_ranks: i32,
        num_experts: i32,
        requested_max_tokens_per_rank: i32,
        num_topk: i32,
        hidden: i32,
        intermediate_hidden: i32,
        out: *mut Sm90MegaMoeWorkspaceLayoutRaw,
    ) -> CUresult;

    pub fn dsv4_sm90_mega_moe_pre_dispatch_cuda(
        hidden_states: *const Half,
        route_indices: *const i32,
        route_weights: *const f32,
        workspace_x: *mut u8,
        workspace_x_sf: *mut f32,
        workspace_topk_idx: *mut i64,
        workspace_topk_weights: *mut f32,
        num_tokens: i32,
        padded_max_tokens: i32,
        hidden: i32,
        num_topk: i32,
        enable_pdl: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_sm90_mega_moe_launch_cuda(
        y: *mut Half,
        cumulative_local_expert_recv_stats: *mut i32,
        peer_buffer_ptrs: *const u64,
        local_workspace: *mut u8,
        num_ranks: i32,
        rank_idx: i32,
        num_max_tokens_per_rank: i32,
        num_tokens: i32,
        num_experts: i32,
        num_topk: i32,
        hidden: i32,
        intermediate_hidden: i32,
        activation_clamp: f32,
        fast_math: i32,
        enable_pdl: i32,
        l1_weights: *const u8,
        l1_weight_stride: i32,
        l1_weights_sf: *const f32,
        l2_weights: *const u8,
        l2_weight_stride: i32,
        l2_weights_sf: *const f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gemm_cuda(
        W: *const Half,
        X: *const Half,
        Y: *mut Half,
        M: i32,
        N: i32,
        K: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gemm_bf16_f32_cuda(
        W: *const Half,
        X: *const Half,
        Y: *mut f32,
        M: i32,
        N: i32,
        K: i32,
        stream: CUstream,
    ) -> CUresult;

    /// PF8.2 — Subtraction-merge zero-point=8 into packed INT4 weight tensor.
    /// Offline weight-prep step for W4+FP8 marlin GEMM; eliminates per-element
    /// zero-point subtract at runtime.
    /// `numel` must be a multiple of 32 (kernel processes 32 INT32 per block).
    /// Returns `cudaErrorInvalidValue` if alignment violated.
    pub fn w8a16_gemv_cuda(
        weight: *const i8,
        scales: *const Half,
        input: *const Half,
        output: *mut Half,
        n: i32,
        k: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    /// W8A16 Marlin tensor-core path (SGLang gptq_marlin, kU8B128). sm_80+.
    /// Repack: GPTQ-packed uint8b128 [k/4, n] i32 (int8+128, 4-per-word) → Marlin
    /// tile layout [k/16, n*4] i32. `out` must be `alloc_zeros` of that size.
    pub fn marlin_gptq_repack_w8a16_cuda(
        b_q_weight: *const u32,
        out: *mut u32,
        size_k: i32,
        size_n: i32,
        stream: CUstream,
    ) -> CUresult;

    /// W8A16 Marlin GEMM: C[m,n] = A[m,k] @ dequant(Marlin-packed W). BF16 acts +
    /// BF16 permuted group scales [k/group_size, n]. `c_tmp`/`workspace` are
    /// caller-owned reusable scratch (the decode loop is graph-captured — no
    /// per-call malloc): size via `marlin_w8a16_c_tmp_floats` /
    /// `marlin_w8a16_workspace_ints`, workspace zeroed once. Returns
    /// `CUDA_ERROR_NOT_SUPPORTED` below sm_80.
    pub fn marlin_w8a16_gemm_cuda(
        a: *const Half,
        b_packed: *const u32,
        scales: *const Half,
        c: *mut Half,
        c_tmp: *mut f32,
        workspace: *mut i32,
        m: i32,
        n: i32,
        k: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Float count `c_tmp` needs for `(m, sms)` (max at m>=64). Size the reusable
    /// scratch with the largest m the decode/prefill loop will pass.
    pub fn marlin_w8a16_c_tmp_floats(m: i32, sms: i32) -> i32;

    /// Int count the Marlin lock `workspace` needs for `sms`. Zero-init once.
    pub fn marlin_w8a16_workspace_ints(sms: i32) -> i32;

    pub fn w4a16_gemv_cuda(
        weight: *const u8,
        scales: *const Half,
        input: *const Half,
        output: *mut Half,
        n: i32,
        k: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn w8a16_gemv_batch_cuda(
        weight: *const i8,
        scales: *const Half,
        input: *const Half,
        output: *mut Half,
        batch_size: i32,
        n: i32,
        k: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn w4a16_gemv_batch_cuda(
        weight: *const u8,
        scales: *const Half,
        input: *const Half,
        output: *mut Half,
        batch_size: i32,
        n: i32,
        k: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_fp8_gemv_batch_cuda(
        weight: *const u8,
        scales: *const u8,
        input: *const Half,
        output: *mut Half,
        batch_size: i32,
        n: i32,
        k: i32,
        scale_rows: i32,
        scale_cols: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_fp4_gemv_batch_cuda(
        weight: *const u8,
        scales: *const u8,
        input: *const Half,
        output: *mut Half,
        batch_size: i32,
        n: i32,
        k: i32,
        scale_rows: i32,
        scale_cols: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gemv_fp8_block_scaled_cuda(
        weight: *const u8,
        scales: *const f32,
        input: *const Half,
        output: *mut Half,
        n: i32,
        k: i32,
        scale_rows: i32,
        scale_cols: i32,
        block_m: i32,
        block_k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gemv_fp8_block_scaled_batch_cuda(
        weight: *const u8,
        scales: *const f32,
        input: *const Half,
        output: *mut Half,
        batch_size: i32,
        n: i32,
        k: i32,
        scale_rows: i32,
        scale_cols: i32,
        block_m: i32,
        block_k: i32,
        stream: CUstream,
    ) -> CUresult;

    // Probe-only weight-read diagnostic (same access pattern as the batch
    // GEMV, x-work removed): mode 0 = raw bytes, mode 1 = +fp8 decode.

    // Software dequant of an FP8 E4M3 2D-block-scaled weight `[n, k]` (row-major)
    // into a dense BF16 weight `[n, k]` (`output`, `Half` = bf16 raw bits). Block
    // scale layout: `scales[(row/block_m)*scale_cols + (col/block_k)]` (matches
    // `fp8_f32_block_scale` / the autograd `fp8_block_scaled.cu` reference). Used
    // by the infer-cuda dense FP8 GEMM fallback on pre-Hopper GPUs (sm < 9.0)
    // where DeepGEMM's wgmma path is unavailable: dequant once → reuse the BF16
    // cuBLAS GEMM (`gemm_cuda`). No arch guard — runs on sm_70+.
    pub fn dequantize_fp8_block_scaled_to_bf16_cuda(
        weight: *const u8,
        scales: *const f32,
        output: *mut Half,
        n: i32,
        k: i32,
        scale_rows: i32,
        scale_cols: i32,
        block_m: i32,
        block_k: i32,
        stream: CUstream,
    ) -> CUresult;

    // BF16 → FP8 E4M3 + per-block f32 scales (amax/448). LoRA merge-requant.
    pub fn quantize_bf16_to_fp8_block_scaled_cuda(
        input: *const Half,
        weight: *mut u8,
        scales: *mut f32,
        n: i32,
        k: i32,
        block_m: i32,
        block_k: i32,
        stream: CUstream,
    ) -> CUresult;

    /// W8A16 INT8 weight → BF16 via per-row per-column-group scale. Prefill
    /// path: dequant once, then one cuBLAS BF16 GEMM over all M rows.
    pub fn dequantize_w8a16_to_bf16_cuda(
        weight: *const i8,
        scales: *const Half,
        output: *mut Half,
        n: i32,
        k: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dequantize_w4a16_to_bf16_cuda(
        weight: *const u8,
        scales: *const Half,
        output: *mut Half,
        n: i32,
        k: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dequantize_fp4_e2m1_group_to_bf16_cuda(
        weight: *const u8,
        scales: *const u8,
        global_scales: *const f32,
        output: *mut Half,
        n: i32,
        k: i32,
        group_size: i32,
        scale_cols: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gemv_fp4_e2m1_group_cuda(
        weight: *const u8,
        scales: *const u8,
        global_scales: *const f32,
        input: *const Half,
        output: *mut Half,
        n: i32,
        k: i32,
        group_size: i32,
        scale_cols: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gemv_fp4_e2m1_group_batch_cuda(
        weight: *const u8,
        scales: *const u8,
        global_scales: *const f32,
        input: *const Half,
        output: *mut Half,
        batch_size: i32,
        n: i32,
        k: i32,
        group_size: i32,
        scale_cols: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn moe_fp8_block_scaled_grouped_gemv_batch_cuda(
        weight_ptrs: *const u64,
        scale_ptrs: *const u64,
        input: *const Half,
        output: *mut Half,
        offsets: *const i32,
        counts: *const i32,
        expert_indices: *const i32,
        num_experts: i32,
        max_count: i32,
        n: i32,
        k: i32,
        scale_rows: i32,
        scale_cols: i32,
        block_m: i32,
        block_k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn moe_fp8_block_scaled_grouped_gemv_pair_batch_cuda(
        weight_a_ptrs: *const u64,
        scale_a_ptrs: *const u64,
        weight_b_ptrs: *const u64,
        scale_b_ptrs: *const u64,
        input: *const Half,
        output_a: *mut Half,
        output_b: *mut Half,
        offsets: *const i32,
        counts: *const i32,
        expert_indices: *const i32,
        num_experts: i32,
        max_count: i32,
        n: i32,
        k: i32,
        scale_rows: i32,
        scale_cols: i32,
        block_m: i32,
        block_k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn moe_fp4_e2m1_grouped_gemv_batch_cuda(
        weight_ptrs: *const u64,
        scale_ptrs: *const u64,
        global_ptrs: *const u64,
        input: *const Half,
        output: *mut Half,
        offsets: *const i32,
        counts: *const i32,
        expert_indices: *const i32,
        num_experts: i32,
        max_count: i32,
        n: i32,
        k: i32,
        group_size: i32,
        scale_cols: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn moe_fp4_e2m1_grouped_gemv_pair_batch_cuda(
        weight_a_ptrs: *const u64,
        scale_a_ptrs: *const u64,
        global_a_ptrs: *const u64,
        weight_b_ptrs: *const u64,
        scale_b_ptrs: *const u64,
        global_b_ptrs: *const u64,
        input: *const Half,
        output_a: *mut Half,
        output_b: *mut Half,
        offsets: *const i32,
        counts: *const i32,
        expert_indices: *const i32,
        num_experts: i32,
        max_count: i32,
        n: i32,
        k: i32,
        group_size: i32,
        scale_cols: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn moe_w4a16_grouped_gemv_batch_cuda(
        weight_ptrs: *const u64,
        scale_ptrs: *const u64,
        input: *const Half,
        output: *mut Half,
        offsets: *const i32,
        counts: *const i32,
        expert_indices: *const i32,
        num_experts: i32,
        max_count: i32,
        n: i32,
        k: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn moe_w4a16_grouped_gemv_pair_batch_cuda(
        weight_a_ptrs: *const u64,
        scale_a_ptrs: *const u64,
        weight_b_ptrs: *const u64,
        scale_b_ptrs: *const u64,
        input: *const Half,
        output_a: *mut Half,
        output_b: *mut Half,
        offsets: *const i32,
        counts: *const i32,
        expert_indices: *const i32,
        num_experts: i32,
        max_count: i32,
        n: i32,
        k: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_fp8_grouped_swiglu_decode_cuda(
        weight_gate_ptrs: *const u64,
        scale_gate_ptrs: *const u64,
        weight_up_ptrs: *const u64,
        scale_up_ptrs: *const u64,
        input: *const Half,
        act: *mut Half,
        offsets: *const i32,
        counts: *const i32,
        expert_indices: *const i32,
        num_experts: i32,
        max_count: i32,
        n: i32,
        k: i32,
        scale_cols: i32,
        limit: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_fp8_grouped_down_decode_cuda(
        weight_ptrs: *const u64,
        scale_ptrs: *const u64,
        input: *const Half,
        output: *mut Half,
        offsets: *const i32,
        counts: *const i32,
        expert_indices: *const i32,
        num_experts: i32,
        max_count: i32,
        n: i32,
        k: i32,
        scale_cols: i32,
        stream: CUstream,
    ) -> CUresult;

    // ── Qwen3.5-MoE / Qwen3.6 single-GPU BF16 grouped expert GEMM ──────────
    // Mirrors the DSv4 FP8 grouped GEMM structure (M-grouping by expert,
    // CUDA-core warp-reduce, sm_70-safe) but with plain BF16 weights — no
    // FP8/E8M0 decode, so no scale pointers / scale_rows / scale_cols.
    // (W4 nibble-decode variant = explicit follow-up.)
    pub fn moe_bf16_grouped_gemm_batch_cuda(
        weight_ptrs: *const u64,
        input: *const Half,
        output: *mut Half,
        offsets: *const i32,
        counts: *const i32,
        expert_indices: *const i32,
        num_experts: i32,
        max_count: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn moe_bf16_grouped_gemm_pair_batch_cuda(
        weight_a_ptrs: *const u64,
        weight_b_ptrs: *const u64,
        input: *const Half,
        output_a: *mut Half,
        output_b: *mut Half,
        offsets: *const i32,
        counts: *const i32,
        expert_indices: *const i32,
        num_experts: i32,
        max_count: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    // Decode-band (small routed-row count) weight-read-bound variants: one
    // warp per weight row with 16B-vector coalesced loads, every touched
    // expert's weight rows read exactly once per <=8-row activation chunk
    // regardless of R. The swiglu variant fuses the gate+up pair AND the
    // SwiGLU epilogue (act = silu(gate)*up, single output buffer — no
    // separate silu_mul pass). Requires k % 8 == 0 (launcher rejects with
    // cudaErrorInvalidValue; the Rust dispatch routes to the batch kernels
    // instead).
    pub fn moe_bf16_grouped_gemm_decode_cuda(
        weight_ptrs: *const u64,
        input: *const Half,
        output: *mut Half,
        offsets: *const i32,
        counts: *const i32,
        expert_indices: *const i32,
        num_experts: i32,
        max_count: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn moe_bf16_grouped_gemm_swiglu_decode_cuda(
        weight_gate_ptrs: *const u64,
        weight_up_ptrs: *const u64,
        input: *const Half,
        act: *mut Half,
        offsets: *const i32,
        counts: *const i32,
        expert_indices: *const i32,
        num_experts: i32,
        max_count: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_fp8_route_gemv_batch_cuda(
        weight_ptrs: *const u64,
        scale_ptrs: *const u64,
        input: *const Half,
        output: *mut Half,
        route_meta: *const i32,
        local_expert_start: i32,
        experts_per_rank: i32,
        num_routes: i32,
        n: i32,
        k: i32,
        scale_rows: i32,
        scale_cols: i32,
        apply_route_weight: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_fp4_route_gemv_batch_cuda(
        weight_ptrs: *const u64,
        scale_ptrs: *const u64,
        input: *const Half,
        output: *mut Half,
        route_meta: *const i32,
        local_expert_start: i32,
        experts_per_rank: i32,
        num_routes: i32,
        n: i32,
        k: i32,
        scale_rows: i32,
        scale_cols: i32,
        apply_route_weight: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_block_scaled_to_fp8_deepgemm_cuda(
        weight: *const u8,
        src_scales: *const u8,
        dst_weight: *mut u8,
        dst_scales: *mut f32,
        rows: i32,
        cols: i32,
        scale_rows: i32,
        scale_cols: i32,
        dst_scale_cols: i32,
        source_format: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_deepgemm_pack_quantize_bf16_to_fp8_cuda(
        input: *const Half,
        output: *mut u8,
        scales: *mut f32,
        active_experts: *const i32,
        active_offsets: *const i32,
        active_counts: *const i32,
        active_count: i32,
        max_m: i32,
        cols: i32,
        scale_stride_m: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_deepgemm_swiglu_quantize_w13_cuda(
        w13: *const Half,
        act: *mut u8,
        scales: *mut f32,
        active_experts: *const i32,
        active_counts: *const i32,
        active_count: i32,
        max_m: i32,
        intermediate_dim: i32,
        scale_stride_m: i32,
        limit: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_deepgemm_silu_mul_masked_quant_cuda(
        input: *const Half,
        out_fp8: *mut u8,
        out_scale: *mut f32,
        masked_m: *const i32,
        expert_num: i32,
        token_num_padded: i32,
        expected_m: i32,
        hidden_dim: i32,
        swiglu_limit: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_deepgemm_unpad_grouped_bf16_cuda(
        grouped: *const Half,
        compact: *mut Half,
        active_experts: *const i32,
        active_offsets: *const i32,
        active_counts: *const i32,
        active_count: i32,
        max_m: i32,
        hidden_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_deepgemm_native_preflight_cuda(
        out: *mut std::ffi::c_char,
        out_len: usize,
    ) -> CUresult;

    pub fn dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked_cuda(
        a: *const u8,
        sfa: *const f32,
        b: *const u8,
        sfb: *const f32,
        d: *mut Half,
        masked_m: *const i32,
        num_groups: i32,
        m: i32,
        n: i32,
        k: i32,
        sfa_aligned_m: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous_cuda(
        a: *const u8,
        sfa: *const f32,
        b: *const u8,
        sfb: *const f32,
        d: *mut Half,
        m_indices: *const i32,
        num_groups: i32,
        m: i32,
        n: i32,
        k: i32,
        sfa_aligned_m: i32,
        mk_align: i32,
        stream: CUstream,
    ) -> CUresult;

    // sm_120 (Blackwell) grouped blockwise-scaled FP8 MoE GEMM — the sm_120
    // replacement for the Hopper-only DeepGEMM contiguous grouped call. Group
    // geometry arrives HOST-resident (`host_group_offsets` 128-aligned row start
    // + `host_group_counts` real Mg) — the caller D2H's it once per layer and
    // reuses it across the w13 + down GEMMs, so the kernel does no per-call
    // D2H/sync. `scale_stride_m` is the SFA K-block leading dim (= DeepGEMM's
    // `sfa_aligned_m`). Non-sm120 builds link a stub returning
    // `cudaErrorNotSupported`.
    pub fn arle_fp8_moe_grouped_gemm_nt_sm120_cuda(
        a: *const u8,
        sfa: *const f32,
        b: *const u8,
        sfb: *const f32,
        d: *mut Half,
        host_group_offsets: *const i32,
        host_group_counts: *const i32,
        num_groups: i32,
        n: i32,
        k: i32,
        scale_stride_m: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepgemm_m_grouped_bf16_gemm_nt_masked_cuda(
        a: *const Half,
        b: *const Half,
        d: *mut Half,
        masked_m: *const i32,
        num_groups: i32,
        m: i32,
        n: i32,
        k: i32,
        expected_m: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn deepgemm_m_grouped_bf16_gemm_nt_contiguous_cuda(
        a: *const Half,
        b: *const Half,
        d: *mut Half,
        m_indices: *const i32,
        num_groups: i32,
        m: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_deepgemm_fp8_gemm_nt_cuda(
        a: *const u8,
        sfa: *const f32,
        b: *const u8,
        sfb: *const f32,
        d: *mut Half,
        m: i32,
        n: i32,
        k: i32,
        sfa_aligned_m: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_deepgemm_paged_mqa_logits_metadata_cuda(
        context_lens: *const i32,
        schedule_metadata: *mut i32,
        batch_size: i32,
        next_n: i32,
        block_kv: i32,
        num_sms: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_deepgemm_fp8_paged_mqa_logits_cuda(
        q: *const u8,
        kv_cache: *const u8,
        kv_cache_scales: *const f32,
        weights: *const f32,
        context_lens: *const i32,
        block_table: *const i32,
        schedule_meta: *const i32,
        logits: *mut f32,
        batch_size: i32,
        next_n: i32,
        num_heads: i32,
        head_dim: i32,
        num_kv_blocks: i32,
        block_kv: i32,
        max_context_len: i32,
        logits_stride: i32,
        block_table_stride: i32,
        kv_cache_stride_bytes: i32,
        num_sms: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn dsv4_deepgemm_fp8_paged_mqa_logits_fused_cache_cuda(
        q: *const u8,
        kv_cache_with_scale: *const u8,
        weights: *const f32,
        context_lens: *const i32,
        block_table: *const i32,
        schedule_meta: *const i32,
        logits: *mut f32,
        batch_size: i32,
        next_n: i32,
        num_heads: i32,
        head_dim: i32,
        num_kv_blocks: i32,
        block_kv: i32,
        max_context_len: i32,
        logits_stride: i32,
        block_table_stride: i32,
        kv_cache_stride_bytes: i32,
        num_sms: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q6k_gemv_cuda(
        weight: *const u8,
        input: *const Half,
        output: *mut Half,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q6k_gemv_batch_cuda(
        weight: *const u8,
        input: *const Half,
        output: *mut Half,
        batch_size: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q6k_dequant_chunk_cuda(
        weight: *const u8,
        out_bf16: *mut Half,
        n: i32,
        k: i32,
        k_start: i32,
        k_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q4k_gemv_cuda(
        weight: *const u8,
        input: *const Half,
        output: *mut Half,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q4k_gemv_batch_cuda(
        weight: *const u8,
        input: *const Half,
        output: *mut Half,
        batch_size: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q4k_dequant_chunk_cuda(
        weight: *const u8,
        out_bf16: *mut Half,
        n: i32,
        k: i32,
        k_start: i32,
        k_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q5k_gemv_cuda(
        weight: *const u8,
        input: *const Half,
        output: *mut Half,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q5k_gemv_batch_cuda(
        weight: *const u8,
        input: *const Half,
        output: *mut Half,
        batch_size: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q5k_dequant_chunk_cuda(
        weight: *const u8,
        out_bf16: *mut Half,
        n: i32,
        k: i32,
        k_start: i32,
        k_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn turboquant_weight_gemv_cuda(
        packed: *const u8,
        scales: *const Half, // f16
        signs: *const i8,
        centroids: *const f32,
        x: *const Half, // bf16
        y: *mut Half,   // bf16
        N: i32,
        K: i32,
        group_size: i32,
        packed_cols: i32,
        num_groups: i32,
        bits: i32,
        stream: CUstream,
    );

    pub fn turboquant_weight_dequant_cuda(
        packed: *const u8,
        scales: *const Half,
        signs: *const i8,
        centroids: *const f32,
        out: *mut Half, // bf16
        N: i32,
        K: i32,
        group_size: i32,
        packed_cols: i32,
        num_groups: i32,
        bits: i32,
        stream: CUstream,
    );
}

#[cfg(test)]
#[path = "gemm_tests.rs"]
mod tests;
