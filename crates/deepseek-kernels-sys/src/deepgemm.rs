//! DeepGEMM SM90 bridge C ABI (`csrc/gemm/deepgemm_native.cu`). The
//! `deepgemm_bridge_stub.cu` fallback exports the same symbols and fails
//! `dsv4_deepgemm_native_preflight_cuda`, so availability is a runtime probe.

use crate::{CUresult, CUstream, Half};

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
}
