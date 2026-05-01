use anyhow::{Result, bail};

use super::{CUstream, Half};

#[derive(Clone, Copy, Debug)]
pub struct MlaDecodePagedBf16Params {
    pub q_nope_or_absorbed: *const Half,
    pub q_rope: *const Half,
    pub ckv_pool: *const Half,
    pub kpe_pool: *const Half,
    pub w_uv_or_absorbed: *const Half,
    pub qo_indptr: *const i32,
    pub kv_indptr: *const i32,
    pub kv_indices: *const i32,
    pub last_page_len: *const i32,
    pub out: *mut Half,
    pub total_q_tokens: i32,
    pub batch_size: i32,
    pub num_q_heads: i32,
    pub kv_lora_rank: i32,
    pub qk_rope_head_dim: i32,
    pub v_head_dim: i32,
    pub page_size: i32,
    pub sm_scale: f32,
    pub num_splits: i32,
    pub stream: CUstream,
}

#[allow(dead_code)]
unsafe extern "C" {
    pub fn mla_decode_paged_bf16_cuda(
        q_nope_or_absorbed: *const Half,
        q_rope: *const Half,
        ckv_pool: *const Half,
        kpe_pool: *const Half,
        w_uv_or_absorbed: *const Half,
        qo_indptr: *const i32,
        kv_indptr: *const i32,
        kv_indices: *const i32,
        last_page_len: *const i32,
        out: *mut Half,
        total_q_tokens: i32,
        batch_size: i32,
        num_q_heads: i32,
        kv_lora_rank: i32,
        qk_rope_head_dim: i32,
        v_head_dim: i32,
        page_size: i32,
        sm_scale: f32,
        num_splits: i32,
        stream: CUstream,
    ) -> i32;
}

pub fn mla_decode_paged_bf16(_params: MlaDecodePagedBf16Params) -> Result<()> {
    bail!("MLA kernel not yet implemented")
}
