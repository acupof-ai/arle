//! Pre-allocated scratch buffers for Qwen3.5 prefill-only chunk-wise operators.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;

use super::config::Config35;
use crate::model::cuda_graph::CudaGraphState;
use cuda_kernels::flashinfer::BatchPrefillPagedPlan;
use cuda_kernels::prelude::{DeviceContext, DeviceVec, HiddenStates};

/// Scratch buffers for a single Qwen3.5 linear-attention chunk-wise GDR prefill call.
///
/// The first implementation target is intentionally narrow:
/// - batch size = 1
/// - fixed Qwen3.5 linear-attention shapes
/// - forward-only
/// - chunk_size = 64
///
/// Buffers are explicit because the chunk-wise path is naturally a multi-stage
/// pipeline rather than one opaque kernel launch.
pub struct GdrChunkwiseScratch35 {
    /// Chunk-local cumulative gate, fp32: [seq_len, num_value_heads]
    pub g_cumsum: CudaSlice<f32>,
    /// Beta values, fp32: [seq_len, num_value_heads]
    pub beta: CudaSlice<f32>,

    /// Expanded + normalized q in token-major layout: [seq_len, num_value_heads * key_dim]
    pub q_expanded: HiddenStates,
    /// Expanded + normalized k in token-major layout: [seq_len, num_value_heads * key_dim]
    pub k_expanded: HiddenStates,
    /// Raw v in token-major layout: [seq_len, num_value_heads * value_dim]
    pub v_raw: HiddenStates,

    /// Chunk attention matrix storage, fp32: [seq_len, num_value_heads, chunk_size]
    pub a_tril: CudaSlice<f32>,
    /// Inverse (I + A)^-1 in bf16: [seq_len, num_value_heads, chunk_size]
    pub a_inv: CudaSlice<bf16>,

    /// Prepared W tensor in token-major layout: [seq_len, num_value_heads * key_dim]
    pub w: HiddenStates,
    /// Prepared U tensor in token-major layout: [seq_len, num_value_heads * value_dim]
    pub u: HiddenStates,
    /// New value tensor consumed by chunk output stage: [seq_len, num_value_heads * value_dim]
    pub v_new: HiddenStates,

    /// Per-chunk recurrent state snapshots, fp32: [num_chunks, num_value_heads, key_dim, value_dim]
    pub chunk_state: CudaSlice<f32>,
}

impl GdrChunkwiseScratch35 {
    pub const CHUNK_SIZE: usize = 64;

    pub(crate) fn new(ctx: &DeviceContext, config: &Config35, seq_len: usize) -> Result<Self> {
        Self::from_dims(
            ctx,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            seq_len,
        )
    }

    pub fn from_dims(
        ctx: &DeviceContext,
        num_value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let kv_hidden_dim = num_value_heads * key_dim;
        let vv_hidden_dim = num_value_heads * value_dim;
        let num_chunks = seq_len.div_ceil(Self::CHUNK_SIZE);

        let g_cumsum: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(seq_len * num_value_heads)
            .map_err(|e| anyhow::anyhow!("Alloc g_cumsum failed: {}", e))?;
        let beta: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(seq_len * num_value_heads)
            .map_err(|e| anyhow::anyhow!("Alloc beta failed: {}", e))?;
        let a_tril: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(seq_len * num_value_heads * Self::CHUNK_SIZE)
            .map_err(|e| anyhow::anyhow!("Alloc a_tril failed: {}", e))?;
        let a_inv: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(seq_len * num_value_heads * Self::CHUNK_SIZE)
            .map_err(|e| anyhow::anyhow!("Alloc a_inv failed: {}", e))?;
        let chunk_state: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(num_chunks * num_value_heads * value_dim * key_dim)
            .map_err(|e| anyhow::anyhow!("Alloc chunk_state failed: {}", e))?;

        Ok(Self {
            g_cumsum,
            beta,
            q_expanded: HiddenStates::zeros(ctx, kv_hidden_dim, seq_len)?,
            k_expanded: HiddenStates::zeros(ctx, kv_hidden_dim, seq_len)?,
            v_raw: HiddenStates::zeros(ctx, vv_hidden_dim, seq_len)?,
            a_tril,
            a_inv,
            w: HiddenStates::zeros(ctx, kv_hidden_dim, seq_len)?,
            u: HiddenStates::zeros(ctx, vv_hidden_dim, seq_len)?,
            v_new: HiddenStates::zeros(ctx, vv_hidden_dim, seq_len)?,
            chunk_state,
        })
    }

    pub fn num_chunks(seq_len: usize) -> usize {
        seq_len.div_ceil(Self::CHUNK_SIZE)
    }
}

pub(super) struct PagedPrefillMetadata35 {
    pub token_ids_gpu: CudaSlice<i32>,
    pub page_indices_gpu: CudaSlice<i32>,
    pub qo_indptr_gpu: CudaSlice<i32>,
    pub kv_indptr_gpu: CudaSlice<i32>,
    pub kv_last_page_len_gpu: CudaSlice<i32>,
    pub num_pages: usize,
    token_ids_host: Vec<i32>,
    page_indices_host: Vec<i32>,
}

impl PagedPrefillMetadata35 {
    fn new(ctx: &DeviceContext, seq_len: usize, initial_pages: usize) -> Result<Self> {
        let token_ids_gpu = ctx
            .stream
            .alloc_zeros(seq_len)
            .map_err(|e| anyhow::anyhow!("Alloc token_ids failed: {e}"))?;
        let page_indices_gpu = ctx
            .stream
            .alloc_zeros(initial_pages.max(1))
            .map_err(|e| anyhow::anyhow!("Alloc page_indices failed: {e}"))?;
        let qo_indptr_gpu = ctx
            .stream
            .alloc_zeros(2)
            .map_err(|e| anyhow::anyhow!("Alloc qo_indptr failed: {e}"))?;
        let kv_indptr_gpu = ctx
            .stream
            .alloc_zeros(2)
            .map_err(|e| anyhow::anyhow!("Alloc kv_indptr failed: {e}"))?;
        let kv_last_page_len_gpu = ctx
            .stream
            .alloc_zeros(1)
            .map_err(|e| anyhow::anyhow!("Alloc kv_last_page_len failed: {e}"))?;

        Ok(Self {
            token_ids_gpu,
            page_indices_gpu,
            qo_indptr_gpu,
            kv_indptr_gpu,
            kv_last_page_len_gpu,
            num_pages: 0,
            token_ids_host: vec![0; seq_len],
            page_indices_host: Vec::with_capacity(initial_pages.max(1)),
        })
    }

    pub(super) fn update(
        &mut self,
        ctx: &DeviceContext,
        token_ids: &[u32],
        page_indices: &[u32],
        seq_len: usize,
        page_size: usize,
        start_pos: usize,
    ) -> Result<bool> {
        debug_assert_eq!(token_ids.len(), seq_len);

        for (dst, &token) in self.token_ids_host.iter_mut().zip(token_ids) {
            *dst = token as i32;
        }
        ctx.stream
            .memcpy_htod(&self.token_ids_host, &mut self.token_ids_gpu)
            .map_err(|e| anyhow::anyhow!("token_ids H2D failed: {e}"))?;

        let num_pages = page_indices.len();
        let mut page_indices_reallocated = false;
        if self.page_indices_gpu.len() < num_pages.max(1) {
            self.page_indices_gpu = ctx
                .stream
                .alloc_zeros(num_pages.max(1))
                .map_err(|e| anyhow::anyhow!("Realloc page_indices failed: {e}"))?;
            self.page_indices_host = Vec::with_capacity(num_pages.max(1));
            page_indices_reallocated = true;
        }

        self.page_indices_host.clear();
        self.page_indices_host
            .extend(page_indices.iter().map(|&page| page as i32));
        let mut page_indices_view = self.page_indices_gpu.slice_mut(..num_pages);
        ctx.stream
            .memcpy_htod(&self.page_indices_host, &mut page_indices_view)
            .map_err(|e| anyhow::anyhow!("page_indices H2D failed: {e}"))?;

        let qo_indptr = [0_i32, seq_len as i32];
        let kv_indptr = [0_i32, num_pages as i32];
        let kv_last_page_len = [if seq_len == 0 {
            0
        } else {
            ((start_pos + seq_len - 1) % page_size + 1) as i32
        }];
        ctx.stream
            .memcpy_htod(&qo_indptr, &mut self.qo_indptr_gpu)
            .map_err(|e| anyhow::anyhow!("qo_indptr H2D failed: {e}"))?;
        ctx.stream
            .memcpy_htod(&kv_indptr, &mut self.kv_indptr_gpu)
            .map_err(|e| anyhow::anyhow!("kv_indptr H2D failed: {e}"))?;
        ctx.stream
            .memcpy_htod(&kv_last_page_len, &mut self.kv_last_page_len_gpu)
            .map_err(|e| anyhow::anyhow!("kv_last_page_len H2D failed: {e}"))?;
        self.num_pages = num_pages;

        Ok(page_indices_reallocated)
    }
}

pub(super) struct PagedPrefillBuffers35 {
    pub seq_len: usize,
    pub page_size: usize,
    pub hidden: HiddenStates,
    pub hidden_next: HiddenStates,
    pub normed: HiddenStates,
    pub q_full: HiddenStates,
    pub k_attn: HiddenStates,
    pub v_attn: HiddenStates,
    pub q_prepped: HiddenStates,
    pub attn_out_full: HiddenStates,
    pub qkv: HiddenStates,
    pub z: HiddenStates,
    pub b_proj: HiddenStates,
    pub a_proj: HiddenStates,
    pub qkv_conv: HiddenStates,
    pub gdr_out: HiddenStates,
    pub normed_gated: HiddenStates,
    pub attn_results: HiddenStates,
    pub hidden_mid: HiddenStates,
    pub gate_out: HiddenStates,
    pub up_out: HiddenStates,
    pub act_out: HiddenStates,
    pub mlp_out: HiddenStates,
    pub last_hidden: DeviceVec,
    pub last_normed: DeviceVec,
    pub logits: DeviceVec,
    pub logits_valid: bool,
    pub gdr_chunkwise_scratch: GdrChunkwiseScratch35,
    pub metadata: PagedPrefillMetadata35,
    pub plan: BatchPrefillPagedPlan,
    pub graph_state: CudaGraphState,
}

impl PagedPrefillBuffers35 {
    pub(super) fn new(
        ctx: &DeviceContext,
        config: &Config35,
        seq_len: usize,
        page_size: usize,
    ) -> Result<Self> {
        let hidden = config.hidden_size;
        let q_proj_dim = config.full_attn_q_proj_dim();
        let q_dim = config.full_attn_q_dim();
        let kv_dim = config.full_attn_kv_dim();
        let qkv_dim = config.linear_attn_qkv_dim();
        let z_dim = config.linear_attn_z_dim();
        let inter = config.intermediate_size;
        let num_pages = seq_len.div_ceil(page_size).max(1);

        Ok(Self {
            seq_len,
            page_size,
            hidden: HiddenStates::zeros(ctx, hidden, seq_len)?,
            hidden_next: HiddenStates::zeros(ctx, hidden, seq_len)?,
            normed: HiddenStates::zeros(ctx, hidden, seq_len)?,
            q_full: HiddenStates::zeros(ctx, q_proj_dim, seq_len)?,
            k_attn: HiddenStates::zeros(ctx, kv_dim, seq_len)?,
            v_attn: HiddenStates::zeros(ctx, kv_dim, seq_len)?,
            q_prepped: HiddenStates::zeros(ctx, q_dim, seq_len)?,
            attn_out_full: HiddenStates::zeros(ctx, q_dim, seq_len)?,
            qkv: HiddenStates::zeros(ctx, qkv_dim, seq_len)?,
            z: HiddenStates::zeros(ctx, z_dim, seq_len)?,
            b_proj: HiddenStates::zeros(ctx, config.linear_num_value_heads, seq_len)?,
            a_proj: HiddenStates::zeros(ctx, config.linear_num_value_heads, seq_len)?,
            qkv_conv: HiddenStates::zeros(ctx, qkv_dim, seq_len)?,
            gdr_out: HiddenStates::zeros(ctx, z_dim, seq_len)?,
            normed_gated: HiddenStates::zeros(ctx, z_dim, seq_len)?,
            attn_results: HiddenStates::zeros(ctx, hidden, seq_len)?,
            hidden_mid: HiddenStates::zeros(ctx, hidden, seq_len)?,
            gate_out: HiddenStates::zeros(ctx, inter, seq_len)?,
            up_out: HiddenStates::zeros(ctx, inter, seq_len)?,
            act_out: HiddenStates::zeros(ctx, inter, seq_len)?,
            mlp_out: HiddenStates::zeros(ctx, hidden, seq_len)?,
            last_hidden: DeviceVec::zeros(ctx, hidden)?,
            last_normed: DeviceVec::zeros(ctx, hidden)?,
            logits: DeviceVec::zeros(ctx, config.vocab_size)?,
            logits_valid: false,
            gdr_chunkwise_scratch: GdrChunkwiseScratch35::new(ctx, config, seq_len)?,
            metadata: PagedPrefillMetadata35::new(ctx, seq_len, num_pages)?,
            plan: BatchPrefillPagedPlan::new_hd256(
                ctx,
                seq_len.max(4096),
                config.num_attention_heads,
            )?,
            graph_state: CudaGraphState::new(),
        })
    }

    pub(super) fn matches_shape(&self, seq_len: usize, page_size: usize) -> bool {
        self.seq_len == seq_len && self.page_size == page_size
    }

    pub(super) fn invalidate_graph(&mut self) {
        self.graph_state = CudaGraphState::new();
    }

    pub(super) fn clear_logits(&mut self) {
        self.logits_valid = false;
    }
}
