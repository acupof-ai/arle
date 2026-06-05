//! Paged attention kernel-call paths for the dense-BF16 Qwen3 forward (HOT axis).
//!
//! Prep kernels fuse Q/K RMSNorm + RoPE + KV-cache write; the TileLang kernels
//! run the HD128/kv8 paged attention.

use anyhow::{Result, anyhow, bail, ensure};
use cuda_kernels::attention as flash_kv;
use cuda_kernels::ffi;
use cuda_kernels::moe as cuda_moe;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates, PagedKVPool};
use cuda_kernels::tensor::{WeightFormat, cache_ptr};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use deepseek_spec::{DeepSeekV4AttentionMode, DeepSeekV4Config};
use std::sync::atomic::{AtomicI8, Ordering};

use crate::dsv4::{
    Dsv4Attention, Dsv4Compressor, Dsv4ForwardKeepalive, Dsv4Indexer, Dsv4MlaKvArena,
};
use crate::loader::PageMeta;
use crate::tp::TpRuntime;

const DSV4_FLASHMLA_MODEL1: i32 = 1;
const DSV4_FLASHMLA_S_Q: usize = 1;
const DSV4_FLASHMLA_OVERRIDE_ENV: i8 = -1;
const DSV4_FLASHMLA_OVERRIDE_OFF: i8 = 0;
const DSV4_FLASHMLA_OVERRIDE_ON: i8 = 1;

static DSV4_FLASHMLA_DECODE_OVERRIDE: AtomicI8 = AtomicI8::new(DSV4_FLASHMLA_OVERRIDE_ENV);
static DSV4_FUSED_WQKV_DECODE_OVERRIDE: AtomicI8 = AtomicI8::new(DSV4_FLASHMLA_OVERRIDE_ENV);

pub(crate) fn set_dsv4_flashmla_decode_override(enabled: Option<bool>) {
    let value = match enabled {
        Some(true) => DSV4_FLASHMLA_OVERRIDE_ON,
        Some(false) => DSV4_FLASHMLA_OVERRIDE_OFF,
        None => DSV4_FLASHMLA_OVERRIDE_ENV,
    };
    DSV4_FLASHMLA_DECODE_OVERRIDE.store(value, Ordering::Relaxed);
}

pub(crate) fn set_dsv4_fused_wqkv_decode_override(enabled: Option<bool>) {
    let value = match enabled {
        Some(true) => DSV4_FLASHMLA_OVERRIDE_ON,
        Some(false) => DSV4_FLASHMLA_OVERRIDE_OFF,
        None => DSV4_FLASHMLA_OVERRIDE_ENV,
    };
    DSV4_FUSED_WQKV_DECODE_OVERRIDE.store(value, Ordering::Relaxed);
}

pub(crate) struct Dsv4CompressorState {
    pending_kv: CudaSlice<half::bf16>,
    pending_score: CudaSlice<half::bf16>,
    prev_overlap_kv: CudaSlice<half::bf16>,
    prev_overlap_score: CudaSlice<half::bf16>,
    compressed: HiddenStates,
}

impl Dsv4CompressorState {
    fn new(
        ctx: &DeviceContext,
        head_dim: usize,
        ratio: usize,
        overlap: bool,
        max_seq_len: usize,
    ) -> Result<Self> {
        let width = if overlap { 2 * head_dim } else { head_dim };
        let compressed_rows = max_seq_len.div_ceil(ratio).max(1);
        Ok(Self {
            pending_kv: ctx
                .stream
                .alloc_zeros::<half::bf16>(ratio * width)
                .map_err(|e| anyhow::anyhow!("DSv4 compressor pending kv alloc failed: {e}"))?,
            pending_score: ctx
                .stream
                .alloc_zeros::<half::bf16>(ratio * width)
                .map_err(|e| anyhow::anyhow!("DSv4 compressor pending score alloc failed: {e}"))?,
            prev_overlap_kv: ctx
                .stream
                .alloc_zeros::<half::bf16>(ratio * head_dim)
                .map_err(|e| anyhow::anyhow!("DSv4 compressor prev kv alloc failed: {e}"))?,
            prev_overlap_score: ctx
                .stream
                .alloc_zeros::<half::bf16>(ratio * head_dim)
                .map_err(|e| anyhow::anyhow!("DSv4 compressor prev score alloc failed: {e}"))?,
            compressed: HiddenStates::zeros(ctx, head_dim, compressed_rows)?,
        })
    }

    fn reset(&mut self, ctx: &DeviceContext) -> Result<()> {
        ctx.stream
            .memset_zeros(&mut self.pending_kv)
            .map_err(|e| anyhow::anyhow!("DSv4 compressor pending kv reset failed: {e}"))?;
        ctx.stream
            .memset_zeros(&mut self.pending_score)
            .map_err(|e| anyhow::anyhow!("DSv4 compressor pending score reset failed: {e}"))?;
        ctx.stream
            .memset_zeros(&mut self.prev_overlap_kv)
            .map_err(|e| anyhow::anyhow!("DSv4 compressor prev kv reset failed: {e}"))?;
        ctx.stream
            .memset_zeros(&mut self.prev_overlap_score)
            .map_err(|e| anyhow::anyhow!("DSv4 compressor prev score reset failed: {e}"))?;
        ctx.stream
            .memset_zeros(&mut self.compressed.data)
            .map_err(|e| anyhow::anyhow!("DSv4 compressor compressed reset failed: {e}"))?;
        self.compressed.seq_len = 0;
        Ok(())
    }
}

struct Dsv4FlashMlaDecodeState {
    fp8_kv_pool: CudaSlice<u8>,
    sw_blocks: usize,
    comp_blocks: usize,
    max_compressed_keys: usize,
    topk_unified: usize,
    fp8_kv_sw_bootstrapped: bool,
    fp8_kv_comp_packed_rows: usize,
    sw_bulk_block_ids: CudaSlice<i32>,
    sw_bulk_rows: CudaSlice<i32>,
    one_block_id: CudaSlice<i32>,
    one_row: CudaSlice<i32>,
    comp_block_ids: CudaSlice<i32>,
    comp_rows: CudaSlice<i32>,
    indices: CudaSlice<i32>,
    topk_length: CudaSlice<i32>,
    lse_out: CudaSlice<f32>,
    lse_accum: CudaSlice<f32>,
    o_accum: CudaSlice<f32>,
    sched_meta: CudaSlice<i32>,
    num_splits: CudaSlice<i32>,
    tp_gathered_q: CudaSlice<half::bf16>,
    tp_packed_q: CudaSlice<half::bf16>,
    tp_full_out: CudaSlice<half::bf16>,
    num_sm_parts: i32,
    fixed_overhead_num_blocks: i32,
    block_size_topk: i32,
}

impl Dsv4FlashMlaDecodeState {
    #[allow(clippy::too_many_arguments)]
    fn new(
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        mode: DeepSeekV4AttentionMode,
        compress_ratio: usize,
        max_seq_len: usize,
        kv_arena: &Dsv4MlaKvArena,
        local_heads: usize,
        tp_world: usize,
    ) -> Result<Self> {
        ensure!(
            config.head_dim == 512 && kv_arena.bytes_per_token == 584,
            "DSv4 FlashMLA decode only wires MODEL1 head_dim=512 bytes/token=584"
        );
        ensure!(
            local_heads > 0 && tp_world > 0,
            "DSv4 FlashMLA decode requires non-zero local_heads and tp_world"
        );
        let h_q = local_heads
            .checked_mul(tp_world)
            .ok_or_else(|| anyhow!("DSv4 FlashMLA h_q overflow"))?;
        ensure!(
            matches!(h_q, 64 | 128),
            "DSv4 FlashMLA decode requires global h_q 64 or 128, got {h_q}"
        );
        ensure!(
            kv_arena.page_block_size == 64,
            "DSv4 FlashMLA decode requires page_block_size=64"
        );

        let sw_blocks = config.sliding_window.div_ceil(kv_arena.page_block_size);
        let compressed_rows = if mode == DeepSeekV4AttentionMode::SlidingWindow {
            0
        } else {
            ensure!(
                compress_ratio > 0,
                "DSv4 FlashMLA compressed decode requires non-zero ratio"
            );
            max_seq_len.div_ceil(compress_ratio).max(1)
        };
        let comp_blocks = compressed_rows.div_ceil(kv_arena.page_block_size);
        let max_compressed_keys = match mode {
            DeepSeekV4AttentionMode::SlidingWindow => 0,
            DeepSeekV4AttentionMode::CompressedSparse => config.index_topk,
            DeepSeekV4AttentionMode::HybridCompressed => compressed_rows.div_ceil(128) * 128,
        };
        let topk_unified = config
            .sliding_window
            .checked_add(max_compressed_keys)
            .ok_or_else(|| anyhow!("DSv4 FlashMLA topk_unified overflow"))?;
        ensure!(
            topk_unified.is_multiple_of(128),
            "DSv4 FlashMLA topk_unified {topk_unified} must be multiple of 128"
        );
        let total_blocks = sw_blocks
            .checked_add(comp_blocks)
            .ok_or_else(|| anyhow!("DSv4 FlashMLA total block overflow"))?;

        let mut num_sm_parts = 0_i32;
        let mut fixed_overhead_num_blocks = 0_i32;
        let mut block_size_topk = 0_i32;
        unsafe {
            ffi::arle_flashmla_sm90_sparse_decode_get_meta(
                h_q as i32,
                DSV4_FLASHMLA_S_Q as i32,
                DSV4_FLASHMLA_MODEL1,
                &mut num_sm_parts,
                &mut fixed_overhead_num_blocks,
                &mut block_size_topk,
            )
            .result()
            .map_err(|e| anyhow!("DSv4 FlashMLA decode meta failed: {e}"))?;
        }
        let num_sm_parts_max = (num_sm_parts as usize).max(256);
        let h_q_d = h_q
            .checked_mul(config.head_dim)
            .ok_or_else(|| anyhow!("DSv4 FlashMLA h_q*d overflow"))?;
        let accum_rows = num_sm_parts_max + 1;
        let sw_slots = config.sliding_window;
        let comp_slots = compressed_rows.max(1);

        Ok(Self {
            fp8_kv_pool: kv_arena.alloc_fp8_arena(ctx, total_blocks)?,
            sw_blocks,
            comp_blocks,
            max_compressed_keys,
            topk_unified,
            fp8_kv_sw_bootstrapped: false,
            fp8_kv_comp_packed_rows: 0,
            sw_bulk_block_ids: ctx.stream.alloc_zeros::<i32>(sw_slots)?,
            sw_bulk_rows: ctx.stream.alloc_zeros::<i32>(sw_slots)?,
            one_block_id: ctx.stream.alloc_zeros::<i32>(1)?,
            one_row: ctx.stream.alloc_zeros::<i32>(1)?,
            comp_block_ids: ctx.stream.alloc_zeros::<i32>(comp_slots)?,
            comp_rows: ctx.stream.alloc_zeros::<i32>(comp_slots)?,
            indices: ctx.stream.alloc_zeros::<i32>(topk_unified)?,
            topk_length: ctx.stream.alloc_zeros::<i32>(1)?,
            lse_out: ctx.stream.alloc_zeros::<f32>(h_q)?,
            lse_accum: ctx.stream.alloc_zeros::<f32>(accum_rows * h_q)?,
            o_accum: ctx.stream.alloc_zeros::<f32>(accum_rows * h_q_d)?,
            sched_meta: ctx.stream.alloc_zeros::<i32>(num_sm_parts_max * 8)?,
            num_splits: ctx.stream.alloc_zeros::<i32>(2)?,
            tp_gathered_q: ctx.stream.alloc_zeros::<half::bf16>(h_q_d)?,
            tp_packed_q: ctx.stream.alloc_zeros::<half::bf16>(h_q_d)?,
            tp_full_out: ctx.stream.alloc_zeros::<half::bf16>(h_q_d)?,
            num_sm_parts,
            fixed_overhead_num_blocks,
            block_size_topk,
        })
    }

    fn reset(&mut self, ctx: &DeviceContext) -> Result<()> {
        self.fp8_kv_sw_bootstrapped = false;
        self.fp8_kv_comp_packed_rows = 0;
        ctx.stream
            .memset_zeros(&mut self.fp8_kv_pool)
            .map_err(|e| anyhow!("DSv4 FlashMLA FP8 KV reset failed: {e}"))?;
        Ok(())
    }
}

struct Dsv4FusedWqkvDecodeScratch {
    input_fp8: CudaSlice<u8>,
    input_scales: CudaSlice<f32>,
    qkv_raw: HiddenStates,
    active_experts: CudaSlice<i32>,
    active_offsets: CudaSlice<i32>,
    active_counts: CudaSlice<i32>,
    max_m: usize,
    scale_stride_m: usize,
    hidden_dim: usize,
    q_lora_rank: usize,
    head_dim: usize,
}

impl Dsv4FusedWqkvDecodeScratch {
    fn new(ctx: &DeviceContext, config: &DeepSeekV4Config) -> Result<Self> {
        let max_m = 128;
        let scale_stride_m = 128;
        let hidden_dim = config.hidden_size;
        let q_lora_rank = config.q_lora_rank;
        let head_dim = config.head_dim;
        let scale_cols = hidden_dim.div_ceil(128);
        Ok(Self {
            input_fp8: ctx
                .stream
                .alloc_zeros::<u8>(max_m * hidden_dim)
                .map_err(|e| anyhow!("DSv4 fused wqkv input fp8 scratch alloc failed: {e}"))?,
            input_scales: ctx
                .stream
                .alloc_zeros::<f32>(scale_stride_m * scale_cols)
                .map_err(|e| anyhow!("DSv4 fused wqkv input scale scratch alloc failed: {e}"))?,
            qkv_raw: unsafe { HiddenStates::uninit(ctx, q_lora_rank + head_dim, 1)? },
            active_experts: ctx
                .stream
                .clone_htod(&[0_i32])
                .map_err(|e| anyhow!("DSv4 fused wqkv active_experts H2D failed: {e}"))?,
            active_offsets: ctx
                .stream
                .clone_htod(&[0_i32])
                .map_err(|e| anyhow!("DSv4 fused wqkv active_offsets H2D failed: {e}"))?,
            active_counts: ctx
                .stream
                .clone_htod(&[1_i32])
                .map_err(|e| anyhow!("DSv4 fused wqkv active_counts H2D failed: {e}"))?,
            max_m,
            scale_stride_m,
            hidden_dim,
            q_lora_rank,
            head_dim,
        })
    }
}

pub(crate) struct Dsv4LayerAttentionState {
    sw_window_cache: CudaSlice<half::bf16>,
    compressor: Option<Dsv4CompressorState>,
    indexer: Option<Dsv4CompressorState>,
    flashmla: Option<Dsv4FlashMlaDecodeState>,
    fused_wqkv: Option<Dsv4FusedWqkvDecodeScratch>,
}

impl Dsv4LayerAttentionState {
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        mode: DeepSeekV4AttentionMode,
        compress_ratio: usize,
        max_seq_len: usize,
        kv_arena: &Dsv4MlaKvArena,
        local_heads: usize,
        tp_world: usize,
    ) -> Result<Self> {
        let sw_len = config.sliding_window * config.head_dim;
        ensure!(
            sw_len > 0,
            "DSv4 SW window cache len is zero (sliding_window={} head_dim={})",
            config.sliding_window,
            config.head_dim
        );
        let sw_window_cache = ctx
            .stream
            .alloc_zeros::<half::bf16>(sw_len)
            .map_err(|e| anyhow::anyhow!("DSv4 SW window cache alloc failed: {e}"))?;
        let overlap = compress_ratio < 16;
        let compressor = if mode == DeepSeekV4AttentionMode::SlidingWindow {
            None
        } else {
            Some(Dsv4CompressorState::new(
                ctx,
                config.head_dim,
                compress_ratio,
                overlap,
                max_seq_len,
            )?)
        };
        let indexer = if mode == DeepSeekV4AttentionMode::CompressedSparse {
            Some(Dsv4CompressorState::new(
                ctx,
                config.index_head_dim,
                compress_ratio,
                true,
                max_seq_len,
            )?)
        } else {
            None
        };
        let flashmla = if dsv4_flashmla_decode_alloc_enabled()? {
            Some(Dsv4FlashMlaDecodeState::new(
                ctx,
                config,
                mode,
                compress_ratio,
                max_seq_len,
                kv_arena,
                local_heads,
                tp_world,
            )?)
        } else {
            None
        };
        let fused_wqkv = if dsv4_fused_wqkv_decode_alloc_enabled()? {
            Some(Dsv4FusedWqkvDecodeScratch::new(ctx, config)?)
        } else {
            None
        };
        Ok(Self {
            sw_window_cache,
            compressor,
            indexer,
            flashmla,
            fused_wqkv,
        })
    }

    pub(crate) fn reset(&mut self, ctx: &DeviceContext) -> Result<()> {
        ctx.stream
            .memset_zeros(&mut self.sw_window_cache)
            .map_err(|e| anyhow::anyhow!("DSv4 SW window cache reset failed: {e}"))?;
        if let Some(compressor) = &mut self.compressor {
            compressor.reset(ctx)?;
        }
        if let Some(indexer) = &mut self.indexer {
            indexer.reset(ctx)?;
        }
        if let Some(flashmla) = &mut self.flashmla {
            flashmla.reset(ctx)?;
        }
        Ok(())
    }

    pub(crate) fn advance_decode_len(
        &mut self,
        mode: DeepSeekV4AttentionMode,
        ratio: usize,
        total_len: usize,
    ) {
        if mode == DeepSeekV4AttentionMode::SlidingWindow {
            return;
        }
        let compressed_rows = total_len / ratio;
        if let Some(compressor) = &mut self.compressor {
            compressor.compressed.seq_len = compressed_rows;
        }
        if let Some(indexer) = &mut self.indexer {
            indexer.compressed.seq_len = compressed_rows;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn paged_attention(
    ctx: &DeviceContext,
    layer_idx: usize,
    pool: &PagedKVPool,
    q_batch: &mut HiddenStates,
    k_batch: &mut HiddenStates,
    v_batch: &HiddenStates,
    q_norm: &DeviceVec,
    k_norm: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    rms_eps: f32,
    meta: &PageMeta,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    if meta.seq_len == 1 {
        decode_attention(
            ctx,
            layer_idx,
            pool,
            q_batch,
            k_batch,
            v_batch,
            q_norm,
            k_norm,
            cos_cache,
            sin_cache,
            rms_eps,
            meta,
            num_q_heads,
            num_kv_heads,
            head_dim,
            out,
        )
    } else {
        prefill_attention(
            ctx,
            layer_idx,
            pool,
            q_batch,
            k_batch,
            v_batch,
            q_norm,
            k_norm,
            cos_cache,
            sin_cache,
            rms_eps,
            meta,
            num_q_heads,
            num_kv_heads,
            head_dim,
            out,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn prefill_attention(
    ctx: &DeviceContext,
    layer_idx: usize,
    pool: &PagedKVPool,
    q_batch: &mut HiddenStates,
    k_batch: &mut HiddenStates,
    v_batch: &HiddenStates,
    q_norm: &DeviceVec,
    k_norm: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    rms_eps: f32,
    meta: &PageMeta,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    {
        let (q_ptr, _gq) = q_batch.data.device_ptr_mut(&ctx.stream);
        let (k_ptr, _gk) = k_batch.data.device_ptr_mut(&ctx.stream);
        let (v_ptr, _gv) = v_batch.data.device_ptr(&ctx.stream);
        let (qn_ptr, _gqn) = q_norm.data.device_ptr(&ctx.stream);
        let (kn_ptr, _gkn) = k_norm.data.device_ptr(&ctx.stream);
        let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
        let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
        let (indices_ptr, _gi) = meta.kv_indices.device_ptr(&ctx.stream);
        let (offsets_ptr, _goff) = meta.page_table_offsets.device_ptr(&ctx.stream);
        let (start_ptr, _gstart) = meta.start_positions.device_ptr(&ctx.stream);
        let k_pool_ptr = pool.k_ptr(layer_idx, &ctx.stream);
        let v_pool_ptr = pool.v_ptr(layer_idx, &ctx.stream);

        unsafe {
            ffi::prefill_attention_paged_prep_cuda(
                q_ptr as *mut ffi::Half,
                k_ptr as *mut ffi::Half,
                v_ptr as *const ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                indices_ptr as *const i32,
                offsets_ptr as *const i32,
                pool.page_size as i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                meta.seq_len as i32,
                start_ptr as *const i32,
                rms_eps,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }
    run_tilelang_paged(
        ctx,
        false,
        layer_idx,
        pool,
        q_batch,
        meta,
        num_q_heads,
        num_kv_heads,
        head_dim,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
fn decode_attention(
    ctx: &DeviceContext,
    layer_idx: usize,
    pool: &PagedKVPool,
    q_batch: &mut HiddenStates,
    k_batch: &HiddenStates,
    v_batch: &HiddenStates,
    q_norm: &DeviceVec,
    k_norm: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    rms_eps: f32,
    meta: &PageMeta,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    {
        let (q_ptr, _gq) = q_batch.data.device_ptr_mut(&ctx.stream);
        let (k_ptr, _gk) = k_batch.data.device_ptr(&ctx.stream);
        let (v_ptr, _gv) = v_batch.data.device_ptr(&ctx.stream);
        let (qn_ptr, _gqn) = q_norm.data.device_ptr(&ctx.stream);
        let (kn_ptr, _gkn) = k_norm.data.device_ptr(&ctx.stream);
        let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
        let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
        let (pos_ptr, _gp) = meta.positions.device_ptr(&ctx.stream);
        let (indices_ptr, _gi) = meta.kv_indices.device_ptr(&ctx.stream);
        let (indptr_ptr, _gind) = meta.kv_indptr.device_ptr(&ctx.stream);
        let (last_ptr, _glp) = meta.kv_last_page_len.device_ptr(&ctx.stream);
        let k_pool_ptr = pool.k_ptr(layer_idx, &ctx.stream);
        let v_pool_ptr = pool.v_ptr(layer_idx, &ctx.stream);
        let stride_page = pool.kv_dim * pool.page_size;

        unsafe {
            ffi::decode_prep_paged_cuda(
                q_ptr as *mut ffi::Half,
                k_ptr as *const ffi::Half,
                v_ptr as *const ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                pos_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                indices_ptr as *const i32,
                indptr_ptr as *const i32,
                last_ptr as *const i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                stride_page as i32,
                1,
                rms_eps,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }
    run_tilelang_paged(
        ctx,
        true,
        layer_idx,
        pool,
        q_batch,
        meta,
        num_q_heads,
        num_kv_heads,
        head_dim,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_tilelang_paged(
    ctx: &DeviceContext,
    decode: bool,
    layer_idx: usize,
    pool: &PagedKVPool,
    q_batch: &HiddenStates,
    meta: &PageMeta,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(head_dim == 128, "only HD128 TileLang kernels are wired");
    ensure!(num_kv_heads == 8, "only kv8 TileLang kernels are wired");

    let (q_ptr, _gq) = q_batch.data.device_ptr(&ctx.stream);
    let (qo_ptr, _gqo) = meta.q_indptr.device_ptr(&ctx.stream);
    let (kv_indptr_ptr, _gki) = meta.kv_indptr.device_ptr(&ctx.stream);
    let (kv_indices_ptr, _gkx) = meta.kv_indices.device_ptr(&ctx.stream);
    let (last_ptr, _glp) = meta.kv_last_page_len.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let k_pool_ptr = pool.k_ptr(layer_idx, &ctx.stream);
    let v_pool_ptr = pool.v_ptr(layer_idx, &ctx.stream);
    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();

    // Set R6_ATTN_DEBUG=1 to dump the scalar args + device arrays fed to the
    // TileLang paged kernel.
    if std::env::var("R6_ATTN_DEBUG").is_ok() {
        eprintln!(
            "[r6-attn] decode={decode} layer={layer_idx} q_heads={num_q_heads} kv_heads={num_kv_heads} head_dim={head_dim} seq_len={} num_pages(meta)={} max_total_pages={} page_size={} kv_dim={} sm_scale={sm_scale}",
            meta.seq_len, meta.num_pages, pool.max_total_pages, pool.page_size, pool.kv_dim
        );
        for (name, slice) in [
            ("q_indptr", &meta.q_indptr),
            ("kv_indptr", &meta.kv_indptr),
            ("kv_indices", &meta.kv_indices),
            ("kv_last_page_len", &meta.kv_last_page_len),
        ] {
            match ctx.stream.clone_dtoh(slice) {
                Ok(v) => eprintln!("[r6-attn]   {name} = {v:?}"),
                Err(e) => eprintln!("[r6-attn]   {name} dtoh err: {e}"),
            }
        }
    }

    // TileLang arg order (load-bearing): `num_pages` (arg 12) = pool capacity
    // (`pool.max_total_pages`, the k_pool/v_pool first-dim extent); `total_pages`
    // (arg 13) = page-table length (`meta.num_pages`). Swapping them gives wrong
    // pool strides + an OOB kv_indices walk that hangs the kernel (Xid 43).
    unsafe {
        match (decode, num_q_heads) {
            (false, 16) => ffi::tilelang_batch_prefill_paged_hd128_q16_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                meta.seq_len as i32,
                meta.seq_len as i32,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (false, 32) => ffi::tilelang_batch_prefill_paged_hd128_q32_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                meta.seq_len as i32,
                meta.seq_len as i32,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (false, 40) => ffi::tilelang_batch_prefill_paged_hd128_q40_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                meta.seq_len as i32,
                meta.seq_len as i32,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (false, 64) => ffi::tilelang_batch_prefill_paged_hd128_q64_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                meta.seq_len as i32,
                meta.seq_len as i32,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (true, 16) => ffi::tilelang_batch_decode_paged_hd128_q16_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                1,
                1,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (true, 32) => ffi::tilelang_batch_decode_paged_hd128_q32_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                1,
                1,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (true, 40) => ffi::tilelang_batch_decode_paged_hd128_q40_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                1,
                1,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (true, 64) => ffi::tilelang_batch_decode_paged_hd128_q64_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                1,
                1,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            _ => bail!("unsupported HD128 q/kv head config q{num_q_heads}_kv{num_kv_heads}"),
        }
    }
    Ok(())
}

// ============================================================================
// DSv4-Flash MLA attention core
// ============================================================================
//
// The MLA attention is a genuinely new subsystem next to the dense-BF16 paged
// path above (it is NOT a GEMM swap): a low-rank Q/KV projection (`wq_a → q_norm
// → wq_b` for Q; `wkv → kv_norm` for the single compressed KV latent), partial
// RoPE on the trailing `rope_dim` columns, a windowed attention with a per-head
// sink logit + (on CSA/HCA layers) a compressed-key stream, and a low-rank O
// projection (`wo_a → wo_b`).
//
// All three modes run through the bf16 correctness core (the perf-optimized
// FlashMLA sparse path stays gated — `Dsv4MlaKvArena::alloc_fp8_arena`):
//   - SlidingWindow (`compress_ratio == 0`): Q/K prep RoPE + `dsv4_swa_attention`
//     over the bf16 SW ring cache, with the output inverse-RoPE fused.
//   - CompressedSparse (`0 < ratio < 16`): a compressor produces compressed keys,
//     an indexer + `dsv4_csa_select_cuda` picks the top-k blocks, then
//     `dsv4_hybrid_attention_cuda` (mode 1) attends over SW window + selected
//     compressed blocks.
//   - HybridCompressed (`ratio >= 16`): compressor + `dsv4_hybrid_attention_cuda`
//     (mode 2) attending over SW window + ALL compressed blocks (no selector).
//
// Shared kernels: `dsv4_{fp8,fp4}_gemv_batch_cuda` / `gemm_cuda` (LoRA matmuls),
// `dsv4_prepare_qk_cuda`, `dsv4_swa_attention_cuda`, `dsv4_compressor_update_cuda`,
// `dsv4_csa_select_cuda`, `dsv4_hybrid_attention_cuda`.

/// Run one DSv4 FP8/FP4 block-scaled LoRA matmul: `out[N, T] = W[N, K] · x[K, T]`.
///
/// The MLA LoRA weights (`wq_a/wq_b/wkv/wo_a/wo_b`) load as
/// [`WeightFormat::Dsv4Fp8BlockScaled`] / [`WeightFormat::Dsv4Fp4BlockScaled`]
/// (raw quant bytes in `qweight`, E8M0 block scales in `dsv4_scales`), so the
/// dense bf16 [`gemm_batch`] cannot run them — this dispatches the shared
/// `dsv4_*_gemv_batch_cuda` kernels instead. `batch_size` is the token count.
pub(crate) fn mla_linear(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        weight.cols == x.hidden_dim,
        "mla_linear input dim mismatch: weight cols {}, x hidden_dim {}",
        weight.cols,
        x.hidden_dim
    );
    ensure!(
        weight.rows == out.hidden_dim && x.seq_len == out.seq_len,
        "mla_linear output shape mismatch: weight rows {}, out hidden_dim {}, x seq {}, out seq {}",
        weight.rows,
        out.hidden_dim,
        x.seq_len,
        out.seq_len
    );
    let qw = weight
        .qweight
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("DSv4 MLA matrix missing raw quant bytes (qweight)"))?;
    let scales = weight
        .dsv4_scales
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("DSv4 MLA matrix missing block scales (dsv4_scales)"))?;
    let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
    let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let stream = ctx.stream.cu_stream();
    // SAFETY: all buffers are valid on ctx.stream; shapes are checked above and
    // the scale-row/col extents come from the matrix the loader built.
    unsafe {
        let res = match weight.weight_format {
            WeightFormat::Dsv4Fp8BlockScaled => ffi::dsv4_fp8_gemv_batch_cuda(
                qw_ptr as *const u8,
                scales_ptr as *const u8,
                x_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                x.seq_len as i32,
                weight.rows as i32,
                weight.cols as i32,
                weight.dsv4_scale_rows as i32,
                weight.dsv4_scale_cols as i32,
                stream,
            ),
            WeightFormat::Dsv4Fp4BlockScaled => ffi::dsv4_fp4_gemv_batch_cuda(
                qw_ptr as *const u8,
                scales_ptr as *const u8,
                x_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                x.seq_len as i32,
                weight.rows as i32,
                weight.cols as i32,
                weight.dsv4_scale_rows as i32,
                weight.dsv4_scale_cols as i32,
                stream,
            ),
            other => bail!("mla_linear: expected DSv4 FP8/FP4 block-scaled weight, got {other:?}"),
        };
        res.result()?;
    }
    Ok(())
}

pub(crate) fn mla_linear_vec(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &DeviceVec,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        weight.cols == x.len,
        "mla_linear_vec input dim mismatch: weight cols {}, x len {}",
        weight.cols,
        x.len
    );
    ensure!(
        weight.rows == out.hidden_dim && out.seq_len == 1,
        "mla_linear_vec output shape mismatch: weight rows {}, out {}x{}",
        weight.rows,
        out.hidden_dim,
        out.seq_len
    );
    let qw = weight
        .qweight
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("DSv4 MLA matrix missing raw quant bytes (qweight)"))?;
    let scales = weight
        .dsv4_scales
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("DSv4 MLA matrix missing block scales (dsv4_scales)"))?;
    let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
    let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let stream = ctx.stream.cu_stream();
    // SAFETY: all buffers are valid on ctx.stream; shapes are checked above.
    unsafe {
        let res = match weight.weight_format {
            WeightFormat::Dsv4Fp8BlockScaled => ffi::dsv4_fp8_gemv_batch_cuda(
                qw_ptr as *const u8,
                scales_ptr as *const u8,
                x_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                1,
                weight.rows as i32,
                weight.cols as i32,
                weight.dsv4_scale_rows as i32,
                weight.dsv4_scale_cols as i32,
                stream,
            ),
            WeightFormat::Dsv4Fp4BlockScaled => ffi::dsv4_fp4_gemv_batch_cuda(
                qw_ptr as *const u8,
                scales_ptr as *const u8,
                x_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                1,
                weight.rows as i32,
                weight.cols as i32,
                weight.dsv4_scale_rows as i32,
                weight.dsv4_scale_cols as i32,
                stream,
            ),
            other => {
                bail!("mla_linear_vec: expected DSv4 FP8/FP4 block-scaled weight, got {other:?}")
            }
        };
        res.result()?;
    }
    Ok(())
}

/// Run one DSv4 linear `out = W · x` dispatching on the weight's on-disk format:
/// bf16 dense → [`crate::ops::gemm_batch`]; FP8/FP4 block-scaled → [`mla_linear`].
/// DSv4 checkpoints ship the compressor / indexer / HC-mix matrices in either
/// precision, so callers route every non-router linear through here.
pub(crate) fn dsv4_linear(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    match weight.weight_format {
        WeightFormat::DenseBf16 => crate::ops::gemm_batch(ctx, weight, x, out),
        WeightFormat::Dsv4Fp8BlockScaled | WeightFormat::Dsv4Fp4BlockScaled => {
            mla_linear(ctx, weight, x, out)
        }
        other => bail!("dsv4_linear: unsupported weight format {other:?}"),
    }
}

pub(crate) fn dsv4_flashmla_decode_enabled() -> Result<bool> {
    match DSV4_FLASHMLA_DECODE_OVERRIDE.load(Ordering::Relaxed) {
        DSV4_FLASHMLA_OVERRIDE_OFF => return Ok(false),
        DSV4_FLASHMLA_OVERRIDE_ON => return Ok(true),
        _ => {}
    }
    // Default ON: FlashMLA SM90 sparse decode is the adopted decode attention — the
    // same vendored kernel SGLang uses. Licensed 2026-06-06 on the TP=8/EP=8 pod:
    // 64-tok resident same-load A/B token-exact vs scalar, 29.47 -> 36.59 tok/s
    // (+24%). `dsv4_flashmla_decode_alloc_enabled` falls through to this, so the
    // arena allocates under the default. Opt out with ARLE_DSV4_FLASHMLA_DECODE=0.
    Ok(!matches!(
        std::env::var("ARLE_DSV4_FLASHMLA_DECODE").as_deref(),
        Ok("0" | "false" | "FALSE" | "off" | "OFF" | "no" | "NO")
    ))
}

fn dsv4_flashmla_prefill_enabled() -> Result<bool> {
    env_flag("ARLE_DSV4_FLASHMLA_PREFILL")
}

/// Per-layer attention-output localizer (Track A FlashMLA-prefill diagnosis).
///
/// When `ARLE_DSV4_ATTN_DUMP=1`, every layer prints a stable FNV-1a hash of its
/// full bf16 `local_attn` output plus the first 8 values of row 0, on rank 0
/// only. Run the same prompt twice — once scalar (default), once with
/// `ARLE_DSV4_FLASHMLA_PREFILL=1` — and diff the two logs: the *first* CSA/HCA
/// layer whose hash differs is exactly where FlashMLA-prefill diverges from the
/// scalar reference. SW layers run scalar in both passes and match by
/// construction, so a mismatch localizes the bug to one layer in one build —
/// replacing the end-to-end-token guess loop. Adds one `ctx.sync()` per layer,
/// so it is strictly opt-in.
fn dsv4_attn_dump_enabled() -> bool {
    matches!(
        std::env::var("ARLE_DSV4_ATTN_DUMP").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
    ) && std::env::var("INFER_TP_RANK").as_deref() == Ok("0")
}

fn dsv4_dump_attn_output(
    ctx: &DeviceContext,
    layer_idx: usize,
    mode: DeepSeekV4AttentionMode,
    out: &HiddenStates,
) -> Result<()> {
    ctx.sync()?;
    let host: Vec<half::bf16> = ctx
        .stream
        .clone_dtoh(&out.data)
        .map_err(|e| anyhow!("DSv4 attn-dump D2H failed: {e}"))?;
    let mut hash: u64 = 0xcbf29ce484222325;
    for v in &host {
        hash ^= u64::from(v.to_bits());
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let row0: Vec<f32> = host.iter().take(8).map(|v| v.to_f32()).collect();
    eprintln!(
        "[dsv4-attn-dump] layer={layer_idx} mode={mode:?} seq_len={} hidden={} hash={hash:016x} row0={row0:?}",
        out.seq_len, out.hidden_dim
    );
    Ok(())
}

fn dsv4_flashmla_decode_alloc_enabled() -> Result<bool> {
    if env_flag("ARLE_DSV4_FLASHMLA_DECODE_ALLOC")? {
        return Ok(true);
    }
    dsv4_flashmla_decode_enabled()
}

pub(crate) fn dsv4_fused_wqkv_decode_alloc_enabled() -> Result<bool> {
    if env_flag("ARLE_DSV4_FUSED_WQKV_DECODE_ALLOC")? {
        return Ok(true);
    }
    dsv4_fused_wqkv_decode_enabled()
}

fn dsv4_fused_wqkv_decode_enabled() -> Result<bool> {
    match DSV4_FUSED_WQKV_DECODE_OVERRIDE.load(Ordering::Relaxed) {
        DSV4_FLASHMLA_OVERRIDE_OFF => return Ok(false),
        DSV4_FLASHMLA_OVERRIDE_ON => return Ok(true),
        _ => {}
    }
    env_flag("ARLE_DSV4_FUSED_WQKV_DECODE")
}

fn env_flag(name: &str) -> Result<bool> {
    match std::env::var(name) {
        Ok(value) => match value.as_str() {
            "1" | "true" | "TRUE" | "yes" | "on" | "ON" => Ok(true),
            "0" | "false" | "FALSE" | "no" | "off" | "OFF" | "" => Ok(false),
            other => bail!("unsupported {name} `{other}` (expected 0/1, true/false, on/off)"),
        },
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(e) => bail!("{name} invalid env: {e}"),
    }
}

fn flashmla_mode_int(mode: DeepSeekV4AttentionMode) -> i32 {
    match mode {
        DeepSeekV4AttentionMode::CompressedSparse => 1,
        DeepSeekV4AttentionMode::SlidingWindow | DeepSeekV4AttentionMode::HybridCompressed => 2,
    }
}

fn flashmla_pack_sw_ring(
    ctx: &DeviceContext,
    flash: &mut Dsv4FlashMlaDecodeState,
    window_cache: &CudaSlice<half::bf16>,
    config: &DeepSeekV4Config,
) -> Result<()> {
    if flash.fp8_kv_sw_bootstrapped {
        return Ok(());
    }
    let sliding_window = config.sliding_window;
    let page_block_size = 64;
    let mut block_ids = Vec::with_capacity(sliding_window);
    let mut rows = Vec::with_capacity(sliding_window);
    for slot in 0..sliding_window {
        block_ids.push((slot / page_block_size) as i32);
        rows.push((slot % page_block_size) as i32);
    }
    ctx.stream
        .memcpy_htod(&block_ids, &mut flash.sw_bulk_block_ids)
        .map_err(|e| anyhow!("DSv4 FlashMLA SW block_ids H2D failed: {e}"))?;
    ctx.stream
        .memcpy_htod(&rows, &mut flash.sw_bulk_rows)
        .map_err(|e| anyhow!("DSv4 FlashMLA SW rows H2D failed: {e}"))?;
    let (window_ptr, _wg) = window_cache.device_ptr(&ctx.stream);
    let (pool_ptr, _pg) = flash.fp8_kv_pool.device_ptr_mut(&ctx.stream);
    let nope_ptr = window_ptr as u64;
    let rope_ptr = nope_ptr + (config.head_dim - config.qk_rope_head_dim) as u64 * 2;
    flash_kv::dsv4_fp8_kv_pack_strided_raw(
        ctx,
        nope_ptr,
        rope_ptr,
        pool_ptr,
        &flash.sw_bulk_block_ids,
        &flash.sw_bulk_rows,
        sliding_window,
        page_block_size,
        config.head_dim,
        config.head_dim,
    )?;
    flash.fp8_kv_sw_bootstrapped = true;
    Ok(())
}

fn flashmla_pack_one_sw_token(
    ctx: &DeviceContext,
    flash: &mut Dsv4FlashMlaDecodeState,
    k_prepared: &HiddenStates,
    start_pos_device: &CudaSlice<i32>,
    config: &DeepSeekV4Config,
) -> Result<()> {
    let (bid_ptr, bid_guard) = flash.one_block_id.device_ptr_mut(&ctx.stream);
    let (row_ptr, row_guard) = flash.one_row.device_ptr_mut(&ctx.stream);
    let (start_ptr, _sg) = start_pos_device.device_ptr(&ctx.stream);
    flash_kv::dsv4_fp8_kv_fill_one_sw_slot_from_start_pos_raw(
        ctx,
        bid_ptr,
        row_ptr,
        start_ptr,
        config.sliding_window,
        64,
    )?;
    drop(bid_guard);
    drop(row_guard);

    let (k_ptr, _kg) = k_prepared.data.device_ptr(&ctx.stream);
    let (pool_ptr, _pg) = flash.fp8_kv_pool.device_ptr_mut(&ctx.stream);
    let nope_ptr = k_ptr as u64;
    let rope_ptr = nope_ptr + (config.head_dim - config.qk_rope_head_dim) as u64 * 2;
    flash_kv::dsv4_fp8_kv_pack_strided_raw(
        ctx,
        nope_ptr,
        rope_ptr,
        pool_ptr,
        &flash.one_block_id,
        &flash.one_row,
        1,
        64,
        config.head_dim,
        config.head_dim,
    )
}

fn flashmla_pack_compressed_delta(
    ctx: &DeviceContext,
    flash: &mut Dsv4FlashMlaDecodeState,
    compressed: Option<&HiddenStates>,
    config: &DeepSeekV4Config,
) -> Result<()> {
    let Some(compressed) = compressed else {
        return Ok(());
    };
    let start_row = flash.fp8_kv_comp_packed_rows;
    let end_row = compressed.seq_len;
    if end_row <= start_row {
        return Ok(());
    }
    let n = end_row - start_row;
    let mut block_ids = Vec::with_capacity(n);
    let mut rows = Vec::with_capacity(n);
    for row in start_row..end_row {
        block_ids.push((flash.sw_blocks + row / 64) as i32);
        rows.push((row % 64) as i32);
    }
    ctx.stream
        .memcpy_htod(&block_ids, &mut flash.comp_block_ids)
        .map_err(|e| anyhow!("DSv4 FlashMLA compressed block_ids H2D failed: {e}"))?;
    ctx.stream
        .memcpy_htod(&rows, &mut flash.comp_rows)
        .map_err(|e| anyhow!("DSv4 FlashMLA compressed rows H2D failed: {e}"))?;

    let (compressed_ptr, _cg) = compressed.data.device_ptr(&ctx.stream);
    let (pool_ptr, _pg) = flash.fp8_kv_pool.device_ptr_mut(&ctx.stream);
    let row_offset_bytes = start_row as u64 * config.head_dim as u64 * 2;
    let nope_ptr = compressed_ptr as u64 + row_offset_bytes;
    let rope_ptr = nope_ptr + (config.head_dim - config.qk_rope_head_dim) as u64 * 2;
    flash_kv::dsv4_fp8_kv_pack_strided_raw(
        ctx,
        nope_ptr,
        rope_ptr,
        pool_ptr,
        &flash.comp_block_ids,
        &flash.comp_rows,
        n,
        64,
        config.head_dim,
        config.head_dim,
    )?;
    flash.fp8_kv_comp_packed_rows = end_row;
    Ok(())
}

fn update_bf16_sw_window(
    ctx: &DeviceContext,
    sw_window_cache: &mut CudaSlice<half::bf16>,
    k_prepared: &HiddenStates,
    start_pos: usize,
    start_pos_device: Option<&CudaSlice<i32>>,
    config: &DeepSeekV4Config,
) -> Result<()> {
    let (k_ptr, _kg) = k_prepared.data.device_ptr(&ctx.stream);
    let (window_ptr, _wg) = sw_window_cache.device_ptr_mut(&ctx.stream);
    unsafe {
        if let Some(start_pos_device) = start_pos_device {
            let (start_ptr, _sg) = start_pos_device.device_ptr(&ctx.stream);
            ffi::dsv4_update_window_cache_start_pos_ptr_cuda(
                k_ptr as *const ffi::Half,
                window_ptr as *mut ffi::Half,
                k_prepared.seq_len as i32,
                start_ptr as *const i32,
                config.sliding_window as i32,
                config.head_dim as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        } else {
            ffi::dsv4_update_window_cache_cuda(
                k_ptr as *const ffi::Half,
                window_ptr as *mut ffi::Half,
                k_prepared.seq_len as i32,
                start_pos as i32,
                config.sliding_window as i32,
                config.head_dim as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn try_flashmla_prefill_attention(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    attention: &Dsv4Attention,
    mode: DeepSeekV4AttentionMode,
    compress_ratio: usize,
    q_prepared: &HiddenStates,
    k_prepared: &HiddenStates,
    selected: Option<&CudaSlice<i32>>,
    compressed: &HiddenStates,
    sw_window_cache: &mut CudaSlice<half::bf16>,
    start_pos: usize,
    tp: &TpRuntime,
    local_heads: usize,
    local_attn: &mut HiddenStates,
    sm_scale: f32,
    rope_base: f32,
    original_seq_len: i32,
    rope_factor: f32,
    rope_beta_fast: f32,
    rope_beta_slow: f32,
) -> Result<bool> {
    if !dsv4_flashmla_prefill_enabled()? {
        return Ok(false);
    }
    if q_prepared.seq_len <= 1 {
        return Ok(false);
    }
    if mode == DeepSeekV4AttentionMode::SlidingWindow {
        return Ok(false);
    }
    ensure!(
        config.head_dim == 512 && config.qk_rope_head_dim == 64,
        "DSv4 FlashMLA prefill only supports MODEL1 head_dim=512 rope_dim=64"
    );
    ensure!(
        q_prepared.seq_len == k_prepared.seq_len && local_attn.seq_len == q_prepared.seq_len,
        "DSv4 FlashMLA prefill shape mismatch: q={} k={} out={}",
        q_prepared.seq_len,
        k_prepared.seq_len,
        local_attn.seq_len
    );

    let token_count = q_prepared.seq_len;
    let tp_world = tp.config().world_size;
    let tp_rank = tp.config().rank;
    let global_heads = local_heads
        .checked_mul(tp_world)
        .ok_or_else(|| anyhow!("DSv4 FlashMLA prefill global head overflow"))?;
    ensure!(
        matches!(global_heads, 64 | 128),
        "DSv4 FlashMLA prefill requires global heads 64/128, got {global_heads}"
    );

    let compressed_count = compressed.seq_len;
    let max_compressed_keys = match mode {
        DeepSeekV4AttentionMode::CompressedSparse => config.index_topk,
        DeepSeekV4AttentionMode::HybridCompressed => compressed_count.div_ceil(128) * 128,
        DeepSeekV4AttentionMode::SlidingWindow => unreachable!(),
    };
    let topk_unified = config
        .sliding_window
        .checked_add(max_compressed_keys)
        .ok_or_else(|| anyhow!("DSv4 FlashMLA prefill topk overflow"))?;
    ensure!(
        topk_unified.is_multiple_of(128),
        "DSv4 FlashMLA prefill topk {topk_unified} must be multiple of 128"
    );
    let kv_rows = config
        .sliding_window
        .checked_add(token_count)
        .and_then(|v| v.checked_add(compressed_count))
        .ok_or_else(|| anyhow!("DSv4 FlashMLA prefill unified KV rows overflow"))?;
    ensure!(kv_rows > 0, "DSv4 FlashMLA prefill needs non-empty KV pool");

    // FlashMLA prefill consumes one unified bf16 pool:
    // [rolling SW cache rebased | current chunk K | compressed pool].
    let mut kv_unified = unsafe { HiddenStates::uninit(ctx, config.head_dim, kv_rows)? };
    {
        let _nvtx = crate::nvtx::range("dsv4/flashmla_prefill_pack_kv");
        let (kv_ptr, _kvg) = kv_unified.data.device_ptr_mut(&ctx.stream);
        let (window_ptr, _wg) = sw_window_cache.device_ptr(&ctx.stream);
        let (k_ptr, _kg) = k_prepared.data.device_ptr(&ctx.stream);
        let (comp_ptr, _cg) = compressed.data.device_ptr(&ctx.stream);
        unsafe {
            ffi::arle_flashmla_csa_pack_kv(
                kv_ptr as *mut ffi::Half,
                window_ptr as *const ffi::Half,
                k_ptr as *const ffi::Half,
                if compressed_count > 0 {
                    comp_ptr as *const ffi::Half
                } else {
                    std::ptr::null()
                },
                start_pos as i32,
                config.sliding_window as i32,
                token_count as i32,
                compressed_count as i32,
                config.head_dim as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|e| anyhow!("DSv4 FlashMLA prefill KV pack failed: {e}"))?;
        }
    }

    let mut indices = ctx
        .stream
        .alloc_zeros::<i32>(token_count * topk_unified)
        .map_err(|e| anyhow!("DSv4 FlashMLA prefill indices alloc failed: {e}"))?;
    let mut topk_length = ctx
        .stream
        .alloc_zeros::<i32>(token_count)
        .map_err(|e| anyhow!("DSv4 FlashMLA prefill topk_length alloc failed: {e}"))?;
    {
        let _nvtx = crate::nvtx::range("dsv4/flashmla_prefill_build_indices");
        let (indices_ptr, _ig) = indices.device_ptr_mut(&ctx.stream);
        let (topk_ptr, _tg) = topk_length.device_ptr_mut(&ctx.stream);
        unsafe {
            match mode {
                DeepSeekV4AttentionMode::CompressedSparse => {
                    let selected = selected.ok_or_else(|| {
                        anyhow!("DSv4 FlashMLA CSA prefill missing selected topk")
                    })?;
                    let (selected_ptr, _sg) = selected.device_ptr(&ctx.stream);
                    ffi::arle_flashmla_csa_build_indices(
                        indices_ptr as *mut i32,
                        topk_ptr as *mut i32,
                        selected_ptr as *const i32,
                        token_count as i32,
                        start_pos as i32,
                        config.sliding_window as i32,
                        config.index_topk as i32,
                        compressed_count as i32,
                        compress_ratio as i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()
                    .map_err(|e| anyhow!("DSv4 FlashMLA CSA prefill indices failed: {e}"))?;
                }
                DeepSeekV4AttentionMode::HybridCompressed => {
                    ffi::arle_flashmla_hca_build_indices(
                        indices_ptr as *mut i32,
                        topk_ptr as *mut i32,
                        token_count as i32,
                        start_pos as i32,
                        config.sliding_window as i32,
                        max_compressed_keys as i32,
                        compressed_count as i32,
                        compress_ratio as i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()
                    .map_err(|e| anyhow!("DSv4 FlashMLA HCA prefill indices failed: {e}"))?;
                }
                DeepSeekV4AttentionMode::SlidingWindow => unreachable!(),
            }
        }
    }

    let mut max_logits = ctx
        .stream
        .alloc_zeros::<f32>(token_count * global_heads)
        .map_err(|e| anyhow!("DSv4 FlashMLA prefill max_logits alloc failed: {e}"))?;
    let mut lse = ctx
        .stream
        .alloc_zeros::<f32>(token_count * global_heads)
        .map_err(|e| anyhow!("DSv4 FlashMLA prefill lse alloc failed: {e}"))?;

    let (q_ptr, q_guard) = q_prepared.data.device_ptr(&ctx.stream);
    let (kv_ptr, kv_guard) = kv_unified.data.device_ptr(&ctx.stream);
    let (indices_ptr, indices_guard) = indices.device_ptr(&ctx.stream);
    let (topk_ptr, topk_guard) = topk_length.device_ptr(&ctx.stream);
    let (out_ptr, out_guard) = local_attn.data.device_ptr_mut(&ctx.stream);
    let (max_ptr, max_guard) = max_logits.device_ptr_mut(&ctx.stream);
    let (lse_ptr, lse_guard) = lse.device_ptr_mut(&ctx.stream);

    let local_width = local_heads * config.head_dim;
    let global_width = global_heads * config.head_dim;
    let (q_for_flashmla, flash_out_ptr, mut tp_gathered_q, mut tp_packed_q, mut tp_full_out) =
        if tp_world > 1 {
            let mut gathered = ctx
                .stream
                .alloc_zeros::<half::bf16>(tp_world * token_count * local_width)
                .map_err(|e| anyhow!("DSv4 FlashMLA prefill TP Q gather alloc failed: {e}"))?;
            let mut packed = ctx
                .stream
                .alloc_zeros::<half::bf16>(token_count * global_width)
                .map_err(|e| anyhow!("DSv4 FlashMLA prefill TP Q pack alloc failed: {e}"))?;
            let full_out = ctx
                .stream
                .alloc_zeros::<half::bf16>(token_count * global_width)
                .map_err(|e| anyhow!("DSv4 FlashMLA prefill TP output alloc failed: {e}"))?;
            let (gather_ptr, gather_guard) = gathered.device_ptr_mut(&ctx.stream);
            {
                let _nvtx = crate::nvtx::range("dsv4/flashmla_prefill_q_allgather");
                unsafe {
                    tp.all_gather_bf16_raw(
                        ctx,
                        q_ptr as *const std::ffi::c_void,
                        token_count * local_width,
                        gather_ptr as *mut std::ffi::c_void,
                    )?;
                }
            }
            drop(gather_guard);
            let (packed_ptr, packed_guard) = packed.device_ptr_mut(&ctx.stream);
            {
                let _nvtx = crate::nvtx::range("dsv4/flashmla_prefill_q_repack");
                unsafe {
                    ffi::dsv4_tp_q_repack_cuda(
                        gather_ptr as *const ffi::Half,
                        packed_ptr as *mut ffi::Half,
                        tp_world as i32,
                        token_count as i32,
                        local_heads as i32,
                        config.head_dim as i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()
                    .map_err(|e| anyhow!("DSv4 FlashMLA prefill TP Q repack failed: {e}"))?;
                }
            }
            drop(packed_guard);
            let (full_out_ptr, full_out_guard) = full_out.device_ptr(&ctx.stream);
            drop(full_out_guard);
            (
                packed_ptr as *const ffi::Half,
                full_out_ptr as *mut ffi::Half,
                Some(gathered),
                Some(packed),
                Some(full_out),
            )
        } else {
            (
                q_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                None,
                None,
                None,
            )
        };

    let (sink_base, sink_guard) = attention.attn_sink_f32.device_ptr(&ctx.stream);
    ensure!(
        if tp_world > 1 {
            attention.attn_sink_f32.len() >= global_heads
        } else {
            attention.attn_sink_f32.len() >= tp_rank * local_heads + local_heads
        },
        "DSv4 FlashMLA prefill attn_sink_f32 len {} cannot cover heads",
        attention.attn_sink_f32.len()
    );
    let sink_ptr = if tp_world > 1 {
        sink_base as *const f32
    } else {
        unsafe { (sink_base as *const f32).add(tp_rank * local_heads) }
    };

    {
        let _nvtx = crate::nvtx::range("dsv4/flashmla_prefill_fwd");
        unsafe {
            ffi::arle_flashmla_sm90_sparse_prefill_fwd(
                q_for_flashmla,
                kv_ptr as *const ffi::Half,
                indices_ptr as *const i32,
                sink_ptr,
                topk_ptr as *const i32,
                flash_out_ptr,
                max_ptr as *mut f32,
                lse_ptr as *mut f32,
                token_count as i32,
                kv_rows as i32,
                global_heads as i32,
                1,
                config.head_dim as i32,
                config.head_dim as i32,
                topk_unified as i32,
                sm_scale,
                global_width as i32,
                config.head_dim as i32,
                config.head_dim as i32,
                0,
                topk_unified as i32,
                0,
                0,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|e| anyhow!("DSv4 FlashMLA sparse prefill failed: {e}"))?;
        }
    }

    if tp_world > 1 {
        let full_out = tp_full_out
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 FlashMLA prefill missing TP full output"))?;
        let (full_out_ptr, full_out_guard) = full_out.device_ptr(&ctx.stream);
        {
            let _nvtx = crate::nvtx::range("dsv4/flashmla_prefill_out_slice");
            unsafe {
                ffi::dsv4_tp_out_slice_cuda(
                    full_out_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    token_count as i32,
                    global_width as i32,
                    local_width as i32,
                    (tp_rank * local_width) as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("DSv4 FlashMLA prefill TP out slice failed: {e}"))?;
            }
        }
        drop(full_out_guard);
    }

    {
        let _nvtx = crate::nvtx::range("dsv4/flashmla_prefill_inverse_rope");
        unsafe {
            ffi::arle_dsv4_output_inverse_rope_cuda(
                out_ptr as *mut ffi::Half,
                token_count as i32,
                local_heads as i32,
                config.head_dim as i32,
                config.qk_rope_head_dim as i32,
                start_pos as i32,
                rope_base,
                original_seq_len,
                rope_factor,
                rope_beta_fast,
                rope_beta_slow,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|e| anyhow!("DSv4 FlashMLA prefill output inverse-rope failed: {e}"))?;
        }
    }

    update_bf16_sw_window(ctx, sw_window_cache, k_prepared, start_pos, None, config)?;

    if env_flag("ARLE_DSV4_FLASHMLA_PREFILL_SYNC")? {
        ctx.sync()?;
    }

    // Keep temporary buffers in scope until all launches that use their raw
    // pointers have been enqueued. Optional sync above is available for
    // diagnostics and for conservative lifetime validation on pod.
    drop(tp_gathered_q.take());
    drop(tp_packed_q.take());
    drop(tp_full_out.take());
    drop(q_guard);
    drop(kv_guard);
    drop(indices_guard);
    drop(topk_guard);
    drop(out_guard);
    drop(max_guard);
    drop(lse_guard);
    drop(sink_guard);

    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn try_flashmla_decode_attention(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    attention: &Dsv4Attention,
    mode: DeepSeekV4AttentionMode,
    compress_ratio: usize,
    q_prepared: &HiddenStates,
    k_prepared: &HiddenStates,
    selected: Option<&CudaSlice<i32>>,
    compressed: Option<&HiddenStates>,
    sw_window_cache: &mut CudaSlice<half::bf16>,
    flash: &mut Dsv4FlashMlaDecodeState,
    start_pos: usize,
    start_pos_device: Option<&CudaSlice<i32>>,
    tp: &TpRuntime,
    local_heads: usize,
    local_attn: &mut HiddenStates,
    sm_scale: f32,
    rope_base: f32,
    original_seq_len: i32,
    rope_factor: f32,
    rope_beta_fast: f32,
    rope_beta_slow: f32,
) -> Result<bool> {
    if !dsv4_flashmla_decode_enabled()? {
        return Ok(false);
    }
    if q_prepared.seq_len != 1 {
        return Ok(false);
    }
    let start_pos_device = start_pos_device.ok_or_else(|| {
        anyhow!("DSv4 FlashMLA decode requires device start_pos for token_count=1")
    })?;
    ensure!(
        config.head_dim == 512 && config.qk_rope_head_dim == 64,
        "DSv4 FlashMLA decode only supports MODEL1 head_dim=512 rope_dim=64"
    );
    ensure!(
        local_attn.seq_len == 1,
        "DSv4 FlashMLA decode writes exactly one token"
    );

    let tp_world = tp.config().world_size;
    let tp_rank = tp.config().rank;
    let global_heads = local_heads
        .checked_mul(tp_world)
        .ok_or_else(|| anyhow!("DSv4 FlashMLA global head overflow"))?;
    ensure!(
        matches!(global_heads, 64 | 128),
        "DSv4 FlashMLA decode requires global heads 64/128, got {global_heads}"
    );

    {
        let _nvtx = crate::nvtx::range("dsv4/flashmla_pack_sw_ring");
        flashmla_pack_sw_ring(ctx, flash, sw_window_cache, config)?;
    }

    {
        let _nvtx = crate::nvtx::range("dsv4/flashmla_pack_one");
        flashmla_pack_one_sw_token(ctx, flash, k_prepared, start_pos_device, config)?;
    }
    {
        let _nvtx = crate::nvtx::range("dsv4/flashmla_pack_compressed");
        flashmla_pack_compressed_delta(ctx, flash, compressed, config)?;
    }

    let mode_int = flashmla_mode_int(mode);
    let selected_ptr_u64 = if mode == DeepSeekV4AttentionMode::CompressedSparse {
        let selected =
            selected.ok_or_else(|| anyhow!("DSv4 FlashMLA CSA missing selected topk"))?;
        let (ptr, guard) = selected.device_ptr(&ctx.stream);
        let ptr_u64 = ptr as u64;
        drop(guard);
        ptr_u64
    } else {
        0
    };
    let (indices_ptr, indices_guard) = flash.indices.device_ptr_mut(&ctx.stream);
    let (start_ptr, start_guard) = start_pos_device.device_ptr(&ctx.stream);
    {
        let _nvtx = crate::nvtx::range("dsv4/flashmla_build_indices");
        flash_kv::dsv4_flashmla_decode_build_indices_start_pos_ptr_raw(
            ctx,
            indices_ptr,
            selected_ptr_u64,
            flash.sw_blocks,
            config.sliding_window,
            start_ptr,
            flash.max_compressed_keys,
            if mode == DeepSeekV4AttentionMode::SlidingWindow {
                1
            } else {
                compress_ratio
            },
            mode_int,
            64,
        )?;
    }
    drop(indices_guard);
    drop(start_guard);

    let topk = i32::try_from(flash.topk_unified)
        .map_err(|_| anyhow!("DSv4 FlashMLA topk {} overflows i32", flash.topk_unified))?;
    ctx.stream
        .memcpy_htod(&[topk], &mut flash.topk_length)
        .map_err(|e| anyhow!("DSv4 FlashMLA topk_length H2D failed: {e}"))?;

    let (topk_ptr, topk_guard) = flash.topk_length.device_ptr(&ctx.stream);
    let (sched_ptr, sched_guard) = flash.sched_meta.device_ptr_mut(&ctx.stream);
    let (splits_ptr, splits_guard) = flash.num_splits.device_ptr_mut(&ctx.stream);
    {
        let _nvtx = crate::nvtx::range("dsv4/flashmla_sched_meta");
        unsafe {
            ffi::arle_flashmla_sm90_sparse_decode_sched_meta(
                1,
                1,
                flash.block_size_topk,
                flash.fixed_overhead_num_blocks,
                topk,
                0,
                topk_ptr as *const i32,
                std::ptr::null(),
                sched_ptr as *mut i32,
                splits_ptr as *mut i32,
                flash.num_sm_parts,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|e| anyhow!("DSv4 FlashMLA sched_meta failed: {e}"))?;
        }
    }
    drop(topk_guard);
    drop(sched_guard);
    drop(splits_guard);

    let (q_ptr, q_guard) = q_prepared.data.device_ptr(&ctx.stream);
    let (pool_ptr, pool_guard) = flash.fp8_kv_pool.device_ptr(&ctx.stream);
    let (out_ptr, out_guard) = local_attn.data.device_ptr_mut(&ctx.stream);
    let (lse_out_ptr, lse_guard) = flash.lse_out.device_ptr_mut(&ctx.stream);
    let (lse_accum_ptr, lse_accum_guard) = flash.lse_accum.device_ptr_mut(&ctx.stream);
    let (o_accum_ptr, o_accum_guard) = flash.o_accum.device_ptr_mut(&ctx.stream);
    let (indices_ptr, indices_guard) = flash.indices.device_ptr(&ctx.stream);
    let (topk_ptr, topk_guard) = flash.topk_length.device_ptr(&ctx.stream);
    let (sched_ptr, sched_guard) = flash.sched_meta.device_ptr(&ctx.stream);
    let (splits_ptr, splits_guard) = flash.num_splits.device_ptr(&ctx.stream);

    let q_for_flashmla = if tp_world > 1 {
        let (gather_ptr, gather_guard) = flash.tp_gathered_q.device_ptr_mut(&ctx.stream);
        {
            let _nvtx = crate::nvtx::range("dsv4/flashmla_q_allgather");
            unsafe {
                tp.all_gather_bf16_raw(
                    ctx,
                    q_ptr as *const std::ffi::c_void,
                    local_heads * config.head_dim,
                    gather_ptr as *mut std::ffi::c_void,
                )?;
            }
        }
        drop(gather_guard);
        let (packed_ptr, packed_guard) = flash.tp_packed_q.device_ptr_mut(&ctx.stream);
        {
            let _nvtx = crate::nvtx::range("dsv4/flashmla_q_repack");
            unsafe {
                ffi::dsv4_tp_q_repack_cuda(
                    gather_ptr as *const ffi::Half,
                    packed_ptr as *mut ffi::Half,
                    tp_world as i32,
                    1,
                    local_heads as i32,
                    config.head_dim as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("DSv4 FlashMLA TP Q repack failed: {e}"))?;
            }
        }
        drop(packed_guard);
        packed_ptr as *const ffi::Half
    } else {
        q_ptr as *const ffi::Half
    };

    let (sink_base, sink_guard) = attention.attn_sink_f32.device_ptr(&ctx.stream);
    ensure!(
        if tp_world > 1 {
            attention.attn_sink_f32.len() >= global_heads
        } else {
            attention.attn_sink_f32.len() >= tp_rank * local_heads + local_heads
        },
        "DSv4 FlashMLA attn_sink_f32 len {} cannot cover heads",
        attention.attn_sink_f32.len()
    );
    let sink_ptr = if tp_world > 1 {
        sink_base as *const f32
    } else {
        unsafe { (sink_base as *const f32).add(tp_rank * local_heads) }
    };

    let flash_out_ptr = if tp_world > 1 {
        let (full_out_ptr, full_out_guard) = flash.tp_full_out.device_ptr_mut(&ctx.stream);
        drop(full_out_guard);
        full_out_ptr as *mut ffi::Half
    } else {
        out_ptr as *mut ffi::Half
    };

    let bytes_per_token = 584_i32;
    let stride_kv_block_bytes = 64_i32 * bytes_per_token;
    let stride_q = (global_heads * config.head_dim) as i32;
    let stride_o = stride_q;
    let stride_indices = flash.topk_unified as i32;
    let stride_lse = global_heads as i32;
    {
        let _nvtx = crate::nvtx::range("dsv4/flashmla_fwd");
        unsafe {
            ffi::arle_flashmla_sm90_sparse_decode_fwd(
                q_for_flashmla,
                pool_ptr as *const ffi::Half,
                indices_ptr as *const i32,
                topk_ptr as *const i32,
                sink_ptr,
                flash_out_ptr,
                lse_out_ptr as *mut f32,
                lse_accum_ptr as *mut f32,
                o_accum_ptr as *mut f32,
                sched_ptr as *const i32,
                splits_ptr as *const i32,
                1,
                1,
                global_heads as i32,
                1,
                config.head_dim as i32,
                config.head_dim as i32,
                (flash.sw_blocks + flash.comp_blocks) as i32,
                64,
                stride_indices,
                flash.num_sm_parts,
                DSV4_FLASHMLA_MODEL1,
                sm_scale,
                stride_q,
                stride_q,
                config.head_dim as i32,
                stride_kv_block_bytes,
                bytes_per_token,
                stride_indices,
                stride_indices,
                stride_lse,
                1,
                stride_o,
                stride_o,
                config.head_dim as i32,
                global_heads as i32,
                global_heads as i32,
                stride_o,
                stride_o,
                config.head_dim as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|e| anyhow!("DSv4 FlashMLA sparse decode failed: {e}"))?;
        }
    }

    if tp_world > 1 {
        let (full_out_ptr, full_out_guard) = flash.tp_full_out.device_ptr(&ctx.stream);
        {
            let _nvtx = crate::nvtx::range("dsv4/flashmla_out_slice");
            unsafe {
                ffi::dsv4_tp_out_slice_cuda(
                    full_out_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    1,
                    (global_heads * config.head_dim) as i32,
                    (local_heads * config.head_dim) as i32,
                    (tp_rank * local_heads * config.head_dim) as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("DSv4 FlashMLA TP out slice failed: {e}"))?;
            }
        }
        drop(full_out_guard);
    }

    {
        let _nvtx = crate::nvtx::range("dsv4/flashmla_inverse_rope");
        unsafe {
            ffi::arle_dsv4_output_inverse_rope_start_pos_ptr_cuda(
                out_ptr as *mut ffi::Half,
                1,
                local_heads as i32,
                config.head_dim as i32,
                config.qk_rope_head_dim as i32,
                start_ptr as *const i32,
                rope_base,
                original_seq_len,
                rope_factor,
                rope_beta_fast,
                rope_beta_slow,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|e| anyhow!("DSv4 FlashMLA output inverse-rope failed: {e}"))?;
        }
    }

    drop(q_guard);
    drop(pool_guard);
    drop(out_guard);
    drop(lse_guard);
    drop(lse_accum_guard);
    drop(o_accum_guard);
    drop(indices_guard);
    drop(topk_guard);
    drop(sched_guard);
    drop(splits_guard);
    drop(sink_guard);

    update_bf16_sw_window(
        ctx,
        sw_window_cache,
        k_prepared,
        start_pos,
        Some(start_pos_device),
        config,
    )?;
    Ok(true)
}

/// RMSNorm a `HiddenStates` in place into a fresh buffer (the MLA Q/KV LoRA
/// norms `q_norm` / `kv_norm`). Thin wrapper over the shared batched RMSNorm.
fn mla_rms_norm(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
) -> Result<HiddenStates> {
    // SAFETY: rms_norm_batched_cuda writes the full output buffer.
    let mut out = unsafe { HiddenStates::uninit(ctx, x.hidden_dim, x.seq_len)? };
    {
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
        // SAFETY: buffers valid on ctx.stream; out matches x shape.
        unsafe {
            ffi::rms_norm_batched_cuda(
                x_ptr as *const ffi::Half,
                w_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                x.hidden_dim as i32,
                x.seq_len as i32,
                eps,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }
    Ok(out)
}

fn mla_rms_norm_decode_slice(
    ctx: &DeviceContext,
    x: &HiddenStates,
    offset: usize,
    width: usize,
    weight: &DeviceVec,
    eps: f32,
) -> Result<HiddenStates> {
    ensure!(
        x.seq_len == 1,
        "DSv4 fused wqkv slice RMSNorm is decode-only, got seq_len={}",
        x.seq_len
    );
    ensure!(
        offset + width <= x.hidden_dim,
        "DSv4 fused wqkv slice out of range: offset={offset} width={width} hidden_dim={}",
        x.hidden_dim
    );
    ensure!(
        weight.len == width,
        "DSv4 fused wqkv slice norm weight len {} != slice width {width}",
        weight.len
    );
    let mut out = unsafe { HiddenStates::uninit(ctx, width, 1)? };
    {
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
        let x_ptr = unsafe { (x_ptr as *const ffi::Half).add(offset) };
        unsafe {
            ffi::rms_norm_batched_cuda(
                x_ptr,
                w_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                width as i32,
                1,
                eps,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }
    Ok(out)
}

fn run_fused_wqkv_decode(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    attention: &Dsv4Attention,
    hidden: &HiddenStates,
    scratch: &mut Dsv4FusedWqkvDecodeScratch,
) -> Result<(HiddenStates, HiddenStates, HiddenStates)> {
    ensure!(
        hidden.seq_len == 1,
        "DSv4 fused wqkv decode path requires seq_len=1, got {}",
        hidden.seq_len
    );
    ensure!(
        hidden.hidden_dim == scratch.hidden_dim && hidden.hidden_dim == config.hidden_size,
        "DSv4 fused wqkv hidden dim mismatch: hidden={} scratch={} config={}",
        hidden.hidden_dim,
        scratch.hidden_dim,
        config.hidden_size
    );
    ensure!(
        scratch.q_lora_rank == attention.wq_a.rows && scratch.head_dim == attention.wkv.rows,
        "DSv4 fused wqkv scratch shape mismatch: scratch q={} kv={} weights q={} kv={}",
        scratch.q_lora_rank,
        scratch.head_dim,
        attention.wq_a.rows,
        attention.wkv.rows
    );
    let cache = attention.wqkv_a_deepgemm.as_ref().ok_or_else(|| {
        anyhow!("DSv4 fused wqkv decode requested but fused cache was not loaded")
    })?;
    ensure!(
        cache.rows == scratch.q_lora_rank + scratch.head_dim && cache.cols == scratch.hidden_dim,
        "DSv4 fused wqkv cache shape {}x{} != expected {}x{}",
        cache.rows,
        cache.cols,
        scratch.q_lora_rank + scratch.head_dim,
        scratch.hidden_dim
    );
    let scale_cols = scratch.hidden_dim.div_ceil(128);
    ensure!(
        scratch.input_scales.len() >= scratch.scale_stride_m * scale_cols,
        "DSv4 fused wqkv scale scratch too small"
    );
    let stream = ctx.stream.cu_stream();
    unsafe {
        cuda_moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
            cache_ptr(&hidden.data, ctx),
            cache_ptr(&scratch.input_fp8, ctx),
            cache_ptr(&scratch.input_scales, ctx),
            cache_ptr(&scratch.active_experts, ctx),
            cache_ptr(&scratch.active_offsets, ctx),
            cache_ptr(&scratch.active_counts, ctx),
            1,
            scratch.max_m,
            scratch.hidden_dim,
            scratch.scale_stride_m,
            stream,
        )
        .map_err(|e| anyhow!("DSv4 fused wqkv activation quantize failed: {e}"))?;
        cuda_moe::dsv4_deepgemm_fp8_gemm_nt(
            cache_ptr(&scratch.input_fp8, ctx),
            cache_ptr(&scratch.input_scales, ctx),
            cache_ptr(&cache.weight, ctx),
            cache_ptr(&cache.scales, ctx),
            cache_ptr(&scratch.qkv_raw.data, ctx),
            1,
            cache.rows,
            cache.cols,
            scratch.scale_stride_m,
            stream,
        )
        .map_err(|e| anyhow!("DSv4 fused wqkv DeepGEMM dense failed: {e}"))?;
    }
    let c_q_normed = mla_rms_norm_decode_slice(
        ctx,
        &scratch.qkv_raw,
        0,
        scratch.q_lora_rank,
        &attention.q_norm,
        config.rms_norm_eps,
    )?;
    let kv_normed = mla_rms_norm_decode_slice(
        ctx,
        &scratch.qkv_raw,
        scratch.q_lora_rank,
        scratch.head_dim,
        &attention.kv_norm,
        config.rms_norm_eps,
    )?;

    let mut q_raw = unsafe { HiddenStates::uninit(ctx, attention.wq_b.rows, 1)? };
    let nvtx_wq_b = crate::nvtx::range("dsv4/linear/wq_b");
    crate::linear_profile::profile(ctx, "dsv4/linear/wq_b", || {
        dsv4_linear(ctx, &attention.wq_b, &c_q_normed, &mut q_raw)
    })?;
    drop(nvtx_wq_b);
    Ok((c_q_normed, q_raw, kv_normed))
}

/// One DSv4 MLA attention block (SlidingWindow / CompressedSparse /
/// HybridCompressed, dispatched on `mode` / `compress_ratio`).
///
/// `hidden` is the post-attn-LN input `[hidden_size, token_count]`;
/// `state` holds this layer's per-slot bf16 sliding-window ring plus compressor
/// pending/compressed pools. `start_pos` is the absolute position of `hidden`'s
/// first token (0 for a fresh prefill). Writes `[hidden_size, token_count]` into
/// `out` (the O-LoRA output, pre-TP-all-reduce — the model layer-loop owns the
/// row-parallel sum). FlashMLA-FP8 decode stays gated (perf path).
///
/// `tp_rank` is this rank's tensor-parallel index. The per-head `attn_sink`
/// vector is loaded WHOLE on every rank (no TP slice), so the SW/hybrid kernels
/// must skip to this rank's head block via `sink_offset = tp_rank * local_heads`
/// — otherwise every non-zero rank reads rank-0's sink logits and the attention
/// output diverges by a small head-dependent margin (multi-GPU only).
#[allow(clippy::too_many_arguments)]
pub(crate) fn mla_attention(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    attention: &Dsv4Attention,
    mode: DeepSeekV4AttentionMode,
    compress_ratio: usize,
    layer_idx: usize,
    hidden: &HiddenStates,
    state: &mut Dsv4LayerAttentionState,
    start_pos: usize,
    start_pos_device: Option<&CudaSlice<i32>>,
    tp: &TpRuntime,
    out: &mut HiddenStates,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<()> {
    ensure!(
        hidden.hidden_dim == config.hidden_size,
        "DSv4 MLA hidden dim {} != hidden_size {}",
        hidden.hidden_dim,
        config.hidden_size
    );

    let head_dim = config.head_dim;
    let token_count = hidden.seq_len;
    let local_width = attention.wq_b.rows;
    ensure!(
        head_dim > 0 && local_width.is_multiple_of(head_dim),
        "DSv4 MLA local q width {local_width} is not a multiple of head_dim {head_dim}"
    );
    let local_heads = local_width / head_dim;
    ensure!(local_heads > 0, "DSv4 MLA requires at least one local head");
    let tp_rank = tp.config().rank;
    // This rank owns global heads [tp_rank*local_heads, +local_heads); the
    // whole-loaded attn_sink must be indexed from that offset (see fn docs).
    let sink_offset = tp_rank * local_heads;
    ensure!(
        attention.wkv.rows == head_dim,
        "DSv4 MLA wkv rows {} != head_dim {head_dim}",
        attention.wkv.rows
    );
    ensure!(
        attention.wo_a.cols == local_width,
        "DSv4 MLA wo_a cols {} != local attention width {local_width}",
        attention.wo_a.cols
    );
    ensure!(
        attention.wo_b.rows == out.hidden_dim && out.seq_len == token_count,
        "DSv4 MLA output shape mismatch: wo_b rows {} out {}x{} expected {}x{}",
        attention.wo_b.rows,
        out.hidden_dim,
        out.seq_len,
        attention.wo_b.rows,
        token_count
    );
    ensure!(
        config.sliding_window > 0,
        "DSv4 MLA requires a non-zero sliding_window"
    );
    ensure!(
        config.qk_rope_head_dim <= head_dim,
        "DSv4 MLA rope dim {} exceeds head_dim {head_dim}",
        config.qk_rope_head_dim
    );
    ensure!(
        state.sw_window_cache.len() == config.sliding_window * head_dim,
        "DSv4 MLA SW window cache len {} != sliding_window*head_dim {}",
        state.sw_window_cache.len(),
        config.sliding_window * head_dim
    );
    ensure!(
        attention.attn_sink.len >= sink_offset + local_heads,
        "DSv4 MLA attn_sink len {} cannot cover rank {tp_rank} heads [{sink_offset}, {})",
        attention.attn_sink.len,
        sink_offset + local_heads
    );

    let rope = &config.rope_parameters;
    // Q / SW-K / output RoPE is ALWAYS the main rope_theta with NO YaRN, for
    // every layer regardless of compress_ratio (only the COMPRESSED keys use
    // compress_rope_theta — applied inside `compressor_forward`). This matches
    // the validated long-context fix (errors/2026-05-29-dsv4-longctx-rope...).
    let rope_base = config.rope_theta;
    let original_seq_len = 0i32;
    let start_pos_i32 = i32::try_from(start_pos)
        .map_err(|_| anyhow::anyhow!("DSv4 MLA start_pos {start_pos} overflows i32"))?;

    // ── 1+2. Q/KV LoRA. SGLang fuses the first projections
    // (`wq_a | wkv`) into one wqkv_a call. ARLE only enables that structure for
    // B=1 decode: the fused output's two slices are contiguous, so the existing
    // RMSNorm kernel can consume raw pointer offsets without adding a split
    // kernel. Multi-token prefill stays on the scalar reference path until ARLE
    // grows strided HiddenStates views or a fused split+norm kernel.
    let fused_wqkv = token_count == 1 && dsv4_fused_wqkv_decode_enabled()?;
    let (c_q_normed, q_raw, kv_normed) = if fused_wqkv {
        let scratch = state.fused_wqkv.as_mut().ok_or_else(|| {
            anyhow!("DSv4 fused wqkv decode requested but decode scratch was not allocated")
        })?;
        let nvtx_wqkv = crate::nvtx::range("dsv4/linear/wqkv_a_fused");
        let out = crate::linear_profile::profile(ctx, "dsv4/linear/wqkv_a_fused", || {
            run_fused_wqkv_decode(ctx, config, attention, hidden, scratch)
        })?;
        drop(nvtx_wqkv);
        out
    } else {
        // Q-LoRA: wq_a (down) → q_norm RMSNorm → wq_b (up to per-head Q).
        // SAFETY: dsv4_linear writes the full c_q buffer.
        let mut c_q = unsafe { HiddenStates::uninit(ctx, attention.wq_a.rows, token_count)? };
        let nvtx_wq_a = crate::nvtx::range("dsv4/linear/wq_a");
        crate::linear_profile::profile(ctx, "dsv4/linear/wq_a", || {
            dsv4_linear(ctx, &attention.wq_a, hidden, &mut c_q)
        })?;
        drop(nvtx_wq_a);
        keepalive.keep_hidden(&c_q);
        let c_q_normed = mla_rms_norm(ctx, &c_q, &attention.q_norm, config.rms_norm_eps)?;
        keepalive.keep_hidden(&c_q_normed);
        // SAFETY: dsv4_linear writes the full q_raw buffer.
        let mut q_raw = unsafe { HiddenStates::uninit(ctx, local_width, token_count)? };
        let nvtx_wq_b = crate::nvtx::range("dsv4/linear/wq_b");
        crate::linear_profile::profile(ctx, "dsv4/linear/wq_b", || {
            dsv4_linear(ctx, &attention.wq_b, &c_q_normed, &mut q_raw)
        })?;
        drop(nvtx_wq_b);
        keepalive.keep_hidden(&q_raw);

        // KV latent: wkv (down to the single compressed latent) → kv_norm.
        // SAFETY: dsv4_linear writes the full kv_raw buffer.
        let mut kv_raw = unsafe { HiddenStates::uninit(ctx, head_dim, token_count)? };
        let nvtx_wkv = crate::nvtx::range("dsv4/linear/wkv");
        crate::linear_profile::profile(ctx, "dsv4/linear/wkv", || {
            dsv4_linear(ctx, &attention.wkv, hidden, &mut kv_raw)
        })?;
        drop(nvtx_wkv);
        keepalive.keep_hidden(&kv_raw);
        let kv_normed = mla_rms_norm(ctx, &kv_raw, &attention.kv_norm, config.rms_norm_eps)?;
        keepalive.keep_hidden(&kv_normed);
        (c_q_normed, q_raw, kv_normed)
    };
    keepalive.keep_hidden(&c_q_normed);
    keepalive.keep_hidden(&q_raw);
    keepalive.keep_hidden(&kv_normed);

    // ── 3. Partial RoPE on the trailing rope_dim cols of Q (per head) and K.
    // SAFETY: dsv4_prepare_qk_cuda writes both full output buffers.
    let mut q_prepared = unsafe { HiddenStates::uninit(ctx, local_width, token_count)? };
    let mut k_prepared = unsafe { HiddenStates::uninit(ctx, head_dim, token_count)? };
    {
        let (q_raw_ptr, _qr) = q_raw.data.device_ptr(&ctx.stream);
        let (k_raw_ptr, _kr) = kv_normed.data.device_ptr(&ctx.stream);
        let (q_out_ptr, _qo) = q_prepared.data.device_ptr_mut(&ctx.stream);
        let (k_out_ptr, _ko) = k_prepared.data.device_ptr_mut(&ctx.stream);
        // SAFETY: all buffers valid on ctx.stream; head/dim args checked above.
        unsafe {
            if let Some(start_pos_device) = start_pos_device {
                let (start_ptr, _sg) = start_pos_device.device_ptr(&ctx.stream);
                ffi::dsv4_prepare_qk_start_pos_ptr_cuda(
                    q_raw_ptr as *const ffi::Half,
                    k_raw_ptr as *const ffi::Half,
                    q_out_ptr as *mut ffi::Half,
                    k_out_ptr as *mut ffi::Half,
                    token_count as i32,
                    local_heads as i32,
                    head_dim as i32,
                    config.qk_rope_head_dim as i32,
                    start_ptr as *const i32,
                    config.rms_norm_eps,
                    rope_base,
                    original_seq_len,
                    rope.factor,
                    rope.beta_fast,
                    rope.beta_slow,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            } else {
                ffi::dsv4_prepare_qk_cuda(
                    q_raw_ptr as *const ffi::Half,
                    k_raw_ptr as *const ffi::Half,
                    q_out_ptr as *mut ffi::Half,
                    k_out_ptr as *mut ffi::Half,
                    token_count as i32,
                    local_heads as i32,
                    head_dim as i32,
                    config.qk_rope_head_dim as i32,
                    start_pos_i32,
                    config.rms_norm_eps,
                    rope_base,
                    original_seq_len,
                    rope.factor,
                    rope.beta_fast,
                    rope.beta_slow,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }
    }
    keepalive.keep_hidden(&q_prepared);
    keepalive.keep_hidden(&k_prepared);

    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
    // SAFETY: the SW/hybrid attention kernel writes the full local_attn buffer.
    let mut local_attn = unsafe { HiddenStates::uninit(ctx, local_width, token_count)? };

    if mode == DeepSeekV4AttentionMode::SlidingWindow {
        // ── 4a. SW: windowed attention + per-head sink + output inverse-RoPE.
        // The kernel reads the pre-roped q/k, attends over the bf16 SW ring cache
        // (which it also updates), adds the sink, and un-rotates the rope tail of
        // the OUTPUT (sign = -1) before returning.
        let flashmla_used = if dsv4_flashmla_decode_enabled()? {
            let flash = state.flashmla.as_mut().ok_or_else(|| {
                anyhow!("ARLE_DSV4_FLASHMLA_DECODE=1 but layer state has no FlashMLA arena")
            })?;
            try_flashmla_decode_attention(
                ctx,
                config,
                attention,
                mode,
                compress_ratio,
                &q_prepared,
                &k_prepared,
                None,
                None,
                &mut state.sw_window_cache,
                flash,
                start_pos,
                start_pos_device,
                tp,
                local_heads,
                &mut local_attn,
                sm_scale,
                rope_base,
                original_seq_len,
                rope.factor,
                rope.beta_fast,
                rope.beta_slow,
            )?
        } else {
            false
        };
        if !flashmla_used {
            let (q_ptr, _qg) = q_prepared.data.device_ptr(&ctx.stream);
            let (k_ptr, _kg) = k_prepared.data.device_ptr(&ctx.stream);
            let (window_ptr, _wg) = state.sw_window_cache.device_ptr_mut(&ctx.stream);
            let (sink_ptr, _sg) = attention.attn_sink.data.device_ptr(&ctx.stream);
            let (out_ptr, _og) = local_attn.data.device_ptr_mut(&ctx.stream);
            // SAFETY: all buffers valid on ctx.stream; window sized above; sink_offset
            // skips to this rank's head block in the whole-loaded attn_sink vector.
            unsafe {
                if let Some(start_pos_device) = start_pos_device {
                    let (start_ptr, _spg) = start_pos_device.device_ptr(&ctx.stream);
                    ffi::dsv4_swa_attention_start_pos_ptr_cuda(
                        q_ptr as *const ffi::Half,
                        k_ptr as *const ffi::Half,
                        window_ptr as *mut ffi::Half,
                        sink_ptr as *const ffi::Half,
                        out_ptr as *mut ffi::Half,
                        token_count as i32,
                        local_heads as i32,
                        head_dim as i32,
                        config.sliding_window as i32,
                        start_ptr as *const i32,
                        sink_offset as i32,
                        sm_scale,
                        config.qk_rope_head_dim as i32,
                        rope_base,
                        original_seq_len,
                        rope.factor,
                        rope.beta_fast,
                        rope.beta_slow,
                        1,
                        ctx.stream.cu_stream(),
                    )
                    .result()?;
                } else {
                    ffi::dsv4_swa_attention_cuda(
                        q_ptr as *const ffi::Half,
                        k_ptr as *const ffi::Half,
                        window_ptr as *mut ffi::Half,
                        sink_ptr as *const ffi::Half,
                        out_ptr as *mut ffi::Half,
                        token_count as i32,
                        local_heads as i32,
                        head_dim as i32,
                        config.sliding_window as i32,
                        start_pos_i32,
                        sink_offset as i32,
                        sm_scale,
                        config.qk_rope_head_dim as i32,
                        rope_base,
                        original_seq_len,
                        rope.factor,
                        rope.beta_fast,
                        rope.beta_slow,
                        1,
                        ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
            }
        }
    } else {
        // ── 4b. CSA / HCA: compressor → (CSA) indexer top-k select → hybrid
        // windowed+compressed attention.
        let compressor = attention.compressor.as_ref().ok_or_else(|| {
            anyhow::anyhow!("DSv4 layer {layer_idx} is {mode:?} but has no compressor weights")
        })?;
        let overlap = compress_ratio < 16;
        {
            let compressor_state = state.compressor.as_mut().ok_or_else(|| {
                anyhow::anyhow!("DSv4 layer {layer_idx} is {mode:?} but has no compressor state")
            })?;
            compressor_forward(
                ctx,
                config,
                compressor,
                hidden,
                compressor_state,
                head_dim,
                compress_ratio,
                overlap,
                start_pos,
                start_pos_device,
                true,
                keepalive,
            )?;
        }

        let selected = if mode == DeepSeekV4AttentionMode::CompressedSparse {
            let indexer = attention.indexer.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "DSv4 layer {layer_idx} is CompressedSparse but has no indexer weights"
                )
            })?;
            // Indexer keys: a second compressor over index_head_dim keys (no APE
            // gate on the keys — `apply_rope = true`, head_dim = index_head_dim).
            {
                let indexer_state = state.indexer.as_mut().ok_or_else(|| {
                    anyhow::anyhow!(
                        "DSv4 layer {layer_idx} is CompressedSparse but has no indexer state"
                    )
                })?;
                compressor_forward(
                    ctx,
                    config,
                    &indexer.compressor,
                    hidden,
                    indexer_state,
                    config.index_head_dim,
                    compress_ratio,
                    true,
                    start_pos,
                    start_pos_device,
                    false,
                    keepalive,
                )?;
            }
            let index_keys = &state
                .indexer
                .as_ref()
                .expect("indexer state checked above")
                .compressed;
            Some(csa_select(
                ctx,
                config,
                indexer,
                hidden,
                &c_q_normed,
                index_keys,
                start_pos,
                start_pos_device,
                compress_ratio,
                keepalive,
            )?)
        } else {
            None
        };

        let compressed = &state
            .compressor
            .as_ref()
            .expect("compressor state checked above")
            .compressed;
        let compressed_count = compressed.seq_len;
        let compressed_capacity = compressed.data.len() / head_dim;
        let compressed_count_arg = if start_pos_device.is_some() {
            // CUDA graph replay bakes scalar launch args. In decode, the causal
            // bound is `abs_pos / compress_ratio`, so the kernel may safely see
            // the fixed capacity instead of the current compressed seq_len.
            compressed_capacity
        } else {
            compressed_count
        };
        let mode_int = match mode {
            DeepSeekV4AttentionMode::CompressedSparse => 1,
            DeepSeekV4AttentionMode::HybridCompressed => 2,
            DeepSeekV4AttentionMode::SlidingWindow => unreachable!(),
        };
        let flashmla_used = if try_flashmla_prefill_attention(
            ctx,
            config,
            attention,
            mode,
            compress_ratio,
            &q_prepared,
            &k_prepared,
            selected.as_ref(),
            compressed,
            &mut state.sw_window_cache,
            start_pos,
            tp,
            local_heads,
            &mut local_attn,
            sm_scale,
            rope_base,
            original_seq_len,
            rope.factor,
            rope.beta_fast,
            rope.beta_slow,
        )? {
            true
        } else if dsv4_flashmla_decode_enabled()? {
            let flash = state.flashmla.as_mut().ok_or_else(|| {
                anyhow!("ARLE_DSV4_FLASHMLA_DECODE=1 but layer state has no FlashMLA arena")
            })?;
            try_flashmla_decode_attention(
                ctx,
                config,
                attention,
                mode,
                compress_ratio,
                &q_prepared,
                &k_prepared,
                selected.as_ref(),
                Some(compressed),
                &mut state.sw_window_cache,
                flash,
                start_pos,
                start_pos_device,
                tp,
                local_heads,
                &mut local_attn,
                sm_scale,
                rope_base,
                original_seq_len,
                rope.factor,
                rope.beta_fast,
                rope.beta_slow,
            )?
        } else {
            false
        };
        if !flashmla_used {
            let (q_ptr, _qg) = q_prepared.data.device_ptr(&ctx.stream);
            let (k_ptr, _kg) = k_prepared.data.device_ptr(&ctx.stream);
            let (window_ptr, _wg) = state.sw_window_cache.device_ptr_mut(&ctx.stream);
            let (sink_ptr, _sg) = attention.attn_sink.data.device_ptr(&ctx.stream);
            let (out_ptr, _og) = local_attn.data.device_ptr_mut(&ctx.stream);
            let (comp_ptr, _cguard) = if compressed_count_arg > 0 {
                let (p, g) = compressed.data.device_ptr(&ctx.stream);
                (p as *const ffi::Half, Some(g))
            } else {
                (std::ptr::null(), None)
            };
            let (sel_ptr, _sguard) = match selected.as_ref() {
                Some(sel) => {
                    let (p, g) = sel.device_ptr(&ctx.stream);
                    (p as *const i32, Some(g))
                }
                None => (std::ptr::null(), None),
            };
            // SAFETY: all buffers valid on ctx.stream; compressed/selected may be null
            // (the kernel branches on compressed_count / mode). write_window_cache=1
            // updates the bf16 SW ring inline.
            unsafe {
                if let Some(start_pos_device) = start_pos_device {
                    let (start_ptr, _spg) = start_pos_device.device_ptr(&ctx.stream);
                    ffi::dsv4_hybrid_attention_start_pos_ptr_cuda(
                        q_ptr as *const ffi::Half,
                        k_ptr as *const ffi::Half,
                        window_ptr as *mut ffi::Half,
                        comp_ptr,
                        sel_ptr,
                        sink_ptr as *const ffi::Half,
                        out_ptr as *mut ffi::Half,
                        token_count as i32,
                        local_heads as i32,
                        head_dim as i32,
                        config.sliding_window as i32,
                        start_ptr as *const i32,
                        sink_offset as i32,
                        sm_scale,
                        config.qk_rope_head_dim as i32,
                        rope_base,
                        original_seq_len,
                        rope.factor,
                        rope.beta_fast,
                        rope.beta_slow,
                        mode_int,
                        compress_ratio as i32,
                        compressed_count_arg as i32,
                        config.index_topk as i32,
                        1,
                        ctx.stream.cu_stream(),
                    )
                    .result()?;
                } else {
                    ffi::dsv4_hybrid_attention_cuda(
                        q_ptr as *const ffi::Half,
                        k_ptr as *const ffi::Half,
                        window_ptr as *mut ffi::Half,
                        comp_ptr,
                        sel_ptr,
                        sink_ptr as *const ffi::Half,
                        out_ptr as *mut ffi::Half,
                        token_count as i32,
                        local_heads as i32,
                        head_dim as i32,
                        config.sliding_window as i32,
                        start_pos_i32,
                        sink_offset as i32,
                        sm_scale,
                        config.qk_rope_head_dim as i32,
                        rope_base,
                        original_seq_len,
                        rope.factor,
                        rope.beta_fast,
                        rope.beta_slow,
                        mode_int,
                        compress_ratio as i32,
                        compressed_count_arg as i32,
                        config.index_topk as i32,
                        1,
                        ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
            }
        }
        if let Some(sel) = selected.as_ref() {
            keepalive.keep_i32(sel);
        }
    }
    keepalive.keep_hidden(&local_attn);

    if dsv4_attn_dump_enabled() {
        dsv4_dump_attn_output(ctx, layer_idx, mode, &local_attn)?;
    }

    // ── 5. O-LoRA: wo_a (per o-group, down to the output latent) → wo_b (up
    // back to hidden). Row-parallel: the all-reduce-sum is the model's concern.
    // SAFETY: dsv4_linear writes the full latent buffer.
    let mut latent = unsafe { HiddenStates::uninit(ctx, attention.wo_a.rows, token_count)? };
    let nvtx_wo_a = crate::nvtx::range("dsv4/linear/wo_a");
    crate::linear_profile::profile(ctx, "dsv4/linear/wo_a", || {
        dsv4_linear(ctx, &attention.wo_a, &local_attn, &mut latent)
    })?;
    drop(nvtx_wo_a);
    keepalive.keep_hidden(&latent);
    let nvtx_wo_b = crate::nvtx::range("dsv4/linear/wo_b");
    crate::linear_profile::profile(ctx, "dsv4/linear/wo_b", || {
        dsv4_linear(ctx, &attention.wo_b, &latent, out)
    })?;
    drop(nvtx_wo_b);
    Ok(())
}

/// Run one compressor sub-block over `hidden`, updating the per-slot bf16
/// compressed-key pool for the absolute `[0, start_pos + token_count)` range.
///
/// `wkv`/`wgate` project the hidden into the per-block KV / gating-score streams
/// (`width = 2*head_dim` when `overlap`, else `head_dim`); `dsv4_compressor_update_cuda`
/// folds them through `ape` + RMSNorm(`norm`) + compress-rope into one row per
/// `compress_ratio` tokens. `apply_rope = false` skips the rope tail (indexer
/// keys).
#[allow(clippy::too_many_arguments)]
fn compressor_forward(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    compressor: &Dsv4Compressor,
    hidden: &HiddenStates,
    state: &mut Dsv4CompressorState,
    head_dim: usize,
    ratio: usize,
    overlap: bool,
    start_pos: usize,
    start_pos_device: Option<&CudaSlice<i32>>,
    apply_rope: bool,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<()> {
    ensure!(ratio > 0, "DSv4 compressor ratio must be non-zero");
    let width = if overlap { 2 * head_dim } else { head_dim };
    ensure!(
        compressor.wkv.rows == width && compressor.wgate.rows == width,
        "DSv4 compressor rows mismatch: wkv={} wgate={} expected width={width}",
        compressor.wkv.rows,
        compressor.wgate.rows
    );
    let token_count = hidden.seq_len;
    let total = start_pos + token_count;
    let compressed_rows = total / ratio;
    let start_pos_i32 = i32::try_from(start_pos)
        .map_err(|_| anyhow::anyhow!("DSv4 compressor start_pos {start_pos} exceeds i32"))?;
    let pending_len = start_pos % ratio;
    let pending_len_i32 = i32::try_from(pending_len)
        .map_err(|_| anyhow::anyhow!("DSv4 compressor pending_len {pending_len} exceeds i32"))?;
    let compressed_base = start_pos / ratio;
    let compressed_base_i32 = i32::try_from(compressed_base).map_err(|_| {
        anyhow::anyhow!("DSv4 compressor compressed_base {compressed_base} exceeds i32")
    })?;
    ensure!(
        state.compressed.hidden_dim == head_dim,
        "DSv4 compressor state hidden_dim {} != head_dim {head_dim}",
        state.compressed.hidden_dim
    );
    let compressed_capacity = state.compressed.data.len() / head_dim;
    ensure!(
        compressed_rows <= compressed_capacity,
        "DSv4 compressor compressed rows {compressed_rows} exceed state capacity {compressed_capacity}"
    );

    // SAFETY: dsv4_linear writes the full compressor kv buffer.
    let mut kv_raw = unsafe { HiddenStates::uninit(ctx, width, token_count)? };
    let nvtx_compressor_wkv = crate::nvtx::range("dsv4/linear/compressor_wkv");
    crate::linear_profile::profile(ctx, "dsv4/linear/compressor_wkv", || {
        dsv4_linear(ctx, &compressor.wkv, hidden, &mut kv_raw)
    })?;
    drop(nvtx_compressor_wkv);
    keepalive.keep_hidden(&kv_raw);
    // SAFETY: dsv4_linear writes the full compressor score buffer.
    let mut score_raw = unsafe { HiddenStates::uninit(ctx, width, token_count)? };
    let nvtx_compressor_wgate = crate::nvtx::range("dsv4/linear/compressor_wgate");
    crate::linear_profile::profile(ctx, "dsv4/linear/compressor_wgate", || {
        dsv4_linear(ctx, &compressor.wgate, hidden, &mut score_raw)
    })?;
    drop(nvtx_compressor_wgate);
    keepalive.keep_hidden(&score_raw);

    let rope = &config.rope_parameters;
    // Compressed keys use compress_rope_theta with NO YaRN (original_seq_len = 0).
    let (rope_dim, rope_base) = if apply_rope {
        (config.qk_rope_head_dim, config.compress_rope_theta)
    } else {
        (0, config.compress_rope_theta)
    };
    {
        let (kv_ptr, _kg) = kv_raw.data.device_ptr(&ctx.stream);
        let (score_ptr, _scg) = score_raw.data.device_ptr(&ctx.stream);
        let (ape_ptr, _ag) = compressor.ape.data.device_ptr(&ctx.stream);
        let (norm_ptr, _ng) = compressor.norm.data.device_ptr(&ctx.stream);
        let (pkv_ptr, _pkg) = state.pending_kv.device_ptr_mut(&ctx.stream);
        let (psc_ptr, _psg) = state.pending_score.device_ptr_mut(&ctx.stream);
        let (prkv_ptr, _prkg) = state.prev_overlap_kv.device_ptr_mut(&ctx.stream);
        let (prsc_ptr, _prsg) = state.prev_overlap_score.device_ptr_mut(&ctx.stream);
        let (comp_ptr, _cg) = state.compressed.data.device_ptr_mut(&ctx.stream);
        let has_prev_overlap = i32::from(compressed_base > 0);
        // SAFETY: all buffers valid on ctx.stream; state carries the pending and
        // overlap rows from previous contiguous appends.
        unsafe {
            if let Some(start_pos_device) = start_pos_device {
                let (start_ptr, _spg) = start_pos_device.device_ptr(&ctx.stream);
                ffi::dsv4_compressor_update_start_pos_ptr_cuda(
                    kv_ptr as *const ffi::Half,
                    score_ptr as *const ffi::Half,
                    ape_ptr as *const ffi::Half,
                    norm_ptr as *const ffi::Half,
                    pkv_ptr as *mut ffi::Half,
                    psc_ptr as *mut ffi::Half,
                    prkv_ptr as *mut ffi::Half,
                    prsc_ptr as *mut ffi::Half,
                    comp_ptr as *mut ffi::Half,
                    token_count as i32,
                    start_ptr as *const i32,
                    head_dim as i32,
                    ratio as i32,
                    width as i32,
                    i32::from(overlap),
                    config.rms_norm_eps,
                    rope_dim as i32,
                    rope_base,
                    0,
                    rope.factor,
                    rope.beta_fast,
                    rope.beta_slow,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            } else {
                ffi::dsv4_compressor_update_cuda(
                    kv_ptr as *const ffi::Half,
                    score_ptr as *const ffi::Half,
                    ape_ptr as *const ffi::Half,
                    norm_ptr as *const ffi::Half,
                    pkv_ptr as *mut ffi::Half,
                    psc_ptr as *mut ffi::Half,
                    prkv_ptr as *mut ffi::Half,
                    prsc_ptr as *mut ffi::Half,
                    comp_ptr as *mut ffi::Half,
                    token_count as i32,
                    start_pos_i32,
                    pending_len_i32,
                    compressed_base_i32,
                    head_dim as i32,
                    ratio as i32,
                    width as i32,
                    i32::from(overlap),
                    has_prev_overlap,
                    config.rms_norm_eps,
                    rope_dim as i32,
                    rope_base,
                    0,
                    rope.factor,
                    rope.beta_fast,
                    rope.beta_slow,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }
    }
    state.compressed.seq_len = compressed_rows;
    Ok(())
}

/// CSA top-k block selection: project the index query (`wq_b`) + per-head gating
/// (`weights_proj`), then `dsv4_csa_select_cuda` scores each compressed-key block
/// and writes the top-`index_topk` block ids per token into `[seq * index_topk]`.
#[allow(clippy::too_many_arguments)]
fn csa_select(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    indexer: &Dsv4Indexer,
    hidden: &HiddenStates,
    c_q_normed: &HiddenStates,
    keys: &HiddenStates,
    start_pos: usize,
    start_pos_device: Option<&CudaSlice<i32>>,
    ratio: usize,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<CudaSlice<i32>> {
    // SAFETY: dsv4_linear writes the full index-query buffer.
    let mut q_i = unsafe { HiddenStates::uninit(ctx, indexer.wq_b.rows, c_q_normed.seq_len)? };
    let nvtx_indexer_wq_b = crate::nvtx::range("dsv4/linear/indexer_wq_b");
    crate::linear_profile::profile(ctx, "dsv4/linear/indexer_wq_b", || {
        dsv4_linear(ctx, &indexer.wq_b, c_q_normed, &mut q_i)
    })?;
    drop(nvtx_indexer_wq_b);
    keepalive.keep_hidden(&q_i);
    // SAFETY: dsv4_linear writes the full index-weight buffer.
    let mut weights =
        unsafe { HiddenStates::uninit(ctx, indexer.weights_proj.rows, hidden.seq_len)? };
    let nvtx_indexer_weights = crate::nvtx::range("dsv4/linear/indexer_weights");
    crate::linear_profile::profile(ctx, "dsv4/linear/indexer_weights", || {
        dsv4_linear(ctx, &indexer.weights_proj, hidden, &mut weights)
    })?;
    drop(nvtx_indexer_weights);
    keepalive.keep_hidden(&weights);

    ensure!(
        q_i.hidden_dim.is_multiple_of(config.index_head_dim),
        "DSv4 indexer q width {} is not divisible by index_head_dim {}",
        q_i.hidden_dim,
        config.index_head_dim
    );
    let local_index_heads = q_i.hidden_dim / config.index_head_dim;
    ensure!(
        weights.hidden_dim == local_index_heads,
        "DSv4 indexer weights width {} != local index heads {local_index_heads}",
        weights.hidden_dim
    );

    let key_count = if start_pos_device.is_some() {
        // Graph replay must not bake the current compressed-key seq_len. The
        // selector computes `available = min(key_count, abs_pos / ratio)`, so
        // capacity preserves the same causal set while staying replay-safe.
        keys.data.len() / keys.hidden_dim
    } else {
        keys.seq_len
    };
    let mut selected = ctx
        .stream
        .alloc_zeros::<i32>(hidden.seq_len * config.index_topk)
        .map_err(|e| anyhow::anyhow!("DSv4 CSA selected alloc failed: {e}"))?;
    let score_scale =
        (config.index_head_dim as f32).powf(-0.5) * (config.index_n_heads as f32).powf(-0.5);
    {
        let (q_ptr, _qg) = q_i.data.device_ptr(&ctx.stream);
        let (w_ptr, _wg) = weights.data.device_ptr(&ctx.stream);
        let (keys_ptr, _kg) = keys.data.device_ptr(&ctx.stream);
        let (sel_ptr, _sg) = selected.device_ptr_mut(&ctx.stream);
        // SAFETY: all buffers valid on ctx.stream; selected sized seq*index_topk.
        unsafe {
            if let Some(start_pos_device) = start_pos_device {
                let (start_ptr, _spg) = start_pos_device.device_ptr(&ctx.stream);
                ffi::dsv4_csa_select_start_pos_ptr_cuda(
                    q_ptr as *const ffi::Half,
                    w_ptr as *const ffi::Half,
                    keys_ptr as *const ffi::Half,
                    sel_ptr as *mut i32,
                    hidden.seq_len as i32,
                    q_i.hidden_dim as i32,
                    local_index_heads as i32,
                    config.index_head_dim as i32,
                    key_count as i32,
                    ratio as i32,
                    config.index_topk as i32,
                    score_scale,
                    start_ptr as *const i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            } else {
                ffi::dsv4_csa_select_cuda(
                    q_ptr as *const ffi::Half,
                    w_ptr as *const ffi::Half,
                    keys_ptr as *const ffi::Half,
                    sel_ptr as *mut i32,
                    hidden.seq_len as i32,
                    q_i.hidden_dim as i32,
                    local_index_heads as i32,
                    config.index_head_dim as i32,
                    key_count as i32,
                    ratio as i32,
                    config.index_topk as i32,
                    score_scale,
                    start_pos as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }
    }
    Ok(selected)
}
