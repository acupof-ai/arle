//! Paged attention kernel-call paths for the dense-BF16 Qwen3 forward (HOT axis).
//!
//! Prep kernels fuse Q/K RMSNorm + RoPE + KV-cache write; the TileLang kernels
//! run the HD128/kv8 paged attention.

use anyhow::{Result, anyhow, bail, ensure};
use cuda_kernels::attention as flash_kv;
use cuda_kernels::ffi;
use cuda_kernels::kv_quant;
use cuda_kernels::moe as cuda_moe;
use cuda_kernels::prelude::{
    DeviceContext, DeviceMatrix, DeviceVec, HiddenStates, HiddenStatesView, PagedKVPool,
};
use cuda_kernels::tensor::{WeightFormat, cache_ptr};
use cuda_kernels::{BandPage, KVFormat, TokenKVPool};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use deepseek_spec::{DeepSeekV4AttentionMode, DeepSeekV4Config};
use infer_seam::{KvBatchDescriptor, KvBatchRowKind};
use std::sync::atomic::{AtomicI8, Ordering};

use crate::dsv4::{
    Dsv4Attention, Dsv4Compressor, Dsv4ForwardKeepalive, Dsv4Indexer, Dsv4MlaKvArena,
};
use crate::loader::PageMeta;
use crate::moe_config::ExpertSplit;
use crate::paged_kv_table::{contiguous_page_table_byte_range, physical_page};
use crate::tp::TpRuntime;

#[path = "attention/kv_layout.rs"]
mod kv_layout;
pub(crate) use kv_layout::*;

#[path = "attention/flashmla.rs"]
mod flashmla;
pub(crate) use flashmla::*;

#[path = "attention/dsa.rs"]
mod dsa;
pub(crate) use dsa::*;

#[path = "attention/prefix_state.rs"]
mod prefix_state;
pub(crate) use prefix_state::*;
const DSV4_FLASHMLA_MODEL1: i32 = 1;
/// GLM-5.2 V32 model-type int passed to the FlashMLA sparse decode shim
/// (`arle_flashmla_sm90_sparse_decode_*`, model_type=0). V32 = 576-wide latent q
/// (512 NoPE + 64 RoPE), 512-wide latent output, 656 B/tok packed KV.
const DSV4_FLASHMLA_V32: i32 = 0;
const DSV4_FLASHMLA_S_Q: usize = 1;
/// Packed bytes per token the FlashMLA sparse-FP8 decode reads for the canonical
/// MODEL1 NoPE=448 / RoPE=64 shape (validated against `kv_arena.bytes_per_token`
/// in `Dsv4FlashMlaDecodeState::new`).
const DSV4_FLASH_KV_BYTES_PER_TOKEN_I32: i32 = 584;
/// Packed bytes per token for GLM-5.2 V32 (NoPE=512 / RoPE=64), inline layout:
/// 512 (NoPE fp8) + 16 (4× F32 block scales) + 128 (bf16 rope) = 656. Matches
/// the shim `V32_BYTES_PER_TOKEN` and the vendored decode's inline reads.
const DSV4_V32_KV_BYTES_PER_TOKEN_I32: i32 = 656;

/// FlashMLA model-family dims, resolved once from the config's attention shape.
/// The single source of the `(head_dim, rope, kv_lora) → (model_type, bytes/tok,
/// d_v)` table — prefill, single-row decode, and batched decode all call this
/// instead of re-matching the tuple. `d_v` (value/output latent) is 512 for both
/// families; MODEL1's d_qk==d_v==512, V32's d_qk=576 but output latent stays 512.
struct FlashMlaModelMeta {
    model_type_int: i32,
    bytes_per_token: i32,
    d_v: i32,
}

fn dsv4_flashmla_model_meta(config: &DeepSeekV4Config) -> Result<FlashMlaModelMeta> {
    match (
        config.head_dim,
        config.qk_rope_head_dim,
        config.kv_lora_rank,
    ) {
        (512, 64, _) => Ok(FlashMlaModelMeta {
            model_type_int: DSV4_FLASHMLA_MODEL1,
            bytes_per_token: DSV4_FLASH_KV_BYTES_PER_TOKEN_I32,
            d_v: 512,
        }),
        (576, 64, 512) => Ok(FlashMlaModelMeta {
            model_type_int: DSV4_FLASHMLA_V32,
            bytes_per_token: DSV4_V32_KV_BYTES_PER_TOKEN_I32,
            d_v: 512,
        }),
        (hd, rd, kv) => anyhow::bail!(
            "DSv4 FlashMLA: unsupported (head_dim={hd}, rope={rd}, kv_lora={kv}); want MODEL1 (512,64) or V32 (576,64,kv512)"
        ),
    }
}
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

static DSV4_VERIFY_FROZEN: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Frozen-KV MTP verify: while set, `mla_attention` SKIPS `dsv4_compressor_update`
/// so the speculative K-token verify forms no new compressed blocks / DSA packs and
/// mutates nothing compressed. The executor sets it around the speculative verify and
/// clears it for the accepted-prefix commit re-forward.
pub(crate) fn set_dsv4_verify_frozen(frozen: bool) {
    DSV4_VERIFY_FROZEN.store(frozen, Ordering::Relaxed);
}

pub(crate) fn dsv4_verify_frozen() -> bool {
    DSV4_VERIFY_FROZEN.load(Ordering::Relaxed)
}

/// P2 commit fold (fast-path plan): commit the ACCEPTED verify prefix into ONE
/// layer's persistent state WITHOUT re-running the full forward. §0.1 mutated
/// buffers and their dispositions:
/// - compressor + indexer state (`pending/overlap/compressed{,seq_len}`):
///   re-ingested here by the same NON-frozen batched `compressor_forward` the
///   re-forward would have run, over the PERSISTED attn-normed rows.
/// - bf16 SW ring: re-derive `k_prepared` (wkv → kv_norm → rope at the chain
///   positions) from the persisted rows, then the same window roll the
///   prefill path uses.
/// - FP8 SW ring: strided pack of those K rows at `pos % sliding_window`
///   (table-routed, mirrors `flashmla_pack_sw_ring`).
/// - `fp8_kv_comp_packed_rows` / `dsa_official.packed_rows`: untouched —
///   the next decode's `flashmla_pack_compressed_delta` / `csa_select` bulk
///   paths self-heal off the advanced `compressed.seq_len`.
///
/// The Q-side compute is discarded (the verify already produced argmax and
/// hiddens); `q_dummy` feeds the prepare kernel's Q arm with zeros.
#[allow(clippy::too_many_arguments)]
pub(crate) fn commit_layer_fold(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    attention: &Dsv4Attention,
    mode: DeepSeekV4AttentionMode,
    compress_ratio: usize,
    state: &mut Dsv4LayerAttentionState,
    // Model-wide shared single-row FlashMLA decode scratch (#85 P3): the FP8 SW
    // ring fold reuses its `sw_bulk_*` buffers. `Some` whenever this layer has a
    // FlashMLA arena (same gate); the fold only touches it inside the FlashMLA
    // `if let Some(flash)` branch below.
    flashmla_scratch: Option<&mut Dsv4FlashMlaDecodeScratch>,
    // Model-wide shared FP32 compressor-probe scratch: the fold's compressor
    // re-ingestion runs prefill-lane (start_pos_device None), so it consumes it.
    mut fp32_scratch: Option<&mut Dsv4CompressorFp32Scratch>,
    pool: &mut Dsv4LayerKvLayout,
    gathered: &HiddenStates,
    start_pos: usize,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<()> {
    let m = gathered.seq_len;
    ensure!(m > 0, "DSv4 commit fold needs at least the pending row");
    let head_dim = config.head_dim;
    let rope = &config.rope_parameters;
    let (rope_base, original_seq_len) = if compress_ratio > 0 {
        let osl = i32::try_from(rope.original_max_position_embeddings).map_err(|_| {
            anyhow!(
                "DSv4 original_max_position_embeddings {} overflows i32",
                rope.original_max_position_embeddings
            )
        })?;
        (config.compress_rope_theta, osl)
    } else {
        (config.rope_theta, 0i32)
    };

    // ── Compressor + indexer ingestion (compressor layers only), exactly the
    // calls the re-forward's mla_attention would have made, non-frozen. This is
    // the MTP/spec commit re-forward; GLM ships no MTP (num_nextn_predict_layers
    // == 0) so SparseIndexed never reaches here — fail loud.
    ensure!(
        mode != DeepSeekV4AttentionMode::SparseIndexed,
        "DSv4 commit_layer_fold (MTP commit) does not support SparseIndexed; GLM ships no MTP \
         (num_nextn_predict_layers==0) so this path is unreachable"
    );
    if mode.has_compressor() {
        let compressor = attention.compressor.as_ref().ok_or_else(|| {
            anyhow!("DSv4 commit fold: {mode:?} layer without compressor weights")
        })?;
        let overlap = compress_ratio < 16;
        let compressor_state = state
            .compressor
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 commit fold: {mode:?} layer without compressor state"))?;
        compressor_forward(
            ctx,
            config,
            compressor,
            gathered,
            compressor_state,
            head_dim,
            compress_ratio,
            overlap,
            start_pos,
            None,
            true,
            original_seq_len,
            fp32_scratch.as_deref_mut(),
            None,
            None,
            keepalive,
        )?;
        if mode == DeepSeekV4AttentionMode::CompressedSparse {
            let indexer = attention
                .indexer
                .as_ref()
                .ok_or_else(|| anyhow!("DSv4 commit fold: CSA layer without indexer weights"))?;
            let use_official_dsa = dsv4_dsa_official_enabled()?;
            let indexer_rope_original_seq_len = if use_official_dsa {
                i32::try_from(config.rope_parameters.original_max_position_embeddings)
                    .map_err(|_| anyhow!("DSv4 commit fold indexer rope len overflows i32"))?
            } else {
                0
            };
            let indexer_state = state
                .indexer
                .as_mut()
                .ok_or_else(|| anyhow!("DSv4 commit fold: CSA layer without indexer state"))?;
            compressor_forward(
                ctx,
                config,
                indexer
                    .compressor
                    .as_ref()
                    .expect("DSv4 CSA indexer has a key compressor"),
                gathered,
                indexer_state,
                config.index_head_dim,
                compress_ratio,
                true,
                start_pos,
                None,
                use_official_dsa,
                indexer_rope_original_seq_len,
                fp32_scratch,
                None,
                None,
                keepalive,
            )?;
        }
    }

    // ── K re-derivation: wkv → kv_norm → rope at chain positions.
    // SAFETY: uninit device scratch; fully written before first read.
    let mut kv_raw = unsafe { HiddenStates::uninit(ctx, head_dim, m)? };
    dsv4_linear(ctx, &attention.wkv, gathered, &mut kv_raw)?;
    keepalive.keep_hidden(&kv_raw);
    let kv_normed = mla_rms_norm(ctx, &kv_raw, &attention.kv_norm, config.rms_norm_eps)?;
    keepalive.keep_hidden(&kv_normed);
    let local_width = attention.wq_b.rows;
    let local_heads = local_width / head_dim;
    let q_dummy = HiddenStates {
        data: ctx
            .stream
            .alloc_zeros::<half::bf16>(local_width * m)
            .map_err(|e| anyhow!("DSv4 commit fold q scratch alloc failed: {e}"))?,
        hidden_dim: local_width,
        seq_len: m,
    };
    // SAFETY: uninit device scratch; fully written before first read.
    let mut q_discard = unsafe { HiddenStates::uninit(ctx, local_width, m)? };
    // SAFETY: uninit device scratch; fully written before first read.
    let mut k_prepared = unsafe { HiddenStates::uninit(ctx, head_dim, m)? };
    {
        let (q_raw_ptr, _qr) = q_dummy.data.device_ptr(&ctx.stream);
        let (k_raw_ptr, _kr) = kv_normed.data.device_ptr(&ctx.stream);
        let (q_out_ptr, _qo) = q_discard.data.device_ptr_mut(&ctx.stream);
        let (k_out_ptr, _ko) = k_prepared.data.device_ptr_mut(&ctx.stream);
        // SAFETY: buffers sized above; q arm runs on zeros and is discarded.
        unsafe {
            ffi::dsv4_prepare_qk_cuda(
                q_raw_ptr as *const ffi::Half,
                k_raw_ptr as *const ffi::Half,
                q_out_ptr as *mut ffi::Half,
                k_out_ptr as *mut ffi::Half,
                m as i32,
                local_heads as i32,
                head_dim as i32,
                config.qk_rope_head_dim as i32,
                start_pos as i32,
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
    keepalive.keep_hidden(&q_dummy);
    keepalive.keep_hidden(&q_discard);
    keepalive.keep_hidden(&k_prepared);

    // ── bf16 SW ring roll (chain shape — identical to the prefill path).
    update_bf16_sw_window(
        ctx,
        &mut state.sw_window_cache,
        &k_prepared,
        start_pos,
        None,
        config,
    )?;

    // ── FP8 SW ring pack for the accepted positions (table-routed strided
    // pack, mirrors flashmla_pack_sw_ring's math for m explicit slots).
    // GLM pure-SparseIndexed (sliding_window==0) has no SW ring (and the
    // `% config.sliding_window` below would divide by zero) — skip the SW
    // fold entirely; the per-token sparse pack carries the latent.
    // ponytail: pod-verify GLM pure-SparseIndexed (sliding_window=0) skips SW-ring entirely; attention is indexer-selected full-latent only
    if config.sliding_window > 0
        && let Some(flash) = &mut state.flashmla
    {
        let scratch = flashmla_scratch.ok_or_else(|| {
            anyhow!("DSv4 commit fold: FlashMLA arena present but shared decode scratch missing")
        })?;
        let page_block_size = 64;
        // Stage-B: hand the kernel slot-LOGICAL pages + the device page table;
        // it resolves `block_id = table[logical]` into the dynamic pool.
        let (block_ids, rows): (Vec<i32>, Vec<i32>) = (0..m)
            .map(|i| {
                let slot_idx = (start_pos + i) % config.sliding_window;
                (
                    (slot_idx / page_block_size) as i32,
                    (slot_idx % page_block_size) as i32,
                )
            })
            .unzip();
        ctx.stream
            .memcpy_htod(&block_ids, &mut scratch.sw_bulk_block_ids)
            .map_err(|e| anyhow!("DSv4 commit fold FP8 block_ids H2D failed: {e}"))?;
        ctx.stream
            .memcpy_htod(&rows, &mut scratch.sw_bulk_rows)
            .map_err(|e| anyhow!("DSv4 commit fold FP8 rows H2D failed: {e}"))?;
        let (k_ptr, _kg) = k_prepared.data.device_ptr(&ctx.stream);
        let pool_buf = pool.flashmla_pool_data_mut()?;
        let (pool_ptr, _pg) = pool_buf.device_ptr_mut(&ctx.stream);
        let nope_ptr = k_ptr;
        let rope_ptr = nope_ptr + (config.head_dim - config.qk_rope_head_dim) as u64 * 2;
        flash_kv::dsv4_fp8_kv_pack_strided_raw(
            ctx,
            nope_ptr,
            rope_ptr,
            pool_ptr,
            &scratch.sw_bulk_block_ids,
            &scratch.sw_bulk_rows,
            m,
            page_block_size,
            config.head_dim,
            config.head_dim,
            Some(&flash.device_page_table),
        )?;
    }
    Ok(())
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

/// Re-materialize the quantized prefix rows of `layer_idx` into the shared
/// bf16 work buffers (pool→work dequant) before the prefill prep kernel
/// appends the new chunk's rows. The work buffer is shared across layers and
/// overwritten every layer/forward, and prefix pages may arrive via radix
/// attach / COW detach / tier promote — the quantized plane is the only
/// durable source, so the prefix is unconditionally re-materialized. K uses
/// the per-channel KIVI dequant (per-channel K quantize never writes
/// per-token K scales); V uses the per-token sibling.
fn refill_prefix_rows_to_work(
    ctx: &DeviceContext,
    layer_idx: usize,
    pool: &PagedKVPool,
    meta: &PageMeta,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    if meta.start_pos == 0 {
        return Ok(());
    }
    let stream = &ctx.stream;
    let prefix_rows = meta.prefix_token_rows.as_ref().ok_or_else(|| {
        anyhow!(
            "quant KV prefill missing prefix_token_rows for start_pos={}",
            meta.start_pos
        )
    })?;
    let k_static_scales_ptr = pool
        .k_static_scales_ptr(layer_idx, stream)
        .ok_or_else(|| anyhow!("quant KV pool missing KIVI k_static_scales (layer {layer_idx})"))?;
    match pool.format {
        KVFormat::FP8E4M3 => {
            kv_quant::dequantize_paged_kv_fp8_per_channel_k_to_hnd(
                ctx,
                pool.k_data_ptr(layer_idx, stream),
                k_static_scales_ptr,
                pool.k_work_ptr(stream),
                prefix_rows,
                num_kv_heads,
                head_dim,
                pool.kv_dim,
                meta.start_pos,
            )?;
            kv_quant::dequantize_paged_kv_fp8_to_hnd(
                ctx,
                pool.v_data_ptr(layer_idx, stream),
                pool.v_scales_ptr(layer_idx, stream),
                pool.v_work_ptr(stream),
                prefix_rows,
                num_kv_heads,
                head_dim,
                pool.kv_dim,
                meta.start_pos,
            )
        }
        KVFormat::INT8 => {
            kv_quant::dequantize_paged_kv_int8_per_channel_k_to_hnd(
                ctx,
                pool.k_data_ptr(layer_idx, stream),
                k_static_scales_ptr,
                pool.k_work_ptr(stream),
                prefix_rows,
                num_kv_heads,
                head_dim,
                pool.kv_dim,
                meta.start_pos,
            )?;
            kv_quant::dequantize_paged_kv_int8_to_hnd(
                ctx,
                pool.v_data_ptr(layer_idx, stream),
                pool.v_scales_ptr(layer_idx, stream),
                pool.v_work_ptr(stream),
                prefix_rows,
                num_kv_heads,
                head_dim,
                pool.kv_dim,
                meta.start_pos,
            )
        }
        other => bail!("quant KV prefix refill does not support format {other:?}"),
    }
}

/// Quantize this forward's new bf16 rows work→pool for `layer_idx`,
/// calibrating the KIVI per-channel K scale table first if the layer's latch
/// is still unset. K goes through the per-channel quantize against the static
/// table; V keeps per-(token, head) scales.
fn calibrate_and_quantize_new_rows(
    ctx: &DeviceContext,
    layer_idx: usize,
    pool: &PagedKVPool,
    meta: &PageMeta,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let stream = &ctx.stream;
    let new_rows = meta
        .new_token_rows
        .as_ref()
        .ok_or_else(|| anyhow!("quant KV forward missing new_token_rows"))?;
    let k_static_scales_ptr = pool
        .k_static_scales_ptr(layer_idx, stream)
        .ok_or_else(|| anyhow!("quant KV pool missing KIVI k_static_scales (layer {layer_idx})"))?;
    let batch = meta.seq_len;
    // Latch-once calibration is REQUIRED under chunked prefill: recalibrating
    // on a later chunk would rescale the table while earlier chunks' K bytes
    // remain quantized under the old scale, corrupting every prior row at
    // decode. First batch through the layer calibrates (absmax → finalize),
    // then the latch flips and the table is read-only.
    if !pool.k_kivi_calibrated[layer_idx].load(Ordering::Relaxed) {
        kv_quant::compute_k_per_channel_absmax(
            ctx,
            pool.k_work_ptr(stream),
            k_static_scales_ptr,
            new_rows,
            num_kv_heads,
            head_dim,
            pool.kv_dim,
            batch,
        )?;
        match pool.format {
            KVFormat::FP8E4M3 => kv_quant::finalize_k_per_channel_scales(
                ctx,
                k_static_scales_ptr,
                num_kv_heads * head_dim,
            )?,
            KVFormat::INT8 => kv_quant::finalize_k_per_channel_scales_int8(
                ctx,
                k_static_scales_ptr,
                num_kv_heads * head_dim,
            )?,
            other => bail!("quant KV calibration does not support format {other:?}"),
        }
        pool.k_kivi_calibrated[layer_idx].store(true, Ordering::Relaxed);
    }
    match pool.format {
        KVFormat::FP8E4M3 => {
            kv_quant::quantize_paged_kv_fp8_per_channel(
                ctx,
                pool.k_work_ptr(stream),
                pool.k_data_ptr(layer_idx, stream),
                k_static_scales_ptr,
                new_rows,
                num_kv_heads,
                head_dim,
                pool.kv_dim,
                batch,
            )?;
            kv_quant::quantize_paged_kv_per_token(
                ctx,
                pool.v_work_ptr(stream),
                pool.v_data_ptr(layer_idx, stream),
                pool.v_scales_ptr(layer_idx, stream),
                new_rows,
                num_kv_heads,
                head_dim,
                pool.kv_dim,
                batch,
                KVFormat::FP8E4M3,
            )?;
        }
        KVFormat::INT8 => {
            kv_quant::quantize_paged_kv_int8_per_channel(
                ctx,
                pool.k_work_ptr(stream),
                pool.k_data_ptr(layer_idx, stream),
                k_static_scales_ptr,
                new_rows,
                num_kv_heads,
                head_dim,
                pool.kv_dim,
                batch,
            )?;
            kv_quant::quantize_paged_kv_per_token(
                ctx,
                pool.v_work_ptr(stream),
                pool.v_data_ptr(layer_idx, stream),
                pool.v_scales_ptr(layer_idx, stream),
                new_rows,
                num_kv_heads,
                head_dim,
                pool.kv_dim,
                batch,
                KVFormat::INT8,
            )?;
        }
        other => bail!("quant KV new-row quantize does not support format {other:?}"),
    }
    Ok(())
}

/// Fused-dequant decode attention over the quantized pool planes (replaces
/// the TileLang bf16 kernel for INT8/FP8 pools). NOT
/// `decode_attention_varlen_quantized` — that kernel consumes per-token K scales
/// and is incompatible with per-channel K.
#[allow(clippy::too_many_arguments)]
fn run_quant_decode(
    ctx: &DeviceContext,
    layer_idx: usize,
    pool: &PagedKVPool,
    q_batch: &HiddenStates,
    meta: &PageMeta,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    let stream = &ctx.stream;
    let quant_meta = meta
        .quant_decode_meta
        .as_ref()
        .ok_or_else(|| anyhow!("quant KV decode missing quant_decode_meta"))?;
    let k_static_scales_ptr = pool
        .k_static_scales_ptr(layer_idx, stream)
        .ok_or_else(|| anyhow!("quant KV pool missing KIVI k_static_scales (layer {layer_idx})"))?;
    let ws = pool.quantized_attn_workspace()?;
    // The kernel adapts its split count to the workspace it is given
    // (`choose_decode_num_splits` clamps by workspace_bytes, ≥1 split), so the
    // only unfittable case is a single split not fitting. The pool sizes the
    // workspace from an approximate-max-heads heuristic (paged_kv.rs) that can
    // undershoot the full 32-split footprint for q40/q64 dense shapes at small
    // num_slots — gating on 32 splits here would falsely reject those.
    let needed = kv_quant::decode_attention_int8_workspace_bytes(1, num_q_heads, head_dim, 1);
    ensure!(
        needed <= pool.quantized_attn_workspace_bytes,
        "quant decode workspace cannot fit a single split: needs {needed} bytes, pool allocated {}",
        pool.quantized_attn_workspace_bytes
    );
    let sm_scale = 1.0 / (head_dim as f32).sqrt();
    match pool.format {
        KVFormat::FP8E4M3 => kv_quant::decode_attention_fp8_per_channel_k(
            ctx,
            q_batch,
            pool.k_data_ptr(layer_idx, stream),
            pool.v_data_ptr(layer_idx, stream),
            k_static_scales_ptr,
            pool.v_scales_ptr(layer_idx, stream),
            &meta.kv_indices,
            quant_meta,
            out,
            1,
            num_q_heads,
            num_kv_heads,
            head_dim,
            pool.kv_dim,
            sm_scale,
            ws,
            pool.quantized_attn_workspace_bytes,
        ),
        KVFormat::INT8 => kv_quant::decode_attention_int8_per_channel_k(
            ctx,
            q_batch,
            pool.k_data_ptr(layer_idx, stream),
            pool.v_data_ptr(layer_idx, stream),
            k_static_scales_ptr,
            pool.v_scales_ptr(layer_idx, stream),
            &meta.kv_indices,
            quant_meta,
            out,
            1,
            num_q_heads,
            num_kv_heads,
            head_dim,
            pool.kv_dim,
            sm_scale,
            ws,
            pool.quantized_attn_workspace_bytes,
        ),
        other => bail!("quant KV decode does not support format {other:?}"),
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
    let quant = matches!(pool.format, KVFormat::INT8 | KVFormat::FP8E4M3);
    if !quant && pool.format != KVFormat::BF16 {
        // Defensive: CudaKvCacheDtype::resolve admits only BF16/INT8/FP8.
        bail!(
            "dense-Qwen3 paged prefill supports BF16/INT8/FP8E4M3 KV pools, got {:?}",
            pool.format
        );
    }
    if quant {
        // Prefix rows must be back in the bf16 work buffer before the prep
        // kernel appends this chunk's rows (TileLang reads the whole [0,
        // start_pos + seq_len) span from the work buffer via pool.k_ptr).
        refill_prefix_rows_to_work(ctx, layer_idx, pool, meta, num_kv_heads, head_dim)?;
    }
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

        // SAFETY: ptrs from live device allocations sized to the dims passed.
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
    )?;
    if quant {
        // FINALIZE after TileLang has consumed the bf16 work buffer: calibrate
        // (latch-once) and persist this chunk's new rows into the quant planes.
        calibrate_and_quantize_new_rows(ctx, layer_idx, pool, meta, num_kv_heads, head_dim)?;
    }
    Ok(())
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
    let quant = matches!(pool.format, KVFormat::INT8 | KVFormat::FP8E4M3);
    if !quant && pool.format != KVFormat::BF16 {
        // Defensive: CudaKvCacheDtype::resolve admits only BF16/INT8/FP8.
        bail!(
            "dense-Qwen3 paged decode supports BF16/INT8/FP8E4M3 KV pools, got {:?}",
            pool.format
        );
    }
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

        // SAFETY: ptrs from live device allocations sized to the dims passed.
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
                meta.batch as i32,
                rms_eps,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }
    if quant {
        // Calibrate-if-unlatched covers the 1-token-prompt edge — a seq_len==1
        // first forward routes here with start_pos==0 and zero-init static
        // scales; quantizing K against a zero table would write garbage for
        // the whole request. Then quantize this step's row and run the fused
        // dequant decode kernel (graph is hard-disabled for quant pools).
        calibrate_and_quantize_new_rows(ctx, layer_idx, pool, meta, num_kv_heads, head_dim)?;
        return run_quant_decode(
            ctx,
            layer_idx,
            pool,
            q_batch,
            meta,
            num_q_heads,
            num_kv_heads,
            head_dim,
            out,
        );
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
    // Generated dispatch table (kernels.toml -> resolve_paged_attn_v1). The
    // ensure!() above pins head_dim==128 / kv==8; only the same 8 configs
    // (prefill/decode x q16/32/40/64) resolve. The (1,1,1) vs (1,seq,seq)
    // scalar choice reproduces the old per-arm literals exactly.
    let phase = if decode {
        ffi::AttnPhase::Decode
    } else {
        ffi::AttnPhase::Prefill
    };
    let kernel = ffi::resolve_paged_attn_v1(
        head_dim as u32,
        num_q_heads as u32,
        num_kv_heads as u32,
        phase,
    )
    .ok_or_else(|| {
        anyhow::anyhow!("unsupported HD128 q/kv head config q{num_q_heads}_kv{num_kv_heads}")
    })?;
    // Decode: one Q row per request → bsz = batch, total_q = batch, max_q = 1.
    // B=1 evaluates to (1,1,1), byte-identical to the prior literal.
    let (bsz, total_q, max_q) = if decode {
        (meta.batch as i32, meta.batch as i32, 1)
    } else {
        (1, meta.seq_len as i32, meta.seq_len as i32)
    };
    // SAFETY: ptrs from live device allocations sized to the dims passed.
    unsafe {
        kernel(
            q_ptr as *mut ffi::Half,
            qo_ptr as *const i32,
            k_pool_ptr as *mut ffi::Half,
            v_pool_ptr as *mut ffi::Half,
            kv_indptr_ptr as *const i32,
            kv_indices_ptr as *const i32,
            last_ptr as *const i32,
            out_ptr as *mut ffi::Half,
            bsz,
            total_q,
            max_q,
            pool.max_total_pages as i32,
            meta.num_pages as i32,
            num_q_heads as i32,
            num_kv_heads as i32,
            pool.page_size as i32,
            sm_scale,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

// DSv4-Flash MLA attention core
//
// The MLA attention is a genuinely new subsystem next to the dense-BF16 paged
// path above (it is NOT a GEMM swap): a low-rank Q/KV projection (`wq_a → q_norm
// → wq_b` for Q; `wkv → kv_norm` for the single compressed KV latent), partial
// RoPE on the trailing `rope_dim` columns, a windowed attention with a per-head
// sink logit + (on CSA/HCA layers) a compressed-key stream, and a low-rank O
// projection (`wo_a → wo_b`).
//
// All modes run through FlashMLA sparse attention (the shared per-layer FP8 KV
// pool):
//   - SlidingWindow (`compress_ratio == 0`): SW window only.
//   - CompressedSparse (`0 < ratio < 16`): compressor + indexer top-k select,
//     then SW window + selected compressed blocks.
//   - HybridCompressed (`ratio >= 16`): compressor + SW window + ALL compressed
//     blocks (no selector).
//   - SparseIndexed (GLM DSA): indexer top-k over the full latent (no
//     compressor), then SW window + selected full-latent blocks.
//
// Shared kernels: `dsv4_{fp8,fp4}_gemv_batch_cuda` / `gemm_cuda` (LoRA matmuls),
// `dsv4_prepare_qk_cuda`, `dsv4_compressor_update_cuda`, official DSA select,
// FlashMLA sparse prefill/decode.

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

/// Decode (M=1) FP8 projection through tensor-core DeepGEMM: quantize `input`
/// (K columns) into the fused-wqkv FP8 scratch, then `dsv4_deepgemm_fp8_gemm_nt`
/// with the pre-repacked weight `cache`. Used for the residual decode projections
/// (wo_a/wo_b; lever #1b) when K ≤ the scratch hidden_dim. The scratch may have
/// been consumed by an earlier projection this step — safe, all on `ctx.stream`.
fn decode_proj_deepgemm_raw<I, O>(
    ctx: &DeviceContext,
    scratch: &mut Dsv4FusedWqkvDecodeScratch,
    cache: &cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache,
    input: &I,
    input_len: usize,
    out: &mut O,
    out_len: usize,
    k: usize,
    m: usize,
) -> Result<()>
where
    I: DevicePtr<half::bf16>,
    O: DevicePtrMut<half::bf16>,
{
    // M (token rows) drives both the pack-quantize active_counts[0] and the
    // GEMM's m arg. Batched O-LoRA passes m=n (token-major [n, k] input); the
    // per-row decode lanes pass m=1. N=1 BYTE-IDENTITY: the scratch's
    // active_counts is constructed [1] and every per-row decode call restores it
    // to [1], so an m=1 call SKIPS the H2D write below and emits exactly the old
    // (m=1, active_counts=[1]) kernel args. The batched caller is responsible
    // for restoring active_counts to [1] after its m=n call.
    ensure!(
        m > 0 && m <= scratch.max_m,
        "DSv4 decode_proj_deepgemm_raw M={m} out of range (1..={})",
        scratch.max_m
    );
    ensure!(
        cache.cols == k && input_len >= m * k && out_len >= m * cache.rows,
        "DSv4 decode_proj_deepgemm_raw shape mismatch: cache {}x{} k={k} m={m} input_len={} out_len={}",
        cache.rows,
        cache.cols,
        input_len,
        out_len
    );
    if m != 1 {
        // Token-row count for the pack-quantize row loop and the GEMM. The
        // shared scratch's active_counts is restored to [1] by the caller.
        let m_i32 =
            i32::try_from(m).map_err(|_| anyhow!("DSv4 decode proj M={m} overflows i32"))?;
        ctx.stream
            .memcpy_htod(&[m_i32], &mut scratch.active_counts)
            .map_err(|e| anyhow!("DSv4 decode proj active_counts H2D failed: {e}"))?;
    }
    let scratch = &*scratch;
    let stream = ctx.stream.cu_stream();
    let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
    let (fp8_ptr, _fp8_guard) = scratch.input_fp8.device_ptr(&ctx.stream);
    let (scale_ptr, _scale_guard) = scratch.input_scales.device_ptr(&ctx.stream);
    let (active_experts_ptr, _active_experts_guard) =
        scratch.active_experts.device_ptr(&ctx.stream);
    let (active_offsets_ptr, _active_offsets_guard) =
        scratch.active_offsets.device_ptr(&ctx.stream);
    let (active_counts_ptr, _active_counts_guard) = scratch.active_counts.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = cache.weight.device_ptr(&ctx.stream);
    let (weight_scale_ptr, _weight_scale_guard) = cache.scales.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    // SAFETY: ptrs from live device allocations sized to the dims passed.
    unsafe {
        ffi::dsv4_deepgemm_pack_quantize_bf16_to_fp8_cuda(
            input_ptr as *const ffi::Half,
            fp8_ptr as *mut u8,
            scale_ptr as *mut f32,
            active_experts_ptr as *const i32,
            active_offsets_ptr as *const i32,
            active_counts_ptr as *const i32,
            1,
            i32::try_from(scratch.max_m)?,
            i32::try_from(k)?,
            i32::try_from(scratch.scale_stride_m)?,
            stream,
        )
        .result()
        .map_err(|e| anyhow!("DSv4 decode proj activation quantize failed: {e}"))?;
        ffi::dsv4_deepgemm_fp8_gemm_nt_cuda(
            fp8_ptr as *const u8,
            scale_ptr as *const f32,
            weight_ptr as *const u8,
            weight_scale_ptr as *const f32,
            out_ptr as *mut ffi::Half,
            i32::try_from(m)?,
            i32::try_from(cache.rows)?,
            i32::try_from(cache.cols)?,
            i32::try_from(scratch.scale_stride_m)?,
            stream,
        )
        .result()
        .map_err(|e| anyhow!("DSv4 decode proj DeepGEMM dense failed: {e}"))?;
    }
    Ok(())
}

fn decode_proj_deepgemm(
    ctx: &DeviceContext,
    scratch: &mut Dsv4FusedWqkvDecodeScratch,
    cache: &cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache,
    input: &HiddenStates,
    out: &mut HiddenStates,
    k: usize,
) -> Result<()> {
    // M = token rows (input is token-major [m, k]). m==1 is the per-row decode
    // lane (byte-identical to the original assert seq_len==1); m==n is the
    // batched O-LoRA. The shape contract is the same in both.
    let m = input.seq_len;
    ensure!(
        cache.cols == k
            && cache.rows == out.hidden_dim
            && input.hidden_dim == k
            && out.seq_len == m,
        "DSv4 decode_proj_deepgemm shape mismatch: cache {}x{} k={k} m={m} in {}x{} out {}x{}",
        cache.rows,
        cache.cols,
        input.hidden_dim,
        input.seq_len,
        out.hidden_dim,
        out.seq_len
    );
    let in_len = input.data.len();
    let out_len = out.data.len();
    decode_proj_deepgemm_raw(
        ctx,
        scratch,
        cache,
        &input.data,
        in_len,
        &mut out.data,
        out_len,
        k,
        m,
    )
}

/// Prefill (M=token_count) residual projection via DeepGEMM: quantize `input`
/// [m, k] into the prefill FP8 scratch, then `dsv4_deepgemm_fp8_gemm_nt` with the
/// pre-repacked weight `cache`. The M>1 analogue of [`decode_proj_deepgemm`] —
/// moves the prefill wq_b / wo / indexer projections off the scalar
/// `dsv4_fp8_gemv_batch` (62% of mla_attn prefill) onto tensor-core DeepGEMM.
/// K ≤ scratch.max_k (the fused-wqkv scratch is sized for the largest K=hidden_dim).
fn prefill_proj_deepgemm(
    ctx: &DeviceContext,
    scratch: &mut Dsv4PrefillDeepGemmLinearScratch,
    cache: &cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache,
    input: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    let m = input.seq_len;
    let k = cache.cols;
    let n = cache.rows;
    ensure!(
        input.hidden_dim == k && out.hidden_dim == n && out.seq_len == m,
        "DSv4 prefill_proj_deepgemm shape mismatch: cache {n}x{k} in {}x{} out {}x{}",
        input.hidden_dim,
        input.seq_len,
        out.hidden_dim,
        out.seq_len
    );
    // M (query/token) dim is chunk-bounded: scratch.max_m = DSV4_PREFILL_QUERY_CHUNK
    // (>= chunked_prefill_size). Chunked prefill guarantees seq_len <=
    // chunked_prefill_size <= DSV4_PREFILL_QUERY_CHUNK, so this assert only trips on
    // a misconfigured chunk size or the one-shot dsv4_parity long-context example —
    // fail loud rather than write past the chunk-sized M×K scratch.
    ensure!(
        m <= scratch.max_m && k <= scratch.max_k,
        "DSv4 prefill_proj_deepgemm M={m} > query chunk {} (or K={k} > {}): chunked \
         prefill must keep seq_len <= chunked_prefill_size <= DSV4_PREFILL_QUERY_CHUNK",
        scratch.max_m,
        scratch.max_k
    );
    let scale_stride_m = m.div_ceil(4) * 4;
    let scale_cols = k.div_ceil(128);
    ensure!(
        scale_stride_m <= scratch.max_scale_stride_m
            && scale_stride_m * scale_cols <= scratch.input_scales.len()
            && m * k <= scratch.input_fp8.len(),
        "DSv4 prefill_proj_deepgemm scratch extent mismatch: M={m} K={k} stride={scale_stride_m}"
    );
    let active_count = i32::try_from(m)
        .map_err(|_| anyhow!("DSv4 prefill_proj_deepgemm token count {m} overflows i32"))?;
    ctx.stream
        .memcpy_htod(&[active_count], &mut scratch.active_counts)
        .map_err(|e| anyhow!("DSv4 prefill_proj_deepgemm active_counts H2D failed: {e}"))?;
    let stream = ctx.stream.cu_stream();
    // SAFETY: all buffers on ctx.stream; M/K within scratch extents (checked above).
    unsafe {
        cuda_moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
            cache_ptr(&input.data, ctx),
            cache_ptr(&scratch.input_fp8, ctx),
            cache_ptr(&scratch.input_scales, ctx),
            cache_ptr(&scratch.active_experts, ctx),
            cache_ptr(&scratch.active_offsets, ctx),
            cache_ptr(&scratch.active_counts, ctx),
            1,
            m,
            k,
            scale_stride_m,
            stream,
        )
        .map_err(|e| anyhow!("DSv4 prefill_proj_deepgemm activation quantize failed: {e}"))?;
        cuda_moe::dsv4_deepgemm_fp8_gemm_nt(
            cache_ptr(&scratch.input_fp8, ctx),
            cache_ptr(&scratch.input_scales, ctx),
            cache_ptr(&cache.weight, ctx),
            cache_ptr(&cache.scales, ctx),
            cache_ptr(&out.data, ctx),
            m,
            n,
            k,
            scale_stride_m,
            stream,
        )
        .map_err(|e| anyhow!("DSv4 prefill_proj_deepgemm DeepGEMM dense failed: {e}"))?;
    }
    Ok(())
}

fn prefill_proj_deepgemm_group_scratch(
    ctx: &DeviceContext,
    scratch: &mut Dsv4PrefillDeepGemmLinearScratch,
    cache: &cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache,
    m: usize,
) -> Result<()> {
    let k = cache.cols;
    let n = cache.rows;
    ensure!(
        m > 0
            && m <= scratch.max_m
            && k <= scratch.oproj_group_cols
            && n <= scratch.oproj_group_rows,
        "DSv4 grouped wo_a DeepGEMM scratch mismatch: M={m} cache={n}x{k} scratch M={} in_cols={} out_rows={}",
        scratch.max_m,
        scratch.oproj_group_cols,
        scratch.oproj_group_rows
    );
    let scale_stride_m = m.div_ceil(4) * 4;
    let scale_cols = k.div_ceil(128);
    ensure!(
        scale_stride_m <= scratch.max_scale_stride_m
            && scale_stride_m * scale_cols <= scratch.input_scales.len()
            && m * k <= scratch.oproj_group_in.len()
            && m * n <= scratch.oproj_group_out.len(),
        "DSv4 grouped wo_a DeepGEMM scratch extent mismatch: M={m} K={k} N={n} stride={scale_stride_m}"
    );
    let active_count =
        i32::try_from(m).map_err(|_| anyhow!("DSv4 grouped wo_a token count {m} overflows i32"))?;
    ctx.stream
        .memcpy_htod(&[active_count], &mut scratch.active_counts)
        .map_err(|e| anyhow!("DSv4 grouped wo_a active_counts H2D failed: {e}"))?;
    let stream = ctx.stream.cu_stream();
    let (input_ptr, _input_guard) = scratch.oproj_group_in.device_ptr(&ctx.stream);
    let (fp8_ptr, _fp8_guard) = scratch.input_fp8.device_ptr(&ctx.stream);
    let (scale_ptr, _scale_guard) = scratch.input_scales.device_ptr(&ctx.stream);
    let (active_experts_ptr, _active_experts_guard) =
        scratch.active_experts.device_ptr(&ctx.stream);
    let (active_offsets_ptr, _active_offsets_guard) =
        scratch.active_offsets.device_ptr(&ctx.stream);
    let (active_counts_ptr, _active_counts_guard) = scratch.active_counts.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = cache.weight.device_ptr(&ctx.stream);
    let (weight_scale_ptr, _weight_scale_guard) = cache.scales.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = scratch.oproj_group_out.device_ptr_mut(&ctx.stream);
    // SAFETY: ptrs from live device allocations sized to the dims passed.
    unsafe {
        ffi::dsv4_deepgemm_pack_quantize_bf16_to_fp8_cuda(
            input_ptr as *const ffi::Half,
            fp8_ptr as *mut u8,
            scale_ptr as *mut f32,
            active_experts_ptr as *const i32,
            active_offsets_ptr as *const i32,
            active_counts_ptr as *const i32,
            1,
            i32::try_from(m)?,
            i32::try_from(k)?,
            i32::try_from(scale_stride_m)?,
            stream,
        )
        .result()
        .map_err(|e| anyhow!("DSv4 grouped wo_a activation quantize failed: {e}"))?;
        ffi::dsv4_deepgemm_fp8_gemm_nt_cuda(
            fp8_ptr as *const u8,
            scale_ptr as *const f32,
            weight_ptr as *const u8,
            weight_scale_ptr as *const f32,
            out_ptr as *mut ffi::Half,
            i32::try_from(m)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(scale_stride_m)?,
            stream,
        )
        .result()
        .map_err(|e| anyhow!("DSv4 grouped wo_a DeepGEMM dense failed: {e}"))?;
    }
    Ok(())
}

fn run_fused_wqkv_prefill(
    ctx: &DeviceContext,
    attention: &Dsv4Attention,
    hidden: &HiddenStates,
    scratch: &mut Dsv4PrefillDeepGemmLinearScratch,
    c_q: &mut HiddenStates,
    kv_raw: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        hidden.seq_len > 1,
        "DSv4 fused wqkv prefill path requires seq_len>1, got {}",
        hidden.seq_len
    );
    ensure!(
        hidden.hidden_dim == scratch.hidden_dim,
        "DSv4 fused wqkv prefill hidden dim mismatch: hidden={} scratch={}",
        hidden.hidden_dim,
        scratch.hidden_dim
    );
    ensure!(
        c_q.hidden_dim == scratch.q_lora_rank
            && kv_raw.hidden_dim == scratch.head_dim
            && c_q.seq_len == hidden.seq_len
            && kv_raw.seq_len == hidden.seq_len,
        "DSv4 fused wqkv prefill output shape mismatch: c_q={}x{} kv={}x{} scratch q={} kv={} tokens={}",
        c_q.hidden_dim,
        c_q.seq_len,
        kv_raw.hidden_dim,
        kv_raw.seq_len,
        scratch.q_lora_rank,
        scratch.head_dim,
        hidden.seq_len
    );
    let cache = attention.wqkv_a_deepgemm.as_ref().ok_or_else(|| {
        anyhow!("DSv4 fused wqkv prefill requested but fused cache was not loaded")
    })?;
    ensure!(
        cache.rows == scratch.q_lora_rank + scratch.head_dim && cache.cols == scratch.hidden_dim,
        "DSv4 fused wqkv prefill cache shape {}x{} != expected {}x{}",
        cache.rows,
        cache.cols,
        scratch.q_lora_rank + scratch.head_dim,
        scratch.hidden_dim
    );
    let m = hidden.seq_len;
    let n = cache.rows;
    let k = cache.cols;
    // M (query/token) dim is chunk-bounded: scratch.max_m = DSV4_PREFILL_QUERY_CHUNK
    // (>= chunked_prefill_size). Chunked prefill guarantees seq_len <=
    // chunked_prefill_size <= DSV4_PREFILL_QUERY_CHUNK, so this assert only trips on
    // a misconfigured chunk size or the one-shot dsv4_parity long-context example —
    // fail loud rather than write past the chunk-sized M×K activation scratch.
    ensure!(
        m <= scratch.max_m && k <= scratch.max_k,
        "DSv4 fused wqkv prefill M={} > query chunk {} (or K={} > {}): chunked prefill \
         must keep seq_len <= chunked_prefill_size <= DSV4_PREFILL_QUERY_CHUNK",
        m,
        scratch.max_m,
        k,
        scratch.max_k
    );
    let scale_stride_m = m.div_ceil(4) * 4;
    let scale_cols = k.div_ceil(128);
    ensure!(
        scale_stride_m <= scratch.max_scale_stride_m
            && scale_stride_m * scale_cols <= scratch.input_scales.len()
            && m * k <= scratch.input_fp8.len(),
        "DSv4 fused wqkv prefill scratch extent mismatch: M={} K={} scale_stride={} scales={} fp8={}",
        m,
        k,
        scale_stride_m,
        scratch.input_scales.len(),
        scratch.input_fp8.len()
    );
    let active_count = i32::try_from(m)
        .map_err(|_| anyhow!("DSv4 fused wqkv prefill token count {m} overflows i32"))?;
    ctx.stream
        .memcpy_htod(&[active_count], &mut scratch.active_counts)
        .map_err(|e| anyhow!("DSv4 fused wqkv prefill active_counts H2D failed: {e}"))?;
    let stream = ctx.stream.cu_stream();
    // SAFETY: ptrs from live device allocations sized to the dims passed.
    unsafe {
        cuda_moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
            cache_ptr(&hidden.data, ctx),
            cache_ptr(&scratch.input_fp8, ctx),
            cache_ptr(&scratch.input_scales, ctx),
            cache_ptr(&scratch.active_experts, ctx),
            cache_ptr(&scratch.active_offsets, ctx),
            cache_ptr(&scratch.active_counts, ctx),
            1,
            m,
            k,
            scale_stride_m,
            stream,
        )
        .map_err(|e| anyhow!("DSv4 fused wqkv prefill activation quantize failed: {e}"))?;
        cuda_moe::dsv4_deepgemm_fp8_gemm_nt(
            cache_ptr(&scratch.input_fp8, ctx),
            cache_ptr(&scratch.input_scales, ctx),
            cache_ptr(&cache.weight, ctx),
            cache_ptr(&cache.scales, ctx),
            cache_ptr(&scratch.qkv_raw.data, ctx),
            m,
            n,
            k,
            scale_stride_m,
            stream,
        )
        .map_err(|e| anyhow!("DSv4 fused wqkv prefill DeepGEMM dense failed: {e}"))?;
        let (qkv_ptr, _qkv_guard) = scratch.qkv_raw.data.device_ptr(&ctx.stream);
        let (cq_ptr, _cq_guard) = c_q.data.device_ptr_mut(&ctx.stream);
        ffi::dsv4_tp_out_slice_cuda(
            qkv_ptr as *const ffi::Half,
            cq_ptr as *mut ffi::Half,
            m as i32,
            n as i32,
            scratch.q_lora_rank as i32,
            0,
            stream,
        )
        .result()
        .map_err(|e| anyhow!("DSv4 fused wqkv prefill c_q slice failed: {e}"))?;
        let (kv_ptr, _kv_guard) = kv_raw.data.device_ptr_mut(&ctx.stream);
        ffi::dsv4_tp_out_slice_cuda(
            qkv_ptr as *const ffi::Half,
            kv_ptr as *mut ffi::Half,
            m as i32,
            n as i32,
            scratch.head_dim as i32,
            scratch.q_lora_rank as i32,
            stream,
        )
        .result()
        .map_err(|e| anyhow!("DSv4 fused wqkv prefill kv slice failed: {e}"))?;
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
    // Runtime A/B gate:
    // `--dsv4-flashmla-decode false` forces FlashMLA decode off, `true` forces on,
    // unset keeps the default below.
    // Default ON: FlashMLA SM90 sparse decode is the adopted decode attention.
    // Compile-gated via `cuda_kernels::HAS_FLASHMLA`; a build without FlashMLA
    // reports false and decode is unavailable.
    Ok(cuda_kernels::HAS_FLASHMLA)
}

/// Batched (`b = N`) FlashMLA sparse decode lane gate (#60). Canonical for
/// MODEL1 n>1 decode when FlashMLA decode is available; N=1 forwards never
/// consult this and always take the cached-meta single-row path.
pub(crate) fn dsv4_flashmla_decode_batched_enabled() -> Result<bool> {
    dsv4_flashmla_decode_enabled()
}

pub(crate) fn dsv4_flashmla_prefill_enabled() -> Result<bool> {
    // Default ON: vendored FlashMLA sparse prefill replaces the scalar
    // SW/CSA/HCA attention math. Licensed 2026-06-07 on the TP=8/EP=8 H20 pod:
    // 4096-token warm prefill 7189 -> 4299 ms, and the 2048-token edge case is
    // within the legacy same-config floor on both synthetic and real-prose prompts.
    // Compile-gated via `cuda_kernels::HAS_FLASHMLA`; scalar fallback when absent.
    Ok(cuda_kernels::HAS_FLASHMLA)
}

fn dsv4_q8kv8_prefill_enabled() -> Result<bool> {
    // Experimental: fp8 q8kv8 sparse prefill (FlashMLA prefill's fp8 twin).
    // Default OFF; A/B only. Runs on the post-gather global-head Q, so it works
    // under TP (gate is global_heads%64, the kernel's GMMA head-tile).
    Ok(std::env::var("ARLE_DSV4_Q8KV8_PREFILL").as_deref() == Ok("1"))
}

fn dsv4_fp8_linear_deepgemm_enabled() -> Result<bool> {
    // Default ON: prefill wq_a|wkv projection fusion routes the shared hidden
    // activation through FP8 DeepGEMM instead of the scalar FP8 GEMV path. Licensed
    // 2026-06-07 by the six-shape within-floor gate. Runtime preflight probe
    // (`cuda_kernels::has_deepgemm_native()`, cached); scalar FP8-GEMV fallback
    // when native DeepGEMM is not compiled in.
    Ok(cuda_kernels::has_deepgemm_native())
}

fn dsv4_decode_proj_deepgemm_enabled() -> bool {
    // Lever #1 (nsys decode breakdown): route the residual decode projection GEMVs
    // (wq_b now; wo_a/wo_b next) through tensor-core DeepGEMM instead of the scalar
    // `dsv4_fp8_gemv_batch` (3.62ms, #1 decode GPU kernel). Default ON: licensed
    // 2026-06-07 on the TP=8/EP=8 pod, same-load A/B 38.2 -> 39.2 tok/s (+2.5%,
    // reproduced ×2) with the 37-tok needle retrieved bit-identically (divergence
    // only in the free-continuation tail = legitimate FP8 numerics).
    cuda_kernels::has_deepgemm_native()
}

/// Prefill residual projections (wq_b now; wo/indexer next) → tensor-core DeepGEMM
/// instead of the scalar `dsv4_fp8_gemv_batch` (62% of mla_attn prefill per the P/D
/// nsys breakdown). Default ON: licensed 2026-06-08 on the TP=8 pod — at M=1024 the
/// prefill wq_b A/B cut total prefill_ms 14382 → 7628 (−47%) with the needle answer
/// retrieved byte-identically (scalar fp8_gemv scales O(M); it's a decode GEMV).
fn dsv4_prefill_proj_deepgemm_enabled() -> bool {
    cuda_kernels::has_deepgemm_native()
}

/// Prefill DSA indexer query projection → DeepGEMM (134.9 → 6.05ms, −95.5% at M=1024).
/// **Default ON (licensed 2026-06-09).** This was the #1 prefill GPU kernel — the
/// nsys 64K breakdown pinned `dsv4_fp8_gemv_batch_tiled` (this indexer query proj)
/// at **38.4% of all GPU time** (25ms/call, scalar token-looped GEMV). It feeds the
/// top-k block SELECTOR, so it was gated OFF pending a planted-answer long-context
/// needle (an FP8 flip could shift selection). That gate is now MET: with it ON, the
/// planted needle (738291) **retrieves** — 64K hit `738291` exact, 128K hit `738291`
/// exact, and every run finds the needle region (selection intact). Same-binary A/B:
/// 64K prefill 17.6s → 11.0s (−37%), 128K 42.7s → 23.0s (−46%). The exact-digit
/// borderline at ≥2K is the pre-existing compression-fidelity residual (tracked
/// separately), NOT a selection break.
fn dsv4_prefill_indexer_deepgemm_enabled() -> bool {
    cuda_kernels::has_deepgemm_native()
}

pub(crate) fn dsv4_dsa_official_enabled() -> Result<bool> {
    // The vendored/official DSA selector is the only CSA select implementation.
    Ok(true)
}

/// #150 opt-in correctness lever (default OFF): `ARLE_DSV4_PROJ_BATCHED_BF16=1`
/// forces the bf16 cublasLt path in [`proj_batched`] even at m>1, skipping the
/// FP8-repack DeepGEMM lane. Measured partial mitigation for the concurrent-
/// decode digit corruption: n=2 needle miss 57.1%→30.0% (= the n=1 floor),
/// truncation-class corruption from this lane eliminated. Residual digit-
/// substitution originates upstream (`mla_attention_prepare_proj_batch`,
/// F8_E4M3-only weights — no bf16 fallback exists) and is NOT covered. Trades
/// tensor-core throughput at n≥2 — no default flip without a perf license.
fn dsv4_proj_batched_bf16_forced() -> Result<bool> {
    env_flag("ARLE_DSV4_PROJ_BATCHED_BF16")
}

/// #150 opt-in correctness lever (default OFF): `ARLE_DSV4_MLA_PROJ_BF16=1`
/// dequantizes the F8-only MLA `wq_a`/`wq_b`/`wkv` to dense bf16 at LOAD
/// (host-side block dequant, `loader.rs`) and routes BOTH decode lanes — the
/// n=1 fused-wqkv path and the n≥2 batched `mla_attention_prepare_proj_batch`
/// — through bf16 cublasLt, so every decode batch size shares one arithmetic
/// (the #150 near-tie digit-flip mechanism is batch-size-DEPENDENT numerics).
/// Prefill keeps FP8 DeepGEMM (licensed −47%). Checked at load only; runtime
/// gates on `Dsv4Attention::mla_proj_bf16` presence.
pub(crate) fn dsv4_mla_proj_bf16_enabled() -> Result<bool> {
    env_flag("ARLE_DSV4_MLA_PROJ_BF16")
}

/// Batched-decode CSA select-metadata DEVICE build (default OFF). When ON, the
/// per-step block_table/context_lens/positions host builds + 3 `memcpy_htod` are
/// replaced by ONE on-device kernel (removes the per-step H2D a CUDA graph can't
/// bake). Default-off so the baseline stays byte-for-byte unchanged; flip on the
/// pod to A/B via `ARLE_DSV4_DSA_DEVICE_META=1`.
pub(crate) fn dsv4_dsa_device_meta_enabled() -> Result<bool> {
    env_flag("ARLE_DSV4_DSA_DEVICE_META")
}

/// Single-row decode-graph CSA READ via the graph-safe n=1 batched device-meta
/// select (opt-in, default OFF). When ON, the CSA decode-graph path runs the
/// READ (logits + topk) through `csa_select_official_batched` at n=1 with
/// PERSISTENT slot_id/key_count device buffers — no per-step `upload_i32` H2D,
/// so the READ is graph-capturable. The cache WRITE (block (a), still host-shape
/// driven) is NOT yet graph-capturable, so the surrounding `ARLE_DSV4_DECODE_GRAPH`
/// bail stays until a device-driven index-key packer lands; this gate lets the
/// READ path be exercised eagerly (warm runs) and A/B'd on the pod meanwhile.
/// Implies device-meta (the READ needs the on-device build).
pub(crate) fn dsv4_decode_graph_csa_read_enabled() -> Result<bool> {
    env_flag("ARLE_DSV4_DECODE_GRAPH_CSA")
}

/// Pages one layer's FlashMLA shared-pool band needs at `max_seq_len` —
/// `sw_blocks + comp_blocks` from [`Dsv4FlashMlaDecodeShape::new`], without
/// the `local_heads`/`tp_world`/`kv_arena` plumbing that formula also
/// validates (irrelevant to a page-count budget check; real construction
/// still runs those checks for real). `Ok(0)` when the FlashMLA decode-alloc
/// path is disabled — the shared pool isn't built at all, so nothing to
/// check. Lets [`crate::dsv4::Dsv4Model::kv_budget_plan`] reject a
/// startup that can't afford even one slot's band with the same clean error
/// the `affordable` gate already produces, instead of a hard panic deep in
/// `kv_layout.rs`'s pool constructor (pod-verified 2026-07-06: the two gates
/// disagreeing crashes every worker rank).
pub(crate) fn dsv4_flashmla_slot_pages(
    config: &DeepSeekV4Config,
    mode: DeepSeekV4AttentionMode,
    compress_ratio: usize,
    max_seq_len: usize,
    page_block_size: usize,
) -> Result<usize> {
    if !dsv4_flashmla_decode_alloc_enabled()? {
        return Ok(0);
    }
    let sw_blocks = config.sliding_window.div_ceil(page_block_size);
    let comp_blocks = if mode == DeepSeekV4AttentionMode::SlidingWindow {
        0
    } else {
        max_seq_len
            .div_ceil(compress_ratio.max(1))
            .max(1)
            .div_ceil(page_block_size)
    };
    Ok(sw_blocks + comp_blocks)
}

/// Whether a layer's FlashMLA band is DEMAND-PAGED (#154 Phase 3b): comp
/// pages allocate from the layer pool's free list as the sequence grows,
/// instead of a full identity band per slot. MODEL1 only — every MODEL1
/// pack/index kernel routes slot-logical blocks through the device page
/// table (Stage B), so arbitrary physical pages are safe. V32/GLM
/// (`head_dim == 576`) keeps the identity full band: its pack lane still
/// uses contiguous band-base addressing (`flashmla_pages_byte_range`).
pub(crate) fn dsv4_flashmla_demand_paged(config: &DeepSeekV4Config) -> bool {
    config.head_dim != 576
}

/// Per-slot safety pages a demand-paged comp region needs on top of the
/// shared `pool_tokens` capacity: ceil-rounding of per-slot comp pages (+1)
/// plus the MTP verify margin crossing one extra page boundary (+1) — see
/// `Dsv4KvAdapter::prepare_kv_batch`'s `DSV4_BAND_ENSURE_MARGIN_TOKENS`.
pub(crate) const DSV4_COMP_SAFETY_PAGES_PER_SLOT: usize = 2;

/// Pages ONE layer's FlashMLA shared pool is sized to (#154 Phase 3b) — the
/// ONE sizing formula shared by `kv_budget_plan` (solving `pool_tokens`)
/// and `Dsv4LayerKvLayout::new` (allocating), so the two cannot drift.
///
/// Identity layers (V32, or FlashMLA off): `num_slots` full bands.
/// Demand-paged layers: per-slot ring blocks (+ comp safety, see
/// [`DSV4_COMP_SAFETY_PAGES_PER_SLOT`]) + the SHARED comp capacity for
/// `pool_tokens` total tokens across all slots.
pub(crate) fn dsv4_flashmla_layer_pool_pages(
    config: &DeepSeekV4Config,
    mode: DeepSeekV4AttentionMode,
    compress_ratio: usize,
    max_seq_len: usize,
    page_block_size: usize,
    num_slots: usize,
    pool_tokens: usize,
) -> Result<usize> {
    let lsp = dsv4_flashmla_slot_pages(config, mode, compress_ratio, max_seq_len, page_block_size)?;
    if lsp == 0 {
        return Ok(0);
    }
    if !dsv4_flashmla_demand_paged(config) {
        return Ok(num_slots.saturating_mul(lsp));
    }
    let sw_blocks = config.sliding_window.div_ceil(page_block_size);
    if mode == DeepSeekV4AttentionMode::SlidingWindow {
        return Ok(num_slots.saturating_mul(sw_blocks));
    }
    let comp_tokens_per_page = page_block_size.saturating_mul(compress_ratio.max(1));
    let shared_comp = pool_tokens.div_ceil(comp_tokens_per_page.max(1));
    Ok(num_slots
        .saturating_mul(sw_blocks + DSV4_COMP_SAFETY_PAGES_PER_SLOT)
        .saturating_add(shared_comp))
}

/// Whether the FlashMLA shared-band pool is built at all. This is a compile-
/// time question (does the arena exist), not the runtime kernel-choice
/// question `dsv4_flashmla_decode_enabled` answers — `--dsv4-flashmla-decode false`
/// (picking the scalar kernel for an A/B or a correctness reference) must NOT
/// also zero the pool's page budget, since the scalar kernel still reads the
/// same compressed/sliding-window layout (pod-verified 2026-07-06: the
/// fallthrough this replaced sized every slot's band at 0 pages whenever the
/// decode-kernel flag was off, so admission then rejected almost every
/// request against the real per-request page need).
fn dsv4_flashmla_decode_alloc_enabled() -> Result<bool> {
    Ok(cuda_kernels::HAS_FLASHMLA)
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
    // Default ON: fuse wq_a|wkv_a into one FP8 DeepGEMM instead of the scalar
    // `dsv4_fp8_gemv_batch_kernel` (which the clean decode profile pinned at 16.9% of
    // decode GPU — the #1 real decode kernel). Licensed 2026-06-06 on the TP=8/EP=8
    // pod, 64-tok same-binary env A/B: 31.774 -> 37.633 tok/s (+18.4%), token-exact.
    // `dsv4_fused_wqkv_decode_alloc_enabled` falls through to this, so the fused
    // scratch allocates under the default. Runtime preflight probe
    // (`cuda_kernels::has_deepgemm_native()`, cached); scalar fallback when native
    // DeepGEMM is absent. The AtomicI8 override above stays for tests/A-B.
    Ok(cuda_kernels::has_deepgemm_native())
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
        // GLM SparseIndexed selects via the indexer top-k over the FULL per-token
        // latent pool — structurally the CSA sparse index build (uses `selected`),
        // so it shares the CSA kernel mode_int=1. The V32 MODEL_TYPE (0) is a
        // separate constant passed to the fwd kernel (DSV4_FLASHMLA_V32), not this.
        // ponytail: pod-verify SparseIndexed uses the CSA-style (mode_int=1) sparse index build
        DeepSeekV4AttentionMode::SparseIndexed => 1,
    }
}

pub(crate) fn flashmla_pack_sw_ring(
    ctx: &DeviceContext,
    flash: &mut Dsv4FlashMlaDecodeState,
    scratch: &mut Dsv4FlashMlaDecodeScratch,
    pool: &mut Dsv4LayerKvLayout,
    window_cache: &CudaSlice<half::bf16>,
    config: &DeepSeekV4Config,
) -> Result<()> {
    if flash.fp8_kv_sw_bootstrapped {
        return Ok(());
    }
    let sliding_window = config.sliding_window;
    let page_block_size = 64;
    // Stage-B: slot-LOGICAL pages + device page table; the kernel resolves
    // `block_id = table[logical]` into the dynamic pool (pool BASE pointer).
    let (block_ids, rows): (Vec<i32>, Vec<i32>) = (0..sliding_window)
        .map(|slot| {
            (
                (slot / page_block_size) as i32,
                (slot % page_block_size) as i32,
            )
        })
        .unzip();
    ctx.stream
        .memcpy_htod(&block_ids, &mut scratch.sw_bulk_block_ids)
        .map_err(|e| anyhow!("DSv4 FlashMLA SW block_ids H2D failed: {e}"))?;
    ctx.stream
        .memcpy_htod(&rows, &mut scratch.sw_bulk_rows)
        .map_err(|e| anyhow!("DSv4 FlashMLA SW rows H2D failed: {e}"))?;
    let (window_ptr, _wg) = window_cache.device_ptr(&ctx.stream);
    let pool_buf = pool.flashmla_pool_data_mut()?;
    let (pool_ptr, _pg) = pool_buf.device_ptr_mut(&ctx.stream);
    let nope_ptr = window_ptr;
    let rope_ptr = nope_ptr + (config.head_dim - config.qk_rope_head_dim) as u64 * 2;
    flash_kv::dsv4_fp8_kv_pack_strided_raw(
        ctx,
        nope_ptr,
        rope_ptr,
        pool_ptr,
        &scratch.sw_bulk_block_ids,
        &scratch.sw_bulk_rows,
        sliding_window,
        page_block_size,
        config.head_dim,
        config.head_dim,
        Some(&flash.device_page_table),
    )?;
    flash.fp8_kv_sw_bootstrapped = true;
    Ok(())
}

fn flashmla_pack_one_sw_token(
    ctx: &DeviceContext,
    flash: &mut Dsv4FlashMlaDecodeState,
    scratch: &mut Dsv4FlashMlaDecodeScratch,
    pool: &mut Dsv4LayerKvLayout,
    k_prepared: &HiddenStates,
    start_pos_device: &CudaSlice<i32>,
    config: &DeepSeekV4Config,
) -> Result<()> {
    let (bid_ptr, bid_guard) = scratch.one_block_id.device_ptr_mut(&ctx.stream);
    let (row_ptr, row_guard) = scratch.one_row.device_ptr_mut(&ctx.stream);
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
    let nope_ptr = k_ptr;
    // rope offset = head_dim - rope_dim: MODEL1 512-64=448, V32 576-64=512.
    // stride = head_dim: MODEL1 512, V32 576. The config formula yields the
    // right offset/stride for both shapes; only the pack FN differs. V32 writes
    // the inline 656 B/tok layout [512 NoPE fp8][4 F32 scales @512][128 rope bf16]
    // (4× 128-elem NoPE blocks, F32 scale=amax/448) per the vendored decode.
    // ponytail: pod-verify V32 pack offsets (nope@0 stride576, rope@512 stride576) + 656 B/tok pool addressing
    let rope_ptr = nope_ptr + (config.head_dim - config.qk_rope_head_dim) as u64 * 2;
    if config.head_dim == 576 {
        // V32/GLM has no device-page-table pack kernel; it keeps band-base
        // addressing (the band stays contiguous), slicing the slot's range.
        let range = pool.flashmla_pages_byte_range(flash.slot_idx)?;
        let pool_buf = pool.flashmla_pool_data_mut()?;
        ensure!(
            range.end <= pool_buf.len() && range.len() == flash.fp8_kv_pool_len,
            "DSv4 FlashMLA shared one-token table range {:?} invalid pool_len={} slot_len={}",
            range,
            pool_buf.len(),
            flash.fp8_kv_pool_len
        );
        let mut pool_view = pool_buf.slice_mut(range);
        let (pool_ptr, _pg) = pool_view.device_ptr_mut(&ctx.stream);
        flash_kv::dsv4_v32_fp8_kv_pack_strided_raw(
            ctx,
            nope_ptr,
            rope_ptr,
            pool_ptr,
            &scratch.one_block_id,
            &scratch.one_row,
            1,
            64,
            config.head_dim,
            config.head_dim,
        )
    } else {
        // MODEL1 Stage-B: the device fill kernel produced a slot-LOGICAL block
        // id; hand the POOL base + device page table so it routes to the dynamic
        // physical block (`block_id = table[logical]`).
        let pool_buf = pool.flashmla_pool_data_mut()?;
        let (pool_ptr, _pg) = pool_buf.device_ptr_mut(&ctx.stream);
        flash_kv::dsv4_fp8_kv_pack_strided_raw(
            ctx,
            nope_ptr,
            rope_ptr,
            pool_ptr,
            &scratch.one_block_id,
            &scratch.one_row,
            1,
            64,
            config.head_dim,
            config.head_dim,
            Some(&flash.device_page_table),
        )
    }
}

fn flashmla_pack_compressed_delta(
    ctx: &DeviceContext,
    flash: &mut Dsv4FlashMlaDecodeState,
    scratch: &mut Dsv4FlashMlaDecodeScratch,
    pool: &mut Dsv4LayerKvLayout,
    compressed: Option<&HiddenStates>,
    start_pos_device: &CudaSlice<i32>,
    compress_ratio: usize,
    config: &DeepSeekV4Config,
) -> Result<()> {
    let Some(compressed) = compressed else {
        return Ok(());
    };
    // The ONE block→(page,row) map for this slot's band. Both the single-row
    // device pack (below) and the bulk host-derived block_ids draw sw_blocks /
    // page_size from here, so the two write paths cannot drift (#146).
    let bmap = flash.block_map();
    // Steady-state decode adds AT MOST one compressed row per step
    // ((pos+1) % ratio == 0). That row is packed by the DEVICE kernel below —
    // fully derived from `start_pos_device`, so it records into CUDA-graph
    // captures and stays correct on replay (the old host Vec + H2D path is
    // skipped on replay entirely, which stalled the pool → garbage/IMA).
    // The host bulk path remains ONLY for multi-row gaps (first decode after
    // prefill / request boundaries), which always execute eagerly (the graph
    // warm pass — see `CudaGraphState::rearm_warm`).
    {
        // Stage-B: hand the POOL base + device page table so the kernel routes
        // the slot-LOGICAL compressed block to its physical pool block (fragmented
        // band safe; identity table == band byte-for-byte).
        let pool_buf = pool.flashmla_pool_data_mut()?;
        let (pool_ptr, _pg) = pool_buf.device_ptr_mut(&ctx.stream);
        let (compressed_ptr, _cg) = compressed.data.device_ptr(&ctx.stream);
        let (start_ptr, _sg) = start_pos_device.device_ptr(&ctx.stream);
        flash_kv::dsv4_fp8_kv_pack_completed_compressor_row_start_pos_raw(
            ctx,
            compressed_ptr,
            pool_ptr,
            start_ptr,
            compress_ratio,
            bmap.sw_blocks(),
            bmap.page_size(),
            config.head_dim,
            Some(&flash.device_page_table),
        )?;
    }
    let start_row = flash.fp8_kv_comp_packed_rows;
    let end_row = compressed.seq_len;
    if end_row <= start_row {
        return Ok(());
    }
    // Host bookkeeping below runs in eager contexts only (warm pass / no
    // graph); the device kernel above already covered the single-row case, so
    // bulk-pack only multi-row gaps. The boundary row may be packed by both
    // paths — idempotent overwrite of identical data.
    flash.fp8_kv_comp_packed_rows = end_row;
    if end_row == start_row + 1 {
        return Ok(());
    }
    let n = end_row - start_row;
    // Bulk rebuild observability (codex R3): volume scales with the restored/
    // prefilled length (~584 B/row/layer, e.g. matched=8064 → ~1.2 MB/layer,
    // ~25 MB across the 21 CSA layers) — NVTX for traces, debug for counters.
    crate::profile::profile_op(ctx, "flashmla_bulk_comp_rebuild", None, n, || {
        log::debug!(
            "DSv4 FlashMLA bulk comp rebuild: rows {start_row}..{end_row} ({n} rows, {} bytes)",
            n * config.head_dim * 2
        );
        let (block_ids, rows): (Vec<i32>, Vec<i32>) = (start_row..end_row)
            .map(|row| {
                let (page, in_page) = bmap.comp_row(row);
                (page as i32, in_page as i32)
            })
            .unzip();
        ctx.stream
            .memcpy_htod(&block_ids, &mut scratch.comp_block_ids)
            .map_err(|e| anyhow!("DSv4 FlashMLA compressed block_ids H2D failed: {e}"))?;
        ctx.stream
            .memcpy_htod(&rows, &mut scratch.comp_rows)
            .map_err(|e| anyhow!("DSv4 FlashMLA compressed rows H2D failed: {e}"))?;

        // Stage-B: `block_ids` carries slot-LOGICAL pages (`sw_blocks + row/64`);
        // hand the POOL base + device page table so the kernel routes each to its
        // dynamic physical block (`block_id = table[logical]`).
        let (compressed_ptr, _cg) = compressed.data.device_ptr(&ctx.stream);
        let pool_buf = pool.flashmla_pool_data_mut()?;
        let (pool_ptr, _pg) = pool_buf.device_ptr_mut(&ctx.stream);
        let row_offset_bytes = start_row as u64 * config.head_dim as u64 * 2;
        let nope_ptr = compressed_ptr + row_offset_bytes;
        let rope_ptr = nope_ptr + (config.head_dim - config.qk_rope_head_dim) as u64 * 2;
        flash_kv::dsv4_fp8_kv_pack_strided_raw(
            ctx,
            nope_ptr,
            rope_ptr,
            pool_ptr,
            &scratch.comp_block_ids,
            &scratch.comp_rows,
            n,
            bmap.page_size(),
            config.head_dim,
            config.head_dim,
            Some(&flash.device_page_table),
        )?;
        flash.fp8_kv_comp_packed_rows = end_row;
        Ok(())
    })?;
    Ok(())
}

/// Host-side tail slice for the SW ring update. Only the last `window` rows of
/// a `seq_len`-row batch can survive in the ring; writing the earlier rows too
/// makes up to `ceil(seq_len/window)` unordered same-slot writers race in
/// `dsv4_update_window_cache_kernel` (`slot = (start_pos+token) % window`,
/// dsv4_swa.cu) — nondeterministic SWA keys for the next chunk once prefill
/// chunks exceed the window. Returns `(rows_skipped, adjusted_start_pos,
/// rows_to_write)`; a no-op `(0, start_pos, seq_len)` at `seq_len <= window`.
fn sw_ring_tail_slice(seq_len: usize, window: usize, start_pos: usize) -> (usize, usize, usize) {
    let skip = seq_len.saturating_sub(window);
    (skip, start_pos + skip, seq_len - skip)
}

fn update_bf16_sw_window(
    ctx: &DeviceContext,
    sw_window_cache: &mut CudaSlice<half::bf16>,
    k_prepared: &HiddenStates,
    start_pos: usize,
    start_pos_device: Option<&CudaSlice<i32>>,
    config: &DeepSeekV4Config,
) -> Result<()> {
    // GLM pure-SparseIndexed has no SW ring (sliding_window==0); the window-cache
    // update kernel does `% sliding_window` → would divide by zero. Nothing to
    // ring when there is no window.
    // ponytail: pod-verify GLM pure-SparseIndexed (sliding_window=0) skips SW-ring entirely; attention is indexer-selected full-latent only
    if config.sliding_window == 0 {
        return Ok(());
    }
    let (k_ptr, _kg) = k_prepared.data.device_ptr(&ctx.stream);
    let (window_ptr, _wg) = sw_window_cache.device_ptr_mut(&ctx.stream);
    // SAFETY: ptrs from live device allocations sized to the dims passed.
    unsafe {
        if let Some(start_pos_device) = start_pos_device {
            // start_pos lives on device — the host can't tail-slice. Decode/MTP
            // rows are far below the window; oversize here is a bug, not a path.
            ensure!(
                k_prepared.seq_len <= config.sliding_window,
                "DSv4 SW ring device-start_pos update rows {} > window {} (host tail-slice \
                 unavailable; the update kernel would race same-slot writers)",
                k_prepared.seq_len,
                config.sliding_window
            );
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
            let (skip, start_pos, rows) =
                sw_ring_tail_slice(k_prepared.seq_len, config.sliding_window, start_pos);
            let k_ptr = k_ptr + (skip * config.head_dim * 2) as u64;
            ffi::dsv4_update_window_cache_cuda(
                k_ptr as *const ffi::Half,
                window_ptr as *mut ffi::Half,
                rows as i32,
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

/// Batched DSv4 sparse-verify attention metadata. Row `r` is at
/// `positions[r]` and attends committed KV plus the listed earlier chunk
/// ancestors and self. The caller decides whether those rows are a linear chain
/// or the bounded D2 top-k branch shape.
pub(crate) struct Dsv4ChainVerifyAttnMeta {
    pub(crate) positions: CudaSlice<i32>,
    pub(crate) ancestors: CudaSlice<i32>,
    pub(crate) max_anc: usize,
    pub(crate) n_rows: usize,
}

impl Dsv4ChainVerifyAttnMeta {
    pub(crate) fn new(
        ctx: &DeviceContext,
        positions: &[usize],
        ancestors: &[Vec<usize>],
    ) -> Result<Self> {
        let n = positions.len();
        ensure!(
            n > 0 && ancestors.len() == n,
            "DSv4 chain verify meta shape mismatch: positions={} ancestors={}",
            n,
            ancestors.len()
        );
        let max_anc = ancestors.iter().map(Vec::len).max().unwrap_or(0).max(1);
        let pos_host: Vec<i32> = positions
            .iter()
            .map(|&p| {
                i32::try_from(p)
                    .map_err(|_| anyhow!("DSv4 chain verify position {p} overflows i32"))
            })
            .collect::<Result<_>>()?;
        let mut anc_host = vec![-1i32; n * max_anc];
        for (r, chain) in ancestors.iter().enumerate() {
            for (j, &a) in chain.iter().enumerate() {
                ensure!(a < n, "DSv4 chain verify ancestor row {a} out of {n} rows");
                anc_host[r * max_anc + j] = a as i32;
            }
        }
        Ok(Self {
            positions: ctx
                .stream
                .clone_htod(&pos_host)
                .map_err(|e| anyhow!("DSv4 chain verify positions H2D failed: {e}"))?,
            ancestors: ctx
                .stream
                .clone_htod(&anc_host)
                .map_err(|e| anyhow!("DSv4 chain verify ancestors H2D failed: {e}"))?,
            max_anc,
            n_rows: n,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn flashmla_prefill_attention(
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
    start_pos: usize,
    chain_verify: Option<&Dsv4ChainVerifyAttnMeta>,
    tp: &TpRuntime,
    local_heads: usize,
    local_attn: &mut HiddenStates,
    sm_scale: f32,
    rope_base: f32,
    original_seq_len: i32,
    rope_factor: f32,
    rope_beta_fast: f32,
    rope_beta_slow: f32,
) -> Result<()> {
    ensure!(
        dsv4_flashmla_prefill_enabled()?,
        "DSv4 FlashMLA prefill is not available"
    );
    ensure!(
        q_prepared.seq_len > 1,
        "DSv4 FlashMLA prefill requires seq_len > 1, got {}",
        q_prepared.seq_len
    );
    // MODEL1 (head_dim=512) + V32 (GLM, head_dim=576 = 512 latent + 64 rope).
    let meta = dsv4_flashmla_model_meta(config)?;
    let (model_type_int, bytes_per_token) = (meta.model_type_int, meta.bytes_per_token);
    let is_v32 = model_type_int == DSV4_FLASHMLA_V32;
    let _ = (model_type_int, bytes_per_token, is_v32);
    ensure!(
        q_prepared.seq_len == k_prepared.seq_len && local_attn.seq_len == q_prepared.seq_len,
        "DSv4 FlashMLA prefill shape mismatch: q={} k={} out={}",
        q_prepared.seq_len,
        k_prepared.seq_len,
        local_attn.seq_len
    );

    let token_count = q_prepared.seq_len;
    if let Some(meta) = chain_verify {
        ensure!(
            meta.n_rows == token_count,
            "DSv4 chain verify meta rows {} != verify rows {token_count}",
            meta.n_rows
        );
    }
    let tp_world = tp.config().world_size;
    let tp_rank = tp.config().rank;
    let global_heads = local_heads
        .checked_mul(tp_world)
        .ok_or_else(|| anyhow!("DSv4 FlashMLA prefill global head overflow"))?;
    ensure!(
        matches!(global_heads, 64 | 128),
        "DSv4 FlashMLA prefill requires global heads 64/128, got {global_heads}"
    );

    let compressed_count = compressed.map_or(0, |c| c.seq_len);
    let max_compressed_keys = match mode {
        DeepSeekV4AttentionMode::CompressedSparse => config.index_topk,
        DeepSeekV4AttentionMode::HybridCompressed => compressed_count.div_ceil(128) * 128,
        DeepSeekV4AttentionMode::SlidingWindow => 0,
        // GLM SparseIndexed: top-k cap over the full latent pool == CSA.
        DeepSeekV4AttentionMode::SparseIndexed => config.index_topk,
    };
    let chain_pad = if chain_verify.is_some() { 128 } else { 0 };
    let topk_unified = config
        .sliding_window
        .checked_add(chain_pad)
        .and_then(|v| v.checked_add(max_compressed_keys))
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
    // SAFETY: uninit device scratch; fully written before first read.
    let mut kv_unified = unsafe { HiddenStates::uninit(ctx, config.head_dim, kv_rows)? };
    {
        crate::profile::profile_op(ctx, "flashmla_prefill_pack_kv", None, token_count, || {
            let (kv_ptr, _kvg) = kv_unified.data.device_ptr_mut(&ctx.stream);
            let (window_ptr, _wg) = sw_window_cache.device_ptr(&ctx.stream);
            let (k_ptr, _kg) = k_prepared.data.device_ptr(&ctx.stream);
            let (comp_ptr, _cg) = match compressed.filter(|_| compressed_count > 0) {
                Some(c) => {
                    let (p, g) = c.data.device_ptr(&ctx.stream);
                    (p as *const ffi::Half, Some(g))
                }
                None => (std::ptr::null(), None),
            };
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            unsafe {
                ffi::arle_flashmla_csa_pack_kv(
                    kv_ptr as *mut ffi::Half,
                    window_ptr as *const ffi::Half,
                    k_ptr as *const ffi::Half,
                    comp_ptr,
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
            Ok(())
        })?;
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
        crate::profile::profile_op(
            ctx,
            "flashmla_prefill_build_indices",
            None,
            token_count,
            || {
                let (indices_ptr, _ig) = indices.device_ptr_mut(&ctx.stream);
                let (topk_ptr, _tg) = topk_length.device_ptr_mut(&ctx.stream);
                if let Some(meta) = chain_verify {
                    let (pos_ptr, _pg) = meta.positions.device_ptr(&ctx.stream);
                    let (anc_ptr, _ag) = meta.ancestors.device_ptr(&ctx.stream);
                    let (sel_ptr, _sg) = match (mode, selected) {
                        (DeepSeekV4AttentionMode::CompressedSparse, Some(sel)) => {
                            let (p, g) = sel.device_ptr(&ctx.stream);
                            (p as *const i32, Some(g))
                        }
                        (DeepSeekV4AttentionMode::CompressedSparse, None) => {
                            bail!("DSv4 FlashMLA CSA chain verify missing selected topk")
                        }
                        _ => (std::ptr::null(), None),
                    };
                    let max_compressed_arg = if mode == DeepSeekV4AttentionMode::HybridCompressed {
                        max_compressed_keys
                    } else {
                        0
                    };
                    // SAFETY: ptrs from live device allocations sized to the dims passed.
                    unsafe {
                        ffi::arle_flashmla_chain_verify_build_indices(
                            indices_ptr as *mut i32,
                            topk_ptr as *mut i32,
                            pos_ptr as *const i32,
                            anc_ptr as *const i32,
                            meta.max_anc as i32,
                            sel_ptr,
                            token_count as i32,
                            start_pos as i32,
                            config.sliding_window as i32,
                            if sel_ptr.is_null() {
                                0
                            } else {
                                config.index_topk as i32
                            },
                            max_compressed_arg as i32,
                            topk_unified as i32,
                            compressed_count as i32,
                            compress_ratio as i32,
                            ctx.stream.cu_stream(),
                        )
                        .result()
                        .map_err(|e| anyhow!("DSv4 FlashMLA chain verify indices failed: {e}"))?;
                    }
                } else {
                    // SAFETY: ptrs from live device allocations sized to the dims passed.
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
                                .map_err(|e| {
                                    anyhow!("DSv4 FlashMLA CSA prefill indices failed: {e}")
                                })?;
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
                                .map_err(|e| {
                                    anyhow!("DSv4 FlashMLA HCA prefill indices failed: {e}")
                                })?;
                            }
                            DeepSeekV4AttentionMode::SlidingWindow => {
                                // SWA: no compressed pool — reuse the CSA index
                                // builder with selected=null (fills -1) and
                                // index_topk=0, so indices hold SW blocks only.
                                ffi::arle_flashmla_csa_build_indices(
                                    indices_ptr as *mut i32,
                                    topk_ptr as *mut i32,
                                    std::ptr::null(),
                                    token_count as i32,
                                    start_pos as i32,
                                    config.sliding_window as i32,
                                    max_compressed_keys as i32,
                                    compressed_count as i32,
                                    compress_ratio as i32,
                                    ctx.stream.cu_stream(),
                                )
                                .result()
                                .map_err(|e| {
                                    anyhow!("DSv4 FlashMLA SWA prefill indices failed: {e}")
                                })?;
                            }
                            DeepSeekV4AttentionMode::SparseIndexed => {
                                // GLM SparseIndexed mirrors the CSA index build (top-k over
                                // `selected`), but over the FULL per-token latent (no
                                // compressor): pass compress_ratio=1.
                                let selected = selected.ok_or_else(|| {
                                    anyhow!(
                                        "DSv4 FlashMLA SparseIndexed prefill missing selected topk"
                                    )
                                })?;
                                let (selected_ptr, _sg) = selected.device_ptr(&ctx.stream);
                                // ponytail: pod-verify SparseIndexed prefill index build uses ratio=1 (full latent)
                                ffi::arle_flashmla_csa_build_indices(
                                    indices_ptr as *mut i32,
                                    topk_ptr as *mut i32,
                                    selected_ptr as *const i32,
                                    token_count as i32,
                                    start_pos as i32,
                                    config.sliding_window as i32,
                                    config.index_topk as i32,
                                    compressed_count as i32,
                                    1,
                                    ctx.stream.cu_stream(),
                                )
                                .result()
                                .map_err(|e| {
                                    anyhow!(
                                        "DSv4 FlashMLA SparseIndexed prefill indices failed: {e}"
                                    )
                                })?;
                            }
                        }
                    }
                }
                Ok(())
            },
        )?;
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
                crate::profile::profile_op(
                    ctx,
                    "flashmla_prefill_q_allgather",
                    None,
                    token_count,
                    || {
                        // SAFETY: q spans token_count*local_width; gathered holds tp_world x that, on ctx.stream.
                        unsafe {
                            tp.all_gather_bf16_raw(
                                ctx,
                                q_ptr as *const std::ffi::c_void,
                                token_count * local_width,
                                gather_ptr as *mut std::ffi::c_void,
                            )?;
                        }
                        Ok(())
                    },
                )?;
            }
            drop(gather_guard);
            let (packed_ptr, packed_guard) = packed.device_ptr_mut(&ctx.stream);
            {
                crate::profile::profile_op(
                    ctx,
                    "flashmla_prefill_q_repack",
                    None,
                    token_count,
                    || {
                        // SAFETY: ptrs from live device allocations sized to the dims passed.
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
                            .map_err(|e| {
                                anyhow!("DSv4 FlashMLA prefill TP Q repack failed: {e}")
                            })?;
                        }
                        Ok(())
                    },
                )?;
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

    let (sink_base, sink_guard) = attention
        .attn_sink_f32
        .as_ref()
        .expect("DSv4 attn_sink_f32")
        .device_ptr(&ctx.stream);
    ensure!(
        if tp_world > 1 {
            attention
                .attn_sink_f32
                .as_ref()
                .expect("DSv4 attn_sink_f32")
                .len()
                >= global_heads
        } else {
            attention
                .attn_sink_f32
                .as_ref()
                .expect("DSv4 attn_sink_f32")
                .len()
                >= tp_rank * local_heads + local_heads
        },
        "DSv4 FlashMLA prefill attn_sink_f32 len {} cannot cover heads",
        attention
            .attn_sink_f32
            .as_ref()
            .expect("DSv4 attn_sink_f32")
            .len()
    );
    let sink_ptr = if tp_world > 1 {
        sink_base as *const f32
    } else {
        // SAFETY: ensure! above bounds tp_rank*local_heads + local_heads <= sink len.
        unsafe { (sink_base as *const f32).add(tp_rank * local_heads) }
    };

    {
        crate::profile::profile_op(ctx, "flashmla_prefill_fwd", None, token_count, || {
            // q8kv8 fp8 sparse prefill is FlashMLA prefill's fp8 twin: it occupies
            // the SAME execution point (post-gather global-head Q, global out) so it
            // runs on the production TP path. Gate on global_heads%64 (the kernel's
            // GMMA head-tile), not local_heads — a TP=4 shard has local_heads=16 but
            // the gathered Q here is all `global_heads`.
            if dsv4_q8kv8_prefill_enabled()? && global_heads % 64 == 0 {
                crate::profile::profile_op(ctx, "q8kv8_prefill", None, token_count, || {
                    let head_dim = config.head_dim;
                    let q_elems = token_count * global_width;
                    let kv_elems = kv_rows * head_dim;
                    let mut q_fp8 = ctx.stream.alloc_zeros::<u8>(q_elems)?;
                    let mut kv_fp8 = ctx.stream.alloc_zeros::<u8>(kv_elems)?;
                    let mut scale = ctx.stream.alloc_zeros::<f32>(2)?;
                    ctx.stream.memcpy_htod(&[1.0f32, 1.0f32], &mut scale)?;
                    // Cast the gathered global-head Q and the shared latent KV to fp8.
                    let cast_fp8 =
                        |src: *const ffi::Half, dst: &mut CudaSlice<u8>, n: usize| -> Result<()> {
                            let (dst_ptr, _dg) = dst.device_ptr_mut(&ctx.stream);
                            // SAFETY: src spans n elements; dst holds n bytes, fully written.
                            unsafe {
                                ffi::arle_bf16_to_fp8_e4m3_cuda(
                                    src,
                                    dst_ptr as *mut u8,
                                    n as i64,
                                    ctx.stream.cu_stream(),
                                )
                                .result()?;
                            }
                            Ok(())
                        };
                    cast_fp8(q_for_flashmla as *const ffi::Half, &mut q_fp8, q_elems)?;
                    cast_fp8(kv_ptr as *const ffi::Half, &mut kv_fp8, kv_elems)?;
                    let (q_fp8_ptr, _qfg) = q_fp8.device_ptr(&ctx.stream);
                    let (kv_fp8_ptr, _kfg) = kv_fp8.device_ptr(&ctx.stream);
                    let (scale_ptr, _sg) = scale.device_ptr(&ctx.stream);
                    // indices/topk reuse FlashMLA's; topk_length=null → fixed-topk path.
                    // SAFETY: fp8 buffers filled above; out is global-width, sliced below.
                    unsafe {
                        ffi::arle_q8kv8_sparse_prefill_fwd(
                            q_fp8_ptr as *const u8,
                            kv_fp8_ptr as *const u8,
                            indices_ptr as *const i32,
                            scale_ptr as *const f32,
                            scale_ptr as *const f32,
                            sink_ptr,
                            std::ptr::null(),
                            flash_out_ptr as *mut ffi::Half,
                            max_ptr as *mut f32,
                            lse_ptr as *mut f32,
                            token_count as i32,
                            kv_rows as i32,
                            global_heads as i32,
                            head_dim as i32,
                            topk_unified as i32,
                            sm_scale,
                            ctx.stream.cu_stream(),
                        )
                        .result()
                        .map_err(|e| anyhow!("DSv4 q8kv8 prefill failed: {e}"))?;
                    }
                    Ok(())
                })?;
            } else {
                // SAFETY: ptrs from live device allocations sized to the dims passed.
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
            Ok(())
        })?;
    }

    if tp_world > 1 {
        let full_out = tp_full_out
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 FlashMLA prefill missing TP full output"))?;
        let (full_out_ptr, full_out_guard) = full_out.device_ptr(&ctx.stream);
        {
            crate::profile::profile_op(
                ctx,
                "flashmla_prefill_out_slice",
                None,
                token_count,
                || {
                    // SAFETY: ptrs from live device allocations sized to the dims passed.
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
                    Ok(())
                },
            )?;
        }
        drop(full_out_guard);
    }

    {
        crate::profile::profile_op(
            ctx,
            "flashmla_prefill_inverse_rope",
            None,
            token_count,
            || {
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                unsafe {
                    if let Some(meta) = chain_verify {
                        let (pos_ptr, _pg) = meta.positions.device_ptr(&ctx.stream);
                        ffi::arle_dsv4_output_inverse_rope_batch_start_pos_cuda(
                            out_ptr as *mut ffi::Half,
                            token_count as i32,
                            local_heads as i32,
                            config.head_dim as i32,
                            config.qk_rope_head_dim as i32,
                            pos_ptr as *const i32,
                            rope_base,
                            original_seq_len,
                            rope_factor,
                            rope_beta_fast,
                            rope_beta_slow,
                            ctx.stream.cu_stream(),
                        )
                        .result()
                        .map_err(|e| {
                            anyhow!("DSv4 FlashMLA chain verify output inverse-rope failed: {e}")
                        })?;
                    } else {
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
                        .map_err(|e| {
                            anyhow!("DSv4 FlashMLA prefill output inverse-rope failed: {e}")
                        })?;
                    }
                }
                Ok(())
            },
        )?;
    }

    if chain_verify.is_none() {
        update_bf16_sw_window(ctx, sw_window_cache, k_prepared, start_pos, None, config)?;
    }

    // Keep temporary buffers in scope until all launches that use their raw
    // pointers have been enqueued.
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

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn flashmla_decode_attention(
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
    scratch: &mut Dsv4FlashMlaDecodeScratch,
    pool: &mut Dsv4LayerKvLayout,
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
) -> Result<()> {
    ensure!(
        dsv4_flashmla_decode_enabled()?,
        "DSv4 FlashMLA decode is not available"
    );
    ensure!(
        q_prepared.seq_len == 1,
        "DSv4 FlashMLA decode requires seq_len == 1, got {}",
        q_prepared.seq_len
    );
    let start_pos_device = start_pos_device.ok_or_else(|| {
        anyhow!("DSv4 FlashMLA decode requires device start_pos for token_count=1")
    })?;
    // MODEL1 (DSv4, head_dim=512) and V32 (GLM, head_dim=576 = 512 latent NoPE
    // + 64 RoPE). The FlashMLA shim reads q[heads, d_qk] and writes out[heads,
    // d_v=512 latent]; for MODEL1 d_qk==d_v==512, for V32 d_qk=576 but d_v=512.
    let meta = dsv4_flashmla_model_meta(config)?;
    let (model_type_int, bytes_per_token) = (meta.model_type_int, meta.bytes_per_token);
    let is_v32 = model_type_int == DSV4_FLASHMLA_V32;
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

    // GLM pure-SparseIndexed (sliding_window==0): there is no SW ring to
    // bootstrap; attention is indexer-selected full-latent only. The
    // per-token KV pack (`flashmla_pack_one_sw_token`) still runs — it
    // populates THIS token's latent into the sparse pool the indexer selects
    // from. The SW-ring bootstrap is the only SW-specific step here.
    // ponytail: pod-verify GLM pure-SparseIndexed (sliding_window=0) skips SW-ring entirely; attention is indexer-selected full-latent only
    if config.sliding_window > 0 {
        crate::profile::profile_op(ctx, "flashmla_pack_sw_ring", None, 1, || {
            flashmla_pack_sw_ring(ctx, flash, scratch, pool, sw_window_cache, config)
        })?;
    }

    {
        crate::profile::profile_op(ctx, "flashmla_pack_one", None, 1, || {
            flashmla_pack_one_sw_token(
                ctx,
                flash,
                scratch,
                pool,
                k_prepared,
                start_pos_device,
                config,
            )
        })?;
    }
    {
        crate::profile::profile_op(ctx, "flashmla_pack_compressed", None, 1, || {
            flashmla_pack_compressed_delta(
                ctx,
                flash,
                scratch,
                pool,
                compressed,
                start_pos_device,
                compress_ratio,
                config,
            )
        })?;
    }

    let mode_int = flashmla_mode_int(mode);
    // Indexer modes (CSA + GLM SparseIndexed) feed the per-row top-k `selected`
    // (shared mode_int=1); SW/HCA pass selected_ptr=0.
    let selected_ptr_u64 = if mode.has_indexer() {
        let selected =
            selected.ok_or_else(|| anyhow!("DSv4 FlashMLA indexer missing selected topk"))?;
        let (ptr, guard) = selected.device_ptr(&ctx.stream);
        let ptr_u64 = ptr;
        drop(guard);
        ptr_u64
    } else {
        0
    };
    // Stage-B: route the device page table so build_indices emits POOL-ABSOLUTE
    // physical indices (slot-logical page -> physical pool page), fed against the
    // WHOLE-pool base below (matches the batched path, which routes the table for
    // every mode). On a contiguous identity table (Stage-A) this is byte-equal to
    // the old slot-relative + band-slice path (gate: `dsv4_decode_route_index`
    // host bit-identity test); once the pool fragments it lets the slot draw
    // non-contiguous pages. V32/GLM also route here: only the WRITE/pack side
    // lacks a V32 device-page-table kernel; the read-side table is mode-neutral
    // and their contiguous identity band stays byte-equal.
    let build_page_table = Some(&flash.device_page_table);
    // Single-source sw_blocks / page_block_size for the index-build kernel.
    let bmap = flash.block_map();
    let (indices_ptr, indices_guard) = scratch.indices.device_ptr_mut(&ctx.stream);
    let (start_ptr, start_guard) = start_pos_device.device_ptr(&ctx.stream);
    {
        crate::profile::profile_op(ctx, "flashmla_build_indices", None, 1, || {
            flash_kv::dsv4_flashmla_decode_build_indices_start_pos_ptr_raw(
                ctx,
                indices_ptr,
                selected_ptr_u64,
                bmap.sw_blocks(),
                config.sliding_window,
                start_ptr,
                flash.max_compressed_keys,
                // GLM SparseIndexed: every token a key (ratio=1); SW also 1; CSA/HCA
                // keep compress_ratio.
                if mode == DeepSeekV4AttentionMode::SlidingWindow
                    || mode == DeepSeekV4AttentionMode::SparseIndexed
                {
                    1
                } else {
                    compress_ratio
                },
                mode_int,
                bmap.page_size(),
                build_page_table,
                // M1: whole-pool page count — mask any routed physical page >= this.
                pool.flashmla_total_pages(),
            )
        })?;
    }
    drop(indices_guard);
    drop(start_guard);

    // topk_length + scheduler metadata are slot constants, computed once at
    // state init (`init_constant_sched_meta`) — see the capture-hazard note
    // there. Saves 43 sched-meta calls/token as a side effect.

    let (q_ptr, q_guard) = q_prepared.data.device_ptr(&ctx.stream);
    // Stage-B: feed the FlashMLA decode kernel the WHOLE-pool base; the indices
    // built above are POOL-ABSOLUTE (page table routed), matching the batched
    // path. V32/GLM still route a contiguous identity table so whole-pool base +
    // band-equal indices stay byte-identical to the old per-slot slice.
    let pool_buf = pool.flashmla_pool_data()?;
    let (pool_ptr, pool_guard) = pool_buf.device_ptr(&ctx.stream);
    let (out_ptr, out_guard) = local_attn.data.device_ptr_mut(&ctx.stream);
    let (lse_out_ptr, lse_guard) = scratch.lse_out.device_ptr_mut(&ctx.stream);
    let (lse_accum_ptr, lse_accum_guard) = scratch.lse_accum.device_ptr_mut(&ctx.stream);
    let (o_accum_ptr, o_accum_guard) = scratch.o_accum.device_ptr_mut(&ctx.stream);
    let (indices_ptr, indices_guard) = scratch.indices.device_ptr(&ctx.stream);
    let (topk_ptr, topk_guard) = flash.topk_length.device_ptr(&ctx.stream);
    let (sched_ptr, sched_guard) = flash.sched_meta.device_ptr(&ctx.stream);
    let (splits_ptr, splits_guard) = flash.num_splits.device_ptr(&ctx.stream);

    let q_for_flashmla = if tp_world > 1 {
        let (gather_ptr, gather_guard) = scratch.tp_gathered_q.device_ptr_mut(&ctx.stream);
        {
            crate::profile::profile_op(ctx, "flashmla_q_allgather", None, 1, || {
                // SAFETY: q spans token_count*local_width; gathered holds tp_world x that, on ctx.stream.
                unsafe {
                    tp.all_gather_bf16_raw(
                        ctx,
                        q_ptr as *const std::ffi::c_void,
                        local_heads * config.head_dim,
                        gather_ptr as *mut std::ffi::c_void,
                    )?;
                }
                Ok(())
            })?;
        }
        drop(gather_guard);
        let (packed_ptr, packed_guard) = scratch.tp_packed_q.device_ptr_mut(&ctx.stream);
        {
            crate::profile::profile_op(ctx, "flashmla_q_repack", None, 1, || {
                // SAFETY: ptrs from live device allocations sized to the dims passed.
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
                Ok(())
            })?;
        }
        drop(packed_guard);
        packed_ptr as *const ffi::Half
    } else {
        q_ptr as *const ffi::Half
    };

    let (sink_base, sink_guard) = attention
        .attn_sink_f32
        .as_ref()
        .expect("DSv4 attn_sink_f32")
        .device_ptr(&ctx.stream);
    ensure!(
        if tp_world > 1 {
            attention
                .attn_sink_f32
                .as_ref()
                .expect("DSv4 attn_sink_f32")
                .len()
                >= global_heads
        } else {
            attention
                .attn_sink_f32
                .as_ref()
                .expect("DSv4 attn_sink_f32")
                .len()
                >= tp_rank * local_heads + local_heads
        },
        "DSv4 FlashMLA attn_sink_f32 len {} cannot cover heads",
        attention
            .attn_sink_f32
            .as_ref()
            .expect("DSv4 attn_sink_f32")
            .len()
    );
    let sink_ptr = if tp_world > 1 {
        sink_base as *const f32
    } else {
        // SAFETY: ensure! above bounds tp_rank*local_heads + local_heads <= sink len.
        unsafe { (sink_base as *const f32).add(tp_rank * local_heads) }
    };

    let flash_out_ptr = if tp_world > 1 {
        let (full_out_ptr, full_out_guard) = scratch.tp_full_out.device_ptr_mut(&ctx.stream);
        drop(full_out_guard);
        full_out_ptr as *mut ffi::Half
    } else {
        out_ptr as *mut ffi::Half
    };

    let stride_kv_block_bytes = 64_i32 * bytes_per_token;
    // q is [b, s_q, h_q, d_qk]: d_qk = head_dim (512 MODEL1 / 576 V32).
    // out / o_accum are [..., h_q, d_v]: d_v = kv_lora latent = 512 ALWAYS
    // (the shim hard-asserts d_v==512). For MODEL1 head_dim==d_v==512 so the
    // two coincide; for V32 they diverge (q=576, output latent=512).
    // ponytail: pod-verify V32 FlashMLA decode stride/dim arg mapping (d_qk=576 latent=512)
    let d_qk = config.head_dim as i32;
    let d_v = if is_v32 { 512 } else { config.head_dim as i32 };
    let stride_q = (global_heads * config.head_dim) as i32;
    let stride_o = (global_heads as i32) * d_v;
    let stride_indices = flash.topk_unified as i32;
    let stride_lse = global_heads as i32;
    {
        crate::profile::profile_op(ctx, "flashmla_fwd", None, 1, || {
            // SAFETY: ptrs from live device allocations sized to the dims passed.
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
                    d_qk,
                    d_v,
                    (flash.sw_blocks + flash.comp_blocks) as i32,
                    64,
                    stride_indices,
                    flash.num_sm_parts,
                    model_type_int,
                    sm_scale,
                    stride_q,
                    stride_q,
                    d_qk,
                    stride_kv_block_bytes,
                    bytes_per_token,
                    stride_indices,
                    stride_indices,
                    stride_lse,
                    1,
                    stride_o,
                    stride_o,
                    d_v,
                    global_heads as i32,
                    global_heads as i32,
                    stride_o,
                    stride_o,
                    d_v,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("DSv4 FlashMLA sparse decode failed: {e}"))?;
            }
            Ok(())
        })?;
    }

    // The FlashMLA output is [heads, d_v]: d_v == head_dim for MODEL1, but the
    // 512-wide latent for V32. Slice / un-rotate on the OUTPUT width (d_v).
    let out_head_dim = d_v as usize;
    if tp_world > 1 {
        let (full_out_ptr, full_out_guard) = scratch.tp_full_out.device_ptr(&ctx.stream);
        {
            crate::profile::profile_op(ctx, "flashmla_out_slice", None, 1, || {
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                unsafe {
                    ffi::dsv4_tp_out_slice_cuda(
                        full_out_ptr as *const ffi::Half,
                        out_ptr as *mut ffi::Half,
                        1,
                        (global_heads * out_head_dim) as i32,
                        (local_heads * out_head_dim) as i32,
                        (tp_rank * local_heads * out_head_dim) as i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()
                    .map_err(|e| anyhow!("DSv4 FlashMLA TP out slice failed: {e}"))?;
                }
                Ok(())
            })?;
        }
        drop(full_out_guard);
    }

    // Output inverse-RoPE un-rotates the rope tail of the MODEL1 absorbed
    // output [heads, 512]. V32's FlashMLA output is the pure kv_lora latent
    // [heads, 512] (NoPE only — the 64 rope dims live in q/k for scoring, not
    // in the value-side latent), so there is NO output rope tail to un-rotate.
    // V32's value side is reconstructed by the w_vc absorption (D3d) downstream.
    // ponytail: pod-verify V32 skips output inverse-RoPE (512 latent is pure NoPE)
    if !is_v32 {
        crate::profile::profile_op(ctx, "flashmla_inverse_rope", None, 1, || {
            // SAFETY: ptrs from live device allocations sized to the dims passed.
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
            Ok(())
        })?;
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
    Ok(())
}

/// Canonical batched KV pack for the MODEL1 batched decode lane (#60 op "c"):
/// the per-step ∝n pack work — ONE batched SW one-token pack + ONE batched
/// compressed-delta pack over all N rows — replacing the N `flashmla_decode_pack_row`
/// launches' 2×N per-step packs (the SW-ring BOOTSTRAP stays per-row; it fires
/// once per slot lifetime, off the per-step hot path). All per-slot inputs are
/// pre-gathered device-pointer arrays this step: `nope_arr[r]`/`rope_arr[r]` =
/// row r's `k_prepared` NoPE / RoPE base (rope offset computed host-side, matching
/// `flashmla_pack_one_sw_token`); `compressed_arr[r]` = row r's compressor
/// `compressed` base (0/null for SparseIndexed / no-compressor rows → kernel
/// no-op, == per-row None); `page_table_arr[r]` = row r's per-slot device page
/// table (each `total_blocks` long). `start_pos` is the contiguous `[N]` decode
/// positions (== each slot's `start_pos_device` scalar, byte-identical). `pool_ptr`
/// is the single shared pool base (uniform; rows write disjoint bands via their
/// page tables). `sw_blocks`/`compress_ratio`/`num_logical_pages` (= total_blocks)
/// are uniform per layer. n=1 == the per-row pack, byte-for-byte. MODEL1 only
/// (head_dim==512); V32/SparseIndexed callers keep [`flashmla_decode_pack_row`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn flashmla_decode_pack_batched(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    compress_ratio: usize,
    sw_blocks: usize,
    n: usize,
    pool_ptr: u64,
    nope_arr: &CudaSlice<u64>,
    rope_arr: &CudaSlice<u64>,
    compressed_arr: &CudaSlice<u64>,
    start_pos: &CudaSlice<i32>,
    page_table_arr: &CudaSlice<u64>,
    num_logical_pages: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    {
        crate::profile::profile_op(ctx, "flashmla_pack_one_batched", None, n, || {
            flash_kv::dsv4_fp8_kv_pack_strided_batched_raw(
                ctx,
                nope_arr,
                rope_arr,
                pool_ptr,
                start_pos,
                n,
                64,
                config.sliding_window,
                config.head_dim,
                config.head_dim,
                page_table_arr,
                num_logical_pages,
            )
        })?;
    }
    // compress_ratio==0 layers (DSv4-Flash 0/1/last) have NO compressor — skip the
    // completed-compressor pack. The batched FFI rejects ratio<=0 with INVALID_VALUE;
    // the single-row path skips it too (passes compressed=None for these layers).
    if compress_ratio > 0 {
        crate::profile::profile_op(ctx, "flashmla_pack_compressed_batched", None, n, || {
            flash_kv::dsv4_fp8_kv_pack_completed_compressor_row_batched_raw(
                ctx,
                compressed_arr,
                pool_ptr,
                start_pos,
                n,
                compress_ratio,
                sw_blocks,
                64,
                config.head_dim,
                page_table_arr,
                num_logical_pages,
            )
        })?;
    }
    Ok(())
}

/// Host-side rope-base offset for a row's `k_prepared` NoPE pointer, matching
/// `flashmla_pack_one_sw_token` (rope = nope + (head_dim - qk_rope_head_dim)*2 B).
pub(crate) fn flashmla_pack_rope_offset_bytes(config: &DeepSeekV4Config) -> u64 {
    (config.head_dim - config.qk_rope_head_dim) as u64 * 2
}

/// PHASE B (#60) per-row KV pack for the batched decode lane: the EXACT pack
/// sequence the single-row [`flashmla_decode_attention`] runs before the
/// fwd (SW ring bootstrap → one-token SW pack → compressed-delta), writing this
/// row's slot KV into the shared pool. Run once per row in the batched lane's
/// pack loop; the ONE batched fwd then reads the filled pool. SW passes
/// `compressed = None`; HCA passes `Some(&state.compressor.compressed)`. CSA is
/// NOT routed here (stays on the per-row `mla_attention` path — see the lane).
#[allow(clippy::too_many_arguments)]
pub(crate) fn flashmla_decode_pack_row(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    compress_ratio: usize,
    flash: &mut Dsv4FlashMlaDecodeState,
    scratch: &mut Dsv4FlashMlaDecodeScratch,
    pool: &mut Dsv4LayerKvLayout,
    sw_window_cache: &CudaSlice<half::bf16>,
    k_prepared: &HiddenStates,
    compressed: Option<&HiddenStates>,
    start_pos_device: &CudaSlice<i32>,
) -> Result<()> {
    {
        crate::profile::profile_op(ctx, "flashmla_pack_sw_ring_batched", None, 1, || {
            flashmla_pack_sw_ring(ctx, flash, scratch, pool, sw_window_cache, config)
        })?;
    }
    {
        crate::profile::profile_op(ctx, "flashmla_pack_one_batched", None, 1, || {
            flashmla_pack_one_sw_token(
                ctx,
                flash,
                scratch,
                pool,
                k_prepared,
                start_pos_device,
                config,
            )
        })?;
    }
    {
        crate::profile::profile_op(ctx, "flashmla_pack_compressed_batched", None, 1, || {
            flashmla_pack_compressed_delta(
                ctx,
                flash,
                scratch,
                pool,
                compressed,
                start_pos_device,
                compress_ratio,
                config,
            )
        })?;
    }
    Ok(())
}

/// PHASE B (#60) per-row finish for the batched decode lane: the EXACT output
/// tail the single-row [`flashmla_decode_attention`] runs AFTER the fwd —
/// output inverse-RoPE on this row's `local_attn`, then the bf16 SW-window
/// update — minus the TP out-slice (the lane runs [`Dsv4FlashMlaDecodeBatchScratch::slice_out_row`]
/// before this) and minus the kernel (batched). `local_attn` must already hold
/// this rank's local-head output for this row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn flashmla_decode_finish_row(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    sw_window_cache: &mut CudaSlice<half::bf16>,
    k_prepared: &HiddenStates,
    local_attn: &mut HiddenStates,
    start_pos: usize,
    start_pos_device: &CudaSlice<i32>,
    local_heads: usize,
    rope_base: f32,
    original_seq_len: i32,
    rope_factor: f32,
    rope_beta_fast: f32,
    rope_beta_slow: f32,
) -> Result<()> {
    let (out_ptr, out_guard) = local_attn.data.device_ptr_mut(&ctx.stream);
    let (start_ptr, start_guard) = start_pos_device.device_ptr(&ctx.stream);
    {
        crate::profile::profile_op(ctx, "flashmla_inverse_rope_batched", None, 1, || {
            // SAFETY: identical args to the single-row inverse-rope; out is one
            // local-head row (token_count=1), start_pos_device is this row's pos.
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
                .map_err(|e| anyhow!("DSv4 batched FlashMLA output inverse-rope failed: {e}"))?;
            }
            Ok(())
        })?;
    }
    drop(out_guard);
    drop(start_guard);
    update_bf16_sw_window(
        ctx,
        sw_window_cache,
        k_prepared,
        start_pos,
        Some(start_pos_device),
        config,
    )?;
    Ok(())
}

/// Canonical batched FINISH tail for the MODEL1 batched decode lane:
/// ONE batched output inverse-RoPE over N rows' (non-contiguous) `local_attn`
/// buffers, replacing the N per-row `arle_dsv4_output_inverse_rope_*` launches in
/// [`flashmla_decode_finish_row`]. `out_ptrs` are this step's per-row `local_attn`
/// device pointers (gathered host-side, all `[local_width,1]`); `start_pos` is the
/// contiguous `[N]` decode-position array (each row's abs pos). `local_heads`,
/// `rope_base`, `original_seq_len`, and the rope factor/beta are uniform across the
/// layer's rows. MUST run BEFORE the per-row O-LoRA (which consumes `local_attn`).
/// Byte-identical to N per-row inverse-RoPE calls (same kernel math, one block per
/// (row,head)).
#[allow(clippy::too_many_arguments)]
pub(crate) fn flashmla_decode_inverse_rope_batched(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    out_ptrs: &CudaSlice<u64>,
    start_pos: &CudaSlice<i32>,
    n: usize,
    local_heads: usize,
    rope_base: f32,
    original_seq_len: i32,
    rope_factor: f32,
    rope_beta_fast: f32,
    rope_beta_slow: f32,
) -> Result<()> {
    crate::profile::profile_op(ctx, "flashmla_inverse_rope_batched_ptr", None, n, || {
        let (out_ptr, og) = out_ptrs.device_ptr(&ctx.stream);
        let (start_ptr, sg) = start_pos.device_ptr(&ctx.stream);
        // SAFETY: out_ptrs holds N valid `[local_width,1]` device pointers; start_pos
        // is `[N]`; the kernel grids N*local_heads blocks and indexes both per row.
        unsafe {
            ffi::arle_dsv4_output_inverse_rope_batched_ptr_cuda(
                out_ptr as *const *mut ffi::Half,
                n as i32,
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
            .map_err(|e| anyhow!("DSv4 batched FlashMLA output inverse-rope (ptr) failed: {e}"))?;
        }
        drop(og);
        drop(sg);
        Ok(())
    })?;
    Ok(())
}

/// Canonical batched FINISH tail for the MODEL1 batched decode lane:
/// ONE batched SW-window write over N rows' (non-contiguous) k_prepared / SW ring
/// cache buffers, replacing the N per-row `dsv4_update_window_cache_*` launches in
/// [`flashmla_decode_finish_row`]. `k_ptrs` / `cache_ptrs` are this step's per-row
/// `k_prepared[head_dim,1]` and SW ring device pointers (gathered host-side);
/// `start_pos` is the contiguous `[N]` decode-position array. Each row writes its
/// single new key into its own ring at slot `start_pos[r] % sliding_window`.
/// Byte-identical to N per-row SW writes.
pub(crate) fn flashmla_decode_sw_window_batched(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    k_ptrs: &CudaSlice<u64>,
    cache_ptrs: &CudaSlice<u64>,
    start_pos: &CudaSlice<i32>,
    n: usize,
) -> Result<()> {
    crate::profile::profile_op(ctx, "flashmla_sw_window_batched_ptr", None, n, || {
        let (k_ptr, kg) = k_ptrs.device_ptr(&ctx.stream);
        let (cache_ptr, cg) = cache_ptrs.device_ptr(&ctx.stream);
        let (start_ptr, sg) = start_pos.device_ptr(&ctx.stream);
        // SAFETY: k_ptrs/cache_ptrs hold N valid device pointers; start_pos is `[N]`;
        // the kernel grids N*head_dim threads and indexes per row.
        unsafe {
            ffi::dsv4_update_window_cache_batched_ptr_cuda(
                k_ptr as *const *const ffi::Half,
                cache_ptr as *const *mut ffi::Half,
                n as i32,
                start_ptr as *const i32,
                config.sliding_window as i32,
                config.head_dim as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|e| anyhow!("DSv4 batched FlashMLA SW window write (ptr) failed: {e}"))?;
        }
        drop(kg);
        drop(cg);
        drop(sg);
        Ok(())
    })?;
    Ok(())
}

/// RMSNorm a `HiddenStates` in place into a fresh buffer (the MLA Q/KV LoRA
/// norms `q_norm` / `kv_norm`). Thin wrapper over the shared batched RMSNorm.
pub(crate) fn mla_rms_norm(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
) -> Result<HiddenStates> {
    // SAFETY: rms_norm_batched_cuda writes the full output buffer.
    let mut out = unsafe { HiddenStates::uninit(ctx, x.hidden_dim, x.seq_len)? };
    mla_rms_norm_into(ctx, x, weight, eps, &mut out)?;
    Ok(out)
}

fn mla_rms_norm_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        out.hidden_dim == x.hidden_dim && out.seq_len == x.seq_len,
        "DSv4 MLA RMSNorm out {}x{} != input {}x{}",
        out.hidden_dim,
        out.seq_len,
        x.hidden_dim,
        x.seq_len
    );
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
    Ok(())
}

fn mla_rms_norm_decode_slice_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    offset: usize,
    width: usize,
    weight: &DeviceVec,
    eps: f32,
    out: &mut HiddenStates,
) -> Result<()> {
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
    ensure!(
        out.hidden_dim == width && out.seq_len == 1,
        "DSv4 fused wqkv slice RMSNorm out {}x{} != {}x1",
        out.hidden_dim,
        out.seq_len,
        width
    );
    {
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
        // SAFETY: ensure! above bounds offset+width within x.
        let x_ptr = unsafe { (x_ptr as *const ffi::Half).add(offset) };
        // SAFETY: ptrs from live device allocations sized to the dims passed.
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
    Ok(())
}

fn run_fused_wqkv_decode(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    attention: &Dsv4Attention,
    hidden: &HiddenStates,
    scratch: &mut Dsv4FusedWqkvDecodeScratch,
) -> Result<(HiddenStates, HiddenStates, HiddenStates)> {
    // SAFETY: uninit device scratch; fully written before first read.
    let mut c_q_normed = unsafe { HiddenStates::uninit(ctx, scratch.q_lora_rank, 1)? };
    // SAFETY: uninit device scratch; fully written before first read.
    let mut q_raw = unsafe { HiddenStates::uninit(ctx, attention.wq_b.rows, 1)? };
    // SAFETY: uninit device scratch; fully written before first read.
    let mut kv_normed = unsafe { HiddenStates::uninit(ctx, scratch.head_dim, 1)? };
    run_fused_wqkv_decode_into(
        ctx,
        config,
        attention,
        hidden,
        scratch,
        &mut c_q_normed,
        &mut q_raw,
        &mut kv_normed,
    )?;
    Ok((c_q_normed, q_raw, kv_normed))
}

#[allow(clippy::too_many_arguments)]
fn run_fused_wqkv_decode_into(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    attention: &Dsv4Attention,
    hidden: &HiddenStates,
    scratch: &mut Dsv4FusedWqkvDecodeScratch,
    c_q_normed: &mut HiddenStates,
    q_raw: &mut HiddenStates,
    kv_normed: &mut HiddenStates,
) -> Result<()> {
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
    ensure!(
        c_q_normed.hidden_dim == scratch.q_lora_rank && c_q_normed.seq_len == 1,
        "DSv4 fused wqkv c_q_normed scratch {}x{} != {}x1",
        c_q_normed.hidden_dim,
        c_q_normed.seq_len,
        scratch.q_lora_rank
    );
    ensure!(
        q_raw.hidden_dim == attention.wq_b.rows && q_raw.seq_len == 1,
        "DSv4 fused wqkv q_raw scratch {}x{} != {}x1",
        q_raw.hidden_dim,
        q_raw.seq_len,
        attention.wq_b.rows
    );
    ensure!(
        kv_normed.hidden_dim == scratch.head_dim && kv_normed.seq_len == 1,
        "DSv4 fused wqkv kv_normed scratch {}x{} != {}x1",
        kv_normed.hidden_dim,
        kv_normed.seq_len,
        scratch.head_dim
    );
    let stream = ctx.stream.cu_stream();
    // SAFETY: ptrs from live device allocations sized to the dims passed.
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
    mla_rms_norm_decode_slice_into(
        ctx,
        &scratch.qkv_raw,
        0,
        scratch.q_lora_rank,
        &attention.q_norm,
        config.rms_norm_eps,
        c_q_normed,
    )?;
    mla_rms_norm_decode_slice_into(
        ctx,
        &scratch.qkv_raw,
        scratch.q_lora_rank,
        scratch.head_dim,
        &attention.kv_norm,
        config.rms_norm_eps,
        kv_normed,
    )?;

    crate::profile::profile_op(ctx, "linear/wq_b", None, 1, || {
        match (
            dsv4_decode_proj_deepgemm_enabled(),
            attention.wq_b_deepgemm.as_ref(),
        ) {
            (true, Some(cache)) => {
                // Lever #1: wq_b (M=1) through tensor-core DeepGEMM instead of the
                // scalar GEMV. Quantize c_q_normed (K=q_lora_rank) into the fused
                // scratch FP8 buffer (already consumed by the wq_a|wkv GEMM above, so
                // safe to reuse on this stream), then DeepGEMM dense GEMM.
                let k = scratch.q_lora_rank;
                ensure!(
                    cache.cols == k && cache.rows == attention.wq_b.rows,
                    "DSv4 wq_b DeepGEMM cache shape {}x{} != expected {}x{}",
                    cache.rows,
                    cache.cols,
                    attention.wq_b.rows,
                    k
                );
                crate::linear_profile::profile(ctx, "dsv4/linear/wq_b", || -> Result<()> {
                    let stream = ctx.stream.cu_stream();
                    // SAFETY: all buffers live on ctx.stream; K=q_lora_rank ≤ hidden_dim
                    // so the fused scratch (sized for hidden_dim) covers the FP8 +
                    // scale extents.
                    unsafe {
                        cuda_moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
                            cache_ptr(&c_q_normed.data, ctx),
                            cache_ptr(&scratch.input_fp8, ctx),
                            cache_ptr(&scratch.input_scales, ctx),
                            cache_ptr(&scratch.active_experts, ctx),
                            cache_ptr(&scratch.active_offsets, ctx),
                            cache_ptr(&scratch.active_counts, ctx),
                            1,
                            scratch.max_m,
                            k,
                            scratch.scale_stride_m,
                            stream,
                        )
                        .map_err(|e| anyhow!("DSv4 wq_b activation quantize failed: {e}"))?;
                        cuda_moe::dsv4_deepgemm_fp8_gemm_nt(
                            cache_ptr(&scratch.input_fp8, ctx),
                            cache_ptr(&scratch.input_scales, ctx),
                            cache_ptr(&cache.weight, ctx),
                            cache_ptr(&cache.scales, ctx),
                            cache_ptr(&q_raw.data, ctx),
                            1,
                            cache.rows,
                            cache.cols,
                            scratch.scale_stride_m,
                            stream,
                        )
                        .map_err(|e| anyhow!("DSv4 wq_b DeepGEMM dense failed: {e}"))?;
                    }
                    Ok(())
                })?;
            }
            _ => {
                crate::linear_profile::profile(ctx, "dsv4/linear/wq_b", || {
                    dsv4_linear(ctx, &attention.wq_b, c_q_normed, &mut *q_raw)
                })?;
            }
        }
        Ok(())
    })?;
    Ok(())
}

/// Prepared-but-not-attended MLA state, the boundary between `mla_attention`'s
/// per-row PREPARE half (wq/wkv proj + RoPE + — for CSA/HCA — compressor /
/// indexer / `selected`) and its FWD half (the FlashMLA attention kernel +
/// O-LoRA). The split (#60 Phase B) lets the batched decode lane run PREPARE
/// per row, gather each row's `q_prepared` into one batched Q buffer, then
/// issue ONE `sparse_decode_fwd(b=N)` — replacing the per-row FlashMLA launch
/// loop. Single-row callers go through `mla_attention`, which runs
/// `mla_attention_prepare` then `mla_attention_fwd` back-to-back.
///
/// `selected` is the CSA per-row top-k (owned, `[max_compressed_keys]`); `None`
/// for SW/HCA. The compressed-key pool is re-borrowed from `state` inside the
/// fwd (not stored here, to avoid a self-borrow), so PREPARE must leave
/// `state.compressor` populated for CSA/HCA.
pub(crate) struct Dsv4MlaPrepared {
    pub(crate) q_prepared: HiddenStates,
    pub(crate) k_prepared: HiddenStates,
    /// Attention output scratch `[local_width, token_count]`, written by the fwd.
    pub(crate) local_attn: HiddenStates,
    pub(crate) selected: Option<CudaSlice<i32>>,
    pub(crate) local_heads: usize,
    token_count: usize,
    pub(crate) sm_scale: f32,
    pub(crate) rope_base: f32,
    pub(crate) original_seq_len: i32,
}

pub(crate) struct Dsv4MlaDecodeGraphScratch {
    c_q: HiddenStates,
    c_q_normed: HiddenStates,
    q_raw: HiddenStates,
    kv_raw: HiddenStates,
    kv_normed: HiddenStates,
    q_prepared: HiddenStates,
    k_prepared: HiddenStates,
    local_attn: HiddenStates,
    oproj_latent: HiddenStates,
    compressor_main_kv: Option<HiddenStates>,
    compressor_main_score: Option<HiddenStates>,
    compressor_index_kv: Option<HiddenStates>,
    compressor_index_score: Option<HiddenStates>,
    csa_q_i: Option<HiddenStates>,
    csa_weights: Option<HiddenStates>,
    csa_selected: Option<CudaSlice<i32>>,
    // Persistent n=1 device-meta inputs for the graph-safe CSA READ (the batched
    // device-meta select reads these instead of per-step `upload_i32`, which the
    // capture audit rejects). Both are GRAPH-LIFETIME CONSTANTS: `csa_slot_id_dev`
    // is the slot's index, `csa_key_count_dev` the indexer compressed capacity.
    // Lazy-initialized on the first eager/warm run (H2D outside capture) and never
    // rewritten — guarded by `csa_meta_initialized`.
    csa_slot_id_dev: Option<CudaSlice<i32>>,
    csa_key_count_dev: Option<CudaSlice<i32>>,
    csa_meta_initialized: bool,
}

impl Dsv4MlaDecodeGraphScratch {
    pub(crate) fn device_bytes_for(
        config: &DeepSeekV4Config,
        attention: &Dsv4Attention,
        mode: DeepSeekV4AttentionMode,
    ) -> usize {
        if attention.w_kc.is_some() || attention.w_vc.is_some() || attention.o_proj.is_some() {
            return 0;
        }
        let bf16 = std::mem::size_of::<half::bf16>();
        let local_width = attention.wq_b.rows;
        let mut elems = 0usize;
        elems = elems
            .saturating_add(attention.wq_a.rows) // c_q
            .saturating_add(attention.wq_a.rows) // c_q_normed
            .saturating_add(local_width) // q_raw
            .saturating_add(config.head_dim) // kv_raw
            .saturating_add(config.head_dim) // kv_normed
            .saturating_add(local_width) // q_prepared
            .saturating_add(config.head_dim) // k_prepared
            .saturating_add(local_width) // local_attn
            .saturating_add(attention.wo_a.as_ref().expect("DSv4 wo_a").rows); // oproj_latent
        if mode.has_compressor() {
            let compressor = attention
                .compressor
                .as_ref()
                .expect("DSv4 compressor scratch requires compressor weights");
            elems = elems
                .saturating_add(compressor.wkv.rows)
                .saturating_add(compressor.wgate.rows);
        }
        if mode == DeepSeekV4AttentionMode::CompressedSparse {
            let indexer = attention
                .indexer
                .as_ref()
                .expect("DSv4 CSA scratch requires indexer weights");
            let compressor = indexer
                .compressor
                .as_ref()
                .expect("DSv4 CSA scratch requires indexer compressor");
            elems = elems
                .saturating_add(compressor.wkv.rows)
                .saturating_add(compressor.wgate.rows)
                .saturating_add(indexer.wq_b.rows)
                .saturating_add(indexer.weights_proj.rows);
        }
        elems.saturating_mul(bf16).saturating_add(
            if mode == DeepSeekV4AttentionMode::CompressedSparse {
                config.index_topk.saturating_mul(std::mem::size_of::<i32>())
            } else {
                0
            },
        )
    }

    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        attention: &Dsv4Attention,
        mode: DeepSeekV4AttentionMode,
        _compress_ratio: usize,
    ) -> Result<Self> {
        ensure!(
            attention.w_kc.is_none() && attention.w_vc.is_none() && attention.o_proj.is_none(),
            "DSv4 decode graph scratch is MODEL1-only; GLM/plain-o attention must use eager decode"
        );
        let local_width = attention.wq_b.rows;
        let oproj_rows = attention.wo_a.as_ref().expect("DSv4 wo_a").rows;
        let (compressor_main_kv, compressor_main_score) = if mode.has_compressor() {
            let compressor = attention.compressor.as_ref().ok_or_else(|| {
                anyhow!("DSv4 graph scratch mode {mode:?} requires compressor weights")
            })?;
            (
                // SAFETY: uninit device scratch; fully written before first read.
                Some(unsafe { HiddenStates::uninit(ctx, compressor.wkv.rows, 1)? }),
                // SAFETY: uninit device scratch; fully written before first read.
                Some(unsafe { HiddenStates::uninit(ctx, compressor.wgate.rows, 1)? }),
            )
        } else {
            (None, None)
        };
        let (compressor_index_kv, compressor_index_score) =
            if mode == DeepSeekV4AttentionMode::CompressedSparse {
                let indexer = attention
                    .indexer
                    .as_ref()
                    .ok_or_else(|| anyhow!("DSv4 CSA graph scratch requires indexer weights"))?;
                let compressor = indexer
                    .compressor
                    .as_ref()
                    .ok_or_else(|| anyhow!("DSv4 CSA graph scratch requires indexer compressor"))?;
                (
                    // SAFETY: uninit device scratch; fully written before first read.
                    Some(unsafe { HiddenStates::uninit(ctx, compressor.wkv.rows, 1)? }),
                    // SAFETY: uninit device scratch; fully written before first read.
                    Some(unsafe { HiddenStates::uninit(ctx, compressor.wgate.rows, 1)? }),
                )
            } else {
                (None, None)
            };
        let (csa_q_i, csa_weights, csa_selected, csa_slot_id_dev, csa_key_count_dev) =
            if mode.has_indexer() {
                let indexer = attention
                    .indexer
                    .as_ref()
                    .ok_or_else(|| anyhow!("DSv4 graph scratch mode {mode:?} requires indexer"))?;
                (
                    // SAFETY: uninit device scratch; fully written before first read.
                    Some(unsafe { HiddenStates::uninit(ctx, indexer.wq_b.rows, 1)? }),
                    // SAFETY: uninit device scratch; fully written before first read.
                    Some(unsafe { HiddenStates::uninit(ctx, indexer.weights_proj.rows, 1)? }),
                    Some(
                        ctx.stream
                            .alloc_zeros::<i32>(config.index_topk)
                            .map_err(|e| {
                                anyhow!("DSv4 graph CSA selected scratch alloc failed: {e}")
                            })?,
                    ),
                    // n=1 persistent device-meta inputs (one element each).
                    Some(ctx.stream.alloc_zeros::<i32>(1).map_err(|e| {
                        anyhow!("DSv4 graph CSA slot-id scratch alloc failed: {e}")
                    })?),
                    Some(ctx.stream.alloc_zeros::<i32>(1).map_err(|e| {
                        anyhow!("DSv4 graph CSA key-count scratch alloc failed: {e}")
                    })?),
                )
            } else {
                (None, None, None, None, None)
            };
        Ok(Self {
            // SAFETY: uninit device scratch; fully written before first read.
            c_q: unsafe { HiddenStates::uninit(ctx, attention.wq_a.rows, 1)? },
            // SAFETY: uninit device scratch; fully written before first read.
            c_q_normed: unsafe { HiddenStates::uninit(ctx, attention.wq_a.rows, 1)? },
            // SAFETY: uninit device scratch; fully written before first read.
            q_raw: unsafe { HiddenStates::uninit(ctx, local_width, 1)? },
            // SAFETY: uninit device scratch; fully written before first read.
            kv_raw: unsafe { HiddenStates::uninit(ctx, config.head_dim, 1)? },
            // SAFETY: uninit device scratch; fully written before first read.
            kv_normed: unsafe { HiddenStates::uninit(ctx, config.head_dim, 1)? },
            // SAFETY: uninit device scratch; fully written before first read.
            q_prepared: unsafe { HiddenStates::uninit(ctx, local_width, 1)? },
            // SAFETY: uninit device scratch; fully written before first read.
            k_prepared: unsafe { HiddenStates::uninit(ctx, config.head_dim, 1)? },
            // SAFETY: uninit device scratch; fully written before first read.
            local_attn: unsafe { HiddenStates::uninit(ctx, local_width, 1)? },
            // SAFETY: uninit device scratch; fully written before first read.
            oproj_latent: unsafe { HiddenStates::uninit(ctx, oproj_rows, 1)? },
            compressor_main_kv,
            compressor_main_score,
            compressor_index_kv,
            compressor_index_score,
            csa_q_i,
            csa_weights,
            csa_selected,
            csa_slot_id_dev,
            csa_key_count_dev,
            csa_meta_initialized: false,
        })
    }

    pub(crate) fn device_bytes(&self) -> usize {
        self.c_q
            .device_bytes()
            .saturating_add(self.c_q_normed.device_bytes())
            .saturating_add(self.q_raw.device_bytes())
            .saturating_add(self.kv_raw.device_bytes())
            .saturating_add(self.kv_normed.device_bytes())
            .saturating_add(self.q_prepared.device_bytes())
            .saturating_add(self.k_prepared.device_bytes())
            .saturating_add(self.local_attn.device_bytes())
            .saturating_add(self.oproj_latent.device_bytes())
            .saturating_add(
                self.compressor_main_kv
                    .as_ref()
                    .map_or(0, HiddenStates::device_bytes),
            )
            .saturating_add(
                self.compressor_main_score
                    .as_ref()
                    .map_or(0, HiddenStates::device_bytes),
            )
            .saturating_add(
                self.compressor_index_kv
                    .as_ref()
                    .map_or(0, HiddenStates::device_bytes),
            )
            .saturating_add(
                self.compressor_index_score
                    .as_ref()
                    .map_or(0, HiddenStates::device_bytes),
            )
            .saturating_add(self.csa_q_i.as_ref().map_or(0, HiddenStates::device_bytes))
            .saturating_add(
                self.csa_weights
                    .as_ref()
                    .map_or(0, HiddenStates::device_bytes),
            )
            .saturating_add(
                self.csa_selected
                    .as_ref()
                    .map_or(0, |s| s.len().saturating_mul(std::mem::size_of::<i32>())),
            )
            .saturating_add(
                self.csa_slot_id_dev
                    .as_ref()
                    .map_or(0, |s| s.len().saturating_mul(std::mem::size_of::<i32>())),
            )
            .saturating_add(
                self.csa_key_count_dev
                    .as_ref()
                    .map_or(0, |s| s.len().saturating_mul(std::mem::size_of::<i32>())),
            )
    }
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
/// vector is loaded WHOLE on every rank (no TP slice), so the attention kernel
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
    pool: &mut Dsv4LayerKvLayout,
    dsa_shared: Option<&mut Dsv4DsaSharedScratch>,
    // Model-wide shared single-row FlashMLA decode scratch (#85 P3, hoisted off
    // the per-(slot,layer) state). `Some` whenever FlashMLA decode is allocated
    // (same gate as the per-slot state); consumed only on the single-row decode
    // FlashMLA path inside the fwd.
    flashmla_scratch: Option<&mut Dsv4FlashMlaDecodeScratch>,
    // Model-wide shared FP8 prefill DeepGEMM linear scratch (hoisted off the
    // per-slot state). `Some` only when native DeepGEMM is available and the caller
    // threads it. `None` on the decode (token_count==1) graph lane.
    mut prefill_shared: Option<&mut Dsv4PrefillDeepGemmLinearScratch>,
    // Shared FP32 probe scratch — contract on `compressor_forward`'s param.
    fp32_scratch: Option<&mut Dsv4CompressorFp32Scratch>,
    start_pos: usize,
    start_pos_device: Option<&CudaSlice<i32>>,
    chain_verify: Option<&Dsv4ChainVerifyAttnMeta>,
    tp: &TpRuntime,
    out: &mut HiddenStates,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<()> {
    // Single-row + chunked-prefill callers: PREPARE then FWD back-to-back, in
    // the exact original order — byte-identical to the pre-split body. Only the
    // batched decode lane (#60) calls the two halves separately. The shared
    // prefill scratch reborrows across PREPARE (consumed first) into FWD.
    let prepared = mla_attention_prepare(
        ctx,
        config,
        attention,
        mode,
        compress_ratio,
        layer_idx,
        hidden,
        state,
        pool,
        dsa_shared,
        prefill_shared.as_deref_mut(),
        fp32_scratch,
        start_pos,
        start_pos_device,
        chain_verify,
        tp,
        keepalive,
    )?;
    mla_attention_fwd(
        ctx,
        config,
        attention,
        mode,
        compress_ratio,
        layer_idx,
        state,
        pool,
        flashmla_scratch,
        prefill_shared,
        start_pos,
        start_pos_device,
        chain_verify,
        tp,
        prepared,
        out,
        keepalive,
    )
}

#[allow(clippy::too_many_arguments)]
fn compressor_forward_decode_graph(
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
    rope_original_seq_len: i32,
    kv_scratch: &mut HiddenStates,
    score_scratch: &mut HiddenStates,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<()> {
    let width = dsv4_compressor_width(head_dim, overlap);
    ensure!(
        kv_scratch.hidden_dim == width
            && kv_scratch.seq_len == 1
            && score_scratch.hidden_dim == width
            && score_scratch.seq_len == 1,
        "DSv4 graph compressor scratch mismatch: kv={}x{} score={}x{} expected {width}x1",
        kv_scratch.hidden_dim,
        kv_scratch.seq_len,
        score_scratch.hidden_dim,
        score_scratch.seq_len
    );
    crate::profile::profile_op(ctx, "linear/compressor_wkv", None, 1, || {
        crate::linear_profile::profile(ctx, "dsv4/linear/compressor_wkv", || {
            dsv4_linear(ctx, &compressor.wkv, hidden, kv_scratch)
        })
    })?;
    crate::profile::profile_op(ctx, "linear/compressor_wgate", None, 1, || {
        crate::linear_profile::profile(ctx, "dsv4/linear/compressor_wgate", || {
            dsv4_linear(ctx, &compressor.wgate, hidden, score_scratch)
        })
    })?;
    compressor_forward(
        ctx,
        config,
        compressor,
        hidden,
        state,
        head_dim,
        ratio,
        overlap,
        start_pos,
        start_pos_device,
        apply_rope,
        rope_original_seq_len,
        None,
        Some((&*kv_scratch, &*score_scratch)),
        None,
        keepalive,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn mla_attention_decode_graph(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    attention: &Dsv4Attention,
    mode: DeepSeekV4AttentionMode,
    compress_ratio: usize,
    layer_idx: usize,
    hidden: &HiddenStates,
    state: &mut Dsv4LayerAttentionState,
    pool: &mut Dsv4LayerKvLayout,
    dsa_shared: Option<&mut Dsv4DsaSharedScratch>,
    flashmla_scratch: Option<&mut Dsv4FlashMlaDecodeScratch>,
    start_pos: usize,
    start_pos_device: Option<&CudaSlice<i32>>,
    tp: &TpRuntime,
    scratch: &mut Dsv4MlaDecodeGraphScratch,
    out: &mut HiddenStates,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<()> {
    ensure!(
        hidden.hidden_dim == config.hidden_size && hidden.seq_len == 1,
        "DSv4 graph MLA requires one hidden row [{}x1], got {}x{}",
        config.hidden_size,
        hidden.hidden_dim,
        hidden.seq_len
    );
    ensure!(
        attention.w_kc.is_none() && attention.w_vc.is_none() && attention.o_proj.is_none(),
        "DSv4 decode graph MLA is MODEL1-only; GLM/plain-o uses eager decode"
    );
    ensure!(
        start_pos_device.is_some(),
        "DSv4 decode graph MLA requires device start_pos"
    );
    let head_dim = config.head_dim;
    let local_width = attention.wq_b.rows;
    ensure!(
        local_width.is_multiple_of(head_dim),
        "DSv4 graph MLA local q width {local_width} is not a multiple of head_dim {head_dim}"
    );
    let local_heads = local_width / head_dim;
    ensure!(local_heads > 0, "DSv4 graph MLA requires local heads");
    let tp_rank = tp.config().rank;
    let sink_offset = tp_rank * local_heads;
    ensure!(
        attention.wkv.rows == head_dim,
        "DSv4 graph MLA wkv rows {} != head_dim {head_dim}",
        attention.wkv.rows
    );
    ensure!(
        config.sliding_window > 0,
        "DSv4 graph MLA requires a non-zero sliding_window"
    );
    ensure!(
        config.qk_rope_head_dim <= head_dim,
        "DSv4 graph MLA rope dim {} exceeds head_dim {head_dim}",
        config.qk_rope_head_dim
    );
    ensure!(
        state.sw_window_cache.len() == config.sliding_window * head_dim,
        "DSv4 graph MLA SW window cache len {} != sliding_window*head_dim {}",
        state.sw_window_cache.len(),
        config.sliding_window * head_dim
    );
    ensure!(
        attention.attn_sink.as_ref().expect("DSv4 attn_sink").len >= sink_offset + local_heads,
        "DSv4 graph MLA attn_sink len {} cannot cover rank {tp_rank} heads [{sink_offset}, {})",
        attention.attn_sink.as_ref().expect("DSv4 attn_sink").len,
        sink_offset + local_heads
    );

    let rope = &config.rope_parameters;
    let (rope_base, original_seq_len) = if compress_ratio > 0 {
        let osl = i32::try_from(rope.original_max_position_embeddings).map_err(|_| {
            anyhow!(
                "DSv4 original_max_position_embeddings {} overflows i32",
                rope.original_max_position_embeddings
            )
        })?;
        (config.compress_rope_theta, osl)
    } else {
        (config.rope_theta, 0i32)
    };
    // #150: with the bf16 dequant copies present, force the scalar route so the
    // n=1 decode arithmetic (bf16 cublasLt) matches the n≥2 batched lane's.
    if attention.mla_proj_bf16.is_none() && dsv4_fused_wqkv_decode_enabled()? {
        let fused = state.fused_wqkv.as_mut().ok_or_else(|| {
            anyhow!("DSv4 fused wqkv decode requested but decode scratch was not allocated")
        })?;
        crate::profile::profile_op(ctx, "linear/wqkv_a_fused", None, 1, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wqkv_a_fused", || {
                run_fused_wqkv_decode_into(
                    ctx,
                    config,
                    attention,
                    hidden,
                    fused,
                    &mut scratch.c_q_normed,
                    &mut scratch.q_raw,
                    &mut scratch.kv_normed,
                )
            })
        })?;
    } else {
        let (wq_a, wq_b, wkv) = attention.decode_proj_weights();
        crate::profile::profile_op(ctx, "linear/wq_a", None, 1, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wq_a", || {
                dsv4_linear(ctx, wq_a, hidden, &mut scratch.c_q)
            })
        })?;
        mla_rms_norm_into(
            ctx,
            &scratch.c_q,
            &attention.q_norm,
            config.rms_norm_eps,
            &mut scratch.c_q_normed,
        )?;
        crate::profile::profile_op(ctx, "linear/wq_b", None, 1, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wq_b", || {
                dsv4_linear(ctx, wq_b, &scratch.c_q_normed, &mut scratch.q_raw)
            })
        })?;
        crate::profile::profile_op(ctx, "linear/wkv", None, 1, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wkv", || {
                dsv4_linear(ctx, wkv, hidden, &mut scratch.kv_raw)
            })
        })?;
        mla_rms_norm_into(
            ctx,
            &scratch.kv_raw,
            &attention.kv_norm,
            config.rms_norm_eps,
            &mut scratch.kv_normed,
        )?;
    }

    {
        let (q_raw_ptr, _qr) = scratch.q_raw.data.device_ptr(&ctx.stream);
        let (k_raw_ptr, _kr) = scratch.kv_normed.data.device_ptr(&ctx.stream);
        let (q_out_ptr, _qo) = scratch.q_prepared.data.device_ptr_mut(&ctx.stream);
        let (k_out_ptr, _ko) = scratch.k_prepared.data.device_ptr_mut(&ctx.stream);
        let start_pos_device = start_pos_device.expect("checked above");
        let (start_ptr, _sg) = start_pos_device.device_ptr(&ctx.stream);
        // SAFETY: ptrs from live device allocations sized to the dims passed.
        unsafe {
            ffi::dsv4_prepare_qk_start_pos_ptr_cuda(
                q_raw_ptr as *const ffi::Half,
                k_raw_ptr as *const ffi::Half,
                q_out_ptr as *mut ffi::Half,
                k_out_ptr as *mut ffi::Half,
                1,
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
        }
    }

    let mut selected_ready = false;
    if mode.has_compressor() {
        let compressor = attention.compressor.as_ref().ok_or_else(|| {
            anyhow!("DSv4 layer {layer_idx} is {mode:?} but has no compressor weights")
        })?;
        let kv = scratch
            .compressor_main_kv
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 graph main compressor kv scratch missing"))?;
        let score = scratch
            .compressor_main_score
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 graph main compressor score scratch missing"))?;
        compressor_forward_decode_graph(
            ctx,
            config,
            compressor,
            hidden,
            state.compressor.as_mut().ok_or_else(|| {
                anyhow!("DSv4 layer {layer_idx} is {mode:?} but has no compressor state")
            })?,
            head_dim,
            compress_ratio,
            compress_ratio < 16,
            start_pos,
            start_pos_device,
            true,
            original_seq_len,
            kv,
            score,
            keepalive,
        )?;
    }
    if mode.has_indexer() {
        // Graph-safe READ lane (opt-in): route the CSA READ through the n=1 batched
        // device-meta select reading persistent slot_id/key_count buffers. The cache
        // WRITE (block (a)) is still host-shape driven (not yet graph-capturable), so
        // this lane runs EAGER (the caller bypasses attn-graph capture); it makes the
        // shape-stable read reachable + needle-testable. When OFF, the legacy bail
        // keeps decode-graph from running the non-capturable per-tile CSA select.
        let csa_read_lane = dsv4_decode_graph_csa_read_enabled()?;
        if !csa_read_lane
            && matches!(
                std::env::var("ARLE_DSV4_DECODE_GRAPH").as_deref(),
                Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
            )
        {
            anyhow::bail!(
                "DSv4 decode-graph does not support the official CSA select; run without ARLE_DSV4_DECODE_GRAPH=1 (or set ARLE_DSV4_DECODE_GRAPH_CSA=1 for the eager graph-safe-read lane)"
            );
        }
        ensure!(
            mode == DeepSeekV4AttentionMode::CompressedSparse,
            "DSv4 decode graph does not support SparseIndexed/GLM indexer"
        );
        let indexer = attention.indexer.as_ref().ok_or_else(|| {
            anyhow!("DSv4 layer {layer_idx} is {mode:?} but has no indexer weights")
        })?;
        let indexer_rows_before = state
            .indexer
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 layer {layer_idx} is {mode:?} but has no indexer state"))?
            .compressed
            .seq_len;
        let indexer_compressor = indexer
            .compressor
            .as_ref()
            .expect("DSv4 CSA indexer has a key compressor");
        let kv = scratch
            .compressor_index_kv
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 graph indexer compressor kv scratch missing"))?;
        let score = scratch
            .compressor_index_score
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 graph indexer compressor score scratch missing"))?;
        {
            let indexer_state = state.indexer.as_mut().ok_or_else(|| {
                anyhow!("DSv4 layer {layer_idx} is {mode:?} but has no indexer state")
            })?;
            compressor_forward_decode_graph(
                ctx,
                config,
                indexer_compressor,
                hidden,
                indexer_state,
                config.index_head_dim,
                compress_ratio,
                true,
                start_pos,
                start_pos_device,
                true,
                i32::try_from(config.rope_parameters.original_max_position_embeddings).map_err(
                    |_| {
                        anyhow!(
                            "DSv4 official DSA original_max_position_embeddings {} overflows i32",
                            config.rope_parameters.original_max_position_embeddings
                        )
                    },
                )?,
                kv,
                score,
                keepalive,
            )?;
        }
        let indexer_rows_after = state
            .indexer
            .as_ref()
            .expect("indexer state checked above")
            .compressed
            .seq_len;
        // #146 Index-layer guard (mirrors `mla_attention_prepare`'s eager-lane
        // check): this decode-graph lane is CompressedSparse-only (asserted
        // above), so the gate is unconditional — no GLM/frozen-verify path
        // reaches here to false-fire on.
        {
            let value_rows = state
                .compressor
                .as_ref()
                .map(|s| s.compressed.seq_len)
                .unwrap_or(0);
            ensure!(
                indexer_rows_after == value_rows,
                "DSv4 CSA select boundary (decode-graph): indexer rows {indexer_rows_after} != \
                 value compressor rows {value_rows} (Shape drift — #146 guard)"
            );
        }
        let shared =
            dsa_shared.ok_or_else(|| anyhow!("DSv4 graph CSA shared DSA scratch missing"))?;
        // Read-only constants for the graph-safe read lane (slot index + key
        // capacity), pulled BEFORE the mutable csa-scratch borrows below.
        let slot_idx = state
            .dsa_official_slot_idx()
            .ok_or_else(|| anyhow!("DSv4 graph CSA official DSA state missing"))?;
        let keys_capacity = state.indexer_compressed_capacity().unwrap_or(0);
        // LAZY-INIT the persistent n=1 device-meta inputs ONCE, on an eager/warm
        // run (H2D OUTSIDE any capture). slot_id + key_count are graph-lifetime
        // constants; the capture audit forbids re-doing the H2D inside replay.
        if csa_read_lane && !scratch.csa_meta_initialized {
            let slot_id_i32 = i32::try_from(slot_idx)
                .map_err(|_| anyhow!("DSv4 graph CSA slot_idx {slot_idx} overflows i32"))?;
            let key_count_i32 = i32::try_from(keys_capacity).map_err(|_| {
                anyhow!("DSv4 graph CSA key capacity {keys_capacity} overflows i32")
            })?;
            let slot_id_dev = scratch
                .csa_slot_id_dev
                .as_mut()
                .ok_or_else(|| anyhow!("DSv4 graph CSA slot-id device buffer missing"))?;
            ctx.stream
                .memcpy_htod(&[slot_id_i32], slot_id_dev)
                .map_err(|e| anyhow!("DSv4 graph CSA slot-id H2D failed: {e}"))?;
            let key_count_dev = scratch
                .csa_key_count_dev
                .as_mut()
                .ok_or_else(|| anyhow!("DSv4 graph CSA key-count device buffer missing"))?;
            ctx.stream
                .memcpy_htod(&[key_count_i32], key_count_dev)
                .map_err(|e| anyhow!("DSv4 graph CSA key-count H2D failed: {e}"))?;
            scratch.csa_meta_initialized = true;
        }
        // Disjoint-field borrows: c_q_normed (read) + csa scratch (mut) + the
        // persistent device-meta refs.
        let Dsv4MlaDecodeGraphScratch {
            c_q_normed,
            csa_q_i,
            csa_weights,
            csa_selected,
            csa_slot_id_dev,
            csa_key_count_dev,
            ..
        } = scratch;
        let csa_q_i = csa_q_i
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 graph CSA q_i scratch missing"))?;
        let csa_weights = csa_weights
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 graph CSA weights scratch missing"))?;
        let csa_selected = csa_selected
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 graph CSA selected scratch missing"))?;
        let (slot_id_dev, key_count_dev) = if csa_read_lane {
            (csa_slot_id_dev.as_ref(), csa_key_count_dev.as_ref())
        } else {
            (None, None)
        };
        let Dsv4LayerAttentionState {
            indexer: indexer_state_ref,
            dsa_official,
            ..
        } = state;
        let index_keys = &indexer_state_ref
            .as_ref()
            .expect("indexer state checked above")
            .compressed;
        let official = dsa_official
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 graph CSA official DSA state missing"))?;
        ensure!(
            official.slot_idx == slot_idx,
            "DSv4 graph CSA slot index mismatch: official {} != staged {slot_idx}",
            official.slot_idx
        );
        csa_select_decode_graph(
            ctx,
            config,
            indexer,
            hidden,
            c_q_normed,
            index_keys,
            keys_capacity,
            // This decode-graph path is CompressedSparse-only (asserted above) →
            // full-retention compressor keys, base 0. (SparseIndexed would pass
            // start_pos; kept explicit so the base stays correct if relaxed.)
            if mode == DeepSeekV4AttentionMode::SparseIndexed {
                start_pos
            } else {
                0
            },
            official,
            shared,
            pool,
            indexer_rows_before,
            indexer_rows_after,
            start_pos,
            start_pos_device,
            compress_ratio,
            csa_q_i,
            csa_weights,
            csa_selected,
            layer_idx,
            slot_id_dev,
            key_count_dev,
            slot_idx,
            keepalive,
        )?;
        selected_ready = true;
    }

    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
    let selected = if selected_ready {
        scratch.csa_selected.as_ref()
    } else {
        None
    };
    let compressed: Option<&HiddenStates> = if mode.has_compressor() {
        Some(
            &state
                .compressor
                .as_ref()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "DSv4 layer {layer_idx} is {mode:?} but has no compressor state"
                    )
                })?
                .compressed,
        )
    } else {
        None
    };
    let flash = state.flashmla.as_mut().ok_or_else(|| {
        anyhow!("FlashMLA decode enabled but layer state has no FlashMLA arena")
    })?;
    let flash_scratch = flashmla_scratch.ok_or_else(|| {
        anyhow!("FlashMLA decode enabled but shared FlashMLA decode scratch missing")
    })?;
    flashmla_decode_attention(
        ctx,
        config,
        attention,
        mode,
        compress_ratio,
        &scratch.q_prepared,
        &scratch.k_prepared,
        selected,
        compressed,
        &mut state.sw_window_cache,
        flash,
        flash_scratch,
        pool,
        start_pos,
        start_pos_device,
        tp,
        local_heads,
        &mut scratch.local_attn,
        sm_scale,
        rope_base,
        original_seq_len,
        rope.factor,
        rope.beta_fast,
        rope.beta_slow,
    )?;

    mla_oproj_decode_graph(
        ctx,
        attention,
        state,
        &scratch.local_attn,
        &mut scratch.oproj_latent,
        out,
    )
}

/// GLM runtime Q absorption: q_raw [heads, qk_nope(192)+qk_rope(64)] → absorbed
/// q [heads, kv_lora(512)+qk_rope(64) = head_dim(576)] per head, via the per-head
/// contraction q_latent[h] = w_kc[h] · q_nope[h] (w_kc loaded in `gemm_batch`
/// orientation `[kv_lora(out), qk_nope(in)]` per head, see `load_dsv4_kv_b_absorb`
/// step 4). The q_rope(64) tail is carried through unchanged.
///
/// Input  `q_raw`: `[local_heads * qk_head_dim(256), token_count]` token-major
///         (per head: q_nope(qk_nope=192) | q_rope(qk_rope=64)).
/// Output `q_absorbed`: `[local_heads * head_dim(576), token_count]` token-major
///         (per head: q_latent(kv_lora=512) | q_rope(qk_rope=64)).
///
/// w_kc `DeviceMatrix` is `[local_heads*kv_lora, qk_nope]`; head `h`'s block is
/// rows `[h*kv_lora, (h+1)*kv_lora)` = `[kv_lora, qk_nope]`. Per head we run the
/// dense bf16 `gemm_cuda` (`weight[R=kv_lora, C=qk_nope] · x[qk_nope, tok]`).
///
/// DECODE (token_count==1): each head's q_nope rows are contiguous (one token),
/// so the per-head GEMM + the q_rope copy are exact. This is the V32/GLM decode
/// hot path and is fully wired.
///
/// PREFILL (token_count>1): token-major layout makes a single head's rows
/// strided across tokens (`stride = qk_head_dim`), so a per-head GEMM needs a
/// gather/scatter or a batched-head kernel that ARLE doesn't yet expose. Rather
/// than serve a wrong projection, this bails loudly for token_count>1.
/// ponytail: pod-verify GLM prefill Q absorption — wire a batched-head bf16 GEMM
/// (or per-token-per-head gather) for token_count>1; decode (==1) is exact.
/// ponytail: pod-verify w_kc per-head contraction q_latent = w_kc · q_nope (decode)
fn glm_absorb_q(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    w_kc: &DeviceMatrix,
    q_raw: &HiddenStates,
    local_heads: usize,
    token_count: usize,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<HiddenStates> {
    let qk_nope = config.qk_nope_head_dim;
    let qk_rope = config.qk_rope_head_dim;
    let kv_lora = config.kv_lora_rank;
    let qk_head = qk_nope + qk_rope; // 256
    let head_dim = kv_lora + qk_rope; // 576
    ensure!(
        w_kc.rows == local_heads * kv_lora && w_kc.cols == qk_nope,
        "GLM w_kc shape {}x{} != [heads*kv_lora={}, qk_nope={}]",
        w_kc.rows,
        w_kc.cols,
        local_heads * kv_lora,
        qk_nope
    );
    ensure!(
        q_raw.hidden_dim == local_heads * qk_head && q_raw.seq_len == token_count,
        "GLM q absorb input {}x{} != [heads*qk_head={}, tok={}]",
        q_raw.hidden_dim,
        q_raw.seq_len,
        local_heads * qk_head,
        token_count
    );
    ensure!(
        config.head_dim == head_dim,
        "GLM q absorb head_dim {} != kv_lora+qk_rope {}",
        config.head_dim,
        head_dim
    );
    if token_count != 1 {
        // See doc: token-major strided per-head rows need a batched-head kernel.
        bail!(
            "GLM runtime Q absorption (w_kc) prefill (token_count={token_count}>1) not \
             wired: needs a batched-head bf16 GEMM. Decode (token_count==1) is exact."
        );
    }
    // Raw bf16 read (per-head GEMM): quantized `.data` is a dummy — #138 OOB class.
    ensure!(
        w_kc.is_dense_bf16(),
        "GLM w_kc must be dense bf16 (raw-read per-head GEMM); got {:?}",
        w_kc.weight_format
    );
    let mut q_absorbed = HiddenStates::zeros(ctx, local_heads * head_dim, token_count)?;
    let stream = ctx.stream.cu_stream();
    // Phase 1: per-head q_nope · w_kc → q_latent (raw pointers; guards scoped here).
    {
        let (w_ptr, _gw) = w_kc.data.device_ptr(&ctx.stream);
        let (q_ptr, _gq) = q_raw.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = q_absorbed.data.device_ptr_mut(&ctx.stream);
        for h in 0..local_heads {
            // q_nope[h] = q_raw[h*qk_head .. h*qk_head+qk_nope] (token 0).
            // SAFETY: h < local_heads keeps this per-head offset in bounds.
            let q_nope_h = unsafe { (q_ptr as *const ffi::Half).add(h * qk_head) };
            // w_kc[h] block: rows [h*kv_lora, (h+1)*kv_lora), [kv_lora, qk_nope].
            // SAFETY: h < local_heads keeps this per-head offset in bounds.
            let w_h = unsafe { (w_ptr as *const ffi::Half).add(h * kv_lora * qk_nope) };
            // q_latent[h] → q_absorbed[h*head_dim .. h*head_dim+kv_lora].
            // SAFETY: h < local_heads keeps this per-head offset in bounds.
            let out_h = unsafe { (out_ptr as *mut ffi::Half).add(h * head_dim) };
            // SAFETY: per-head bf16 GEMM weight[kv_lora, qk_nope] · q_nope[qk_nope, 1].
            unsafe {
                ffi::gemm_cuda(
                    w_h,
                    q_nope_h,
                    out_h,
                    kv_lora as i32,
                    token_count as i32,
                    qk_nope as i32,
                    stream,
                )
                .result()
                .map_err(|e| anyhow!("GLM glm_absorb_q head {h} gemm failed: {e}"))?;
            }
        }
    }
    // Phase 2: copy each head's q_rope(64) tail through unchanged.
    for h in 0..local_heads {
        let rope_src = q_raw
            .data
            .slice((h * qk_head + qk_nope)..(h * qk_head + qk_head));
        let mut rope_dst = q_absorbed
            .data
            .slice_mut((h * head_dim + kv_lora)..(h * head_dim + head_dim));
        ctx.stream
            .memcpy_dtod(&rope_src, &mut rope_dst)
            .map_err(|e| anyhow!("GLM glm_absorb_q head {h} rope copy failed: {e}"))?;
    }
    keepalive.keep_hidden(&q_absorbed);
    Ok(q_absorbed)
}

/// GLM runtime V absorption: latent attn_out `[local_heads*kv_lora(512), tok]` →
/// v `[local_heads*v_head_dim(256), tok]`, per head `v[h] = w_vc[h] · attn_out[h]`.
/// w_vc loaded in `gemm_batch` orientation `[v_head(out), kv_lora(in)]` per head
/// (`load_dsv4_kv_b_absorb` step 4); head `h`'s block is rows `[h*v_head,
/// (h+1)*v_head)`. Per head we run dense bf16 `gemm_cuda` (`weight[v_head,
/// kv_lora] · x[kv_lora, tok]`). The resulting v `[heads*v_head]` feeds the plain
/// `o_proj` (D4).
///
/// DECODE (token_count==1): each head's latent rows are contiguous (one token) →
/// per-head GEMM is exact (V32/GLM decode hot path, fully wired). PREFILL
/// (token_count>1): strided per-head rows need a batched-head kernel → bails.
/// ponytail: pod-verify GLM prefill V absorption (token_count>1 batched-head GEMM)
/// ponytail: pod-verify w_vc per-head contraction v = w_vc · attn_out feeds plain o_proj
fn glm_absorb_v(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    w_vc: &DeviceMatrix,
    local_attn: &HiddenStates,
    local_heads: usize,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<HiddenStates> {
    let kv_lora = config.kv_lora_rank;
    let v_head = config.v_head_dim;
    let token_count = local_attn.seq_len;
    ensure!(
        w_vc.rows == local_heads * v_head && w_vc.cols == kv_lora,
        "GLM w_vc shape {}x{} != [heads*v_head={}, kv_lora={}]",
        w_vc.rows,
        w_vc.cols,
        local_heads * v_head,
        kv_lora
    );
    ensure!(
        local_attn.hidden_dim == local_heads * kv_lora,
        "GLM v absorb input {} != heads*kv_lora {}",
        local_attn.hidden_dim,
        local_heads * kv_lora
    );
    if token_count != 1 {
        bail!(
            "GLM runtime V absorption (w_vc) prefill (token_count={token_count}>1) not \
             wired: needs a batched-head bf16 GEMM. Decode (token_count==1) is exact."
        );
    }
    // Raw bf16 read (per-head GEMM): quantized `.data` is a dummy — #138 OOB class.
    ensure!(
        w_vc.is_dense_bf16(),
        "GLM w_vc must be dense bf16 (raw-read per-head GEMM); got {:?}",
        w_vc.weight_format
    );
    let mut v_out = HiddenStates::zeros(ctx, local_heads * v_head, token_count)?;
    let stream = ctx.stream.cu_stream();
    // Per-head attn_out · w_vc → v (raw pointers; guards scoped to drop before return).
    {
        let (w_ptr, _gw) = w_vc.data.device_ptr(&ctx.stream);
        let (a_ptr, _ga) = local_attn.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = v_out.data.device_ptr_mut(&ctx.stream);
        for h in 0..local_heads {
            // attn_out[h] = local_attn[h*kv_lora .. (h+1)*kv_lora] (token 0).
            // SAFETY: h < local_heads keeps this per-head offset in bounds.
            let a_h = unsafe { (a_ptr as *const ffi::Half).add(h * kv_lora) };
            // w_vc[h] block rows [h*v_head, (h+1)*v_head), [v_head, kv_lora].
            // SAFETY: h < local_heads keeps this per-head offset in bounds.
            let w_h = unsafe { (w_ptr as *const ffi::Half).add(h * v_head * kv_lora) };
            // v[h] → v_out[h*v_head .. (h+1)*v_head].
            // SAFETY: h < local_heads keeps this per-head offset in bounds.
            let out_h = unsafe { (out_ptr as *mut ffi::Half).add(h * v_head) };
            // SAFETY: per-head bf16 GEMM weight[v_head, kv_lora] · attn_out[kv_lora, 1].
            unsafe {
                ffi::gemm_cuda(
                    w_h,
                    a_h,
                    out_h,
                    v_head as i32,
                    token_count as i32,
                    kv_lora as i32,
                    stream,
                )
                .result()
                .map_err(|e| anyhow!("GLM glm_absorb_v head {h} gemm failed: {e}"))?;
            }
        }
    }
    keepalive.keep_hidden(&v_out);
    Ok(v_out)
}

/// PREPARE half of `mla_attention` (see [`Dsv4MlaPrepared`]). Q/KV LoRA + partial
/// RoPE, and for CSA/HCA the compressor + (CSA) indexer top-k `selected`. Leaves
/// `state.compressor` / `state.indexer` populated for the fwd's compressed-key
/// re-borrow. Pure host/proj work — no FlashMLA kernel, no pool writes — so the
/// batched lane can run it per row before the one batched fwd.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mla_attention_prepare(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    attention: &Dsv4Attention,
    mode: DeepSeekV4AttentionMode,
    compress_ratio: usize,
    layer_idx: usize,
    hidden: &HiddenStates,
    state: &mut Dsv4LayerAttentionState,
    pool: &mut Dsv4LayerKvLayout,
    dsa_shared: Option<&mut Dsv4DsaSharedScratch>,
    // Model-wide shared FP8 prefill DeepGEMM linear scratch (hoisted off the
    // per-slot state). `Some` only when native DeepGEMM is available and the caller
    // threads it (the prefill projection lanes). `None` on the decode
    // (token_count==1) graph/batched lanes, which never take a prefill branch.
    mut prefill_shared: Option<&mut Dsv4PrefillDeepGemmLinearScratch>,
    // Shared FP32 probe scratch — contract on `compressor_forward`'s param.
    mut fp32_scratch: Option<&mut Dsv4CompressorFp32Scratch>,
    start_pos: usize,
    start_pos_device: Option<&CudaSlice<i32>>,
    chain_verify: Option<&Dsv4ChainVerifyAttnMeta>,
    tp: &TpRuntime,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<Dsv4MlaPrepared> {
    ensure!(
        hidden.hidden_dim == config.hidden_size,
        "DSv4 MLA hidden dim {} != hidden_size {}",
        hidden.hidden_dim,
        config.hidden_size
    );

    let head_dim = config.head_dim;
    let token_count = hidden.seq_len;
    let local_width = attention.wq_b.rows;
    // DSv4 wq_b emits the PRE-ABSORBED q at `head_dim` (512) per head. GLM's wq_b
    // emits the standard q at `qk_nope+qk_rope` (= 256) per head, absorbed to the
    // 576 latent at runtime (`glm_absorb_q`). So derive local_heads from the wq_b
    // per-head width, which is qk_head for GLM (w_kc.is_some()) and head_dim else.
    let q_head_width = if attention.w_kc.is_some() {
        config.qk_nope_head_dim + config.qk_rope_head_dim
    } else {
        head_dim
    };
    ensure!(
        q_head_width > 0 && local_width.is_multiple_of(q_head_width),
        "DSv4 MLA local q width {local_width} is not a multiple of q_head_width {q_head_width}"
    );
    let local_heads = local_width / q_head_width;
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
    // wo_a/wo_b ↔ out shape checks live in `mla_attention_fwd` (PREPARE has no
    // `out`); the O-LoRA there owns the projection.
    // GLM pure-SparseIndexed has sliding_window==0 (no SW window — attention is
    // indexer-selected full-latent only). DSv4 modes require a non-zero window.
    // ponytail: pod-verify GLM pure-SparseIndexed (sliding_window=0) skips SW-ring entirely; attention is indexer-selected full-latent only
    ensure!(
        config.sliding_window > 0 || mode == DeepSeekV4AttentionMode::SparseIndexed,
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
        attention.attn_sink.as_ref().expect("DSv4 attn_sink").len >= sink_offset + local_heads,
        "DSv4 MLA attn_sink len {} cannot cover rank {tp_rank} heads [{sink_offset}, {})",
        attention.attn_sink.as_ref().expect("DSv4 attn_sink").len,
        sink_offset + local_heads
    );

    let rope = &config.rope_parameters;
    // RoPE base/YaRN is PER-LAYER, matching the canonical SGLang impl
    // (deepseek_v4.py:271 `rope_base = compress_rope_theta if compress_ratio else
    // rope_theta`, and fused_qk_norm_rope_swa_store which ropes Q + SW-K with ONE
    // per-layer cos/sin cache): compressed layers (CSA cr=4 / HCA cr=128) rope Q,
    // SW-K, the output inverse-rope AND the compressor with compress_rope_theta +
    // YaRN(original_max_position_embeddings); pure-SW layers (cr=0) use rope_theta
    // with no YaRN. Q MUST share the compressed-key theta or Q·compressed-K phase
    // mismatches and long context (>~80 tok) collapses to garbage. (The prior
    // "always rope_theta, no YaRN" matched the old-tree reference.rs /
    // errors/2026-05-29-dsv4-longctx-rope-conflation, not the canonical model —
    // SGLang ropes everything on a compressed layer at compress_rope_theta.)
    let (rope_base, original_seq_len) = if compress_ratio > 0 {
        let osl = i32::try_from(rope.original_max_position_embeddings).map_err(|_| {
            anyhow!(
                "DSv4 original_max_position_embeddings {} overflows i32",
                rope.original_max_position_embeddings
            )
        })?;
        (config.compress_rope_theta, osl)
    } else {
        (config.rope_theta, 0i32)
    };
    let start_pos_i32 = i32::try_from(start_pos)
        .map_err(|_| anyhow::anyhow!("DSv4 MLA start_pos {start_pos} overflows i32"))?;

    // ── 1+2. Q/KV LoRA. Decode uses the existing B=1 fused (`wq_a | wkv`)
    // path. Prefill uses the same fused weight cache when native DeepGEMM is
    // available; otherwise the scalar reference order remains intact.
    // #150: bf16 dequant copies present ⇒ decode (token_count==1) takes the
    // scalar route with the bf16 weights; prefill (token_count>1) keeps FP8.
    let decode_bf16 = token_count == 1 && attention.mla_proj_bf16.is_some();
    let fused_wqkv = token_count == 1 && !decode_bf16 && dsv4_fused_wqkv_decode_enabled()?;
    let (c_q_normed, q_raw, kv_normed) = if fused_wqkv {
        let scratch = state.fused_wqkv.as_mut().ok_or_else(|| {
            anyhow!("DSv4 fused wqkv decode requested but decode scratch was not allocated")
        })?;
        let out =
            crate::profile::profile_op(ctx, "linear/wqkv_a_fused", None, token_count, || {
                crate::linear_profile::profile(ctx, "dsv4/linear/wqkv_a_fused", || {
                    run_fused_wqkv_decode(ctx, config, attention, hidden, scratch)
                })
            })?;
        out
    } else if token_count > 1 && dsv4_fp8_linear_deepgemm_enabled()? {
        // SAFETY: uninit device scratch; fully written before first read.
        let mut c_q = unsafe { HiddenStates::uninit(ctx, attention.wq_a.rows, token_count)? };
        // SAFETY: uninit device scratch; fully written before first read.
        let mut kv_raw = unsafe { HiddenStates::uninit(ctx, head_dim, token_count)? };
        let scratch = prefill_shared.as_deref_mut().ok_or_else(|| {
            anyhow!("DSv4 fused wqkv prefill requested but prefill scratch was not allocated")
        })?;
        crate::profile::profile_op(
            ctx,
            "linear/wqkv_a_fused_prefill",
            None,
            token_count,
            || {
                crate::linear_profile::profile(ctx, "dsv4/linear/wqkv_a_fused_prefill", || {
                    run_fused_wqkv_prefill(ctx, attention, hidden, scratch, &mut c_q, &mut kv_raw)
                })
            },
        )?;
        keepalive.keep_hidden(&c_q);
        let c_q_normed = mla_rms_norm(ctx, &c_q, &attention.q_norm, config.rms_norm_eps)?;
        keepalive.keep_hidden(&c_q_normed);
        // SAFETY: uninit device scratch; fully written before first read.
        let mut q_raw = unsafe { HiddenStates::uninit(ctx, local_width, token_count)? };
        crate::profile::profile_op(ctx, "linear/wq_b", None, token_count, || {
            // Prefill wq_b → DeepGEMM (off the scalar dsv4_fp8_gemv_batch, the 62% of
            // mla_attn prefill). Reuses the prefill fused-wqkv FP8 scratch since
            // K=q_lora_rank ≤ hidden_dim. Opt-in until the prefill A/B licenses it.
            if let Some(cache) = attention
                .wq_b_deepgemm
                .as_ref()
                .filter(|_| dsv4_prefill_proj_deepgemm_enabled())
            {
                let scratch = prefill_shared.as_deref_mut().ok_or_else(|| {
                    anyhow!(
                        "DSv4 prefill wq_b DeepGEMM requested but prefill scratch not allocated"
                    )
                })?;
                crate::linear_profile::profile(ctx, "dsv4/linear/wq_b", || {
                    prefill_proj_deepgemm(ctx, scratch, cache, &c_q_normed, &mut q_raw)
                })
            } else {
                crate::linear_profile::profile(ctx, "dsv4/linear/wq_b", || {
                    dsv4_linear(ctx, &attention.wq_b, &c_q_normed, &mut q_raw)
                })
            }
        })?;
        keepalive.keep_hidden(&q_raw);

        // KV latent: wkv (down to the single compressed latent) → kv_norm.
        keepalive.keep_hidden(&kv_raw);
        let kv_normed = mla_rms_norm(ctx, &kv_raw, &attention.kv_norm, config.rms_norm_eps)?;
        keepalive.keep_hidden(&kv_normed);
        (c_q_normed, q_raw, kv_normed)
    } else {
        // Q-LoRA: wq_a (down) → q_norm RMSNorm → wq_b (up to per-head Q).
        // #150: decode with the bf16 copies present takes them (DenseBf16 ⇒
        // gemm_batch); prefill/no-copies keeps the FP8 originals.
        let (wq_a, wq_b, wkv) = if decode_bf16 {
            attention.decode_proj_weights()
        } else {
            (&attention.wq_a, &attention.wq_b, &attention.wkv)
        };
        // SAFETY: dsv4_linear writes the full c_q buffer.
        let mut c_q = unsafe { HiddenStates::uninit(ctx, wq_a.rows, token_count)? };
        crate::profile::profile_op(ctx, "linear/wq_a", None, token_count, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wq_a", || {
                dsv4_linear(ctx, wq_a, hidden, &mut c_q)
            })
        })?;
        keepalive.keep_hidden(&c_q);
        let c_q_normed = mla_rms_norm(ctx, &c_q, &attention.q_norm, config.rms_norm_eps)?;
        keepalive.keep_hidden(&c_q_normed);
        // SAFETY: dsv4_linear writes the full q_raw buffer.
        let mut q_raw = unsafe { HiddenStates::uninit(ctx, local_width, token_count)? };
        crate::profile::profile_op(ctx, "linear/wq_b", None, token_count, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wq_b", || {
                dsv4_linear(ctx, wq_b, &c_q_normed, &mut q_raw)
            })
        })?;
        keepalive.keep_hidden(&q_raw);

        // KV latent: wkv (down to the single compressed latent) → kv_norm.
        // SAFETY: dsv4_linear writes the full kv_raw buffer.
        let mut kv_raw = unsafe { HiddenStates::uninit(ctx, head_dim, token_count)? };
        crate::profile::profile_op(ctx, "linear/wkv", None, token_count, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wkv", || {
                dsv4_linear(ctx, wkv, hidden, &mut kv_raw)
            })
        })?;
        keepalive.keep_hidden(&kv_raw);
        let kv_normed = mla_rms_norm(ctx, &kv_raw, &attention.kv_norm, config.rms_norm_eps)?;
        keepalive.keep_hidden(&kv_normed);
        (c_q_normed, q_raw, kv_normed)
    };
    keepalive.keep_hidden(&c_q_normed);
    keepalive.keep_hidden(&q_raw);
    keepalive.keep_hidden(&kv_normed);

    // ── 2b. GLM runtime Q absorption (w_kc.is_some()). GLM's wq_b produces
    // q_nope(qk_nope=192) + q_rope(qk_rope=64) = qk_head_dim(256) per head; the
    // FlashMLA latent path needs the absorbed q[heads, kv_lora(512) + rope(64) =
    // head_dim(576)]. Per SGLang forward_mla.py: q_latent[h] = q_nope[h] · w_kc[h]
    // (w_kc logical [h, qk_nope=192, kv_lora=512]); reassemble [q_latent(512) |
    // q_rope(64)] per head, so the partial-RoPE + pack + FlashMLA below see the
    // 576-wide latent q. DSv4 (w_kc None) skips — q_raw is already pre-absorbed.
    let (q_raw, local_width) = if let Some(w_kc) = attention.w_kc.as_ref() {
        let q_absorbed = glm_absorb_q(
            ctx,
            config,
            w_kc,
            &q_raw,
            local_heads,
            token_count,
            keepalive,
        )?;
        let lw = q_absorbed.hidden_dim;
        (q_absorbed, lw)
    } else {
        (q_raw, local_width)
    };
    keepalive.keep_hidden(&q_raw);

    // ── 3. Partial RoPE on the trailing rope_dim cols of Q (per head) and K.
    // SAFETY: dsv4_prepare_qk_cuda writes both full output buffers.
    let mut q_prepared = unsafe { HiddenStates::uninit(ctx, local_width, token_count)? };
    // SAFETY: uninit device scratch; fully written before first read.
    let mut k_prepared = unsafe { HiddenStates::uninit(ctx, head_dim, token_count)? };
    {
        let (q_raw_ptr, _qr) = q_raw.data.device_ptr(&ctx.stream);
        let (k_raw_ptr, _kr) = kv_normed.data.device_ptr(&ctx.stream);
        let (q_out_ptr, _qo) = q_prepared.data.device_ptr_mut(&ctx.stream);
        let (k_out_ptr, _ko) = k_prepared.data.device_ptr_mut(&ctx.stream);
        // SAFETY: all buffers valid on ctx.stream; head/dim args checked above.
        unsafe {
            if let Some(meta) = chain_verify {
                let (start_ptr, _sg) = meta.positions.device_ptr(&ctx.stream);
                ffi::dsv4_prepare_qk_fused_batch_start_pos_cuda(
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
            } else if let Some(start_pos_device) = start_pos_device {
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
    // SAFETY: the FlashMLA attention kernel writes the full local_attn buffer
    // in `mla_attention_fwd`.
    let local_attn = unsafe { HiddenStates::uninit(ctx, local_width, token_count)? };

    // CSA / HCA: run the compressor (+ for CSA the indexer + top-k `selected`)
    // here in PREPARE — these are projection/select kernels that produce
    // `state.compressor` / `selected`, which the fwd consumes. SW has neither.
    //
    // Frozen chain verify is different: the verify rows are speculative and
    // must not append compressed/indexer KV. FlashMLA chain verify already packs
    // the verify rows' K directly as the current chunk, so recomputing
    // compressor/indexer key projections here only creates rows that are then
    // discarded by the frozen update guard. Keep CSA query/top-k below, but read
    // the committed compressed/indexer pools as-is.
    let skip_frozen_compressor = dsv4_verify_frozen() && chain_verify.is_some();
    // GLM SparseIndexed: full-sequence indexer, every token a key (ratio=1, no
    // compressor). CompressedSparse keeps its real compress_ratio.
    let index_ratio = if mode == DeepSeekV4AttentionMode::SparseIndexed {
        1
    } else {
        compress_ratio
    };
    let selected = if !mode.has_indexer() && !mode.has_compressor() {
        // SlidingWindow: neither compressor nor indexer.
        None
    } else {
        // ── 4b(prep). MAIN compressor (CSA/HCA only) → (indexer modes) top-k
        // select. GLM SparseIndexed has no MAIN compressor — gate it on
        // has_compressor() so GLM skips straight to the indexer block.
        let overlap = compress_ratio < 16;
        if mode.has_compressor() {
            let compressor = attention.compressor.as_ref().ok_or_else(|| {
                anyhow::anyhow!("DSv4 layer {layer_idx} is {mode:?} but has no compressor weights")
            })?;
            if skip_frozen_compressor {
                ensure!(
                    state.compressor.is_some(),
                    "DSv4 layer {layer_idx} is {mode:?} but has no compressor state"
                );
            } else {
                let compressor_state = state.compressor.as_mut().ok_or_else(|| {
                    anyhow::anyhow!(
                        "DSv4 layer {layer_idx} is {mode:?} but has no compressor state"
                    )
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
                    // YaRN on for compressed layers (matches Q/SW-K + SGLang
                    // compressor freqs_cis); original_seq_len = orig_max_pos here.
                    original_seq_len,
                    fp32_scratch.as_deref_mut(),
                    None,
                    None,
                    keepalive,
                )?;
            }
        }

        if mode.has_indexer() {
            let indexer = attention.indexer.as_ref().ok_or_else(|| {
                anyhow::anyhow!("DSv4 layer {layer_idx} is {mode:?} but has no indexer weights")
            })?;
            let use_official_dsa = dsv4_dsa_official_enabled()?;
            let indexer_rope_original_seq_len = if use_official_dsa {
                i32::try_from(config.rope_parameters.original_max_position_embeddings).map_err(
                    |_| {
                        anyhow!(
                            "DSv4 official DSA original_max_position_embeddings {} overflows i32",
                            config.rope_parameters.original_max_position_embeddings
                        )
                    },
                )?
            } else {
                0
            };
            let indexer_rows_before = state
                .indexer
                .as_ref()
                .map(|s| s.compressed.seq_len)
                .unwrap_or(0);
            // Indexer keys: CSA runs a second compressor over index_head_dim keys
            // (apply_rope=true, head_dim=index_head_dim). GLM SparseIndexed has no
            // key compressor — build one full-length index key per token (ratio=1).
            if skip_frozen_compressor {
                ensure!(
                    state.indexer.is_some(),
                    "DSv4 layer {layer_idx} is {mode:?} but has no indexer state"
                );
            } else {
                let indexer_state = state.indexer.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("DSv4 layer {layer_idx} is {mode:?} but has no indexer state")
                })?;
                if mode == DeepSeekV4AttentionMode::CompressedSparse {
                    compressor_forward(
                        ctx,
                        config,
                        indexer
                            .compressor
                            .as_ref()
                            .expect("DSv4 CSA indexer has a key compressor"),
                        hidden,
                        indexer_state,
                        config.index_head_dim,
                        compress_ratio,
                        true,
                        start_pos,
                        start_pos_device,
                        use_official_dsa,
                        indexer_rope_original_seq_len,
                        fp32_scratch,
                        None,
                        None,
                        keepalive,
                    )?;
                } else {
                    // GLM SparseIndexed: full-sequence index-key build (no
                    // compressor); every token → one key row at its abs position.
                    sparse_indexed_index_key_forward(
                        ctx,
                        config,
                        indexer,
                        hidden,
                        indexer_state,
                        start_pos,
                        keepalive,
                    )?;
                }
            }
            let indexer_rows_after = state
                .indexer
                .as_ref()
                .map(|s| s.compressed.seq_len)
                .unwrap_or(0);
            // #146 Index-layer guard: for CSA the value compressor and the indexer
            // key compressor consume the same token stream at the same start_pos
            // with the same compress_ratio, so their row counts MUST match. GLM
            // SparseIndexed runs the indexer at ratio=1 vs value ratio>1 (rows
            // differ by design), and frozen chain-verify advances neither — gate
            // to CSA's live path so this never false-fires. Turns a silent Shape
            // drift past 2048 into a loud boundary fail.
            if mode == DeepSeekV4AttentionMode::CompressedSparse && !skip_frozen_compressor {
                let value_rows = state
                    .compressor
                    .as_ref()
                    .map(|s| s.compressed.seq_len)
                    .unwrap_or(0);
                ensure!(
                    indexer_rows_after == value_rows,
                    "DSv4 CSA select boundary: indexer rows {indexer_rows_after} != \
                     value compressor rows {value_rows} (Shape drift — #146 guard)"
                );
            }
            let keys_capacity = state
                .indexer
                .as_ref()
                .map(|s| s.compressed_capacity())
                .unwrap_or(0);
            let index_keys = &state
                .indexer
                .as_ref()
                .expect("indexer state checked above")
                .compressed;
            let official = state.dsa_official.as_mut();
            csa_select(
                ctx,
                config,
                layer_idx,
                indexer,
                hidden,
                &c_q_normed,
                index_keys,
                keys_capacity,
                // Staging-ring base: SparseIndexed stages window-relative to
                // start_pos; the compressor-indexer retains full history (base 0).
                if mode == DeepSeekV4AttentionMode::SparseIndexed {
                    start_pos
                } else {
                    0
                },
                official,
                dsa_shared,
                pool,
                indexer_rows_before,
                indexer_rows_after,
                start_pos,
                start_pos_device,
                index_ratio,
                prefill_shared,
                // Prefill / single-row path: no batched gather (byte-identical).
                None,
                // Prefill / single-row path: no batched query pre-pass.
                None,
                keepalive,
            )?
        } else {
            None
        }
    };

    // TEMP #146 probe (self-gating; revert after the run): dump the LAST query
    // row's DSA top-k selection per CSA layer, on BOTH lanes — prefill sel was
    // proven healthy (round 2: needle blocks present at all 21 CSA layers), so
    // round 3 asks whether the DECODE-step selection (scored over the FP8
    // rotated key cache, a different input than prefill's bf16 staging) still
    // carries the needle blocks once C > index_topk.
    if let Some(sel) = selected.as_ref()
        && env_flag("ARLE_DSV4_DSA_TOPK_PROBE")?
    {
        ctx.sync()?;
        let host: Vec<i32> = ctx.stream.clone_dtoh(sel)?;
        let k = config.index_topk;
        let last = &host[host.len().saturating_sub(k)..];
        let line = format!(
            "[dsaprobe] pid={} layer={layer_idx} sp={start_pos} n={token_count} sel={last:?}\n",
            std::process::id()
        );
        eprint!("{line}");
    }
    Ok(Dsv4MlaPrepared {
        q_prepared,
        k_prepared,
        local_attn,
        selected,
        local_heads,
        token_count,
        sm_scale,
        rope_base,
        original_seq_len,
    })
}

/// Batched (`m = N`) slot-INDEPENDENT projection pre-pass for the #60 batched
/// decode lane (PHASE C — projection batching).
///
/// The single-token (`m=1`) per-row `mla_attention_prepare` re-reads the wq_a /
/// wkv / wq_b projection weights ONCE PER ROW (× 43 layers — the 137ms @ n=22
/// PREPARE hot spot). The Q/KV LoRA projections + the partial RoPE are
/// SLOT-INDEPENDENT: each row's `c_q_normed` / `q_prepared` / `k_prepared`
/// depends only on this row's `normed` activation + the shared `&Dsv4Attention`
/// weights + this row's absolute position. So they batch to one `m=N` GEMV-batch
/// per weight (each weight read ONCE across the `blockIdx.y` token grid of
/// `dsv4_fp8_gemv_batch_cuda`), amortizing the weight read ×N — the same
/// per-row→batched amortization as the committed lm_head GEMM batch.
///
/// Routes the projections through the scalar batched [`dsv4_linear`] (FP8 GEMV at
/// `num_tokens=N`, or bf16 cuBLAS for dense weights) rather than the per-slot
/// fused-DeepGEMM `run_fused_wqkv_decode` (whose `Dsv4FusedWqkvDecodeScratch` is
/// PER-SLOT, sized for a single `m=1` row — it cannot stage an `m=N` batch
/// without a new model-wide scratch). The batched-GEMV kernel's `(out_row,
/// token)` grid is row-independent, so each row's projection result is
/// byte-identical to the scalar `m=1` per-row path; the change vs the
/// fused-DeepGEMM decode default is the legitimate FP8-DeepGEMM→FP8-GEMV numerics
/// difference (needle-gated, like every DSv4 projection lever).
///
/// `normed` is `[N, hidden]` (the post-attn-LN batch); `positions` is `[N]` i32
/// device, each row's absolute decode position (`start_positions[r]`). Returns
/// `(c_q_normed[N], q_prepared[N], k_prepared[N])` plus the per-row scalars the
/// compressed-only finish needs. The per-slot compressor / indexer / `csa_select`
/// stay PER-ROW in [`mla_attention_prepare_compressed_only`] (they mutate the
/// slot's compressed-key ring); this pre-pass touches NO slot state.
pub(crate) struct Dsv4MlaProjBatch {
    pub(crate) c_q_normed: HiddenStates,
    pub(crate) q_prepared: HiddenStates,
    pub(crate) k_prepared: HiddenStates,
    pub(crate) local_heads: usize,
    pub(crate) local_width: usize,
    pub(crate) sm_scale: f32,
    pub(crate) rope_base: f32,
    pub(crate) original_seq_len: i32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn mla_attention_prepare_proj_batch(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    attention: &Dsv4Attention,
    compress_ratio: usize,
    normed: &HiddenStates,
    positions: &CudaSlice<i32>,
    prefill_shared: Option<&mut Dsv4PrefillDeepGemmLinearScratch>,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<Dsv4MlaProjBatch> {
    ensure!(
        normed.hidden_dim == config.hidden_size,
        "DSv4 MLA proj-batch hidden dim {} != hidden_size {}",
        normed.hidden_dim,
        config.hidden_size
    );
    let n = normed.seq_len;
    ensure!(n > 0, "DSv4 MLA proj-batch requires N>0 rows");
    let head_dim = config.head_dim;
    let local_width = attention.wq_b.rows;
    // See mla_attention_prepare: GLM wq_b emits qk_head-wide q (runtime-absorbed),
    // DSv4 emits pre-absorbed head_dim-wide q. (Batched lane is MODEL1-only today;
    // this keeps the derivation correct if a V32 batched lane is ever wired.)
    let q_head_width = if attention.w_kc.is_some() {
        config.qk_nope_head_dim + config.qk_rope_head_dim
    } else {
        head_dim
    };
    ensure!(
        q_head_width > 0 && local_width.is_multiple_of(q_head_width),
        "DSv4 MLA proj-batch local q width {local_width} is not a multiple of q_head_width {q_head_width}"
    );
    let local_heads = local_width / q_head_width;
    ensure!(
        local_heads > 0,
        "DSv4 MLA proj-batch requires at least one local head"
    );
    ensure!(
        attention.wkv.rows == head_dim,
        "DSv4 MLA proj-batch wkv rows {} != head_dim {head_dim}",
        attention.wkv.rows
    );
    ensure!(
        config.qk_rope_head_dim <= head_dim,
        "DSv4 MLA proj-batch rope dim {} exceeds head_dim {head_dim}",
        config.qk_rope_head_dim
    );

    // RoPE base/YaRN is PER-LAYER (identical policy to `mla_attention_prepare`):
    // compressed layers (cr>0) rope at compress_rope_theta + YaRN, pure-SW
    // (cr=0) at rope_theta with no YaRN.
    let rope = &config.rope_parameters;
    let (rope_base, original_seq_len) = if compress_ratio > 0 {
        let osl = i32::try_from(rope.original_max_position_embeddings).map_err(|_| {
            anyhow!(
                "DSv4 original_max_position_embeddings {} overflows i32",
                rope.original_max_position_embeddings
            )
        })?;
        (config.compress_rope_theta, osl)
    } else {
        (config.rope_theta, 0i32)
    };

    // ── 1+2. Q/KV LoRA at m=N. The weight read MUST amortize across the N decode
    // rows: a real batched FP8 GEMM (DeepGEMM, K-tiled, tensor-core, weight read
    // ONCE for all N), not the per-(out_row, token) scalar GEMV which re-reads the
    // projection weight once PER token (N independent GEMVs, zero amortization).
    // This mirrors `mla_attention_prepare`'s prefill DeepGEMM branch verbatim at
    // m=N: `run_fused_wqkv_prefill` (fused wqkv_a → c_q[N] q_lora + kv_raw[N]
    // head_dim, weight read once) → q_norm RMSNorm → `prefill_proj_deepgemm`
    // (wq_b → q_raw[N], weight read once) → kv_norm RMSNorm. The numerics shift vs
    // the scalar FP8-GEMV (FP8-DeepGEMM activation quantize) is the intended
    // improvement and is needle-gated. The shared `prefill_linear` scratch
    // (max_m = DSV4_PREFILL_QUERY_CHUNK = 4096 >= N) stages the FP8 activation;
    // the per-(out_row,token) GEMV `else` branch is the DeepGEMM-disabled fallback
    // (byte-identical to the prior batched path).
    // #150: bf16 dequant copies present ⇒ skip FP8 DeepGEMM, take the scalar
    // route with the DenseBf16 weights (dsv4_linear ⇒ gemm_batch) so the n≥2
    // decode arithmetic matches the n=1 lane's.
    let use_deepgemm = attention.mla_proj_bf16.is_none()
        && dsv4_fp8_linear_deepgemm_enabled()?
        && attention.wqkv_a_deepgemm.is_some()
        && attention.wq_b_deepgemm.is_some()
        && prefill_shared.is_some();
    let (c_q_normed, q_raw, kv_normed) = if use_deepgemm {
        let scratch = prefill_shared.ok_or_else(|| {
            anyhow!("DSv4 MLA proj-batch DeepGEMM path requires the prefill_linear scratch")
        })?;
        // Fused wqkv_a (hidden → q_lora + head_dim) at m=N via DeepGEMM: c_q[N],
        // kv_raw[N] sliced out of the fused output. Weight read ONCE across N rows.
        // SAFETY: run_fused_wqkv_prefill writes the full c_q / kv_raw buffers.
        let mut c_q = unsafe { HiddenStates::uninit(ctx, attention.wq_a.rows, n)? };
        // SAFETY: uninit device scratch; fully written before first read.
        let mut kv_raw = unsafe { HiddenStates::uninit(ctx, head_dim, n)? };
        crate::profile::profile_op(ctx, "linear/wqkv_a_fused_batched", None, n, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wqkv_a_fused_batched", || {
                run_fused_wqkv_prefill(ctx, attention, normed, &mut *scratch, &mut c_q, &mut kv_raw)
            })
        })?;
        keepalive.keep_hidden(&c_q);
        let c_q_normed = mla_rms_norm(ctx, &c_q, &attention.q_norm, config.rms_norm_eps)?;
        keepalive.keep_hidden(&c_q_normed);
        // Q up-projection wq_b (q_lora → per-head Q) at m=N via DeepGEMM (reuses the
        // same FP8 prefill scratch; K=q_lora_rank <= hidden_dim). Weight read ONCE.
        // SAFETY: prefill_proj_deepgemm writes the full q_raw buffer.
        let mut q_raw = unsafe { HiddenStates::uninit(ctx, local_width, n)? };
        let cache = attention
            .wq_b_deepgemm
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 MLA proj-batch DeepGEMM path requires wq_b_deepgemm"))?;
        crate::profile::profile_op(ctx, "linear/wq_b_batched", None, n, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wq_b_batched", || {
                prefill_proj_deepgemm(ctx, &mut *scratch, cache, &c_q_normed, &mut q_raw)
            })
        })?;
        keepalive.keep_hidden(&q_raw);
        keepalive.keep_hidden(&kv_raw);
        let kv_normed = mla_rms_norm(ctx, &kv_raw, &attention.kv_norm, config.rms_norm_eps)?;
        keepalive.keep_hidden(&kv_normed);
        (c_q_normed, q_raw, kv_normed)
    } else {
        // Fallback (bf16 copies present / DeepGEMM disabled / caches not loaded /
        // scratch absent): the scalar batched dsv4_linear path. With the #150
        // bf16 copies this dispatches DenseBf16 ⇒ gemm_batch (cublasLt); with the
        // FP8 originals, the per-(out_row, token) GEMV grid — byte path == the
        // non-fused `else` branch of `mla_attention_prepare`, at seq_len=N.
        let (wq_a, wq_b, wkv) = attention.decode_proj_weights();
        // SAFETY: dsv4_linear writes the full c_q buffer.
        let mut c_q = unsafe { HiddenStates::uninit(ctx, wq_a.rows, n)? };
        crate::profile::profile_op(ctx, "linear/wq_a_batched", None, n, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wq_a_batched", || {
                dsv4_linear(ctx, wq_a, normed, &mut c_q)
            })
        })?;
        keepalive.keep_hidden(&c_q);
        let c_q_normed = mla_rms_norm(ctx, &c_q, &attention.q_norm, config.rms_norm_eps)?;
        keepalive.keep_hidden(&c_q_normed);
        // SAFETY: dsv4_linear writes the full q_raw buffer.
        let mut q_raw = unsafe { HiddenStates::uninit(ctx, local_width, n)? };
        crate::profile::profile_op(ctx, "linear/wq_b_batched", None, n, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wq_b_batched", || {
                dsv4_linear(ctx, wq_b, &c_q_normed, &mut q_raw)
            })
        })?;
        keepalive.keep_hidden(&q_raw);
        // SAFETY: dsv4_linear writes the full kv_raw buffer.
        let mut kv_raw = unsafe { HiddenStates::uninit(ctx, head_dim, n)? };
        crate::profile::profile_op(ctx, "linear/wkv_batched", None, n, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wkv_batched", || {
                dsv4_linear(ctx, wkv, normed, &mut kv_raw)
            })
        })?;
        keepalive.keep_hidden(&kv_raw);
        let kv_normed = mla_rms_norm(ctx, &kv_raw, &attention.kv_norm, config.rms_norm_eps)?;
        keepalive.keep_hidden(&kv_normed);
        (c_q_normed, q_raw, kv_normed)
    };

    // ── 3. Partial RoPE over all N rows with PER-ROW positions. The fused-batch
    // kernel's Q half (RMS-scale + RoPE) and K half (RoPE-only, no norm) are
    // byte-identical to the single-row `dsv4_prepare_q_kernel`/`_k_kernel` the
    // decode `mla_attention_prepare` runs — only the position is sourced as
    // `positions[token]` (this row's `start_positions[r]`) instead of a single
    // scalar + token offset. For an N-row decode batch each row is a distinct
    // sequence at its own absolute position, so the per-row positions array is
    // exactly what makes the batched RoPE equal the N per-row RoPE calls.
    // SAFETY: dsv4_prepare_qk_fused_batch writes both full output buffers.
    let mut q_prepared = unsafe { HiddenStates::uninit(ctx, local_width, n)? };
    // SAFETY: uninit device scratch; fully written before first read.
    let mut k_prepared = unsafe { HiddenStates::uninit(ctx, head_dim, n)? };
    {
        let (q_raw_ptr, _qr) = q_raw.data.device_ptr(&ctx.stream);
        let (k_raw_ptr, _kr) = kv_normed.data.device_ptr(&ctx.stream);
        let (q_out_ptr, _qo) = q_prepared.data.device_ptr_mut(&ctx.stream);
        let (k_out_ptr, _ko) = k_prepared.data.device_ptr_mut(&ctx.stream);
        let (pos_ptr, _pg) = positions.device_ptr(&ctx.stream);
        // SAFETY: all buffers valid on ctx.stream; positions is [N] i32; shapes
        // checked above.
        unsafe {
            ffi::dsv4_prepare_qk_fused_batch_start_pos_cuda(
                q_raw_ptr as *const ffi::Half,
                k_raw_ptr as *const ffi::Half,
                q_out_ptr as *mut ffi::Half,
                k_out_ptr as *mut ffi::Half,
                n as i32,
                local_heads as i32,
                head_dim as i32,
                config.qk_rope_head_dim as i32,
                pos_ptr as *const i32,
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
    keepalive.keep_hidden(&q_prepared);
    keepalive.keep_hidden(&k_prepared);

    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
    Ok(Dsv4MlaProjBatch {
        c_q_normed,
        q_prepared,
        k_prepared,
        local_heads,
        local_width,
        sm_scale,
        rope_base,
        original_seq_len,
    })
}

/// Full-flatten decode P1a: run ONE row's main + indexer
/// compressor STATE update in DEFER mode — skip the per-row FFI, push the row's
/// ring-state device pointers into the batch sinks, and advance
/// `compressed.seq_len` (host bookkeeping). Returns this row's `indexer_rows_before`
/// (the indexer compressed row count BEFORE this step's advance), captured for the
/// later [`mla_attention_prepare_compressed_only`] (`skip_compressor=true`) CSA
/// path. `0` for non-CSA rows (no indexer). The actual GPU state writes run later
/// in ONE [`dsv4_compressor_update_batched`] over all N rows (BEFORE any reader:
/// the P1b pack + csa cache-write). SW rows are a no-op (no compressor).
#[allow(clippy::too_many_arguments)]
pub(crate) fn mla_attention_compressor_defer_row(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    attention: &Dsv4Attention,
    mode: DeepSeekV4AttentionMode,
    compress_ratio: usize,
    layer_idx: usize,
    normed_row: &HiddenStates,
    state: &mut Dsv4LayerAttentionState,
    start_pos: usize,
    start_pos_device: Option<&CudaSlice<i32>>,
    original_seq_len: i32,
    main_sink: &mut Dsv4CompressorBatchPtrs,
    indexer_sink: &mut Dsv4CompressorBatchPtrs,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<usize> {
    // GLM SparseIndexed has no compressor, and the batched-defer kernel is
    // compressor-specific. The MODEL1-only batch scratch keeps GLM off this path;
    // fail loud if a caller violates that boundary.
    ensure!(
        mode != DeepSeekV4AttentionMode::SparseIndexed,
        "DSv4 full-flatten compressor-defer path does not support SparseIndexed (no compressor); \
         full-flatten must stay off for GLM"
    );
    if mode == DeepSeekV4AttentionMode::SlidingWindow {
        return Ok(0);
    }
    let head_dim = config.head_dim;
    let compressor = attention.compressor.as_ref().ok_or_else(|| {
        anyhow!("DSv4 layer {layer_idx} is {mode:?} but has no compressor weights")
    })?;
    let overlap = compress_ratio < 16;
    {
        let compressor_state = state.compressor.as_mut().ok_or_else(|| {
            anyhow!("DSv4 layer {layer_idx} is {mode:?} but has no compressor state")
        })?;
        compressor_forward(
            ctx,
            config,
            compressor,
            normed_row,
            compressor_state,
            head_dim,
            compress_ratio,
            overlap,
            start_pos,
            start_pos_device,
            true,
            original_seq_len,
            None,
            // Defer mode ignores `precomputed` (the batched update reads the batched
            // prepass output directly); pass None.
            None,
            Some(main_sink),
            keepalive,
        )?;
    }
    let mut indexer_rows_before = 0usize;
    if mode == DeepSeekV4AttentionMode::CompressedSparse {
        let indexer = attention.indexer.as_ref().ok_or_else(|| {
            anyhow!("DSv4 layer {layer_idx} is CompressedSparse but has no indexer weights")
        })?;
        let use_official_dsa = dsv4_dsa_official_enabled()?;
        let indexer_rope_original_seq_len = if use_official_dsa {
            i32::try_from(config.rope_parameters.original_max_position_embeddings).map_err(
                |_| anyhow!("DSv4 official DSA original_max_position_embeddings overflows i32"),
            )?
        } else {
            0
        };
        indexer_rows_before = state
            .indexer
            .as_ref()
            .map(|s| s.compressed.seq_len)
            .unwrap_or(0);
        let indexer_state = state.indexer.as_mut().ok_or_else(|| {
            anyhow!("DSv4 layer {layer_idx} is CompressedSparse but has no indexer state")
        })?;
        compressor_forward(
            ctx,
            config,
            indexer
                .compressor
                .as_ref()
                .expect("DSv4 CSA indexer has a key compressor"),
            normed_row,
            indexer_state,
            config.index_head_dim,
            compress_ratio,
            true,
            start_pos,
            start_pos_device,
            use_official_dsa,
            indexer_rope_original_seq_len,
            None,
            None,
            Some(indexer_sink),
            keepalive,
        )?;
    }
    Ok(indexer_rows_before)
}

/// Per-row SLOT-DEPENDENT finish of the batched-decode PREPARE, paired with
/// [`mla_attention_prepare_proj_batch`] (#60 PHASE C). The projection + RoPE for
/// row `r` were already computed batched; this runs ONLY the per-slot half —
/// the CSA/HCA compressor, the CSA indexer + top-k `selected` — over this row's
/// `normed_row` (the slot's compressed-key ring is mutated here) and this row's
/// sliced `c_q_normed_row`. It then assembles the `Dsv4MlaPrepared` the batched
/// lane's pack/gather/fwd/finish consume, taking ownership of the per-row
/// `q_prepared_row` / `k_prepared_row` slices the caller copied out of the batch.
///
/// `q_prepared_row` / `k_prepared_row` / `c_q_normed_row` are this row's `[*, 1]`
/// slices of the batched outputs (the caller memcpy's them into row scratch,
/// mirroring the existing `normed_row` copy-in discipline). `c_q_normed_row` MUST
/// hold byte-identical data to what `mla_attention_prepare` would have produced
/// for this row, so `csa_select` selects the same blocks — guaranteed by the
/// row-independent batched GEMV.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mla_attention_prepare_compressed_only(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    attention: &Dsv4Attention,
    mode: DeepSeekV4AttentionMode,
    compress_ratio: usize,
    layer_idx: usize,
    normed_row: &HiddenStates,
    c_q_normed_row: &HiddenStates,
    q_prepared_row: HiddenStates,
    k_prepared_row: HiddenStates,
    proj: &Dsv4MlaProjBatch,
    state: &mut Dsv4LayerAttentionState,
    pool: &mut Dsv4LayerKvLayout,
    dsa_shared: Option<&mut Dsv4DsaSharedScratch>,
    start_pos: usize,
    start_pos_device: Option<&CudaSlice<i32>>,
    // Batched-decode lane (#60): when `Some`, the CSA per-row READ/SELECT is
    // skipped (cache writes still run), this row's q_i/weights are gathered into
    // the N-row staging and key_count captured, and the returned `selected` is
    // `None` (the batched select fills selected_batched). `None` → byte-identical
    // single-row CSA prepare.
    batched_gather: Option<Dsv4DsaBatchedGather<'_>>,
    // Canonical batched decode pre-pass: when `Some`, this row's
    // compressor/indexer `(kv_raw, score_raw)` were already projected batched and
    // are passed to
    // `compressor_forward`, skipping the per-row m=1 GEMVs. `None` → byte-identical
    // per-row `dsv4_linear` projection.
    compressor_precomputed: Option<Dsv4CompressorPrecomputed<'_>>,
    // Batched-decode pre-pass: when `Some`, this row's indexer query (`q_i`) and
    // gating `weights` were already projected batched (one m=N `dsv4_linear` each
    // in `indexer_query_batch_prepass`); the `[width,1]` slices are threaded into
    // `csa_select`, skipping the per-row m=1 `wq_b`/`weights_proj` GEMVs. `None` →
    // byte-identical per-row GEMVs. Compressor-layer (CSA) only; SparseIndexed and
    // the non-full-flatten lane pass `None`.
    indexer_query_precomputed: Option<Dsv4IndexerQueryPrecomputed<'_>>,
    // Full-flatten decode: when `true`, the per-row
    // compressor / indexer STATE updates already ran in ONE batched
    // `dsv4_compressor_update_batched` BEFORE this call (a P1a pre-pass), so the
    // two `compressor_forward` calls here are SKIPPED — only the CSA per-row
    // `csa_select` (which reads the now-written compressed keys) runs. The
    // pre-pass also advanced `compressed.seq_len`, so `indexer_rows_before` (the
    // value BEFORE this step's advance) must be supplied via
    // `indexer_rows_before_override`. `false` → the compressor_forward calls run
    // here (per-row update, byte-identical).
    skip_compressor: bool,
    indexer_rows_before_override: Option<usize>,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<Dsv4MlaPrepared> {
    let head_dim = config.head_dim;
    let local_heads = proj.local_heads;
    let local_width = proj.local_width;
    ensure!(
        q_prepared_row.hidden_dim == local_width && q_prepared_row.seq_len == 1,
        "DSv4 compressed-only q_prepared row {}x{} != [{local_width},1]",
        q_prepared_row.hidden_dim,
        q_prepared_row.seq_len
    );
    ensure!(
        k_prepared_row.hidden_dim == head_dim && k_prepared_row.seq_len == 1,
        "DSv4 compressed-only k_prepared row {}x{} != [{head_dim},1]",
        k_prepared_row.hidden_dim,
        k_prepared_row.seq_len
    );
    ensure!(
        state.sw_window_cache.len() == config.sliding_window * head_dim,
        "DSv4 MLA SW window cache len {} != sliding_window*head_dim {}",
        state.sw_window_cache.len(),
        config.sliding_window * head_dim
    );
    let original_seq_len = proj.original_seq_len;

    // SAFETY: the FlashMLA attention kernel writes the full local_attn buffer
    // in the batched fwd / single-row `mla_attention_fwd`.
    let local_attn = unsafe { HiddenStates::uninit(ctx, local_width, 1)? };

    // ── 4b(prep). compressor → (CSA) indexer top-k select — PER SLOT. Byte path
    // identical to the `mla_attention_prepare` decode branch, reading this row's
    // `normed_row` (compressor) and `c_q_normed_row` (indexer query proj).
    // GLM SparseIndexed: full-sequence indexer, every token a key (ratio=1, no
    // compressor). CompressedSparse keeps its real compress_ratio.
    let index_ratio = if mode == DeepSeekV4AttentionMode::SparseIndexed {
        1
    } else {
        compress_ratio
    };
    let selected = if !mode.has_indexer() && !mode.has_compressor() {
        None
    } else {
        let overlap = compress_ratio < 16;
        // Split the batched pre-pass slices into main + indexer (the inner
        // `(&HiddenStates, &HiddenStates)` pairs are `Copy` references, so reading
        // both fields does not conflict with the per-call `state` borrows below).
        // `None` everywhere → byte-identical per-row GEMV path. The
        // compressor-batch / full-flatten gates only fire for compressor layers,
        // so for GLM SparseIndexed they are always None/false.
        let precomputed_main = compressor_precomputed.as_ref().map(|p| p.main);
        let precomputed_indexer = compressor_precomputed.as_ref().and_then(|p| p.indexer);
        // MAIN compressor (CSA/HCA only): GLM SparseIndexed has none — gate on
        // has_compressor(). Full-flatten: the main compressor STATE update already
        // ran batched in the P1a pre-pass; skip it here.
        if mode.has_compressor() {
            let compressor = attention.compressor.as_ref().ok_or_else(|| {
                anyhow::anyhow!("DSv4 layer {layer_idx} is {mode:?} but has no compressor weights")
            })?;
            if !skip_compressor {
                let compressor_state = state.compressor.as_mut().ok_or_else(|| {
                    anyhow::anyhow!(
                        "DSv4 layer {layer_idx} is {mode:?} but has no compressor state"
                    )
                })?;
                compressor_forward(
                    ctx,
                    config,
                    compressor,
                    normed_row,
                    compressor_state,
                    head_dim,
                    compress_ratio,
                    overlap,
                    start_pos,
                    start_pos_device,
                    true,
                    original_seq_len,
                    None,
                    precomputed_main,
                    None,
                    keepalive,
                )?;
            }
        }

        if mode.has_indexer() {
            let indexer = attention.indexer.as_ref().ok_or_else(|| {
                anyhow::anyhow!("DSv4 layer {layer_idx} is {mode:?} but has no indexer weights")
            })?;
            let use_official_dsa = dsv4_dsa_official_enabled()?;
            let indexer_rope_original_seq_len = if use_official_dsa {
                i32::try_from(config.rope_parameters.original_max_position_embeddings).map_err(
                    |_| {
                        anyhow!(
                            "DSv4 official DSA original_max_position_embeddings {} overflows i32",
                            config.rope_parameters.original_max_position_embeddings
                        )
                    },
                )?
            } else {
                0
            };
            // `indexer_rows_before` = the indexer compressed row count BEFORE this
            // step's advance. In full-flatten the P1a pre-pass already advanced
            // `compressed.seq_len`, so the live value would be the AFTER count;
            // take the pre-pass-captured override instead. Otherwise read it live.
            // (full-flatten is compressor-only, so SparseIndexed always reads live.)
            let indexer_rows_before = if skip_compressor {
                indexer_rows_before_override.ok_or_else(|| {
                    anyhow!("DSv4 full-flatten CSA prepare: indexer_rows_before_override missing")
                })?
            } else {
                state
                    .indexer
                    .as_ref()
                    .map(|s| s.compressed.seq_len)
                    .unwrap_or(0)
            };
            // CSA runs the indexer key compressor; GLM SparseIndexed builds one
            // full-length index key per token (ratio=1, no compressor). Full-flatten
            // (skip_compressor) only applies to compressor layers.
            if !skip_compressor {
                let indexer_state = state.indexer.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("DSv4 layer {layer_idx} is {mode:?} but has no indexer state")
                })?;
                if mode == DeepSeekV4AttentionMode::CompressedSparse {
                    compressor_forward(
                        ctx,
                        config,
                        indexer
                            .compressor
                            .as_ref()
                            .expect("DSv4 CSA indexer has a key compressor"),
                        normed_row,
                        indexer_state,
                        config.index_head_dim,
                        compress_ratio,
                        true,
                        start_pos,
                        start_pos_device,
                        use_official_dsa,
                        indexer_rope_original_seq_len,
                        None,
                        precomputed_indexer,
                        None,
                        keepalive,
                    )?;
                } else {
                    // GLM SparseIndexed: per-row index-key build (no compressor,
                    // no precomputed batch path — those are compressor-only).
                    sparse_indexed_index_key_forward(
                        ctx,
                        config,
                        indexer,
                        normed_row,
                        indexer_state,
                        start_pos,
                        keepalive,
                    )?;
                }
            }
            let indexer_rows_after = state
                .indexer
                .as_ref()
                .map(|s| s.compressed.seq_len)
                .unwrap_or(0);
            // #146 Index-layer guard (mirrors `mla_attention_prepare`'s eager-lane
            // check): this is the batched-decode twin of that function — same
            // `Dsv4LayerAttentionState`, same CSA row-count invariant. No
            // `chain_verify`/frozen-compressor lane exists on this path, so the
            // gate is mode-only.
            if mode == DeepSeekV4AttentionMode::CompressedSparse {
                let value_rows = state
                    .compressor
                    .as_ref()
                    .map(|s| s.compressed.seq_len)
                    .unwrap_or(0);
                ensure!(
                    indexer_rows_after == value_rows,
                    "DSv4 CSA select boundary (batched): indexer rows {indexer_rows_after} != \
                     value compressor rows {value_rows} (Shape drift — #146 guard)"
                );
            }
            let keys_capacity = state
                .indexer
                .as_ref()
                .map(|s| s.compressed_capacity())
                .unwrap_or(0);
            let index_keys = &state
                .indexer
                .as_ref()
                .expect("indexer state checked above")
                .compressed;
            let official = state.dsa_official.as_mut();
            // In the batched-decode lane (`batched_gather` Some) `csa_select`
            // does cache writes + gather and returns `None`; the batched select
            // fills `selected_batched`. The single-row path keeps the real
            // per-row `selected`.
            csa_select(
                ctx,
                config,
                layer_idx,
                indexer,
                normed_row,
                c_q_normed_row,
                index_keys,
                keys_capacity,
                // Staging-ring base: SparseIndexed stages window-relative to
                // start_pos; the compressor-indexer retains full history (base 0).
                if mode == DeepSeekV4AttentionMode::SparseIndexed {
                    start_pos
                } else {
                    0
                },
                official,
                dsa_shared,
                pool,
                indexer_rows_before,
                indexer_rows_after,
                start_pos,
                start_pos_device,
                index_ratio,
                // Decode (token_count=1): the prefill indexer DeepGEMM lane is
                // never taken (gated on c_q_normed.seq_len>1), so the shared
                // prefill scratch is not needed.
                None,
                batched_gather,
                indexer_query_precomputed,
                keepalive,
            )?
        } else {
            None
        }
    };

    Ok(Dsv4MlaPrepared {
        q_prepared: q_prepared_row,
        k_prepared: k_prepared_row,
        local_attn,
        selected,
        local_heads,
        token_count: 1,
        sm_scale: proj.sm_scale,
        rope_base: proj.rope_base,
        original_seq_len,
    })
}

/// FWD half of `mla_attention` (see [`Dsv4MlaPrepared`]). Runs the FlashMLA
/// attention kernel over the PREPARE output, then the O-LoRA. The compressed-key
/// pool is re-borrowed from `state` here. Single-row + chunked-prefill callers
/// reach this through `mla_attention`; the batched lane runs PREPARE per row and
/// the batched fwd separately.
#[allow(clippy::too_many_arguments)]
fn mla_attention_fwd(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    attention: &Dsv4Attention,
    mode: DeepSeekV4AttentionMode,
    compress_ratio: usize,
    layer_idx: usize,
    state: &mut Dsv4LayerAttentionState,
    pool: &mut Dsv4LayerKvLayout,
    // Model-wide shared single-row FlashMLA decode scratch (#85 P3). `Some`
    // whenever FlashMLA decode is allocated (same gate as the per-slot state);
    // consumed only on the single-row decode FlashMLA path below.
    flashmla_scratch: Option<&mut Dsv4FlashMlaDecodeScratch>,
    // Model-wide shared FP8 prefill DeepGEMM linear scratch, forwarded to the
    // O-LoRA `mla_oproj` (its token_count>1 prefill DeepGEMM lane gates on it).
    prefill_shared: Option<&mut Dsv4PrefillDeepGemmLinearScratch>,
    start_pos: usize,
    start_pos_device: Option<&CudaSlice<i32>>,
    chain_verify: Option<&Dsv4ChainVerifyAttnMeta>,
    tp: &TpRuntime,
    prepared: Dsv4MlaPrepared,
    out: &mut HiddenStates,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<()> {
    let Dsv4MlaPrepared {
        q_prepared,
        k_prepared,
        mut local_attn,
        selected,
        local_heads,
        token_count,
        sm_scale,
        rope_base,
        original_seq_len,
    } = prepared;
    let rope = &config.rope_parameters;
    ensure!(
        attention.wo_b.as_ref().expect("DSv4 wo_b").rows == out.hidden_dim
            && out.seq_len == token_count,
        "DSv4 MLA output shape mismatch: wo_b rows {} out {}x{} expected {}x{}",
        attention.wo_b.as_ref().expect("DSv4 wo_b").rows,
        out.hidden_dim,
        out.seq_len,
        attention.wo_b.as_ref().expect("DSv4 wo_b").rows,
        token_count
    );
    keepalive.keep_hidden(&q_prepared);
    keepalive.keep_hidden(&k_prepared);

    // Compressed KV pool: CSA/HCA have a compressor; SparseIndexed (GLM DSA)
    // and SWA do not.
    let compressed: Option<&HiddenStates> = if mode.has_compressor() {
        Some(
            &state
                .compressor
                .as_ref()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "DSv4 layer {layer_idx} is {mode:?} but has no compressor state"
                    )
                })?
                .compressed,
        )
    } else {
        None
    };

    if token_count > 1 {
        flashmla_prefill_attention(
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
            chain_verify,
            tp,
            local_heads,
            &mut local_attn,
            sm_scale,
            rope_base,
            original_seq_len,
            rope.factor,
            rope.beta_fast,
            rope.beta_slow,
        )?;
    } else {
        let flash = state.flashmla.as_mut().ok_or_else(|| {
            anyhow!("FlashMLA decode enabled but layer state has no FlashMLA arena")
        })?;
        let scratch = flashmla_scratch.ok_or_else(|| {
            anyhow!("FlashMLA decode enabled but shared FlashMLA decode scratch missing")
        })?;
        flashmla_decode_attention(
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
            flash,
            scratch,
            pool,
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
        )?;
    }
    if let Some(sel) = selected.as_ref() {
        keepalive.keep_i32(sel);
    }
    keepalive.keep_hidden(&local_attn);

    // ── 4b. GLM runtime V absorption (w_vc.is_some()). The FlashMLA output is
    // the [heads, kv_lora(512)] latent; GLM projects it back to v[heads,
    // v_head(256)] via v[h] = attn_out[h] · w_vc[h] before the plain o_proj.
    // This changes local_attn's hidden_dim from heads*512 to heads*256. DSv4
    // (w_vc None) skips — local_attn is already the wo_a/wo_b latent.
    let local_attn = if let Some(w_vc) = attention.w_vc.as_ref() {
        let v = glm_absorb_v(ctx, config, w_vc, &local_attn, local_heads, keepalive)?;
        keepalive.keep_hidden(&v);
        v
    } else {
        local_attn
    };

    // ── 5. O-LoRA: wo_a (per o-group, down to the output latent) → wo_b (up
    // back to hidden). Row-parallel: the all-reduce-sum is the model's concern.
    // GLM (o_proj.is_some()) takes the plain-o early return in mla_oproj.
    // SAFETY: dsv4_linear writes the full latent buffer.
    mla_oproj(
        ctx,
        attention,
        state,
        prefill_shared,
        &local_attn,
        token_count,
        keepalive,
        out,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Dsv4OProjGroupShape {
    groups: usize,
    rows_per_group: usize,
    cols_per_group: usize,
    routes: usize,
}

fn dsv4_oproj_group_shape(
    wo_a_rows: usize,
    wo_a_cols: usize,
    table_groups: usize,
    table_rows_per_group: usize,
    table_cols_per_group: usize,
    local_width: usize,
    token_count: usize,
) -> Result<Dsv4OProjGroupShape> {
    ensure!(token_count > 0, "DSv4 O-LoRA projection requires tokens");
    ensure!(
        wo_a_cols > 0 && local_width.is_multiple_of(wo_a_cols),
        "DSv4 O-LoRA local attention width {local_width} is not a whole number of wo_a groups (group width {wo_a_cols})"
    );
    let groups = local_width / wo_a_cols;
    ensure!(groups > 0, "DSv4 O-LoRA has zero local output groups");
    ensure!(
        groups == table_groups
            && table_rows_per_group > 0
            && table_cols_per_group == wo_a_cols
            && wo_a_rows == groups * table_rows_per_group,
        "DSv4 O-LoRA group table mismatch: local groups={groups}, table groups={table_groups}, \
         table rows/group={table_rows_per_group}, table cols/group={table_cols_per_group}, \
         wo_a={}x{}",
        wo_a_rows,
        wo_a_cols
    );
    let routes = token_count.checked_mul(groups).ok_or_else(|| {
        anyhow!("DSv4 O-LoRA route count overflow: token_count={token_count} groups={groups}")
    })?;
    Ok(Dsv4OProjGroupShape {
        groups,
        rows_per_group: table_rows_per_group,
        cols_per_group: wo_a_cols,
        routes,
    })
}

fn dsv4_wo_a_grouped_linear(
    ctx: &DeviceContext,
    attention: &Dsv4Attention,
    local_attn: &HiddenStates,
    shape: Dsv4OProjGroupShape,
    latent: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        local_attn.hidden_dim == shape.groups * shape.cols_per_group
            && latent.hidden_dim == shape.groups * shape.rows_per_group
            && local_attn.seq_len == latent.seq_len,
        "DSv4 O-LoRA grouped shape mismatch: local_attn {}x{}, latent {}x{}, groups={} rows/group={} cols/group={}",
        local_attn.hidden_dim,
        local_attn.seq_len,
        latent.hidden_dim,
        latent.seq_len,
        shape.groups,
        shape.rows_per_group,
        shape.cols_per_group
    );
    // Dense BF16 `wo_a` (FP8 re-serialization leaves the tiny low-rank wo_a
    // unquantized) has no per-group FP8 scales — the loader builds the per-group
    // base-pointer table with `scale_rows_per_group=0`. On TP<o_groups the rank
    // owns >1 output group, so this grouped path is taken; route through a
    // per-group dense BF16 GEMM instead of the FP8/FP4 route-GEMV (which asserts
    // non-empty scales below). Each group is independent: gather group `g`'s
    // column slice from the token-major `[seq, groups*cols]` activation into
    // contiguous `[seq, cols]`, run `gemm_cuda` with group `g`'s `[rows, cols]`
    // weight block (contiguous rows `[g*rows, (g+1)*rows)` of `wo_a`), then
    // scatter the `[seq, rows]` result back into the `[seq, groups*rows]` latent.
    if attention.wo_a.as_ref().expect("DSv4 wo_a").weight_format == WeightFormat::DenseBf16 {
        let seq = local_attn.seq_len;
        let cols = shape.cols_per_group;
        let rows = shape.rows_per_group;
        let wo_a = attention.wo_a.as_ref().expect("DSv4 wo_a");
        // SAFETY: uninit device scratch; fully written before first read.
        let mut in_g = unsafe { HiddenStates::uninit(ctx, cols, seq)? };
        // SAFETY: uninit device scratch; fully written before first read.
        let mut out_g = unsafe { HiddenStates::uninit(ctx, rows, seq)? };
        // Cache raw pointers once: the buffers are not reallocated across the
        // group loop (same single inference stream), so per-iteration mutable +
        // immutable `device_ptr` calls would falsely collide on SyncOnDrop.
        let (wo_a_base, _wg) = wo_a.data.device_ptr(&ctx.stream);
        let (src_ptr, _sg) = local_attn.data.device_ptr(&ctx.stream);
        let (in_ptr, _ig) = in_g.data.device_ptr_mut(&ctx.stream);
        let (out_ptr, _og) = out_g.data.device_ptr_mut(&ctx.stream);
        let (dst_ptr, _dg) = latent.data.device_ptr_mut(&ctx.stream);
        let stream = ctx.stream.cu_stream();
        for group in 0..shape.groups {
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            unsafe {
                ffi::dsv4_oproj_group_gather_cuda(
                    src_ptr as *const ffi::Half,
                    in_ptr as *mut ffi::Half,
                    i32::try_from(seq)?,
                    i32::try_from(shape.groups)?,
                    i32::try_from(cols)?,
                    i32::try_from(group)?,
                    stream,
                )
                .result()
                .map_err(|e| anyhow!("DSv4 dense grouped O-LoRA gather failed: {e}"))?;
            }
            // SAFETY: group `g`'s weight block is contiguous rows
            // `[g*rows, (g+1)*rows)` of the `[groups*rows, cols]` dense `wo_a`,
            // i.e. offset `g*rows*cols` bf16 elements from the base pointer.
            let w_g = unsafe { (wo_a_base as *const ffi::Half).add(group * rows * cols) };
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            unsafe {
                ffi::gemm_cuda(
                    w_g,
                    in_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    i32::try_from(rows)?,
                    i32::try_from(seq)?,
                    i32::try_from(cols)?,
                    stream,
                )
                .result()
                .map_err(|e| anyhow!("DSv4 dense grouped O-LoRA gemm failed: {e}"))?;
            }
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            unsafe {
                ffi::dsv4_oproj_group_scatter_cuda(
                    out_ptr as *const ffi::Half,
                    dst_ptr as *mut ffi::Half,
                    i32::try_from(seq)?,
                    i32::try_from(shape.groups)?,
                    i32::try_from(rows)?,
                    i32::try_from(group)?,
                    stream,
                )
                .result()
                .map_err(|e| anyhow!("DSv4 dense grouped O-LoRA scatter failed: {e}"))?;
            }
        }
        return Ok(());
    }
    ensure!(
        attention
            .wo_a_groups
            .as_ref()
            .expect("DSv4 wo_a_groups")
            .scale_rows_per_group
            > 0
            && attention
                .wo_a_groups
                .as_ref()
                .expect("DSv4 wo_a_groups")
                .scale_cols
                > 0,
        "DSv4 O-LoRA grouped scale shape must be non-empty"
    );
    let (weight_ptrs, _wg) = attention
        .wo_a_groups
        .as_ref()
        .expect("DSv4 wo_a_groups")
        .weight_ptrs
        .device_ptr(&ctx.stream);
    let (scale_ptrs, _sg) = attention
        .wo_a_groups
        .as_ref()
        .expect("DSv4 wo_a_groups")
        .scale_ptrs
        .device_ptr(&ctx.stream);
    let (input_ptr, _ig) = local_attn.data.device_ptr(&ctx.stream);
    let (output_ptr, _og) = latent.data.device_ptr_mut(&ctx.stream);
    let stream = ctx.stream.cu_stream();
    // SAFETY: pointer tables were built from this rank's contiguous `wo_a`
    // groups at load time. `route_meta=null` selects group `route % groups`;
    // route order is `[token0/group0, token0/group1, ..., token1/group0, ...]`,
    // which is exactly the `HiddenStates` token-major layout when each group is
    // `cols_per_group` wide.
    unsafe {
        match attention.wo_a.as_ref().expect("DSv4 wo_a").weight_format {
            WeightFormat::Dsv4Fp8BlockScaled => ffi::dsv4_fp8_route_gemv_batch_cuda(
                weight_ptrs as *const u64,
                scale_ptrs as *const u64,
                input_ptr as *const ffi::Half,
                output_ptr as *mut ffi::Half,
                std::ptr::null(),
                0,
                i32::try_from(shape.groups)?,
                i32::try_from(shape.routes)?,
                i32::try_from(shape.rows_per_group)?,
                i32::try_from(shape.cols_per_group)?,
                i32::try_from(
                    attention
                        .wo_a_groups
                        .as_ref()
                        .expect("DSv4 wo_a_groups")
                        .scale_rows_per_group,
                )?,
                i32::try_from(
                    attention
                        .wo_a_groups
                        .as_ref()
                        .expect("DSv4 wo_a_groups")
                        .scale_cols,
                )?,
                0,
                stream,
            ),
            WeightFormat::Dsv4Fp4BlockScaled => ffi::dsv4_fp4_route_gemv_batch_cuda(
                weight_ptrs as *const u64,
                scale_ptrs as *const u64,
                input_ptr as *const ffi::Half,
                output_ptr as *mut ffi::Half,
                std::ptr::null(),
                0,
                i32::try_from(shape.groups)?,
                i32::try_from(shape.routes)?,
                i32::try_from(shape.rows_per_group)?,
                i32::try_from(shape.cols_per_group)?,
                i32::try_from(
                    attention
                        .wo_a_groups
                        .as_ref()
                        .expect("DSv4 wo_a_groups")
                        .scale_rows_per_group,
                )?,
                i32::try_from(
                    attention
                        .wo_a_groups
                        .as_ref()
                        .expect("DSv4 wo_a_groups")
                        .scale_cols,
                )?,
                0,
                stream,
            ),
            other => bail!("DSv4 O-LoRA grouped wo_a expected FP8/FP4 block-scaled, got {other:?}"),
        }
        .result()?;
    }
    Ok(())
}

fn dsv4_oproj_group_gather(
    ctx: &DeviceContext,
    src: &HiddenStates,
    group: usize,
    shape: Dsv4OProjGroupShape,
    scratch: &mut Dsv4PrefillDeepGemmLinearScratch,
) -> Result<()> {
    ensure!(
        group < shape.groups
            && src.hidden_dim == shape.groups * shape.cols_per_group
            && src.seq_len <= scratch.max_m
            && shape.cols_per_group <= scratch.oproj_group_cols
            && src.seq_len * shape.cols_per_group <= scratch.oproj_group_in.len(),
        "DSv4 grouped O-LoRA gather mismatch: group={} groups={} src={}x{} scratch M={} cols={}",
        group,
        shape.groups,
        src.hidden_dim,
        src.seq_len,
        scratch.max_m,
        scratch.oproj_group_cols
    );
    let (src_ptr, _src_guard) = src.data.device_ptr(&ctx.stream);
    let (dst_ptr, _dst_guard) = scratch.oproj_group_in.device_ptr_mut(&ctx.stream);
    // SAFETY: ptrs from live device allocations sized to the dims passed.
    unsafe {
        ffi::dsv4_oproj_group_gather_cuda(
            src_ptr as *const ffi::Half,
            dst_ptr as *mut ffi::Half,
            i32::try_from(src.seq_len)?,
            i32::try_from(shape.groups)?,
            i32::try_from(shape.cols_per_group)?,
            i32::try_from(group)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("DSv4 grouped O-LoRA gather failed: {e}"))?;
    }
    Ok(())
}

fn dsv4_oproj_group_scatter(
    ctx: &DeviceContext,
    scratch: &Dsv4PrefillDeepGemmLinearScratch,
    group: usize,
    shape: Dsv4OProjGroupShape,
    dst: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        group < shape.groups
            && dst.hidden_dim == shape.groups * shape.rows_per_group
            && dst.seq_len <= scratch.max_m
            && shape.rows_per_group <= scratch.oproj_group_rows
            && dst.seq_len * shape.rows_per_group <= scratch.oproj_group_out.len(),
        "DSv4 grouped O-LoRA scatter mismatch: group={} groups={} dst={}x{} scratch M={} rows={}",
        group,
        shape.groups,
        dst.hidden_dim,
        dst.seq_len,
        scratch.max_m,
        scratch.oproj_group_rows
    );
    let (src_ptr, _src_guard) = scratch.oproj_group_out.device_ptr(&ctx.stream);
    let (dst_ptr, _dst_guard) = dst.data.device_ptr_mut(&ctx.stream);
    // SAFETY: ptrs from live device allocations sized to the dims passed.
    unsafe {
        ffi::dsv4_oproj_group_scatter_cuda(
            src_ptr as *const ffi::Half,
            dst_ptr as *mut ffi::Half,
            i32::try_from(dst.seq_len)?,
            i32::try_from(shape.groups)?,
            i32::try_from(shape.rows_per_group)?,
            i32::try_from(group)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("DSv4 grouped O-LoRA scatter failed: {e}"))?;
    }
    Ok(())
}

fn dsv4_wo_a_grouped_deepgemm_decode(
    ctx: &DeviceContext,
    scratch: &mut Dsv4FusedWqkvDecodeScratch,
    caches: &[cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache],
    local_attn: &HiddenStates,
    shape: Dsv4OProjGroupShape,
    latent: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        caches.len() == shape.groups,
        "DSv4 grouped O-LoRA DeepGEMM cache count {} != groups {}",
        caches.len(),
        shape.groups
    );
    // M-parametric over n: each group is ONE DeepGEMM at M=n. Group `g`'s columns
    // are strided in the token-major `[n, groups*cols_per_group]` activation, so
    // gather them into contiguous `[cols_per_group, n]`, GEMM(m=n) into
    // `[rows_per_group, n]`, then scatter back into the `[n, groups*rows_per_group]`
    // latent. The gather/scatter kernels are already num_tokens-parametric and the
    // GEMM is M-parametric. N=1 BYTE-IDENTITY: at n=1 gather/scatter index group g
    // at exactly the offsets the old per-row `slice()` used, m=1 skips the
    // active_counts H2D, so the GEMM args are unchanged and the copies are bit-exact.
    let n = local_attn.seq_len;
    ensure!(
        latent.seq_len == n,
        "DSv4 grouped O-LoRA decode DeepGEMM seq mismatch: local={n} latent={}",
        latent.seq_len
    );
    let cols = shape.cols_per_group;
    let rows = shape.rows_per_group;
    // SAFETY: uninit device scratch; fully written before first read.
    let mut in_g = unsafe { HiddenStates::uninit(ctx, cols, n)? };
    // SAFETY: uninit device scratch; fully written before first read.
    let mut out_g = unsafe { HiddenStates::uninit(ctx, rows, n)? };
    let in_len = in_g.data.len();
    let out_len = out_g.data.len();
    for (group, cache) in caches.iter().enumerate() {
        // Scope the gather guards so `in_g.data`'s mutable SyncOnDrop guard drops
        // before the GEMM re-borrows it immutably.
        {
            let (src_ptr, _src_guard) = local_attn.data.device_ptr(&ctx.stream);
            let (in_ptr, _in_guard) = in_g.data.device_ptr_mut(&ctx.stream);
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            unsafe {
                ffi::dsv4_oproj_group_gather_cuda(
                    src_ptr as *const ffi::Half,
                    in_ptr as *mut ffi::Half,
                    i32::try_from(n)?,
                    i32::try_from(shape.groups)?,
                    i32::try_from(cols)?,
                    i32::try_from(group)?,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("DSv4 grouped O-LoRA decode gather failed: {e}"))?;
            }
        }
        decode_proj_deepgemm_raw(
            ctx,
            scratch,
            cache,
            &in_g.data,
            in_len,
            &mut out_g.data,
            out_len,
            cols,
            n,
        )?;
        // Scope the scatter guards symmetrically.
        {
            let (out_ptr, _out_guard) = out_g.data.device_ptr(&ctx.stream);
            let (dst_ptr, _dst_guard) = latent.data.device_ptr_mut(&ctx.stream);
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            unsafe {
                ffi::dsv4_oproj_group_scatter_cuda(
                    out_ptr as *const ffi::Half,
                    dst_ptr as *mut ffi::Half,
                    i32::try_from(n)?,
                    i32::try_from(shape.groups)?,
                    i32::try_from(rows)?,
                    i32::try_from(group)?,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("DSv4 grouped O-LoRA decode scatter failed: {e}"))?;
            }
        }
    }
    Ok(())
}

fn dsv4_wo_a_grouped_deepgemm_prefill(
    ctx: &DeviceContext,
    scratch: &mut Dsv4PrefillDeepGemmLinearScratch,
    caches: &[cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache],
    local_attn: &HiddenStates,
    shape: Dsv4OProjGroupShape,
    latent: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        caches.len() == shape.groups,
        "DSv4 grouped O-LoRA DeepGEMM cache count {} != groups {}",
        caches.len(),
        shape.groups
    );
    ensure!(
        local_attn.seq_len <= scratch.max_m
            && shape.cols_per_group <= scratch.oproj_group_cols
            && shape.rows_per_group <= scratch.oproj_group_rows,
        "DSv4 grouped O-LoRA prefill scratch mismatch: M={} shape cols={} rows={} scratch M={} cols={} rows={}",
        local_attn.seq_len,
        shape.cols_per_group,
        shape.rows_per_group,
        scratch.max_m,
        scratch.oproj_group_cols,
        scratch.oproj_group_rows
    );
    for (group, cache) in caches.iter().enumerate() {
        dsv4_oproj_group_gather(ctx, local_attn, group, shape, scratch)?;
        prefill_proj_deepgemm_group_scratch(ctx, scratch, cache, local_attn.seq_len)?;
        dsv4_oproj_group_scatter(ctx, scratch, group, shape, latent)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "attention/tests.rs"]
mod tests;

/// O-LoRA output projection, extracted from `mla_attention` so the batched-decode
/// path can call it ONCE over [N] rows (Phase 4): `wo_a` (down to the output latent)
/// → `wo_b` (up to hidden) into `out`. Row-parallel — the all-reduce-sum is the
/// caller's concern. Decode (token_count==1) and prefill (token_count>1) DeepGEMM
/// paths + the scalar fallback are preserved byte-for-byte; batched decode passes
/// token_count=N to hit the prefill DeepGEMM branch (M=N amortizes the wo weight read).
#[allow(clippy::too_many_arguments)]
pub(crate) fn mla_oproj(
    ctx: &DeviceContext,
    attention: &Dsv4Attention,
    state: &mut Dsv4LayerAttentionState,
    // Model-wide shared FP8 prefill DeepGEMM linear scratch (hoisted off the
    // per-slot state). `Some` only when native DeepGEMM is available and the caller
    // threads it; the prefill wo_a/wo_b DeepGEMM lane (token_count>1)
    // gates on it. `None` on the decode (token_count==1) graph/batched lanes.
    mut prefill_shared: Option<&mut Dsv4PrefillDeepGemmLinearScratch>,
    local_attn: &HiddenStates,
    token_count: usize,
    keepalive: &mut Dsv4ForwardKeepalive,
    out: &mut HiddenStates,
) -> Result<()> {
    // GLM plain output projection: a single GEMM v[heads*v_head_dim] -> hidden.
    // No wo_a/wo_b low-rank, no group tables. `local_attn` for GLM is the
    // post-w_vc v ([heads*v_head_dim]); o_proj is [hidden, heads*v_head_dim].
    // DSv4 (o_proj None) falls through to the wo_a/wo_b path UNCHANGED.
    // ponytail: pod-verify plain o_proj input is post-w_vc v (heads*v_head_dim)
    if let Some(o_proj) = attention.o_proj.as_ref() {
        let _ = &mut prefill_shared;
        let _ = keepalive;
        crate::profile::profile_op(ctx, "linear/o_proj", None, token_count, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/o_proj", || {
                dsv4_linear(ctx, o_proj, local_attn, out)
            })
        })?;
        return Ok(());
    }
    let shape = dsv4_oproj_group_shape(
        attention.wo_a.as_ref().expect("DSv4 wo_a").rows,
        attention.wo_a.as_ref().expect("DSv4 wo_a").cols,
        attention
            .wo_a_groups
            .as_ref()
            .expect("DSv4 wo_a_groups")
            .groups,
        attention
            .wo_a_groups
            .as_ref()
            .expect("DSv4 wo_a_groups")
            .rows_per_group,
        attention
            .wo_a_groups
            .as_ref()
            .expect("DSv4 wo_a_groups")
            .cols_per_group,
        local_attn.hidden_dim,
        token_count,
    )?;
    // SAFETY: uninit device scratch; fully written before first read.
    let mut latent = unsafe {
        HiddenStates::uninit(
            ctx,
            attention.wo_a.as_ref().expect("DSv4 wo_a").rows,
            token_count,
        )?
    };
    // Decode (no prefill scratch) vs prefill — captured before any branch consumes
    // `prefill_shared` (the wo_b prefill lane moves it via .expect()). Drives the
    // batched-O-LoRA active_counts restore at the end.
    let is_decode = prefill_shared.is_none();
    // Single-group decode DeepGEMM (lever #1b) is now M-parametric: token_count==1
    // is the per-row decode lane (byte-identical to the original M=1 path), and
    // token_count==n is the batched FINISH O-LoRA (token-major [n, cols] input).
    let wo_a_decode_dg = shape.groups == 1
        && dsv4_decode_proj_deepgemm_enabled()
        && state.fused_wqkv.is_some()
        && attention.wo_a_deepgemm.is_some()
        && prefill_shared.is_none();
    let wo_a_prefill_dg = shape.groups == 1
        && token_count > 1
        && dsv4_prefill_proj_deepgemm_enabled()
        && attention.wo_a_deepgemm.is_some()
        && prefill_shared.is_some();
    // Grouped decode DeepGEMM is now M-parametric (gather/GEMM/scatter per group
    // over n rows), so the batched FINISH routes groups>1 here at M=n directly —
    // one grouped-GEMM per group, no per-row loop. Decode lane only (no prefill
    // scratch); grouped prefill stays on its own gate below.
    let wo_a_group_decode_dg = shape.groups > 1
        && dsv4_decode_proj_deepgemm_enabled()
        && state.fused_wqkv.is_some()
        && attention.wo_a_group_deepgemm.is_some()
        && prefill_shared.is_none();
    let wo_a_group_prefill_dg = shape.groups > 1
        && token_count > 1
        && dsv4_prefill_proj_deepgemm_enabled()
        && attention.wo_a_group_deepgemm.is_some()
        && prefill_shared.is_some();
    if wo_a_decode_dg {
        // Lever #1b: wo_a through tensor-core DeepGEMM (M=token_count), reusing the
        // fused-wqkv FP8 scratch for the single-output-group case.
        let scratch = state.fused_wqkv.as_mut().expect("wo_dg gate checked");
        let wo_a_cache = attention
            .wo_a_deepgemm
            .as_ref()
            .expect("wo_dg gate checked");
        let wo_a_cols = attention.wo_a.as_ref().expect("DSv4 wo_a").cols;
        crate::profile::profile_op(ctx, "linear/wo_a", None, token_count, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wo_a", || {
                decode_proj_deepgemm(ctx, scratch, wo_a_cache, local_attn, &mut latent, wo_a_cols)
            })
        })?;
    } else if wo_a_group_decode_dg {
        let scratch = state
            .fused_wqkv
            .as_mut()
            .expect("wo grouped dg gate checked");
        let wo_a_caches = attention
            .wo_a_group_deepgemm
            .as_ref()
            .expect("wo grouped dg gate checked");
        crate::profile::profile_op(ctx, "linear/wo_a", None, token_count, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wo_a", || {
                dsv4_wo_a_grouped_deepgemm_decode(
                    ctx,
                    scratch,
                    wo_a_caches,
                    local_attn,
                    shape,
                    &mut latent,
                )
            })
        })?;
    } else if wo_a_prefill_dg {
        // Prefill wo_a (M=token_count) → DeepGEMM for the single-output-group case.
        let wo_a_cache = attention
            .wo_a_deepgemm
            .as_ref()
            .expect("wo prefill gate checked");
        crate::profile::profile_op(ctx, "linear/wo_a", None, token_count, || {
            let scratch = prefill_shared
                .as_deref_mut()
                .expect("wo prefill gate checked");
            crate::linear_profile::profile(ctx, "dsv4/linear/wo_a", || {
                prefill_proj_deepgemm(ctx, scratch, wo_a_cache, local_attn, &mut latent)
            })
        })?;
    } else if wo_a_group_prefill_dg {
        let wo_a_caches = attention
            .wo_a_group_deepgemm
            .as_ref()
            .expect("wo grouped prefill gate checked");
        crate::profile::profile_op(ctx, "linear/wo_a", None, token_count, || {
            let scratch = prefill_shared
                .as_deref_mut()
                .expect("wo grouped prefill gate checked");
            crate::linear_profile::profile(ctx, "dsv4/linear/wo_a", || {
                dsv4_wo_a_grouped_deepgemm_prefill(
                    ctx,
                    scratch,
                    wo_a_caches,
                    local_attn,
                    shape,
                    &mut latent,
                )
            })
        })?;
    } else if shape.groups == 1 {
        crate::profile::profile_op(ctx, "linear/wo_a", None, token_count, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wo_a", || {
                dsv4_linear(
                    ctx,
                    attention.wo_a.as_ref().expect("DSv4 wo_a"),
                    local_attn,
                    &mut latent,
                )
            })
        })?;
    } else {
        crate::profile::profile_op(ctx, "linear/wo_a", None, token_count, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wo_a", || {
                dsv4_wo_a_grouped_linear(ctx, attention, local_attn, shape, &mut latent)
            })
        })?;
    }
    keepalive.keep_hidden(&latent);

    // wo_b is always a single-group [hidden, o_lora_rank] GEMM, so its decode
    // DeepGEMM lane is M-parametric like wo_a: M=token_count (1 per-row, n batched).
    let wo_b_decode_dg = dsv4_decode_proj_deepgemm_enabled()
        && state.fused_wqkv.is_some()
        && attention.wo_b_deepgemm.is_some()
        && prefill_shared.is_none();
    let wo_b_prefill_dg = token_count > 1
        && dsv4_prefill_proj_deepgemm_enabled()
        && attention.wo_b_deepgemm.is_some()
        && prefill_shared.is_some();
    if wo_b_decode_dg {
        let scratch = state.fused_wqkv.as_mut().expect("wo_b dg gate checked");
        let wo_b_cache = attention
            .wo_b_deepgemm
            .as_ref()
            .expect("wo_b dg gate checked");
        let wo_b_cols = attention.wo_b.as_ref().expect("DSv4 wo_b").cols;
        crate::profile::profile_op(ctx, "linear/wo_b", None, token_count, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wo_b", || {
                decode_proj_deepgemm(ctx, scratch, wo_b_cache, &latent, out, wo_b_cols)
            })
        })?;
    } else if wo_b_prefill_dg {
        let wo_b_cache = attention
            .wo_b_deepgemm
            .as_ref()
            .expect("wo_b prefill gate checked");
        crate::profile::profile_op(ctx, "linear/wo_b", None, token_count, || {
            let scratch = prefill_shared.expect("wo_b prefill gate checked");
            crate::linear_profile::profile(ctx, "dsv4/linear/wo_b", || {
                prefill_proj_deepgemm(ctx, scratch, wo_b_cache, &latent, out)
            })
        })?;
    } else {
        crate::profile::profile_op(ctx, "linear/wo_b", None, token_count, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wo_b", || {
                dsv4_linear(
                    ctx,
                    attention.wo_b.as_ref().expect("DSv4 wo_b"),
                    &latent,
                    out,
                )
            })
        })?;
    }
    // Restore the shared fused-wqkv scratch active_counts to [1] after a batched
    // (M=n) decode-DeepGEMM O-LoRA. The single-group wo_a/wo_b decode lanes write
    // active_counts=[n] when token_count>1; every per-row M=1 reader (the next
    // layer's wq decode, the per-row grouped fallback) relies on it being [1].
    // No-op (and no H2D) at token_count==1 — preserves byte+launch identity.
    if token_count > 1
        && is_decode
        && let Some(scratch) = state.fused_wqkv.as_mut()
    {
        ctx.stream
            .memcpy_htod(&[1_i32], &mut scratch.active_counts)
            .map_err(|e| anyhow!("DSv4 batched O-LoRA active_counts restore failed: {e}"))?;
    }
    Ok(())
}

fn mla_oproj_decode_graph(
    ctx: &DeviceContext,
    attention: &Dsv4Attention,
    state: &mut Dsv4LayerAttentionState,
    local_attn: &HiddenStates,
    latent: &mut HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        attention.o_proj.is_none(),
        "DSv4 decode graph O projection is MODEL1-only; GLM/plain-o uses eager decode"
    );
    let token_count = 1usize;
    let shape = dsv4_oproj_group_shape(
        attention.wo_a.as_ref().expect("DSv4 wo_a").rows,
        attention.wo_a.as_ref().expect("DSv4 wo_a").cols,
        attention
            .wo_a_groups
            .as_ref()
            .expect("DSv4 wo_a_groups")
            .groups,
        attention
            .wo_a_groups
            .as_ref()
            .expect("DSv4 wo_a_groups")
            .rows_per_group,
        attention
            .wo_a_groups
            .as_ref()
            .expect("DSv4 wo_a_groups")
            .cols_per_group,
        local_attn.hidden_dim,
        token_count,
    )?;
    ensure!(
        latent.hidden_dim == attention.wo_a.as_ref().expect("DSv4 wo_a").rows
            && latent.seq_len == token_count,
        "DSv4 graph O-LoRA latent scratch {}x{} != {}x1",
        latent.hidden_dim,
        latent.seq_len,
        attention.wo_a.as_ref().expect("DSv4 wo_a").rows
    );
    let wo_a_decode_dg = shape.groups == 1
        && dsv4_decode_proj_deepgemm_enabled()
        && state.fused_wqkv.is_some()
        && attention.wo_a_deepgemm.is_some();
    let wo_a_group_decode_dg = shape.groups > 1
        && dsv4_decode_proj_deepgemm_enabled()
        && state.fused_wqkv.is_some()
        && attention.wo_a_group_deepgemm.is_some();
    if wo_a_decode_dg {
        let wo_a_cache = attention
            .wo_a_deepgemm
            .as_ref()
            .expect("wo_dg gate checked");
        let wo_a_cols = attention.wo_a.as_ref().expect("DSv4 wo_a").cols;
        let scratch = state.fused_wqkv.as_mut().expect("wo_dg gate checked");
        crate::profile::profile_op(ctx, "linear/wo_a", None, 1, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wo_a", || {
                decode_proj_deepgemm(ctx, scratch, wo_a_cache, local_attn, latent, wo_a_cols)
            })
        })?;
    } else if wo_a_group_decode_dg {
        let wo_a_caches = attention
            .wo_a_group_deepgemm
            .as_ref()
            .expect("wo grouped dg gate checked");
        let scratch = state
            .fused_wqkv
            .as_mut()
            .expect("wo grouped dg gate checked");
        crate::profile::profile_op(ctx, "linear/wo_a", None, 1, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wo_a", || {
                dsv4_wo_a_grouped_deepgemm_decode(
                    ctx,
                    scratch,
                    wo_a_caches,
                    local_attn,
                    shape,
                    latent,
                )
            })
        })?;
    } else if shape.groups == 1 {
        crate::profile::profile_op(ctx, "linear/wo_a", None, 1, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wo_a", || {
                dsv4_linear(
                    ctx,
                    attention.wo_a.as_ref().expect("DSv4 wo_a"),
                    local_attn,
                    latent,
                )
            })
        })?;
    } else {
        crate::profile::profile_op(ctx, "linear/wo_a", None, 1, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wo_a", || {
                dsv4_wo_a_grouped_linear(ctx, attention, local_attn, shape, latent)
            })
        })?;
    }

    let wo_b_decode_dg = dsv4_decode_proj_deepgemm_enabled()
        && state.fused_wqkv.is_some()
        && attention.wo_b_deepgemm.is_some();
    if wo_b_decode_dg {
        let wo_b_cache = attention
            .wo_b_deepgemm
            .as_ref()
            .expect("wo_b dg gate checked");
        let wo_b_cols = attention.wo_b.as_ref().expect("DSv4 wo_b").cols;
        let scratch = state.fused_wqkv.as_mut().expect("wo_b dg gate checked");
        crate::profile::profile_op(ctx, "linear/wo_b", None, 1, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wo_b", || {
                decode_proj_deepgemm(ctx, scratch, wo_b_cache, latent, out, wo_b_cols)
            })
        })?;
    } else {
        crate::profile::profile_op(ctx, "linear/wo_b", None, 1, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wo_b", || {
                dsv4_linear(
                    ctx,
                    attention.wo_b.as_ref().expect("DSv4 wo_b"),
                    latent,
                    out,
                )
            })
        })?;
    }
    Ok(())
}

/// GLM SparseIndexed only: build the full-sequence index KEY ring (ratio=1, no
/// compressor). For each of `hidden`'s tokens, project a single MQA index key via
/// `indexer.wk`, RMSNorm it with `indexer.k_norm` (width `index_head_dim`), and
/// append the `[index_head_dim, seq_len]` normed keys into `state.compressed` at
/// absolute rows `[start_pos .. start_pos + seq_len)`. This mirrors what
/// `compressor_forward` does for CSA but with NO compression — every token is one
/// key, exactly what `csa_select_official` reads (`keys = [index_head_dim, rows]`,
/// Hadamard-rotated then FP8-stored downstream).
fn sparse_indexed_index_key_forward(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    indexer: &Dsv4Indexer,
    hidden: &HiddenStates,
    state: &mut Dsv4CompressorState,
    start_pos: usize,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<()> {
    let wk = indexer.wk.as_ref().ok_or_else(|| {
        anyhow!("DSv4 SparseIndexed layer indexer missing wk projection (GLM tranche-C weight)")
    })?;
    let k_norm = indexer.k_norm.as_ref().ok_or_else(|| {
        anyhow!("DSv4 SparseIndexed layer indexer missing k_norm (GLM tranche-C weight)")
    })?;
    let seq_len = hidden.seq_len;
    // Project the index key(s): [wk.rows, seq_len].
    // SAFETY: dsv4_linear writes the full wk_out buffer.
    let mut wk_out = unsafe { HiddenStates::uninit(ctx, wk.rows, seq_len)? };
    dsv4_linear(ctx, wk, hidden, &mut wk_out)?;
    keepalive.keep_hidden(&wk_out);

    // The index KEY ring stores ONE key of width `index_head_dim` per token. The
    // DeepSeek-V3.2 lightning indexer key is a SINGLE MQA key per token. GLM's
    // `wk` doc says `[index_n_heads*index_head_dim, hidden]`. The width reduction
    // (index_n_heads*index_head_dim → index_head_dim) is the load-bearing
    // unverifiable detail and is handled by branching on `wk.rows`.
    let normed = if wk.rows == config.index_head_dim {
        // Single-head MQA key (DSv3.2/GLM-DSA `wk = Linear(hidden, index_head_dim)`,
        // ONE key shared across all index_n_heads query heads — confirmed against the
        // vLLM DeepSeek-V3.2 indexer reference: wk out = head_dim, not n_head*head_dim).
        // This is the branch GLM takes (index_head_dim=128). Normalize the key here.
        //
        // NUMERIC GAP (pod-verify): the DSv3.2 reference normalizes the index key with
        // `LayerNorm(index_head_dim, eps=1e-6)` (mean-subtract + variance + the loaded
        // `k_norm` weight AND `k_norm_bias`). GLM ships a `k_norm.bias`, which implies
        // LayerNorm, not RMSNorm. This path applies the bias-free `mla_rms_norm` with
        // `config.rms_norm_eps` instead — a correctness approximation that must be
        // replaced with a LayerNorm(+bias, eps=1e-6) kernel once a GPU forward confirms
        // GLM's exact index-key norm. `k_norm_bias` is intentionally still consumed
        // below (kept live) so the wiring is in place for that fix.
        // ponytail: pod-verify GLM index k_norm = LayerNorm(eps=1e-6) with k_norm weight+bias — current path is bias-free RMSNorm
        let _ = indexer.k_norm_bias.as_ref();
        mla_rms_norm(ctx, &wk_out, k_norm, config.rms_norm_eps)?
    } else {
        // GLM's real `wk` is `[index_head_dim, hidden]` (single MQA key), so this
        // branch is not expected. Fail loud rather than fabricate a per-head→single-key
        // reduction the official scorer never expects.
        // ponytail: pod-verify GLM wk index-key width if this ever fires (expected wk.rows == index_head_dim)
        bail!(
            "DSv4 SparseIndexed index-key build: wk.rows {} != index_head_dim {} — GLM's \
             lightning-indexer key is a single MQA head of width index_head_dim; a wider wk \
             means an unexpected checkpoint layout. Pod-verify GLM wk before enabling decode",
            wk.rows,
            config.index_head_dim
        );
    };
    keepalive.keep_hidden(&normed);
    ensure!(
        normed.hidden_dim == config.index_head_dim,
        "DSv4 SparseIndexed normed index key width {} != index_head_dim {}",
        normed.hidden_dim,
        config.index_head_dim
    );
    ensure!(
        state.compressed.hidden_dim == config.index_head_dim,
        "DSv4 SparseIndexed indexer state hidden_dim {} != index_head_dim {}",
        state.compressed.hidden_dim,
        config.index_head_dim
    );

    // Stage this forward's delta into the STAGING RING. The ring holds only the
    // live delta — `csa_select_official` drains [packed_rows..rows_after) into
    // the DSA pools in the SAME forward (drain-immediate), so the full history
    // never lives here. Each forward stages its `seq_len` rows CONTIGUOUSLY from
    // ring row 0 (window base = this forward's `start_pos`); the drain recovers
    // row `r` at ring offset `r - start_pos`. Staging from 0 (NOT
    // `start_pos % ring_rows`) keeps the delta contiguous for ANY start_pos —
    // decode/MTP start_pos is not chunk-aligned, so a modulo base could straddle
    // the wrap, but the offset-0 window never does. The LOGICAL committed count
    // is `compressed.seq_len`, bounded by the logical capacity. Frozen-KV verify
    // never advances state.
    ensure!(
        start_pos + seq_len <= state.compressed_capacity,
        "DSv4 SparseIndexed index-key logical overflow: start_pos {start_pos} + seq_len {seq_len} \
         exceeds capacity {}",
        state.compressed_capacity
    );
    ensure!(
        seq_len <= state.ring_rows,
        "DSv4 SparseIndexed index-key forward delta {seq_len} exceeds staging-ring depth {} \
         (one forward must fit contiguously; raise DSV4_INDEXER_STAGING_RING_ROWS)",
        state.ring_rows
    );
    if !dsv4_verify_frozen() {
        // Window-relative: ring rows [0, seq_len) for absolute [start_pos, start_pos+seq_len).
        let hi = seq_len * config.index_head_dim;
        let mut dst = state.compressed.data.slice_mut(0..hi);
        ctx.stream
            .memcpy_dtod(&normed.data, &mut dst)
            .map_err(|e| anyhow!("DSv4 SparseIndexed index-key ring D2D failed: {e}"))?;
        state.compressed.seq_len = start_pos + seq_len;
    }
    Ok(())
}

/// FP32 main-value compressor. Re-runs the compressor forward in
/// FP32 — BF16 input projections via cuBLASLt FP32-accumulate GEMM, FP32 APE, FP32
/// state carry — to avoid BF16/FP8 value mismatches (#146, #150). Writes the BF16
/// overlap + pending carry mirrors and the compressed output back into `state`
/// so the downstream attention AND the bf16 decode lane read the FP32 values.
///
/// Runs on every prefill compression boundary (any `start_pos`, with prior
/// compressed state); the decode fast path is unchanged. Carry coherence: when
/// `fp32_carry_stale` (a bf16-lane update / prefix restore / reset advanced the
/// bf16 carry since the last probe), the bf16 carry is the authority — reseed
/// FP32 from it before the probe reads pending/prev.
#[allow(clippy::too_many_arguments)]
fn compressor_fp32_probe(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    compressor: &Dsv4Compressor,
    hidden: &HiddenStates,
    state: &mut Dsv4CompressorState,
    scratch: &mut Dsv4CompressorFp32Scratch,
    head_dim: usize,
    ratio: usize,
    width: usize,
    overlap: bool,
    start_pos: usize,
    token_count: usize,
    compressed_rows: usize,
    apply_rope: bool,
    rope_original_seq_len: i32,
) -> Result<()> {
    crate::profile::profile_op(ctx, "compressor_fp32_probe", None, token_count, || {
        let probe = &compressor.fp32_probe;
        ensure!(
            probe.wkv.cols == hidden.hidden_dim && probe.wgate.cols == hidden.hidden_dim,
            "DSv4 compressor FP32 projection K mismatch"
        );
        ensure!(
            probe.ape.len() == ratio * width,
            "DSv4 compressor FP32 APE len {} != ratio*width {}",
            probe.ape.len(),
            ratio * width
        );
        // Model-wide shared FP32 GEMM scratch (hoisted off the per-slot state);
        // written and consumed within this call, no cross-call state.
        ensure!(
            token_count * width <= scratch.kv_raw.len(),
            "DSv4 compressor FP32 scratch too small: token_count {token_count} × width {width} \
             exceeds {}",
            scratch.kv_raw.len()
        );
        if state.fp32_carry_stale {
            let pending_elems = i32::try_from(state.fp32_pending_kv.len())
                .map_err(|_| anyhow!("DSv4 FP32 carry pending elems exceed i32"))?;
            let prev_elems = i32::try_from(state.fp32_prev_kv.len())
                .map_err(|_| anyhow!("DSv4 FP32 carry prev elems exceed i32"))?;
            let (pkv_b, _b0) = state.pending_kv.device_ptr(&ctx.stream);
            let (psc_b, _b1) = state.pending_score.device_ptr(&ctx.stream);
            let (prkv_b, _b2) = state.prev_overlap_kv.device_ptr(&ctx.stream);
            let (prsc_b, _b3) = state.prev_overlap_score.device_ptr(&ctx.stream);
            let (pkv, _f0) = state.fp32_pending_kv.device_ptr_mut(&ctx.stream);
            let (psc, _f1) = state.fp32_pending_score.device_ptr_mut(&ctx.stream);
            let (prkv, _f2) = state.fp32_prev_kv.device_ptr_mut(&ctx.stream);
            let (prsc, _f3) = state.fp32_prev_score.device_ptr_mut(&ctx.stream);
            // SAFETY: bf16/f32 carry buffers are allocated with identical element
            // counts (`Dsv4CompressorState::new`).
            unsafe {
                ffi::dsv4_compressor_fp32_carry_reseed_cuda(
                    pkv_b as *const ffi::Half,
                    psc_b as *const ffi::Half,
                    prkv_b as *const ffi::Half,
                    prsc_b as *const ffi::Half,
                    pkv as *mut f32,
                    psc as *mut f32,
                    prkv as *mut f32,
                    prsc as *mut f32,
                    pending_elems,
                    prev_elems,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
            state.fp32_carry_stale = false;
        }
        let kv_raw = &mut scratch.kv_raw;
        let score_raw = &mut scratch.score_raw;
        let pending_kv = &mut state.fp32_pending_kv;
        let pending_score = &mut state.fp32_pending_score;
        let prev_kv = &mut state.fp32_prev_kv;
        let prev_score = &mut state.fp32_prev_score;
        {
            let (wkv, _wg0) = probe.wkv.data.device_ptr(&ctx.stream);
            let (wgate, _wg1) = probe.wgate.data.device_ptr(&ctx.stream);
            let (x, _xg) = hidden.data.device_ptr(&ctx.stream);
            let (kv, _kg) = kv_raw.device_ptr_mut(&ctx.stream);
            let (score, _sg) = score_raw.device_ptr_mut(&ctx.stream);
            // SAFETY: dense BF16 matrices and outputs match the checked M/N/K shapes.
            unsafe {
                ffi::gemm_bf16_f32_cuda(
                    wkv as *const ffi::Half,
                    x as *const ffi::Half,
                    kv as *mut f32,
                    width as i32,
                    token_count as i32,
                    hidden.hidden_dim as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
                ffi::gemm_bf16_f32_cuda(
                    wgate as *const ffi::Half,
                    x as *const ffi::Half,
                    score as *mut f32,
                    width as i32,
                    token_count as i32,
                    hidden.hidden_dim as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }
        let rope = &config.rope_parameters;
        let (rope_dim, rope_base) = if apply_rope {
            (config.qk_rope_head_dim, config.compress_rope_theta)
        } else {
            (0, config.compress_rope_theta)
        };
        let start_pos_i32 = i32::try_from(start_pos).map_err(|_| {
            anyhow::anyhow!("DSv4 FP32 compressor start_pos {start_pos} exceeds i32")
        })?;
        let pending_len = start_pos % ratio;
        let pending_len_i32 = i32::try_from(pending_len).map_err(|_| {
            anyhow::anyhow!("DSv4 FP32 compressor pending_len {pending_len} exceeds i32")
        })?;
        let compressed_base = start_pos / ratio;
        let compressed_base_i32 = i32::try_from(compressed_base).map_err(|_| {
            anyhow::anyhow!("DSv4 FP32 compressor compressed_base {compressed_base} exceeds i32")
        })?;
        let has_prev_overlap = i32::from(compressed_base > 0);
        let overlap_page_stride = 0i32;
        {
            let (kv, _kg) = kv_raw.device_ptr(&ctx.stream);
            let (score, _sg) = score_raw.device_ptr(&ctx.stream);
            let (ape, _ag) = probe.ape.device_ptr(&ctx.stream);
            let (norm, _ng) = compressor.norm.data.device_ptr(&ctx.stream);
            let (pkv, _p0) = pending_kv.device_ptr_mut(&ctx.stream);
            let (psc, _p1) = pending_score.device_ptr_mut(&ctx.stream);
            let (prkv, _p2) = prev_kv.device_ptr_mut(&ctx.stream);
            let (prsc, _p3) = prev_score.device_ptr_mut(&ctx.stream);
            let (prkv_bf16, _p4) = state.prev_overlap_kv.device_ptr_mut(&ctx.stream);
            let (prsc_bf16, _p5) = state.prev_overlap_score.device_ptr_mut(&ctx.stream);
            let (pkv_bf16, _p6) = state.pending_kv.device_ptr_mut(&ctx.stream);
            let (psc_bf16, _p7) = state.pending_score.device_ptr_mut(&ctx.stream);
            let (compressed, _cg) = state.compressed.data.device_ptr_mut(&ctx.stream);
            // SAFETY: all buffers match the checked ratio, width, and token count.
            unsafe {
                ffi::dsv4_compressor_fp32_prefill_probe_cuda(
                    kv as *const f32,
                    score as *const f32,
                    ape as *const f32,
                    norm as *const ffi::Half,
                    pkv as *mut f32,
                    psc as *mut f32,
                    prkv as *mut f32,
                    prsc as *mut f32,
                    prkv_bf16 as *mut ffi::Half,
                    prsc_bf16 as *mut ffi::Half,
                    pkv_bf16 as *mut ffi::Half,
                    psc_bf16 as *mut ffi::Half,
                    compressed as *mut ffi::Half,
                    token_count as i32,
                    start_pos_i32,
                    pending_len_i32,
                    compressed_base_i32,
                    head_dim as i32,
                    ratio as i32,
                    width as i32,
                    i32::from(overlap),
                    has_prev_overlap,
                    overlap_page_stride,
                    config.rms_norm_eps,
                    rope_dim as i32,
                    rope_base,
                    rope_original_seq_len,
                    rope.factor,
                    rope.beta_fast,
                    rope.beta_slow,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }
        state.compressed.seq_len = compressed_rows;
        Ok(())
    })?;
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
    rope_original_seq_len: i32,
    // Model-wide shared FP32 probe scratch: required (`Some`) on every prefill
    // lane (`start_pos_device` None — the probe branch consumes it, fail loud);
    // decode lanes never reach the probe and pass `None`.
    fp32_scratch: Option<&mut Dsv4CompressorFp32Scratch>,
    // Batched decode pre-pass: when `Some`, the two m=1
    // `dsv4_linear` projection GEMVs are SKIPPED and the passed
    // `(kv_raw, score_raw)` — this row's `[width, 1]` slices of the batched
    // batched `dsv4_linear` outputs — are used directly. Everything downstream
    // (the per-slot `dsv4_compressor_update_*` FFI + state advance) is
    // UNCHANGED. `None` → per-row GEMV path.
    precomputed: Option<(&HiddenStates, &HiddenStates)>,
    // Full-flatten decode: when `Some`, the per-row
    // `dsv4_compressor_update_*` FFI is SKIPPED and this row's five ring-state
    // device pointers are pushed into the batch sink instead — the actual state
    // update runs later in ONE `dsv4_compressor_update_batched` over all N rows.
    // `compressed.seq_len` IS still advanced here (host bookkeeping; the batched
    // kernel writes the data before any reader runs, in the P2 loop). Requires the
    // start_pos_ptr path (decode); the batched kernel resolves pending/base per row
    // from start_pos exactly like the per-row start_pos_ptr launcher. `None` → the
    // per-row FFI runs (unchanged).
    defer_update: Option<&mut Dsv4CompressorBatchPtrs>,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<()> {
    ensure!(ratio > 0, "DSv4 compressor ratio must be non-zero");
    let width = dsv4_compressor_width(head_dim, overlap);
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

    // FP32 probe: prefill only (start_pos_device None). The #146/#150 corruption
    // was a multi-token prefill boundary issue; decode uses the BF16 path.
    if !dsv4_verify_frozen() && token_count > 0 && start_pos_device.is_none() {
        let scratch = fp32_scratch.ok_or_else(|| {
            anyhow!("DSv4 FP32 compressor probe needs the model-wide scratch on prefill lanes")
        })?;
        compressor_fp32_probe(
            ctx,
            config,
            compressor,
            hidden,
            state,
            scratch,
            head_dim,
            ratio,
            width,
            overlap,
            start_pos,
            token_count,
            compressed_rows,
            apply_rope,
            rope_original_seq_len,
        )?;
        return Ok(());
    }

    // Full-flatten decode (defer_update Some): skip BOTH the per-row projection
    // GEMVs (already batched in the prepass; the batched update reads that output
    // directly) AND the per-row state-update FFI — instead push this row's five
    // ring-state device pointers into the batch sink and advance
    // `compressed.seq_len` (host bookkeeping). The actual GPU state write runs
    // later in ONE `dsv4_compressor_update_batched` over all N rows, BEFORE any
    // reader (the P1b pack / csa cache-write). MUST early-return before the GEMV
    // match below (else the ∝n GEMVs this batched path replaces would still run).
    // Decode-only: token_count==1; requires the start_pos_ptr path.
    if let Some(sink) = defer_update {
        ensure!(
            start_pos_device.is_some(),
            "DSv4 full-flatten compressor defer requires start_pos_device (decode path)"
        );
        ensure!(
            token_count == 1,
            "DSv4 full-flatten compressor defer is decode-only (token_count must be 1, got {token_count})"
        );
        // Frozen-KV verify never advances state (P1-1) — same guard as the FFI path.
        if !dsv4_verify_frozen() {
            sink.push(state.batched_update_ptrs(ctx));
            state.compressed.seq_len = compressed_rows;
            state.fp32_carry_stale = true;
        }
        return Ok(());
    }

    // SOURCE of kv_raw/score_raw: either this row's `[width, 1]` slice of the
    // batched pre-pass (precomputed `Some`) or the per-row m=1 `dsv4_linear`
    // GEMVs (precomputed `None`).
    // `owned_*` hold the GEMV-produced buffers in the `None` branch (declared in
    // the outer scope so they outlive the FFI read below); `kv_raw` / `score_raw`
    // reference whichever source is active. The downstream
    // `dsv4_compressor_update_*` FFI reads these by device pointer and is
    // identical for both sources (only the numerics shift, FP8-DeepGEMM vs
    // bf16-GEMV, which is needle-gated).
    let mut owned_kv: Option<HiddenStates> = None;
    let mut owned_score: Option<HiddenStates> = None;
    let (kv_raw, score_raw): (&HiddenStates, &HiddenStates) = match precomputed {
        Some((kv_precomputed, score_precomputed)) => {
            ensure!(
                kv_precomputed.hidden_dim == width
                    && kv_precomputed.seq_len == token_count
                    && score_precomputed.hidden_dim == width
                    && score_precomputed.seq_len == token_count,
                "DSv4 compressor precomputed slice shape mismatch: kv={}x{} score={}x{} expected [{width},{token_count}]",
                kv_precomputed.hidden_dim,
                kv_precomputed.seq_len,
                score_precomputed.hidden_dim,
                score_precomputed.seq_len
            );
            (kv_precomputed, score_precomputed)
        }
        None => {
            // SAFETY: dsv4_linear writes the full compressor kv buffer.
            let mut kv = unsafe { HiddenStates::uninit(ctx, width, token_count)? };
            crate::profile::profile_op(ctx, "linear/compressor_wkv", None, token_count, || {
                crate::linear_profile::profile(ctx, "dsv4/linear/compressor_wkv", || {
                    dsv4_linear(ctx, &compressor.wkv, hidden, &mut kv)
                })
            })?;
            keepalive.keep_hidden(&kv);
            // SAFETY: dsv4_linear writes the full compressor score buffer.
            let mut score = unsafe { HiddenStates::uninit(ctx, width, token_count)? };
            crate::profile::profile_op(ctx, "linear/compressor_wgate", None, token_count, || {
                crate::linear_profile::profile(ctx, "dsv4/linear/compressor_wgate", || {
                    dsv4_linear(ctx, &compressor.wgate, hidden, &mut score)
                })
            })?;
            keepalive.keep_hidden(&score);
            (
                owned_kv.insert(kv) as &HiddenStates,
                owned_score.insert(score) as &HiddenStates,
            )
        }
    };

    let rope = &config.rope_parameters;
    // Compressed keys use compress_rope_theta with NO YaRN (original_seq_len = 0).
    let (rope_dim, rope_base) = if apply_rope {
        (config.qk_rope_head_dim, config.compress_rope_theta)
    } else {
        (0, config.compress_rope_theta)
    };
    // Raw bf16 read: on a quantized matrix `.data` is a 1-element dummy (bytes
    // live in qweight) — the #138 OOB class. The loader dequants ape to dense.
    ensure!(
        compressor.ape.is_dense_bf16(),
        "DSv4 compressor ape must be dense bf16 (raw-read by the update kernel); got {:?}",
        compressor.ape.weight_format
    );
    {
        let (kv_ptr, _kg) = kv_raw.data.device_ptr(&ctx.stream);
        let (score_ptr, _scg) = score_raw.data.device_ptr(&ctx.stream);
        let (ape_ptr, _ag) = compressor.ape.data.device_ptr(&ctx.stream);
        let (norm_ptr, _ng) = compressor.norm.data.device_ptr(&ctx.stream);
        let (pkv_ptr, _pkg) = state.pending_kv.device_ptr_mut(&ctx.stream);
        let (psc_ptr, _psg) = state.pending_score.device_ptr_mut(&ctx.stream);
        // #154: prev_overlap always resolves from `state`'s own per-slot
        // buffer; stride 0 collapses the kernel's indexing to the
        // single-register form (the shared pool was deleted with Route A).
        let (prkv_ptr, _kg2) = state.prev_overlap_kv.device_ptr_mut(&ctx.stream);
        let (prsc_ptr, _sg2) = state.prev_overlap_score.device_ptr_mut(&ctx.stream);
        let overlap_page_stride = 0i32;
        let (comp_ptr, _cg) = state.compressed.data.device_ptr_mut(&ctx.stream);
        let has_prev_overlap = i32::from(compressed_base > 0);
        // SAFETY: all buffers valid on ctx.stream; state carries the pending and
        // overlap rows from previous contiguous appends.
        if !dsv4_verify_frozen() {
            // SAFETY: ptrs from live device allocations sized to the dims passed.
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
                        overlap_page_stride,
                        config.rms_norm_eps,
                        rope_dim as i32,
                        rope_base,
                        rope_original_seq_len,
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
                        overlap_page_stride,
                        config.rms_norm_eps,
                        rope_dim as i32,
                        rope_base,
                        rope_original_seq_len,
                        rope.factor,
                        rope.beta_fast,
                        rope.beta_slow,
                        ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
            }
        }
    }
    // Frozen-KV (P1-1): a frozen verify SKIPS the compressor/indexer CUDA update
    // above, so it must NOT advance `compressed.seq_len` either — otherwise CSA /
    // FlashMLA in the same verify would attend to a compressed/indexer row whose
    // data was never produced, and `csa_select` would advance DSA `packed_rows`
    // off `indexer_rows_after`. Freezing the length keeps `indexer_rows_after ==
    // indexer_rows_before` so the whole compressed+sparse path stays frozen; the
    // accepted-prefix commit re-forward (non-frozen) advances it for real.
    if !dsv4_verify_frozen() {
        state.compressed.seq_len = compressed_rows;
        // bf16 lane advanced the carry; the next FP32 probe must reseed.
        state.fp32_carry_stale = true;
    }
    Ok(())
}

/// Batched (m=N) decode compressor/indexer projection pre-pass. Projects the N-row
/// post-attn-LN `normed_batch` [hidden_size, N] through `compressor.wkv` /
/// `compressor.wgate` ONCE via `dsv4_linear` (weight read once across all N rows),
/// returning `kv_raw_batch [width, N]` and `score_raw_batch [width, N]`
/// (`width = cache.rows`). Each row's `[width, 1]` column slice then feeds
/// [`compressor_forward`] as `precomputed`, replacing the per-row m=1
/// `dsv4_linear` GEMVs that re-read the full weight per row (bandwidth-bound,
/// zero amortization, ~54% of the decode step, DEAD-LINEAR in N).
///
/// Both outputs are fresh per-step allocations kept alive via
/// `keepalive.keep_hidden`. Mirrors `mla_attention_prepare_proj_batch`'s q/kv
/// pre-pass; touches NO slot state.
/// One batched (m=N) decode projection: tensor-core DeepGEMM when an FP8 repack
/// cache AND the shared prefill scratch are present (m>1 amortizes the weight
/// read), else the scalar `dsv4_linear` GEMV. The single routing point for every
/// compressor/indexer batch pre-pass projection.
fn proj_batched(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    cache: Option<&cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache>,
    scratch: Option<&mut Dsv4PrefillDeepGemmLinearScratch>,
    input: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    let force_bf16 = dsv4_proj_batched_bf16_forced()?;
    match (cache, scratch) {
        (Some(cache), Some(scratch)) if input.seq_len > 1 && !force_bf16 => {
            prefill_proj_deepgemm(ctx, scratch, cache, input, out)
        }
        _ => dsv4_linear(ctx, weight, input, out),
    }
}

pub(crate) fn compressor_batch_prepass(
    ctx: &DeviceContext,
    compressor: &Dsv4Compressor,
    normed_batch: &HiddenStates,
    mut scratch: Option<&mut Dsv4PrefillDeepGemmLinearScratch>,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<Option<(HiddenStates, HiddenStates)>> {
    // Batch the per-row compressor projections into ONE m=N GEMM each. `dsv4_linear`
    // dispatches per `weight_format`: DenseBf16 → gemm_cuda (cublasLtMatmul, weight
    // read ONCE for all N rows = amortized); Fp8BlockScaled → deepgemm. The DSv4
    // compressor weights are bf16, so this is the cublasLt path — the N per-row m=1
    // GEMVs (weight re-read N times, bandwidth-bound = the measured `perrow ∝ n`
    // bottleneck) collapse to one m=N GEMM (weight read once). No FP8 quant, so no
    // selection-shift correctness risk; mixed-precision-safe by construction.
    let n = normed_batch.seq_len;
    let width = compressor.wkv.rows;
    ensure!(
        compressor.wgate.rows == width,
        "DSv4 compressor batch pre-pass wkv/wgate row mismatch: wkv={width} wgate={}",
        compressor.wgate.rows
    );
    ensure!(
        compressor.wkv.cols == normed_batch.hidden_dim
            && compressor.wgate.cols == normed_batch.hidden_dim,
        "DSv4 compressor batch pre-pass K mismatch: wkv.cols={} wgate.cols={} normed hidden={}",
        compressor.wkv.cols,
        compressor.wgate.cols,
        normed_batch.hidden_dim
    );
    // SAFETY: dsv4_linear writes the full output buffer.
    let mut kv_raw_batch = unsafe { HiddenStates::uninit(ctx, width, n)? };
    crate::profile::profile_op(ctx, "linear/compressor_wkv_batched", None, n, || {
        crate::linear_profile::profile(ctx, "dsv4/linear/compressor_wkv_batched", || {
            proj_batched(
                ctx,
                &compressor.wkv,
                compressor.wkv_deepgemm.as_ref(),
                scratch.as_deref_mut(),
                normed_batch,
                &mut kv_raw_batch,
            )
        })
    })?;
    keepalive.keep_hidden(&kv_raw_batch);
    // SAFETY: dsv4_linear writes the full output buffer.
    let mut score_raw_batch = unsafe { HiddenStates::uninit(ctx, width, n)? };
    crate::profile::profile_op(ctx, "linear/compressor_wgate_batched", None, n, || {
        crate::linear_profile::profile(ctx, "dsv4/linear/compressor_wgate_batched", || {
            proj_batched(
                ctx,
                &compressor.wgate,
                compressor.wgate_deepgemm.as_ref(),
                scratch,
                normed_batch,
                &mut score_raw_batch,
            )
        })
    })?;
    keepalive.keep_hidden(&score_raw_batch);
    Ok(Some((kv_raw_batch, score_raw_batch)))
}

/// Batched (m=N) decode indexer-query / gating-weights projection pre-pass. The
/// per-row `csa_select` runs `dsv4_linear(indexer.wq_b, c_q_normed_row)` and
/// `dsv4_linear(indexer.weights_proj, normed_row)` as m=1 GEMVs that re-read the
/// full weight per decode row (bandwidth-bound, zero amortization, ∝n — the last
/// batchable GEMM left in the full-flatten prepare loop). Batch them into ONE m=N
/// GEMM each (`q_i_batch [wq_b.rows, N]`, `weights_batch [weights_proj.rows, N]`),
/// weight read once across the N-token grid. Each row's `[width,1]` column slice
/// then feeds `csa_select` as `query_precomputed`, skipping the per-row GEMVs;
/// the per-slot DSA cache writes + gathers stay per-row.
///
/// `c_q_normed_batch` is `[q_lora_rank, N]` (= `Dsv4MlaProjBatch::c_q_normed`);
/// `normed_batch` is `[hidden, N]`. Both outputs are fresh per-step allocations
/// kept alive via `keepalive.keep_hidden`. Mirrors `compressor_batch_prepass`;
/// touches NO slot state. DSv4 indexer weights are bf16 → cublasLt path, no FP8
/// quant, mixed-precision-safe by construction (same byte path as the per-row m=1
/// `dsv4_linear`, just one larger M).
pub(crate) fn indexer_query_batch_prepass(
    ctx: &DeviceContext,
    indexer: &Dsv4Indexer,
    c_q_normed_batch: &HiddenStates,
    normed_batch: &HiddenStates,
    mut scratch: Option<&mut Dsv4PrefillDeepGemmLinearScratch>,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<(HiddenStates, HiddenStates)> {
    let n = c_q_normed_batch.seq_len;
    ensure!(
        normed_batch.seq_len == n,
        "DSv4 indexer query batch pre-pass row mismatch: c_q={} normed={}",
        n,
        normed_batch.seq_len
    );
    ensure!(
        indexer.wq_b.cols == c_q_normed_batch.hidden_dim,
        "DSv4 indexer wq_b K mismatch: wq_b.cols={} c_q hidden={}",
        indexer.wq_b.cols,
        c_q_normed_batch.hidden_dim
    );
    ensure!(
        indexer.weights_proj.cols == normed_batch.hidden_dim,
        "DSv4 indexer weights_proj K mismatch: weights_proj.cols={} normed hidden={}",
        indexer.weights_proj.cols,
        normed_batch.hidden_dim
    );
    // SAFETY: dsv4_linear writes the full output buffer.
    let mut q_i_batch = unsafe { HiddenStates::uninit(ctx, indexer.wq_b.rows, n)? };
    crate::profile::profile_op(ctx, "linear/indexer_wq_b_batched", None, n, || {
        crate::linear_profile::profile(ctx, "dsv4/linear/indexer_wq_b_batched", || {
            proj_batched(
                ctx,
                &indexer.wq_b,
                indexer.wq_b_deepgemm.as_ref(),
                scratch.as_deref_mut(),
                c_q_normed_batch,
                &mut q_i_batch,
            )
        })
    })?;
    keepalive.keep_hidden(&q_i_batch);
    // SAFETY: dsv4_linear writes the full output buffer.
    let mut weights_batch = unsafe { HiddenStates::uninit(ctx, indexer.weights_proj.rows, n)? };
    crate::profile::profile_op(ctx, "linear/indexer_weights_batched", None, n, || {
        crate::linear_profile::profile(ctx, "dsv4/linear/indexer_weights_batched", || {
            proj_batched(
                ctx,
                &indexer.weights_proj,
                indexer.weights_proj_deepgemm.as_ref(),
                scratch,
                normed_batch,
                &mut weights_batch,
            )
        })
    })?;
    keepalive.keep_hidden(&weights_batch);
    Ok((q_i_batch, weights_batch))
}

/// Per-row precomputed indexer query/weights projection slices for the batched
/// decode pre-pass. `q_i` is this row's `[wq_b.rows, 1]` column VIEW of the
/// batched [`indexer_query_batch_prepass`] output; `weights` is the
/// `[weights_proj.rows, 1]` view. When threaded into [`csa_select`], they replace
/// the per-row m=1 `dsv4_linear` GEMVs AND the per-row D2D re-copy — the view
/// feeds the prepass column's device pointer straight into the gather (zero copy);
/// the per-slot cache writes + gathers stay per-row.
pub(crate) struct Dsv4IndexerQueryPrecomputed<'a> {
    pub(crate) q_i: HiddenStatesView<'a>,
    pub(crate) weights: HiddenStatesView<'a>,
}

/// Host-gathered per-row device pointer arrays for one compressor's batched
/// state update, uploaded to device `*_arr` arrays by
/// [`dsv4_compressor_update_batched`]. Built over the prepare loop's row order
/// (push exactly one entry per row); all five arrays hold per-row per-slot
/// buffers (#154: the shared compress-state pool was deleted with Route A).
#[derive(Default)]
pub(crate) struct Dsv4CompressorBatchPtrs {
    pub(crate) pending_kv: Vec<u64>,
    pub(crate) pending_score: Vec<u64>,
    pub(crate) prev_overlap_kv: Vec<u64>,
    pub(crate) prev_overlap_score: Vec<u64>,
    pub(crate) compressed: Vec<u64>,
}

impl Dsv4CompressorBatchPtrs {
    pub(crate) fn with_capacity(n: usize) -> Self {
        Self {
            pending_kv: Vec::with_capacity(n),
            pending_score: Vec::with_capacity(n),
            prev_overlap_kv: Vec::with_capacity(n),
            prev_overlap_score: Vec::with_capacity(n),
            compressed: Vec::with_capacity(n),
        }
    }

    /// Push one row's five state pointers (resolved via
    /// [`Dsv4CompressorState::batched_update_ptrs`]).
    pub(crate) fn push(&mut self, ptrs: (u64, u64, u64, u64, u64)) {
        self.pending_kv.push(ptrs.0);
        self.pending_score.push(ptrs.1);
        self.prev_overlap_kv.push(ptrs.2);
        self.prev_overlap_score.push(ptrs.3);
        self.compressed.push(ptrs.4);
    }

    fn len(&self) -> usize {
        self.pending_kv.len()
    }
}

/// Host-gathered per-slot device arrays for the batched DSA cache-write (block
/// (a) of [`csa_select_official`]): the Hadamard rotate of each slot's
/// newly-packed index-key rows + the FP8 fused-store into each slot's cache
/// band. Each `Vec` holds one entry per row (push exactly one per slot via
/// [`dsv4_dsa_cache_write_gather_row`]); the `u64` ptr arrays + `i32` offset/count
/// arrays are uploaded by [`dsv4_dsa_cache_write_batched`]. The READ side
/// ([`csa_select_official_batched`]) is unaffected — it reads the populated cache.
#[derive(Default)]
pub(crate) struct Dsv4DsaCacheWriteBatchPtrs {
    keys_src: Vec<u64>, // per-slot staging-ring base (state.indexer.compressed.data)
    rotated_dst: Vec<u64>, // per-slot rotated_keys base (offset 0; hadamard DEST, + (dst_row+r)*ihd)
    rotated_src: Vec<u64>, // per-slot rotated_keys PRE-OFFSET base = rotated_dst + dst_row*ihd (fused-store SRC, reads row r from 0)
    cache_band: Vec<u64>,  // per-slot dsa cache band base (pool.dsa_slot_range)
    cache_locs: Vec<u64>,  // per-slot cache_locs slice base (shared.cache_locs + packed_rows)
    src_ring_row: Vec<i32>, // per-slot packed_rows - keys_window_base
    dst_row: Vec<i32>,     // per-slot packed_rows (absolute)
    newly_packed: Vec<i32>, // per-slot row count (0 => slot skipped by the kernel guard)
}

impl Dsv4DsaCacheWriteBatchPtrs {
    pub(crate) fn with_capacity(n: usize) -> Self {
        Self {
            keys_src: Vec::with_capacity(n),
            rotated_dst: Vec::with_capacity(n),
            rotated_src: Vec::with_capacity(n),
            cache_band: Vec::with_capacity(n),
            cache_locs: Vec::with_capacity(n),
            src_ring_row: Vec::with_capacity(n),
            dst_row: Vec::with_capacity(n),
            newly_packed: Vec::with_capacity(n),
        }
    }

    fn len(&self) -> usize {
        self.keys_src.len()
    }
}

/// Gather one slot's DSA cache-write inputs into `ptrs` and ADVANCE this slot's
/// `official.packed_rows` to `indexer_rows_after` — the host half of block (a)
/// of [`csa_select_official`], hoisted into the P1b batched pre-pass. Re-asserts
/// the same preflight invariants block (a) does (rotated_keys length, packed_rows
/// vs indexer rows, the staging-ring straddle, the drain-immediate window-base
/// invariant, the cache-band range). `keys_window_base` is `0` for the
/// full-retention CompressedSparse compressor-indexer (P1b is gated to that). A
/// slot with `newly_packed == 0` pushes a clean skip-entry (the kernel guard
/// early-returns) and leaves `packed_rows` unchanged. Mirrors the per-row block
/// (a) at [`csa_select_official`] exactly, so the batched launch is byte-identical.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dsv4_dsa_cache_write_gather_row(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    state: &mut Dsv4LayerAttentionState,
    pool: &mut Dsv4LayerKvLayout,
    shared: &Dsv4DsaSharedScratch,
    indexer_rows_before: usize,
    indexer_rows_after: usize,
    keys_window_base: usize,
    ptrs: &mut Dsv4DsaCacheWriteBatchPtrs,
) -> Result<()> {
    let ihd = config.index_head_dim;
    // The staging-ring source `keys` is the indexer's `compressed` HiddenStates.
    let keys_len = state
        .indexer
        .as_ref()
        .map(|s| s.compressed.data.len())
        .ok_or_else(|| anyhow!("DSv4 batched DSA cache-write requires indexer state"))?;
    let official = state
        .dsa_official
        .as_mut()
        .ok_or_else(|| anyhow!("DSv4 batched DSA cache-write requires official per-slot state"))?;
    ensure!(
        official.rotated_keys.len()
            == dsv4_dsa_rotated_ring_rows(shared.compressed_capacity) * shared.head_dim,
        "DSv4 batched DSA rotated staging ring len {} mismatches {}x{}",
        official.rotated_keys.len(),
        dsv4_dsa_rotated_ring_rows(shared.compressed_capacity),
        shared.head_dim
    );
    ensure!(
        indexer_rows_after <= shared.compressed_capacity
            && indexer_rows_before <= indexer_rows_after,
        "DSv4 batched DSA key rows before={indexer_rows_before} after={indexer_rows_after} \
         capacity={}",
        shared.compressed_capacity
    );
    let newly_packed = indexer_rows_after.saturating_sub(official.packed_rows);
    if newly_packed == 0 {
        // Clean skip-entry: keep array lengths == n, kernel guard early-returns,
        // packed_rows unchanged (== indexer_rows_after when newly_packed==0).
        ptrs.keys_src.push(0);
        ptrs.rotated_dst.push(0);
        ptrs.rotated_src.push(0);
        ptrs.cache_band.push(0);
        ptrs.cache_locs.push(0);
        ptrs.src_ring_row.push(0);
        ptrs.dst_row.push(0);
        ptrs.newly_packed.push(0);
        return Ok(());
    }
    ensure!(
        official.packed_rows <= indexer_rows_before,
        "DSv4 batched DSA packed rows {} ahead of indexer rows before {indexer_rows_before}",
        official.packed_rows
    );
    let keys_ring_rows = keys_len / ihd;
    ensure!(
        official.packed_rows >= keys_window_base,
        "DSv4 batched DSA packed_rows {} precedes keys window base {keys_window_base} \
         (drain-immediate invariant violated)",
        official.packed_rows
    );
    let src_ring_row = official.packed_rows - keys_window_base;
    ensure!(
        src_ring_row + newly_packed <= keys_ring_rows,
        "DSv4 batched DSA index-key delta straddles staging ring: ring_off {src_ring_row} \
         + newly_packed {newly_packed} > ring_rows {keys_ring_rows} (packed_rows {} base {keys_window_base})",
        official.packed_rows
    );
    // Cache band base for this slot (bands disjoint by slot_idx).
    let cache_range = pool.dsa_slot_range(official.slot_idx)?;
    let cache_pool = pool
        .dsa_key_cache
        .as_mut()
        .ok_or_else(|| anyhow!("DSv4 batched DSA shared key-cache missing"))?;
    ensure!(
        cache_range.end <= cache_pool.len() && cache_range.len() == official.key_cache_len,
        "DSv4 batched DSA shared key-cache range {:?} invalid pool_len={} slot_len={}",
        cache_range,
        cache_pool.len(),
        official.key_cache_len
    );
    // The raw u64 ptr stays valid after the guard drops — single-stream, buffer
    // not reallocated; the launch later re-reads it via the uploaded array.
    let cache_band_ptr = {
        let mut cache_view = cache_pool.slice_mut(cache_range);
        let (p, g) = cache_view.device_ptr_mut(&ctx.stream);
        drop(g);
        p
    };
    // keys_src base (offset 0; the kernel applies (src_ring_row + r) * ihd).
    let keys_src_ptr = {
        let keys = &state_indexer_compressed(state)?.data;
        let (p, g) = keys.device_ptr(&ctx.stream);
        drop(g);
        p
    };
    let official = state
        .dsa_official
        .as_mut()
        .ok_or_else(|| anyhow!("DSv4 batched DSA cache-write requires official per-slot state"))?;
    // rotated_keys is a transient drain-immediate ring: Hadamard writes the delta
    // at ring-relative 0 and the fused-store reads it back the same launch, so
    // BOTH the hadamard dst and the store src are ring-relative 0 (dst_row, the
    // absolute packed-row offset, indexes only cache_locs / the FP8 cache band,
    // not rotated_keys).
    let rotated_dst_ptr = {
        let (p, g) = official.rotated_keys.device_ptr_mut(&ctx.stream);
        drop(g);
        p
    };
    let rotated_src_ptr = rotated_dst_ptr;
    // cache_locs slice base at packed_rows (the kernel reads [tok] from here).
    let cache_locs_ptr = {
        let locs = shared.cache_locs.slice(official.packed_rows..);
        let (p, g) = locs.device_ptr(&ctx.stream);
        drop(g);
        p
    };

    ptrs.keys_src.push(keys_src_ptr);
    ptrs.rotated_dst.push(rotated_dst_ptr);
    ptrs.rotated_src.push(rotated_src_ptr);
    ptrs.cache_band.push(cache_band_ptr);
    ptrs.cache_locs.push(cache_locs_ptr);
    ptrs.src_ring_row.push(i32::try_from(src_ring_row)?);
    // Ring-relative: rotated_keys dst row is always 0 (transient drain-immediate).
    ptrs.dst_row.push(0);
    ptrs.newly_packed.push(i32::try_from(newly_packed)?);
    official.packed_rows = indexer_rows_after;
    Ok(())
}

/// Borrow this layer-state's indexer `compressed` HiddenStates (shared by the
/// cache-write gather to resolve the staging-ring source ptr). Errors if the
/// indexer state is absent.
fn state_indexer_compressed(state: &Dsv4LayerAttentionState) -> Result<&HiddenStates> {
    Ok(&state
        .indexer
        .as_ref()
        .ok_or_else(|| anyhow!("DSv4 batched DSA cache-write requires indexer state"))?
        .compressed)
}

/// Batched DSA cache-write (block (a) of [`csa_select_official`]): ONE
/// `<<<dim3(.,n)>>>` Hadamard-rotate launch + ONE FP8 fused-store launch over all
/// n slots, replacing the n per-row pairs. `ptrs` holds the host-gathered per-slot
/// base ptrs / offsets / counts (built by [`dsv4_dsa_cache_write_gather_row`]).
/// `max_rows` = max over slots of newly_packed (the x-grid bound; 0 => the
/// launchers early-return). Uploads the seven arrays, launches both kernels on
/// `ctx.stream`, then holds the uploaded arrays in `ptr_keepalive` (the N>1
/// keepalive is inert, so explicit retention guards the disabled-event-tracking
/// premature free until the next stream sync — same as the compressor wrapper).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dsv4_dsa_cache_write_batched(
    ctx: &DeviceContext,
    n: usize,
    ptrs: &Dsv4DsaCacheWriteBatchPtrs,
    ptr_keepalive: &mut Vec<CudaSlice<u64>>,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<()> {
    ensure!(
        ptrs.len() == n
            && ptrs.rotated_dst.len() == n
            && ptrs.rotated_src.len() == n
            && ptrs.cache_band.len() == n
            && ptrs.cache_locs.len() == n
            && ptrs.src_ring_row.len() == n
            && ptrs.dst_row.len() == n
            && ptrs.newly_packed.len() == n,
        "DSv4 batched DSA cache-write pointer-array length mismatch (expected {n})"
    );
    if n == 0 {
        return Ok(());
    }
    let max_rows = ptrs.newly_packed.iter().copied().max().unwrap_or(0);
    if max_rows == 0 {
        return Ok(()); // every slot skipped this step.
    }
    let keys_src_arr = crate::ops::upload_u64(ctx, &ptrs.keys_src)?;
    let rotated_dst_arr = crate::ops::upload_u64(ctx, &ptrs.rotated_dst)?;
    let rotated_src_arr = crate::ops::upload_u64(ctx, &ptrs.rotated_src)?;
    let cache_band_arr = crate::ops::upload_u64(ctx, &ptrs.cache_band)?;
    let cache_locs_arr = crate::ops::upload_u64(ctx, &ptrs.cache_locs)?;
    let src_ring_row_arr = crate::ops::upload_i32(ctx, &ptrs.src_ring_row)?;
    let dst_row_arr = crate::ops::upload_i32(ctx, &ptrs.dst_row)?;
    let newly_packed_arr = crate::ops::upload_i32(ctx, &ptrs.newly_packed)?;
    {
        let (keys_src_a, g0) = keys_src_arr.device_ptr(&ctx.stream);
        let (rotated_dst_a, g1) = rotated_dst_arr.device_ptr(&ctx.stream);
        let (rotated_src_a, g1b) = rotated_src_arr.device_ptr(&ctx.stream);
        let (cache_band_a, g2) = cache_band_arr.device_ptr(&ctx.stream);
        let (cache_locs_a, g3) = cache_locs_arr.device_ptr(&ctx.stream);
        let (src_ring_row_a, g4) = src_ring_row_arr.device_ptr(&ctx.stream);
        let (dst_row_a, g5) = dst_row_arr.device_ptr(&ctx.stream);
        let (newly_packed_a, g6) = newly_packed_arr.device_ptr(&ctx.stream);
        // SAFETY: all per-slot ptrs valid on ctx.stream; arrays hold n entries;
        // src/dst offsets + counts mirror the per-row block (a) exactly. The
        // hadamard writes rotated_keys at (dst_row+r); the fused-store reads the
        // PRE-OFFSET `rotated_src` base (= rotated + dst_row*ihd) at row r-from-0.
        unsafe {
            ffi::dsv4_dsa_hadamard128_batched_cuda(
                keys_src_a as *const *const ffi::Half,
                src_ring_row_a as *const i32,
                rotated_dst_a as *const *mut ffi::Half,
                dst_row_a as *const i32,
                newly_packed_a as *const i32,
                n as i32,
                max_rows,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
        // SAFETY: ptrs from live device allocations sized to the dims passed.
        unsafe {
            ffi::dsv4_dsa_fused_store_index_k_cache_batched_cuda(
                rotated_src_a as *const *const ffi::Half,
                cache_band_a as *const *mut u8,
                cache_locs_a as *const *const i64,
                newly_packed_a as *const i32,
                n as i32,
                max_rows,
                64,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
        drop(g0);
        drop(g1);
        drop(g1b);
        drop(g2);
        drop(g3);
        drop(g4);
        drop(g5);
        drop(g6);
    }
    ptr_keepalive.push(keys_src_arr);
    ptr_keepalive.push(rotated_dst_arr);
    ptr_keepalive.push(rotated_src_arr);
    ptr_keepalive.push(cache_band_arr);
    ptr_keepalive.push(cache_locs_arr);
    // i32 offset/count arrays held via the forward keepalive (Arc-retained), same
    // premature-free guard as the u64 ptr arrays above.
    keepalive.keep_i32(&src_ring_row_arr);
    keepalive.keep_i32(&dst_row_arr);
    keepalive.keep_i32(&newly_packed_arr);
    Ok(())
}

/// Batched (m=N) compressor STATE update: ONE `<<<n, BLOCK>>>` launch
/// running each row's per-slot compressor ring update (RoPE/RMSNorm/store into
/// pending/overlap/compressed), replacing the N per-row
/// `dsv4_compressor_update_start_pos_ptr_cuda` launches. `kv_raw_batch` /
/// `score_raw_batch` are the batched [`compressor_batch_prepass`] outputs
/// `[width, n]`; `ape`/`norm` are the SHARED compressor weights; `ptrs` holds the
/// N rows' state buffer pointers (one per row, gathered host-side this step);
/// `start_pos` is the contiguous `[N]` decode-position array. Dims/rope params are
/// uniform across the layer's rows and EXACTLY mirror the per-row
/// [`compressor_forward`] args, so the body math is byte-identical to N per-row
/// launches. The host-side pointer-array uploads are kept alive by the caller's
/// `ptr_keepalive` Vec (the N>1 keepalive is inert, so the batched lane holds the
/// arrays to function return explicitly).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dsv4_compressor_update_batched(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    compressor: &Dsv4Compressor,
    kv_raw_batch: &HiddenStates,
    score_raw_batch: &HiddenStates,
    ptrs: &Dsv4CompressorBatchPtrs,
    start_pos: &CudaSlice<i32>,
    n: usize,
    head_dim: usize,
    ratio: usize,
    overlap: bool,
    apply_rope: bool,
    rope_original_seq_len: i32,
    ptr_keepalive: &mut Vec<CudaSlice<u64>>,
) -> Result<()> {
    ensure!(ratio > 0, "DSv4 batched compressor ratio must be non-zero");
    let width = dsv4_compressor_width(head_dim, overlap);
    ensure!(
        kv_raw_batch.hidden_dim == width
            && kv_raw_batch.seq_len == n
            && score_raw_batch.hidden_dim == width
            && score_raw_batch.seq_len == n,
        "DSv4 batched compressor raw shape mismatch: kv={}x{} score={}x{} expected [{width},{n}]",
        kv_raw_batch.hidden_dim,
        kv_raw_batch.seq_len,
        score_raw_batch.hidden_dim,
        score_raw_batch.seq_len
    );
    ensure!(
        ptrs.len() == n
            && ptrs.pending_score.len() == n
            && ptrs.prev_overlap_kv.len() == n
            && ptrs.prev_overlap_score.len() == n
            && ptrs.compressed.len() == n,
        "DSv4 batched compressor pointer-array length mismatch (expected {n})"
    );
    // Compressed keys use compress_rope_theta with NO YaRN (original_seq_len = 0)
    // on the indexer-no-rope path; identical to the per-row `compressor_forward`.
    let rope = &config.rope_parameters;
    let (rope_dim, rope_base) = if apply_rope {
        (config.qk_rope_head_dim, config.compress_rope_theta)
    } else {
        (0usize, config.compress_rope_theta)
    };
    // Upload the five per-row pointer arrays + hold them alive to function return.
    let pkv_arr = crate::ops::upload_u64(ctx, &ptrs.pending_kv)?;
    let psc_arr = crate::ops::upload_u64(ctx, &ptrs.pending_score)?;
    let prkv_arr = crate::ops::upload_u64(ctx, &ptrs.prev_overlap_kv)?;
    let prsc_arr = crate::ops::upload_u64(ctx, &ptrs.prev_overlap_score)?;
    let comp_arr = crate::ops::upload_u64(ctx, &ptrs.compressed)?;
    // Resolve raw device pointers and release the cudarc borrow guards BEFORE the
    // push below (the SyncOnDrop guards would otherwise borrow the arrays past the
    // move into `ptr_keepalive`). The raw u64 ptrs stay valid — the buffers are not
    // reallocated; single-stream ordering keeps the launched kernel correct.
    // Same raw-read contract as the per-row path: ape must be dense bf16 (#138).
    ensure!(
        compressor.ape.is_dense_bf16(),
        "DSv4 compressor ape must be dense bf16 (raw-read by the batched update kernel); got {:?}",
        compressor.ape.weight_format
    );
    {
        let (kv_ptr, kg) = kv_raw_batch.data.device_ptr(&ctx.stream);
        let (score_ptr, scg) = score_raw_batch.data.device_ptr(&ctx.stream);
        let (ape_ptr, ag) = compressor.ape.data.device_ptr(&ctx.stream);
        let (norm_ptr, ng) = compressor.norm.data.device_ptr(&ctx.stream);
        let (pkv_a, g0) = pkv_arr.device_ptr(&ctx.stream);
        let (psc_a, g1) = psc_arr.device_ptr(&ctx.stream);
        let (prkv_a, g2) = prkv_arr.device_ptr(&ctx.stream);
        let (prsc_a, g3) = prsc_arr.device_ptr(&ctx.stream);
        let (comp_a, g4) = comp_arr.device_ptr(&ctx.stream);
        let (start_ptr, spg) = start_pos.device_ptr(&ctx.stream);
        // SAFETY: all buffers valid on ctx.stream; the pointer arrays hold n valid
        // per-row device pointers; kv/score are [width,n]; dims match the per-row path.
        unsafe {
            ffi::dsv4_compressor_update_batched_start_pos_ptr_cuda(
                kv_ptr as *const ffi::Half,
                score_ptr as *const ffi::Half,
                ape_ptr as *const ffi::Half,
                norm_ptr as *const ffi::Half,
                pkv_a as *const *mut ffi::Half,
                psc_a as *const *mut ffi::Half,
                prkv_a as *const *mut ffi::Half,
                prsc_a as *const *mut ffi::Half,
                comp_a as *const *mut ffi::Half,
                n as i32,
                1, // num_tokens per row (decode)
                start_ptr as *const i32,
                head_dim as i32,
                ratio as i32,
                width as i32,
                i32::from(overlap),
                0, // overlap_page_stride: per-slot register form (#154 D1)
                config.rms_norm_eps,
                rope_dim as i32,
                rope_base,
                rope_original_seq_len,
                rope.factor,
                rope.beta_fast,
                rope.beta_slow,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
        drop(kg);
        drop(scg);
        drop(ag);
        drop(ng);
        drop(g0);
        drop(g1);
        drop(g2);
        drop(g3);
        drop(g4);
        drop(spg);
    }
    // Hold the pointer arrays alive until the caller's keepalive Vec drops (the
    // N>1 forward keepalive is inert, so explicit retention guards the
    // disabled-event-tracking premature free until the next stream sync).
    ptr_keepalive.push(pkv_arr);
    ptr_keepalive.push(psc_arr);
    ptr_keepalive.push(prkv_arr);
    ptr_keepalive.push(prsc_arr);
    ptr_keepalive.push(comp_arr);
    Ok(())
}

/// Batched-decode gather sink threaded into [`csa_select`] / [`mla_attention_prepare_compressed_only`]
/// for row `r` of the N-row batched-decode lane (#60). When present, `csa_select`
/// does the per-row CACHE WRITES only (skips the per-row read/select), gathers
/// this row's bf16 indexer query (`q_i`) and gating weights into the N-row staging
/// buffers at row offset `r`, and captures this row's exact `key_count` — so the
/// ONE batched `csa_select_official_batched` after the prepare loop produces
/// byte-equivalent context_lens. `selected` is then `None` (the batched select
/// writes directly into `selected_batched`).
/// Per-row precomputed compressor/indexer projection slices for the batched
/// decode pre-pass.
/// Each `(kv_raw, score_raw)` pair is this row's `[width, 1]` column slice of the
/// batched `dsv4_linear` output. When threaded into
/// [`mla_attention_prepare_compressed_only`], they replace the per-row m=1
/// `dsv4_linear` GEMVs inside [`compressor_forward`]; the per-slot state update
/// stays per-row. `main` feeds `attention.compressor`; `indexer` feeds
/// `attention.indexer.compressor` (present only when the layer is CompressedSparse).
pub(crate) struct Dsv4CompressorPrecomputed<'a> {
    pub(crate) main: (&'a HiddenStates, &'a HiddenStates),
    pub(crate) indexer: Option<(&'a HiddenStates, &'a HiddenStates)>,
}

pub(crate) struct Dsv4DsaBatchedGather<'a> {
    /// Optional N-row staging for indexer query, shape `[local_index_heads*index_head_dim, n]`.
    /// `None` when the batched indexer-query prepass already produced the exact
    /// buffer consumed by the batched selector.
    pub(crate) q_i_batch: Option<&'a mut HiddenStates>,
    /// Optional N-row staging for gating weights, shape `[local_index_heads, n]`.
    pub(crate) weights_batch: Option<&'a mut HiddenStates>,
    /// Row index in `[0, n)` to gather into.
    pub(crate) row: usize,
    /// Per-row captured `key_count` (push exactly one per call), for the batched
    /// context_lens (`min(key_count_r, abs_pos_r/ratio)`).
    pub(crate) key_counts: &'a mut Vec<i32>,
    /// `true` when the per-row CACHE WRITES (block (a)) already ran in the ONE
    /// batched P1b pre-pass ([`dsv4_dsa_cache_write_batched`]) BEFORE the prepare
    /// loop — so `csa_select` must NOT run the per-row
    /// `csa_select_official(cache_writes_only=true)` again (and `packed_rows` was
    /// already advanced in P1b). `false` (SparseIndexed full-flatten=false lane,
    /// which has no P1b) → the per-row cache write still runs here.
    pub(crate) cache_writes_in_prepass: bool,
}

#[allow(clippy::too_many_arguments)]
fn csa_select_decode_graph(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    indexer: &Dsv4Indexer,
    hidden: &HiddenStates,
    c_q_normed: &HiddenStates,
    keys: &HiddenStates,
    keys_capacity: usize,
    // Ring window base for `keys` (see `csa_select_official`).
    keys_window_base: usize,
    official: &mut Dsv4DsaOfficialState,
    shared: &mut Dsv4DsaSharedScratch,
    pool: &mut Dsv4LayerKvLayout,
    indexer_rows_before: usize,
    indexer_rows_after: usize,
    start_pos: usize,
    start_pos_device: Option<&CudaSlice<i32>>,
    ratio: usize,
    q_i: &mut HiddenStates,
    weights: &mut HiddenStates,
    selected: &mut CudaSlice<i32>,
    layer_idx: usize,
    // GRAPH-SAFE READ lane (opt-in `ARLE_DSV4_DECODE_GRAPH_CSA`): when both are
    // `Some`, route the READ (logits + topk) through the n=1 batched device-meta
    // select reading these PERSISTENT slot_id/key_count buffers (no per-step H2D).
    // When `None`, fall back to the per-tile `csa_select_official` READ (eager only;
    // not graph-capturable, used when the read lane is off).
    slot_id_dev: Option<&CudaSlice<i32>>,
    key_count_dev: Option<&CudaSlice<i32>>,
    slot_idx: usize,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<()> {
    ensure!(
        hidden.seq_len == 1 && c_q_normed.seq_len == 1,
        "DSv4 graph CSA select is decode-only, hidden seq={} c_q seq={}",
        hidden.seq_len,
        c_q_normed.seq_len
    );
    ensure!(
        selected.len() >= config.index_topk,
        "DSv4 graph CSA selected scratch len {} < topk {}",
        selected.len(),
        config.index_topk
    );
    crate::linear_profile::profile(ctx, "dsv4/linear/indexer_wq_b", || {
        dsv4_linear(ctx, &indexer.wq_b, c_q_normed, q_i)
    })?;
    crate::linear_profile::profile(ctx, "dsv4/linear/indexer_weights", || {
        dsv4_linear(ctx, &indexer.weights_proj, hidden, weights)
    })?;
    ensure!(
        q_i.hidden_dim.is_multiple_of(config.index_head_dim),
        "DSv4 graph CSA q_i width {} is not divisible by index_head_dim {}",
        q_i.hidden_dim,
        config.index_head_dim
    );
    let local_index_heads = q_i.hidden_dim / config.index_head_dim;
    ensure!(
        weights.hidden_dim == local_index_heads,
        "DSv4 graph CSA weights width {} != local index heads {local_index_heads}",
        weights.hidden_dim
    );
    let key_count = if start_pos_device.is_some() {
        keys_capacity
    } else {
        keys.seq_len
    };
    let score_scale =
        (config.index_head_dim as f32).powf(-0.5) * (config.index_n_heads as f32).powf(-0.5);

    // GRAPH-SAFE READ lane: cache WRITES (block (a)) via `cache_writes_only`, then
    // the READ via the n=1 batched device-meta select reading persistent buffers.
    // Both `slot_id_dev`/`key_count_dev` Some ⟹ caller pre-staged them (graph lane).
    if let (Some(slot_id_dev), Some(key_count_dev)) = (slot_id_dev, key_count_dev) {
        // (a) per-row CACHE WRITES only — populate this slot's DSA key cache for
        // the rows committed this step. (Still host-shape driven; NOT yet graph-
        // capturable — gated behind the surrounding decode-graph bail.)
        let cache_ret = csa_select_official(
            ctx,
            config,
            q_i,
            weights,
            keys,
            official,
            shared,
            pool,
            indexer_rows_before,
            indexer_rows_after,
            key_count,
            start_pos,
            start_pos_device,
            keys_window_base,
            layer_idx,
            ratio,
            local_index_heads,
            score_scale,
            None,
            /* cache_writes_only */ true,
            keepalive,
        )?;
        ensure!(
            cache_ret.is_none(),
            "DSv4 graph CSA cache-write pass unexpectedly returned selected output"
        );
        // (b)-(f) graph-safe READ at n=1 via the batched device-meta select. The
        // q_i/weights scratch ARE the [width,1] n=1 batch (no gather). Persistent
        // slot_id/key_count device buffers + start_pos_device ⟹ no per-step H2D.
        let mut empty_keepalive: Vec<CudaSlice<i32>> = Vec::new();
        csa_select_official_batched(
            ctx,
            config,
            layer_idx,
            &*q_i,
            &*weights,
            shared,
            pool,
            1,
            &[slot_idx],
            &[],
            &[],
            local_index_heads,
            score_scale,
            selected,
            keepalive,
            /* use_device_meta */ true,
            i32::try_from(ratio)?,
            start_pos_device,
            &[],
            &mut empty_keepalive,
            Some(slot_id_dev),
            Some(key_count_dev),
        )?;
        ensure!(
            empty_keepalive.is_empty(),
            "DSv4 graph CSA read uploaded meta despite persistent buffers"
        );
        return Ok(());
    }

    // Eager READ fallback (read lane off): per-tile `csa_select_official` READ. Not
    // graph-capturable (per-step alloc/host-shape) — only reached eagerly.
    let selected_return = csa_select_official(
        ctx,
        config,
        q_i,
        weights,
        keys,
        official,
        shared,
        pool,
        indexer_rows_before,
        indexer_rows_after,
        key_count,
        start_pos,
        start_pos_device,
        keys_window_base,
        layer_idx,
        ratio,
        local_index_heads,
        score_scale,
        Some(selected),
        /* cache_writes_only */ false,
        keepalive,
    )?;
    ensure!(
        selected_return.is_none(),
        "DSv4 graph CSA official select unexpectedly allocated selected output"
    );
    Ok(())
}

/// CSA top-k block selection: project the index query (`wq_b`) + per-head gating
/// (`weights_proj`), then the official DSA selector scores each compressed-key
/// block and writes the top-`index_topk` block ids per token into `[seq * index_topk]`.
///
/// When `batched_gather` is `Some`, the per-row READ/SELECT is SKIPPED: only the
/// per-row cache writes run (via `csa_select_official` cache-writes-only mode),
/// this row's `q_i`/`weights` are gathered into the N-row staging, and the row's
/// `key_count` is captured — the batched lane runs ONE `csa_select_official_batched`
/// afterward. When `None`, this runs the same official selector directly.
#[allow(clippy::too_many_arguments)]
fn csa_select(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    layer_idx: usize,
    indexer: &Dsv4Indexer,
    hidden: &HiddenStates,
    c_q_normed: &HiddenStates,
    keys: &HiddenStates,
    keys_capacity: usize,
    // Ring window base for `keys` (see `csa_select_official`): `start_pos` for the
    // SparseIndexed staging ring, `0` for the full-retention compressor.
    keys_window_base: usize,
    official: Option<&mut Dsv4DsaOfficialState>,
    dsa_shared: Option<&mut Dsv4DsaSharedScratch>,
    pool: &mut Dsv4LayerKvLayout,
    indexer_rows_before: usize,
    indexer_rows_after: usize,
    start_pos: usize,
    start_pos_device: Option<&CudaSlice<i32>>,
    ratio: usize,
    prefill_scratch: Option<&mut Dsv4PrefillDeepGemmLinearScratch>,
    batched_gather: Option<Dsv4DsaBatchedGather<'_>>,
    // Batched-decode pre-pass (full-flatten): when `Some`, this row's
    // `q_i`/`weights` were already projected batched (one m=N `dsv4_linear` each
    // over the whole layer's rows in `indexer_query_batch_prepass`), so the per-row
    // m=1 `wq_b`/`weights_proj` GEMVs below are SKIPPED and these `[width,1]` slices
    // are used instead. The slice IS the exact column of the batched GEMM output =
    // the per-row m=1 GEMV result → byte-identical. `None` → byte-identical per-row
    // GEMVs (single-row / prefill / non-full-flatten lanes).
    query_precomputed: Option<Dsv4IndexerQueryPrecomputed<'_>>,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<Option<CudaSlice<i32>>> {
    // Batched pre-pass: this row's `q_i`/`weights` are a borrowed `[width,1]`
    // column VIEW of the batched prepass output — ZERO copy (the view's device
    // pointer is the exact column the per-row D2D copy would have produced). The
    // GEMVs are skipped. Decode-only (`c_q_normed.seq_len == 1`); the prefill
    // DeepGEMM path never supplies it. `owned_*` stays `None` here: the only
    // consumer needing `&HiddenStates` (the non-batched `csa_select_official`
    // fallback below) is unreachable when precomputed (proven: precomputed Some
    // ⟹ CompressedSparse + dsa-official + batch-scratch ⟹ batched_gather Some).
    let (owned_q_i, owned_weights) = if let Some(pre) = query_precomputed.as_ref() {
        ensure!(
            c_q_normed.seq_len == 1 && hidden.seq_len == 1,
            "DSv4 indexer query precomputed is decode-only (c_q seq={} hidden seq={})",
            c_q_normed.seq_len,
            hidden.seq_len
        );
        ensure!(
            pre.q_i.hidden_dim == indexer.wq_b.rows
                && pre.q_i.seq_len == 1
                && pre.weights.hidden_dim == indexer.weights_proj.rows
                && pre.weights.seq_len == 1,
            "DSv4 indexer query precomputed shape mismatch: q_i {}x{} (want {}x1) weights {}x{} (want {}x1)",
            pre.q_i.hidden_dim,
            pre.q_i.seq_len,
            indexer.wq_b.rows,
            pre.weights.hidden_dim,
            pre.weights.seq_len,
            indexer.weights_proj.rows
        );
        (None, None)
    } else {
        // SAFETY: dsv4_linear writes the full index-query buffer.
        let mut q_i = unsafe { HiddenStates::uninit(ctx, indexer.wq_b.rows, c_q_normed.seq_len)? };
        crate::profile::profile_op(ctx, "linear/indexer_wq_b", None, c_q_normed.seq_len, || {
            // Prefill index-query (M=token_count) → DeepGEMM, off the scalar fp8_gemv (the #1
            // remaining projection at M=1024). Decode (seq_len==1) / no-cache stays scalar.
            let indexer_wq_b_dg = c_q_normed.seq_len > 1
                && dsv4_prefill_indexer_deepgemm_enabled()
                && indexer.wq_b_deepgemm.is_some()
                && prefill_scratch.is_some();
            if indexer_wq_b_dg {
                let cache = indexer
                    .wq_b_deepgemm
                    .as_ref()
                    .expect("indexer wq_b dg gate checked");
                let scratch = prefill_scratch.expect("indexer wq_b dg gate checked");
                crate::linear_profile::profile(ctx, "dsv4/linear/indexer_wq_b", || {
                    prefill_proj_deepgemm(ctx, scratch, cache, c_q_normed, &mut q_i)
                })
            } else {
                crate::linear_profile::profile(ctx, "dsv4/linear/indexer_wq_b", || {
                    dsv4_linear(ctx, &indexer.wq_b, c_q_normed, &mut q_i)
                })
            }
        })?;
        keepalive.keep_hidden(&q_i);
        // SAFETY: dsv4_linear writes the full index-weight buffer.
        let mut weights =
            unsafe { HiddenStates::uninit(ctx, indexer.weights_proj.rows, hidden.seq_len)? };
        crate::profile::profile_op(ctx, "linear/indexer_weights", None, hidden.seq_len, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/indexer_weights", || {
                dsv4_linear(ctx, &indexer.weights_proj, hidden, &mut weights)
            })
        })?;
        keepalive.keep_hidden(&weights);
        (Some(q_i), Some(weights))
    };

    // Unified VIEW over q_i/weights for all reads below: owned buffer's full view
    // (non-precomputed) or the prepass column view (precomputed) — identical
    // device pointer + width either way → byte-identical reads.
    let q_i = match owned_q_i.as_ref() {
        Some(o) => o.as_view(),
        None => query_precomputed
            .as_ref()
            .expect("owned_q_i None ⟹ precomputed Some")
            .q_i
            .as_self_view(),
    };
    let weights = match owned_weights.as_ref() {
        Some(o) => o.as_view(),
        None => query_precomputed
            .as_ref()
            .expect("owned_weights None ⟹ precomputed Some")
            .weights
            .as_self_view(),
    };

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
        if dsv4_verify_frozen() {
            // Frozen-KV (P1-A): the selector computes `available = min(key_count,
            // abs_pos / ratio)`. A frozen verify's `abs_pos` can cross a compression
            // boundary, so capacity-or-`abs_pos/ratio` would expose a compressed row
            // the frozen compressor never produced. Pin to the committed indexer row
            // count — `keys.seq_len` is frozen to that by P1-1, and abs_pos/ratio >=
            // it, so `available` stays at the committed set. (Frozen verify is the
            // spec path, never graph-replayed, so the replay-safety capacity rule
            // below does not apply here.)
            keys.seq_len
        } else {
            // Graph replay must not bake the current compressed-key seq_len. The
            // selector computes `available = min(key_count, abs_pos / ratio)`, so
            // capacity preserves the same causal set while staying replay-safe.
            keys_capacity
        }
    } else {
        keys.seq_len
    };
    let score_scale =
        (config.index_head_dim as f32).powf(-0.5) * (config.index_n_heads as f32).powf(-0.5);

    // ── Batched-decode lane (#60): per-row CACHE WRITES only, then gather this
    // row's q_i/weights into the N-row staging + capture key_count. The READ
    // (logits + topk) is deferred to ONE `csa_select_official_batched` after all
    // N rows' caches are populated. Requires the official DSA path (per-slot
    // `official` + shared scratch). Never reached on the single-row / prefill
    // path (batched_gather stays None there → byte-identical below).
    if let Some(mut gather) = batched_gather {
        // Per-row CACHE WRITES (block (a) of csa_select_official). When the ONE
        // batched P1b pre-pass already populated all N slots' DSA caches
        // (`cache_writes_in_prepass`, the CompressedSparse full-flatten lane),
        // skip this per-row write + `packed_rows` advance — they ran in P1b. The
        // SparseIndexed lane (no P1b) still runs the per-row write here.
        if !gather.cache_writes_in_prepass {
            // `cache_writes_in_prepass` is false only on the SparseIndexed lane,
            // which never supplies `query_precomputed` (CompressedSparse-only) →
            // `owned_*` is Some, so `csa_select_official` reads the owned
            // `&HiddenStates` (signature unchanged). Asserted, not assumed.
            let (owned_q_i, owned_weights) = (owned_q_i.as_ref(), owned_weights.as_ref());
            let q_i = owned_q_i.ok_or_else(|| {
                anyhow!("DSv4 batched DSA per-row cache write needs owned q_i (precomputed only on the prepass lane)")
            })?;
            let weights = owned_weights.ok_or_else(|| {
                anyhow!("DSv4 batched DSA per-row cache write needs owned weights (precomputed only on the prepass lane)")
            })?;
            let official = official.ok_or_else(|| {
                anyhow!("DSv4 batched DSA gather requires official per-slot DSA state")
            })?;
            let shared = dsa_shared
                .ok_or_else(|| anyhow!("DSv4 batched DSA gather requires shared DSA scratch"))?;
            csa_select_official(
                ctx,
                config,
                q_i,
                weights,
                keys,
                official,
                shared,
                pool,
                indexer_rows_before,
                indexer_rows_after,
                key_count,
                start_pos,
                start_pos_device,
                keys_window_base,
                layer_idx,
                ratio,
                local_index_heads,
                score_scale,
                None,
                /* cache_writes_only */ true,
                keepalive,
            )?;
        }
        // Gather this row's q_i/weights (bf16) into the N-row staging at row r.
        // q_i is `[local_index_heads*index_head_dim, 1]`, weights `[local_index_heads, 1]`.
        // Source is the unified VIEW (`q_i`/`weights`): the prepass column view
        // (precomputed, zero copy) or the owned buffer's view (non-precomputed).
        let r = gather.row;
        if let (Some(q_i_batch), Some(weights_batch)) = (
            gather.q_i_batch.as_deref_mut(),
            gather.weights_batch.as_deref_mut(),
        ) {
            let q_width = q_i.data.len();
            let w_width = weights.data.len();
            ensure!(
                q_i_batch.data.len() >= (r + 1) * q_width
                    && weights_batch.data.len() >= (r + 1) * w_width,
                "DSv4 batched DSA staging too small for row {r} (q {} w {})",
                q_width,
                w_width
            );
            {
                let mut dst = q_i_batch.data.slice_mut(r * q_width..(r + 1) * q_width);
                ctx.stream
                    .memcpy_dtod(&q_i.data, &mut dst)
                    .map_err(|e| anyhow!("DSv4 batched DSA q_i gather D2D failed: {e}"))?;
            }
            {
                let mut dst = weights_batch.data.slice_mut(r * w_width..(r + 1) * w_width);
                ctx.stream
                    .memcpy_dtod(&weights.data, &mut dst)
                    .map_err(|e| anyhow!("DSv4 batched DSA weights gather D2D failed: {e}"))?;
            }
        } else {
            ensure!(
                gather.q_i_batch.is_none() && gather.weights_batch.is_none(),
                "DSv4 batched DSA staging must provide both q_i and weights or neither"
            );
        }
        gather.key_counts.push(
            i32::try_from(key_count)
                .map_err(|_| anyhow!("DSv4 batched DSA key_count {key_count} overflows i32"))?,
        );
        // No per-row selected: the batched select writes selected_batched directly.
        return Ok(None);
    }

    // Non-batched single-row / prefill fallback: `batched_gather` is None here,
    // and precomputed Some ⟹ batched_gather Some (CompressedSparse + dsa-official
    // + batch-scratch), so this path is never reached when precomputed → `owned_*`
    // is Some, and `csa_select_official` reads the owned `&HiddenStates`
    // (signature unchanged). Asserted, not assumed.
    let q_i = owned_q_i.as_ref().ok_or_else(|| {
        anyhow!("DSv4 CSA non-batched select needs owned q_i (precomputed ⟹ batched gather)")
    })?;
    let weights = owned_weights.as_ref().ok_or_else(|| {
        anyhow!("DSv4 CSA non-batched select needs owned weights (precomputed ⟹ batched gather)")
    })?;
    let official =
        official.ok_or_else(|| anyhow!("DSv4 CSA select requires official DSA per-slot state"))?;
    let shared = dsa_shared
        .ok_or_else(|| anyhow!("DSv4 CSA select requires shared official DSA scratch"))?;
    let selected = csa_select_official(
        ctx,
        config,
        q_i,
        weights,
        keys,
        official,
        shared,
        pool,
        indexer_rows_before,
        indexer_rows_after,
        key_count,
        start_pos,
        start_pos_device,
        keys_window_base,
        layer_idx,
        ratio,
        local_index_heads,
        score_scale,
        None,
        /* cache_writes_only */ false,
        keepalive,
    )?
    .ok_or_else(|| anyhow!("DSv4 CSA official select returned no selected output"))?;
    Ok(Some(selected))
}

#[allow(clippy::too_many_arguments)]
fn csa_select_official(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    q_i: &HiddenStates,
    weights: &HiddenStates,
    keys: &HiddenStates,
    official: &mut Dsv4DsaOfficialState,
    shared: &mut Dsv4DsaSharedScratch,
    pool: &mut Dsv4LayerKvLayout,
    indexer_rows_before: usize,
    indexer_rows_after: usize,
    key_count: usize,
    start_pos: usize,
    start_pos_device: Option<&CudaSlice<i32>>,
    // Ring window base for reading the `keys` staging buffer: `start_pos` for the
    // SparseIndexed indexer (its delta is staged window-relative from ring row 0
    // every forward), `0` for the full-retention compressor (absolute offsets).
    // Row `r` of the delta is read at `keys.data[(r - keys_window_base) * ihd]`.
    keys_window_base: usize,
    _layer_idx: usize,
    ratio: usize,
    local_index_heads: usize,
    score_scale: f32,
    selected_out: Option<&mut CudaSlice<i32>>,
    // Batched-decode lane (#60): run block (a) (per-row CACHE WRITES) only, then
    // return `Ok(None)` BEFORE the per-row read/select (b)-(f). The READ is
    // deferred to ONE `csa_select_official_batched`. The single-row / prefill path
    // passes `false` → byte-identical behavior.
    cache_writes_only: bool,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<Option<CudaSlice<i32>>> {
    if !cache_writes_only
        && start_pos_device.is_some()
        && matches!(
            std::env::var("ARLE_DSV4_DECODE_GRAPH").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
        )
    {
        return Ok(None);
    }
    ensure!(
        local_index_heads == shared.num_heads && config.index_head_dim == shared.head_dim,
        "DSv4 official DSA shape mismatch local_heads={} official_heads={} dim={} official_dim={}",
        local_index_heads,
        shared.num_heads,
        config.index_head_dim,
        shared.head_dim
    );
    ensure!(
        q_i.seq_len <= shared.max_tokens,
        "DSv4 official DSA token_count {} exceeds scratch max {}",
        q_i.seq_len,
        shared.max_tokens
    );
    ensure!(
        key_count <= shared.compressed_capacity
            && indexer_rows_after <= shared.compressed_capacity
            && indexer_rows_before <= indexer_rows_after,
        "DSv4 official DSA key rows before={} after={} key_count={} capacity={}",
        indexer_rows_before,
        indexer_rows_after,
        key_count,
        shared.compressed_capacity
    );
    ensure!(
        start_pos + q_i.seq_len <= shared.max_tokens,
        "DSv4 official DSA positions {}..{} exceed freqs_cis max {}",
        start_pos,
        start_pos + q_i.seq_len,
        shared.max_tokens
    );

    ensure!(
        official.rotated_keys.len()
            == dsv4_dsa_rotated_ring_rows(shared.compressed_capacity) * shared.head_dim,
        "DSv4 official DSA rotated staging ring len {} mismatches {}x{}",
        official.rotated_keys.len(),
        dsv4_dsa_rotated_ring_rows(shared.compressed_capacity),
        shared.head_dim
    );

    let newly_packed = indexer_rows_after.saturating_sub(official.packed_rows);
    if newly_packed > 0 {
        ensure!(
            official.packed_rows <= indexer_rows_before,
            "DSv4 official DSA packed rows {} ahead of indexer rows before {}",
            official.packed_rows,
            indexer_rows_before
        );
        let ihd = config.index_head_dim;
        let keys_ring_rows = keys.data.len() / ihd;
        // The SOURCE `keys.data` is the indexer's STAGING RING: the live delta
        // [packed_rows..rows_after) was staged this same forward window-relative
        // to `keys_window_base` (= `start_pos` for the SparseIndexed ring; `0`
        // for the full-retention compressor, where the buffer holds every row at
        // its absolute offset). Recover row `r` at ring offset `r - window_base`.
        // The DESTINATION `rotated_keys` stays at the ABSOLUTE `packed_rows`
        // offset (it retains the full history). Drain-immediate guarantees the
        // window base never trails `packed_rows`.
        ensure!(
            official.packed_rows >= keys_window_base,
            "DSv4 official DSA packed_rows {} precedes keys window base {keys_window_base} \
             (drain-immediate invariant violated)",
            official.packed_rows
        );
        let src_ring_row = official.packed_rows - keys_window_base;
        ensure!(
            src_ring_row + newly_packed <= keys_ring_rows,
            "DSv4 official DSA index-key delta straddles staging ring: ring_off {src_ring_row} \
             + newly_packed {newly_packed} > ring_rows {keys_ring_rows} (packed_rows {} base {keys_window_base})",
            official.packed_rows
        );
        // `rotated_keys` is a transient drain-immediate staging buffer: Hadamard
        // writes the delta at ring offset 0 and the fused-store reads it back the
        // same forward, so the dst is ALWAYS ring-relative 0 (not absolute
        // `packed_rows`). `newly_packed <= ring_rows` is guaranteed by the source
        // staging-ring straddle check above (same depth), re-asserted for clarity.
        let rotated_ring_rows = official.rotated_keys.len() / ihd;
        ensure!(
            newly_packed <= rotated_ring_rows,
            "DSv4 official DSA delta {newly_packed} exceeds rotated staging ring {rotated_ring_rows}"
        );
        let src_offset = src_ring_row * ihd;
        let dst_offset = 0usize;
        let src = keys.data.slice(src_offset..src_offset + newly_packed * ihd);
        {
            let mut rotated = official
                .rotated_keys
                .slice_mut(dst_offset..dst_offset + newly_packed * ihd);
            let (src_ptr, _sg) = src.device_ptr(&ctx.stream);
            let (rot_ptr, _rg) = rotated.device_ptr_mut(&ctx.stream);
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            unsafe {
                ffi::dsv4_dsa_hadamard128_bf16_cuda(
                    src_ptr as *const ffi::Half,
                    rot_ptr as *mut ffi::Half,
                    i32::try_from(newly_packed)?,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }
        let locs = shared
            .cache_locs
            .slice(official.packed_rows..official.packed_rows + newly_packed);
        {
            let rotated = official
                .rotated_keys
                .slice(dst_offset..dst_offset + newly_packed * ihd);
            let (rot_store_ptr, _rsg) = rotated.device_ptr(&ctx.stream);
            let cache_range = pool.dsa_slot_range(official.slot_idx)?;
            let cache_pool = pool
                .dsa_key_cache
                .as_mut()
                .ok_or_else(|| anyhow!("DSv4 official DSA shared key-cache missing"))?;
            ensure!(
                cache_range.end <= cache_pool.len() && cache_range.len() == official.key_cache_len,
                "DSv4 official DSA shared key-cache range {:?} invalid pool_len={} slot_len={}",
                cache_range,
                cache_pool.len(),
                official.key_cache_len
            );
            let mut cache_view = cache_pool.slice_mut(cache_range);
            let (cache_ptr_u8, _cg) = cache_view.device_ptr_mut(&ctx.stream);
            let (locs_ptr, _lg) = locs.device_ptr(&ctx.stream);
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            unsafe {
                ffi::dsv4_dsa_fused_store_index_k_cache_cuda(
                    rot_store_ptr as *const ffi::Half,
                    cache_ptr_u8 as *mut u8,
                    locs_ptr as *const i64,
                    i32::try_from(newly_packed)?,
                    64,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }
        official.packed_rows = indexer_rows_after;
    }

    // Batched-decode lane (#60): the per-row CACHE WRITES (block (a)) are done;
    // the per-row READ/SELECT (b)-(f) is deferred to ONE `csa_select_official_batched`.
    if cache_writes_only {
        return Ok(None);
    }

    let token_count = q_i.seq_len;
    // `raw_indices` (topk output) is sized by `query_chunk`, not `max_tokens`. The
    // scheduler guarantees a single forward never passes more than
    // `chunked_prefill_size <= DSV4_PREFILL_QUERY_CHUNK == query_chunk` query tokens,
    // so the per-tile `raw_indices[t0..t0+tlen]` writes and the DUMP read
    // `raw_indices[0..seq_len*topk]` both stay within the chunk-sized buffer. Fail loud
    // rather than write past it (e.g. the one-shot long-context `dsv4_parity` example,
    // which is not the chunked-prefill path).
    ensure!(
        token_count <= shared.query_chunk,
        "DSv4 official DSA token_count {} exceeds prefill query chunk {} (raw_indices \
         scratch is chunk-sized; chunked prefill must keep seq_len <= \
         chunked_prefill_size <= DSV4_PREFILL_QUERY_CHUNK)",
        token_count,
        shared.query_chunk
    );
    let mut owned_selected = if selected_out.is_none() {
        // Full-N eager output. Decode graph passes persistent scratch in
        // `selected_out` so capture/replay does not allocate.
        Some(
            ctx.stream
                .alloc_zeros::<i32>(token_count * config.index_topk)
                .map_err(|e| anyhow!("DSv4 official DSA selected alloc failed: {e}"))?,
        )
    } else {
        None
    };

    {
        let selected = selected_out
            .unwrap_or_else(|| owned_selected.as_mut().expect("owned selected allocated"));
        ensure!(
            selected.len() >= token_count * config.index_topk,
            "DSv4 official DSA selected output len {} < required {}",
            selected.len(),
            token_count * config.index_topk
        );

        // Query-axis tiling — the ONLY compute path. The logits scratch is bounded by
        // `tile × logits_stride`; long prompts loop in tiles and never materialize full-N
        // logits. When token_count <= tile this loop runs a single iteration with t0=0
        // (tlen=token_count), behavior-IDENTICAL to the pre-tiling code.
        //
        // Mutated-buffer enumeration (per-tile correctness):
        //   shared.logits [tile × stride]: overwritten each sub-chunk before topk reads it — safe.
        //   shared.q_fp8/weights/context_lens/positions [tile-sized]: overwritten each
        //     sub-chunk before use — safe.
        //   shared.page_table_identity [tile × num_pages]: identity, read-only, same for
        //     every sub-chunk — safe.
        //   selected / shared.raw_indices [full N × topk]: each sub-chunk writes its
        //     disjoint [t0..t0+tlen) slice — full output assembled, no overlap.
        //   key-packing buffers (rotated_keys, key cache, cache_locs, packed_rows):
        //     untouched by this change (handled in the query-independent block above).
        let tile = shared.query_tile;
        // q_i.data / weights.data are flat [seq_len * per_token_width]; derive per-token
        // strides so each tile slices the right sub-range of the (untiled) inputs.
        let q_stride = q_i.data.len() / token_count;
        ensure!(
            q_stride * token_count == q_i.data.len(),
            "DSv4 official DSA q input len {} not divisible by token_count {}",
            q_i.data.len(),
            token_count
        );
        let w_stride = weights.data.len() / token_count;
        ensure!(
            w_stride * token_count == weights.data.len(),
            "DSv4 official DSA weights input len {} not divisible by token_count {}",
            weights.data.len(),
            token_count
        );

        let mut t0 = 0usize;
        while t0 < token_count {
            let tlen = (token_count - t0).min(tile);

            // (a) per-tile context_lens / positions. Decode graph/eager decode carry
            // `start_pos` on device; fill tile metadata on GPU to avoid two tiny
            // H2D copies per CSA layer.
            {
                let mut context_lens = shared.context_lens.slice_mut(0..tlen);
                let mut positions = shared.positions.slice_mut(0..tlen);
                if let Some(start_pos_device) = start_pos_device {
                    let (lens_ptr, _lg) = context_lens.device_ptr_mut(&ctx.stream);
                    let (positions_ptr, _pg) = positions.device_ptr_mut(&ctx.stream);
                    let (start_ptr, _sg) = start_pos_device.device_ptr(&ctx.stream);
                    // SAFETY: ptrs from live device allocations sized to the dims passed.
                    unsafe {
                        ffi::dsv4_dsa_fill_context_lens_positions_start_pos_cuda(
                            lens_ptr as *mut i32,
                            positions_ptr as *mut i32,
                            start_ptr as *const i32,
                            i32::try_from(t0)?,
                            i32::try_from(tlen)?,
                            i32::try_from(key_count)?,
                            i32::try_from(ratio)?,
                            ctx.stream.cu_stream(),
                        )
                        .result()
                        .map_err(|e| anyhow!("DSv4 official DSA GPU metadata fill failed: {e}"))?;
                    }
                } else {
                    let context_lens_tile: Vec<i32> = (0..tlen)
                        .map(|i| {
                            let abs_pos = start_pos + t0 + i;
                            i32::try_from(std::cmp::min(key_count, abs_pos / ratio))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let positions_tile: Vec<i32> = (0..tlen)
                        .map(|i| i32::try_from(start_pos + t0 + i))
                        .collect::<Result<Vec<_>, _>>()?;
                    ctx.stream
                        .memcpy_htod(&context_lens_tile, &mut context_lens)
                        .map_err(|e| anyhow!("DSv4 official DSA context_lens H2D failed: {e}"))?;
                    ctx.stream
                        .memcpy_htod(&positions_tile, &mut positions)
                        .map_err(|e| anyhow!("DSv4 official DSA positions H2D failed: {e}"))?;
                }
            }

            // (c) fused Q indexer rope+hadamard+quant over the tile's input sub-range.
            {
                let q_in = q_i.data.slice(t0 * q_stride..(t0 + tlen) * q_stride);
                let (q_ptr, _qg) = q_in.device_ptr(&ctx.stream);
                let (q_fp8_ptr, _qfg) = shared.q_fp8.device_ptr_mut(&ctx.stream);
                let w_in = weights.data.slice(t0 * w_stride..(t0 + tlen) * w_stride);
                let (w_ptr, _wg) = w_in.device_ptr(&ctx.stream);
                let (weights_out_ptr, _wog) = shared.weights.device_ptr_mut(&ctx.stream);
                let (freqs_ptr, _fg) = shared.freqs_cis.device_ptr(&ctx.stream);
                let positions = shared.positions.slice(0..tlen);
                let (positions_ptr, _pg) = positions.device_ptr(&ctx.stream);
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                unsafe {
                    ffi::dsv4_dsa_fused_q_indexer_rope_hadamard_quant_cuda(
                        q_ptr as *const ffi::Half,
                        q_fp8_ptr as *mut u8,
                        w_ptr as *const ffi::Half,
                        weights_out_ptr as *mut f32,
                        score_scale,
                        freqs_ptr as *const f32,
                        positions_ptr as *const i32,
                        i32::try_from(tlen)?,
                        i32::try_from(local_index_heads)?,
                        ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
            }

            // (d) paged MQA logits scheduling metadata for the tile.
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            unsafe {
                cuda_moe::dsv4_deepgemm_paged_mqa_logits_metadata(
                    cache_ptr(&shared.context_lens, ctx),
                    cache_ptr(&shared.sched_meta, ctx),
                    tlen,
                    1,
                    64,
                    shared.num_sms,
                    ctx.stream.cu_stream(),
                )
                .map_err(|e| anyhow!("DSv4 official DSA metadata failed: {e}"))?;
            }

            // (e) fused paged FP8 MQA logits → shared.logits (tlen rows).
            {
                let cache_range = pool.dsa_slot_range(official.slot_idx)?;
                let cache_pool = pool
                    .dsa_key_cache
                    .as_ref()
                    .ok_or_else(|| anyhow!("DSv4 official DSA shared key-cache missing"))?;
                ensure!(
                    cache_range.end <= cache_pool.len()
                        && cache_range.len() == official.key_cache_len,
                    "DSv4 official DSA shared key-cache range {:?} invalid pool_len={} slot_len={}",
                    cache_range,
                    cache_pool.len(),
                    official.key_cache_len
                );
                let cache_view = cache_pool.slice(cache_range);
                let (q_ptr, _qg) = shared.q_fp8.device_ptr(&ctx.stream);
                let (cache_ptr_u8, _kg) = cache_view.device_ptr(&ctx.stream);
                let (weights_ptr, _wg) = shared.weights.device_ptr(&ctx.stream);
                let (lens_ptr, _lg) = shared.context_lens.device_ptr(&ctx.stream);
                let (page_ptr, _pg) = shared.page_table_identity.device_ptr(&ctx.stream);
                let (sched_ptr, _sg) = shared.sched_meta.device_ptr(&ctx.stream);
                let (logits_ptr, _og) = shared.logits.device_ptr_mut(&ctx.stream);
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                unsafe {
                    ffi::dsv4_deepgemm_fp8_paged_mqa_logits_fused_cache_cuda(
                        q_ptr as *const u8,
                        cache_ptr_u8 as *const u8,
                        weights_ptr as *const f32,
                        lens_ptr as *const i32,
                        page_ptr as *const i32,
                        sched_ptr as *const i32,
                        logits_ptr as *mut f32,
                        i32::try_from(tlen)?,
                        1,
                        i32::try_from(local_index_heads)?,
                        i32::try_from(config.index_head_dim)?,
                        i32::try_from(shared.num_pages)?,
                        64,
                        i32::try_from(shared.num_pages * 64)?,
                        i32::try_from(shared.logits_stride)?,
                        i32::try_from(shared.num_pages)?,
                        i32::try_from(64 * (config.index_head_dim + std::mem::size_of::<f32>()))?,
                        i32::try_from(shared.num_sms)?,
                        ctx.stream.cu_stream(),
                    )
                    .result()
                    .map_err(|e| anyhow!("DSv4 official DSA paged logits failed: {e}"))?;
                }
            }

            // (f) topk transform: read shared.logits (base), write the tile's disjoint
            //     output slices of `selected` and `shared.raw_indices`.
            {
                let context_lens = shared.context_lens.slice(0..tlen);
                let (logits_ptr, _lg) = shared.logits.device_ptr(&ctx.stream);
                let (lens_ptr, _csg) = context_lens.device_ptr(&ctx.stream);
                let (page_ptr, _ptg) = shared.page_table_identity.device_ptr(&ctx.stream);
                let mut sel =
                    selected.slice_mut(t0 * config.index_topk..(t0 + tlen) * config.index_topk);
                let (sel_ptr, _seg) = sel.device_ptr_mut(&ctx.stream);
                let mut raw = shared
                    .raw_indices
                    .slice_mut(t0 * config.index_topk..(t0 + tlen) * config.index_topk);
                let (raw_ptr, _rig) = raw.device_ptr_mut(&ctx.stream);
                // SAFETY: ptrs from live device allocations sized to the dims passed.
                unsafe {
                    ffi::dsv4_deepseek_v4_topk_transform_cuda(
                        logits_ptr as *const f32,
                        lens_ptr as *const i32,
                        page_ptr as *const i32,
                        sel_ptr as *mut i32,
                        raw_ptr as *mut i32,
                        i64::try_from(shared.logits_stride)?,
                        i64::try_from(shared.num_pages)?,
                        i64::try_from(config.index_topk)?,
                        i32::try_from(tlen)?,
                        i32::try_from(config.index_topk)?,
                        64,
                        ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
            }

            t0 += tlen;
        }
    }
    keepalive.keep_u8(&shared.q_fp8);
    keepalive.keep_f32(&shared.weights);
    Ok(owned_selected)
}

/// BATCHED CSA select over N decode rows (#60). Mirrors blocks (b)-(f) of
/// [`csa_select_official`] but batches the READ side (paged-MQA logits + topk)
/// of all N rows into ONE `batch_size=N` DeepGEMM call, replacing N
/// `batch_size=1` calls. The PER-ROW CACHE WRITES (hadamard rotate +
/// fused_store + `packed_rows` advance — block (a) of `csa_select_official`)
/// stay per-row and have ALREADY run before this is called (via
/// `csa_select`→`csa_select_official` in cache-write-only mode). This fn does NOT
/// touch any per-slot cache state; the read side reads the shared DSA key pool by
/// the per-row block_table band.
///
/// Never graph-replayed (the batched lane is eager, N>1), so unlike
/// `csa_select_official` it does NOT early-return on `ARLE_DSV4_DECODE_GRAPH`.
///
/// `context_lens_host` / `positions_host` are captured ON HOST during the per-row
/// prepare (`context_lens[r] = min(key_count_r, abs_pos_r/ratio)`,
/// `positions[r] = abs_pos_r`) — byte-equivalent VALUES to the single-row GPU
/// `dsv4_dsa_fill_context_lens_positions_start_pos_cuda` fill, since each row's
/// exact `key_count_r` is captured rather than assumed uniform.
///
/// Mutated device buffers (ALL stream-ordered, overwritten-before-read each
/// forward; per-slot cache state is UNTOUCHED by this read side):
///   - `shared.q_fp8_batch`         [N*heads*head_dim u8]  — fused Q quant out
///   - `shared.weights_batch`       [N*heads f32]          — fused weights out
///   - `shared.context_lens_batch`  [N i32]                — H2D from host
///   - `shared.positions_batch`     [N i32]                — H2D from host
///   - `shared.block_table_batch`   [N*num_pages i32]      — H2D per-row bands
///   - `shared.logits_batch`        [N*logits_stride f32]  — paged-MQA logits
///   - `shared.raw_indices_batch`   [N*index_topk i32]     — topk raw out
///   - `shared.sched_meta`          [(num_sms+1)*2 i32]    — batch-independent size
///   - `out_selected` [N*index_topk i32] — topk slot-relative indices (the
///     FlashMLA batch scratch's `selected_batched`; written here BEFORE
///     `build_layer_batch_meta` reads it)
///   - `shared.page_table_identity_batch` is READ-ONLY (identity).
///
/// `use_device_meta` (gate, default OFF): when true the (b1)/(b2) host builds +
/// 3 `memcpy_htod` are replaced by ONE on-device `dsv4_dsa_build_select_meta_cuda`
/// launch from device inputs (`start_pos_device`/slot_ids/key_counts), removing
/// the per-step H2D (graph-capturable). `ratio` is the layer compress ratio
/// (=1 for SparseIndexed) — the host's `context_lens[r]=min(abs_pos/ratio,
/// key_count_r)` divisor. `start_pos_device` is the [n] abs-position array
/// (== `positions`). `key_counts`/`meta_keepalive` only consulted when on.
#[allow(clippy::too_many_arguments)]
pub(crate) fn csa_select_official_batched(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    layer_idx: usize,
    q_i_batch: &HiddenStates,
    weights_batch: &HiddenStates,
    shared: &mut Dsv4DsaSharedScratch,
    pool: &Dsv4LayerKvLayout,
    n: usize,
    slot_ids: &[usize],
    context_lens_host: &[i32],
    positions_host: &[i32],
    local_index_heads: usize,
    score_scale: f32,
    out_selected: &mut CudaSlice<i32>,
    keepalive: &mut Dsv4ForwardKeepalive,
    use_device_meta: bool,
    ratio: i32,
    start_pos_device: Option<&CudaSlice<i32>>,
    key_counts: &[i32],
    meta_keepalive: &mut Vec<CudaSlice<i32>>,
    // GRAPH-SAFE device-meta inputs (single-row decode-graph lane). When BOTH are
    // `Some`, the device-meta path reads these PERSISTENT device buffers instead
    // of doing two per-step `upload_i32` (H2D allocs the capture audit rejects).
    // The eager batched lane passes `None` → the `upload_i32` path is unchanged.
    // `slot_ids_dev` holds the n slot indices (constant per slot); `key_counts_dev`
    // the n key capacities (constant for the graph). Require `use_device_meta`.
    slot_ids_dev: Option<&CudaSlice<i32>>,
    key_counts_dev: Option<&CudaSlice<i32>>,
) -> Result<()> {
    ensure!(
        n > 0 && n <= shared.decode_max_batch,
        "DSv4 batched DSA n {} outside [1, decode_max_batch {}]",
        n,
        shared.decode_max_batch
    );
    ensure!(
        local_index_heads == shared.num_heads && config.index_head_dim == shared.head_dim,
        "DSv4 batched DSA shape mismatch local_heads={} official_heads={} dim={} official_dim={}",
        local_index_heads,
        shared.num_heads,
        config.index_head_dim,
        shared.head_dim
    );
    ensure!(
        q_i_batch.seq_len == n && weights_batch.seq_len == n,
        "DSv4 batched DSA staging seq_len q={} w={} != n {}",
        q_i_batch.seq_len,
        weights_batch.seq_len,
        n
    );
    // Host context_lens/positions only consumed by the host (H2D) path; the
    // device-meta path computes them on-device and passes empty Vecs.
    ensure!(
        slot_ids.len() == n
            && (use_device_meta || (context_lens_host.len() == n && positions_host.len() == n)),
        "DSv4 batched DSA host arrays slot_ids={} lens={} pos={} != n {} (device_meta={})",
        slot_ids.len(),
        context_lens_host.len(),
        positions_host.len(),
        n,
        use_device_meta
    );
    ensure!(
        q_i_batch.hidden_dim == local_index_heads * config.index_head_dim,
        "DSv4 batched DSA q width {} != local_index_heads {} * index_head_dim {}",
        q_i_batch.hidden_dim,
        local_index_heads,
        config.index_head_dim
    );
    ensure!(
        weights_batch.hidden_dim == local_index_heads,
        "DSv4 batched DSA weights width {} != local_index_heads {}",
        weights_batch.hidden_dim,
        local_index_heads
    );
    ensure!(
        out_selected.len() >= n * config.index_topk,
        "DSv4 batched DSA out_selected len {} < n {} * index_topk {}",
        out_selected.len(),
        n,
        config.index_topk
    );

    let num_pages = shared.num_pages;

    if use_device_meta {
        // (b0) DEVICE path: ONE kernel computes block_table/context_lens/positions
        // on-device from device inputs, byte-identical to the (b1)/(b2) host builds
        // — no per-step H2D.
        let start_pos = start_pos_device.ok_or_else(|| {
            anyhow!("DSv4 batched DSA device-meta: start_pos_device (batched_positions) missing")
        })?;
        ensure!(
            start_pos.len() >= n,
            "DSv4 batched DSA device-meta start_pos len {} < n {}",
            start_pos.len(),
            n
        );
        // GRAPH-SAFE lane: caller pre-staged slot_ids/key_counts into PERSISTENT
        // device buffers (no per-step `upload_i32` → no host-memcpy node the
        // capture audit rejects). Eager lane (both None): upload here as before,
        // holding the uploads in `meta_keepalive` so they outlive the async launch.
        let (slot_ids_owned, key_counts_owned) = match (slot_ids_dev, key_counts_dev) {
            (Some(_), Some(_)) => (None, None),
            (None, None) => {
                ensure!(
                    key_counts.len() == n,
                    "DSv4 batched DSA device-meta key_counts {} != n {}",
                    key_counts.len(),
                    n
                );
                let slot_ids_i32: Vec<i32> = slot_ids
                    .iter()
                    .map(|&s| {
                        i32::try_from(s)
                            .map_err(|_| anyhow!("DSv4 batched DSA slot_id {s} overflows i32"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                (
                    Some(crate::ops::upload_i32(ctx, &slot_ids_i32)?),
                    Some(crate::ops::upload_i32(ctx, key_counts)?),
                )
            }
            _ => anyhow::bail!(
                "DSv4 batched DSA device-meta: slot_ids_dev/key_counts_dev must both be Some or both None"
            ),
        };
        let slot_ids_ref = slot_ids_dev.unwrap_or_else(|| {
            slot_ids_owned
                .as_ref()
                .expect("eager lane uploaded slot_ids")
        });
        let key_counts_ref = key_counts_dev.unwrap_or_else(|| {
            key_counts_owned
                .as_ref()
                .expect("eager lane uploaded key_counts")
        });
        ensure!(
            slot_ids_ref.len() >= n && key_counts_ref.len() >= n,
            "DSv4 batched DSA device-meta buffers too small slot_ids={} key_counts={} n={}",
            slot_ids_ref.len(),
            key_counts_ref.len(),
            n
        );
        {
            let mut bt = shared.block_table_batch.slice_mut(0..n * num_pages);
            let mut lens = shared.context_lens_batch.slice_mut(0..n);
            let mut pos = shared.positions_batch.slice_mut(0..n);
            let (bt_ptr, _btg) = bt.device_ptr_mut(&ctx.stream);
            let (lens_ptr, _lg) = lens.device_ptr_mut(&ctx.stream);
            let (pos_ptr, _pg) = pos.device_ptr_mut(&ctx.stream);
            let (slot_ptr, _sg) = slot_ids_ref.device_ptr(&ctx.stream);
            let (sp_ptr, _spg) = start_pos.device_ptr(&ctx.stream);
            let (kc_ptr, _kcg) = key_counts_ref.device_ptr(&ctx.stream);
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            unsafe {
                ffi::dsv4_dsa_build_select_meta_cuda(
                    bt_ptr as *mut i32,
                    lens_ptr as *mut i32,
                    pos_ptr as *mut i32,
                    slot_ptr as *const i32,
                    sp_ptr as *const i32,
                    kc_ptr as *const i32,
                    i32::try_from(n)?,
                    i32::try_from(num_pages)?,
                    ratio,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("DSv4 batched DSA build_select_meta failed: {e}"))?;
            }
        }
        // Eager lane uploads outlive the async launch via `meta_keepalive`; the
        // graph-safe lane's buffers are persistent (owned by the caller) → nothing
        // to push.
        if let (Some(s), Some(k)) = (slot_ids_owned, key_counts_owned) {
            meta_keepalive.push(s);
            meta_keepalive.push(k);
        }
    } else {
        // (b1) per-row block_table band: row r → slot r's DSA band = `num_pages`
        // contiguous blocks based at `slot_idx * num_pages` (alignment proven in the
        // brief: dsa_slot_bytes = num_pages*64*(index_head_dim+4), block base =
        // slot_idx*num_pages, total pool blocks = num_slots*num_pages). H2D into
        // `block_table_batch[0..n*num_pages]`, block_table_stride = num_pages.
        {
            let mut block_table_h = Vec::with_capacity(n * num_pages);
            for &slot_idx in slot_ids.iter() {
                let base = slot_idx
                    .checked_mul(num_pages)
                    .ok_or_else(|| anyhow!("DSv4 batched DSA block table base overflow"))?;
                for b in 0..num_pages {
                    block_table_h.push(
                        i32::try_from(base + b)
                            .map_err(|_| anyhow!("DSv4 batched DSA block id overflows i32"))?,
                    );
                }
            }
            let mut bt = shared.block_table_batch.slice_mut(0..n * num_pages);
            ctx.stream
                .memcpy_htod(&block_table_h, &mut bt)
                .map_err(|e| anyhow!("DSv4 batched DSA block_table H2D failed: {e}"))?;
        }

        // (b2) context_lens / positions: H2D the host-captured byte-equivalent values.
        {
            let mut lens = shared.context_lens_batch.slice_mut(0..n);
            ctx.stream
                .memcpy_htod(context_lens_host, &mut lens)
                .map_err(|e| anyhow!("DSv4 batched DSA context_lens H2D failed: {e}"))?;
            let mut pos = shared.positions_batch.slice_mut(0..n);
            ctx.stream
                .memcpy_htod(positions_host, &mut pos)
                .map_err(|e| anyhow!("DSv4 batched DSA positions H2D failed: {e}"))?;
        }
    }

    // (c) fused Q indexer rope+hadamard+quant over the N gathered rows.
    {
        let (q_ptr, _qg) = q_i_batch.data.device_ptr(&ctx.stream);
        let (q_fp8_ptr, _qfg) = shared.q_fp8_batch.device_ptr_mut(&ctx.stream);
        let (w_ptr, _wg) = weights_batch.data.device_ptr(&ctx.stream);
        let (weights_out_ptr, _wog) = shared.weights_batch.device_ptr_mut(&ctx.stream);
        let (freqs_ptr, _fg) = shared.freqs_cis.device_ptr(&ctx.stream);
        let positions = shared.positions_batch.slice(0..n);
        let (positions_ptr, _pg) = positions.device_ptr(&ctx.stream);
        // SAFETY: ptrs from live device allocations sized to the dims passed.
        unsafe {
            ffi::dsv4_dsa_fused_q_indexer_rope_hadamard_quant_cuda(
                q_ptr as *const ffi::Half,
                q_fp8_ptr as *mut u8,
                w_ptr as *const ffi::Half,
                weights_out_ptr as *mut f32,
                score_scale,
                freqs_ptr as *const f32,
                positions_ptr as *const i32,
                i32::try_from(n)?,
                i32::try_from(local_index_heads)?,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }

    // (d) paged-MQA logits scheduling metadata for the N-row batch. sched_meta is
    // sized `(num_sms+1)*2` — batch-INDEPENDENT — but the kernel reads all N
    // context_lens to partition KV across SMs, so pass batch_size=n.
    // SAFETY: ptrs from live device allocations sized to the dims passed.
    unsafe {
        cuda_moe::dsv4_deepgemm_paged_mqa_logits_metadata(
            cache_ptr(&shared.context_lens_batch, ctx),
            cache_ptr(&shared.sched_meta, ctx),
            n,
            1,
            64,
            shared.num_sms,
            ctx.stream.cu_stream(),
        )
        .map_err(|e| anyhow!("DSv4 batched DSA metadata failed: {e}"))?;
    }

    // (e) fused paged FP8 MQA logits → logits_batch (N rows). The KV cache base
    // is the WHOLE shared DSA pool (NOT a per-slot slice); per-row routing is via
    // the block_table bands above. num_kv_blocks = decode_max_batch * num_pages
    // (TOTAL pool blocks); max_context_len = num_pages*64 (each row's band).
    {
        let cache_pool = pool
            .dsa_key_cache
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 batched DSA shared key-cache missing"))?;
        let (q_ptr, _qg) = shared.q_fp8_batch.device_ptr(&ctx.stream);
        let (cache_ptr_u8, _kg) = cache_pool.device_ptr(&ctx.stream);
        let (weights_ptr, _wg) = shared.weights_batch.device_ptr(&ctx.stream);
        let lens = shared.context_lens_batch.slice(0..n);
        let (lens_ptr, _lg) = lens.device_ptr(&ctx.stream);
        let block_table = shared.block_table_batch.slice(0..n * num_pages);
        let (block_ptr, _bg) = block_table.device_ptr(&ctx.stream);
        let (sched_ptr, _sg) = shared.sched_meta.device_ptr(&ctx.stream);
        let (logits_ptr, _og) = shared.logits_batch.device_ptr_mut(&ctx.stream);
        let num_kv_blocks = shared
            .decode_max_batch
            .checked_mul(num_pages)
            .ok_or_else(|| anyhow!("DSv4 batched DSA num_kv_blocks overflow"))?;
        // SAFETY: ptrs from live device allocations sized to the dims passed.
        unsafe {
            ffi::dsv4_deepgemm_fp8_paged_mqa_logits_fused_cache_cuda(
                q_ptr as *const u8,
                cache_ptr_u8 as *const u8,
                weights_ptr as *const f32,
                lens_ptr as *const i32,
                block_ptr as *const i32,
                sched_ptr as *const i32,
                logits_ptr as *mut f32,
                i32::try_from(n)?,
                1,
                i32::try_from(local_index_heads)?,
                i32::try_from(config.index_head_dim)?,
                i32::try_from(num_kv_blocks)?,
                64,
                i32::try_from(num_pages * 64)?,
                i32::try_from(shared.logits_stride)?,
                i32::try_from(num_pages)?,
                i32::try_from(64 * (config.index_head_dim + std::mem::size_of::<f32>()))?,
                i32::try_from(shared.num_sms)?,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|e| anyhow!("DSv4 batched DSA paged logits failed: {e}"))?;
        }
    }

    // Diagnostic-only (`ARLE_PROBE_STAGES`): DSA raw indexer SCORE fingerprint,
    // captured AFTER (e) writes `logits_batch` and BEFORE (f)'s topk selection
    // reads/transforms it — the actual pre-selection float values, not just
    // which indices topk picks. `positions_host` is empty on the device-meta
    // (graph) lane, where this capture is skipped (untagged pos is useless).
    if !positions_host.is_empty() {
        for (r, &pos) in positions_host.iter().enumerate().take(n) {
            let pos = pos as u64;
            if !crate::probe::stage_want(pos) {
                continue;
            }
            let count = context_lens_host
                .get(r)
                .copied()
                .unwrap_or(shared.logits_stride as i32)
                .clamp(0, shared.logits_stride as i32) as usize;
            if count == 0 {
                continue; // no valid keys yet (early position) — nothing to fingerprint.
            }
            let start = r * shared.logits_stride;
            let view = shared.logits_batch.slice(start..start + count);
            crate::probe::stage_f32(ctx, "dsa_raw_score", layer_idx, r, pos, view);
        }
    }

    // (f) topk transform: read logits_batch (per-row stride logits_stride), write
    // the N rows of `out_selected` (slot-relative indices) and `raw_indices_batch`.
    // The page_table is the N-row identity (stride=num_pages, validator rejects 0);
    // each row reads identity `[0..num_pages)` → `page_to_slot(identity,i)=i`,
    // byte-equivalent to the single-row path's slot-relative mapping.
    {
        let lens = shared.context_lens_batch.slice(0..n);
        let (logits_ptr, _lg) = shared.logits_batch.device_ptr(&ctx.stream);
        let (lens_ptr, _csg) = lens.device_ptr(&ctx.stream);
        let page_table = shared.page_table_identity_batch.slice(0..n * num_pages);
        let (page_ptr, _ptg) = page_table.device_ptr(&ctx.stream);
        let mut sel = out_selected.slice_mut(0..n * config.index_topk);
        let (sel_ptr, _seg) = sel.device_ptr_mut(&ctx.stream);
        let mut raw = shared.raw_indices_batch.slice_mut(0..n * config.index_topk);
        let (raw_ptr, _rig) = raw.device_ptr_mut(&ctx.stream);
        // SAFETY: ptrs from live device allocations sized to the dims passed.
        unsafe {
            ffi::dsv4_deepseek_v4_topk_transform_cuda(
                logits_ptr as *const f32,
                lens_ptr as *const i32,
                page_ptr as *const i32,
                sel_ptr as *mut i32,
                raw_ptr as *mut i32,
                i64::try_from(shared.logits_stride)?,
                i64::try_from(num_pages)?,
                i64::try_from(config.index_topk)?,
                i32::try_from(n)?,
                i32::try_from(config.index_topk)?,
                64,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }

    keepalive.keep_u8(&shared.q_fp8_batch);
    keepalive.keep_f32(&shared.weights_batch);
    Ok(())
}
