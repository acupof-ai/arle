use anyhow::{Result, anyhow, bail, ensure};
use cuda_kernels::attention as flash_kv;
use cuda_kernels::ffi;
use cuda_kernels::moe as cuda_moe;
use cuda_kernels::prelude::{
    DeviceContext, DeviceMatrix, DeviceVec, HiddenStates, HiddenStatesView,
};
use cuda_kernels::quant_linear as cuda_ql;
use cuda_kernels::tensor::{RawDevicePtr, WeightFormat, cache_ptr};
use cuda_kernels::tensor_ops;
use cuda_kernels::{BandPage, KVFormat, TokenKVPool};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use deepseek_spec::{DeepSeekV4AttentionMode, DeepSeekV4Config};
use infer_seam::{KvBatchDescriptor, KvBatchRowKind};
use std::sync::atomic::{AtomicI8, Ordering};

use crate::dsv4::{
    Dsv4Attention, Dsv4Compressor, Dsv4ForwardKeepalive, Dsv4Indexer, Dsv4MlaKvArena,
};
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
/// GLM-5.2 V32 model-type int for the FlashMLA sparse decode shim: 576-wide latent q
/// (512 NoPE + 64 RoPE), 512-wide latent output, 656 B/tok packed KV.
const DSV4_FLASHMLA_V32: i32 = 0;
const DSV4_FLASHMLA_S_Q: usize = 1;
/// Packed bytes/token the FlashMLA sparse-FP8 decode reads for the MODEL1
/// NoPE=448 / RoPE=64 shape.
const DSV4_FLASH_KV_BYTES_PER_TOKEN_I32: i32 = 584;
/// Packed bytes/token for V32 (NoPE=512 / RoPE=64), inline layout: 512 (NoPE fp8)
/// + 16 (4x F32 block scales) + 128 (bf16 rope) = 656.
const DSV4_V32_KV_BYTES_PER_TOKEN_I32: i32 = 656;

/// FlashMLA model-family dims resolved from the config's attention shape: the single
/// `(head_dim, rope, kv_lora) -> (model_type, bytes/tok, d_v)` table. `d_v` (output
/// latent) is 512 for both families; MODEL1's d_qk==d_v==512, V32's d_qk=576.
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

static DSV4_FUSED_WQKV_DECODE_OVERRIDE: AtomicI8 = AtomicI8::new(DSV4_FLASHMLA_OVERRIDE_ENV);

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

/// Frozen-KV MTP verify: while set, `mla_attention` skips `dsv4_compressor_update`,
/// so the speculative verify forms no new compressed blocks and mutates nothing.
pub(crate) fn set_dsv4_verify_frozen(frozen: bool) {
    DSV4_VERIFY_FROZEN.store(frozen, Ordering::Relaxed);
}

pub(crate) fn dsv4_verify_frozen() -> bool {
    DSV4_VERIFY_FROZEN.load(Ordering::Relaxed)
}

/// Commit the ACCEPTED verify prefix into one layer's persistent state without
/// re-running the full forward: re-ingest compressor + indexer over the persisted
/// attn-normed rows, re-derive `k_prepared`, and roll both the bf16 and FP8 SW
/// rings. The packed compressed / DSA rows self-heal off the advanced
/// `compressed.seq_len`. The Q-side compute is discarded; `q_dummy` feeds the
/// prepare kernel's Q arm with zeros.
#[allow(clippy::too_many_arguments)]
pub(crate) fn commit_layer_fold(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    attention: &Dsv4Attention,
    mode: DeepSeekV4AttentionMode,
    compress_ratio: usize,
    state: &mut Dsv4LayerAttentionState,
    // Shared single-row FlashMLA decode scratch; the FP8 SW ring fold reuses its
    // `sw_bulk_*` buffers.
    flashmla_scratch: Option<&mut Dsv4FlashMlaDecodeScratch>,
    // Shared FP32 compressor-probe scratch: the re-ingestion runs the prefill lane,
    // so it consumes this.
    mut fp32_scratch: Option<&mut Dsv4CompressorFp32Scratch>,
    pool: &mut Dsv4LayerKvLayout,
    gathered: &HiddenStates,
    start_pos: usize,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<()> {
    let m = gathered.seq_len;
    ensure!(m > 0, "DSv4 commit fold needs at least the pending row");
    let head_dim = config.head_dim;
    let rope = RopeParams::from_config(config, compress_ratio)?;

    // Compressor + indexer ingestion (compressor layers only), non-frozen. GLM ships
    // no MTP so SparseIndexed never reaches here - fail loud.
    ensure!(
        mode != DeepSeekV4AttentionMode::SparseIndexed,
        "DSv4 commit_layer_fold (MTP commit) does not support SparseIndexed; GLM ships no MTP \
         (num_nextn_predict_layers==0) so this path is unreachable"
    );
    if mode.has_compressor() {
        let compressor = attention.compressor();
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
            Dsv4Position {
                start: start_pos,
                device: None,
            },
            rope,
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
                Dsv4Position {
                    start: start_pos,
                    device: None,
                },
                rope,
                fp32_scratch,
                None,
                None,
                keepalive,
            )?;
        }
    }

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
        {
            flash_kv::dsv4_prepare_qk_raw(
                &ctx.stream,
                q_raw_ptr,
                k_raw_ptr,
                q_out_ptr,
                k_out_ptr,
                m as i32,
                local_heads as i32,
                head_dim as i32,
                config.qk_rope_head_dim as i32,
                start_pos as i32,
                config.rms_norm_eps,
                rope.base,
                rope.original_seq_len,
                rope.factor,
                rope.beta_fast,
                rope.beta_slow,
            )?;
        }
    }
    keepalive.keep_hidden(&q_dummy);
    keepalive.keep_hidden(&q_discard);
    keepalive.keep_hidden(&k_prepared);

    update_bf16_sw_window(
        ctx,
        &mut state.sw_window_cache,
        &k_prepared,
        Dsv4Position {
            start: start_pos,
            device: None,
        },
        config,
    )?;

    // GLM pure-SparseIndexed (sliding_window==0) has no SW ring, and the `%
    // config.sliding_window` below would divide by zero - skip the fold.
    if config.sliding_window > 0
        && let Some(flash) = &mut state.flashmla
    {
        let scratch = flashmla_scratch.ok_or_else(|| {
            anyhow!("DSv4 commit fold: FlashMLA arena present but shared decode scratch missing")
        })?;
        let page_block_size = 64;
        // Hand the kernel slot-LOGICAL pages + the device page table; it resolves
        // `block_id = table[logical]` into the dynamic pool.
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

// DSv4-Flash MLA attention core: a low-rank Q/KV projection, partial RoPE on the
// trailing `rope_dim` columns, a windowed attention with a per-head sink logit,
// and a low-rank O projection. All modes run FlashMLA sparse attention over the
// shared per-layer FP8 KV pool, dispatched on `compress_ratio`:
//   - 0        SlidingWindow: SW window only.
//   - 0<r<16   CompressedSparse: compressor + indexer top-k, then SW + selected.
//   - r>=16    HybridCompressed: compressor + SW + ALL compressed blocks.
//   - GLM DSA  SparseIndexed: indexer top-k over the full latent, no compressor.

/// Run one DSv4 FP8/FP4 block-scaled LoRA matmul: `out[N, T] = W[N, K] * x[K, T]`.
/// The MLA LoRA weights carry raw quant bytes plus E8M0 block scales, so the dense
/// bf16 [`gemm_batch`] cannot run them. `batch_size` is the token count.
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
    match weight.weight_format {
        WeightFormat::Dsv4Fp8BlockScaled => cuda_ql::dsv4_fp8_gemv_batch(
            ctx,
            qw,
            scales,
            &x.data,
            &mut out.data,
            x.seq_len,
            weight.rows,
            weight.cols,
            weight.dsv4_scale_rows,
            weight.dsv4_scale_cols,
        )?,
        WeightFormat::Dsv4Fp4BlockScaled => cuda_ql::dsv4_fp4_gemv_batch(
            ctx,
            qw,
            scales,
            &x.data,
            &mut out.data,
            x.seq_len,
            weight.rows,
            weight.cols,
            weight.dsv4_scale_rows,
            weight.dsv4_scale_cols,
        )?,
        other => bail!("mla_linear: expected DSv4 FP8/FP4 block-scaled weight, got {other:?}"),
    }
    Ok(())
}

/// Decode (M=1) FP8 projection through tensor-core DeepGEMM: quantize `input` into
/// the fused-wqkv FP8 scratch, then `dsv4_deepgemm_fp8_gemm_nt` with the
/// pre-repacked weight `cache`. Requires K <= the scratch hidden_dim; the scratch
/// may already have been consumed this step (safe, same stream).
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
    // M (token rows) drives both the pack-quantize active_counts[0] and the GEMM's m
    // arg. An m=1 call skips the H2D write below; a batched m=n caller MUST restore
    // active_counts to [1] afterwards.
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
        cuda_moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
            RawDevicePtr::from_raw(input_ptr),
            RawDevicePtr::from_raw(fp8_ptr),
            RawDevicePtr::from_raw(scale_ptr),
            RawDevicePtr::from_raw(active_experts_ptr),
            RawDevicePtr::from_raw(active_offsets_ptr),
            RawDevicePtr::from_raw(active_counts_ptr),
            1,
            scratch.max_m,
            k,
            scratch.scale_stride_m,
            stream,
        )
        .map_err(|e| anyhow!("DSv4 decode proj activation quantize failed: {e}"))?;
        cuda_moe::dsv4_deepgemm_fp8_gemm_nt(
            RawDevicePtr::from_raw(fp8_ptr),
            RawDevicePtr::from_raw(scale_ptr),
            RawDevicePtr::from_raw(weight_ptr),
            RawDevicePtr::from_raw(weight_scale_ptr),
            RawDevicePtr::from_raw(out_ptr),
            m,
            cache.rows,
            cache.cols,
            scratch.scale_stride_m,
            stream,
        )
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
    let m = input.seq_len;
    // `cache.cols == k` and the buffer-length bounds it implies are re-checked by
    // `decode_proj_deepgemm_raw`, so only the HiddenStates-dimension assertions live
    // here.
    ensure!(
        cache.rows == out.hidden_dim && input.hidden_dim == k && out.seq_len == m,
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

/// Prefill (M=token_count) residual projection via DeepGEMM: quantize `input` [m, k]
/// into the prefill FP8 scratch, then `dsv4_deepgemm_fp8_gemm_nt` with the
/// pre-repacked `cache`. Moves the prefill projections off the scalar
/// `dsv4_fp8_gemv_batch` (62% of mla_attn prefill). K <= scratch.max_k.
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
    // M is chunk-bounded: scratch.max_m = DSV4_PREFILL_QUERY_CHUNK >=
    // chunked_prefill_size, so this only trips on a misconfigured chunk size - fail
    // loud rather than write past the chunk-sized M*K scratch.
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
    // active_counts is initialized to [1] at scratch construction; skip the
    // per-step H2D at m=1 (decode) so the graph capture sees no host memcpy.
    if m != 1 {
        ctx.stream
            .memcpy_htod(&[active_count], &mut scratch.active_counts)
            .map_err(|e| anyhow!("DSv4 grouped wo_a active_counts H2D failed: {e}"))?;
    }
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
        cuda_moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
            RawDevicePtr::from_raw(input_ptr),
            RawDevicePtr::from_raw(fp8_ptr),
            RawDevicePtr::from_raw(scale_ptr),
            RawDevicePtr::from_raw(active_experts_ptr),
            RawDevicePtr::from_raw(active_offsets_ptr),
            RawDevicePtr::from_raw(active_counts_ptr),
            1,
            m,
            k,
            scale_stride_m,
            stream,
        )
        .map_err(|e| anyhow!("DSv4 grouped wo_a activation quantize failed: {e}"))?;
        cuda_moe::dsv4_deepgemm_fp8_gemm_nt(
            RawDevicePtr::from_raw(fp8_ptr),
            RawDevicePtr::from_raw(scale_ptr),
            RawDevicePtr::from_raw(weight_ptr),
            RawDevicePtr::from_raw(weight_scale_ptr),
            RawDevicePtr::from_raw(out_ptr),
            m,
            n,
            k,
            scale_stride_m,
            stream,
        )
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
    // M is chunk-bounded: scratch.max_m = DSV4_PREFILL_QUERY_CHUNK >=
    // chunked_prefill_size, so this only trips on a misconfigured chunk size - fail
    // loud rather than write past the chunk-sized M*K activation scratch.
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
        flash_kv::dsv4_tp_out_slice_raw(
            &ctx.stream,
            qkv_ptr,
            cq_ptr,
            m as i32,
            n as i32,
            scratch.q_lora_rank as i32,
            0,
        )
        .map_err(|e| anyhow!("DSv4 fused wqkv prefill c_q slice failed: {e}"))?;
        let (kv_ptr, _kv_guard) = kv_raw.data.device_ptr_mut(&ctx.stream);
        flash_kv::dsv4_tp_out_slice_raw(
            &ctx.stream,
            qkv_ptr,
            kv_ptr,
            m as i32,
            n as i32,
            scratch.head_dim as i32,
            scratch.q_lora_rank as i32,
        )
        .map_err(|e| anyhow!("DSv4 fused wqkv prefill kv slice failed: {e}"))?;
    }
    Ok(())
}

/// Run one DSv4 linear `out = W * x` dispatching on the weight's on-disk format:
/// bf16 dense -> [`crate::ops::gemm_batch`]; FP8/FP4 block-scaled -> [`mla_linear`].
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
    // Compile-gated via `cuda_kernels::HAS_FLASHMLA` - a build without FlashMLA
    // reports false and decode is unavailable.
    Ok(cuda_kernels::HAS_FLASHMLA)
}

/// Native DeepGEMM availability gate for all DSv4 FP8 projection lanes (one shared
/// compile-time feature). Licensed on the TP=8/EP=8 H20 pod: prefill -47% at
/// M=1024, decode +2.5% tok/s. Scalar FP8-GEMV fallback when it is absent.
fn dsv4_deepgemm_enabled() -> bool {
    cuda_kernels::has_deepgemm_native()
}

/// Pages one layer's FlashMLA shared-pool band needs at `max_seq_len` (`sw_blocks +
/// comp_blocks`). `Ok(0)` when the FlashMLA decode-alloc path is disabled - no pool
/// is built. Lets [`crate::dsv4::Dsv4Model::kv_budget_plan`] reject an unaffordable
/// startup with a clean error instead of a panic in `kv_layout.rs` (pod-verified:
/// the two gates disagreeing crashes every worker rank).
pub(crate) fn dsv4_flashmla_slot_pages(
    config: &DeepSeekV4Config,
    mode: DeepSeekV4AttentionMode,
    compress_ratio: usize,
    max_seq_len: usize,
    page_block_size: usize,
) -> Result<usize> {
    if !cuda_kernels::HAS_FLASHMLA {
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

/// Whether a layer's FlashMLA band is DEMAND-PAGED: comp pages allocate from the
/// layer pool's free list as the sequence grows. MODEL1 only - its pack/index
/// kernels route slot-logical blocks through the device page table, so arbitrary
/// physical pages are safe. V32/GLM (`head_dim == 576`) keeps the identity full
/// band its pack lane addresses contiguously.
pub(crate) fn dsv4_flashmla_demand_paged(config: &DeepSeekV4Config) -> bool {
    config.head_dim != 576
}

/// Per-slot safety pages a demand-paged comp region needs on top of the shared
/// `pool_tokens` capacity: ceil-rounding of per-slot comp pages (+1) plus the MTP
/// verify margin crossing one page boundary (+1).
pub(crate) const DSV4_COMP_SAFETY_PAGES_PER_SLOT: usize = 2;

/// Pages ONE layer's FlashMLA shared pool is sized to - the one formula shared by
/// `kv_budget_plan` (solving `pool_tokens`) and `Dsv4LayerKvLayout::new`
/// (allocating), so the two cannot drift. Identity layers: `num_slots` full bands.
/// Demand-paged layers: per-slot ring blocks + comp safety
/// ([`DSV4_COMP_SAFETY_PAGES_PER_SLOT`]) + the shared comp capacity.
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

pub(crate) fn dsv4_fused_wqkv_decode_enabled() -> Result<bool> {
    match DSV4_FUSED_WQKV_DECODE_OVERRIDE.load(Ordering::Relaxed) {
        DSV4_FLASHMLA_OVERRIDE_OFF => return Ok(false),
        DSV4_FLASHMLA_OVERRIDE_ON => return Ok(true),
        _ => {}
    }
    // Default ON: fuse wq_a|wkv_a into one FP8 DeepGEMM instead of the scalar
    // `dsv4_fp8_gemv_batch_kernel` (16.9% of decode GPU). Licensed on the TP=8/EP=8
    // pod, 64-tok same-binary A/B: 31.774 -> 37.633 tok/s (+18.4%), token-exact.
    // Runtime preflight probe `cuda_kernels::has_deepgemm_native()`; scalar fallback.
    Ok(cuda_kernels::has_deepgemm_native())
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
    // Slot-LOGICAL pages + device page table; the kernel resolves
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
    // rope offset = head_dim - rope_dim (MODEL1 448, V32 512), stride = head_dim.
    // Only the pack fn differs: V32 writes the inline 656 B/tok layout
    // [512 NoPE fp8][4 F32 scales @512][128 rope bf16] (4x 128-elem NoPE blocks,
    // F32 scale = amax/448) per the vendored decode.
    let rope_ptr = nope_ptr + (config.head_dim - config.qk_rope_head_dim) as u64 * 2;
    if config.head_dim == 576 {
        // V32/GLM has no device-page-table pack kernel; its band stays contiguous, so
        // keep band-base addressing and slice the slot's range.
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
        // MODEL1: the device fill kernel produced a slot-LOGICAL block id, so hand the
        // POOL base + device page table (`block_id = table[logical]`).
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
    // The ONE block->(page,row) map for this slot's band: the device pack and the bulk
    // host-derived block_ids both draw sw_blocks / page_size from here (#146).
    let bmap = flash.block_map();
    // Steady-state decode adds AT MOST one compressed row per step, packed by the
    // DEVICE kernel below - fully derived from `start_pos_device`, so it records into
    // CUDA-graph captures and stays correct on replay. The host bulk path remains
    // ONLY for multi-row gaps, which always execute eagerly.
    {
        // Hand the POOL base + device page table so the slot-LOGICAL compressed block
        // routes to its physical pool block (fragmented band safe).
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
    // Eager contexts only: the device kernel above covered the single-row case, so
    // bulk-pack only multi-row gaps. The boundary row may be packed by both paths -
    // idempotent overwrite of identical data.
    flash.fp8_kv_comp_packed_rows = end_row;
    if end_row == start_row + 1 {
        return Ok(());
    }
    let n = end_row - start_row;
    // Bulk rebuild volume scales with the restored length (~584 B/row/layer, e.g.
    // matched=8064 -> ~1.2 MB/layer, ~25 MB across the 21 CSA layers).
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

        // `block_ids` carries slot-LOGICAL pages; hand the POOL base + device page
        // table
        // so the kernel routes each to its dynamic physical block.
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

/// Host-side tail slice for the SW ring update. Only the last `window` rows of a
/// `seq_len`-row batch can survive in the ring; writing the earlier rows too makes
/// unordered same-slot writers race in `dsv4_update_window_cache_kernel`
/// (`slot = (start_pos+token) % window`). Returns `(rows_skipped,
/// adjusted_start_pos, rows_to_write)`; a no-op at `seq_len <= window`.
fn update_bf16_sw_window(
    ctx: &DeviceContext,
    sw_window_cache: &mut CudaSlice<half::bf16>,
    k_prepared: &HiddenStates,
    pos: Dsv4Position<'_>,
    config: &DeepSeekV4Config,
) -> Result<()> {
    // GLM pure-SparseIndexed has no window (sliding_window==0) and the window-cache
    // update kernel would divide by zero.
    if config.sliding_window == 0 {
        return Ok(());
    }
    let (k_ptr, _kg) = k_prepared.data.device_ptr(&ctx.stream);
    let (window_ptr, _wg) = sw_window_cache.device_ptr_mut(&ctx.stream);
    {
        if let Some(start_pos_device) = pos.device {
            // start_pos lives on device - the host can't tail-slice. Decode/MTP rows
            // are far
            // below the window; oversize here is a bug, not a path.
            ensure!(
                k_prepared.seq_len <= config.sliding_window,
                "DSv4 SW ring device-start_pos update rows {} > window {} (host tail-slice \
                 unavailable; the update kernel would race same-slot writers)",
                k_prepared.seq_len,
                config.sliding_window
            );
            let (start_ptr, _sg) = start_pos_device.device_ptr(&ctx.stream);
            flash_kv::dsv4_update_window_cache_start_pos_ptr_raw(
                &ctx.stream,
                k_ptr,
                window_ptr,
                k_prepared.seq_len as i32,
                start_ptr,
                config.sliding_window as i32,
                config.head_dim as i32,
            )?;
        } else {
            let skip = k_prepared.seq_len.saturating_sub(config.sliding_window);
            let start_pos = pos.start + skip;
            let rows = k_prepared.seq_len - skip;
            let k_ptr = k_ptr + (skip * config.head_dim * 2) as u64;
            flash_kv::dsv4_update_window_cache_raw(
                &ctx.stream,
                k_ptr,
                window_ptr,
                rows as i32,
                start_pos as i32,
                config.sliding_window as i32,
                config.head_dim as i32,
            )?;
        }
    }
    Ok(())
}

/// Batched DSv4 sparse-verify attention metadata. Row `r` is at `positions[r]` and
/// attends committed KV plus the listed earlier chunk ancestors and self.
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

fn flashmla_prefill_attention(
    attn: &Dsv4AttnCtx<'_>,
    q_prepared: &HiddenStates,
    k_prepared: &HiddenStates,
    selected: Option<&CudaSlice<i32>>,
    compressed: Option<&HiddenStates>,
    sw_window_cache: &mut CudaSlice<half::bf16>,
    local_heads: usize,
    local_attn: &mut HiddenStates,
    sm_scale: f32,
    rope: RopeParams,
) -> Result<()> {
    let Dsv4AttnCtx {
        ctx,
        config,
        attention,
        mode,
        compress_ratio,
        tp,
        pos,
        chain_verify,
        ..
    } = *attn;
    let start_pos = pos.start;
    ensure!(
        cuda_kernels::HAS_FLASHMLA,
        "DSv4 FlashMLA prefill is not available"
    );
    ensure!(
        q_prepared.seq_len > 1,
        "DSv4 FlashMLA prefill requires seq_len > 1, got {}",
        q_prepared.seq_len
    );
    // MODEL1 (head_dim=512) + V32 (GLM, head_dim=576 = 512 latent + 64 rope).
    dsv4_flashmla_model_meta(config)?;
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
            {
                flash_kv::flashmla_csa_pack_kv_raw(
                    &ctx.stream,
                    kv_ptr,
                    window_ptr,
                    k_ptr,
                    comp_ptr as u64,
                    start_pos as i32,
                    config.sliding_window as i32,
                    token_count as i32,
                    compressed_count as i32,
                    config.head_dim as i32,
                )
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
                    {
                        flash_kv::flashmla_chain_verify_build_indices_raw(
                            &ctx.stream,
                            indices_ptr,
                            topk_ptr,
                            pos_ptr,
                            anc_ptr,
                            meta.max_anc as i32,
                            sel_ptr as u64,
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
                        )
                        .map_err(|e| anyhow!("DSv4 FlashMLA chain verify indices failed: {e}"))?;
                    }
                } else {
                    {
                        match mode {
                            DeepSeekV4AttentionMode::CompressedSparse => {
                                let selected = selected.ok_or_else(|| {
                                    anyhow!("DSv4 FlashMLA CSA prefill missing selected topk")
                                })?;
                                let (selected_ptr, _sg) = selected.device_ptr(&ctx.stream);
                                flash_kv::flashmla_csa_build_indices_raw(
                                    &ctx.stream,
                                    indices_ptr,
                                    topk_ptr,
                                    selected_ptr,
                                    token_count as i32,
                                    start_pos as i32,
                                    config.sliding_window as i32,
                                    config.index_topk as i32,
                                    compressed_count as i32,
                                    compress_ratio as i32,
                                )
                                .map_err(|e| {
                                    anyhow!("DSv4 FlashMLA CSA prefill indices failed: {e}")
                                })?;
                            }
                            DeepSeekV4AttentionMode::HybridCompressed => {
                                flash_kv::flashmla_hca_build_indices_raw(
                                    &ctx.stream,
                                    indices_ptr,
                                    topk_ptr,
                                    token_count as i32,
                                    start_pos as i32,
                                    config.sliding_window as i32,
                                    max_compressed_keys as i32,
                                    compressed_count as i32,
                                    compress_ratio as i32,
                                )
                                .map_err(|e| {
                                    anyhow!("DSv4 FlashMLA HCA prefill indices failed: {e}")
                                })?;
                            }
                            DeepSeekV4AttentionMode::SlidingWindow => {
                                // SWA has no compressed pool - reuse the CSA index
                                // builder with selected=null
                                // (fills -1) and index_topk=0, so indices hold SW
                                // blocks only.
                                flash_kv::flashmla_csa_build_indices_raw(
                                    &ctx.stream,
                                    indices_ptr,
                                    topk_ptr,
                                    0,
                                    token_count as i32,
                                    start_pos as i32,
                                    config.sliding_window as i32,
                                    max_compressed_keys as i32,
                                    compressed_count as i32,
                                    compress_ratio as i32,
                                )
                                .map_err(|e| {
                                    anyhow!("DSv4 FlashMLA SWA prefill indices failed: {e}")
                                })?;
                            }
                            DeepSeekV4AttentionMode::SparseIndexed => {
                                // GLM SparseIndexed mirrors the CSA index build but
                                // over the FULL per-token latent
                                // (no compressor): pass compress_ratio=1.
                                let selected = selected.ok_or_else(|| {
                                    anyhow!(
                                        "DSv4 FlashMLA SparseIndexed prefill missing selected topk"
                                    )
                                })?;
                                let (selected_ptr, _sg) = selected.device_ptr(&ctx.stream);
                                flash_kv::flashmla_csa_build_indices_raw(
                                    &ctx.stream,
                                    indices_ptr,
                                    topk_ptr,
                                    selected_ptr,
                                    token_count as i32,
                                    start_pos as i32,
                                    config.sliding_window as i32,
                                    config.index_topk as i32,
                                    compressed_count as i32,
                                    1,
                                )
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
                        // SAFETY: q spans token_count*local_width; gathered holds
                        // tp_world x that, on ctx.stream.
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
                        {
                            flash_kv::dsv4_tp_q_repack_raw(
                                &ctx.stream,
                                gather_ptr,
                                packed_ptr,
                                tp_world as i32,
                                token_count as i32,
                                local_heads as i32,
                                config.head_dim as i32,
                            )
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
            {
                flash_kv::flashmla_sm90_sparse_prefill_fwd_raw(
                    &ctx.stream,
                    q_for_flashmla as u64,
                    kv_ptr,
                    indices_ptr,
                    sink_ptr as u64,
                    topk_ptr,
                    flash_out_ptr as u64,
                    max_ptr,
                    lse_ptr,
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
                )
                .map_err(|e| anyhow!("DSv4 FlashMLA sparse prefill failed: {e}"))?;
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
                    {
                        flash_kv::dsv4_tp_out_slice_raw(
                            &ctx.stream,
                            full_out_ptr,
                            out_ptr,
                            token_count as i32,
                            global_width as i32,
                            local_width as i32,
                            (tp_rank * local_width) as i32,
                        )
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
                {
                    if let Some(meta) = chain_verify {
                        let (pos_ptr, _pg) = meta.positions.device_ptr(&ctx.stream);
                        flash_kv::dsv4_output_inverse_rope_batch_start_pos_raw(
                            &ctx.stream,
                            out_ptr,
                            token_count as i32,
                            local_heads as i32,
                            config.head_dim as i32,
                            config.qk_rope_head_dim as i32,
                            pos_ptr,
                            rope.base,
                            rope.original_seq_len,
                            rope.factor,
                            rope.beta_fast,
                            rope.beta_slow,
                        )
                        .map_err(|e| {
                            anyhow!("DSv4 FlashMLA chain verify output inverse-rope failed: {e}")
                        })?;
                    } else {
                        flash_kv::dsv4_output_inverse_rope_raw(
                            &ctx.stream,
                            out_ptr,
                            token_count as i32,
                            local_heads as i32,
                            config.head_dim as i32,
                            config.qk_rope_head_dim as i32,
                            start_pos as i32,
                            rope.base,
                            rope.original_seq_len,
                            rope.factor,
                            rope.beta_fast,
                            rope.beta_slow,
                        )
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
        update_bf16_sw_window(
            ctx,
            sw_window_cache,
            k_prepared,
            Dsv4Position {
                start: start_pos,
                device: None,
            },
            config,
        )?;
    }

    // Keep temporaries in scope until every launch using their raw pointers is
    // enqueued.
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

fn flashmla_decode_attention(
    attn: &Dsv4AttnCtx<'_>,
    q_prepared: &HiddenStates,
    k_prepared: &HiddenStates,
    selected: Option<&CudaSlice<i32>>,
    compressed: Option<&HiddenStates>,
    sw_window_cache: &mut CudaSlice<half::bf16>,
    flash: &mut Dsv4FlashMlaDecodeState,
    scratch: &mut Dsv4FlashMlaDecodeScratch,
    pool: &mut Dsv4LayerKvLayout,
    local_heads: usize,
    local_attn: &mut HiddenStates,
    sm_scale: f32,
    rope: RopeParams,
) -> Result<()> {
    let Dsv4AttnCtx {
        ctx,
        config,
        attention,
        mode,
        compress_ratio,
        tp,
        pos,
        ..
    } = *attn;
    ensure!(
        dsv4_flashmla_decode_enabled()?,
        "DSv4 FlashMLA decode is not available"
    );
    ensure!(
        q_prepared.seq_len == 1,
        "DSv4 FlashMLA decode requires seq_len == 1, got {}",
        q_prepared.seq_len
    );
    let start_pos_device = pos.device.ok_or_else(|| {
        anyhow!("DSv4 FlashMLA decode requires device start_pos for token_count=1")
    })?;
    // The FlashMLA shim reads q[heads, d_qk] and writes out[heads, d_v=512 latent]:
    // MODEL1 d_qk==d_v==512, V32 d_qk=576 (512 latent NoPE + 64 RoPE).
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

    // GLM pure-SparseIndexed (sliding_window==0) has no SW ring to bootstrap; the
    // per-token KV pack still runs, populating this token's latent into the sparse
    // pool the indexer selects from.
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

    // Indexer modes (CSA + GLM SparseIndexed) feed the per-row top-k `selected` and
    // share mode_int=1; SW/HCA pass selected_ptr=0. The V32 MODEL_TYPE is a separate
    // constant passed to the fwd kernel, not this.
    let mode_int = mode.flashmla_mode_int();
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
    // Route the device page table so build_indices emits POOL-ABSOLUTE physical
    // indices, fed against the whole-pool base below - this lets a slot draw
    // non-contiguous pages once the pool fragments. V32/GLM route here too: only the
    // WRITE side lacks a V32 page-table kernel, and their identity band stays equal.
    let build_page_table = Some(&flash.device_page_table);
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
                // GLM SparseIndexed: every token a key (ratio=1); CSA/HCA keep
                // compress_ratio.
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

    // topk_length + scheduler metadata are slot constants computed once at state init
    // (`init_constant_sched_meta`) - see the capture-hazard note there.

    let (q_ptr, q_guard) = q_prepared.data.device_ptr(&ctx.stream);
    // Feed the FlashMLA decode kernel the WHOLE-pool base; the indices built above are
    // POOL-ABSOLUTE (page-table routed).
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
                // SAFETY: q spans token_count*local_width; gathered holds tp_world x
                // that, on ctx.stream.
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
                {
                    flash_kv::dsv4_tp_q_repack_raw(
                        &ctx.stream,
                        gather_ptr,
                        packed_ptr,
                        tp_world as i32,
                        1,
                        local_heads as i32,
                        config.head_dim as i32,
                    )
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
    // q is [b, s_q, h_q, d_qk] with d_qk = head_dim (512 MODEL1 / 576 V32); out and
    // o_accum are [..., h_q, d_v] with d_v = kv_lora latent = 512 ALWAYS (the shim
    // hard-asserts d_v==512).
    let d_qk = config.head_dim as i32;
    let d_v = if is_v32 { 512 } else { config.head_dim as i32 };
    let stride_q = (global_heads * config.head_dim) as i32;
    let stride_o = (global_heads as i32) * d_v;
    let stride_indices = flash.topk_unified as i32;
    let stride_lse = global_heads as i32;
    {
        crate::profile::profile_op(ctx, "flashmla_fwd", None, 1, || {
            {
                flash_kv::flashmla_sm90_sparse_decode_fwd_raw(
                    &ctx.stream,
                    q_for_flashmla as u64,
                    pool_ptr,
                    indices_ptr,
                    topk_ptr,
                    sink_ptr as u64,
                    flash_out_ptr as u64,
                    lse_out_ptr,
                    lse_accum_ptr,
                    o_accum_ptr,
                    sched_ptr,
                    splits_ptr,
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
                )
                .map_err(|e| anyhow!("DSv4 FlashMLA sparse decode failed: {e}"))?;
            }
            Ok(())
        })?;
    }

    // The FlashMLA output is [heads, d_v] - slice and un-rotate on the OUTPUT width.
    let out_head_dim = d_v as usize;
    if tp_world > 1 {
        let (full_out_ptr, full_out_guard) = scratch.tp_full_out.device_ptr(&ctx.stream);
        {
            crate::profile::profile_op(ctx, "flashmla_out_slice", None, 1, || {
                {
                    flash_kv::dsv4_tp_out_slice_raw(
                        &ctx.stream,
                        full_out_ptr,
                        out_ptr,
                        1,
                        (global_heads * out_head_dim) as i32,
                        (local_heads * out_head_dim) as i32,
                        (tp_rank * local_heads * out_head_dim) as i32,
                    )
                    .map_err(|e| anyhow!("DSv4 FlashMLA TP out slice failed: {e}"))?;
                }
                Ok(())
            })?;
        }
        drop(full_out_guard);
    }

    // MODEL1's absorbed output [heads, 512] carries a rope tail to un-rotate. V32's
    // output is the pure kv_lora latent (NoPE only - the 64 rope dims live in q/k for
    // scoring), so there is no tail; its value side is reconstructed by w_vc (D3d).
    if !is_v32 {
        crate::profile::profile_op(ctx, "flashmla_inverse_rope", None, 1, || {
            {
                flash_kv::dsv4_output_inverse_rope_start_pos_ptr_raw(
                    &ctx.stream,
                    out_ptr,
                    1,
                    local_heads as i32,
                    config.head_dim as i32,
                    config.qk_rope_head_dim as i32,
                    start_ptr,
                    rope.base,
                    rope.original_seq_len,
                    rope.factor,
                    rope.beta_fast,
                    rope.beta_slow,
                )
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

    update_bf16_sw_window(ctx, sw_window_cache, k_prepared, pos, config)?;
    Ok(())
}

/// Canonical batched KV pack for the MODEL1 batched decode lane: ONE batched SW
/// one-token pack + ONE batched compressed-delta pack over all N rows, replacing
/// the N `flashmla_decode_pack_row` launches (the SW-ring bootstrap stays per-row).
/// All per-slot inputs are pre-gathered device-pointer arrays: `nope_arr[r]` /
/// `rope_arr[r]` are row r's `k_prepared` bases, `compressed_arr[r]` is 0/null for
/// rows with no compressor (kernel no-op), `page_table_arr[r]` is row r's per-slot
/// device page table. `start_pos` is the contiguous `[N]` decode positions;
/// `pool_ptr` is the single shared pool base, rows writing disjoint bands via their
/// page tables. MODEL1 only (head_dim==512); V32/SparseIndexed callers keep
/// [`flashmla_decode_pack_row`].
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
    // compress_ratio==0 layers have NO compressor - skip the completed-compressor
    // pack; the batched FFI rejects ratio<=0 with INVALID_VALUE.
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

/// Host-side rope-base offset for a row's `k_prepared` NoPE pointer
/// (rope = nope + (head_dim - qk_rope_head_dim)*2 B).
pub(crate) fn flashmla_pack_rope_offset_bytes(config: &DeepSeekV4Config) -> u64 {
    (config.head_dim - config.qk_rope_head_dim) as u64 * 2
}

/// Per-row KV pack for the batched decode lane: SW ring bootstrap -> one-token SW
/// pack -> compressed-delta, writing this row's slot KV into the shared pool. SW
/// passes `compressed = None`, HCA `Some(&state.compressor.compressed)`; CSA is not
/// routed here.
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

/// Per-row output tail for the batched decode lane: output inverse-RoPE on this
/// row's `local_attn`, then the bf16 SW-window update. `local_attn` must already
/// hold this rank's local-head output for this row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn flashmla_decode_finish_row(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    sw_window_cache: &mut CudaSlice<half::bf16>,
    k_prepared: &HiddenStates,
    local_attn: &mut HiddenStates,
    pos: Dsv4Position<'_>,
    local_heads: usize,
    rope: RopeParams,
) -> Result<()> {
    let start_pos_device = pos
        .device
        .expect("DSv4 flashmla_decode_finish_row requires device start_pos");
    let (out_ptr, out_guard) = local_attn.data.device_ptr_mut(&ctx.stream);
    let (start_ptr, start_guard) = start_pos_device.device_ptr(&ctx.stream);
    {
        crate::profile::profile_op(ctx, "flashmla_inverse_rope_batched", None, 1, || {
            // SAFETY: identical args to the single-row inverse-rope; out is one
            // local-head row (token_count=1), start_pos_device is this row's pos.
            {
                flash_kv::dsv4_output_inverse_rope_start_pos_ptr_raw(
                    &ctx.stream,
                    out_ptr,
                    1,
                    local_heads as i32,
                    config.head_dim as i32,
                    config.qk_rope_head_dim as i32,
                    start_ptr,
                    rope.base,
                    rope.original_seq_len,
                    rope.factor,
                    rope.beta_fast,
                    rope.beta_slow,
                )
                .map_err(|e| anyhow!("DSv4 batched FlashMLA output inverse-rope failed: {e}"))?;
            }
            Ok(())
        })?;
    }
    drop(out_guard);
    drop(start_guard);
    update_bf16_sw_window(ctx, sw_window_cache, k_prepared, pos, config)?;
    Ok(())
}

/// Canonical batched FINISH tail for the MODEL1 batched decode lane: ONE batched
/// output inverse-RoPE over N rows' non-contiguous `local_attn` buffers. `out_ptrs`
/// are the per-row `[local_width,1]` device pointers, `start_pos` the contiguous
/// `[N]` decode positions; the remaining params are uniform across the layer's
/// rows. MUST run BEFORE the per-row O-LoRA, which consumes `local_attn`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn flashmla_decode_inverse_rope_batched(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    out_ptrs: &CudaSlice<u64>,
    start_pos: &CudaSlice<i32>,
    n: usize,
    local_heads: usize,
    rope: RopeParams,
) -> Result<()> {
    crate::profile::profile_op(ctx, "flashmla_inverse_rope_batched_ptr", None, n, || {
        let (out_ptr, og) = out_ptrs.device_ptr(&ctx.stream);
        let (start_ptr, sg) = start_pos.device_ptr(&ctx.stream);
        // SAFETY: out_ptrs holds N valid `[local_width,1]` device pointers; start_pos
        // is `[N]`; the kernel grids N*local_heads blocks and indexes both per row.
        {
            flash_kv::dsv4_output_inverse_rope_batched_ptr_raw(
                &ctx.stream,
                out_ptr,
                n as i32,
                local_heads as i32,
                config.head_dim as i32,
                config.qk_rope_head_dim as i32,
                start_ptr,
                rope.base,
                rope.original_seq_len,
                rope.factor,
                rope.beta_fast,
                rope.beta_slow,
            )
            .map_err(|e| anyhow!("DSv4 batched FlashMLA output inverse-rope (ptr) failed: {e}"))?;
        }
        drop(og);
        drop(sg);
        Ok(())
    })?;
    Ok(())
}

/// Canonical batched FINISH tail for the MODEL1 batched decode lane: ONE batched
/// SW-window write over N rows' non-contiguous k_prepared / ring buffers. `k_ptrs`
/// / `cache_ptrs` are the per-row device pointers, `start_pos` the `[N]` decode
/// positions; each row writes its new key at `start_pos[r] % sliding_window`.
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
        {
            flash_kv::dsv4_update_window_cache_batched_ptr_raw(
                &ctx.stream,
                k_ptr,
                cache_ptr,
                n as i32,
                start_ptr,
                config.sliding_window as i32,
                config.head_dim as i32,
            )
            .map_err(|e| anyhow!("DSv4 batched FlashMLA SW window write (ptr) failed: {e}"))?;
        }
        drop(kg);
        drop(cg);
        drop(sg);
        Ok(())
    })?;
    Ok(())
}

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
    cuda_kernels::tensor_ops::rms_norm_batched(
        ctx,
        &x.data,
        0,
        &weight.data,
        &mut out.data,
        x.hidden_dim,
        x.seq_len,
        eps,
    )
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
    cuda_kernels::tensor_ops::rms_norm_batched(
        ctx,
        &x.data,
        offset,
        &weight.data,
        &mut out.data,
        width,
        1,
        eps,
    )
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
        match (dsv4_deepgemm_enabled(), attention.wq_b_deepgemm.as_ref()) {
            (true, Some(cache)) => {
                // wq_b (M=1) through tensor-core DeepGEMM: quantize c_q_normed into the
                // fused FP8
                // scratch (already consumed by the wq_a|wkv GEMM above, so safe to
                // reuse on this
                // stream), then a DeepGEMM dense GEMM.
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
                    // SAFETY: all buffers live on ctx.stream; K=q_lora_rank ≤
                    // hidden_dim
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

/// RoPE/YaRN parameters for a layer. Computed once from `config` + `compress_ratio`
/// and threaded through the attention path. Compressed layers (cr > 0) use
/// `compress_rope_theta` + YaRN(`original_max_position_embeddings`); pure-SW layers
/// (cr == 0) use `rope_theta` with no YaRN (`original_seq_len = 0`).
#[derive(Clone, Copy)]
pub(crate) struct RopeParams {
    pub(crate) base: f32,
    pub(crate) original_seq_len: i32,
    pub(crate) factor: f32,
    pub(crate) beta_fast: f32,
    pub(crate) beta_slow: f32,
}

impl RopeParams {
    pub(crate) fn from_config(config: &DeepSeekV4Config, compress_ratio: usize) -> Result<Self> {
        let rope = &config.rope_parameters;
        let (base, original_seq_len) = if compress_ratio > 0 {
            (config.compress_rope_theta, rope.original_seq_len_i32()?)
        } else {
            (config.rope_theta, 0i32)
        };
        Ok(Self {
            base,
            original_seq_len,
            factor: rope.factor,
            beta_fast: rope.beta_fast,
            beta_slow: rope.beta_slow,
        })
    }
}

/// A single row's absolute start position, carried as both a host `usize` (for
/// host-side math / ring-window indexing) and an optional device `i32` slice
/// (for kernels that read the position from device memory, e.g. decode/MTP
/// where the host cannot know the folded position). The two always describe the
/// same logical position; `device` is `None` on the prefill path where the host
/// `start` is authoritative.
#[derive(Clone, Copy)]
pub(crate) struct Dsv4Position<'a> {
    pub start: usize,
    pub device: Option<&'a CudaSlice<i32>>,
}

/// Immutable execution context for one DSv4 MLA attention call. Bundles the
/// per-call handles that `mla_attention` / `_prepare` / `_fwd` / `_decode` all
/// thread through, so the mutable per-call state (`state`, `pool`, scratch
/// buffers, `out`) stays explicit in the signature.
pub(crate) struct Dsv4AttnCtx<'a> {
    pub ctx: &'a DeviceContext,
    pub config: &'a DeepSeekV4Config,
    pub attention: &'a Dsv4Attention,
    pub mode: DeepSeekV4AttentionMode,
    pub compress_ratio: usize,
    pub layer_idx: usize,
    pub tp: &'a TpRuntime,
    pub pos: Dsv4Position<'a>,
    pub chain_verify: Option<&'a Dsv4ChainVerifyAttnMeta>,
}

/// Prepared-but-not-attended MLA state: the boundary between `mla_attention`'s
/// per-row PREPARE half (wq/wkv proj + RoPE +, for CSA/HCA, compressor / indexer /
/// `selected`) and its FWD half (the FlashMLA kernel + O-LoRA). The split lets the
/// batched decode lane run PREPARE per row, gather each row's `q_prepared`, then
/// issue ONE `sparse_decode_fwd(b=N)`. `selected` is the CSA per-row top-k (owned,
/// `[max_compressed_keys]`), `None` for SW/HCA; the compressed-key pool is
/// re-borrowed from `state` inside the fwd, so PREPARE must leave
/// `state.compressor` populated for CSA/HCA.
pub(crate) struct Dsv4MlaPrepared {
    pub(crate) q_prepared: HiddenStates,
    pub(crate) k_prepared: HiddenStates,
    /// Attention output scratch `[local_width, token_count]`, written by the fwd.
    pub(crate) local_attn: HiddenStates,
    pub(crate) selected: Option<CudaSlice<i32>>,
    pub(crate) local_heads: usize,
    pub(crate) sm_scale: f32,
    pub(crate) rope: RopeParams,
}

pub(crate) struct Dsv4MlaDecodeScratch {
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
}

impl Dsv4MlaDecodeScratch {
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
            .saturating_add(attention.wo_a().rows); // oproj_latent
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
            "DSv4 MODEL1 decode scratch is MODEL1-only; GLM/plain-o attention must use eager decode"
        );
        let local_width = attention.wq_b.rows;
        let oproj_rows = attention.wo_a().rows;
        let (compressor_main_kv, compressor_main_score) = if mode.has_compressor() {
            let compressor = attention.compressor();
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
                let indexer = attention.indexer();
                let compressor = indexer.compressor.as_ref().ok_or_else(|| {
                    anyhow!("DSv4 MODEL1 decode CSA scratch requires indexer compressor")
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
        let (csa_q_i, csa_weights, csa_selected) = if mode.has_indexer() {
            let indexer = attention.indexer();
            (
                // SAFETY: uninit device scratch; fully written before first read.
                Some(unsafe { HiddenStates::uninit(ctx, indexer.wq_b.rows, 1)? }),
                // SAFETY: uninit device scratch; fully written before first read.
                Some(unsafe { HiddenStates::uninit(ctx, indexer.weights_proj.rows, 1)? }),
                Some(
                    ctx.stream
                        .alloc_zeros::<i32>(config.index_topk)
                        .map_err(|e| {
                            anyhow!("DSv4 MODEL1 decode CSA selected scratch alloc failed: {e}")
                        })?,
                ),
            )
        } else {
            (None, None, None)
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
    }
}

/// One DSv4 MLA attention block (SlidingWindow / CompressedSparse /
/// HybridCompressed, dispatched on `mode` / `compress_ratio`).
///
/// `hidden` is the post-attn-LN input `[hidden_size, token_count]`; `attn.pos.start`
/// is the absolute position of its first token. Writes the O-LoRA output into `out`,
/// pre-TP-all-reduce (the model layer-loop owns the row-parallel sum).
///
/// The per-head `attn_sink` vector is loaded WHOLE on every rank, so the attention
/// kernel must skip to this rank's head block via `sink_offset = tp_rank *
/// local_heads` - otherwise every non-zero rank reads rank-0's sink logits and the
/// output diverges by a small head-dependent margin (multi-GPU only).
pub(crate) fn mla_attention(
    attn: &Dsv4AttnCtx<'_>,
    hidden: &HiddenStates,
    state: &mut Dsv4LayerAttentionState,
    pool: &mut Dsv4LayerKvLayout,
    dsa_shared: Option<&mut Dsv4DsaSharedScratch>,
    // Shared single-row FlashMLA decode scratch; consumed only on the single-row
    // decode FlashMLA path inside the fwd.
    flashmla_scratch: Option<&mut Dsv4FlashMlaDecodeScratch>,
    // Shared FP8 prefill DeepGEMM linear scratch. `Some` only when native DeepGEMM is
    // available and the caller threads it; `None` on the decode lane.
    mut prefill_shared: Option<&mut Dsv4PrefillDeepGemmLinearScratch>,
    // Shared FP32 probe scratch — contract on `compressor_forward`'s param.
    fp32_scratch: Option<&mut Dsv4CompressorFp32Scratch>,
    out: &mut HiddenStates,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<()> {
    // Single-row and chunked-prefill callers run PREPARE then FWD back-to-back; only
    // the batched decode lane calls the two halves separately.
    let prepared = mla_attention_prepare(
        attn,
        hidden,
        state,
        pool,
        dsa_shared,
        prefill_shared.as_deref_mut(),
        fp32_scratch,
        keepalive,
    )?;
    mla_attention_fwd(
        attn,
        state,
        pool,
        flashmla_scratch,
        prefill_shared,
        prepared,
        out,
        keepalive,
    )
}

#[allow(clippy::too_many_arguments)]
fn compressor_forward_decode(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    compressor: &Dsv4Compressor,
    hidden: &HiddenStates,
    state: &mut Dsv4CompressorState,
    head_dim: usize,
    ratio: usize,
    overlap: bool,
    pos: Dsv4Position<'_>,
    rope: RopeParams,
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
        "DSv4 MODEL1 decode compressor scratch mismatch: kv={}x{} score={}x{} expected {width}x1",
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
        pos,
        rope,
        None,
        Some((&*kv_scratch, &*score_scratch)),
        None,
        keepalive,
    )
}

pub(crate) fn mla_attention_decode(
    attn: &Dsv4AttnCtx<'_>,
    hidden: &HiddenStates,
    state: &mut Dsv4LayerAttentionState,
    pool: &mut Dsv4LayerKvLayout,
    dsa_shared: Option<&mut Dsv4DsaSharedScratch>,
    flashmla_scratch: Option<&mut Dsv4FlashMlaDecodeScratch>,
    scratch: &mut Dsv4MlaDecodeScratch,
    out: &mut HiddenStates,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<()> {
    let Dsv4AttnCtx {
        ctx,
        config,
        attention,
        mode,
        compress_ratio,
        layer_idx,
        tp,
        pos,
        ..
    } = *attn;
    ensure!(
        hidden.hidden_dim == config.hidden_size && hidden.seq_len == 1,
        "DSv4 MODEL1 decode MLA requires one hidden row [{}x1], got {}x{}",
        config.hidden_size,
        hidden.hidden_dim,
        hidden.seq_len
    );
    ensure!(
        attention.w_kc.is_none() && attention.w_vc.is_none() && attention.o_proj.is_none(),
        "DSv4 MODEL1 decode MLA is MODEL1-only; GLM/plain-o uses eager decode"
    );
    ensure!(
        pos.device.is_some(),
        "DSv4 MODEL1 decode MLA requires device start_pos"
    );
    let head_dim = config.head_dim;
    let local_width = attention.wq_b.rows;
    ensure!(
        local_width.is_multiple_of(head_dim),
        "DSv4 MODEL1 decode MLA local q width {local_width} is not a multiple of head_dim {head_dim}"
    );
    let local_heads = local_width / head_dim;
    ensure!(
        local_heads > 0,
        "DSv4 MODEL1 decode MLA requires local heads"
    );
    let tp_rank = tp.config().rank;
    let sink_offset = tp_rank * local_heads;
    ensure!(
        attention.wkv.rows == head_dim,
        "DSv4 MODEL1 decode MLA wkv rows {} != head_dim {head_dim}",
        attention.wkv.rows
    );
    ensure!(
        config.sliding_window > 0,
        "DSv4 MODEL1 decode MLA requires a non-zero sliding_window"
    );
    ensure!(
        config.qk_rope_head_dim <= head_dim,
        "DSv4 MODEL1 decode MLA rope dim {} exceeds head_dim {head_dim}",
        config.qk_rope_head_dim
    );
    ensure!(
        state.sw_window_cache.len() == config.sliding_window * head_dim,
        "DSv4 MODEL1 decode MLA SW window cache len {} != sliding_window*head_dim {}",
        state.sw_window_cache.len(),
        config.sliding_window * head_dim
    );
    ensure!(
        attention.attn_sink().len >= sink_offset + local_heads,
        "DSv4 MODEL1 decode MLA attn_sink len {} cannot cover rank {tp_rank} heads [{sink_offset}, {})",
        attention.attn_sink().len,
        sink_offset + local_heads
    );

    let rope = RopeParams::from_config(config, compress_ratio)?;
    if dsv4_fused_wqkv_decode_enabled()? {
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
        let (wq_a, wq_b, wkv) = (&attention.wq_a, &attention.wq_b, &attention.wkv);
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
        let start_pos_device = pos.device.expect("checked above");
        let (start_ptr, _sg) = start_pos_device.device_ptr(&ctx.stream);
        {
            flash_kv::dsv4_prepare_qk_start_pos_ptr_raw(
                &ctx.stream,
                q_raw_ptr,
                k_raw_ptr,
                q_out_ptr,
                k_out_ptr,
                1,
                local_heads as i32,
                head_dim as i32,
                config.qk_rope_head_dim as i32,
                start_ptr,
                config.rms_norm_eps,
                rope.base,
                rope.original_seq_len,
                rope.factor,
                rope.beta_fast,
                rope.beta_slow,
            )?;
        }
    }

    let mut selected_ready = false;
    if mode.has_compressor() {
        let compressor = attention.compressor();
        let kv = scratch
            .compressor_main_kv
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 MODEL1 decode main compressor kv scratch missing"))?;
        let score = scratch
            .compressor_main_score
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 MODEL1 decode main compressor score scratch missing"))?;
        compressor_forward_decode(
            ctx,
            config,
            compressor,
            hidden,
            state.compressor_mut(),
            head_dim,
            compress_ratio,
            compress_ratio < 16,
            pos,
            rope,
            kv,
            score,
            keepalive,
        )?;
    }
    if mode.has_indexer() {
        ensure!(
            mode == DeepSeekV4AttentionMode::CompressedSparse,
            "DSv4 MODEL1 decode does not support SparseIndexed/GLM indexer"
        );
        let indexer = attention.indexer();
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
            .ok_or_else(|| anyhow!("DSv4 MODEL1 decode indexer compressor kv scratch missing"))?;
        let score = scratch.compressor_index_score.as_mut().ok_or_else(|| {
            anyhow!("DSv4 MODEL1 decode indexer compressor score scratch missing")
        })?;
        {
            let indexer_state = state.indexer_mut();
            compressor_forward_decode(
                ctx,
                config,
                indexer_compressor,
                hidden,
                indexer_state,
                config.index_head_dim,
                compress_ratio,
                true,
                pos,
                rope,
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
        // #146 Index-layer guard: this MODEL1 decode lane is CompressedSparse-only
        // (asserted above), so the gate is unconditional.
        {
            let value_rows = state
                .compressor
                .as_ref()
                .map(|s| s.compressed.seq_len)
                .unwrap_or(0);
            ensure!(
                indexer_rows_after == value_rows,
                "DSv4 CSA select boundary (MODEL1 decode): indexer rows {indexer_rows_after} != \
                 value compressor rows {value_rows} (Shape drift — #146 guard)"
            );
        }
        let shared = dsa_shared
            .ok_or_else(|| anyhow!("DSv4 MODEL1 decode CSA shared DSA scratch missing"))?;
        // Read-only constants pulled BEFORE the mutable csa-scratch borrows below.
        let slot_idx = state
            .dsa_official_slot_idx()
            .ok_or_else(|| anyhow!("DSv4 MODEL1 decode CSA official DSA state missing"))?;
        let keys_capacity = state.indexer_compressed_capacity().unwrap_or(0);
        // Disjoint-field borrows: c_q_normed (read) + csa scratch (mut).
        let Dsv4MlaDecodeScratch {
            c_q_normed,
            csa_q_i,
            csa_weights,
            csa_selected,
            ..
        } = scratch;
        let csa_q_i = csa_q_i
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 MODEL1 decode CSA q_i scratch missing"))?;
        let csa_weights = csa_weights
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 MODEL1 decode CSA weights scratch missing"))?;
        let csa_selected = csa_selected
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 MODEL1 decode CSA selected scratch missing"))?;
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
            .ok_or_else(|| anyhow!("DSv4 MODEL1 decode CSA official DSA state missing"))?;
        ensure!(
            official.slot_idx == slot_idx,
            "DSv4 MODEL1 decode CSA slot index mismatch: official {} != staged {slot_idx}",
            official.slot_idx
        );
        csa_select_decode(
            ctx,
            config,
            indexer,
            hidden,
            c_q_normed,
            index_keys,
            keys_capacity,
            0,
            official,
            shared,
            pool,
            indexer_rows_before,
            indexer_rows_after,
            pos,
            compress_ratio,
            csa_q_i,
            csa_weights,
            csa_selected,
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
    let compressed = state.compressor.as_ref().map(|s| &s.compressed);
    let flash = state
        .flashmla
        .as_mut()
        .ok_or_else(|| anyhow!("FlashMLA decode enabled but layer state has no FlashMLA arena"))?;
    let flash_scratch = flashmla_scratch.ok_or_else(|| {
        anyhow!("FlashMLA decode enabled but shared FlashMLA decode scratch missing")
    })?;
    flashmla_decode_attention(
        attn,
        &scratch.q_prepared,
        &scratch.k_prepared,
        selected,
        compressed,
        &mut state.sw_window_cache,
        flash,
        flash_scratch,
        pool,
        local_heads,
        &mut scratch.local_attn,
        sm_scale,
        rope,
    )?;

    mla_oproj_decode(
        ctx,
        attention,
        state,
        &scratch.local_attn,
        &mut scratch.oproj_latent,
        out,
    )
}

/// GLM runtime Q absorption: q_raw [heads, qk_nope(192)+qk_rope(64)] -> absorbed q
/// [heads, kv_lora(512)+qk_rope(64) = head_dim(576)] per head, via
/// `q_latent[h] = w_kc[h] * q_nope[h]`; the q_rope(64) tail passes through
/// unchanged. Both buffers are token-major. `w_kc` is `[local_heads*kv_lora,
/// qk_nope]`, head `h`'s block at rows `[h*kv_lora, (h+1)*kv_lora)`.
///
/// DECODE (token_count==1): each head's q_nope rows are contiguous, so the per-head
/// GEMM and the q_rope copy are exact. PREFILL (token_count>1): token-major layout
/// strides a head's rows across tokens (`stride = qk_head_dim`), which needs a
/// batched-head kernel ARLE does not expose - bails loudly.
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
    {
        let (w_ptr, _gw) = w_kc.data.device_ptr(&ctx.stream);
        let (q_ptr, _gq) = q_raw.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = q_absorbed.data.device_ptr_mut(&ctx.stream);
        for h in 0..local_heads {
            // SAFETY: h < local_heads keeps this per-head offset in bounds.
            let q_nope_h = unsafe { (q_ptr as *const ffi::Half).add(h * qk_head) };
            // SAFETY: h < local_heads keeps this per-head offset in bounds.
            let w_h = unsafe { (w_ptr as *const ffi::Half).add(h * kv_lora * qk_nope) };
            // SAFETY: h < local_heads keeps this per-head offset in bounds.
            let out_h = unsafe { (out_ptr as *mut ffi::Half).add(h * head_dim) };
            // Per-head bf16 GEMM weight[kv_lora, qk_nope] · q_nope[qk_nope, 1].
            tensor_ops::gemm_bf16_raw(
                &ctx.stream,
                w_h as u64,
                q_nope_h as u64,
                out_h as u64,
                kv_lora as i32,
                token_count as i32,
                qk_nope as i32,
            )
            .map_err(|e| anyhow!("GLM glm_absorb_q head {h} gemm failed: {e}"))?;
        }
    }
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

/// GLM runtime V absorption: latent attn_out `[local_heads*kv_lora(512), tok]` -> v
/// `[local_heads*v_head_dim(256), tok]`, per head `v[h] = w_vc[h] * attn_out[h]`.
/// `w_vc` is `[local_heads*v_head, kv_lora]`, head `h`'s block at rows `[h*v_head,
/// (h+1)*v_head)`; the result feeds the plain `o_proj` (D4). DECODE
/// (token_count==1) is exact (contiguous per-head rows); PREFILL (token_count>1)
/// needs a batched-head kernel - bails.
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
    {
        let (w_ptr, _gw) = w_vc.data.device_ptr(&ctx.stream);
        let (a_ptr, _ga) = local_attn.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = v_out.data.device_ptr_mut(&ctx.stream);
        for h in 0..local_heads {
            // SAFETY: h < local_heads keeps this per-head offset in bounds.
            let a_h = unsafe { (a_ptr as *const ffi::Half).add(h * kv_lora) };
            // SAFETY: h < local_heads keeps this per-head offset in bounds.
            let w_h = unsafe { (w_ptr as *const ffi::Half).add(h * v_head * kv_lora) };
            // SAFETY: h < local_heads keeps this per-head offset in bounds.
            let out_h = unsafe { (out_ptr as *mut ffi::Half).add(h * v_head) };
            // Per-head bf16 GEMM weight[v_head, kv_lora] · attn_out[kv_lora, 1].
            tensor_ops::gemm_bf16_raw(
                &ctx.stream,
                w_h as u64,
                a_h as u64,
                out_h as u64,
                v_head as i32,
                token_count as i32,
                kv_lora as i32,
            )
            .map_err(|e| anyhow!("GLM glm_absorb_v head {h} gemm failed: {e}"))?;
        }
    }
    keepalive.keep_hidden(&v_out);
    Ok(v_out)
}

/// PREPARE half of `mla_attention` (see [`Dsv4MlaPrepared`]): Q/KV LoRA + partial
/// RoPE, and for CSA/HCA the compressor + (CSA) indexer top-k `selected`, leaving
/// `state.compressor` / `state.indexer` populated for the fwd's re-borrow. No
/// FlashMLA kernel and no pool writes, so the batched lane can run it per row.
pub(crate) fn mla_attention_prepare(
    attn: &Dsv4AttnCtx<'_>,
    hidden: &HiddenStates,
    state: &mut Dsv4LayerAttentionState,
    pool: &mut Dsv4LayerKvLayout,
    dsa_shared: Option<&mut Dsv4DsaSharedScratch>,
    // Shared FP8 prefill DeepGEMM linear scratch; `None` on the decode lanes, which
    // never take a prefill branch.
    mut prefill_shared: Option<&mut Dsv4PrefillDeepGemmLinearScratch>,
    // Shared FP32 probe scratch — contract on `compressor_forward`'s param.
    mut fp32_scratch: Option<&mut Dsv4CompressorFp32Scratch>,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<Dsv4MlaPrepared> {
    let Dsv4AttnCtx {
        ctx,
        config,
        attention,
        mode,
        compress_ratio,
        layer_idx,
        tp,
        pos,
        chain_verify,
    } = *attn;
    ensure!(
        hidden.hidden_dim == config.hidden_size,
        "DSv4 MLA hidden dim {} != hidden_size {}",
        hidden.hidden_dim,
        config.hidden_size
    );

    let head_dim = config.head_dim;
    let token_count = hidden.seq_len;
    let local_width = attention.wq_b.rows;
    // DSv4's wq_b emits the PRE-ABSORBED q at head_dim (512) per head; GLM's emits
    // qk_nope+qk_rope (256), absorbed at runtime. So derive local_heads from the wq_b
    // per-head width.
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
    // GLM pure-SparseIndexed has sliding_window==0; DSv4 modes require a non-zero
    // window.
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
        attention.attn_sink().len >= sink_offset + local_heads,
        "DSv4 MLA attn_sink len {} cannot cover rank {tp_rank} heads [{sink_offset}, {})",
        attention.attn_sink().len,
        sink_offset + local_heads
    );

    // RoPE base/YaRN is PER-LAYER, matching SGLang (deepseek_v4.py:271): compressed
    // layers (CSA cr=4 / HCA cr=128) rope Q, SW-K, the output inverse-rope AND the
    // compressor with compress_rope_theta + YaRN(original_max_position_embeddings);
    // pure-SW layers (cr=0) use rope_theta with no YaRN. Q MUST share the
    // compressed-key theta or Q*compressed-K phase mismatches and long context
    // (>~80 tok) collapses to garbage.
    let rope = RopeParams::from_config(config, compress_ratio)?;
    let start_pos_i32 = i32::try_from(pos.start)
        .map_err(|_| anyhow::anyhow!("DSv4 MLA start_pos {} overflows i32", pos.start))?;

    // Q/KV LoRA: the fused wq_a|wkv weight cache when native DeepGEMM is available,
    // else the scalar reference order.
    let fused_wqkv = token_count == 1 && dsv4_fused_wqkv_decode_enabled()?;
    let (c_q_normed, q_raw, kv_normed) = if fused_wqkv {
        let scratch = state.fused_wqkv.as_mut().ok_or_else(|| {
            anyhow!("DSv4 fused wqkv decode requested but decode scratch was not allocated")
        })?;
        crate::profile::profile_op(ctx, "linear/wqkv_a_fused", None, token_count, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wqkv_a_fused", || {
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
            })
        })?
    } else if token_count > 1 && dsv4_deepgemm_enabled() {
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
            // Prefill wq_b -> DeepGEMM, off the scalar dsv4_fp8_gemv_batch (62% of
            // mla_attn
            // prefill). Reuses the prefill fused-wqkv FP8 scratch since K=q_lora_rank
            // <= hidden_dim.
            if let Some(cache) = attention
                .wq_b_deepgemm
                .as_ref()
                .filter(|_| dsv4_deepgemm_enabled())
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

        keepalive.keep_hidden(&kv_raw);
        let kv_normed = mla_rms_norm(ctx, &kv_raw, &attention.kv_norm, config.rms_norm_eps)?;
        keepalive.keep_hidden(&kv_normed);
        (c_q_normed, q_raw, kv_normed)
    } else {
        let (wq_a, wq_b, wkv) = (&attention.wq_a, &attention.wq_b, &attention.wkv);
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

    // GLM runtime Q absorption (w_kc.is_some()): wq_b produces qk_head_dim(256) per
    // head, but the FlashMLA latent path needs the absorbed q[heads, 576]. Per SGLang
    // forward_mla.py: q_latent[h] = q_nope[h] * w_kc[h], reassembled as
    // [q_latent(512) | q_rope(64)]. DSv4 (w_kc None) skips - q_raw is pre-absorbed.
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
        {
            if let Some(meta) = chain_verify {
                let (start_ptr, _sg) = meta.positions.device_ptr(&ctx.stream);
                flash_kv::dsv4_prepare_qk_fused_batch_start_pos_raw(
                    &ctx.stream,
                    q_raw_ptr,
                    k_raw_ptr,
                    q_out_ptr,
                    k_out_ptr,
                    token_count as i32,
                    local_heads as i32,
                    head_dim as i32,
                    config.qk_rope_head_dim as i32,
                    start_ptr,
                    config.rms_norm_eps,
                    rope.base,
                    rope.original_seq_len,
                    rope.factor,
                    rope.beta_fast,
                    rope.beta_slow,
                )?;
            } else if let Some(start_pos_device) = pos.device {
                let (start_ptr, _sg) = start_pos_device.device_ptr(&ctx.stream);
                flash_kv::dsv4_prepare_qk_start_pos_ptr_raw(
                    &ctx.stream,
                    q_raw_ptr,
                    k_raw_ptr,
                    q_out_ptr,
                    k_out_ptr,
                    token_count as i32,
                    local_heads as i32,
                    head_dim as i32,
                    config.qk_rope_head_dim as i32,
                    start_ptr,
                    config.rms_norm_eps,
                    rope.base,
                    rope.original_seq_len,
                    rope.factor,
                    rope.beta_fast,
                    rope.beta_slow,
                )?;
            } else {
                flash_kv::dsv4_prepare_qk_raw(
                    &ctx.stream,
                    q_raw_ptr,
                    k_raw_ptr,
                    q_out_ptr,
                    k_out_ptr,
                    token_count as i32,
                    local_heads as i32,
                    head_dim as i32,
                    config.qk_rope_head_dim as i32,
                    start_pos_i32,
                    config.rms_norm_eps,
                    rope.base,
                    rope.original_seq_len,
                    rope.factor,
                    rope.beta_fast,
                    rope.beta_slow,
                )?;
            }
        }
    }
    keepalive.keep_hidden(&q_prepared);
    keepalive.keep_hidden(&k_prepared);
    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
    // SAFETY: the FlashMLA attention kernel writes the full local_attn buffer
    // in `mla_attention_fwd`.
    let local_attn = unsafe { HiddenStates::uninit(ctx, local_width, token_count)? };

    // CSA / HCA run the compressor (+ for CSA the indexer + top-k `selected`) here in
    // PREPARE; SW has neither. Frozen chain verify must NOT append compressed/indexer
    // KV: its rows are speculative and FlashMLA already packs them as the current
    // chunk, so keep the CSA query/top-k below but read the committed pools as-is.
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
        // MAIN compressor (CSA/HCA only) -> (indexer modes) top-k select. GLM
        // SparseIndexed has no MAIN compressor, so gate on has_compressor().
        let overlap = compress_ratio < 16;
        if mode.has_compressor() {
            let compressor = attention.compressor();
            if skip_frozen_compressor {
                ensure!(
                    state.compressor.is_some(),
                    "DSv4 layer {layer_idx} is {mode:?} but has no compressor state"
                );
            } else {
                let compressor_state = state.compressor_mut();
                compressor_forward(
                    ctx,
                    config,
                    compressor,
                    hidden,
                    compressor_state,
                    head_dim,
                    compress_ratio,
                    overlap,
                    pos,
                    rope,
                    fp32_scratch.as_deref_mut(),
                    None,
                    None,
                    keepalive,
                )?;
            }
        }

        if mode.has_indexer() {
            let indexer = attention.indexer();
            let indexer_rows_before = state
                .indexer
                .as_ref()
                .map(|s| s.compressed.seq_len)
                .unwrap_or(0);
            // CSA runs a second compressor over index_head_dim keys.
            // GLM
            // SparseIndexed has no key compressor - one full index key per token
            // (ratio=1).
            if skip_frozen_compressor {
                ensure!(
                    state.indexer.is_some(),
                    "DSv4 layer {layer_idx} is {mode:?} but has no indexer state"
                );
            } else {
                let indexer_state = state.indexer_mut();
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
                        pos,
                        rope,
                        fp32_scratch,
                        None,
                        None,
                        keepalive,
                    )?;
                } else {
                    // GLM SparseIndexed: full-sequence index-key build; every token ->
                    // one key row at
                    // its absolute position.
                    sparse_indexed_index_key_forward(
                        ctx,
                        config,
                        indexer,
                        hidden,
                        indexer_state,
                        pos.start,
                        keepalive,
                    )?;
                }
            }
            let indexer_rows_after = state
                .indexer
                .as_ref()
                .map(|s| s.compressed.seq_len)
                .unwrap_or(0);
            // #146 Index-layer guard: for CSA the value compressor and the indexer key
            // compressor consume the same token stream at the same start_pos and ratio,
            // so
            // their row counts MUST match. GLM SparseIndexed (indexer ratio=1 vs value
            // ratio>1) and frozen chain-verify are excluded so this never false-fires.
            // Turns
            // a silent Shape drift past 2048 into a loud boundary fail.
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
                indexer,
                hidden,
                &c_q_normed,
                index_keys,
                keys_capacity,
                if mode == DeepSeekV4AttentionMode::SparseIndexed {
                    pos.start
                } else {
                    0
                },
                official,
                dsa_shared,
                pool,
                indexer_rows_before,
                indexer_rows_after,
                pos,
                index_ratio,
                prefill_shared,
                None,
                None,
                keepalive,
            )?
        } else {
            None
        }
    };

    Ok(Dsv4MlaPrepared {
        q_prepared,
        k_prepared,
        local_attn,
        selected,
        local_heads,
        sm_scale,
        rope,
    })
}

/// Batched (`m = N`) slot-INDEPENDENT projection pre-pass for the batched decode
/// lane. Each row's `c_q_normed` / `q_prepared` / `k_prepared` depends only on its
/// own `normed` activation, the shared `&Dsv4Attention` weights, and its absolute
/// position, so the Q/KV LoRA projections + partial RoPE batch into one `m=N` call
/// per weight - amortizing the weight read xN instead of re-reading it per row
/// (the 137 ms @ n=22 PREPARE hot spot).
///
/// Routed through the scalar batched [`dsv4_linear`] rather than the PER-SLOT
/// fused-DeepGEMM decode scratch, which is sized for a single `m=1` row. The
/// batched-GEMV `(out_row, token)` grid is row-independent, so each row's result
/// matches the scalar per-row path.
///
/// `normed` is `[N, hidden]`; `positions` is `[N]` i32 device (each row's absolute
/// decode position). Returns `(c_q_normed[N], q_prepared[N], k_prepared[N])` plus
/// the per-row scalars the compressed-only finish needs. Touches NO slot state -
/// the per-slot compressor / indexer / `csa_select` stay per-row in
/// [`mla_attention_prepare_compressed_only`].
pub(crate) struct Dsv4MlaProjBatch {
    pub(crate) c_q_normed: HiddenStates,
    pub(crate) q_prepared: HiddenStates,
    pub(crate) k_prepared: HiddenStates,
    pub(crate) local_heads: usize,
    pub(crate) local_width: usize,
    pub(crate) sm_scale: f32,
    pub(crate) rope: RopeParams,
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
    // GLM's wq_b emits qk_head-wide q (runtime-absorbed), DSv4's pre-absorbed
    // head_dim-wide q - see `mla_attention_prepare`.
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

    // RoPE base/YaRN is PER-LAYER, identical policy to `mla_attention_prepare`.
    let rope = RopeParams::from_config(config, compress_ratio)?;

    // Q/KV LoRA at m=N. The weight read MUST amortize across the N decode rows: a real
    // batched FP8 GEMM (DeepGEMM, K-tiled, tensor-core, weight read ONCE for all N),
    // not the per-(out_row, token) scalar GEMV which re-reads the weight per token.
    // Mirrors `mla_attention_prepare`'s prefill DeepGEMM branch at m=N; the shared
    // `prefill_linear` scratch (max_m = DSV4_PREFILL_QUERY_CHUNK >= N) stages the FP8
    // activation. The GEMV `else` branch is the DeepGEMM-disabled fallback.
    let use_deepgemm = dsv4_deepgemm_enabled()
        && attention.wqkv_a_deepgemm.is_some()
        && attention.wq_b_deepgemm.is_some()
        && prefill_shared.is_some();
    let (c_q_normed, q_raw, kv_normed) = if use_deepgemm {
        let scratch = prefill_shared.ok_or_else(|| {
            anyhow!("DSv4 MLA proj-batch DeepGEMM path requires the prefill_linear scratch")
        })?;
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
        // scratch
        // absent): the scalar batched `dsv4_linear` path.
        let (wq_a, wq_b, wkv) = (&attention.wq_a, &attention.wq_b, &attention.wkv);
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

    // Partial RoPE over all N rows with PER-ROW positions. The position is sourced as
    // `positions[token]` instead of a single scalar + token offset, which is what
    // makes the batched RoPE equal N per-row calls - each row is a distinct sequence
    // at its own absolute position.
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
        {
            flash_kv::dsv4_prepare_qk_fused_batch_start_pos_raw(
                &ctx.stream,
                q_raw_ptr,
                k_raw_ptr,
                q_out_ptr,
                k_out_ptr,
                n as i32,
                local_heads as i32,
                head_dim as i32,
                config.qk_rope_head_dim as i32,
                pos_ptr,
                config.rms_norm_eps,
                rope.base,
                rope.original_seq_len,
                rope.factor,
                rope.beta_fast,
                rope.beta_slow,
            )?;
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
        rope,
    })
}

/// Run ONE row's main + indexer compressor STATE update in DEFER mode: skip the
/// per-row FFI, push the row's ring-state device pointers into the batch sinks, and
/// advance `compressed.seq_len` (host bookkeeping). Returns this row's
/// `indexer_rows_before` (the count BEFORE this step's advance; `0` for non-CSA
/// rows). The GPU state writes run later in ONE [`dsv4_compressor_update_batched`]
/// over all N rows, before any reader. SW rows are a no-op.
pub(crate) fn mla_attention_compressor_defer_row(
    attn: &Dsv4AttnCtx<'_>,
    normed_row: &HiddenStates,
    state: &mut Dsv4LayerAttentionState,
    rope: RopeParams,
    main_sink: &mut Dsv4CompressorBatchPtrs,
    indexer_sink: &mut Dsv4CompressorBatchPtrs,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<usize> {
    let Dsv4AttnCtx {
        ctx,
        config,
        attention,
        mode,
        compress_ratio,
        pos,
        ..
    } = *attn;
    // GLM SparseIndexed has no compressor and the batched-defer kernel is
    // compressor-specific; the MODEL1-only batch scratch keeps GLM off this path -
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
    let compressor = attention.compressor();
    let overlap = compress_ratio < 16;
    {
        let compressor_state = state.compressor_mut();
        compressor_forward(
            ctx,
            config,
            compressor,
            normed_row,
            compressor_state,
            head_dim,
            compress_ratio,
            overlap,
            pos,
            rope,
            None,
            // Defer mode ignores `precomputed` (the batched update reads the prepass
            // output
            // directly); pass None.
            None,
            Some(main_sink),
            keepalive,
        )?;
    }
    let mut indexer_rows_before = 0usize;
    if mode == DeepSeekV4AttentionMode::CompressedSparse {
        let indexer = attention.indexer();
        indexer_rows_before = state
            .indexer
            .as_ref()
            .map(|s| s.compressed.seq_len)
            .unwrap_or(0);
        let indexer_state = state.indexer_mut();
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
            pos,
            rope,
            None,
            None,
            Some(indexer_sink),
            keepalive,
        )?;
    }
    Ok(indexer_rows_before)
}

/// Per-row SLOT-DEPENDENT finish of the batched-decode PREPARE, paired with
/// [`mla_attention_prepare_proj_batch`]. The projection + RoPE for row `r` were
/// already computed batched; this runs only the per-slot half - the CSA/HCA
/// compressor and the CSA indexer + top-k `selected` - then assembles the
/// `Dsv4MlaPrepared` the batched lane consumes, taking ownership of the
/// `q_prepared_row` / `k_prepared_row` slices the caller copied out of the batch.
/// `c_q_normed_row` MUST hold byte-identical data to what `mla_attention_prepare`
/// would have produced for this row, so `csa_select` selects the same blocks.
pub(crate) fn mla_attention_prepare_compressed_only(
    attn: &Dsv4AttnCtx<'_>,
    normed_row: &HiddenStates,
    c_q_normed_row: &HiddenStates,
    q_prepared_row: HiddenStates,
    k_prepared_row: HiddenStates,
    proj: &Dsv4MlaProjBatch,
    state: &mut Dsv4LayerAttentionState,
    pool: &mut Dsv4LayerKvLayout,
    dsa_shared: Option<&mut Dsv4DsaSharedScratch>,
    // When `Some`, the CSA per-row READ/SELECT is skipped (cache writes still run),
    // this row's q_i/weights are gathered into the N-row staging, and the returned
    // `selected` is `None` (the batched select fills selected_batched).
    batched_gather: Option<Dsv4DsaBatchedGather<'_>>,
    // When `Some`, this row's compressor/indexer `(kv_raw, score_raw)` were already
    // projected batched and are passed to `compressor_forward`, skipping the per-row
    // m=1 GEMVs.
    compressor_precomputed: Option<Dsv4CompressorPrecomputed<'_>>,
    // When `Some`, this row's indexer query (`q_i`) and gating `weights` were already
    // projected batched; the `[width,1]` slices thread into `csa_select`, skipping the
    // per-row m=1 GEMVs. Compressor-layer (CSA) only.
    indexer_query_precomputed: Option<Dsv4IndexerQueryPrecomputed<'_>>,
    // When `true`, the per-row compressor / indexer STATE updates already ran in ONE
    // batched `dsv4_compressor_update_batched`, so the `compressor_forward` calls here
    // are skipped and only `csa_select` runs. That pre-pass also advanced
    // `compressed.seq_len`, so `indexer_rows_before` must be supplied via
    // `indexer_rows_before_override`.
    skip_compressor: bool,
    indexer_rows_before_override: Option<usize>,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<Dsv4MlaPrepared> {
    let Dsv4AttnCtx {
        ctx,
        config,
        attention,
        mode,
        compress_ratio,
        pos,
        ..
    } = *attn;
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

    // SAFETY: the FlashMLA attention kernel writes the full local_attn buffer
    // in the batched fwd / single-row `mla_attention_fwd`.
    let local_attn = unsafe { HiddenStates::uninit(ctx, local_width, 1)? };

    // compressor -> (CSA) indexer top-k select, PER SLOT, reading this row's
    // `normed_row` and `c_q_normed_row`. GLM SparseIndexed: full-sequence indexer,
    // every token a key (ratio=1, no compressor).
    let index_ratio = if mode == DeepSeekV4AttentionMode::SparseIndexed {
        1
    } else {
        compress_ratio
    };
    let selected = if !mode.has_indexer() && !mode.has_compressor() {
        None
    } else {
        let overlap = compress_ratio < 16;
        // Split the batched pre-pass slices into main + indexer (the inner reference
        // pairs
        // are `Copy`, so reading both fields does not conflict with the per-call
        // `state`
        // borrows below). For GLM SparseIndexed these gates are always None/false.
        let precomputed_main = compressor_precomputed.as_ref().map(|p| p.main);
        let precomputed_indexer = compressor_precomputed.as_ref().and_then(|p| p.indexer);
        // MAIN compressor (CSA/HCA only): GLM SparseIndexed has none. Full-flatten
        // already
        // ran the STATE update batched in the pre-pass, so skip it here.
        if mode.has_compressor() {
            let compressor = attention.compressor();
            if !skip_compressor {
                let compressor_state = state.compressor_mut();
                compressor_forward(
                    ctx,
                    config,
                    compressor,
                    normed_row,
                    compressor_state,
                    head_dim,
                    compress_ratio,
                    overlap,
                    pos,
                    proj.rope,
                    None,
                    precomputed_main,
                    None,
                    keepalive,
                )?;
            }
        }

        if mode.has_indexer() {
            let indexer = attention.indexer();
            // `indexer_rows_before` = the indexer compressed row count BEFORE this
            // step's
            // advance. In full-flatten the pre-pass already advanced
            // `compressed.seq_len`, so
            // take the captured override; otherwise read it live.
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
            // full-length
            // index key per token (ratio=1, no compressor).
            if !skip_compressor {
                let indexer_state = state.indexer_mut();
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
                        pos,
                        proj.rope,
                        None,
                        precomputed_indexer,
                        None,
                        keepalive,
                    )?;
                } else {
                    // GLM SparseIndexed: per-row index-key build (no compressor, no
                    // precomputed batch
                    // path - those are compressor-only).
                    sparse_indexed_index_key_forward(
                        ctx,
                        config,
                        indexer,
                        normed_row,
                        indexer_state,
                        pos.start,
                        keepalive,
                    )?;
                }
            }
            let indexer_rows_after = state
                .indexer
                .as_ref()
                .map(|s| s.compressed.seq_len)
                .unwrap_or(0);
            // #146 Index-layer guard: the batched-decode twin of
            // `mla_attention_prepare`'s
            // check, same state and same CSA row-count invariant. No frozen-compressor
            // lane
            // exists here, so the gate is mode-only.
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
            // With `batched_gather` Some, `csa_select` does cache writes + gather and
            // returns
            // `None`; the batched select fills `selected_batched`.
            csa_select(
                ctx,
                config,
                indexer,
                normed_row,
                c_q_normed_row,
                index_keys,
                keys_capacity,
                if mode == DeepSeekV4AttentionMode::SparseIndexed {
                    pos.start
                } else {
                    0
                },
                official,
                dsa_shared,
                pool,
                indexer_rows_before,
                indexer_rows_after,
                pos,
                index_ratio,
                // Decode (token_count=1) never takes the prefill indexer DeepGEMM lane,
                // so the
                // shared prefill scratch is not needed.
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
        sm_scale: proj.sm_scale,
        rope: proj.rope,
    })
}

/// FWD half of `mla_attention` (see [`Dsv4MlaPrepared`]): the FlashMLA attention
/// kernel over the PREPARE output, then the O-LoRA. The compressed-key pool is
/// re-borrowed from `state` here.
fn mla_attention_fwd(
    attn: &Dsv4AttnCtx<'_>,
    state: &mut Dsv4LayerAttentionState,
    pool: &mut Dsv4LayerKvLayout,
    // Shared single-row FlashMLA decode scratch; consumed only on the single-row
    // decode FlashMLA path below.
    flashmla_scratch: Option<&mut Dsv4FlashMlaDecodeScratch>,
    // Shared FP8 prefill DeepGEMM linear scratch, forwarded to `mla_oproj` (its
    // token_count>1 prefill lane gates on it).
    prefill_shared: Option<&mut Dsv4PrefillDeepGemmLinearScratch>,
    prepared: Dsv4MlaPrepared,
    out: &mut HiddenStates,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<()> {
    let Dsv4AttnCtx {
        ctx,
        config,
        attention,
        ..
    } = *attn;
    let Dsv4MlaPrepared {
        q_prepared,
        k_prepared,
        mut local_attn,
        selected,
        local_heads,
        sm_scale,
        rope,
    } = prepared;
    let token_count = q_prepared.seq_len;
    ensure!(
        attention.wo_b().rows == out.hidden_dim && out.seq_len == token_count,
        "DSv4 MLA output shape mismatch: wo_b rows {} out {}x{} expected {}x{}",
        attention.wo_b().rows,
        out.hidden_dim,
        out.seq_len,
        attention.wo_b().rows,
        token_count
    );
    keepalive.keep_hidden(&q_prepared);
    keepalive.keep_hidden(&k_prepared);

    // Compressed KV pool: CSA/HCA have a compressor; SparseIndexed and SWA do not.
    let compressed = state.compressor.as_ref().map(|s| &s.compressed);

    if token_count > 1 {
        flashmla_prefill_attention(
            attn,
            &q_prepared,
            &k_prepared,
            selected.as_ref(),
            compressed,
            &mut state.sw_window_cache,
            local_heads,
            &mut local_attn,
            sm_scale,
            rope,
        )?;
    } else {
        let flash = state.flashmla.as_mut().ok_or_else(|| {
            anyhow!("FlashMLA decode enabled but layer state has no FlashMLA arena")
        })?;
        let scratch = flashmla_scratch.ok_or_else(|| {
            anyhow!("FlashMLA decode enabled but shared FlashMLA decode scratch missing")
        })?;
        flashmla_decode_attention(
            attn,
            &q_prepared,
            &k_prepared,
            selected.as_ref(),
            compressed,
            &mut state.sw_window_cache,
            flash,
            scratch,
            pool,
            local_heads,
            &mut local_attn,
            sm_scale,
            rope,
        )?;
    }
    if let Some(sel) = selected.as_ref() {
        keepalive.keep_i32(sel);
    }
    keepalive.keep_hidden(&local_attn);

    // GLM runtime V absorption (w_vc.is_some()): project the [heads, kv_lora(512)]
    // FlashMLA latent back to v[heads, v_head(256)] before the plain o_proj, which
    // changes local_attn's hidden_dim. DSv4 (w_vc None) skips.
    let local_attn = if let Some(w_vc) = attention.w_vc.as_ref() {
        let v = glm_absorb_v(ctx, config, w_vc, &local_attn, local_heads, keepalive)?;
        keepalive.keep_hidden(&v);
        v
    } else {
        local_attn
    };

    // O-LoRA: wo_a (per o-group, down to the output latent) -> wo_b (up to hidden).
    // Row-parallel: the all-reduce-sum is the model's concern. GLM takes the plain-o
    // early return in mla_oproj.
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
    // unquantized) has no per-group FP8 scales, so the route-GEMV below would assert
    // on empty scales. On TP<o_groups the rank owns >1 output group and takes this
    // path, so route through a per-group dense BF16 GEMM: gather group `g`'s column
    // slice out of the token-major activation, `gemm_cuda` with group `g`'s weight
    // block, then scatter the result back into the latent.
    if attention.wo_a().weight_format == WeightFormat::DenseBf16 {
        let seq = local_attn.seq_len;
        let cols = shape.cols_per_group;
        let rows = shape.rows_per_group;
        let wo_a = attention.wo_a();
        // SAFETY: uninit device scratch; fully written before first read.
        let mut in_g = unsafe { HiddenStates::uninit(ctx, cols, seq)? };
        // SAFETY: uninit device scratch; fully written before first read.
        let mut out_g = unsafe { HiddenStates::uninit(ctx, rows, seq)? };
        // Cache raw pointers once: per-iteration mutable + immutable `device_ptr` calls
        // would falsely collide on SyncOnDrop (the buffers are not reallocated here).
        let (wo_a_base, _wg) = wo_a.data.device_ptr(&ctx.stream);
        let (src_ptr, _sg) = local_attn.data.device_ptr(&ctx.stream);
        let (in_ptr, _ig) = in_g.data.device_ptr_mut(&ctx.stream);
        let (out_ptr, _og) = out_g.data.device_ptr_mut(&ctx.stream);
        let (dst_ptr, _dg) = latent.data.device_ptr_mut(&ctx.stream);
        for group in 0..shape.groups {
            {
                flash_kv::dsv4_oproj_group_gather_raw(
                    &ctx.stream,
                    src_ptr,
                    in_ptr,
                    i32::try_from(seq)?,
                    i32::try_from(shape.groups)?,
                    i32::try_from(cols)?,
                    i32::try_from(group)?,
                )
                .map_err(|e| anyhow!("DSv4 dense grouped O-LoRA gather failed: {e}"))?;
            }
            // SAFETY: group `g`'s weight block is contiguous rows
            // `[g*rows, (g+1)*rows)` of the `[groups*rows, cols]` dense `wo_a`,
            // i.e. offset `g*rows*cols` bf16 elements from the base pointer.
            let w_g = unsafe { (wo_a_base as *const ffi::Half).add(group * rows * cols) };
            tensor_ops::gemm_bf16_raw(
                &ctx.stream,
                w_g as u64,
                in_ptr,
                out_ptr,
                i32::try_from(rows)?,
                i32::try_from(seq)?,
                i32::try_from(cols)?,
            )
            .map_err(|e| anyhow!("DSv4 dense grouped O-LoRA gemm failed: {e}"))?;
            {
                flash_kv::dsv4_oproj_group_scatter_raw(
                    &ctx.stream,
                    out_ptr,
                    dst_ptr,
                    i32::try_from(seq)?,
                    i32::try_from(shape.groups)?,
                    i32::try_from(rows)?,
                    i32::try_from(group)?,
                )
                .map_err(|e| anyhow!("DSv4 dense grouped O-LoRA scatter failed: {e}"))?;
            }
        }
        return Ok(());
    }
    let wo_a_groups = attention.wo_a_groups.as_ref().expect("DSv4 wo_a_groups");
    ensure!(
        wo_a_groups.scale_rows_per_group > 0 && wo_a_groups.scale_cols > 0,
        "DSv4 O-LoRA grouped scale shape must be non-empty"
    );
    // Pointer tables were built from this rank's contiguous `wo_a` groups at
    // load time. `route_meta: None` selects group `route % groups`; route order
    // is `[token0/group0, token0/group1, ..., token1/group0, ...]`, which is
    // exactly the `HiddenStates` token-major layout when each group is
    // `cols_per_group` wide.
    let args = cuda_ql::Dsv4RouteGemvArgs {
        route_meta: None,
        local_expert_start: 0,
        experts_per_rank: shape.groups,
        num_routes: shape.routes,
        n: shape.rows_per_group,
        k: shape.cols_per_group,
        scale_rows: wo_a_groups.scale_rows_per_group,
        scale_cols: wo_a_groups.scale_cols,
        apply_route_weight: false,
    };
    match attention.wo_a().weight_format {
        WeightFormat::Dsv4Fp8BlockScaled => cuda_ql::dsv4_fp8_route_gemv_batch(
            ctx,
            &wo_a_groups.weight_ptrs,
            &wo_a_groups.scale_ptrs,
            &local_attn.data,
            &mut latent.data,
            args,
        )?,
        WeightFormat::Dsv4Fp4BlockScaled => cuda_ql::dsv4_fp4_route_gemv_batch(
            ctx,
            &wo_a_groups.weight_ptrs,
            &wo_a_groups.scale_ptrs,
            &local_attn.data,
            &mut latent.data,
            args,
        )?,
        other => bail!("DSv4 O-LoRA grouped wo_a expected FP8/FP4 block-scaled, got {other:?}"),
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
    {
        flash_kv::dsv4_oproj_group_gather_raw(
            &ctx.stream,
            src_ptr,
            dst_ptr,
            i32::try_from(src.seq_len)?,
            i32::try_from(shape.groups)?,
            i32::try_from(shape.cols_per_group)?,
            i32::try_from(group)?,
        )
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
    {
        flash_kv::dsv4_oproj_group_scatter_raw(
            &ctx.stream,
            src_ptr,
            dst_ptr,
            i32::try_from(dst.seq_len)?,
            i32::try_from(shape.groups)?,
            i32::try_from(shape.rows_per_group)?,
            i32::try_from(group)?,
        )
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
    // M-parametric over n: each group is ONE DeepGEMM at M=n. Group `g`'s columns are
    // strided in the token-major `[n, groups*cols_per_group]` activation, so gather
    // them into contiguous `[cols_per_group, n]`, GEMM(m=n) into `[rows_per_group,
    // n]`, then scatter back into the latent.
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
        // Scope the gather guards so `in_g.data`'s mutable guard drops before the GEMM
        // re-borrows it immutably.
        {
            let (src_ptr, _src_guard) = local_attn.data.device_ptr(&ctx.stream);
            let (in_ptr, _in_guard) = in_g.data.device_ptr_mut(&ctx.stream);
            {
                flash_kv::dsv4_oproj_group_gather_raw(
                    &ctx.stream,
                    src_ptr,
                    in_ptr,
                    i32::try_from(n)?,
                    i32::try_from(shape.groups)?,
                    i32::try_from(cols)?,
                    i32::try_from(group)?,
                )
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
            {
                flash_kv::dsv4_oproj_group_scatter_raw(
                    &ctx.stream,
                    out_ptr,
                    dst_ptr,
                    i32::try_from(n)?,
                    i32::try_from(shape.groups)?,
                    i32::try_from(rows)?,
                    i32::try_from(group)?,
                )
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

/// O-LoRA output projection, extracted from `mla_attention` so the batched-decode
/// path can call it ONCE over [N] rows: `wo_a` (down to the output latent) ->
/// `wo_b` (up to hidden) into `out`. Row-parallel - the all-reduce-sum is the
/// caller's concern. Batched decode passes token_count=N to hit the prefill
/// DeepGEMM branch (M=N amortizes the wo weight read).
#[allow(clippy::too_many_arguments)]
pub(crate) fn mla_oproj(
    ctx: &DeviceContext,
    attention: &Dsv4Attention,
    state: &mut Dsv4LayerAttentionState,
    // Shared FP8 prefill DeepGEMM linear scratch; the prefill wo_a/wo_b lane
    // (token_count>1) gates on it, and it is `None` on the decode lanes.
    mut prefill_shared: Option<&mut Dsv4PrefillDeepGemmLinearScratch>,
    local_attn: &HiddenStates,
    token_count: usize,
    keepalive: &mut Dsv4ForwardKeepalive,
    out: &mut HiddenStates,
) -> Result<()> {
    // GLM plain output projection: a single GEMM v[heads*v_head_dim] -> hidden, no
    // wo_a/wo_b low-rank and no group tables. `local_attn` for GLM is the post-w_vc v;
    // DSv4 (o_proj None) falls through to the wo_a/wo_b path.
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
        attention.wo_a().rows,
        attention.wo_a().cols,
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
    let mut latent = unsafe { HiddenStates::uninit(ctx, attention.wo_a().rows, token_count)? };
    // Captured before any branch consumes `prefill_shared` (the wo_b prefill lane
    // moves it); drives the batched-O-LoRA active_counts restore at the end.
    let is_decode = prefill_shared.is_none();
    let wo_a_decode_dg = shape.groups == 1
        && dsv4_deepgemm_enabled()
        && state.fused_wqkv.is_some()
        && attention.wo_a_deepgemm.is_some()
        && prefill_shared.is_none();
    let wo_a_prefill_dg = shape.groups == 1
        && token_count > 1
        && dsv4_deepgemm_enabled()
        && attention.wo_a_deepgemm.is_some()
        && prefill_shared.is_some();
    // Grouped decode DeepGEMM is M-parametric (gather/GEMM/scatter per group over n
    // rows), so the batched FINISH routes groups>1 here at M=n. Grouped prefill stays
    // on its own gate below.
    let wo_a_group_decode_dg = shape.groups > 1
        && dsv4_deepgemm_enabled()
        && state.fused_wqkv.is_some()
        && attention.wo_a_group_deepgemm.is_some()
        && prefill_shared.is_none();
    let wo_a_group_prefill_dg = shape.groups > 1
        && token_count > 1
        && dsv4_deepgemm_enabled()
        && attention.wo_a_group_deepgemm.is_some()
        && prefill_shared.is_some();
    if wo_a_decode_dg {
        // wo_a through tensor-core DeepGEMM (M=token_count), reusing the fused-wqkv FP8
        // scratch for the single-output-group case.
        let scratch = state.fused_wqkv.as_mut().expect("wo_dg gate checked");
        let wo_a_cache = attention
            .wo_a_deepgemm
            .as_ref()
            .expect("wo_dg gate checked");
        let wo_a_cols = attention.wo_a().cols;
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
                dsv4_linear(ctx, attention.wo_a(), local_attn, &mut latent)
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

    // wo_b is always a single-group [hidden, o_lora_rank] GEMM, so its DeepGEMM lane
    // is M-parametric like wo_a: M=token_count.
    let wo_b_decode_dg = dsv4_deepgemm_enabled()
        && state.fused_wqkv.is_some()
        && attention.wo_b_deepgemm.is_some()
        && prefill_shared.is_none();
    let wo_b_prefill_dg = token_count > 1
        && dsv4_deepgemm_enabled()
        && attention.wo_b_deepgemm.is_some()
        && prefill_shared.is_some();
    if wo_b_decode_dg {
        let scratch = state.fused_wqkv.as_mut().expect("wo_b dg gate checked");
        let wo_b_cache = attention
            .wo_b_deepgemm
            .as_ref()
            .expect("wo_b dg gate checked");
        let wo_b_cols = attention.wo_b().cols;
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
                dsv4_linear(ctx, attention.wo_b(), &latent, out)
            })
        })?;
    }
    // Restore the shared fused-wqkv scratch active_counts to [1] after a batched (M=n)
    // decode-DeepGEMM O-LoRA: every per-row M=1 reader relies on it being [1]. No-op
    // at token_count==1.
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

fn mla_oproj_decode(
    ctx: &DeviceContext,
    attention: &Dsv4Attention,
    state: &mut Dsv4LayerAttentionState,
    local_attn: &HiddenStates,
    latent: &mut HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        attention.o_proj.is_none(),
        "DSv4 MODEL1 decode O projection is MODEL1-only; GLM/plain-o uses eager decode"
    );
    let token_count = 1usize;
    let shape = dsv4_oproj_group_shape(
        attention.wo_a().rows,
        attention.wo_a().cols,
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
        latent.hidden_dim == attention.wo_a().rows && latent.seq_len == token_count,
        "DSv4 MODEL1 decode O-LoRA latent scratch {}x{} != {}x1",
        latent.hidden_dim,
        latent.seq_len,
        attention.wo_a().rows
    );
    let wo_a_decode_dg = shape.groups == 1
        && dsv4_deepgemm_enabled()
        && state.fused_wqkv.is_some()
        && attention.wo_a_deepgemm.is_some();
    let wo_a_group_decode_dg = shape.groups > 1
        && dsv4_deepgemm_enabled()
        && state.fused_wqkv.is_some()
        && attention.wo_a_group_deepgemm.is_some();
    if wo_a_decode_dg {
        let wo_a_cache = attention
            .wo_a_deepgemm
            .as_ref()
            .expect("wo_dg gate checked");
        let wo_a_cols = attention.wo_a().cols;
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
                dsv4_linear(ctx, attention.wo_a(), local_attn, latent)
            })
        })?;
    } else {
        crate::profile::profile_op(ctx, "linear/wo_a", None, 1, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wo_a", || {
                dsv4_wo_a_grouped_linear(ctx, attention, local_attn, shape, latent)
            })
        })?;
    }

    let wo_b_decode_dg =
        dsv4_deepgemm_enabled() && state.fused_wqkv.is_some() && attention.wo_b_deepgemm.is_some();
    if wo_b_decode_dg {
        let wo_b_cache = attention
            .wo_b_deepgemm
            .as_ref()
            .expect("wo_b dg gate checked");
        let wo_b_cols = attention.wo_b().cols;
        let scratch = state.fused_wqkv.as_mut().expect("wo_b dg gate checked");
        crate::profile::profile_op(ctx, "linear/wo_b", None, 1, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wo_b", || {
                decode_proj_deepgemm(ctx, scratch, wo_b_cache, latent, out, wo_b_cols)
            })
        })?;
    } else {
        crate::profile::profile_op(ctx, "linear/wo_b", None, 1, || {
            crate::linear_profile::profile(ctx, "dsv4/linear/wo_b", || {
                dsv4_linear(ctx, attention.wo_b(), latent, out)
            })
        })?;
    }
    Ok(())
}

/// GLM SparseIndexed only: build the full-sequence index KEY ring (ratio=1, no
/// compressor). Projects one MQA index key per token via `indexer.wk`, RMSNorms it
/// with `indexer.k_norm` (width `index_head_dim`), and appends the
/// `[index_head_dim, seq_len]` normed keys into `state.compressed` at absolute rows
/// `[start_pos, start_pos + seq_len)` - exactly what `csa_select_official` reads.
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
    // SAFETY: dsv4_linear writes the full wk_out buffer.
    let mut wk_out = unsafe { HiddenStates::uninit(ctx, wk.rows, seq_len)? };
    dsv4_linear(ctx, wk, hidden, &mut wk_out)?;
    keepalive.keep_hidden(&wk_out);

    // The index KEY ring stores ONE key of width `index_head_dim` per token, but GLM's
    // `wk` doc says `[index_n_heads*index_head_dim, hidden]`. That width reduction is
    // the load-bearing unverifiable detail - branch on `wk.rows`.
    let normed = if wk.rows == config.index_head_dim {
        // Single-head MQA key (`wk = Linear(hidden, index_head_dim)`, ONE key shared
        // across
        // all index_n_heads query heads - confirmed against the vLLM DeepSeek-V3.2
        // indexer
        // reference). GLM takes this branch (index_head_dim=128).
        // NUMERIC GAP: the DSv3.2 reference normalizes the index key with
        // `LayerNorm(index_head_dim, eps=1e-6)` using both `k_norm` weight and bias,
        // and
        // GLM ships a `k_norm.bias`. This path applies the bias-free `mla_rms_norm`
        // instead - an approximation to replace once a GPU forward confirms GLM's exact
        // norm.
        mla_rms_norm(ctx, &wk_out, k_norm, config.rms_norm_eps)?
    } else {
        // GLM's real `wk` is `[index_head_dim, hidden]` (single MQA key), so this
        // branch is
        // not expected - fail loud rather than fabricate a per-head->single-key
        // reduction.
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

    // Stage this forward's delta into the STAGING RING. `csa_select_official` drains
    // [packed_rows..rows_after) in the SAME forward, so only the live delta lives
    // here. Each forward stages its `seq_len` rows CONTIGUOUSLY from ring row 0 (window
    // base = this forward's `start_pos`); staging from 0 rather than
    // `start_pos % ring_rows` keeps the delta contiguous for any unaligned decode/MTP
    // start_pos. The logical committed count is `compressed.seq_len`; frozen-KV verify
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
        // Window-relative: ring rows [0, seq_len) for absolute [start_pos,
        // start_pos+seq_len).
        let hi = seq_len * config.index_head_dim;
        let mut dst = state.compressed.data.slice_mut(0..hi);
        ctx.stream
            .memcpy_dtod(&normed.data, &mut dst)
            .map_err(|e| anyhow!("DSv4 SparseIndexed index-key ring D2D failed: {e}"))?;
        state.compressed.seq_len = start_pos + seq_len;
    }
    Ok(())
}

/// FP32 main-value compressor: re-runs the compressor forward in FP32 (BF16 input
/// projections via FP32-accumulate GEMM, FP32 APE, FP32 state carry) to avoid
/// BF16/FP8 value mismatches (#146, #150), writing the BF16 mirrors and the
/// compressed output back into `state`. Runs on every prefill compression boundary;
/// the decode fast path is unchanged. When `fp32_carry_stale` the bf16 carry is the
/// authority - reseed FP32 from it before the probe reads pending/prev.
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
    rope: RopeParams,
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
        // Shared FP32 GEMM scratch, written and consumed within this call.
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
            {
                flash_kv::dsv4_compressor_fp32_carry_reseed_raw(
                    &ctx.stream,
                    pkv_b,
                    psc_b,
                    prkv_b,
                    prsc_b,
                    pkv,
                    psc,
                    prkv,
                    prsc,
                    pending_elems,
                    prev_elems,
                )?;
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
            // Dense BF16 matrices and outputs match the checked M/N/K shapes.
            tensor_ops::gemm_bf16_f32_raw(
                &ctx.stream,
                wkv,
                x,
                kv,
                width as i32,
                token_count as i32,
                hidden.hidden_dim as i32,
            )?;
            tensor_ops::gemm_bf16_f32_raw(
                &ctx.stream,
                wgate,
                x,
                score,
                width as i32,
                token_count as i32,
                hidden.hidden_dim as i32,
            )?;
        }
        let rope_dim = config.qk_rope_head_dim;
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
            {
                flash_kv::dsv4_compressor_fp32_prefill_probe_raw(
                    &ctx.stream,
                    kv,
                    score,
                    ape,
                    norm,
                    pkv,
                    psc,
                    prkv,
                    prsc,
                    prkv_bf16,
                    prsc_bf16,
                    pkv_bf16,
                    psc_bf16,
                    compressed,
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
                    rope.base,
                    rope.original_seq_len,
                    rope.factor,
                    rope.beta_fast,
                    rope.beta_slow,
                )?;
            }
        }
        state.compressed.seq_len = compressed_rows;
        Ok(())
    })?;
    Ok(())
}

/// Run one compressor sub-block over `hidden`, updating the per-slot bf16
/// compressed-key pool for the absolute `[0, start_pos + token_count)` range.
/// `dsv4_compressor_update_cuda` folds the wkv / wgate streams through `ape` +
/// RMSNorm(`norm`) + compress-rope into one row per `compress_ratio` tokens.
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
    pos: Dsv4Position<'_>,
    rope: RopeParams,
    // Shared FP32 probe scratch: required on every prefill lane (the probe branch
    // consumes it); decode lanes pass `None`.
    fp32_scratch: Option<&mut Dsv4CompressorFp32Scratch>,
    // When `Some`, the two m=1 projection GEMVs are skipped and this row's `[width, 1]`
    // slices of the batched output are used directly; everything downstream is
    // unchanged.
    precomputed: Option<(&HiddenStates, &HiddenStates)>,
    // When `Some`, the per-row `dsv4_compressor_update_*` FFI is skipped and this row's
    // five ring-state device pointers go into the batch sink instead; the state update
    // runs later in ONE `dsv4_compressor_update_batched`. `compressed.seq_len` IS
    // still advanced here (host bookkeeping - the batched kernel writes the data
    // before any reader). Requires the start_pos_ptr (decode) path.
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
    let total = pos.start + token_count;
    let compressed_rows = total / ratio;
    let start_pos_i32 = i32::try_from(pos.start)
        .map_err(|_| anyhow::anyhow!("DSv4 compressor start_pos {} exceeds i32", pos.start))?;
    let pending_len = pos.start % ratio;
    let pending_len_i32 = i32::try_from(pending_len)
        .map_err(|_| anyhow::anyhow!("DSv4 compressor pending_len {pending_len} exceeds i32"))?;
    let compressed_base = pos.start / ratio;
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

    // FP32 probe: prefill only. The #146/#150 corruption was a multi-token prefill
    // boundary issue; decode uses the BF16 path.
    if !dsv4_verify_frozen() && token_count > 0 && pos.device.is_none() {
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
            pos.start,
            token_count,
            compressed_rows,
            rope,
        )?;
        return Ok(());
    }

    // Full-flatten decode: skip BOTH the per-row projection GEMVs and the state-update
    // FFI, pushing this row's five ring-state pointers into the batch sink and
    // advancing `compressed.seq_len`. The GPU write runs later in ONE
    // `dsv4_compressor_update_batched`, before any reader. MUST early-return before the
    // GEMV match below, or the per-row GEMVs this replaces would still run.
    if let Some(sink) = defer_update {
        ensure!(
            pos.device.is_some(),
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

    // `owned_*` hold the GEMV-produced buffers in the `None` branch (declared in the
    // outer scope so they outlive the FFI read below); `kv_raw` / `score_raw`
    // reference whichever source is active. The downstream FFI reads them by device
    // pointer and is identical for both.
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

    // Compressed keys use compress_rope_theta with NO YaRN (original_seq_len = 0).
    let rope_dim = config.qk_rope_head_dim;
    // Raw bf16 read: on a quantized matrix `.data` is a 1-element dummy (#138 OOB
    // class). The loader dequants ape to dense.
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
        // prev_overlap always resolves from `state`'s own per-slot buffer; stride 0
        // collapses the kernel's indexing to the single-register form.
        let (prkv_ptr, _kg2) = state.prev_overlap_kv.device_ptr_mut(&ctx.stream);
        let (prsc_ptr, _sg2) = state.prev_overlap_score.device_ptr_mut(&ctx.stream);
        let overlap_page_stride = 0i32;
        let (comp_ptr, _cg) = state.compressed.data.device_ptr_mut(&ctx.stream);
        let has_prev_overlap = i32::from(compressed_base > 0);
        // SAFETY: all buffers valid on ctx.stream; state carries the pending and
        // overlap rows from previous contiguous appends.
        if !dsv4_verify_frozen() {
            {
                if let Some(start_pos_device) = pos.device {
                    let (start_ptr, _spg) = start_pos_device.device_ptr(&ctx.stream);
                    flash_kv::dsv4_compressor_update_start_pos_ptr_raw(
                        &ctx.stream,
                        kv_ptr,
                        score_ptr,
                        ape_ptr,
                        norm_ptr,
                        pkv_ptr,
                        psc_ptr,
                        prkv_ptr,
                        prsc_ptr,
                        comp_ptr,
                        token_count as i32,
                        start_ptr,
                        head_dim as i32,
                        ratio as i32,
                        width as i32,
                        i32::from(overlap),
                        overlap_page_stride,
                        config.rms_norm_eps,
                        rope_dim as i32,
                        rope.base,
                        rope.original_seq_len,
                        rope.factor,
                        rope.beta_fast,
                        rope.beta_slow,
                    )?;
                } else {
                    flash_kv::dsv4_compressor_update_raw(
                        &ctx.stream,
                        kv_ptr,
                        score_ptr,
                        ape_ptr,
                        norm_ptr,
                        pkv_ptr,
                        psc_ptr,
                        prkv_ptr,
                        prsc_ptr,
                        comp_ptr,
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
                        rope.base,
                        rope.original_seq_len,
                        rope.factor,
                        rope.beta_fast,
                        rope.beta_slow,
                    )?;
                }
            }
        }
    }
    // Frozen-KV: a frozen verify SKIPS the compressor/indexer CUDA update above, so it
    // must NOT advance `compressed.seq_len` either - otherwise CSA / FlashMLA in the
    // same verify would attend a compressed row whose data was never produced, and
    // `csa_select` would advance DSA `packed_rows` off `indexer_rows_after`.
    if !dsv4_verify_frozen() {
        state.compressed.seq_len = compressed_rows;
        // bf16 lane advanced the carry; the next FP32 probe must reseed.
        state.fp32_carry_stale = true;
    }
    Ok(())
}

/// Batched (m=N) decode compressor/indexer projection pre-pass: projects the N-row
/// post-attn-LN `normed_batch` [hidden_size, N] through `compressor.wkv` /
/// `compressor.wgate` ONCE, returning `kv_raw_batch` and `score_raw_batch`
/// `[width, N]`. Each row's `[width, 1]` slice then feeds [`compressor_forward`] as
/// `precomputed`, replacing the per-row m=1 GEMVs that re-read the full weight per
/// row (~54% of the decode step, dead-linear in N). Both outputs are kept alive via
/// `keepalive.keep_hidden`; touches NO slot state.
/// One batched (m=N) decode projection: tensor-core DeepGEMM when an FP8 repack
/// cache AND the shared prefill scratch are present, else the scalar `dsv4_linear`
/// GEMV. The single routing point for every compressor/indexer batch pre-pass.
fn proj_batched(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    cache: Option<&cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache>,
    scratch: Option<&mut Dsv4PrefillDeepGemmLinearScratch>,
    input: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    match (cache, scratch) {
        (Some(cache), Some(scratch)) if input.seq_len > 1 => {
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
    // Batch the per-row compressor projections into ONE m=N GEMM each. The DSv4
    // compressor weights are bf16, so this is the cublasLt path - weight read once
    // instead of N times. No FP8 quant, so no selection-shift correctness risk.
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

/// Batched (m=N) decode indexer-query / gating-weights projection pre-pass: one m=N
/// GEMM each (`q_i_batch [wq_b.rows, N]`, `weights_batch [weights_proj.rows, N]`)
/// replacing the per-row m=1 GEMVs that re-read the full weight per decode row.
/// Each row's `[width,1]` slice then feeds `csa_select` as `query_precomputed`; the
/// per-slot DSA cache writes + gathers stay per-row. `c_q_normed_batch` is
/// `[q_lora_rank, N]`, `normed_batch` is `[hidden, N]`; both outputs are kept alive
/// via `keepalive.keep_hidden`. Touches NO slot state; bf16 weights -> cublasLt.
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
/// decode pre-pass: `q_i` and `weights` are this row's `[*, 1]` column VIEWs of the
/// [`indexer_query_batch_prepass`] output. Threaded into [`csa_select`] they replace
/// the per-row m=1 GEMVs AND the per-row D2D re-copy (zero copy).
pub(crate) struct Dsv4IndexerQueryPrecomputed<'a> {
    pub(crate) q_i: HiddenStatesView<'a>,
    pub(crate) weights: HiddenStatesView<'a>,
}

/// Host-gathered per-row device pointer arrays for one compressor's batched state
/// update, uploaded to device by [`dsv4_compressor_update_batched`]. Built over the
/// prepare loop's row order - push exactly one entry per row.
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

    /// Push one row's five state pointers (via
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

/// Host-gathered per-slot device arrays for the batched DSA cache-write (block (a)
/// of [`csa_select_official`]): the Hadamard rotate of each slot's newly-packed
/// index-key rows + the FP8 fused-store into its cache band. Push exactly one entry
/// per slot via [`dsv4_dsa_cache_write_gather_row`]. The READ side
/// ([`csa_select_official_batched`]) reads the populated cache.
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
/// `official.packed_rows` to `indexer_rows_after` - the host half of block (a) of
/// [`csa_select_official`], hoisted into the batched pre-pass and re-asserting the
/// same preflight invariants. `keys_window_base` is `0` for the full-retention
/// compressor-indexer. A slot with `newly_packed == 0` pushes a clean skip-entry
/// (the kernel guard early-returns) and leaves `packed_rows` unchanged.
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
        // Clean skip-entry: array lengths stay == n and the kernel guard early-returns.
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
    // The raw u64 ptr stays valid after the guard drops - single-stream, buffer not
    // reallocated; the launch later re-reads it via the uploaded array.
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
    // rotated_keys is a transient drain-immediate ring: the Hadamard writes the delta
    // at ring-relative 0 and the fused-store reads it back the same launch, so BOTH
    // the dst and the src are ring-relative 0 (dst_row indexes only cache_locs / the
    // FP8 cache band).
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

fn state_indexer_compressed(state: &Dsv4LayerAttentionState) -> Result<&HiddenStates> {
    Ok(&state
        .indexer
        .as_ref()
        .ok_or_else(|| anyhow!("DSv4 batched DSA cache-write requires indexer state"))?
        .compressed)
}

/// Batched DSA cache-write (block (a) of [`csa_select_official`]): ONE
/// `<<<dim3(.,n)>>>` Hadamard-rotate launch + ONE FP8 fused-store launch over all n
/// slots, replacing the n per-row pairs. `ptrs` holds the host-gathered per-slot
/// base ptrs / offsets / counts; `max_rows` = max newly_packed across slots (the
/// x-grid bound; 0 => the launchers early-return). The uploaded arrays are held in
/// `ptr_keepalive` - the N>1 keepalive is inert, so explicit retention guards a
/// premature free until the next stream sync.
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
        {
            flash_kv::dsv4_dsa_hadamard128_batched_raw(
                &ctx.stream,
                keys_src_a,
                src_ring_row_a,
                rotated_dst_a,
                dst_row_a,
                newly_packed_a,
                n as i32,
                max_rows,
            )?;
        }
        {
            flash_kv::dsv4_dsa_fused_store_index_k_cache_batched_raw(
                &ctx.stream,
                rotated_src_a,
                cache_band_a,
                cache_locs_a,
                newly_packed_a,
                n as i32,
                max_rows,
                64,
            )?;
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
    // i32 offset/count arrays held via the forward keepalive, same premature-free guard
    // as the u64 ptr arrays above.
    keepalive.keep_i32(&src_ring_row_arr);
    keepalive.keep_i32(&dst_row_arr);
    keepalive.keep_i32(&newly_packed_arr);
    Ok(())
}

/// Batched (m=N) compressor STATE update: ONE `<<<n, BLOCK>>>` launch running each
/// row's per-slot compressor ring update (RoPE/RMSNorm/store into
/// pending/overlap/compressed), replacing the N per-row launches. `kv_raw_batch` /
/// `score_raw_batch` are the [`compressor_batch_prepass`] outputs `[width, n]`;
/// `ape`/`norm` are the SHARED compressor weights; `ptrs` holds the N rows' state
/// buffer pointers; `start_pos` is the contiguous `[N]` decode-position array. Dims
/// and rope params mirror the per-row [`compressor_forward`] args exactly. The
/// pointer-array uploads are kept alive by the caller's `ptr_keepalive`.
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
    rope: RopeParams,
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
    // Compressed keys use compress_rope_theta with NO YaRN (original_seq_len = 0),
    // identical to the per-row `compressor_forward`.
    let rope_dim = config.qk_rope_head_dim;
    // Upload the five per-row pointer arrays + hold them alive to function return.
    let pkv_arr = crate::ops::upload_u64(ctx, &ptrs.pending_kv)?;
    let psc_arr = crate::ops::upload_u64(ctx, &ptrs.pending_score)?;
    let prkv_arr = crate::ops::upload_u64(ctx, &ptrs.prev_overlap_kv)?;
    let prsc_arr = crate::ops::upload_u64(ctx, &ptrs.prev_overlap_score)?;
    let comp_arr = crate::ops::upload_u64(ctx, &ptrs.compressed)?;
    // Resolve raw device pointers and release the cudarc borrow guards BEFORE the push
    // below (the guards would otherwise borrow the arrays past the move into
    // `ptr_keepalive`). The raw u64 ptrs stay valid - the buffers are not reallocated
    // and the stream is ordered. ape must be dense bf16 (#138).
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
        {
            flash_kv::dsv4_compressor_update_batched_start_pos_ptr_raw(
                &ctx.stream,
                kv_ptr,
                score_ptr,
                ape_ptr,
                norm_ptr,
                pkv_a,
                psc_a,
                prkv_a,
                prsc_a,
                comp_a,
                n as i32,
                1, // num_tokens per row (decode)
                start_ptr,
                head_dim as i32,
                ratio as i32,
                width as i32,
                i32::from(overlap),
                0, // overlap_page_stride: per-slot register form (#154 D1)
                config.rms_norm_eps,
                rope_dim as i32,
                rope.base,
                rope.original_seq_len,
                rope.factor,
                rope.beta_fast,
                rope.beta_slow,
            )?;
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
    // Hold the pointer arrays alive until the caller's keepalive Vec drops - the N>1
    // forward keepalive is inert, so explicit retention guards the premature free.
    ptr_keepalive.push(pkv_arr);
    ptr_keepalive.push(psc_arr);
    ptr_keepalive.push(prkv_arr);
    ptr_keepalive.push(prsc_arr);
    ptr_keepalive.push(comp_arr);
    Ok(())
}

/// Batched-decode gather sink threaded into [`csa_select`] /
/// [`mla_attention_prepare_compressed_only`] for row `r`. When present,
/// `csa_select` does the per-row CACHE WRITES only, gathers this row's bf16 indexer
/// query and gating weights into the N-row staging at offset `r`, and captures this
/// row's exact `key_count` so the ONE batched `csa_select_official_batched`
/// produces byte-equivalent context_lens. `selected` is then `None`.
/// Per-row precomputed compressor/indexer projection slices for the batched decode
/// pre-pass. Each `(kv_raw, score_raw)` pair is this row's `[width, 1]` column slice
/// of the batched `dsv4_linear` output, replacing the per-row m=1 GEMVs inside
/// [`compressor_forward`]; the per-slot state update stays per-row. `main` feeds
/// `attention.compressor`, `indexer` feeds `attention.indexer.compressor`.
pub(crate) struct Dsv4CompressorPrecomputed<'a> {
    pub(crate) main: (&'a HiddenStates, &'a HiddenStates),
    pub(crate) indexer: Option<(&'a HiddenStates, &'a HiddenStates)>,
}

pub(crate) struct Dsv4DsaBatchedGather<'a> {
    /// Optional N-row staging for indexer query, shape
    /// `[local_index_heads*index_head_dim, n]`.
    /// `None` when the batched indexer-query prepass already produced the exact buffer.
    pub(crate) q_i_batch: Option<&'a mut HiddenStates>,
    /// Optional N-row staging for gating weights, shape `[local_index_heads, n]`.
    pub(crate) weights_batch: Option<&'a mut HiddenStates>,
    /// Row index in `[0, n)` to gather into.
    pub(crate) row: usize,
    /// Per-row captured `key_count` (push exactly one per call), for the batched
    /// context_lens (`min(key_count_r, abs_pos_r/ratio)`).
    pub(crate) key_counts: &'a mut Vec<i32>,
    /// `true` when the per-row CACHE WRITES already ran in the ONE batched pre-pass
    /// ([`dsv4_dsa_cache_write_batched`], which also advanced `packed_rows`), so
    /// `csa_select` must NOT run them again. `false` on the SparseIndexed lane, which
    /// has no such pre-pass.
    pub(crate) cache_writes_in_prepass: bool,
}

#[allow(clippy::too_many_arguments)]
fn csa_select_decode(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    indexer: &Dsv4Indexer,
    hidden: &HiddenStates,
    c_q_normed: &HiddenStates,
    keys: &HiddenStates,
    keys_capacity: usize,
    keys_window_base: usize,
    official: &mut Dsv4DsaOfficialState,
    shared: &mut Dsv4DsaSharedScratch,
    pool: &mut Dsv4LayerKvLayout,
    indexer_rows_before: usize,
    indexer_rows_after: usize,
    pos: Dsv4Position<'_>,
    ratio: usize,
    q_i: &mut HiddenStates,
    weights: &mut HiddenStates,
    selected: &mut CudaSlice<i32>,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<()> {
    ensure!(
        hidden.seq_len == 1 && c_q_normed.seq_len == 1,
        "DSv4 MODEL1 decode CSA select is decode-only, hidden seq={} c_q seq={}",
        hidden.seq_len,
        c_q_normed.seq_len
    );
    ensure!(
        selected.len() >= config.index_topk,
        "DSv4 MODEL1 decode CSA selected scratch len {} < topk {}",
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
        "DSv4 MODEL1 decode CSA q_i width {} is not divisible by index_head_dim {}",
        q_i.hidden_dim,
        config.index_head_dim
    );
    let local_index_heads = q_i.hidden_dim / config.index_head_dim;
    ensure!(
        weights.hidden_dim == local_index_heads,
        "DSv4 MODEL1 decode CSA weights width {} != local index heads {local_index_heads}",
        weights.hidden_dim
    );
    let key_count = if pos.device.is_some() {
        keys_capacity
    } else {
        keys.seq_len
    };
    let score_scale =
        (config.index_head_dim as f32).powf(-0.5) * (config.index_n_heads as f32).powf(-0.5);

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
        pos,
        keys_window_base,
        ratio,
        local_index_heads,
        score_scale,
        Some(selected),
        /* cache_writes_only */ false,
        keepalive,
    )?;
    ensure!(
        selected_return.is_none(),
        "DSv4 MODEL1 decode CSA official select unexpectedly allocated selected output"
    );
    Ok(())
}

/// CSA top-k block selection: project the index query (`wq_b`) + per-head gating
/// (`weights_proj`), then the official DSA selector scores each compressed-key block
/// and writes the top-`index_topk` block ids per token into `[seq * index_topk]`.
/// When `batched_gather` is `Some`, only the per-row cache writes run and this row's
/// `q_i`/`weights`/`key_count` are gathered for ONE later
/// `csa_select_official_batched`.
#[allow(clippy::too_many_arguments)]
fn csa_select(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
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
    pos: Dsv4Position<'_>,
    ratio: usize,
    prefill_scratch: Option<&mut Dsv4PrefillDeepGemmLinearScratch>,
    batched_gather: Option<Dsv4DsaBatchedGather<'_>>,
    // When `Some`, this row's `q_i`/`weights` were already projected batched, so the
    // per-row m=1 GEMVs below are SKIPPED; the slice IS the exact column of the
    // batched GEMM output.
    query_precomputed: Option<Dsv4IndexerQueryPrecomputed<'_>>,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<Option<CudaSlice<i32>>> {
    // Batched pre-pass: the precomputed `q_i`/`weights` are borrowed `[width,1]` column
    // VIEWs of the prepass output - zero copy, GEMVs skipped, decode-only. `owned_*`
    // stays `None` here: the only consumer needing `&HiddenStates` (the non-batched
    // fallback below) is unreachable when precomputed.
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
            // Prefill index-query (M=token_count) -> DeepGEMM, off the scalar fp8_gemv
            // (the #1
            // remaining projection at M=1024). Decode / no-cache stays scalar.
            let indexer_wq_b_dg = c_q_normed.seq_len > 1
                && dsv4_deepgemm_enabled()
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

    // Unified VIEW over q_i/weights: the owned buffer's full view (non-precomputed) or
    // the prepass column view - identical device pointer and width either way.
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

    let key_count = if pos.device.is_some() {
        if dsv4_verify_frozen() {
            // Frozen-KV: the selector computes `available = min(key_count,
            // abs_pos / ratio)`, and a frozen verify's `abs_pos` can cross a
            // compression boundary, exposing a compressed row the frozen
            // compressor never produced. Pin to the committed indexer row count.
            keys.seq_len
        } else {
            // Use capacity, not the current compressed-key seq_len: the
            // selector's `available = min(key_count, abs_pos / ratio)` already
            // clamps to the same causal set.
            keys_capacity
        }
    } else {
        keys.seq_len
    };
    let score_scale =
        (config.index_head_dim as f32).powf(-0.5) * (config.index_n_heads as f32).powf(-0.5);

    // Batched-decode lane: per-row CACHE WRITES only, then gather this row's
    // q_i/weights into the N-row staging + capture key_count. The READ (logits + topk)
    // is deferred to ONE `csa_select_official_batched` after all N rows' caches are
    // populated. Requires the official DSA path.
    if let Some(mut gather) = batched_gather {
        // Per-row CACHE WRITES (block (a) of csa_select_official). When the batched
        // pre-pass already populated all N slots' DSA caches, skip this write and the
        // `packed_rows` advance; the SparseIndexed lane (no pre-pass) still runs it.
        if !gather.cache_writes_in_prepass {
            // `cache_writes_in_prepass` is false only on the SparseIndexed lane, which
            // never
            // supplies `query_precomputed` => `owned_*` is Some. Asserted, not assumed.
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
                pos,
                keys_window_base,
                ratio,
                local_index_heads,
                score_scale,
                None,
                /* cache_writes_only */ true,
                keepalive,
            )?;
        }
        // Gather this row's q_i/weights (bf16) into the N-row staging at row r, from
        // the
        // unified VIEW (prepass column view or the owned buffer's view).
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

    // Non-batched single-row / prefill fallback: precomputed Some => batched_gather
    // Some, so this is never reached when precomputed and `owned_*` is Some.
    // Asserted, not assumed.
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
        pos,
        keys_window_base,
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
    pos: Dsv4Position<'_>,
    // Ring window base for reading the `keys` staging buffer: `start_pos` for the
    // SparseIndexed indexer (staged window-relative from ring row 0 every forward),
    // `0` for the full-retention compressor. Row `r` is at `(r - keys_window_base) *
    // ihd`.
    keys_window_base: usize,
    ratio: usize,
    local_index_heads: usize,
    score_scale: f32,
    selected_out: Option<&mut CudaSlice<i32>>,
    // Batched-decode lane: run block (a) (per-row CACHE WRITES) only, then return
    // `Ok(None)` BEFORE the per-row read/select (b)-(f), which is deferred to ONE
    // `csa_select_official_batched`.
    cache_writes_only: bool,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<Option<CudaSlice<i32>>> {
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
        pos.start + q_i.seq_len <= shared.max_tokens,
        "DSv4 official DSA positions {}..{} exceed freqs_cis max {}",
        pos.start,
        pos.start + q_i.seq_len,
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
        // The SOURCE `keys.data` is the indexer's STAGING RING: the live delta was
        // staged
        // this same forward window-relative to `keys_window_base`, so row `r` sits at
        // `r - window_base`. The DESTINATION `rotated_keys` stays at the ABSOLUTE
        // `packed_rows` offset (it retains the full history). Drain-immediate
        // guarantees
        // the window base never trails `packed_rows`.
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
        // `rotated_keys` is a transient drain-immediate staging buffer: the Hadamard
        // writes
        // the delta at ring offset 0 and the fused-store reads it back the same
        // forward, so
        // the dst is ALWAYS ring-relative 0. `newly_packed <= ring_rows` follows from
        // the
        // source straddle check above.
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
            {
                flash_kv::dsv4_dsa_hadamard128_bf16_raw(
                    &ctx.stream,
                    src_ptr,
                    rot_ptr,
                    i32::try_from(newly_packed)?,
                )?;
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
            {
                flash_kv::dsv4_dsa_fused_store_index_k_cache_raw(
                    &ctx.stream,
                    rot_store_ptr,
                    cache_ptr_u8,
                    locs_ptr,
                    i32::try_from(newly_packed)?,
                    64,
                )?;
            }
        }
        official.packed_rows = indexer_rows_after;
    }

    if cache_writes_only {
        return Ok(None);
    }

    let token_count = q_i.seq_len;
    // `raw_indices` (topk output) is sized by `query_chunk`, not `max_tokens`. The
    // scheduler guarantees a single forward never passes more than
    // `chunked_prefill_size <= DSV4_PREFILL_QUERY_CHUNK` query tokens, so the per-tile
    // writes and the DUMP read both stay in bounds - fail loud rather than write past
    // it.
    ensure!(
        token_count <= shared.query_chunk,
        "DSv4 official DSA token_count {} exceeds prefill query chunk {} (raw_indices \
         scratch is chunk-sized; chunked prefill must keep seq_len <= \
         chunked_prefill_size <= DSV4_PREFILL_QUERY_CHUNK)",
        token_count,
        shared.query_chunk
    );
    let mut owned_selected = if selected_out.is_none() {
        // The MODEL1 decode lane passes persistent scratch in `selected_out`, so
        // it does not allocate here.
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

        // Query-axis tiling - the ONLY compute path. The logits scratch is bounded by
        // `tile x logits_stride`, so long prompts loop in tiles and never materialize
        // full-N logits. Per-tile buffers (logits, q_fp8, weights, context_lens,
        // positions) are overwritten before use; `page_table_identity` is read-only;
        // and
        // `selected` / `raw_indices` take disjoint `[t0..t0+tlen)` slices. The
        // key-packing
        // buffers are handled in the query-independent block above.
        let tile = shared.query_tile;
        // q_i.data / weights.data are flat [seq_len * per_token_width]; derive
        // per-token
        // strides so each tile slices the right sub-range.
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

            // (a) per-tile context_lens / positions. Decode carries `start_pos` on
            // device, so
            // fill tile metadata on GPU to avoid two tiny H2D copies per CSA layer.
            {
                let mut context_lens = shared.context_lens.slice_mut(0..tlen);
                let mut positions = shared.positions.slice_mut(0..tlen);
                if let Some(start_pos_device) = pos.device {
                    let (lens_ptr, _lg) = context_lens.device_ptr_mut(&ctx.stream);
                    let (positions_ptr, _pg) = positions.device_ptr_mut(&ctx.stream);
                    let (start_ptr, _sg) = start_pos_device.device_ptr(&ctx.stream);
                    {
                        flash_kv::dsv4_dsa_fill_context_lens_positions_start_pos_raw(
                            &ctx.stream,
                            lens_ptr,
                            positions_ptr,
                            start_ptr,
                            i32::try_from(t0)?,
                            i32::try_from(tlen)?,
                            i32::try_from(key_count)?,
                            i32::try_from(ratio)?,
                        )
                        .map_err(|e| anyhow!("DSv4 official DSA GPU metadata fill failed: {e}"))?;
                    }
                } else {
                    let context_lens_tile: Vec<i32> = (0..tlen)
                        .map(|i| {
                            let abs_pos = pos.start + t0 + i;
                            i32::try_from(std::cmp::min(key_count, abs_pos / ratio))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let positions_tile: Vec<i32> = (0..tlen)
                        .map(|i| i32::try_from(pos.start + t0 + i))
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
                {
                    flash_kv::dsv4_dsa_fused_q_indexer_rope_hadamard_quant_raw(
                        &ctx.stream,
                        q_ptr,
                        q_fp8_ptr,
                        w_ptr,
                        weights_out_ptr,
                        score_scale,
                        freqs_ptr,
                        positions_ptr,
                        i32::try_from(tlen)?,
                        i32::try_from(local_index_heads)?,
                    )?;
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
                    cuda_moe::dsv4_deepgemm_fp8_paged_mqa_logits_fused_cache(
                        RawDevicePtr::from_raw(q_ptr),
                        RawDevicePtr::from_raw(cache_ptr_u8),
                        RawDevicePtr::from_raw(weights_ptr),
                        RawDevicePtr::from_raw(lens_ptr),
                        RawDevicePtr::from_raw(page_ptr),
                        RawDevicePtr::from_raw(sched_ptr),
                        RawDevicePtr::from_raw(logits_ptr),
                        tlen,
                        1,
                        local_index_heads,
                        config.index_head_dim,
                        shared.num_pages,
                        64,
                        shared.num_pages * 64,
                        shared.logits_stride,
                        shared.num_pages,
                        64 * (config.index_head_dim + std::mem::size_of::<f32>()),
                        shared.num_sms,
                        ctx.stream.cu_stream(),
                    )
                    .map_err(|e| anyhow!("DSv4 official DSA paged logits failed: {e}"))?;
                }
            }

            // (f) topk transform: read shared.logits, write the tile's disjoint output
            // slices
            // of `selected` and `shared.raw_indices`.
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
                {
                    flash_kv::dsv4_deepseek_v4_topk_transform_raw(
                        &ctx.stream,
                        logits_ptr,
                        lens_ptr,
                        page_ptr,
                        sel_ptr,
                        raw_ptr,
                        i64::try_from(shared.logits_stride)?,
                        i64::try_from(shared.num_pages)?,
                        i64::try_from(config.index_topk)?,
                        i32::try_from(tlen)?,
                        i32::try_from(config.index_topk)?,
                        64,
                    )?;
                }
            }

            t0 += tlen;
        }
    }
    keepalive.keep_u8(&shared.q_fp8);
    keepalive.keep_f32(&shared.weights);
    Ok(owned_selected)
}

/// BATCHED CSA select over N decode rows: blocks (b)-(f) of
/// [`csa_select_official`] with the READ side (paged-MQA logits + topk) batched into
/// ONE `batch_size=N` DeepGEMM call. The per-row CACHE WRITES (block (a)) have
/// ALREADY run, so this touches no per-slot cache state; the read side reads the
/// shared DSA key pool by the per-row block_table band.
///
/// `context_lens_host` / `positions_host` are captured ON HOST during the per-row
/// prepare (`min(key_count_r, abs_pos_r/ratio)`, `abs_pos_r`) - byte-equivalent to
/// the single-row GPU fill, since each row's exact `key_count_r` is captured rather
/// than assumed uniform.
///
/// `out_selected` (the FlashMLA batch scratch's `selected_batched`) is written here
/// BEFORE `build_layer_batch_meta` reads it; every other mutated device buffer is
/// stream-ordered and overwritten before read, and `page_table_identity_batch` is
/// read-only.
#[allow(clippy::too_many_arguments)]
pub(crate) fn csa_select_official_batched(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
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
    ensure!(
        slot_ids.len() == n && context_lens_host.len() == n && positions_host.len() == n,
        "DSv4 batched DSA host arrays slot_ids={} lens={} pos={} != n {}",
        slot_ids.len(),
        context_lens_host.len(),
        positions_host.len(),
        n,
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

    // (b1) per-row block_table band: row r -> slot r's DSA band = `num_pages`
    // contiguous blocks based at `slot_idx * num_pages`. H2D into
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

    // (c) fused Q indexer rope+hadamard+quant over the N gathered rows.
    {
        let (q_ptr, _qg) = q_i_batch.data.device_ptr(&ctx.stream);
        let (q_fp8_ptr, _qfg) = shared.q_fp8_batch.device_ptr_mut(&ctx.stream);
        let (w_ptr, _wg) = weights_batch.data.device_ptr(&ctx.stream);
        let (weights_out_ptr, _wog) = shared.weights_batch.device_ptr_mut(&ctx.stream);
        let (freqs_ptr, _fg) = shared.freqs_cis.device_ptr(&ctx.stream);
        let positions = shared.positions_batch.slice(0..n);
        let (positions_ptr, _pg) = positions.device_ptr(&ctx.stream);
        {
            flash_kv::dsv4_dsa_fused_q_indexer_rope_hadamard_quant_raw(
                &ctx.stream,
                q_ptr,
                q_fp8_ptr,
                w_ptr,
                weights_out_ptr,
                score_scale,
                freqs_ptr,
                positions_ptr,
                i32::try_from(n)?,
                i32::try_from(local_index_heads)?,
            )?;
        }
    }

    // (d) paged-MQA logits scheduling metadata. sched_meta is `(num_sms+1)*2`,
    // batch-INDEPENDENT, but the kernel reads all N context_lens to partition KV
    // across SMs, so pass batch_size=n.
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

    // (e) fused paged FP8 MQA logits -> logits_batch (N rows). The KV cache base is the
    // WHOLE shared DSA pool, with per-row routing via the block_table bands.
    // num_kv_blocks = decode_max_batch * num_pages; max_context_len = num_pages*64.
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
            cuda_moe::dsv4_deepgemm_fp8_paged_mqa_logits_fused_cache(
                RawDevicePtr::from_raw(q_ptr),
                RawDevicePtr::from_raw(cache_ptr_u8),
                RawDevicePtr::from_raw(weights_ptr),
                RawDevicePtr::from_raw(lens_ptr),
                RawDevicePtr::from_raw(block_ptr),
                RawDevicePtr::from_raw(sched_ptr),
                RawDevicePtr::from_raw(logits_ptr),
                n,
                1,
                local_index_heads,
                config.index_head_dim,
                num_kv_blocks,
                64,
                num_pages * 64,
                shared.logits_stride,
                num_pages,
                64 * (config.index_head_dim + std::mem::size_of::<f32>()),
                shared.num_sms,
                ctx.stream.cu_stream(),
            )
            .map_err(|e| anyhow!("DSv4 batched DSA paged logits failed: {e}"))?;
        }
    }

    // (f) topk transform: read logits_batch (per-row stride logits_stride), write the N
    // rows of `out_selected` (slot-relative indices) and `raw_indices_batch`. The
    // page_table is the N-row identity (stride=num_pages), so `page_to_slot(identity,
    // i) = i` - byte-equivalent to the single-row mapping.
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
        {
            flash_kv::dsv4_deepseek_v4_topk_transform_raw(
                &ctx.stream,
                logits_ptr,
                lens_ptr,
                page_ptr,
                sel_ptr,
                raw_ptr,
                i64::try_from(shared.logits_stride)?,
                i64::try_from(num_pages)?,
                i64::try_from(config.index_topk)?,
                i32::try_from(n)?,
                i32::try_from(config.index_topk)?,
                64,
            )?;
        }
    }

    keepalive.keep_u8(&shared.q_fp8_batch);
    keepalive.keep_f32(&shared.weights_batch);
    Ok(())
}
