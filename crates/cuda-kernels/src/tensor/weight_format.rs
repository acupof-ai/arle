//! Weight storage formats and the DSv4 FP8 DeepGEMM weight cache.

use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::{CudaAllocTraceExt, DeviceContext, DeviceMatrix};
use crate::ffi;

/// Explicit storage format for a linear weight matrix.
///
/// This is the Rust-side kernel ABI selector: checkpoint format detection and
/// loader packing set this once, then inference dispatch matches this enum
/// instead of re-interpreting packed buffers through bit-width sentinels.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum WeightFormat {
    /// Dense row-major BF16 weights.
    #[default]
    DenseBf16,
    /// Uniform per-group signed INT8 weights with BF16 scales.
    W8A16,
    /// Uniform per-group packed INT4 weights with BF16 scales.
    W4A16,
    /// Marlin W4 weights with dynamic INT8 activations.
    MarlinW4A8,
    /// Uniform per-group packed INT2 weights with BF16 scales.
    W2A16,
    /// GGUF Q3_K packed superblocks, scales embedded in each 256-wide block.
    GgufQ3K,
    /// GGUF Q4_K packed superblocks, scales embedded in each 256-wide block.
    GgufQ4K,
    /// GGUF Q5_K packed superblocks, scales embedded in each 256-wide block.
    GgufQ5K,
    /// GGUF Q6_K packed superblocks, scales embedded in each 256-wide block.
    GgufQ6K,
    /// TurboQuant packed indices + FP16 group norms + Hadamard signs.
    TurboQuant,
    /// DeepSeek V4 row-major FP8 E4M3 weights with FP8 E8M0 block scales.
    Dsv4Fp8BlockScaled,
    /// DeepSeek V4 row-major packed FP4 E2M1 weights with FP8 E8M0 block scales.
    Dsv4Fp4BlockScaled,
    /// ABI-generic row-major FP8 E4M3 weights with f32 block scales.
    Fp8BlockScaled,
    /// ABI-generic row-major FP8 E4M3 weights with one f32 scale per shard.
    Fp8PerShard,
    /// ABI-generic row-major packed FP4 E2M1 weights with FP8 group scales.
    Fp4E2M1Group,
}

/// Shape/layout constraints expected by the matching CUDA kernels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WeightKernelAlignment {
    pub weight_layout: &'static str,
    pub scale_layout: &'static str,
    pub k_multiple: usize,
    pub n_multiple: usize,
    pub group_size: usize,
}

impl WeightFormat {
    #[must_use]
    pub fn is_quantized(self) -> bool {
        !matches!(self, Self::DenseBf16)
    }

    pub fn validate_shape(self, rows: usize, cols: usize, group_size: usize) -> Result<()> {
        ensure!(rows > 0, "{self} requires rows > 0");
        ensure!(cols > 0, "{self} requires cols > 0");
        match self {
            Self::DenseBf16 => Ok(()),
            Self::W8A16 | Self::W4A16 | Self::W2A16 | Self::TurboQuant => {
                ensure!(group_size > 0, "{self} requires group_size > 0");
                ensure!(
                    cols.is_multiple_of(group_size),
                    "{self} requires cols % group_size == 0, got cols={cols}, group_size={group_size}"
                );
                Ok(())
            }
            Self::MarlinW4A8 => {
                ensure!(group_size > 0, "{self} requires group_size > 0");
                ensure!(
                    group_size == 128,
                    "{self} currently requires group_size=128, got {group_size}"
                );
                ensure!(
                    cols.is_multiple_of(group_size),
                    "{self} requires cols % group_size == 0, got cols={cols}, group_size={group_size}"
                );
                ensure!(
                    cols.is_multiple_of(128),
                    "{self} requires cols % 128 == 0, got {cols}"
                );
                ensure!(
                    rows.is_multiple_of(256),
                    "{self} requires rows % 256 == 0, got {rows}"
                );
                Ok(())
            }
            Self::GgufQ3K | Self::GgufQ4K | Self::GgufQ5K | Self::GgufQ6K => {
                ensure!(
                    cols.is_multiple_of(256),
                    "{self} requires cols % 256 == 0, got {cols}"
                );
                ensure!(
                    group_size == 256,
                    "{self} requires synthetic group_size=256, got {group_size}"
                );
                Ok(())
            }
            Self::Dsv4Fp8BlockScaled => Ok(()),
            Self::Dsv4Fp4BlockScaled => {
                ensure!(
                    cols.is_multiple_of(2),
                    "{self} requires cols % 2 == 0, got {cols}"
                );
                Ok(())
            }
            Self::Fp8BlockScaled | Self::Fp8PerShard => Ok(()),
            Self::Fp4E2M1Group => {
                ensure!(group_size > 0, "{self} requires group_size > 0");
                ensure!(
                    cols.is_multiple_of(2),
                    "{self} requires cols % 2 == 0 for packed E2M1, got {cols}"
                );
                ensure!(
                    cols.is_multiple_of(group_size),
                    "{self} requires cols % group_size == 0, got cols={cols}, group_size={group_size}"
                );
                Ok(())
            }
        }
    }
}

impl std::fmt::Display for WeightFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DenseBf16 => f.write_str("dense_bf16"),
            Self::W8A16 => f.write_str("w8a16"),
            Self::W4A16 => f.write_str("w4a16"),
            Self::MarlinW4A8 => f.write_str("marlin_w4a8"),
            Self::W2A16 => f.write_str("w2a16"),
            Self::GgufQ3K => f.write_str("gguf_q3_k"),
            Self::GgufQ4K => f.write_str("gguf_q4_k"),
            Self::GgufQ5K => f.write_str("gguf_q5_k"),
            Self::GgufQ6K => f.write_str("gguf_q6_k"),
            Self::TurboQuant => f.write_str("turboquant"),
            Self::Dsv4Fp8BlockScaled => f.write_str("dsv4_fp8_block_scaled"),
            Self::Dsv4Fp4BlockScaled => f.write_str("dsv4_fp4_block_scaled"),
            Self::Fp8BlockScaled => f.write_str("fp8_block_scaled"),
            Self::Fp8PerShard => f.write_str("fp8_per_shard"),
            Self::Fp4E2M1Group => f.write_str("fp4_e2m1_group"),
        }
    }
}

const DSV4_DEEPGEMM_FP8_SCALE_GRAN_M: usize = 128;
const DSV4_DEEPGEMM_FP8_SCALE_GRAN_K: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Dsv4DeepGemmSourceFormat {
    Fp8 = 0,
    Fp4 = 1,
}

impl Dsv4DeepGemmSourceFormat {
    fn from_weight_format(format: WeightFormat) -> Result<Self> {
        match format {
            WeightFormat::Dsv4Fp8BlockScaled => Ok(Self::Fp8),
            WeightFormat::Dsv4Fp4BlockScaled => Ok(Self::Fp4),
            other => Err(anyhow!(
                "DeepSeek V4 DeepGEMM FP8 cache needs raw DSv4 block-scaled weights, got {other}"
            )),
        }
    }
}

/// Resident FP8 E4M3 weight cache plus FP32 block scales in DeepGEMM's SM90
/// grouped-GEMM source layout.
///
/// `weight` is row-major `[rows, cols]` FP8 bytes. `scales` is contiguous
/// `[ceil(rows/128), ceil(cols/128)]` FP32, matching DeepGEMM's Hopper SFB
/// recipe for m-grouped FP8 GEMM.
pub struct Dsv4Fp8DeepGemmWeightCache {
    pub weight: CudaSlice<u8>,
    pub scales: CudaSlice<f32>,
    pub rows: usize,
    pub cols: usize,
    pub scale_rows: usize,
    pub scale_cols: usize,
}

impl Dsv4Fp8DeepGemmWeightCache {
    pub fn uninit(ctx: &DeviceContext, rows: usize, cols: usize) -> Result<Self> {
        let scale_rows = rows.div_ceil(DSV4_DEEPGEMM_FP8_SCALE_GRAN_M);
        let scale_cols = cols.div_ceil(DSV4_DEEPGEMM_FP8_SCALE_GRAN_K);
        let weight_len = rows.checked_mul(cols).ok_or_else(|| {
            anyhow!(
                "DeepSeek V4 DeepGEMM cache weight size overflow: rows={} cols={}",
                rows,
                cols
            )
        })?;
        let scale_len = scale_rows.checked_mul(scale_cols).ok_or_else(|| {
            anyhow!(
                "DeepSeek V4 DeepGEMM cache scale size overflow: rows={} cols={}",
                scale_rows,
                scale_cols
            )
        })?;
        Ok(Self {
            // SAFETY: both buffers start uninitialized by design — every row is
            // written by `dsv4_fill_fp8_deepgemm_weight_cache` (the only
            // producer) before any DeepGEMM launch reads the cache.
            weight: unsafe { ctx.stream.alloc_traced::<u8>(weight_len)? },
            // SAFETY: see `weight` above; filled before first read.
            scales: unsafe { ctx.stream.alloc_traced::<f32>(scale_len)? },
            rows,
            cols,
            scale_rows,
            scale_cols,
        })
    }

    #[must_use]
    pub fn weight_bytes(&self) -> usize {
        self.rows.saturating_mul(self.cols)
    }

    #[must_use]
    pub fn scale_bytes(&self) -> usize {
        self.scale_rows
            .saturating_mul(self.scale_cols)
            .saturating_mul(std::mem::size_of::<f32>())
    }

    pub fn from_dsv4_weight(ctx: &DeviceContext, weight: &DeviceMatrix) -> Result<Self> {
        let mut cache = Self::uninit(ctx, weight.rows, weight.cols)?;
        cache.fill_from_dsv4_weight(ctx, weight, 0)?;
        Ok(cache)
    }

    pub fn from_dsv4_weight_row_range(
        ctx: &DeviceContext,
        weight: &DeviceMatrix,
        row_start: usize,
        rows: usize,
    ) -> Result<Self> {
        let mut cache = Self::uninit(ctx, rows, weight.cols)?;
        cache.fill_from_dsv4_weight_row_range(ctx, weight, row_start, rows, 0)?;
        Ok(cache)
    }

    pub fn from_dsv4_weight_pair_rows(
        ctx: &DeviceContext,
        first: &DeviceMatrix,
        second: &DeviceMatrix,
    ) -> Result<Self> {
        ensure!(
            first.cols == second.cols,
            "DeepSeek V4 DeepGEMM fused cache needs matching K: first={} second={}",
            first.cols,
            second.cols
        );
        ensure!(
            first.rows.is_multiple_of(DSV4_DEEPGEMM_FP8_SCALE_GRAN_M),
            "DeepSeek V4 DeepGEMM fused cache needs first row count aligned to {}, got {}",
            DSV4_DEEPGEMM_FP8_SCALE_GRAN_M,
            first.rows
        );
        let mut cache = Self::uninit(ctx, first.rows + second.rows, first.cols)?;
        cache.fill_from_dsv4_weight(ctx, first, 0)?;
        cache.fill_from_dsv4_weight(ctx, second, first.rows)?;
        Ok(cache)
    }

    pub fn from_fp8_block_scaled_weight(
        ctx: &DeviceContext,
        weight: &DeviceMatrix,
    ) -> Result<Self> {
        let mut cache = Self::uninit(ctx, weight.rows, weight.cols)?;
        cache.fill_from_fp8_block_scaled_weight(ctx, weight, 0)?;
        Ok(cache)
    }

    pub fn from_fp8_block_scaled_weight_pair_rows(
        ctx: &DeviceContext,
        first: &DeviceMatrix,
        second: &DeviceMatrix,
    ) -> Result<Self> {
        ensure!(
            first.cols == second.cols,
            "DeepGEMM FP8 fused cache needs matching K: first={} second={}",
            first.cols,
            second.cols
        );
        ensure!(
            first.rows.is_multiple_of(DSV4_DEEPGEMM_FP8_SCALE_GRAN_M),
            "DeepGEMM FP8 fused cache needs first row count aligned to {}, got {}",
            DSV4_DEEPGEMM_FP8_SCALE_GRAN_M,
            first.rows
        );
        let mut cache = Self::uninit(ctx, first.rows + second.rows, first.cols)?;
        cache.fill_from_fp8_block_scaled_weight(ctx, first, 0)?;
        cache.fill_from_fp8_block_scaled_weight(ctx, second, first.rows)?;
        Ok(cache)
    }

    pub fn fill_from_dsv4_weight(
        &mut self,
        ctx: &DeviceContext,
        weight: &DeviceMatrix,
        dst_row_offset: usize,
    ) -> Result<()> {
        dsv4_fill_fp8_deepgemm_weight_cache(
            ctx,
            weight,
            self,
            dst_row_offset,
            dst_row_offset / DSV4_DEEPGEMM_FP8_SCALE_GRAN_M,
        )
    }

    pub fn fill_from_dsv4_weight_row_range(
        &mut self,
        ctx: &DeviceContext,
        weight: &DeviceMatrix,
        row_start: usize,
        rows: usize,
        dst_row_offset: usize,
    ) -> Result<()> {
        dsv4_fill_fp8_deepgemm_weight_cache_row_range(
            ctx,
            weight,
            self,
            row_start,
            rows,
            dst_row_offset,
            dst_row_offset / DSV4_DEEPGEMM_FP8_SCALE_GRAN_M,
        )
    }

    pub fn fill_from_fp8_block_scaled_weight(
        &mut self,
        ctx: &DeviceContext,
        weight: &DeviceMatrix,
        dst_row_offset: usize,
    ) -> Result<()> {
        ensure!(
            weight.weight_format == WeightFormat::Fp8BlockScaled,
            "DeepGEMM FP8 cache needs FP8 block-scaled weights, got {}",
            weight.weight_format
        );
        ensure!(
            weight.quant_block_m == DSV4_DEEPGEMM_FP8_SCALE_GRAN_M
                && weight.quant_block_k == DSV4_DEEPGEMM_FP8_SCALE_GRAN_K,
            "DeepGEMM FP8 cache needs 128x128 block scales, got {}x{}",
            weight.quant_block_m,
            weight.quant_block_k
        );
        ensure!(
            weight.cols == self.cols,
            "DeepGEMM FP8 cache K mismatch: source={} cache={}",
            weight.cols,
            self.cols
        );
        ensure!(
            dst_row_offset + weight.rows <= self.rows,
            "DeepGEMM FP8 cache row range overflow: offset={} src={} cache={}",
            dst_row_offset,
            weight.rows,
            self.rows
        );
        ensure!(
            dst_row_offset.is_multiple_of(DSV4_DEEPGEMM_FP8_SCALE_GRAN_M),
            "DeepGEMM FP8 cache row offset must be {}-aligned, got {}",
            DSV4_DEEPGEMM_FP8_SCALE_GRAN_M,
            dst_row_offset
        );
        let src_scale_rows = weight.rows.div_ceil(DSV4_DEEPGEMM_FP8_SCALE_GRAN_M);
        let src_scale_cols = weight.cols.div_ceil(DSV4_DEEPGEMM_FP8_SCALE_GRAN_K);
        ensure!(
            weight.quant_scale_rows == src_scale_rows && weight.quant_scale_cols == src_scale_cols,
            "DeepGEMM FP8 cache scale shape {}x{} != expected {}x{}",
            weight.quant_scale_rows,
            weight.quant_scale_cols,
            src_scale_rows,
            src_scale_cols
        );
        ensure!(
            self.scale_cols == src_scale_cols,
            "DeepGEMM FP8 cache scale K mismatch: source={} cache={}",
            src_scale_cols,
            self.scale_cols
        );
        let dst_scale_row_offset = dst_row_offset / DSV4_DEEPGEMM_FP8_SCALE_GRAN_M;
        ensure!(
            dst_scale_row_offset + src_scale_rows <= self.scale_rows,
            "DeepGEMM FP8 cache scale row overflow: offset={} src={} cache={}",
            dst_scale_row_offset,
            src_scale_rows,
            self.scale_rows
        );

        let src_weight = weight
            .qweight_u8
            .as_ref()
            .ok_or_else(|| anyhow!("DeepGEMM FP8 cache source missing FP8 weight bytes"))?;
        let src_scales = weight
            .scale_f32
            .as_ref()
            .ok_or_else(|| anyhow!("DeepGEMM FP8 cache source missing f32 block scales"))?;
        ensure!(
            src_weight.len() == weight.rows * weight.cols,
            "DeepGEMM FP8 cache source weight len {} != expected {}",
            src_weight.len(),
            weight.rows * weight.cols
        );
        ensure!(
            src_scales.len() == src_scale_rows * src_scale_cols,
            "DeepGEMM FP8 cache source scale len {} != expected {}",
            src_scales.len(),
            src_scale_rows * src_scale_cols
        );

        {
            let src = src_weight.slice(0..src_weight.len());
            let weight_start = dst_row_offset * self.cols;
            let weight_end = weight_start + src_weight.len();
            let mut dst = self.weight.slice_mut(weight_start..weight_end);
            ctx.stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(|e| anyhow!("DeepGEMM FP8 cache weight D2D failed: {e}"))?;
        }
        {
            let src = src_scales.slice(0..src_scales.len());
            let scale_start = dst_scale_row_offset * self.scale_cols;
            let scale_end = scale_start + src_scales.len();
            let mut dst = self.scales.slice_mut(scale_start..scale_end);
            ctx.stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(|e| anyhow!("DeepGEMM FP8 cache scale D2D failed: {e}"))?;
        }
        Ok(())
    }
}

fn dsv4_fill_fp8_deepgemm_weight_cache(
    ctx: &DeviceContext,
    src: &DeviceMatrix,
    dst: &mut Dsv4Fp8DeepGemmWeightCache,
    dst_row_offset: usize,
    dst_scale_row_offset: usize,
) -> Result<()> {
    let source_format = Dsv4DeepGemmSourceFormat::from_weight_format(src.weight_format)?;
    ensure!(
        src.cols == dst.cols,
        "DeepSeek V4 DeepGEMM cache K mismatch: source={} cache={}",
        src.cols,
        dst.cols
    );
    ensure!(
        dst_row_offset + src.rows <= dst.rows,
        "DeepSeek V4 DeepGEMM cache row range overflow: offset={} src={} cache={}",
        dst_row_offset,
        src.rows,
        dst.rows
    );
    ensure!(
        dst_row_offset.is_multiple_of(DSV4_DEEPGEMM_FP8_SCALE_GRAN_M),
        "DeepSeek V4 DeepGEMM cache row offset must be {}-aligned, got {}",
        DSV4_DEEPGEMM_FP8_SCALE_GRAN_M,
        dst_row_offset
    );
    ensure!(
        src.dsv4_scale_rows > 0 && src.dsv4_scale_cols > 0,
        "DeepSeek V4 DeepGEMM cache source needs DSv4 block scales"
    );
    let src_scale_rows = src.rows.div_ceil(DSV4_DEEPGEMM_FP8_SCALE_GRAN_M);
    ensure!(
        dst_scale_row_offset + src_scale_rows <= dst.scale_rows,
        "DeepSeek V4 DeepGEMM cache scale row overflow: offset={} src={} cache={}",
        dst_scale_row_offset,
        src_scale_rows,
        dst.scale_rows
    );

    let qweight = src
        .qweight
        .as_ref()
        .ok_or_else(|| anyhow!("DeepSeek V4 DeepGEMM cache source missing raw weight bytes"))?;
    let src_scales = src
        .dsv4_scales
        .as_ref()
        .ok_or_else(|| anyhow!("DeepSeek V4 DeepGEMM cache source missing block scales"))?;
    let rows_i32 = i32::try_from(src.rows)
        .map_err(|_| anyhow!("DeepSeek V4 DeepGEMM cache rows overflow i32"))?;
    let cols_i32 = i32::try_from(src.cols)
        .map_err(|_| anyhow!("DeepSeek V4 DeepGEMM cache cols overflow i32"))?;
    let scale_rows_i32 = i32::try_from(src.dsv4_scale_rows)
        .map_err(|_| anyhow!("DeepSeek V4 DeepGEMM source scale rows overflow i32"))?;
    let scale_cols_i32 = i32::try_from(src.dsv4_scale_cols)
        .map_err(|_| anyhow!("DeepSeek V4 DeepGEMM source scale cols overflow i32"))?;
    let dst_scale_cols_i32 = i32::try_from(dst.scale_cols)
        .map_err(|_| anyhow!("DeepSeek V4 DeepGEMM cache scale cols overflow i32"))?;
    let (src_ptr, _src_guard) = qweight.device_ptr(&ctx.stream);
    let (src_scale_ptr, _src_scale_guard) = src_scales.device_ptr(&ctx.stream);
    let (dst_weight_ptr, _dst_weight_guard) = dst.weight.device_ptr_mut(&ctx.stream);
    let (dst_scale_ptr, _dst_scale_guard) = dst.scales.device_ptr_mut(&ctx.stream);
    // SAFETY: `dst_row_offset + src.rows <= dst.rows` was ensured above, so the
    // offset stays inside `dst.weight` (`dst.rows * dst.cols` bytes).
    let dst_weight_ptr = unsafe { (dst_weight_ptr as *mut u8).add(dst_row_offset * dst.cols) };
    // SAFETY: `dst_scale_row_offset + src_scale_rows <= dst.scale_rows` was
    // ensured above, so the offset stays inside `dst.scales`.
    let dst_scale_ptr =
        unsafe { (dst_scale_ptr as *mut f32).add(dst_scale_row_offset * dst.scale_cols) };
    // SAFETY: src pointers are live CudaSlices pinned by the `_g*` guards with
    // lengths matching the ensured shapes; the kernel writes `src.rows` weight
    // rows and `src_scale_rows` scale rows at the bounded offsets above,
    // stream-ordered on `ctx.stream`.
    unsafe {
        ffi::dsv4_block_scaled_to_fp8_deepgemm_cuda(
            src_ptr as *const u8,
            src_scale_ptr as *const u8,
            dst_weight_ptr,
            dst_scale_ptr,
            rows_i32,
            cols_i32,
            scale_rows_i32,
            scale_cols_i32,
            dst_scale_cols_i32,
            source_format as i32,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|err| anyhow!("DeepSeek V4 DeepGEMM FP8 cache build failed: {err}"))?;
    }
    Ok(())
}

fn dsv4_fill_fp8_deepgemm_weight_cache_row_range(
    ctx: &DeviceContext,
    src: &DeviceMatrix,
    dst: &mut Dsv4Fp8DeepGemmWeightCache,
    src_row_start: usize,
    src_rows: usize,
    dst_row_offset: usize,
    dst_scale_row_offset: usize,
) -> Result<()> {
    let source_format = Dsv4DeepGemmSourceFormat::from_weight_format(src.weight_format)?;
    ensure!(
        src_rows > 0,
        "DeepSeek V4 DeepGEMM row-range cache needs rows > 0"
    );
    ensure!(
        src_row_start + src_rows <= src.rows,
        "DeepSeek V4 DeepGEMM source row range [{}..{}) exceeds rows {}",
        src_row_start,
        src_row_start + src_rows,
        src.rows
    );
    ensure!(
        src.cols == dst.cols,
        "DeepSeek V4 DeepGEMM row-range cache K mismatch: source={} cache={}",
        src.cols,
        dst.cols
    );
    ensure!(
        dst_row_offset + src_rows <= dst.rows,
        "DeepSeek V4 DeepGEMM row-range cache dst row overflow: offset={} rows={} cache={}",
        dst_row_offset,
        src_rows,
        dst.rows
    );
    ensure!(
        src_row_start.is_multiple_of(DSV4_DEEPGEMM_FP8_SCALE_GRAN_M)
            && src_rows.is_multiple_of(DSV4_DEEPGEMM_FP8_SCALE_GRAN_M)
            && dst_row_offset.is_multiple_of(DSV4_DEEPGEMM_FP8_SCALE_GRAN_M),
        "DeepSeek V4 DeepGEMM row-range cache rows must be {}-aligned (src_start={} rows={} dst_offset={})",
        DSV4_DEEPGEMM_FP8_SCALE_GRAN_M,
        src_row_start,
        src_rows,
        dst_row_offset
    );
    ensure!(
        src.dsv4_scale_rows > 0 && src.dsv4_scale_cols > 0,
        "DeepSeek V4 DeepGEMM row-range cache source needs DSv4 block scales"
    );
    let src_scale_row_start = src_row_start / DSV4_DEEPGEMM_FP8_SCALE_GRAN_M;
    let src_scale_rows = src_rows / DSV4_DEEPGEMM_FP8_SCALE_GRAN_M;
    ensure!(
        src_scale_row_start + src_scale_rows <= src.dsv4_scale_rows,
        "DeepSeek V4 DeepGEMM row-range scale source overflow: start={} rows={} source={}",
        src_scale_row_start,
        src_scale_rows,
        src.dsv4_scale_rows
    );
    ensure!(
        dst_scale_row_offset + src_scale_rows <= dst.scale_rows,
        "DeepSeek V4 DeepGEMM row-range cache scale row overflow: offset={} rows={} cache={}",
        dst_scale_row_offset,
        src_scale_rows,
        dst.scale_rows
    );

    let qweight = src
        .qweight
        .as_ref()
        .ok_or_else(|| anyhow!("DeepSeek V4 DeepGEMM row-range source missing raw weight bytes"))?;
    let src_scales = src
        .dsv4_scales
        .as_ref()
        .ok_or_else(|| anyhow!("DeepSeek V4 DeepGEMM row-range source missing block scales"))?;
    let bytes_per_src_row = match source_format {
        Dsv4DeepGemmSourceFormat::Fp8 => src.cols,
        Dsv4DeepGemmSourceFormat::Fp4 => {
            ensure!(
                src.cols.is_multiple_of(2),
                "DeepSeek V4 FP4 DeepGEMM row-range source cols must be even, got {}",
                src.cols
            );
            src.cols / 2
        }
    };
    ensure!(
        qweight.len() == src.rows * bytes_per_src_row,
        "DeepSeek V4 DeepGEMM row-range source weight len {} != expected {}",
        qweight.len(),
        src.rows * bytes_per_src_row
    );
    ensure!(
        src_scales.len() == src.dsv4_scale_rows * src.dsv4_scale_cols,
        "DeepSeek V4 DeepGEMM row-range source scale len {} != expected {}",
        src_scales.len(),
        src.dsv4_scale_rows * src.dsv4_scale_cols
    );
    let rows_i32 = i32::try_from(src_rows)
        .map_err(|_| anyhow!("DeepSeek V4 DeepGEMM row-range rows overflow i32"))?;
    let cols_i32 = i32::try_from(src.cols)
        .map_err(|_| anyhow!("DeepSeek V4 DeepGEMM row-range cols overflow i32"))?;
    let scale_rows_i32 = i32::try_from(src_scale_rows)
        .map_err(|_| anyhow!("DeepSeek V4 DeepGEMM row-range scale rows overflow i32"))?;
    let scale_cols_i32 = i32::try_from(src.dsv4_scale_cols)
        .map_err(|_| anyhow!("DeepSeek V4 DeepGEMM row-range scale cols overflow i32"))?;
    let dst_scale_cols_i32 = i32::try_from(dst.scale_cols)
        .map_err(|_| anyhow!("DeepSeek V4 DeepGEMM row-range cache scale cols overflow i32"))?;
    let (src_ptr, _src_guard) = qweight.device_ptr(&ctx.stream);
    let (src_scale_ptr, _src_scale_guard) = src_scales.device_ptr(&ctx.stream);
    let (dst_weight_ptr, _dst_weight_guard) = dst.weight.device_ptr_mut(&ctx.stream);
    let (dst_scale_ptr, _dst_scale_guard) = dst.scales.device_ptr_mut(&ctx.stream);
    // SAFETY: `src_row_start + src_rows <= src.rows` and the qweight length
    // check above keep this offset inside the source weight buffer.
    let src_ptr = unsafe { (src_ptr as *const u8).add(src_row_start * bytes_per_src_row) };
    // SAFETY: `src_scale_row_start + src_scale_rows <= src.dsv4_scale_rows` was
    // ensured above, keeping the offset inside the source scale buffer.
    let src_scale_ptr =
        unsafe { (src_scale_ptr as *const u8).add(src_scale_row_start * src.dsv4_scale_cols) };
    // SAFETY: `dst_row_offset + src_rows <= dst.rows` was ensured above, so the
    // offset stays inside `dst.weight`.
    let dst_weight_ptr = unsafe { (dst_weight_ptr as *mut u8).add(dst_row_offset * dst.cols) };
    // SAFETY: `dst_scale_row_offset + src_scale_rows <= dst.scale_rows` was
    // ensured above, so the offset stays inside `dst.scales`.
    let dst_scale_ptr =
        unsafe { (dst_scale_ptr as *mut f32).add(dst_scale_row_offset * dst.scale_cols) };
    // SAFETY: all four pointers were bounds-offset above from live CudaSlices
    // pinned by the `_g*` guards; the kernel touches `src_rows` weight rows and
    // `src_scale_rows` scale rows only, stream-ordered on `ctx.stream`.
    unsafe {
        ffi::dsv4_block_scaled_to_fp8_deepgemm_cuda(
            src_ptr,
            src_scale_ptr,
            dst_weight_ptr,
            dst_scale_ptr,
            rows_i32,
            cols_i32,
            scale_rows_i32,
            scale_cols_i32,
            dst_scale_cols_i32,
            source_format as i32,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|err| anyhow!("DeepSeek V4 DeepGEMM FP8 row-range cache build failed: {err}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_quant_formats_require_group_aligned_k() {
        assert!(WeightFormat::W4A16.validate_shape(64, 4096, 128).is_ok());
        assert!(WeightFormat::W4A16.validate_shape(64, 4097, 128).is_err());
        assert!(WeightFormat::W8A16.validate_shape(64, 4096, 0).is_err());
    }

    #[test]
    fn gguf_k_formats_require_256_wide_superblocks() {
        assert!(WeightFormat::GgufQ4K.validate_shape(64, 4096, 256).is_ok());
        assert!(WeightFormat::GgufQ4K.validate_shape(64, 4096, 128).is_err());
        assert!(WeightFormat::GgufQ4K.validate_shape(64, 4100, 256).is_err());
    }

    #[test]
    fn resident_quant_abi_formats_validate_shapes() {
        assert!(
            WeightFormat::Fp8BlockScaled
                .validate_shape(512, 2048, 0)
                .is_ok()
        );
        assert!(
            WeightFormat::Fp8PerShard
                .validate_shape(512, 2048, 0)
                .is_ok()
        );
        assert!(
            WeightFormat::Fp4E2M1Group
                .validate_shape(512, 2048, 16)
                .is_ok()
        );
        assert!(
            WeightFormat::Fp4E2M1Group
                .validate_shape(512, 2049, 16)
                .is_err()
        );
        assert!(
            WeightFormat::Fp4E2M1Group
                .validate_shape(512, 2048, 0)
                .is_err()
        );
    }
}
