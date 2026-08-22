//! 2D device matrix: loading, quant packing, and host offload/reload.

use anyhow::{Result, anyhow, bail, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr, ValidAsZeroBits};
use half::{bf16, f16};

use super::{DeviceContext, WeightFormat, bf16_safetensor_host_slice, e4m3_to_f32};
use crate::ffi;

/// 2D device tensor (matrix) — stored in row-major order as bf16 unless
/// `weight_format` names an explicit packed layout.
pub struct DeviceMatrix {
    pub data: CudaSlice<bf16>,
    pub rows: usize,
    pub cols: usize,
    pub weight_format: WeightFormat,
    /// INT8 quantized weights (if quantized). When set, `data` is unused.
    pub qweight: Option<CudaSlice<i8>>,
    /// ABI-generic unsigned quantized weights (FP8 bytes or packed FP4 bytes).
    pub qweight_u8: Option<CudaSlice<u8>>,
    /// Merge-requant: pristine FP8 base; `qweight_u8`/`scale_f32` then hold merged bytes.
    pub pristine_fp8: Option<(CudaSlice<u8>, CudaSlice<f32>)>,
    /// Per-group bf16 scales for quantized weights. Shape: [rows, cols/group_size].
    pub qscales: Option<CudaSlice<bf16>>,
    /// ABI-generic FP8 E4M3 scale bytes.
    pub qscale_fp8: Option<CudaSlice<u8>>,
    /// ABI-generic direct f32 scale buffer.
    pub scale_f32: Option<CudaSlice<f32>>,
    /// ABI-generic secondary f32 scale buffer (activation metadata in v1).
    pub scale2_f32: Option<CudaSlice<f32>>,
    pub quant_scale_rows: usize,
    pub quant_scale_cols: usize,
    pub quant_block_m: usize,
    pub quant_block_k: usize,
    /// DeepSeek V4 block scales encoded as raw FP8 E8M0 bytes.
    pub dsv4_scales: Option<CudaSlice<u8>>,
    pub dsv4_scale_rows: usize,
    pub dsv4_scale_cols: usize,
    /// Quantization group size (0 = not quantized).
    pub group_size: usize,
    /// Marlin-repacked INT4 weights for prefill GEMM (None if not W4 or repack failed).
    pub marlin_packed: Option<CudaSlice<u8>>,
    /// FP16 scales in Marlin layout [K/group_size, N] (transposed from qscales).
    pub marlin_scales: Option<CudaSlice<u16>>,
    /// Per-128x128-block power of two for the NVFP4 DeepGEMM prefill arm,
    /// `[ceil(rows/128) + 1, ceil(cols/128)]` f32. Its presence is what routes
    /// a weight to that arm at prefill M.
    pub fp4_deepgemm_sfb: Option<CudaSlice<f32>>,
    /// `scale_factor * 128`, the per-tensor power of two `repack_for_marlin_fp4`
    /// multiplied into each stored S0E5M3 group scale. Nothing else records it —
    /// the Marlin global scale divides it out again — and every reader of the
    /// scale tail has to undo it. 1.0 for every other format.
    /// Reciprocal of the lift `repack_for_marlin_fp4` folds into the S0E5M3
    /// scale tail. Stored inverted because every reader multiplies by it, and a
    /// power of two inverts exactly.
    pub fp4_marlin_scale_lift_inv: f32,
    /// Whether the loader cleared this weight for the per-channel FP8 DeepGEMM
    /// prefill arm. The FP4 twin is `fp4_deepgemm_sfb`'s presence; per-channel
    /// FP8 needs no per-weight buffer, so the decision has to be carried
    /// explicitly. False keeps every M on Marlin — the output head sets it that
    /// way so its logits do not change precision with prompt length.
    pub fp8_deepgemm_prefill: bool,
    /// TQ packed indices [rows, packed_cols] u8.
    /// 3-bit uses 4-bit nibble packing (2 per byte), 2-bit uses 4 per byte.
    pub tq_packed: Option<CudaSlice<u8>>,
    /// TQ per-group f16 norms `[rows, cols/group_size]`, stored as u16 on device.
    pub tq_scales: Option<CudaSlice<u16>>,
    /// TQ Hadamard signs `[cols]` i8 (+1/-1), shared across rows.
    pub tq_signs: Option<CudaSlice<i8>>,
    /// TQ Lloyd-Max centroids `[2^bits]` f32, shared across all layers.
    pub tq_centroids: Option<CudaSlice<f32>>,
    /// TQ bit width (2, 3, or 4). 0 = not TQ.
    pub tq_bits: u8,
}

/// Host-resident snapshot of an `Option<CudaSlice<T>>` weight buffer, used by
/// the OPD engine time-share offload to hold idle weights in CPU RAM while the
/// device VRAM is freed. `None` means the source buffer was absent (the buffer
/// stays absent on reload).
type OptHostBuf<T> = Option<Vec<T>>;

/// Host-resident snapshot of every device buffer in a [`DeviceMatrix`].
///
/// Captures the full quant-format-agnostic set of side tensors (dense bf16,
/// INT8/INT4 qweight + scales, Marlin packed/scales, hybrid W4A8/W4-FP8
/// sidecars, TurboQuant packed storage) so offload→reload is bit-exact for any
/// weight format. The scalar shape/format fields are restored from the live
/// `DeviceMatrix` they were detached from, so this snapshot only carries the
/// raw buffer bytes.
pub struct HostMatrixSnapshot {
    data: Vec<bf16>,
    qweight: OptHostBuf<i8>,
    qweight_u8: OptHostBuf<u8>,
    pristine_fp8_qweight: OptHostBuf<u8>,
    pristine_fp8_scales: OptHostBuf<f32>,
    qscales: OptHostBuf<bf16>,
    qscale_fp8: OptHostBuf<u8>,
    scale_f32: OptHostBuf<f32>,
    scale2_f32: OptHostBuf<f32>,
    dsv4_scales: OptHostBuf<u8>,
    marlin_packed: OptHostBuf<u8>,
    marlin_scales: OptHostBuf<u16>,
    fp4_deepgemm_sfb: OptHostBuf<f32>,
    tq_packed: OptHostBuf<u8>,
    tq_scales: OptHostBuf<u16>,
    tq_signs: OptHostBuf<i8>,
    tq_centroids: OptHostBuf<f32>,
    /// Total device bytes this snapshot freed when captured (for accounting).
    freed_bytes: usize,
}

impl HostMatrixSnapshot {
    /// Total device VRAM (bytes) freed by capturing this snapshot.
    #[must_use]
    pub fn freed_bytes(&self) -> usize {
        self.freed_bytes
    }
}

fn snapshot_opt_slice<T: DeviceRepr + Clone>(
    ctx: &DeviceContext,
    src: &Option<CudaSlice<T>>,
    freed: &mut usize,
) -> Result<OptHostBuf<T>> {
    match src {
        Some(slice) => {
            let host = ctx
                .stream
                .clone_dtoh(slice)
                .map_err(|e| anyhow!("offload D2H copy failed: {e}"))?;
            *freed += host.len() * std::mem::size_of::<T>();
            Ok(Some(host))
        }
        None => Ok(None),
    }
}

fn restore_opt_slice<T: DeviceRepr>(
    ctx: &DeviceContext,
    host: &Option<Vec<T>>,
) -> Result<Option<CudaSlice<T>>> {
    match host {
        Some(data) => Ok(Some(
            ctx.stream
                .clone_htod(data.as_slice())
                .map_err(|e| anyhow!("reload H2D copy failed: {e}"))?,
        )),
        None => Ok(None),
    }
}

/// Move a raw `CudaSlice<T>` to host RAM, replacing it with a 1-element
/// placeholder and freeing the VRAM. Returns the host copy and bytes freed.
/// Used for the model's bare `CudaSlice<f32>` weight fields (e.g. SSM A_log,
/// norm weights) that are not wrapped in `DeviceVec`/`DeviceMatrix`.
pub fn offload_raw_slice<T: DeviceRepr + Clone + ValidAsZeroBits>(
    ctx: &DeviceContext,
    slice: &mut CudaSlice<T>,
) -> Result<(Vec<T>, usize)> {
    let host = ctx
        .stream
        .clone_dtoh(slice)
        .map_err(|e| anyhow!("offload D2H copy (raw slice) failed: {e}"))?;
    let freed = host.len() * std::mem::size_of::<T>();
    ctx.sync()?;
    *slice = ctx
        .stream
        .alloc_zeros::<T>(1)
        .map_err(|e| anyhow!("offload raw-slice placeholder alloc failed: {e}"))?;
    Ok((host, freed))
}

pub fn reload_raw_slice<T: DeviceRepr>(
    ctx: &DeviceContext,
    slice: &mut CudaSlice<T>,
    host: &[T],
) -> Result<()> {
    *slice = ctx
        .stream
        .clone_htod(host)
        .map_err(|e| anyhow!("reload H2D copy (raw slice) failed: {e}"))?;
    ctx.sync()?;
    Ok(())
}

impl DeviceMatrix {
    /// Raw device pointer to the dense BF16 `data` buffer as a `u64`.
    ///
    /// Used to build the per-expert weight-pointer table (`*const u64`) the MoE
    /// grouped-GEMM kernels consume: each entry is one expert's `DeviceMatrix`
    /// device pointer. Only valid for the dense BF16 path (`data` populated);
    /// quantized formats store weights in the side buffers, not `data`.
    pub fn device_ptr(&self, ctx: &DeviceContext) -> u64 {
        use cudarc::driver::DevicePtr;
        let (ptr, _sync) = self.data.device_ptr(&ctx.stream);
        ptr
    }

    /// Resident FP8 block-scaled weight pointers for read-only foreign borrow
    /// (train-infer weight sharing, `--share-frozen-base`).
    ///
    /// Returns `Some((qweight_u8_ptr, scale_f32_ptr, rows, cols, block_m,
    /// block_k))` ONLY when this matrix is stored as block-scaled FP8 with both
    /// the `qweight_u8` byte buffer and the `scale_f32` scale buffer resident
    /// (the layout `from_fp8_block_scaled` produces). Any other weight format —
    /// or a matrix currently offloaded (placeholder buffers) — yields `None`.
    ///
    /// The returned `u64`s are raw `CUdeviceptr`s into THIS matrix's resident
    /// VRAM; the borrower must keep this `DeviceMatrix` resident (no offload,
    /// no LoRA re-merge replacing the buffers) for the borrow's lifetime.
    /// Pristine FP8 pair once merge-requant split it out, else the live slots.
    ///
    /// The live-slot fallback is format-gated: an `Fp4E2M1Group` matrix also
    /// fills `qweight_u8` (packed nibbles, half the byte count) and `scale_f32`
    /// (a 1-element global scale), and reading those as an FP8 block-scaled pair
    /// walks off the end of both buffers. `pristine_fp8` needs no check — only
    /// `requant_merged_matrix` writes it, and only on the FP8 path.
    pub fn merge_base_fp8(&self) -> Option<(&CudaSlice<u8>, &CudaSlice<f32>)> {
        if let Some((qw, sc)) = self.pristine_fp8.as_ref() {
            return Some((qw, sc));
        }
        if self.weight_format != WeightFormat::Fp8BlockScaled {
            return None;
        }
        self.qweight_u8.as_ref().zip(self.scale_f32.as_ref())
    }

    pub fn fp8_block_scaled_ptrs(
        &self,
        ctx: &DeviceContext,
    ) -> Option<(u64, u64, usize, usize, usize, usize)> {
        use cudarc::driver::DevicePtr;
        if self.weight_format != WeightFormat::Fp8BlockScaled {
            return None;
        }
        let (qweight, scales) = self.merge_base_fp8()?;
        let (wptr, _wsync) = qweight.device_ptr(&ctx.stream);
        let (sptr, _ssync) = scales.device_ptr(&ctx.stream);
        Some((
            wptr,
            sptr,
            self.rows,
            self.cols,
            self.quant_block_m,
            self.quant_block_k,
        ))
    }

    /// Copy the dense BF16 `data` buffer to host as f32 (for testing/training).
    ///
    /// Mirrors [`DeviceVec::to_host`]; only reads the dense `data` field, not
    /// quantized side buffers.
    pub fn to_host(&self, ctx: &DeviceContext) -> Result<Vec<f32>> {
        let host_f16 = ctx
            .stream
            .clone_dtoh(&self.data)
            .map_err(|e| anyhow!("D2H copy failed: {}", e))?;
        ctx.sync()?;
        Ok(host_f16.iter().map(|x| x.to_f32()).collect())
    }

    /// Move every device weight buffer to host RAM and free the VRAM.
    ///
    /// Returns a [`HostMatrixSnapshot`] the caller holds until reload. The
    /// live device buffers are replaced with 1-element placeholders so the
    /// struct stays valid (it must not be forwarded through while offloaded).
    /// Format-agnostic: handles dense, INT8/INT4, Marlin, hybrid W4, and TQ.
    pub fn offload_to_host(&mut self, ctx: &DeviceContext) -> Result<HostMatrixSnapshot> {
        let mut freed = 0usize;
        let data = ctx
            .stream
            .clone_dtoh(&self.data)
            .map_err(|e| anyhow!("offload D2H copy (data) failed: {e}"))?;
        freed += data.len() * std::mem::size_of::<bf16>();

        let (pristine_qweight, pristine_scales) = match self.pristine_fp8.take() {
            Some((qw, sc)) => (Some(qw), Some(sc)),
            None => (None, None),
        };
        let snapshot = HostMatrixSnapshot {
            data,
            qweight: snapshot_opt_slice(ctx, &self.qweight, &mut freed)?,
            qweight_u8: snapshot_opt_slice(ctx, &self.qweight_u8, &mut freed)?,
            pristine_fp8_qweight: snapshot_opt_slice(ctx, &pristine_qweight, &mut freed)?,
            pristine_fp8_scales: snapshot_opt_slice(ctx, &pristine_scales, &mut freed)?,
            qscales: snapshot_opt_slice(ctx, &self.qscales, &mut freed)?,
            qscale_fp8: snapshot_opt_slice(ctx, &self.qscale_fp8, &mut freed)?,
            scale_f32: snapshot_opt_slice(ctx, &self.scale_f32, &mut freed)?,
            scale2_f32: snapshot_opt_slice(ctx, &self.scale2_f32, &mut freed)?,
            dsv4_scales: snapshot_opt_slice(ctx, &self.dsv4_scales, &mut freed)?,
            marlin_packed: snapshot_opt_slice(ctx, &self.marlin_packed, &mut freed)?,
            marlin_scales: snapshot_opt_slice(ctx, &self.marlin_scales, &mut freed)?,
            fp4_deepgemm_sfb: snapshot_opt_slice(ctx, &self.fp4_deepgemm_sfb, &mut freed)?,
            tq_packed: snapshot_opt_slice(ctx, &self.tq_packed, &mut freed)?,
            tq_scales: snapshot_opt_slice(ctx, &self.tq_scales, &mut freed)?,
            tq_signs: snapshot_opt_slice(ctx, &self.tq_signs, &mut freed)?,
            tq_centroids: snapshot_opt_slice(ctx, &self.tq_centroids, &mut freed)?,
            freed_bytes: 0,
        };
        ctx.sync()?;

        // Drop the device buffers (return blocks to the async pool). Replace
        // `data` with a 1-element placeholder so the struct stays well-formed.
        let placeholder = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("offload placeholder alloc failed: {e}"))?;
        self.data = placeholder;
        self.qweight = None;
        self.qweight_u8 = None;
        self.qscales = None;
        self.qscale_fp8 = None;
        self.scale_f32 = None;
        self.scale2_f32 = None;
        self.dsv4_scales = None;
        self.marlin_packed = None;
        self.marlin_scales = None;
        self.fp4_deepgemm_sfb = None;
        self.tq_packed = None;
        self.tq_scales = None;
        self.tq_signs = None;
        self.tq_centroids = None;

        Ok(HostMatrixSnapshot {
            freed_bytes: freed,
            ..snapshot
        })
    }

    pub fn reload_from_host(
        &mut self,
        ctx: &DeviceContext,
        snapshot: &HostMatrixSnapshot,
    ) -> Result<()> {
        self.data = ctx
            .stream
            .clone_htod(snapshot.data.as_slice())
            .map_err(|e| anyhow!("reload H2D copy (data) failed: {e}"))?;
        self.qweight = restore_opt_slice(ctx, &snapshot.qweight)?;
        self.qweight_u8 = restore_opt_slice(ctx, &snapshot.qweight_u8)?;
        self.pristine_fp8 = match (
            restore_opt_slice(ctx, &snapshot.pristine_fp8_qweight)?,
            restore_opt_slice(ctx, &snapshot.pristine_fp8_scales)?,
        ) {
            (Some(qw), Some(sc)) => Some((qw, sc)),
            _ => None,
        };
        self.qscales = restore_opt_slice(ctx, &snapshot.qscales)?;
        self.qscale_fp8 = restore_opt_slice(ctx, &snapshot.qscale_fp8)?;
        self.scale_f32 = restore_opt_slice(ctx, &snapshot.scale_f32)?;
        self.scale2_f32 = restore_opt_slice(ctx, &snapshot.scale2_f32)?;
        self.dsv4_scales = restore_opt_slice(ctx, &snapshot.dsv4_scales)?;
        self.marlin_packed = restore_opt_slice(ctx, &snapshot.marlin_packed)?;
        self.marlin_scales = restore_opt_slice(ctx, &snapshot.marlin_scales)?;
        self.fp4_deepgemm_sfb = restore_opt_slice(ctx, &snapshot.fp4_deepgemm_sfb)?;
        self.tq_packed = restore_opt_slice(ctx, &snapshot.tq_packed)?;
        self.tq_scales = restore_opt_slice(ctx, &snapshot.tq_scales)?;
        self.tq_signs = restore_opt_slice(ctx, &snapshot.tq_signs)?;
        self.tq_centroids = restore_opt_slice(ctx, &snapshot.tq_centroids)?;
        ctx.sync()?;
        Ok(())
    }

    pub fn from_host(ctx: &DeviceContext, data: &[bf16], rows: usize, cols: usize) -> Result<Self> {
        assert_eq!(data.len(), rows * cols);
        let gpu_data = ctx
            .stream
            .clone_htod(data)
            .map_err(|e| anyhow!("H2D copy failed: {}", e))?;
        Ok(Self {
            data: gpu_data,
            rows,
            cols,
            weight_format: WeightFormat::DenseBf16,
            qweight: None,
            qweight_u8: None,
            pristine_fp8: None,
            qscales: None,
            qscale_fp8: None,
            scale_f32: None,
            scale2_f32: None,
            quant_scale_rows: 0,
            quant_scale_cols: 0,
            quant_block_m: 0,
            quant_block_k: 0,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            fp4_deepgemm_sfb: None,
            fp4_marlin_scale_lift_inv: 1.0,
            fp8_deepgemm_prefill: false,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    pub fn from_quantized_int8(
        ctx: &DeviceContext,
        qweight_data: &[i8],
        scales_data: &[bf16],
        rows: usize,
        cols: usize,
        group_size: usize,
    ) -> Result<Self> {
        WeightFormat::W8A16.validate_shape(rows, cols, group_size)?;
        ensure!(qweight_data.len() == rows * cols);
        let num_groups = cols / group_size;
        ensure!(scales_data.len() == rows * num_groups);

        let qw = ctx
            .stream
            .clone_htod(qweight_data)
            .map_err(|e| anyhow!("H2D qweight failed: {}", e))?;
        let qs = ctx
            .stream
            .clone_htod(scales_data)
            .map_err(|e| anyhow!("H2D scales failed: {}", e))?;
        // Allocate dummy bf16 data (1 element, unused)
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {}", e))?;
        Ok(Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::W8A16,
            qweight: Some(qw),
            qweight_u8: None,
            pristine_fp8: None,
            qscales: Some(qs),
            qscale_fp8: None,
            scale_f32: None,
            scale2_f32: None,
            quant_scale_rows: 0,
            quant_scale_cols: 0,
            quant_block_m: 0,
            quant_block_k: 0,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size,
            marlin_packed: None,
            marlin_scales: None,
            fp4_deepgemm_sfb: None,
            fp4_marlin_scale_lift_inv: 1.0,
            fp8_deepgemm_prefill: false,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    /// All-default matrix around a dense `data` buffer; `fuse_rows` arms then
    /// set their format-specific fields.
    fn from_parts_dense(data: CudaSlice<bf16>, rows: usize, cols: usize) -> Self {
        Self {
            data,
            rows,
            cols,
            weight_format: WeightFormat::DenseBf16,
            qweight: None,
            qweight_u8: None,
            pristine_fp8: None,
            qscales: None,
            qscale_fp8: None,
            scale_f32: None,
            scale2_f32: None,
            quant_scale_rows: 0,
            quant_scale_cols: 0,
            quant_block_m: 0,
            quant_block_k: 0,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            fp4_deepgemm_sfb: None,
            fp4_marlin_scale_lift_inv: 1.0,
            fp8_deepgemm_prefill: false,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        }
    }

    /// True once a Marlin repack has released the pre-repack bytes.
    /// [`Self::merge_base_fp8`] returns `None` both for a matrix that was never
    /// block-scaled FP8 and for one whose source is gone, and a caller that
    /// reads the second as the first merges into a buffer the GEMM no longer
    /// reads. Ask this before treating `None` as "not quantised".
    pub fn quant_source_freed(&self) -> bool {
        self.marlin_packed.is_some()
            && self.qweight_u8.is_none()
            && matches!(
                self.weight_format,
                WeightFormat::Fp4E2M1Group | WeightFormat::Fp8BlockScaled
            )
    }

    /// Row-concatenate two weight matrices (`[a; b]` along output rows) so one
    /// GEMM serves both projections — the decode launch-count lever. Formats
    /// covered: DenseBf16 (`data`), pre-repack W8A16 (`qweight`+`qscales`;
    /// fuse BEFORE `repack_for_marlin_w8a16` so the fused matrix repacks and
    /// frees its INT8 source once), Fp8BlockScaled (`qweight_u8`+`scale_f32`,
    /// needs `a.rows % block_m == 0` so the scale grids stack cleanly).
    pub fn fuse_rows(ctx: &DeviceContext, a: &DeviceMatrix, b: &DeviceMatrix) -> Result<Self> {
        ensure!(
            a.weight_format == b.weight_format && a.cols == b.cols,
            "fuse_rows needs matching format/K: {} [{}x{}] vs {} [{}x{}]",
            a.weight_format,
            a.rows,
            a.cols,
            b.weight_format,
            b.rows,
            b.cols
        );
        fn concat<T: DeviceRepr + ValidAsZeroBits>(
            ctx: &DeviceContext,
            x: &CudaSlice<T>,
            y: &CudaSlice<T>,
        ) -> Result<CudaSlice<T>> {
            let mut out = ctx
                .stream
                .alloc_zeros::<T>(x.len() + y.len())
                .map_err(|e| anyhow!("fuse_rows alloc failed: {e}"))?;
            {
                let src = x.slice(0..x.len());
                let mut dst = out.slice_mut(0..x.len());
                ctx.stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(|e| anyhow!("fuse_rows D2D (first) failed: {e}"))?;
            }
            {
                let src = y.slice(0..y.len());
                let mut dst = out.slice_mut(x.len()..x.len() + y.len());
                ctx.stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(|e| anyhow!("fuse_rows D2D (second) failed: {e}"))?;
            }
            Ok(out)
        }
        let rows = a.rows + b.rows;
        let mut fused = match a.weight_format {
            WeightFormat::DenseBf16 => {
                ensure!(
                    a.data.len() == a.rows * a.cols && b.data.len() == b.rows * b.cols,
                    "fuse_rows dense data len mismatch"
                );
                let mut m = Self::from_parts_dense(concat(ctx, &a.data, &b.data)?, rows, a.cols);
                m.weight_format = WeightFormat::DenseBf16;
                m
            }
            WeightFormat::W8A16 => {
                ensure!(
                    a.group_size == b.group_size && a.marlin_packed.is_none(),
                    "fuse_rows W8A16 needs matching group_size and pre-repack sources"
                );
                let qa = a
                    .qweight
                    .as_ref()
                    .ok_or_else(|| anyhow!("a missing qweight"))?;
                let qb = b
                    .qweight
                    .as_ref()
                    .ok_or_else(|| anyhow!("b missing qweight"))?;
                let sa = a
                    .qscales
                    .as_ref()
                    .ok_or_else(|| anyhow!("a missing qscales"))?;
                let sb = b
                    .qscales
                    .as_ref()
                    .ok_or_else(|| anyhow!("b missing qscales"))?;
                let mut m = Self::from_parts_dense(
                    ctx.stream
                        .alloc_zeros::<bf16>(1)
                        .map_err(|e| anyhow!("fuse_rows dummy alloc failed: {e}"))?,
                    rows,
                    a.cols,
                );
                m.weight_format = WeightFormat::W8A16;
                m.qweight = Some(concat(ctx, qa, qb)?);
                m.qscales = Some(concat(ctx, sa, sb)?);
                m.group_size = a.group_size;
                m
            }
            WeightFormat::W4A16 => {
                ensure!(
                    a.group_size == b.group_size && a.marlin_packed.is_none(),
                    "fuse_rows W4A16 needs matching group_size and pre-repack sources"
                );
                let qa = a
                    .qweight
                    .as_ref()
                    .ok_or_else(|| anyhow!("a missing qweight"))?;
                let qb = b
                    .qweight
                    .as_ref()
                    .ok_or_else(|| anyhow!("b missing qweight"))?;
                let sa = a
                    .qscales
                    .as_ref()
                    .ok_or_else(|| anyhow!("a missing qscales"))?;
                let sb = b
                    .qscales
                    .as_ref()
                    .ok_or_else(|| anyhow!("b missing qscales"))?;
                let mut m = Self::from_parts_dense(
                    ctx.stream
                        .alloc_zeros::<bf16>(1)
                        .map_err(|e| anyhow!("fuse_rows dummy alloc failed: {e}"))?,
                    rows,
                    a.cols,
                );
                m.weight_format = WeightFormat::W4A16;
                m.qweight = Some(concat(ctx, qa, qb)?);
                m.qscales = Some(concat(ctx, sa, sb)?);
                m.group_size = a.group_size;
                m
            }
            WeightFormat::Fp8BlockScaled => {
                ensure!(
                    a.quant_block_m == b.quant_block_m
                        && a.quant_block_k == b.quant_block_k
                        && a.quant_block_m > 0
                        && a.rows.is_multiple_of(a.quant_block_m),
                    "fuse_rows FP8 needs matching blocks and a.rows % block_m == 0 \
                     (block {}x{}, a.rows {})",
                    a.quant_block_m,
                    a.quant_block_k,
                    a.rows
                );
                ensure!(
                    a.quant_scale_cols == b.quant_scale_cols,
                    "fuse_rows FP8 scale col mismatch"
                );
                // The repack releases the source, so fusing has to come first.
                ensure!(
                    a.marlin_packed.is_none() && b.marlin_packed.is_none(),
                    "fuse_rows FP8 must run before the Marlin repack"
                );
                let qa = a
                    .qweight_u8
                    .as_ref()
                    .ok_or_else(|| anyhow!("a missing qweight_u8"))?;
                let qb = b
                    .qweight_u8
                    .as_ref()
                    .ok_or_else(|| anyhow!("b missing qweight_u8"))?;
                let sa = a
                    .scale_f32
                    .as_ref()
                    .ok_or_else(|| anyhow!("a missing scale_f32"))?;
                let sb = b
                    .scale_f32
                    .as_ref()
                    .ok_or_else(|| anyhow!("b missing scale_f32"))?;
                let mut m = Self::from_parts_dense(
                    ctx.stream
                        .alloc_zeros::<bf16>(1)
                        .map_err(|e| anyhow!("fuse_rows dummy alloc failed: {e}"))?,
                    rows,
                    a.cols,
                );
                m.weight_format = WeightFormat::Fp8BlockScaled;
                m.qweight_u8 = Some(concat(ctx, qa, qb)?);
                m.scale_f32 = Some(concat(ctx, sa, sb)?);
                m.quant_scale_rows = a.quant_scale_rows + b.quant_scale_rows;
                m.quant_scale_cols = a.quant_scale_cols;
                m.quant_block_m = a.quant_block_m;
                m.quant_block_k = a.quant_block_k;
                m
            }
            WeightFormat::Fp4E2M1Group => {
                ensure!(
                    a.group_size == b.group_size && a.cols == b.cols,
                    "fuse_rows FP4 group shape mismatch"
                );
                // The repack releases the source, so fusing has to come first.
                ensure!(
                    a.marlin_packed.is_none() && b.marlin_packed.is_none(),
                    "fuse_rows FP4 must run before the Marlin repack"
                );
                let qa = a
                    .qweight_u8
                    .as_ref()
                    .ok_or_else(|| anyhow!("a missing qweight_u8"))?;
                let qb = b
                    .qweight_u8
                    .as_ref()
                    .ok_or_else(|| anyhow!("b missing qweight_u8"))?;
                let qsa = a
                    .qscale_fp8
                    .as_ref()
                    .ok_or_else(|| anyhow!("a missing qscale_fp8"))?;
                let qsb = b
                    .qscale_fp8
                    .as_ref()
                    .ok_or_else(|| anyhow!("b missing qscale_fp8"))?;
                // Global weight/input scales are per-tensor scalars; both parts
                // share the same value in practice (same quant config group).
                let mut m = Self::from_parts_dense(
                    ctx.stream
                        .alloc_zeros::<bf16>(1)
                        .map_err(|e| anyhow!("fuse_rows dummy alloc failed: {e}"))?,
                    rows,
                    a.cols,
                );
                m.weight_format = WeightFormat::Fp4E2M1Group;
                m.qweight_u8 = Some(concat(ctx, qa, qb)?);
                m.qscale_fp8 = Some(concat(ctx, qsa, qsb)?);
                m.scale_f32 = a.scale_f32.clone();
                m.scale2_f32 = a.scale2_f32.clone();
                m.quant_scale_rows = a.quant_scale_rows + b.quant_scale_rows;
                m.quant_scale_cols = a.quant_scale_cols;
                m.quant_block_m = a.quant_block_m;
                m.quant_block_k = a.quant_block_k;
                m.group_size = a.group_size;
                m
            }
            other => bail!("fuse_rows unsupported for weight format {other}"),
        };
        fused.rows = rows;
        Ok(fused)
    }

    /// Unpacks INT4 → INT8 at load time for the W8 kernel.
    /// TODO: integrate Marlin kernel for native W4 prefill, AWQ-style GEMV for decode.
    pub fn from_quantized_int4(
        ctx: &DeviceContext,
        packed_data: &[u8],
        scales_data: &[bf16],
        rows: usize,
        cols: usize,
        group_size: usize,
    ) -> Result<Self> {
        WeightFormat::W4A16.validate_shape(rows, cols, group_size)?;
        ensure!(
            cols.is_multiple_of(2),
            "W4A16 requires cols % 2 == 0, got {cols}"
        );
        ensure!(packed_data.len() == rows * cols / 2);
        let num_groups = cols / group_size;
        ensure!(scales_data.len() == rows * num_groups);

        // Upload packed INT4 data directly — native W4 kernel handles nibble extraction
        let qw: CudaSlice<i8> = ctx
            .stream
            // SAFETY: same-size reinterpret of `&[u8]` as `&[i8]` (align 1,
            // every bit pattern valid) for the typed H2D upload.
            .clone_htod(unsafe {
                std::slice::from_raw_parts(packed_data.as_ptr().cast::<i8>(), packed_data.len())
            })
            .map_err(|e| anyhow!("H2D qweight int4 failed: {}", e))?;
        let qs = ctx
            .stream
            .clone_htod(scales_data)
            .map_err(|e| anyhow!("H2D scales failed: {}", e))?;
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {}", e))?;
        Ok(Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::W4A16,
            qweight: Some(qw),
            qweight_u8: None,
            pristine_fp8: None,
            qscales: Some(qs),
            qscale_fp8: None,
            scale_f32: None,
            scale2_f32: None,
            quant_scale_rows: 0,
            quant_scale_cols: 0,
            quant_block_m: 0,
            quant_block_k: 0,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size,
            marlin_packed: None,
            marlin_scales: None,
            fp4_deepgemm_sfb: None,
            fp4_marlin_scale_lift_inv: 1.0,
            fp8_deepgemm_prefill: false,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    pub fn from_dsv4_fp8_block_scaled(
        ctx: &DeviceContext,
        weight_bytes: &[u8],
        scale_bytes: &[u8],
        rows: usize,
        cols: usize,
        scale_rows: usize,
        scale_cols: usize,
    ) -> Result<Self> {
        WeightFormat::Dsv4Fp8BlockScaled.validate_shape(rows, cols, 0)?;
        ensure!(
            weight_bytes.len() == rows * cols,
            "DeepSeek V4 FP8 weight bytes {} != expected {} for rows={rows} cols={cols}",
            weight_bytes.len(),
            rows * cols
        );
        ensure!(
            scale_rows > 0 && scale_cols > 0,
            "DeepSeek V4 FP8 scale shape must be non-empty"
        );
        ensure!(
            scale_bytes.len() == scale_rows * scale_cols,
            "DeepSeek V4 FP8 scale bytes {} != expected {}",
            scale_bytes.len(),
            scale_rows * scale_cols
        );

        let qw: CudaSlice<i8> = ctx
            .stream
            // SAFETY: same-size reinterpret of `&[u8]` as `&[i8]` (align 1,
            // every bit pattern valid) for the typed H2D upload.
            .clone_htod(unsafe {
                std::slice::from_raw_parts(weight_bytes.as_ptr().cast::<i8>(), weight_bytes.len())
            })
            .map_err(|e| anyhow!("H2D DeepSeek V4 FP8 weight failed: {e}"))?;
        let scales = ctx
            .stream
            .clone_htod(scale_bytes)
            .map_err(|e| anyhow!("H2D DeepSeek V4 FP8 scales failed: {e}"))?;
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {e}"))?;

        let matrix = Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::Dsv4Fp8BlockScaled,
            qweight: Some(qw),
            qweight_u8: None,
            pristine_fp8: None,
            qscales: None,
            qscale_fp8: None,
            scale_f32: None,
            scale2_f32: None,
            quant_scale_rows: 0,
            quant_scale_cols: 0,
            quant_block_m: 0,
            quant_block_k: 0,
            dsv4_scales: Some(scales),
            dsv4_scale_rows: scale_rows,
            dsv4_scale_cols: scale_cols,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            fp4_deepgemm_sfb: None,
            fp4_marlin_scale_lift_inv: 1.0,
            fp8_deepgemm_prefill: false,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        };
        Ok(matrix)
    }

    /// `cols` is the logical (unpacked) K; the stored buffer holds `rows * cols / 2` bytes (2 FP4 nibbles per byte).
    pub fn from_dsv4_fp4_block_scaled(
        ctx: &DeviceContext,
        weight_bytes: &[u8],
        scale_bytes: &[u8],
        rows: usize,
        cols: usize,
        scale_rows: usize,
        scale_cols: usize,
    ) -> Result<Self> {
        WeightFormat::Dsv4Fp4BlockScaled.validate_shape(rows, cols, 0)?;
        ensure!(
            weight_bytes.len() == rows * (cols / 2),
            "DeepSeek V4 FP4 weight bytes {} != expected {} for rows={rows} cols={cols}",
            weight_bytes.len(),
            rows * (cols / 2)
        );
        ensure!(
            scale_rows > 0 && scale_cols > 0,
            "DeepSeek V4 FP4 scale shape must be non-empty"
        );
        ensure!(
            scale_bytes.len() == scale_rows * scale_cols,
            "DeepSeek V4 FP4 scale bytes {} != expected {}",
            scale_bytes.len(),
            scale_rows * scale_cols
        );

        let qw: CudaSlice<i8> = ctx
            .stream
            // SAFETY: same-size reinterpret of `&[u8]` as `&[i8]` (align 1,
            // every bit pattern valid) for the typed H2D upload.
            .clone_htod(unsafe {
                std::slice::from_raw_parts(weight_bytes.as_ptr().cast::<i8>(), weight_bytes.len())
            })
            .map_err(|e| anyhow!("H2D DeepSeek V4 FP4 weight failed: {e}"))?;
        let scales = ctx
            .stream
            .clone_htod(scale_bytes)
            .map_err(|e| anyhow!("H2D DeepSeek V4 FP4 scales failed: {e}"))?;
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {e}"))?;

        let matrix = Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::Dsv4Fp4BlockScaled,
            qweight: Some(qw),
            qweight_u8: None,
            pristine_fp8: None,
            qscales: None,
            qscale_fp8: None,
            scale_f32: None,
            scale2_f32: None,
            quant_scale_rows: 0,
            quant_scale_cols: 0,
            quant_block_m: 0,
            quant_block_k: 0,
            dsv4_scales: Some(scales),
            dsv4_scale_rows: scale_rows,
            dsv4_scale_cols: scale_cols,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            fp4_deepgemm_sfb: None,
            fp4_marlin_scale_lift_inv: 1.0,
            fp8_deepgemm_prefill: false,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        };
        Ok(matrix)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_fp8_block_scaled(
        ctx: &DeviceContext,
        weight_bytes: &[u8],
        scale_f32: &[f32],
        rows: usize,
        cols: usize,
        block_m: usize,
        block_k: usize,
    ) -> Result<Self> {
        WeightFormat::Fp8BlockScaled.validate_shape(rows, cols, 0)?;
        ensure!(block_m > 0, "Fp8BlockScaled requires block_m > 0");
        ensure!(block_k > 0, "Fp8BlockScaled requires block_k > 0");
        ensure!(
            weight_bytes.len() == rows * cols,
            "FP8 block-scaled weight bytes {} != expected {} for rows={rows} cols={cols}",
            weight_bytes.len(),
            rows * cols
        );
        let scale_rows = rows.div_ceil(block_m);
        let scale_cols = cols.div_ceil(block_k);
        ensure!(
            scale_f32.len() == scale_rows * scale_cols,
            "FP8 block-scaled scales {} != expected {}",
            scale_f32.len(),
            scale_rows * scale_cols
        );

        let qweight = ctx
            .stream
            .clone_htod(weight_bytes)
            .map_err(|e| anyhow!("H2D FP8 block-scaled weight failed: {e}"))?;
        let scales = ctx
            .stream
            .clone_htod(scale_f32)
            .map_err(|e| anyhow!("H2D FP8 block-scaled scales failed: {e}"))?;
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {e}"))?;

        Ok(Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::Fp8BlockScaled,
            qweight: None,
            qweight_u8: Some(qweight),
            pristine_fp8: None,
            qscales: None,
            qscale_fp8: None,
            scale_f32: Some(scales),
            scale2_f32: None,
            quant_scale_rows: scale_rows,
            quant_scale_cols: scale_cols,
            quant_block_m: block_m,
            quant_block_k: block_k,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            fp4_deepgemm_sfb: None,
            fp4_marlin_scale_lift_inv: 1.0,
            fp8_deepgemm_prefill: false,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    pub fn from_fp8_per_shard(
        ctx: &DeviceContext,
        weight_bytes: &[u8],
        scale_f32: &[f32],
        input_scale_f32: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        WeightFormat::Fp8PerShard.validate_shape(rows, cols, 0)?;
        ensure!(
            weight_bytes.len() == rows * cols,
            "FP8 per-shard weight bytes {} != expected {} for rows={rows} cols={cols}",
            weight_bytes.len(),
            rows * cols
        );
        ensure!(
            !scale_f32.is_empty(),
            "FP8 per-shard weight scales must be non-empty"
        );
        ensure!(
            !input_scale_f32.is_empty(),
            "FP8 per-shard input scales must be non-empty"
        );

        let qweight = ctx
            .stream
            .clone_htod(weight_bytes)
            .map_err(|e| anyhow!("H2D FP8 per-shard weight failed: {e}"))?;
        let scales = ctx
            .stream
            .clone_htod(scale_f32)
            .map_err(|e| anyhow!("H2D FP8 per-shard scales failed: {e}"))?;
        let input_scales = ctx
            .stream
            .clone_htod(input_scale_f32)
            .map_err(|e| anyhow!("H2D FP8 per-shard input scales failed: {e}"))?;
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {e}"))?;

        Ok(Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::Fp8PerShard,
            qweight: None,
            qweight_u8: Some(qweight),
            pristine_fp8: None,
            qscales: None,
            qscale_fp8: None,
            scale_f32: Some(scales),
            scale2_f32: Some(input_scales),
            quant_scale_rows: scale_f32.len(),
            quant_scale_cols: 1,
            quant_block_m: 0,
            quant_block_k: 0,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            fp4_deepgemm_sfb: None,
            fp4_marlin_scale_lift_inv: 1.0,
            fp8_deepgemm_prefill: false,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_fp4_e2m1_group(
        ctx: &DeviceContext,
        packed_bytes: &[u8],
        scale_fp8: &[u8],
        global_scale_f32: &[f32],
        input_scale_f32: Option<&[f32]>,
        rows: usize,
        logical_cols: usize,
        group_size: usize,
    ) -> Result<Self> {
        WeightFormat::Fp4E2M1Group.validate_shape(rows, logical_cols, group_size)?;
        ensure!(
            packed_bytes.len() == rows * logical_cols / 2,
            "FP4 E2M1 packed bytes {} != expected {} for rows={rows} cols={logical_cols}",
            packed_bytes.len(),
            rows * logical_cols / 2
        );
        let scale_cols = logical_cols / group_size;
        ensure!(
            scale_fp8.len() == rows * scale_cols,
            "FP4 E2M1 group scales {} != expected {}",
            scale_fp8.len(),
            rows * scale_cols
        );
        ensure!(
            !global_scale_f32.is_empty(),
            "FP4 E2M1 global scale must be non-empty"
        );

        let qweight = ctx
            .stream
            .clone_htod(packed_bytes)
            .map_err(|e| anyhow!("H2D FP4 E2M1 weight failed: {e}"))?;
        let qscale = ctx
            .stream
            .clone_htod(scale_fp8)
            .map_err(|e| anyhow!("H2D FP4 E2M1 group scales failed: {e}"))?;
        let global = ctx
            .stream
            .clone_htod(global_scale_f32)
            .map_err(|e| anyhow!("H2D FP4 E2M1 global scales failed: {e}"))?;
        let input = match input_scale_f32 {
            Some(scales) => Some(
                ctx.stream
                    .clone_htod(scales)
                    .map_err(|e| anyhow!("H2D FP4 E2M1 input scales failed: {e}"))?,
            ),
            None => None,
        };
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {e}"))?;

        Ok(Self {
            data: dummy,
            rows,
            cols: logical_cols,
            weight_format: WeightFormat::Fp4E2M1Group,
            qweight: None,
            qweight_u8: Some(qweight),
            pristine_fp8: None,
            qscales: None,
            qscale_fp8: Some(qscale),
            scale_f32: Some(global),
            scale2_f32: input,
            quant_scale_rows: rows,
            quant_scale_cols: scale_cols,
            quant_block_m: 1,
            quant_block_k: group_size,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size,
            marlin_packed: None,
            marlin_scales: None,
            fp4_deepgemm_sfb: None,
            fp4_marlin_scale_lift_inv: 1.0,
            fp8_deepgemm_prefill: false,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    pub fn is_quantized(&self) -> bool {
        self.weight_format.is_quantized()
            && (self.qweight.is_some()
                || self.qweight_u8.is_some()
                || self.tq_packed.is_some()
                || self.marlin_packed.is_some())
    }

    /// Whether this matrix's active weight format is dense BF16 (the forward
    /// path reads from `data`). Retired FP8/qweight buffers may still be present
    /// (kept alive for the share-frozen-base student alias); they are not used.
    pub fn is_dense_bf16(&self) -> bool {
        self.weight_format == WeightFormat::DenseBf16
    }

    #[must_use]
    pub fn weight_format(&self) -> WeightFormat {
        self.weight_format
    }

    /// Build the Marlin tensor-core layout for a W8A16 weight: re-encode signed
    /// INT8 → uint8b128 (+128), pack to GPTQ `[K/4, N]` i32, GPU-repack to Marlin
    /// tiles, and transpose+permute the BF16 group scales to `[K/gs, N]` (Marlin's
    /// length-64 `scale_perm`). Stores into `marlin_packed`/`marlin_scales`; the
    /// GEMM (`marlin_w8a16_gemm_cuda`) consumes them, scales stay BF16 (matches the
    /// bf16 kernel). No-op (leaves marlin_* None → scalar fallback) when the shape
    /// isn't Marlin tile-aligned. SM-gated by the caller (Ampere+).
    pub fn repack_for_marlin_w8a16(&mut self, ctx: &DeviceContext) -> Result<()> {
        if self.weight_format != WeightFormat::W8A16
            || self.qweight.is_none()
            || self.qscales.is_none()
            || self.group_size == 0
        {
            return Ok(());
        }
        // Ampere+ only (Marlin uses mma.sync/cp.async). Below sm_80 leave marlin_*
        // None so dispatch keeps the dequant→BF16 / scalar path — the shim would
        // otherwise return NOT_SUPPORTED and fail the load.
        if ctx.compute_capability().0 < 8 {
            return Ok(());
        }
        let n = self.rows; // output dim
        let k = self.cols; // input dim
        // kU8B128 is instantiated only for gs ∈ {32,64,128}; other gs → no-op kernel.
        if !k.is_multiple_of(16)
            || !n.is_multiple_of(64)
            || !k.is_multiple_of(self.group_size)
            || !matches!(self.group_size, 32 | 64 | 128)
        {
            log::warn!(
                "Marlin W8A16 repack skipped: [{n}x{k}] gs={} (need K%16, N%64, gs∈{{32,64,128}}); scalar path",
                self.group_size
            );
            return Ok(());
        }

        // element (n,k): u8 = int8+128, packed 4-per-word at bits (k%4)*8.
        let qw = self.qweight.as_ref().unwrap();
        let weight_host: Vec<i8> = ctx
            .stream
            .clone_dtoh(qw)
            .map_err(|e| anyhow!("D2H W8A16 qweight: {}", e))?;
        let gptq_rows = k / 4;
        let mut gptq = vec![0u32; gptq_rows * n];
        for row_n in 0..n {
            for col_k in 0..k {
                let u8v = (i16::from(weight_host[row_n * k + col_k]) + 128) as u32 & 0xFF;
                let gptq_row = col_k / 4;
                let bit_pos = (col_k % 4) * 8;
                gptq[gptq_row * n + row_n] |= u8v << bit_pos;
            }
        }
        // SAFETY: views the live `Vec<u32>` as its byte representation (u8 align 1,
        // len*4 bytes); `gptq` outlives the borrow.
        let gptq_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(gptq.as_ptr().cast::<u8>(), gptq.len() * 4) };
        let gptq_gpu: CudaSlice<u8> = ctx
            .stream
            .clone_htod(gptq_bytes)
            .map_err(|e| anyhow!("H2D W8A16 GPTQ: {}", e))?;

        // Marlin output: [K/16, N*4] i32 = K*N/4 i32 = K*N bytes.
        let mut marlin_gpu: CudaSlice<u8> = ctx
            .stream
            .alloc_zeros(k * n)
            .map_err(|e| anyhow!("Alloc W8A16 Marlin: {}", e))?;

        {
            let (gptq_ptr, _g1) = gptq_gpu.device_ptr(&ctx.stream);
            let (marlin_ptr, _g2) = marlin_gpu.device_ptr_mut(&ctx.stream);
            // SAFETY: both from live CudaSlices pinned by the guards; K*N-byte
            // input / output verified tile-aligned above, stream-ordered.
            unsafe {
                ffi::marlin_gptq_repack_w8a16_cuda(
                    gptq_ptr as *const u32,
                    marlin_ptr as *mut u32,
                    k as i32,
                    n as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("W8A16 Marlin repack failed: {:?}", e))?;
            }
        }

        // scale_perm is an 8×8 transpose within each 64-column block:
        //   perm[out] = (out%8)*8 + (out/8)   (vLLM get_scale_perms, len 64).
        // Kept BF16 (the bf16 GEMM reinterprets scales as scalar_t2).
        let qs = self.qscales.as_ref().unwrap();
        let scales_host: Vec<bf16> = ctx
            .stream
            .clone_dtoh(qs)
            .map_err(|e| anyhow!("D2H W8A16 scales: {}", e))?;
        let num_groups = k / self.group_size;
        let mut scales_t = vec![bf16::from_f32(0.0); num_groups * n];
        for row_n in 0..n {
            for g in 0..num_groups {
                scales_t[g * n + row_n] = scales_host[row_n * num_groups + g];
            }
        }
        // Permute within each 64-block of the flattened [num_groups, N] array
        // (N%64==0 → blocks align to N-column runs).
        let mut scales_perm = vec![0u16; num_groups * n];
        for block in 0..(num_groups * n / 64) {
            let base = block * 64;
            for out in 0..64 {
                let src = (out % 8) * 8 + (out / 8);
                scales_perm[base + out] = scales_t[base + src].to_bits();
            }
        }
        let scales_gpu: CudaSlice<u16> = ctx
            .stream
            .clone_htod(&scales_perm)
            .map_err(|e| anyhow!("H2D W8A16 Marlin scales: {}", e))?;

        self.marlin_packed = Some(marlin_gpu);
        self.marlin_scales = Some(scales_gpu);
        // Marlin consumes only marlin_packed/marlin_scales; drop the source int8
        // weight + scales to realize the W8A16 VRAM win (else both resident).
        self.qweight = None;
        self.qscales = None;

        Ok(())
    }

    /// Build the Marlin tensor-core layout for a PER-CHANNEL FP8 weight
    /// (compressed-tensors float-quantized: `F8_E4M3` + one BF16 scale per output
    /// row). Marlin's channelwise mode is `group_size = -1` -> `group_blocks = -1`,
    /// already instantiated for `kFE4M3fn` by `BIGGROUP_GET_IF`, so this needs no
    /// new kernel. No-op (leaves `marlin_packed` None -> GEMV fallback) outside the
    /// tile alignment. SM-gated by the caller (Ampere+).
    ///
    /// Two details this format does not share with W8A16:
    ///
    /// **No `+128`.** `kU8B128` stores `int8 + 128` and its dequant subtracts the
    /// bias; the `kFE4M3fn` dequant reads the raw E4M3 byte (sign bit kept, the
    /// 4-bit exponent field shifted into BF16's 8-bit one), so the byte is packed
    /// unchanged.
    ///
    /// **The scale absorbs 2^120.** `dequant_skip_flop` is `!is_int_type`, and
    /// `is_int_type` covers only the kU4/kU8 family, so `kFE4M3fn` takes the
    /// skip-flop arm and the kernel never applies its exponent-rebias multiply.
    /// Shifting an E4M3 exponent (bias 7) into a BF16 field (bias 127) without
    /// rebiasing scales every weight by `2^-120`, and unlike NVFP4 there is no `s2`
    /// global-scale channel to park the correction in — only `kFE2M1f` reads
    /// `scale2_ptr`. So the per-channel scale carries it. The fold overflows only
    /// at `scale >= 255.5`; this checkpoint's channel scales are ~1e-3..1e-1,
    /// landing at 2^110..2^117, and 2^120 is an exact power of two so the fold
    /// shifts the exponent without touching the mantissa.
    ///
    /// Unlike NVFP4, nothing underflows. For `size_bits() == 8 && group_blocks ==
    /// -1` the scale multiplies the FP32 accumulator after the K loop
    /// (`marlin_template.h:1548`), not the BF16 weight fragment before the MMA
    /// (`:1136`, the kFE2M1f site that returned `nonzero 0/256`). The fragment
    /// carries `w * 2^-120` down to 2^-129, inside bf16; the accumulator's 2^-120
    /// and the scale's 2^+120 cancel bit-exactly in f32.
    ///
    /// Releases `qweight_u8` on success: the DeepGEMM prefill arm materialises
    /// the plain `[N, K]` E4M3 bytes back out of these tiles per call
    /// (`marlin_fp8_to_e4m3_cuda`), so nothing reads the checkpoint copy.
    /// `scale_f32` is `[N]` and stays — that arm's post-GEMM channel scale
    /// reads it.
    pub fn repack_for_marlin_fp8(&mut self, ctx: &DeviceContext) -> Result<()> {
        if self.weight_format != WeightFormat::Fp8BlockScaled
            || self.qweight_u8.is_none()
            || self.scale_f32.is_none()
        {
            return Ok(());
        }
        if ctx.compute_capability().0 < 8 {
            return Ok(());
        }
        let n = self.rows; // output dim
        let k = self.cols; // input dim
        // Per-channel only: one scale per output row, spanning all of K.
        // `block_m == 1` is the discriminator — a 128x128 block-scaled weight
        // belongs to DeepGEMM and fails it. `block_k >= k` rather than `== k` so a
        // TP shard, whose `cols` is a slice of the K the scale was defined over,
        // still qualifies.
        // K%64, not the repack's K%16: `min_thread_k = 64` and every thread config
        // has thread_k in {64,128} (marlin.cuh:18, gptq_marlin.cuh:115-129), so a K
        // the GEMM's `is_valid_config` rejects would repack cleanly here and then
        // throw on every call. N%64 matches min_thread_n and the repack's tile_n.
        if self.quant_block_m != 1
            || self.quant_block_k < k
            || !k.is_multiple_of(64)
            || !n.is_multiple_of(64)
        {
            return Ok(());
        }

        // Step 1: raw E4M3 [N, K] row-major -> GPTQ [K/4, N] i32, element (n,k) at
        // bit (k%4)*8 of word (k/4)*N + n. No bias: see the note above. Each (n,k)
        // owns exactly one byte, so this is a byte-granular transpose, written
        // straight into the little-endian byte position the u32 packing implies.
        // Blocked: the destination stride is 4*N bytes, and this runs over every
        // per-channel FP8 weight in the model at load.
        let qw = self.qweight_u8.as_ref().unwrap();
        let weight_host: Vec<u8> = ctx
            .stream
            .clone_dtoh(qw)
            .map_err(|e| anyhow!("D2H FP8 qweight: {}", e))?;
        ensure!(
            weight_host.len() == n * k,
            "FP8 per-channel weight is {} bytes, expected {n}*{k}",
            weight_host.len()
        );
        const TILE: usize = 64;
        let mut gptq_bytes = vec![0u8; n * k];
        for n0 in (0..n).step_by(TILE) {
            for k0 in (0..k).step_by(TILE) {
                let k_end = (k0 + TILE).min(k);
                for row_n in n0..(n0 + TILE).min(n) {
                    for (i, &byte) in weight_host[row_n * k + k0..row_n * k + k_end]
                        .iter()
                        .enumerate()
                    {
                        let col_k = k0 + i;
                        gptq_bytes[((col_k / 4) * n + row_n) * 4 + (col_k % 4)] = byte;
                    }
                }
            }
        }
        let gptq_gpu: CudaSlice<u8> = ctx
            .stream
            .clone_htod(&gptq_bytes)
            .map_err(|e| anyhow!("H2D FP8 GPTQ: {}", e))?;

        // Step 2: GPTQ -> Marlin tiles. The repack is a pure 8-bit lane shuffle, so
        // the kU8B128 kernel serves kFE4M3fn unchanged.
        let mut marlin_gpu: CudaSlice<u8> = ctx
            .stream
            .alloc_zeros(k * n)
            .map_err(|e| anyhow!("Alloc FP8 Marlin: {}", e))?;
        {
            let (gptq_ptr, _g1) = gptq_gpu.device_ptr(&ctx.stream);
            let (marlin_ptr, _g2) = marlin_gpu.device_ptr_mut(&ctx.stream);
            // SAFETY: both from live CudaSlices pinned by the guards; K*N bytes in
            // and out, tile alignment checked above, stream-ordered.
            unsafe {
                ffi::marlin_gptq_repack_w8a16_cuda(
                    gptq_ptr as *const u32,
                    marlin_ptr as *mut u32,
                    k as i32,
                    n as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("FP8 Marlin repack failed: {:?}", e))?;
            }
        }

        // Step 3: scales [N] f32 -> [1, N] bf16, folding 2^120, then vLLM's
        // `scale_perm_single` (channelwise) — NOT the length-64 `scale_perm` the
        // grouped path uses.
        let scales_host: Vec<f32> = ctx
            .stream
            .clone_dtoh(self.scale_f32.as_ref().unwrap())
            .map_err(|e| anyhow!("D2H FP8 scales: {}", e))?;
        ensure!(
            scales_host.len() == n,
            "per-channel FP8 needs {n} scales, got {}",
            scales_host.len()
        );
        // 2^120 as a bit pattern, not a decimal literal: the whole no-precision-loss
        // argument is that the fold is a pure exponent shift.
        const SKIP_FLOP_FACTOR: f32 = f32::from_bits(0x7B80_0000);
        // The fold is bit-exact only if the source scale is already bf16-exact,
        // which this checkpoint's weight_scale is. An f32-scale checkpoint takes a
        // coherent per-output-channel 2^-9 bias instead — say so rather than assume.
        if let Some(s) = scales_host
            .iter()
            .find(|&&s| bf16::from_f32(s).to_f32() != s)
        {
            log::warn!(
                "Marlin FP8 scales are not bf16-exact (e.g. {s}); the fold costs up to 2^-9 per channel"
            );
        }
        // bf16 rounds to +inf at scale >= 255.5. Testing the f32 product would let
        // (3.3895e38, 3.4028e38] through and store an infinite channel scale. A
        // scale with no Marlin encoding means skip the weight, not fail the load.
        if scales_host
            .iter()
            .any(|&s| !bf16::from_f32(s * SKIP_FLOP_FACTOR).is_finite())
        {
            log::warn!(
                "Marlin FP8 repack skipped: [{n}x{k}] channel scale overflows bf16 once 2^120 is folded in (limit 255.5); scalar path"
            );
            return Ok(());
        }
        let mut perm_single = [0usize; 32];
        for i in 0..4 {
            for (jj, off) in [0usize, 1, 8, 9, 16, 17, 24, 25].into_iter().enumerate() {
                perm_single[i * 8 + jj] = 2 * i + off;
            }
        }
        let mut scales_perm = vec![0u16; n];
        for block in 0..(n / 32) {
            let base = block * 32;
            for out in 0..32 {
                let v = scales_host[base + perm_single[out]] * SKIP_FLOP_FACTOR;
                scales_perm[base + out] = bf16::from_f32(v).to_bits();
            }
        }
        let scales_gpu: CudaSlice<u16> = ctx
            .stream
            .clone_htod(&scales_perm)
            .map_err(|e| anyhow!("H2D FP8 Marlin scales: {}", e))?;

        self.marlin_packed = Some(marlin_gpu);
        self.marlin_scales = Some(scales_gpu);
        self.qweight_u8 = None;
        Ok(())
    }

    /// Build the per-128x128-block power of two the NVFP4 DeepGEMM prefill arm
    /// divides out of the weight and hands DeepGEMM back as `sfb`.
    ///
    /// The group scale cannot ride inside the E4M3 weight value: this
    /// checkpoint's `weight_scale` reaches E4M3's full 448 and an E2M1 value
    /// reaches 6, so the product reaches 2688 against a 448 ceiling. Dividing
    /// out a per-block power of two puts every block's peak in (224, 448] and
    /// costs nothing — a power of two is exact both ways, and DeepGEMM's `sfb`
    /// multiplies it back into the fp32 accumulator.
    ///
    /// The floor is measured, not assumed: over this checkpoint's 168 NVFP4
    /// scale tensors the widest 128x128 block spans 6.81 binades, which leaves
    /// the smallest folded value at 0.332 against E4M3's 0.0156 normal minimum.
    ///
    /// Caller decides whether the arm is reachable at all (SM tier, DeepGEMM
    /// built and enabled); this only checks that the shape and metadata fit.
    /// Reads the S0E5M3 scale tail of `marlin_packed`, so it must run after
    /// [`Self::repack_for_marlin_fp4`] and its absence means no arm.
    pub fn prepare_fp4_deepgemm_sfb(&mut self, ctx: &DeviceContext) -> Result<()> {
        if self.weight_format != WeightFormat::Fp4E2M1Group
            || self.marlin_packed.is_none()
            || self.scale_f32.is_none()
        {
            return Ok(());
        }
        let n = self.rows;
        let k = self.cols;
        // DeepGEMM's dense NT entry wants `k % 128` and `n % 8`; a Marlin FP4
        // layout only exists at group_size 16 with `n % 64`.
        if self.group_size != 16 || !k.is_multiple_of(128) || !n.is_multiple_of(64) {
            return Ok(());
        }
        let packed = self.marlin_packed.as_ref().expect("checked above");
        let global = self.scale_f32.as_ref().expect("checked above");
        let len = (n.div_ceil(128) + 1) * k.div_ceil(128);
        let sfb = ctx
            .stream
            .alloc_zeros::<f32>(len)
            .map_err(|e| anyhow!("NVFP4 DeepGEMM sfb alloc failed: {e}"))?;
        {
            let (packed_ptr, _gs) = packed.device_ptr(&ctx.stream);
            let (global_ptr, _gg) = global.device_ptr(&ctx.stream);
            let (sfb_ptr, _gf) = sfb.device_ptr(&ctx.stream);
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            unsafe {
                crate::ffi::fp4_marlin_scale_block_pow2_cuda(
                    packed_ptr as *const u8,
                    global_ptr as *const f32,
                    self.fp4_marlin_scale_lift_inv,
                    sfb_ptr as *mut f32,
                    n as i32,
                    k as i32,
                    self.group_size as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("NVFP4 DeepGEMM sfb kernel failed: {e}"))?;
            }
        }
        self.fp4_deepgemm_sfb = Some(sfb);
        Ok(())
    }

    /// Build the Marlin tensor-core layout for an NVFP4 weight
    /// (compressed-tensors nvfp4-pack-quantized): transpose the packed E2M1
    /// nibbles into GPTQ `[K/8, N]` i32, GPU-repack to Marlin tiles, and
    /// re-encode the FP8 E4M3 group scales into the S0E5M3 form Marlin's FP8
    /// scale dequant expects. `marlin_packed` holds both, concatenated —
    /// `[K*N/2 weight bytes][N*K/16 scale bytes]`, one allocation because the
    /// two are always read together — and `marlin_scales` the single BF16
    /// when the shape or group size is outside the kFE2M1f kernel's
    /// instantiation. SM-gated by the caller (Ampere+).
    ///
    /// Releases `qweight_u8` / `qscale_fp8` on success — every arm that reads
    /// an NVFP4 weight with a Marlin layout reads it from here, the DeepGEMM
    /// prefill arm included (`dequantize_fp4_marlin_to_fp8_cuda`). Keeping them
    /// stored the model twice: 39.3 GB resident for a 23.9 GB checkpoint.
    /// `scale_f32` is small and stays.
    ///
    /// Scale encoding (vLLM `marlin_utils_fp4.nvfp4_marlin_process_scales`):
    /// the kernel's weight dequant leaves a 2^-126 factor and the scale dequant
    /// reads an 8-bit `[E5|M3]` field, so the byte is the high half of
    /// `f16(scale * 2^7) << 1` and the leftover 2^119 is folded into the global
    pub fn repack_for_marlin_fp4(&mut self, ctx: &DeviceContext) -> Result<()> {
        if self.weight_format != WeightFormat::Fp4E2M1Group
            || self.qweight_u8.is_none()
            || self.qscale_fp8.is_none()
            || self.scale_f32.is_none()
        {
            return Ok(());
        }
        let (major, minor) = ctx.compute_capability();
        // NVFP4 has one serving path and it is Marlin's, so a shape or a tier it
        // cannot take is a load failure with the reason, not a silent demotion to
        // a scalar arm no gate has ever executed.
        ensure!(
            major >= 8,
            "NVFP4 requires sm_80 or newer for the Marlin tensor-core path; this device is \
             sm_{major}{minor}. Serve an FP8 or W4A16 checkpoint instead."
        );
        let n = self.rows; // output dim
        let k = self.cols; // input dim
        // kFE2M1f is instantiated only at group_blocks == 1 (group_size 16);
        // the tile grid needs N % 64 and K % 64.
        ensure!(
            self.group_size == 16
                && k.is_multiple_of(64)
                && n.is_multiple_of(64)
                && self.quant_scale_rows == n
                && self.quant_scale_cols == k / 16,
            "NVFP4 weight [{n}x{k}] gs={} scales=[{}x{}] cannot take the Marlin layout \
             (kFE2M1f needs group_size 16, and the tile grid needs K%64 and N%64). NVFP4 has no \
             other serving path.",
            self.group_size,
            self.quant_scale_rows,
            self.quant_scale_cols
        );
        let global_host: Vec<f32> = ctx
            .stream
            .clone_dtoh(self.scale_f32.as_ref().unwrap())
            .map_err(|e| anyhow!("D2H NVFP4 global scale: {}", e))?;
        ensure!(
            global_host.len() == 1,
            "NVFP4 weight [{n}x{k}] carries {} global scales, expected 1",
            global_host.len()
        );

        // Step 1: packed [N, K/2] u8 → GPTQ [K/8, N] i32. Each row's u32 view is
        // already k-major inside the word (nibble k at bit (k%8)*4), so this is a
        // plain transpose of that view — no bit shuffling.
        let qw = self.qweight_u8.as_ref().unwrap();
        let packed_host: Vec<u8> = ctx
            .stream
            .clone_dtoh(qw)
            .map_err(|e| anyhow!("D2H NVFP4 qweight: {}", e))?;
        ensure!(
            packed_host.len() == n * (k / 2),
            "NVFP4 qweight {} bytes, expected {}",
            packed_host.len(),
            n * (k / 2)
        );
        let words_per_row = k / 8;
        let mut gptq = vec![0u32; words_per_row * n];
        for row_n in 0..n {
            let base = row_n * (k / 2);
            for j in 0..words_per_row {
                let b = &packed_host[base + j * 4..base + j * 4 + 4];
                gptq[j * n + row_n] = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        // SAFETY: views the live `Vec<u32>` as its byte representation (u8 align 1,
        // len*4 bytes); `gptq` outlives the borrow.
        let gptq_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(gptq.as_ptr().cast::<u8>(), gptq.len() * 4) };
        let gptq_gpu: CudaSlice<u8> = ctx
            .stream
            .clone_htod(gptq_bytes)
            .map_err(|e| anyhow!("H2D NVFP4 GPTQ: {}", e))?;

        // Marlin weight: [K/16, N*2] i32 = K*N/8 i32 = K*N/2 bytes, then the
        // S0E5M3 scale bytes in the tail of the same allocation.
        let weight_bytes = k * n / 2;
        let mut marlin_gpu: CudaSlice<u8> = ctx
            .stream
            .alloc_zeros(weight_bytes + n * (k / 16))
            .map_err(|e| anyhow!("Alloc NVFP4 Marlin: {}", e))?;
        {
            let (gptq_ptr, _g1) = gptq_gpu.device_ptr(&ctx.stream);
            let (marlin_ptr, _g2) = marlin_gpu.device_ptr_mut(&ctx.stream);
            // SAFETY: both from live CudaSlices pinned by the guards; sizes checked
            // tile-aligned above, stream-ordered.
            unsafe {
                ffi::marlin_gptq_repack_fp4_cuda(
                    gptq_ptr as *const u32,
                    marlin_ptr as *mut u32,
                    k as i32,
                    n as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("NVFP4 Marlin repack failed: {:?}", e))?;
            }
        }

        // Step 2: scales [N, K/16] E4M3 → transpose [K/16, N] → Marlin permute
        // (8×8 transpose inside each 64-run) → the FP8-dequant pair order
        // [0,2,1,3] inside each 4-run → S0E5M3 bytes.
        let scales_host: Vec<u8> = ctx
            .stream
            .clone_dtoh(self.qscale_fp8.as_ref().unwrap())
            .map_err(|e| anyhow!("D2H NVFP4 scales: {}", e))?;
        let num_groups = k / 16;
        ensure!(
            scales_host.len() == n * num_groups,
            "NVFP4 scales {} bytes, expected {}",
            scales_host.len(),
            n * num_groups
        );
        let mut sflat = vec![0f32; num_groups * n];
        for row_n in 0..n {
            for g in 0..num_groups {
                sflat[g * n + row_n] = e4m3_to_f32(scales_host[row_n * num_groups + g]);
            }
        }
        let mut sperm = vec![0f32; num_groups * n];
        for block in 0..(num_groups * n / 64) {
            let base = block * 64;
            for out in 0..64 {
                sperm[base + out] = sflat[base + (out % 8) * 8 + (out / 8)];
            }
        }
        for quad in sperm.chunks_exact_mut(4) {
            quad.swap(1, 2);
        }
        // The S0E5M3 field only spans exponents whose MSB is set, i.e. scale*2^7
        // >= 2; lift everything by the largest power of two that keeps the max
        // inside E4M3 range and divide it back out of the global scale.
        let smax = sperm.iter().fold(0f32, |a, &b| a.max(b)) * 128.0;
        let ceiling = 448.0f32 * 128.0;
        let scale_factor = if smax > 0.0 && smax < ceiling {
            (ceiling / smax).log2().floor().exp2()
        } else {
            1.0
        };
        let mut sbytes = vec![0u8; num_groups * n];
        let mut flushed = 0usize;
        for (dst, &src) in sbytes.iter_mut().zip(sperm.iter()) {
            let v = src * scale_factor * 128.0;
            let v = if v < 2.0 {
                flushed += usize::from(src > 0.0);
                0.0
            } else {
                v
            };
            *dst = (f16::from_f32(v).to_bits() >> 7) as u8;
        }
        if flushed > 0 {
            // The one lossy step in the repack, and now also in the DeepGEMM
            // prefill arm that reads these bytes back.
            log::warn!(
                "Marlin NVFP4 [{n}x{k}]: {flushed}/{} group scales below the S0E5M3 floor flushed to zero",
                sbytes.len()
            );
        }
        {
            let mut tail = marlin_gpu.slice_mut(weight_bytes..weight_bytes + sbytes.len());
            ctx.stream
                .memcpy_htod(sbytes.as_slice(), &mut tail)
                .map_err(|e| anyhow!("H2D NVFP4 Marlin scales: {}", e))?;
        }

        // Step 3: global scale, pre-multiplied by the 2^119 dequant bias.
        let global = bf16::from_f32(
            (f64::from(global_host[0]) * 2f64.powi(119) / f64::from(scale_factor)) as f32,
        );
        let global_gpu: CudaSlice<u16> = ctx
            .stream
            .clone_htod(&[global.to_bits()])
            .map_err(|e| anyhow!("H2D NVFP4 Marlin global scale: {}", e))?;

        self.marlin_packed = Some(marlin_gpu);
        self.marlin_scales = Some(global_gpu);
        self.fp4_marlin_scale_lift_inv = 1.0 / (scale_factor * 128.0);
        // Marlin holds every nibble and every group scale now, and both the
        // Marlin GEMM and the DeepGEMM prefill arm read them from here.
        self.qweight_u8 = None;
        self.qscale_fp8 = None;
        Ok(())
    }

    pub fn from_safetensors(
        ctx: &DeviceContext,
        data: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        if data.len() != rows * cols * std::mem::size_of::<bf16>() {
            return Err(anyhow!(
                "Data length mismatch: expected {} bytes, got {} bytes",
                rows * cols * std::mem::size_of::<bf16>(),
                data.len()
            ));
        }
        let slice = bf16_safetensor_host_slice(data)?;
        let gpu_data = ctx
            .stream
            .clone_htod(slice.as_ref())
            .map_err(|e| anyhow!("H2D copy failed: {}", e))?;
        Ok(Self {
            data: gpu_data,
            rows,
            cols,
            weight_format: WeightFormat::DenseBf16,
            qweight: None,
            qweight_u8: None,
            pristine_fp8: None,
            qscales: None,
            qscale_fp8: None,
            scale_f32: None,
            scale2_f32: None,
            quant_scale_rows: 0,
            quant_scale_cols: 0,
            quant_block_m: 0,
            quant_block_k: 0,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            fp4_deepgemm_sfb: None,
            fp4_marlin_scale_lift_inv: 1.0,
            fp8_deepgemm_prefill: false,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    pub fn slice_rows(
        ctx: &DeviceContext,
        src: &DeviceMatrix,
        row_start: usize,
        row_end: usize,
    ) -> Result<Self> {
        assert!(
            row_start < row_end && row_end <= src.rows,
            "slice_rows: invalid range [{}..{}) for matrix with {} rows",
            row_start,
            row_end,
            src.rows,
        );
        let out_rows = row_end - row_start;
        let n = out_rows * src.cols;
        let offset = row_start * src.cols;
        let mut dst: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(n)
            .map_err(|e| anyhow!("slice_rows alloc failed: {e}"))?;
        ctx.stream
            .memcpy_dtod(&src.data.slice(offset..offset + n), &mut dst)
            .map_err(|e| anyhow!("slice_rows D2D copy failed: {e}"))?;
        Ok(Self {
            data: dst,
            rows: out_rows,
            cols: src.cols,
            weight_format: WeightFormat::DenseBf16,
            qweight: None,
            qweight_u8: None,
            pristine_fp8: None,
            qscales: None,
            qscale_fp8: None,
            scale_f32: None,
            scale2_f32: None,
            quant_scale_rows: 0,
            quant_scale_cols: 0,
            quant_block_m: 0,
            quant_block_k: 0,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            fp4_deepgemm_sfb: None,
            fp4_marlin_scale_lift_inv: 1.0,
            fp8_deepgemm_prefill: false,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn copy_matrix_to_host(ctx: &DeviceContext, matrix: &DeviceMatrix) -> Vec<bf16> {
        let host = ctx
            .stream
            .clone_dtoh(&matrix.data)
            .expect("D2H copy failed");
        ctx.sync().expect("CUDA sync failed");
        host
    }

    #[test]
    fn test_device_matrix_from_host_roundtrip() {
        let ctx = DeviceContext::new().expect("Failed to create CUDA context");
        let rows = 2;
        let cols = 3;
        let host = vec![
            bf16::from_f32(-1.5),
            bf16::from_f32(0.0),
            bf16::from_f32(2.25),
            bf16::from_f32(7.0),
            bf16::from_f32(-3.0),
            bf16::from_f32(0.5),
        ];

        let matrix =
            DeviceMatrix::from_host(&ctx, &host, rows, cols).expect("from_host should succeed");

        assert_eq!(matrix.rows, rows);
        assert_eq!(matrix.cols, cols);

        let got = copy_matrix_to_host(&ctx, &matrix);
        assert_eq!(got.len(), host.len());
        for (idx, (actual, expected)) in got.iter().zip(host.iter()).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "roundtrip mismatch at index {}",
                idx
            );
        }
    }

    #[test]
    fn test_device_matrix_from_safetensors_matches_from_host() {
        let ctx = DeviceContext::new().expect("Failed to create CUDA context");
        let rows = 3;
        let cols = 2;
        let host = vec![
            bf16::from_f32(-8.0),
            bf16::from_f32(-0.25),
            bf16::from_f32(1.0),
            bf16::from_f32(3.5),
            bf16::from_f32(9.0),
            bf16::from_f32(10.75),
        ];
        let safetensor_bytes: Vec<u8> = host
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();

        let from_host =
            DeviceMatrix::from_host(&ctx, &host, rows, cols).expect("from_host should succeed");
        let from_safetensors = DeviceMatrix::from_safetensors(&ctx, &safetensor_bytes, rows, cols)
            .expect("from_safetensors should succeed");

        assert_eq!(from_safetensors.rows, from_host.rows);
        assert_eq!(from_safetensors.cols, from_host.cols);

        let host_out = copy_matrix_to_host(&ctx, &from_host);
        let safetensors_out = copy_matrix_to_host(&ctx, &from_safetensors);
        assert_eq!(host_out.len(), safetensors_out.len());
        for (idx, (a, b)) in host_out.iter().zip(safetensors_out.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "from_safetensors/from_host mismatch at index {}",
                idx
            );
        }
    }
}
