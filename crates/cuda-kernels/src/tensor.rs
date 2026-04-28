//! Device tensor types and CUDA context.

use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use std::marker::PhantomData;
use std::sync::Arc;

use super::ffi;

/// CUDA device context holding compute stream and optional copy stream.
///
/// Two-stream architecture for overlapping H2D/D2H transfers with compute:
/// - `stream` (compute): all GPU kernels, CUDA Graph capture/replay
/// - `copy_stream`: async H2D/D2H transfers, runs concurrently with compute
///
/// Cross-stream sync uses raw CUDA events (not cudarc's automatic tracking,
/// which breaks CUDA Graph capture).
#[derive(Clone)]
pub struct DeviceContext {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    /// Separate stream for async H2D/D2H memory copies.
    pub copy_stream: Arc<CudaStream>,
    /// CUDA device ordinal this context is bound to.
    pub ordinal: u32,
}

/// Parse `INFER_CUDA_DEVICE` (default 0). Selects the device for `DeviceContext::new()`.
pub fn parse_device_ordinal_from_env() -> Result<u32> {
    parse_device_ordinal(std::env::var("INFER_CUDA_DEVICE").ok().as_deref())
}

/// String-pure parse of an `INFER_CUDA_DEVICE`-style ordinal. `None` => 0.
/// Split out from [`parse_device_ordinal_from_env`] so unit tests don't need
/// to mutate the process environment (which races with concurrent tests).
fn parse_device_ordinal(value: Option<&str>) -> Result<u32> {
    match value {
        Some(s) => s.trim().parse::<u32>().map_err(|e| {
            anyhow!("INFER_CUDA_DEVICE must be a non-negative integer, got {s:?}: {e}")
        }),
        None => Ok(0),
    }
}

impl DeviceContext {
    /// Query available (free) GPU memory in bytes.
    /// Returns `(free_bytes, total_bytes)`.
    pub fn gpu_memory_info() -> Result<(usize, usize)> {
        cudarc::driver::result::mem_get_info()
            .map_err(|e| anyhow!("Failed to query GPU memory: {}", e))
    }

    /// Default constructor: honours `INFER_CUDA_DEVICE` (default 0).
    /// F1+ multi-GPU rank threads bypass this and call `on_device(ordinal)`.
    pub fn new() -> Result<Self> {
        let ordinal = parse_device_ordinal_from_env()?;
        Self::on_device(ordinal)
    }

    pub fn on_device(ordinal: u32) -> Result<Self> {
        let ctx = CudaContext::new(ordinal as usize)
            .map_err(|e| anyhow!("Failed to create CUDA context on device {ordinal}: {e}"))?;

        // Disable multi-stream event tracking before creating streams.
        // We use a single compute stream, so no cross-stream synchronization is needed.
        // This avoids stream.wait(event) calls that break CUDA Graph capture.
        // SAFETY: We only use one stream for all GPU work.
        unsafe {
            ctx.disable_event_tracking();
        }

        let stream = ctx
            .new_stream()
            .map_err(|e| anyhow!("Failed to create CUDA stream: {}", e))?;

        let copy_stream = ctx
            .new_stream()
            .map_err(|e| anyhow!("Failed to create CUDA copy stream: {}", e))?;

        // Initialize cuBLAS handle
        unsafe {
            ffi::cublas_init();
        }

        Ok(Self {
            ctx,
            stream,
            copy_stream,
            ordinal,
        })
    }

    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    /// Query the number of streaming multiprocessors on the GPU this context is bound to.
    pub fn sm_count(&self) -> usize {
        use cudarc::driver::sys::*;
        let mut count: i32 = 0;
        unsafe {
            cuDeviceGetAttribute(
                &mut count,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                self.ctx.cu_device(),
            );
        }
        count.max(1) as usize
    }

    /// Synchronize compute stream.
    pub fn sync(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| anyhow!("Sync failed: {}", e))
    }

    /// Synchronize copy stream.
    pub fn sync_copy(&self) -> Result<()> {
        self.copy_stream
            .synchronize()
            .map_err(|e| anyhow!("Copy stream sync failed: {}", e))
    }

    /// Record an event on the compute stream and make the copy stream wait for it.
    ///
    /// Use after GPU kernels finish (e.g. sampling) to ensure the copy stream
    /// sees the results before starting D2H transfer.
    pub fn copy_waits_for_compute(&self) -> Result<()> {
        use cudarc::driver::sys::*;
        unsafe {
            let mut event: CUevent = std::ptr::null_mut();
            let r = cuEventCreate(&mut event, CUevent_flags::CU_EVENT_DISABLE_TIMING as u32);
            if r != CUresult::CUDA_SUCCESS {
                return Err(anyhow!("cuEventCreate failed: {:?}", r));
            }
            let r = cuEventRecord(event, self.stream.cu_stream());
            if r != CUresult::CUDA_SUCCESS {
                cuEventDestroy_v2(event);
                return Err(anyhow!("cuEventRecord failed: {:?}", r));
            }
            let r = cuStreamWaitEvent(self.copy_stream.cu_stream(), event, 0);
            cuEventDestroy_v2(event);
            if r != CUresult::CUDA_SUCCESS {
                return Err(anyhow!("cuStreamWaitEvent failed: {:?}", r));
            }
        }
        Ok(())
    }

    /// Record an event on the copy stream and make the compute stream wait for it.
    ///
    /// Use after H2D transfer completes to ensure compute kernels see the uploaded data.
    pub fn compute_waits_for_copy(&self) -> Result<()> {
        use cudarc::driver::sys::*;
        unsafe {
            let mut event: CUevent = std::ptr::null_mut();
            let r = cuEventCreate(&mut event, CUevent_flags::CU_EVENT_DISABLE_TIMING as u32);
            if r != CUresult::CUDA_SUCCESS {
                return Err(anyhow!("cuEventCreate failed: {:?}", r));
            }
            let r = cuEventRecord(event, self.copy_stream.cu_stream());
            if r != CUresult::CUDA_SUCCESS {
                cuEventDestroy_v2(event);
                return Err(anyhow!("cuEventRecord failed: {:?}", r));
            }
            let r = cuStreamWaitEvent(self.stream.cu_stream(), event, 0);
            cuEventDestroy_v2(event);
            if r != CUresult::CUDA_SUCCESS {
                return Err(anyhow!("cuStreamWaitEvent failed: {:?}", r));
            }
        }
        Ok(())
    }
}

/// 1D device tensor (vector) — stored as bf16.
pub struct DeviceVec {
    pub data: CudaSlice<bf16>,
    pub len: usize,
    /// Debug label describing the tensor's semantic shape (e.g., `norm_weight[hidden]`, `kv_cache[heads,seq,dim]`).
    pub label: &'static str,
}

impl DeviceVec {
    /// Create from host data (bf16)
    pub fn from_host(ctx: &DeviceContext, data: &[bf16]) -> Result<Self> {
        let gpu_data = ctx
            .stream
            .clone_htod(data)
            .map_err(|e| anyhow!("H2D copy failed: {}", e))?;
        Ok(Self {
            data: gpu_data,
            len: data.len(),
            label: "",
        })
    }

    #[allow(clippy::cast_ptr_alignment)]
    pub fn from_safetensors(ctx: &DeviceContext, data: &[u8]) -> Result<Self> {
        if !data.len().is_multiple_of(2) {
            return Err(anyhow!(
                "Data length must be even for bf16: got {} bytes",
                data.len()
            ));
        }
        let len = data.len() / 2;
        // NOTE: This assumes a little-endian host. Safetensors are little-endian.
        // On a big-endian machine, this will be incorrect. A full solution would
        // involve byte-swapping.
        let slice = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<bf16>(), len) };
        Self::from_host(ctx, slice)
    }

    /// Create zeroed tensor
    pub fn zeros(ctx: &DeviceContext, len: usize) -> Result<Self> {
        let gpu_data: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(len)
            .map_err(|e| anyhow!("Alloc failed: {}", e))?;
        Ok(Self {
            data: gpu_data,
            len,
            label: "",
        })
    }

    /// Create a tensor filled with bf16 ones (1.0).
    /// Useful for dummy RMSNorm weights (identity normalization).
    pub fn ones(ctx: &DeviceContext, len: usize) -> Result<Self> {
        let host = vec![bf16::ONE; len];
        Self::from_host(ctx, &host)
    }

    /// Extract a contiguous sub-range `[start..end)` as a new `DeviceVec`.
    /// The result is an independent copy on the GPU.
    pub fn slice_to_vec(
        ctx: &DeviceContext,
        src: &DeviceVec,
        start: usize,
        end: usize,
    ) -> Result<Self> {
        assert!(
            start < end && end <= src.len,
            "slice_to_vec: invalid range [{}..{}) for vec of len {}",
            start,
            end,
            src.len,
        );
        let len = end - start;
        let mut out = Self::zeros(ctx, len)?;
        let src_view = src.data.slice(start..end);
        ctx.stream
            .memcpy_dtod(&src_view, &mut out.data)
            .map_err(|e| anyhow!("slice_to_vec D2D copy failed: {e}"))?;
        Ok(out)
    }

    /// Attach a debug label describing this tensor's semantic shape/purpose.
    ///
    /// ```ignore
    /// let w = DeviceVec::zeros(&ctx, 4096)?.with_label("norm_weight[hidden]");
    /// ```
    pub fn with_label(mut self, label: &'static str) -> Self {
        self.label = label;
        self
    }

    /// Copy a region of the device buffer to a host slice (D2H).
    ///
    /// `offset` and `len` are in elements (bf16), not bytes.
    /// `dst` must have length >= `len`.
    pub fn copy_region_to_host(
        &self,
        ctx: &DeviceContext,
        offset: usize,
        len: usize,
        dst: &mut [bf16],
    ) -> Result<()> {
        assert!(
            offset + len <= self.len,
            "copy_region_to_host: offset {} + len {} exceeds buffer len {}",
            offset,
            len,
            self.len
        );
        assert!(
            dst.len() >= len,
            "copy_region_to_host: dst len {} < requested len {}",
            dst.len(),
            len
        );
        let view = self.data.slice(offset..offset + len);
        ctx.stream
            .memcpy_dtoh(&view, &mut dst[..len])
            .map_err(|e| anyhow!("D2H region copy failed: {}", e))?;
        Ok(())
    }

    /// Copy from a host slice into a region of the device buffer (H2D).
    ///
    /// `offset` is in elements (bf16). `src.len()` elements are copied
    /// starting at `offset` in the device buffer.
    pub fn copy_region_from_host(
        &mut self,
        ctx: &DeviceContext,
        offset: usize,
        src: &[bf16],
    ) -> Result<()> {
        assert!(
            offset + src.len() <= self.len,
            "copy_region_from_host: offset {} + src len {} exceeds buffer len {}",
            offset,
            src.len(),
            self.len
        );
        let mut view = self.data.slice_mut(offset..offset + src.len());
        ctx.stream
            .memcpy_htod(src, &mut view)
            .map_err(|e| anyhow!("H2D region copy failed: {}", e))?;
        Ok(())
    }

    /// Copy a region within the same device buffer or between buffers (D2D).
    ///
    /// Copies `len` elements from `src_offset` in `src` to `dst_offset` in `self`.
    pub fn copy_region_from_device(
        &mut self,
        ctx: &DeviceContext,
        dst_offset: usize,
        src: &DeviceVec,
        src_offset: usize,
        len: usize,
    ) -> Result<()> {
        assert!(
            src_offset + len <= src.len,
            "copy_region_from_device: src_offset {} + len {} exceeds src len {}",
            src_offset,
            len,
            src.len
        );
        assert!(
            dst_offset + len <= self.len,
            "copy_region_from_device: dst_offset {} + len {} exceeds dst len {}",
            dst_offset,
            len,
            self.len
        );
        let src_view = src.data.slice(src_offset..src_offset + len);
        let mut dst_view = self.data.slice_mut(dst_offset..dst_offset + len);
        ctx.stream
            .memcpy_dtod(&src_view, &mut dst_view)
            .map_err(|e| anyhow!("D2D region copy failed: {}", e))?;
        Ok(())
    }

    /// Copy to host as f32 (for testing). Exposed publicly so downstream
    /// crates in this workspace (notably `infer`) can use it from their
    /// own test suites, since that would otherwise sit behind the
    /// cuda-kernels `#[cfg(test)]` boundary.
    pub fn to_host(&self, ctx: &DeviceContext) -> Result<Vec<f32>> {
        let host_f16 = ctx
            .stream
            .clone_dtoh(&self.data)
            .map_err(|e| anyhow!("D2H copy failed: {}", e))?;
        ctx.sync()?;
        Ok(host_f16.iter().map(|x| x.to_f32()).collect())
    }
}

impl Clone for DeviceVec {
    fn clone(&self) -> Self {
        Self {
            data: self.data.try_clone().unwrap(),
            len: self.len,
            label: self.label,
        }
    }
}

impl std::fmt::Debug for DeviceVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.label.is_empty() {
            write!(f, "DeviceVec(len={})", self.len)
        } else {
            write!(f, "DeviceVec({}, len={})", self.label, self.len)
        }
    }
}

/// Explicit storage format for a linear weight matrix.
///
/// This is the Rust-side kernel ABI selector: checkpoint format detection and
/// loader packing set this once, then inference dispatch matches this enum
/// instead of re-interpreting packed buffers through bit-width sentinels.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WeightFormat {
    /// Dense row-major BF16 weights.
    #[default]
    DenseBf16,
    /// Uniform per-group signed INT8 weights with BF16 scales.
    W8A16,
    /// Uniform per-group packed INT4 weights with BF16 scales.
    W4A16,
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

    #[must_use]
    pub fn is_gguf_k_quant(self) -> bool {
        matches!(
            self,
            Self::GgufQ3K | Self::GgufQ4K | Self::GgufQ5K | Self::GgufQ6K
        )
    }

    #[must_use]
    pub fn kernel_alignment(self, group_size: usize) -> WeightKernelAlignment {
        match self {
            Self::DenseBf16 => WeightKernelAlignment {
                weight_layout: "bf16.row_major",
                scale_layout: "none",
                k_multiple: 1,
                n_multiple: 1,
                group_size: 0,
            },
            Self::W8A16 | Self::W4A16 | Self::W2A16 => WeightKernelAlignment {
                weight_layout: "wN.row_major.group_packed",
                scale_layout: "bf16[row, k/group_size]",
                k_multiple: group_size.max(1),
                n_multiple: 1,
                group_size,
            },
            Self::GgufQ3K | Self::GgufQ4K | Self::GgufQ5K | Self::GgufQ6K => {
                WeightKernelAlignment {
                    weight_layout: "gguf.qk.row_major.superblock256",
                    scale_layout: "embedded.superblock",
                    k_multiple: 256,
                    n_multiple: 1,
                    group_size: 256,
                }
            }
            Self::TurboQuant => WeightKernelAlignment {
                weight_layout: "turboquant.row_major.group_packed",
                scale_layout: "fp16[row, k/group_size]",
                k_multiple: group_size.max(1),
                n_multiple: 1,
                group_size,
            },
        }
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
        }
    }
}

impl std::fmt::Display for WeightFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DenseBf16 => f.write_str("dense_bf16"),
            Self::W8A16 => f.write_str("w8a16"),
            Self::W4A16 => f.write_str("w4a16"),
            Self::W2A16 => f.write_str("w2a16"),
            Self::GgufQ3K => f.write_str("gguf_q3_k"),
            Self::GgufQ4K => f.write_str("gguf_q4_k"),
            Self::GgufQ5K => f.write_str("gguf_q5_k"),
            Self::GgufQ6K => f.write_str("gguf_q6_k"),
            Self::TurboQuant => f.write_str("turboquant"),
        }
    }
}

/// 2D device tensor (matrix) — stored in row-major order as bf16 unless
/// `weight_format` names an explicit packed layout.
pub struct DeviceMatrix {
    pub data: CudaSlice<bf16>,
    pub rows: usize,
    pub cols: usize,
    pub weight_format: WeightFormat,
    /// INT8 quantized weights (if quantized). When set, `data` is unused.
    pub qweight: Option<CudaSlice<i8>>,
    /// Per-group bf16 scales for quantized weights. Shape: [rows, cols/group_size].
    pub qscales: Option<CudaSlice<bf16>>,
    /// Quantization group size (0 = not quantized).
    pub group_size: usize,
    /// Marlin-repacked INT4 weights for prefill GEMM (None if not W4 or repack failed).
    pub marlin_packed: Option<CudaSlice<u8>>,
    /// FP16 scales in Marlin layout [K/group_size, N] (transposed from qscales).
    pub marlin_scales: Option<CudaSlice<u16>>,
    // -- TurboQuant packed weight storage (Phase 2: fused dequant at runtime) --
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

impl DeviceMatrix {
    /// Create from host data (row-major, bf16)
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
            qscales: None,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    /// Create from INT8 quantized weight + bf16 scales.
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
            qscales: Some(qs),
            group_size,
            marlin_packed: None,
            marlin_scales: None,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    /// Create from INT4 packed quantized weight + bf16 scales.
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
            qscales: Some(qs),
            group_size,
            marlin_packed: None,
            marlin_scales: None,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    /// Create from Q6_K packed GGUF superblocks.
    ///
    /// Each 256-element superblock is 210 bytes: ql(128)|qh(64)|scales(16×i8)|d(f16).
    /// Per-row byte stride = `(cols/256) * 210`.
    pub fn from_quantized_q6k(
        ctx: &DeviceContext,
        packed_bytes: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        WeightFormat::GgufQ6K.validate_shape(rows, cols, 256)?;
        let expected = rows * cols * 210 / 256;
        ensure!(
            packed_bytes.len() == expected,
            "Q6_K packed size {} != expected {} for rows={} cols={}",
            packed_bytes.len(),
            expected,
            rows,
            cols
        );

        let qw: CudaSlice<i8> = ctx
            .stream
            .clone_htod(unsafe {
                std::slice::from_raw_parts(packed_bytes.as_ptr().cast::<i8>(), packed_bytes.len())
            })
            .map_err(|e| anyhow!("H2D Q6_K packed upload failed: {}", e))?;
        let dummy_scales: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc Q6_K dummy scales: {}", e))?;
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {}", e))?;
        Ok(Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::GgufQ6K,
            qweight: Some(qw),
            qscales: Some(dummy_scales),
            group_size: 256,
            marlin_packed: None,
            marlin_scales: None,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    /// Create from Q3_K packed GGUF superblocks.
    ///
    /// Each 256-element superblock is 110 bytes: hmask(32)|qs(64)|scales(12)|d(f16).
    /// Per-row byte stride = `(cols/256) * 110 = cols * 55/128`.
    pub fn from_quantized_q3k(
        ctx: &DeviceContext,
        packed_bytes: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        WeightFormat::GgufQ3K.validate_shape(rows, cols, 256)?;
        let expected = rows * cols * 55 / 128; // (cols/256) * 110 per row
        ensure!(
            packed_bytes.len() == expected,
            "Q3_K packed size {} != expected {} for rows={} cols={}",
            packed_bytes.len(),
            expected,
            rows,
            cols
        );

        let qw: CudaSlice<i8> = ctx
            .stream
            .clone_htod(unsafe {
                std::slice::from_raw_parts(packed_bytes.as_ptr().cast::<i8>(), packed_bytes.len())
            })
            .map_err(|e| anyhow!("H2D Q3_K packed upload failed: {}", e))?;
        let dummy_scales: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc Q3_K dummy scales: {}", e))?;
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {}", e))?;
        Ok(Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::GgufQ3K,
            qweight: Some(qw),
            qscales: Some(dummy_scales),
            group_size: 256,
            marlin_packed: None,
            marlin_scales: None,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    /// Create from Q4_K_M/Q4_K_S packed GGUF superblocks.
    ///
    /// Uploads the raw 144-byte superblock bytes verbatim to the GPU — no BF16
    /// intermediate ever materialises. One row consists of `cols/256` contiguous
    /// superblocks, so the per-row byte stride is `(cols/256)*144 = cols*9/16`.
    ///
    /// `weight_format` is set to `GgufQ4K` so dispatch can distinguish this
    /// embedded-scale superblock layout from uniform-group W4A16. `group_size`
    /// is set to 256 (superblock size) for informational purposes; the kernel
    /// decodes scales per superblock.
    pub fn from_quantized_q4k(
        ctx: &DeviceContext,
        packed_bytes: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        WeightFormat::GgufQ4K.validate_shape(rows, cols, 256)?;
        let expected = rows * cols * 9 / 16; // (cols/256) * 144 per row
        ensure!(
            packed_bytes.len() == expected,
            "Q4_K packed size {} != expected {} for rows={} cols={}",
            packed_bytes.len(),
            expected,
            rows,
            cols
        );

        let qw: CudaSlice<i8> = ctx
            .stream
            .clone_htod(unsafe {
                std::slice::from_raw_parts(packed_bytes.as_ptr().cast::<i8>(), packed_bytes.len())
            })
            .map_err(|e| anyhow!("H2D Q4_K packed upload failed: {}", e))?;
        let dummy_scales: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc Q4_K dummy scales: {}", e))?;
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {}", e))?;
        Ok(Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::GgufQ4K,
            qweight: Some(qw),
            qscales: Some(dummy_scales),
            group_size: 256,
            marlin_packed: None,
            marlin_scales: None,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    /// Create from Q5_K packed GGUF superblocks.
    ///
    /// Each 256-element superblock is 176 bytes:
    /// d(2)|dmin(2)|scales(12)|qh(32)|qs(128).
    pub fn from_quantized_q5k(
        ctx: &DeviceContext,
        packed_bytes: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        WeightFormat::GgufQ5K.validate_shape(rows, cols, 256)?;
        let expected = rows * cols * 11 / 16; // (cols/256) * 176 per row
        ensure!(
            packed_bytes.len() == expected,
            "Q5_K packed size {} != expected {} for rows={} cols={}",
            packed_bytes.len(),
            expected,
            rows,
            cols
        );

        let qw: CudaSlice<i8> = ctx
            .stream
            .clone_htod(unsafe {
                std::slice::from_raw_parts(packed_bytes.as_ptr().cast::<i8>(), packed_bytes.len())
            })
            .map_err(|e| anyhow!("H2D Q5_K packed upload failed: {}", e))?;
        let dummy_scales: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc Q5_K dummy scales: {}", e))?;
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {}", e))?;
        Ok(Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::GgufQ5K,
            qweight: Some(qw),
            qscales: Some(dummy_scales),
            group_size: 256,
            marlin_packed: None,
            marlin_scales: None,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    /// Create from INT2 packed quantized weight + bf16 scales.
    /// Weight data is packed: 4 int2 values per byte → [rows, cols/4] bytes.
    pub fn from_quantized_int2(
        ctx: &DeviceContext,
        packed_data: &[u8],
        scales_data: &[bf16],
        rows: usize,
        cols: usize,
        group_size: usize,
    ) -> Result<Self> {
        WeightFormat::W2A16.validate_shape(rows, cols, group_size)?;
        ensure!(
            cols.is_multiple_of(4),
            "W2A16 requires cols % 4 == 0, got {cols}"
        );
        ensure!(packed_data.len() == rows * cols / 4);
        let num_groups = cols / group_size;
        ensure!(scales_data.len() == rows * num_groups);

        // Upload packed data directly (native W2 kernel handles bit extraction)
        let qw: CudaSlice<i8> = ctx
            .stream
            .clone_htod(unsafe {
                std::slice::from_raw_parts(packed_data.as_ptr().cast::<i8>(), packed_data.len())
            })
            .map_err(|e| anyhow!("H2D qweight int2 failed: {}", e))?;
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
            weight_format: WeightFormat::W2A16,
            qweight: Some(qw),
            qscales: Some(qs),
            group_size,
            marlin_packed: None,
            marlin_scales: None,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    /// Whether this matrix uses quantized weights.
    pub fn is_quantized(&self) -> bool {
        self.weight_format.is_quantized() && (self.qweight.is_some() || self.tq_packed.is_some())
    }

    /// Whether this matrix is plain BF16 with no packed side buffers.
    pub fn is_dense_bf16(&self) -> bool {
        self.weight_format == WeightFormat::DenseBf16
            && self.qweight.is_none()
            && self.tq_packed.is_none()
    }

    #[must_use]
    pub fn weight_format(&self) -> WeightFormat {
        self.weight_format
    }

    /// Whether this matrix has Marlin-repacked weights for fast prefill GEMM.
    pub fn has_marlin(&self) -> bool {
        self.marlin_packed.is_some()
    }

    /// Whether this matrix uses TurboQuant packed weight storage.
    pub fn has_tq(&self) -> bool {
        self.tq_packed.is_some()
    }

    /// Create from TurboQuant packed weights on GPU.
    ///
    /// Weights stay packed at runtime; dequant happens in the fused GEMV kernel
    /// (decode) or via bulk dequant + cuBLAS GEMM (prefill).
    #[allow(clippy::too_many_arguments)]
    pub fn from_quantized_tq(
        ctx: &DeviceContext,
        packed: &[u8],
        scales: &[u8], // f16 as raw bytes
        signs: &[i8],
        centroids: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
        group_size: usize,
        bits: u8,
    ) -> Result<Self> {
        let tq_p = ctx
            .stream
            .clone_htod(packed)
            .map_err(|e| anyhow!("H2D tq_packed failed: {}", e))?;
        let tq_s: CudaSlice<u16> = ctx
            .stream
            .clone_htod(unsafe {
                std::slice::from_raw_parts(scales.as_ptr().cast::<u16>(), scales.len() / 2)
            })
            .map_err(|e| anyhow!("H2D tq_scales failed: {}", e))?;
        let tq_sg = ctx
            .stream
            .clone_htod(signs)
            .map_err(|e| anyhow!("H2D tq_signs failed: {}", e))?;
        let tq_c = ctx
            .stream
            .clone_dtod(centroids)
            .map_err(|e| anyhow!("D2D tq_centroids failed: {}", e))?;
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {}", e))?;
        Ok(Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::TurboQuant,
            qweight: None,
            qscales: None,
            group_size,
            marlin_packed: None,
            marlin_scales: None,
            tq_packed: Some(tq_p),
            tq_scales: Some(tq_s),
            tq_signs: Some(tq_sg),
            tq_centroids: Some(tq_c),
            tq_bits: bits,
        })
    }

    /// Repack W4 weights to Marlin tile layout for fast prefill.
    /// Our format: [N, K/2] uint8 packed (lo/hi nibble = even/odd elements)
    /// Marlin format: tiled int32 layout optimized for tensor core MMA.
    /// Also transposes scales from [N, K/group_size] bf16 → [K/group_size, N] fp16.
    pub fn repack_for_marlin(&mut self, ctx: &DeviceContext) -> Result<()> {
        if self.weight_format != WeightFormat::W4A16
            || self.qweight.is_none()
            || self.qscales.is_none()
        {
            return Ok(()); // Only for W4
        }
        let n = self.rows; // output dim
        let k = self.cols; // input dim

        // Skip if dimensions not Marlin-compatible (need K%16==0, N%64==0)
        if !k.is_multiple_of(16) || !n.is_multiple_of(64) {
            log::warn!("Marlin repack skipped: [{n}x{k}] not tile-aligned (need K%16==0, N%64==0)");
            return Ok(());
        }

        // Step 1: Convert our [N, K/2] uint8 → GPTQ [K/8, N] int32 on CPU
        let qw = self.qweight.as_ref().unwrap();
        let packed_host: Vec<i8> = ctx
            .stream
            .clone_dtoh(qw)
            .map_err(|e| anyhow!("D2H qweight: {}", e))?;
        let packed: &[u8] = unsafe {
            std::slice::from_raw_parts(packed_host.as_ptr().cast::<u8>(), packed_host.len())
        };

        // GPTQ format: qweight[k/8, n] = 8 nibbles packed into int32
        // bit position (k%8)*4 holds the 4-bit unsigned value for element (k, n)
        let gptq_rows = k / 8;
        let mut gptq = vec![0u32; gptq_rows * n];
        for row_n in 0..n {
            for col_k in 0..k {
                let byte_idx = row_n * (k / 2) + col_k / 2;
                let nibble = if col_k % 2 == 0 {
                    packed[byte_idx] & 0x0F
                } else {
                    packed[byte_idx] >> 4
                };
                let gptq_row = col_k / 8;
                let bit_pos = (col_k % 8) * 4;
                gptq[gptq_row * n + row_n] |= (nibble as u32) << bit_pos;
            }
        }

        // Upload GPTQ weights as raw bytes
        let gptq_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(gptq.as_ptr().cast::<u8>(), gptq.len() * 4) };
        let gptq_gpu: CudaSlice<u8> = ctx
            .stream
            .clone_htod(gptq_bytes)
            .map_err(|e| anyhow!("H2D GPTQ: {}", e))?;

        // Allocate Marlin output buffer (same byte count as GPTQ: K*N/2 bytes)
        let marlin_bytes = k * n / 2;
        let mut marlin_gpu: CudaSlice<u8> = ctx
            .stream
            .alloc_zeros(marlin_bytes)
            .map_err(|e| anyhow!("Alloc Marlin: {}", e))?;

        // Step 2: GPTQ → Marlin repack on GPU
        {
            let (gptq_ptr, _g1) = gptq_gpu.device_ptr(&ctx.stream);
            let (marlin_ptr, _g2) = marlin_gpu.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::gptq_marlin_repack_cuda(
                    gptq_ptr as *const u32,
                    marlin_ptr as *mut u32,
                    k as i32,
                    n as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("Marlin repack failed: {:?}", e))?;
            }
        }

        // Step 3: Transpose + convert scales [N, K/gs] bf16 → [K/gs, N] fp16
        let qs = self.qscales.as_ref().unwrap();
        let scales_host: Vec<bf16> = ctx
            .stream
            .clone_dtoh(qs)
            .map_err(|e| anyhow!("D2H scales: {}", e))?;
        let num_groups = k / self.group_size;
        let mut scales_fp16 = vec![0u16; num_groups * n];
        for row_n in 0..n {
            for g in 0..num_groups {
                let bf = scales_host[row_n * num_groups + g];
                let f = f32::from(bf);
                let fp16 = half::f16::from_f32(f);
                scales_fp16[g * n + row_n] = fp16.to_bits();
            }
        }
        let scales_gpu: CudaSlice<u16> = ctx
            .stream
            .clone_htod(&scales_fp16)
            .map_err(|e| anyhow!("H2D Marlin scales: {}", e))?;

        self.marlin_packed = Some(marlin_gpu);
        self.marlin_scales = Some(scales_gpu);

        Ok(())
    }

    #[allow(clippy::cast_ptr_alignment)]
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
        // NOTE: This assumes a little-endian host. Safetensors are little-endian.
        // On a big-endian machine, this will be incorrect. A full solution would
        // involve byte-swapping.
        let slice =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<bf16>(), rows * cols) };
        let gpu_data = ctx
            .stream
            .clone_htod(slice)
            .map_err(|e| anyhow!("H2D copy failed: {}", e))?;
        Ok(Self {
            data: gpu_data,
            rows,
            cols,
            weight_format: WeightFormat::DenseBf16,
            qweight: None,
            qscales: None,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    /// Extract a contiguous range of rows `[row_start..row_end)` as a new `DeviceMatrix`.
    /// The result is an independent copy on the GPU.
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
            qscales: None,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    /// Concatenate multiple matrices vertically (stacking rows).
    /// All matrices must have the same number of columns.
    /// Result has rows = sum of all input rows, cols = shared cols.
    pub fn concat_rows(ctx: &DeviceContext, matrices: &[&DeviceMatrix]) -> Result<Self> {
        assert!(!matrices.is_empty(), "concat_rows: empty input");
        let cols = matrices[0].cols;
        for m in matrices {
            assert_eq!(m.cols, cols, "concat_rows: cols mismatch");
        }
        let total_rows: usize = matrices.iter().map(|m| m.rows).sum();

        // Quantized weights use separate GEMVs (not merged), so skip the
        // expensive bf16 concat — just allocate a 1-element dummy.
        if matrices[0].is_quantized() {
            let dummy = ctx
                .stream
                .alloc_zeros::<bf16>(1)
                .map_err(|e| anyhow!("concat_rows dummy alloc: {e}"))?;
            return Ok(Self {
                data: dummy,
                rows: total_rows,
                cols,
                weight_format: WeightFormat::DenseBf16,
                qweight: None,
                qscales: None,
                group_size: 0,
                marlin_packed: None,
                marlin_scales: None,
                tq_packed: None,
                tq_scales: None,
                tq_signs: None,
                tq_centroids: None,
                tq_bits: 0,
            });
        }

        let total_elements = total_rows * cols;
        let mut merged: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(total_elements)
            .map_err(|e| anyhow!("concat_rows alloc failed: {e}"))?;

        let mut offset = 0usize;
        for m in matrices {
            let n = m.rows * m.cols;
            ctx.stream
                .memcpy_dtod(&m.data, &mut merged.slice_mut(offset..offset + n))
                .map_err(|e| anyhow!("concat_rows D2D copy failed: {e}"))?;
            offset += n;
        }

        Ok(Self {
            data: merged,
            rows: total_rows,
            cols,
            weight_format: WeightFormat::DenseBf16,
            qweight: None,
            qscales: None,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }
}

/// Batched hidden states: seq_len vectors of dim hidden_dim, stored contiguously.
/// Memory layout: [hidden_dim * seq_len] elements, token i at offset i * hidden_dim.
/// cuBLAS interprets as [hidden_dim, seq_len] column-major.
pub struct HiddenStates {
    pub data: CudaSlice<bf16>,
    pub hidden_dim: usize,
    pub seq_len: usize,
}

impl HiddenStates {
    /// Create zeroed batch
    pub fn zeros(ctx: &DeviceContext, hidden_dim: usize, seq_len: usize) -> Result<Self> {
        let data: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(hidden_dim * seq_len)
            .map_err(|e| anyhow!("Alloc failed: {}", e))?;
        Ok(Self {
            data,
            hidden_dim,
            seq_len,
        })
    }
}

/// Cached raw CUDA device pointer for a pre-allocated buffer.
///
/// Avoids per-call overhead of cudarc's `device_ptr()` / `device_ptr_mut()`
/// which perform atomic loads + SyncOnDrop bookkeeping even when event tracking
/// is disabled.
///
/// # Safety invariants
/// - The originating CudaSlice must outlive all uses of this pointer.
/// - The originating CudaSlice must not be reallocated.
/// - Only used from the single inference thread (single CUDA stream).
#[derive(Debug, Clone, Copy)]
pub struct RawDevicePtr<T> {
    ptr: u64,
    _marker: PhantomData<*const T>,
}

// SAFETY: RawDevicePtr is only used from the single inference thread.
unsafe impl<T> Send for RawDevicePtr<T> {}

impl<T> RawDevicePtr<T> {
    /// Get as const pointer for kernel read parameters.
    pub fn as_ptr(self) -> *const T {
        self.ptr as *const T
    }

    /// Get as mut pointer for kernel write parameters.
    pub fn as_mut_ptr(self) -> *mut T {
        self.ptr as *mut T
    }
}

/// Extract and cache a raw device pointer from a CudaSlice.
/// Calls device_ptr() once -- amortized over thousands of decode steps.
pub fn cache_ptr<T>(slice: &CudaSlice<T>, ctx: &DeviceContext) -> RawDevicePtr<T> {
    use cudarc::driver::DevicePtr;
    let (ptr, _sync) = slice.device_ptr(&ctx.stream);
    RawDevicePtr {
        ptr,
        _marker: PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_device_ordinal_handles_unset_default_and_invalid() {
        assert_eq!(parse_device_ordinal(None).unwrap(), 0);
        assert_eq!(parse_device_ordinal(Some("3")).unwrap(), 3);
        assert_eq!(parse_device_ordinal(Some("  7 ")).unwrap(), 7);
        assert!(parse_device_ordinal(Some("not-a-number")).is_err());
        assert!(parse_device_ordinal(Some("")).is_err());
    }

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
    fn kernel_alignment_names_scale_layout_explicitly() {
        let w4 = WeightFormat::W4A16.kernel_alignment(128);
        assert_eq!(w4.weight_layout, "wN.row_major.group_packed");
        assert_eq!(w4.scale_layout, "bf16[row, k/group_size]");
        assert_eq!(w4.k_multiple, 128);

        let q4k = WeightFormat::GgufQ4K.kernel_alignment(256);
        assert_eq!(q4k.weight_layout, "gguf.qk.row_major.superblock256");
        assert_eq!(q4k.scale_layout, "embedded.superblock");
        assert_eq!(q4k.k_multiple, 256);
    }

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
