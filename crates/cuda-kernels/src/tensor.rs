//! Device tensor types and CUDA context.

use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use std::borrow::Cow;
use std::marker::PhantomData;

#[path = "tensor/device_context.rs"]
mod device_context;
#[path = "tensor/device_matrix.rs"]
mod device_matrix;
#[path = "tensor/weight_format.rs"]
mod weight_format;
pub use device_context::*;
pub use device_matrix::*;
pub use weight_format::*;

pub(super) fn bf16_safetensor_host_slice(data: &[u8]) -> Result<Cow<'_, [bf16]>> {
    ensure!(
        data.len().is_multiple_of(2),
        "Data length must be even for bf16: got {} bytes",
        data.len()
    );
    // Safetensors are little-endian. If a mmap-backed tensor starts at an
    // unaligned byte offset, casting `u8*` to `bf16*` would be undefined
    // behavior; fall back to a small decode buffer only for that case.
    // SAFETY: bf16 is a 2-byte POD for which every bit pattern is valid;
    // `align_to` itself confines the reinterpret to the correctly-aligned
    // middle, and the unaligned prefix/suffix case falls back to a decode copy.
    let (prefix, aligned, suffix) = unsafe { data.align_to::<bf16>() };
    if prefix.is_empty() && suffix.is_empty() {
        return Ok(Cow::Borrowed(aligned));
    }
    Ok(Cow::Owned(
        data.as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16::from_le_bytes(*c))
            .collect(),
    ))
}

/// 1D device tensor (vector) — stored as bf16.
pub struct DeviceVec {
    pub data: CudaSlice<bf16>,
    pub len: usize,
    /// Debug label describing the tensor's semantic shape (e.g., `norm_weight[hidden]`, `kv_cache[heads,seq,dim]`).
    pub label: &'static str,
}

impl DeviceVec {
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

    pub fn from_safetensors(ctx: &DeviceContext, data: &[u8]) -> Result<Self> {
        let slice = bf16_safetensor_host_slice(data)?;
        Self::from_host(ctx, slice.as_ref())
    }

    #[track_caller]
    pub fn zeros(ctx: &DeviceContext, len: usize) -> Result<Self> {
        let gpu_data: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(len)
            .map_err(|e| anyhow!("Alloc failed: {}", e))?;
        record_cuda_alloc::<bf16>("alloc_zeros", "DeviceVec::zeros", len);
        Ok(Self {
            data: gpu_data,
            len,
            label: "",
        })
    }

    /// Create an UNINITIALIZED tensor (no zeroing memset).
    ///
    /// # Safety
    /// The buffer holds uninitialized device memory; every element must be
    /// written before it is read.
    #[track_caller]
    pub unsafe fn uninit(ctx: &DeviceContext, len: usize) -> Result<Self> {
        // SAFETY: forwards the uninitialized-memory contract to our caller.
        let gpu_data: CudaSlice<bf16> = unsafe {
            ctx.stream
                .alloc(len)
                .map_err(|e| anyhow!("Alloc failed: {}", e))?
        };
        record_cuda_alloc::<bf16>("alloc", "DeviceVec::uninit", len);
        Ok(Self {
            data: gpu_data,
            len,
            label: "",
        })
    }

    /// Useful for dummy RMSNorm weights (identity normalization).
    pub fn ones(ctx: &DeviceContext, len: usize) -> Result<Self> {
        let host = vec![bf16::ONE; len];
        Self::from_host(ctx, &host)
    }

    /// Move the device buffer to host RAM and free the VRAM (OPD time-share).
    ///
    /// Returns the host bytes plus the device bytes freed; the live buffer is
    /// replaced with a 1-element placeholder.
    pub fn offload_to_host(&mut self, ctx: &DeviceContext) -> Result<(Vec<bf16>, usize)> {
        let host = ctx
            .stream
            .clone_dtoh(&self.data)
            .map_err(|e| anyhow!("offload D2H copy (vec) failed: {e}"))?;
        let freed = host.len() * std::mem::size_of::<bf16>();
        ctx.sync()?;
        self.data = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("offload vec placeholder alloc failed: {e}"))?;
        Ok((host, freed))
    }

    pub fn reload_from_host(&mut self, ctx: &DeviceContext, host: &[bf16]) -> Result<()> {
        self.data = ctx
            .stream
            .clone_htod(host)
            .map_err(|e| anyhow!("reload H2D copy (vec) failed: {e}"))?;
        ctx.sync()?;
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

/// Batched hidden states: seq_len vectors of dim hidden_dim, stored contiguously.
/// Memory layout: [hidden_dim * seq_len] elements, token i at offset i * hidden_dim.
/// cuBLAS interprets as [hidden_dim, seq_len] column-major.
pub struct HiddenStates {
    pub data: CudaSlice<bf16>,
    pub hidden_dim: usize,
    pub seq_len: usize,
}

impl HiddenStates {
    #[track_caller]
    pub fn zeros(ctx: &DeviceContext, hidden_dim: usize, seq_len: usize) -> Result<Self> {
        let len = hidden_dim * seq_len;
        let data: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(len)
            .map_err(|e| anyhow!("Alloc failed: {}", e))?;
        record_cuda_alloc::<bf16>("alloc_zeros", "HiddenStates::zeros", len);
        Ok(Self {
            data,
            hidden_dim,
            seq_len,
        })
    }

    /// Create an uninitialized batch for call sites that immediately overwrite
    /// every element with a CUDA kernel.
    ///
    /// # Safety
    ///
    /// The returned buffer must not be read before all `hidden_dim * seq_len`
    /// elements have been written by a kernel or device copy.
    #[track_caller]
    pub unsafe fn uninit(ctx: &DeviceContext, hidden_dim: usize, seq_len: usize) -> Result<Self> {
        let len = hidden_dim * seq_len;
        // SAFETY: forwards the uninitialized-memory contract to our caller per
        // this method's `# Safety` doc (must be fully written before any read).
        let data: CudaSlice<bf16> = unsafe {
            ctx.stream
                .alloc(len)
                .map_err(|e| anyhow!("Alloc failed: {}", e))?
        };
        record_cuda_alloc::<bf16>("alloc", "HiddenStates::uninit", len);
        Ok(Self {
            data,
            hidden_dim,
            seq_len,
        })
    }

    /// Exact requested device bytes this buffer owns:
    /// `data.len() * size_of::<bf16>()`. Read-only accounting for the DSv4
    /// VRAM ledger.
    pub fn device_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<bf16>()
    }

    /// Copy to host as f32, token-major `[seq_len, hidden_dim]`.
    pub fn to_host(&self, ctx: &DeviceContext) -> Result<Vec<f32>> {
        let host = ctx
            .stream
            .clone_dtoh(&self.data)
            .map_err(|e| anyhow!("D2H copy failed: {}", e))?;
        ctx.sync()?;
        Ok(host.iter().map(|x| x.to_f32()).collect())
    }

    /// Borrowed view over the whole buffer (`seq_len` columns).
    pub fn as_view(&self) -> HiddenStatesView<'_> {
        HiddenStatesView {
            data: self.data.slice(..),
            hidden_dim: self.hidden_dim,
            seq_len: self.seq_len,
        }
    }

    /// Borrowed view of column `r` (`[hidden_dim, 1]`). Same device address +
    /// length the per-row D2D copy would have produced — read-only, zero copy.
    pub fn col(&self, r: usize) -> HiddenStatesView<'_> {
        let w = self.hidden_dim;
        HiddenStatesView {
            data: self.data.slice(r * w..(r + 1) * w),
            hidden_dim: w,
            seq_len: 1,
        }
    }
}

/// Borrowed column view into a contiguous `[hidden_dim, seq_len]` [`HiddenStates`].
/// Feeds the identical device pointer the per-row D2D copy produced → bit-identical
/// reads, zero copy. Read-only.
pub struct HiddenStatesView<'a> {
    pub data: cudarc::driver::CudaView<'a, bf16>,
    pub hidden_dim: usize,
    pub seq_len: usize,
}

impl<'a> HiddenStatesView<'a> {
    /// Reborrow this view at the same span, preserving lifetime `'a`
    /// (cudarc `CudaView::slice(..)` returns `Self`). Lets owned and borrowed
    /// indexer-query sources be unified to one `HiddenStatesView` value.
    pub fn as_self_view(&self) -> HiddenStatesView<'a> {
        HiddenStatesView {
            data: self.data.slice(..),
            hidden_dim: self.hidden_dim,
            seq_len: self.seq_len,
        }
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
    /// Wrap an already-extracted device address (a `device_ptr()` value whose
    /// guard the caller still holds). The caller asserts the originating
    /// allocation holds `T` and outlives all uses of this pointer.
    pub fn from_raw(ptr: u64) -> RawDevicePtr<T> {
        RawDevicePtr {
            ptr,
            _marker: PhantomData,
        }
    }

    pub fn as_ptr(self) -> *const T {
        self.ptr as *const T
    }

    pub fn as_mut_ptr(self) -> *mut T {
        self.ptr as *mut T
    }

    /// Reinterpret the device address as a pointer to `U`. The caller asserts
    /// the underlying bytes are a valid `[U]` (e.g. a `CudaSlice<u8>` byte
    /// buffer that actually holds bf16 weights). No allocation; just a u64 view.
    pub fn cast<U>(self) -> RawDevicePtr<U> {
        RawDevicePtr {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }

    /// Advance the pointer by `count` elements of `T` (`count * size_of::<T>()`
    /// bytes). The caller asserts the result stays within the backing slice.
    pub fn offset_elems(self, count: usize) -> RawDevicePtr<T> {
        RawDevicePtr {
            ptr: self.ptr + (count * std::mem::size_of::<T>()) as u64,
            _marker: PhantomData,
        }
    }
}

/// Calls device_ptr() once -- amortized over thousands of decode steps.
pub fn cache_ptr<T>(slice: &CudaSlice<T>, ctx: &DeviceContext) -> RawDevicePtr<T> {
    cache_ptr_on(slice, &ctx.stream)
}

/// `cache_ptr` for callers that hold a bare stream (the autograd backend) rather
/// than a full `DeviceContext`. Same one-shot device_ptr extraction.
pub fn cache_ptr_on<T>(slice: &CudaSlice<T>, stream: &CudaStream) -> RawDevicePtr<T> {
    use cudarc::driver::DevicePtr;
    let (ptr, _sync) = slice.device_ptr(stream);
    RawDevicePtr {
        ptr,
        _marker: PhantomData,
    }
}

/// A null [`RawDevicePtr`] — for optional kernel tables the kernel treats as
/// absent (e.g. `expert_indices` when the compact index is the expert index).
pub fn null_raw_ptr<T>() -> RawDevicePtr<T> {
    RawDevicePtr {
        ptr: 0,
        _marker: PhantomData,
    }
}

/// FP8 E4M3FN → f32 (host twin of `dsv4_decode_fp8_e4m3`; 0x7f/0xff is the
/// format's NaN and is clamped to +/-448, matching the device decoder).
pub(super) fn e4m3_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exp = i32::from((bits >> 3) & 0x0f);
    let mant = f32::from(bits & 0x07);
    if bits & 0x7f == 0x7f {
        return sign * 448.0;
    }
    if exp == 0 {
        return sign * (mant / 8.0) * 2.0_f32.powi(-6);
    }
    sign * (1.0 + mant / 8.0) * 2.0_f32.powi(exp - 7)
}
