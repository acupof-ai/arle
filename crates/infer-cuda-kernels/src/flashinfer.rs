//! FlashInfer decode metadata management.
//!
//! Owns the GPU-side paged KV metadata buffers (positions, indptr, indices,
//! last_page_len) and the FlashInfer workspace. Extracted from
//! `model::qwen3::batch_decode` so multiple model implementations can share it.

use anyhow::{Result, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::ffi;
use super::paged_kv::PagedKVPool;
use super::tensor::DeviceContext;

/// GPU-side metadata buffers for FlashInfer batched decode attention.
///
/// Manages positions, KV indptr/indices/last_page_len arrays, the FlashInfer
/// workspace, and host-side scratch buffers for incremental index updates.
pub struct FlashInferDecodeMetadata {
    pub positions: CudaSlice<i32>,
    pub kv_indices: CudaSlice<i32>,
    pub kv_indptr: CudaSlice<i32>,
    pub kv_last_page_len: CudaSlice<i32>,
    /// Q indptr for tensor-core decode: [0, 1, 2, ..., B] (1 token per request).
    pub qo_indptr: CudaSlice<i32>,
    pub flashinfer_ws: FlashInferWorkspace,
    pub max_total_pages: usize,
    /// Scratch buffer for building positions on the host side.
    positions_scratch: Vec<i32>,
    /// Scratch buffer for KV index H2D.
    indices_scratch: Vec<i32>,
    /// Cached host-side indptr from last `update()`, reused by `plan()`.
    indptr_h: Vec<i32>,
    /// Previous slot indices that produced `indices_scratch`.
    prev_slot_indices: Vec<usize>,
    /// Previous slot epochs; changes when a slot is recycled for a new request.
    prev_slot_epochs: Vec<u64>,
    /// Host-side q_indptr for TC decode: [0, 1, 2, ..., max_batch_size].
    pub qo_indptr_h: Vec<i32>,
    /// Physical page id of the last logical page per request [max_batch_size].
    /// Used only by the page_size=1 quantize-after-write fast paths.
    pub last_token_indices: CudaSlice<i32>,
    /// Host scratch for building last_token_indices.
    last_token_scratch: Vec<i32>,
    /// Total pages across all requests from last update.
    total_tokens: usize,
    /// Batch size from last successful plan (for plan caching).
    plan_batch_size: usize,
    /// Whether the plan needs to be re-run (batch composition changed).
    plan_dirty: bool,
}

/// Workspace buffers for FlashInfer batch decode, allocated once and reused.
///
/// FlashInfer's plan/run API needs:
/// - `float_workspace`: ~128MB GPU buffer for split-KV temp storage
/// - `int_workspace`: ~8MB GPU buffer for plan metadata
/// - `page_locked_workspace`: ~8MB CPU pinned buffer for plan H2D copy
/// - `plan_info`: ~256 bytes opaque buffer for plan output
pub struct FlashInferWorkspace {
    /// GPU scratch for split-KV temporaries (~128MB)
    pub float_workspace: CudaSlice<u8>,
    pub float_workspace_bytes: usize,
    /// GPU scratch for plan metadata (~8MB)
    pub int_workspace: CudaSlice<u8>,
    pub int_workspace_bytes: usize,
    /// CPU pinned buffer for plan H2D copy (~8MB)
    page_locked_workspace: *mut u8,
    #[allow(dead_code)]
    page_locked_workspace_bytes: usize,
    /// Opaque plan info buffer (256 bytes, HOST pinned — used by CPU memcpy in FlashInfer)
    pub plan_info: *mut u8,
    /// LSE output buffer (log-sum-exp), nullable but we allocate for max batch
    pub lse: CudaSlice<f32>,
}

// SAFETY: The page_locked_workspace is a pinned CPU allocation that is only accessed
// from the single inference thread (same thread that calls plan/run). No concurrent access.
unsafe impl Send for FlashInferWorkspace {}

fn alloc_host_buffer(bytes: usize, label: &str) -> Result<*mut u8> {
    unsafe {
        let mut ptr: *mut u8 = std::ptr::null_mut();
        let err = cudarc::driver::sys::cuMemAllocHost_v2(
            &mut ptr as *mut *mut u8 as *mut *mut std::ffi::c_void,
            bytes,
        );
        if err != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow::anyhow!(
                "cuMemAllocHost failed for {label}: {:?}",
                err
            ));
        }
        std::ptr::write_bytes(ptr, 0, bytes);
        Ok(ptr)
    }
}

fn free_host_buffer(ptr: &mut *mut u8) {
    if !ptr.is_null() {
        unsafe {
            let _ = cudarc::driver::sys::cuMemFreeHost((*ptr).cast::<std::ffi::c_void>());
        }
        *ptr = std::ptr::null_mut();
    }
}

pub struct PlanBuf {
    plan_info: *mut u8,
}

// SAFETY: plan_info is pinned host memory used from the same inference thread.
unsafe impl Send for PlanBuf {}

impl PlanBuf {
    fn new() -> Result<Self> {
        Ok(Self {
            plan_info: alloc_host_buffer(
                FlashInferWorkspace::PLAN_INFO_BYTES,
                "paged_prefill plan_info",
            )?,
        })
    }

    fn as_ptr(&self) -> *mut u8 {
        self.plan_info
    }
}

impl Drop for PlanBuf {
    fn drop(&mut self) {
        free_host_buffer(&mut self.plan_info);
    }
}

pub struct BatchPrefillPagedPlan {
    workspace: FlashInferWorkspace,
    pub hd128: PlanBuf,
    pub hd256: PlanBuf,
}

impl BatchPrefillPagedPlan {
    pub fn new(ctx: &DeviceContext, max_total_qo_rows: usize, num_qo_heads: usize) -> Result<Self> {
        Ok(Self {
            workspace: FlashInferWorkspace::new(ctx, max_total_qo_rows, num_qo_heads)?,
            hd128: PlanBuf::new()?,
            hd256: PlanBuf::new()?,
        })
    }

    fn plan_impl(
        &mut self,
        ctx: &DeviceContext,
        plan_info: *mut u8,
        qo_indptr: &[i32],
        kv_indptr: &[i32],
        batch_size: usize,
        num_qo_heads: usize,
        num_kv_heads: usize,
        page_size: usize,
        head_dim: usize,
        plan_fn: unsafe extern "C" fn(
            *mut u8,
            usize,
            *mut u8,
            *mut u8,
            usize,
            *const i32,
            *const i32,
            i32,
            i32,
            i32,
            i32,
            i32,
            *mut u8,
            cudarc::driver::sys::CUstream,
        ) -> i32,
        plan_name: &str,
    ) -> Result<()> {
        ensure!(
            qo_indptr.len() == batch_size + 1,
            "{plan_name}: qo_indptr len {} does not match batch_size {}",
            qo_indptr.len(),
            batch_size
        );
        ensure!(
            kv_indptr.len() == batch_size + 1,
            "{plan_name}: kv_indptr len {} does not match batch_size {}",
            kv_indptr.len(),
            batch_size
        );
        ensure!(qo_indptr[0] == 0, "{plan_name}: qo_indptr must start at 0");
        ensure!(kv_indptr[0] == 0, "{plan_name}: kv_indptr must start at 0");

        let (fw_ptr, _gfw) = self.workspace.float_workspace.device_ptr_mut(&ctx.stream);
        let (iw_ptr, _giw) = self.workspace.int_workspace.device_ptr_mut(&ctx.stream);
        let ret = unsafe {
            plan_fn(
                fw_ptr as *mut u8,
                self.workspace.float_workspace_bytes,
                iw_ptr as *mut u8,
                self.workspace.page_locked_workspace,
                self.workspace.int_workspace_bytes,
                qo_indptr.as_ptr(),
                kv_indptr.as_ptr(),
                batch_size as i32,
                num_qo_heads as i32,
                num_kv_heads as i32,
                page_size as i32,
                head_dim as i32,
                plan_info,
                ctx.stream.cu_stream(),
            )
        };
        if ret != 0 {
            return Err(anyhow::anyhow!("{plan_name} failed with CUDA error {ret}"));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_impl(
        &mut self,
        ctx: &DeviceContext,
        plan_info: *mut u8,
        q_ptr: u64,
        q_indptr_ptr: u64,
        k_data_ptr: u64,
        v_data_ptr: u64,
        kv_indptr_ptr: u64,
        kv_indices_ptr: u64,
        kv_last_page_len_ptr: u64,
        output_ptr: u64,
        lse_ptr: Option<u64>,
        batch_size: usize,
        num_qo_heads: usize,
        num_kv_heads: usize,
        page_size: usize,
        head_dim: usize,
        run_fn: unsafe extern "C" fn(
            *mut u8,
            *mut u8,
            *const u8,
            *mut ffi::Half,
            *const i32,
            *mut ffi::Half,
            *mut ffi::Half,
            *const i32,
            *const i32,
            *const i32,
            *mut ffi::Half,
            *mut f32,
            i32,
            i32,
            i32,
            i32,
            f32,
            cudarc::driver::sys::CUstream,
        ) -> i32,
        run_name: &str,
    ) -> Result<()> {
        let (fw_ptr, _gfw) = self.workspace.float_workspace.device_ptr_mut(&ctx.stream);
        let (iw_ptr, _giw) = self.workspace.int_workspace.device_ptr_mut(&ctx.stream);
        let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
        let ret = unsafe {
            run_fn(
                fw_ptr as *mut u8,
                iw_ptr as *mut u8,
                plan_info.cast_const(),
                q_ptr as *mut ffi::Half,
                q_indptr_ptr as *const i32,
                k_data_ptr as *mut ffi::Half,
                v_data_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                kv_last_page_len_ptr as *const i32,
                output_ptr as *mut ffi::Half,
                lse_ptr.map_or(std::ptr::null_mut(), |ptr| ptr as *mut f32),
                batch_size as i32,
                num_qo_heads as i32,
                num_kv_heads as i32,
                page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
        };
        if ret != 0 {
            return Err(anyhow::anyhow!("{run_name} failed with CUDA error {ret}"));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn plan_hd128(
        &mut self,
        ctx: &DeviceContext,
        qo_indptr: &[i32],
        kv_indptr: &[i32],
        batch_size: usize,
        num_qo_heads: usize,
        num_kv_heads: usize,
        page_size: usize,
    ) -> Result<()> {
        let plan_info = self.hd128.as_ptr();
        self.plan_impl(
            ctx,
            plan_info,
            qo_indptr,
            kv_indptr,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            page_size,
            128,
            ffi::flashinfer_batch_prefill_paged_hd128_plan,
            "flashinfer_batch_prefill_paged_hd128_plan",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn plan_hd256(
        &mut self,
        ctx: &DeviceContext,
        qo_indptr: &[i32],
        kv_indptr: &[i32],
        batch_size: usize,
        num_qo_heads: usize,
        num_kv_heads: usize,
        page_size: usize,
    ) -> Result<()> {
        let plan_info = self.hd256.as_ptr();
        self.plan_impl(
            ctx,
            plan_info,
            qo_indptr,
            kv_indptr,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            page_size,
            256,
            ffi::flashinfer_batch_prefill_paged_hd256_plan,
            "flashinfer_batch_prefill_paged_hd256_plan",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_hd128(
        &mut self,
        ctx: &DeviceContext,
        q_ptr: u64,
        q_indptr_ptr: u64,
        k_data_ptr: u64,
        v_data_ptr: u64,
        kv_indptr_ptr: u64,
        kv_indices_ptr: u64,
        kv_last_page_len_ptr: u64,
        output_ptr: u64,
        lse_ptr: Option<u64>,
        batch_size: usize,
        num_qo_heads: usize,
        num_kv_heads: usize,
        page_size: usize,
    ) -> Result<()> {
        let plan_info = self.hd128.as_ptr();
        self.run_impl(
            ctx,
            plan_info,
            q_ptr,
            q_indptr_ptr,
            k_data_ptr,
            v_data_ptr,
            kv_indptr_ptr,
            kv_indices_ptr,
            kv_last_page_len_ptr,
            output_ptr,
            lse_ptr,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            page_size,
            128,
            ffi::flashinfer_batch_prefill_paged_hd128_run,
            "flashinfer_batch_prefill_paged_hd128_run",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_hd256(
        &mut self,
        ctx: &DeviceContext,
        q_ptr: u64,
        q_indptr_ptr: u64,
        k_data_ptr: u64,
        v_data_ptr: u64,
        kv_indptr_ptr: u64,
        kv_indices_ptr: u64,
        kv_last_page_len_ptr: u64,
        output_ptr: u64,
        lse_ptr: Option<u64>,
        batch_size: usize,
        num_qo_heads: usize,
        num_kv_heads: usize,
        page_size: usize,
    ) -> Result<()> {
        let plan_info = self.hd256.as_ptr();
        self.run_impl(
            ctx,
            plan_info,
            q_ptr,
            q_indptr_ptr,
            k_data_ptr,
            v_data_ptr,
            kv_indptr_ptr,
            kv_indices_ptr,
            kv_last_page_len_ptr,
            output_ptr,
            lse_ptr,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            page_size,
            256,
            ffi::flashinfer_batch_prefill_paged_hd256_run,
            "flashinfer_batch_prefill_paged_hd256_run",
        )
    }
}

impl FlashInferWorkspace {
    /// Default sizes matching FlashInfer's typical requirements.
    /// 256MB for split-KV temporaries (sglang uses 512MB but we need headroom).
    const FLOAT_WORKSPACE_BYTES: usize = 256 * 1024 * 1024; // 256 MB
    const INT_WORKSPACE_BYTES: usize = 8 * 1024 * 1024; // 8 MB
    const PAGE_LOCKED_WORKSPACE_BYTES: usize = 8 * 1024 * 1024; // 8 MB
    const PLAN_INFO_BYTES: usize = 256;

    /// Allocate FlashInfer workspace buffers.
    ///
    /// `max_batch_size` controls the LSE buffer size.
    /// `num_qo_heads` is needed for LSE dimensioning.
    pub fn new(ctx: &DeviceContext, max_batch_size: usize, num_qo_heads: usize) -> Result<Self> {
        let float_workspace: CudaSlice<u8> = ctx
            .stream
            .alloc_zeros(Self::FLOAT_WORKSPACE_BYTES)
            .map_err(|e| anyhow::anyhow!("FlashInfer float_workspace alloc failed: {e}"))?;

        let int_workspace: CudaSlice<u8> = ctx
            .stream
            .alloc_zeros(Self::INT_WORKSPACE_BYTES)
            .map_err(|e| anyhow::anyhow!("FlashInfer int_workspace alloc failed: {e}"))?;

        let plan_info = alloc_host_buffer(Self::PLAN_INFO_BYTES, "plan_info")?;

        let lse: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(max_batch_size * num_qo_heads)
            .map_err(|e| anyhow::anyhow!("FlashInfer lse alloc failed: {e}"))?;

        let page_locked_workspace =
            alloc_host_buffer(Self::PAGE_LOCKED_WORKSPACE_BYTES, "page_locked_workspace")?;

        Ok(Self {
            float_workspace,
            float_workspace_bytes: Self::FLOAT_WORKSPACE_BYTES,
            int_workspace,
            int_workspace_bytes: Self::INT_WORKSPACE_BYTES,
            page_locked_workspace,
            page_locked_workspace_bytes: Self::PAGE_LOCKED_WORKSPACE_BYTES,
            plan_info,
            lse,
        })
    }
}

impl Drop for FlashInferWorkspace {
    fn drop(&mut self) {
        free_host_buffer(&mut self.page_locked_workspace);
        free_host_buffer(&mut self.plan_info);
    }
}

/// FlashInfer plan step only (CPU-side scheduling). Call once per decode step,
/// not per layer — the plan result works for all layers since KV layout is the same.
#[allow(clippy::too_many_arguments)]
pub fn flashinfer_plan(
    ctx: &DeviceContext,
    indptr_h: &[i32],
    workspace: &mut FlashInferWorkspace,
    batch_size: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    page_size: usize,
    head_dim: usize,
) -> Result<()> {
    let (fw_ptr, _gfw) = workspace.float_workspace.device_ptr_mut(&ctx.stream);
    let (iw_ptr, _giw) = workspace.int_workspace.device_ptr_mut(&ctx.stream);

    let ret = unsafe {
        ffi::flashinfer_batch_decode_plan(
            fw_ptr as *mut u8,
            workspace.float_workspace_bytes,
            iw_ptr as *mut u8,
            workspace.page_locked_workspace,
            workspace.int_workspace_bytes,
            indptr_h.as_ptr(),
            batch_size as i32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            page_size as i32,
            head_dim as i32,
            workspace.plan_info,
            ctx.stream.cu_stream(),
        )
    };
    if ret != 0 {
        return Err(anyhow::anyhow!(
            "flashinfer_batch_decode_plan failed with CUDA error {}",
            ret
        ));
    }
    Ok(())
}

/// FlashInfer tensor-core decode plan (uses prefill kernel for flat ITL).
///
/// Uses `PrefillPlan` instead of `DecodePlan`. For decode, each request has
/// `qo_len=1`, so `qo_indptr = [0, 1, 2, ..., B]`.
#[allow(clippy::too_many_arguments)]
pub fn flashinfer_tc_plan(
    ctx: &DeviceContext,
    qo_indptr_h: &[i32],
    kv_indptr_h: &[i32],
    workspace: &mut FlashInferWorkspace,
    batch_size: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    page_size: usize,
    head_dim: usize,
) -> Result<()> {
    let (fw_ptr, _gfw) = workspace.float_workspace.device_ptr_mut(&ctx.stream);
    let (iw_ptr, _giw) = workspace.int_workspace.device_ptr_mut(&ctx.stream);

    let ret = unsafe {
        ffi::flashinfer_tc_decode_plan(
            fw_ptr as *mut u8,
            workspace.float_workspace_bytes,
            iw_ptr as *mut u8,
            workspace.page_locked_workspace,
            workspace.int_workspace_bytes,
            qo_indptr_h.as_ptr(),
            kv_indptr_h.as_ptr(),
            batch_size as i32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            page_size as i32,
            head_dim as i32,
            workspace.plan_info,
            ctx.stream.cu_stream(),
        )
    };
    if ret != 0 {
        return Err(anyhow::anyhow!(
            "flashinfer_tc_decode_plan failed with CUDA error {}",
            ret
        ));
    }
    Ok(())
}

/// FlashInfer HD256 decode plan (Qwen3.5 full attention).
#[allow(clippy::too_many_arguments)]
pub fn flashinfer_plan_hd256(
    ctx: &DeviceContext,
    indptr_h: &[i32],
    workspace: &mut FlashInferWorkspace,
    batch_size: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    page_size: usize,
    head_dim: usize,
) -> Result<()> {
    let (fw_ptr, _gfw) = workspace.float_workspace.device_ptr_mut(&ctx.stream);
    let (iw_ptr, _giw) = workspace.int_workspace.device_ptr_mut(&ctx.stream);

    let ret = unsafe {
        ffi::flashinfer_batch_decode_hd256_plan(
            fw_ptr as *mut u8,
            workspace.float_workspace_bytes,
            iw_ptr as *mut u8,
            workspace.page_locked_workspace,
            workspace.int_workspace_bytes,
            indptr_h.as_ptr(),
            batch_size as i32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            page_size as i32,
            head_dim as i32,
            workspace.plan_info,
            ctx.stream.cu_stream(),
        )
    };
    if ret != 0 {
        return Err(anyhow::anyhow!(
            "flashinfer_batch_decode_hd256_plan failed with CUDA error {}",
            ret
        ));
    }
    Ok(())
}

impl FlashInferDecodeMetadata {
    /// Allocate metadata buffers for up to `max_batch_size` requests.
    /// `max_total_pages` should be large enough for the worst-case total KV pages.
    pub fn new(
        ctx: &DeviceContext,
        max_batch_size: usize,
        max_total_pages: usize,
        num_qheads: usize,
    ) -> Result<Self> {
        let qo_indptr: CudaSlice<i32> = ctx
            .stream
            .alloc_zeros(max_batch_size + 1)
            .map_err(|e| anyhow::anyhow!("Alloc qo_indptr failed: {e}"))?;

        Ok(Self {
            positions: ctx
                .stream
                .alloc_zeros(max_batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc positions failed: {e}"))?,
            kv_indices: ctx
                .stream
                .alloc_zeros(max_total_pages.max(1))
                .map_err(|e| anyhow::anyhow!("Alloc kv_indices failed: {e}"))?,
            kv_indptr: ctx
                .stream
                .alloc_zeros(max_batch_size + 1)
                .map_err(|e| anyhow::anyhow!("Alloc kv_indptr failed: {e}"))?,
            kv_last_page_len: ctx
                .stream
                .alloc_zeros(max_batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc kv_last_page_len failed: {e}"))?,
            qo_indptr,
            flashinfer_ws: FlashInferWorkspace::new(ctx, max_batch_size, num_qheads)?,
            max_total_pages,
            positions_scratch: Vec::with_capacity(max_batch_size),
            indices_scratch: Vec::with_capacity(max_total_pages.max(1)),
            indptr_h: Vec::with_capacity(max_batch_size + 1),
            prev_slot_indices: Vec::with_capacity(max_batch_size),
            prev_slot_epochs: Vec::with_capacity(max_batch_size),
            qo_indptr_h: vec![0i32; max_batch_size + 1],
            last_token_indices: ctx
                .stream
                .alloc_zeros(max_batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc last_token_indices failed: {e}"))?,
            last_token_scratch: Vec::with_capacity(max_batch_size),
            total_tokens: 0,
            plan_batch_size: 0,
            plan_dirty: true,
        })
    }

    /// Upload metadata (positions, indptr, last_page_len, KV indices) to GPU.
    ///
    /// Returns `true` if the kv_indices GPU buffer was reallocated (caller
    /// should invalidate any CUDA graph that captured the old pointer).
    pub fn update(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot_indices: &[usize],
    ) -> Result<bool> {
        let slot_epochs: Vec<u64> = slot_indices
            .iter()
            .map(|&slot| pool.slot_epoch(slot))
            .collect();
        let next_indptr_h = pool.build_indptr(slot_indices);
        let page_size = pool.page_size;
        let same_batch_identity =
            self.prev_slot_indices == slot_indices && self.prev_slot_epochs == slot_epochs;
        let can_incremental_update = page_size == 1
            && same_batch_identity
            && can_append_decode_step(&self.indptr_h, &next_indptr_h)
            && self.indices_scratch.len() == self.total_tokens;
        let reuse_plan = if page_size == 1 {
            same_batch_identity
                && self.plan_batch_size == slot_indices.len()
                && can_reuse_decode_plan(&self.indptr_h, &next_indptr_h)
        } else {
            same_batch_identity
                && self.plan_batch_size == slot_indices.len()
                && self.indptr_h == next_indptr_h
        };
        let last_page_lens_h = pool.build_last_page_lens(slot_indices);

        // Build positions (each request's current sequence position).
        self.positions_scratch.clear();
        self.positions_scratch
            .extend(slot_indices.iter().map(|&si| (pool.seq_len(si) - 1) as i32));
        let prev_indptr_h = std::mem::replace(&mut self.indptr_h, next_indptr_h);

        // H2D copies -- positions, indptr, last_page_len (small, fixed size).
        ctx.stream
            .memcpy_htod(&self.positions_scratch, &mut self.positions)
            .map_err(|e| anyhow::anyhow!("H2D positions: {e}"))?;
        ctx.stream
            .memcpy_htod(&self.indptr_h, &mut self.kv_indptr)
            .map_err(|e| anyhow::anyhow!("H2D indptr: {e}"))?;
        ctx.stream
            .memcpy_htod(&last_page_lens_h, &mut self.kv_last_page_len)
            .map_err(|e| anyhow::anyhow!("H2D last_page_len: {e}"))?;
        self.qo_indptr_h.clear();
        self.qo_indptr_h.extend(0..=slot_indices.len() as i32);
        ctx.stream
            .memcpy_htod(&self.qo_indptr_h, &mut self.qo_indptr)
            .map_err(|e| anyhow::anyhow!("H2D qo_indptr: {e}"))?;

        let new_total = *self
            .indptr_h
            .last()
            .expect("invariant: indptr_h from build_indptr() always has at least one element")
            as usize;
        let mut reallocated = false;

        self.last_token_scratch.clear();
        self.last_token_scratch
            .extend(pool.build_last_indices(slot_indices));

        let use_gpu_incremental_update =
            can_incremental_update && new_total <= self.max_total_pages;
        if can_incremental_update {
            append_last_token_indices_in_place(
                &mut self.indices_scratch,
                &prev_indptr_h,
                &self.indptr_h,
                &self.last_token_scratch,
            );
        } else {
            self.indices_scratch.clear();
            self.indices_scratch.extend(flatten_page_indices(
                slot_indices.iter().map(|&slot| pool.page_indices(slot)),
            ));
        }
        if self.indices_scratch.len() > self.max_total_pages {
            self.kv_indices = ctx
                .stream
                .alloc_zeros(self.indices_scratch.len())
                .map_err(|e| anyhow::anyhow!("Realloc kv_indices: {e}"))?;
            self.max_total_pages = self.indices_scratch.len();
            reallocated = true;
        }

        // Upload last-token pool indices (for INT8 quantize-after-write).
        ctx.stream
            .memcpy_htod(&self.last_token_scratch, &mut self.last_token_indices)
            .map_err(|e| anyhow::anyhow!("H2D last_token_indices: {e}"))?;
        if use_gpu_incremental_update && !reallocated {
            let (indices_ptr, _gidx) = self.kv_indices.device_ptr_mut(&ctx.stream);
            let (indptr_ptr, _gindptr) = self.kv_indptr.device_ptr(&ctx.stream);
            let (last_ptr, _glast) = self.last_token_indices.device_ptr(&ctx.stream);
            unsafe {
                ffi::flashinfer_append_last_token_indices_cuda(
                    indices_ptr as *mut i32,
                    indptr_ptr as *const i32,
                    last_ptr as *const i32,
                    slot_indices.len() as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        } else {
            ctx.stream
                .memcpy_htod(&self.indices_scratch, &mut self.kv_indices)
                .map_err(|e| anyhow::anyhow!("H2D indices: {e}"))?;
        }
        self.total_tokens = new_total;

        // Re-plan when the batch size changed, lengths did not follow the
        // steady-state decode progression, or kv_indices had to move.
        if reallocated || !reuse_plan {
            self.plan_dirty = true;
        }
        self.prev_slot_indices.clear();
        self.prev_slot_indices.extend_from_slice(slot_indices);
        self.prev_slot_epochs.clear();
        self.prev_slot_epochs.extend_from_slice(&slot_epochs);

        Ok(reallocated)
    }

    /// Upload metadata for a mixed decode + single-chunk prefill batch.
    ///
    /// Layout:
    /// - decode requests: one Q row each
    /// - prefill request: `prefill_token_count` consecutive Q rows
    /// - `qo_indptr`: `[0, 1, 2, ..., B, B + C]`
    pub fn update_mixed(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        decode_slot_indices: &[usize],
        prefill_slot_idx: usize,
        prefill_start_pos: usize,
        prefill_token_count: usize,
    ) -> Result<bool> {
        if decode_slot_indices.is_empty() || prefill_token_count == 0 {
            return Ok(false);
        }

        let request_count = decode_slot_indices.len() + 1;
        let total_qo_rows = decode_slot_indices.len() + prefill_token_count;
        let mut all_slots = Vec::with_capacity(request_count);
        all_slots.extend_from_slice(decode_slot_indices);
        all_slots.push(prefill_slot_idx);

        self.positions_scratch.clear();
        self.positions_scratch.extend(
            decode_slot_indices
                .iter()
                .map(|&slot| (pool.seq_len(slot) - 1) as i32),
        );
        self.positions_scratch.extend(
            (prefill_start_pos..prefill_start_pos + prefill_token_count).map(|pos| pos as i32),
        );

        self.indptr_h = pool.build_indptr(&all_slots);

        let mut last_page_lens_h = pool.build_last_page_lens(decode_slot_indices);
        let prefill_seq_len = pool.seq_len(prefill_slot_idx);
        last_page_lens_h.push(if prefill_seq_len == 0 {
            0
        } else {
            ((prefill_seq_len - 1) % pool.page_size + 1) as i32
        });

        self.qo_indptr_h.clear();
        self.qo_indptr_h
            .extend(0..=decode_slot_indices.len() as i32);
        self.qo_indptr_h.push(total_qo_rows as i32);

        self.indices_scratch.clear();
        self.indices_scratch.extend(flatten_page_indices(
            all_slots.iter().map(|&slot| pool.page_indices(slot)),
        ));

        let mut reallocated = false;
        if self.indices_scratch.len() > self.max_total_pages {
            self.kv_indices = ctx
                .stream
                .alloc_zeros(self.indices_scratch.len())
                .map_err(|e| anyhow::anyhow!("Realloc mixed kv_indices: {e}"))?;
            self.max_total_pages = self.indices_scratch.len();
            reallocated = true;
        }

        ctx.stream
            .memcpy_htod(&self.positions_scratch, &mut self.positions)
            .map_err(|e| anyhow::anyhow!("H2D mixed positions: {e}"))?;
        ctx.stream
            .memcpy_htod(&self.indptr_h, &mut self.kv_indptr)
            .map_err(|e| anyhow::anyhow!("H2D mixed indptr: {e}"))?;
        ctx.stream
            .memcpy_htod(&last_page_lens_h, &mut self.kv_last_page_len)
            .map_err(|e| anyhow::anyhow!("H2D mixed last_page_len: {e}"))?;
        ctx.stream
            .memcpy_htod(&self.qo_indptr_h, &mut self.qo_indptr)
            .map_err(|e| anyhow::anyhow!("H2D mixed qo_indptr: {e}"))?;
        ctx.stream
            .memcpy_htod(&self.indices_scratch, &mut self.kv_indices)
            .map_err(|e| anyhow::anyhow!("H2D mixed indices: {e}"))?;

        self.total_tokens = *self
            .indptr_h
            .last()
            .expect("mixed indptr must contain the terminal entry")
            as usize;
        self.plan_dirty = true;
        self.plan_batch_size = 0;
        self.prev_slot_indices.clear();
        self.prev_slot_epochs.clear();
        self.last_token_scratch.clear();

        Ok(reallocated)
    }

    /// Call FlashInfer's plan step (CPU-side scheduling).
    ///
    /// Must be called once per decode step after `update()`, before any layer
    /// runs. The plan result works for all layers since KV layout is the same.
    ///
    /// Skips re-planning when batch composition is unchanged (steady-state decode),
    /// since the work partitioning across CUDA blocks is stable when only KV lengths
    /// grow by 1 each step.
    #[allow(clippy::too_many_arguments)]
    pub fn plan(
        &mut self,
        ctx: &DeviceContext,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        page_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        if !self.plan_dirty && self.plan_batch_size == batch_size {
            return Ok(());
        }
        flashinfer_plan(
            ctx,
            &self.indptr_h,
            &mut self.flashinfer_ws,
            batch_size,
            num_heads,
            num_kv_heads,
            page_size,
            head_dim,
        )?;
        self.plan_batch_size = batch_size;
        self.plan_dirty = false;
        Ok(())
    }

    /// HD256 decode plan (Qwen3.5 full attention, head_dim=256).
    #[allow(clippy::too_many_arguments)]
    pub fn plan_hd256(
        &mut self,
        ctx: &DeviceContext,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        page_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        if !self.plan_dirty && self.plan_batch_size == batch_size {
            return Ok(());
        }
        flashinfer_plan_hd256(
            ctx,
            &self.indptr_h,
            &mut self.flashinfer_ws,
            batch_size,
            num_heads,
            num_kv_heads,
            page_size,
            head_dim,
        )?;
        self.plan_batch_size = batch_size;
        self.plan_dirty = false;
        Ok(())
    }

    /// Tensor-core decode plan (uses PrefillPlan for flat ITL at long contexts).
    ///
    /// For GQA group_size >= 4, this provides better performance than the standard
    /// decode kernel by tiling across KV chunks with tensor cores.
    #[allow(clippy::too_many_arguments)]
    pub fn tc_plan(
        &mut self,
        ctx: &DeviceContext,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        page_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        // TC decode always re-plans (PrefillPlan depends on KV lengths for split-KV).
        let qo_indptr = &self.qo_indptr_h[..batch_size + 1];
        flashinfer_tc_plan(
            ctx,
            qo_indptr,
            &self.indptr_h,
            &mut self.flashinfer_ws,
            batch_size,
            num_heads,
            num_kv_heads,
            page_size,
            head_dim,
        )?;
        self.plan_batch_size = batch_size;
        self.plan_dirty = false;
        Ok(())
    }

    /// Total tokens across all requests from the last `update()` call.
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }
}

fn flatten_page_indices<'a>(page_groups: impl IntoIterator<Item = &'a [u32]>) -> Vec<i32> {
    let mut flat = Vec::new();
    for group in page_groups {
        flat.extend(group.iter().map(|&idx| idx as i32));
    }
    flat
}

fn can_append_decode_step(prev_indptr: &[i32], next_indptr: &[i32]) -> bool {
    if prev_indptr.len() != next_indptr.len() || prev_indptr.len() < 2 {
        return false;
    }

    prev_indptr
        .windows(2)
        .zip(next_indptr.windows(2))
        .all(|(prev, next)| (next[1] - next[0]) == (prev[1] - prev[0]) + 1)
}

fn append_last_token_indices_in_place(
    indices_scratch: &mut Vec<i32>,
    prev_indptr: &[i32],
    next_indptr: &[i32],
    last_token_indices: &[i32],
) {
    if next_indptr.is_empty() {
        indices_scratch.clear();
        return;
    }

    let new_total = *next_indptr
        .last()
        .expect("invariant: next_indptr always has at least one element")
        as usize;
    indices_scratch.resize(new_total, 0);

    for i in (0..last_token_indices.len()).rev() {
        let old_start = prev_indptr[i] as usize;
        let old_end = prev_indptr[i + 1] as usize;
        let new_start = next_indptr[i] as usize;
        let new_end = next_indptr[i + 1] as usize;
        indices_scratch.copy_within(old_start..old_end, new_start);
        indices_scratch[new_end - 1] = last_token_indices[i];
    }
}

fn can_reuse_decode_plan(prev_indptr: &[i32], next_indptr: &[i32]) -> bool {
    can_append_decode_step(prev_indptr, next_indptr)
}

#[cfg(test)]
mod tests {
    use super::{
        append_last_token_indices_in_place, can_append_decode_step, can_reuse_decode_plan,
        flatten_page_indices,
    };

    #[test]
    fn flatten_page_indices_preserves_per_slot_segments() {
        let flat =
            flatten_page_indices([&[10_u32, 11, 12][..], &[20_u32, 21, 22][..], &[30_u32][..]]);
        assert_eq!(flat, vec![10, 11, 12, 20, 21, 22, 30]);
    }

    #[test]
    fn decode_plan_reuse_allows_steady_state_growth() {
        assert!(can_reuse_decode_plan(&[0, 3, 7], &[0, 4, 9]));
    }

    #[test]
    fn decode_plan_reuse_rejects_batch_shape_changes() {
        assert!(!can_reuse_decode_plan(&[0, 3, 7], &[0, 3, 7]));
        assert!(!can_reuse_decode_plan(&[0, 3, 7], &[0, 5, 8]));
        assert!(!can_reuse_decode_plan(&[0, 3, 7], &[0, 3]));
        assert!(!can_reuse_decode_plan(&[0, 3, 7], &[0, 2, 4]));
    }

    #[test]
    fn incremental_append_updates_each_slot_segment_without_corruption() {
        let mut scratch = vec![10, 11, 20, 21, 22, 30];
        append_last_token_indices_in_place(
            &mut scratch,
            &[0, 2, 5, 6],
            &[0, 3, 7, 9],
            &[12, 23, 31],
        );
        assert_eq!(scratch, vec![10, 11, 12, 20, 21, 22, 23, 30, 31]);
    }

    #[test]
    fn append_decode_step_requires_every_slot_to_grow_by_one() {
        assert!(can_append_decode_step(&[0, 2, 5], &[0, 3, 7]));
        assert!(!can_append_decode_step(&[0, 2, 5], &[0, 2, 6]));
        assert!(!can_append_decode_step(&[0, 2, 5], &[0, 4, 6]));
    }
}
