//! deepep-sys — torch-free Rust binding for DeepEP intranode kernels
//! (Phase B-2 of the multiproc-serve pivot).
//!
//! Two build modes:
//! - **Native** (set `ARLE_DEEPEP_DIR=<deepseek-ai/DeepEP source tree>`
//!   at build time): build.rs nvcc-compiles `csrc/deepep_buffer.cpp` +
//!   DeepEP's kernel `.cu` files into a static archive and links against
//!   libcudart. The extern "C" surface from `deepep_buffer.hpp` is
//!   exposed via the `Buffer` struct.
//! - **Stub** (env unset / nvcc absent): every call returns
//!   `Err(DeepEpError::NotBuilt)`. Lets dependent crates compile cleanly
//!   on machines without the DeepEP source tree.

use anyhow::{Result, bail};

#[derive(Debug, thiserror::Error)]
pub enum DeepEpError {
    #[error("deepep-sys not built — set ARLE_DEEPEP_DIR at build time")]
    NotBuilt,
    #[error("deepep call returned status {code}: {msg}")]
    Status { code: i32, msg: String },
    #[error("bad argument: {0}")]
    BadArgs(String),
}

pub const IPC_HANDLE_BYTES: usize = 64;

/// NVSHMEM unique-id blob size (sizeof(nvshmemx_uniqueid_t) == 128).
pub const LL_UNIQUEID_BYTES: usize = 128;

pub struct DispatchParams {
    pub num_tokens: u32,
    pub hidden: u32,
    pub num_topk: u32,
    pub num_experts: u32,
    pub num_sms: u32,
    pub nvl_chunked_send: u32,
    pub nvl_chunked_recv: u32,
    /// Input device pointers (caller-owned).
    pub d_x: usize,
    pub d_topk_idx: usize,
    pub d_topk_weights: usize,
    /// Output device pointers (caller-allocated, worst-case sized).
    pub d_recv_x: usize,
    pub d_recv_src_idx: usize,
    pub d_recv_topk_idx: usize,
    pub d_recv_topk_weights: usize,
    pub d_rank_prefix_matrix: usize,
    pub d_recv_channel_prefix: usize,
    pub d_send_head: usize,
    /// Scratch (caller-allocated).
    pub d_num_tokens_per_rank: usize,
    pub d_num_tokens_per_expert: usize,
    pub d_is_token_in_rank: usize,
    pub d_channel_prefix_matrix: usize,
    /// Caller's COMPUTE stream handle (cudaStream_t as usize) — the stream
    /// that produces `d_x`/`d_topk_idx`/`d_topk_weights` and consumes the
    /// recv buffers. When non-zero, the wrapper does event-based stream_wait
    /// instead of host `cudaStreamSynchronize`. 0 = fall back to host sync.
    pub compute_stream: usize,
}

pub struct CombineParams {
    pub num_input_tokens: u32,
    pub num_output_tokens: u32,
    pub hidden: u32,
    pub num_topk: u32,
    pub num_sms: u32,
    pub nvl_chunked_send: u32,
    pub nvl_chunked_recv: u32,
    pub d_x: usize,
    pub d_topk_weights: usize,
    pub d_recv_src_idx: usize,
    pub d_rank_prefix_matrix: usize,
    pub d_recv_channel_prefix: usize,
    pub d_send_head: usize,
    pub d_combined_x: usize,
    pub d_combined_topk_w: usize,
    /// CUDA stream handle (cudaStream_t as usize) of the caller's COMPUTE stream
    /// — the stream that produces `d_x` (the expert output) and consumes
    /// `d_combined_x`. When non-zero, the wrapper does event-based
    /// `stream_wait` (comm-stream waits compute before, compute waits comm
    /// after) instead of host `cudaStreamSynchronize`, so the combine no longer
    /// host-blocks the caller. 0 = fall back to the host sync.
    pub compute_stream: usize,
}

/// Parameters for a single NVSHMEM low-latency dispatch. The output
/// device buffers are caller-allocated and sized per the packed LL recv
/// layout `[num_local_experts, world*num_max_dispatch_tokens_per_rank,
/// hidden]` (see `Buffer::low_latency_dispatch` docs).
pub struct LowLatencyDispatchParams {
    pub num_tokens: u32,
    pub hidden: u32,
    pub num_topk: u32,
    pub num_experts: u32,
    pub num_max_dispatch_tokens_per_rank: u32,
    /// FP8 e4m3 packed dispatch (`true`) vs BF16 (`false`).
    pub use_fp8: bool,
    pub round_scale: bool,
    pub use_ue8m0: bool,
    /// Inputs (caller-owned).
    pub d_x: usize, // __nv_bfloat16[num_tokens, hidden]
    pub d_topk_idx: usize, // int64_t[num_tokens, num_topk]
    /// Outputs (caller-allocated; see header for exact shapes).
    pub d_recv_x: usize,
    /// FP8 scales — nullable (`0`) when `use_fp8 == false`.
    pub d_recv_x_scales: usize,
    pub d_recv_src_info: usize,
    pub d_recv_layout_range: usize,
    pub d_recv_count: usize,
    /// Caller COMPUTE stream (cudaStream_t as usize); `0` → host sync.
    pub compute_stream: usize,
}

/// Parameters for a single NVSHMEM low-latency combine.
pub struct LowLatencyCombineParams {
    pub num_combined_tokens: u32,
    pub hidden: u32,
    pub num_topk: u32,
    pub num_experts: u32,
    pub num_max_dispatch_tokens_per_rank: u32,
    pub use_logfmt: bool,
    pub zero_copy: bool,
    /// Inputs (caller-owned).
    pub d_x: usize, // bf16[num_local_experts, world*max_tok, hidden]
    pub d_topk_idx: usize,     // int64_t[num_combined_tokens, num_topk]
    pub d_topk_weights: usize, // float[num_combined_tokens, num_topk]
    pub d_src_info: usize,     // dispatch out: int[num_local_experts, world*max_tok]
    pub d_layout_range: usize, // dispatch out: int64[num_local_experts, world]
    /// Output (caller-allocated): bf16[num_combined_tokens, hidden].
    pub d_combined_x: usize,
    /// Caller COMPUTE stream (cudaStream_t as usize); `0` → host sync.
    pub compute_stream: usize,
}

/// Rank-0-only: produce the 128-byte NVSHMEM unique id for the caller to
/// broadcast to every rank (over NCCL / the sidecar pipe), exactly like the
/// intranode IPC-handle exchange. Other ranks pass the received bytes to
/// [`Buffer::new_low_latency`].
#[cfg(not(deepep_stub))]
pub fn ll_get_uniqueid() -> Result<[u8; LL_UNIQUEID_BYTES]> {
    let mut uid = [0u8; LL_UNIQUEID_BYTES];
    // SAFETY: writes exactly LL_UNIQUEID_BYTES into `uid`.
    let status = unsafe { native::arle_deepep_ll_get_uniqueid(uid.as_mut_ptr()) };
    if status != 0 {
        bail!(DeepEpError::Status {
            code: status,
            msg: native::last_error(),
        });
    }
    Ok(uid)
}

/// Stub build: NVSHMEM LL unavailable.
#[cfg(deepep_stub)]
pub fn ll_get_uniqueid() -> Result<[u8; LL_UNIQUEID_BYTES]> {
    bail!(DeepEpError::NotBuilt)
}

/// Whether this binary was built with the DeepEP native path enabled.
/// `false` means every method on `Buffer` returns `DeepEpError::NotBuilt`.
pub fn is_native() -> bool {
    !cfg!(deepep_stub)
}

/// Whether the DeepEP NVSHMEM low-latency (internode_ll) path is
/// compiled + linked into this binary. `true` proves libarle_deepep.a
/// pulled the internode_ll + NVSHMEM device objects at final link
/// (build+link de-risk only — no `nvshmem_init` runtime bootstrap yet).
#[cfg(not(deepep_stub))]
pub fn nvshmem_built() -> bool {
    // SAFETY: trivial extern "C" probe, no args, returns int.
    unsafe { native::arle_deepep_nvshmem_built() != 0 }
}

/// Stub build: NVSHMEM LL is never compiled in.
#[cfg(deepep_stub)]
pub fn nvshmem_built() -> bool {
    false
}

#[cfg(deepep_stub)]
pub struct Buffer {
    _rank: u32,
    _world_size: u32,
}

#[cfg(deepep_stub)]
impl Buffer {
    pub fn new(_rank: u32, _world_size: u32, _device_ordinal: u32) -> Result<Self> {
        bail!(DeepEpError::NotBuilt)
    }
    pub fn local_ipc_handle(&self) -> Result<([u8; IPC_HANDLE_BYTES], u32)> {
        bail!(DeepEpError::NotBuilt)
    }
    pub fn sync(&mut self, _peers: &[([u8; IPC_HANDLE_BYTES], u32)]) -> Result<()> {
        bail!(DeepEpError::NotBuilt)
    }
    pub fn dispatch(&mut self, _p: &DispatchParams) -> Result<i32> {
        bail!(DeepEpError::NotBuilt)
    }
    pub fn combine(&mut self, _p: &CombineParams) -> Result<()> {
        bail!(DeepEpError::NotBuilt)
    }
    pub fn new_low_latency(
        _rank: u32,
        _world_size: u32,
        _device_ordinal: u32,
        _num_max_dispatch_tokens_per_rank: u32,
        _hidden: u32,
        _num_experts: u32,
        _root_uniqueid: &[u8; LL_UNIQUEID_BYTES],
    ) -> Result<Self> {
        bail!(DeepEpError::NotBuilt)
    }
    pub fn is_low_latency(&self) -> bool {
        false
    }
    pub fn low_latency_dispatch(&mut self, _p: &LowLatencyDispatchParams) -> Result<i32> {
        bail!(DeepEpError::NotBuilt)
    }
    pub fn low_latency_combine(&mut self, _p: &LowLatencyCombineParams) -> Result<()> {
        bail!(DeepEpError::NotBuilt)
    }
}

#[cfg(not(deepep_stub))]
mod native {
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_int};

    #[repr(C)]
    pub(super) struct ArleDeepEpBuffer {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub(super) struct ArleDeepEpDispatchParams {
        pub num_tokens: u32,
        pub hidden: u32,
        pub num_topk: u32,
        pub num_experts: u32,
        pub num_sms: u32,
        pub nvl_chunked_send: u32,
        pub nvl_chunked_recv: u32,
        pub d_x: usize,
        pub d_topk_idx: usize,
        pub d_topk_weights: usize,
        pub d_recv_x: usize,
        pub d_recv_src_idx: usize,
        pub d_recv_topk_idx: usize,
        pub d_recv_topk_weights: usize,
        pub d_rank_prefix_matrix: usize,
        pub d_recv_channel_prefix: usize,
        pub d_send_head: usize,
        pub d_num_tokens_per_rank: usize,
        pub d_num_tokens_per_expert: usize,
        pub d_is_token_in_rank: usize,
        pub d_channel_prefix_matrix: usize,
        /// Caller's COMPUTE stream handle (cudaStream_t as usize). When
        /// non-zero, event-based stream_wait replaces host sync. 0 = host sync.
        pub compute_stream: usize,
        pub out_num_recv_tokens: *mut i32,
    }

    #[repr(C)]
    pub(super) struct ArleDeepEpCombineParams {
        pub num_input_tokens: u32,
        pub num_output_tokens: u32,
        pub hidden: u32,
        pub num_topk: u32,
        pub num_sms: u32,
        pub nvl_chunked_send: u32,
        pub nvl_chunked_recv: u32,
        pub d_x: usize,
        pub d_topk_weights: usize,
        pub d_recv_src_idx: usize,
        pub d_rank_prefix_matrix: usize,
        pub d_recv_channel_prefix: usize,
        pub d_send_head: usize,
        pub d_combined_x: usize,
        pub d_combined_topk_w: usize,
        pub compute_stream: usize,
    }

    // NVSHMEM low-latency params — field order + types MUST match
    // deepep_buffer.hpp ArleDeepEpLowLatency{Dispatch,Combine}Params.
    #[repr(C)]
    pub(super) struct ArleDeepEpLowLatencyDispatchParams {
        pub num_tokens: u32,
        pub hidden: u32,
        pub num_topk: u32,
        pub num_experts: u32,
        pub num_max_dispatch_tokens_per_rank: u32,
        pub use_fp8: u8,
        pub round_scale: u8,
        pub use_ue8m0: u8,
        pub d_x: usize,
        pub d_topk_idx: usize,
        pub d_recv_x: usize,
        pub d_recv_x_scales: usize,
        pub d_recv_src_info: usize,
        pub d_recv_layout_range: usize,
        pub d_recv_count: usize,
        pub compute_stream: usize,
        pub out_expected_m: *mut i32,
    }

    #[repr(C)]
    pub(super) struct ArleDeepEpLowLatencyCombineParams {
        pub num_combined_tokens: u32,
        pub hidden: u32,
        pub num_topk: u32,
        pub num_experts: u32,
        pub num_max_dispatch_tokens_per_rank: u32,
        pub use_logfmt: u8,
        pub zero_copy: u8,
        pub d_x: usize,
        pub d_topk_idx: usize,
        pub d_topk_weights: usize,
        pub d_src_info: usize,
        pub d_layout_range: usize,
        pub d_combined_x: usize,
        pub compute_stream: usize,
    }

    unsafe extern "C" {
        pub(super) fn arle_deepep_buffer_create(
            rank: u32,
            world_size: u32,
            device_id: u32,
            out_handle: *mut *mut ArleDeepEpBuffer,
        ) -> c_int;
        pub(super) fn arle_deepep_buffer_local_ipc_handle(
            handle: *mut ArleDeepEpBuffer,
            out_ipc_handle: *mut u8,
            out_device_id: *mut u32,
        ) -> c_int;
        pub(super) fn arle_deepep_buffer_sync(
            handle: *mut ArleDeepEpBuffer,
            peer_ipc_handles: *const u8,
            peer_device_ids: *const u32,
            world_size: u32,
        ) -> c_int;
        pub(super) fn arle_deepep_buffer_dispatch(
            handle: *mut ArleDeepEpBuffer,
            params: *const ArleDeepEpDispatchParams,
        ) -> c_int;
        pub(super) fn arle_deepep_buffer_combine(
            handle: *mut ArleDeepEpBuffer,
            params: *const ArleDeepEpCombineParams,
        ) -> c_int;
        pub(super) fn arle_deepep_buffer_destroy(handle: *mut ArleDeepEpBuffer);
        pub(super) fn arle_deepep_last_error() -> *const c_char;
        /// Force-link probe: returns 1 when the internode_ll + NVSHMEM
        /// objects are compiled into libarle_deepep.a (T4-LL de-risk).
        pub(super) fn arle_deepep_nvshmem_built() -> c_int;
        // --- NVSHMEM low-latency (internode_ll) surface ---
        pub(super) fn arle_deepep_ll_get_uniqueid(out_uniqueid: *mut u8) -> c_int;
        pub(super) fn arle_deepep_buffer_ll_create(
            rank: u32,
            world_size: u32,
            device_id: u32,
            num_max_dispatch_tokens_per_rank: u32,
            hidden: u32,
            num_experts: u32,
            root_uniqueid: *const u8,
            out_handle: *mut *mut ArleDeepEpBuffer,
        ) -> c_int;
        pub(super) fn arle_deepep_buffer_low_latency_dispatch(
            handle: *mut ArleDeepEpBuffer,
            params: *const ArleDeepEpLowLatencyDispatchParams,
        ) -> c_int;
        pub(super) fn arle_deepep_buffer_low_latency_combine(
            handle: *mut ArleDeepEpBuffer,
            params: *const ArleDeepEpLowLatencyCombineParams,
        ) -> c_int;
        pub(super) fn arle_deepep_buffer_is_low_latency(handle: *const ArleDeepEpBuffer) -> c_int;
    }

    pub(super) fn last_error() -> String {
        // SAFETY: thread-local static buffer; null-terminated.
        unsafe {
            let p = arle_deepep_last_error();
            if p.is_null() {
                return String::new();
            }
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

#[cfg(not(deepep_stub))]
pub struct Buffer {
    handle: *mut native::ArleDeepEpBuffer,
}

#[cfg(not(deepep_stub))]
// Safety: the underlying C state is owned exclusively by this Buffer (no
// shared state across threads), and the implementation runs CUDA calls
// against a stream owned by the same struct.
unsafe impl Send for Buffer {}

#[cfg(not(deepep_stub))]
impl Buffer {
    pub fn new(rank: u32, world_size: u32, device_ordinal: u32) -> Result<Self> {
        let mut handle: *mut native::ArleDeepEpBuffer = std::ptr::null_mut();
        let status = unsafe {
            native::arle_deepep_buffer_create(rank, world_size, device_ordinal, &mut handle)
        };
        if status != 0 {
            bail!(DeepEpError::Status {
                code: status,
                msg: native::last_error(),
            });
        }
        Ok(Self { handle })
    }

    pub fn local_ipc_handle(&self) -> Result<([u8; IPC_HANDLE_BYTES], u32)> {
        let mut buf = [0u8; IPC_HANDLE_BYTES];
        let mut device_id = 0u32;
        let status = unsafe {
            native::arle_deepep_buffer_local_ipc_handle(
                self.handle,
                buf.as_mut_ptr(),
                &mut device_id,
            )
        };
        if status != 0 {
            bail!(DeepEpError::Status {
                code: status,
                msg: native::last_error(),
            });
        }
        Ok((buf, device_id))
    }

    pub fn sync(&mut self, peers: &[([u8; IPC_HANDLE_BYTES], u32)]) -> Result<()> {
        let world_size = peers.len();
        if world_size < 2 || world_size > 8 {
            bail!(DeepEpError::BadArgs(format!(
                "world_size must be in [2, 8], got {world_size}"
            )));
        }
        // Flatten peer handles into a contiguous byte buffer for the C
        // call (C side reads world_size × 64 bytes).
        let handle_blob: Vec<u8> = peers.iter().flat_map(|(h, _)| h.iter().copied()).collect();
        let device_ids: Vec<_> = peers.iter().map(|(_, did)| *did).collect();
        let status = unsafe {
            native::arle_deepep_buffer_sync(
                self.handle,
                handle_blob.as_ptr(),
                device_ids.as_ptr(),
                world_size as u32,
            )
        };
        if status != 0 {
            bail!(DeepEpError::Status {
                code: status,
                msg: native::last_error(),
            });
        }
        Ok(())
    }

    /// Returns the actual number of received tokens (host-poll result of
    /// notify_dispatch).
    pub fn dispatch(&mut self, p: &DispatchParams) -> Result<i32> {
        let mut out_num_recv = 0i32;
        let c = native::ArleDeepEpDispatchParams {
            num_tokens: p.num_tokens,
            hidden: p.hidden,
            num_topk: p.num_topk,
            num_experts: p.num_experts,
            num_sms: p.num_sms,
            nvl_chunked_send: p.nvl_chunked_send,
            nvl_chunked_recv: p.nvl_chunked_recv,
            d_x: p.d_x,
            d_topk_idx: p.d_topk_idx,
            d_topk_weights: p.d_topk_weights,
            d_recv_x: p.d_recv_x,
            d_recv_src_idx: p.d_recv_src_idx,
            d_recv_topk_idx: p.d_recv_topk_idx,
            d_recv_topk_weights: p.d_recv_topk_weights,
            d_rank_prefix_matrix: p.d_rank_prefix_matrix,
            d_recv_channel_prefix: p.d_recv_channel_prefix,
            d_send_head: p.d_send_head,
            d_num_tokens_per_rank: p.d_num_tokens_per_rank,
            d_num_tokens_per_expert: p.d_num_tokens_per_expert,
            d_is_token_in_rank: p.d_is_token_in_rank,
            d_channel_prefix_matrix: p.d_channel_prefix_matrix,
            compute_stream: p.compute_stream,
            out_num_recv_tokens: &mut out_num_recv,
        };
        let status = unsafe { native::arle_deepep_buffer_dispatch(self.handle, &c) };
        if status != 0 {
            bail!(DeepEpError::Status {
                code: status,
                msg: native::last_error(),
            });
        }
        Ok(out_num_recv)
    }

    pub fn combine(&mut self, p: &CombineParams) -> Result<()> {
        let c = native::ArleDeepEpCombineParams {
            num_input_tokens: p.num_input_tokens,
            num_output_tokens: p.num_output_tokens,
            hidden: p.hidden,
            num_topk: p.num_topk,
            num_sms: p.num_sms,
            nvl_chunked_send: p.nvl_chunked_send,
            nvl_chunked_recv: p.nvl_chunked_recv,
            d_x: p.d_x,
            d_topk_weights: p.d_topk_weights,
            d_recv_src_idx: p.d_recv_src_idx,
            d_rank_prefix_matrix: p.d_rank_prefix_matrix,
            d_recv_channel_prefix: p.d_recv_channel_prefix,
            d_send_head: p.d_send_head,
            d_combined_x: p.d_combined_x,
            d_combined_topk_w: p.d_combined_topk_w,
            compute_stream: p.compute_stream,
        };
        let status = unsafe { native::arle_deepep_buffer_combine(self.handle, &c) };
        if status != 0 {
            bail!(DeepEpError::Status {
                code: status,
                msg: native::last_error(),
            });
        }
        Ok(())
    }

    /// Create an NVSHMEM low-latency buffer. Every rank calls this with the
    /// SAME `root_uniqueid` (rank 0 produced it via [`ll_get_uniqueid`] and
    /// broadcast it) and the SAME sizing. Collective: it runs
    /// `nvshmem::init` + a cross-rank barrier internally, so all ranks must
    /// reach it. `num_experts` must be divisible by `world_size` and
    /// `hidden` a multiple of 128 (multiple of 512 for the FP8 dispatch).
    pub fn new_low_latency(
        rank: u32,
        world_size: u32,
        device_ordinal: u32,
        num_max_dispatch_tokens_per_rank: u32,
        hidden: u32,
        num_experts: u32,
        root_uniqueid: &[u8; LL_UNIQUEID_BYTES],
    ) -> Result<Self> {
        if !(2..=8).contains(&world_size) {
            bail!(DeepEpError::BadArgs(format!(
                "world_size must be in [2, 8], got {world_size}"
            )));
        }
        let mut handle: *mut native::ArleDeepEpBuffer = std::ptr::null_mut();
        let status = unsafe {
            native::arle_deepep_buffer_ll_create(
                rank,
                world_size,
                device_ordinal,
                num_max_dispatch_tokens_per_rank,
                hidden,
                num_experts,
                root_uniqueid.as_ptr(),
                &mut handle,
            )
        };
        if status != 0 {
            bail!(DeepEpError::Status {
                code: status,
                msg: native::last_error(),
            });
        }
        Ok(Self { handle })
    }

    /// `true` if this buffer was created in NVSHMEM low-latency mode.
    pub fn is_low_latency(&self) -> bool {
        // SAFETY: read-only query on the owned handle.
        unsafe { native::arle_deepep_buffer_is_low_latency(self.handle) != 0 }
    }

    /// Run one NVSHMEM low-latency dispatch. Returns `expected_m` — the
    /// average per-local-expert token count (`ceil(num_tokens * world *
    /// num_topk / num_experts)`) for sizing the GroupedGEMM; the exact
    /// per-expert `masked_m` counts land in `d_recv_count`.
    pub fn low_latency_dispatch(&mut self, p: &LowLatencyDispatchParams) -> Result<i32> {
        let mut out_expected_m = 0i32;
        let c = native::ArleDeepEpLowLatencyDispatchParams {
            num_tokens: p.num_tokens,
            hidden: p.hidden,
            num_topk: p.num_topk,
            num_experts: p.num_experts,
            num_max_dispatch_tokens_per_rank: p.num_max_dispatch_tokens_per_rank,
            use_fp8: p.use_fp8 as u8,
            round_scale: p.round_scale as u8,
            use_ue8m0: p.use_ue8m0 as u8,
            d_x: p.d_x,
            d_topk_idx: p.d_topk_idx,
            d_recv_x: p.d_recv_x,
            d_recv_x_scales: p.d_recv_x_scales,
            d_recv_src_info: p.d_recv_src_info,
            d_recv_layout_range: p.d_recv_layout_range,
            d_recv_count: p.d_recv_count,
            compute_stream: p.compute_stream,
            out_expected_m: &mut out_expected_m,
        };
        let status = unsafe { native::arle_deepep_buffer_low_latency_dispatch(self.handle, &c) };
        if status != 0 {
            bail!(DeepEpError::Status {
                code: status,
                msg: native::last_error(),
            });
        }
        Ok(out_expected_m)
    }

    /// Run one NVSHMEM low-latency combine, reducing the per-expert outputs
    /// back to `[num_combined_tokens, hidden]` in `d_combined_x`.
    pub fn low_latency_combine(&mut self, p: &LowLatencyCombineParams) -> Result<()> {
        let c = native::ArleDeepEpLowLatencyCombineParams {
            num_combined_tokens: p.num_combined_tokens,
            hidden: p.hidden,
            num_topk: p.num_topk,
            num_experts: p.num_experts,
            num_max_dispatch_tokens_per_rank: p.num_max_dispatch_tokens_per_rank,
            use_logfmt: p.use_logfmt as u8,
            zero_copy: p.zero_copy as u8,
            d_x: p.d_x,
            d_topk_idx: p.d_topk_idx,
            d_topk_weights: p.d_topk_weights,
            d_src_info: p.d_src_info,
            d_layout_range: p.d_layout_range,
            d_combined_x: p.d_combined_x,
            compute_stream: p.compute_stream,
        };
        let status = unsafe { native::arle_deepep_buffer_low_latency_combine(self.handle, &c) };
        if status != 0 {
            bail!(DeepEpError::Status {
                code: status,
                msg: native::last_error(),
            });
        }
        Ok(())
    }
}

#[cfg(not(deepep_stub))]
impl Drop for Buffer {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { native::arle_deepep_buffer_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}
