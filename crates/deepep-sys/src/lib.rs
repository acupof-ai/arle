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

/// Whether this binary was built with the DeepEP native path enabled.
/// `false` means every method on `Buffer` returns `DeepEpError::NotBuilt`.
pub fn is_native() -> bool {
    !cfg!(deepep_stub)
}

#[cfg(deepep_stub)]
pub struct Buffer {
    _rank: u32,
    _world_size: u32,
}

#[cfg(deepep_stub)]
impl Buffer {
    pub fn new(_rank: u32, _world_size: u32) -> Result<Self> {
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
}

#[cfg(not(deepep_stub))]
mod native {
    use super::*;
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

    unsafe extern "C" {
        pub(super) fn arle_deepep_buffer_create(
            rank: u32,
            world_size: u32,
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
    pub fn new(rank: u32, world_size: u32) -> Result<Self> {
        let mut handle: *mut native::ArleDeepEpBuffer = std::ptr::null_mut();
        let status = unsafe { native::arle_deepep_buffer_create(rank, world_size, &mut handle) };
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
        let mut handle_blob = Vec::with_capacity(world_size * IPC_HANDLE_BYTES);
        let mut device_ids = Vec::with_capacity(world_size);
        for (h, did) in peers {
            handle_blob.extend_from_slice(h);
            device_ids.push(*did);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_or_native_compiles() {
        // Just ensure the API surface compiles in both modes.
        // We can't actually construct a Buffer in unit tests because that
        // requires 2+ CUDA devices and DeepEP. Smoke is in
        // infer/tests/deepep_sys_smoke.rs once added.
        let _ = is_native();
    }
}
