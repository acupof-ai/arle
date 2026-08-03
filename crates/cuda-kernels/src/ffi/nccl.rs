//! Minimal NCCL FFI declarations. Linking is wired in F1 via build.rs.
//!
//! Per the F0 multi-GPU plan, this module declares only the symbol surface
//! required by
//! `CollectiveBackend::Nccl`. No build-time linkage is performed in F0:
//! the `extern "C"` block compiles without `libnccl.so` present, and the
//! actual library hookup lives in F1's `build.rs` work.
//!
//! Function signatures track NCCL 2.x. Stream pointers are passed as
//! `*mut std::ffi::c_void` so this module does not need a cudarc dependency
//! (callers that hold a `CUstream` cast it through `as *mut c_void` at the
//! call site).

#![allow(non_camel_case_types, non_snake_case)]

#[repr(C)]
pub struct ncclComm {
    _private: [u8; 0],
}
pub type ncclComm_t = *mut ncclComm;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ncclUniqueId {
    pub internal: [i8; 128],
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ncclResult_t {
    Success = 0,
    UnhandledCudaError = 1,
    SystemError = 2,
    InternalError = 3,
    InvalidArgument = 4,
    InvalidUsage = 5,
    RemoteError = 6,
    InProgress = 7,
    NumResults = 8,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ncclDataType_t {
    Int8 = 0,
    Uint8 = 1,
    Int32 = 2,
    Uint32 = 3,
    Int64 = 4,
    Uint64 = 5,
    Float16 = 6,
    Float32 = 7,
    Float64 = 8,
    Bfloat16 = 9,
    Float8e4m3 = 10,
    Float8e5m2 = 11,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ncclRedOp_t {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Avg = 4,
}

/// Opaque NCCL window handle (`ncclWindow_t`, NCCL >= 2.27).
#[allow(non_camel_case_types)]
pub type ncclWindow_t = *mut std::os::raw::c_void;

/// `ncclCommWindowRegister` flag requesting symmetric-collective eligibility
/// (NCCL >= 2.27 low-latency symmetric kernels; `NCCL_WIN_COLL_SYMMETRIC`).
pub const NCCL_WIN_COLL_SYMMETRIC: i32 = 1;

unsafe extern "C" {
    pub fn ncclGetUniqueId(unique_id: *mut ncclUniqueId) -> ncclResult_t;
    /// NCCL >= 2.27: allocate device memory eligible for window registration
    /// (symmetric low-latency kernels require `ncclMemAlloc`'d or VMM-aligned
    /// buffers, not plain `cudaMalloc`).
    pub fn ncclMemAlloc(ptr: *mut *mut std::os::raw::c_void, size: usize) -> ncclResult_t;
    pub fn ncclMemFree(ptr: *mut std::os::raw::c_void) -> ncclResult_t;
    /// NCCL >= 2.27: register `buff` as a window over `comm`. With
    /// [`NCCL_WIN_COLL_SYMMETRIC`], collectives whose operands are all
    /// symmetric windows take the low-latency symmetric kernels.
    /// COLLECTIVE over `comm`: every rank must call it together.
    pub fn ncclCommWindowRegister(
        comm: ncclComm_t,
        buff: *mut std::os::raw::c_void,
        size: usize,
        win: *mut ncclWindow_t,
        win_flags: i32,
    ) -> ncclResult_t;
    pub fn ncclCommWindowDeregister(comm: ncclComm_t, win: ncclWindow_t) -> ncclResult_t;
    pub fn ncclCommInitRank(
        comm: *mut ncclComm_t,
        world_size: i32,
        unique_id: ncclUniqueId,
        rank: i32,
    ) -> ncclResult_t;
    pub fn ncclCommDestroy(comm: ncclComm_t) -> ncclResult_t;
    /// Split the parent comm into sub-comms. COLLECTIVE over `comm`: every
    /// parent rank must call it together. Ranks sharing a `color` join the same
    /// `newcomm`, ordered by `key`. `config` is passed as null (`*mut c_void`)
    /// so the sub-comm inherits the parent's config — avoids binding the
    /// `ncclConfig_t` struct / `NCCL_CONFIG_INITIALIZER` magic.
    pub fn ncclCommSplit(
        comm: ncclComm_t,
        color: i32,
        key: i32,
        newcomm: *mut ncclComm_t,
        config: *mut std::os::raw::c_void,
    ) -> ncclResult_t;
    pub fn ncclAllReduce(
        sendbuff: *const std::ffi::c_void,
        recvbuff: *mut std::ffi::c_void,
        count: usize,
        dtype: ncclDataType_t,
        op: ncclRedOp_t,
        comm: ncclComm_t,
        stream: *mut std::ffi::c_void,
    ) -> ncclResult_t;
    pub fn ncclAllGather(
        sendbuff: *const std::ffi::c_void,
        recvbuff: *mut std::ffi::c_void,
        sendcount: usize,
        dtype: ncclDataType_t,
        comm: ncclComm_t,
        stream: *mut std::ffi::c_void,
    ) -> ncclResult_t;
    pub fn ncclReduceScatter(
        sendbuff: *const std::ffi::c_void,
        recvbuff: *mut std::ffi::c_void,
        recvcount: usize,
        dtype: ncclDataType_t,
        op: ncclRedOp_t,
        comm: ncclComm_t,
        stream: *mut std::ffi::c_void,
    ) -> ncclResult_t;
    pub fn ncclBroadcast(
        sendbuff: *const std::ffi::c_void,
        recvbuff: *mut std::ffi::c_void,
        count: usize,
        dtype: ncclDataType_t,
        root: i32,
        comm: ncclComm_t,
        stream: *mut std::ffi::c_void,
    ) -> ncclResult_t;
    pub fn ncclSend(
        sendbuff: *const std::ffi::c_void,
        count: usize,
        dtype: ncclDataType_t,
        peer: i32,
        comm: ncclComm_t,
        stream: *mut std::ffi::c_void,
    ) -> ncclResult_t;
    pub fn ncclRecv(
        recvbuff: *mut std::ffi::c_void,
        count: usize,
        dtype: ncclDataType_t,
        peer: i32,
        comm: ncclComm_t,
        stream: *mut std::ffi::c_void,
    ) -> ncclResult_t;
    pub fn ncclGroupStart() -> ncclResult_t;
    pub fn ncclGroupEnd() -> ncclResult_t;
    pub fn ncclGetErrorString(result: ncclResult_t) -> *const std::os::raw::c_char;
}

/// Map a non-Success NCCL return into `anyhow::Error` carrying the library's
/// own diagnostic string.
pub fn check(result: ncclResult_t) -> anyhow::Result<()> {
    if result == ncclResult_t::Success {
        Ok(())
    } else {
        // SAFETY: `ncclGetErrorString` returns a static NUL-terminated string owned by
        // the library, valid for the process; we only borrow it for the format below.
        let cstr = unsafe { std::ffi::CStr::from_ptr(ncclGetErrorString(result)) };
        Err(anyhow::anyhow!(
            "NCCL error: {} ({:?})",
            cstr.to_string_lossy(),
            result
        ))
    }
}

#[cfg(all(test, feature = "nccl"))]
mod tests {
    use super::*;

    #[test]
    fn unique_id_size() {
        assert_eq!(std::mem::size_of::<ncclUniqueId>(), 128);
    }

    #[test]
    fn nccl_result_size() {
        assert_eq!(std::mem::size_of::<ncclResult_t>(), 4);
    }
}
