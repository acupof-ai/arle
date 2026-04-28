//! Minimal NCCL FFI declarations. Linking is wired in F1 via build.rs.
//!
//! Per the F0 multi-GPU plan (docs/plans/2026-04-28-single-node-multi-gpu.md
//! §4.2), this module declares only the symbol surface required by
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

unsafe extern "C" {
    pub fn ncclGetUniqueId(unique_id: *mut ncclUniqueId) -> ncclResult_t;
    pub fn ncclCommInitRank(
        comm: *mut ncclComm_t,
        world_size: i32,
        unique_id: ncclUniqueId,
        rank: i32,
    ) -> ncclResult_t;
    pub fn ncclCommDestroy(comm: ncclComm_t) -> ncclResult_t;
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
