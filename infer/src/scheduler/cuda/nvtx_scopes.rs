use std::os::raw::c_char;

unsafe extern "C" {
    fn arle_nvtx_range_push(message: *const c_char);
    fn arle_nvtx_range_pop();
}

pub(in crate::scheduler::cuda) struct NvtxScope;

impl NvtxScope {
    pub(in crate::scheduler::cuda) fn push(message: *const c_char) -> Self {
        // NVTX v3 resolves the profiler injection lazily and is a no-op when
        // Nsight is not attached. The C shim includes the header-only ABI.
        unsafe {
            arle_nvtx_range_push(message);
        }
        Self
    }
}

impl Drop for NvtxScope {
    fn drop(&mut self) {
        unsafe {
            arle_nvtx_range_pop();
        }
    }
}

macro_rules! nvtx_scope {
    ($name:literal) => {
        let _nvtx_scope = $crate::scheduler::cuda::nvtx_scopes::NvtxScope::push(
            concat!($name, "\0").as_ptr().cast(),
        );
    };
}

pub(in crate::scheduler::cuda) use nvtx_scope;
