#[cfg(target_os = "linux")]
use std::ffi::c_void;
use std::ffi::{CString, c_char, c_int};
#[cfg(target_os = "linux")]
use std::ptr;
use std::sync::OnceLock;

type NvtxPush = unsafe extern "C" fn(*const c_char) -> c_int;
type NvtxPop = unsafe extern "C" fn() -> c_int;

#[derive(Clone, Copy)]
struct NvtxFns {
    push: Option<NvtxPush>,
    pop: Option<NvtxPop>,
}

#[cfg(target_os = "linux")]
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

#[cfg(target_os = "linux")]
const RTLD_LAZY: c_int = 1;
static ENABLED: OnceLock<bool> = OnceLock::new();
static NVTX: OnceLock<NvtxFns> = OnceLock::new();

fn enabled() -> bool {
    *ENABLED.get_or_init(|| {
        std::env::var_os("ARLE_NVTX").is_some() || std::env::var_os("ARLE_DSV4_NVTX").is_some()
    })
}

fn load() -> NvtxFns {
    // SAFETY: dlopen/dlsym on NUL-terminated static names; every returned
    // pointer is null-checked before use, and the transmuted signatures match
    // the NVTX C ABI for nvtxRangePushA / nvtxRangePop.
    #[cfg(target_os = "linux")]
    unsafe {
        let handle = [
            "libnvtx3interop.so.1",
            "libnvtx3interop.so",
            "/opt/cuda/targets/x86_64-linux/lib/libnvtx3interop.so.1",
        ]
        .iter()
        .filter_map(|name| CString::new(*name).ok())
        .map(|name| dlopen(name.as_ptr(), RTLD_LAZY))
        .find(|handle| !handle.is_null())
        .unwrap_or(ptr::null_mut());
        if handle.is_null() {
            return NvtxFns {
                push: None,
                pop: None,
            };
        }

        let push_name = CString::new("nvtxRangePushA").expect("static nvtx symbol");
        let pop_name = CString::new("nvtxRangePop").expect("static nvtx symbol");
        let push_ptr = dlsym(handle, push_name.as_ptr());
        let pop_ptr = dlsym(handle, pop_name.as_ptr());
        NvtxFns {
            push: (!push_ptr.is_null()).then(|| std::mem::transmute(push_ptr)),
            pop: (!pop_ptr.is_null()).then(|| std::mem::transmute(pop_ptr)),
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        NvtxFns {
            push: None,
            pop: None,
        }
    }
}

pub(crate) struct Range {
    active: bool,
}

pub(crate) fn range(name: &str) -> Range {
    if !enabled() {
        return Range { active: false };
    }
    let fns = NVTX.get_or_init(load);
    let Some(push) = fns.push else {
        return Range { active: false };
    };
    let Ok(name) = CString::new(name) else {
        return Range { active: false };
    };
    // SAFETY: push is a resolved nvToolsExt symbol; name is a live NUL-terminated CString.
    unsafe {
        push(name.as_ptr());
    }
    Range {
        active: fns.pop.is_some(),
    }
}

impl Drop for Range {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let Some(pop) = NVTX.get_or_init(load).pop else {
            return;
        };
        // SAFETY: pop is a resolved nvToolsExt symbol, paired with the successful push in range().
        unsafe {
            pop();
        }
    }
}
