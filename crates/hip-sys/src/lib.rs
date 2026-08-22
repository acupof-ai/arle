//! Thin HIP runtime FFI for the AIPC HIP backend (#71/#76).
//!
//! Hand-declared `extern "C"` over the stable scalar/pointer subset of the
//! HIP runtime API — no bindgen, no ROCm headers needed to typecheck. The
//! surface is deliberately the smallest thing the device probe and smoke
//! test call today; module-load/launch and hipBLASLt land with their first
//! real consumers (interfaces trail callers).
//!
//! Feature contract (mirror of `cuda,no-cuda`):
//! - `--features hip` on a ROCm box with `libamdhip64`: calls are real.
//! - default, or `--features hip` off-box without `libamdhip64`: every entry
//!   point returns [`HIP_NOT_COMPILED`]; compiles and tests on any machine.
//!
//! On-box TODOs (need ROCm headers / real hardware): gfx arch-name probe
//! (`hipDeviceProp_t.gcnArchName` is struct-ABI-versioned — resolve with
//! bindgen or a pinned `hipGetDevicePropertiesR0600` decl on the box),
//! hipBLASLt handle/matmul subset, async memcpy + module launch.

/// Raw HIP error code. `0` is `hipSuccess`; [`HIP_NOT_COMPILED`] is our
/// sentinel for the stub build, never produced by the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipError(pub i32);

pub const HIP_NOT_COMPILED: HipError = HipError(-1);

pub type Result<T> = std::result::Result<T, HipError>;

impl std::fmt::Display for HipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if *self == HIP_NOT_COMPILED {
            return write!(
                f,
                "HIP runtime unavailable (build with --features hip on a ROCm box with libamdhip64)"
            );
        }
        #[cfg(hip_runtime_available)]
        {
            let msg = unsafe { ffi::hipGetErrorString(self.0) };
            if !msg.is_null() {
                let msg = unsafe { std::ffi::CStr::from_ptr(msg) };
                return write!(f, "HIP error {}: {}", self.0, msg.to_string_lossy());
            }
        }
        write!(f, "HIP error {}", self.0)
    }
}

impl std::error::Error for HipError {}

/// Memcpy kinds — identical values to the CUDA enum (HIP keeps parity).
pub const MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub const MEMCPY_DEVICE_TO_HOST: i32 = 2;
pub const MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

#[cfg(hip_runtime_available)]
mod ffi {
    use std::ffi::{c_char, c_void};

    unsafe extern "C" {
        pub fn hipInit(flags: u32) -> i32;
        pub fn hipGetDeviceCount(count: *mut i32) -> i32;
        pub fn hipSetDevice(device: i32) -> i32;
        pub fn hipDeviceGetName(name: *mut c_char, len: i32, device: i32) -> i32;
        pub fn hipDeviceTotalMem(bytes: *mut usize, device: i32) -> i32;
        pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
        pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
        pub fn hipFree(ptr: *mut c_void) -> i32;
        pub fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: i32) -> i32;
        pub fn hipStreamCreate(stream: *mut *mut c_void) -> i32;
        pub fn hipStreamDestroy(stream: *mut c_void) -> i32;
        pub fn hipStreamSynchronize(stream: *mut c_void) -> i32;
        pub fn hipDeviceSynchronize() -> i32;
        pub fn hipGetErrorString(code: i32) -> *const c_char;
    }
}

#[cfg(hip_runtime_available)]
mod real {
    use super::{HipError, Result, ffi};
    use std::ffi::c_void;

    fn check(code: i32) -> Result<()> {
        if code == 0 {
            Ok(())
        } else {
            Err(HipError(code))
        }
    }

    /// Initialize the HIP runtime. Idempotent.
    pub fn init() -> Result<()> {
        check(unsafe { ffi::hipInit(0) })
    }

    pub fn device_count() -> Result<i32> {
        let mut n = 0i32;
        check(unsafe { ffi::hipGetDeviceCount(&mut n) })?;
        Ok(n)
    }

    pub fn set_device(device: i32) -> Result<()> {
        check(unsafe { ffi::hipSetDevice(device) })
    }

    pub fn device_name(device: i32) -> Result<String> {
        let mut buf = [0u8; 256];
        check(unsafe { ffi::hipDeviceGetName(buf.as_mut_ptr().cast(), buf.len() as i32, device) })?;
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        Ok(String::from_utf8_lossy(&buf[..end]).into_owned())
    }

    pub fn device_total_mem(device: i32) -> Result<usize> {
        let mut bytes = 0usize;
        check(unsafe { ffi::hipDeviceTotalMem(&mut bytes, device) })?;
        Ok(bytes)
    }

    pub fn mem_get_info() -> Result<(usize, usize)> {
        let (mut free, mut total) = (0usize, 0usize);
        check(unsafe { ffi::hipMemGetInfo(&mut free, &mut total) })?;
        Ok((free, total))
    }

    pub fn device_synchronize() -> Result<()> {
        check(unsafe { ffi::hipDeviceSynchronize() })
    }

    /// Owned device allocation. Frees on drop; copies are synchronous —
    /// async variants land with the first stream-ordered consumer.
    pub struct DeviceBuffer {
        ptr: *mut c_void,
        len: usize,
    }

    impl DeviceBuffer {
        pub fn alloc(len: usize) -> Result<Self> {
            let mut ptr = std::ptr::null_mut();
            check(unsafe { ffi::hipMalloc(&mut ptr, len) })?;
            Ok(Self { ptr, len })
        }

        pub fn len(&self) -> usize {
            self.len
        }

        pub fn is_empty(&self) -> bool {
            self.len == 0
        }

        pub fn as_ptr(&self) -> *mut c_void {
            self.ptr
        }

        pub fn copy_from_host(&mut self, src: &[u8]) -> Result<()> {
            assert!(src.len() <= self.len, "host slice exceeds device buffer");
            check(unsafe {
                ffi::hipMemcpy(
                    self.ptr,
                    src.as_ptr().cast(),
                    src.len(),
                    super::MEMCPY_HOST_TO_DEVICE,
                )
            })
        }

        pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
            assert!(dst.len() <= self.len, "host slice exceeds device buffer");
            check(unsafe {
                ffi::hipMemcpy(
                    dst.as_mut_ptr().cast(),
                    self.ptr,
                    dst.len(),
                    super::MEMCPY_DEVICE_TO_HOST,
                )
            })
        }
    }

    impl Drop for DeviceBuffer {
        fn drop(&mut self) {
            unsafe { ffi::hipFree(self.ptr) };
        }
    }

    pub struct Stream(*mut c_void);

    impl Stream {
        pub fn create() -> Result<Self> {
            let mut s = std::ptr::null_mut();
            check(unsafe { ffi::hipStreamCreate(&mut s) })?;
            Ok(Self(s))
        }

        pub fn synchronize(&self) -> Result<()> {
            check(unsafe { ffi::hipStreamSynchronize(self.0) })
        }

        pub fn as_ptr(&self) -> *mut c_void {
            self.0
        }
    }

    impl Drop for Stream {
        fn drop(&mut self) {
            unsafe { ffi::hipStreamDestroy(self.0) };
        }
    }
}

#[cfg(hip_runtime_available)]
pub use real::{
    DeviceBuffer, Stream, device_count, device_name, device_synchronize, device_total_mem, init,
    mem_get_info, set_device,
};

#[cfg(not(hip_runtime_available))]
mod stub {
    use super::{HIP_NOT_COMPILED, Result};
    use std::ffi::c_void;

    pub fn init() -> Result<()> {
        Err(HIP_NOT_COMPILED)
    }

    pub fn device_count() -> Result<i32> {
        Err(HIP_NOT_COMPILED)
    }

    pub fn set_device(_device: i32) -> Result<()> {
        Err(HIP_NOT_COMPILED)
    }

    pub fn device_name(_device: i32) -> Result<String> {
        Err(HIP_NOT_COMPILED)
    }

    pub fn device_total_mem(_device: i32) -> Result<usize> {
        Err(HIP_NOT_COMPILED)
    }

    pub fn mem_get_info() -> Result<(usize, usize)> {
        Err(HIP_NOT_COMPILED)
    }

    pub fn device_synchronize() -> Result<()> {
        Err(HIP_NOT_COMPILED)
    }

    pub struct DeviceBuffer {
        _private: (),
    }

    impl DeviceBuffer {
        pub fn alloc(_len: usize) -> Result<Self> {
            Err(HIP_NOT_COMPILED)
        }

        pub fn len(&self) -> usize {
            0
        }

        pub fn is_empty(&self) -> bool {
            true
        }

        pub fn as_ptr(&self) -> *mut c_void {
            std::ptr::null_mut()
        }

        pub fn copy_from_host(&mut self, _src: &[u8]) -> Result<()> {
            Err(HIP_NOT_COMPILED)
        }

        pub fn copy_to_host(&self, _dst: &mut [u8]) -> Result<()> {
            Err(HIP_NOT_COMPILED)
        }
    }

    pub struct Stream {
        _private: (),
    }

    impl Stream {
        pub fn create() -> Result<Self> {
            Err(HIP_NOT_COMPILED)
        }

        pub fn synchronize(&self) -> Result<()> {
            Err(HIP_NOT_COMPILED)
        }

        pub fn as_ptr(&self) -> *mut c_void {
            std::ptr::null_mut()
        }
    }
}

#[cfg(not(hip_runtime_available))]
pub use stub::{
    DeviceBuffer, Stream, device_count, device_name, device_synchronize, device_total_mem, init,
    mem_get_info, set_device,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(hip_runtime_available))]
    #[test]
    fn stub_reports_not_compiled() {
        assert_eq!(init().unwrap_err(), HIP_NOT_COMPILED);
        assert_eq!(device_count().unwrap_err(), HIP_NOT_COMPILED);
        assert!(DeviceBuffer::alloc(16).is_err());
        let msg = HIP_NOT_COMPILED.to_string();
        assert!(msg.contains("unavailable"), "{msg}");
    }

    /// Real-hardware smoke: probe + H2D/D2H roundtrip. Skips (does not
    /// fail) when no AMD GPU is present so `--features hip` can run in
    /// link-only CI environments.
    #[cfg(hip_runtime_available)]
    #[test]
    fn probe_and_roundtrip_or_skip() {
        if init().is_err() {
            eprintln!("hip-sys smoke: hipInit failed — skipping (no usable GPU)");
            return;
        }
        let n = device_count().unwrap_or(0);
        if n == 0 {
            eprintln!("hip-sys smoke: 0 devices — skipping");
            return;
        }
        set_device(0).expect("set_device");
        let name = device_name(0).expect("device_name");
        let mem = device_total_mem(0).expect("device_total_mem");
        eprintln!("hip-sys smoke: device 0 = {name}, {} MiB", mem / (1 << 20));

        let src: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        let mut buf = DeviceBuffer::alloc(src.len()).expect("alloc");
        buf.copy_from_host(&src).expect("h2d");
        let mut back = vec![0u8; src.len()];
        buf.copy_to_host(&mut back).expect("d2h");
        device_synchronize().expect("sync");
        assert_eq!(src, back, "H2D/D2H roundtrip mismatch");
    }
}
