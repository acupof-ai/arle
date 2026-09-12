//! Optional node-wide GPU sampling through NVML, dlopen'd with no new link
//! dependency. Off unless `ARLE_OBSERVE_GPU=1`; a missing library or any init
//! failure disables sampling rather than failing startup. Sampled once per
//! observe tick in-process, so it never forks a subprocess on the hot path.

#[cfg(target_os = "linux")]
mod ffi {
    use std::ffi::c_void;

    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlopen(filename: *const std::ffi::c_char, flags: std::ffi::c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const std::ffi::c_char) -> *mut c_void;
    }

    const RTLD_LAZY: std::ffi::c_int = 1;
    pub(super) const NVML_SUCCESS: std::ffi::c_int = 0;
    /// nvmlTemperatureSensors_t::NVML_TEMPERATURE_GPU.
    pub(super) const NVML_TEMPERATURE_GPU: std::ffi::c_int = 0;

    pub(super) type Device = *mut c_void;

    #[repr(C)]
    pub(super) struct Memory {
        pub total: u64,
        pub free: u64,
        pub used: u64,
    }

    #[repr(C)]
    pub(super) struct Utilization {
        pub gpu: u32,
        pub memory: u32,
    }

    pub(super) type Init = unsafe extern "C" fn() -> std::ffi::c_int;
    pub(super) type GetCount = unsafe extern "C" fn(*mut u32) -> std::ffi::c_int;
    pub(super) type GetHandle = unsafe extern "C" fn(u32, *mut Device) -> std::ffi::c_int;
    pub(super) type GetMemory = unsafe extern "C" fn(Device, *mut Memory) -> std::ffi::c_int;
    pub(super) type GetUtil = unsafe extern "C" fn(Device, *mut Utilization) -> std::ffi::c_int;
    pub(super) type GetTemp =
        unsafe extern "C" fn(Device, std::ffi::c_int, *mut u32) -> std::ffi::c_int;
    pub(super) type GetPower = unsafe extern "C" fn(Device, *mut u32) -> std::ffi::c_int;

    pub(super) struct Fns {
        pub init: Init,
        pub count: GetCount,
        pub handle: GetHandle,
        pub memory: GetMemory,
        pub util: GetUtil,
        pub temp: GetTemp,
        pub power: GetPower,
    }

    fn sym<T>(handle: *mut c_void, name: &str) -> Option<T> {
        let cname = std::ffi::CString::new(name).ok()?;
        // SAFETY: `handle` is a successful dlopen result and `cname` is
        // NUL-terminated, so dlsym is sound. A null result returns None; any
        // non-null pointer is reinterpreted as the caller-declared ABI fn
        // pointer (all `T` call sites are the `extern "C" fn` aliases above,
        // same pointer width), so transmute_copy's size assumption holds.
        unsafe {
            let ptr = dlsym(handle, cname.as_ptr());
            (!ptr.is_null()).then(|| std::mem::transmute_copy::<*mut c_void, T>(&ptr))
        }
    }

    /// Resolve every NVML entry point used. Returns None if the library or any
    /// symbol is missing, so a partial NVML cannot publish zeroed gauges.
    pub(super) fn load() -> Option<Fns> {
        let name = std::ffi::CString::new("libnvidia-ml.so.1").ok()?;
        // SAFETY: NUL-terminated library name; null result is checked.
        let handle = unsafe { dlopen(name.as_ptr(), RTLD_LAZY) };
        if handle.is_null() {
            return None;
        }
        Some(Fns {
            init: sym(handle, "nvmlInit_v2")?,
            count: sym(handle, "nvmlDeviceGetCount_v2")?,
            handle: sym(handle, "nvmlDeviceGetHandleByIndex_v2")?,
            memory: sym(handle, "nvmlDeviceGetMemoryInfo")?,
            util: sym(handle, "nvmlDeviceGetUtilizationRates")?,
            temp: sym(handle, "nvmlDeviceGetTemperature")?,
            power: sym(handle, "nvmlDeviceGetPowerUsage")?,
        })
    }
}

pub(crate) struct NvmlSampler {
    #[cfg(target_os = "linux")]
    fns: ffi::Fns,
}

#[cfg(target_os = "linux")]
fn enabled() -> bool {
    matches!(std::env::var("ARLE_OBSERVE_GPU").as_deref(), Ok("1"))
}

#[cfg(target_os = "linux")]
fn bytes_to_mb(bytes: u64) -> u32 {
    (bytes / (1024 * 1024)).min(u32::MAX as u64) as u32
}

#[cfg(target_os = "linux")]
fn mw_to_watts(mw: u32) -> u16 {
    (mw / 1000).min(u16::MAX as u32) as u16
}

impl NvmlSampler {
    /// Construct when opted in and NVML initializes; otherwise None.
    #[cfg(target_os = "linux")]
    pub(crate) fn if_enabled() -> Option<Self> {
        if !enabled() {
            return None;
        }
        let fns = ffi::load()?;
        // SAFETY: resolved NVML entry point with no args.
        if unsafe { (fns.init)() } != ffi::NVML_SUCCESS {
            log::warn!("observe: NVML init failed; GPU sampling disabled");
            return None;
        }
        Some(Self { fns })
    }

    #[cfg(not(target_os = "linux"))]
    pub(crate) fn if_enabled() -> Option<Self> {
        None
    }

    /// Read all visible devices. Returns None if the count is unavailable,
    /// exceeds the fixed 8-device capacity, or any device read is incomplete —
    /// a partially read sample would render a zeroed device that looks real.
    #[cfg(target_os = "linux")]
    pub(crate) fn sample(&self) -> Option<infer_seam::GpuSample> {
        use infer_seam::{GpuDeviceSample, GpuSample};
        let mut count = 0u32;
        // SAFETY: pointer is valid out-param for nvmlDeviceGetCount_v2.
        if unsafe { (self.fns.count)(&mut count) } != ffi::NVML_SUCCESS {
            return None;
        }
        if count > 8 {
            log::warn!("observe: {count} NVML devices exceed the 8-device sample capacity");
            return None;
        }
        let count = count as usize;
        let mut devices = [GpuDeviceSample::default(); 8];
        for (i, slot) in devices.iter_mut().enumerate().take(count) {
            let mut device: ffi::Device = std::ptr::null_mut();
            // SAFETY: index < reported count; pointer is a valid out-param.
            if unsafe { (self.fns.handle)(i as u32, &mut device) } != ffi::NVML_SUCCESS {
                return None;
            }
            let mut mem = std::mem::MaybeUninit::<ffi::Memory>::zeroed();
            let mut util = std::mem::MaybeUninit::<ffi::Utilization>::zeroed();
            let mut temp = 0u32;
            let mut power_mw = 0u32;
            // All four gauges are exported, so a device is filled only when every
            // read succeeds; a partial result would publish a confident zero.
            // SAFETY: each call gets the device handle and an out-param of the
            // declared repr(C) type, read only after the success check.
            unsafe {
                let ok = (self.fns.memory)(device, mem.as_mut_ptr()) == ffi::NVML_SUCCESS
                    && (self.fns.util)(device, util.as_mut_ptr()) == ffi::NVML_SUCCESS
                    && (self.fns.temp)(device, ffi::NVML_TEMPERATURE_GPU, &mut temp)
                        == ffi::NVML_SUCCESS
                    && (self.fns.power)(device, &mut power_mw) == ffi::NVML_SUCCESS;
                if !ok {
                    return None;
                }
                let mem = mem.assume_init();
                let util = util.assume_init();
                *slot = GpuDeviceSample {
                    gpu_index: i as u8,
                    util_pct: util.gpu.min(100) as u8,
                    memory_used_mb: bytes_to_mb(mem.used),
                    memory_total_mb: bytes_to_mb(mem.total),
                    temp_c: temp.min(255) as u8,
                    power_w: mw_to_watts(power_mw),
                };
            }
        }
        Some(GpuSample {
            devices,
            device_count: count as u8,
        })
    }

    #[cfg(not(target_os = "linux"))]
    pub(crate) fn sample(&self) -> Option<infer_seam::GpuSample> {
        None
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn unit_conversions() {
        assert_eq!(bytes_to_mb(0), 0);
        assert_eq!(bytes_to_mb(5 * 1024 * 1024), 5);
        assert_eq!(bytes_to_mb((u32::MAX as u64 + 1) * 1024 * 1024), u32::MAX);
        assert_eq!(mw_to_watts(325_000), 325);
        assert_eq!(mw_to_watts(999), 0);
        assert_eq!(mw_to_watts((u16::MAX as u32 + 1) * 1000), u16::MAX);
    }
}
