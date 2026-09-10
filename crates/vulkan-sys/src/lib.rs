//! Ash-backed Vulkan runtime wrapper for the AIPC Vulkan backend (#71/#76/#77).
//!
//! Feature contract mirrors `hip-sys`:
//! - `--features vulkan`: dynamically loads the Vulkan loader through `ash`.
//! - default: every entry point returns [`VULKAN_NOT_COMPILED`].

pub const VULKAN_NOT_COMPILED: VulkanError = VulkanError::NotCompiled;

pub type Result<T> = std::result::Result<T, VulkanError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VulkanError {
    NotCompiled,
    Runtime(String),
    NoComputeDevice,
    NoMemoryType { type_bits: u32, required_flags: u32 },
    InvalidSpirvLength(usize),
}

impl std::fmt::Display for VulkanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VulkanError::NotCompiled => {
                write!(
                    f,
                    "Vulkan support not compiled (build with --features vulkan)"
                )
            }
            VulkanError::Runtime(msg) => write!(f, "{msg}"),
            VulkanError::NoComputeDevice => {
                write!(f, "no Vulkan device with a compute queue found")
            }
            VulkanError::NoMemoryType {
                type_bits,
                required_flags,
            } => write!(
                f,
                "no Vulkan memory type for bits 0x{type_bits:x} with flags 0x{required_flags:x}"
            ),
            VulkanError::InvalidSpirvLength(len) => {
                write!(f, "SPIR-V bytecode length {len} is not a multiple of 4")
            }
        }
    }
}

impl std::error::Error for VulkanError {}

#[cfg(feature = "vulkan")]
mod real {
    use super::{Result, VulkanError};
    use ash::{Entry, vk};
    use std::ffi::{CStr, CString};

    const REQUIRED_API_VERSION: u32 = vk::API_VERSION_1_2;

    fn runtime_error(context: &str, err: impl std::fmt::Display) -> VulkanError {
        VulkanError::Runtime(format!("{context}: {err}"))
    }

    fn vk_error(context: &str, err: vk::Result) -> VulkanError {
        VulkanError::Runtime(format!("{context}: {err:?}"))
    }

    fn vk_bool(value: vk::Bool32) -> bool {
        value != 0
    }

    /// Map `memory` for `len` bytes at `offset`, copy `src` in, unmap.
    fn write_mapped(
        device: &ash::Device,
        memory: vk::DeviceMemory,
        offset: u64,
        src: &[u8],
        dir: &'static str,
    ) -> Result<()> {
        // SAFETY: `memory` is bound to the caller's buffer with at least
        // `offset + src.len()` bytes; the caller asserts the bounds. The mapping
        // is used only between this call and the matching unmap below.
        let ptr = unsafe {
            device.map_memory(
                memory,
                offset,
                src.len() as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|e| vk_error(&format!("mapping Vulkan buffer for {dir}"), e))?;
        // SAFETY: `ptr` is a valid mapping of exactly `src.len()` bytes, so the
        // non-overlapping copy is in bounds; unmap releases the mapping used here.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), ptr.cast::<u8>(), src.len());
            device.unmap_memory(memory);
        }
        Ok(())
    }

    /// Map `memory` for `len` bytes at `offset`, copy out into `dst`, unmap.
    fn read_mapped(
        device: &ash::Device,
        memory: vk::DeviceMemory,
        offset: u64,
        dst: &mut [u8],
        dir: &'static str,
    ) -> Result<()> {
        // SAFETY: `memory` is bound to the caller's buffer with at least
        // `offset + dst.len()` bytes; the caller asserts the bounds. The mapping
        // is used only between this call and the matching unmap below.
        let ptr = unsafe {
            device.map_memory(
                memory,
                offset,
                dst.len() as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|e| vk_error(&format!("mapping Vulkan buffer for {dir}"), e))?;
        // SAFETY: `ptr` maps exactly `dst.len()` bytes, so the copy into the
        // `dst` buffer is in bounds; unmap releases the mapping used here.
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), dst.as_mut_ptr(), dst.len());
            device.unmap_memory(memory);
        }
        Ok(())
    }

    fn create_instance(entry: &Entry) -> Result<ash::Instance> {
        let app_name = CString::new("arle-vulkan")
            .map_err(|e| runtime_error("building Vulkan app name", e))?;
        let engine_name =
            CString::new("arle").map_err(|e| runtime_error("building Vulkan engine name", e))?;
        let app = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(1)
            .engine_name(&engine_name)
            .engine_version(1)
            .api_version(REQUIRED_API_VERSION);
        let create = vk::InstanceCreateInfo::default().application_info(&app);
        // SAFETY: `entry` is the successfully loaded Vulkan loader and `create`
        // points at valid NUL-terminated app/engine names built above.
        unsafe { entry.create_instance(&create, None) }
            .map_err(|e| vk_error("creating Vulkan instance", e))
    }

    fn has_device_extension(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        name: &CStr,
    ) -> Result<bool> {
        // SAFETY: `physical_device` came from this instance's
        // `enumerate_physical_devices`, so it is a valid handle for it.
        let extensions = unsafe { instance.enumerate_device_extension_properties(physical_device) }
            .map_err(|e| vk_error("enumerating Vulkan device extensions", e))?;
        Ok(extensions.iter().any(|extension| {
            extension
                .extension_name_as_c_str()
                .is_ok_and(|extension_name| extension_name == name)
        }))
    }

    fn supports_required_shader_features(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> bool {
        let mut storage16 = vk::PhysicalDevice16BitStorageFeatures::default();
        let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut integer_dot = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut integer_dot)
            .push_next(&mut vulkan12)
            .push_next(&mut storage16);
        // SAFETY: `physical_device` belongs to `instance`; the pNext chain
        // references the stack-local structs above for the duration of the call.
        unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };

        vk_bool(features2.features.shader_int16)
            && vk_bool(storage16.storage_buffer16_bit_access)
            && vk_bool(vulkan12.storage_buffer8_bit_access)
            && vk_bool(vulkan12.shader_float16)
            && vk_bool(vulkan12.shader_int8)
            && vk_bool(integer_dot.shader_integer_dot_product)
    }

    fn load_entry() -> Result<Entry> {
        // SAFETY: `Entry::load` resolves the Vulkan loader symbols from the
        // process's linked/`dlopen`ed loader; it returns Err when none is present.
        unsafe { Entry::load() }.map_err(|e| runtime_error("loading Vulkan loader", e))
    }

    fn pick_compute_queue(
        instance: &ash::Instance,
    ) -> Result<(vk::PhysicalDevice, u32, vk::PhysicalDeviceProperties)> {
        // SAFETY: `instance` is live and enumeration takes no external inputs.
        let devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| vk_error("enumerating Vulkan physical devices", e))?;
        for physical_device in devices {
            // SAFETY: `physical_device` was just enumerated from `instance`.
            let props = unsafe { instance.get_physical_device_properties(physical_device) };
            if props.api_version < REQUIRED_API_VERSION {
                continue;
            }
            if !has_device_extension(
                instance,
                physical_device,
                vk::KHR_SHADER_INTEGER_DOT_PRODUCT_NAME,
            )? {
                continue;
            }
            if !supports_required_shader_features(instance, physical_device) {
                continue;
            }
            let families = {
                // SAFETY: `physical_device` belongs to `instance`; the returned
                // Vec is owned by ash and valid immediately below.
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
            };
            for (idx, family) in families.iter().enumerate() {
                if family.queue_count > 0 && family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    let queue_family_index = u32::try_from(idx)
                        .map_err(|e| runtime_error("converting Vulkan queue family index", e))?;
                    return Ok((physical_device, queue_family_index, props));
                }
            }
        }
        Err(VulkanError::NoComputeDevice)
    }

    fn device_name_from_properties(props: &vk::PhysicalDeviceProperties) -> String {
        props
            .device_name_as_c_str()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "<invalid name>".to_string())
    }

    pub struct VulkanContext {
        _entry: Entry,
        instance: ash::Instance,
        device: ash::Device,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
        queue: vk::Queue,
        device_name: String,
        /// Device-wide pipeline cache fed to every `createComputePipeline`. The
        /// driver reuses prior compile results (shader binary / layout) across
        /// pipelines that share a backend, collapsing redundant compile work.
        /// Created once at context init; destroyed before the device in `Drop`.
        pipeline_cache: vk::PipelineCache,
    }

    impl VulkanContext {
        pub fn create() -> Result<Self> {
            let entry = load_entry()?;
            let instance = create_instance(&entry)?;
            let picked = match pick_compute_queue(&instance) {
                Ok(picked) => picked,
                Err(e) => {
                    // SAFETY: `instance` was just created and no device exists
                    // yet, so it is exclusively owned and safe to destroy.
                    unsafe { instance.destroy_instance(None) };
                    return Err(e);
                }
            };
            let (physical_device, queue_family_index, props) = picked;
            let priorities = [1.0f32];
            let queue_info = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities)];
            let extensions = [
                vk::KHR_SHADER_INTEGER_DOT_PRODUCT_NAME.as_ptr(),
                vk::EXT_SUBGROUP_SIZE_CONTROL_NAME.as_ptr(),
            ];
            let base_features = vk::PhysicalDeviceFeatures::default().shader_int16(true);
            let mut storage16 =
                vk::PhysicalDevice16BitStorageFeatures::default().storage_buffer16_bit_access(true);
            let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default()
                .storage_buffer8_bit_access(true)
                .shader_float16(true)
                .shader_int8(true);
            let mut integer_dot = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default()
                .shader_integer_dot_product(true);
            // The flash-attn shader hardcodes `SubGroupSize=32` (subgroup shuffle
            // reductions + `num_subgroups = WorkGroupSize/32`). The 8060S defaults
            // to wave64, which scrambles those reductions. Enable subgroup-size
            // control + required-size pipelines so the FA pipeline runs at 32.
            let mut size_control = vk::PhysicalDeviceSubgroupSizeControlFeatures::default()
                .subgroup_size_control(true)
                .compute_full_subgroups(true);
            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .features(base_features)
                .push_next(&mut integer_dot)
                .push_next(&mut vulkan12)
                .push_next(&mut storage16)
                .push_next(&mut size_control);
            let create = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&extensions)
                .push_next(&mut features2);
            let device = {
                // SAFETY: `physical_device` was selected from `instance`; the
                // pNext chain and queue/extension slices are stack-local and live
                // for the call.
                unsafe { instance.create_device(physical_device, &create, None) }
            };
            let device = match device {
                Ok(device) => device,
                Err(e) => {
                    // SAFETY: the failed device left only `instance` to clean up.
                    unsafe { instance.destroy_instance(None) };
                    return Err(vk_error("creating Vulkan device", e));
                }
            };
            // SAFETY: the device owns queue family `queue_family_index`, which
            // enumeration confirmed has a compute queue; queue 0 exists.
            let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
            let pipeline_cache_create = vk::PipelineCacheCreateInfo::default();
            let pipeline_cache = {
                // SAFETY: the device is freshly created and idle; an empty cache
                // create info is valid.
                unsafe { device.create_pipeline_cache(&pipeline_cache_create, None) }
            };
            let pipeline_cache = match pipeline_cache {
                Ok(cache) => cache,
                Err(e) => {
                    unsafe {
                        // SAFETY: nothing else has been created on the device, and
                        // the instance outlives it; both are exclusively owned on
                        // this error path.
                        device.destroy_device(None);
                        instance.destroy_instance(None);
                    }
                    return Err(vk_error("creating Vulkan pipeline cache", e));
                }
            };
            Ok(Self {
                _entry: entry,
                instance,
                device,
                physical_device,
                queue_family_index,
                queue,
                device_name: device_name_from_properties(&props),
                pipeline_cache,
            })
        }

        pub fn device_name(&self) -> &str {
            &self.device_name
        }

        pub fn physical_device(&self) -> vk::PhysicalDevice {
            self.physical_device
        }

        pub fn queue_family_index(&self) -> u32 {
            self.queue_family_index
        }

        pub fn queue(&self) -> vk::Queue {
            self.queue
        }

        pub fn raw_device(&self) -> &ash::Device {
            &self.device
        }

        /// The device-wide pipeline cache. Thread this into every
        /// `createComputePipeline` so the driver can reuse prior compile work
        /// (mirrors `ggml-vulkan` building each pipeline once; here the driver
        /// also de-duplicates shared backend state across pipelines).
        pub fn pipeline_cache(&self) -> vk::PipelineCache {
            self.pipeline_cache
        }

        /// `minStorageBufferOffsetAlignment` (bytes) — every storage-buffer
        /// descriptor offset (e.g. an arena slot's start) must be a multiple of
        /// this. Queried from `vkPhysicalDeviceProperties.limits`.
        pub fn min_storage_buffer_offset_alignment(&self) -> u64 {
            // SAFETY: `self.physical_device` is owned by `self.instance`, both
            // held for `self`'s lifetime; the returned properties are copied out.
            let props = unsafe {
                self.instance
                    .get_physical_device_properties(self.physical_device)
            };
            props.limits.min_storage_buffer_offset_alignment
        }

        /// `(timestampPeriod ns/tick, timestampValidBits)` for GPU timestamp
        /// profiling. `valid_bits == 0` means the compute queue does not support
        /// timestamps (profiling must be disabled).
        pub fn timestamp_info(&self) -> (f32, u32) {
            // SAFETY: both queries read `self.physical_device`, owned by
            // `self.instance` for the context's lifetime; results are copied.
            let props = unsafe {
                self.instance
                    .get_physical_device_properties(self.physical_device)
            };
            // SAFETY: same owned physical device; the queue-family Vec is read
            // only via the bounded index below.
            let qf = unsafe {
                self.instance
                    .get_physical_device_queue_family_properties(self.physical_device)
            };
            let valid_bits = qf
                .get(self.queue_family_index as usize)
                .map(|q| q.timestamp_valid_bits)
                .unwrap_or(0);
            (props.limits.timestamp_period, valid_bits)
        }

        /// `(subgroupSize, min, max)` from `VkPhysicalDeviceSubgroupProperties`
        /// and `VkPhysicalDeviceSubgroupSizeControlProperties`. The flash-attn
        /// shader hardcodes `SubGroupSize=32` and uses subgroup shuffles, so the
        /// pipeline must run at a 32-wide subgroup; this lets the caller diagnose
        /// whether the device defaults to wave64 (needing size control).
        pub fn subgroup_size(&self) -> (u32, u32, u32) {
            let mut size_control = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut size_control);
            // SAFETY: `self.physical_device` is valid for `self.instance`; the
            // pNext chain points at the local struct filled by the query.
            unsafe {
                self.instance
                    .get_physical_device_properties2(self.physical_device, &mut props2);
            }
            let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
            // SAFETY: same valid physical device; the local pNext chain outlives
            // the call.
            unsafe {
                self.instance
                    .get_physical_device_properties2(self.physical_device, &mut p2);
            }
            (
                subgroup.subgroup_size,
                size_control.min_subgroup_size,
                size_control.max_subgroup_size,
            )
        }

        pub fn memory_heaps(&self) -> Vec<(u64, bool)> {
            // SAFETY: `self.physical_device` is owned by `self.instance`; the
            // returned arrays are read only within the copied-out ranges below.
            let props = unsafe {
                self.instance
                    .get_physical_device_memory_properties(self.physical_device)
            };
            (0..props.memory_heap_count as usize)
                .map(|i| {
                    let heap = props.memory_heaps[i];
                    (
                        heap.size,
                        heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
                    )
                })
                .collect()
        }

        pub fn memory_types(&self) -> Vec<(u32, bool, bool)> {
            // SAFETY: valid owned physical device; only copied-out fields are read.
            let props = unsafe {
                self.instance
                    .get_physical_device_memory_properties(self.physical_device)
            };
            (0..props.memory_type_count as usize)
                .map(|i| {
                    let ty = props.memory_types[i];
                    (
                        ty.heap_index,
                        ty.property_flags
                            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL),
                        ty.property_flags
                            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
                    )
                })
                .collect()
        }

        fn memory_type_index(
            &self,
            type_bits: u32,
            required: vk::MemoryPropertyFlags,
        ) -> Result<u32> {
            // SAFETY: valid owned physical device; heap/type indices are bounded
            // by the counts returned in the same `props`.
            let props = unsafe {
                self.instance
                    .get_physical_device_memory_properties(self.physical_device)
            };
            // Among the compatible memory types, prefer the one whose heap is
            // largest. On a unified-memory APU several device-local types map to
            // different-sized heaps (e.g. 32 GB vs 64 GB); steering multi-GB
            // weight allocations to the biggest heap maximizes headroom for the
            // later per-dispatch working set.
            let mut best: Option<(u32, u64)> = None;
            for i in 0..props.memory_type_count {
                let bit = 1u32.checked_shl(i).unwrap_or(0);
                if type_bits & bit == 0 {
                    continue;
                }
                let ty = props.memory_types[i as usize];
                if !ty.property_flags.contains(required) {
                    continue;
                }
                let heap_size = props.memory_heaps[ty.heap_index as usize].size;
                match best {
                    Some((_, best_size)) if best_size >= heap_size => {}
                    _ => best = Some((i, heap_size)),
                }
            }
            best.map(|(i, _)| i).ok_or(VulkanError::NoMemoryType {
                type_bits,
                required_flags: required.as_raw(),
            })
        }
    }

    impl Drop for VulkanContext {
        fn drop(&mut self) {
            // SAFETY: `drop` has exclusive ownership: no other thread can touch
            // the device or instance once `self` is being dropped. The pipeline
            // cache is destroyed before its parent device, the device before its
            // instance, matching creation order.
            unsafe {
                self.device
                    .destroy_pipeline_cache(self.pipeline_cache, None);
                self.device.destroy_device(None);
                self.instance.destroy_instance(None);
            }
        }
    }

    /// Initialize the Vulkan loader. Idempotent.
    pub fn init() -> Result<()> {
        let _entry = load_entry()?;
        Ok(())
    }

    pub fn device_count() -> Result<usize> {
        let entry = load_entry()?;
        let instance = create_instance(&entry)?;
        // SAFETY: `instance` is live and takes no handle input.
        let devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| vk_error("enumerating Vulkan physical devices", e));
        // SAFETY: no device was created from this throwaway instance, so it is
        // exclusively owned at destroy.
        unsafe { instance.destroy_instance(None) };
        devices.map(|d| d.len())
    }

    pub fn device_name(device_index: usize) -> Result<String> {
        let entry = load_entry()?;
        let instance = create_instance(&entry)?;
        // SAFETY: live instance, no handle argument.
        let devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| vk_error("enumerating Vulkan physical devices", e))?;
        let name = match devices.get(device_index) {
            Some(device) => {
                // SAFETY: `device` was just enumerated from `instance`.
                let props = unsafe { instance.get_physical_device_properties(*device) };
                Ok(device_name_from_properties(&props))
            }
            None => Err(VulkanError::Runtime(format!(
                "Vulkan device index {device_index} out of range ({} devices)",
                devices.len()
            ))),
        };
        // SAFETY: throwaway instance with no child device.
        unsafe { instance.destroy_instance(None) };
        name
    }

    pub struct DeviceBuffer<'a> {
        ctx: &'a VulkanContext,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        len: usize,
    }

    impl<'a> DeviceBuffer<'a> {
        pub fn alloc(ctx: &'a VulkanContext, len: usize) -> Result<Self> {
            Self::alloc_with_usage(
                ctx,
                len,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        }

        /// Allocate a UMA storage buffer: `DEVICE_LOCAL | HOST_VISIBLE |
        /// HOST_COHERENT`. On the Strix Halo APU the big device-local heap is
        /// host-mappable, so the GEMV activation arena lives here — the GPU reads
        /// it at device-local speed while the host writes the input / reads the
        /// result with zero staging. Falls back to a plain host-visible buffer if
        /// the device exposes no device-local + host-visible memory type (keeps
        /// non-UMA boxes working, just without the device-local win).
        pub fn alloc_uma(ctx: &'a VulkanContext, len: usize) -> Result<Self> {
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST;
            Self::alloc_with_usage(
                ctx,
                len,
                usage,
                vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .or_else(|_| {
                Self::alloc_with_usage(
                    ctx,
                    len,
                    usage,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
            })
        }

        pub fn alloc_with_usage(
            ctx: &'a VulkanContext,
            len: usize,
            usage: vk::BufferUsageFlags,
            memory_flags: vk::MemoryPropertyFlags,
        ) -> Result<Self> {
            if len == 0 {
                return Err(VulkanError::Runtime(
                    "Vulkan buffer allocation size must be non-zero".to_string(),
                ));
            }
            let create = vk::BufferCreateInfo::default()
                .size(len as vk::DeviceSize)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            // SAFETY: the device is held by `ctx`; the create info references
            // only stack-local data valid for the call.
            let buffer = unsafe { ctx.device.create_buffer(&create, None) }
                .map_err(|e| vk_error("creating Vulkan buffer", e))?;
            // SAFETY: `buffer` is a fresh handle from `create_buffer`.
            let req = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
            let memory_type_index = match ctx.memory_type_index(req.memory_type_bits, memory_flags)
            {
                Ok(idx) => idx,
                Err(e) => {
                    // SAFETY: memory was not yet allocated for `buffer`, which is
                    // unbound and exclusively held on this error path.
                    unsafe { ctx.device.destroy_buffer(buffer, None) };
                    return Err(e);
                }
            };
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(memory_type_index);
            // SAFETY: `memory_type_index` is one of the device's types from the
            // requirements above, and `alloc.size == req.size`, so the allocation
            // satisfies `buffer`'s requirements.
            let memory = match unsafe { ctx.device.allocate_memory(&alloc, None) } {
                Ok(memory) => memory,
                Err(e) => {
                    // SAFETY: `buffer` is unbound (bind happens below) and only
                    // this function holds it.
                    unsafe { ctx.device.destroy_buffer(buffer, None) };
                    return Err(vk_error("allocating Vulkan buffer memory", e));
                }
            };
            // SAFETY: `buffer` and `memory` are fresh, unbound handles and
            // `memory` was sized from `buffer`'s own memory requirements.
            if let Err(e) = unsafe { ctx.device.bind_buffer_memory(buffer, memory, 0) } {
                unsafe {
                    // SAFETY: on bind failure ownership stays here; free memory
                    // then the unbound buffer in reverse creation order.
                    ctx.device.free_memory(memory, None);
                    ctx.device.destroy_buffer(buffer, None);
                }
                return Err(vk_error("binding Vulkan buffer memory", e));
            }
            Ok(Self {
                ctx,
                buffer,
                memory,
                len,
            })
        }

        pub fn len(&self) -> usize {
            self.len
        }

        pub fn is_empty(&self) -> bool {
            self.len == 0
        }

        pub fn raw(&self) -> vk::Buffer {
            self.buffer
        }

        pub fn copy_from_host(&mut self, src: &[u8]) -> Result<()> {
            self.copy_from_host_at(0, src)
        }

        pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
            self.copy_to_host_at(0, dst)
        }

        /// Map `src.len()` bytes at byte `offset` and write `src` into them. The
        /// arena uses this to land one GEMV's input activation into a named slot
        /// with no per-call allocation (host-visible/UMA memory). `offset` need
        /// not honor the storage-buffer alignment for the *map* itself, but the
        /// caller binds the slot via an aligned descriptor offset.
        pub fn copy_from_host_at(&mut self, offset: u64, src: &[u8]) -> Result<()> {
            assert!(
                offset as usize + src.len() <= self.len,
                "host slice + offset exceeds Vulkan buffer"
            );
            if src.is_empty() {
                return Ok(());
            }
            // SAFETY rationale lives in `write_mapped`; the assert above bounds
            // the slice to this buffer's allocated length.
            write_mapped(&self.ctx.device, self.memory, offset, src, "H2D")
        }

        pub fn copy_to_host_at(&self, offset: u64, dst: &mut [u8]) -> Result<()> {
            assert!(
                offset as usize + dst.len() <= self.len,
                "host slice + offset exceeds Vulkan buffer"
            );
            if dst.is_empty() {
                return Ok(());
            }
            // SAFETY rationale lives in `read_mapped`; the assert above bounds
            // the slice to this buffer's allocated length.
            read_mapped(&self.ctx.device, self.memory, offset, dst, "D2H")
        }

        /// Allocate a **DEVICE_LOCAL** (not host-visible) buffer and fill it from
        /// `src` via a temporary host-visible staging buffer + a one-shot copy.
        ///
        /// Resident weights are written once and only read by the GPU, so they
        /// belong in device-local memory. On a unified-memory APU the
        /// host-visible heap is a smaller carve-out; routing multi-GB weights
        /// through it (as `alloc` + `copy_from_host` does) exhausts it and makes
        /// later per-dispatch submits fail with `ERROR_OUT_OF_DEVICE_MEMORY`.
        /// Staging keeps the big buffers on the large device-local heap and
        /// frees the staging copy immediately.
        pub fn alloc_device_local_from_host(ctx: &'a VulkanContext, src: &[u8]) -> Result<Self> {
            let dst = Self::alloc_with_usage(
                ctx,
                src.len().max(1),
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            if src.is_empty() {
                return Ok(dst);
            }
            let mut staging = Self::alloc_with_usage(
                ctx,
                src.len(),
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            staging.copy_from_host(src)?;
            let pool = CommandPool::create(ctx)?;
            pool.one_shot_submit(|cmd| {
                let region = vk::BufferCopy::default().size(src.len() as vk::DeviceSize);
                // SAFETY: `cmd` is the recording primary buffer handed to this
                // closure; both buffers were created with TRANSFER usage and
                // `dst` holds `region.size` bytes.
                unsafe {
                    ctx.device
                        .cmd_copy_buffer(cmd, staging.buffer, dst.buffer, &[region]);
                }
                Ok(())
            })?;
            Ok(dst)
        }
    }

    impl Drop for DeviceBuffer<'_> {
        fn drop(&mut self) {
            // SAFETY: `drop` gives exclusive ownership; the buffer and its memory
            // were created together from `ctx` and are not in use (all GPU work
            // using them is fence-waited before the owning weights are dropped).
            unsafe {
                self.ctx.device.destroy_buffer(self.buffer, None);
                self.ctx.device.free_memory(self.memory, None);
            }
        }
    }

    pub struct CommandPool<'a> {
        ctx: &'a VulkanContext,
        pool: vk::CommandPool,
    }

    impl<'a> CommandPool<'a> {
        pub fn create(ctx: &'a VulkanContext) -> Result<Self> {
            let create = vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            // SAFETY: the device and queue family index both belong to `ctx`.
            let pool = unsafe { ctx.device.create_command_pool(&create, None) }
                .map_err(|e| vk_error("creating Vulkan command pool", e))?;
            Ok(Self { ctx, pool })
        }

        pub fn raw(&self) -> vk::CommandPool {
            self.pool
        }

        pub fn one_shot_submit<F>(&self, record: F) -> Result<()>
        where
            F: FnOnce(vk::CommandBuffer) -> Result<()>,
        {
            let alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: `self.pool` is live and the alloc info only references it.
            let buffers = unsafe { self.ctx.device.allocate_command_buffers(&alloc) }
                .map_err(|e| vk_error("allocating Vulkan command buffer", e))?;
            let command_buffer = buffers.first().copied().ok_or_else(|| {
                VulkanError::Runtime(
                    "Vulkan command buffer allocation returned no buffers".to_string(),
                )
            })?;
            let result = (|| {
                let begin = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                // SAFETY: `command_buffer` is freshly allocated and not being
                // recorded elsewhere.
                unsafe { self.ctx.device.begin_command_buffer(command_buffer, &begin) }
                    .map_err(|e| vk_error("beginning Vulkan command buffer", e))?;
                record(command_buffer)?;
                // SAFETY: the closure ended recording successfully.
                unsafe { self.ctx.device.end_command_buffer(command_buffer) }
                    .map_err(|e| vk_error("ending Vulkan command buffer", e))?;
                let command_buffers = [command_buffer];
                let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
                // SAFETY: the buffer is in the executable state after end, and the
                // single queue belongs to this device; the NULL fence is paired
                // with the immediate `queue_wait_idle` below.
                unsafe {
                    self.ctx
                        .device
                        .queue_submit(self.ctx.queue, &submits, vk::Fence::null())
                }
                .map_err(|e| vk_error("submitting Vulkan command buffer", e))?;
                // SAFETY: waiting the device's own queue needs no external input.
                unsafe { self.ctx.device.queue_wait_idle(self.ctx.queue) }
                    .map_err(|e| vk_error("waiting for Vulkan queue idle", e))?;
                Ok(())
            })();
            // SAFETY: queue is idle, so the GPU has finished with
            // `command_buffer`, which belongs to `self.pool`.
            unsafe {
                self.ctx
                    .device
                    .free_command_buffers(self.pool, &[command_buffer]);
            }
            result
        }
    }

    impl Drop for CommandPool<'_> {
        fn drop(&mut self) {
            // SAFETY: exclusive ownership at drop; all buffers from this pool are
            // freed by their one-shot paths before the pool goes.
            unsafe { self.ctx.device.destroy_command_pool(self.pool, None) };
        }
    }

    /// Per-dispatch GPU timestamp profiler (ARLE_GPU_TIMESTAMPS=1). Writes a
    /// `vkCmdWriteTimestamp` after each dispatch; the delta between consecutive
    /// timestamps is that dispatch's GPU time, accumulated by category label.
    /// Mirrors what `GGML_VK_PERF_LOGGER` does for llama.cpp so the two per-op
    /// breakdowns can be compared directly.
    struct GpuProf {
        pool: vk::QueryPool,
        capacity: u32,
        period_ns: f32,
        valid_mask: u64,
        idx: u32,
        labels: Vec<&'static str>,
        next_label: &'static str,
        totals: std::collections::HashMap<&'static str, (u64, u128)>,
    }

    /// Records many compute dispatches into **one** primary command buffer and
    /// submits them with a **single** `vkQueueSubmit` on a reused fence — the
    /// `ggml-vulkan` decode sync model.
    ///
    /// This replaces the per-dispatch `CommandPool::one_shot_submit` drain
    /// (alloc + record + submit(NULL fence) + `queue_wait_idle` + free, once per
    /// op) on the forward path. `one_shot_submit` stays only for cold weight
    /// upload; the hot per-token decode graph records here.
    ///
    /// Lifecycle: `begin()` → N × (`dispatch()` / `barrier()`) →
    /// `submit_and_wait()`. The command buffer and fence are allocated once and
    /// reused across tokens (the pool is created with
    /// `RESET_COMMAND_BUFFER`, so `begin()` can reset the buffer in place).
    pub struct CommandRecorder<'a> {
        ctx: &'a VulkanContext,
        pool: vk::CommandPool,
        command_buffer: vk::CommandBuffer,
        fence: vk::Fence,
        /// Optional per-dispatch GPU timestamp profiler (ARLE_GPU_TIMESTAMPS).
        prof: Option<GpuProf>,
        /// True between `submit_and_wait()`'s `queue_submit` and the next
        /// `begin()`'s fence wait — guards against re-recording a buffer whose
        /// prior submission has not yet been waited on.
        pending: bool,
        /// Number of dispatches recorded into the CURRENTLY-OPEN batch (reset by
        /// `begin()`, read by the batch-cadence cap so a token's recorder can
        /// submit before the command buffer trips the APU TDR watchdog).
        dispatches_in_batch: u64,
        /// Total `vkQueueSubmit` calls over the recorder's life — the submits/token
        /// instrument. The forward loop snapshots this around a token to report the
        /// per-token submit count.
        submit_count: u64,
    }

    impl<'a> CommandRecorder<'a> {
        pub fn new(ctx: &'a VulkanContext) -> Result<Self> {
            let pool_create = vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            // SAFETY: device and queue family both belong to `ctx`.
            let pool = unsafe { ctx.device.create_command_pool(&pool_create, None) }
                .map_err(|e| vk_error("creating Vulkan command pool", e))?;

            let alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: `pool` is the freshly created, live command pool.
            let command_buffer = match unsafe { ctx.device.allocate_command_buffers(&alloc) } {
                Ok(buffers) => match buffers.first().copied() {
                    Some(buffer) => buffer,
                    None => {
                        // SAFETY: the pool owns no buffers on this path.
                        unsafe { ctx.device.destroy_command_pool(pool, None) };
                        return Err(VulkanError::Runtime(
                            "Vulkan command buffer allocation returned no buffers".to_string(),
                        ));
                    }
                },
                Err(e) => {
                    // SAFETY: nothing was allocated from the failed pool.
                    unsafe { ctx.device.destroy_command_pool(pool, None) };
                    return Err(vk_error("allocating Vulkan command buffer", e));
                }
            };

            let fence_create = vk::FenceCreateInfo::default();
            // SAFETY: empty create info and a live device are valid inputs.
            let fence = match unsafe { ctx.device.create_fence(&fence_create, None) } {
                Ok(fence) => fence,
                Err(e) => {
                    // SAFETY: on failure the command buffer is still owned by
                    // `pool`, so destroying the pool frees it too.
                    unsafe { ctx.device.destroy_command_pool(pool, None) };
                    return Err(vk_error("creating Vulkan fence", e));
                }
            };

            let prof = if std::env::var("ARLE_GPU_TIMESTAMPS").is_ok() {
                let (period_ns, valid_bits) = ctx.timestamp_info();
                if valid_bits == 0 {
                    eprintln!("ARLE_GPU_TIMESTAMPS: compute queue has no timestamp support");
                    None
                } else {
                    let capacity = 8192u32;
                    let info = vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(capacity);
                    // SAFETY: live device; the query count and type are the only
                    // inputs and both are valid.
                    match unsafe { ctx.device.create_query_pool(&info, None) } {
                        Ok(qpool) => Some(GpuProf {
                            pool: qpool,
                            capacity,
                            period_ns,
                            valid_mask: if valid_bits >= 64 {
                                u64::MAX
                            } else {
                                (1u64 << valid_bits) - 1
                            },
                            idx: 0,
                            labels: Vec::new(),
                            next_label: "other",
                            totals: std::collections::HashMap::new(),
                        }),
                        Err(e) => {
                            eprintln!("ARLE_GPU_TIMESTAMPS: query pool create failed: {e}");
                            None
                        }
                    }
                }
            } else {
                None
            };

            Ok(Self {
                ctx,
                pool,
                command_buffer,
                fence,
                prof,
                pending: false,
                dispatches_in_batch: 0,
                submit_count: 0,
            })
        }

        pub fn label_next(&mut self, label: &'static str) {
            if let Some(p) = self.prof.as_mut() {
                p.next_label = label;
            }
        }

        pub fn take_gpu_profile(&mut self) -> Vec<(&'static str, u64, f64)> {
            match self.prof.as_mut() {
                Some(p) => {
                    let period = p.period_ns as f64;
                    let mut v: Vec<_> = p
                        .totals
                        .drain()
                        .map(|(k, (c, ticks))| (k, c, ticks as f64 * period / 1e6))
                        .collect();
                    v.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
                    v
                }
                None => Vec::new(),
            }
        }

        /// Dispatches recorded into the currently-open batch (since the last
        /// `begin()`). The forward loop checks this against a cadence cap and
        /// flushes (submit + re-begin) before a single command buffer grows large
        /// enough to trip the APU TDR watchdog (mirrors `ggml-vulkan`'s
        /// `submitted_nodes >= 100` batch flush).
        pub fn dispatches_in_batch(&self) -> u64 {
            self.dispatches_in_batch
        }

        pub fn submit_count(&self) -> u64 {
            self.submit_count
        }

        /// Open the buffer for a fresh batch. Waits for any prior submission's
        /// fence (never re-record before the GPU is done — bugs there read as
        /// numeric corruption), resets the fence, then resets + begins the
        /// command buffer.
        pub fn begin(&mut self) -> Result<()> {
            if self.pending {
                // SAFETY: `self.fence` is the reusable fence submitted with the
                // in-flight batch; waiting an infinite timeout returns only when
                // the GPU is done.
                unsafe {
                    self.ctx
                        .device
                        .wait_for_fences(&[self.fence], true, u64::MAX)
                }
                .map_err(|e| vk_error("waiting for Vulkan fence", e))?;
                self.pending = false;
            }
            // SAFETY: the fence is now signalled (or was never submitted), the
            // only state in which reset is valid.
            unsafe { self.ctx.device.reset_fences(&[self.fence]) }
                .map_err(|e| vk_error("resetting Vulkan fence", e))?;
            // SAFETY: the waited fence means no recording is in flight, so the
            // buffer is idle and resettable (the pool allows reset).
            unsafe {
                self.ctx
                    .device
                    .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
            }
            .map_err(|e| vk_error("resetting Vulkan command buffer", e))?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            // SAFETY: the buffer was just reset and is not being recorded.
            unsafe {
                self.ctx
                    .device
                    .begin_command_buffer(self.command_buffer, &begin)
            }
            .map_err(|e| vk_error("beginning Vulkan command buffer", e))?;
            self.dispatches_in_batch = 0;
            if let Some((pool, cap)) = self.prof.as_ref().map(|p| (p.pool, p.capacity)) {
                let cmd = self.command_buffer;
                // SAFETY: `cmd` is recording and the timestamp query pool was
                // created with `cap` TIMESTAMP queries, so resetting [0, cap) and
                // writing query 0 at TOP_OF_PIPE are in range.
                unsafe {
                    self.ctx.device.cmd_reset_query_pool(cmd, pool, 0, cap);
                    self.ctx.device.cmd_write_timestamp(
                        cmd,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        pool,
                        0,
                    );
                }
                if let Some(p) = self.prof.as_mut() {
                    p.idx = 1;
                    p.labels.clear();
                    p.next_label = "other";
                }
            }
            Ok(())
        }

        /// Record one compute dispatch (bind pipeline + descriptor set, push
        /// constants, dispatch) into the open buffer. No submit, no drain — this
        /// is exactly the body of `one_shot_submit`'s closure as used on the
        /// kernel path, minus the submit.
        pub fn dispatch(
            &mut self,
            pipeline: &ComputePipeline<'_>,
            set: &DescriptorSet<'_>,
            push: &[u8],
            groups: [u32; 3],
        ) {
            self.dispatch_raw(pipeline, set.raw(), push, groups);
        }

        /// Same as [`Self::dispatch`] but binds an already-resolved raw
        /// `VkDescriptorSet` (e.g. the next slot of a [`DescriptorSetRing`]),
        /// avoiding the per-dispatch `DescriptorSet` (pool) creation. The caller
        /// must keep the underlying set valid until `submit_and_wait`.
        pub fn dispatch_raw(
            &mut self,
            pipeline: &ComputePipeline<'_>,
            set: vk::DescriptorSet,
            push: &[u8],
            groups: [u32; 3],
        ) {
            let device = &self.ctx.device;
            let cmd = self.command_buffer;
            // SAFETY: `cmd` is in the recording state between `begin` and
            // `submit_and_wait`. The pipeline was created with `pipeline.layout`,
            // the descriptor set was allocated for that layout, and `push` is
            // within the pipeline's declared push-constant range; `groups` is
            // caller-supplied dispatch geometry with no handle validity concern.
            unsafe {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.raw());
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.layout(),
                    0,
                    &[set],
                    &[],
                );
                if !push.is_empty() {
                    device.cmd_push_constants(
                        cmd,
                        pipeline.layout(),
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        push,
                    );
                }
                device.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
            }
            self.dispatches_in_batch += 1;
            let slot = self.prof.as_mut().and_then(|p| {
                if p.idx < p.capacity {
                    let s = (p.pool, p.idx);
                    p.labels.push(p.next_label);
                    p.idx += 1;
                    p.next_label = "other";
                    Some(s)
                } else {
                    None
                }
            });
            if let Some((pool, idx)) = slot {
                // SAFETY: `cmd` is still recording and `idx < capacity` (the slot
                // closure above bounds it); BOTTOM_OF_PIPE is a valid stage.
                unsafe {
                    self.ctx.device.cmd_write_timestamp(
                        cmd,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        pool,
                        idx,
                    );
                }
            }
        }

        /// Record a single compute→compute execution+memory barrier so a later
        /// dispatch reads the writes of the earlier one. Mirrors
        /// `ggml-vulkan.cpp:2717-2737` `ggml_vk_sync_buffers` (one
        /// `vkCmdPipelineBarrier`, global `MemoryBarrier`, SHADER_WRITE →
        /// SHADER_READ|SHADER_WRITE over the COMPUTE_SHADER stage).
        pub fn barrier(&mut self) {
            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            // SAFETY: `self.command_buffer` is recording; the global barrier uses
            // no handles, and the stage/access masks are compile-time constants.
            unsafe {
                self.ctx.device.cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[memory_barrier],
                    &[],
                    &[],
                );
            }
        }

        /// End the buffer, submit it with **one** `vkQueueSubmit` **on the
        /// fence** (no NULL fence, no `queue_wait_idle`), then wait the fence.
        /// Mirrors `ggml-vulkan.cpp:2278-2355` (one submit) +
        /// `2037-2067`/`13474-13485` (one fence wait per batch). A YIELD-spin
        /// tail-latency variant can replace the blocking wait later.
        pub fn submit_and_wait(&mut self) -> Result<()> {
            // SAFETY: recording finished without error; the buffer is in the
            // executable state required for submit.
            unsafe { self.ctx.device.end_command_buffer(self.command_buffer) }
                .map_err(|e| vk_error("ending Vulkan command buffer", e))?;
            let command_buffers = [self.command_buffer];
            let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            // SAFETY: the buffer is executable, the queue is this device's
            // compute queue, and `self.fence` is unsignalled after `begin`'s
            // reset, so one submit can signal it.
            unsafe {
                self.ctx
                    .device
                    .queue_submit(self.ctx.queue, &submits, self.fence)
            }
            .map_err(|e| vk_error("submitting Vulkan command buffer", e))?;
            self.pending = true;
            self.submit_count += 1;
            // SAFETY: the fence was just submitted and will become signalled; an
            // infinite timeout blocks until the batch completes.
            unsafe {
                self.ctx
                    .device
                    .wait_for_fences(&[self.fence], true, u64::MAX)
            }
            .map_err(|e| vk_error("waiting for Vulkan fence", e))?;
            self.pending = false;
            let read = self.prof.as_ref().map(|p| (p.pool, p.idx, p.valid_mask));
            if let Some((pool, idx, mask)) = read
                && idx > 1
            {
                let mut data = vec![0u64; idx as usize];
                // SAFETY: the waited fence means the recorded query writes are
                // available; `data.len() == idx` matches the [0, idx) range
                // and the pool was created with >= idx TIMESTAMP queries.
                let ok = unsafe {
                    self.ctx.device.get_query_pool_results(
                        pool,
                        0,
                        &mut data,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                };
                if ok.is_ok()
                    && let Some(p) = self.prof.as_mut()
                {
                    for j in 0..p.labels.len() {
                        let a = data[j] & mask;
                        let b = data[j + 1] & mask;
                        let dt = b.wrapping_sub(a) as u128;
                        let e = p.totals.entry(p.labels[j]).or_insert((0u64, 0u128));
                        e.0 += 1;
                        e.1 += dt;
                    }
                }
            }
            Ok(())
        }
    }

    impl Drop for CommandRecorder<'_> {
        fn drop(&mut self) {
            // SAFETY: drop is exclusive. The fence wait below (only when a batch
            // is pending) guarantees the GPU has finished with the command buffer
            // before its pool, the fence, and the optional query pool are
            // destroyed; creation order is reversed.
            unsafe {
                // The fence guarantees the GPU is done with the buffer before we
                // free its pool; only wait if a submission is still in flight.
                if self.pending {
                    let _ = self
                        .ctx
                        .device
                        .wait_for_fences(&[self.fence], true, u64::MAX);
                }
                if let Some(p) = self.prof.as_ref() {
                    self.ctx.device.destroy_query_pool(p.pool, None);
                }
                self.ctx.device.destroy_fence(self.fence, None);
                self.ctx.device.destroy_command_pool(self.pool, None);
            }
        }
    }

    pub struct ShaderModule<'a> {
        ctx: &'a VulkanContext,
        module: vk::ShaderModule,
    }

    impl<'a> ShaderModule<'a> {
        pub fn from_spirv_bytes(ctx: &'a VulkanContext, bytes: &[u8]) -> Result<Self> {
            if !bytes.len().is_multiple_of(4) {
                return Err(VulkanError::InvalidSpirvLength(bytes.len()));
            }
            let words: Vec<u32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect();
            Self::from_spirv_words(ctx, &words)
        }

        pub fn from_spirv_words(ctx: &'a VulkanContext, words: &[u32]) -> Result<Self> {
            let create = vk::ShaderModuleCreateInfo::default().code(words);
            // SAFETY: the device is held by `ctx`; `words` is a valid SPIR-V
            // blob (length-multiple-of-4 checked by the caller), referenced for
            // the create call only.
            let module = unsafe { ctx.device.create_shader_module(&create, None) }
                .map_err(|e| vk_error("creating Vulkan shader module", e))?;
            Ok(Self { ctx, module })
        }

        pub fn raw(&self) -> vk::ShaderModule {
            self.module
        }
    }

    impl Drop for ShaderModule<'_> {
        fn drop(&mut self) {
            // SAFETY: exclusive ownership at drop; the module is not referenced by
            // any live pipeline (pipelines are built from it but do not keep it).
            unsafe { self.ctx.device.destroy_shader_module(self.module, None) };
        }
    }

    pub struct DescriptorSetLayout<'a> {
        ctx: &'a VulkanContext,
        layout: vk::DescriptorSetLayout,
    }

    impl<'a> DescriptorSetLayout<'a> {
        pub fn storage_buffers(ctx: &'a VulkanContext, binding_count: usize) -> Result<Self> {
            let binding_count = u32::try_from(binding_count)
                .map_err(|e| runtime_error("converting descriptor binding count", e))?;
            if binding_count == 0 {
                return Err(VulkanError::Runtime(
                    "descriptor layout needs at least one storage-buffer binding".to_string(),
                ));
            }
            let bindings: Vec<_> = (0..binding_count)
                .map(|binding| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let create = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            // SAFETY: the device is held by `ctx`; `bindings` is stack-local and
            // all values are compile-time-valid descriptor descriptions.
            let layout = unsafe { ctx.device.create_descriptor_set_layout(&create, None) }
                .map_err(|e| vk_error("creating Vulkan descriptor set layout", e))?;
            Ok(Self { ctx, layout })
        }

        pub fn raw(&self) -> vk::DescriptorSetLayout {
            self.layout
        }
    }

    impl Drop for DescriptorSetLayout<'_> {
        fn drop(&mut self) {
            // SAFETY: exclusive ownership at drop; descriptor sets allocated for
            // a layout do not keep it alive, and this layout's sets belong to
            // pools dropped first (see DescriptorSet/ring Drop).
            unsafe {
                self.ctx
                    .device
                    .destroy_descriptor_set_layout(self.layout, None)
            };
        }
    }

    pub struct DescriptorSet<'a> {
        ctx: &'a VulkanContext,
        pool: vk::DescriptorPool,
        set: vk::DescriptorSet,
    }

    impl<'a> DescriptorSet<'a> {
        pub fn storage_buffers(
            ctx: &'a VulkanContext,
            layout: &DescriptorSetLayout<'_>,
            buffers: &[&DeviceBuffer<'_>],
        ) -> Result<Self> {
            let descriptor_count = u32::try_from(buffers.len())
                .map_err(|e| runtime_error("converting descriptor buffer count", e))?;
            if descriptor_count == 0 {
                return Err(VulkanError::Runtime(
                    "descriptor set needs at least one storage buffer".to_string(),
                ));
            }
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(descriptor_count)];
            let pool_create = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes);
            // SAFETY: live device; the one pool size covers the single set's
            // STORAGE_BUFFER descriptors counted above.
            let pool = unsafe { ctx.device.create_descriptor_pool(&pool_create, None) }
                .map_err(|e| vk_error("creating Vulkan descriptor pool", e))?;
            let layouts = [layout.raw()];
            let alloc = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            // SAFETY: `pool` has max_sets(1) and enough descriptors for `layout`.
            let sets = match unsafe { ctx.device.allocate_descriptor_sets(&alloc) } {
                Ok(sets) => sets,
                Err(e) => {
                    // SAFETY: failed allocation leaves the pool holding no sets.
                    unsafe { ctx.device.destroy_descriptor_pool(pool, None) };
                    return Err(vk_error("allocating Vulkan descriptor set", e));
                }
            };
            let set = match sets.first().copied() {
                Some(set) => set,
                None => {
                    // SAFETY: empty result means no set was handed out.
                    unsafe { ctx.device.destroy_descriptor_pool(pool, None) };
                    return Err(VulkanError::Runtime(
                        "Vulkan descriptor allocation returned no sets".to_string(),
                    ));
                }
            };
            let infos: Vec<_> = buffers
                .iter()
                .map(|buf| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(buf.raw())
                        .offset(0)
                        .range(buf.len() as vk::DeviceSize)
                })
                .collect();
            let writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(idx, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(idx as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();
            // SAFETY: every write targets `set` (allocated from `pool`) and
            // references a live DeviceBuffer whose byte range covers the
            // descriptor; there are no copy descriptors.
            unsafe { ctx.device.update_descriptor_sets(&writes, &[]) };
            Ok(Self { ctx, pool, set })
        }

        /// Bind a descriptor set to **sub-ranges** of buffers: each entry is
        /// `(buffer, offset_bytes, range_bytes)`. The shader sees each bound
        /// range as starting at index 0 (Vulkan applies the descriptor offset),
        /// so this is what threads an activation arena's named slots into the
        /// per-GEMV bindings without a per-call allocation. Every `offset` MUST
        /// honor the device's `minStorageBufferOffsetAlignment` (query via
        /// [`VulkanContext::min_storage_buffer_offset_alignment`]) or the bind is
        /// invalid. Unlike [`Self::storage_buffers`] (which hardcodes offset 0 /
        /// full range), this is the ranged form the arena needs.
        pub fn storage_buffers_ranged(
            ctx: &'a VulkanContext,
            layout: &DescriptorSetLayout<'_>,
            buffers: &[(&DeviceBuffer<'_>, u64, u64)],
        ) -> Result<Self> {
            let descriptor_count = u32::try_from(buffers.len())
                .map_err(|e| runtime_error("converting descriptor buffer count", e))?;
            if descriptor_count == 0 {
                return Err(VulkanError::Runtime(
                    "descriptor set needs at least one storage buffer".to_string(),
                ));
            }
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(descriptor_count)];
            let pool_create = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes);
            // SAFETY: live device; the one pool size covers the single set's
            // STORAGE_BUFFER descriptors counted above.
            let pool = unsafe { ctx.device.create_descriptor_pool(&pool_create, None) }
                .map_err(|e| vk_error("creating Vulkan descriptor pool", e))?;
            let layouts = [layout.raw()];
            let alloc = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            // SAFETY: `pool` has max_sets(1) and enough descriptors for `layout`.
            let sets = match unsafe { ctx.device.allocate_descriptor_sets(&alloc) } {
                Ok(sets) => sets,
                Err(e) => {
                    // SAFETY: failed allocation leaves the pool holding no sets.
                    unsafe { ctx.device.destroy_descriptor_pool(pool, None) };
                    return Err(vk_error("allocating Vulkan descriptor set", e));
                }
            };
            let set = match sets.first().copied() {
                Some(set) => set,
                None => {
                    // SAFETY: empty result means no set was handed out.
                    unsafe { ctx.device.destroy_descriptor_pool(pool, None) };
                    return Err(VulkanError::Runtime(
                        "Vulkan descriptor allocation returned no sets".to_string(),
                    ));
                }
            };
            let infos: Vec<_> = buffers
                .iter()
                .map(|(buf, offset, range)| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(buf.raw())
                        .offset(*offset)
                        .range(*range)
                })
                .collect();
            let writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(idx, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(idx as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();
            // SAFETY: every write targets `set` (allocated from `pool`) and
            // references a live DeviceBuffer whose byte range covers the
            // descriptor; there are no copy descriptors.
            unsafe { ctx.device.update_descriptor_sets(&writes, &[]) };
            Ok(Self { ctx, pool, set })
        }

        pub fn raw(&self) -> vk::DescriptorSet {
            self.set
        }
    }

    impl Drop for DescriptorSet<'_> {
        fn drop(&mut self) {
            // SAFETY: exclusive ownership at drop; the GPU work using this set is
            // fence-waited by the recorder before the owning weights are dropped.
            unsafe { self.ctx.device.destroy_descriptor_pool(self.pool, None) };
        }
    }

    /// A persistent descriptor pool + a round-robin ring of pre-allocated
    /// `VkDescriptorSet`s, reused across dispatches. The
    /// per-dispatch `DescriptorSet::storage_buffers*` path creates and destroys a
    /// whole `VkDescriptorPool` every call (~900/token), which dominates the host
    /// GEMV-prep bucket. This mirrors `ggml-vulkan`'s
    /// `ggml_pipeline_allocate_descriptor_sets` (grow-by-50% on exhaustion) +
    /// round-robin `descriptor_set_idx` (ggml-vulkan.cpp:2209-2255, 6303-6332):
    /// the pool and its sets are allocated ONCE for a fixed
    /// `(binding_count, max_descriptors_per_set)` layout; each
    /// [`Self::next_updated`] only runs `vkUpdateDescriptorSets` on the next ring
    /// slot — no object creation, no destruction.
    ///
    /// The ring's `binding_count` is fixed at construction (the layout binds that
    /// many storage buffers). A decode token's dispatches span several distinct
    /// binding counts, so the consumer builds one ring per binding count it uses
    /// (`infer-vulkan`'s `forward.rs` builds six: binding counts 2..=7), each
    /// sized to hold a whole token's worth of live dispatches at that binding
    /// count. Call [`Self::reset`] at the start of each token to rewind the
    /// round-robin index.
    pub struct DescriptorSetRing<'a> {
        ctx: &'a VulkanContext,
        pool: vk::DescriptorPool,
        sets: Vec<vk::DescriptorSet>,
        binding_count: u32,
        next: usize,
    }

    impl<'a> DescriptorSetRing<'a> {
        pub fn new(
            ctx: &'a VulkanContext,
            layout: &DescriptorSetLayout<'_>,
            binding_count: usize,
            ring_size: usize,
        ) -> Result<Self> {
            let binding_count = u32::try_from(binding_count)
                .map_err(|e| runtime_error("converting descriptor binding count", e))?;
            if binding_count == 0 {
                return Err(VulkanError::Runtime(
                    "descriptor ring needs at least one storage-buffer binding".to_string(),
                ));
            }
            let ring_size = ring_size.max(1);
            let ring_size_u32 = u32::try_from(ring_size)
                .map_err(|e| runtime_error("converting descriptor ring size", e))?;
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(binding_count * ring_size_u32)];
            let pool_create = vk::DescriptorPoolCreateInfo::default()
                .max_sets(ring_size_u32)
                .pool_sizes(&pool_sizes);
            // SAFETY: live device; max_sets and the pool descriptor count match
            // `ring_size` sets of this `binding_count`-binding layout.
            let pool = unsafe { ctx.device.create_descriptor_pool(&pool_create, None) }
                .map_err(|e| vk_error("creating Vulkan descriptor pool", e))?;
            let layouts: Vec<_> = std::iter::repeat_n(layout.raw(), ring_size).collect();
            let alloc = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            // SAFETY: the pool was sized for exactly `ring_size` sets of `layout`,
            // and `layouts` repeats that layout `ring_size` times.
            let sets = match unsafe { ctx.device.allocate_descriptor_sets(&alloc) } {
                Ok(sets) => sets,
                Err(e) => {
                    // SAFETY: failed allocation leaves the pool with no sets.
                    unsafe { ctx.device.destroy_descriptor_pool(pool, None) };
                    return Err(vk_error("allocating Vulkan descriptor sets", e));
                }
            };
            Ok(Self {
                ctx,
                pool,
                sets,
                binding_count,
                next: 0,
            })
        }

        /// Rewind the round-robin cursor. Call once per token so a token's
        /// dispatches reuse the ring from slot 0 (the prior token's single
        /// per-token submit has completed — the decode `CommandRecorder`
        /// fence-waits the whole batch before the next token records).
        pub fn reset(&mut self) {
            self.next = 0;
        }

        /// Bind `buffers` (each `(buffer, offset_bytes, range_bytes)`) into the
        /// next ring slot via one `vkUpdateDescriptorSets` and return its raw
        /// `VkDescriptorSet`. No pool / set creation. The caller must record the
        /// dispatch that uses this set BEFORE the slot is reused (a ring of size N
        /// allows N live dispatches between `reset`s). The decode path batches a
        /// whole token into ONE submit, so the ring must hold every live dispatch
        /// of that token at this binding count; the caller (`forward.rs`) sizes
        /// each ring accordingly (≥64), not 4.
        pub fn next_updated(
            &mut self,
            buffers: &[(&DeviceBuffer<'_>, u64, u64)],
        ) -> Result<vk::DescriptorSet> {
            let count = u32::try_from(buffers.len())
                .map_err(|e| runtime_error("converting descriptor buffer count", e))?;
            if count != self.binding_count {
                return Err(VulkanError::Runtime(format!(
                    "descriptor ring bound with {count} buffers but layout has {} bindings",
                    self.binding_count
                )));
            }
            let set = self.sets[self.next % self.sets.len()];
            self.next += 1;
            let infos: Vec<_> = buffers
                .iter()
                .map(|(buf, offset, range)| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(buf.raw())
                        .offset(*offset)
                        .range(*range)
                })
                .collect();
            let writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(idx, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(idx as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();
            // SAFETY: the target is one of this ring's own sets, and each write
            // references a live buffer spanning its declared offset/range; sets
            // are reused only after the batch using them has been waited on.
            unsafe { self.ctx.device.update_descriptor_sets(&writes, &[]) };
            Ok(set)
        }
    }

    impl Drop for DescriptorSetRing<'_> {
        fn drop(&mut self) {
            // SAFETY: exclusive ownership at drop; all sets are freed with the
            // pool, and the in-flight token using them is fence-waited first.
            unsafe { self.ctx.device.destroy_descriptor_pool(self.pool, None) };
        }
    }

    pub struct ComputePipeline<'a> {
        ctx: &'a VulkanContext,
        layout: vk::PipelineLayout,
        pipeline: vk::Pipeline,
    }

    impl<'a> ComputePipeline<'a> {
        pub fn create(
            ctx: &'a VulkanContext,
            shader: &ShaderModule<'_>,
            descriptor_layouts: &[&DescriptorSetLayout<'_>],
        ) -> Result<Self> {
            Self::create_with_push_constants_and_specialization(
                ctx,
                shader,
                descriptor_layouts,
                0,
                &[],
            )
        }

        pub fn create_with_push_constants(
            ctx: &'a VulkanContext,
            shader: &ShaderModule<'_>,
            descriptor_layouts: &[&DescriptorSetLayout<'_>],
            push_constant_bytes: u32,
        ) -> Result<Self> {
            Self::create_with_push_constants_and_specialization(
                ctx,
                shader,
                descriptor_layouts,
                push_constant_bytes,
                &[],
            )
        }

        pub fn create_with_push_constants_and_specialization(
            ctx: &'a VulkanContext,
            shader: &ShaderModule<'_>,
            descriptor_layouts: &[&DescriptorSetLayout<'_>],
            push_constant_bytes: u32,
            specialization_u32: &[(u32, u32)],
        ) -> Result<Self> {
            Self::create_with_push_constants_specialization_and_subgroup_size(
                ctx,
                shader,
                descriptor_layouts,
                push_constant_bytes,
                specialization_u32,
                None,
            )
        }

        /// Like [`Self::create_with_push_constants_and_specialization`] but
        /// optionally pins the compute stage's subgroup size via
        /// `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`. The flash-attn
        /// shader hardcodes `SubGroupSize=32` and reduces with subgroup shuffles,
        /// so on a wave64 device (the 8060S) its pipeline MUST be created with
        /// `required_subgroup_size = Some(32)` or every reduction scrambles.
        /// Requires the `subgroupSizeControl` feature (enabled in
        /// [`VulkanContext::create`]).
        pub fn create_with_push_constants_specialization_and_subgroup_size(
            ctx: &'a VulkanContext,
            shader: &ShaderModule<'_>,
            descriptor_layouts: &[&DescriptorSetLayout<'_>],
            push_constant_bytes: u32,
            specialization_u32: &[(u32, u32)],
            required_subgroup_size: Option<u32>,
        ) -> Result<Self> {
            let set_layouts: Vec<_> = descriptor_layouts
                .iter()
                .map(|layout| layout.raw())
                .collect();
            let push_ranges = if push_constant_bytes == 0 {
                Vec::new()
            } else {
                vec![
                    vk::PushConstantRange::default()
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .offset(0)
                        .size(push_constant_bytes),
                ]
            };
            let layout_create = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push_ranges);
            // SAFETY: live device; both slices are stack-local, and the push range
            // size is what the shaders declare via `push_constant_bytes`.
            let layout = unsafe { ctx.device.create_pipeline_layout(&layout_create, None) }
                .map_err(|e| vk_error("creating Vulkan pipeline layout", e))?;
            let entry =
                CString::new("main").map_err(|e| runtime_error("building shader entry", e))?;

            let mut specialization_data = Vec::with_capacity(specialization_u32.len() * 4);
            let mut specialization_entries = Vec::with_capacity(specialization_u32.len());
            for (idx, (constant_id, value)) in specialization_u32.iter().copied().enumerate() {
                let offset = u32::try_from(idx * std::mem::size_of::<u32>())
                    .map_err(|e| runtime_error("converting specialization offset", e))?;
                specialization_entries.push(vk::SpecializationMapEntry {
                    constant_id,
                    offset,
                    size: std::mem::size_of::<u32>(),
                });
                specialization_data.extend_from_slice(&value.to_ne_bytes());
            }
            let specialization_info = vk::SpecializationInfo::default()
                .map_entries(&specialization_entries)
                .data(&specialization_data);

            let mut stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader.raw())
                .name(&entry);
            if !specialization_u32.is_empty() {
                stage = stage.specialization_info(&specialization_info);
            }
            // Pin the subgroup size (e.g. 32 for flash-attn on a wave64 device).
            // REQUIRE_FULL_SUBGROUPS guarantees the workgroup is a multiple of the
            // pinned size so `num_subgroups = WorkGroupSize/SubGroupSize` holds.
            let mut required_size_info =
                vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default();
            if let Some(size) = required_subgroup_size {
                required_size_info.required_subgroup_size = size;
                stage = stage
                    .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
                    .push_next(&mut required_size_info);
            }
            let create = [vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)];
            // SAFETY: the shader module and `layout` are both live (owned by the
            // caller/`ctx`), the pipeline cache is the context's, and the create
            // structs/specialization bytes are stack-local.
            let pipeline_result = unsafe {
                ctx.device
                    .create_compute_pipelines(ctx.pipeline_cache, &create, None)
            };
            let pipeline = match pipeline_result {
                Ok(mut pipelines) => match pipelines.pop() {
                    Some(pipeline) => pipeline,
                    None => {
                        // SAFETY: empty result; only the successfully created
                        // layout needs destroying.
                        unsafe { ctx.device.destroy_pipeline_layout(layout, None) };
                        return Err(VulkanError::Runtime(
                            "Vulkan compute pipeline creation returned no pipelines".to_string(),
                        ));
                    }
                },
                Err((pipelines, e)) => {
                    for pipeline in pipelines {
                        // SAFETY: these are pipeline handles the failed create
                        // reports as successfully built; none is in use.
                        unsafe { ctx.device.destroy_pipeline(pipeline, None) };
                    }
                    // SAFETY: layout has no pipeline referencing it after the
                    // created ones are destroyed above.
                    unsafe { ctx.device.destroy_pipeline_layout(layout, None) };
                    return Err(vk_error("creating Vulkan compute pipeline", e));
                }
            };
            Ok(Self {
                ctx,
                layout,
                pipeline,
            })
        }

        pub fn layout(&self) -> vk::PipelineLayout {
            self.layout
        }

        pub fn raw(&self) -> vk::Pipeline {
            self.pipeline
        }
    }

    impl Drop for ComputePipeline<'_> {
        fn drop(&mut self) {
            // SAFETY: exclusive ownership at drop; every submit is fence-waited
            // before the owning weights drop, so the pipeline is not executing.
            // Destroy the pipeline before its layout.
            unsafe {
                self.ctx.device.destroy_pipeline(self.pipeline, None);
                self.ctx.device.destroy_pipeline_layout(self.layout, None);
            }
        }
    }
}

#[cfg(feature = "vulkan")]
pub use real::{
    CommandPool, CommandRecorder, ComputePipeline, DescriptorSet, DescriptorSetLayout,
    DescriptorSetRing, DeviceBuffer, ShaderModule, VulkanContext, device_count, device_name, init,
};

#[cfg(not(feature = "vulkan"))]
mod stub {
    use super::{Result, VULKAN_NOT_COMPILED};
    use std::marker::PhantomData;

    pub fn init() -> Result<()> {
        Err(VULKAN_NOT_COMPILED)
    }

    pub fn device_count() -> Result<usize> {
        Err(VULKAN_NOT_COMPILED)
    }

    pub fn device_name(_device_index: usize) -> Result<String> {
        Err(VULKAN_NOT_COMPILED)
    }

    pub struct VulkanContext {
        _private: (),
    }

    impl VulkanContext {
        pub fn create() -> Result<Self> {
            Err(VULKAN_NOT_COMPILED)
        }

        pub fn device_name(&self) -> &str {
            ""
        }

        pub fn queue_family_index(&self) -> u32 {
            0
        }

        pub fn min_storage_buffer_offset_alignment(&self) -> u64 {
            0
        }
    }

    pub struct DeviceBuffer<'a> {
        _marker: PhantomData<&'a VulkanContext>,
    }

    impl<'a> DeviceBuffer<'a> {
        pub fn alloc(_ctx: &'a VulkanContext, _len: usize) -> Result<Self> {
            Err(VULKAN_NOT_COMPILED)
        }

        pub fn alloc_uma(_ctx: &'a VulkanContext, _len: usize) -> Result<Self> {
            Err(VULKAN_NOT_COMPILED)
        }

        pub fn len(&self) -> usize {
            0
        }

        pub fn is_empty(&self) -> bool {
            true
        }

        pub fn copy_from_host(&mut self, _src: &[u8]) -> Result<()> {
            Err(VULKAN_NOT_COMPILED)
        }

        pub fn copy_from_host_at(&mut self, _offset: u64, _src: &[u8]) -> Result<()> {
            Err(VULKAN_NOT_COMPILED)
        }

        pub fn copy_to_host(&self, _dst: &mut [u8]) -> Result<()> {
            Err(VULKAN_NOT_COMPILED)
        }

        pub fn copy_to_host_at(&self, _offset: u64, _dst: &mut [u8]) -> Result<()> {
            Err(VULKAN_NOT_COMPILED)
        }
    }

    pub struct CommandPool<'a> {
        _marker: PhantomData<&'a VulkanContext>,
    }

    impl<'a> CommandPool<'a> {
        pub fn create(_ctx: &'a VulkanContext) -> Result<Self> {
            Err(VULKAN_NOT_COMPILED)
        }
    }

    pub struct CommandRecorder<'a> {
        _marker: PhantomData<&'a VulkanContext>,
    }

    impl<'a> CommandRecorder<'a> {
        pub fn new(_ctx: &'a VulkanContext) -> Result<Self> {
            Err(VULKAN_NOT_COMPILED)
        }

        pub fn begin(&mut self) -> Result<()> {
            Err(VULKAN_NOT_COMPILED)
        }

        pub fn dispatch(
            &mut self,
            _pipeline: &ComputePipeline<'_>,
            _set: &DescriptorSet<'_>,
            _push: &[u8],
            _groups: [u32; 3],
        ) {
        }

        pub fn barrier(&mut self) {}

        pub fn label_next(&mut self, _label: &'static str) {}

        pub fn take_gpu_profile(&mut self) -> Vec<(&'static str, u64, f64)> {
            Vec::new()
        }

        pub fn submit_and_wait(&mut self) -> Result<()> {
            Err(VULKAN_NOT_COMPILED)
        }
    }

    pub struct ShaderModule<'a> {
        _marker: PhantomData<&'a VulkanContext>,
    }

    impl<'a> ShaderModule<'a> {
        pub fn from_spirv_bytes(_ctx: &'a VulkanContext, _bytes: &[u8]) -> Result<Self> {
            Err(VULKAN_NOT_COMPILED)
        }
    }

    pub struct DescriptorSetLayout<'a> {
        _marker: PhantomData<&'a VulkanContext>,
    }

    impl<'a> DescriptorSetLayout<'a> {
        pub fn storage_buffers(_ctx: &'a VulkanContext, _binding_count: usize) -> Result<Self> {
            Err(VULKAN_NOT_COMPILED)
        }
    }

    pub struct DescriptorSet<'a> {
        _marker: PhantomData<&'a VulkanContext>,
    }

    impl<'a> DescriptorSet<'a> {
        pub fn storage_buffers(
            _ctx: &'a VulkanContext,
            _layout: &DescriptorSetLayout<'_>,
            _buffers: &[&DeviceBuffer<'_>],
        ) -> Result<Self> {
            Err(VULKAN_NOT_COMPILED)
        }

        pub fn storage_buffers_ranged(
            _ctx: &'a VulkanContext,
            _layout: &DescriptorSetLayout<'_>,
            _buffers: &[(&DeviceBuffer<'_>, u64, u64)],
        ) -> Result<Self> {
            Err(VULKAN_NOT_COMPILED)
        }
    }

    pub struct DescriptorSetRing<'a> {
        _marker: PhantomData<&'a VulkanContext>,
    }

    impl<'a> DescriptorSetRing<'a> {
        pub fn new(
            _ctx: &'a VulkanContext,
            _layout: &DescriptorSetLayout<'_>,
            _binding_count: usize,
            _ring_size: usize,
        ) -> Result<Self> {
            Err(VULKAN_NOT_COMPILED)
        }

        pub fn reset(&mut self) {}
    }

    pub struct ComputePipeline<'a> {
        _marker: PhantomData<&'a VulkanContext>,
    }

    impl<'a> ComputePipeline<'a> {
        pub fn create(
            _ctx: &'a VulkanContext,
            _shader: &ShaderModule<'_>,
            _descriptor_layouts: &[&DescriptorSetLayout<'_>],
        ) -> Result<Self> {
            Err(VULKAN_NOT_COMPILED)
        }

        pub fn create_with_push_constants(
            _ctx: &'a VulkanContext,
            _shader: &ShaderModule<'_>,
            _descriptor_layouts: &[&DescriptorSetLayout<'_>],
            _push_constant_bytes: u32,
        ) -> Result<Self> {
            Err(VULKAN_NOT_COMPILED)
        }

        pub fn create_with_push_constants_and_specialization(
            _ctx: &'a VulkanContext,
            _shader: &ShaderModule<'_>,
            _descriptor_layouts: &[&DescriptorSetLayout<'_>],
            _push_constant_bytes: u32,
            _specialization_u32: &[(u32, u32)],
        ) -> Result<Self> {
            Err(VULKAN_NOT_COMPILED)
        }

        pub fn create_with_push_constants_specialization_and_subgroup_size(
            _ctx: &'a VulkanContext,
            _shader: &ShaderModule<'_>,
            _descriptor_layouts: &[&DescriptorSetLayout<'_>],
            _push_constant_bytes: u32,
            _specialization_u32: &[(u32, u32)],
            _required_subgroup_size: Option<u32>,
        ) -> Result<Self> {
            Err(VULKAN_NOT_COMPILED)
        }
    }
}

#[cfg(not(feature = "vulkan"))]
pub use stub::{
    CommandPool, CommandRecorder, ComputePipeline, DescriptorSet, DescriptorSetLayout,
    DescriptorSetRing, DeviceBuffer, ShaderModule, VulkanContext, device_count, device_name, init,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "vulkan"))]
    #[test]
    fn stub_reports_not_compiled() {
        assert_eq!(init().unwrap_err(), VULKAN_NOT_COMPILED);
        assert_eq!(device_count().unwrap_err(), VULKAN_NOT_COMPILED);
        assert!(VulkanContext::create().is_err());
        let msg = VULKAN_NOT_COMPILED.to_string();
        assert!(msg.contains("not compiled"), "{msg}");
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn probe_and_roundtrip_or_skip() {
        if let Err(e) = init() {
            eprintln!("vulkan-sys smoke: loader unavailable — skipping ({e})");
            return;
        }
        let n = match device_count() {
            Ok(n) => n,
            Err(e) => {
                eprintln!("vulkan-sys smoke: device enumeration failed — skipping ({e})");
                return;
            }
        };
        if n == 0 {
            eprintln!("vulkan-sys smoke: 0 devices — skipping");
            return;
        }
        let ctx = match VulkanContext::create() {
            Ok(ctx) => ctx,
            Err(VulkanError::NoComputeDevice) => {
                eprintln!("vulkan-sys smoke: no compute queue — skipping");
                return;
            }
            Err(e) => panic!("failed to create Vulkan context: {e}"),
        };
        eprintln!(
            "vulkan-sys smoke: device = {}, queue_family = {}",
            ctx.device_name(),
            ctx.queue_family_index()
        );

        let src: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        let mut buf = match DeviceBuffer::alloc(&ctx, src.len()) {
            Ok(buf) => buf,
            Err(e) => panic!("failed to allocate Vulkan buffer: {e}"),
        };
        if let Err(e) = buf.copy_from_host(&src) {
            panic!("H2D copy failed: {e}");
        }
        let mut back = vec![0u8; src.len()];
        if let Err(e) = buf.copy_to_host(&mut back) {
            panic!("D2H copy failed: {e}");
        }
        assert_eq!(src, back, "H2D/D2H roundtrip mismatch");
    }

    /// Find `glslc` the same way `vulkan-kernels/build.rs` does: explicit
    /// `ARLE_VULKAN_GLSLC`, then `VULKAN_SDK/bin`, then `PATH`. Returns `None`
    /// so the test can skip cleanly on a box without a shader compiler.
    #[cfg(feature = "vulkan")]
    fn find_glslc() -> Option<std::path::PathBuf> {
        use std::path::{Path, PathBuf};
        if let Some(path) = std::env::var_os("ARLE_VULKAN_GLSLC") {
            let path = PathBuf::from(path);
            if path.exists() {
                return Some(path);
            }
        }
        const NAMES: &[&str] = &["glslc", "glslc.exe"];
        if let Some(sdk) = std::env::var_os("VULKAN_SDK") {
            let bin = Path::new(&sdk).join("bin");
            for name in NAMES {
                let path = bin.join(name);
                if path.exists() {
                    return Some(path);
                }
            }
        }
        std::env::var_os("PATH").and_then(|paths| {
            std::env::split_paths(&paths).find_map(|dir| {
                NAMES
                    .iter()
                    .map(|name| dir.join(name))
                    .find(|candidate| candidate.exists())
            })
        })
    }

    /// Compile a trivial "add a push-constant scalar to every element of an
    /// in-place f32 buffer" compute shader to SPIR-V. One binding, one uint of
    /// push (the count) + one int (the addend); a later dispatch reading an
    /// earlier dispatch's writes is exactly the barrier-ordering we must prove.
    #[cfg(feature = "vulkan")]
    fn compile_add_shader(glslc: &std::path::Path) -> Option<Vec<u8>> {
        const SRC: &str = r#"#version 450
layout(local_size_x = 64) in;
layout(push_constant) uniform Params { uint n; int addend; } p;
layout(binding = 0) buffer Buf { int data[]; };
void main() {
    const uint i = gl_GlobalInvocationID.x;
    if (i >= p.n) { return; }
    data[i] = data[i] + p.addend;
}
"#;
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let src_path = dir.join(format!("arle_cmdrec_add_{pid}.comp"));
        let spv_path = dir.join(format!("arle_cmdrec_add_{pid}.spv"));
        std::fs::write(&src_path, SRC).ok()?;
        let output = std::process::Command::new(glslc)
            .arg("-O")
            .arg("--target-env=vulkan1.2")
            .arg("-fshader-stage=compute")
            .arg("-o")
            .arg(&spv_path)
            .arg(&src_path)
            .output()
            .ok()?;
        let _ = std::fs::remove_file(&src_path);
        if !output.status.success() {
            eprintln!(
                "vulkan-sys CommandRecorder test: glslc failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return None;
        }
        let bytes = std::fs::read(&spv_path).ok();
        let _ = std::fs::remove_file(&spv_path);
        bytes
    }

    /// Proves the new single-submit barrier-chained primitive:
    ///   1. Record 3 `add(+k)` dispatches into ONE `CommandRecorder`, each
    ///      separated by `barrier()`, and `submit_and_wait()` exactly ONCE.
    ///      Because each dispatch reads the previous dispatch's writes in place,
    ///      the result is correct only if the barriers serialize the chain.
    ///   2. Reproduce the same chain with 3 sequential `one_shot_submit`s (each
    ///      doing its own submit + `queue_wait_idle`) and assert byte-identical
    ///      results — the new primitive matches the proven drain-per-op path.
    ///
    /// One `submit_and_wait` call == one `vkQueueSubmit` for all 3 dispatches
    /// (structurally guaranteed: `submit_and_wait` issues exactly one
    /// `queue_submit`), versus 3 submits + 3 full `queue_wait_idle` drains in
    /// the reference.
    #[cfg(feature = "vulkan")]
    #[test]
    fn command_recorder_chains_three_dispatches_with_one_submit() {
        if init().is_err() {
            eprintln!("CommandRecorder test: loader unavailable — skipping");
            return;
        }
        match device_count() {
            Ok(0) | Err(_) => {
                eprintln!("CommandRecorder test: no devices — skipping");
                return;
            }
            Ok(_) => {}
        }
        let ctx = match VulkanContext::create() {
            Ok(ctx) => ctx,
            Err(VulkanError::NoComputeDevice) => {
                eprintln!("CommandRecorder test: no compute queue — skipping");
                return;
            }
            Err(e) => panic!("failed to create Vulkan context: {e}"),
        };
        let Some(glslc) = find_glslc() else {
            eprintln!("CommandRecorder test: glslc not found — skipping");
            return;
        };
        let Some(spirv) = compile_add_shader(&glslc) else {
            eprintln!("CommandRecorder test: shader compile failed — skipping");
            return;
        };

        const N: usize = 256;
        let initial: Vec<i32> = (0..N as i32).collect();
        // Three addends applied in order; final value per element = i + 11.
        const ADDENDS: [i32; 3] = [2, 4, 5];
        let expected: Vec<i32> = initial.iter().map(|&v| v + 2 + 4 + 5).collect();

        let bytes_of = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let ints_of = |b: &[u8]| -> Vec<i32> {
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c))
                .collect()
        };

        let shader =
            ShaderModule::from_spirv_bytes(&ctx, &spirv).expect("create add shader module");
        let layout = DescriptorSetLayout::storage_buffers(&ctx, 1).expect("create DSL");
        // push = { uint n; int addend; } = 8 bytes.
        let push_bytes: u32 = 8;
        let pipeline =
            ComputePipeline::create_with_push_constants(&ctx, &shader, &[&layout], push_bytes)
                .expect("create add pipeline");
        let groups = [(N as u32).div_ceil(64), 1, 1];
        let push_for =
            |addend: i32| -> Vec<u8> { [(N as u32).to_le_bytes(), addend.to_le_bytes()].concat() };

        let chained = {
            let mut buf = DeviceBuffer::alloc(&ctx, N * 4).expect("alloc chained buf");
            buf.copy_from_host(&bytes_of(&initial))
                .expect("H2D chained");
            let set = DescriptorSet::storage_buffers(&ctx, &layout, &[&buf]).expect("DS chained");
            let mut rec = CommandRecorder::new(&ctx).expect("CommandRecorder::new");
            rec.begin().expect("recorder begin");
            for (idx, &addend) in ADDENDS.iter().enumerate() {
                rec.dispatch(&pipeline, &set, &push_for(addend), groups);
                if idx + 1 < ADDENDS.len() {
                    rec.barrier();
                }
            }
            rec.submit_and_wait().expect("recorder submit_and_wait");
            let mut back = vec![0u8; N * 4];
            buf.copy_to_host(&mut back).expect("D2H chained");
            ints_of(&back)
        };

        let sequential = {
            let mut buf = DeviceBuffer::alloc(&ctx, N * 4).expect("alloc seq buf");
            buf.copy_from_host(&bytes_of(&initial)).expect("H2D seq");
            let set = DescriptorSet::storage_buffers(&ctx, &layout, &[&buf]).expect("DS seq");
            let pool = CommandPool::create(&ctx).expect("CommandPool::create");
            for &addend in &ADDENDS {
                let push = push_for(addend);
                pool.one_shot_submit(|cmd| {
                    let device = ctx.raw_device();
                    // SAFETY: `cmd` is recording; `pipeline`, its layout, the set
                    // (allocated for that layout), and the push slice are all live
                    // for the closure. This mirrors `CommandRecorder::dispatch_raw`.
                    unsafe {
                        device.cmd_bind_pipeline(
                            cmd,
                            ash::vk::PipelineBindPoint::COMPUTE,
                            pipeline.raw(),
                        );
                        device.cmd_bind_descriptor_sets(
                            cmd,
                            ash::vk::PipelineBindPoint::COMPUTE,
                            pipeline.layout(),
                            0,
                            &[set.raw()],
                            &[],
                        );
                        device.cmd_push_constants(
                            cmd,
                            pipeline.layout(),
                            ash::vk::ShaderStageFlags::COMPUTE,
                            0,
                            &push,
                        );
                        device.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
                    }
                    Ok(())
                })
                .expect("one_shot_submit");
            }
            let mut back = vec![0u8; N * 4];
            buf.copy_to_host(&mut back).expect("D2H seq");
            ints_of(&back)
        };

        assert_eq!(
            chained, expected,
            "barrier-chained result wrong — barriers did not serialize the in-place adds"
        );
        assert_eq!(
            chained, sequential,
            "single-submit chain != 3 sequential one_shot_submits"
        );
        eprintln!(
            "CommandRecorder test: 3 barrier-chained dispatches via ONE submit_and_wait == 3 one_shot_submits ({} elems, +{} each elem)",
            N,
            ADDENDS.iter().sum::<i32>()
        );
    }
}
