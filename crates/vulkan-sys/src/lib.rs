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

/// Round `value` up to the next multiple of `alignment` (a power of two).
fn align_up(value: u64, alignment: u64) -> u64 {
    debug_assert!(
        alignment.is_power_of_two(),
        "alignment must be a power of 2"
    );
    value.div_ceil(alignment) * alignment
}

/// Smallest slab a [`SlabAllocator`] will fall back to under heap pressure.
///
/// The floor exists because shrinking slabs trades one scarce resource for
/// another: at 64 MiB a 71 GiB residency still needs only 71 GiB / 64 MiB =
/// 1136 `vkAllocateMemory` calls, inside the 4096 that `maxMemoryAllocationCount`
/// is *guaranteed* to allow on any conformant device. Halving past this point
/// would start racing that limit instead of the size limit.
pub const MIN_SLAB_BYTES: u64 = 64 << 20;

/// Where a [`SlabAllocator`]'s slabs live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlabMemory {
    /// `DEVICE_LOCAL`, not host-mappable. The right home for resident weights:
    /// written once through staging, then only ever read by the GPU.
    DeviceLocal,
    /// `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`, falling back to plain
    /// host-visible. On a UMA part the big device-local heap is mappable, so
    /// the loader can write weights straight into the slab with no staging
    /// copy. Note the memory is WRITE-COMBINED: host *writes* run at full
    /// speed, host *reads* do not (see [`DeviceBuffer::alloc_host_cached`]).
    Uma,
}

/// One suballocation: which slab, at what byte offset, for how many bytes.
///
/// `offset` already honours the plan's alignment, so it goes straight into
/// [`DescriptorSet::storage_buffers_ranged`] as a descriptor offset — a
/// slab-backed tensor binds as `(slab buffer, offset, len)` with no copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlabAlloc {
    slab: usize,
    offset: u64,
    len: u64,
}

impl SlabAlloc {
    /// Index of the owning slab, for [`SlabAllocator::slab`].
    pub fn slab(&self) -> usize {
        self.slab
    }

    /// Byte offset within the slab buffer; aligned, bindable as-is.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// One past the last byte, within the slab.
    pub fn end(&self) -> u64 {
        self.offset + self.len
    }
}

/// The placement arithmetic behind [`SlabAllocator`], with no device attached.
///
/// [`SlabAllocator`] drives its allocations through this exact type, so a
/// dry-run plan and the real residency cannot drift apart. Alone, it answers
/// "does this checkpoint fit, and in how many allocations?" before a byte of
/// GPU memory is touched.
///
/// The constraint that forces slabs is `maxMemoryAllocationSize`: one
/// `vkAllocateMemory` may not exceed it, and this driver reports 2 GiB
/// (measured), so a 71 GiB model has no single-buffer form. The other Vulkan
/// cap, `maxMemoryAllocationCount`, is often quoted as the reason too — its
/// spec-guaranteed floor is 4096, against 296 475 tensors here — but it is NOT
/// binding on this part: the 8060S driver reports 4294967295 (measured). Treat
/// the count as a portability constraint and the size as the local one.
///
/// Placement is first-fit across every open slab, not a bump pointer into the
/// newest one. That matters: a slab holds five 400 MB PLE shards and no more,
/// so a newest-slab-only bump strands a 147 MB tail (6.9%) in every slab it
/// touches. First-fit gives those tails to the small tensors that dominate the
/// checkpoint by count.
///
/// **Feed it largest-first if you can.** Measured over all 296 475 tensors of
/// the qwen4_exp checkpoint at a 2 GiB slab size (122.75 GiB placeable):
///
/// ```text
///   floor (ceil(bytes/slab))         62 slabs
///   arrival order, first-fit         64 slabs   4.10% waste
///   arrival order, best-fit          64 slabs   4.10% waste
///   largest-first, first-fit         62 slabs   1.01% waste
/// ```
///
/// Best-fit buys nothing over first-fit, which is why the cheaper scan is the
/// one implemented; the ordering is where the two slabs are. A loader that
/// knows all its tensor sizes up front should sort descending before placing.
#[derive(Debug, Clone)]
pub struct SlabPlan {
    slab_size: u64,
    alignment: u64,
    /// `(capacity, bump cursor)` per slab, in allocation order. Capacity is
    /// per-slab rather than global because a real slab can come back smaller
    /// than nominal when the heap is nearly full.
    slabs: Vec<(u64, u64)>,
    used: u64,
}

impl SlabPlan {
    /// `slab_size` is the nominal size of a fresh slab (cap it at the device's
    /// `maxMemoryAllocationSize`); `alignment` must be a power of two and at
    /// least the device's `minStorageBufferOffsetAlignment`.
    pub fn new(slab_size: u64, alignment: u64) -> Result<Self> {
        if slab_size == 0 {
            return Err(VulkanError::Runtime(
                "slab size must be non-zero".to_string(),
            ));
        }
        if !alignment.is_power_of_two() {
            return Err(VulkanError::Runtime(format!(
                "slab alignment {alignment} is not a power of two"
            )));
        }
        if alignment > slab_size {
            return Err(VulkanError::Runtime(format!(
                "slab alignment {alignment} exceeds slab size {slab_size}"
            )));
        }
        Ok(Self {
            slab_size,
            alignment,
            slabs: Vec::new(),
            used: 0,
        })
    }

    pub fn slab_size(&self) -> u64 {
        self.slab_size
    }

    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    pub fn slab_count(&self) -> usize {
        self.slabs.len()
    }

    /// Bytes handed to `vkAllocateMemory`, summed over slabs — what the device
    /// heap actually gives up, including alignment padding and slab tails.
    pub fn committed_bytes(&self) -> u64 {
        self.slabs.iter().map(|(capacity, _)| *capacity).sum()
    }

    /// Bytes handed out to callers, excluding padding and tails.
    pub fn used_bytes(&self) -> u64 {
        self.used
    }

    /// `committed - used`: what the packing costs.
    pub fn wasted_bytes(&self) -> u64 {
        self.committed_bytes().saturating_sub(self.used)
    }

    /// Dry run: place `len` bytes, opening nominal-size slabs as needed. No
    /// device memory is touched, so this can run on a box with no GPU.
    pub fn place(&mut self, len: u64) -> Result<SlabAlloc> {
        if let Some(alloc) = self.find(len)? {
            self.commit(alloc);
            return Ok(alloc);
        }
        let slab = self.push_slab(self.slab_size);
        // `find` already rejected `len > slab_size`, so offset 0 of a fresh
        // nominal-size slab always fits.
        let alloc = SlabAlloc {
            slab,
            offset: 0,
            len,
        };
        self.commit(alloc);
        Ok(alloc)
    }

    /// First open slab with room for `len`, or `None` if a new slab is needed.
    /// Errors on a request no slab could ever hold.
    fn find(&self, len: u64) -> Result<Option<SlabAlloc>> {
        if len == 0 {
            return Err(VulkanError::Runtime(
                "slab suballocation size must be non-zero".to_string(),
            ));
        }
        if len > self.slab_size {
            return Err(VulkanError::Runtime(format!(
                "suballocation of {len} B exceeds the {} B slab size \
                 (maxMemoryAllocationSize); such a tensor must be split across \
                 several bindings",
                self.slab_size
            )));
        }
        Ok(self
            .slabs
            .iter()
            .enumerate()
            .find_map(|(slab, (capacity, cursor))| {
                (cursor + len <= *capacity).then_some(SlabAlloc {
                    slab,
                    offset: *cursor,
                    len,
                })
            }))
    }

    /// Record a slab of `capacity` bytes; returns its index.
    fn push_slab(&mut self, capacity: u64) -> usize {
        self.slabs.push((capacity, 0));
        self.slabs.len() - 1
    }

    /// Consume the space `alloc` occupies, re-aligning the slab's cursor.
    fn commit(&mut self, alloc: SlabAlloc) {
        let (capacity, cursor) = &mut self.slabs[alloc.slab];
        *cursor = align_up(alloc.end(), self.alignment).min(*capacity);
        self.used += alloc.len;
    }
}

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
        unsafe { entry.create_instance(&create, None) }
            .map_err(|e| vk_error("creating Vulkan instance", e))
    }

    fn has_device_extension(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        name: &CStr,
    ) -> Result<bool> {
        let extensions = unsafe { instance.enumerate_device_extension_properties(physical_device) }
            .map_err(|e| vk_error("enumerating Vulkan device extensions", e))?;
        Ok(extensions.iter().any(|extension| {
            let extension_name = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) };
            extension_name == name
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
        unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };

        vk_bool(features2.features.shader_int16)
            && vk_bool(storage16.storage_buffer16_bit_access)
            && vk_bool(vulkan12.storage_buffer8_bit_access)
            && vk_bool(vulkan12.shader_float16)
            && vk_bool(vulkan12.shader_int8)
            && vk_bool(integer_dot.shader_integer_dot_product)
    }

    fn load_entry() -> Result<Entry> {
        unsafe { Entry::load() }.map_err(|e| runtime_error("loading Vulkan loader", e))
    }

    /// The `f16 x f16 -> f32` cooperative-matrix tile the device advertises.
    ///
    /// `VK_KHR_cooperative_matrix` exposes a *set* of supported shapes; only the
    /// ones with `AType == BType == float16`, `scope == Subgroup` and an f32
    /// accumulator are usable by the tiled prefill GEMM (matching what
    /// `ggml-vulkan` selects). On RDNA3/3.5 that is 16x16x16.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CoopmatShape {
        pub m: u32,
        pub n: u32,
        pub k: u32,
    }

    /// Pick the usable `f16 x f16 -> f32` subgroup-scoped tile, or `None` when
    /// the device has no matrix cores (or exposes only f16-accumulate shapes,
    /// which we reject for the same accuracy reason `ggml-vulkan` does).
    ///
    /// Called before `vkCreateDevice`: the query is a *physical device*
    /// function, so the loader resolves it through `vkGetInstanceProcAddr`
    /// without the extension being enabled anywhere yet.
    fn query_coopmat(
        entry: &Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Option<CoopmatShape> {
        if !has_device_extension(instance, physical_device, vk::KHR_COOPERATIVE_MATRIX_NAME)
            .unwrap_or(false)
        {
            return None;
        }
        let mut coopmat = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut coopmat);
        unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };
        if !vk_bool(coopmat.cooperative_matrix) {
            return None;
        }
        let ext = ash::khr::cooperative_matrix::Instance::new(entry, instance);
        let props =
            unsafe { ext.get_physical_device_cooperative_matrix_properties(physical_device) }
                .ok()?;
        props
            .iter()
            .find(|p| {
                p.a_type == vk::ComponentTypeKHR::FLOAT16
                    && p.b_type == vk::ComponentTypeKHR::FLOAT16
                    && p.c_type == vk::ComponentTypeKHR::FLOAT32
                    && p.result_type == vk::ComponentTypeKHR::FLOAT32
                    && p.scope == vk::ScopeKHR::SUBGROUP
            })
            .map(|p| CoopmatShape {
                m: p.m_size,
                n: p.n_size,
                k: p.k_size,
            })
    }

    fn pick_compute_queue(
        instance: &ash::Instance,
    ) -> Result<(vk::PhysicalDevice, u32, vk::PhysicalDeviceProperties)> {
        let devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| vk_error("enumerating Vulkan physical devices", e))?;
        for physical_device in devices {
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
            let families =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
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
        let raw = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
        raw.to_string_lossy().into_owned()
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
        /// `Some` when `VK_KHR_cooperative_matrix` was found *and* enabled on
        /// the device, carrying the f16xf16->f32 tile the prefill GEMM should
        /// compile for. `None` means no matrix cores: `mul_mmq` stays the route.
        coopmat: Option<CoopmatShape>,
    }

    impl VulkanContext {
        pub fn create() -> Result<Self> {
            let entry = load_entry()?;
            let instance = create_instance(&entry)?;
            let picked = match pick_compute_queue(&instance) {
                Ok(picked) => picked,
                Err(e) => {
                    unsafe { instance.destroy_instance(None) };
                    return Err(e);
                }
            };
            let (physical_device, queue_family_index, props) = picked;
            let priorities = [1.0f32];
            let queue_info = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities)];
            let coopmat = query_coopmat(&entry, &instance, physical_device);
            let mut extensions = vec![
                vk::KHR_SHADER_INTEGER_DOT_PRODUCT_NAME.as_ptr(),
                vk::EXT_SUBGROUP_SIZE_CONTROL_NAME.as_ptr(),
            ];
            if coopmat.is_some() {
                extensions.push(vk::KHR_COOPERATIVE_MATRIX_NAME.as_ptr());
            }
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
            let mut coopmat_features =
                vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .features(base_features)
                .push_next(&mut integer_dot)
                .push_next(&mut vulkan12)
                .push_next(&mut storage16)
                .push_next(&mut size_control);
            if coopmat.is_some() {
                features2 = features2.push_next(&mut coopmat_features);
            }
            let create = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&extensions)
                .push_next(&mut features2);
            let device = match unsafe { instance.create_device(physical_device, &create, None) } {
                Ok(device) => device,
                Err(e) => {
                    unsafe { instance.destroy_instance(None) };
                    return Err(vk_error("creating Vulkan device", e));
                }
            };
            let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
            let pipeline_cache_create = vk::PipelineCacheCreateInfo::default();
            let pipeline_cache =
                match unsafe { device.create_pipeline_cache(&pipeline_cache_create, None) } {
                    Ok(cache) => cache,
                    Err(e) => {
                        unsafe {
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
                coopmat,
            })
        }

        /// The cooperative-matrix tile enabled on this device, or `None` when
        /// the device has no usable matrix cores. Callers must treat `None` as
        /// "compile the `mul_mmq` fallback", never as an error.
        pub fn coopmat(&self) -> Option<CoopmatShape> {
            self.coopmat
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
            let props = unsafe {
                self.instance
                    .get_physical_device_properties(self.physical_device)
            };
            props.limits.min_storage_buffer_offset_alignment
        }

        /// `maxComputeSharedMemorySize` (bytes) — the ceiling on a compute
        /// pipeline's `shared` declarations. The tiled `mul_mmq` prefill GEMM
        /// sizes its shared A/B caches from spec constants, so the tile must be
        /// chosen against this limit (see `MmqSpec::choose`); an oversized tile
        /// fails at pipeline creation, not at dispatch.
        pub fn max_compute_shared_memory_size(&self) -> u32 {
            let props = unsafe {
                self.instance
                    .get_physical_device_properties(self.physical_device)
            };
            props.limits.max_compute_shared_memory_size
        }

        /// `maxMemoryAllocationSize` (bytes) — the hard ceiling on ONE
        /// `vkAllocateMemory`, from `VkPhysicalDeviceMaintenance3Properties`
        /// (core since Vulkan 1.1, so always present at our required 1.2).
        /// This driver reports 2 GiB, which is why a 71 GiB model cannot be one
        /// buffer and has to be sliced into slabs — see [`SlabAllocator`].
        ///
        /// Returns the Vulkan required-limits minimum (2^30) if the driver
        /// leaves the struct zeroed, so callers never divide by zero.
        pub fn max_memory_allocation_size(&self) -> u64 {
            let mut maintenance3 = vk::PhysicalDeviceMaintenance3Properties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut maintenance3);
            // SAFETY: `physical_device` is owned by this context and alive;
            // `props2` and the `maintenance3` it chains both outlive the call
            // and are only read afterwards.
            unsafe {
                self.instance
                    .get_physical_device_properties2(self.physical_device, &mut props2);
            }
            if maintenance3.max_memory_allocation_size == 0 {
                1 << 30
            } else {
                maintenance3.max_memory_allocation_size
            }
        }

        /// `maxMemoryAllocationCount` — the ceiling on how many
        /// `vkAllocateMemory` results may be live at once. The spec's required
        /// minimum is 4096, which a one-`DeviceBuffer`-per-tensor loader would
        /// blow past 72x on the qwen4_exp checkpoint's 296 475 tensors — but do
        /// not assume that floor is the local value. Measured on the 8060S:
        /// 4294967295, i.e. unlimited in practice. Query it, then decide.
        pub fn max_memory_allocation_count(&self) -> u32 {
            // SAFETY: `physical_device` is owned by this context and alive; the
            // query only fills a `VkPhysicalDeviceProperties` by value.
            let props = unsafe {
                self.instance
                    .get_physical_device_properties(self.physical_device)
            };
            props.limits.max_memory_allocation_count
        }

        /// `maxStorageBufferRange` (bytes) — the largest range one storage
        /// buffer descriptor may cover. A slab suballocation is bound as a
        /// range, so a tensor bigger than this cannot be one binding even when
        /// it fits in a slab.
        pub fn max_storage_buffer_range(&self) -> u32 {
            // SAFETY: `physical_device` is owned by this context and alive; the
            // query only fills a `VkPhysicalDeviceProperties` by value.
            let props = unsafe {
                self.instance
                    .get_physical_device_properties(self.physical_device)
            };
            props.limits.max_storage_buffer_range
        }

        /// `(timestampPeriod ns/tick, timestampValidBits)` for GPU timestamp
        /// profiling. `valid_bits == 0` means the compute queue does not support
        /// timestamps (profiling must be disabled).
        pub fn timestamp_info(&self) -> (f32, u32) {
            let props = unsafe {
                self.instance
                    .get_physical_device_properties(self.physical_device)
            };
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
            unsafe {
                self.instance
                    .get_physical_device_properties2(self.physical_device, &mut props2);
            }
            let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
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

        /// AMD shader-core topology, or `None` off an AMD driver.
        ///
        /// Returns `(compute_units, simd_per_cu, wavefronts_per_simd,
        /// wavefront_size, vgprs_per_simd)`.
        ///
        /// Why this is worth a query: on RDNA the scheduling unit is the WGP
        /// (Work Group Processor) = 2 CUs = 4 SIMD32, and a workgroup is
        /// resident on ONE of them. Reasoning in "40 CUs" hides the fact that
        /// the part is really 20 WGPs / 80 SIMDs, so a kernel launching e.g. 48
        /// workgroups is not "48 of 40" but ~2.4 per WGP — and the wave-per-SIMD
        /// occupancy that actually hides memory latency is a third number
        /// again. Nothing in ARLE printed any of this, which is how a launch
        /// geometry gets chosen against an imagined machine.
        ///
        /// `vgprs_per_simd` is the other half of the same story: it is the
        /// budget that decides how many waves fit, and it is what made
        /// llama.cpp's coopmat warptile run at 0.57x here (32 live accumulators
        /// ~= 128 VGPR/lane). See [`vulkan_kernels::MmSpec`].
        pub fn amd_shader_core(&self) -> Option<(u32, u32, u32, u32, u32)> {
            if !self.device_name().contains("AMD") && !self.device_name().contains("Radeon") {
                return None;
            }
            let mut core = vk::PhysicalDeviceShaderCorePropertiesAMD::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut core);
            // SAFETY: `physical_device` is owned by this context and alive;
            // `props2` (with its pushed `core`) outlives the call and is only
            // read afterwards. Querying an unsupported extension struct leaves
            // it zeroed rather than failing, which the check below catches.
            unsafe {
                self.instance
                    .get_physical_device_properties2(self.physical_device, &mut props2);
            }
            (core.compute_units_per_shader_array != 0).then(|| {
                (
                    core.compute_units_per_shader_array
                        * core.shader_arrays_per_engine_count
                        * core.shader_engine_count,
                    core.simd_per_compute_unit,
                    core.wavefronts_per_simd,
                    core.wavefront_size,
                    core.vgprs_per_simd,
                )
            })
        }

        pub fn memory_heaps(&self) -> Vec<(u64, bool)> {
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
        let devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| vk_error("enumerating Vulkan physical devices", e));
        unsafe { instance.destroy_instance(None) };
        devices.map(|d| d.len())
    }

    pub fn device_name(device_index: usize) -> Result<String> {
        let entry = load_entry()?;
        let instance = create_instance(&entry)?;
        let devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| vk_error("enumerating Vulkan physical devices", e))?;
        let name = match devices.get(device_index) {
            Some(device) => {
                let props = unsafe { instance.get_physical_device_properties(*device) };
                Ok(device_name_from_properties(&props))
            }
            None => Err(VulkanError::Runtime(format!(
                "Vulkan device index {device_index} out of range ({} devices)",
                devices.len()
            ))),
        };
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

        /// Allocate a READ-BACK buffer: `HOST_VISIBLE | HOST_COHERENT |
        /// HOST_CACHED`, falling back to plain host-visible if the device has
        /// no cached type.
        ///
        /// The distinction is not cosmetic. `alloc`/`alloc_uma` memory on this
        /// APU is write-combined: the host can WRITE it at full speed but reads
        /// are uncached, and every read is a partial-line fetch. Measured on the
        /// 8060S reading one token's logits (248320 f32 = 970 KB):
        ///
        /// ```text
        ///   alloc_uma  9.80 ms   0.10 GB/s     <- 811x slower than memcpy
        ///   alloc     10.03 ms   0.10 GB/s
        ///   memcpy     0.01 ms  82.19 GB/s
        /// ```
        ///
        /// A per-token readback out of write-combined memory is therefore ~10 ms
        /// of pure host stall that no GPU profile can see, because no dispatch
        /// is involved. Anything the host READS every step belongs here; use
        /// [`Self::alloc_uma`] for buffers the host only writes.
        pub fn alloc_host_cached(ctx: &'a VulkanContext, len: usize) -> Result<Self> {
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST;
            Self::alloc_with_usage(
                ctx,
                len,
                usage,
                vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::HOST_CACHED,
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
            let buffer = unsafe { ctx.device.create_buffer(&create, None) }
                .map_err(|e| vk_error("creating Vulkan buffer", e))?;
            let req = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
            let memory_type_index = match ctx.memory_type_index(req.memory_type_bits, memory_flags)
            {
                Ok(idx) => idx,
                Err(e) => {
                    unsafe { ctx.device.destroy_buffer(buffer, None) };
                    return Err(e);
                }
            };
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(memory_type_index);
            let memory = match unsafe { ctx.device.allocate_memory(&alloc, None) } {
                Ok(memory) => memory,
                Err(e) => {
                    unsafe { ctx.device.destroy_buffer(buffer, None) };
                    return Err(vk_error("allocating Vulkan buffer memory", e));
                }
            };
            if let Err(e) = unsafe { ctx.device.bind_buffer_memory(buffer, memory, 0) } {
                unsafe {
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
            let ptr = unsafe {
                self.ctx.device.map_memory(
                    self.memory,
                    offset,
                    src.len() as vk::DeviceSize,
                    vk::MemoryMapFlags::empty(),
                )
            }
            .map_err(|e| vk_error("mapping Vulkan buffer for H2D", e))?;
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), ptr.cast::<u8>(), src.len());
                self.ctx.device.unmap_memory(self.memory);
            }
            Ok(())
        }

        pub fn copy_to_host_at(&self, offset: u64, dst: &mut [u8]) -> Result<()> {
            assert!(
                offset as usize + dst.len() <= self.len,
                "host slice + offset exceeds Vulkan buffer"
            );
            if dst.is_empty() {
                return Ok(());
            }
            let ptr = unsafe {
                self.ctx.device.map_memory(
                    self.memory,
                    offset,
                    dst.len() as vk::DeviceSize,
                    vk::MemoryMapFlags::empty(),
                )
            }
            .map_err(|e| vk_error("mapping Vulkan buffer for D2H", e))?;
            unsafe {
                std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), dst.as_mut_ptr(), dst.len());
                self.ctx.device.unmap_memory(self.memory);
            }
            Ok(())
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
                unsafe {
                    ctx.device
                        .cmd_copy_buffer(cmd, staging.buffer, dst.buffer, &[region]);
                }
                Ok(())
            })?;
            Ok(dst)
        }

        /// Read this buffer back through a temporary host-visible staging
        /// buffer — the inverse of [`Self::alloc_device_local_from_host`].
        ///
        /// [`Self::copy_to_host`] maps the buffer's own memory, so it fails with
        /// `ERROR_MEMORY_MAP_FAILED` on anything allocated DEVICE_LOCAL-only,
        /// which is every resident weight. This is the read-back path for those.
        ///
        /// Verification-only: it allocates, submits, and blocks. Never put it on
        /// a per-token path — see the write-combined read-back trap that made the
        /// MoE router 20x slower than it looked.
        pub fn copy_to_host_staged(&self, dst: &mut [u8]) -> Result<()> {
            if dst.is_empty() {
                return Ok(());
            }
            if dst.len() > self.len {
                return Err(VulkanError::Runtime(format!(
                    "staged D2H of {} B from a {} B buffer",
                    dst.len(),
                    self.len
                )));
            }
            let staging = Self::alloc_with_usage(
                self.ctx,
                dst.len(),
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let pool = CommandPool::create(self.ctx)?;
            pool.one_shot_submit(|cmd| {
                let region = vk::BufferCopy::default().size(dst.len() as vk::DeviceSize);
                // SAFETY: `cmd` is the live, recording command buffer supplied by
                // `one_shot_submit`. Both buffers outlive the submit (`self` by
                // `&self`, `staging` by this scope) and carry the required usage
                // flags — `self` TRANSFER_SRC from `alloc_device_local_from_host`,
                // `staging` TRANSFER_DST above. `region` copies `dst.len()` bytes,
                // checked against `self.len` and equal to `staging`'s size.
                unsafe {
                    self.ctx
                        .device
                        .cmd_copy_buffer(cmd, self.buffer, staging.buffer, &[region]);
                }
                Ok(())
            })?;
            staging.copy_to_host(dst)
        }
    }

    impl Drop for DeviceBuffer<'_> {
        fn drop(&mut self) {
            unsafe {
                self.ctx.device.destroy_buffer(self.buffer, None);
                self.ctx.device.free_memory(self.memory, None);
            }
        }
    }

    /// Staging window for [`SlabAllocator::write`] into DEVICE_LOCAL slabs.
    ///
    /// Sized to bound host memory while keeping the submit count negligible:
    /// at 32 MiB a 71 GiB upload is ~2270 `vkQueueSubmit`s total, versus one
    /// per tensor (296 475) for a naive per-tensor staging buffer. The buffer
    /// is allocated once per allocator and reused for every write.
    const STAGING_CHUNK_BYTES: usize = 32 << 20;

    /// Suballocates a large model out of a handful of big device allocations.
    ///
    /// The hard limit it exists for is
    /// [`VulkanContext::max_memory_allocation_size`]: one `vkAllocateMemory`
    /// may not exceed it, and this driver reports 2 GiB (measured), so a ~71
    /// GiB model has no single-buffer form no matter how the loader is written.
    ///
    /// The other half — one `DeviceBuffer` per tensor — is a cost argument, not
    /// a limit argument, and the distinction matters because it is easy to get
    /// backwards. The qwen4_exp checkpoint has 296 475 tensors, which would be
    /// 296 475 `vkAllocateMemory` + `vkBindBufferMemory` pairs and 296 475 live
    /// allocations in the WDDM residency list. That is 72x the 4096 floor
    /// [`VulkanContext::max_memory_allocation_count`] is *guaranteed* to allow,
    /// so the per-tensor path is not portable — but on this part it would not
    /// actually fail that check: the 8060S reports 4294967295 (measured).
    ///
    /// So: allocate slabs of `maxMemoryAllocationSize`, and hand out
    /// `(slab, offset, len)` triples aligned to
    /// [`VulkanContext::min_storage_buffer_offset_alignment`]. Each
    /// suballocation binds through the existing ranged descriptor path with no
    /// copy and no extra object:
    ///
    /// ```ignore
    /// let w = slabs.alloc(bytes)?;
    /// let (buf, offset, len) = slabs.binding(&w)?;
    /// DescriptorSet::storage_buffers_ranged(ctx, &layout, &[(buf, offset, len)])?;
    /// ```
    ///
    /// Placement lives in [`SlabPlan`] so a dry run and the real residency
    /// cannot diverge. Slabs are freed when the allocator drops; there is no
    /// per-suballocation free, which suits a load-once weight set.
    pub struct SlabAllocator<'a> {
        ctx: &'a VulkanContext,
        slabs: Vec<DeviceBuffer<'a>>,
        plan: super::SlabPlan,
        memory: super::SlabMemory,
        /// Reused host-visible upload window; `None` until the first staged
        /// write, and never allocated at all for [`SlabMemory::Uma`].
        staging: Option<DeviceBuffer<'a>>,
        max_binding_range: u64,
    }

    impl<'a> SlabAllocator<'a> {
        /// DEVICE_LOCAL slabs of the device's `maxMemoryAllocationSize`.
        pub fn new(ctx: &'a VulkanContext) -> Result<Self> {
            Self::with_slab_size(ctx, ctx.max_memory_allocation_size())
        }

        /// As [`Self::new`], but with an explicit nominal slab size (clamped to
        /// the device's `maxMemoryAllocationSize`). Useful for tests and for
        /// leaving headroom on a shared heap.
        pub fn with_slab_size(ctx: &'a VulkanContext, slab_size: u64) -> Result<Self> {
            Self::with_slab_size_and_memory(ctx, slab_size, super::SlabMemory::DeviceLocal)
        }

        pub fn with_slab_size_and_memory(
            ctx: &'a VulkanContext,
            slab_size: u64,
            memory: super::SlabMemory,
        ) -> Result<Self> {
            let slab_size = slab_size.min(ctx.max_memory_allocation_size());
            // A descriptor offset must satisfy `minStorageBufferOffsetAlignment`;
            // the 16-byte floor on top of it is for the shaders that read these
            // bindings as `uvec4`, and matches the NVFP4 group of 16 so a
            // quantized tensor never starts mid-group.
            let alignment = ctx.min_storage_buffer_offset_alignment().max(16);
            Ok(Self {
                ctx,
                slabs: Vec::new(),
                plan: super::SlabPlan::new(slab_size, alignment)?,
                memory,
                staging: None,
                max_binding_range: u64::from(ctx.max_storage_buffer_range()),
            })
        }

        /// Reserve `len` bytes, opening a slab if none has room.
        pub fn alloc(&mut self, len: u64) -> Result<super::SlabAlloc> {
            if len > self.max_binding_range {
                return Err(VulkanError::Runtime(format!(
                    "suballocation of {len} B exceeds maxStorageBufferRange ({} B); \
                     it cannot be bound as one storage buffer descriptor",
                    self.max_binding_range
                )));
            }
            if let Some(alloc) = self.plan.find(len)? {
                self.plan.commit(alloc);
                return Ok(alloc);
            }
            let (buffer, capacity) = self.alloc_slab(len)?;
            self.slabs.push(buffer);
            self.plan.push_slab(capacity);
            let alloc = self.plan.find(len)?.ok_or_else(|| {
                VulkanError::Runtime(format!(
                    "a fresh {capacity} B slab did not fit a {len} B suballocation"
                ))
            })?;
            self.plan.commit(alloc);
            Ok(alloc)
        }

        /// Allocate one slab, at least `needed` bytes.
        ///
        /// Halves on failure down to [`MIN_SLAB_BYTES`]: near the end of a 71
        /// GiB residency the heap can have several GiB free yet not one
        /// contiguous nominal slab, and failing there with free memory left
        /// would be a worse outcome than a smaller final slab.
        fn alloc_slab(&self, needed: u64) -> Result<(DeviceBuffer<'a>, u64)> {
            let mut size = self.plan.slab_size().max(needed);
            loop {
                let bytes = usize::try_from(size)
                    .map_err(|e| runtime_error("converting slab size to usize", e))?;
                match self.try_alloc_slab(bytes) {
                    Ok(buffer) => return Ok((buffer, size)),
                    Err(e) => {
                        let halved = size / 2;
                        if halved < needed || halved < super::MIN_SLAB_BYTES {
                            return Err(e);
                        }
                        size = halved;
                    }
                }
            }
        }

        fn try_alloc_slab(&self, bytes: usize) -> Result<DeviceBuffer<'a>> {
            // TRANSFER_SRC/DST so slabs can be staged into and read back for
            // verification, matching `alloc_device_local_from_host`.
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST;
            match self.memory {
                super::SlabMemory::DeviceLocal => DeviceBuffer::alloc_with_usage(
                    self.ctx,
                    bytes,
                    usage,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                ),
                super::SlabMemory::Uma => DeviceBuffer::alloc_with_usage(
                    self.ctx,
                    bytes,
                    usage,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL
                        | vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
                .or_else(|_| {
                    DeviceBuffer::alloc_with_usage(
                        self.ctx,
                        bytes,
                        usage,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
                }),
            }
        }

        /// The slab buffer backing `alloc`, plus its descriptor offset and
        /// range — feed straight to
        /// [`DescriptorSet::storage_buffers_ranged`].
        pub fn binding(&self, alloc: &super::SlabAlloc) -> Result<(&DeviceBuffer<'a>, u64, u64)> {
            let buffer = self.slab(alloc.slab()).ok_or_else(|| {
                VulkanError::Runtime(format!(
                    "slab index {} out of range ({} slabs) — handle from another allocator?",
                    alloc.slab(),
                    self.slabs.len()
                ))
            })?;
            Ok((buffer, alloc.offset(), alloc.len()))
        }

        pub fn slab(&self, index: usize) -> Option<&DeviceBuffer<'a>> {
            self.slabs.get(index)
        }

        pub fn slab_count(&self) -> usize {
            self.slabs.len()
        }

        /// Device memory actually allocated, summed over slabs.
        pub fn committed_bytes(&self) -> u64 {
            self.plan.committed_bytes()
        }

        /// Bytes handed out to callers, excluding alignment padding and tails.
        pub fn used_bytes(&self) -> u64 {
            self.plan.used_bytes()
        }

        pub fn slab_size(&self) -> u64 {
            self.plan.slab_size()
        }

        pub fn alignment(&self) -> u64 {
            self.plan.alignment()
        }

        /// `committed - used`: alignment padding plus slab tails.
        pub fn wasted_bytes(&self) -> u64 {
            self.plan.wasted_bytes()
        }

        /// Fill `alloc` from host memory.
        ///
        /// UMA slabs are mapped and written in place. DEVICE_LOCAL slabs are
        /// not host-mappable, so the bytes go through a reused
        /// [`STAGING_CHUNK_BYTES`] window and a `cmd_copy_buffer` per chunk —
        /// the same trick as [`DeviceBuffer::alloc_device_local_from_host`],
        /// minus its per-tensor allocate/free pair.
        pub fn write(&mut self, alloc: &super::SlabAlloc, src: &[u8]) -> Result<()> {
            let len = u64::try_from(src.len())
                .map_err(|e| runtime_error("converting host slice length", e))?;
            if len > alloc.len() {
                return Err(VulkanError::Runtime(format!(
                    "writing {len} B into a {} B suballocation",
                    alloc.len()
                )));
            }
            if src.is_empty() {
                return Ok(());
            }
            // Split the borrow: the staging buffer and the destination slab are
            // disjoint fields, but both are reached through `self`.
            // Copy the context out first: `slabs` and `staging` are disjoint
            // fields but both are reached through `self`, so the write below
            // needs them borrowed separately.
            let ctx = self.ctx;
            let Self {
                slabs,
                staging,
                memory,
                ..
            } = self;
            let slab = slabs.get_mut(alloc.slab()).ok_or_else(|| {
                VulkanError::Runtime(format!("slab index {} out of range", alloc.slab()))
            })?;
            if matches!(memory, super::SlabMemory::Uma) {
                return slab.copy_from_host_at(alloc.offset(), src);
            }
            let chunk = STAGING_CHUNK_BYTES.min(src.len());
            if staging.as_ref().is_none_or(|buf| buf.len() < chunk) {
                *staging = Some(DeviceBuffer::alloc_with_usage(
                    ctx,
                    chunk,
                    vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )?);
            }
            let staging = staging.as_mut().ok_or_else(|| {
                VulkanError::Runtime("staging buffer missing after allocation".to_string())
            })?;
            let pool = CommandPool::create(ctx)?;
            for (index, part) in src.chunks(chunk).enumerate() {
                staging.copy_from_host(part)?;
                let dst_offset = alloc.offset()
                    + u64::try_from(index * chunk)
                        .map_err(|e| runtime_error("converting staging chunk offset", e))?;
                pool.one_shot_submit(|cmd| {
                    let region = vk::BufferCopy::default()
                        .src_offset(0)
                        .dst_offset(dst_offset)
                        .size(part.len() as vk::DeviceSize);
                    // SAFETY: `cmd` is the live recording command buffer from
                    // `one_shot_submit`. Both buffers outlive the submit and
                    // carry the needed usage: `staging` TRANSFER_SRC, the slab
                    // TRANSFER_DST. `dst_offset + part.len()` is inside the
                    // suballocation, itself checked inside the slab.
                    unsafe {
                        ctx.device
                            .cmd_copy_buffer(cmd, staging.raw(), slab.raw(), &[region]);
                    }
                    Ok(())
                })?;
            }
            Ok(())
        }

        /// Read `alloc` back into `dst` through a temporary staging buffer.
        ///
        /// Verification-only, exactly like
        /// [`DeviceBuffer::copy_to_host_staged`]: it allocates, submits and
        /// blocks. Never put it on a per-token path.
        pub fn read_back(&self, alloc: &super::SlabAlloc, dst: &mut [u8]) -> Result<()> {
            let len = u64::try_from(dst.len())
                .map_err(|e| runtime_error("converting host slice length", e))?;
            if len > alloc.len() {
                return Err(VulkanError::Runtime(format!(
                    "reading {len} B from a {} B suballocation",
                    alloc.len()
                )));
            }
            if dst.is_empty() {
                return Ok(());
            }
            let (slab, offset, _) = self.binding(alloc)?;
            let staging = DeviceBuffer::alloc_with_usage(
                self.ctx,
                dst.len(),
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let pool = CommandPool::create(self.ctx)?;
            pool.one_shot_submit(|cmd| {
                let region = vk::BufferCopy::default()
                    .src_offset(offset)
                    .dst_offset(0)
                    .size(dst.len() as vk::DeviceSize);
                // SAFETY: `cmd` is the live recording command buffer from
                // `one_shot_submit`. The slab (TRANSFER_SRC) and `staging`
                // (TRANSFER_DST) both outlive the submit, and the source range
                // was bounds-checked against the suballocation above.
                unsafe {
                    self.ctx
                        .device
                        .cmd_copy_buffer(cmd, slab.raw(), staging.raw(), &[region]);
                }
                Ok(())
            })?;
            staging.copy_to_host(dst)
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
                unsafe { self.ctx.device.begin_command_buffer(command_buffer, &begin) }
                    .map_err(|e| vk_error("beginning Vulkan command buffer", e))?;
                record(command_buffer)?;
                unsafe { self.ctx.device.end_command_buffer(command_buffer) }
                    .map_err(|e| vk_error("ending Vulkan command buffer", e))?;
                let command_buffers = [command_buffer];
                let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
                unsafe {
                    self.ctx
                        .device
                        .queue_submit(self.ctx.queue, &submits, vk::Fence::null())
                }
                .map_err(|e| vk_error("submitting Vulkan command buffer", e))?;
                unsafe { self.ctx.device.queue_wait_idle(self.ctx.queue) }
                    .map_err(|e| vk_error("waiting for Vulkan queue idle", e))?;
                Ok(())
            })();
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
            let pool = unsafe { ctx.device.create_command_pool(&pool_create, None) }
                .map_err(|e| vk_error("creating Vulkan command pool", e))?;

            let alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffer = match unsafe { ctx.device.allocate_command_buffers(&alloc) } {
                Ok(buffers) => match buffers.first().copied() {
                    Some(buffer) => buffer,
                    None => {
                        unsafe { ctx.device.destroy_command_pool(pool, None) };
                        return Err(VulkanError::Runtime(
                            "Vulkan command buffer allocation returned no buffers".to_string(),
                        ));
                    }
                },
                Err(e) => {
                    unsafe { ctx.device.destroy_command_pool(pool, None) };
                    return Err(vk_error("allocating Vulkan command buffer", e));
                }
            };

            let fence_create = vk::FenceCreateInfo::default();
            let fence = match unsafe { ctx.device.create_fence(&fence_create, None) } {
                Ok(fence) => fence,
                Err(e) => {
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
                unsafe {
                    self.ctx
                        .device
                        .wait_for_fences(&[self.fence], true, u64::MAX)
                }
                .map_err(|e| vk_error("waiting for Vulkan fence", e))?;
                self.pending = false;
            }
            unsafe { self.ctx.device.reset_fences(&[self.fence]) }
                .map_err(|e| vk_error("resetting Vulkan fence", e))?;
            unsafe {
                self.ctx
                    .device
                    .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
            }
            .map_err(|e| vk_error("resetting Vulkan command buffer", e))?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                self.ctx
                    .device
                    .begin_command_buffer(self.command_buffer, &begin)
            }
            .map_err(|e| vk_error("beginning Vulkan command buffer", e))?;
            self.dispatches_in_batch = 0;
            if let Some((pool, cap)) = self.prof.as_ref().map(|p| (p.pool, p.capacity)) {
                let cmd = self.command_buffer;
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

        /// Record a device-to-device buffer copy into THIS command buffer, so
        /// it rides the submit that is already happening.
        ///
        /// The point is read-back staging: the arena is write-combined (fast for
        /// the GPU and for host writes, ~0.10 GB/s for host READS), so anything
        /// the host reads per token should be copied into a
        /// [`DeviceBuffer::alloc_host_cached`] buffer first. On the 8060S that
        /// turns a 970 KB logits read from 10.04 ms into 0.023 ms. Recording the
        /// copy here rather than in a `one_shot_submit` keeps it free: no extra
        /// submit, no extra fence.
        ///
        /// Both buffers need the matching `TRANSFER_SRC`/`TRANSFER_DST` usage,
        /// which every `alloc*` constructor here sets.
        pub fn copy_buffer(
            &mut self,
            src: &DeviceBuffer<'_>,
            src_offset: u64,
            dst: &DeviceBuffer<'_>,
            dst_offset: u64,
            size: u64,
        ) {
            if size == 0 {
                return;
            }
            let region = vk::BufferCopy::default()
                .src_offset(src_offset)
                .dst_offset(dst_offset)
                .size(size);
            // The copy must see the compute writes that produced the data, and
            // the host must see the copy — hence TRANSFER on both sides of the
            // surrounding barriers rather than the COMPUTE-only `barrier()`.
            let to_transfer = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            let to_host = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            // SAFETY: `command_buffer` is live and recording (between `begin`
            // and `submit_and_wait`). Both buffers outlive this call via the
            // borrows, and both carry TRANSFER_SRC|TRANSFER_DST usage. `region`
            // is bounds-checked by the caller against both buffer lengths.
            unsafe {
                self.ctx.device.cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[to_transfer],
                    &[],
                    &[],
                );
                self.ctx.device.cmd_copy_buffer(
                    self.command_buffer,
                    src.buffer,
                    dst.buffer,
                    &[region],
                );
                self.ctx.device.cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[to_host],
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
            unsafe { self.ctx.device.end_command_buffer(self.command_buffer) }
                .map_err(|e| vk_error("ending Vulkan command buffer", e))?;
            let command_buffers = [self.command_buffer];
            let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            unsafe {
                self.ctx
                    .device
                    .queue_submit(self.ctx.queue, &submits, self.fence)
            }
            .map_err(|e| vk_error("submitting Vulkan command buffer", e))?;
            self.pending = true;
            self.submit_count += 1;
            unsafe {
                self.ctx
                    .device
                    .wait_for_fences(&[self.fence], true, u64::MAX)
            }
            .map_err(|e| vk_error("waiting for Vulkan fence", e))?;
            self.pending = false;
            let read = self.prof.as_ref().map(|p| (p.pool, p.idx, p.valid_mask));
            if let Some((pool, idx, mask)) = read {
                if idx > 1 {
                    let mut data = vec![0u64; idx as usize];
                    let ok = unsafe {
                        self.ctx.device.get_query_pool_results(
                            pool,
                            0,
                            &mut data,
                            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                        )
                    };
                    if ok.is_ok() {
                        if let Some(p) = self.prof.as_mut() {
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
                }
            }
            Ok(())
        }
    }

    impl Drop for CommandRecorder<'_> {
        fn drop(&mut self) {
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
            let pool = unsafe { ctx.device.create_descriptor_pool(&pool_create, None) }
                .map_err(|e| vk_error("creating Vulkan descriptor pool", e))?;
            let layouts = [layout.raw()];
            let alloc = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            let sets = match unsafe { ctx.device.allocate_descriptor_sets(&alloc) } {
                Ok(sets) => sets,
                Err(e) => {
                    unsafe { ctx.device.destroy_descriptor_pool(pool, None) };
                    return Err(vk_error("allocating Vulkan descriptor set", e));
                }
            };
            let set = match sets.first().copied() {
                Some(set) => set,
                None => {
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
            let pool = unsafe { ctx.device.create_descriptor_pool(&pool_create, None) }
                .map_err(|e| vk_error("creating Vulkan descriptor pool", e))?;
            let layouts = [layout.raw()];
            let alloc = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            let sets = match unsafe { ctx.device.allocate_descriptor_sets(&alloc) } {
                Ok(sets) => sets,
                Err(e) => {
                    unsafe { ctx.device.destroy_descriptor_pool(pool, None) };
                    return Err(vk_error("allocating Vulkan descriptor set", e));
                }
            };
            let set = match sets.first().copied() {
                Some(set) => set,
                None => {
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
            unsafe { ctx.device.update_descriptor_sets(&writes, &[]) };
            Ok(Self { ctx, pool, set })
        }

        pub fn raw(&self) -> vk::DescriptorSet {
            self.set
        }
    }

    impl Drop for DescriptorSet<'_> {
        fn drop(&mut self) {
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
            let pool = unsafe { ctx.device.create_descriptor_pool(&pool_create, None) }
                .map_err(|e| vk_error("creating Vulkan descriptor pool", e))?;
            let layouts: Vec<_> = std::iter::repeat_n(layout.raw(), ring_size).collect();
            let alloc = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            let sets = match unsafe { ctx.device.allocate_descriptor_sets(&alloc) } {
                Ok(sets) => sets,
                Err(e) => {
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

        /// Slots left before the cursor wraps and starts OVERWRITING sets that
        /// the currently-recorded command buffer may still reference.
        ///
        /// `next_updated` wraps silently — it cannot fail, it just corrupts the
        /// bindings of an already-recorded dispatch. A caller that records more
        /// than `ring_size` dispatches in one batch (batched prefill, which
        /// scales dispatch count with the chunk width) must consult this and
        /// submit + [`reset`](Self::reset) before it runs out.
        #[must_use]
        pub fn remaining(&self) -> usize {
            self.sets.len().saturating_sub(self.next)
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
            unsafe { self.ctx.device.update_descriptor_sets(&writes, &[]) };
            Ok(set)
        }
    }

    impl Drop for DescriptorSetRing<'_> {
        fn drop(&mut self) {
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
            let pipeline = match unsafe {
                ctx.device
                    .create_compute_pipelines(ctx.pipeline_cache, &create, None)
            } {
                Ok(mut pipelines) => match pipelines.pop() {
                    Some(pipeline) => pipeline,
                    None => {
                        unsafe { ctx.device.destroy_pipeline_layout(layout, None) };
                        return Err(VulkanError::Runtime(
                            "Vulkan compute pipeline creation returned no pipelines".to_string(),
                        ));
                    }
                },
                Err((pipelines, e)) => {
                    for pipeline in pipelines {
                        unsafe { ctx.device.destroy_pipeline(pipeline, None) };
                    }
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
            unsafe {
                self.ctx.device.destroy_pipeline(self.pipeline, None);
                self.ctx.device.destroy_pipeline_layout(self.layout, None);
            }
        }
    }
}

#[cfg(feature = "vulkan")]
pub use real::{
    CommandPool, CommandRecorder, ComputePipeline, CoopmatShape, DescriptorSet,
    DescriptorSetLayout, DescriptorSetRing, DeviceBuffer, ShaderModule, SlabAllocator,
    VulkanContext, device_count, device_name, init,
};

#[cfg(not(feature = "vulkan"))]
mod stub {
    use super::{Result, SlabAlloc, SlabMemory, VULKAN_NOT_COMPILED};
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

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CoopmatShape {
        pub m: u32,
        pub n: u32,
        pub k: u32,
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

        pub fn coopmat(&self) -> Option<CoopmatShape> {
            None
        }

        pub fn queue_family_index(&self) -> u32 {
            0
        }

        pub fn min_storage_buffer_offset_alignment(&self) -> u64 {
            0
        }

        pub fn max_compute_shared_memory_size(&self) -> u32 {
            0
        }

        pub fn max_memory_allocation_size(&self) -> u64 {
            0
        }

        pub fn max_memory_allocation_count(&self) -> u32 {
            0
        }

        pub fn max_storage_buffer_range(&self) -> u32 {
            0
        }
    }

    pub struct SlabAllocator<'a> {
        _marker: PhantomData<&'a VulkanContext>,
    }

    impl<'a> SlabAllocator<'a> {
        pub fn new(_ctx: &'a VulkanContext) -> Result<Self> {
            Err(VULKAN_NOT_COMPILED)
        }

        pub fn with_slab_size(_ctx: &'a VulkanContext, _slab_size: u64) -> Result<Self> {
            Err(VULKAN_NOT_COMPILED)
        }

        pub fn with_slab_size_and_memory(
            _ctx: &'a VulkanContext,
            _slab_size: u64,
            _memory: SlabMemory,
        ) -> Result<Self> {
            Err(VULKAN_NOT_COMPILED)
        }

        pub fn alloc(&mut self, _len: u64) -> Result<SlabAlloc> {
            Err(VULKAN_NOT_COMPILED)
        }

        pub fn binding(&self, _alloc: &SlabAlloc) -> Result<(&DeviceBuffer<'a>, u64, u64)> {
            Err(VULKAN_NOT_COMPILED)
        }

        pub fn slab(&self, _index: usize) -> Option<&DeviceBuffer<'a>> {
            None
        }

        pub fn slab_count(&self) -> usize {
            0
        }

        pub fn committed_bytes(&self) -> u64 {
            0
        }

        pub fn used_bytes(&self) -> u64 {
            0
        }

        pub fn slab_size(&self) -> u64 {
            0
        }

        pub fn alignment(&self) -> u64 {
            0
        }

        pub fn wasted_bytes(&self) -> u64 {
            0
        }

        pub fn write(&mut self, _alloc: &SlabAlloc, _src: &[u8]) -> Result<()> {
            Err(VULKAN_NOT_COMPILED)
        }

        pub fn read_back(&self, _alloc: &SlabAlloc, _dst: &mut [u8]) -> Result<()> {
            Err(VULKAN_NOT_COMPILED)
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

        pub fn alloc_host_cached(_ctx: &'a VulkanContext, _len: usize) -> Result<Self> {
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

        pub fn copy_to_host_staged(&self, _dst: &mut [u8]) -> Result<()> {
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

        pub fn copy_buffer(
            &mut self,
            _src: &DeviceBuffer<'_>,
            _src_offset: u64,
            _dst: &DeviceBuffer<'_>,
            _dst_offset: u64,
            _size: u64,
        ) {
        }

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

        #[must_use]
        pub fn remaining(&self) -> usize {
            0
        }
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
    CommandPool, CommandRecorder, ComputePipeline, CoopmatShape, DescriptorSet,
    DescriptorSetLayout, DescriptorSetRing, DeviceBuffer, ShaderModule, SlabAllocator,
    VulkanContext, device_count, device_name, init,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Nominal slab size used by the device-free plan tests.
    ///
    /// `maxMemoryAllocationSize` measured on this box's driver (Radeon 8060S,
    /// Vulkan). [`SlabAllocator::new`] queries the device rather than trusting
    /// this; the constant only lets the planning tests run with no GPU.
    const DRIVER_MAX_ALLOC_BYTES: u64 = 2 << 30;

    #[test]
    fn slab_plan_packs_to_ceil_not_one_slab_per_tensor() {
        const SLAB: u64 = 512 << 20;
        const SUB: u64 = 4 << 20;
        const N: u64 = 768;
        let mut plan = SlabPlan::new(SLAB, 64).expect("plan");
        for _ in 0..N {
            plan.place(SUB).expect("place");
        }
        assert_eq!(
            plan.slab_count() as u64,
            (N * SUB).div_ceil(SLAB),
            "packing must be ceil(total / slab_size), not one slab per tensor"
        );
        assert_eq!(plan.slab_count(), 6);
        assert_eq!(plan.used_bytes(), N * SUB);
        assert_eq!(
            plan.wasted_bytes(),
            0,
            "evenly-dividing suballocations must leave no tail"
        );
        assert_eq!(plan.committed_bytes(), 6 * SLAB);
    }

    /// The reason placement is first-fit rather than a bump pointer into the
    /// newest slab.
    ///
    /// `SHARD` is a real tensor size from this checkpoint — one PLE n-gram
    /// embedding shard, `model.language_model.layers.1.ple.ple_embedding
    /// .ngram_embedding.shard_N.weight`. Five fill 2.00 of a 2 GiB slab and the
    /// sixth cannot follow, so a newest-slab-only bump pointer would strand a
    /// 147 MB tail (6.9%) in every slab it touches. First-fit gives that tail
    /// back to the small tensors, which dominate this checkpoint by count.
    #[test]
    fn slab_plan_backfills_tails_instead_of_stranding_them() {
        const SHARD: u64 = 400_001_920;
        let mut plan = SlabPlan::new(DRIVER_MAX_ALLOC_BYTES, 256).expect("plan");
        for _ in 0..5 {
            plan.place(SHARD).expect("shard");
        }
        assert_eq!(plan.slab_count(), 1, "5 PLE shards fit one 2 GiB slab");
        plan.place(SHARD).expect("sixth shard");
        assert_eq!(plan.slab_count(), 2, "the sixth shard needs a new slab");

        let small = plan.place(1 << 20).expect("small tensor");
        assert_eq!(
            small.slab(),
            0,
            "first-fit must backfill slab 0's tail, not append to the newest slab"
        );
        assert!(
            small.offset() >= 5 * SHARD,
            "backfill must land after the shards already in slab 0"
        );
        assert_eq!(plan.slab_count(), 2, "backfilling must not open a slab");
    }

    #[test]
    fn slab_plan_rejects_a_tensor_larger_than_one_allocation() {
        let mut plan = SlabPlan::new(DRIVER_MAX_ALLOC_BYTES, 256).expect("plan");
        let err = plan
            .place(DRIVER_MAX_ALLOC_BYTES + 1)
            .expect_err("must reject");
        let msg = err.to_string();
        assert!(msg.contains("maxMemoryAllocationSize"), "{msg}");
        assert_eq!(
            plan.slab_count(),
            0,
            "a rejected request must not commit a slab"
        );
        assert!(plan.place(0).is_err(), "zero-length suballocation");
    }

    #[test]
    fn slab_plan_rejects_bad_geometry() {
        assert!(SlabPlan::new(0, 256).is_err(), "zero slab size");
        assert!(
            SlabPlan::new(1 << 30, 96).is_err(),
            "non-power-of-two alignment"
        );
        assert!(
            SlabPlan::new(128, 256).is_err(),
            "alignment larger than the slab"
        );
    }

    /// Directory of the qwen4_exp checkpoint, overridable for other boxes.
    fn qwen4_exp_dir() -> std::path::PathBuf {
        std::env::var_os("ARLE_QWEN4_EXP_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(r"C:\Users\Asus\models\qwen3.8-flash-next-nvfp4")
            })
    }

    /// Byte length of every tensor in a `.safetensors` file, with its name.
    ///
    /// Only the header is read (an 8-byte little-endian length followed by that
    /// many bytes of JSON — the safetensors format), so this walks a 126 GiB
    /// checkpoint in well under a second. A tensor's size is the span of its
    /// `data_offsets` pair; the name is the object key that opens the record,
    /// which is enough structure to skip a JSON dependency this crate does not
    /// have.
    fn safetensors_tensor_sizes(path: &std::path::Path) -> Option<Vec<(String, u64)>> {
        use std::io::Read;

        fn skip_ws(bytes: &[u8], index: &mut usize) {
            while bytes.get(*index).is_some_and(u8::is_ascii_whitespace) {
                *index += 1;
            }
        }
        fn take(bytes: &[u8], index: &mut usize, want: u8) -> Option<()> {
            skip_ws(bytes, index);
            if *bytes.get(*index)? != want {
                return None;
            }
            *index += 1;
            Some(())
        }
        fn take_u64(bytes: &[u8], index: &mut usize) -> Option<u64> {
            skip_ws(bytes, index);
            let start = *index;
            while bytes.get(*index).is_some_and(u8::is_ascii_digit) {
                *index += 1;
            }
            std::str::from_utf8(bytes.get(start..*index)?)
                .ok()?
                .parse()
                .ok()
        }
        /// The object key that opens the record containing byte `at`.
        fn key_before(header: &str, at: usize) -> String {
            let bytes = header.as_bytes();
            let back = |end: usize, needle: u8| -> Option<usize> {
                bytes.get(..end)?.iter().rposition(|byte| *byte == needle)
            };
            let Some(brace) = back(at, b'{') else {
                return String::new();
            };
            let Some(close) = back(brace, b'"') else {
                return String::new();
            };
            let Some(open) = back(close, b'"') else {
                return String::new();
            };
            header.get(open + 1..close).unwrap_or_default().to_string()
        }

        let mut file = std::fs::File::open(path).ok()?;
        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes).ok()?;
        let header_len = usize::try_from(u64::from_le_bytes(len_bytes)).ok()?;
        // Guard against a truncated or foreign file claiming a huge header.
        if header_len == 0 || header_len > (256 << 20) {
            return None;
        }
        let mut header = vec![0u8; header_len];
        file.read_exact(&mut header).ok()?;
        let header = String::from_utf8(header).ok()?;

        const KEY: &str = "\"data_offsets\"";
        let bytes = header.as_bytes();
        let mut out = Vec::new();
        for (at, _) in header.match_indices(KEY) {
            let mut index = at + KEY.len();
            take(bytes, &mut index, b':')?;
            take(bytes, &mut index, b'[')?;
            let begin = take_u64(bytes, &mut index)?;
            take(bytes, &mut index, b',')?;
            let end = take_u64(bytes, &mut index)?;
            take(bytes, &mut index, b']')?;
            out.push((key_before(&header, at), end.saturating_sub(begin)));
        }
        Some(out)
    }

    /// Plan the REAL qwen4_exp checkpoint (296 475 tensors, 125.9 GiB) through
    /// the same placement code [`SlabAllocator`] uses, with no GPU involved.
    ///
    /// One `vkAllocateMemory` per tensor would need 296 475 live allocations,
    /// a 72x overrun of the 4096 floor `maxMemoryAllocationCount` is guaranteed
    /// to allow (though not of what this driver actually reports — see
    /// [`SlabPlan`]). Slabs turn it into 64.
    ///
    /// It also pins the one shape the slab scheme genuinely cannot express: a
    /// tensor larger than `maxMemoryAllocationSize` has no contiguous home. The
    /// test asserts every such tensor belongs to the MTP draft head, which the
    /// base forward pass does not load.
    #[test]
    fn slab_plan_fits_the_real_qwen4_exp_checkpoint() {
        /// The floor `maxMemoryAllocationCount` is guaranteed to allow on
        /// any conformant device. Asserting against the floor rather than the
        /// local value keeps the plan portable off this box.
        const MIN_ALLOCATION_COUNT: usize = 4096;
        // 16 B is the floor `SlabAllocator` applies on top of
        // `minStorageBufferOffsetAlignment`; 256 is the largest value current
        // desktop drivers report, so planning at 256 is the pessimistic case
        // for padding.
        const ALIGNMENT: u64 = 256;

        let dir = qwen4_exp_dir();
        let Ok(entries) = std::fs::read_dir(&dir) else {
            eprintln!(
                "slab plan test: checkpoint not at {} — skipping (set ARLE_QWEN4_EXP_DIR)",
                dir.display()
            );
            return;
        };
        let mut shards: Vec<std::path::PathBuf> = entries
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        shards.sort();
        if shards.is_empty() {
            eprintln!(
                "slab plan test: no .safetensors in {} — skipping",
                dir.display()
            );
            return;
        }

        let mut tensors = Vec::new();
        for shard in &shards {
            match safetensors_tensor_sizes(shard) {
                Some(sizes) => tensors.extend(sizes),
                None => panic!("failed to read safetensors header of {}", shard.display()),
            }
        }
        assert!(
            tensors.len() > 100_000,
            "expected a six-figure tensor count, got {}",
            tensors.len()
        );

        let mut plan = SlabPlan::new(DRIVER_MAX_ALLOC_BYTES, ALIGNMENT).expect("plan");
        let mut last_end: Vec<u64> = Vec::new();
        let mut oversize: Vec<(String, u64)> = Vec::new();
        let mut placed = 0usize;
        let mut total_bytes = 0u64;
        for (name, len) in &tensors {
            total_bytes += *len;
            let Ok(alloc) = plan.place(*len) else {
                oversize.push((name.clone(), *len));
                continue;
            };
            assert_eq!(
                alloc.offset() % ALIGNMENT,
                0,
                "{name} landed at unaligned offset {}",
                alloc.offset()
            );
            if last_end.len() <= alloc.slab() {
                last_end.resize(alloc.slab() + 1, 0);
            }
            assert!(
                alloc.offset() >= last_end[alloc.slab()],
                "{name} overlaps the previous tensor in slab {}",
                alloc.slab()
            );
            assert!(
                alloc.end() <= DRIVER_MAX_ALLOC_BYTES,
                "{name} runs past the end of its slab"
            );
            last_end[alloc.slab()] = alloc.end();
            placed += 1;
        }

        assert_eq!(placed + oversize.len(), tensors.len());
        assert!(
            oversize.iter().all(|(name, _)| name.starts_with("mtp.")),
            "a non-MTP tensor exceeds maxMemoryAllocationSize and has no contiguous home: {oversize:?}"
        );
        assert!(
            plan.slab_count() <= MIN_ALLOCATION_COUNT,
            "slab count {} exceeds the guaranteed maxMemoryAllocationCount",
            plan.slab_count()
        );
        assert!(
            plan.slab_count() < tensors.len() / 1000,
            "slabs ({}) must be orders of magnitude below tensors ({})",
            plan.slab_count(),
            tensors.len()
        );
        // The sharper statement of "packs well": how far above the information
        // -theoretic floor of ceil(bytes / slab_size) the online placement
        // lands. Measured here: 64 slabs against a 62-slab floor, i.e. 2 slabs
        // (4 GiB) of fragmentation over 122.75 GiB, 4.10% waste. Feeding the
        // same sizes largest-first reaches the floor exactly (1.01% waste), so
        // a loader that can sort its tensors should — see [`SlabPlan`].
        let placed_bytes = plan.used_bytes();
        let floor = usize::try_from(placed_bytes.div_ceil(DRIVER_MAX_ALLOC_BYTES))
            .expect("slab floor fits usize");
        assert!(
            plan.slab_count() <= floor + 4,
            "online placement used {} slabs against a {floor}-slab floor",
            plan.slab_count()
        );
        // Alignment padding plus slab tails, over the whole checkpoint.
        let waste_pct = plan.wasted_bytes() as f64 / plan.committed_bytes() as f64 * 100.0;

        let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
        eprintln!(
            "slab plan: {} tensors / {:.2} GiB across {} shards -> {} slabs of {} GiB \
             (floor {floor}, {:.2} GiB committed, {:.2}% waste); \
             {} oversize tensor(s) excluded",
            tensors.len(),
            gib(total_bytes),
            shards.len(),
            plan.slab_count(),
            DRIVER_MAX_ALLOC_BYTES >> 30,
            gib(plan.committed_bytes()),
            waste_pct,
            oversize.len(),
        );
        for (name, len) in &oversize {
            eprintln!(
                "slab plan: {name} is {:.2} GiB — must be split across bindings",
                gib(*len)
            );
        }
    }

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
        // Diagnostic, not an assertion: a device without matrix cores is a
        // supported configuration (the prefill GEMM falls back to `mul_mmq`).
        match ctx.coopmat() {
            Some(s) => eprintln!(
                "vulkan-sys smoke: coopmat f16xf16->f32 = {}x{}x{}",
                s.m, s.n, s.k
            ),
            None => eprintln!("vulkan-sys smoke: coopmat = unsupported"),
        }

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

    /// The GPU half of the slab story: gigabytes of real device memory handed
    /// out as many small suballocations from a handful of `vkAllocateMemory`
    /// calls, and a suballocation at a NONZERO offset that a compute shader
    /// actually writes through the existing ranged-descriptor path.
    ///
    /// The load-bearing assertion is the last one. The dispatch is pointed at
    /// the second suballocation in slab 0; if the descriptor offset were
    /// dropped anywhere between [`SlabAllocator::binding`] and
    /// [`DescriptorSet::storage_buffers_ranged`], the shader would write slab
    /// offset 0 instead and corrupt its neighbour — which is exactly what the
    /// neighbour check catches.
    ///
    /// Kept to ~3 GiB: this box shares one GPU between agents.
    #[cfg(feature = "vulkan")]
    #[test]
    fn slab_allocator_suballocates_gigabytes_in_few_slabs() {
        // A forced 512 MiB slab (rather than the device's 2 GiB) keeps the test
        // inside 3 GiB while still crossing several slab boundaries.
        const SLAB: u64 = 512 << 20;
        const SUB: u64 = 4 << 20;
        const COUNT: u64 = 768;

        if init().is_err() {
            eprintln!("slab allocator test: loader unavailable — skipping");
            return;
        }
        match device_count() {
            Ok(0) | Err(_) => {
                eprintln!("slab allocator test: no devices — skipping");
                return;
            }
            Ok(_) => {}
        }
        let ctx = match VulkanContext::create() {
            Ok(ctx) => ctx,
            Err(VulkanError::NoComputeDevice) => {
                eprintln!("slab allocator test: no compute queue — skipping");
                return;
            }
            Err(e) => panic!("failed to create Vulkan context: {e}"),
        };
        eprintln!(
            "slab allocator: {} — maxMemoryAllocationSize {} MiB,              maxMemoryAllocationCount {}, maxStorageBufferRange {} MiB,              minStorageBufferOffsetAlignment {} B",
            ctx.device_name(),
            ctx.max_memory_allocation_size() >> 20,
            ctx.max_memory_allocation_count(),
            u64::from(ctx.max_storage_buffer_range()) >> 20,
            ctx.min_storage_buffer_offset_alignment(),
        );

        let mut slabs = match SlabAllocator::with_slab_size(&ctx, SLAB) {
            Ok(slabs) => slabs,
            Err(e) => panic!("failed to create slab allocator: {e}"),
        };
        let slab_size = slabs.slab_size();
        assert_eq!(
            SUB % slabs.alignment(),
            0,
            "test assumes suballocations that pack a slab exactly"
        );

        let mut allocs = Vec::with_capacity(COUNT as usize);
        for index in 0..COUNT {
            match slabs.alloc(SUB) {
                Ok(alloc) => allocs.push(alloc),
                Err(e) => {
                    // Another agent may hold the heap; that is not a defect.
                    eprintln!(
                        "slab allocator test: device out of memory after {index} of {COUNT}                          suballocations — skipping ({e})"
                    );
                    return;
                }
            }
        }

        for alloc in &allocs {
            assert_eq!(
                alloc.offset() % slabs.alignment(),
                0,
                "suballocation at {} violates minStorageBufferOffsetAlignment",
                alloc.offset()
            );
            assert!(alloc.slab() < slabs.slab_count(), "slab index out of range");
            assert!(alloc.end() <= slab_size, "suballocation runs past its slab");
        }
        let mut sorted = allocs.clone();
        sorted.sort_by_key(|alloc| (alloc.slab(), alloc.offset()));
        for pair in sorted.windows(2) {
            let (lhs, rhs) = (pair[0], pair[1]);
            if lhs.slab() == rhs.slab() {
                assert!(
                    lhs.end() <= rhs.offset(),
                    "suballocations overlap in slab {}: [{}, {}) vs [{}, {})",
                    lhs.slab(),
                    lhs.offset(),
                    lhs.end(),
                    rhs.offset(),
                    rhs.end()
                );
            }
        }

        let total = COUNT * SUB;
        assert_eq!(
            slabs.slab_count() as u64,
            total.div_ceil(slab_size),
            "slab count must be ceil(total / slab size), not one per suballocation"
        );
        assert!(
            (slabs.slab_count() as u64) * 100 < COUNT,
            "{} slabs for {COUNT} suballocations is not a suballocator",
            slabs.slab_count()
        );
        assert_eq!(slabs.used_bytes(), total);
        assert_eq!(
            slabs.committed_bytes(),
            slabs.slab_count() as u64 * slab_size
        );
        let live = u32::try_from(slabs.slab_count()).expect("slab count fits u32");
        assert!(
            live <= ctx.max_memory_allocation_count(),
            "{live} live allocations exceeds maxMemoryAllocationCount {}",
            ctx.max_memory_allocation_count()
        );

        // The slab size must come from the device, not a hardcoded 2 GiB:
        // a default allocator's geometry has to match what this driver reports,
        // and a request past it has to be refused rather than silently split.
        {
            let mut device_sized = SlabAllocator::new(&ctx).expect("default slab allocator");
            assert_eq!(
                device_sized.slab_size(),
                ctx.max_memory_allocation_size(),
                "default slab size must be the queried maxMemoryAllocationSize"
            );
            let err = device_sized
                .alloc(ctx.max_memory_allocation_size() + 1)
                .expect_err("a suballocation past maxMemoryAllocationSize must be refused");
            assert!(err.to_string().contains("maxMemoryAllocationSize"), "{err}");
            assert_eq!(
                device_sized.slab_count(),
                0,
                "a refused request must not have committed device memory"
            );
        }

        // Round-trip the last suballocation: it is in the last slab at a
        // nonzero offset, so both the staged write and the staged read have to
        // get the offset right.
        let target = *allocs.last().expect("at least one suballocation");
        assert!(target.slab() > 0, "expected the last slab");
        let payload: Vec<u8> = (0..4096u32).map(|byte| (byte % 251) as u8).collect();
        slabs.write(&target, &payload).expect("staged write");
        let mut back = vec![0u8; payload.len()];
        slabs.read_back(&target, &mut back).expect("staged read");
        assert_eq!(payload, back, "staged round-trip through a slab mismatched");

        eprintln!(
            "slab allocator: {COUNT} x {} MiB suballocations -> {} slabs of {} MiB              ({} MiB committed, {} MiB used)",
            SUB >> 20,
            slabs.slab_count(),
            slab_size >> 20,
            slabs.committed_bytes() >> 20,
            slabs.used_bytes() >> 20,
        );

        // UMA slabs take the mapped-write branch of `write` instead of the
        // staging one; the offset arithmetic differs and needs its own proof.
        {
            let mut uma =
                match SlabAllocator::with_slab_size_and_memory(&ctx, 64 << 20, SlabMemory::Uma) {
                    Ok(uma) => uma,
                    Err(e) => panic!("failed to create UMA slab allocator: {e}"),
                };
            let head = uma.alloc(1 << 20).expect("uma head");
            let tail = uma.alloc(1 << 20).expect("uma tail");
            assert_eq!(uma.slab_count(), 1, "two 1 MiB subs fit one 64 MiB slab");
            assert!(tail.offset() >= head.end());
            uma.write(&tail, &payload).expect("mapped write");
            let mut uma_back = vec![0u8; payload.len()];
            uma.read_back(&tail, &mut uma_back).expect("uma read");
            assert_eq!(
                payload, uma_back,
                "mapped write at a slab offset mismatched"
            );
        }

        let Some(glslc) = find_glslc() else {
            eprintln!("slab allocator test: glslc not found — skipping the bind-at-offset half");
            return;
        };
        let Some(spirv) = compile_add_shader(&glslc) else {
            eprintln!("slab allocator test: shader compile failed — skipping bind-at-offset");
            return;
        };

        const N: usize = 1024;
        const ADDEND: i32 = 7;
        let first = allocs[0];
        let second = allocs[1];
        assert_eq!(first.slab(), second.slab(), "expected two subs in slab 0");
        assert_eq!(first.offset(), 0);
        assert!(
            second.offset() > 0,
            "the neighbour check needs a real offset"
        );

        let pattern: Vec<u8> = (0..N as i32).flat_map(i32::to_le_bytes).collect();
        slabs.write(&first, &pattern).expect("seed neighbour");
        slabs.write(&second, &pattern).expect("seed target");

        let shader = ShaderModule::from_spirv_bytes(&ctx, &spirv).expect("add shader");
        let layout = DescriptorSetLayout::storage_buffers(&ctx, 1).expect("layout");
        // push = { uint n; int addend; } = 8 bytes.
        let pipeline = ComputePipeline::create_with_push_constants(&ctx, &shader, &[&layout], 8)
            .expect("pipeline");
        let (buffer, offset, len) = slabs.binding(&second).expect("binding");
        let set = DescriptorSet::storage_buffers_ranged(&ctx, &layout, &[(buffer, offset, len)])
            .expect("ranged descriptor set");
        let push = [(N as u32).to_le_bytes(), ADDEND.to_le_bytes()].concat();
        let mut rec = CommandRecorder::new(&ctx).expect("recorder");
        rec.begin().expect("begin");
        rec.dispatch(&pipeline, &set, &push, [(N as u32).div_ceil(64), 1, 1]);
        rec.submit_and_wait().expect("submit");

        let ints = |bytes: &[u8]| -> Vec<i32> {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| i32::from_le_bytes(*chunk))
                .collect()
        };
        let mut target_back = vec![0u8; pattern.len()];
        slabs
            .read_back(&second, &mut target_back)
            .expect("read target");
        let mut neighbour_back = vec![0u8; pattern.len()];
        slabs
            .read_back(&first, &mut neighbour_back)
            .expect("read neighbour");
        assert_eq!(
            ints(&target_back),
            (0..N as i32).map(|v| v + ADDEND).collect::<Vec<_>>(),
            "the ranged bind never reached the suballocation"
        );
        assert_eq!(
            ints(&neighbour_back),
            (0..N as i32).collect::<Vec<_>>(),
            "the dispatch wrote slab offset 0 — the descriptor offset was ignored"
        );
        eprintln!(
            "slab allocator: dispatch through a slab binding at offset {offset} hit the              suballocation and left its neighbour at offset 0 untouched"
        );
    }
}
