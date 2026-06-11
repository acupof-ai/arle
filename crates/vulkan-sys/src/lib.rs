//! Ash-backed Vulkan runtime wrapper for the AIPC Vulkan backend (#71/#76/#77).
//!
//! Feature contract mirrors `hip-sys`:
//! - `--features vulkan`: dynamically loads the Vulkan loader through `ash`.
//! - default: every entry point returns [`VULKAN_NOT_COMPILED`].

/// Sentinel error returned by every stub entry point when built without
/// `vulkan`.
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

    fn runtime_error(context: &str, err: impl std::fmt::Display) -> VulkanError {
        VulkanError::Runtime(format!("{context}: {err}"))
    }

    fn vk_error(context: &str, err: vk::Result) -> VulkanError {
        VulkanError::Runtime(format!("{context}: {err:?}"))
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
            .api_version(vk::API_VERSION_1_1);
        let create = vk::InstanceCreateInfo::default().application_info(&app);
        unsafe { entry.create_instance(&create, None) }
            .map_err(|e| vk_error("creating Vulkan instance", e))
    }

    fn load_entry() -> Result<Entry> {
        unsafe { Entry::load() }.map_err(|e| runtime_error("loading Vulkan loader", e))
    }

    fn pick_compute_queue(
        instance: &ash::Instance,
    ) -> Result<(vk::PhysicalDevice, u32, vk::PhysicalDeviceProperties)> {
        let devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| vk_error("enumerating Vulkan physical devices", e))?;
        for physical_device in devices {
            let families =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
            for (idx, family) in families.iter().enumerate() {
                if family.queue_count > 0 && family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    let props = unsafe { instance.get_physical_device_properties(physical_device) };
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

    /// Dynamically loaded Vulkan runtime context with one compute queue.
    pub struct VulkanContext {
        _entry: Entry,
        instance: ash::Instance,
        device: ash::Device,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
        queue: vk::Queue,
        device_name: String,
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
            let create = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info);
            let device = match unsafe { instance.create_device(physical_device, &create, None) } {
                Ok(device) => device,
                Err(e) => {
                    unsafe { instance.destroy_instance(None) };
                    return Err(vk_error("creating Vulkan device", e));
                }
            };
            let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
            Ok(Self {
                _entry: entry,
                instance,
                device,
                physical_device,
                queue_family_index,
                queue,
                device_name: device_name_from_properties(&props),
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

        fn memory_type_index(
            &self,
            type_bits: u32,
            required: vk::MemoryPropertyFlags,
        ) -> Result<u32> {
            let props = unsafe {
                self.instance
                    .get_physical_device_memory_properties(self.physical_device)
            };
            for i in 0..props.memory_type_count {
                let bit = 1u32.checked_shl(i).unwrap_or(0);
                if type_bits & bit == 0 {
                    continue;
                }
                let flags = props.memory_types[i as usize].property_flags;
                if flags.contains(required) {
                    return Ok(i);
                }
            }
            Err(VulkanError::NoMemoryType {
                type_bits,
                required_flags: required.as_raw(),
            })
        }
    }

    impl Drop for VulkanContext {
        fn drop(&mut self) {
            unsafe {
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

    /// Host-visible storage buffer. Device-local staging lands with the first
    /// throughput-sensitive caller; P0 needs deterministic H2D/D2H smoke.
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
            assert!(src.len() <= self.len, "host slice exceeds Vulkan buffer");
            if src.is_empty() {
                return Ok(());
            }
            let ptr = unsafe {
                self.ctx.device.map_memory(
                    self.memory,
                    0,
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

        pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
            assert!(dst.len() <= self.len, "host slice exceeds Vulkan buffer");
            if dst.is_empty() {
                return Ok(());
            }
            let ptr = unsafe {
                self.ctx.device.map_memory(
                    self.memory,
                    0,
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
    }

    impl Drop for DeviceBuffer<'_> {
        fn drop(&mut self) {
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
            let command_buffer = match buffers.first().copied() {
                Some(command_buffer) => command_buffer,
                None => {
                    return Err(VulkanError::Runtime(
                        "Vulkan command buffer allocation returned no buffers".to_string(),
                    ));
                }
            };
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

    pub struct ShaderModule<'a> {
        ctx: &'a VulkanContext,
        module: vk::ShaderModule,
    }

    impl<'a> ShaderModule<'a> {
        pub fn from_spirv_bytes(ctx: &'a VulkanContext, bytes: &[u8]) -> Result<Self> {
            if !bytes.len().is_multiple_of(4) {
                return Err(VulkanError::InvalidSpirvLength(bytes.len()));
            }
            let mut words = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
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

        pub fn raw(&self) -> vk::DescriptorSet {
            self.set
        }
    }

    impl Drop for DescriptorSet<'_> {
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
            let set_layouts: Vec<_> = descriptor_layouts
                .iter()
                .map(|layout| layout.raw())
                .collect();
            let layout_create = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            let layout = unsafe { ctx.device.create_pipeline_layout(&layout_create, None) }
                .map_err(|e| vk_error("creating Vulkan pipeline layout", e))?;
            let entry =
                CString::new("main").map_err(|e| runtime_error("building shader entry", e))?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader.raw())
                .name(&entry);
            let create = [vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)];
            let pipeline = match unsafe {
                ctx.device
                    .create_compute_pipelines(vk::PipelineCache::null(), &create, None)
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
    CommandPool, ComputePipeline, DescriptorSet, DescriptorSetLayout, DeviceBuffer, ShaderModule,
    VulkanContext, device_count, device_name, init,
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
    }

    pub struct DeviceBuffer<'a> {
        _marker: PhantomData<&'a VulkanContext>,
    }

    impl<'a> DeviceBuffer<'a> {
        pub fn alloc(_ctx: &'a VulkanContext, _len: usize) -> Result<Self> {
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

        pub fn copy_to_host(&self, _dst: &mut [u8]) -> Result<()> {
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
    }
}

#[cfg(not(feature = "vulkan"))]
pub use stub::{
    CommandPool, ComputePipeline, DescriptorSet, DescriptorSetLayout, DeviceBuffer, ShaderModule,
    VulkanContext, device_count, device_name, init,
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
}
