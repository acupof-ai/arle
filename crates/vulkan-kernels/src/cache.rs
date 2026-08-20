//! Compile-once pipeline cache for the Vulkan decode path.
//!
//! Per-dispatch, the legacy `launch_with_params_and_specialization` rebuilt the
//! entire pipeline object graph from scratch — `fs::read(.spv)` → `ShaderModule`
//! → `DescriptorSetLayout` → `ComputePipeline` (with a NULL pipeline cache) —
//! and dropped all of it at call end (~900 rebuilds/token). That is the
//! second-dominant host overhead behind the per-op queue drain.
//!
//! [`KernelCache`] collapses that to a one-time build: each distinct
//! `(Kernel, specialization, push-constant bytes, binding count)` reads its
//! `.spv` and builds its `(ComputePipeline, DescriptorSetLayout)` exactly once,
//! then serves every later dispatch from the map. This mirrors
//! `ggml-vulkan.cpp` `ggml_vk_create_pipeline_func` (build pipeline + a shared
//! DSL once, `pipeline->compiled = true`) feeding `ggml_vk_dispatch_pipeline`
//! (per dispatch = bind + push + dispatch only, no object creation).
//!
//! The descriptor *set* (which binds concrete buffers) is intentionally NOT
//! cached here — it is per-call data. The cache holds only the immutable,
//! expensive-to-build pipeline + layout. Recording a dispatch from a cached
//! pipeline + a freshly-bound set is [`record_dispatch`].

use std::collections::HashMap;
use std::path::PathBuf;

use vulkan_sys::{
    CommandRecorder, ComputePipeline, DescriptorSet, DescriptorSetLayout, ShaderModule,
    VulkanContext,
};

use crate::{Dispatch, Kernel, KernelError, Result};

/// Cache key for a built pipeline. Two launches collapse to the same cached
/// pipeline iff they share the kernel, the full specialization-constant vector,
/// the push-constant byte size, and the descriptor binding count — exactly the
/// inputs that change the compiled `ComputePipeline` / `DescriptorSetLayout`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CacheKey {
    kernel: Kernel,
    specialization: Vec<(u32, u32)>,
    push_bytes: u32,
    binding_count: usize,
}

/// `Kernel` is `Copy` but not `Hash`; hash by its stable shader name (1:1 with
/// the variant) so the key is hashable without touching the verbatim enum.
impl std::hash::Hash for CacheKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.kernel.shader_name().hash(state);
        self.specialization.hash(state);
        self.push_bytes.hash(state);
        self.binding_count.hash(state);
    }
}

/// Compile-once pipeline cache. Owns its `(ComputePipeline, DescriptorSetLayout)`
/// entries (RAII — they drop with the cache, before the `VulkanContext`).
///
/// Lifetime `'a` ties cached pipelines to the `VulkanContext` they were built
/// against; the cache must be dropped before that context. A persistent cache
/// is built once at model load and threaded through the whole decode;
/// the legacy `launch_*` path routes through a transient cache so its build
/// also flows through the single cache-miss builder below.
pub struct KernelCache<'a> {
    entries: HashMap<CacheKey, (ComputePipeline<'a>, DescriptorSetLayout<'a>)>,
    /// Number of cache-miss pipeline builds. The build-once test asserts this is
    /// `1` after 100 `get`/`record_dispatch` calls for one kernel.
    build_count: usize,
}

impl<'a> KernelCache<'a> {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            build_count: 0,
        }
    }

    /// Total cache-miss pipeline builds performed so far (the instrumented
    /// build counter). One distinct `(kernel, spec, push, bindings)` increments
    /// this exactly once, on its first `get`.
    pub fn build_count(&self) -> usize {
        self.build_count
    }

    /// Number of distinct cached pipelines.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Fetch the `(ComputePipeline, DescriptorSetLayout)` for this signature,
    /// building (and caching) it on a miss. The build path — the only place that
    /// reads a `.spv`, makes a `ShaderModule`, a `DescriptorSetLayout`, and a
    /// `ComputePipeline` — runs at most once per distinct signature.
    pub fn get(
        &mut self,
        ctx: &'a VulkanContext,
        kernel: Kernel,
        specialization_u32: &[(u32, u32)],
        push_bytes: u32,
        binding_count: usize,
    ) -> Result<&(ComputePipeline<'a>, DescriptorSetLayout<'a>)> {
        let key = CacheKey {
            kernel,
            specialization: specialization_u32.to_vec(),
            push_bytes,
            binding_count,
        };
        if !self.entries.contains_key(&key) {
            let built = build_pipeline(ctx, kernel, specialization_u32, push_bytes, binding_count)?;
            self.build_count += 1;
            self.entries.insert(key.clone(), built);
        }
        // Present by construction (just inserted or already there).
        Ok(self
            .entries
            .get(&key)
            .expect("cache entry present after build"))
    }
}

impl Default for KernelCache<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// The cache-miss builder: read the kernel's `.spv` once, build its shader
/// module (dropped after pipeline creation — Vulkan copies the SPIR-V), its
/// shared storage-buffer `DescriptorSetLayout`, and its `ComputePipeline`
/// against the device-wide `vk::PipelineCache`. This is the object-creation
/// block lifted verbatim out of `launch_with_params_and_specialization`.
fn build_pipeline<'a>(
    ctx: &'a VulkanContext,
    kernel: Kernel,
    specialization_u32: &[(u32, u32)],
    push_bytes: u32,
    binding_count: usize,
) -> Result<(ComputePipeline<'a>, DescriptorSetLayout<'a>)> {
    let Some(path) = shader_path(kernel) else {
        return Err(KernelError::ShaderMissing(kernel.shader_name()));
    };
    let bytes =
        std::fs::read(&path).map_err(|_| KernelError::ShaderMissing(kernel.shader_name()))?;
    let shader = ShaderModule::from_spirv_bytes(ctx, &bytes)
        .map_err(|e| KernelError::Runtime(e.to_string()))?;
    let layout = DescriptorSetLayout::storage_buffers(ctx, binding_count)
        .map_err(|e| KernelError::Runtime(e.to_string()))?;
    let pipeline = ComputePipeline::create_with_push_constants_specialization_and_subgroup_size(
        ctx,
        &shader,
        &[&layout],
        push_bytes,
        specialization_u32,
        kernel.required_subgroup_size(specialization_u32),
    )
    .map_err(|e| KernelError::Runtime(e.to_string()))?;
    Ok((pipeline, layout))
}

/// Resolve a kernel's compiled `.spv` from the build-time shader directory.
/// Returns `None` (→ `ShaderMissing`) when the dir is unset or the file is
/// absent, matching the legacy launcher's behavior exactly.
fn shader_path(kernel: Kernel) -> Option<PathBuf> {
    let dir = option_env!("ARLE_VULKAN_SPV_DIR")?;
    let path = PathBuf::from(dir).join(format!("{}.spv", kernel.shader_name()));
    path.exists().then_some(path)
}

/// Record ONE compute dispatch from an already-built (cached) pipeline + a
/// freshly-bound descriptor set into the Step-1 [`CommandRecorder`]. No object
/// creation, no submit — purely bind + push + dispatch, the
/// `ggml_vk_dispatch_pipeline` body. The recorder must be `begin()`-open;
/// callers chain `record_dispatch` calls (with `barrier()`s between dependent
/// ops) and `submit_and_wait()` once per batch.
pub fn record_dispatch(
    recorder: &mut CommandRecorder<'_>,
    pipeline: &ComputePipeline<'_>,
    set: &DescriptorSet<'_>,
    push: &[u8],
    groups: [u32; 3],
) {
    recorder.dispatch(pipeline, set, push, groups);
}

/// One-shot cached launch: fetch (or build-once) the pipeline for this call,
/// bind the buffers, and record + submit a single `record_dispatch` through a
/// [`CommandRecorder`]. This is the cached replacement for the per-call object
/// graph the legacy launcher built; the pipeline build is served from `cache`,
/// so a repeated launch of the same signature rebuilds nothing.
#[allow(clippy::too_many_arguments)]
pub fn launch_cached<'a>(
    cache: &mut KernelCache<'a>,
    ctx: &'a VulkanContext,
    kernel: Kernel,
    buffers: &[&vulkan_sys::DeviceBuffer<'_>],
    dispatch: Dispatch,
    push_bytes: &[u8],
    specialization_u32: &[(u32, u32)],
) -> Result<()> {
    if dispatch.x == 0 || dispatch.y == 0 || dispatch.z == 0 {
        return Err(KernelError::InvalidDispatch);
    }
    if !push_bytes.len().is_multiple_of(4) {
        return Err(KernelError::InvalidPushConstants);
    }
    let push_len = u32::try_from(push_bytes.len())
        .map_err(|e| KernelError::Runtime(format!("push-constant size overflow: {e}")))?;
    let (pipeline, layout) = cache.get(ctx, kernel, specialization_u32, push_len, buffers.len())?;
    let set = DescriptorSet::storage_buffers(ctx, layout, buffers)
        .map_err(|e| KernelError::Runtime(e.to_string()))?;
    let mut recorder =
        CommandRecorder::new(ctx).map_err(|e| KernelError::Runtime(e.to_string()))?;
    recorder
        .begin()
        .map_err(|e| KernelError::Runtime(e.to_string()))?;
    record_dispatch(
        &mut recorder,
        pipeline,
        &set,
        push_bytes,
        [dispatch.x, dispatch.y, dispatch.z],
    );
    recorder
        .submit_and_wait()
        .map_err(|e| KernelError::Runtime(e.to_string()))
}
