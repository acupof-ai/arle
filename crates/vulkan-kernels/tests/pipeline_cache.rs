//! Build-once proof for the perf-parity Step 2 `KernelCache`.
//!
//! The legacy launcher rebuilt the whole pipeline object graph (`fs::read(.spv)`
//! → `ShaderModule` → `DescriptorSetLayout` → `ComputePipeline`) on EVERY
//! dispatch (~900/token). [`KernelCache`] must build each distinct
//! `(kernel, spec, push, bindings)` pipeline EXACTLY ONCE and serve all later
//! dispatches from the map. This test instruments the cache's build counter and
//! asserts it stays at 1 across 100 `get` + `record_dispatch` calls.
//!
//! Runs only with `--features vulkan` + a working device + compiled `.spv`;
//! skips cleanly otherwise.
#![cfg(feature = "vulkan")]

use vulkan_kernels::{
    Kernel, KernelCache, q8_1_quantize_dispatch, q8_1_quantize_params, record_dispatch,
};
use vulkan_sys::{CommandRecorder, DeviceBuffer, VulkanContext};

/// 100 cached dispatches of one kernel build the pipeline exactly once.
///
/// Uses `QuantizeQ8_1` (the q8_1 activation quantizer): a real, registered
/// kernel with a known 2-uint push block and a 2-buffer binding interface, so
/// the dispatch actually runs on the device. Each iteration:
///   * `cache.get(...)` — pipeline served from the map after the first build,
///   * a fresh `DescriptorSet` bound to the buffers (per-call data, never cached),
///   * `record_dispatch` into a `CommandRecorder` + one `submit_and_wait`.
///
/// After 100 iterations the instrumented `build_count()` must be 1 and exactly
/// one pipeline must be cached.
#[test]
fn kernel_pipeline_built_exactly_once_across_100_dispatches() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping pipeline-cache build-once test");
            return;
        }
    };
    eprintln!("KernelCache build-once proof on: {}", ctx.device_name());

    // q8_1 quantize: in = f32[ne], out = block_q8_1_x4 bytes; 2 buffers.
    const NE: usize = 256;
    let input: Vec<f32> = (0..NE).map(|i| (i as f32 * 0.013).sin()).collect();
    let input_bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let num_x4 = NE.div_ceil(128);
    let out_len = num_x4 * 4 * 36; // 36 B per q8_1 block

    let mut buf_in = DeviceBuffer::alloc(&ctx, input_bytes.len()).expect("alloc q8_1 input");
    buf_in
        .copy_from_host(&input_bytes)
        .expect("upload q8_1 input");
    let mut buf_out = DeviceBuffer::alloc(&ctx, out_len).expect("alloc q8_1 output");
    buf_out
        .copy_from_host(&vec![0u8; out_len])
        .expect("zero output");

    let params = q8_1_quantize_params(NE as u32);
    let push_bytes = params.to_le_bytes();
    let dispatch = q8_1_quantize_dispatch(NE as u32);
    let spec = Kernel::QuantizeQ8_1.specialization_u32();

    let mut cache = KernelCache::new();

    // First get builds the pipeline; if the .spv is missing, skip (no device
    // shader corpus on this box) rather than fail the suite.
    if let Err(e) = cache.get(&ctx, Kernel::QuantizeQ8_1, spec, push_bytes.len() as u32, 2) {
        eprintln!("q8_1 quantize .spv unavailable ({e}); skipping build-once test");
        return;
    }
    assert_eq!(cache.build_count(), 1, "first get must build exactly once");

    const ITERS: usize = 100;
    let mut last_out = vec![0u8; out_len];
    for iter in 0..ITERS {
        let (pipeline, layout) = cache
            .get(&ctx, Kernel::QuantizeQ8_1, spec, push_bytes.len() as u32, 2)
            .expect("cached get");
        let set = vulkan_sys::DescriptorSet::storage_buffers(&ctx, layout, &[&buf_in, &buf_out])
            .expect("bind descriptor set");
        let mut recorder = CommandRecorder::new(&ctx).expect("recorder");
        recorder.begin().expect("recorder begin");
        record_dispatch(
            &mut recorder,
            pipeline,
            &set,
            &push_bytes,
            [dispatch.x, dispatch.y, dispatch.z],
        );
        recorder.submit_and_wait().expect("submit");

        if iter == ITERS - 1 {
            buf_out.copy_to_host(&mut last_out).expect("read back");
        }
    }

    assert_eq!(
        cache.build_count(),
        1,
        "pipeline rebuilt {} times across {ITERS} dispatches — cache miss-path ran more than once",
        cache.build_count()
    );
    assert_eq!(cache.len(), 1, "exactly one pipeline cached");
    // The quantizer ran: the q8_1 output is non-zero for a non-zero input.
    assert!(
        last_out.iter().any(|&b| b != 0),
        "q8_1 output all-zero — cached dispatch did not execute"
    );

    eprintln!(
        "KernelCache build-once PASS: {ITERS} record_dispatch/submit calls, build_count={}, cached_pipelines={}",
        cache.build_count(),
        cache.len()
    );
}
