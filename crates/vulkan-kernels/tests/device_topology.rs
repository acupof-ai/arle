//! Print the device's shader-core topology and the occupancy arithmetic that
//! follows from it.
//!
//! Exists because launch geometry in this lane was chosen against "40 CUs",
//! which is not the unit anything is scheduled on. On RDNA the resident unit is
//! the WGP (2 CUs = 4 SIMD32) and the latency-hiding unit is waves-per-SIMD, so
//! a workgroup count that sounds ample against CUs can leave most SIMDs idle.
#![cfg(feature = "vulkan")]

use vulkan_sys::VulkanContext;

#[test]
fn report_shader_core_topology() {
    let Ok(ctx) = VulkanContext::create() else {
        eprintln!("no Vulkan device; skipping");
        return;
    };
    eprintln!("device: {}", ctx.device_name());
    eprintln!(
        "  maxComputeSharedMemorySize = {} B",
        ctx.max_compute_shared_memory_size()
    );
    let (sg, min_sg, max_sg) = ctx.subgroup_size();
    eprintln!("  subgroup = {sg} (min {min_sg}, max {max_sg})");

    let Some((cus, simd_per_cu, waves_per_simd, wave, vgprs)) = ctx.amd_shader_core() else {
        eprintln!("  (no VK_AMD_shader_core_properties)");
        return;
    };
    let wgps = cus / 2;
    let simds = cus * simd_per_cu;
    eprintln!("  compute units      = {cus}  -> {wgps} WGP (2 CU each)");
    eprintln!(
        "  SIMD per CU        = {simd_per_cu}  -> {simds} SIMD total, {} per WGP",
        simd_per_cu * 2
    );
    eprintln!(
        "  wavefronts per SIMD= {waves_per_simd}  -> {} resident waves at full occupancy",
        simds * waves_per_simd
    );
    eprintln!("  wavefront size     = {wave}");
    eprintln!("  VGPRs per SIMD     = {vgprs}");
    eprintln!(
        "  -> VGPR budget at full occupancy: {} per lane",
        vgprs / (waves_per_simd * wave / 32).max(1)
    );

    // What the lane actually launches, against that.
    for (name, groups, threads) in [
        ("gated-delta (1 wg / value head, 27B)", 48u32, 128u32),
        ("gated-delta (1 wg / value head, 122B)", 64, 128),
        ("coopmat mul_mm 'wide' tile", 0, 512),
    ] {
        if groups == 0 {
            eprintln!(
                "  [{name}] {threads} threads = {} wave{wave}",
                threads / wave
            );
            continue;
        }
        let waves = groups * threads.div_ceil(wave);
        eprintln!(
            "  [{name}] {groups} workgroups = {:.1} per WGP, {waves} waves = {:.2} per SIMD \
             (capacity {waves_per_simd})",
            f64::from(groups) / f64::from(wgps),
            f64::from(waves) / f64::from(simds),
        );
    }
}
