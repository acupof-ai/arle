//! On-device smoke test: proves the ARLE Vulkan execution path
//! (`vulkan-sys` context + buffers + `vulkan-kernels` launch + compiled SPIR-V)
//! actually runs a real compute kernel on the local GPU and returns the
//! mathematically correct result.
//!
//! Runs only with `--features vulkan` and a working Vulkan device + compiled
//! shaders; skips cleanly otherwise (CI / no-GPU boxes). On the AMD Radeon
//! 8060S (Strix Halo) this is the first end-to-end ARLE-on-Vulkan execution.
//!
//! Case: quantize 128 values all equal to 1.0 with `q8_1`. The block amax is
//! 1.0, so the scale `d = 1/127`, every quantized int8 = `round(1.0/d) = 127`,
//! and the per-32 sum `s = 32 * 127 * d = 32`. `block_q8_1_x4` is
//! `{ f16vec2 ds[4]; int32 qs[32] }` (16 B of scales, then 128 B of int8), so
//! the output must be 128 bytes of 0x7F preceded by 4×(d≈1/127, s≈32).
#![cfg(feature = "vulkan")]

use vulkan_kernels::{
    BLOCK_Q8_1_BYTES, q8_1_quantize, q8_1_quantize_dispatch, q8_1_quantize_params,
};
use vulkan_sys::{DeviceBuffer, VulkanContext};

/// Minimal IEEE-754 half → f32 decode (test-local; no dep on infer-gguf).
fn f16_to_f32(h: u16) -> f32 {
    let sign = if (h >> 15) & 1 == 1 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f32;
    let mag = if exp == 0 {
        frac * 2f32.powi(-24)
    } else if exp == 0x1f {
        if frac == 0.0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + frac / 1024.0) * 2f32.powi(exp - 15)
    };
    sign * mag
}

#[test]
fn q8_1_quantize_executes_on_device() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping device smoke test");
            return;
        }
    };
    eprintln!("ARLE Vulkan device smoke on: {}", ctx.device_name());

    let ne = 128usize;
    let input: Vec<f32> = vec![1.0; ne];
    let input_bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let num_x4 = ne.div_ceil(128);
    let out_len = num_x4 * 4 * BLOCK_Q8_1_BYTES; // 144 B per x4 block

    let mut buf_in = DeviceBuffer::alloc(&ctx, input_bytes.len()).expect("alloc input buffer");
    buf_in.copy_from_host(&input_bytes).expect("upload input");
    let mut buf_out = DeviceBuffer::alloc(&ctx, out_len).expect("alloc output buffer");
    buf_out
        .copy_from_host(&vec![0u8; out_len])
        .expect("zero output");

    let params = q8_1_quantize_params(ne as u32);
    let dispatch = q8_1_quantize_dispatch(ne as u32);
    q8_1_quantize(&ctx, &[&buf_in, &buf_out], dispatch, &params)
        .expect("q8_1_quantize dispatch on device");

    let mut got = vec![0u8; out_len];
    buf_out.copy_to_host(&mut got).expect("read back output");

    // ds[4]: 16 bytes of (d, s) f16 pairs; qs[32]: 128 bytes of int8 = 127.
    let d = f16_to_f32(u16::from_le_bytes([got[0], got[1]]));
    let s = f16_to_f32(u16::from_le_bytes([got[2], got[3]]));
    let ones = got[16..16 + 128].iter().filter(|&&b| b == 0x7F).count();
    eprintln!(
        "device result: d={d:.6} (1/127={:.6}) s={s} qs==127 count={ones}/128",
        1.0 / 127.0
    );

    assert_eq!(
        ones, 128,
        "every q8_1 value must quantize to 127 (got {ones})"
    );
    assert!(
        (d - 1.0 / 127.0).abs() < 1e-3,
        "scale d={d}, expected 1/127"
    );
    assert!((s - 32.0).abs() < 0.5, "sum s={s}, expected 32");
}
