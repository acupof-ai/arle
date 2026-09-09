//! CPU f32 reference for NVFP4: dequantize each weight in f32 (e2m1 * e4m3
//! group scale * global scale) and dot with the bf16 input widened to f32,
//! accumulating in f32. The anchor both GPU arms are measured against — never
//! a second bf16 path.

use half::bf16;

pub const GROUP: usize = 16;

/// FP8 E4M3FN → f32 (host twin of `dsv4_decode_fp8_e4m3`; 0x7f/0xff is the
/// format's NaN and is clamped to +/-448, matching the device decoder).
pub fn e4m3_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exp = i32::from((bits >> 3) & 0x0f);
    let mant = f32::from(bits & 0x07);
    if bits & 0x7f == 0x7f {
        return sign * 448.0;
    }
    if exp == 0 {
        return sign * (mant / 8.0) * 2.0_f32.powi(-6);
    }
    sign * (1.0 + mant / 8.0) * 2.0_f32.powi(exp - 7)
}

/// FP4 E2M1 → f32, matching the kernel's LUT (dequant.h).
pub fn e2m1_to_f32(nibble: u8) -> f32 {
    const LUT: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    LUT[(nibble & 0x0f) as usize]
}

pub fn cpu_ref_fp4(
    packed: &[u8],
    scales: &[u8],
    global: f32,
    x: &[bf16],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let scale_cols = k / GROUP;
    let mut out = Vec::with_capacity(m * n);
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f32;
            for ki in 0..k {
                let byte = packed[ni * (k / 2) + ki / 2];
                let nibble = if ki % 2 == 0 { byte & 0xF } else { byte >> 4 };
                let s = e4m3_to_f32(scales[ni * scale_cols + ki / GROUP]);
                acc += x[mi * k + ki].to_f32() * e2m1_to_f32(nibble) * s * global;
            }
            out.push(acc);
        }
    }
    out
}
