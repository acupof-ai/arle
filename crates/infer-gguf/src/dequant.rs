//! CPU dequantizers — faithful Rust ports of `dequantize_row_*` from
//! `vendor/llama.cpp/ggml-quants.c`, block layouts from
//! `vendor/llama.cpp/ggml-common.h` (QK_K=256 superblocks, l.89;
//! K_SCALE_SIZE=12, l.90). Always compiled; the HIP build dequantizes on
//! host for the DequantBf16/DequantF32 residency tiers. IQ2_XXS stays
//! device-only (mmvq path), no CPU port.
//!
//! Golden cross-check vs llama.cpp output = pending-remote (on-box H3);
//! local tests pin block-size arithmetic + hand-derived single blocks.

use anyhow::{Result, bail};

pub const QK_K: usize = 256;
pub const QK8_0: usize = 32;

/// ggml-common.h:241-246 — d half + 32×i8.
pub const BLOCK_Q8_0_BYTES: usize = 34;
/// ggml-common.h:288-299 — 16 scales + 64 qs + 2 half.
pub const BLOCK_Q2_K_BYTES: usize = 84;
/// ggml-common.h:317-328 — 2 half + 12 scales + 128 qs.
pub const BLOCK_Q4_K_BYTES: usize = 144;
/// ggml-common.h:334-346 — q4_K + 32 qh.
pub const BLOCK_Q5_K_BYTES: usize = 176;
/// ggml-common.h:352-358 — 128 ql + 64 qh + 16 scales + half.
pub const BLOCK_Q6_K_BYTES: usize = 210;

/// ggml-common.h:211 — QK_NVFP4.
pub const QK_NVFP4: usize = 64;
/// ggml-common.h:212 — QK_NVFP4_SUB, one UE4M3 scale per 16 elements.
pub const QK_NVFP4_SUB: usize = 16;
/// ggml-common.h:213-217 — 4 UE4M3 sub-block scales + 32 packed-nibble bytes.
pub const BLOCK_NVFP4_BYTES: usize = 36;

/// IEEE 754 binary16 → f32 (ggml GGML_FP16_TO_FP32 semantics).
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1F);
    let frac = u32::from(bits & 0x3FF);
    let word = match (exp, frac) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: renormalize so the leading fraction bit becomes the
            // implicit 1, lowering the exponent per shift.
            let mut e = 113u32;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((f & 0x3FF) << 13)
        }
        (0x1F, 0) => sign | 0x7F80_0000,
        (0x1F, _) => sign | 0x7FC0_0000,
        _ => sign | ((exp + 112) << 23) | (frac << 13),
    };
    f32::from_bits(word)
}

/// bfloat16 → f32: bf16 is the top 16 bits of the f32 word.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// OCP FP8 E4M3 → f32 — `S EEEE MMM`, exponent bias 7. This is the OFP8
/// v1.0 encoding (PyTorch `float8_e4m3fn`, what modelopt emits), NOT an IEEE
/// binary8: there are no infinities, the all-ones exponent is ordinary finite
/// range, and `S.1111.111` is the sole NaN. The consequences are worth stating
/// because getting them wrong is silent rather than loud — max normal is
/// 2^8 x 1.75 = 448, min normal 2^-6, and subnormals reach 2^-9.
pub fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    let sign = u32::from(bits >> 7) << 31;
    let exp = u32::from((bits >> 3) & 0xF);
    let mant = u32::from(bits & 0x7);
    let word = match (exp, mant) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal, value mant x 2^-9: renormalize so the leading mantissa
            // bit becomes f32's implicit 1, lowering the exponent per shift from
            // the 2^-6 an already-normalized mantissa would carry (127-6=121).
            let mut e = 121u32;
            let mut m = mant;
            while m & 0x8 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x7) << 20)
        }
        // The one NaN code. Every other exp==0xF encoding is a finite normal —
        // exactly where an IEEE-shaped decoder would wrongly produce inf/NaN.
        (0xF, 0x7) => sign | 0x7FC0_0000,
        // Rebias 7 -> 127 and left-justify the 3 mantissa bits in f32's 23.
        _ => sign | ((exp + 120) << 23) | (mant << 20),
    };
    f32::from_bits(word)
}

fn f16_at(block: &[u8], offset: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([block[offset], block[offset + 1]]))
}

fn check_blocks(
    data_len: usize,
    k: usize,
    qk: usize,
    block_bytes: usize,
    what: &str,
) -> Result<usize> {
    if k == 0 || !k.is_multiple_of(qk) {
        bail!("{what}: k={k} not a multiple of {qk}");
    }
    let nb = k / qk;
    if data_len != nb * block_bytes {
        bail!("{what}: data len {data_len} != {nb} blocks x {block_bytes}");
    }
    Ok(nb)
}

pub fn dequantize_row_f16(data: &[u8], k: usize) -> Result<Vec<f32>> {
    if data.len() != k * 2 {
        bail!("f16 row: data len {} != {k}*2", data.len());
    }
    Ok((0..k).map(|i| f16_at(data, i * 2)).collect())
}

pub fn dequantize_row_bf16(data: &[u8], k: usize) -> Result<Vec<f32>> {
    if data.len() != k * 2 {
        bail!("bf16 row: data len {} != {k}*2", data.len());
    }
    Ok((0..k)
        .map(|i| bf16_to_f32(u16::from_le_bytes([data[i * 2], data[i * 2 + 1]])))
        .collect())
}

pub fn dequantize_row_f8_e4m3(data: &[u8], k: usize) -> Result<Vec<f32>> {
    if data.len() != k {
        bail!("f8_e4m3 row: data len {} != {k}", data.len());
    }
    Ok(data.iter().map(|&b| fp8_e4m3_to_f32(b)).collect())
}

pub fn dequantize_row_f32(data: &[u8], k: usize) -> Result<Vec<f32>> {
    if data.len() != k * 4 {
        bail!("f32 row: data len {} != {k}*4", data.len());
    }
    Ok((0..k)
        .map(|i| f32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap()))
        .collect())
}

/// ggml-quants.c:495-509.
pub fn dequantize_row_q8_0(data: &[u8], k: usize) -> Result<Vec<f32>> {
    let nb = check_blocks(data.len(), k, QK8_0, BLOCK_Q8_0_BYTES, "q8_0")?;
    let y = (0..nb)
        .flat_map(|i| {
            let b = &data[i * BLOCK_Q8_0_BYTES..];
            let d = f16_at(b, 0);
            (0..QK8_0).map(move |j| f32::from(b[2 + j] as i8) * d)
        })
        .collect();
    Ok(y)
}

/// ggml-quants.c:822-829 — exact port incl. the j>=4 bit-splice.
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// ggml-quants.c:903-933. Block layout ggml-common.h:288-299
/// (scales[16] qs[64] d dmin).
pub fn dequantize_row_q2_k(data: &[u8], k: usize) -> Result<Vec<f32>> {
    let nb = check_blocks(data.len(), k, QK_K, BLOCK_Q2_K_BYTES, "q2_K")?;
    let mut y = Vec::with_capacity(k);
    for i in 0..nb {
        let b = &data[i * BLOCK_Q2_K_BYTES..];
        let scales = &b[0..16];
        let d = f16_at(b, 80);
        let min = f16_at(b, 82);
        let mut is = 0usize;
        let mut q = &b[16..80];
        let mut n = 0;
        while n < QK_K {
            let mut shift = 0;
            for _ in 0..4 {
                let sc = scales[is];
                is += 1;
                let dl = d * f32::from(sc & 0xF);
                let ml = min * f32::from(sc >> 4);
                for &qv in &q[..16] {
                    y.push(dl * f32::from((qv >> shift) & 3) - ml);
                }
                let sc = scales[is];
                is += 1;
                let dl = d * f32::from(sc & 0xF);
                let ml = min * f32::from(sc >> 4);
                for l in 0..16 {
                    y.push(dl * f32::from((q[l + 16] >> shift) & 3) - ml);
                }
                shift += 2;
            }
            q = &q[32..];
            n += 128;
        }
    }
    Ok(y)
}

/// ggml-quants.c:1471-1493. Block layout ggml-common.h:317-328
/// (d dmin scales[12] qs[128]).
pub fn dequantize_row_q4_k(data: &[u8], k: usize) -> Result<Vec<f32>> {
    let nb = check_blocks(data.len(), k, QK_K, BLOCK_Q4_K_BYTES, "q4_K")?;
    let mut y = Vec::with_capacity(k);
    for i in 0..nb {
        let b = &data[i * BLOCK_Q4_K_BYTES..];
        let d = f16_at(b, 0);
        let min = f16_at(b, 2);
        let scales = &b[4..16];
        let mut q = &b[16..144];
        let mut is = 0usize;
        let mut j = 0;
        while j < QK_K {
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * f32::from(sc);
            let m1 = min * f32::from(m);
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let d2 = d * f32::from(sc);
            let m2 = min * f32::from(m);
            for &qv in &q[..32] {
                y.push(d1 * f32::from(qv & 0xF) - m1);
            }
            for &qv in &q[..32] {
                y.push(d2 * f32::from(qv >> 4) - m2);
            }
            q = &q[32..];
            is += 2;
            j += 64;
        }
    }
    Ok(y)
}

/// ggml-quants.c:1673-1698. Block layout ggml-common.h:334-346
/// (d dmin scales[12] qh[32] qs[128]).
pub fn dequantize_row_q5_k(data: &[u8], k: usize) -> Result<Vec<f32>> {
    let nb = check_blocks(data.len(), k, QK_K, BLOCK_Q5_K_BYTES, "q5_K")?;
    let mut y = Vec::with_capacity(k);
    for i in 0..nb {
        let b = &data[i * BLOCK_Q5_K_BYTES..];
        let d = f16_at(b, 0);
        let min = f16_at(b, 2);
        let scales = &b[4..16];
        let qh = &b[16..48];
        let mut ql = &b[48..176];
        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        let mut j = 0;
        while j < QK_K {
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * f32::from(sc);
            let m1 = min * f32::from(m);
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let d2 = d * f32::from(sc);
            let m2 = min * f32::from(m);
            for l in 0..32 {
                let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
                y.push(d1 * f32::from((ql[l] & 0xF) + hi) - m1);
            }
            for l in 0..32 {
                let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
                y.push(d2 * f32::from((ql[l] >> 4) + hi) - m2);
            }
            ql = &ql[32..];
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
            j += 64;
        }
    }
    Ok(y)
}

/// ggml-quants.c:1881-1910. Block layout ggml-common.h:352-358
/// (ql[128] qh[64] scales[16] d).
pub fn dequantize_row_q6_k(data: &[u8], k: usize) -> Result<Vec<f32>> {
    let nb = check_blocks(data.len(), k, QK_K, BLOCK_Q6_K_BYTES, "q6_K")?;
    let mut y = vec![0.0f32; k];
    for i in 0..nb {
        let b = &data[i * BLOCK_Q6_K_BYTES..];
        let d = f16_at(b, 208);
        let mut ql = &b[0..128];
        let mut qh = &b[128..192];
        let mut sc = &b[192..208];
        let base = i * QK_K;
        let mut n = 0;
        while n < QK_K {
            for l in 0..32 {
                let is = l / 16;
                let q1 = i16::from((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
                let q2 = i16::from((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                let q3 = i16::from((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                let q4 = i16::from((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                let o = base + n + l;
                y[o] = d * f32::from(sc[is] as i8) * f32::from(q1);
                y[o + 32] = d * f32::from(sc[is + 2] as i8) * f32::from(q2);
                y[o + 64] = d * f32::from(sc[is + 4] as i8) * f32::from(q3);
                y[o + 96] = d * f32::from(sc[is + 6] as i8) * f32::from(q4);
            }
            n += 128;
            ql = &ql[64..];
            qh = &qh[32..];
            sc = &sc[8..];
        }
    }
    Ok(y)
}

/// E2M1 nibble → f32. Bit layout `s ee m` (OCP Microscaling Formats v1.0, the
/// spec ggml-common.h:1115 cites): bit 3 sign, bits 2..1 exponent with bias 1,
/// bit 0 mantissa. `exp == 0` is the subnormal
/// rung (no implicit leading 1, so the value is `m/2`); otherwise the value is
/// `2^(exp-1) * (1 + m/2)`. That enumerates {0, .5, 1, 1.5, 2, 3, 4, 6}, which
/// is ggml's `kvalues_mxfp4` table (ggml-common.h:1116) halved — that table
/// stores the magnitudes *doubled* so it can be int8.
fn e2m1_to_f32(nibble: u8) -> f32 {
    let exp = u32::from((nibble >> 1) & 3);
    let man = f32::from(nibble & 1);
    let mag = if exp == 0 {
        man * 0.5
    } else {
        // exp is 1..=3 here, so the shift and the product stay exact in f32.
        (1.0 + man * 0.5) * f32::from(1u16 << (exp - 1))
    };
    // Nibble 8 is a legal -0, but ggml's int8 table stores it as +0; keep the
    // sign off zero so a row matches the reference bit for bit.
    if nibble & 8 != 0 && mag != 0.0 {
        -mag
    } else {
        mag
    }
}

/// UE4M3 sub-block scale → f32. Bit layout `x eeee mmm`: bit 7 is an unused
/// sign (vulkan-shaders/types.glsl:1775 — ggml never sets it, and the shader's
/// scale LUT only has 128 entries), bits 6..3 exponent with bias 7, bits 2..0
/// mantissa. `exp == 0` is subnormal, `(m/8) * 2^(1-7) = m * 2^-9`; otherwise
/// `(1 + m/8) * 2^(exp-7)`. 0x7F is E4M3's NaN and ggml decodes it as zero
/// (ggml-cuda/common.cuh:843, vulkan-shaders/types.glsl:1779-1780).
///
/// This returns the *raw* UE4M3 value. ggml's own helpers return half of it
/// (`raw / 2` in ggml-cuda/common.cuh:854, `* 0.5` in
/// vulkan-shaders/dequant_funcs.glsl:502) because they pair the scale with the
/// doubled `kvalues_mxfp4` table; we fold the halving into `e2m1_to_f32`
/// instead, so the product is identical.
///
/// Deliberately not `fp8_e4m3_to_f32` above: that one is the signed OFP8
/// element type and propagates NaN, whereas a UE4M3 *scale* has no sign bit to
/// spend and ggml flushes its NaN code to zero. Sharing the body would mean
/// one of the two callers silently getting the other's NaN policy.
fn ue4m3_to_f32(byte: u8) -> f32 {
    let u = u32::from(byte & 0x7F);
    if u == 0 || u == 0x7F {
        return 0.0;
    }
    let exp = u >> 3;
    let man = u & 7;
    if exp == 0 {
        // 2^-9 is exact, so this is a plain scale of the 3 mantissa bits.
        (man as f32) / 512.0
    } else {
        // f32's exponent bias is 127 against UE4M3's 7, so the biased field is
        // exp - 7 + 127 = exp + 120 (max 135, never overflows). The 3 mantissa
        // bits are the top 3 of f32's 23.
        f32::from_bits(((exp + 120) << 23) | (man << 20))
    }
}

/// ggml-quants.c:531-554. Block layout ggml-common.h:211-217
/// (d[4] UE4M3 sub-block scales, then qs[32] of packed E2M1 nibbles).
///
/// Nibble order inside a 16-element sub-block follows q4_0's split-halves
/// convention, not consecutive pairs: the low nibbles of the 8 bytes are
/// elements 0..7 and the high nibbles are elements 8..15. Cross-checked
/// against ggml-cuda/convert.cu:636-646 and the Vulkan consumer in
/// vulkan-shaders/dequant_funcs.glsl:493-503, which both index the same way.
pub fn dequantize_row_nvfp4(data: &[u8], k: usize) -> Result<Vec<f32>> {
    let nb = check_blocks(data.len(), k, QK_NVFP4, BLOCK_NVFP4_BYTES, "nvfp4")?;
    let n_sub = QK_NVFP4 / QK_NVFP4_SUB;
    let half = QK_NVFP4_SUB / 2;
    let mut y = vec![0.0f32; k];
    for i in 0..nb {
        let (d, qs) = data[i * BLOCK_NVFP4_BYTES..].split_at(n_sub);
        for s in 0..n_sub {
            let scale = ue4m3_to_f32(d[s]);
            let base = i * QK_NVFP4 + s * QK_NVFP4_SUB;
            for j in 0..half {
                let q = qs[s * half + j];
                y[base + j] = e2m1_to_f32(q & 0xF) * scale;
                y[base + half + j] = e2m1_to_f32(q >> 4) * scale;
            }
        }
    }
    Ok(y)
}

#[cfg(test)]
mod fp8_tests {
    use super::{dequantize_row_f8_e4m3, fp8_e4m3_to_f32};
    use crate::gguf::GgmlType;

    /// Textbook OFP8 E4M3 definition evaluated in float arithmetic, i.e. a
    /// second derivation that shares no code with the bit-surgery decoder.
    /// `None` = the single NaN code `S.1111.111`.
    fn reference(bits: u8) -> Option<f32> {
        let sign = if bits & 0x80 == 0 { 1.0f32 } else { -1.0 };
        let e = i32::from((bits >> 3) & 0xF);
        let m = f32::from(bits & 0x7);
        if e == 0xF && bits & 0x7 == 0x7 {
            return None;
        }
        Some(if e == 0 {
            // Subnormals carry the min-normal exponent with no implicit 1.
            sign * (m / 8.0) * 2f32.powi(-6)
        } else {
            sign * (1.0 + m / 8.0) * 2f32.powi(e - 7)
        })
    }

    /// Hand-derived, exactly representable in f32, so compared bit-for-bit
    /// (which also pins the sign of zero).
    #[test]
    fn hand_computed_codes() {
        // (code, expected) with the derivation, bias 7:
        let cases: [(u8, f32); 12] = [
            (0x00, 0.0),         // +0
            (0x80, -0.0),        // -0
            (0x01, 0.001953125), // min subnormal, 1 x 2^-6/8 = 2^-9
            (0x05, 0.009765625), // subnormal 5 x 2^-9
            (0x07, 0.013671875), // largest subnormal, 7 x 2^-9
            (0x08, 0.015625),    // smallest normal, e=1: 2^(1-7) = 2^-6
            (0x38, 1.0),         // e=7 m=0: 2^0 x 1.0
            (0xB8, -1.0),        // sign bit on
            (0x78, 256.0),       // e=15 m=0: 2^8 x 1.0, +inf in IEEE
            (0x7D, 416.0),       // e=15 m=5: 2^8 x 1.625
            (0x7E, 448.0),       // max normal, e=15 m=6: 2^8 x 1.75
            (0xFE, -448.0),      // largest magnitude the format encodes
        ];
        for (bits, want) in cases {
            let got = fp8_e4m3_to_f32(bits);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "0x{bits:02X}: got {got}, want {want}"
            );
        }
        // The only NaN encodings; nothing else in the format is non-finite.
        assert!(fp8_e4m3_to_f32(0x7F).is_nan());
        assert!(fp8_e4m3_to_f32(0xFF).is_nan());
    }

    /// Whole domain against the independent formula — a wrong bias or a
    /// mis-shifted subnormal is off by a factor of two, not obviously broken.
    #[test]
    fn matches_reference_over_all_256_codes() {
        for bits in 0..=u8::MAX {
            let got = fp8_e4m3_to_f32(bits);
            match reference(bits) {
                None => assert!(got.is_nan(), "0x{bits:02X} should be NaN, got {got}"),
                Some(want) => assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "0x{bits:02X}: got {got}, want {want}"
                ),
            }
        }
        // Exactly two NaN codes, and 448 is the largest finite magnitude.
        let finite = (0..=u8::MAX).filter(|&b| fp8_e4m3_to_f32(b).is_finite());
        assert_eq!(finite.clone().count(), 254);
        assert_eq!(
            finite.fold(0.0f32, |acc, b| acc.max(fp8_e4m3_to_f32(b).abs())),
            448.0
        );
    }

    #[test]
    fn row_is_one_byte_per_element() {
        let row: Vec<u8> = vec![0x38, 0xB8, 0x00, 0x7E];
        assert_eq!(
            dequantize_row_f8_e4m3(&row, 4).unwrap(),
            vec![1.0, -1.0, 0.0, 448.0]
        );
        assert!(dequantize_row_f8_e4m3(&row, 5).is_err());
    }

    /// 1 byte per element and no blocking, so the 160-wide PLE row is 160 B.
    #[test]
    fn ggml_type_layout() {
        assert_eq!(GgmlType::from_id(42).unwrap(), GgmlType::F8E4M3);
        assert_eq!(GgmlType::F8E4M3.block_size(), 1);
        assert_eq!(GgmlType::F8E4M3.type_size(), Some(1));
        assert_eq!(GgmlType::F8E4M3.row_bytes(160), Some(160));
    }
}

#[cfg(test)]
mod nvfp4_tests {
    use super::{BLOCK_NVFP4_BYTES, QK_NVFP4, dequantize_row_nvfp4, e2m1_to_f32, ue4m3_to_f32};

    /// ggml-common.h:1116 — `kvalues_mxfp4`, the E2M1 magnitudes stored
    /// *doubled* so the table can be int8. Transcribed here so the decode is
    /// checked against the vendored constants and not against itself.
    const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

    #[test]
    fn e2m1_matches_the_vendored_doubled_table() {
        for nibble in 0u8..16 {
            assert_eq!(
                e2m1_to_f32(nibble),
                f32::from(KVALUES_MXFP4[nibble as usize]) * 0.5,
                "e2m1 nibble {nibble:#x}"
            );
        }
        // ggml stores -0 (nibble 8) as +0; a stray sign here would flip the
        // sign of any zero weight fed to a downstream `copysign`/`signum`.
        assert!(e2m1_to_f32(8).is_sign_positive());
    }

    #[test]
    fn ue4m3_decodes_hand_derived_scales() {
        // Each pair below is (byte, value worked out from `x eeee mmm`,
        // bias 7): subnormals are m/512, normals are 2^(e-7)*(1+m/8).
        for (byte, want) in [
            (0x00, 0.0),         // zero
            (0x01, 1.0 / 512.0), // min subnormal, 2^-9
            (0x07, 7.0 / 512.0), // max subnormal
            (0x08, 0.015625),    // min normal, 2^-6
            (0x38, 1.0),         // e=7 m=0 -> 2^0
            (0x3C, 1.5),         // e=7 m=4 -> 1 + 4/8
            (0x77, 240.0),       // e=14 m=7 -> 2^7 * 1.875
            (0x7E, 448.0),       // e=15 m=6 -> 2^8 * 1.75, max finite
            (0x7F, 0.0),         // E4M3's NaN code, flushed to zero
            (0xB8, 1.0),         // bit 7 is unused; must be masked off
        ] {
            assert_eq!(ue4m3_to_f32(byte), want, "ue4m3 byte {byte:#04x}");
        }
    }

    /// One hand-built block. The four sub-block scales are 1.0, 1.5, 2^-7 and
    /// the NaN code (zero); every sub-block carries the same nibbles, chosen to
    /// walk all sixteen E2M1 codes: low nibbles 0..7 in bytes 0..7 (elements
    /// 0..7) and high nibbles 8..15 in the same bytes (elements 8..15). Every
    /// expected value below is a product of dyadic rationals, so it is exact in
    /// f32 and can be compared with `==`.
    #[test]
    fn dequantizes_a_hand_built_block() {
        let mut block = [0u8; BLOCK_NVFP4_BYTES];
        block[0..4].copy_from_slice(&[0x38, 0x3C, 0x04, 0x7F]);
        for s in 0..4 {
            block[4 + s * 8..4 + s * 8 + 8]
                .copy_from_slice(&[0x80, 0x91, 0xA2, 0xB3, 0xC4, 0xD5, 0xE6, 0xF7]);
        }

        #[rustfmt::skip]
        let want: [f32; QK_NVFP4] = [
            // sub 0, scale 1.0
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
            // sub 1, scale 1.5
            0.0, 0.75, 1.5, 2.25, 3.0, 4.5, 6.0, 9.0,
            0.0, -0.75, -1.5, -2.25, -3.0, -4.5, -6.0, -9.0,
            // sub 2, scale 4/512 = 2^-7
            0.0, 0.003_906_25, 0.007_812_5, 0.011_718_75,
            0.015_625, 0.023_437_5, 0.031_25, 0.046_875,
            0.0, -0.003_906_25, -0.007_812_5, -0.011_718_75,
            -0.015_625, -0.023_437_5, -0.031_25, -0.046_875,
            // sub 3, NaN scale -> zero, whatever the nibbles say
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        let got = dequantize_row_nvfp4(&block, QK_NVFP4).expect("dequant nvfp4");
        assert_eq!(got, want.to_vec());
    }

    /// Two blocks back to back, to pin that the block stride is 36 bytes and
    /// not 32 (qs alone) or 40 (a padded struct).
    #[test]
    fn walks_blocks_at_a_36_byte_stride() {
        let mut data = [0u8; 2 * BLOCK_NVFP4_BYTES];
        // Second block only: scale 1.0 on sub 0, nibble 7 (= 6.0) at element 3.
        data[BLOCK_NVFP4_BYTES] = 0x38;
        data[BLOCK_NVFP4_BYTES + 4 + 3] = 0x07;

        let got = dequantize_row_nvfp4(&data, 2 * QK_NVFP4).expect("dequant nvfp4");
        assert_eq!(got[QK_NVFP4 + 3], 6.0);
        assert!(
            got.iter()
                .enumerate()
                .all(|(i, &v)| v == 0.0 || i == QK_NVFP4 + 3)
        );
    }

    #[test]
    fn rejects_lengths_that_are_not_whole_blocks() {
        assert!(dequantize_row_nvfp4(&[0u8; BLOCK_NVFP4_BYTES], 32).is_err());
        assert!(dequantize_row_nvfp4(&[0u8; BLOCK_NVFP4_BYTES - 1], QK_NVFP4).is_err());
    }
}
