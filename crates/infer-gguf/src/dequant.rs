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
