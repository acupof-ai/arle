//! Correctness tests for the native Q4_K packed GPU path.
//!
//! Build a synthetic Q4_K tensor on the CPU, upload it through
//! `DeviceMatrix::from_quantized_q4k`, run `ops::gemv` against a known input,
//! and compare the result to a CPU reference that uses the same superblock
//! dequant math as `gguf::dequant_q4_k` (the well-tested path that backs all
//! existing `read_tensor_bf16` calls).

#![cfg(feature = "cuda")]

use half::bf16;
use infer::backend::cuda::tensor::{DeviceContext, DeviceMatrix, DeviceVec};
use infer::ops;

/// Assemble a single Q4_K superblock (144 bytes) with chosen scales and nibbles.
/// Layout: d(f16) | dmin(f16) | scales_packed(12) | qs(128).
///
/// Element → qs byte layout follows llama.cpp: each outer iter holds 32 qs
/// bytes; the first 32 elements of the iter are the LOW nibbles of those
/// bytes, the next 32 are the HIGH nibbles.
fn make_superblock(
    d: f32,
    dmin: f32,
    sub_scales: &[u8; 8],
    sub_mins: &[u8; 8],
    nibbles: &[u8; 256],
) -> [u8; 144] {
    let mut out = [0u8; 144];
    out[0..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
    out[2..4].copy_from_slice(&half::f16::from_f32(dmin).to_le_bytes());

    // scales_packed (same as Q5_K's get_scale_min_k4).
    for i in 0..4 {
        out[4 + i] = (sub_scales[i] & 0x3F) | ((sub_scales[4 + i] & 0x03) << 6);
        out[8 + i] = (sub_mins[i] & 0x3F) | ((sub_mins[4 + i] & 0x03) << 6);
        out[12 + i] = ((sub_scales[4 + i] >> 2) & 0x0F) | (((sub_mins[4 + i] >> 2) & 0x0F) << 4);
    }

    // qs: 4 outer iterations of 32 bytes each. For each iter, byte l carries
    // element (2*iter+0)*32 + l in its LOW nibble and element (2*iter+1)*32 + l
    // in its HIGH nibble.
    for iter in 0..4 {
        for l in 0..32 {
            let lo = nibbles[(iter * 2) * 32 + l] & 0x0F;
            let hi = nibbles[(iter * 2 + 1) * 32 + l] & 0x0F;
            out[16 + iter * 32 + l] = lo | (hi << 4);
        }
    }
    out
}

/// Reference CPU dequant of one superblock — mirrors llama.cpp
/// `dequantize_row_q4_K` exactly.
fn dequant_superblock_cpu(sb: &[u8; 144]) -> [f32; 256] {
    let d = half::f16::from_le_bytes([sb[0], sb[1]]).to_f32();
    let dmin = half::f16::from_le_bytes([sb[2], sb[3]]).to_f32();
    let scales_raw = &sb[4..16];
    let qs = &sb[16..144];

    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for i in 0..4 {
        sc[i] = scales_raw[i] & 0x3F;
        mn[i] = scales_raw[i + 4] & 0x3F;
    }
    for i in 0..4 {
        sc[4 + i] = (scales_raw[i] >> 6) | ((scales_raw[8 + i] & 0x0F) << 2);
        mn[4 + i] = (scales_raw[i + 4] >> 6) | ((scales_raw[8 + i] >> 4) << 2);
    }

    let mut out = [0f32; 256];
    for iter in 0..4 {
        let j_lo = iter * 2;
        let j_hi = j_lo + 1;
        let d1 = d * sc[j_lo] as f32;
        let m1 = dmin * mn[j_lo] as f32;
        let d2 = d * sc[j_hi] as f32;
        let m2 = dmin * mn[j_hi] as f32;
        for l in 0..32 {
            let byte = qs[iter * 32 + l];
            let lo = (byte & 0x0F) as f32;
            let hi = (byte >> 4) as f32;
            out[j_lo * 32 + l] = lo * d1 - m1;
            out[j_hi * 32 + l] = hi * d2 - m2;
        }
    }
    out
}

/// Dequantize an entire packed Q4_K tensor and return the per-row flat buffer.
fn dequant_tensor_cpu(rows: usize, cols: usize, packed: &[u8]) -> Vec<f32> {
    assert_eq!(packed.len(), rows * cols * 9 / 16);
    let sb_per_row = cols / 256;
    let row_bytes = sb_per_row * 144;
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for sb in 0..sb_per_row {
            let base = r * row_bytes + sb * 144;
            let sb_bytes: &[u8; 144] = packed[base..base + 144].try_into().unwrap();
            let deq = dequant_superblock_cpu(sb_bytes);
            for i in 0..256 {
                out[r * cols + sb * 256 + i] = deq[i];
            }
        }
    }
    out
}

/// Round-trip the superblock scale-pack encoding so we don't trust our own writer.
#[test]
fn q4k_superblock_encode_decode_roundtrip() {
    let sub_scales = [5u8, 11, 23, 1, 40, 2, 31, 63];
    let sub_mins = [3u8, 7, 15, 0, 19, 8, 50, 44];
    let mut nibbles = [0u8; 256];
    for i in 0..256 {
        nibbles[i] = (i as u8) & 0x0F;
    }
    let sb = make_superblock(0.0078125, 0.015625, &sub_scales, &sub_mins, &nibbles);

    // Decode the packed 12-byte scales header back and check against the originals.
    let scales_raw = &sb[4..16];
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for i in 0..4 {
        sc[i] = scales_raw[i] & 0x3F;
        mn[i] = scales_raw[i + 4] & 0x3F;
    }
    for i in 0..4 {
        sc[4 + i] = (scales_raw[i] >> 6) | ((scales_raw[8 + i] & 0x0F) << 2);
        mn[4 + i] = (scales_raw[i + 4] >> 6) | ((scales_raw[8 + i] >> 4) << 2);
    }
    assert_eq!(sc, sub_scales);
    assert_eq!(mn, sub_mins);

    // And a known dequant: element 0 = 0*sc[0]*d - dmin*mn[0] = -0.015625*3 = -0.046875.
    let ref_f32 = dequant_superblock_cpu(&sb);
    assert!((ref_f32[0] - (-0.046875)).abs() < 1e-6);
    // Element 1 = 1*sc[0]*d - dmin*mn[0] = 0.0078125*5 - 0.046875 = -0.0078125.
    assert!((ref_f32[1] - (-0.0078125)).abs() < 1e-6);
}

#[test]
fn q4k_gemv_matches_cpu_identity_input() {
    // With x = all 1.0, y[r] = sum of dequantized row r elements. This is the
    // simplest possible check: it isolates the kernel's row striding + superblock
    // walk from any activation-side effects.
    const ROWS: usize = 8;
    const COLS: usize = 512; // 2 superblocks per row

    let sub_scales = [2u8, 5, 7, 11, 13, 17, 19, 23];
    let sub_mins = [1u8, 3, 5, 7, 9, 11, 13, 15];

    let mut packed = Vec::with_capacity(ROWS * COLS * 9 / 16);
    for r in 0..ROWS {
        for _ in 0..(COLS / 256) {
            let mut nibbles = [0u8; 256];
            for i in 0..256 {
                nibbles[i] = ((i + r) as u8) & 0x0F;
            }
            let d = 0.0078125 * (1 + r as i32) as f32;
            let dmin = 0.015625 * (1 + r as i32) as f32;
            let sb = make_superblock(d, dmin, &sub_scales, &sub_mins, &nibbles);
            packed.extend_from_slice(&sb);
        }
    }

    let cpu_weights = dequant_tensor_cpu(ROWS, COLS, &packed);
    let x_f32 = vec![1.0f32; COLS];
    let cpu_y: Vec<f32> = (0..ROWS)
        .map(|r| (0..COLS).map(|c| cpu_weights[r * COLS + c]).sum())
        .collect();

    let x_bf16: Vec<bf16> = x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let ctx = DeviceContext::new().expect("device context");
    let mat =
        DeviceMatrix::from_quantized_q4k(&ctx, &packed, ROWS, COLS).expect("from_quantized_q4k");
    let x_dev = DeviceVec::from_host(&ctx, &x_bf16).expect("x upload");
    let mut y_dev = DeviceVec::zeros(&ctx, ROWS).expect("y alloc");
    ops::gemv(&ctx, &mat, &x_dev, &mut y_dev).expect("gemv");

    let mut y_bf16 = vec![bf16::ZERO; ROWS];
    y_dev
        .copy_region_to_host(&ctx, 0, ROWS, &mut y_bf16)
        .expect("download");

    for r in 0..ROWS {
        let g = y_bf16[r].to_f32();
        let c = cpu_y[r];
        let tol = (c.abs() * 5e-2).max(5e-2);
        assert!((g - c).abs() < tol, "row {r}: gpu={g} cpu={c} (tol={tol})");
    }
}

// ── Q3_K ──────────────────────────────────────────────────────────────────

/// Build a single Q3_K superblock (110 bytes) with chosen d, scales, and
/// 256 3-bit values in `vals` (range 0..=7).
///
/// Layout: hmask(32) | qs(64, 2-bit packed) | scales(12) | d:f16(2).
///
/// Scales are 6-bit unsigned (0..63) mapped to signed range -32..31.
/// Encoder is the inverse of the decode in gguf.rs (post-fix).
fn make_q3k_superblock(d: f32, scales: &[i8; 16], vals: &[u8; 256]) -> [u8; 110] {
    let mut out = [0u8; 110];

    // hmask: high bit per element, 8 per byte.
    for j in 0..256 {
        let hbit = (vals[j] >> 2) & 1;
        out[j / 8] |= hbit << (j % 8);
    }
    // qs: low 2 bits per element, 4 per byte.
    for j in 0..256 {
        let q2 = vals[j] & 0x3;
        out[32 + j / 4] |= q2 << ((j % 4) * 2);
    }

    // scales_raw: 12 bytes encoding 16 6-bit unsigned scales (each = signed + 32).
    //   bytes 0..8 : low 4 bits of u6[i] (low nibble = i, high nibble = i+8) for i<8
    //   bytes 8..12: high 2 bits of u6[i+0,+4,+8,+12] packed in 2-bit fields
    let u6: [u8; 16] = std::array::from_fn(|i| ((scales[i] as i32) + 32) as u8);
    for i in 0..8 {
        out[96 + i] = (u6[i] & 0x0F) | ((u6[i + 8] & 0x0F) << 4);
    }
    for i in 0..4 {
        let s0 = (u6[i] >> 4) & 0x03;
        let s4 = (u6[i + 4] >> 4) & 0x03;
        let s8 = (u6[i + 8] >> 4) & 0x03;
        let s12 = (u6[i + 12] >> 4) & 0x03;
        out[104 + i] = s0 | (s4 << 2) | (s8 << 4) | (s12 << 6);
    }

    // d
    out[108..110].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
    out
}

fn dequant_q3k_superblock_cpu(sb: &[u8; 110]) -> [f32; 256] {
    // Mirror of the fixed dequant_q3_k in gguf.rs.
    let d = half::f16::from_le_bytes([sb[108], sb[109]]).to_f32();
    let hmask = &sb[0..32];
    let qs = &sb[32..96];
    let scales_raw = &sb[96..108];

    let mut scales = [0i8; 16];
    for i in 0..16 {
        let low4 = if i < 8 {
            scales_raw[i] & 0x0F
        } else {
            (scales_raw[i - 8] >> 4) & 0x0F
        };
        let high2 = (scales_raw[8 + (i & 3)] >> (2 * (i / 4))) & 0x03;
        let u6 = low4 | (high2 << 4);
        scales[i] = (u6 as i32 - 32) as i8;
    }

    let mut out = [0f32; 256];
    for j in 0..256 {
        let q2 = (qs[j / 4] >> ((j % 4) * 2)) & 0x03;
        let hbit = (hmask[j / 8] >> (j % 8)) & 1;
        let q3 = q2 | (hbit << 2);
        let sc = scales[j / 16] as f32;
        out[j] = d * sc * (q3 as f32 - 4.0);
    }
    out
}

fn dequant_q3k_tensor_cpu(rows: usize, cols: usize, packed: &[u8]) -> Vec<f32> {
    assert_eq!(packed.len(), rows * cols * 55 / 128);
    let sb_per_row = cols / 256;
    let row_bytes = sb_per_row * 110;
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for sb in 0..sb_per_row {
            let base = r * row_bytes + sb * 110;
            let sb_bytes: &[u8; 110] = packed[base..base + 110].try_into().unwrap();
            let deq = dequant_q3k_superblock_cpu(sb_bytes);
            for i in 0..256 {
                out[r * cols + sb * 256 + i] = deq[i];
            }
        }
    }
    out
}

#[test]
fn q3k_superblock_encode_decode_roundtrip() {
    // Q3_K encodes signed 6-bit scales, so legal range is -32..31.
    let scales: [i8; 16] = [
        -32, -20, -10, -5, -1, 0, 1, 3, 7, 11, 15, 19, 23, 27, 29, 31,
    ];
    let mut vals = [0u8; 256];
    for i in 0..256 {
        vals[i] = (i % 8) as u8;
    }
    let sb = make_q3k_superblock(0.0078125, &scales, &vals);
    let out = dequant_q3k_superblock_cpu(&sb);

    // Element 0: d * scales[0] * (0 - 4) = 0.0078125 * -32 * -4 = 1.0
    assert!((out[0] - 1.0).abs() < 1e-6, "out[0]={}", out[0]);
    // Element 1: d * scales[0] * (1 - 4) = 0.0078125 * -32 * -3 = 0.75
    assert!((out[1] - 0.75).abs() < 1e-6, "out[1]={}", out[1]);

    // And the decoded scales must match the input (using the fixed scheme).
    let scales_raw = &sb[96..108];
    let mut decoded = [0i8; 16];
    for i in 0..16 {
        let low4 = if i < 8 {
            scales_raw[i] & 0x0F
        } else {
            (scales_raw[i - 8] >> 4) & 0x0F
        };
        let high2 = (scales_raw[8 + (i & 3)] >> (2 * (i / 4))) & 0x03;
        let u6 = low4 | (high2 << 4);
        decoded[i] = (u6 as i32 - 32) as i8;
    }
    assert_eq!(decoded, scales);
}

#[test]
fn q3k_gemv_matches_cpu_varied_input() {
    const ROWS: usize = 8;
    const COLS: usize = 512; // 2 superblocks/row
    let scales: [i8; 16] = [-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26];

    let mut packed = Vec::with_capacity(ROWS * COLS * 55 / 128);
    for r in 0..ROWS {
        for sb_idx in 0..(COLS / 256) {
            let mut vals = [0u8; 256];
            for i in 0..256 {
                vals[i] = ((i.wrapping_mul(3) + r * 5 + sb_idx * 7) % 8) as u8;
            }
            let d = 0.015625 * (1.0 + r as f32);
            let sb = make_q3k_superblock(d, &scales, &vals);
            packed.extend_from_slice(&sb);
        }
    }

    let cpu_weights = dequant_q3k_tensor_cpu(ROWS, COLS, &packed);
    let x_f32: Vec<f32> = (0..COLS).map(|i| (i as f32) * 0.001 - 0.25).collect();
    let cpu_y: Vec<f32> = (0..ROWS)
        .map(|r| {
            (0..COLS)
                .map(|c| cpu_weights[r * COLS + c] * x_f32[c])
                .sum()
        })
        .collect();

    let x_bf16: Vec<bf16> = x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let ctx = DeviceContext::new().expect("device context");
    let mat =
        DeviceMatrix::from_quantized_q3k(&ctx, &packed, ROWS, COLS).expect("from_quantized_q3k");
    let x_dev = DeviceVec::from_host(&ctx, &x_bf16).expect("x upload");
    let mut y_dev = DeviceVec::zeros(&ctx, ROWS).expect("y alloc");
    ops::gemv(&ctx, &mat, &x_dev, &mut y_dev).expect("gemv");

    let mut y_bf16 = vec![bf16::ZERO; ROWS];
    y_dev
        .copy_region_to_host(&ctx, 0, ROWS, &mut y_bf16)
        .expect("download");

    for r in 0..ROWS {
        let g = y_bf16[r].to_f32();
        let c = cpu_y[r];
        let tol = (c.abs() * 5e-2).max(5e-2);
        assert!((g - c).abs() < tol, "row {r}: gpu={g} cpu={c} (tol={tol})");
    }
}

#[test]
fn q4k_gemv_matches_cpu_varied_input() {
    const ROWS: usize = 8;
    const COLS: usize = 512;
    let sub_scales = [3u8, 4, 5, 6, 7, 8, 9, 10];
    let sub_mins = [0u8, 1, 2, 3, 4, 5, 6, 7];

    let mut packed = Vec::with_capacity(ROWS * COLS * 9 / 16);
    for r in 0..ROWS {
        for sb_idx in 0..(COLS / 256) {
            let mut nibbles = [0u8; 256];
            for i in 0..256 {
                nibbles[i] = ((i as u8).wrapping_mul(17 + r as u8 + sb_idx as u8)) & 0x0F;
            }
            let d = 0.015625 * (1.0 + r as f32);
            let dmin = 0.0078125 * (1.0 + r as f32);
            let sb = make_superblock(d, dmin, &sub_scales, &sub_mins, &nibbles);
            packed.extend_from_slice(&sb);
        }
    }

    let cpu_weights = dequant_tensor_cpu(ROWS, COLS, &packed);
    let x_f32: Vec<f32> = (0..COLS).map(|i| (i as f32) * 0.001 - 0.25).collect();
    let cpu_y: Vec<f32> = (0..ROWS)
        .map(|r| {
            (0..COLS)
                .map(|c| cpu_weights[r * COLS + c] * x_f32[c])
                .sum()
        })
        .collect();

    let x_bf16: Vec<bf16> = x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let ctx = DeviceContext::new().expect("device context");
    let mat =
        DeviceMatrix::from_quantized_q4k(&ctx, &packed, ROWS, COLS).expect("from_quantized_q4k");
    let x_dev = DeviceVec::from_host(&ctx, &x_bf16).expect("x upload");
    let mut y_dev = DeviceVec::zeros(&ctx, ROWS).expect("y alloc");
    ops::gemv(&ctx, &mat, &x_dev, &mut y_dev).expect("gemv");

    let mut y_bf16 = vec![bf16::ZERO; ROWS];
    y_dev
        .copy_region_to_host(&ctx, 0, ROWS, &mut y_bf16)
        .expect("download");

    for r in 0..ROWS {
        let g = y_bf16[r].to_f32();
        let c = cpu_y[r];
        let tol = (c.abs() * 5e-2).max(5e-2);
        assert!((g - c).abs() < tol, "row {r}: gpu={g} cpu={c} (tol={tol})");
    }
}
