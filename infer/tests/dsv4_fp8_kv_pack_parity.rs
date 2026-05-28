//! DSv4-Flash MODEL1 FP8 KV pack parity test.
//!
//! Drives the `arle_dsv4_fp8_kv_pack_cuda` kernel through its Rust wrapper,
//! reads back the packed bytes, performs the same dequant the FlashMLA
//! kernel does on the consumer side (`splitkv_mla.cuh:540-549` + `dequant.h`),
//! and asserts the round-trip error stays within the E4M3 quantization
//! envelope.
//!
//! Phase D-3' of
//! [`docs/plans/2026-05-28-dsv4-flashmla-decode-integration.md`].
//!
//! Tolerances (matching upstream's E4M3 + per-tile-e8m0 dynamic range):
//!   - E4M3 has ~3 mantissa bits (4 mantissa values in subnormals; up to ~12.5%
//!     relative error at the top of each E4M3 binade).
//!   - With per-tile e8m0 scale chosen as `scale = 2^(ceil(log2(amax/448)))`,
//!     the worst-case absolute error for any element is bounded by
//!     `scale * 2^(-3) ≈ amax/8` (top of mantissa step) and the smallest
//!     subnormal step is `scale * 2^(-9) ≈ amax/(448*8)`.
//!   - Net cap used here: `abs_err ≤ 0.20 * tile_amax + scale * 2^(-9)`.

#![cfg(feature = "cuda")]

use cuda_kernels::attention::dsv4_fp8_kv_pack;
use cuda_kernels::tensor::{DeviceContext, DeviceVec};
use cudarc::driver::DevicePtr;
use half::bf16;

// Match the MODEL1 contract — kept in lockstep with
// `crates/cuda-kernels/csrc/attention/dsv4_fp8_kv_pack.cu`.
const HEAD_DIM_NOPE: usize = 448;
const HEAD_DIM_ROPE: usize = 64;
const QUANT_TILE_SIZE: usize = 64;
const NUM_TILES: usize = HEAD_DIM_NOPE / QUANT_TILE_SIZE; // 7
const NUM_SCALES: usize = 8;
const ROPE_BYTES: usize = HEAD_DIM_ROPE * 2; // 128
const TOKEN_DATA_BYTES: usize = HEAD_DIM_NOPE + ROPE_BYTES; // 576
const TOKEN_BYTES: usize = TOKEN_DATA_BYTES + NUM_SCALES; // 584
const PAGE_BLOCK_SIZE: usize = 64;

/// CPU mirror of CUDA `__nv_cvt_e8m0x2_to_bf162raw` for a single byte.
///
/// E8M0 is an exponent-only float (8 bits all-exponent, IEEE bias 127):
///   byte = 0       → 0.0
///   byte = 255     → NaN
///   byte ∈ [1,254] → 2^(byte - 127)
fn e8m0_to_f32(b: u8) -> f32 {
    if b == 0 {
        0.0
    } else if b == 255 {
        f32::NAN
    } else {
        // 2^(b - 127) is exact and representable in f32 for the
        // valid e8m0 range.
        let exp = b as i32 - 127;
        2f32.powi(exp)
    }
}

/// CPU mirror of CUDA `__nv_fp8_e4m3` decode → f32.
///
/// E4M3 format (1 sign + 4 exponent + 3 mantissa, bias 7):
///   - byte 0x00       → +0.0
///   - byte 0x80       → -0.0
///   - byte 0x7F       → +448.0   (max normal)
///   - byte 0xFF       → -448.0
///   - otherwise:
///       sign = byte >> 7
///       exp_raw = (byte >> 3) & 0xF
///       mant = byte & 0x7
///       if exp_raw == 0:  value = mant * 2^(-9)    (subnormals, bias 7 → 2^(1-7-3))
///       else:             value = (1 + mant/8) * 2^(exp_raw - 7)
fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if (byte & 0x80) != 0 { -1.0f32 } else { 1.0 };
    let exp_raw = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as i32;
    let mag = if exp_raw == 0 {
        // Subnormal: 2^(1 - bias) * mant / 2^3 = 2^(-6) * mant/8 = mant * 2^(-9).
        (mant as f32) * 2f32.powi(-9)
    } else {
        // Normal: (1 + mant/8) * 2^(exp_raw - 7).
        (1.0 + (mant as f32) / 8.0) * 2f32.powi(exp_raw - 7)
    };
    sign * mag
}

/// CPU reference for the bf16 round (round-to-nearest-even).
///
/// bf16 keeps the top 16 bits of f32; round to nearest even on the bottom 16.
fn round_bf16(x: f32) -> bf16 {
    bf16::from_f32(x)
}

/// Compute the e8m0 byte that the kernel will produce for a given tile amax.
///
/// Mirrors the CUDA logic exactly so the test can call this independently
/// for sanity (e.g. to check the byte stored on device matches the
/// expectation). Returns 0 if amax ≤ 0.
fn encode_e8m0_byte_cpu(amax: f32) -> u8 {
    if amax <= 0.0 || !amax.is_finite() {
        return 0;
    }
    // ceil(log2(amax / 448)) via frexpf-style decomposition.
    let (_m, e_amax) = frexp(amax);
    let mut e = e_amax - 9;
    let trial = 448.0 * 2f32.powi(e);
    if trial < amax {
        e += 1;
    }
    let e_clamped = e.clamp(-126, 127);
    (e_clamped + 127) as u8
}

/// f32 frexp returning (mantissa in [0.5, 1.0), exponent) such that
/// `x = mantissa * 2^exponent`. Matches the C `frexpf` semantics that the
/// CUDA kernel relies on.
fn frexp(x: f32) -> (f32, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let abs = x.abs();
    let bits = abs.to_bits();
    let raw_exp = ((bits >> 23) & 0xFF) as i32;
    if raw_exp == 0 {
        // Subnormal: renormalize.
        let mut m = abs;
        let mut e = 0i32;
        while m < 0.5 {
            m *= 2.0;
            e -= 1;
        }
        let signed = if x.is_sign_negative() { -m } else { m };
        return (signed, e);
    }
    // Normal: m * 2^(raw_exp - 126) = x → m = x / 2^(raw_exp - 126).
    // Cheaper: replace exponent field with bias-1 (= 126) → result is in
    // [0.5, 1.0) with the same sign.
    let new_bits = (bits & 0x807F_FFFF) | (126u32 << 23);
    let mantissa_abs = f32::from_bits(new_bits);
    let signed = if x.is_sign_negative() {
        -mantissa_abs
    } else {
        mantissa_abs
    };
    (signed, raw_exp - 126)
}

/// Quick self-check of the CPU frexp + e8m0 encode round-trip.
#[test]
fn cpu_e8m0_roundtrip_sanity() {
    // For amax = 448 exactly: scale should be 1.0 → e=0 → byte=127.
    let byte = encode_e8m0_byte_cpu(448.0);
    let decoded = e8m0_to_f32(byte);
    assert!(
        (decoded - 1.0).abs() < 1e-9,
        "amax=448 → byte={byte} → decoded={decoded}, expected 1.0"
    );

    // For amax = 224 (= 448 / 2): scale should be 0.5 → e=-1 → byte=126.
    let byte = encode_e8m0_byte_cpu(224.0);
    let decoded = e8m0_to_f32(byte);
    assert!(
        (decoded - 0.5).abs() < 1e-9,
        "amax=224 → byte={byte} → decoded={decoded}, expected 0.5"
    );

    // For amax = 449 (just above 448): scale should bump to 2.0 → e=1 → byte=128.
    let byte = encode_e8m0_byte_cpu(449.0);
    let decoded = e8m0_to_f32(byte);
    assert!(
        (decoded - 2.0).abs() < 1e-9,
        "amax=449 → byte={byte} → decoded={decoded}, expected 2.0"
    );

    // For amax = 1.0: scale = 1/256 = 2^-8 → e=-8 → byte=119.
    // (448 * 2^-9 = 0.875 < 1.0 → bump → e=-8; 448*2^-8=1.75 ≥ 1.0).
    let byte = encode_e8m0_byte_cpu(1.0);
    let decoded = e8m0_to_f32(byte);
    assert!(
        (decoded - 2f32.powi(-8)).abs() < 1e-9,
        "amax=1.0 → byte={byte} → decoded={decoded}, expected 2^-8 = {}",
        2f32.powi(-8)
    );

    // Zero / non-finite amax → byte=0 → decoded 0.0.
    assert_eq!(encode_e8m0_byte_cpu(0.0), 0);
    assert_eq!(encode_e8m0_byte_cpu(-1.0), 0);
    assert_eq!(encode_e8m0_byte_cpu(f32::NAN), 0);
}

/// CPU reference encode of one tile: returns (e8m0_byte, [fp8 bytes; 64]).
fn cpu_encode_tile(tile_vals: &[bf16; QUANT_TILE_SIZE]) -> (u8, [u8; QUANT_TILE_SIZE]) {
    let amax = tile_vals
        .iter()
        .map(|v| v.to_f32().abs())
        .fold(0.0f32, f32::max);
    let byte = encode_e8m0_byte_cpu(amax);
    let scale = if byte == 0 { 1.0f32 } else { e8m0_to_f32(byte) };

    // Emulate __nv_fp8_e4m3(x): convert f32 → e4m3 byte via round-to-nearest.
    let mut out = [0u8; QUANT_TILE_SIZE];
    for (i, v) in tile_vals.iter().enumerate() {
        let scaled = v.to_f32() / scale;
        out[i] = f32_to_e4m3_byte(scaled);
    }
    (byte, out)
}

/// Round-to-nearest-even f32 → E4M3 byte.
///
/// E4M3 range: ±448 (max normal). Subnormals fill [-2^(-6), 2^(-6)] approx.
fn f32_to_e4m3_byte(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F; // E4M3 reserves no NaN encoding officially; saturate.
    }
    let sign_bit = if x.is_sign_negative() { 0x80u8 } else { 0u8 };
    let ax = x.abs().min(448.0);
    if ax == 0.0 {
        return sign_bit;
    }
    // Find exponent.
    let (m, e_x) = frexp(ax); // m ∈ [0.5, 1.0), ax = m * 2^e_x
    // Aim representation:
    //   normal: (1 + mant/8) * 2^(exp_raw - 7), exp_raw ∈ [1, 15]
    //   subnormal: mant/8 * 2^(-6), mant ∈ [1, 7]
    let mut exp_raw = e_x - 1 + 7; // because ax = (2m) * 2^(e_x - 1), 2m ∈ [1,2)
    let mantissa_f = 2.0 * m - 1.0; // ∈ [0, 1)
    if exp_raw < 1 {
        // Subnormal path.
        let shift = 1 - exp_raw;
        let m_sub = ax * 2f32.powi(6 + 3); // mant/8 = ax * 2^6 → mant = ax * 2^9
        let mant_round = m_sub.round() as i32;
        let mant_round = mant_round.clamp(0, 7);
        if mant_round == 0 {
            return sign_bit;
        }
        let _ = shift;
        return sign_bit | (mant_round as u8);
    }
    // Mantissa rounding: 3 mantissa bits → 8 levels.
    let mant_int_f = mantissa_f * 8.0;
    let mant_round = mant_int_f.round() as i32;
    let (mant, carry) = if mant_round >= 8 {
        (0, 1)
    } else {
        (mant_round, 0)
    };
    exp_raw += carry;
    if exp_raw >= 16 {
        // Overflow → saturate to ±448 (E4M3 max normal = 0x7E typically, but
        // upstream lets it reach 0x7F; pick 0x7E for "safe max").
        return sign_bit | 0x7E;
    }
    sign_bit | ((exp_raw as u8) << 3) | (mant as u8)
}

/// Build a deterministic pseudo-random bf16 buffer with mixed magnitudes.
fn make_random_bf16(seed: u64, count: usize, large_every: usize) -> Vec<bf16> {
    // xorshift PRNG.
    let mut state = seed;
    (0..count)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 32) as u32;
            let v = (u as f32 / u32::MAX as f32 - 0.5) * 4.0; // [-2, 2]
            let v = if large_every > 0 && i % large_every == 0 {
                // Occasional larger value to exercise the per-tile scale.
                v * 50.0
            } else {
                v
            };
            bf16::from_f32(v)
        })
        .collect()
}

/// Build the expected per-token packed bytes on the CPU for cross-check.
///
/// Layout matches the kernel exactly: 576 B AoS (NoPE 448 + RoPE 128) per
/// token + 8 e8m0 scale bytes per token at the block tail.
fn cpu_pack_one_block(
    n_tokens_in_block: usize,
    nope: &[bf16],
    rope: &[bf16],
    token_offset: usize,
) -> Vec<u8> {
    let mut packed = vec![0u8; PAGE_BLOCK_SIZE * TOKEN_BYTES];
    for local_t in 0..n_tokens_in_block {
        let t = token_offset + local_t;
        let row = local_t;

        // NoPE + scales.
        for tile in 0..NUM_TILES {
            let mut tile_vals = [bf16::ZERO; QUANT_TILE_SIZE];
            for i in 0..QUANT_TILE_SIZE {
                tile_vals[i] = nope[t * HEAD_DIM_NOPE + tile * QUANT_TILE_SIZE + i];
            }
            let (byte, fp8_bytes) = cpu_encode_tile(&tile_vals);
            // Write NoPE bytes.
            let nope_off = row * TOKEN_DATA_BYTES + tile * QUANT_TILE_SIZE;
            packed[nope_off..nope_off + QUANT_TILE_SIZE].copy_from_slice(&fp8_bytes);
            // Write scale byte (block-tail region).
            let scales_off = PAGE_BLOCK_SIZE * TOKEN_DATA_BYTES + row * NUM_SCALES + tile;
            packed[scales_off] = byte;
        }
        // Pad scale (slot 7).
        let pad_off = PAGE_BLOCK_SIZE * TOKEN_DATA_BYTES + row * NUM_SCALES + NUM_SCALES - 1;
        packed[pad_off] = 0;

        // RoPE bytes (bf16 verbatim, little-endian).
        for i in 0..HEAD_DIM_ROPE {
            let v = rope[t * HEAD_DIM_ROPE + i];
            let bytes = v.to_bits().to_le_bytes();
            let off = row * TOKEN_DATA_BYTES + HEAD_DIM_NOPE + i * 2;
            packed[off] = bytes[0];
            packed[off + 1] = bytes[1];
        }
    }
    packed
}

/// Reconstruct one tile's bf16 values from packed bytes via the same path the
/// FlashMLA kernel takes (e8m0 → bf16 scale, fp8_e4m3 → f32, multiply, round).
fn reconstruct_tile(
    packed_block: &[u8],
    row: usize,
    tile: usize,
) -> ([bf16; QUANT_TILE_SIZE], f32) {
    let scale_byte = packed_block[PAGE_BLOCK_SIZE * TOKEN_DATA_BYTES + row * NUM_SCALES + tile];
    let scale = e8m0_to_f32(scale_byte);
    let mut out = [bf16::ZERO; QUANT_TILE_SIZE];
    for i in 0..QUANT_TILE_SIZE {
        let fp8_byte = packed_block[row * TOKEN_DATA_BYTES + tile * QUANT_TILE_SIZE + i];
        let f32_val = e4m3_to_f32(fp8_byte) * scale;
        out[i] = round_bf16(f32_val);
    }
    (out, scale)
}

#[test]
fn dsv4_fp8_kv_pack_parity_two_blocks() {
    let ctx = DeviceContext::new().expect("CUDA context");

    // 128 tokens → 2 full blocks of 64 each.
    let n_tokens = 128usize;
    let nope_host = make_random_bf16(0xCAFEF00DDEADBEEF, n_tokens * HEAD_DIM_NOPE, 137);
    let rope_host = make_random_bf16(0x1234_5678_9ABC_DEF0, n_tokens * HEAD_DIM_ROPE, 0);

    let nope_dev = DeviceVec::from_host(&ctx, &nope_host).expect("nope H2D");
    let rope_dev = DeviceVec::from_host(&ctx, &rope_host).expect("rope H2D");

    // 2 blocks worth of packed FP8 KV storage.
    let num_blocks = 2usize;
    let pool_bytes = num_blocks * PAGE_BLOCK_SIZE * TOKEN_BYTES;
    let pool = ctx
        .stream
        .alloc_zeros::<u8>(pool_bytes)
        .expect("FP8 pool alloc");
    let pool_ptr = {
        let (p, _g) = pool.device_ptr(&ctx.stream);
        p
    };

    // Routing: tokens 0..63 → block 0, row=i; tokens 64..127 → block 1, row=i-64.
    let block_ids: Vec<i32> = (0..n_tokens)
        .map(|i| (i / PAGE_BLOCK_SIZE) as i32)
        .collect();
    let in_block_rows: Vec<i32> = (0..n_tokens)
        .map(|i| (i % PAGE_BLOCK_SIZE) as i32)
        .collect();
    let block_ids_dev = ctx.stream.clone_htod(&block_ids).expect("block_ids H2D");
    let rows_dev = ctx.stream.clone_htod(&in_block_rows).expect("rows H2D");

    dsv4_fp8_kv_pack(
        &ctx,
        &nope_dev,
        &rope_dev,
        pool_ptr,
        &block_ids_dev,
        &rows_dev,
        n_tokens,
        PAGE_BLOCK_SIZE,
    )
    .expect("pack kernel");

    ctx.sync().expect("sync");

    let packed_host = ctx.stream.clone_dtoh(&pool).expect("pool D2H");
    assert_eq!(packed_host.len(), pool_bytes);

    // Cross-check 1: per-tile e8m0 byte the GPU stored matches the CPU
    // reference exactly. This isolates the encoding logic from the fp8
    // rounding noise.
    let mut byte_mismatches = 0usize;
    for t in 0..n_tokens {
        let block_id = t / PAGE_BLOCK_SIZE;
        let row = t % PAGE_BLOCK_SIZE;
        let block_off = block_id * PAGE_BLOCK_SIZE * TOKEN_BYTES;
        let block = &packed_host[block_off..block_off + PAGE_BLOCK_SIZE * TOKEN_BYTES];

        for tile in 0..NUM_TILES {
            // CPU expectation.
            let mut tile_vals = [bf16::ZERO; QUANT_TILE_SIZE];
            for i in 0..QUANT_TILE_SIZE {
                tile_vals[i] = nope_host[t * HEAD_DIM_NOPE + tile * QUANT_TILE_SIZE + i];
            }
            let (expected_byte, _) = cpu_encode_tile(&tile_vals);

            let scale_off = PAGE_BLOCK_SIZE * TOKEN_DATA_BYTES + row * NUM_SCALES + tile;
            let actual_byte = block[scale_off];
            if actual_byte != expected_byte {
                byte_mismatches += 1;
                if byte_mismatches <= 4 {
                    eprintln!(
                        "e8m0 byte mismatch t={t} tile={tile}: gpu={actual_byte} cpu={expected_byte}"
                    );
                }
            }
        }
        // Pad byte (slot 7) must always be 0.
        let pad_off = PAGE_BLOCK_SIZE * TOKEN_DATA_BYTES + row * NUM_SCALES + NUM_SCALES - 1;
        assert_eq!(block[pad_off], 0, "pad byte non-zero at t={t}");
    }
    assert_eq!(
        byte_mismatches, 0,
        "e8m0 scale bytes differ between CPU and GPU encode paths"
    );

    // Cross-check 2: per-element abs-error after the kernel's reverse path.
    let mut max_abs_err = 0.0f32;
    let mut sum_abs_err = 0.0f64;
    let mut max_rel_err = 0.0f32;
    let mut n_elems = 0usize;

    for t in 0..n_tokens {
        let block_id = t / PAGE_BLOCK_SIZE;
        let row = t % PAGE_BLOCK_SIZE;
        let block_off = block_id * PAGE_BLOCK_SIZE * TOKEN_BYTES;
        let block = &packed_host[block_off..block_off + PAGE_BLOCK_SIZE * TOKEN_BYTES];

        // NoPE elements.
        for tile in 0..NUM_TILES {
            let mut tile_vals = [bf16::ZERO; QUANT_TILE_SIZE];
            for i in 0..QUANT_TILE_SIZE {
                tile_vals[i] = nope_host[t * HEAD_DIM_NOPE + tile * QUANT_TILE_SIZE + i];
            }
            let tile_amax = tile_vals
                .iter()
                .map(|v| v.to_f32().abs())
                .fold(0.0f32, f32::max);

            let (recon, scale) = reconstruct_tile(block, row, tile);
            // Per-tile envelope: abs_err ≤ 0.20 * tile_amax + scale * 2^-9.
            let envelope = 0.20 * tile_amax + scale * 2f32.powi(-9);
            for i in 0..QUANT_TILE_SIZE {
                let expected = tile_vals[i].to_f32();
                let actual = recon[i].to_f32();
                let abs_err = (actual - expected).abs();
                if abs_err > envelope + 1e-6 {
                    panic!(
                        "out-of-envelope element: t={t} tile={tile} i={i}: \
                         expected={expected} actual={actual} abs_err={abs_err} \
                         tile_amax={tile_amax} scale={scale} envelope={envelope}"
                    );
                }
                max_abs_err = max_abs_err.max(abs_err);
                sum_abs_err += abs_err as f64;
                let rel = if expected.abs() > 1e-6 {
                    abs_err / expected.abs()
                } else {
                    0.0
                };
                max_rel_err = max_rel_err.max(rel);
                n_elems += 1;
            }
        }

        // RoPE: bf16 verbatim — must round-trip exactly.
        for i in 0..HEAD_DIM_ROPE {
            let expected_bf16 = rope_host[t * HEAD_DIM_ROPE + i];
            let off = row * TOKEN_DATA_BYTES + HEAD_DIM_NOPE + i * 2;
            let actual_bits = u16::from_le_bytes([block[off], block[off + 1]]);
            let actual_bf16 = bf16::from_bits(actual_bits);
            assert_eq!(
                actual_bits,
                expected_bf16.to_bits(),
                "RoPE bf16 mismatch t={t} i={i}: expected={expected_bf16} actual={actual_bf16}"
            );
        }
    }

    let mean_abs_err = sum_abs_err / n_elems as f64;
    eprintln!(
        "dsv4_fp8_kv_pack_parity: tokens={n_tokens} elems={n_elems} \
         max_abs_err={max_abs_err:.6} mean_abs_err={mean_abs_err:.6} \
         max_rel_err={max_rel_err:.6}"
    );
    // Sanity floor — even worst-case E4M3 / per-tile scale shouldn't exceed
    // 0.2× any tile amax; over all tokens of N(0, 2) data the global max
    // should land comfortably under 30.0 (50× outlier scaled).
    assert!(
        max_abs_err < 30.0,
        "max_abs_err={max_abs_err} unexpectedly large — encode path likely broken"
    );
}

#[test]
fn dsv4_fp8_kv_pack_parity_uses_cpu_encode_self_consistency() {
    // Quick check: the CPU encode of a constant-magnitude tile produces
    // the expected e8m0 byte and fp8 bytes. Pure CPU — no CUDA required
    // beyond the cfg gate.
    let tile = [bf16::from_f32(1.0); QUANT_TILE_SIZE];
    let (byte, fp8) = cpu_encode_tile(&tile);
    // amax=1.0 → scale=2^-8 → fp8 must dequant to ~1.0.
    let scale = e8m0_to_f32(byte);
    assert!(
        (scale - 2f32.powi(-8)).abs() < 1e-9,
        "expected scale 2^-8, got {scale}"
    );
    // 1.0 / 2^-8 = 256 — clamped to 448 in fp8 (still in range), so the
    // reconstructed value should be close to 1.0 within ~12% relative.
    let recon = e4m3_to_f32(fp8[0]) * scale;
    assert!(
        (recon - 1.0).abs() < 0.06,
        "constant-tile recon {recon} too far from 1.0 (fp8 byte {})",
        fp8[0]
    );
}
