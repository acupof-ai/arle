//! On-device `mul_mmq` (batched prefill GEMM) correctness proof.
//!
//! The decode path multiplies a `[m, k]` quantized weight by ONE activation
//! vector (`mul_mat_vecq`, see `device_gemv.rs`). Prefill needs the same weight
//! multiplied by `n` activation vectors at once, which is `mul_mmq`: a tiled
//! integer-dot-product GEMM that consumes the SAME `block_q8_1_x4` activation
//! format the GEMV does, so a single `q8_1_quantize` dispatch feeds both lanes.
//!
//! What this proves:
//!   1. `D[n, m] = A[m, k] · Bᵀ[n, k]` matches the CPU reference
//!      `d[t][r] = Σ_c dequant(W)[r, c] · x[t][c]` for Q4_K and Q8_0 weights.
//!   2. Row `t` of `D` equals what the proven GEMV produces for activation `t`
//!      alone — the property batched prefill actually depends on.
//!   3. Non-tile-aligned `m` and `n` are handled (the shader's `dr < M && dc < N`
//!      write guard), across the small / medium / large tile geometries that
//!      `MmqSpec::choose` selects.
//!
//! Runs only with `--features vulkan` + a working device; skips cleanly
//! otherwise.
#![cfg(feature = "vulkan")]

use infer_gguf::dequant::{dequantize_row_q4_k, dequantize_row_q8_0};
use vulkan_kernels::{
    BLOCK_Q4_K_BYTES, BLOCK_Q8_0_BYTES, BLOCK_Q8_1_BYTES, Kernel, MmqSpec, mmq_dispatch,
    mmq_params, mmq_with_params_and_spec, q8_1_quantize, q8_1_quantize_dispatch,
    q8_1_quantize_params,
};
use vulkan_sys::{DeviceBuffer, VulkanContext};

/// Per-element tolerance, expressed as a fraction of the sum of term
/// MAGNITUDES (`Σ|w·x|`), not of the result. `mul_mmq`'s integer dot products
/// are exact; the only error is the f16 rounding of the per-block scale product
/// `d_a * d_b` (`FLOAT_TYPE = float16_t`), which is ~5e-4 relative, accumulated
/// into an f32 (`ACC_TYPE = float`).
const MAGNITUDE_REL_TOL: f32 = 2e-3;

/// Deterministic xorshift PRNG so failures reproduce.
struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }
    fn next_byte(&mut self) -> u8 {
        (self.next_u32() & 0xFF) as u8
    }
    fn next_unit_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// f32 -> IEEE-754 half (round-to-nearest-even), matching the f16 block scales.
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    let mut h = sign | ((exp as u16) << 10) | (mant >> 13) as u16;
    if (mant & 0x1000) != 0 && ((mant & 0x0fff) != 0 || (h & 1) != 0) {
        h += 1;
    }
    h
}

/// One valid `block_q4_K` (144 B): `f16 d; f16 dmin; u8 scales[12]; u8 qs[128]`.
fn make_q4_k_block(rng: &mut Rng) -> Vec<u8> {
    let mut b = Vec::with_capacity(BLOCK_Q4_K_BYTES);
    let d = 0.02 + (rng.next_u32() as f32 / u32::MAX as f32) * 0.04;
    let dmin = 0.01 + (rng.next_u32() as f32 / u32::MAX as f32) * 0.02;
    b.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    b.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());
    for _ in 0..12 {
        b.push(rng.next_byte());
    }
    for _ in 0..128 {
        b.push(rng.next_byte());
    }
    debug_assert_eq!(b.len(), BLOCK_Q4_K_BYTES);
    b
}

/// IEEE-754 half → f32, for decoding the `block_q8_1_x4` scales we read back.
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

/// Dequantize `block_q8_1_x4` bytes back to f32: `{ f16vec2 ds[4]; i8 qs[128] }`
/// per 144-byte group, sub-block `i` scaled by `ds[i].x`.
///
/// This is what the *shader* multiplies against, so it — not the pre-quantized
/// f32 — is the right oracle for the GEMM itself. The q8_1 rounding is the
/// `q8_1_quantize` kernel's contract and is proven separately in `device_q8_1`.
fn dequantize_q8_1_x4(bytes: &[u8], ne: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(ne);
    for group in bytes.chunks_exact(4 * BLOCK_Q8_1_BYTES) {
        for sub in 0..4 {
            let d = f16_to_f32(u16::from_le_bytes([group[sub * 4], group[sub * 4 + 1]]));
            for j in 0..32 {
                out.push(d * (group[16 + sub * 32 + j] as i8) as f32);
            }
        }
    }
    out.truncate(ne);
    out
}

/// One valid `block_q8_0` (34 B): `f16 d; i8 qs[32]`.
fn make_q8_0_block(rng: &mut Rng) -> Vec<u8> {
    let mut b = Vec::with_capacity(BLOCK_Q8_0_BYTES);
    let d = 0.01 + (rng.next_u32() as f32 / u32::MAX as f32) * 0.04;
    b.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    for _ in 0..32 {
        b.push(rng.next_byte());
    }
    debug_assert_eq!(b.len(), BLOCK_Q8_0_BYTES);
    b
}

/// Quantize `n` activation rows of `k` f32 each to `block_q8_1_x4` on the GPU.
///
/// The rows are quantized in ONE dispatch over the flat `n * k` buffer. That is
/// only equivalent to quantizing each row separately when `k` is a multiple of
/// the 128-value x4 group, which is also what `mul_mmq` requires for a B row to
/// start on an x4 boundary — asserted here so a bad shape fails loudly.
fn quantize_rows_x4(ctx: &VulkanContext, rows: &[Vec<f32>], k: usize) -> Vec<u8> {
    assert_eq!(k % 128, 0, "mul_mmq B rows must be 128-value aligned");
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let ne = flat.len();
    let input_bytes: Vec<u8> = flat.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out_len = ne.div_ceil(128) * 4 * BLOCK_Q8_1_BYTES;

    let mut buf_in = DeviceBuffer::alloc(ctx, input_bytes.len()).expect("alloc q8_1 input");
    buf_in
        .copy_from_host(&input_bytes)
        .expect("upload q8_1 input");
    let mut buf_out = DeviceBuffer::alloc(ctx, out_len).expect("alloc q8_1 output");
    buf_out
        .copy_from_host(&vec![0u8; out_len])
        .expect("zero q8_1 output");

    q8_1_quantize(
        ctx,
        &[&buf_in, &buf_out],
        q8_1_quantize_dispatch(ne as u32),
        &q8_1_quantize_params(ne as u32),
    )
    .expect("q8_1_quantize on device");

    let mut got = vec![0u8; out_len];
    buf_out.copy_to_host(&mut got).expect("read back q8_1");
    got
}

/// Run one `mul_mmq` case and assert every `[n, m]` output element against the
/// CPU reference. Returns `(device output, per-element term magnitude)`; the
/// magnitudes are the denominator any *further* comparison against this output
/// must use, for the cancellation reason described below.
#[allow(clippy::too_many_arguments)]
fn run_mmq_case(
    ctx: &VulkanContext,
    label: &str,
    kernel: Kernel,
    m: usize,
    n: usize,
    k: usize,
    weight_bytes: &[u8],
    row_bytes: usize,
    activations: &[Vec<f32>],
    dequant_row: &dyn Fn(&[u8]) -> Vec<f32>,
    rel_tol: f32,
) -> (Vec<f32>, Vec<f32>) {
    let spec = MmqSpec::choose(
        kernel,
        m as u32,
        n as u32,
        ctx.max_compute_shared_memory_size(),
    )
    .unwrap_or_else(|| panic!("{label}: no mul_mmq tile fits shared memory"));

    let activ = quantize_rows_x4(ctx, activations, k);

    // CPU reference: dequantize each weight row once, then dot with the
    // DEQUANTIZED q8_1 activations — the exact values the shader multiplies.
    // Comparing against the pre-quantization f32 would fold the q8_1 rounding
    // (proven separately in `device_q8_1`) into this test's tolerance, and with
    // random 4-bit quants the dot products cancel hard enough that the rounding
    // alone swamps any real GEMM error.
    let flat_b = dequantize_q8_1_x4(&activ, n * k);
    let weight_rows: Vec<Vec<f32>> = (0..m)
        .map(|r| {
            let row = dequant_row(&weight_bytes[r * row_bytes..(r + 1) * row_bytes]);
            assert_eq!(row.len(), k, "{label}: dequant row width");
            row
        })
        .collect();
    // Cancellation-proof scale: assert against the sum of term MAGNITUDES, not
    // the (near-zero, heavily cancelled) result.
    let mut expected = vec![0f32; n * m];
    let mut magnitude = vec![0f32; n * m];
    for t in 0..n {
        let x = &flat_b[t * k..(t + 1) * k];
        for (r, w) in weight_rows.iter().enumerate() {
            expected[t * m + r] = w.iter().zip(x).map(|(a, b)| a * b).sum();
            magnitude[t * m + r] = w.iter().zip(x).map(|(a, b)| (a * b).abs()).sum();
        }
    }

    let mut buf_a = DeviceBuffer::alloc(ctx, weight_bytes.len()).expect("alloc weights");
    buf_a.copy_from_host(weight_bytes).expect("upload weights");
    let mut buf_b = DeviceBuffer::alloc(ctx, activ.len()).expect("alloc activations");
    buf_b.copy_from_host(&activ).expect("upload activations");
    let out_len = m * n * std::mem::size_of::<f32>();
    let mut buf_d = DeviceBuffer::alloc(ctx, out_len).expect("alloc dst");
    buf_d.copy_from_host(&vec![0u8; out_len]).expect("zero dst");

    mmq_with_params_and_spec(
        kernel,
        ctx,
        &[&buf_a, &buf_b, &buf_d],
        mmq_dispatch(m as u32, n as u32, &spec),
        &mmq_params(m as u32, n as u32, k as u32),
        &spec,
    )
    .unwrap_or_else(|e| panic!("{label}: mul_mmq dispatch failed: {e}"));

    let mut out_bytes = vec![0u8; out_len];
    buf_d.copy_to_host(&mut out_bytes).expect("read back dst");
    let got: Vec<f32> = out_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut worst_rel = 0f32;
    for t in 0..n {
        for r in 0..m {
            let idx = t * m + r;
            let denom = magnitude[idx].max(1e-3);
            let rel = (got[idx] - expected[idx]).abs() / denom;
            worst_rel = worst_rel.max(rel);
            assert!(
                rel < rel_tol,
                "{label} token {t} row {r}: got {} vs expected {} \
                 (err {rel} of term magnitude {} >= tol {rel_tol})",
                got[idx],
                expected[idx],
                magnitude[idx]
            );
        }
    }
    eprintln!(
        "[{label}] m={m} n={n} k={k} tile={}x{} PASS (worst err={worst_rel:.2e} of \
         term magnitude, tol={rel_tol:.0e})",
        spec.bm(),
        spec.bn()
    );
    (got, magnitude)
}

fn build_q4_k_weights(rng: &mut Rng, m: usize, k: usize) -> (Vec<u8>, usize) {
    let row_bytes = vulkan_kernels::q4_k_row_bytes(k).expect("q4_k row bytes");
    let mut bytes = Vec::with_capacity(m * row_bytes);
    for _ in 0..m * (k / 256) {
        bytes.extend_from_slice(&make_q4_k_block(rng));
    }
    (bytes, row_bytes)
}

fn build_q8_0_weights(rng: &mut Rng, m: usize, k: usize) -> (Vec<u8>, usize) {
    let row_bytes = vulkan_kernels::q8_0_row_bytes(k).expect("q8_0 row bytes");
    let mut bytes = Vec::with_capacity(m * row_bytes);
    for _ in 0..m * (k / 32) {
        bytes.extend_from_slice(&make_q8_0_block(rng));
    }
    (bytes, row_bytes)
}

#[test]
fn quantized_mmq_matches_reference_on_device() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping device mul_mmq test");
            return;
        }
    };
    eprintln!(
        "ARLE Vulkan mul_mmq proof on: {} (maxComputeSharedMemorySize={})",
        ctx.device_name(),
        ctx.max_compute_shared_memory_size()
    );

    let mut rng = Rng(0x2545_F491_4F6C_DD1D);

    // Three shapes, chosen so `MmqSpec::choose` lands on a different tile each
    // time and none of `m` / `n` divides its tile evenly — that is the write
    // guard (`dr < M && dc < N`) and the K-loop tail under test.
    //
    // `k` is a multiple of 256 throughout: the K-quant `block_a_to_shmem` indexes
    // A in 8 sub-blocks per 256-value super-block, and `mul_mmq` needs B rows on
    // 128-value x4 boundaries.
    for &(m, n, k) in &[(68usize, 5usize, 256usize), (130, 70, 512), (200, 132, 256)] {
        let activations: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..k).map(|_| rng.next_unit_f32()).collect())
            .collect();

        let (q4k, q4k_row) = build_q4_k_weights(&mut rng, m, k);
        run_mmq_case(
            &ctx,
            "Q4_K",
            Kernel::MmqQ4K,
            m,
            n,
            k,
            &q4k,
            q4k_row,
            &activations,
            &|row| dequantize_row_q4_k(row, k).expect("dequant q4_k"),
            MAGNITUDE_REL_TOL,
        );

        let (q8, q8_row) = build_q8_0_weights(&mut rng, m, k);
        run_mmq_case(
            &ctx,
            "Q8_0",
            Kernel::MmqQ8_0,
            m,
            n,
            k,
            &q8,
            q8_row,
            &activations,
            &|row| dequantize_row_q8_0(row, k).expect("dequant q8_0"),
            MAGNITUDE_REL_TOL,
        );
    }
}

/// The property batched prefill depends on: running `n` tokens through
/// `mul_mmq` in one dispatch gives, for each token, the same vector the
/// single-token GEMV path gives — so replacing the serial prefill loop cannot
/// change the model's output.
#[test]
fn mmq_row_matches_single_token_gemv() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping mul_mmq/GEMV parity test");
            return;
        }
    };

    let (m, n, k) = (64usize, 7usize, 256usize);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let activations: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..k).map(|_| rng.next_unit_f32()).collect())
        .collect();
    let (weights, row_bytes) = build_q4_k_weights(&mut rng, m, k);

    let (batched, magnitude) = run_mmq_case(
        &ctx,
        "Q4_K/parity",
        Kernel::MmqQ4K,
        m,
        n,
        k,
        &weights,
        row_bytes,
        &activations,
        &|row| dequantize_row_q4_k(row, k).expect("dequant q4_k"),
        MAGNITUDE_REL_TOL,
    );

    // Same weight, one token at a time through the proven GEMV.
    for (t, x) in activations.iter().enumerate() {
        let activ = quantize_rows_x4(&ctx, std::slice::from_ref(x), k);
        let mut buf_w = DeviceBuffer::alloc(&ctx, weights.len()).expect("alloc weights");
        buf_w.copy_from_host(&weights).expect("upload weights");
        let mut buf_b = DeviceBuffer::alloc(&ctx, activ.len()).expect("alloc activations");
        buf_b.copy_from_host(&activ).expect("upload activations");
        let out_len = m * std::mem::size_of::<f32>();
        let mut buf_d = DeviceBuffer::alloc(&ctx, out_len).expect("alloc dst");
        buf_d.copy_from_host(&vec![0u8; out_len]).expect("zero dst");
        let mut buf_f0 = DeviceBuffer::alloc(&ctx, 4).expect("alloc fuse0");
        buf_f0.copy_from_host(&[0u8; 4]).expect("zero fuse0");
        let mut buf_f1 = DeviceBuffer::alloc(&ctx, 4).expect("alloc fuse1");
        buf_f1.copy_from_host(&[0u8; 4]).expect("zero fuse1");

        vulkan_kernels::q4_k_gemv_with_params(
            &ctx,
            &[&buf_w, &buf_b, &buf_d, &buf_f0, &buf_f1],
            vulkan_kernels::gemv_dispatch(m as u32),
            &vulkan_kernels::gemv_params(k as u32, m as u32),
        )
        .expect("GEMV dispatch");

        let mut out_bytes = vec![0u8; out_len];
        buf_d.copy_to_host(&mut out_bytes).expect("read back dst");
        for r in 0..m {
            let gemv = f32::from_le_bytes(out_bytes[r * 4..r * 4 + 4].try_into().unwrap());
            let mmq = batched[t * m + r];
            // Both paths quantize the SAME activation to the SAME q8_1 blocks and
            // accumulate in f32, so they differ only by scale-rounding and
            // summation order. Scaled by term magnitude, not by the result: the
            // dot of random 4-bit quants cancels to ~1/30 of the term magnitude,
            // which would turn ordinary rounding into a 40% "relative" error.
            let denom = magnitude[t * m + r].max(1e-3);
            assert!(
                (mmq - gemv).abs() / denom < MAGNITUDE_REL_TOL,
                "token {t} row {r}: mmq {mmq} vs gemv {gemv} \
                 (err {} of term magnitude {})",
                (mmq - gemv).abs() / denom,
                magnitude[t * m + r]
            );
        }
    }
    eprintln!("[Q4_K/parity] mul_mmq rows match single-token GEMV for all {n} tokens");
}
