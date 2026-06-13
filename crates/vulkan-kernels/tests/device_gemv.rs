//! On-device GEMV (decode matmul) correctness proof.
//!
//! Proves that the quantized `mul_mat_vecq` GEMV (matrix × vector — the decode
//! matmul) actually runs on the GPU and returns the mathematically correct
//! result. Two weight formats are validated end-to-end:
//!   * Q4_K (`mul_mat_vecq_q4_k`) — the K-quant attention/FFN weights.
//!   * Q8_0 (`mul_mat_vecq_q8_0`) — the format qwen35 / qwen35moe store their
//!     attention + FFN weights in, so this kernel is required for any forward.
//!
//! Pipeline per case:
//!   1. Build a `[nrows, ncols]` quantized weight matrix from random *valid*
//!      block bytes. The CPU reference dequantizes the SAME bytes with
//!      `infer-gguf` (whose layout matches the shader's `block_q*` structs),
//!      so the comparison isolates the GPU dot-product math.
//!   2. Build an f32 activation vector `x[ncols]` and quantize it to
//!      `block_q8_1_x4` on the GPU via the proven `q8_1_quantize` kernel.
//!   3. Launch the GEMV with the discovered call contract (5 storage buffers in
//!      order [A weights, B q8_1_x4, D f32 dst, Fuse0, Fuse1]; the 13-uint
//!      push-constant block from `gemv_params`; one workgroup per output row).
//!   4. Read back `y[nrows]` and assert each row matches the CPU reference
//!      `y[r] = sum_c dequant(W)[r,c] * x[c]` within a tolerance that accounts
//!      for the q8_1 activation quantization.
//!
//! Runs only with `--features vulkan` + a working device; skips cleanly
//! otherwise.
#![cfg(feature = "vulkan")]

use infer_gguf::dequant::{dequantize_row_q4_k, dequantize_row_q8_0};
use vulkan_kernels::{
    BLOCK_Q4_K_BYTES, BLOCK_Q8_0_BYTES, BLOCK_Q8_1_BYTES, gemv_dispatch, gemv_params,
    q4_k_gemv_with_params, q8_0_gemv_with_params, q8_1_quantize, q8_1_quantize_dispatch,
    q8_1_quantize_params,
};
use vulkan_sys::{DeviceBuffer, VulkanContext};

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
    /// Small-magnitude f32 in [-1, 1).
    fn next_unit_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// f32 -> IEEE-754 half (round-to-nearest-even), matching the f16 fields the
/// shaders read for block scales.
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp <= 0 {
        // Subnormal/zero — fine for our small, normalized scales (never hit).
        return sign;
    }
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    let mut h = sign | ((exp as u16) << 10) | (mant >> 13) as u16;
    // round-to-nearest-even on the dropped low 13 bits
    if (mant & 0x1000) != 0 && ((mant & 0x0fff) != 0 || (h & 1) != 0) {
        h += 1;
    }
    h
}

/// Build one valid `block_q4_K` (144 B): `f16 d; f16 dmin; u8 scales[12];
/// u8 qs[128]`. Scales/dmin are kept modest so dequantized magnitudes stay
/// tame and the q8_1 activation quantization error stays bounded.
fn make_q4_k_block(rng: &mut Rng) -> Vec<u8> {
    let mut b = Vec::with_capacity(BLOCK_Q4_K_BYTES);
    let d = 0.02 + (rng.next_u32() as f32 / u32::MAX as f32) * 0.04; // ~0.02..0.06
    let dmin = 0.01 + (rng.next_u32() as f32 / u32::MAX as f32) * 0.02;
    b.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    b.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());
    // 12 packed 6-bit scale/min bytes — any byte pattern is a valid encoding.
    for _ in 0..12 {
        b.push(rng.next_byte());
    }
    // 128 quant bytes (two 4-bit nibbles each).
    for _ in 0..128 {
        b.push(rng.next_byte());
    }
    debug_assert_eq!(b.len(), BLOCK_Q4_K_BYTES);
    b
}

/// Build one valid `block_q8_0` (34 B): `f16 d; i8 qs[32]`.
fn make_q8_0_block(rng: &mut Rng) -> Vec<u8> {
    let mut b = Vec::with_capacity(BLOCK_Q8_0_BYTES);
    let d = 0.01 + (rng.next_u32() as f32 / u32::MAX as f32) * 0.04; // ~0.01..0.05
    b.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    for _ in 0..32 {
        b.push(rng.next_byte()); // reinterpreted as i8 in the block
    }
    debug_assert_eq!(b.len(), BLOCK_Q8_0_BYTES);
    b
}

/// Decode an f16 half (mirror of the scale fields written above) for printing.
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

/// Quantize an f32 activation vector to `block_q8_1_x4` bytes on the GPU using
/// the proven `q8_1_quantize` kernel and read it back.
fn quantize_activations_x4(ctx: &VulkanContext, x: &[f32]) -> Vec<u8> {
    let ne = x.len();
    let input_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let num_x4 = ne.div_ceil(128);
    let out_len = num_x4 * 4 * BLOCK_Q8_1_BYTES; // 144 B per x4 block

    let mut buf_in = DeviceBuffer::alloc(ctx, input_bytes.len()).expect("alloc q8_1 input");
    buf_in
        .copy_from_host(&input_bytes)
        .expect("upload q8_1 input");
    let mut buf_out = DeviceBuffer::alloc(ctx, out_len).expect("alloc q8_1 output");
    buf_out
        .copy_from_host(&vec![0u8; out_len])
        .expect("zero q8_1 output");

    let params = q8_1_quantize_params(ne as u32);
    let dispatch = q8_1_quantize_dispatch(ne as u32);
    q8_1_quantize(ctx, &[&buf_in, &buf_out], dispatch, &params).expect("q8_1_quantize on device");

    let mut got = vec![0u8; out_len];
    buf_out.copy_to_host(&mut got).expect("read back q8_1");
    got
}

/// Run a GEMV case and assert correctness. `weight_bytes` is the row-major
/// quantized matrix; `dequant_row` produces the f32 reference for one row.
fn run_gemv_case(
    ctx: &VulkanContext,
    label: &str,
    nrows: usize,
    ncols: usize,
    weight_bytes: &[u8],
    x: &[f32],
    dequant_row: impl Fn(&[u8]) -> Vec<f32>,
    launch: impl Fn(
        &VulkanContext,
        &[&DeviceBuffer<'_>],
        vulkan_kernels::Dispatch,
        &vulkan_kernels::KernelParams,
    ) -> vulkan_kernels::Result<()>,
    rel_tol: f32,
    row_bytes: usize,
) {
    // CPU reference: y[r] = sum_c dequant(W)[r, c] * x[c].
    let mut expected = vec![0f32; nrows];
    for (r, exp) in expected.iter_mut().enumerate() {
        let row = dequant_row(&weight_bytes[r * row_bytes..(r + 1) * row_bytes]);
        assert_eq!(row.len(), ncols, "{label}: dequant row width");
        *exp = row.iter().zip(x).map(|(w, xc)| w * xc).sum();
    }

    // Device buffers in the discovered binding order.
    let activ = quantize_activations_x4(ctx, x);

    let mut buf_w = DeviceBuffer::alloc(ctx, weight_bytes.len()).expect("alloc weights");
    buf_w.copy_from_host(weight_bytes).expect("upload weights");
    let mut buf_b = DeviceBuffer::alloc(ctx, activ.len()).expect("alloc activations");
    buf_b.copy_from_host(&activ).expect("upload activations");
    let out_len = nrows * std::mem::size_of::<f32>();
    let mut buf_d = DeviceBuffer::alloc(ctx, out_len).expect("alloc dst");
    buf_d.copy_from_host(&vec![0u8; out_len]).expect("zero dst");
    // Bindings 3/4 (Fuse0/Fuse1) are declared by the shader but only read when
    // fusion_flags != 0 (we pass 0). Bind small dummies to satisfy the layout.
    let mut buf_f0 = DeviceBuffer::alloc(ctx, 4).expect("alloc fuse0");
    buf_f0.copy_from_host(&[0u8; 4]).expect("zero fuse0");
    let mut buf_f1 = DeviceBuffer::alloc(ctx, 4).expect("alloc fuse1");
    buf_f1.copy_from_host(&[0u8; 4]).expect("zero fuse1");

    let params = gemv_params(ncols as u32, nrows as u32);
    let dispatch = gemv_dispatch(nrows as u32);
    launch(
        ctx,
        &[&buf_w, &buf_b, &buf_d, &buf_f0, &buf_f1],
        dispatch,
        &params,
    )
    .unwrap_or_else(|e| panic!("{label}: GEMV dispatch failed: {e}"));

    let mut out_bytes = vec![0u8; out_len];
    buf_d.copy_to_host(&mut out_bytes).expect("read back dst");
    let got: Vec<f32> = out_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    eprintln!("[{label}] nrows={nrows} ncols={ncols}");
    let mut worst_rel = 0f32;
    for r in 0..nrows {
        let denom = expected[r].abs().max(1e-3);
        let rel = (got[r] - expected[r]).abs() / denom;
        worst_rel = worst_rel.max(rel);
        eprintln!(
            "  row {r}: got={:+.5} expected={:+.5} rel_err={:.4}",
            got[r], expected[r], rel
        );
        assert!(
            rel < rel_tol,
            "{label} row {r}: got {} vs expected {} (rel_err {rel} >= tol {rel_tol})",
            got[r],
            expected[r]
        );
    }
    eprintln!("[{label}] PASS (worst rel_err={worst_rel:.4}, tol={rel_tol})");
}

#[test]
fn quantized_gemv_executes_on_device() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping device GEMV test");
            return;
        }
    };
    eprintln!("ARLE Vulkan GEMV proof on: {}", ctx.device_name());

    let nrows = 4usize;
    let ncols = 256usize; // one K-quant super-block per row / 8 q8_0 blocks per row

    // Shared activation vector (small magnitudes keep q8_1 error bounded).
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let x: Vec<f32> = (0..ncols).map(|_| rng.next_unit_f32()).collect();

    // ---- Q4_K case --------------------------------------------------------
    let q4k_row = vulkan_kernels::q4_k_row_bytes(ncols).expect("q4_k row bytes");
    assert_eq!(q4k_row, BLOCK_Q4_K_BYTES); // 256 cols => 1 block
    let mut q4k_weights = Vec::with_capacity(nrows * q4k_row);
    for _ in 0..nrows {
        q4k_weights.extend_from_slice(&make_q4_k_block(&mut rng));
    }
    eprintln!(
        "Q4_K block[0] d={:.5} dmin={:.5}",
        f16_to_f32(u16::from_le_bytes([q4k_weights[0], q4k_weights[1]])),
        f16_to_f32(u16::from_le_bytes([q4k_weights[2], q4k_weights[3]])),
    );
    run_gemv_case(
        &ctx,
        "Q4_K",
        nrows,
        ncols,
        &q4k_weights,
        &x,
        |row| dequantize_row_q4_k(row, ncols).expect("dequant q4_k"),
        q4_k_gemv_with_params,
        2e-2,
        q4k_row,
    );

    // ---- Q8_0 case --------------------------------------------------------
    let q8_0_row = vulkan_kernels::q8_0_row_bytes(ncols).expect("q8_0 row bytes");
    assert_eq!(q8_0_row, (ncols / 32) * BLOCK_Q8_0_BYTES);
    let mut q8_0_weights = Vec::with_capacity(nrows * q8_0_row);
    for _ in 0..nrows {
        for _ in 0..(ncols / 32) {
            q8_0_weights.extend_from_slice(&make_q8_0_block(&mut rng));
        }
    }
    eprintln!(
        "Q8_0 block[0] d={:.5}",
        f16_to_f32(u16::from_le_bytes([q8_0_weights[0], q8_0_weights[1]])),
    );
    run_gemv_case(
        &ctx,
        "Q8_0",
        nrows,
        ncols,
        &q8_0_weights,
        &x,
        |row| dequantize_row_q8_0(row, ncols).expect("dequant q8_0"),
        q8_0_gemv_with_params,
        2e-2,
        q8_0_row,
    );
}
