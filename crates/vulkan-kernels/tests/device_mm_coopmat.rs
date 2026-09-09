//! On-device `mul_mm` COOPMAT (cooperative-matrix prefill GEMM) correctness
//! proof.
//!
//! This is the fast twin of `device_mmq.rs`. Same shapes, same weights, same
//! output contract (`D[n, m] = A[m, k] · Bᵀ[n, k]`, `stride_d = m`), same
//! push-constant block — the whole difference is *how* the product is formed:
//!
//! | | `mul_mmq` | `mul_mm` COOPMAT |
//! | --- | --- | --- |
//! | `B` operand | `block_q8_1_x4` | plain row-major **f16** |
//! | `A` handling | integer dot vs quant blocks | dequantized to f16 shmem |
//! | multiply | `dotPacked4x8AccSatEXT` | `coopMatMulAdd` on matrix cores |
//!
//! The `B` operand change is the trap this file exists to catch: handing the
//! coopmat shader q8_1 bytes is not a validation error, it silently computes a
//! wrong answer. So the reference here is built from the *f16-rounded*
//! activations, exactly what the shader multiplies.
//!
//! Runs only with `--features vulkan`, and only on a device that advertises a
//! usable `f16 x f16 -> f32` subgroup shape. Skips cleanly otherwise — a
//! device without matrix cores is a supported configuration (the runtime falls
//! back to `mul_mmq`), not a failure.
#![cfg(feature = "vulkan")]

use infer_gguf::dequant::{dequantize_row_q4_k, dequantize_row_q6_k, dequantize_row_q8_0};
use vulkan_kernels::{
    BLOCK_Q4_K_BYTES, BLOCK_Q6_K_BYTES, BLOCK_Q8_0_BYTES, CoopmatShape, Kernel, MmSpec,
    mm_dispatch, mm_with_params_and_spec, mmq_params,
};
use vulkan_sys::{DeviceBuffer, VulkanContext};

/// Per-element tolerance as a fraction of the sum of term MAGNITUDES
/// (`Σ|w·x|`), not of the result — the dot of random quants cancels to a small
/// fraction of the terms, so a result-relative tolerance would be meaningless.
///
/// Looser than `device_mmq`'s 2e-3 for a structural reason: `mul_mmq` keeps the
/// products exact (8-bit integer dot) and only rounds the per-block scale,
/// whereas COOPMAT rounds **every** `A` value to f16 on its way into shared
/// memory and multiplies in f16 (`FLOAT_TYPE = float16_t`), accumulating into
/// f32. With ~1e-3 relative error per term and `k` terms partially cancelling,
/// 1e-2 of the term magnitude is the honest bound.
const MAGNITUDE_REL_TOL: f32 = 1e-2;

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

/// f32 -> IEEE-754 half (round-to-nearest-even).
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

/// IEEE-754 half -> f32.
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

/// Round-trip through f16 — the value the shader actually sees for `B`.
fn as_f16(v: f32) -> f32 {
    f16_to_f32(f32_to_f16(v))
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

/// One valid `block_q6_K` (210 B): `u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d`.
///
/// Included because Q6_K is the one quant whose coopmat `LOAD_VEC_A` is 2, not
/// 4 — a wrong value there aliases shared-memory rows rather than erroring, so
/// it needs its own oracle run.
fn make_q6_k_block(rng: &mut Rng) -> Vec<u8> {
    let mut b = Vec::with_capacity(BLOCK_Q6_K_BYTES);
    for _ in 0..128 + 64 {
        b.push(rng.next_byte());
    }
    for _ in 0..16 {
        b.push(rng.next_byte());
    }
    let d = 0.01 + (rng.next_u32() as f32 / u32::MAX as f32) * 0.03;
    b.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    debug_assert_eq!(b.len(), BLOCK_Q6_K_BYTES);
    b
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

fn build_weights(rng: &mut Rng, m: usize, k: usize, quant: Quant) -> (Vec<u8>, usize) {
    let (row_bytes, per_row) = match quant {
        Quant::Q4K => (vulkan_kernels::q4_k_row_bytes(k).unwrap(), k / 256),
        Quant::Q6K => (vulkan_kernels::q6_k_row_bytes(k).unwrap(), k / 256),
        Quant::Q8_0 => (vulkan_kernels::q8_0_row_bytes(k).unwrap(), k / 32),
    };
    let mut bytes = Vec::with_capacity(m * row_bytes);
    for _ in 0..m * per_row {
        match quant {
            Quant::Q4K => bytes.extend_from_slice(&make_q4_k_block(rng)),
            Quant::Q6K => bytes.extend_from_slice(&make_q6_k_block(rng)),
            Quant::Q8_0 => bytes.extend_from_slice(&make_q8_0_block(rng)),
        }
    }
    (bytes, row_bytes)
}

#[derive(Clone, Copy)]
enum Quant {
    Q4K,
    Q6K,
    Q8_0,
}

impl Quant {
    fn label(self) -> &'static str {
        match self {
            Quant::Q4K => "Q4_K",
            Quant::Q6K => "Q6_K",
            Quant::Q8_0 => "Q8_0",
        }
    }
    fn kernel(self) -> Kernel {
        match self {
            Quant::Q4K => Kernel::MmCmQ4K,
            Quant::Q6K => Kernel::MmCmQ6K,
            Quant::Q8_0 => Kernel::MmCmQ8_0,
        }
    }
    fn dequant_row(self, row: &[u8], k: usize) -> Vec<f32> {
        match self {
            Quant::Q4K => dequantize_row_q4_k(row, k).expect("dequant q4_k"),
            Quant::Q6K => dequantize_row_q6_k(row, k).expect("dequant q6_k"),
            Quant::Q8_0 => dequantize_row_q8_0(row, k).expect("dequant q8_0"),
        }
    }
}

/// Run one COOPMAT `mul_mm` case and assert every `[n, m]` element against the
/// CPU reference.
fn run_mm_case(
    ctx: &VulkanContext,
    shape: CoopmatShape,
    quant: Quant,
    m: usize,
    n: usize,
    k: usize,
    activations: &[Vec<f32>],
) {
    let label = quant.label();
    let kernel = quant.kernel();
    // `warp` must be the width the pipeline is actually created at. That is the
    // tile's own `WARP` spec constant — `KernelCache` reads it back out of the
    // specialization list to set `requiredSubgroupSize` — so passing anything
    // but the value `choose` bakes in would test a pipeline the runtime never
    // builds.
    let spec = MmSpec::choose(
        shape,
        ctx.subgroup_size().0,
        n as u32,
        ctx.max_compute_shared_memory_size(),
    )
    .unwrap_or_else(|| panic!("{label}: {shape:?} tiles no warptile within shared memory"));

    let mut rng = Rng(0x2545_F491_4F6C_DD1D ^ (m as u64) << 32 ^ (k as u64));
    let (weight_bytes, row_bytes) = build_weights(&mut rng, m, k, quant);

    // Reference against the f16-ROUNDED activations: that is what the shader
    // multiplies, so folding the f32->f16 conversion into the tolerance would
    // hide real GEMM error behind a known, separately-correct rounding step.
    let flat_b: Vec<f32> = activations.iter().flatten().map(|&v| as_f16(v)).collect();
    let weight_rows: Vec<Vec<f32>> = (0..m)
        .map(|r| {
            let row = quant.dequant_row(&weight_bytes[r * row_bytes..(r + 1) * row_bytes], k);
            assert_eq!(row.len(), k, "{label}: dequant row width");
            row
        })
        .collect();
    let mut expected = vec![0f32; n * m];
    let mut magnitude = vec![0f32; n * m];
    for t in 0..n {
        let x = &flat_b[t * k..(t + 1) * k];
        for (r, w) in weight_rows.iter().enumerate() {
            expected[t * m + r] = w.iter().zip(x).map(|(a, b)| a * b).sum();
            magnitude[t * m + r] = w.iter().zip(x).map(|(a, b)| (a * b).abs()).sum();
        }
    }

    // `B` on device: plain row-major f16, `stride_b = k` ELEMENTS. This is the
    // format `pf_pack_b`'s `f16_kv_pack` dispatch produces in the runtime.
    let b_bytes: Vec<u8> = activations
        .iter()
        .flatten()
        .flat_map(|&v| f32_to_f16(v).to_le_bytes())
        .collect();

    let mut buf_a = DeviceBuffer::alloc(ctx, weight_bytes.len()).expect("alloc weights");
    buf_a.copy_from_host(&weight_bytes).expect("upload weights");
    let mut buf_b = DeviceBuffer::alloc(ctx, b_bytes.len()).expect("alloc activations");
    buf_b.copy_from_host(&b_bytes).expect("upload activations");
    let out_len = m * n * std::mem::size_of::<f32>();
    let mut buf_d = DeviceBuffer::alloc(ctx, out_len).expect("alloc dst");
    buf_d.copy_from_host(&vec![0u8; out_len]).expect("zero dst");

    mm_with_params_and_spec(
        kernel,
        ctx,
        &[&buf_a, &buf_b, &buf_d],
        mm_dispatch(m as u32, n as u32, &spec),
        &mmq_params(m as u32, n as u32, k as u32),
        &spec,
    )
    .unwrap_or_else(|e| panic!("{label}: mul_mm COOPMAT dispatch failed: {e}"));

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
                rel < MAGNITUDE_REL_TOL,
                "{label} token {t} row {r}: got {} vs expected {} \
                 (err {rel} of term magnitude {} >= tol {MAGNITUDE_REL_TOL})",
                got[idx],
                expected[idx],
                magnitude[idx]
            );
        }
    }
    eprintln!(
        "[{label}/coopmat] m={m} n={n} k={k} tile={}x{} PASS (worst err={worst_rel:.2e} of \
         term magnitude, tol={MAGNITUDE_REL_TOL:.0e})",
        spec.bm(),
        spec.bn()
    );
}

#[test]
fn coopmat_mm_matches_reference_on_device() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping coopmat mul_mm test");
            return;
        }
    };
    let Some(shape) = ctx.coopmat() else {
        eprintln!(
            "{} advertises no f16xf16->f32 subgroup matrix shape; skipping \
             (the runtime falls back to mul_mmq here)",
            ctx.device_name()
        );
        return;
    };
    eprintln!(
        "ARLE Vulkan mul_mm COOPMAT proof on: {} (coopmat {}x{}x{}, \
         maxComputeSharedMemorySize={})",
        ctx.device_name(),
        shape.m,
        shape.n,
        shape.k,
        ctx.max_compute_shared_memory_size()
    );

    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);

    // `device_mmq`'s three shapes plus one more, chosen so that every tile
    // `MmSpec::choose` can return is actually exercised: n=5 -> narrow,
    // n=50 -> medium, n=70 and n=132 -> wide. Neither `m` nor `n` divides its
    // tile evenly in any case, so the write guard and the K-loop tail are both
    // exercised too.
    //
    // NOTE `m` is deliberately NOT a multiple of BM here even though every real
    // projection in the model is. The quant `load_a_to_shmem` has no `idx_m`
    // bound check, so a ragged `m` reads past the last weight row — the
    // allocation is padded by the buffer alignment, and the shader's `dr < M`
    // store guard drops the result. Keeping it ragged proves the store guard.
    for &(m, n, k) in &[
        (68usize, 5usize, 256usize),
        (130, 50, 512),
        (130, 70, 512),
        (200, 132, 256),
    ] {
        let activations: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..k).map(|_| rng.next_unit_f32()).collect())
            .collect();
        for quant in [Quant::Q4K, Quant::Q6K, Quant::Q8_0] {
            run_mm_case(&ctx, shape, quant, m, n, k, &activations);
        }
    }
}
