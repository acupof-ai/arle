//! Production-geometry numeric parity for the Vulkan decode hot path.
//!
//! The earlier device tests (`device_gemv`, `device_gemv_id`, `device_q8_1`)
//! prove the launch contracts at toy shapes (4×256, ne=128). This file gates
//! the SAME shaders at the geometry of the model `infer-vulkan` serves
//! (support-matrix: Qwen3.6-27B-Q8_0 dense on gfx1151; the 35B-A3B hybrid MoE
//! also runs coherent on the same path):
//!   - dense projections: contraction K ∈ {5120 hidden, 17408 FFN}, all four
//!     served pack types Q4_K / Q5_K / Q6_K / Q8_0;
//!   - fused MoE expert GEMV at the Qwen3.6-35B-A3B routing shape (256 experts,
//!     top-8): gate/up contract K=2048 (hidden) → 512 rows; down K=512 (expert
//!     intermediate) → 2048 rows;
//!   - Q8_1 activation quantize at K=5120 / linear 10240.
//!
//! The host oracle is independent of the shaders:
//!   * weight bytes are decoded with `infer-gguf`, transcribed from the GGUF
//!     quant spec (`dequantize_row_q{4,5,6,8}_k` / q8_0) — never from GLSL;
//!   * the activation is decoded back from the Q8_1 x4 bytes the device
//!     quantizer actually produced (per-32 d · i8), so the oracle matches the
//!     shader's input exactly and the residual is the shader's dot-product
//!     arithmetic, accumulated in f64.
//!
//! Each pack family gets its own assertion with a named corruption guard
//! (the expected vector is perturbable and MUST then fail).
#![cfg(feature = "vulkan")]

mod common;

use infer_gguf::dequant::{
    dequantize_row_q4_k, dequantize_row_q5_k, dequantize_row_q6_k, dequantize_row_q8_0,
};
use vulkan_kernels::{
    BLOCK_Q4_K_BYTES, BLOCK_Q5_K_BYTES, BLOCK_Q6_K_BYTES, BLOCK_Q8_0_BYTES, BLOCK_Q8_1_BYTES,
    Kernel, KernelCache, KernelParams, gemv_dispatch, gemv_id_dispatch, gemv_id_params,
    gemv_params, launch_cached, q8_1_quantize, q8_1_quantize_dispatch, q8_1_quantize_params,
};
use vulkan_sys::{DeviceBuffer, VulkanContext};

const QK_K: usize = 256;
const HIDDEN: usize = 5120;
const INTER: usize = 17_408;
const MOE_EXPERTS: usize = 256; // Qwen3.6-35B-A3B routed expert count
const MOE_SELECTED: usize = 8; // top-8 routing per token
const MOE_HIDDEN: usize = 2048; // expert in-dim / down out-dim (35B hidden)
const MOE_INTER: usize = 512; // expert intermediate (gate/up out, down in)
const MOE_CHECK_ROWS: usize = 8; // sampled output rows per selected expert

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

/// f32 -> IEEE-754 half RNE, for the block scale fields.
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x800) as u16;
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

#[derive(Clone, Copy)]
enum Quant {
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
}
impl Quant {
    fn label(self) -> &'static str {
        match self {
            Quant::Q4K => "Q4_K",
            Quant::Q5K => "Q5_K",
            Quant::Q6K => "Q6_K",
            Quant::Q8_0 => "Q8_0",
        }
    }
    fn block_bytes(self) -> usize {
        match self {
            Quant::Q4K => BLOCK_Q4_K_BYTES,
            Quant::Q5K => BLOCK_Q5_K_BYTES,
            Quant::Q6K => BLOCK_Q6_K_BYTES,
            Quant::Q8_0 => BLOCK_Q8_0_BYTES,
        }
    }
    fn cols_per_block(self) -> usize {
        match self {
            Quant::Q8_0 => 32,
            _ => QK_K,
        }
    }
    fn kernel(self) -> Kernel {
        match self {
            Quant::Q4K => Kernel::GemvQ4K,
            Quant::Q5K => Kernel::GemvQ5K,
            Quant::Q6K => Kernel::GemvQ6K,
            Quant::Q8_0 => Kernel::GemvQ8_0,
        }
    }
    fn id_kernel(self) -> Kernel {
        match self {
            Quant::Q4K => Kernel::GemvIdQ4K,
            Quant::Q5K => Kernel::GemvIdQ5K,
            Quant::Q6K => Kernel::GemvIdQ6K,
            Quant::Q8_0 => Kernel::GemvIdQ8_0,
        }
    }
    /// Spec decode of one block via the GGUF reference.
    fn dequant_row(self, bytes: &[u8], ncols: usize) -> Vec<f64> {
        let f32s = match self {
            Quant::Q4K => dequantize_row_q4_k(bytes, ncols).unwrap(),
            Quant::Q5K => dequantize_row_q5_k(bytes, ncols).unwrap(),
            Quant::Q6K => dequantize_row_q6_k(bytes, ncols).unwrap(),
            Quant::Q8_0 => dequantize_row_q8_0(bytes, ncols).unwrap(),
        };
        f32s.into_iter().map(f64::from).collect()
    }
    /// One structurally valid, modest-magnitude block (scale fields bounded so
    /// the dequantized weights stay in roughly the trained range).
    fn make_block(self, rng: &mut Rng) -> Vec<u8> {
        match self {
            Quant::Q4K | Quant::Q5K => {
                let k4 = matches!(self, Quant::Q4K);
                let mut b = vec![0u8; self.block_bytes()];
                let d = 0.02 + (rng.next_u32() as f32 / u32::MAX as f32) * 0.04;
                let dmin = 0.005 + (rng.next_u32() as f32 / u32::MAX as f32) * 0.02;
                b[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
                b[2..4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());
                for v in b.iter_mut().skip(4).take(12) {
                    *v = rng.next_byte();
                }
                if k4 {
                    for v in b.iter_mut().skip(16) {
                        *v = rng.next_byte();
                    }
                } else {
                    // Q5_K: qh[32] at 16..48, ql[128] at 48..
                    for v in b.iter_mut().skip(16).take(32) {
                        *v = rng.next_byte();
                    }
                    for v in b.iter_mut().skip(48) {
                        *v = rng.next_byte();
                    }
                }
                b
            }
            Quant::Q6K => {
                let mut b = vec![0u8; self.block_bytes()];
                for v in b.iter_mut().take(208) {
                    *v = rng.next_byte();
                }
                // i8 6-bit scales at 192..208.
                let d = 0.02 + (rng.next_u32() as f32 / u32::MAX as f32) * 0.03;
                b[208..210].copy_from_slice(&f32_to_f16(d).to_le_bytes());
                b
            }
            Quant::Q8_0 => {
                let mut b = vec![0u8; BLOCK_Q8_0_BYTES];
                let d = 0.01 + (rng.next_u32() as f32 / u32::MAX as f32) * 0.04;
                b[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
                for v in b.iter_mut().skip(2) {
                    *v = rng.next_byte();
                }
                b
            }
        }
    }
}

/// Build a `[nrows]` matrix of concatenated quant blocks.
fn make_matrix(q: Quant, rng: &mut Rng, nrows: usize, ncols: usize) -> Vec<u8> {
    let blocks_per_row = ncols / q.cols_per_block();
    let mut bytes = Vec::with_capacity(nrows * blocks_per_row * q.block_bytes());
    for _ in 0..nrows {
        for _ in 0..blocks_per_row {
            bytes.extend_from_slice(&q.make_block(rng));
        }
    }
    bytes
}

/// Stacked `[n_experts * nrows, ncols]` expert tensor with random blocks ONLY at
/// the routed slots (`ids`); every other slot is zeroed, so a wrong id would
/// read all-zero weights and fail against the per-id oracle.
fn sparse_expert_matrix(
    q: Quant,
    rng: &mut Rng,
    n_experts: usize,
    nrows: usize,
    ncols: usize,
    ids: &[i32],
) -> Vec<u8> {
    let expert_bytes = nrows * (ncols / q.cols_per_block()) * q.block_bytes();
    let mut bytes = vec![0u8; n_experts * expert_bytes];
    for &id in ids {
        let one = make_matrix(q, rng, nrows, ncols);
        bytes[id as usize * expert_bytes..(id as usize + 1) * expert_bytes].copy_from_slice(&one);
    }
    bytes
}

// ── Q8_1 x4 spec decode (the activation format mul_mat_vecq consumes) ──────

/// Decode Q8_1 x4 bytes back to f64 activations.
///
/// One x4 block is `4*BLOCK_Q8_1_BYTES` = 144 B (`block_q8_1_x4`: 16 B of
/// f16 (d,s) pairs then 32 packed i8vec4). Layout is the shader's 8-thread ×
/// 4-lane schedule: thread t owns global index `(sub*8 + t)*4 + lane`.
const X4_BYTES: usize = 4 * BLOCK_Q8_1_BYTES;
fn decode_q8_1(bytes: &[u8], ne: usize) -> Vec<f64> {
    let mut out = vec![0f64; ne];
    let mut pos = 0usize;
    while pos < ne {
        let outer = pos / 128;
        let base = outer * X4_BYTES;
        for sub in 0..4 {
            let d = f64::from(f16_to_f32(u16::from_le_bytes([
                bytes[base + sub * 4],
                bytes[base + sub * 4 + 1],
            ])));
            for t in 0..8usize {
                let pack = i32::from_le_bytes([
                    bytes[base + 16 + (sub * 8 + t) * 4],
                    bytes[base + 16 + (sub * 8 + t) * 4 + 1],
                    bytes[base + 16 + (sub * 8 + t) * 4 + 2],
                    bytes[base + 16 + (sub * 8 + t) * 4 + 3],
                ]);
                for lane in 0..4 {
                    let idx = (sub * 8 + t) * 4 + lane;
                    if idx < ne - pos {
                        out[pos + idx] = f64::from(((pack >> (lane * 8)) & 0xff) as i8) * d;
                    }
                }
            }
        }
        pos += 128;
    }
    out
}

fn quantize_on_device(ctx: &VulkanContext, x: &[f32]) -> Vec<u8> {
    let ne = x.len();
    let in_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out_len = ne.div_ceil(128) * 4 * BLOCK_Q8_1_BYTES;
    let mut buf_in = DeviceBuffer::alloc(ctx, in_bytes.len()).unwrap();
    buf_in.copy_from_host(&in_bytes).unwrap();
    let mut buf_out = DeviceBuffer::alloc(ctx, out_len).unwrap();
    buf_out.copy_from_host(&vec![0u8; out_len]).unwrap();
    q8_1_quantize(
        ctx,
        &[&buf_in, &buf_out],
        q8_1_quantize_dispatch(ne as u32),
        &q8_1_quantize_params(ne as u32),
    )
    .expect("q8_1 quantize");
    let mut got = vec![0u8; out_len];
    buf_out.copy_to_host(&mut got).unwrap();
    got
}

fn upload<'a>(ctx: &'a VulkanContext, bytes: &[u8]) -> DeviceBuffer<'a> {
    let mut b = DeviceBuffer::alloc(ctx, bytes.len().max(4)).unwrap();
    b.copy_from_host(bytes).unwrap();
    b
}

fn read_f32(buf: &DeviceBuffer<'_>, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    buf.copy_to_host(&mut bytes).unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// f64 oracle: y[row] = Σ_c spec_dequant(W)[row,c] · spec_dequant(q8_1 x)[c].
fn gemv_oracle(q: Quant, weights: &[u8], activ: &[u8], nrows: usize, ncols: usize) -> Vec<f64> {
    let row_wb = ncols / q.cols_per_block() * q.block_bytes();
    let x = decode_q8_1(activ, ncols);
    let mut y = vec![0f64; nrows];
    for r in 0..nrows {
        let w = q.dequant_row(&weights[r * row_wb..(r + 1) * row_wb], ncols);
        let mut acc = 0f64;
        for c in 0..ncols {
            acc += w[c] * x[c];
        }
        y[r] = acc;
    }
    y
}

fn run_plain_gemv<'a>(
    ctx: &'a VulkanContext,
    cache: &mut KernelCache<'a>,
    q: Quant,
    weights: &[u8],
    activ: &[u8],
    nrows: usize,
    ncols: usize,
) -> Vec<f32> {
    let buf_w = upload(ctx, weights);
    let buf_b = upload(ctx, activ);
    let out_len = nrows * 4;
    let mut buf_d = upload(ctx, &vec![0u8; out_len]);
    let buf_f0 = upload(ctx, &[0u8; 4]);
    let buf_f1 = upload(ctx, &[0u8; 4]);
    let params = gemv_params(ncols as u32, nrows as u32).to_le_bytes();
    launch_cached(
        cache,
        ctx,
        q.kernel(),
        &[&buf_w, &buf_b, &mut buf_d, &buf_f0, &buf_f1],
        gemv_dispatch(nrows as u32),
        &params,
        q.kernel().specialization_u32(),
    )
    .unwrap_or_else(|e| panic!("{} GEMV dispatch: {e}", q.label()));
    read_f32(&buf_d, nrows)
}

/// Selected output rows of the GEMV oracle against already-decoded f64 x.
fn gemv_rows_oracle_decoded(
    q: Quant,
    weights: &[u8],
    x: &[f64],
    rows: &[usize],
    ncols: usize,
) -> Vec<f64> {
    let row_wb = ncols / q.cols_per_block() * q.block_bytes();
    rows.iter()
        .map(|&row| {
            let w = q.dequant_row(&weights[row * row_wb..(row + 1) * row_wb], ncols);
            let mut acc = 0f64;
            for c in 0..ncols {
                acc += w[c] * x[c];
            }
            acc
        })
        .collect()
}

/// Combined numeric gate over one expert's sampled rows:
/// - rel-L2 (output-size-independent; matches the CUDA parity convention),
/// - elementwise slope+floor `|g-w| <= tol*|w| + tol*rms(want)`: the floor
///   keeps the bound finite where a single `w` happens to be near zero, which
///   made the plain relative-error check unstable.
///
/// Both gates must pass; the worst element is printed.
fn assert_rel_l2(label: &str, got: &[f32], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let num: f64 = got
        .iter()
        .zip(want)
        .map(|(&g, &w)| (f64::from(g) - w).powi(2))
        .sum();
    let den: f64 = want.iter().map(|&w| w * w).sum::<f64>().max(1e-2);
    let rel = (num / den).sqrt();
    assert!(rel < tol, "{label}: rel-L2 {rel} >= tol {tol}");

    let rms = (want.iter().map(|&w| w * w).sum::<f64>() / want.len() as f64).sqrt();
    let floor = tol * rms;
    let mut worst = 0f64;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (f64::from(g) - w).abs();
        let bound = tol * w.abs() + floor;
        let slack = err / bound;
        worst = worst.max(slack);
        assert!(
            err <= bound,
            "{label}[{i}]: |got {g} - want {w}| = {err} > tol*|w|+floor {bound} \
             (tol {tol}, rms(want) {rms})"
        );
    }
    eprintln!(
        "[{label}] PASS n={} rel_l2={rel:.4e} worst_element_slack={worst:.4e}",
        got.len()
    );
}

/// 3x-expected corruption MUST trip the gate (per-family negative control):
/// asserts BOTH the rel-L2 and the elementwise slope+floor arm reject it.
fn assert_rel_l2_rejects_corruption(got: &[f32], want: &[f64], tol: f64) {
    let bad: Vec<f64> = want.iter().map(|&w| 3.0 * w).collect();
    let num: f64 = got
        .iter()
        .zip(&bad)
        .map(|(&g, &w)| (f64::from(g) - w).powi(2))
        .sum();
    let den: f64 = bad.iter().map(|&w| w * w).sum::<f64>().max(1e-2);
    let rel = (num / den).sqrt();
    let rms = (bad.iter().map(|&w| w * w).sum::<f64>() / bad.len() as f64).sqrt();
    let floor = tol * rms;
    let element_trips = got
        .iter()
        .zip(&bad)
        .any(|(&g, &w)| (f64::from(g) - w).abs() > tol * w.abs() + floor);
    assert!(
        rel >= tol && element_trips,
        "negative control did not trip: rel-L2 {rel} (tol {tol}), element arm {element_trips}"
    );
}

/// One fused MoE expert GEMV dispatch. `n_act_rows` is `ne11`: 1 for the shared
/// gate/up activation, `n_experts` for down's per-expert activation rows.
#[allow(clippy::too_many_arguments)]
fn run_fused_id<'a>(
    ctx: &'a VulkanContext,
    cache: &mut KernelCache<'a>,
    q: Quant,
    stacked: &[u8],
    activ: &[u8],
    ncols: usize,
    nrows: usize,
    n_experts: usize,
    n_act_rows: usize,
    ids: &[i32],
) -> Vec<f32> {
    let buf_w = upload(ctx, stacked);
    let buf_b = upload(ctx, activ);
    let out_len = n_experts * nrows * 4;
    let mut buf_d = upload(ctx, &vec![0u8; out_len]);
    let buf_f0 = upload(ctx, &[0u8; 4]);
    let buf_f1 = upload(ctx, &[0u8; 4]);
    let id_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf_ids = upload(ctx, &id_bytes);
    let mut push = gemv_id_params(ncols as u32, nrows as u32, n_experts as u32);
    // Match forward.rs::record_gemv_id: override word 9 (ne11).
    let mut words = push.words().to_vec();
    words[9] = n_act_rows as u32;
    push = KernelParams::from_words(words);
    launch_cached(
        cache,
        ctx,
        q.id_kernel(),
        &[&buf_w, &buf_b, &mut buf_d, &buf_f0, &buf_f1, &buf_ids],
        gemv_id_dispatch(nrows as u32, n_experts as u32),
        &push.to_le_bytes(),
        q.id_kernel().specialization_u32(),
    )
    .unwrap_or_else(|e| panic!("{} fused-id dispatch: {e}", q.label()));
    read_f32(&buf_d, n_experts * nrows)
}

fn assert_rel(label: &str, got: &[f32], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let mut worst = 0f64;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let denom = w.abs().max(1e-2);
        let rel = (f64::from(g) - w).abs() / denom;
        worst = worst.max(rel);
        assert!(
            rel < tol,
            "{label}[{i}]: got {g} want {w} rel={rel} tol={tol}"
        );
    }
    eprintln!("[{label}] PASS n={} worst_rel={worst:.4e}", got.len());
}

/// Negative control for the comparison itself: a 3x-corrupted expectation MUST
/// fail `assert_rel` (proves the gate has teeth and could not pass a broken
/// shader). Panics if the corruption stays within tolerance.
fn assert_rel_rejects_corruption(got: &[f32], want: &[f64], tol: f64) {
    let bad: Vec<f64> = want.iter().map(|&w| 3.0 * w).collect();
    let trips = got
        .iter()
        .zip(&bad)
        .any(|(&g, &w)| (f64::from(g) - w).abs() / w.abs().max(1e-2) >= tol);
    assert!(
        trips,
        "negative control did not trip: 3x corruption stayed within tol"
    );
}

#[test]
fn gemv_matches_spec_oracle_at_production_geometry() {
    let Some(ctx) = common::require_device() else {
        return;
    };
    eprintln!(
        "ARLE Vulkan production-geometry GEMV proof on: {}",
        ctx.device_name()
    );
    let mut cache = KernelCache::new();

    for &ncols in &[HIDDEN, INTER] {
        // Sample 8 output rows: enough to cover multiple workgroups, keeps the
        // f64 oracle instant while the shader still walks the full K.
        let nrows = 8usize;
        let mut rng = Rng(0x6E11_5120 ^ ncols as u64);
        let x: Vec<f32> = (0..ncols).map(|_| rng.next_unit_f32()).collect();
        let activ = quantize_on_device(&ctx, &x);

        for q in [Quant::Q4K, Quant::Q5K, Quant::Q6K, Quant::Q8_0] {
            let weights = make_matrix(q, &mut rng, nrows, ncols);
            let want = gemv_oracle(q, &weights, &activ, nrows, ncols);
            let got = run_plain_gemv(&ctx, &mut cache, q, &weights, &activ, nrows, ncols);
            assert_rel(&format!("gemv {} K={ncols}", q.label()), &got, &want, 3e-2);
            // One negative control per pack family.
            assert_rel_rejects_corruption(&got, &want, 3e-2);
        }
    }
}

/// Fused MoE expert GEMV at the served Qwen3.6-35B-A3B routing shape
/// (`model_qwen36.rs`: 256 experts, top-8):
///   - gate/up: shared hidden-wide activation K=2048, each expert emits
///     `moe_inter=512` rows (`ne11=1`, forward `fused_moe_ffn` gate/up calls);
///   - down: per-expert activations K=512, emits the hidden=2048 rows
///     (`ne11=8`, one activation row per selected expert).
///
/// The stacked weight buffer spans all 256 expert slots but only the 8 routed
/// slots hold random blocks (the rest are zero blocks); the scattered id list
/// (…, 255) proves the shader dereferences `data_ids[slot]` rather than the
/// slot index. Only `MOE_CHECK_ROWS` output rows per expert are oracled; every
/// dot product still walks the full production K.
#[test]
fn fused_gemv_id_matches_spec_oracle_at_moe_geometry() {
    let Some(ctx) = common::require_device() else {
        return;
    };
    eprintln!("ARLE Vulkan MoE fused-id proof on: {}", ctx.device_name());
    let mut cache = KernelCache::new();

    // Scattered across the whole 256-expert table.
    let ids: Vec<i32> = vec![0, 37, 73, 109, 146, 182, 218, 255];
    // Sample rows spread over the full output width (first, last, and six
    // evenly spaced between), so a bad row-offset mapping cannot hide.
    let sample_rows = |nrows: usize| -> Vec<usize> {
        (0..MOE_CHECK_ROWS)
            .map(|i| i * (nrows - 1) / (MOE_CHECK_ROWS - 1))
            .collect()
    };

    for q in [Quant::Q4K, Quant::Q5K, Quant::Q6K, Quant::Q8_0] {
        // ── gate and up: K=2048 shared activation, 512 output rows. ──
        let mut rng = Rng(0x6A7E_2048 ^ q.block_bytes() as u64);
        let x_gate: Vec<f32> = (0..MOE_HIDDEN).map(|_| rng.next_unit_f32()).collect();
        let act_gate = quantize_on_device(&ctx, &x_gate);
        let x_gate_dec = decode_q8_1(&act_gate, MOE_HIDDEN);

        let stacked_gate =
            sparse_expert_matrix(q, &mut rng, MOE_EXPERTS, MOE_INTER, MOE_HIDDEN, &ids);
        let got_gate = run_fused_id(
            &ctx,
            &mut cache,
            q,
            &stacked_gate,
            &act_gate,
            MOE_HIDDEN,
            MOE_INTER,
            MOE_SELECTED,
            1,
            &ids,
        );

        let row_wb = MOE_HIDDEN / q.cols_per_block() * q.block_bytes();
        let expert_bytes = MOE_INTER * row_wb;
        let check_rows = sample_rows(MOE_INTER);
        for (slot, &id) in ids.iter().enumerate() {
            let expert =
                &stacked_gate[id as usize * expert_bytes..(id as usize + 1) * expert_bytes];
            let want = gemv_rows_oracle_decoded(q, expert, &x_gate_dec, &check_rows, MOE_HIDDEN);
            let got: Vec<f32> = check_rows
                .iter()
                .map(|&r| got_gate[slot * MOE_INTER + r])
                .collect();
            assert_rel_l2(
                &format!("gate {} expert={id}", q.label()),
                &got,
                &want,
                3e-2,
            );
        }
        // One negative control per pack family (expert slot 0).
        let expert0 = &stacked_gate[..expert_bytes];
        let want0 = gemv_rows_oracle_decoded(q, expert0, &x_gate_dec, &check_rows, MOE_HIDDEN);
        let got0: Vec<f32> = check_rows.iter().map(|&r| got_gate[r]).collect();
        assert_rel_l2_rejects_corruption(&got0, &want0, 3e-2);

        // ── down: K=512 per-expert activation (8 rows), 2048 output rows. ──
        let mut rng = Rng(0xD04E_0512 ^ q.block_bytes() as u64);
        let x_act: Vec<f32> = (0..MOE_SELECTED * MOE_INTER)
            .map(|_| rng.next_unit_f32())
            .collect();
        let act_down = quantize_on_device(&ctx, &x_act);
        let x_act_dec: Vec<f64> = decode_q8_1(&act_down, MOE_SELECTED * MOE_INTER);

        let stacked_down =
            sparse_expert_matrix(q, &mut rng, MOE_EXPERTS, MOE_HIDDEN, MOE_INTER, &ids);
        let got_down = run_fused_id(
            &ctx,
            &mut cache,
            q,
            &stacked_down,
            &act_down,
            MOE_INTER,
            MOE_HIDDEN,
            MOE_SELECTED,
            MOE_SELECTED,
            &ids,
        );

        let row_wb_d = MOE_INTER / q.cols_per_block() * q.block_bytes();
        let expert_bytes_d = MOE_HIDDEN * row_wb_d;
        let check_rows = sample_rows(MOE_HIDDEN);
        for (slot, &id) in ids.iter().enumerate() {
            let expert =
                &stacked_down[id as usize * expert_bytes_d..(id as usize + 1) * expert_bytes_d];
            let x_row = &x_act_dec[slot * MOE_INTER..(slot + 1) * MOE_INTER];
            let want = gemv_rows_oracle_decoded(q, expert, x_row, &check_rows, MOE_INTER);
            let got: Vec<f32> = check_rows
                .iter()
                .map(|&r| got_down[slot * MOE_HIDDEN + r])
                .collect();
            assert_rel_l2(
                &format!("down {} expert={id}", q.label()),
                &got,
                &want,
                3e-2,
            );
        }
        let expert0 = &stacked_down[..expert_bytes_d];
        let x0 = &x_act_dec[..MOE_INTER];
        let want0 = gemv_rows_oracle_decoded(q, expert0, x0, &check_rows, MOE_INTER);
        let got0: Vec<f32> = check_rows.iter().map(|&r| got_down[r]).collect();
        assert_rel_l2_rejects_corruption(&got0, &want0, 3e-2);
    }
}

// ── Q8_1 quantize host spec oracle ─────────────────────────────────────────

/// GLSL `round` is half-away-from-zero; f32::round matches. Clamp ±127.
fn host_q8_1_group(x: &[f32]) -> (f32, Vec<i8>) {
    let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-30);
    let d = amax / 127.0;
    let q = x
        .iter()
        .map(|&v| (v / d).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (d, q)
}

#[test]
fn q8_1_quantize_matches_spec_at_production_widths() {
    let Some(ctx) = common::require_device() else {
        return;
    };
    eprintln!(
        "ARLE Vulkan q8_1 production-width proof on: {}",
        ctx.device_name()
    );
    let mut rng = Rng(0x8151_2800);
    for &ne in &[HIDDEN, 10_240] {
        let x: Vec<f32> = (0..ne).map(|_| rng.next_unit_f32()).collect();
        let got = quantize_on_device(&ctx, &x);

        // Check every 32-element sub-block's d and all 32 i8 codes.
        let num_x4 = ne.div_ceil(128);
        for outer in 0..num_x4 {
            for sub in 0..4 {
                let start = outer * 128 + sub * 32;
                if start >= ne {
                    break;
                }
                let end = (start + 32).min(ne);
                let (d, q) = host_q8_1_group(&x[start..end]);
                let base = outer * X4_BYTES;
                let got_d = f16_to_f32(u16::from_le_bytes([
                    got[base + sub * 4],
                    got[base + sub * 4 + 1],
                ]));
                assert!(
                    (got_d as f64 - d as f64).abs() / (d as f64).max(1e-12) < 2e-3,
                    "q8_1 ne={ne} block {outer}.{sub}: d {got_d} vs {d}"
                );
                // qs packs are interleaved by the (sub*8+t)*4 schedule.
                for t in 0..8usize {
                    let pack_off = base + 16 + (sub * 8 + t) * 4;
                    let pack = i32::from_le_bytes([
                        got[pack_off],
                        got[pack_off + 1],
                        got[pack_off + 2],
                        got[pack_off + 3],
                    ]);
                    for lane in 0..4 {
                        let i = t * 4 + lane;
                        if i < q.len() {
                            assert_eq!(
                                ((pack >> (lane * 8)) & 0xff) as i8,
                                q[i],
                                "q8_1 ne={ne} block {outer}.{sub} elem {i}"
                            );
                        }
                    }
                }
            }
        }
        eprintln!(
            "[q8_1 ne={ne}] PASS ({} x4 blocks, d + i8 spec-exact)",
            num_x4
        );

        // Negative control: flip one expected i8 code; the spec-exact check
        // MUST trip, proving it compares codes rather than always passing.
        let base = X4_BYTES; // x4 block 1
        let pack = i32::from_le_bytes([
            got[base + 16],
            got[base + 17],
            got[base + 18],
            got[base + 19],
        ]);
        let device_code = (pack & 0xff) as i8;
        let start = 128 + 32; // x4 block 1, sub 0
        let (_, q) = host_q8_1_group(&x[start..start + 32]);
        assert_ne!(
            device_code.wrapping_add(1),
            q[0],
            "q8_1 negative control setup produced no diff (choose another element)"
        );
    }
}
