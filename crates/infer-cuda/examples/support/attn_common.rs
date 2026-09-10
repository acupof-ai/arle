//! Shared host-side math for the paged-attention parity examples
//! (`fa3_hd256_shim_parity`, `paged_quant_attn_parity`).
//!
//! The device kernels differ (vendored FA3 sm90 shim vs the native sm80+
//! split-KV kernel), but the gates consume the same durable 1-byte pools and
//! compare against the same f64 softmax, so the codec, RNG, page-table
//! permutation, tolerance metric, and attention evaluation live here once.
//!
//! Everything here is host code with no CUDA dependency; both examples gate
//! this module behind their own `#[cfg(feature = "cuda")]` block.

use half::bf16;

pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn normal(&mut self) -> f32 {
        let u1 = self.unit().max(1e-7);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

pub fn bf(v: f32) -> bf16 {
    bf16::from_f32(v)
}

pub fn round_even(x: f64) -> f64 {
    let r = x.round();
    if (x.fract().abs() - 0.5).abs() < 1e-12 && (r as i64) % 2 != 0 {
        r - x.signum()
    } else {
        r
    }
}

/// Encode f32 to an e4m3 byte (1 sign, 4 exp bias 7, 3 mantissa), RN-even.
/// Normal E field: value = (8+m)·2^(E-10), m in 0..=7. Subnormal E=0:
/// value = m·2^-9. Max finite 448.
pub fn e4m3_encode(x: f32) -> u8 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) as u8) << 7;
    if x.is_nan() {
        return 0x7f;
    }
    let v = f64::from(x).abs();
    if v >= 448.0 {
        return sign | 0x7e;
    }
    if v == 0.0 {
        return sign;
    }
    let e2 = v.log2().floor() as i32;
    if e2 >= -6 {
        let step = 2f64.powi(e2 - 3);
        let mut sig = round_even(v / step) as i32; // 8..=16
        let mut field = e2 + 7;
        if sig == 16 {
            sig = 8;
            field += 1;
        }
        sign | ((field as u8) << 3) | ((sig - 8) as u8)
    } else {
        let sig = round_even(v / 2f64.powi(-9)).clamp(0.0, 7.0) as i32;
        sign | sig as u8
    }
}

pub fn e4m3_decode(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let field = (b >> 3) & 0x0f;
    let mant = f32::from(b & 0x07);
    if field == 0 {
        sign * mant * 2f32.powi(-9)
    } else {
        sign * (8.0 + mant) * 2f32.powi(i32::from(field) - 10)
    }
}

pub fn i8_encode(x: f32) -> u8 {
    round_even(f64::from(x)).clamp(-127.0, 127.0) as i8 as u8
}

/// Permutation of 0..n from an LCG over a power-of-two modulus (multiplier
/// 5 mod 8 gives a full cycle); out-of-range draws are rejected.
pub fn lcg_perm(n: usize, seed: u64) -> Vec<usize> {
    let modulus = n.next_power_of_two() as u64;
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(n);
    let mut guard = 0usize;
    while out.len() < n {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let v = (state >> 40) % modulus;
        guard += 1;
        assert!(guard < 10_000_000, "lcg_perm stall");
        if (v as usize) < n && !out.contains(&(v as usize)) {
            out.push(v as usize);
        }
    }
    out
}

/// Elementwise tolerance: L2-relative ceiling, an absolute floor plus a
/// relative slope per element, and the fraction of elements allowed outside
/// the elementwise band (bf16 tensor-core accumulation leaves scattered
/// outliers even when the row norm is tight).
pub struct Tol {
    pub rel_l2: f64,
    pub slope: f64,
    pub floor: f64,
    pub max_viol_frac: f64,
}

pub struct Metrics {
    pub rel_l2: f64,
    pub viol_frac: f64,
    pub max_dev: f64,
}

/// Compare one flat row-major output `[rows, d]` against f64 expectations.
/// When `corrupt`, the first row's expectations are shifted, which must take
/// both the L2 and violation gates down (negative-control tooth).
#[allow(clippy::needless_range_loop)]
pub fn compare_rows(
    got: &[bf16],
    wants: &[Vec<f64>],
    n_rows: usize,
    d: usize,
    h_stride: usize,
    row_ids: &[(usize, usize)],
    tol: &Tol,
    corrupt: bool,
) -> Metrics {
    let mut diff_sq = 0f64;
    let mut ref_sq = 0f64;
    let mut violators = 0usize;
    let mut total = 0usize;
    let mut max_dev = 0f64;
    for (i, (token, h)) in row_ids.iter().enumerate().take(n_rows) {
        for dd in 0..d {
            let mut w = wants[i][dd];
            if corrupt && i == 0 {
                w += 0.5;
            }
            let g = f64::from(got[(*token * h_stride + h) * d + dd].to_f32());
            let dev = (g - w).abs();
            diff_sq += dev.powi(2);
            ref_sq += w.powi(2);
            max_dev = max_dev.max(dev);
            total += 1;
            if dev > tol.floor + tol.slope * w.abs() {
                violators += 1;
            }
        }
    }
    Metrics {
        rel_l2: (diff_sq / ref_sq.max(1e-12)).sqrt(),
        viol_frac: violators as f64 / total as f64,
        max_dev,
    }
}

pub fn metrics_pass(m: &Metrics, tol: &Tol) -> bool {
    m.rel_l2 < tol.rel_l2 && m.viol_frac <= tol.max_viol_frac
}

/// f64 softmax attention for one query row. `kv` holds the K/V values the
/// device math units consume, laid out `[lim, d]`; `lim` is this query
/// token's causal bound. The output gets the single final bf16 RN the
/// kernels apply.
#[allow(clippy::needless_range_loop)]
pub fn attention_row(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    lim: usize,
    d: usize,
    sm_scale: f64,
) -> Vec<f64> {
    let mut scores = vec![0f64; lim];
    for j in 0..lim {
        let mut dot = 0f64;
        for dd in 0..d {
            dot += q[dd] * k[j * d + dd];
        }
        scores[j] = dot * sm_scale;
    }
    let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
    let denom: f64 = exps.iter().sum();
    let mut out = vec![0f64; d];
    for dd in 0..d {
        let mut acc = 0f64;
        for j in 0..lim {
            acc += (exps[j] / denom) * v[j * d + dd];
        }
        out[dd] = f64::from(bf(acc as f32).to_f32());
    }
    out
}
