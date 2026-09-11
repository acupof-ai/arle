//! Parity gates for the hand-written Metal kernels shipped in mlx-sys.
//!
//! Every custom `fast::metal_kernel` family is compared against an f64 host
//! oracle at Qwen3.6-35B-A3B-4bit geometry (GDR: Hk=16 Hv=32 Dk=Dv=128;
//! full attention: Hq=16 Hk=2 D=256; 4-bit QMM group_size=64, K=2048).
//! Kernel inputs are bf16 on the production path; the oracle quantizes every
//! input through bf16 first so the residual measures kernel arithmetic, not
//! input cast error. The `negative_controls` test perturbs each family and
//! asserts the corruption is detected.
#![cfg(target_os = "macos")]

use std::ptr;

use mlx_sys::{
    MLX_BFLOAT16, MLX_FLOAT32, MLX_UINT32, mlx_array_export_bytes, mlx_array_free,
    mlx_array_from_data, mlx_batched_sdpa_2pass, mlx_eval, mlx_guard, mlx_qwen35_gated_delta_step,
    mlx_tape_replay, mlx_verify_quantized_matmul,
};

// Canonical GDR geometry (linear_* in Qwen3.6-35B-A3B-4bit config.json).
const HK: usize = 16;
const HV: usize = 32;
const DK: usize = 128;
const DV: usize = 128;
// Canonical full-attention geometry.
const HQ: usize = 16;
const HKA: usize = 2;
const DA: usize = 256;
const Q_LEN: usize = 16;

struct Arr(*mut mlx_sys::mlx_array);
impl Drop for Arr {
    fn drop(&mut self) {
        // SAFETY: handle was returned by the bridge; FFI takes ownership.
        unsafe { mlx_array_free(self.0) }
    }
}

fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    let lsb = (b >> 16) & 1;
    ((b.wrapping_add(0x7fff).wrapping_add(lsb)) >> 16) as u16
}
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }
}

fn arr_f32(data: &[f32], shape: &[i32]) -> Arr {
    // SAFETY: valid caller buffer + shape; bridge copies.
    unsafe {
        Arr(mlx_array_from_data(
            data.as_ptr().cast(),
            shape.as_ptr(),
            shape.len() as i32,
            MLX_FLOAT32,
        ))
    }
}
fn arr_bf16(data: &[f32], shape: &[i32]) -> (Arr, Vec<u16>) {
    let bits: Vec<u16> = data.iter().map(|x| f32_to_bf16(*x)).collect();
    // SAFETY: live bf16 buffer + valid shape; the bridge copies the bytes.
    let a = unsafe {
        Arr(mlx_array_from_data(
            bits.as_ptr().cast(),
            shape.as_ptr(),
            shape.len() as i32,
            MLX_BFLOAT16,
        ))
    };
    (a, bits)
}
fn arr_u32(data: &[u32], shape: &[i32]) -> Arr {
    // SAFETY: live u32 buffer + valid shape; the bridge copies the bytes.
    unsafe {
        Arr(mlx_array_from_data(
            data.as_ptr().cast(),
            shape.as_ptr(),
            shape.len() as i32,
            MLX_UINT32,
        ))
    }
}
fn eval(a: &Arr) {
    // SAFETY: one live handle, null-terminated array not needed for count 1.
    unsafe { mlx_eval(&a.0 as *const _ as *mut _, 1) }
}
fn read_f32(a: &Arr, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    // SAFETY: buffer is n*4 bytes; the bridge writes at most that.
    let copied = unsafe { mlx_array_export_bytes(a.0, out.as_mut_ptr().cast(), n * 4) };
    assert_eq!(copied, n * 4);
    out
}
fn read_bf16(a: &Arr, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 2];
    // SAFETY: buffer is n*2 bytes; the bridge writes at most that.
    let copied = unsafe { mlx_array_export_bytes(a.0, bytes.as_mut_ptr().cast(), n * 2) };
    assert_eq!(copied, n * 2);
    (0..n)
        .map(|i| bf16_to_f32(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])))
        .collect()
}

fn err_ratio(got: &[f32], want: &[f64]) -> f64 {
    let scale = (want.iter().map(|v| v * v).sum::<f64>() / want.len() as f64)
        .sqrt()
        .max(1e-8);
    got.iter()
        .zip(want)
        .map(|(g, w)| (*g as f64 - w).abs())
        .fold(0f64, f64::max)
        / scale
}

// ---- GDR gated-delta-step f64 oracle -------------------------------------
// Layouts (production kernel contract):
// q,k [B,T,Hk,Dk], v [B,T,Hv,Dv], g,beta [B,T,Hv], state [B,Hv,Dv,Dk].
// Returns (y [B,T,Hv,Dv], state_out, tape [B,T,Hv,Dv]).
fn gated_delta_oracle(
    q: &[u16],
    k: &[u16],
    v: &[u16],
    g: &[f32],
    beta: &[f32],
    state: &[f32],
    b: usize,
    t: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let qv = |b0, t0, h, d| bf16_to_f32(q[((b0 * t + t0) * HK + h) * DK + d]) as f64;
    let kv = |b0, t0, h, d| bf16_to_f32(k[((b0 * t + t0) * HK + h) * DK + d]) as f64;
    let vv = |b0, t0, h, d| bf16_to_f32(v[((b0 * t + t0) * HV + h) * DV + d]) as f64;
    let mut st: Vec<f64> = state.iter().map(|x| *x as f64).collect();
    let mut ys = Vec::with_capacity(b * t * HV * DV);
    let mut tape = Vec::with_capacity(b * t * HV * DV);
    for b0 in 0..b {
        for t0 in 0..t {
            for hv in 0..HV {
                let hk = hv / (HV / HK);
                let gg = g[(b0 * t + t0) * HV + hv] as f64;
                let bb = beta[(b0 * t + t0) * HV + hv] as f64;
                for dv in 0..DV {
                    let base = ((b0 * HV + hv) * DV + dv) * DK;
                    let mut kv_mem = 0f64;
                    for dk in 0..DK {
                        st[base + dk] *= gg;
                        kv_mem += st[base + dk] * kv(b0, t0, hk, dk);
                    }
                    let delta = (vv(b0, t0, hv, dv) - kv_mem) * bb;
                    tape.push(delta);
                    let mut y = 0f64;
                    for dk in 0..DK {
                        st[base + dk] += kv(b0, t0, hk, dk) * delta;
                        y += st[base + dk] * qv(b0, t0, hk, dk);
                    }
                    ys.push(y);
                }
            }
        }
    }
    (ys, st, tape)
}

// f64 replay oracle used by the rollback kernel gate: state update driven by
// the bf16-rounded innovation tape (what the kernel actually consumes).
fn replay_oracle(
    tape: &[u16],
    k: &[u16],
    g: &[f32],
    state: &[f32],
    b: usize,
    t: usize,
) -> Vec<f64> {
    let mut st: Vec<f64> = state.iter().map(|x| *x as f64).collect();
    for b0 in 0..b {
        for t0 in 0..t {
            for hv in 0..HV {
                let hk = hv / (HV / HK);
                let gg = g[(b0 * t + t0) * HV + hv] as f64;
                for dv in 0..DV {
                    let base = ((b0 * HV + hv) * DV + dv) * DK;
                    let delta = bf16_to_f32(tape[((b0 * t + t0) * HV + hv) * DV + dv]) as f64;
                    for dk in 0..DK {
                        st[base + dk] = st[base + dk] * gg
                            + bf16_to_f32(k[((b0 * t + t0) * HK + hk) * DK + dk]) as f64 * delta;
                    }
                }
            }
        }
    }
    st
}

struct GdrInputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
    state: Vec<f32>,
}
fn make_gdr_inputs(b: usize, t: usize, seed: u64) -> GdrInputs {
    let mut r = Rng(seed);
    GdrInputs {
        q: (0..b * t * HK * DK).map(|_| r.next() * 0.3).collect(),
        k: (0..b * t * HK * DK).map(|_| r.next() * 0.3).collect(),
        v: (0..b * t * HV * DV).map(|_| r.next() * 0.3).collect(),
        // g in (0.9, 1.0): the recurrent decay is near 1 on the real path.
        g: (0..b * t * HV)
            .map(|_| 0.9 + r.next().abs() * 0.1)
            .collect(),
        beta: (0..b * t * HV).map(|_| 0.5 + r.next().abs()).collect(),
        state: (0..b * HV * DV * DK).map(|_| r.next() * 0.05).collect(),
    }
}

struct GdrOut {
    y: Arr,
    state: Arr,
    tape: Option<Arr>,
}
fn run_gated(x: &GdrInputs, b: usize, t: usize, record_tape: bool) -> GdrOut {
    let (qa, _) = arr_bf16(&x.q, &[b as i32, t as i32, HK as i32, DK as i32]);
    let (ka, _) = arr_bf16(&x.k, &[b as i32, t as i32, HK as i32, DK as i32]);
    let (va, _) = arr_bf16(&x.v, &[b as i32, t as i32, HV as i32, DV as i32]);
    let ga = arr_f32(&x.g, &[b as i32, t as i32, HV as i32]);
    let ba = arr_f32(&x.beta, &[b as i32, t as i32, HV as i32]);
    let sa = arr_f32(&x.state, &[b as i32, HV as i32, DV as i32, DK as i32]);
    let (mut y, mut so, mut tp) = (ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
    // SAFETY: valid owned input handles; the extern fills the three out slots.
    unsafe {
        mlx_qwen35_gated_delta_step(
            qa.0,
            ka.0,
            va.0,
            ga.0,
            ba.0,
            sa.0,
            i32::from(record_tape),
            &mut y,
            &mut so,
            &mut tp,
        );
    }
    assert!(!y.is_null() && !so.is_null());
    GdrOut {
        y: Arr(y),
        state: Arr(so),
        tape: if record_tape {
            assert!(!tp.is_null());
            Some(Arr(tp))
        } else {
            None
        },
    }
}

#[test]
fn gated_delta_step_parity() {
    let _g = mlx_guard();
    for (t, y_tol, s_tol) in [(1usize, 0.02, 5e-4), (16, 0.05, 2e-2)] {
        let b = 1;
        let x = make_gdr_inputs(b, t, 0x9e37_79b9_7f4a_7c15);
        let (_, qb) = arr_bf16(&x.q, &[]);
        let (_, kb) = arr_bf16(&x.k, &[]);
        let (_, vb) = arr_bf16(&x.v, &[]);
        let (want_y, want_s, _) = gated_delta_oracle(&qb, &kb, &vb, &x.g, &x.beta, &x.state, b, t);
        let out = run_gated(&x, b, t, false);
        eval(&out.y);
        eval(&out.state);
        let ey = err_ratio(&read_bf16(&out.y, b * t * HV * DV), &want_y);
        let es = err_ratio(&read_f32(&out.state, b * HV * DV * DK), &want_s);
        println!("gated_delta T={t}: y err {ey:.3e} (<{y_tol}), state err {es:.3e} (<{s_tol})");
        assert!(ey < y_tol, "gated_delta y T={t}: {ey} >= {y_tol}");
        assert!(es < s_tol, "gated_delta state T={t}: {es} >= {s_tol}");
    }
}

#[test]
fn gated_delta_innovation_tape() {
    let _g = mlx_guard();
    let (b, t) = (1, 16);
    let x = make_gdr_inputs(b, t, 0x1234_5678_9abc_def0);
    let (_, qb) = arr_bf16(&x.q, &[]);
    let (_, kb) = arr_bf16(&x.k, &[]);
    let (_, vb) = arr_bf16(&x.v, &[]);
    let (_, _, want_f64) = gated_delta_oracle(&qb, &kb, &vb, &x.g, &x.beta, &x.state, b, t);
    let out = run_gated(&x, b, t, true);
    let tape = out.tape.as_ref().unwrap();
    eval(tape);
    // Kernel stores bf16-rounded delta; compare to the same rounding of the
    // f64 oracle.
    let got = read_bf16(tape, b * t * HV * DV);
    let want: Vec<f64> = want_f64
        .iter()
        .map(|d| bf16_to_f32(f32_to_bf16(*d as f32)) as f64)
        .collect();
    let e = err_ratio(&got, &want);
    println!("gated_delta tape err {e:.3e}");
    assert!(e < 0.02, "tape parity: {e}");
}

#[test]
fn tape_replay_reconstructs_state() {
    let _g = mlx_guard();
    let (b, t) = (1, 16);
    let x = make_gdr_inputs(b, t, 0xfeed_f00d_cafe_babe);
    let forward = run_gated(&x, b, t, true);
    let tape = forward.tape.as_ref().unwrap();
    eval(&forward.y);
    eval(&forward.state);
    eval(tape);
    let tape_bf16 = read_bf16(tape, b * t * HV * DV);
    let tape_bits: Vec<u16> = tape_bf16.iter().map(|v| f32_to_bf16(*v)).collect();
    let (_, kb) = arr_bf16(&x.k, &[]);
    let want = replay_oracle(&tape_bits, &kb, &x.g, &x.state, b, t);

    let (ka, _) = arr_bf16(&x.k, &[b as i32, t as i32, HK as i32, DK as i32]);
    let ga = arr_f32(&x.g, &[b as i32, t as i32, HV as i32]);
    let sa = arr_f32(&x.state, &[b as i32, HV as i32, DV as i32, DK as i32]);
    // SAFETY: valid owned handles and matching T; bridge returns an owned array.
    let replay = unsafe { Arr(mlx_tape_replay(tape.0, ka.0, ga.0, sa.0, t as i32)) };
    assert!(!replay.0.is_null());
    eval(&replay);
    let e = err_ratio(&read_f32(&replay, b * HV * DV * DK), &want);
    println!("tape_replay state err {e:.3e}");
    assert!(e < 1e-5, "tape_replay state: {e}");
}

// ---- 2-pass batched SDPA f64 oracle --------------------------------------
// queries [B,Hq,16,D], keys/values [B,Hk,N,D]. The kernel's mask lets query
// row qi attend key n when n <= N-16+qi (causal packed chunk).
fn sdpa_oracle(q: &[u16], k: &[u16], v: &[u16], n: usize, scale: f64) -> Vec<f64> {
    let qi = |h, row, d| bf16_to_f32(q[(h * Q_LEN + row) * DA + d]) as f64;
    let ki = |h, key, d| bf16_to_f32(k[(h * n + key) * DA + d]) as f64;
    let vi = |h, key, d| bf16_to_f32(v[(h * n + key) * DA + d]) as f64;
    let mut out = vec![0f64; HQ * Q_LEN * DA];
    for h in 0..HQ {
        let hk = h / (HQ / HKA);
        for row in 0..Q_LEN {
            let last = n - Q_LEN + row;
            let scores: Vec<f64> = (0..=last)
                .map(|key| (0..DA).map(|d| qi(h, row, d) * ki(hk, key, d)).sum::<f64>() * scale)
                .collect();
            let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let ex: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
            let z: f64 = ex.iter().sum();
            for d in 0..DA {
                let acc: f64 = (0..=last).map(|key| ex[key] * vi(hk, key, d)).sum();
                out[(h * Q_LEN + row) * DA + d] = acc / z;
            }
        }
    }
    out
}

fn run_sdpa(n: usize, corrupt_v: bool) -> (Vec<f32>, Vec<f64>) {
    let mut r = Rng(0x51ab_5eed_5000_0001);
    let q: Vec<f32> = (0..HQ * Q_LEN * DA).map(|_| r.next() * 0.2).collect();
    let mut k: Vec<f32> = (0..HKA * n * DA).map(|_| r.next() * 0.2).collect();
    let mut v: Vec<f32> = (0..HKA * n * DA).map(|_| r.next() * 0.2).collect();
    if corrupt_v {
        // Every query row attends key 0 (last >= 0), so a key-0 value spike
        // changes the whole output.
        v[0] += 4.0;
        k[0] += 0.0;
    }
    let (qa, qb) = arr_bf16(&q, &[1, HQ as i32, Q_LEN as i32, DA as i32]);
    let (ka, kb) = arr_bf16(&k, &[1, HKA as i32, n as i32, DA as i32]);
    let (va, vb) = arr_bf16(&v, &[1, HKA as i32, n as i32, DA as i32]);
    let want = sdpa_oracle(&qb, &kb, &vb, n, 1.0 / (DA as f64).sqrt());
    // SAFETY: handles match the kernel's [B,H,S,D] bf16 contract.
    let out = unsafe {
        Arr(mlx_batched_sdpa_2pass(
            qa.0,
            ka.0,
            va.0,
            1.0 / (DA as f32).sqrt(),
            (HQ / HKA) as i32,
        ))
    };
    assert!(!out.0.is_null());
    eval(&out);
    (read_bf16(&out, HQ * Q_LEN * DA), want)
}

#[test]
fn batched_sdpa_2pass_causal() {
    let _g = mlx_guard();
    // N>128 exercises the partials kernel's strided `n += blocks` loop.
    for n in [64, 128, 192] {
        let (got, want) = run_sdpa(n, false);
        let e = err_ratio(&got, &want);
        println!("sdpa_2pass N={n}: err {e:.3e}");
        assert!(e < 0.05, "sdpa_2pass N={n}: {e}");
    }
}

// ---- fixed-M=16 4-bit quantized matmul f64 oracle ------------------------
// w packed [N, K/8] u32, nibble k occupies bits (k%8)*4; dequant =
// nib*scale+bias per group of group_size along K. y = x @ dequant(w)^T.
fn qmm_oracle(
    x: &[u16],
    packed: &[u32],
    scales: &[f32],
    biases: &[f32],
    k: usize,
    n: usize,
    gs: usize,
) -> Vec<f64> {
    let kg = k / gs;
    let mut out = vec![0f64; 16 * n];
    for j in 0..n {
        for i in 0..16 {
            let mut acc = 0f64;
            for kk in 0..k {
                let nib = f64::from((packed[j * (k / 8) + kk / 8] >> ((kk % 8) * 4)) & 0xF);
                let w = nib * scales[j * kg + kk / gs] as f64 + biases[j * kg + kk / gs] as f64;
                acc += bf16_to_f32(x[i * k + kk]) as f64 * w;
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn run_qmm(gs: usize, n: usize, corrupt_scale: bool) -> (Vec<f32>, Vec<f64>) {
    const K: usize = 2048;
    let mut r = Rng(0x71aa_0000_0000_0001 | gs as u64);
    let xf: Vec<f32> = (0..16 * K).map(|_| r.next() * 0.2).collect();
    let nibbles: Vec<u8> = (0..n * K).map(|_| (r.0 >> 60) as u8 & 0xF).collect();
    let kg = K / gs;
    let mut scales: Vec<f32> = (0..n * kg).map(|_| 0.02 + r.next().abs() * 0.05).collect();
    let biases: Vec<f32> = (0..n * kg).map(|_| r.next() * 0.02).collect();
    if corrupt_scale {
        // First group of output 0 feeds all 16 rows.
        scales[0] += 1.0;
    }
    let mut packed = vec![0u32; n * (K / 8)];
    for j in 0..n {
        for kk in 0..K {
            packed[j * (K / 8) + kk / 8] |= u32::from(nibbles[j * K + kk]) << ((kk % 8) * 4);
        }
    }
    let (xa, xb) = arr_bf16(&xf, &[16, K as i32]);
    let wa = arr_u32(&packed, &[n as i32, (K / 8) as i32]);
    let sa = arr_f32(&scales, &[n as i32, kg as i32]);
    let ba = arr_f32(&biases, &[n as i32, kg as i32]);
    let want = qmm_oracle(&xb, &packed, &scales, &biases, K, n, gs);
    // SAFETY: M=16 x, packed [N,K/8] u32, per-group f32 scale/bias shapes.
    let out = unsafe {
        Arr(mlx_verify_quantized_matmul(
            xa.0, wa.0, sa.0, ba.0, gs as i32, 4,
        ))
    };
    assert!(!out.0.is_null());
    eval(&out);
    (read_bf16(&out, 16 * n), want)
}

#[test]
fn verify_qmm_mma2big_group_sizes() {
    let _g = mlx_guard();
    for (gs, n) in [(32usize, 512usize), (64, 2048), (128, 512)] {
        let (got, want) = run_qmm(gs, n, false);
        let e = err_ratio(&got, &want);
        println!("verify_qmm gs={gs} N={n}: err {e:.3e}");
        assert!(e < 0.05, "verify_qmm gs={gs}: {e}");
    }
}

// One corruption per kernel family; each must clear the clean-run error by a
// wide margin.
#[test]
fn negative_controls() {
    let _g = mlx_guard();
    // gated_delta: spike one k element.
    {
        let (b, t) = (1, 4);
        let seed = 0x9e37_79b9_7f4a_7c15;
        let mut bad = make_gdr_inputs(b, t, seed);
        bad.k[0] += 1.0;
        let out = run_gated(&bad, b, t, false);
        eval(&out.y);
        let clean = make_gdr_inputs(b, t, seed);
        let (_, qb) = arr_bf16(&clean.q, &[]);
        let (_, kb) = arr_bf16(&clean.k, &[]);
        let (_, vb) = arr_bf16(&clean.v, &[]);
        let (want, _, _) =
            gated_delta_oracle(&qb, &kb, &vb, &clean.g, &clean.beta, &clean.state, b, t);
        let e = err_ratio(&read_bf16(&out.y, b * t * HV * DV), &want);
        println!("neg gated_delta: {e:.3e}");
        assert!(e > 0.2, "gated_delta corruption not detected: {e}");
    }
    // innovation tape: spike one v element.
    {
        let (b, t) = (1, 4);
        let seed = 0x1234_5678_9abc_def0;
        let mut bad = make_gdr_inputs(b, t, seed);
        bad.v[0] += 2.0;
        let out = run_gated(&bad, b, t, true);
        let tape = out.tape.as_ref().unwrap();
        eval(tape);
        let clean = make_gdr_inputs(b, t, seed);
        let (_, qb) = arr_bf16(&clean.q, &[]);
        let (_, kb) = arr_bf16(&clean.k, &[]);
        let (_, vb) = arr_bf16(&clean.v, &[]);
        let (_, _, want_f) =
            gated_delta_oracle(&qb, &kb, &vb, &clean.g, &clean.beta, &clean.state, b, t);
        let want: Vec<f64> = want_f
            .iter()
            .map(|d| bf16_to_f32(f32_to_bf16(*d as f32)) as f64)
            .collect();
        let e = err_ratio(&read_bf16(tape, b * t * HV * DV), &want);
        println!("neg tape: {e:.3e}");
        assert!(e > 0.2, "tape corruption not detected: {e}");
    }
    // tape_replay: flip the high byte of the first bf16 tape entry.
    {
        let (b, t) = (1, 4);
        let x = make_gdr_inputs(b, t, 0xfeed_f00d_cafe_babe);
        let forward = run_gated(&x, b, t, true);
        let tape = forward.tape.as_ref().unwrap();
        eval(tape);
        let mut bytes = vec![0u8; b * t * HV * DV * 2];
        // SAFETY: buffer length equals the bf16 tape byte count.
        unsafe {
            mlx_array_export_bytes(tape.0, bytes.as_mut_ptr().cast(), bytes.len());
        }
        bytes[1] ^= 0xFF;
        let bad_tape = arr_bf16_from_bytes(&bytes, &[b as i32, t as i32, HV as i32, DV as i32]);
        let (ka, kb) = arr_bf16(&x.k, &[b as i32, t as i32, HK as i32, DK as i32]);
        let ga = arr_f32(&x.g, &[b as i32, t as i32, HV as i32]);
        let sa = arr_f32(&x.state, &[b as i32, HV as i32, DV as i32, DK as i32]);
        // Oracle consumes the un-corrupted bf16 tape.
        let good_bytes = {
            let mut g = bytes.clone();
            g[1] ^= 0xFF;
            g
        };
        let good_bits: Vec<u16> = (0..b * t * HV * DV)
            .map(|i| u16::from_le_bytes([good_bytes[2 * i], good_bytes[2 * i + 1]]))
            .collect();
        let want = replay_oracle(&good_bits, &kb, &x.g, &x.state, b, t);
        // SAFETY: corrupted-but-valid bf16 tape and matching k/g/state handles.
        let replay = unsafe { Arr(mlx_tape_replay(bad_tape.0, ka.0, ga.0, sa.0, t as i32)) };
        eval(&replay);
        let e = err_ratio(&read_f32(&replay, b * HV * DV * DK), &want);
        println!("neg replay: {e:.3e}");
        assert!(e > 0.1, "replay corruption not detected: {e}");
    }
    // 2-pass SDPA: value spike attended by every query row.
    {
        let (clean, wclean) = run_sdpa(128, false);
        let (bad, _) = run_sdpa(128, true);
        let ec = err_ratio(&clean, &wclean);
        let eb = err_ratio(&bad, &wclean);
        println!("neg sdpa: clean {ec:.3e} corrupt {eb:.3e}");
        assert!(
            eb > 0.2 && eb > 10.0 * ec,
            "sdpa corruption not detected: {eb}"
        );
    }
    // mma2big QMM at every supported group size: scale spike.
    for gs in [32usize, 64, 128] {
        let (clean, wclean) = run_qmm(gs, 512, false);
        let (bad, _) = run_qmm(gs, 512, true);
        let ec = err_ratio(&clean, &wclean);
        let eb = err_ratio(&bad, &wclean);
        println!("neg qmm gs={gs}: clean {ec:.3e} corrupt {eb:.3e}");
        assert!(
            eb > 0.1 && eb > 10.0 * ec,
            "qmm gs={gs} corruption not detected: {eb}"
        );
    }
}

fn arr_bf16_from_bytes(bytes: &[u8], shape: &[i32]) -> Arr {
    // SAFETY: live bf16 byte buffer + valid shape; the bridge copies.
    unsafe {
        Arr(mlx_array_from_data(
            bytes.as_ptr().cast(),
            shape.as_ptr(),
            shape.len() as i32,
            MLX_BFLOAT16,
        ))
    }
}
