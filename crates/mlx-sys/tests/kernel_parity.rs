//! Parity gates for the hand-written Metal kernels shipped in mlx-sys.
//!
//! Every custom `fast::metal_kernel` family is compared against an f64 host
//! oracle at Qwen3.6-35B-A3B-4bit geometry (GDR: Hk=16 Hv=32 Dk=Dv=128;
//! full attention: Hq=16 Hk=2 D=256; 4-bit QMM group_size=64, K=2048).
//! Kernel inputs are bf16 on the production path, including the checkpoint's
//! per-group QMM scales/biases (BF16 in the safetensors); the oracle rounds
//! every input through bf16 first so the residual measures kernel
//! arithmetic, not input cast error. The `negative_controls` test perturbs
//! each family and asserts the corruption is detected.
#![cfg(target_os = "macos")]

use std::ptr;

use mlx_sys::{
    MLX_BFLOAT16, MLX_FLOAT32, MLX_UINT32, mlx_array_export_bytes, mlx_array_free,
    mlx_array_from_data, mlx_batched_sdpa_2pass, mlx_dequantize, mlx_eval, mlx_guard, mlx_quantize,
    mlx_qwen35_gated_delta_step, mlx_tape_replay, mlx_verify_quantized_matmul,
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
// int8 KV group for head_dim 256 (kv_int8_group_size in the model TU).
const KV_GROUP: i32 = 128;

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
    let a = arr_bf16_from_bits(&bits, shape);
    (a, bits)
}
fn arr_bf16_from_bits(bits: &[u16], shape: &[i32]) -> Arr {
    // SAFETY: live bf16 buffer + valid shape; the bridge copies the bytes.
    unsafe {
        Arr(mlx_array_from_data(
            bits.as_ptr().cast(),
            shape.as_ptr(),
            shape.len() as i32,
            MLX_BFLOAT16,
        ))
    }
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
    bytes_to_bf16(&bytes)
}
fn bytes_to_bf16(bytes: &[u8]) -> Vec<f32> {
    (0..bytes.len() / 2)
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
    // Bounds are ~3x the measured clean error. State error stays ~1e-6
    // relative-RMS at both T values: every accumulator in the kernel is
    // f32 and only the bf16 inputs round, so it does not grow with T.
    for &t in &[1usize, 16] {
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
        println!("gated_delta B={b} T={t}: y err {ey:.3e}, state err {es:.3e}");
        assert!(ey < 0.03, "gated_delta y B={b} T={t}: {ey}");
        assert!(es < 3e-6, "gated_delta state B={b} T={t}: {es}");
    }
}

#[test]
fn gated_delta_batched_rows() {
    let _g = mlx_guard();
    // Batched B>1 never reaches this kernel on the production Metal path
    // (every FFI entry pins batch_size=1 and the executor calls step_session
    // per slot), but the kernel's z-grid is B*Hv: pin the cross-row
    // addressing with B=4 and an independent state per row.
    let (b, t) = (4, 4);
    let x = make_gdr_inputs(b, t, 0x4b61_7463_6800_0004);
    let (_, qb) = arr_bf16(&x.q, &[]);
    let (_, kb) = arr_bf16(&x.k, &[]);
    let (_, vb) = arr_bf16(&x.v, &[]);
    let (want_y, want_s, _) = gated_delta_oracle(&qb, &kb, &vb, &x.g, &x.beta, &x.state, b, t);
    let out = run_gated(&x, b, t, false);
    eval(&out.y);
    eval(&out.state);
    let ey = err_ratio(&read_bf16(&out.y, b * t * HV * DV), &want_y);
    let es = err_ratio(&read_f32(&out.state, b * HV * DV * DK), &want_s);
    println!("gated_delta B={b} T={t}: y err {ey:.3e}, state err {es:.3e}");
    assert!(ey < 0.03, "gated_delta y B={b}: {ey}");
    assert!(es < 3e-6, "gated_delta state B={b}: {es}");
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
    // The kernel stores the delta as bf16; compare to the same rounding of
    // the f64 oracle. That rounding is the error mechanism and the bound.
    let got = read_bf16(tape, b * t * HV * DV);
    let want: Vec<f64> = want_f64
        .iter()
        .map(|d| bf16_to_f32(f32_to_bf16(*d as f32)) as f64)
        .collect();
    let e = err_ratio(&got, &want);
    println!("gated_delta tape err {e:.3e}");
    assert!(e < 0.012, "tape parity: {e}");
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
    assert!(e < 2.5e-6, "tape_replay state: {e}");
}

// ---- 2-pass batched SDPA f64 oracle --------------------------------------
// queries [B,Hq,16,D], keys/values [B,Hk,N,D]. The kernel's mask lets query
// row qi attend key n when n <= N-16+qi (causal packed chunk).
fn sdpa_oracle(q: &[u16], k: &[u16], v: &[u16], b: usize, n: usize, scale: f64) -> Vec<f64> {
    let qi = |b0, h, row, d| bf16_to_f32(q[((b0 * HQ + h) * Q_LEN + row) * DA + d]) as f64;
    let ki = |b0, h, key, d| bf16_to_f32(k[((b0 * HKA + h) * n + key) * DA + d]) as f64;
    let vi = |b0, h, key, d| bf16_to_f32(v[((b0 * HKA + h) * n + key) * DA + d]) as f64;
    let mut out = vec![0f64; b * HQ * Q_LEN * DA];
    for b0 in 0..b {
        for h in 0..HQ {
            let hk = h / (HQ / HKA);
            for row in 0..Q_LEN {
                let last = n - Q_LEN + row;
                let scores: Vec<f64> = (0..=last)
                    .map(|key| {
                        (0..DA)
                            .map(|d| qi(b0, h, row, d) * ki(b0, hk, key, d))
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let ex: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let z: f64 = ex.iter().sum();
                for d in 0..DA {
                    let acc: f64 = (0..=last).map(|key| ex[key] * vi(b0, hk, key, d)).sum();
                    out[((b0 * HQ + h) * Q_LEN + row) * DA + d] = acc / z;
                }
            }
        }
    }
    out
}

// Round k/v through the production int8 KV path: MLX affine quantize
// (bits=8, group 128 on head_dim 256) then dequantize back to bf16. The
// model feeds exactly these dequantized arrays to batched_sdpa_2pass_cpp.
fn int8_kv_roundtrip(k: Arr, v: Arr, b: usize, n: usize) -> (Arr, Arr, Vec<u16>, Vec<u16>) {
    let (mut kq, mut ks, mut kb) = (ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
    let (mut vq, mut vs, mut vb) = (ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
    // SAFETY: bf16 [B,H,N,256] inputs; last axis 256 % group 128 == 0.
    unsafe {
        mlx_quantize(k.0, KV_GROUP, 8, &mut kq, &mut ks, &mut kb);
        mlx_quantize(v.0, KV_GROUP, 8, &mut vq, &mut vs, &mut vb);
    }
    assert!(!kq.is_null() && !vq.is_null());
    // Keep the quantize triples alive for the dequantize calls and free them
    // when the function returns.
    let (kq_a, ks_a, kb_a) = (Arr(kq), Arr(ks), Arr(kb));
    let (vq_a, vs_a, vb_a) = (Arr(vq), Arr(vs), Arr(vb));
    // SAFETY: both triples came from the matching quantize calls above.
    let (kd, vd) = unsafe {
        (
            Arr(mlx_dequantize(kq_a.0, ks_a.0, kb_a.0, KV_GROUP, 8, 0)),
            Arr(mlx_dequantize(vq_a.0, vs_a.0, vb_a.0, KV_GROUP, 8, 0)),
        )
    };
    assert!(!kd.0.is_null() && !vd.0.is_null());
    eval(&kd);
    eval(&vd);
    let kout = read_bf16(&kd, b * HKA * n * DA);
    let vout = read_bf16(&vd, b * HKA * n * DA);
    let kbits: Vec<u16> = kout.iter().map(|z| f32_to_bf16(*z)).collect();
    let vbits: Vec<u16> = vout.iter().map(|z| f32_to_bf16(*z)).collect();
    (kd, vd, kbits, vbits)
}

struct SdpaCase {
    b: usize,
    n: usize,
    int8_kv: bool,
}
fn run_sdpa(case: &SdpaCase, seed: u64, corrupt_v: bool) -> (Vec<f32>, Vec<f64>) {
    let (b, n) = (case.b, case.n);
    let mut r = Rng(seed);
    let q: Vec<f32> = (0..b * HQ * Q_LEN * DA).map(|_| r.next() * 0.2).collect();
    let k: Vec<f32> = (0..b * HKA * n * DA).map(|_| r.next() * 0.2).collect();
    let mut v: Vec<f32> = (0..b * HKA * n * DA).map(|_| r.next() * 0.2).collect();
    let (qa, qb) = arr_bf16(&q, &[b as i32, HQ as i32, Q_LEN as i32, DA as i32]);

    let (ka, va, kb, vb) = if case.int8_kv {
        let (ka0, _) = arr_bf16(&k, &[b as i32, HKA as i32, n as i32, DA as i32]);
        let (va0, _) = arr_bf16(&v, &[b as i32, HKA as i32, n as i32, DA as i32]);
        let (kda, vda, kout, vout) = int8_kv_roundtrip(ka0, va0, b, n);
        (kda, vda, kout, vout)
    } else {
        let (ka, kb) = arr_bf16(&k, &[b as i32, HKA as i32, n as i32, DA as i32]);
        let (va, vb) = arr_bf16(&v, &[b as i32, HKA as i32, n as i32, DA as i32]);
        (ka, va, kb, vb)
    };

    if corrupt_v {
        // Every query row attends key 0 (last >= 0); spike key-0's value.
        v[0] += 4.0;
    }
    let v_for_kernel: Option<Arr> = if corrupt_v {
        let (vx, _) = arr_bf16(&v, &[b as i32, HKA as i32, n as i32, DA as i32]);
        Some(vx)
    } else {
        None
    };
    let v_handle = v_for_kernel.as_ref().unwrap_or(&va);

    let want = sdpa_oracle(&qb, &kb, &vb, b, n, 1.0 / (DA as f64).sqrt());
    // SAFETY: handles match the kernel's [B,H,S,D] bf16 contract.
    let out = unsafe {
        Arr(mlx_batched_sdpa_2pass(
            qa.0,
            ka.0,
            v_handle.0,
            1.0 / (DA as f32).sqrt(),
            (HQ / HKA) as i32,
        ))
    };
    assert!(!out.0.is_null());
    eval(&out);
    (read_bf16(&out, b * HQ * Q_LEN * DA), want)
}

#[test]
fn batched_sdpa_2pass_causal() {
    let _g = mlx_guard();
    // Production N = cache_pos + S is arbitrary: unaligned N (<128 leaves
    // most of the 128 partial blocks empty), long N (multiple block rounds),
    // and B=4 batched verify. One case runs K/V through the int8 KV
    // quantize/dequantize path the model uses for cached int8 attention.
    let cases = [
        SdpaCase {
            b: 1,
            n: 16,
            int8_kv: false,
        },
        SdpaCase {
            b: 1,
            n: 100,
            int8_kv: false,
        },
        SdpaCase {
            b: 1,
            n: 129,
            int8_kv: false,
        },
        SdpaCase {
            b: 1,
            n: 1000,
            int8_kv: false,
        },
        SdpaCase {
            b: 1,
            n: 4113,
            int8_kv: false,
        },
        SdpaCase {
            b: 4,
            n: 1000,
            int8_kv: false,
        },
        SdpaCase {
            b: 1,
            n: 1000,
            int8_kv: true,
        },
    ];
    for (i, case) in cases.iter().enumerate() {
        let (got, want) = run_sdpa(case, 0x51ab_5eed_5000_0001 + i as u64, false);
        let e = err_ratio(&got, &want);
        println!(
            "sdpa_2pass B={} N={} int8_kv={}: err {e:.3e}",
            case.b, case.n, case.int8_kv
        );
        // Bound ~3x the measured clean error across the long/quantized cases.
        assert!(
            e < 0.04,
            "sdpa_2pass {:?}: {e}",
            (case.b, case.n, case.int8_kv)
        );
    }
}

// ---- fixed-M=16 4-bit quantized matmul f64 oracle ------------------------
// w packed [N, K/8] u32, nibble k occupies bits (k%8)*4; dequant =
// nib*scale+bias per group of group_size along K. y = x @ dequant(w)^T.
// scales/biases are bf16, matching the checkpoint's BF16 quantization
// tensors; the oracle receives their bf16-rounded values.
fn qmm_oracle(
    x: &[u16],
    packed: &[u32],
    scales: &[u16],
    biases: &[u16],
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
                let w = nib * bf16_to_f32(scales[j * kg + kk / gs]) as f64
                    + bf16_to_f32(biases[j * kg + kk / gs]) as f64;
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
    // Production dtype: checkpoint scales/biases are BF16.
    let (sa, sb) = arr_bf16(&scales, &[n as i32, kg as i32]);
    let (ba, bb) = arr_bf16(&biases, &[n as i32, kg as i32]);
    let want = qmm_oracle(&xb, &packed, &sb, &bb, K, n, gs);
    // SAFETY: M=16 x, packed [N,K/8] u32, per-group bf16 scale/bias shapes.
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
        assert!(e < 0.04, "verify_qmm gs={gs}: {e}");
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
        let bad_tape = arr_bf16_from_bits(
            &bytes_to_bits(&bytes),
            &[b as i32, t as i32, HV as i32, DV as i32],
        );
        let (ka, kb) = arr_bf16(&x.k, &[b as i32, t as i32, HK as i32, DK as i32]);
        let ga = arr_f32(&x.g, &[b as i32, t as i32, HV as i32]);
        let sa = arr_f32(&x.state, &[b as i32, HV as i32, DV as i32, DK as i32]);
        // Oracle consumes the un-corrupted bf16 tape.
        let mut good_bytes = bytes.clone();
        good_bytes[1] ^= 0xFF;
        let good_bits = bytes_to_bits(&good_bytes);
        let want = replay_oracle(&good_bits, &kb, &x.g, &x.state, b, t);
        // SAFETY: corrupted-but-valid bf16 tape and matching k/g/state handles.
        let replay = unsafe { Arr(mlx_tape_replay(bad_tape.0, ka.0, ga.0, sa.0, t as i32)) };
        eval(&replay);
        let e = err_ratio(&read_f32(&replay, b * HV * DV * DK), &want);
        println!("neg replay: {e:.3e}");
        assert!(e > 0.1, "replay corruption not detected: {e}");
    }
    // 2-pass SDPA: value spike attended by every query row (B=1, N=1000).
    {
        let case = SdpaCase {
            b: 1,
            n: 1000,
            int8_kv: false,
        };
        let (clean, wclean) = run_sdpa(&case, 0x51ab_5eed_5000_0099, false);
        let (bad, _) = run_sdpa(&case, 0x51ab_5eed_5000_0099, true);
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

fn bytes_to_bits(bytes: &[u8]) -> Vec<u16> {
    (0..bytes.len() / 2)
        .map(|i| u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]))
        .collect()
}
