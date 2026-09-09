//! On-device proof of the chunked (WY-form) gated-delta prefill pair
//! (`qwen4_gdn_chunk_intra` + `qwen4_gdn_chunk_state`).
//!
//! Two oracles, two claims:
//! - a host transcription of the CHUNKED math (same association order as the
//!   shaders) pins the kernels themselves — device vs host-chunked must agree
//!   to float-ULP level (the only differences are GPU `exp`/`inversesqrt`
//!   rounding and FMA contraction);
//! - the SERIAL per-token rule (transcribed from
//!   `qwen35_gated_delta_net.comp`) pins the operator — host-chunked vs
//!   host-serial differ only by the WY reassociation, and that drift envelope
//!   is asserted here so a math error (wrong decay mask, wrong state blend)
//!   reads as O(1), not as rounding.
//!
//! Shapes: a small one that exercises partial trailing chunks and multi-chunk
//! state propagation cheaply, the real qwen4_exp shape (nk=16, nv=48,
//! kd=vd=128), and a two-call continuation (state carried across dispatch
//! pairs — how the prefill actually uses the kernels).
//!
//! Runs only with `--features vulkan` + a device; skips cleanly otherwise.
#![cfg(feature = "vulkan")]

use vulkan_kernels::{
    Kernel, KernelCache, QWEN4_GDN_CHUNK, launch_cached, qwen4_gdn_chunk_intra_dispatch,
    qwen4_gdn_chunk_params, qwen4_gdn_chunk_scratch_elems, qwen4_gdn_chunk_state_dispatch,
};
use vulkan_sys::{DeviceBuffer, VulkanContext};

/// Deterministic xorshift PRNG so failures reproduce.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn upload_f32<'a>(ctx: &'a VulkanContext, data: &[f32]) -> DeviceBuffer<'a> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut b = DeviceBuffer::alloc(ctx, bytes.len().max(4)).expect("alloc f32 buffer");
    b.copy_from_host(&bytes).expect("upload f32 buffer");
    b
}

fn read_f32(buf: &DeviceBuffer<'_>, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    buf.copy_to_host(&mut bytes).expect("read back f32 buffer");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Clone, Copy)]
struct Shape {
    nk: usize,
    nv: usize,
    kd: usize,
    vd: usize,
}

impl Shape {
    fn qkv_stride(self) -> usize {
        2 * self.nk * self.kd + self.nv * self.vd
    }
}

/// Inputs for one sequence, in the exact device layouts.
struct Case {
    qkv: Vec<f32>,
    b_proj: Vec<f32>,
    a_proj: Vec<f32>,
    dt_bias: Vec<f32>,
    a_log: Vec<f32>,
    seq: usize,
}

fn random_case(rng: &mut Rng, sh: Shape, seq: usize) -> Case {
    // `a_log` is the GGUF `ssm_a` = -exp(A_log): strictly negative, spanning
    // slow (-0.02) to brutal (-6) per-token decay so exp(G) underflow inside
    // long chunks is exercised, not avoided.
    Case {
        qkv: (0..seq * sh.qkv_stride()).map(|_| rng.next_f32()).collect(),
        b_proj: (0..seq * sh.nv).map(|_| rng.next_f32() * 2.0).collect(),
        a_proj: (0..seq * sh.nv).map(|_| rng.next_f32()).collect(),
        dt_bias: (0..sh.nv).map(|_| rng.next_f32() * 0.5 + 0.5).collect(),
        a_log: (0..sh.nv)
            .map(|_| -(rng.next_f32() * 1.5 + 1.0).exp() * 0.5)
            .collect(),
        seq,
    }
}

/// The serial per-token rule, transcribed from `qwen35_gated_delta_net.comp`
/// (which the host oracle in `model_qwen4_exp.rs` matches byte for byte).
fn host_serial(sh: Shape, case: &Case, state: &mut [f32]) -> Vec<f32> {
    let Shape { nk, nv, kd, vd } = sh;
    let stride = sh.qkv_stride();
    let mut out = vec![0.0f32; case.seq * nv * vd];
    for t in 0..case.seq {
        let tb = t * stride;
        for vh in 0..nv {
            let kh = vh % nk;
            let q = &case.qkv[tb + kh * kd..tb + (kh + 1) * kd];
            let k = &case.qkv[tb + nk * kd + kh * kd..tb + nk * kd + (kh + 1) * kd];
            let v = &case.qkv[tb + 2 * nk * kd + vh * vd..tb + 2 * nk * kd + (vh + 1) * vd];
            let q_sumsq: f32 = q.iter().map(|&x| x * x).sum();
            let k_sumsq: f32 = k.iter().map(|&x| x * x).sum();
            let qn = (q_sumsq + 1e-6).sqrt().recip() / (kd as f32).sqrt();
            let kn = (k_sumsq + 1e-6).sqrt().recip();
            let ab = t * nv + vh;
            let exp_g = (case.a_log[vh] * softplus(case.a_proj[ab] + case.dt_bias[vh])).exp();
            let beta = sigmoid(case.b_proj[ab]);
            let s = &mut state[vh * kd * vd..(vh + 1) * kd * vd];
            for val in 0..vd {
                let mut kv_mem = 0.0f32;
                for (j, &kj) in k.iter().enumerate() {
                    let idx = j * vd + val;
                    let decayed = s[idx] * exp_g;
                    s[idx] = decayed;
                    kv_mem += decayed * (kj * kn);
                }
                let delta = (v[val] - kv_mem) * beta;
                let mut acc = 0.0f32;
                for (j, (&kj, &qj)) in k.iter().zip(q).enumerate() {
                    let idx = j * vd + val;
                    let updated = s[idx] + delta * (kj * kn);
                    s[idx] = updated;
                    acc += updated * (qj * qn);
                }
                out[t * nv * vd + vh * vd + val] = acc;
            }
        }
    }
    out
}

/// The chunked math, association-for-association with the two shaders.
fn host_chunked(sh: Shape, case: &Case, state: &mut [f32]) -> Vec<f32> {
    let Shape { nk, nv, kd, vd } = sh;
    let cs = QWEN4_GDN_CHUNK as usize;
    let stride = sh.qkv_stride();
    let n_chunks = case.seq.div_ceil(cs);
    let mut out = vec![0.0f32; case.seq * nv * vd];
    for vh in 0..nv {
        let kh = vh % nk;
        for c in 0..n_chunks {
            let l = cs.min(case.seq - c * cs);
            let tok0 = c * cs;
            let q_at = |i: usize| (tok0 + i) * stride + kh * kd;
            let k_at = |i: usize| (tok0 + i) * stride + nk * kd + kh * kd;
            let v_at = |i: usize| (tok0 + i) * stride + 2 * nk * kd + vh * vd;
            // Per-token scalars + inclusive cumsum of the log-decays.
            let mut gcs = vec![0.0f32; l];
            let mut beta = vec![0.0f32; l];
            let mut qn = vec![0.0f32; l];
            let mut kn = vec![0.0f32; l];
            for i in 0..l {
                let mut qs = 0.0f32;
                let mut ks = 0.0f32;
                for d in 0..kd {
                    qs += case.qkv[q_at(i) + d] * case.qkv[q_at(i) + d];
                    ks += case.qkv[k_at(i) + d] * case.qkv[k_at(i) + d];
                }
                qn[i] = (qs + 1e-6).sqrt().recip() / (kd as f32).sqrt();
                kn[i] = (ks + 1e-6).sqrt().recip();
                let ab = (tok0 + i) * nv + vh;
                gcs[i] = case.a_log[vh] * softplus(case.a_proj[ab] + case.dt_bias[vh]);
                beta[i] = sigmoid(case.b_proj[ab]);
            }
            for i in 1..l {
                gcs[i] += gcs[i - 1];
            }
            let eg: Vec<f32> = gcs.iter().map(|&g| g.exp()).collect();
            let gd: Vec<f32> = gcs.iter().map(|&g| (gcs[l - 1] - g).exp()).collect();
            // Strict-lower L and lower-incl-diag kq, decay-masked.
            let mut m = vec![0.0f32; l * l];
            let mut kq = vec![0.0f32; l * l];
            for i in 0..l {
                for j in 0..=i {
                    let mut dot_qk = 0.0f32;
                    let mut dot_kk = 0.0f32;
                    for d in 0..kd {
                        let kj = case.qkv[k_at(j) + d];
                        dot_qk += case.qkv[q_at(i) + d] * kj;
                        dot_kk += case.qkv[k_at(i) + d] * kj;
                    }
                    let decay = (gcs[i] - gcs[j]).exp();
                    kq[i * l + j] = dot_qk * qn[i] * kn[j] * decay;
                    if j < i {
                        m[i * l + j] = dot_kk * kn[i] * kn[j] * beta[i] * decay;
                    }
                }
            }
            // T = (I + L)^{-1} by forward substitution, in place (T - I).
            for i in 1..l {
                let mut new_row = vec![0.0f32; i];
                for (j, slot) in new_row.iter_mut().enumerate() {
                    let mut acc = -m[i * l + j];
                    for x in j + 1..i {
                        acc -= m[i * l + x] * m[x * l + j];
                    }
                    *slot = acc;
                }
                m[i * l..i * l + i].copy_from_slice(&new_row);
            }
            // T-applied chunk quantities.
            let mut u = vec![0.0f32; l * vd];
            for i in 0..l {
                for d in 0..vd {
                    let mut acc = beta[i] * case.qkv[v_at(i) + d];
                    for x in 0..i {
                        acc += m[i * l + x] * beta[x] * case.qkv[v_at(x) + d];
                    }
                    u[i * vd + d] = acc;
                }
            }
            let mut kcd = vec![0.0f32; l * kd];
            let mut qg = vec![0.0f32; l * kd];
            let mut kg = vec![0.0f32; l * kd];
            for i in 0..l {
                for d in 0..kd {
                    let mut acc = beta[i] * eg[i] * case.qkv[k_at(i) + d] * kn[i];
                    for x in 0..i {
                        acc += m[i * l + x] * beta[x] * eg[x] * case.qkv[k_at(x) + d] * kn[x];
                    }
                    kcd[i * kd + d] = acc;
                    qg[i * kd + d] = case.qkv[q_at(i) + d] * qn[i] * eg[i];
                    kg[i * kd + d] = case.qkv[k_at(i) + d] * kn[i] * gd[i];
                }
            }
            let glast = eg[l - 1];
            // Inter-chunk state pass.
            let s = &mut state[vh * kd * vd..(vh + 1) * kd * vd];
            let mut v_new = vec![0.0f32; l * vd];
            for i in 0..l {
                for val in 0..vd {
                    let mut acc = u[i * vd + val];
                    for d in 0..kd {
                        acc -= kcd[i * kd + d] * s[d * vd + val];
                    }
                    v_new[i * vd + val] = acc;
                }
            }
            for i in 0..l {
                for val in 0..vd {
                    let mut acc = 0.0f32;
                    for d in 0..kd {
                        acc += qg[i * kd + d] * s[d * vd + val];
                    }
                    for x in 0..l {
                        acc += kq[i * l + x] * v_new[x * vd + val];
                    }
                    out[(tok0 + i) * nv * vd + vh * vd + val] = acc;
                }
            }
            for d in 0..kd {
                for val in 0..vd {
                    let mut acc = s[d * vd + val] * glast;
                    for i in 0..l {
                        acc += kg[i * kd + d] * v_new[i * vd + val];
                    }
                    s[d * vd + val] = acc;
                }
            }
        }
    }
    out
}

/// Run the device pair once over `case`, starting from `state`, updating it
/// in place; returns the output. Two `launch_cached` calls = two submits, so
/// the intra->state dependency is fenced exactly as the prefill's barrier is.
fn device_chunked<'a>(
    ctx: &'a VulkanContext,
    cache: &mut KernelCache<'a>,
    sh: Shape,
    case: &Case,
    state: &mut [f32],
) -> Vec<f32> {
    let cs = QWEN4_GDN_CHUNK as usize;
    let n_chunks = case.seq.div_ceil(cs) as u32;
    let qkv = upload_f32(ctx, &case.qkv);
    let b = upload_f32(ctx, &case.b_proj);
    let a = upload_f32(ctx, &case.a_proj);
    let dt = upload_f32(ctx, &case.dt_bias);
    let alog = upload_f32(ctx, &case.a_log);
    let scratch_elems =
        qwen4_gdn_chunk_scratch_elems(n_chunks, sh.nv as u32, sh.kd as u32, sh.vd as u32);
    let scratch = upload_f32(ctx, &vec![0.0f32; scratch_elems]);
    let state_buf = upload_f32(ctx, state);
    let out_elems = case.seq * sh.nv * sh.vd;
    let out_buf = upload_f32(ctx, &vec![0.0f32; out_elems]);

    let params = qwen4_gdn_chunk_params(
        sh.nk as u32,
        sh.nv as u32,
        sh.kd as u32,
        sh.vd as u32,
        case.seq as u32,
        n_chunks,
    );
    launch_cached(
        cache,
        ctx,
        Kernel::Qwen4GdnChunkIntra,
        &[&qkv, &b, &a, &dt, &alog, &scratch],
        qwen4_gdn_chunk_intra_dispatch(n_chunks, sh.nv as u32),
        &params.to_le_bytes(),
        &[],
    )
    .expect("launch qwen4_gdn_chunk_intra");
    launch_cached(
        cache,
        ctx,
        Kernel::Qwen4GdnChunkState,
        &[&scratch, &state_buf, &out_buf],
        qwen4_gdn_chunk_state_dispatch(sh.nv as u32, sh.vd as u32),
        &params.to_le_bytes(),
        &[],
    )
    .expect("launch qwen4_gdn_chunk_state");

    state.copy_from_slice(&read_f32(&state_buf, state.len()));
    read_f32(&out_buf, out_elems)
}

/// Max relative error with a 1e-3 magnitude floor (same convention as the
/// prefill parity harness).
fn max_rel(got: &[f32], want: &[f32], what: &str) -> f32 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = 0f32;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{what}[{i}] = {g}");
        assert!(w.is_finite(), "{what}[{i}] oracle = {w}");
        worst = worst.max((g - w).abs() / w.abs().max(1e-3));
    }
    worst
}

fn run_case<'a>(
    ctx: &'a VulkanContext,
    cache: &mut KernelCache<'a>,
    sh: Shape,
    seq: usize,
    seed: u64,
) {
    let mut rng = Rng(seed);
    let case = random_case(&mut rng, sh, seq);
    let n_state = sh.nv * sh.kd * sh.vd;
    // Non-zero initial state: the inter-chunk pass must prove it BLENDS a
    // carried state, not merely that it works from zero.
    let mut st_serial: Vec<f32> = (0..n_state).map(|_| rng.next_f32() * 0.1).collect();
    let mut st_chunk = st_serial.clone();
    let mut st_dev = st_serial.clone();

    let out_serial = host_serial(sh, &case, &mut st_serial);
    let out_chunk = host_chunked(sh, &case, &mut st_chunk);
    let out_dev = device_chunked(ctx, cache, sh, &case, &mut st_dev);

    // Kernel claim: device == host-chunked up to GPU exp/rsqrt rounding and
    // FMA contraction. Calibrated on this box (8060S): worst 4.4e-5 (out) /
    // 1.5e-4 (state, real shape seq 160) across the cases below. A structural
    // break (wrong decay mask, wrong state blend) reads O(1) — verified by
    // mutating the mask sign during bring-up — so 5e-4 keeps >3 orders of
    // separation.
    let d_out = max_rel(&out_dev, &out_chunk, "device out");
    let d_st = max_rel(&st_dev, &st_chunk, "device state");
    // Operator claim: chunked == serial up to WY reassociation. Calibrated:
    // worst 4.3e-5 (out) / 8.9e-5 (state) across the cases below.
    let r_out = max_rel(&out_chunk, &out_serial, "reassociation out");
    let r_st = max_rel(&st_chunk, &st_serial, "reassociation state");
    eprintln!(
        "nk={} nv={} kd={} vd={} seq={seq}: device-vs-chunked out {d_out:.3e} state {d_st:.3e}; \
         chunked-vs-serial out {r_out:.3e} state {r_st:.3e}",
        sh.nk, sh.nv, sh.kd, sh.vd
    );
    assert!(d_out < 5e-4, "device out drift {d_out:.3e}");
    assert!(d_st < 5e-4, "device state drift {d_st:.3e}");
    assert!(r_out < 1e-3, "reassociation out drift {r_out:.3e}");
    assert!(r_st < 1e-3, "reassociation state drift {r_st:.3e}");
}

#[test]
fn chunked_pair_matches_host_chunked_and_serial_oracles() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping gdn_chunked test");
            return;
        }
    };
    eprintln!("ARLE Vulkan gdn_chunked proof on: {}", ctx.device_name());
    let mut cache = KernelCache::new();

    let small = Shape {
        nk: 2,
        nv: 6,
        kd: 16,
        vd: 32,
    };
    // Multi-chunk with a partial tail (64 + 36), a single partial chunk (7,
    // the truncated fixture's width), and an exact one-chunk boundary (64).
    run_case(&ctx, &mut cache, small, 100, 0x5EED_0001);
    run_case(&ctx, &mut cache, small, 7, 0x5EED_0002);
    run_case(&ctx, &mut cache, small, 64, 0x5EED_0003);
    // The real qwen4_exp shape.
    let real = Shape {
        nk: 16,
        nv: 48,
        kd: 128,
        vd: 128,
    };
    run_case(&ctx, &mut cache, real, 160, 0x5EED_0004);
}

/// State continuation across two dispatch pairs — how `forward_prompt`
/// actually drives the kernels (one pair per prefill chunk, resident state
/// carried in place).
#[test]
fn chunked_pair_carries_state_across_calls() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping continuation test");
            return;
        }
    };
    let mut cache = KernelCache::new();
    let sh = Shape {
        nk: 2,
        nv: 6,
        kd: 16,
        vd: 32,
    };
    let mut rng = Rng(0xC0FF_EE00);
    let full = random_case(&mut rng, sh, 96);
    let n_state = sh.nv * sh.kd * sh.vd;
    let mut st_serial = vec![0.0f32; n_state];
    let out_serial = host_serial(sh, &full, &mut st_serial);

    // Same 96 tokens as two 48-token calls through the device pair.
    let stride = sh.qkv_stride();
    let split = |lo: usize, hi: usize| Case {
        qkv: full.qkv[lo * stride..hi * stride].to_vec(),
        b_proj: full.b_proj[lo * sh.nv..hi * sh.nv].to_vec(),
        a_proj: full.a_proj[lo * sh.nv..hi * sh.nv].to_vec(),
        dt_bias: full.dt_bias.clone(),
        a_log: full.a_log.clone(),
        seq: hi - lo,
    };
    let mut st_dev = vec![0.0f32; n_state];
    let mut out_dev = device_chunked(&ctx, &mut cache, sh, &split(0, 48), &mut st_dev);
    out_dev.extend(device_chunked(
        &ctx,
        &mut cache,
        sh,
        &split(48, 96),
        &mut st_dev,
    ));

    let r_out = max_rel(&out_dev, &out_serial, "continuation out");
    let r_st = max_rel(&st_dev, &st_serial, "continuation state");
    eprintln!("two-call continuation: out {r_out:.3e} state {r_st:.3e}");
    assert!(r_out < 1e-3, "continuation out drift {r_out:.3e}");
    assert!(r_st < 1e-3, "continuation state drift {r_st:.3e}");
}
