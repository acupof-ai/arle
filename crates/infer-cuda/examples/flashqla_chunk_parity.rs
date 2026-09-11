//! GDN multi-row advance numeric-parity gate: chunked FlashQLA and the
//! single-sequence recurrent fallback.
//!
//! Standalone kernel harness — NO engine, NO serve. Every multi-row linear
//! (gated-delta-rule) advance takes one of two kernels, and both are anchored
//! here on one f64 host recurrence written from the model math (the same
//! recurrence as `gdr_decode_parity.rs`, f64 accumulation):
//!
//! 1. **Chunked FlashQLA** — the default serving path on sm90:
//!    `conv1d_prefill_cuda` -> `gdr_fq_prep` -> chunk cumsum -> kkt -> fwd.
//!    Compared on GDR bf16 output at every row and the final f32 state.
//! 2. **Single-sequence recurrent prefill** — the fallback when the chunked
//!    path is unavailable (`--qwen35-gdr-chunked` off, non-sm90 such as A100,
//!    or an unsupported geometry): `conv1d_prefill_cuda` then
//!    `gated_delta_rule_prefill_recurrent_cuda`. Compared on the conv output,
//!    the rebuilt conv ring, GDR output and the final f32 state. This is also
//!    the only multi-row GDR path on sm80.
//!
//! Geometry: all four GDN attn_tp shards — global (key,value) heads (16,48),
//! attn_tp=2 (8,24), attn_tp=4 (4,12), attn_tp=8 (2,6) — head dim 128, at the
//! DSpark-relevant row length 5, the 17-row verify, and one 64-token
//! FlashMLA paged-KV page (the chunked path covers it in one chunk). Every row
//! starts from a nonzero conv ring and a nonzero recurrent state.
//!
//! `--negative-control` applies one sabotage per compared family in separate
//! runs and asserts ONLY that family trips: FlashQLA output/state and
//! recurrent conv/ring/output/state. It prints NEGATIVE CONTROL OK and exits
//! 0; the teeth are internal.
//!
//! Run on a pod:
//! `INFER_CUDA_DEVICE=<free-gpu> target/release/examples/flashqla_chunk_parity`

fn main() -> anyhow::Result<()> {
    if std::env::args().nth(1).as_deref() == Some("--kernel-build-id") {
        println!("{}", cuda_kernels::KERNEL_BUILD_ID);
        return Ok(());
    }
    let negative = std::env::args().any(|a| a == "--negative-control");
    real::run(negative)
}

#[cfg(not(feature = "cuda"))]
mod real {
    pub(super) fn run(_negative: bool) -> anyhow::Result<()> {
        eprintln!("flashqla_chunk_parity is a CUDA harness; rebuild with --features cuda.");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use anyhow::{Result, ensure};
    use cuda_kernels::ffi;
    use cuda_kernels::prelude::DeviceContext;
    use cuda_kernels::recurrent::{conv1d_prefill_raw, gdr_fq_prep_raw, gdr_prefill_recurrent_raw};
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

    const KEY_DIM: usize = 128;
    const VAL_DIM: usize = 128;
    const K: usize = 4;
    // All GDN shards serve can land on: global and attn_tp = 2 / 4 / 8.
    const GEOMS: &[(usize, usize)] = &[(16, 48), (8, 24), (4, 12), (2, 6)];
    // 5-token DSpark draft block, 17-row verify, one 64-token paged-KV page.
    const LENS: &[usize] = &[5, 17, 64];
    const SEED: u64 = 0x6A27_11E5_C0DE_5240;

    // FlashQLA runs f32 with bf16 intermediates and a chunk-local cumsum.
    const FQ_OUT_REL_L2_MAX: f64 = 6e-2;
    const FQ_OUT_ABS_FLOOR: f64 = 2e-2;
    const FQ_OUT_ABS_SLOPE: f64 = 8e-2;
    const FQ_STATE_REL_L2_MAX: f64 = 3e-2;
    const FQ_STATE_ABS_MAX: f64 = 1.5e-2;
    // The single-sequence recurrent kernel is bf16 throughout; tighter bounds
    // carried from the kernel's earlier varlen-form gate.
    const REC_OUT_REL_L2_MAX: f64 = 3e-2;
    const REC_OUT_ABS_FLOOR: f64 = 1.5e-2;
    const REC_OUT_ABS_SLOPE: f64 = 5e-2;
    const REC_STATE_REL_L2_MAX: f64 = 1.5e-2;
    const REC_STATE_ABS_MAX: f64 = 8e-3;
    const CONV_REL_L2_MAX: f64 = 2e-2;
    const CONV_ABS_FLOOR: f64 = 1e-2;
    const CONV_ABS_SLOPE: f64 = 3e-2;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Sabotage {
        None,
        FqOut,
        FqState,
        RecConv,
        RecRing,
        RecOut,
        RecState,
    }

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn unit(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
        }
        fn normal(&mut self) -> f32 {
            let u1 = self.unit().max(1e-7);
            let u2 = self.unit();
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        }
    }

    fn bf(v: f64) -> bf16 {
        bf16::from_f32(v as f32)
    }

    fn silu(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }

    /// f64 reference for one multi-row advance: conv front end + GDR recurrence.
    struct SlotRef {
        /// Conv output rows (bf16-truncated, token-major).
        conv_out: Vec<bf16>,
        /// Rebuilt conv ring [ch, K-1].
        ring: Vec<bf16>,
        /// GDR outputs (token-major [len, vh*128]), pre-bf16 f64.
        gdr_out: Vec<f64>,
        /// Final f32 recurrent state [vh*128*128].
        state: Vec<f64>,
    }

    #[allow(clippy::too_many_arguments)]
    fn reference_slot(
        kh: usize,
        vh: usize,
        ch: usize,
        len: usize,
        x: &[bf16],
        bp: &[bf16],
        ap: &[bf16],
        w: &[bf16],
        dt: &[bf16],
        alog: &[f32],
        ring0: &[bf16],
        state0: &[f32],
    ) -> SlotRef {
        let q_dim = kh * KEY_DIM;
        let v_base = 2 * q_dim;
        let ring: Vec<f64> = ring0.iter().map(|v| f64::from(*v)).collect();
        let mut state: Vec<f64> = state0.iter().map(|v| f64::from(*v)).collect();
        let mut conv_out = vec![bf16::ZERO; len * ch];
        let mut gdr_out = vec![0f64; len * vh * VAL_DIM];

        for t in 0..len {
            // K=4 causal depthwise conv; src<0 reads the prior ring or zero.
            for c in 0..ch {
                let mut sum = 0f64;
                for kk in 0..K {
                    let src = t as isize - (K as isize - 1) + kk as isize;
                    let val = if src >= 0 {
                        f64::from(x[src as usize * ch + c])
                    } else {
                        let idx = (K as isize - 1) + src;
                        if idx >= 0 {
                            ring[c * (K - 1) + idx as usize]
                        } else {
                            0.0
                        }
                    };
                    sum += val * f64::from(w[c * K + kk]);
                }
                let v = f64::from(bf(sum));
                conv_out[t * ch + c] = bf(silu(v));
            }

            for vhead in 0..vh {
                let khead = vhead * kh / vh;
                let mut q = [0f64; KEY_DIM];
                let mut kk = [0f64; KEY_DIM];
                let mut v = [0f64; VAL_DIM];
                let mut q_sq = 0f64;
                let mut k_sq = 0f64;
                for d in 0..KEY_DIM {
                    q[d] = f64::from(conv_out[t * ch + khead * KEY_DIM + d]);
                    kk[d] = f64::from(conv_out[t * ch + q_dim + khead * KEY_DIM + d]);
                    q_sq += q[d] * q[d];
                    k_sq += kk[d] * kk[d];
                }
                let q_norm = (q_sq + 1e-12).sqrt().recip() * (KEY_DIM as f64).sqrt().recip();
                let k_norm = (k_sq + 1e-12).sqrt().recip();
                for d in 0..KEY_DIM {
                    q[d] *= q_norm;
                    kk[d] *= k_norm;
                }
                for d in 0..VAL_DIM {
                    v[d] = f64::from(conv_out[t * ch + v_base + vhead * VAL_DIM + d]);
                }
                let xv = f64::from(ap[t * vh + vhead]) + f64::from(dt[vhead]);
                let softplus = if xv > 20.0 { xv } else { (1.0 + xv.exp()).ln() };
                let exp_g = (-f64::from(alog[vhead]).exp() * softplus).exp();
                let beta = 1.0 / (1.0 + (-f64::from(bp[t * vh + vhead])).exp());

                let base = vhead * KEY_DIM * VAL_DIM;
                let mut kv = [0f64; VAL_DIM];
                for j in 0..KEY_DIM {
                    for i in 0..VAL_DIM {
                        let s = state[base + j * VAL_DIM + i] * exp_g;
                        state[base + j * VAL_DIM + i] = s;
                        kv[i] += s * kk[j];
                    }
                }
                let mut delta = [0f64; VAL_DIM];
                for i in 0..VAL_DIM {
                    delta[i] = (v[i] - kv[i]) * beta;
                }
                let mut out = [0f64; VAL_DIM];
                for j in 0..KEY_DIM {
                    for i in 0..VAL_DIM {
                        let s = state[base + j * VAL_DIM + i] + delta[i] * kk[j];
                        state[base + j * VAL_DIM + i] = s;
                        out[i] += s * q[j];
                    }
                }
                gdr_out
                    [t * vh * VAL_DIM + vhead * VAL_DIM..t * vh * VAL_DIM + (vhead + 1) * VAL_DIM]
                    .copy_from_slice(&out);
            }
        }

        // Ring rebuild (conv1d_state_update_kernel): tail of x shifted across
        // the old ring; positions before x[0] read the old ring or zero.
        let mut ring_final = vec![0f64; ch * (K - 1)];
        for c in 0..ch {
            for i in 0..(K - 1) {
                let src = len as isize - (K as isize - 1) + i as isize;
                ring_final[c * (K - 1) + i] = if src >= 0 {
                    f64::from(x[src as usize * ch + c])
                } else {
                    let idx = (K as isize - 1) + src;
                    if idx >= 0 {
                        ring[c * (K - 1) + idx as usize]
                    } else {
                        0.0
                    }
                };
            }
        }

        SlotRef {
            conv_out,
            ring: ring_final.iter().map(|v| bf(*v)).collect(),
            gdr_out,
            state,
        }
    }

    /// One generated case: all host inputs plus the f64 reference.
    struct Case {
        kh: usize,
        vh: usize,
        ch: usize,
        len: usize,
        w: Vec<bf16>,
        x: Vec<bf16>,
        bp: Vec<bf16>,
        ap: Vec<bf16>,
        dt: Vec<bf16>,
        alog: Vec<f32>,
        ring0: Vec<bf16>,
        state0: Vec<f32>,
        reference: SlotRef,
    }

    fn make_case(kh: usize, vh: usize, len: usize) -> Case {
        let ch = 2 * kh * KEY_DIM + vh * VAL_DIM;
        let mut rng = Rng::new(
            SEED ^ 0xF1A5 ^ ((len as u64) << 20) ^ ((kh as u64) << 32) ^ ((vh as u64) << 24),
        );
        let w: Vec<bf16> = (0..ch * K)
            .map(|_| bf((rng.normal() * 0.3) as f64))
            .collect();
        let dt: Vec<bf16> = (0..vh).map(|_| bf((rng.normal() * 0.5) as f64)).collect();
        let alog: Vec<f32> = (0..vh).map(|i| -(1.0 + (i as f32 % 3.0) * 0.5)).collect();
        let x: Vec<bf16> = (0..len * ch)
            .map(|_| bf((rng.normal() * 0.5) as f64))
            .collect();
        let bp: Vec<bf16> = (0..len * vh)
            .map(|_| bf((rng.normal() * 0.5) as f64))
            .collect();
        let ap: Vec<bf16> = (0..len * vh)
            .map(|_| bf((rng.normal() * 0.5) as f64))
            .collect();
        let ring0: Vec<bf16> = (0..ch * (K - 1))
            .map(|_| bf((rng.normal() * 0.3) as f64))
            .collect();
        let state0: Vec<f32> = (0..vh * KEY_DIM * VAL_DIM)
            .map(|_| rng.normal() * 0.02)
            .collect();
        let reference = reference_slot(
            kh, vh, ch, len, &x, &bp, &ap, &w, &dt, &alog, &ring0, &state0,
        );
        Case {
            kh,
            vh,
            ch,
            len,
            w,
            x,
            bp,
            ap,
            dt,
            alog,
            ring0,
            state0,
            reference,
        }
    }

    /// rel-L2 plus a per-element slope+floor bound; returns (pass, rel, n_viol).
    /// `bias` shifts the reference (negative control) and must fail the bound.
    fn bound(
        got: &[f64],
        want: &[f64],
        bias: f64,
        rel_cap: f64,
        floor: f64,
        slope: f64,
    ) -> (bool, f64, usize) {
        let mut viol = 0usize;
        let (mut diff_sq, mut ref_sq) = (0f64, 0f64);
        for (g, w0) in got.iter().zip(want) {
            let w = w0 + bias;
            let d = g - w;
            diff_sq += d * d;
            ref_sq += w * w;
            if d.abs() > floor + slope * w.abs() {
                viol += 1;
            }
        }
        let rel = (diff_sq / ref_sq.max(1e-12)).sqrt();
        (rel < rel_cap && viol == 0, rel, viol)
    }

    pub(super) fn run(negative: bool) -> Result<()> {
        let ctx = DeviceContext::new()?;
        eprintln!(
            "[flashqla-chunk-parity] device={} build={}{}",
            ctx.ordinal(),
            cuda_kernels::KERNEL_BUILD_ID,
            if negative { " NEGATIVE-CONTROL" } else { "" }
        );

        for &(kh, vh) in GEOMS {
            for &len in LENS {
                let label = format!("geom=({kh},{vh}) len={len}");
                if negative {
                    let (fo0, fs0) = probe_flashqla(&ctx, &make_case(kh, vh, len), Sabotage::None)?;
                    let (c0, r0, o0, s0) =
                        probe_recurrent(&ctx, &make_case(kh, vh, len), Sabotage::None)?;
                    ensure!(
                        fo0 && fs0 && c0 && r0 && o0 && s0,
                        "{label}: baseline not green before teeth"
                    );
                    let (fo1, fs1) =
                        probe_flashqla(&ctx, &make_case(kh, vh, len), Sabotage::FqOut)?;
                    ensure!(!fo1 && fs1, "{label}: fq-output tooth dead or leaked");
                    let (fo2, fs2) =
                        probe_flashqla(&ctx, &make_case(kh, vh, len), Sabotage::FqState)?;
                    ensure!(fo2 && !fs2, "{label}: fq-state tooth dead or leaked");
                    let (c1, r1, o1, s1) =
                        probe_recurrent(&ctx, &make_case(kh, vh, len), Sabotage::RecConv)?;
                    ensure!(
                        !c1 && r1 && o1 && s1,
                        "{label}: rec-conv tooth dead or leaked"
                    );
                    let (c2, r2, o2, s2) =
                        probe_recurrent(&ctx, &make_case(kh, vh, len), Sabotage::RecRing)?;
                    ensure!(
                        c2 && !r2 && o2 && s2,
                        "{label}: rec-ring tooth dead or leaked"
                    );
                    let (c3, r3, o3, s3) =
                        probe_recurrent(&ctx, &make_case(kh, vh, len), Sabotage::RecOut)?;
                    ensure!(
                        c3 && r3 && !o3 && s3,
                        "{label}: rec-output tooth dead or leaked"
                    );
                    let (c4, r4, o4, s4) =
                        probe_recurrent(&ctx, &make_case(kh, vh, len), Sabotage::RecState)?;
                    ensure!(
                        c4 && r4 && o4 && !s4,
                        "{label}: rec-state tooth dead or leaked"
                    );
                    eprintln!("[{label}] teeth OK");
                } else {
                    let (fo, fs) = probe_flashqla(&ctx, &make_case(kh, vh, len), Sabotage::None)?;
                    let (c, r, o, s) =
                        probe_recurrent(&ctx, &make_case(kh, vh, len), Sabotage::None)?;
                    ensure!(
                        fo && fs && c && r && o && s,
                        "{label}: flashqla_chunk_parity FAILED"
                    );
                    eprintln!("[{label}] PASS");
                }
            }
        }

        if negative {
            eprintln!("[flashqla-chunk-parity] NEGATIVE CONTROL OK");
        } else {
            eprintln!("[flashqla-chunk-parity] ALL PASS");
        }
        Ok(())
    }

    /// Chunked FlashQLA pipeline for one case, anchored on the f64 recurrence.
    /// Returns (output_ok, state_ok) under the chosen sabotage.
    fn probe_flashqla(
        ctx: &DeviceContext,
        case: &Case,
        sabotage: Sabotage,
    ) -> Result<(bool, bool)> {
        let (kh, vh, ch, len) = (case.kh, case.vh, case.ch, case.len);
        let (cumsum_fn, kkt_fn, fwd_fn): (ffi::FqCumsumFn, ffi::FqKktFn, ffi::FqFwdFn) =
            match (kh, vh) {
                (16, 48) => (
                    ffi::gdr_fq_cumsum_h48_cuda as _,
                    ffi::gdr_fq_kkt_h48_cuda as _,
                    ffi::gdr_fq_fwd_h48_cuda as _,
                ),
                (8, 24) => (
                    ffi::gdr_fq_cumsum_h24g8_cuda as _,
                    ffi::gdr_fq_kkt_h24g8_cuda as _,
                    ffi::gdr_fq_fwd_h24g8_cuda as _,
                ),
                (4, 12) => (
                    ffi::gdr_fq_cumsum_h12g4_cuda as _,
                    ffi::gdr_fq_kkt_h12g4_cuda as _,
                    ffi::gdr_fq_fwd_h12g4_cuda as _,
                ),
                (2, 6) => (
                    ffi::gdr_fq_cumsum_h6g2_cuda as _,
                    ffi::gdr_fq_kkt_h6g2_cuda as _,
                    ffi::gdr_fq_fwd_h6g2_cuda as _,
                ),
                _ => unreachable!("geom not in GEOMS"),
            };

        let w_d = ctx.stream.clone_htod(&case.w)?;
        let x_d = ctx.stream.clone_htod(&case.x)?;
        let bp_d = ctx.stream.clone_htod(&case.bp)?;
        let ap_d = ctx.stream.clone_htod(&case.ap)?;
        let dt_d = ctx.stream.clone_htod(&case.dt)?;
        let alog_d = ctx.stream.clone_htod(&case.alog)?;
        let mut ring_d = ctx.stream.clone_htod(&case.ring0)?;
        let h0_d = ctx.stream.clone_htod(&case.state0)?;
        let mut conv_d = ctx.stream.alloc_zeros::<bf16>(len * ch)?;
        let mut o_d = ctx.stream.alloc_zeros::<bf16>(len * vh * VAL_DIM)?;
        let mut ht_d = ctx.stream.alloc_zeros::<f32>(vh * KEY_DIM * VAL_DIM)?;

        let (xp, _) = x_d.device_ptr(&ctx.stream);
        let (wp, _) = w_d.device_ptr(&ctx.stream);
        let (bp_p, _) = bp_d.device_ptr(&ctx.stream);
        let (ap_p, _) = ap_d.device_ptr(&ctx.stream);
        let (dt_p, _) = dt_d.device_ptr(&ctx.stream);
        let (al_p, _) = alog_d.device_ptr(&ctx.stream);

        // The AOT dispatch resolves SM/module via the driver context.
        ctx.ctx.bind_to_thread()?;
        {
            let (rp, _) = ring_d.device_ptr_mut(&ctx.stream);
            let (cp, _) = conv_d.device_ptr_mut(&ctx.stream);
            conv1d_prefill_raw(&ctx.stream, xp, wp, rp, cp, ch, len, K)?;

            let mut q_d = ctx.stream.alloc_zeros::<bf16>(len * kh * KEY_DIM)?;
            let mut k_d = ctx.stream.alloc_zeros::<bf16>(len * kh * KEY_DIM)?;
            let mut v_d = ctx.stream.alloc_zeros::<bf16>(len * vh * VAL_DIM)?;
            let mut g_d = ctx.stream.alloc_zeros::<f32>(len * vh)?;
            let mut gc_d = ctx.stream.alloc_zeros::<f32>(len * vh)?;
            let mut beta_d = ctx.stream.alloc_zeros::<f32>(len * vh)?;
            let mut ainv_d = ctx.stream.alloc_zeros::<bf16>(len * vh * 64)?;
            gdr_fq_prep_raw(
                &ctx.stream,
                cp,
                bp_p,
                ap_p,
                dt_p,
                al_p,
                q_d.device_ptr_mut(&ctx.stream).0,
                k_d.device_ptr_mut(&ctx.stream).0,
                v_d.device_ptr_mut(&ctx.stream).0,
                g_d.device_ptr_mut(&ctx.stream).0,
                beta_d.device_ptr_mut(&ctx.stream).0,
                kh,
                vh,
                KEY_DIM,
                VAL_DIM,
                len,
            )?;

            let (qp, _) = q_d.device_ptr(&ctx.stream);
            let (kp, _) = k_d.device_ptr(&ctx.stream);
            let (vp, _) = v_d.device_ptr(&ctx.stream);
            let (gp, _) = g_d.device_ptr(&ctx.stream);
            let (gcp, _) = gc_d.device_ptr_mut(&ctx.stream);
            let (betap, _) = beta_d.device_ptr(&ctx.stream);
            let (ainvp, _) = ainv_d.device_ptr_mut(&ctx.stream);
            let (h0p, _) = h0_d.device_ptr(&ctx.stream);
            let (op, _) = o_d.device_ptr_mut(&ctx.stream);
            let (htp, _) = ht_d.device_ptr_mut(&ctx.stream);
            // SAFETY: slices sized for `len`; one chunk covers len<=64, so the
            // chunk-local cumsum spans the full sequence.
            unsafe {
                cumsum_fn(
                    gp as *const f32,
                    gcp as *mut f32,
                    len as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
                kkt_fn(
                    kp as *const ffi::Half,
                    betap as *const f32,
                    ainvp as *mut ffi::Half,
                    len as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
                fwd_fn(
                    qp as *const ffi::Half,
                    kp as *const ffi::Half,
                    vp as *const ffi::Half,
                    ainvp as *const ffi::Half,
                    gcp as *const f32,
                    betap as *const f32,
                    h0p as *const f32,
                    op as *mut ffi::Half,
                    htp as *mut f32,
                    len as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
            ctx.sync()?;
        }

        let o_got: Vec<f64> = ctx
            .stream
            .clone_dtoh(&o_d)?
            .iter()
            .map(|v| f64::from(*v))
            .collect();
        let ht_got: Vec<f64> = ctx
            .stream
            .clone_dtoh(&ht_d)?
            .iter()
            .map(|v| f64::from(*v))
            .collect();

        let out_bias = if sabotage == Sabotage::FqOut {
            1.0
        } else {
            0.0
        };
        let state_bias = if sabotage == Sabotage::FqState {
            0.5
        } else {
            0.0
        };
        let (o_ok, o_rel, o_viol) = bound(
            &o_got,
            &case.reference.gdr_out,
            out_bias,
            FQ_OUT_REL_L2_MAX,
            FQ_OUT_ABS_FLOOR,
            FQ_OUT_ABS_SLOPE,
        );
        let (s_ok, s_rel, s_viol) = bound(
            &ht_got,
            &case.reference.state,
            state_bias,
            FQ_STATE_REL_L2_MAX,
            FQ_STATE_ABS_MAX,
            0.0,
        );
        eprintln!(
            "[flashqla-chunk-parity:fq] ({kh},{vh}) len={len} out(l2={o_rel:.3e},v={o_viol}) state(l2={s_rel:.3e},v={s_viol})"
        );
        Ok((o_ok, s_ok))
    }

    /// Single-sequence recurrent fallback for one case: `conv1d_prefill_cuda`
    /// then `gated_delta_rule_prefill_recurrent_cuda` (state advanced in
    /// place). Returns (conv_ok, ring_ok, output_ok, state_ok).
    fn probe_recurrent(
        ctx: &DeviceContext,
        case: &Case,
        sabotage: Sabotage,
    ) -> Result<(bool, bool, bool, bool)> {
        let (kh, vh, ch, len) = (case.kh, case.vh, case.ch, case.len);

        let w_d = ctx.stream.clone_htod(&case.w)?;
        let x_d = ctx.stream.clone_htod(&case.x)?;
        let bp_d = ctx.stream.clone_htod(&case.bp)?;
        let ap_d = ctx.stream.clone_htod(&case.ap)?;
        let dt_d = ctx.stream.clone_htod(&case.dt)?;
        let alog_d = ctx.stream.clone_htod(&case.alog)?;
        let mut ring_d = ctx.stream.clone_htod(&case.ring0)?;
        let mut state_d = ctx.stream.clone_htod(&case.state0)?;
        let mut conv_d = ctx.stream.alloc_zeros::<bf16>(len * ch)?;
        let mut o_d = ctx.stream.alloc_zeros::<bf16>(len * vh * VAL_DIM)?;

        {
            let (xp, _) = x_d.device_ptr(&ctx.stream);
            let (wp, _) = w_d.device_ptr(&ctx.stream);
            let (rp, _) = ring_d.device_ptr_mut(&ctx.stream);
            let (cp, _) = conv_d.device_ptr_mut(&ctx.stream);
            conv1d_prefill_raw(&ctx.stream, xp, wp, rp, cp, ch, len, K)?;

            let (bp_p, _) = bp_d.device_ptr(&ctx.stream);
            let (ap_p, _) = ap_d.device_ptr(&ctx.stream);
            let (dt_p, _) = dt_d.device_ptr(&ctx.stream);
            let (al_p, _) = alog_d.device_ptr(&ctx.stream);
            let (sp, _) = state_d.device_ptr_mut(&ctx.stream);
            let (op, _) = o_d.device_ptr_mut(&ctx.stream);
            gdr_prefill_recurrent_raw(
                &ctx.stream,
                cp,
                bp_p,
                ap_p,
                dt_p,
                al_p,
                sp,
                op,
                kh,
                vh,
                KEY_DIM,
                VAL_DIM,
                len,
            )?;
            ctx.sync()?;
        }

        let conv_got: Vec<f64> = ctx
            .stream
            .clone_dtoh(&conv_d)?
            .iter()
            .map(|v| f64::from(*v))
            .collect();
        let ring_got: Vec<f64> = ctx
            .stream
            .clone_dtoh(&ring_d)?
            .iter()
            .map(|v| f64::from(*v))
            .collect();
        let o_got: Vec<f64> = ctx
            .stream
            .clone_dtoh(&o_d)?
            .iter()
            .map(|v| f64::from(*v))
            .collect();
        let st_got: Vec<f64> = ctx
            .stream
            .clone_dtoh(&state_d)?
            .iter()
            .map(|v| f64::from(*v))
            .collect();
        let conv_want: Vec<f64> = case
            .reference
            .conv_out
            .iter()
            .map(|v| f64::from(*v))
            .collect();
        let ring_want: Vec<f64> = case.reference.ring.iter().map(|v| f64::from(*v)).collect();

        let conv_bias = if sabotage == Sabotage::RecConv {
            1.0
        } else {
            0.0
        };
        let ring_bias = if sabotage == Sabotage::RecRing {
            1.0
        } else {
            0.0
        };
        let out_bias = if sabotage == Sabotage::RecOut {
            1.0
        } else {
            0.0
        };
        let state_bias = if sabotage == Sabotage::RecState {
            0.5
        } else {
            0.0
        };
        let (c_ok, c_rel, c_viol) = bound(
            &conv_got,
            &conv_want,
            conv_bias,
            CONV_REL_L2_MAX,
            CONV_ABS_FLOOR,
            CONV_ABS_SLOPE,
        );
        let (r_ok, r_rel, r_viol) = bound(
            &ring_got,
            &ring_want,
            ring_bias,
            CONV_REL_L2_MAX,
            CONV_ABS_FLOOR,
            0.0,
        );
        let (o_ok, o_rel, o_viol) = bound(
            &o_got,
            &case.reference.gdr_out,
            out_bias,
            REC_OUT_REL_L2_MAX,
            REC_OUT_ABS_FLOOR,
            REC_OUT_ABS_SLOPE,
        );
        let (s_ok, s_rel, s_viol) = bound(
            &st_got,
            &case.reference.state,
            state_bias,
            REC_STATE_REL_L2_MAX,
            REC_STATE_ABS_MAX,
            0.0,
        );
        eprintln!(
            "[flashqla-chunk-parity:rec] ({kh},{vh}) len={len} conv(l2={c_rel:.3e},v={c_viol}) ring(l2={r_rel:.3e},v={r_viol}) out(l2={o_rel:.3e},v={o_viol}) state(l2={s_rel:.3e},v={s_viol})"
        );
        Ok((c_ok, r_ok, o_ok, s_ok))
    }
}
