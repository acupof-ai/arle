//! DSv4/Qwen3.5 varlen prefill GDR + conv1d numeric-parity gate.
//!
//! Standalone kernel harness — NO engine, NO serve. The attn_tp>=2 batched
//! spec-verify and DSpark rollback-replay path feeds each slot its own row:
//! `conv1d_prefill_varlen_cuda` then
//! `gated_delta_rule_prefill_recurrent_varlen_cuda`, each slot a different
//! length into its own conv ring and f32 recurrent state. Both are anchored on
//! one f64 host recurrence written from the model math (the same recurrence as
//! `gdr_decode_parity.rs`, f64 accumulation):
//!   - conv output at every valid row AND the rebuilt conv ring
//!   - GDR bf16 output at every row AND the final f32 state per slot
//!
//! Geometry: local (key,value) heads (16,48) and the attn_tp=2 shard (8,24),
//! head dim 128, batch 1/4/8. Per-slot lengths cycle 1, 2, 5 (DSpark draft
//! block), 17 and 64 (one FlashMLA paged-KV page). Every slot starts from a
//! nonzero conv ring and a nonzero recurrent state.
//!
//! At the global (16,48) shard AND the attn_tp=2 (8,24) / attn_tp=4 (4,12)
//! shards, the SAME single-slot input at the DSpark-relevant row lengths 5 and
//! 17 (and a 64-token reference chunk) is additionally run through BOTH the
//! varlen recurrent kernel and the chunked FlashQLA pipeline
//! (gdr_fq_prep -> cumsum -> kkt -> fwd, one chunk), from the same nonzero
//! initial state: each path is compared to the f64 anchor, and the varlen-vs-
//! FlashQLA max output/state diff is printed per geometry and length. All four
//! GDN shards (global 16,48; attn_tp=2 8,24; attn_tp=4 4,12; attn_tp=8 2,6)
//! are checked, so the cross-path gate covers every TP size that routes
//! multi-row advances through chunked FlashQLA.
//!
//! `--negative-control` applies one sabotage per family (conv output, rebuilt
//! conv ring, GDR output, GDR final state) in separate comparisons and asserts
//! ONLY that family trips. It prints NEGATIVE CONTROL OK and exits 0; the
//! teeth are internal.
//!
//! Run on a pod:
//! `INFER_CUDA_DEVICE=<free-gpu> target/release/examples/gdr_varlen_parity`

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
        eprintln!("gdr_varlen_parity is a CUDA harness; rebuild with --features cuda.");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use anyhow::{Result, ensure};
    use cuda_kernels::ffi;
    use cuda_kernels::prelude::DeviceContext;
    use cuda_kernels::recurrent::{
        conv1d_prefill_raw, conv1d_prefill_varlen_raw, gdr_fq_prep_raw,
        gdr_prefill_recurrent_varlen_raw,
    };
    use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
    use half::bf16;

    const KEY_DIM: usize = 128;
    const VAL_DIM: usize = 128;
    const K: usize = 4;
    const GEOMS: &[(usize, usize)] = &[(16, 48), (8, 24)];
    const BATCHES: &[usize] = &[1, 4, 8];
    // Per-slot row lengths cycle through this list: 1, 2, the 5-token DSpark
    // draft block, 17, and one 64-token FlashMLA paged-KV page.
    const LENS: &[usize] = &[1, 2, 5, 17, 64];
    const SEED: u64 = 0x6A27_11E5_C0DE_5240;

    // f64 oracle vs f32/bf16 kernels over a <=64-step carried recurrence.
    const OUT_REL_L2_MAX: f64 = 3e-2;
    const OUT_ABS_FLOOR: f64 = 1.5e-2;
    const OUT_ABS_SLOPE: f64 = 5e-2;
    const STATE_REL_L2_MAX: f64 = 1.5e-2;
    const STATE_ABS_MAX: f64 = 8e-3;
    const CONV_REL_L2_MAX: f64 = 2e-2;
    const CONV_ABS_FLOOR: f64 = 1e-2;
    const CONV_ABS_SLOPE: f64 = 3e-2;
    // FlashQLA runs f32 with bf16 intermediates and a chunk-local cumsum.
    const FQ_OUT_REL_L2_MAX: f64 = 6e-2;
    const FQ_OUT_ABS_FLOOR: f64 = 2e-2;
    const FQ_OUT_ABS_SLOPE: f64 = 8e-2;
    const FQ_STATE_REL_L2_MAX: f64 = 3e-2;
    const FQ_STATE_ABS_MAX: f64 = 1.5e-2;
    // Cross-path: geometries where serve routes verify — global (16,48) and
    // the attn_tp=2 (8,24), attn_tp=4 (4,12) and attn_tp=8 (2,6) shards — at
    // the DSpark-relevant row lengths (5-token block, 17-row verify) and one
    // 64-token page.
    const FQ_XCHECK_GEOMS: &[(usize, usize)] = &[(16, 48), (8, 24), (4, 12), (2, 6)];
    const FQ_XCHECK_LENS: &[usize] = &[5, 17, 64];

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Sabotage {
        None,
        ConvOut,
        ConvRing,
        GdrOut,
        GdrState,
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

    /// f64 reference for one varlen slot: conv front end + GDR recurrence.
    struct SlotRef {
        /// Conv output rows the GDR kernel reads (bf16-truncated, token-major).
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

    /// rel-L2 plus a per-element slope+floor bound; returns (pass, rel, n_viol).
    /// `bias` shifts the reference (negative control) and must fail both.
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

    fn ptr_table_ro<T>(ctx: &DeviceContext, slices: &[CudaSlice<T>]) -> Result<CudaSlice<u64>> {
        let ptrs: Vec<u64> = slices.iter().map(|s| s.device_ptr(&ctx.stream).0).collect();
        Ok(ctx.stream.clone_htod(&ptrs)?)
    }

    fn ptr_table_rw<T>(ctx: &DeviceContext, slices: &mut [CudaSlice<T>]) -> Result<CudaSlice<u64>> {
        let ptrs: Vec<u64> = slices
            .iter_mut()
            .map(|s| s.device_ptr_mut(&ctx.stream).0)
            .collect();
        Ok(ctx.stream.clone_htod(&ptrs)?)
    }

    struct DeviceRun {
        lens: Vec<usize>,
        max_len: usize,
        ch: usize,
        vh: usize,
        conv_got: Vec<bf16>,
        gdr_got: Vec<bf16>,
        ring_got: Vec<Vec<bf16>>,
        state_got: Vec<Vec<f32>>,
        refs: Vec<SlotRef>,
    }

    /// One device run for a geometry/batch; device results are compared under a
    /// chosen sabotage by [`compare`].
    fn run_device(ctx: &DeviceContext, kh: usize, vh: usize, b: usize) -> Result<DeviceRun> {
        let ch = 2 * kh * KEY_DIM + vh * VAL_DIM;
        // Offset the cycle per geometry so B=1 covers a long single row
        // ((16,48) -> 5, (8,24) -> the 64-token page); B=4/8 mix every length.
        let geom_off = if (kh, vh) == (16, 48) { 2 } else { 4 };
        let lens: Vec<usize> = (0..b).map(|s| LENS[(s + geom_off) % LENS.len()]).collect();
        let max_len = lens.iter().copied().max().unwrap();
        let mut rng =
            Rng::new(SEED ^ ((kh as u64) << 40) ^ ((vh as u64) << 32) ^ ((b as u64) << 24));

        let w: Vec<bf16> = (0..ch * K)
            .map(|_| bf((rng.normal() * 0.3) as f64))
            .collect();
        let dt: Vec<bf16> = (0..vh).map(|_| bf((rng.normal() * 0.5) as f64)).collect();
        let alog: Vec<f32> = (0..vh).map(|i| -(1.0 + (i as f32 % 3.0) * 0.5)).collect();

        let mut xs = Vec::with_capacity(b);
        let mut bps = Vec::with_capacity(b);
        let mut aps = Vec::with_capacity(b);
        let mut rings0 = Vec::with_capacity(b);
        let mut states0 = Vec::with_capacity(b);
        let mut refs = Vec::with_capacity(b);
        for &len in &lens {
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
            let r = reference_slot(
                kh, vh, ch, len, &x, &bp, &ap, &w, &dt, &alog, &ring0, &state0,
            );
            xs.push(x);
            bps.push(bp);
            aps.push(ap);
            rings0.push(ring0);
            states0.push(state0);
            refs.push(r);
        }

        let w_d = ctx.stream.clone_htod(&w)?;
        let dt_d = ctx.stream.clone_htod(&dt)?;
        let alog_d = ctx.stream.clone_htod(&alog)?;
        let x_d: Vec<_> = xs
            .iter()
            .map(|x| ctx.stream.clone_htod(x))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let bp_d: Vec<_> = bps
            .iter()
            .map(|v| ctx.stream.clone_htod(v))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let ap_d: Vec<_> = aps
            .iter()
            .map(|v| ctx.stream.clone_htod(v))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut ring_d: Vec<_> = rings0
            .iter()
            .map(|r| ctx.stream.clone_htod(r))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut state_d: Vec<_> = states0
            .iter()
            .map(|s| ctx.stream.clone_htod(s))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let lens_i: Vec<i32> = lens.iter().map(|v| *v as i32).collect();
        let lens_d = ctx.stream.clone_htod(&lens_i)?;
        let mut conv_out_d = ctx.stream.alloc_zeros::<bf16>(b * max_len * ch)?;
        let mut gdr_out_d = ctx.stream.alloc_zeros::<bf16>(b * max_len * vh * VAL_DIM)?;

        {
            let x_tbl = ptr_table_ro(ctx, &x_d)?;
            let ring_tbl = ptr_table_rw(ctx, &mut ring_d)?;
            let bp_tbl = ptr_table_ro(ctx, &bp_d)?;
            let ap_tbl = ptr_table_ro(ctx, &ap_d)?;
            let state_tbl = ptr_table_rw(ctx, &mut state_d)?;

            let (xp, _) = x_tbl.device_ptr(&ctx.stream);
            let (wp, _) = w_d.device_ptr(&ctx.stream);
            let (rp, _) = ring_tbl.device_ptr(&ctx.stream);
            let (lp, _) = lens_d.device_ptr(&ctx.stream);
            let (cp, _) = conv_out_d.device_ptr_mut(&ctx.stream);
            conv1d_prefill_varlen_raw(&ctx.stream, xp, wp, rp, lp, cp, ch, max_len, K, b)?;

            let (bp_p, _) = bp_tbl.device_ptr(&ctx.stream);
            let (ap_p, _) = ap_tbl.device_ptr(&ctx.stream);
            let (dt_p, _) = dt_d.device_ptr(&ctx.stream);
            let (al_p, _) = alog_d.device_ptr(&ctx.stream);
            let (st_p, _) = state_tbl.device_ptr(&ctx.stream);
            let (op, _) = gdr_out_d.device_ptr_mut(&ctx.stream);
            gdr_prefill_recurrent_varlen_raw(
                &ctx.stream,
                cp,
                bp_p,
                ap_p,
                dt_p,
                al_p,
                st_p,
                lp,
                op,
                kh,
                vh,
                KEY_DIM,
                VAL_DIM,
                max_len,
                b,
            )?;
            ctx.sync()?;
        }

        let conv_got = ctx.stream.clone_dtoh(&conv_out_d)?;
        let gdr_got = ctx.stream.clone_dtoh(&gdr_out_d)?;
        let ring_got: Vec<Vec<bf16>> = ring_d
            .iter()
            .map(|s| ctx.stream.clone_dtoh(s))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let state_got: Vec<Vec<f32>> = state_d
            .iter()
            .map(|s| ctx.stream.clone_dtoh(s))
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(DeviceRun {
            lens,
            max_len,
            ch,
            vh,
            conv_got,
            gdr_got,
            ring_got,
            state_got,
            refs,
        })
    }

    /// Returns (conv_out_ok, conv_ring_ok, gdr_out_ok, gdr_state_ok).
    fn compare(run: &DeviceRun, sabotage: Sabotage) -> (bool, bool, bool, bool) {
        let mut cout_ok = true;
        let mut ring_ok = true;
        let mut out_ok = true;
        let mut state_ok = true;
        for (s, &len) in run.lens.iter().enumerate() {
            let conv_g: Vec<f64> = run.conv_got
                [s * run.max_len * run.ch..s * run.max_len * run.ch + len * run.ch]
                .iter()
                .map(|v| f64::from(*v))
                .collect();
            let conv_w: Vec<f64> = run.refs[s].conv_out.iter().map(|v| f64::from(*v)).collect();
            let ring_g: Vec<f64> = run.ring_got[s].iter().map(|v| f64::from(*v)).collect();
            let ring_w: Vec<f64> = run.refs[s].ring.iter().map(|v| f64::from(*v)).collect();
            let gdr_g: Vec<f64> = run.gdr_got[s * run.max_len * run.vh * VAL_DIM
                ..s * run.max_len * run.vh * VAL_DIM + len * run.vh * VAL_DIM]
                .iter()
                .map(|v| f64::from(*v))
                .collect();
            let state_g: Vec<f64> = run.state_got[s].iter().map(|v| f64::from(*v)).collect();

            let cout_bias = if sabotage == Sabotage::ConvOut && s == 0 {
                1.0
            } else {
                0.0
            };
            let ring_bias = if sabotage == Sabotage::ConvRing && s == 0 {
                1.0
            } else {
                0.0
            };
            let out_bias = if sabotage == Sabotage::GdrOut && s == 0 {
                1.0
            } else {
                0.0
            };
            let state_bias = if sabotage == Sabotage::GdrState && s == 0 {
                0.5
            } else {
                0.0
            };

            let (c, crel, cviol) = bound(
                &conv_g,
                &conv_w,
                cout_bias,
                CONV_REL_L2_MAX,
                CONV_ABS_FLOOR,
                CONV_ABS_SLOPE,
            );
            let (r, rrel, rviol) = bound(
                &ring_g,
                &ring_w,
                ring_bias,
                CONV_REL_L2_MAX,
                CONV_ABS_FLOOR,
                0.0,
            );
            let (o, orel, oviol) = bound(
                &gdr_g,
                &run.refs[s].gdr_out,
                out_bias,
                OUT_REL_L2_MAX,
                OUT_ABS_FLOOR,
                OUT_ABS_SLOPE,
            );
            let (st, srel, sviol) = bound(
                &state_g,
                &run.refs[s].state,
                state_bias,
                STATE_REL_L2_MAX,
                STATE_ABS_MAX,
                0.0,
            );
            cout_ok &= c;
            ring_ok &= r;
            out_ok &= o;
            state_ok &= st;
            if !c || !r || !o || !st {
                eprintln!(
                    "  slot={s} len={len} conv_out(l2={crel:.3e},v={cviol}) ring(l2={rrel:.3e},v={rviol}) out(l2={orel:.3e},v={oviol}) state(l2={srel:.3e},v={sviol})"
                );
            }
        }
        (cout_ok, ring_ok, out_ok, state_ok)
    }

    pub(super) fn run(negative: bool) -> Result<()> {
        let ctx = DeviceContext::new()?;
        eprintln!(
            "[gdr-varlen-parity] device={} build={}{}",
            ctx.ordinal(),
            cuda_kernels::KERNEL_BUILD_ID,
            if negative { " NEGATIVE-CONTROL" } else { "" }
        );

        for &(kh, vh) in GEOMS {
            for &b in BATCHES {
                let label = format!("geom=({kh},{vh}) B={b}");
                let run = run_device(&ctx, kh, vh, b)?;
                if negative {
                    // Each sabotage trips ONLY its own family.
                    let (c0, r0, o0, s0) = compare(&run, Sabotage::None);
                    ensure!(
                        c0 && r0 && o0 && s0,
                        "{label}: baseline not green before teeth"
                    );
                    let (c1, r1, o1, s1) = compare(&run, Sabotage::ConvOut);
                    ensure!(
                        !c1 && r1 && o1 && s1,
                        "{label}: conv-out tooth dead or leaked"
                    );
                    let (c2, r2, o2, s2) = compare(&run, Sabotage::ConvRing);
                    ensure!(
                        c2 && !r2 && o2 && s2,
                        "{label}: conv-ring tooth dead or leaked"
                    );
                    let (c3, r3, o3, s3) = compare(&run, Sabotage::GdrOut);
                    ensure!(
                        c3 && r3 && !o3 && s3,
                        "{label}: gdr-output tooth dead or leaked"
                    );
                    let (c4, r4, o4, s4) = compare(&run, Sabotage::GdrState);
                    ensure!(
                        c4 && r4 && o4 && !s4,
                        "{label}: gdr-state tooth dead or leaked"
                    );
                    eprintln!("[{label}] teeth OK");
                } else {
                    let (c, r, o, s) = compare(&run, Sabotage::None);
                    ensure!(c && r && o && s, "{label}: gdr_varlen_parity FAILED");
                    eprintln!("[{label}] PASS");
                }
            }
        }

        if negative {
            eprintln!("[gdr-varlen-parity] NEGATIVE CONTROL OK");
            return Ok(());
        }

        let mut fq_ok = true;
        for &(kh, vh) in FQ_XCHECK_GEOMS {
            for &len in FQ_XCHECK_LENS {
                fq_ok &= probe_flashqla(&ctx, kh, vh, len)?;
            }
        }
        ensure!(fq_ok, "flashqla cross-check FAILED");
        eprintln!("[gdr-varlen-parity] ALL PASS");
        Ok(())
    }

    /// Single-slot run at geometry (kh,vh) and `len` through BOTH production
    /// prefill paths on identical inputs and the same nonzero initial state —
    /// the chunked FlashQLA pipeline and the varlen recurrent twin — with each
    /// path anchored on the same f64 reference and the varlen-vs-FQ max diff
    /// printed. Covers global (16,48) and the attn_tp=2 (8,24) / attn_tp=4
    /// (4,12) shards, the fork behind #300, at lengths 5/17/64.
    fn probe_flashqla(ctx: &DeviceContext, kh: usize, vh: usize, len: usize) -> Result<bool> {
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
                _ => unreachable!("cross-check geom not in FQ_XCHECK_GEOMS"),
            };
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

        let w_d = ctx.stream.clone_htod(&w)?;
        let x_d = ctx.stream.clone_htod(&x)?;
        let bp_d = ctx.stream.clone_htod(&bp)?;
        let ap_d = ctx.stream.clone_htod(&ap)?;
        let dt_d = ctx.stream.clone_htod(&dt)?;
        let alog_d = ctx.stream.clone_htod(&alog)?;
        // Two independent ring/state copies: both paths advance them.
        let mut ring_fq_d = ctx.stream.clone_htod(&ring0)?;
        let mut ring_vl_d = ctx.stream.clone_htod(&ring0)?;
        let h0_fq_d = ctx.stream.clone_htod(&state0)?;
        let mut state_vl_d = ctx.stream.clone_htod(&state0)?;
        let mut conv_fq_d = ctx.stream.alloc_zeros::<bf16>(len * ch)?;
        let mut conv_vl_d = ctx.stream.alloc_zeros::<bf16>(len * ch)?;
        let mut o_fq_d = ctx.stream.alloc_zeros::<bf16>(len * vh * VAL_DIM)?;
        let mut o_vl_d = ctx.stream.alloc_zeros::<bf16>(len * vh * VAL_DIM)?;
        let mut ht_fq_d = ctx.stream.alloc_zeros::<f32>(vh * KEY_DIM * VAL_DIM)?;

        let (xp, _) = x_d.device_ptr(&ctx.stream);
        let (wp, _) = w_d.device_ptr(&ctx.stream);
        let (bp_p, _) = bp_d.device_ptr(&ctx.stream);
        let (ap_p, _) = ap_d.device_ptr(&ctx.stream);
        let (dt_p, _) = dt_d.device_ptr(&ctx.stream);
        let (al_p, _) = alog_d.device_ptr(&ctx.stream);

        // The AOT dispatch resolves SM/module via the driver context.
        ctx.ctx.bind_to_thread()?;
        {
            // --- FlashQLA chunked path (single row, one chunk) ---
            let (rp, _) = ring_fq_d.device_ptr_mut(&ctx.stream);
            let (cp, _) = conv_fq_d.device_ptr_mut(&ctx.stream);
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
            let (h0p, _) = h0_fq_d.device_ptr(&ctx.stream);
            let (op, _) = o_fq_d.device_ptr_mut(&ctx.stream);
            let (htp, _) = ht_fq_d.device_ptr_mut(&ctx.stream);
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

            // --- Varlen recurrent twin (B=1) ---
            let x_tbl = {
                let tbl = vec![xp];
                ctx.stream.clone_htod(&tbl)?
            };
            let ring_tbl = {
                let tbl = vec![ring_vl_d.device_ptr_mut(&ctx.stream).0];
                ctx.stream.clone_htod(&tbl)?
            };
            let bp_tbl = {
                let tbl = vec![bp_p];
                ctx.stream.clone_htod(&tbl)?
            };
            let ap_tbl = {
                let tbl = vec![ap_p];
                ctx.stream.clone_htod(&tbl)?
            };
            let state_tbl = {
                let tbl = vec![state_vl_d.device_ptr_mut(&ctx.stream).0];
                ctx.stream.clone_htod(&tbl)?
            };
            let len_d = ctx.stream.clone_htod(&[len as i32])?;
            let (cvp, _) = conv_vl_d.device_ptr_mut(&ctx.stream);
            let (ovp, _) = o_vl_d.device_ptr_mut(&ctx.stream);
            conv1d_prefill_varlen_raw(
                &ctx.stream,
                x_tbl.device_ptr(&ctx.stream).0,
                wp,
                ring_tbl.device_ptr(&ctx.stream).0,
                len_d.device_ptr(&ctx.stream).0,
                cvp,
                ch,
                len,
                K,
                1,
            )?;
            gdr_prefill_recurrent_varlen_raw(
                &ctx.stream,
                cvp,
                bp_tbl.device_ptr(&ctx.stream).0,
                ap_tbl.device_ptr(&ctx.stream).0,
                dt_p,
                al_p,
                state_tbl.device_ptr(&ctx.stream).0,
                len_d.device_ptr(&ctx.stream).0,
                ovp,
                kh,
                vh,
                KEY_DIM,
                VAL_DIM,
                len,
                1,
            )?;
            ctx.sync()?;
        }

        let to_f64 = |s: &[bf16]| s.iter().map(|v| f64::from(*v)).collect::<Vec<f64>>();
        let o_fq = to_f64(&ctx.stream.clone_dtoh(&o_fq_d)?);
        let o_vl = to_f64(&ctx.stream.clone_dtoh(&o_vl_d)?);
        let ht_fq: Vec<f64> = ctx
            .stream
            .clone_dtoh(&ht_fq_d)?
            .iter()
            .map(|v| f64::from(*v))
            .collect();
        let ht_vl: Vec<f64> = ctx
            .stream
            .clone_dtoh(&state_vl_d)?
            .iter()
            .map(|v| f64::from(*v))
            .collect();

        let (fqop, fqorel, fqoviol) = bound(
            &o_fq,
            &reference.gdr_out,
            0.0,
            FQ_OUT_REL_L2_MAX,
            FQ_OUT_ABS_FLOOR,
            FQ_OUT_ABS_SLOPE,
        );
        let (fqsp, fqsrel, fqsviol) = bound(
            &ht_fq,
            &reference.state,
            0.0,
            FQ_STATE_REL_L2_MAX,
            FQ_STATE_ABS_MAX,
            0.0,
        );
        let (vlop, vlorel, vloviol) = bound(
            &o_vl,
            &reference.gdr_out,
            0.0,
            OUT_REL_L2_MAX,
            OUT_ABS_FLOOR,
            OUT_ABS_SLOPE,
        );
        let (vlsp, vlsrel, vlsviol) = bound(
            &ht_vl,
            &reference.state,
            0.0,
            STATE_REL_L2_MAX,
            STATE_ABS_MAX,
            0.0,
        );
        let out_xdiff = o_fq
            .iter()
            .zip(&o_vl)
            .map(|(a, b)| (a - b).abs())
            .fold(0f64, f64::max);
        let state_xdiff = ht_fq
            .iter()
            .zip(&ht_vl)
            .map(|(a, b)| (a - b).abs())
            .fold(0f64, f64::max);
        eprintln!(
            "[gdr-varlen-parity:xcheck] ({kh},{vh}) len={len} fq_out(l2={fqorel:.3e},v={fqoviol}) fq_state(l2={fqsrel:.3e},v={fqsviol}) | varlen_out(l2={vlorel:.3e},v={vloviol}) varlen_state(l2={vlsrel:.3e},v={vlsviol}) | VARLEN-vs-FQ maxdiff out={out_xdiff:.4e} state={state_xdiff:.4e}"
        );
        Ok(fqop && fqsp && vlop && vlsp)
    }
}
