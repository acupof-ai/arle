//! GDR decode / conv1d-decode numeric-parity gate.
//!
//! Standalone kernel harness — NO engine, NO serve. Feeds random BF16 token
//! streams through the production decode front end
//! (`conv1d_decode_batch_cuda` → `gdr_decode_batch_cuda`, plus the singular
//! `gated_delta_rule_decode_cuda` at B=1) and compares, after every decode step
//! with state carried forward, against an f32 CPU reference written
//! independently of the kernels:
//!   - conv output AND the shifted conv ring
//!   - GDR bf16 output AND the mutated f32 recurrent state
//!
//! Geometry: local (key,value) heads (16,48) hd128 and the attn_tp=2 shard
//! (8,24), batch 1 and 8, several recurrent steps. Anchors on f32: the
//! reference starts from the same BF16-truncated conv output the GDR kernel
//! reads and models the kernel's BF16 conv-sum truncation; GDR internal math
//! is f32, so state tolerance is tight. Exits non-zero on any violation.
//!
//! `--negative-control` perturbs one reference element IN EACH comparator
//! family (batch conv_out/conv_state/gdr_out/gdr_state plus the three singular
//! cross-checks); every family MUST independently FAIL — a family whose
//! corrupted comparison still passes has no teeth.
//!
//! Run on a pod: `INFER_CUDA_DEVICE=<free-gpu> target/release/examples/gdr_decode_parity`

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
        eprintln!("gdr_decode_parity is a CUDA harness; rebuild with --features cuda.");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use anyhow::{Result, ensure};
    use cuda_kernels::prelude::DeviceContext;
    use cuda_kernels::recurrent::{conv1d_decode_batch_raw, gdr_decode_batch_raw, gdr_decode_raw};
    use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
    use half::bf16;

    const KEY_DIM: usize = 128;
    const VAL_DIM: usize = 128;
    const K: usize = 4;
    const STEPS: usize = 5;
    const GEOMS: &[(usize, usize)] = &[(16, 48), (8, 24)];
    const BATCHES: &[usize] = &[1, 8];
    const SEEDS: &[u64] = &[0x9e37_79b9_7f4a_7c15, 0x517c_c1cc_9e37_79b9];

    // bf16 conv/GDR output: truncation floor is ~4e-3 rel; caps stay well
    // above it and well below a real recurrence break.
    const OUT_REL_L2_MAX: f64 = 2e-2;
    const OUT_ABS_FLOOR: f32 = 1e-2;
    const OUT_ABS_SLOPE: f32 = 3e-2;
    // f32 recurrent state: reference and kernel share the same f32 recurrence;
    // only rsqrtf/expf ulp error carries across steps.
    const STATE_REL_L2_MAX: f64 = 2e-3;
    const STATE_ABS_MAX: f32 = 2e-3;

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

    fn bf(v: f32) -> bf16 {
        bf16::from_f32(v)
    }

    /// Independent f32 reference for one batched stream (B lanes).
    struct Reference {
        kh: usize,
        vh: usize,
        ch: usize,
        // Per lane: conv ring [ch, K-1] bf16, GDR state [vh*128*128] f32.
        conv: Vec<Vec<bf16>>,
        state: Vec<Vec<f32>>,
        // Last step's expected conv output [B][ch] bf16 and GDR out [B][vh*128] f32.
        conv_out: Vec<Vec<bf16>>,
        gdr_out: Vec<Vec<f32>>,
    }

    impl Reference {
        fn step(
            &mut self,
            x: &[Vec<bf16>],
            w: &[bf16],
            bp: &[Vec<bf16>],
            ap: &[Vec<bf16>],
            dt: &[bf16],
            alog: &[f32],
        ) {
            let b = x.len();
            self.conv_out = vec![vec![bf(0.0); self.ch]; b];
            self.gdr_out = vec![vec![0.0; self.vh * VAL_DIM]; b];
            for lane in 0..b {
                self.conv_step(lane, &x[lane], w);
                self.gdr_step(lane, &bp[lane], &ap[lane], dt, alog);
            }
        }

        /// K=4 depthwise conv + SiLU, faithful to conv1d_decode_batch.cu:
        /// `sum` truncates to bf16 before SiLU; ring stores bf16.
        fn conv_step(&mut self, lane: usize, x: &[bf16], w: &[bf16]) {
            let ring = &mut self.conv[lane];
            let out = &mut self.conv_out[lane];
            for c in 0..self.ch {
                let s0 = f32::from(ring[c * 3]);
                let s1 = f32::from(ring[c * 3 + 1]);
                let s2 = f32::from(ring[c * 3 + 2]);
                let xv = f32::from(x[c]);
                let sum = f32::from(w[c * K]) * s0
                    + f32::from(w[c * K + 1]) * s1
                    + f32::from(w[c * K + 2]) * s2
                    + f32::from(w[c * K + 3]) * xv;
                let sum = f32::from(bf(sum));
                let silu = sum / (1.0 + (-sum).exp());
                out[c] = bf(silu);
                ring[c * 3] = bf(s1);
                ring[c * 3 + 1] = bf(s2);
                ring[c * 3 + 2] = bf(xv);
            }
        }

        /// One GDR decode step per value head; mirrors gdr_decode_batch.cu.
        fn gdr_step(&mut self, lane: usize, bp: &[bf16], ap: &[bf16], dt: &[bf16], alog: &[f32]) {
            let qkv = &self.conv_out[lane];
            let state = &mut self.state[lane];
            let out = &mut self.gdr_out[lane];
            let q_dim = self.kh * KEY_DIM;
            let v_base = 2 * q_dim;
            for vh in 0..self.vh {
                let kh = vh * self.kh / self.vh;
                let mut q = [0f32; KEY_DIM];
                let mut k = [0f32; KEY_DIM];
                let mut v = [0f32; VAL_DIM];
                let mut q_sq = 0f32;
                let mut k_sq = 0f32;
                for d in 0..KEY_DIM {
                    q[d] = f32::from(qkv[kh * KEY_DIM + d]);
                    k[d] = f32::from(qkv[q_dim + kh * KEY_DIM + d]);
                    q_sq += q[d] * q[d];
                    k_sq += k[d] * k[d];
                }
                let q_norm = (q_sq + 1e-12).sqrt().recip() * (KEY_DIM as f32).sqrt().recip();
                let k_norm = (k_sq + 1e-12).sqrt().recip();
                for d in 0..KEY_DIM {
                    q[d] *= q_norm;
                    k[d] *= k_norm;
                }
                for d in 0..VAL_DIM {
                    v[d] = f32::from(qkv[v_base + vh * VAL_DIM + d]);
                }
                let xv = f32::from(ap[vh]) + f32::from(dt[vh]);
                let softplus = if xv > 20.0 { xv } else { (1.0 + xv.exp()).ln() };
                let exp_g = (-alog[vh].exp() * softplus).exp();
                let beta = 1.0 / (1.0 + (-f32::from(bp[vh])).exp());

                let base = vh * KEY_DIM * VAL_DIM;
                let mut kv = [0f32; VAL_DIM];
                for j in 0..KEY_DIM {
                    for i in 0..VAL_DIM {
                        let s = state[base + j * VAL_DIM + i] * exp_g;
                        state[base + j * VAL_DIM + i] = s;
                        kv[i] += s * k[j];
                    }
                }
                let mut delta = [0f32; VAL_DIM];
                for i in 0..VAL_DIM {
                    delta[i] = (v[i] - kv[i]) * beta;
                }
                for j in 0..KEY_DIM {
                    for i in 0..VAL_DIM {
                        state[base + j * VAL_DIM + i] += delta[i] * k[j];
                    }
                }
                for j in 0..KEY_DIM {
                    for i in 0..VAL_DIM {
                        out[vh * VAL_DIM + i] += state[base + j * VAL_DIM + i] * q[j];
                    }
                }
            }
        }
    }

    fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
        let (mut num, mut den) = (0f64, 0f64);
        for (g, r) in got.iter().zip(want) {
            let d = *g as f64 - *r as f64;
            num += d * d;
            den += (*r as f64) * (*r as f64);
        }
        (num / den.max(1e-12)).sqrt()
    }

    /// One fail bit per comparator family, so --negative-control can prove
    /// each family independently has teeth.
    #[derive(Clone, Copy)]
    struct Families {
        batch_conv_out: bool,
        batch_conv_state: bool,
        batch_gdr_out: bool,
        batch_gdr_state: bool,
        sing_gdr_out: bool,
        sing_gdr_state: bool,
        sing_conv_state: bool,
    }
    impl Families {
        fn entries(self) -> [(&'static str, bool); 7] {
            [
                ("batch conv_out", self.batch_conv_out),
                ("batch conv_state", self.batch_conv_state),
                ("batch gdr_out", self.batch_gdr_out),
                ("batch gdr_state", self.batch_gdr_state),
                ("singular gdr_out", self.sing_gdr_out),
                ("singular gdr_state", self.sing_gdr_state),
                ("singular conv_state", self.sing_conv_state),
            ]
        }
    }

    fn check(
        label: &str,
        got: &[f32],
        want: &[f32],
        rel_cap: f64,
        tol: impl Fn(f32) -> f32,
        failed: &mut bool,
    ) {
        let rel = rel_l2(got, want);
        let mut max_excess = f32::NEG_INFINITY;
        for (g, r) in got.iter().zip(want) {
            max_excess = max_excess.max((g - r).abs() - tol(*r));
        }
        let pass = rel.is_finite() && rel <= rel_cap && max_excess.is_finite() && max_excess <= 0.0;
        if !pass {
            *failed = true;
        }
        eprintln!(
            "[{label}] relL2={rel:.4e} maxExcess={max_excess:.4e} {}",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn probe(
        ctx: &DeviceContext,
        kh: usize,
        vh: usize,
        b: usize,
        seed: u64,
        negative: bool,
        fams: &mut Families,
    ) -> Result<()> {
        let ch = 2 * kh * KEY_DIM + vh * VAL_DIM;
        let label = format!("gdr geom=({kh},{vh}) B={b} seed={seed:#x}");
        let mut rng =
            Rng::new(seed ^ ((kh as u64) << 40) ^ ((vh as u64) << 32) ^ ((b as u64) << 24));

        // Shared weights/biases.
        let w: Vec<bf16> = (0..ch * K).map(|_| bf(rng.normal() * 0.3)).collect();
        let dt: Vec<bf16> = (0..vh).map(|_| bf(rng.normal() * 0.5)).collect();
        let alog: Vec<f32> = (0..vh).map(|i| -(1.0 + (i as f32 % 3.0) * 0.5)).collect();

        // Per-lane initial rings and f32 states.
        let conv0: Vec<Vec<bf16>> = (0..b)
            .map(|_| (0..ch * (K - 1)).map(|_| bf(rng.normal() * 0.3)).collect())
            .collect();
        let state0: Vec<Vec<f32>> = (0..b)
            .map(|_| {
                (0..vh * KEY_DIM * VAL_DIM)
                    .map(|_| rng.normal() * 0.02)
                    .collect()
            })
            .collect();

        let mut reference = Reference {
            kh,
            vh,
            ch,
            conv: conv0.clone(),
            state: state0.clone(),
            conv_out: vec![],
            gdr_out: vec![],
        };

        // Device buffers that carry state across steps.
        let w_d = ctx.stream.clone_htod(&w)?;
        let dt_d = ctx.stream.clone_htod(&dt)?;
        let alog_d = ctx.stream.clone_htod(&alog)?;
        let mut conv_d: Vec<CudaSlice<bf16>> = conv0
            .iter()
            .map(|s| ctx.stream.clone_htod(s))
            .collect::<Result<_, _>>()?;
        let mut state_d: Vec<CudaSlice<f32>> = state0
            .iter()
            .map(|s| ctx.stream.clone_htod(s))
            .collect::<Result<_, _>>()?;

        for step in 0..STEPS {
            // Random per-step inputs.
            let x: Vec<Vec<bf16>> = (0..b)
                .map(|_| (0..ch).map(|_| bf(rng.normal() * 0.5)).collect())
                .collect();
            let bp: Vec<bf16> = (0..b * vh).map(|_| bf(rng.normal() * 0.5)).collect();
            let ap: Vec<bf16> = (0..b * vh).map(|_| bf(rng.normal() * 0.5)).collect();
            let bp_lanes: Vec<Vec<bf16>> =
                (0..b).map(|l| bp[l * vh..(l + 1) * vh].to_vec()).collect();
            let ap_lanes: Vec<Vec<bf16>> =
                (0..b).map(|l| ap[l * vh..(l + 1) * vh].to_vec()).collect();

            reference.step(&x, &w, &bp_lanes, &ap_lanes, &dt, &alog);

            // Flatten batch and upload.
            let x_flat: Vec<bf16> = x.concat();
            let x_d = ctx.stream.clone_htod(&x_flat)?;
            let bp_d = ctx.stream.clone_htod(&bp)?;
            let ap_d = ctx.stream.clone_htod(&ap)?;
            let mut conv_out_d = ctx.stream.alloc_zeros::<bf16>(b * ch)?;
            let mut gdr_out_d = ctx.stream.alloc_zeros::<bf16>(b * vh * VAL_DIM)?;

            {
                // Pointer tables stay live across the loop: addresses are fixed
                // because the per-lane slices are never reallocated. The kernels
                // mutate the rings/states through these table addresses.
                let conv_guards: Vec<_> = conv_d
                    .iter_mut()
                    .map(|s| s.device_ptr_mut(&ctx.stream))
                    .collect();
                let conv_ptrs: Vec<u64> = conv_guards.iter().map(|(p, _)| *p).collect();
                let conv_tbl_d = ctx.stream.clone_htod(&conv_ptrs)?;
                let state_guards: Vec<_> = state_d
                    .iter_mut()
                    .map(|s| s.device_ptr_mut(&ctx.stream))
                    .collect();
                let state_ptrs: Vec<u64> = state_guards.iter().map(|(p, _)| *p).collect();
                let state_tbl_d = ctx.stream.clone_htod(&state_ptrs)?;

                let (xp, _gx) = x_d.device_ptr(&ctx.stream);
                let (wp, _gw) = w_d.device_ptr(&ctx.stream);
                let (ctp, _gct) = conv_tbl_d.device_ptr(&ctx.stream);
                let (cop, _gco) = conv_out_d.device_ptr_mut(&ctx.stream);
                conv1d_decode_batch_raw(&ctx.stream, xp, wp, ctp, cop, ch, K, b)?;

                let (bp_p, _gb) = bp_d.device_ptr(&ctx.stream);
                let (ap_p, _ga) = ap_d.device_ptr(&ctx.stream);
                let (dt_p, _gd) = dt_d.device_ptr(&ctx.stream);
                let (al_p, _gl) = alog_d.device_ptr(&ctx.stream);
                let (stp, _gst) = state_tbl_d.device_ptr(&ctx.stream);
                let (op, _go) = gdr_out_d.device_ptr_mut(&ctx.stream);
                gdr_decode_batch_raw(
                    &ctx.stream,
                    cop,
                    bp_p,
                    ap_p,
                    dt_p,
                    al_p,
                    stp,
                    op,
                    kh,
                    vh,
                    KEY_DIM,
                    VAL_DIM,
                    b,
                )?;
                ctx.sync()?;
            }
            let conv_out_got = ctx.stream.clone_dtoh(&conv_out_d)?;
            let gdr_out_bf = ctx.stream.clone_dtoh(&gdr_out_d)?;
            let conv_got: Vec<Vec<bf16>> = conv_d
                .iter()
                .map(|s| ctx.stream.clone_dtoh(s))
                .collect::<Result<_, _>>()?;
            let state_got: Vec<Vec<f32>> = state_d
                .iter()
                .map(|s| ctx.stream.clone_dtoh(s))
                .collect::<Result<_, _>>()?;

            for lane in 0..b {
                let co: Vec<f32> = conv_out_got[lane * ch..(lane + 1) * ch]
                    .iter()
                    .map(|v| f32::from(*v))
                    .collect();
                let mut want_co: Vec<f32> = reference.conv_out[lane]
                    .iter()
                    .map(|v| f32::from(*v))
                    .collect();
                if negative && step == 0 && lane == 0 {
                    want_co[0] += 1.0;
                }
                check(
                    &format!("{label} step={step} lane={lane} conv_out"),
                    &co,
                    &want_co,
                    OUT_REL_L2_MAX,
                    |r| OUT_ABS_FLOOR + OUT_ABS_SLOPE * r.abs(),
                    &mut fams.batch_conv_out,
                );

                let cr_got: Vec<f32> = conv_got[lane].iter().map(|v| f32::from(*v)).collect();
                let mut cr_want: Vec<f32> =
                    reference.conv[lane].iter().map(|v| f32::from(*v)).collect();
                if negative && step == 0 && lane == 0 {
                    cr_want[0] += 1.0;
                }
                check(
                    &format!("{label} step={step} lane={lane} conv_state"),
                    &cr_got,
                    &cr_want,
                    OUT_REL_L2_MAX,
                    |_| OUT_ABS_FLOOR,
                    &mut fams.batch_conv_state,
                );

                let go: Vec<f32> = gdr_out_bf[lane * vh * VAL_DIM..(lane + 1) * vh * VAL_DIM]
                    .iter()
                    .map(|v| f32::from(*v))
                    .collect();
                let mut want_go = reference.gdr_out[lane].clone();
                if negative && step == 0 && lane == 0 {
                    want_go[0] += 1.0;
                }
                check(
                    &format!("{label} step={step} lane={lane} gdr_out"),
                    &go,
                    &want_go,
                    OUT_REL_L2_MAX,
                    |r| OUT_ABS_FLOOR + OUT_ABS_SLOPE * r.abs(),
                    &mut fams.batch_gdr_out,
                );

                if negative && step == 0 && lane == 0 {
                    reference.state[lane][0] += 1.0;
                }
                check(
                    &format!("{label} step={step} lane={lane} gdr_state"),
                    &state_got[lane],
                    &reference.state[lane],
                    STATE_REL_L2_MAX,
                    |_| STATE_ABS_MAX,
                    &mut fams.batch_gdr_state,
                );
                if negative && step == 0 && lane == 0 {
                    reference.state[lane][0] -= 1.0;
                }
            }
        }

        // Singular kernel cross-check at B=1: separate state advanced over the
        // same stream must match the same reference trajectory.
        if b == 1 {
            probe_singular(
                ctx, kh, vh, seed, negative, &w, &dt, &alog, &conv0[0], &state0[0], fams,
            )?;
        }
        Ok(())
    }

    /// Reruns one lane through `gated_delta_rule_decode_cuda` (singular ABI),
    /// regenerating the identical RNG stream, and compares every step.
    #[allow(clippy::too_many_arguments)]
    fn probe_singular(
        ctx: &DeviceContext,
        kh: usize,
        vh: usize,
        seed: u64,
        negative: bool,
        w: &[bf16],
        dt: &[bf16],
        alog: &[f32],
        conv0: &[bf16],
        state0: &[f32],
        fams: &mut Families,
    ) -> Result<()> {
        let ch = 2 * kh * KEY_DIM + vh * VAL_DIM;
        let mut rng = Rng::new(seed ^ ((kh as u64) << 40) ^ ((vh as u64) << 32) ^ (1u64 << 24));
        // Burn the weight/bias draws probe() made before per-step draws.
        for _ in 0..ch * K {
            let _ = rng.normal();
        }
        for _ in 0..vh {
            let _ = rng.normal();
        }
        for _ in 0..conv0.len() {
            let _ = rng.normal();
        }
        for _ in 0..vh * KEY_DIM * VAL_DIM {
            let _ = rng.normal();
        }

        let mut reference = Reference {
            kh,
            vh,
            ch,
            conv: vec![conv0.to_vec()],
            state: vec![state0.to_vec()],
            conv_out: vec![],
            gdr_out: vec![],
        };
        let w_d = ctx.stream.clone_htod(w)?;
        let dt_d = ctx.stream.clone_htod(dt)?;
        let alog_d = ctx.stream.clone_htod(alog)?;
        let mut conv_sd = ctx.stream.clone_htod(conv0)?;
        let mut state_sd = ctx.stream.clone_htod(state0)?;
        let mut out_sd = ctx.stream.alloc_zeros::<bf16>(vh * VAL_DIM)?;
        let mut conv_out_sd = ctx.stream.alloc_zeros::<bf16>(ch)?;

        for step in 0..STEPS {
            let x: Vec<bf16> = (0..ch).map(|_| bf(rng.normal() * 0.5)).collect();
            let bp: Vec<bf16> = (0..vh).map(|_| bf(rng.normal() * 0.5)).collect();
            let ap: Vec<bf16> = (0..vh).map(|_| bf(rng.normal() * 0.5)).collect();
            reference.step(
                std::slice::from_ref(&x),
                w,
                std::slice::from_ref(&bp),
                std::slice::from_ref(&ap),
                dt,
                alog,
            );

            let x_d = ctx.stream.clone_htod(&x)?;
            let bp_d = ctx.stream.clone_htod(&bp)?;
            let ap_d = ctx.stream.clone_htod(&ap)?;
            {
                let (xp, _gx) = x_d.device_ptr(&ctx.stream);
                let (wp, _gw) = w_d.device_ptr(&ctx.stream);
                let (cp, _gc) = conv_sd.device_ptr_mut(&ctx.stream);
                let (cop, _gco) = conv_out_sd.device_ptr_mut(&ctx.stream);
                let conv_tbl = vec![cp];
                let conv_tbl_d = ctx.stream.clone_htod(&conv_tbl)?;
                let (ctp, _gct) = conv_tbl_d.device_ptr(&ctx.stream);
                conv1d_decode_batch_raw(&ctx.stream, xp, wp, ctp, cop, ch, K, 1)?;

                let (bp_p, _gb) = bp_d.device_ptr(&ctx.stream);
                let (ap_p, _ga) = ap_d.device_ptr(&ctx.stream);
                let (dt_p, _gd) = dt_d.device_ptr(&ctx.stream);
                let (al_p, _gl) = alog_d.device_ptr(&ctx.stream);
                let (sp, _gs) = state_sd.device_ptr_mut(&ctx.stream);
                let (op, _go) = out_sd.device_ptr_mut(&ctx.stream);
                gdr_decode_raw(
                    &ctx.stream,
                    cop,
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
                )?;
                ctx.sync()?;
            }
            let out_got = ctx.stream.clone_dtoh(&out_sd)?;
            let state_got = ctx.stream.clone_dtoh(&state_sd)?;
            let conv_got = ctx.stream.clone_dtoh(&conv_sd)?;
            let go: Vec<f32> = out_got.iter().map(|v| f32::from(*v)).collect();
            let mut want_go = reference.gdr_out[0].clone();
            if negative && step == 0 {
                want_go[0] += 1.0;
            }
            check(
                &format!("gdr-singular geom=({kh},{vh}) step={step} gdr_out"),
                &go,
                &want_go,
                OUT_REL_L2_MAX,
                |r| OUT_ABS_FLOOR + OUT_ABS_SLOPE * r.abs(),
                &mut fams.sing_gdr_out,
            );
            if negative && step == 0 {
                reference.state[0][0] += 1.0;
            }
            check(
                &format!("gdr-singular geom=({kh},{vh}) step={step} gdr_state"),
                &state_got,
                &reference.state[0],
                STATE_REL_L2_MAX,
                |_| STATE_ABS_MAX,
                &mut fams.sing_gdr_state,
            );
            if negative && step == 0 {
                reference.state[0][0] -= 1.0;
            }
            let cr: Vec<f32> = conv_got.iter().map(|v| f32::from(*v)).collect();
            let mut cr_want: Vec<f32> = reference.conv[0].iter().map(|v| f32::from(*v)).collect();
            if negative && step == 0 {
                cr_want[0] += 1.0;
            }
            check(
                &format!("gdr-singular geom=({kh},{vh}) step={step} conv_state"),
                &cr,
                &cr_want,
                OUT_REL_L2_MAX,
                |_| OUT_ABS_FLOOR,
                &mut fams.sing_conv_state,
            );
        }
        Ok(())
    }

    pub(super) fn run(negative: bool) -> Result<()> {
        let ctx = DeviceContext::new()?;
        eprintln!(
            "[gdr-parity] device={} build={}{}",
            ctx.ordinal(),
            cuda_kernels::KERNEL_BUILD_ID,
            if negative { " NEGATIVE-CONTROL" } else { "" }
        );

        let mut fams = Families {
            batch_conv_out: false,
            batch_conv_state: false,
            batch_gdr_out: false,
            batch_gdr_state: false,
            sing_gdr_out: false,
            sing_gdr_state: false,
            sing_conv_state: false,
        };
        for &seed in SEEDS {
            for &(kh, vh) in GEOMS {
                for &b in BATCHES {
                    probe(&ctx, kh, vh, b, seed, negative, &mut fams)?;
                }
            }
        }

        if negative {
            let unfired = fams
                .entries()
                .into_iter()
                .filter_map(|(n, fired)| (!fired).then_some(n))
                .collect::<Vec<_>>();
            ensure!(
                unfired.is_empty(),
                "gdr_decode_parity negative control did NOT fail family: {}",
                unfired.join(", ")
            );
            eprintln!(
                "[gdr-parity] NEGATIVE CONTROL OK (all 7 comparator families failed as required)"
            );
            return Ok(());
        }
        let failed = fams
            .entries()
            .into_iter()
            .filter_map(|(n, fired)| fired.then_some(n))
            .collect::<Vec<_>>();
        ensure!(
            failed.is_empty(),
            "gdr_decode_parity FAILED families: {}",
            failed.join(", ")
        );
        eprintln!("[gdr-parity] ALL PASS");
        Ok(())
    }
}
