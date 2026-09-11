//! DSv4 decode-side FP8 grouped MoE + TP Q-repack numeric-parity gate.
//!
//! Standalone kernel harness — NO engine, NO serve. Two decode-side families on
//! the 1×H20 DSv4-Flash FP8 hot path, each anchored on an independent host
//! oracle:
//!
//! 1. `dsv4_fp8_grouped_swiglu_decode` + `dsv4_fp8_grouped_down_decode`
//!    (w8a16: FP8 e4m3 weights, f32 128×128 block scales, BF16 activations).
//!    Production DSv4-Flash-0731 geometry: K=4096 hidden, N=2048 moe
//!    intermediate, 256 experts with top-6 routing at decode batch B=1 and B=8
//!    (up to 48 distinct routed experts, global ids spread non-contiguously over
//!    0..=255), swiglu limit 10.0. Only routed experts carry (distinct,
//!    full-size) weight matrices in a compact group-major pointer table. The f64
//!    oracle sums over the full K and checks EVERY output row (swiglu N=2048,
//!    down N=4096) with a relative-L2 metric plus a per-element slope+floor
//!    bound.
//!
//! 2. `dsv4_tp_q_repack` — exact `[tp,s,h_local,d] -> [s,tp·h_local,d]`
//!    permutation for the FlashMLA AllGather Q shuffle. Checked BIT-EXACT at
//!    TP=2/4/8 over 64 global heads × 512 head dim.
//!
//! The grouped O-LoRA o-proj (o_groups 8) runs FP8 DeepGEMM (its own prefill
//! gate); the dense-BF16 low-rank `wo_a` fallback runs plain cuBLAS
//! `gemm_cuda`, vendor-trusted — no o-proj oracle is added here.
//!
//! `--negative-control` corrupts each comparison family (swiglu / down /
//! q-repack) independently and asserts ONLY that family trips. It prints
//! NEGATIVE CONTROL OK and exits 0; the teeth assertions are internal.
//!
//! Run on a pod:
//! `INFER_CUDA_DEVICE=<free-gpu> target/release/examples/dsv4_decode_moe_parity`

fn main() -> anyhow::Result<()> {
    use parity_common::Parsed;
    match parity_common::cli() {
        Parsed::BuildIdPrinted => Ok(()),
        Parsed::Run(cli) => real::run(cli.negative),
    }
}

#[allow(dead_code)]
#[path = "support/parity_common.rs"]
mod parity_common;

#[cfg(not(feature = "cuda"))]
mod real {
    pub(super) fn run(_negative: bool) -> anyhow::Result<()> {
        eprintln!("dsv4_decode_moe_parity is a CUDA harness; rebuild with --features cuda.");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use anyhow::{Result, ensure};
    use cuda_kernels::moe;
    use cuda_kernels::prelude::DeviceContext;
    use cuda_kernels::tensor::cache_ptr;
    use cudarc::driver::{CudaSlice, DevicePtr};
    use half::bf16;

    use super::parity_common::{Families, Rng};

    // Production DSv4-Flash-0731 MoE widths.
    const HIDDEN_K: usize = 4096;
    const INTER_N: usize = 2048;
    const SCALE_BLK: usize = 128;
    const SWIGLU_LIMIT: f64 = 10.0;
    // Production fan-out: 256 experts, top-6 per token, B=1 and B=8 decode.
    const TOTAL_EXPERTS: usize = 256;
    const TOPK: usize = 6;
    const BATCHES: &[usize] = &[1, 8];
    // Relative-L2 + per-element slope/floor bounds (two fp8 operands, ~K/128
    // scale-block accumulations; floor covers near-zero outputs rel-L2 can't).
    const MAX_REL_L2: f64 = 0.06;
    const ABS_SLOPE: f64 = 0.10;
    const ABS_FLOOR: f64 = 2e-2;
    // Q-repack: 64 global FlashMLA heads, 512 head dim.
    const H_GLOBAL: usize = 64;
    const HEAD_D: usize = 512;
    const TP_WORLDS: &[i32] = &[2, 4, 8];
    const SEED: u64 = 0xD5D4_DEC0_DE00_0022;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Sabotage {
        None,
        Swiglu,
        Down,
        QRepack,
    }

    fn rint_ne(x: f32) -> i32 {
        let fl = x.floor();
        let frac = x - fl;
        let fl_i = fl as i32;
        if frac > 0.5 || (frac == 0.5 && fl_i % 2 == 1) {
            fl_i + 1
        } else {
            fl_i
        }
    }

    /// OCP E4M3 (no infinities), clips ±448, matching `__nv_fp8_e4m3`.
    fn f32_to_e4m3(value: f32) -> u8 {
        if !value.is_finite() || value == 0.0 {
            return 0;
        }
        let sign = u8::from(value < 0.0) << 7;
        let a = value.abs();
        if a >= 448.0 {
            return sign | 0x7e;
        }
        let exp = a.log2().floor() as i32;
        let rb = exp + 7;
        if rb <= 0 {
            let mant = rint_ne(a * (6.0f32).exp2() * 8.0);
            if mant == 0 {
                return 0;
            }
            if mant >= 8 {
                return sign | (1 << 3);
            }
            return sign | mant as u8;
        }
        let mut mant = rint_ne((a * (-exp as f32).exp2() - 1.0) * 8.0);
        let mut ef = rb;
        if mant >= 8 {
            ef += 1;
            mant = 0;
        }
        if ef > 15 {
            return sign | 0x7e;
        }
        sign | ((ef as u8) << 3) | mant as u8
    }

    fn e4m3_to_f64(byte: u8) -> f64 {
        if byte & 0x7f == 0 {
            return 0.0;
        }
        let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
        let e = ((byte >> 3) & 0x0f) as i32;
        let m = f64::from(byte & 0x07);
        let v = if e == 0 {
            (m / 8.0) * (-6.0f64).exp2()
        } else {
            (1.0 + m / 8.0) * f64::from(e - 7).exp2()
        };
        sign * v
    }

    fn silu(g: f64) -> f64 {
        g / (1.0 + (-g).exp())
    }

    struct Fp8Matrix {
        w: Vec<u8>,
        scale: Vec<f32>,
        n: usize,
        k: usize,
    }
    impl Fp8Matrix {
        fn scale_cols(&self) -> usize {
            self.k / SCALE_BLK
        }
        fn scale_at(&self, row: usize, col: usize) -> f64 {
            self.scale[(row / SCALE_BLK) * self.scale_cols() + (col / SCALE_BLK)] as f64
        }
    }

    /// Per-128×128-block FP8 quantization of an N(0,0.25) target; f32 scales.
    fn gen_matrix(rng: &mut Rng, n: usize, k: usize) -> Fp8Matrix {
        let sr = n / SCALE_BLK;
        let sc = k / SCALE_BLK;
        let mut raw = vec![0f32; n * k];
        for v in raw.iter_mut() {
            *v = rng.normal() * 0.25;
        }
        let mut scale = vec![0f32; sr * sc];
        let mut w = vec![0u8; n * k];
        for br in 0..sr {
            for bc in 0..sc {
                let mut amax = 0f32;
                for r in br * SCALE_BLK..(br + 1) * SCALE_BLK {
                    for c in bc * SCALE_BLK..(bc + 1) * SCALE_BLK {
                        amax = amax.max(raw[r * k + c].abs());
                    }
                }
                let s = (amax / 448.0).max(1e-12);
                scale[br * sc + bc] = s;
                for r in br * SCALE_BLK..(br + 1) * SCALE_BLK {
                    for c in bc * SCALE_BLK..(bc + 1) * SCALE_BLK {
                        w[r * k + c] = f32_to_e4m3(raw[r * k + c] / s);
                    }
                }
            }
        }
        Fp8Matrix { w, scale, n, k }
    }

    fn ptr_table(ctx: &DeviceContext, slices: &[CudaSlice<u8>]) -> Result<CudaSlice<u64>> {
        let mut table = Vec::with_capacity(slices.len());
        for s in slices {
            let (p, _g) = s.device_ptr(&ctx.stream);
            table.push(p);
        }
        Ok(ctx.stream.clone_htod(&table)?)
    }

    /// f64 swiglu oracle over ALL N output rows (full K), pre-bf16 rounding.
    fn oracle_swiglu(gate: &Fp8Matrix, up: &Fp8Matrix, x: &[bf16]) -> Vec<f64> {
        let k = gate.k;
        let n = gate.n;
        let mut out = vec![0f64; n];
        for (row, o) in out.iter_mut().enumerate() {
            let (mut g, mut u) = (0f64, 0f64);
            for (c, xv) in x.iter().enumerate() {
                let xv = f64::from(*xv);
                g += e4m3_to_f64(gate.w[row * k + c]) * gate.scale_at(row, c) * xv;
                u += e4m3_to_f64(up.w[row * k + c]) * up.scale_at(row, c) * xv;
            }
            *o = silu(g.min(SWIGLU_LIMIT)) * u.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
        }
        out
    }

    /// f64 down oracle over ALL N rows; activations are the device bf16 swiglu out.
    fn oracle_down(down: &Fp8Matrix, act_bf16: &[bf16]) -> Vec<f64> {
        let k = down.k;
        let n = down.n;
        let mut out = vec![0f64; n];
        for (row, o) in out.iter_mut().enumerate() {
            let mut acc = 0f64;
            for (c, av) in act_bf16.iter().enumerate() {
                acc += e4m3_to_f64(down.w[row * k + c]) * down.scale_at(row, c) * f64::from(*av);
            }
            *o = acc;
        }
        out
    }

    struct MoeOutputs {
        n_experts: usize,
        /// Compact expert id for each route index i = token*TOPK + choice.
        compact: Vec<usize>,
        /// Packed (group-major) slot order: slot -> route index.
        packed_route: Vec<usize>,
        x_host: Vec<bf16>,
        got_act: Vec<bf16>,
        got_out: Vec<bf16>,
        gate: Vec<Fp8Matrix>,
        up: Vec<Fp8Matrix>,
        down: Vec<Fp8Matrix>,
    }

    /// Global expert for route index `i` (= token*TOPK + choice). `53` is coprime
    /// to 256, so the first B*TOPK picks are distinct; ids span 0..=255
    /// non-contiguously (i=0 -> 255).
    fn global_expert(i: usize) -> usize {
        255 - (i * 53 % TOTAL_EXPERTS)
    }

    /// Run both decode kernels for one batch at production fan-out. Each routed
    /// expert gets a distinct full-size weight matrix in a compact pointer table
    /// (the safe wrappers pass expert_indices=null, so the table is indexed by
    /// compact group id — the group-major production layout).
    fn run_moe_batch(ctx: &DeviceContext, batch: usize) -> Result<MoeOutputs> {
        let mut rng = Rng::new(SEED ^ batch as u64);
        let total_routes = batch * TOPK;

        let mut globals: Vec<usize> = (0..total_routes).map(global_expert).collect();
        globals.sort_unstable();
        globals.dedup();
        let n_experts = globals.len();
        let compact: Vec<usize> = (0..total_routes)
            .map(|i| globals.binary_search(&global_expert(i)).unwrap())
            .collect();

        let mut x_host = vec![bf16::ZERO; total_routes * HIDDEN_K];
        for v in x_host.iter_mut() {
            *v = bf16::from_f32(rng.normal() * 0.5);
        }
        let gate: Vec<Fp8Matrix> = (0..n_experts)
            .map(|_| gen_matrix(&mut rng, INTER_N, HIDDEN_K))
            .collect();
        let up: Vec<Fp8Matrix> = (0..n_experts)
            .map(|_| gen_matrix(&mut rng, INTER_N, HIDDEN_K))
            .collect();
        let down: Vec<Fp8Matrix> = (0..n_experts)
            .map(|_| gen_matrix(&mut rng, HIDDEN_K, INTER_N))
            .collect();

        let mut counts = vec![0i32; n_experts];
        for &e in &compact {
            counts[e] += 1;
        }
        let mut offsets = vec![0i32; n_experts];
        for e in 1..n_experts {
            offsets[e] = offsets[e - 1] + counts[e - 1];
        }
        let mut packed_route: Vec<usize> = Vec::with_capacity(total_routes);
        for e in 0..n_experts {
            for (i, &ce) in compact.iter().enumerate() {
                if ce == e {
                    packed_route.push(i);
                }
            }
        }
        let mut x_packed = Vec::with_capacity(total_routes * HIDDEN_K);
        for &i in &packed_route {
            x_packed.extend_from_slice(&x_host[i * HIDDEN_K..(i + 1) * HIDDEN_K]);
        }

        let x_d = ctx.stream.clone_htod(&x_packed)?;
        type Tables = (Vec<CudaSlice<u8>>, Vec<CudaSlice<u8>>);
        let upload = |mats: &[Fp8Matrix]| -> Result<Tables> {
            let mut wbuf = Vec::new();
            let mut sbuf = Vec::new();
            for m in mats {
                wbuf.push(ctx.stream.clone_htod(&m.w)?);
                let sbytes: Vec<u8> = m.scale.iter().flat_map(|s| s.to_ne_bytes()).collect();
                sbuf.push(ctx.stream.clone_htod(&sbytes)?);
            }
            Ok((wbuf, sbuf))
        };
        let (gw, gs) = upload(&gate)?;
        let (uw, us) = upload(&up)?;
        let (dw, ds) = upload(&down)?;
        let gwp = ptr_table(ctx, &gw)?;
        let gsp = ptr_table(ctx, &gs)?;
        let uwp = ptr_table(ctx, &uw)?;
        let usp = ptr_table(ctx, &us)?;
        let dwp = ptr_table(ctx, &dw)?;
        let dsp = ptr_table(ctx, &ds)?;
        let counts_d = ctx.stream.clone_htod(&counts)?;
        let offsets_d = ctx.stream.clone_htod(&offsets)?;
        let act_d = ctx.stream.alloc_zeros::<bf16>(total_routes * INTER_N)?;
        let out_d = ctx.stream.alloc_zeros::<bf16>(total_routes * HIDDEN_K)?;

        // SAFETY: compact tables hold n_experts full [N,K] matrices; packed x is
        // total_routes×K; counts/offsets are group-major over the same compact ids.
        unsafe {
            moe::dsv4_fp8_grouped_swiglu_decode(
                cache_ptr(&gwp, ctx),
                cache_ptr(&gsp, ctx),
                cache_ptr(&uwp, ctx),
                cache_ptr(&usp, ctx),
                cache_ptr(&x_d, ctx),
                cache_ptr(&act_d, ctx),
                cache_ptr(&offsets_d, ctx),
                cache_ptr(&counts_d, ctx),
                n_experts,
                total_routes,
                INTER_N,
                HIDDEN_K,
                HIDDEN_K / SCALE_BLK,
                SWIGLU_LIMIT as f32,
                ctx.stream.cu_stream(),
            )?;
        }
        ctx.sync()?;
        let got_act = ctx.stream.clone_dtoh(&act_d)?;

        // SAFETY: act total_routes×INTER, out total_routes×HIDDEN.
        unsafe {
            moe::dsv4_fp8_grouped_down_decode(
                cache_ptr(&dwp, ctx),
                cache_ptr(&dsp, ctx),
                cache_ptr(&act_d, ctx),
                cache_ptr(&out_d, ctx),
                cache_ptr(&offsets_d, ctx),
                cache_ptr(&counts_d, ctx),
                n_experts,
                total_routes,
                HIDDEN_K,
                INTER_N,
                INTER_N / SCALE_BLK,
                ctx.stream.cu_stream(),
            )?;
        }
        ctx.sync()?;
        let got_out = ctx.stream.clone_dtoh(&out_d)?;

        Ok(MoeOutputs {
            n_experts,
            compact,
            packed_route,
            x_host,
            got_act,
            got_out,
            gate,
            up,
            down,
        })
    }

    struct Bound {
        rel_l2: f64,
        violators: usize,
        pass: bool,
    }

    /// Relative-L2 plus a per-element slope+floor bound over every element.
    /// `bias` shifts the expected values (negative control) and must fail both.
    fn elementwise(got: &[bf16], want: &[f64], bias: f64) -> Bound {
        let mut diff_sq = 0f64;
        let mut ref_sq = 0f64;
        let mut violators = 0usize;
        for (g, w0) in got.iter().zip(want.iter()) {
            let g = f64::from(*g);
            let w = w0 + bias;
            let d = (g - w).abs();
            diff_sq += d.powi(2);
            ref_sq += w.powi(2);
            if d > ABS_FLOOR + ABS_SLOPE * w.abs() {
                violators += 1;
            }
        }
        let rel_l2 = (diff_sq / ref_sq.max(1e-12)).sqrt();
        Bound {
            rel_l2,
            violators,
            pass: rel_l2 < MAX_REL_L2 && violators == 0,
        }
    }

    /// Evaluate both MoE families for one batch over ALL output rows. A sabotage
    /// perturbs only that family's oracle, so the other stays green.
    fn compare_moe(o: &MoeOutputs, sabotage: Sabotage) -> (bool, bool, f64, f64) {
        let swiglu_bias = if sabotage == Sabotage::Swiglu {
            1.0
        } else {
            0.0
        };
        let down_bias = if sabotage == Sabotage::Down { 1.0 } else { 0.0 };

        // Per-route f64 oracles in parallel (full N×K each); index by slot.
        let swiglu_want: std::sync::Mutex<Vec<Vec<f64>>> =
            std::sync::Mutex::new(vec![Vec::new(); o.packed_route.len()]);
        let down_want: std::sync::Mutex<Vec<Vec<f64>>> =
            std::sync::Mutex::new(vec![Vec::new(); o.packed_route.len()]);
        std::thread::scope(|s| {
            for (slot, &route) in o.packed_route.iter().enumerate() {
                let e = o.compact[route];
                let x = &o.x_host[route * HIDDEN_K..(route + 1) * HIDDEN_K];
                let act = &o.got_act[slot * INTER_N..(slot + 1) * INTER_N];
                let (gate, up, down) = (&o.gate[e], &o.up[e], &o.down[e]);
                let (sw, dw) = (&swiglu_want, &down_want);
                s.spawn(move || {
                    sw.lock().unwrap()[slot] = oracle_swiglu(gate, up, x);
                    dw.lock().unwrap()[slot] = oracle_down(down, act);
                });
            }
        });
        let mut swiglu_want = swiglu_want.into_inner().unwrap();
        let mut down_want = down_want.into_inner().unwrap();

        let mut sw_flat = Vec::with_capacity(o.packed_route.len() * INTER_N);
        let mut dw_flat = Vec::with_capacity(o.packed_route.len() * HIDDEN_K);
        for slot in 0..o.packed_route.len() {
            sw_flat.append(&mut swiglu_want[slot]);
            dw_flat.append(&mut down_want[slot]);
        }

        let sb = elementwise(&o.got_act, &sw_flat, swiglu_bias);
        let db = elementwise(&o.got_out, &dw_flat, down_bias);
        eprintln!(
            "[dsv4-decode-parity:moe] B={} experts={} routes={} swiglu(l2={:.3e},viol={}) down(l2={:.3e},viol={})",
            o.compact.len() / TOPK,
            o.n_experts,
            o.compact.len(),
            sb.rel_l2,
            sb.violators,
            db.rel_l2,
            db.violators,
        );
        (sb.pass, db.pass, sb.rel_l2, db.rel_l2)
    }

    fn run_q_repack(ctx: &DeviceContext, sabotage: Sabotage) -> Result<bool> {
        let s_q: usize = 2;
        let mut all_ok = true;
        for &tp in TP_WORLDS {
            let tp = tp as usize;
            let h_local = H_GLOBAL / tp;
            let n = tp * s_q * h_local * HEAD_D;
            let gathered: Vec<bf16> = (0..n)
                .map(|i| bf16::from_f32((i as u32 % 4091) as f32 * 0.001))
                .collect();
            let g_d = ctx.stream.clone_htod(&gathered)?;
            let p_d = ctx.stream.alloc_zeros::<bf16>(s_q * H_GLOBAL * HEAD_D)?;
            {
                let (gp, _gg) = g_d.device_ptr(&ctx.stream);
                let (pp, _pg) = p_d.device_ptr(&ctx.stream);
                // SAFETY: gathered [tp,s,h_local,d], packed [s,tp*h_local,d].
                cuda_kernels::attention::dsv4_tp_q_repack_raw(
                    &ctx.stream,
                    gp,
                    pp,
                    tp as i32,
                    s_q as i32,
                    h_local as i32,
                    HEAD_D as i32,
                )?;
            }
            ctx.sync()?;
            let packed = ctx.stream.clone_dtoh(&p_d)?;

            let mut ok = true;
            // Expected source index for packed dst (s,w,h,k). In sabotage mode
            // use the UN-transposed [tp,s,h_local,d] index — which must NOT equal
            // the repacked value at a moved location, so the comparator trips.
            let src_idx = |s: usize, w: usize, h: usize, k: usize| -> usize {
                if sabotage == Sabotage::QRepack && tp == TP_WORLDS[0] as usize {
                    // Same flat offset as dst interpreted in the gathered layout.
                    ((s * tp + w) * h_local + h) * HEAD_D + k
                } else {
                    ((w * s_q + s) * h_local + h) * HEAD_D + k
                }
            };
            for s in 0..s_q {
                for w in 0..tp {
                    for h in 0..h_local {
                        for k in 0..HEAD_D {
                            let dst = (s * H_GLOBAL + w * h_local + h) * HEAD_D + k;
                            if packed[dst].to_bits() != gathered[src_idx(s, w, h, k)].to_bits() {
                                ok = false;
                            }
                        }
                    }
                }
            }
            eprintln!("[dsv4-decode-parity:qrepack] tp={tp} h_local={h_local} bitexact={ok}");
            all_ok &= ok;
        }
        Ok(all_ok)
    }

    pub(super) fn run(negative: bool) -> Result<()> {
        let ctx = DeviceContext::new()?;
        eprintln!(
            "[dsv4-decode-parity] device={} build={}{}",
            ctx.ordinal(),
            cuda_kernels::KERNEL_BUILD_ID,
            if negative { " NEGATIVE-CONTROL" } else { "" }
        );

        if negative {
            // Each family must trip on its own corruption while the others stay
            // green. Use the widest batch so every family sees production fan-out.
            // The cross-family isolation asserts are stricter than the shared
            // Families aggregator (which records one bit per family), so they
            // stay here; Families still records that each family has teeth.
            let moe = run_moe_batch(&ctx, *BATCHES.last().unwrap())?;
            let (sw, dn, _, _) = compare_moe(&moe, Sabotage::Swiglu);
            ensure!(!sw, "swiglu negative did not trip — comparator dead");
            ensure!(dn, "swiglu sabotage wrongly tripped down");
            let (sw2, dn2, _, _) = compare_moe(&moe, Sabotage::Down);
            ensure!(!dn2, "down negative did not trip — comparator dead");
            ensure!(sw2, "down sabotage wrongly tripped swiglu");
            let qbad = run_q_repack(&ctx, Sabotage::QRepack)?;
            ensure!(!qbad, "q-repack negative did not trip — comparator dead");
            let mut families = Families::new();
            // Comparator pass=false under its own sabotage = that family
            // correctly failed; record the failure bit (`!pass`).
            families.record("swiglu", !sw);
            families.record("down", !dn2);
            families.record("q-repack", !qbad);
            return families.finish("dsv4-decode-parity", true);
        }

        let mut all_ok = true;
        for &b in BATCHES {
            let moe = run_moe_batch(&ctx, b)?;
            let (swiglu_ok, down_ok, _, _) = compare_moe(&moe, Sabotage::None);
            all_ok &= swiglu_ok && down_ok;
        }
        let q_ok = run_q_repack(&ctx, Sabotage::None)?;
        let mut families = Families::new();
        families.record("moe swiglu/down", !all_ok);
        families.record("q-repack", !q_ok);
        families.finish("dsv4-decode-parity", false)
    }
}
