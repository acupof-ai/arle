//! Elementwise layer-kernel numeric-parity gate.
//!
//! Standalone kernel harness — NO engine, NO serve. Covers the HOT per-layer
//! elementwise family against independently written f64/f32 references:
//!   - `rms_norm_cuda` (single row; BF16 round BETWEEN scale and weight,
//!     HuggingFace pattern) and `rms_norm_batched_cuda` (fused fp32 scale ×
//!     weight, one BF16 round at the store — the aligned production path)
//!   - `silu_mul_cuda` (separate gate/up) and `silu_mul_fused_cuda`
//!     (row-fused gate_up): SiLU(gate) * up
//!   - `split2_cuda` / `split_qkv_cuda`: layout copies, bit-exact
//!   - `embedding_batched_cuda`: table gather, bit-exact
//!
//! Production geometry: Qwen3.6-27B (hidden 5120, inter 17408, split_qkv
//! q=6144/kv=1024, 24×256-attn / 4 KV heads) and DSv4 backbone (hidden 7168,
//! inter 20480), batch 1 and 8; RMSNorm eps = 1e-6 (the model config value).
//!
//! `--negative-control` perturbs one reference element; the gate MUST FAIL.
//!
//! Run on a pod: `INFER_CUDA_DEVICE=<free-gpu> target/release/examples/elementwise_parity`

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
        eprintln!("elementwise_parity is a CUDA harness; rebuild with --features cuda.");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use anyhow::{Result, ensure};
    use cuda_kernels::prelude::DeviceContext;
    use cuda_kernels::tensor_ops::{
        embedding_batched, rms_norm, rms_norm_batched, silu_mul, silu_mul_fused, split_qkv, split2,
    };
    use half::bf16;

    const EPS: f64 = 1e-6;
    const BATCHES: &[usize] = &[1, 8];
    const SEED: u64 = 0xe1e4_1515_e1e4_1515;
    // (label, hidden, inter) — Qwen3.6-27B dense and the DSv4 backbone.
    const MODELS: &[(&str, usize, usize)] = &[("qwen36-27b", 5120, 17_408), ("dsv4", 7168, 20_480)];
    // 27B attention geometry: q 24×256=6144, kv 4×256=1024.
    const Q_DIM: usize = 6144;
    const KV_DIM: usize = 1024;
    const EMBED_ROWS: usize = 256;

    // BF16 outputs off f64 anchors: truncation floor ~4e-3 rel.
    const REL_L2_MAX: f64 = 2e-2;
    const ABS_FLOOR: f32 = 3e-3;
    const ABS_SLOPE: f32 = 3e-2;

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

    fn rel_l2(got: &[bf16], want: &[f64]) -> f64 {
        let (mut num, mut den) = (0f64, 0f64);
        for (g, r) in got.iter().zip(want) {
            let d = f32::from(*g) as f64 - *r;
            num += d * d;
            den += r * r;
        }
        (num / den.max(1e-12)).sqrt()
    }

    fn check(label: &str, got: &[bf16], want: &[f64], any_fail: &mut bool) {
        let rel = rel_l2(got, want);
        let mut max_excess = f32::NEG_INFINITY;
        for (g, r) in got.iter().zip(want) {
            let tol = ABS_FLOOR + ABS_SLOPE * *r as f32;
            max_excess = max_excess.max((f32::from(*g) - *r as f32).abs() - tol);
        }
        let pass =
            rel.is_finite() && rel <= REL_L2_MAX && max_excess.is_finite() && max_excess <= 0.0;
        if !pass {
            *any_fail = true;
        }
        eprintln!(
            "[{label}] relL2={rel:.4e} maxExcess={max_excess:.4e} {}",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    fn check_bitexact(label: &str, got: &[bf16], want: &[bf16], any_fail: &mut bool) {
        let mut bad = 0usize;
        for (a, b) in got.iter().zip(want) {
            if a.to_bits() != b.to_bits() {
                bad += 1;
            }
        }
        let pass = bad == 0 && got.len() == want.len();
        if !pass {
            *any_fail = true;
        }
        eprintln!(
            "[{label}] n={} mismatches={bad} {}",
            got.len(),
            if pass { "PASS" } else { "FAIL" }
        );
    }

    /// f64 RMSNorm anchor. `round_between` mirrors rms_norm_cuda: truncate
    /// x*inv_rms to BF16 before the weight multiply; the batched aligned path
    /// keeps x*inv_rms*weight in fp32 to the store.
    fn rms_ref(x: &[bf16], w: &[bf16], round_between: bool) -> Vec<f64> {
        let n = x.len();
        let sum: f64 = x
            .iter()
            .map(|v| {
                let v = f32::from(*v) as f64;
                v * v
            })
            .sum();
        let inv = (sum / n as f64 + EPS).sqrt().recip();
        x.iter()
            .zip(w)
            .map(|(xi, wi)| {
                let xi = f32::from(*xi) as f64;
                let wi = f32::from(*wi) as f64;
                if round_between {
                    f32::from(bf16::from_f32((xi * inv) as f32)) as f64 * wi
                } else {
                    xi * inv * wi
                }
            })
            .collect()
    }

    fn silu_ref(gate: &[bf16], up: &[bf16]) -> Vec<f64> {
        gate.iter()
            .zip(up)
            .map(|(g, u)| {
                let g = f32::from(*g) as f64;
                let u = f32::from(*u) as f64;
                g / (1.0 + (-g).exp()) * u
            })
            .collect()
    }

    fn probe_rms(ctx: &DeviceContext, negative: bool, any_fail: &mut bool) -> Result<()> {
        let mut rng = Rng::new(SEED ^ 0x524d_5300);
        for &(name, hidden, _) in MODELS {
            let w: Vec<bf16> = (0..hidden)
                .map(|_| bf16::from_f32(1.0 + rng.normal() * 0.1))
                .collect();
            let w_d = ctx.stream.clone_htod(&w)?;
            for &rows in BATCHES {
                let x: Vec<bf16> = (0..rows * hidden)
                    .map(|_| bf16::from_f32(rng.normal() * 0.5))
                    .collect();
                let x_d = ctx.stream.clone_htod(&x)?;
                let mut out_d = ctx.stream.alloc_zeros::<bf16>(rows * hidden)?;

                // Batched: aligned multiples of 8 take the fused fp32 path.
                rms_norm_batched(ctx, &x_d, 0, &w_d, &mut out_d, hidden, rows, EPS as f32)?;
                ctx.sync()?;
                let got = ctx.stream.clone_dtoh(&out_d)?;
                for lane in 0..rows {
                    let xr = &x[lane * hidden..(lane + 1) * hidden];
                    let mut want = rms_ref(xr, &w, false);
                    if negative && lane == 0 {
                        want[0] += 0.1;
                    }
                    check(
                        &format!("rms-batched {name} h={hidden} B={rows} lane={lane}"),
                        &got[lane * hidden..(lane + 1) * hidden],
                        &want,
                        any_fail,
                    );
                }

                if rows == 1 {
                    // Singular kernel: BF16 round between scale and weight.
                    let mut one_d = ctx.stream.alloc_zeros::<bf16>(hidden)?;
                    rms_norm(ctx, &x_d, &w_d, &mut one_d, hidden, EPS as f32)?;
                    ctx.sync()?;
                    let got = ctx.stream.clone_dtoh(&one_d)?;
                    let want = rms_ref(&x[..hidden], &w, true);
                    check(
                        &format!("rms-single {name} h={hidden}"),
                        &got,
                        &want,
                        any_fail,
                    );
                }
            }
        }
        Ok(())
    }

    fn probe_silu_split_embed(
        ctx: &DeviceContext,
        negative: bool,
        any_fail: &mut bool,
    ) -> Result<()> {
        let mut rng = Rng::new(SEED ^ 0x5349_4c55);
        for &(name, hidden, inter) in MODELS {
            for &rows in BATCHES {
                // ── silu_mul (separate gate/up) ──
                let gate: Vec<bf16> = (0..inter)
                    .map(|_| bf16::from_f32(rng.normal() * 0.3))
                    .collect();
                let up: Vec<bf16> = (0..inter)
                    .map(|_| bf16::from_f32(rng.normal() * 0.3))
                    .collect();
                let mut want = silu_ref(&gate, &up);
                if negative {
                    want[0] += 0.1;
                }
                let gate_d = ctx.stream.clone_htod(&gate)?;
                let up_d = ctx.stream.clone_htod(&up)?;
                let mut out_d = ctx.stream.alloc_zeros::<bf16>(inter)?;
                silu_mul(ctx, &gate_d, &up_d, &mut out_d, inter)?;
                ctx.sync()?;
                check(
                    &format!("silu_mul {name} inter={inter} B=1"),
                    &ctx.stream.clone_dtoh(&out_d)?,
                    &want,
                    any_fail,
                );

                // ── silu_mul_fused [B, 2*inter] ──
                let mut gu: Vec<bf16> = Vec::with_capacity(rows * 2 * inter);
                let mut wants = Vec::with_capacity(rows);
                for lane in 0..rows {
                    let g: Vec<bf16> = (0..inter)
                        .map(|_| bf16::from_f32(rng.normal() * 0.3))
                        .collect();
                    let u: Vec<bf16> = (0..inter)
                        .map(|_| bf16::from_f32(rng.normal() * 0.3))
                        .collect();
                    let mut w = silu_ref(&g, &u);
                    if negative && lane == 0 {
                        w[0] += 0.1;
                    }
                    wants.push(w);
                    gu.extend(g);
                    gu.extend(u);
                }
                let gu_d = ctx.stream.clone_htod(&gu)?;
                let mut fused_d = ctx.stream.alloc_zeros::<bf16>(rows * inter)?;
                silu_mul_fused(ctx, &gu_d, &mut fused_d, rows, inter)?;
                ctx.sync()?;
                let fused_got = ctx.stream.clone_dtoh(&fused_d)?;
                for lane in 0..rows {
                    check(
                        &format!("silu_fused {name} inter={inter} B={rows} lane={lane}"),
                        &fused_got[lane * inter..(lane + 1) * inter],
                        &wants[lane],
                        any_fail,
                    );
                }

                // ── split2 at the gate_up width ──
                let fused: Vec<bf16> = (0..rows * 2 * inter)
                    .map(|i| bf16::from_f32(i as f32 * 0.0001))
                    .collect();
                let fused_d = ctx.stream.clone_htod(&fused)?;
                let mut first_d = ctx.stream.alloc_zeros::<bf16>(rows * inter)?;
                let mut second_d = ctx.stream.alloc_zeros::<bf16>(rows * inter)?;
                split2(
                    ctx,
                    &fused_d,
                    &mut first_d,
                    &mut second_d,
                    rows,
                    inter,
                    inter,
                )?;
                ctx.sync()?;
                let first_got = ctx.stream.clone_dtoh(&first_d)?;
                let second_got = ctx.stream.clone_dtoh(&second_d)?;
                check_bitexact(
                    &format!("split2-first {name} d={inter} B={rows}"),
                    &first_got,
                    &fused[..rows * inter],
                    any_fail,
                );
                let mut second_want = Vec::with_capacity(rows * inter);
                for lane in 0..rows {
                    second_want.extend_from_slice(
                        &fused[lane * 2 * inter + inter..(lane + 1) * 2 * inter],
                    );
                }
                check_bitexact(
                    &format!("split2-second {name} d={inter} B={rows}"),
                    &second_got,
                    &second_want,
                    any_fail,
                );

                // ── split_qkv at 27B attention geometry (27B model only) ──
                if name == "qwen36-27b" {
                    let qkv_dim = Q_DIM + 2 * KV_DIM;
                    let qkv: Vec<bf16> = (0..rows * qkv_dim)
                        .map(|i| bf16::from_f32(i as f32 * 0.000_01))
                        .collect();
                    let qkv_d = ctx.stream.clone_htod(&qkv)?;
                    let mut q_d2 = ctx.stream.alloc_zeros::<bf16>(rows * Q_DIM)?;
                    let mut k_d2 = ctx.stream.alloc_zeros::<bf16>(rows * KV_DIM)?;
                    let mut v_d2 = ctx.stream.alloc_zeros::<bf16>(rows * KV_DIM)?;
                    split_qkv(
                        ctx, &qkv_d, &mut q_d2, &mut k_d2, &mut v_d2, rows, Q_DIM, KV_DIM,
                    )?;
                    ctx.sync()?;
                    let q_got = ctx.stream.clone_dtoh(&q_d2)?;
                    let k_got = ctx.stream.clone_dtoh(&k_d2)?;
                    let v_got = ctx.stream.clone_dtoh(&v_d2)?;
                    let mut q_want = Vec::new();
                    let mut k_want = Vec::new();
                    let mut v_want = Vec::new();
                    for lane in 0..rows {
                        let row = &qkv[lane * qkv_dim..(lane + 1) * qkv_dim];
                        q_want.extend_from_slice(&row[..Q_DIM]);
                        k_want.extend_from_slice(&row[Q_DIM..Q_DIM + KV_DIM]);
                        v_want.extend_from_slice(&row[Q_DIM + KV_DIM..]);
                    }
                    check_bitexact(
                        &format!("split_qkv-q {name} B={rows}"),
                        &q_got,
                        &q_want,
                        any_fail,
                    );
                    check_bitexact(
                        &format!("split_qkv-k {name} B={rows}"),
                        &k_got,
                        &k_want,
                        any_fail,
                    );
                    check_bitexact(
                        &format!("split_qkv-v {name} B={rows}"),
                        &v_got,
                        &v_want,
                        any_fail,
                    );
                }

                // ── embedding gather at this model's hidden ──
                let table: Vec<bf16> = (0..EMBED_ROWS * hidden)
                    .map(|i| bf16::from_f32((i % 101) as f32 * 0.01))
                    .collect();
                let ids: Vec<i32> = (0..rows)
                    .map(|lane| ((lane * 37 + 3) % EMBED_ROWS) as i32)
                    .collect();
                let mut expect = Vec::with_capacity(rows * hidden);
                for &id in &ids {
                    let id = id as usize;
                    expect.extend_from_slice(&table[id * hidden..(id + 1) * hidden]);
                }
                if negative {
                    expect[0] = bf16::from_f32(f32::from(expect[0]) + 0.1);
                }
                let table_d = ctx.stream.clone_htod(&table)?;
                let ids_d = ctx.stream.clone_htod(&ids)?;
                let mut emb_d = ctx.stream.alloc_zeros::<bf16>(rows * hidden)?;
                embedding_batched(ctx, &table_d, &ids_d, &mut emb_d, hidden, rows)?;
                ctx.sync()?;
                let emb_got = ctx.stream.clone_dtoh(&emb_d)?;
                check_bitexact(
                    &format!("embedding {name} h={hidden} B={rows}"),
                    &emb_got,
                    &expect,
                    any_fail,
                );
                let _ = hidden;
            }
        }
        Ok(())
    }

    pub(super) fn run(negative: bool) -> Result<()> {
        let ctx = DeviceContext::new()?;
        eprintln!(
            "[elementwise-parity] device={} build={}{}",
            ctx.ordinal(),
            cuda_kernels::KERNEL_BUILD_ID,
            if negative { " NEGATIVE-CONTROL" } else { "" }
        );

        let mut any_fail = false;
        probe_rms(&ctx, negative, &mut any_fail)?;
        probe_silu_split_embed(&ctx, negative, &mut any_fail)?;

        if negative {
            ensure!(
                any_fail,
                "elementwise_parity negative control did NOT fail — gate has no teeth"
            );
            eprintln!("[elementwise-parity] NEGATIVE CONTROL OK (gate failed as required)");
            return Ok(());
        }
        ensure!(
            !any_fail,
            "elementwise_parity FAILED — see violations above"
        );
        eprintln!("[elementwise-parity] ALL PASS");
        Ok(())
    }
}
