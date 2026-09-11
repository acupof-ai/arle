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
//! q=6144/kv=1024, 24×256-attn / 4 KV heads) and DeepSeek-V4-Flash-0731
//! (hidden 4096, moe inter 2048, per the model's config.json), batch 1 and 8;
//! RMSNorm eps = 1e-6 (the model config value).
//! Embedding additionally runs at the production 151936-row × 5120 table
//! (~1.5 GB) with ids at 0, the last rows and both ends.
//!
//! `--negative-control` perturbs one reference element IN EACH comparator
//! family (batched/single RMSNorm, separate/fused SiLU, split2 halves,
//! split_qkv q/k/v, embedding, production embedding); every family MUST
//! independently FAIL.
//!
//! Run on a pod: `INFER_CUDA_DEVICE=<free-gpu> target/release/examples/elementwise_parity`

fn main() -> anyhow::Result<()> {
    use parity_common::Parsed;
    match parity_common::cli() {
        Parsed::BuildIdPrinted => Ok(()),
        Parsed::Run(cli) => real::run(cli.negative),
    }
}

#[allow(dead_code)] // shared harness; each gate uses only the subset it needs
#[path = "support/parity_common.rs"]
mod parity_common;

#[cfg(not(feature = "cuda"))]
mod real {
    pub(super) fn run(_negative: bool) -> anyhow::Result<()> {
        eprintln!("elementwise_parity is a CUDA harness; rebuild with --features cuda.");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use anyhow::Result;
    use cuda_kernels::prelude::DeviceContext;
    use cuda_kernels::tensor_ops::{
        embedding_batched, rms_norm, rms_norm_batched, silu_mul, silu_mul_fused, split_qkv, split2,
    };
    use half::bf16;

    use super::parity_common::{Rng, compare as common_compare};

    const EPS: f64 = 1e-6;
    const BATCHES: &[usize] = &[1, 8];
    const SEED: u64 = 0xe1e4_1515_e1e4_1515;
    // (label, hidden, inter) — Qwen3.6-27B dense dims and DeepSeek-V4-Flash-0731
    // (hidden_size 4096 / moe_intermediate_size 2048; rms_norm_eps 1e-6,
    // vocab 129280 — read from the model's config.json).
    const MODELS: &[(&str, usize, usize)] = &[("qwen36-27b", 5120, 17_408), ("dsv4", 4_096, 2_048)];
    // 27B attention geometry: q 24×256=6144, kv 4×256=1024.
    const Q_DIM: usize = 6144;
    const KV_DIM: usize = 1024;
    // Small synthetic table for the per-model hidden-width gather.
    const EMBED_ROWS: usize = 256;

    // BF16 outputs off f64 anchors: truncation floor ~4e-3 rel.
    const REL_L2_MAX: f64 = 2e-2;
    const ABS_FLOOR: f32 = 3e-3;
    const ABS_SLOPE: f32 = 3e-2;

    /// One fail bit per comparator family, so --negative-control can prove
    /// each family independently has teeth.
    #[derive(Clone, Copy)]
    struct Families {
        rms_batched: bool,
        rms_single: bool,
        silu_separate: bool,
        silu_fused: bool,
        split2_first: bool,
        split2_second: bool,
        splitqkv_q: bool,
        splitqkv_k: bool,
        splitqkv_v: bool,
        embedding: bool,
        embedding_prod: bool,
    }
    impl Families {
        fn entries(self) -> [(&'static str, bool); 11] {
            [
                ("rms_norm batched", self.rms_batched),
                ("rms_norm single", self.rms_single),
                ("silu_mul separate", self.silu_separate),
                ("silu_mul fused", self.silu_fused),
                ("split2 first half", self.split2_first),
                ("split2 second half", self.split2_second),
                ("split_qkv q", self.splitqkv_q),
                ("split_qkv k", self.splitqkv_k),
                ("split_qkv v", self.splitqkv_v),
                ("embedding", self.embedding),
                ("embedding production", self.embedding_prod),
            ]
        }
    }

    fn check(label: &str, got: &[bf16], want: &[f64], failed: &mut bool) {
        let gv: Vec<f64> = got.iter().map(|g| f32::from(*g) as f64).collect();
        let v = common_compare(&gv, want, ABS_FLOOR as f64, ABS_SLOPE as f64);
        let pass = v.is_ok(REL_L2_MAX);
        if !pass {
            *failed = true;
        }
        eprintln!("[{label}] {} {}", v.worst_line(), if pass { "PASS" } else { "FAIL" });
    }

    fn check_bitexact(label: &str, got: &[bf16], want: &[bf16], failed: &mut bool) {
        let mut bad = 0usize;
        for (a, b) in got.iter().zip(want) {
            if a.to_bits() != b.to_bits() {
                bad += 1;
            }
        }
        let pass = bad == 0 && got.len() == want.len();
        if !pass {
            *failed = true;
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

    fn probe_rms(ctx: &DeviceContext, negative: bool, fams: &mut Families) -> Result<()> {
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
                        &mut fams.rms_batched,
                    );
                }

                if rows == 1 {
                    // Singular kernel: BF16 round between scale and weight.
                    let mut one_d = ctx.stream.alloc_zeros::<bf16>(hidden)?;
                    rms_norm(ctx, &x_d, &w_d, &mut one_d, hidden, EPS as f32)?;
                    ctx.sync()?;
                    let got = ctx.stream.clone_dtoh(&one_d)?;
                    let mut want = rms_ref(&x[..hidden], &w, true);
                    if negative {
                        want[0] += 0.1;
                    }
                    check(
                        &format!("rms-single {name} h={hidden}"),
                        &got,
                        &want,
                        &mut fams.rms_single,
                    );
                }
            }
        }
        Ok(())
    }

    fn probe_silu_split_embed(
        ctx: &DeviceContext,
        negative: bool,
        fams: &mut Families,
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
                    &mut fams.silu_separate,
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
                        &mut fams.silu_fused,
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
                let mut first_want: Vec<bf16> = fused[..rows * inter].to_vec();
                let mut second_want = Vec::with_capacity(rows * inter);
                for lane in 0..rows {
                    second_want.extend_from_slice(
                        &fused[lane * 2 * inter + inter..(lane + 1) * 2 * inter],
                    );
                }
                if negative {
                    // Bit-exact family: flip the expected bits at one element.
                    first_want[0] = bf16::from_f32(f32::from(first_want[0]) + 0.5);
                    second_want[0] = bf16::from_f32(f32::from(second_want[0]) + 0.5);
                }
                check_bitexact(
                    &format!("split2-first {name} d={inter} B={rows}"),
                    &first_got,
                    &first_want,
                    &mut fams.split2_first,
                );
                check_bitexact(
                    &format!("split2-second {name} d={inter} B={rows}"),
                    &second_got,
                    &second_want,
                    &mut fams.split2_second,
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
                    if negative {
                        for w in [&mut q_want, &mut k_want, &mut v_want] {
                            w[0] = bf16::from_f32(f32::from(w[0]) + 0.5);
                        }
                    }
                    check_bitexact(
                        &format!("split_qkv-q {name} B={rows}"),
                        &q_got,
                        &q_want,
                        &mut fams.splitqkv_q,
                    );
                    check_bitexact(
                        &format!("split_qkv-k {name} B={rows}"),
                        &k_got,
                        &k_want,
                        &mut fams.splitqkv_k,
                    );
                    check_bitexact(
                        &format!("split_qkv-v {name} B={rows}"),
                        &v_got,
                        &v_want,
                        &mut fams.splitqkv_v,
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
                    &mut fams.embedding,
                );
                let _ = hidden;
            }
        }
        Ok(())
    }

    /// Production-shape embedding gather: Qwen 151936-row table × 5120 bf16
    /// (~1.5 GB). Ids cover 0, the last rows, and rows near both ends — the
    /// table offset is the arithmetic that can overflow. One B=8 launch spans
    /// both ends. The table is a periodic pattern over the flat index, so every
    /// row differs and a wrong base cannot mask.
    fn probe_embedding_production(
        ctx: &DeviceContext,
        negative: bool,
        fams: &mut Families,
    ) -> Result<()> {
        const ROWS: usize = 151_936;
        let hidden: usize = MODELS[0].1;
        let ids: Vec<i32> = vec![
            0,
            1,
            2,
            3,
            (ROWS - 4) as i32,
            (ROWS - 3) as i32,
            (ROWS - 2) as i32,
            (ROWS - 1) as i32,
        ];
        let b = ids.len();
        let table: Vec<bf16> = (0..ROWS * hidden)
            .map(|i| bf16::from_f32((i % 101) as f32 * 0.01))
            .collect();
        let mut expect: Vec<bf16> = Vec::with_capacity(b * hidden);
        for &id in &ids {
            let base = id as usize * hidden;
            expect.extend((base..base + hidden).map(|i| table[i]));
        }
        if negative {
            expect[0] = bf16::from_f32(f32::from(expect[0]) + 0.1);
        }
        let table_d = ctx.stream.clone_htod(&table)?;
        let ids_d = ctx.stream.clone_htod(&ids)?;
        let mut out_d = ctx.stream.alloc_zeros::<bf16>(b * hidden)?;
        embedding_batched(ctx, &table_d, &ids_d, &mut out_d, hidden, b)?;
        ctx.sync()?;
        check_bitexact(
            &format!("embedding-prod rows={ROWS} h={hidden} B={b}"),
            &ctx.stream.clone_dtoh(&out_d)?,
            &expect,
            &mut fams.embedding_prod,
        );
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

        let mut fams = Families {
            rms_batched: false,
            rms_single: false,
            silu_separate: false,
            silu_fused: false,
            split2_first: false,
            split2_second: false,
            splitqkv_q: false,
            splitqkv_k: false,
            splitqkv_v: false,
            embedding: false,
            embedding_prod: false,
        };
        probe_rms(&ctx, negative, &mut fams)?;
        probe_silu_split_embed(&ctx, negative, &mut fams)?;
        probe_embedding_production(&ctx, negative, &mut fams)?;

        let mut families = super::parity_common::Families::new();
        for (name, fired) in fams.entries() {
            families.record(name, fired);
        }
        families.finish("elementwise-parity", negative)
    }
}
