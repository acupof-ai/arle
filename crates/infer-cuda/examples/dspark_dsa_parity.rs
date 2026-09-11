//! DSpark draft attention + DSA indexer numeric-parity gate.
//!
//! Standalone kernel harness — NO engine, NO serve. Two production DSv4 spec
//! surfaces that previously had shape guards and counters only:
//!
//! 1. `dsv4_dspark_draft_attention_cuda` — non-causal dense MLA-latent
//!    attention over one head-shared compressed latent, full-head_dim dot
//!    products, inverse-RoPE on the value tail. Geometry from the DSv4-Flash
//!    production config (DeepSeek-V4-Flash-0731): `dspark_block_size = 5`,
//!    latent head_dim 512 (448 nope + 64 rope), 64 local heads at TP=1 (draft
//!    wq_b rows 32768 / 512), kv_len = sliding_window(128)+block(5)=133 up to a
//!    4096 stress case (kernel shared-mem cap DSV4_DSPARK_MAX_KEYS=9216).
//! 2. The DSA official indexer family:
//!    - `dsv4_dsa_fused_q_indexer_rope_hadamard_quant_cuda` (partial RoPE +
//!      H128 orthonormal rotate + per-row FP8 e4m3 quant and scale; production
//!      index_n_heads=64, index_head_dim=128),
//!    - `dsv4_deepseek_v4_topk_transform_cuda` (radix top-k with paged index
//!      mapping; production index_topk=512 over seq up to 4096).
//!
//! The oracles are written from the DSv4 model spec / kernel contract in f64
//! (attention) and f32/f64 (indexer), independently of the CUDA code. Top-k
//! indices are compared as SETS with the exact boundary-tie rule: every
//! strictly-larger element must be present; ties at the cutoff value are
//! interchangeable (the kernel orders bit-equal scores by atomic arrival).
//! Attention output is compared within a bound derived from the bf16/f32
//! accumulation path, not a byte identity.
//!
//! `--negative-control` corrupts one expectation per family; the gate MUST
//! then FAIL (proves it has teeth).
//!
//! Run on a pod:
//!   INFER_CUDA_DEVICE=<free-gpu> target/release/examples/dspark_dsa_parity
//!   INFER_CUDA_DEVICE=<free-gpu> target/release/examples/dspark_dsa_parity --negative-control

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
        eprintln!("dspark_dsa_parity is a CUDA harness; rebuild with --features cuda.");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use anyhow::Result;
    use cuda_kernels::attention::{
        dsv4_deepseek_v4_topk_transform_raw, dsv4_dsa_fused_q_indexer_rope_hadamard_quant_raw,
        dsv4_dspark_draft_attention_raw,
    };
    use cuda_kernels::prelude::DeviceContext;
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

    use super::parity_common::{Families, Rng};

    const SEED: u64 = 0xd5a1_7d5a_c01d_51a1;

    // DSpark draft-attention geometries from the DSv4-Flash production
    // config (/data00/DeepSeek-V4-Flash-0731/config.json): head_dim=512,
    // qk_rope_head_dim=64 (nope 448), dspark_block_size=5, sliding_window=128.
    // The committed per-stage latent is the window plus the block (kv_len=133);
    // a 4096-row case stresses the kernel up to its 9216 shared-mem cap.
    // local_heads = draft wq_b.rows / 512 = 32768/512 = 64 at TP=1 (DSpark
    // draft mtp.0/1/2.attn wq_b [32768,1024]); TP>1 shards heads 64/tp, which
    // this TP=1 gate does not cover.
    const ATTN_HEAD_DIM: usize = 512;
    const ATTN_ROPE_DIM: usize = 64;
    const ATTN_LOCAL_HEADS: usize = 64;
    const ATTN_BLOCK: usize = 5;
    const ATTN_KV_LENS: &[usize] = &[133, 1024, 4096];
    const ROPE_BASE: f32 = 10_000.0;
    const BASE_START_POS: i32 = 1280;

    // Kernel accumulates head_dim*kv_len f32 products then rounds to bf16 once.
    // The final bf16 store dominates; bound is element-wise slope+floor.
    const ATTN_REL_L2_MAX: f64 = 4e-2;
    const ATTN_ABS_FLOOR: f32 = 2e-2;
    const ATTN_ABS_SLOPE: f32 = 5e-2;

    // DSA indexer geometry (DSv4 fixture: 64 MQA heads, 128 index dim, topk 512).
    // seq > topk takes the radix path; seq == topk takes the naive paged branch.
    const INDEX_HEADS: usize = 64;
    const INDEX_HEAD_DIM: usize = 128;
    const INDEX_TOPK: usize = 512;
    const INDEX_SEQS: &[usize] = &[512, 2049, 4096];
    const INDEX_BATCH: usize = 4;
    const PAGE_SIZE: usize = 64;

    // FP8 e4m3 RN has 3 mantissa bits (<= ~3.1% mid-bin rel error; an f32/f64
    // disagreement on a bin boundary costs one bin, ~6.25%). A wrong-Hadamard
    // bug produces order-unity deviations, so the slope stays discriminating.
    const FP8_ABS_SLOPE: f32 = 8e-2;
    const FP8_ABS_FLOOR: f32 = 3e-2;
    // Scale = row abs-max/448 over bf16-rounded inputs.
    const SCALE_REL_MAX: f32 = 1e-2;

    fn bf(v: f32) -> bf16 {
        bf16::from_f32(v)
    }

    // ── FP8 e4m3 (round-to-nearest-even), matching __nv_fp8_e4m3 ────────────

    fn round_even(x: f64) -> f64 {
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
    fn e4m3_encode(x: f32) -> u8 {
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
        let e2 = v.log2().floor() as i32; // binade exponent
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
            let sig = round_even(v / 2f64.powi(-9)).clamp(0.0, 7.0) as u8;
            sign | sig
        }
    }

    /// Decode an e4m3 byte to f32.
    fn e4m3_decode(b: u8) -> f32 {
        let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
        let field = (b >> 3) & 0x0f;
        let mant = f32::from(b & 0x07);
        if field == 0 {
            sign * mant * 2f32.powi(-9)
        } else {
            sign * (8.0 + mant) * 2f32.powi(field as i32 - 10)
        }
    }

    // ── Attention oracle ────────────────────────────────────────────────────

    /// Plain (non-YaRN, original_seq_len=0) inverse RoPE for one (a,b) pair,
    /// matching dsv4_apply_rope_pair(sign=-1): o_a = a·c + b·sin,
    /// o_b = b·c − a·sin.
    fn inverse_rope_pair(a: f64, b: f64, pair: usize, pos: i32, rope_dim: usize) -> (f64, f64) {
        let inv = (ROPE_BASE as f64).powf(-(2.0 * pair as f64 / rope_dim as f64));
        let angle = pos as f64 * inv;
        let (c, s) = (angle.cos(), angle.sin());
        (a * c + b * s, b * c - a * s)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_ref(
        q: &[bf16],
        latent: &[bf16],
        kv_len: usize,
        block: usize,
        heads: usize,
        head_dim: usize,
        rope_dim: usize,
        base_pos: i32,
    ) -> Vec<f32> {
        let sm_scale = 1.0 / (head_dim as f64).sqrt();
        let rope_start = head_dim - rope_dim;
        let mut out = vec![0f32; block * heads * head_dim];
        for tok in 0..block {
            for h in 0..heads {
                let qb = &q[(tok * heads + h) * head_dim..][..head_dim];
                let mut scores = vec![0f64; kv_len];
                for j in 0..kv_len {
                    let kb = &latent[j * head_dim..][..head_dim];
                    let dot: f64 = qb
                        .iter()
                        .zip(kb)
                        .map(|(x, y)| f64::from(x.to_f32()) * f64::from(y.to_f32()))
                        .sum();
                    scores[j] = dot * sm_scale;
                }
                let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let denom: f64 = exps.iter().sum();

                // Weighted value over the FULL head_dim first.
                let mut v = vec![0f64; head_dim];
                for col in 0..head_dim {
                    v[col] = (0..kv_len)
                        .map(|j| (exps[j] / denom) * f64::from(latent[j * head_dim + col].to_f32()))
                        .sum();
                }
                // Inverse-RoPE the value tail pairwise at the query position.
                let ob = &mut out[(tok * heads + h) * head_dim..][..head_dim];
                for col in 0..head_dim {
                    ob[col] = if rope_dim > 0 && col >= rope_start {
                        let pair = (col - rope_start) / 2;
                        let (a, b) = (v[rope_start + pair * 2], v[rope_start + pair * 2 + 1]);
                        let (oa, ob2) =
                            inverse_rope_pair(a, b, pair, base_pos + tok as i32, rope_dim);
                        if (col - rope_start).is_multiple_of(2) {
                            oa
                        } else {
                            ob2
                        }
                    } else {
                        v[col]
                    } as f32;
                }
            }
        }
        out
    }

    fn check_attention_geom(ctx: &DeviceContext, kv_len: usize, negative: bool) -> Result<bool> {
        let head_dim = ATTN_HEAD_DIM;
        let rope_dim = ATTN_ROPE_DIM;
        let heads = ATTN_LOCAL_HEADS;
        let block = ATTN_BLOCK;
        let mut rng = Rng::new(
            SEED ^ ((head_dim as u64) << 32) ^ ((kv_len as u64) << 12) ^ ((heads as u64) << 4),
        );
        let q: Vec<bf16> = (0..block * heads * head_dim)
            .map(|_| bf(rng.normal() * 0.3))
            .collect();
        let latent: Vec<bf16> = (0..kv_len * head_dim)
            .map(|_| bf(rng.normal() * 0.3))
            .collect();

        let q_d = ctx.stream.clone_htod(&q)?;
        let latent_d = ctx.stream.clone_htod(&latent)?;
        let mut out_d = ctx.stream.alloc_zeros::<bf16>(block * heads * head_dim)?;

        let sm_scale = 1.0 / (head_dim as f32).sqrt();
        let (q_ptr, _) = q_d.device_ptr(&ctx.stream);
        let (l_ptr, _) = latent_d.device_ptr(&ctx.stream);
        let (o_ptr, _) = out_d.device_ptr_mut(&ctx.stream);
        dsv4_dspark_draft_attention_raw(
            &ctx.stream,
            q_ptr,
            l_ptr,
            o_ptr,
            kv_len as i32,
            block as i32,
            heads as i32,
            head_dim as i32,
            (head_dim - rope_dim) as i32,
            rope_dim as i32,
            BASE_START_POS,
            sm_scale,
            ROPE_BASE,
            0,
            1.0,
            0.0,
            0.0,
        )?;
        ctx.sync()?;
        let got = ctx.stream.clone_dtoh(&out_d)?;
        let mut want = attention_ref(
            &q,
            &latent,
            kv_len,
            block,
            heads,
            head_dim,
            rope_dim,
            BASE_START_POS,
        );
        if negative {
            want[0] += 1.0;
        }

        let mut diff_sq = 0f64;
        let mut ref_sq = 0f64;
        let mut violators = 0usize;
        for (g, w) in got.iter().zip(want.iter()) {
            let d = (g.to_f32() - w).abs();
            diff_sq += (d as f64).powi(2);
            ref_sq += (*w as f64).powi(2);
            if d > ATTN_ABS_FLOOR + ATTN_ABS_SLOPE * w.abs() {
                violators += 1;
            }
        }
        let rel_l2 = (diff_sq / ref_sq.max(1e-12)).sqrt();
        let pass = rel_l2 < ATTN_REL_L2_MAX && violators == 0;
        eprintln!(
            "[attn hd={head_dim} rope={rope_dim} h={heads} block={block} kv={kv_len}] rel_l2={rel_l2:.2e} \
             violators={violators} {}",
            if pass { "PASS" } else { "FAIL" }
        );
        Ok(pass)
    }

    // ── Indexer oracle ──────────────────────────────────────────────────────

    /// Build the freqs_cis table the kernel consumes: per position 64 floats,
    /// 16 groups of (cos,sin,cos,sin) for RoPE pairs 2g / 2g+1 over the 64
    /// rope dims.
    fn build_freqs(positions: &[i32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(positions.len() * 64);
        for &pos in positions {
            for g in 0..16usize {
                for pair in 0..2usize {
                    let p = 2 * g + pair;
                    let inv = (ROPE_BASE as f64).powf(-(2.0 * p as f64 / 64.0));
                    let angle = pos as f64 * inv;
                    out.push(angle.cos() as f32);
                    out.push(angle.sin() as f32);
                }
            }
        }
        out
    }

    /// Oracle fused Q indexer: partial RoPE on elements 64..128, H128
    /// orthonormal rotate, per-row FP8 e4m3 quant with scale. Returns (fp8
    /// bytes, per-row scale).
    fn fused_q_indexer_ref(
        q: &[bf16],
        freqs: &[f32],
        positions: &[i32],
        num_heads: usize,
    ) -> (Vec<u8>, Vec<f32>) {
        let rows = q.len() / 128;
        let mut rotated = vec![0f64; rows * 128];
        for r in 0..rows {
            let batch = r / num_heads;
            let fbase = (positions[batch] as usize) * 64;
            let mut x = [0f64; 128];
            for (i, v) in q[r * 128..(r + 1) * 128].iter().enumerate() {
                x[i] = f64::from(v.to_f32());
            }
            for g in 0..16 {
                let i = 64 + g * 4;
                let (xr, xi) = (x[i], x[i + 1]);
                let (yr, yi) = (x[i + 2], x[i + 3]);
                let c0 = f64::from(freqs[fbase + g * 4]);
                let s0 = f64::from(freqs[fbase + g * 4 + 1]);
                let c1 = f64::from(freqs[fbase + g * 4 + 2]);
                let s1 = f64::from(freqs[fbase + g * 4 + 3]);
                x[i] = xr * c0 - xi * s0;
                x[i + 1] = xr * s0 + xi * c0;
                x[i + 2] = yr * c1 - yi * s1;
                x[i + 3] = yr * s1 + yi * c1;
            }
            // H128 Walsh–Hadamard: 7 radix-2 stages (strides 1..64). The kernel
            // implements the same transform as H4 within each 4-lane float4
            // group plus H32 across the 32 lanes, scaled by 1/sqrt(128).
            let mut h = x;
            let mut stride = 1usize;
            while stride < 128 {
                for base in (0..128).step_by(stride * 2) {
                    for k in 0..stride {
                        let i0 = base + k;
                        let i1 = i0 + stride;
                        let (a, b) = (h[i0], h[i1]);
                        h[i0] = a + b;
                        h[i1] = a - b;
                    }
                }
                stride *= 2;
            }
            let norm = (128f64).sqrt();
            for (i, v) in h.iter().enumerate() {
                rotated[r * 128 + i] = v / norm;
            }
        }
        let mut bytes = vec![0u8; rows * 128];
        let mut scales = vec![0f32; rows];
        for r in 0..rows {
            let abs_max = rotated[r * 128..(r + 1) * 128]
                .iter()
                .fold(1e-4f64, |m, v| m.max(v.abs()));
            let scale = (abs_max / 448.0) as f32;
            scales[r] = scale;
            for i in 0..128 {
                bytes[r * 128 + i] = e4m3_encode((rotated[r * 128 + i] / scale as f64) as f32);
            }
        }
        (bytes, scales)
    }

    /// Top-k as a SET with the boundary-tie rule. Returns the mandatory
    /// strictly-greater indices and the cutoff score.
    fn topk_set(scores: &[f32], k: usize) -> (std::collections::HashSet<i32>, f32) {
        let mut idx: Vec<usize> = (0..scores.len()).collect();
        idx.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let cutoff = scores[idx[k - 1]];
        let strict = idx
            .iter()
            .filter(|&&i| scores[i] > cutoff)
            .map(|&i| i as i32)
            .collect();
        (strict, cutoff)
    }

    // Returns (fused_q_ok, topk_ok) so each band is its own gate family.
    fn check_indexer(ctx: &DeviceContext, negative: bool) -> Result<(bool, bool)> {
        // (1) Fused Q indexer: INDEX_BATCH tokens × INDEX_HEADS rows.
        let works = INDEX_BATCH * INDEX_HEADS;
        let mut rng = Rng::new(SEED ^ 0x1de5);
        let q: Vec<bf16> = (0..works * INDEX_HEAD_DIM)
            .map(|_| bf(rng.normal() * 0.4))
            .collect();
        let positions: Vec<i32> = (0..INDEX_BATCH as i32).collect();
        let freqs = build_freqs(&positions);
        // Non-unit weights and weight_scale match production's per-row multiply.
        const WEIGHT_SCALE: f32 = 1.0 / 1024.0;
        let weights: Vec<bf16> = (0..works).map(|_| bf(rng.normal() * 0.5)).collect();
        let (want_bytes, mut want_scales) =
            fused_q_indexer_ref(&q, &freqs, &positions, INDEX_HEADS);
        if negative {
            want_scales[0] *= 2.0;
        }

        let q_d = ctx.stream.clone_htod(&q)?;
        let mut q_fp8_d = ctx.stream.alloc_zeros::<u8>(works * 128)?;
        let w_d = ctx.stream.clone_htod(&weights)?;
        let mut wout_d = ctx.stream.alloc_zeros::<f32>(works)?;
        let f_d = ctx.stream.clone_htod(&freqs)?;
        let p_d = ctx.stream.clone_htod(&positions)?;
        let (qp, _) = q_d.device_ptr(&ctx.stream);
        let (fp, _) = q_fp8_d.device_ptr_mut(&ctx.stream);
        let (wp, _) = w_d.device_ptr(&ctx.stream);
        let (op, _) = wout_d.device_ptr_mut(&ctx.stream);
        let (frp, _) = f_d.device_ptr(&ctx.stream);
        let (pp, _) = p_d.device_ptr(&ctx.stream);
        dsv4_dsa_fused_q_indexer_rope_hadamard_quant_raw(
            &ctx.stream,
            qp,
            fp,
            wp,
            op,
            WEIGHT_SCALE,
            frp,
            pp,
            INDEX_BATCH as i32,
            INDEX_HEADS as i32,
        )?;
        ctx.sync()?;
        let got_bytes = ctx.stream.clone_dtoh(&q_fp8_d)?;
        let got_weighted = ctx.stream.clone_dtoh(&wout_d)?;

        // Recover the pure per-row quant scale; zero-weight rows cannot recover.
        let mut scale_worst = 0f32;
        let mut scale_rows = 0usize;
        let mut weighted_worst = 0f32;
        let mut violators = 0usize;
        for r in 0..works {
            let want_scale = want_scales[r];
            let w = weights[r].to_f32();
            let want_weighted = w * WEIGHT_SCALE * want_scale;
            weighted_worst = weighted_worst
                .max((got_weighted[r] - want_weighted).abs() / want_weighted.abs().max(1e-12));
            if w == 0.0 {
                continue;
            }
            let scale = got_weighted[r] / (w * WEIGHT_SCALE);
            scale_worst = scale_worst.max((scale - want_scale).abs() / want_scale.abs().max(1e-8));
            scale_rows += 1;
            for i in 0..128 {
                let got_v = e4m3_decode(got_bytes[r * 128 + i]) * scale;
                let want_v = e4m3_decode(want_bytes[r * 128 + i]) * want_scale;
                let d = (got_v - want_v).abs();
                if d > FP8_ABS_FLOOR * want_scale.abs() + FP8_ABS_SLOPE * want_v.abs() {
                    violators += 1;
                }
            }
        }
        let fq_pass =
            scale_worst < SCALE_REL_MAX && weighted_worst < SCALE_REL_MAX && violators == 0;
        eprintln!(
            "[indexer fused_q rows={works}] scale_rel_worst={scale_worst:.2e} weighted_rel_worst={weighted_worst:.2e} \
             violators={violators} ({} scale rows) {}",
            scale_rows,
            if fq_pass { "PASS" } else { "FAIL" }
        );
        let mut topk_pass = true;

        // (2) Radix top-k + paged transform over score rows.
        for &seq in INDEX_SEQS {
            let stride = seq + 128;
            let mut scores = vec![0f32; INDEX_BATCH * stride];
            let mut rng = Rng::new(SEED ^ (seq as u64));
            for b in 0..INDEX_BATCH {
                for i in 0..seq {
                    scores[b * stride + i] = rng.normal() * 2.0;
                }
                // Pin deterministic exact ties at one score so the cutoff-tie
                // rule is exercised.
                if b == 0 || b == 2 {
                    let tie = scores[b * stride + 7];
                    scores[b * stride + 11] = tie;
                    scores[b * stride + 100 + b] = tie;
                }
            }
            let seq_lens = vec![seq as i32; INDEX_BATCH];
            let pages = stride.div_ceil(PAGE_SIZE);
            // Deterministic non-identity page table (rotated 1..N permutation)
            // so a raw-index/slot mix-up cannot pass.
            let mut page_table = vec![0i32; INDEX_BATCH * pages];
            for b in 0..INDEX_BATCH {
                for p in 0..pages {
                    page_table[b * pages + p] = ((p + 1 + b) % pages) as i32;
                }
            }

            let scores_d = ctx.stream.clone_htod(&scores)?;
            let lens_d = ctx.stream.clone_htod(&seq_lens)?;
            let table_d = ctx.stream.clone_htod(&page_table)?;
            let mut page_idx_d = ctx.stream.alloc_zeros::<i32>(INDEX_BATCH * INDEX_TOPK)?;
            let mut raw_idx_d = ctx.stream.alloc_zeros::<i32>(INDEX_BATCH * INDEX_TOPK)?;
            let (sp, _) = scores_d.device_ptr(&ctx.stream);
            let (lp, _) = lens_d.device_ptr(&ctx.stream);
            let (tp, _) = table_d.device_ptr(&ctx.stream);
            let (pi, _) = page_idx_d.device_ptr_mut(&ctx.stream);
            let (ri, _) = raw_idx_d.device_ptr_mut(&ctx.stream);
            dsv4_deepseek_v4_topk_transform_raw(
                &ctx.stream,
                sp,
                lp,
                tp,
                pi,
                ri,
                stride as i64,
                pages as i64,
                INDEX_TOPK as i64,
                INDEX_BATCH as i32,
                INDEX_TOPK as i32,
                PAGE_SIZE as i32,
            )?;
            ctx.sync()?;
            let mut got_raw = ctx.stream.clone_dtoh(&raw_idx_d)?;
            let got_page = ctx.stream.clone_dtoh(&page_idx_d)?;

            let mut geom_pass = true;
            for b in 0..INDEX_BATCH {
                let k = INDEX_TOPK.min(seq);
                if negative && b == 0 {
                    // Replace the highest-score selection with the row's
                    // minimum-score index: outside the top-k on every shape.
                    let row = &scores[b * stride..b * stride + seq];
                    let argmin = row
                        .iter()
                        .enumerate()
                        .min_by(|a, c| a.1.partial_cmp(c.1).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap()
                        .0;
                    got_raw[b * INDEX_TOPK] = argmin as i32;
                }
                let mut got: Vec<i32> = got_raw[b * INDEX_TOPK..b * INDEX_TOPK + k].to_vec();
                if seq <= INDEX_TOPK {
                    // Naive branch: expect 0..seq in order, tail padded with -1.
                    let mut sorted: Vec<i32> = got.iter().copied().filter(|&i| i >= 0).collect();
                    sorted.sort_unstable();
                    let expect: Vec<i32> = (0..seq as i32).collect();
                    if sorted != expect {
                        geom_pass = false;
                    }
                    got = got_raw[b * INDEX_TOPK..b * INDEX_TOPK + seq].to_vec();
                } else {
                    let row = &scores[b * stride..b * stride + seq];
                    let got_set: std::collections::HashSet<i32> = got.iter().copied().collect();
                    if got_set.len() != got.len() {
                        geom_pass = false;
                    }
                    let (strict, cutoff) = topk_set(row, k);
                    for i in strict {
                        if !got_set.contains(&i) {
                            geom_pass = false;
                        }
                    }
                    for &i in &got {
                        if row[i as usize] < cutoff {
                            geom_pass = false;
                        }
                    }
                }
                // Every selected page slot must equal the page-table mapping.
                for (slot, &raw) in got.iter().enumerate() {
                    let page = got_page[b * INDEX_TOPK + slot];
                    if raw < 0 {
                        if page != -1 {
                            geom_pass = false;
                        }
                        continue;
                    }
                    let mapped = page_table[b * pages + raw as usize / PAGE_SIZE]
                        * PAGE_SIZE as i32
                        + raw % PAGE_SIZE as i32;
                    if page != mapped {
                        geom_pass = false;
                    }
                }
            }
            eprintln!(
                "[indexer topk seq={seq} k={INDEX_TOPK} B={INDEX_BATCH}] {}",
                if geom_pass { "PASS" } else { "FAIL" }
            );
            topk_pass &= geom_pass;
        }

        Ok((fq_pass, topk_pass))
    }

    pub(super) fn run(negative: bool) -> Result<()> {
        let ctx = DeviceContext::new()?;
        eprintln!(
            "[dspark-dsa-parity] device={} build={}{}",
            ctx.ordinal(),
            cuda_kernels::KERNEL_BUILD_ID,
            if negative { " NEGATIVE-CONTROL" } else { "" }
        );

        let mut attn_ok = true;
        for &kv_len in ATTN_KV_LENS {
            attn_ok &= check_attention_geom(&ctx, kv_len, negative)?;
        }
        let (fq_ok, topk_ok) = check_indexer(&ctx, negative)?;

        let mut families = Families::new();
        families.record("attention", !attn_ok);
        families.record("indexer-fused-q", !fq_ok);
        families.record("indexer-topk", !topk_ok);
        families.finish("dspark-dsa-parity", negative)
    }
}
