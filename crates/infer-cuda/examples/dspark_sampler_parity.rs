//! DSpark spec-decode sampler numeric-parity gate.
//!
//! Standalone kernel harness — NO engine, NO serve. Drives the three DSpark
//! sampler kernels against independently written f64 host references:
//!   - `dspark_filter_probs_cuda`: full-vocab temperature softmax → top-k /
//!     top-p / min-p threshold filter → renormalize, degenerate row one-hot
//!   - `dspark_draft_sample_cuda`: same filter into the persistent q row plus a
//!     CDF multinomial draw at a host-supplied uniform
//!   - `dspark_chain_accept_cuda`: flashinfer/SGLang chain rejection with
//!     renormalized max(0,p-q) residual, p fallback on ~0 mass, bonus draw
//!
//! Prob vectors are compared at small vocab 255..16384 AND the production
//! vocabularies Qwen 151936 / DSv4 129280 (representative filter cases); one
//! chain rejection runs at 151936 to cover the multi-pass residual-mass
//! reduction. Synthetic chain coverage: depth 1..=4 at vocab 32 with
//! rejection at every position, all-accept + bonus, and the zero-residual p
//! fallback; B=1/8. Exact draw ids only hold at vocab <= 256 (index-order
//! CDF); above that they assert positive reference mass.
//!
//! `--negative-control` perturbs one expected probability; the gate MUST FAIL.
//!
//! Run on a pod: `INFER_CUDA_DEVICE=<free-gpu> target/release/examples/dspark_sampler_parity`

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
        eprintln!("dspark_sampler_parity is a CUDA harness; rebuild with --features cuda.");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use anyhow::{Result, ensure};
    use cuda_kernels::prelude::DeviceContext;
    use cuda_kernels::sampling::{
        DsparkFilter, dspark_chain_accept, dspark_draft_sample, dspark_filter_probs,
    };
    use half::bf16;
    // Small-vocab sweep: every filter case, incl. multi-pass reductions.
    const FILTER_VOCABS: &[usize] = &[255, 257, 2048, 16_384];
    // Production vocabularies (prob-vector compare): Qwen 151936 and the
    // DeepSeek-V3/V4 tokenizer family 129280. Only the representative cases
    // run at this size; the residual-mass chain reduction is also exercised at
    // 151936 below (single-pass at vocab=32 would not cover it).
    const PROD_VOCABS: &[(usize, &str)] = &[(151_936, "qwen"), (129_280, "dsv4")];
    const PROD_CASES: &[usize] = &[0, 5, 6]; // unfiltered, combined, all-neg-inf
    // Synthetic chain tests use a small alphabet with hand-placed mass.
    const CHAIN_VOCAB: usize = 32;
    // One chain rejection at the production vocab: the residual mass and CDF
    // reductions are multi-pass above SAMPLE_BLOCK.
    const CHAIN_PROD_VOCAB: usize = 151_936;
    const FILTER_BATCHES: &[usize] = &[1, 8];
    const MAX_DEPTH: usize = 4;
    const SEED: u64 = 0xd54a_1200_d54a_1200;

    const PROB_REL_L2_MAX: f64 = 2e-2;
    const PROB_ABS_FLOOR: f32 = 2e-4;

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

    // ── f64 reference: the CUDA filter algorithm, computed independently ──

    fn f64_filter(row: &[f32], inv_t: f64, top_k: i64, top_p: f64, min_p: f64) -> Vec<f64> {
        let v: Vec<f64> = row.iter().map(|&x| x as f64).collect();
        let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = v.iter().map(|&x| (x - m) * inv_t).map(f64::exp).collect();
        let s: f64 = e.iter().sum();
        let mut p: Vec<f64> = if s > 0.0 {
            e.iter().map(|&x| x / s).collect()
        } else {
            e.clone()
        };

        // Kernel threshold binary search converges to the exact kth-largest
        // probability; boundary ties are over-kept (p >= threshold).
        if top_k > 0 && (top_k as usize) < p.len() {
            let mut sorted = p.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let thresh = sorted[top_k as usize - 1];
            for x in &mut p {
                if *x < thresh {
                    *x = 0.0;
                }
            }
        }
        // Smallest threshold whose above-threshold mass reaches top_p: the
        // smallest sorted prefix with cumulative mass >= top_p, ties over-kept.
        if top_p < 1.0 {
            let mut order: Vec<usize> = (0..p.len()).collect();
            order.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
            let mut cum = 0.0;
            let mut thresh = 0.0;
            for &i in &order {
                cum += p[i];
                if cum >= top_p {
                    thresh = p[i];
                    break;
                }
            }
            for x in &mut p {
                if *x < thresh {
                    *x = 0.0;
                }
            }
        }
        if min_p > 0.0 {
            let pmax = p.iter().copied().fold(0.0, f64::max);
            let thresh = min_p * pmax;
            for x in &mut p {
                if *x < thresh {
                    *x = 0.0;
                }
            }
        }

        let total: f64 = p.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            // Degenerate: one-hot at the row argmax (strict-> contract).
            let mut best = 0usize;
            let mut bv = f64::NEG_INFINITY;
            for (i, &x) in v.iter().enumerate() {
                if x > bv {
                    bv = x;
                    best = i;
                }
            }
            p.fill(0.0);
            p[best] = 1.0;
        } else {
            for x in &mut p {
                *x /= total;
            }
        }
        p
    }

    /// Exact CDF draw of the kernel at vocab <= SAMPLE_BLOCK: each thread owns
    /// one index, so the permutation is index order; the FP-tail fallback is a
    /// plain reverse-index scan to the last positive-weight token.
    fn cdf_draw_small(p: &[f32], q: Option<&[f32]>, target: f32) -> usize {
        let w = |i: usize| match q {
            Some(q) => (p[i] - q[i]).max(0.0),
            None => p[i],
        };
        let mut acc = 0f32;
        for i in 0..p.len() {
            acc += w(i);
            if target < acc {
                return i;
            }
        }
        for i in (0..p.len()).rev() {
            if w(i) > 0.0 {
                return i;
            }
        }
        0
    }

    fn rel_l2(got: &[f32], want: &[f64]) -> f64 {
        let (mut num, mut den) = (0f64, 0f64);
        for (g, r) in got.iter().zip(want) {
            let d = *g as f64 - *r;
            num += d * d;
            den += r * r;
        }
        (num / den.max(1e-12)).sqrt()
    }

    fn check_probs(label: &str, got: &[f32], want: &[f64], any_fail: &mut bool) {
        let rel = rel_l2(got, want);
        let mut max_excess = f32::NEG_INFINITY;
        for (g, r) in got.iter().zip(want) {
            let tol = PROB_ABS_FLOOR.max((*r as f32) * 0.02);
            max_excess = max_excess.max((*g - *r as f32).abs() - tol);
        }
        let pass = rel.is_finite() && rel <= PROB_REL_L2_MAX && max_excess <= 0.0;
        if !pass {
            *any_fail = true;
        }
        eprintln!(
            "[{label}] relL2={rel:.4e} maxExcess={max_excess:.4e} sum={:.6} {}",
            got.iter().sum::<f32>(),
            if pass { "PASS" } else { "FAIL" }
        );
    }

    fn check_eq(label: &str, got: i32, want: usize, any_fail: &mut bool) {
        let pass = got as usize == want;
        if !pass {
            *any_fail = true;
        }
        eprintln!(
            "[{label}] got={got} want={want} {}",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    fn filter_cases() -> Vec<(&'static str, DsparkFilter)> {
        vec![
            (
                "unfiltered",
                DsparkFilter {
                    inv_temperature: 1.0,
                    top_k: 0,
                    top_p: 1.0,
                    min_p: 0.0,
                },
            ),
            (
                "hot-t0.7",
                DsparkFilter {
                    inv_temperature: 1.0 / 0.7,
                    top_k: 0,
                    top_p: 1.0,
                    min_p: 0.0,
                },
            ),
            (
                "topk8",
                DsparkFilter {
                    inv_temperature: 1.0,
                    top_k: 8,
                    top_p: 1.0,
                    min_p: 0.0,
                },
            ),
            (
                "topp0.9",
                DsparkFilter {
                    inv_temperature: 1.0,
                    top_k: 0,
                    top_p: 0.9,
                    min_p: 0.0,
                },
            ),
            (
                "minp0.05",
                DsparkFilter {
                    inv_temperature: 1.0,
                    top_k: 0,
                    top_p: 1.0,
                    min_p: 0.05,
                },
            ),
            (
                "combined",
                DsparkFilter {
                    inv_temperature: 1.0 / 0.8,
                    top_k: 16,
                    top_p: 0.95,
                    min_p: 0.02,
                },
            ),
            (
                "all-neg-inf",
                DsparkFilter {
                    inv_temperature: 1.0,
                    top_k: 4,
                    top_p: 0.9,
                    min_p: 0.05,
                },
            ),
        ]
    }

    fn build_filter_row(rng: &mut Rng, vocab: usize, kind: &str) -> Vec<f32> {
        match kind {
            "all-neg-inf" => vec![f32::NEG_INFINITY; vocab],
            _ => (0..vocab).map(|_| rng.normal() * 1.5).collect(),
        }
    }

    fn run_filter_and_draft(
        ctx: &DeviceContext,
        negative: bool,
        any_fail: &mut bool,
    ) -> Result<()> {
        // Small sweep: every filter case at B=1/8. Production vocabs run the
        // representative cases at B=1 (~1.2 MB logits + probs per row).
        let mut jobs: Vec<(usize, usize, Vec<usize>)> = Vec::new();
        for &vocab in FILTER_VOCABS {
            for &rows in FILTER_BATCHES {
                jobs.push((vocab, rows, (0..filter_cases().len()).collect()));
            }
        }
        for &(vocab, _) in PROD_VOCABS {
            jobs.push((vocab, 1, PROD_CASES.to_vec()));
        }
        for (vocab, rows, case_indices) in jobs {
            // The batch kernel launches with ONE DsparkFilter for every row.
            for case_idx in case_indices {
                let (name, f) = filter_cases()[case_idx];
                let mut rng = Rng::new(
                    SEED ^ ((vocab as u64) << 24) ^ ((rows as u64) << 12) ^ case_idx as u64,
                );
                let mut logits = Vec::with_capacity(rows * vocab);
                let mut refs: Vec<Vec<f64>> = Vec::with_capacity(rows);
                let mut uniforms: Vec<f32> = Vec::with_capacity(rows);
                let mut first_positive = vec![0usize; rows];
                for (lane, first_positive) in first_positive.iter_mut().enumerate() {
                    let row = build_filter_row(&mut rng, vocab, name);
                    let mut pref = f64_filter(
                        &row,
                        f.inv_temperature as f64,
                        f.top_k as i64,
                        f.top_p as f64,
                        f.min_p as f64,
                    );
                    if negative && lane == 0 {
                        pref[0] += 0.1;
                    }
                    // Uniform at the middle of the first positive token's mass
                    // interval, off the reference: no f32 CDF boundary.
                    let w0 = pref.iter().position(|&x| x > 0.0).unwrap_or(0);
                    let lo: f64 = pref[..w0].iter().sum();
                    uniforms.push(((lo + lo + pref[w0]) * 0.5).clamp(1e-6, 0.999_999) as f32);
                    *first_positive = w0;
                    refs.push(pref);
                    logits.extend(row.iter().map(|v| bf16::from_f32(*v)));
                }

                // ── filter_probs batch ──
                let logits_d = ctx.stream.clone_htod(&logits)?;
                let mut probs_d = ctx.stream.alloc_zeros::<f32>(rows * vocab)?;
                dspark_filter_probs(ctx, &logits_d, &mut probs_d, rows, vocab, f)?;
                ctx.sync()?;
                let probs = ctx.stream.clone_dtoh(&probs_d)?;
                for lane in 0..rows {
                    check_probs(
                        &format!("filter vocab={vocab} B={rows} {name} lane={lane}"),
                        &probs[lane * vocab..(lane + 1) * vocab],
                        &refs[lane],
                        any_fail,
                    );
                }

                // ── draft_sample per row: the q row equals the filter output,
                // token id is the CDF draw at the supplied uniform ──
                for lane in 0..rows {
                    let row_bf: Vec<bf16> = logits[lane * vocab..(lane + 1) * vocab].to_vec();
                    let row_d = ctx.stream.clone_htod(&row_bf)?;
                    let mut q_d = ctx.stream.alloc_zeros::<f32>(vocab)?;
                    let mut tok_d = ctx.stream.alloc_zeros::<i32>(1)?;
                    dspark_draft_sample(
                        ctx,
                        &row_d,
                        &mut q_d,
                        &mut tok_d,
                        vocab,
                        f,
                        uniforms[lane],
                    )?;
                    ctx.sync()?;
                    let q = ctx.stream.clone_dtoh(&q_d)?;
                    let tok = ctx.stream.clone_dtoh(&tok_d)?;
                    check_probs(
                        &format!("draft-q vocab={vocab} B={rows} {name} lane={lane}"),
                        &q,
                        &refs[lane],
                        any_fail,
                    );
                    // w0 IS the degenerate-row argmax for all-neg-inf (the
                    // one-hot's first index).
                    if vocab <= 256 {
                        check_eq(
                            &format!("draft-tok vocab={vocab} B={rows} {name} lane={lane}"),
                            tok[0],
                            first_positive[lane],
                            any_fail,
                        );
                    } else {
                        // Above SAMPLE_BLOCK the kernel CDF walks indices in a
                        // thread-strided permutation; assert the drawn id carries
                        // positive reference mass.
                        let id = tok[0] as usize;
                        let pass = id < vocab && refs[lane][id] > 1e-6;
                        if !pass {
                            *any_fail = true;
                        }
                        eprintln!(
                            "[draft-tok vocab={vocab} B={rows} {name} lane={lane}] id={id} refmass={:.4e} {}",
                            refs[lane].get(id).copied().unwrap_or(-1.0),
                            if pass { "PASS" } else { "FAIL" }
                        );
                    }
                }
            }
        }
        Ok(())
    }

    // ── chain rejection reference ──

    /// Returns (accepted_len, token) for one chain.
    fn chain_reference(
        q: &[Vec<f32>],
        p: &[Vec<f32>],
        draft: &[i32],
        u_accept: &[f32],
        u_res: &[f32],
        depth: usize,
    ) -> (usize, usize) {
        for j in 0..depth {
            let tok = draft[j] as usize;
            let accept = (p[j][tok] / q[j][tok].max(1e-8)).min(1.0);
            if u_accept[j] < accept {
                continue;
            }
            let mass: f32 = p[j]
                .iter()
                .zip(&q[j])
                .map(|(pi, qi)| (pi - qi).max(0.0))
                .sum();
            if mass <= 1e-8 {
                return (j, cdf_draw_small(&p[j], None, u_res[j]));
            }
            return (j, cdf_draw_small(&p[j], Some(&q[j]), u_res[j] * mass));
        }
        (depth, cdf_draw_small(&p[depth], None, u_res[depth]))
    }

    /// A normalized two-spike distribution: mass `ma` at token `a`, the rest
    /// spread uniformly over other indices (deterministic, avoids zero rows).
    fn two_spike(vocab: usize, a: usize, ma: f32, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        let rest: Vec<f32> = (0..vocab - 1).map(|_| 0.5 + rng.unit()).collect();
        let rs: f32 = rest.iter().sum();
        let mut p = vec![0f32; vocab];
        p[a] = ma;
        for (i, x) in p.iter_mut().enumerate() {
            if i != a {
                *x = (1.0 - ma) * rest[if i < a { i } else { i - 1 }] / rs;
            }
        }
        p
    }

    fn run_chain(ctx: &DeviceContext, negative: bool, any_fail: &mut bool) -> Result<()> {
        for &batch in FILTER_BATCHES {
            for depth in 1..=MAX_DEPTH {
                // Scenario per lane: reject at position r in 0..depth, plus an
                // all-accept scenario; cycled to `batch` launches.
                for scenario in 0..=depth {
                    for lane_base in (0..batch).step_by(1) {
                        let rej = (scenario + lane_base) % (depth + 1);
                        let all_accept = rej == depth; // last slot = full accept
                        let v = CHAIN_VOCAB;
                        let draft_tok = 3i32;
                        let mut q = vec![vec![0f32; v]; depth];
                        let mut p = vec![vec![0f32; v]; depth + 1];
                        let mut u_accept = vec![0f32; depth];
                        let mut u_res = vec![0f32; depth + 1];

                        for j in 0..depth {
                            if all_accept || j < rej {
                                // p dominates q at the draft token → accept=1.
                                q[j] = two_spike(v, draft_tok as usize, 0.5, SEED + j as u64);
                                p[j] = two_spike(v, draft_tok as usize, 0.8, SEED + 100 + j as u64);
                                u_accept[j] = 0.0;
                            } else if j == rej {
                                if scenario % 2 == 0 {
                                    // Positive residual: p weak at draft, q
                                    // strong; one clean residual spike at 9.
                                    q[j] = two_spike(v, draft_tok as usize, 0.5, SEED + 7);
                                    p[j] = two_spike(v, 9, 0.8, SEED + 9);
                                    p[j][draft_tok as usize] = 0.2;
                                    let s: f32 = p[j].iter().sum();
                                    for x in &mut p[j] {
                                        *x /= s;
                                    }
                                    u_accept[j] = 0.99; // accept=0.4 → reject
                                    // Target the middle of token 9's residual
                                    // spike: q[9] is background only (< p[9]).
                                    let q9 = q[j][9];
                                    let mass: f32 = p[j]
                                        .iter()
                                        .zip(&q[j])
                                        .map(|(pi, qi)| (pi - qi).max(0.0))
                                        .sum();
                                    // Place target at half of token 9's own
                                    // residual interval start: aim at token 9.
                                    let before: f32 =
                                        (0..9).map(|i| (p[j][i] - q[j][i]).max(0.0)).sum();
                                    u_res[j] =
                                        ((before + (p[j][9] - q9) * 0.5) / mass).clamp(1e-4, 0.999);
                                } else {
                                    // Zero residual: q = 2*p dominates p
                                    // everywhere (an overconfident draft; q is
                                    // not renormalized in this defensive
                                    // branch). p[3] weak → accept=0.5, reject,
                                    // and max(p-q,0) mass is exactly 0 → p
                                    // fallback draw.
                                    p[j] =
                                        two_spike(v, draft_tok as usize, 0.2, SEED + 31 + j as u64);
                                    q[j] = p[j].iter().map(|x| 2.0 * x).collect();
                                    u_accept[j] = 0.99;
                                    // Middle of token 5's mass interval in p.
                                    let before: f32 = p[j][..5].iter().sum();
                                    u_res[j] = (before + p[j][5] * 0.5).clamp(1e-4, 0.999);
                                }
                            } else {
                                q[j] = two_spike(v, draft_tok as usize, 0.5, SEED + j as u64);
                                p[j] = q[j].clone();
                                u_accept[j] = 0.0;
                            }
                        }
                        // Bonus row: spike at token 11, target its middle.
                        p[depth] = two_spike(v, 11, 0.8, SEED + 51);
                        let lo: f32 = p[depth][..11].iter().sum();
                        u_res[depth] = (lo + p[depth][11] * 0.5).clamp(1e-4, 0.999);

                        let (want_len, want_tok) = chain_reference(
                            &q,
                            &p,
                            &vec![draft_tok; depth],
                            &u_accept,
                            &u_res,
                            depth,
                        );
                        let mut want_tok = want_tok;
                        if negative {
                            want_tok = if want_tok == 0 { 1 } else { 0 };
                        }

                        let q_flat: Vec<f32> = q.concat();
                        let p_flat: Vec<f32> = p.concat();
                        let draft_v = vec![draft_tok; depth];
                        let q_d = ctx.stream.clone_htod(&q_flat)?;
                        let p_d = ctx.stream.clone_htod(&p_flat)?;
                        let draft_d = ctx.stream.clone_htod(&draft_v)?;
                        let ua_d = ctx.stream.clone_htod(&u_accept)?;
                        let ur_d = ctx.stream.clone_htod(&u_res)?;
                        let mut out_d = ctx.stream.alloc_zeros::<i32>(2)?;
                        dspark_chain_accept(
                            ctx, &q_d, &p_d, &draft_d, &ua_d, &ur_d, &mut out_d, depth, v,
                        )?;
                        ctx.sync()?;
                        let out = ctx.stream.clone_dtoh(&out_d)?;

                        let label =
                            format!("chain depth={depth} B={batch} sc={scenario} lane={lane_base}");
                        check_eq(&format!("{label} accepted_len"), out[0], want_len, any_fail);
                        check_eq(&format!("{label} token"), out[1], want_tok, any_fail);
                    }
                }
            }
        }
        Ok(())
    }

    /// One rejection at the production vocab. The accept decision and accepted
    /// length are exact; the residual draw walks the kernel's >256 thread-strided
    /// CDF, so the token is asserted to carry positive residual mass. This is
    /// the multi-pass path for both the residual-mass reduction and the draw:
    /// the vocab=32 cases above reduce both in a single pass.
    fn run_chain_prod(ctx: &DeviceContext, any_fail: &mut bool) -> Result<()> {
        let v = CHAIN_PROD_VOCAB;
        let depth = 1;
        let draft_tok = 3i32;
        let mut q = vec![vec![0f32; v]; depth];
        let mut p = vec![vec![0f32; v]; depth + 1];
        let u_accept = vec![0.99f32; depth];
        let mut u_res = vec![0.5f32; depth + 1];

        // q confident at the draft token; p puts a bigger spike at token 20003,
        // weak mass at the draft so accept = p/q < 1 and a reject fires.
        q[0] = two_spike(v, draft_tok as usize, 0.5, SEED ^ 0x5052_0001);
        p[0] = two_spike(v, 20_003, 0.8, SEED ^ 0x5052_0002);
        p[0][draft_tok as usize] = 0.2;
        let s: f32 = p[0].iter().sum();
        for x in &mut p[0] {
            *x /= s;
        }
        p[depth] = two_spike(v, 40_007, 0.8, SEED ^ 0x5052_0003);
        u_res[depth] = 0.5;

        let residual_mass: f32 = p[0]
            .iter()
            .zip(&q[0])
            .map(|(pi, qi)| (pi - qi).max(0.0))
            .sum();
        ensure!(
            residual_mass > 1e-3,
            "production chain case needs a real residual mass, got {residual_mass}"
        );

        let q_d = ctx.stream.clone_htod(&q.concat())?;
        let p_d = ctx.stream.clone_htod(&p.concat())?;
        let draft_d = ctx.stream.clone_htod(&vec![draft_tok; depth])?;
        let ua_d = ctx.stream.clone_htod(&u_accept)?;
        let ur_d = ctx.stream.clone_htod(&u_res)?;
        let mut out_d = ctx.stream.alloc_zeros::<i32>(2)?;
        dspark_chain_accept(
            ctx, &q_d, &p_d, &draft_d, &ua_d, &ur_d, &mut out_d, depth, v,
        )?;
        ctx.sync()?;
        let out = ctx.stream.clone_dtoh(&out_d)?;

        let len_pass = out[0] == 0;
        if !len_pass {
            *any_fail = true;
        }
        eprintln!(
            "[chain-prod vocab={v}] accepted_len={} (want 0) residual_mass={residual_mass:.4e} {}",
            out[0],
            if len_pass { "PASS" } else { "FAIL" }
        );
        let id = out[1] as usize;
        let res_at =
            (p[0].get(id).copied().unwrap_or(0.0) - q[0].get(id).copied().unwrap_or(0.0)).max(0.0);
        let tok_pass = id < v && res_at > 1e-9;
        if !tok_pass {
            *any_fail = true;
        }
        eprintln!(
            "[chain-prod vocab={v}] residual token={id} residual_mass_at={res_at:.4e} {}",
            if tok_pass { "PASS" } else { "FAIL" }
        );
        Ok(())
    }

    pub(super) fn run(negative: bool) -> Result<()> {
        let ctx = DeviceContext::new()?;
        eprintln!(
            "[dspark-parity] device={} build={}{}",
            ctx.ordinal(),
            cuda_kernels::KERNEL_BUILD_ID,
            if negative { " NEGATIVE-CONTROL" } else { "" }
        );

        let mut any_fail = false;
        run_filter_and_draft(&ctx, negative, &mut any_fail)?;
        run_chain(&ctx, negative, &mut any_fail)?;
        run_chain_prod(&ctx, &mut any_fail)?;

        if negative {
            ensure!(
                any_fail,
                "dspark_sampler_parity negative control did NOT fail — gate has no teeth"
            );
            eprintln!("[dspark-parity] NEGATIVE CONTROL OK (gate failed as required)");
            return Ok(());
        }
        ensure!(
            !any_fail,
            "dspark_sampler_parity FAILED — see violations above"
        );
        eprintln!("[dspark-parity] ALL PASS");
        Ok(())
    }
}
