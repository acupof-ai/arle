//! DeepGEMM FP8 m-grouped prefill MoE numeric-parity gate.
//!
//! Standalone kernel harness — NO engine, NO serve. Gates the two SM90 FP8
//! grouped GEMM entry points used by the DSv4 MoE layer:
//!
//! - `dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked_cuda` — per-expert bands
//!   `[num_groups, m_cap, k]`, valid row count read from `masked_m`.
//! - `dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous_cuda` — one packed
//!   `[m, k]` activation with a per-row `m_indices` expert map (128-aligned
//!   segments, -1 pad rows).
//!
//! Both production projections are run on both paths:
//! - down:    `act[m,2048] @ w2[g,4096,2048] -> [m,4096]`  (n=4096, k=2048)
//! - gate/up: `hid[m,4096] @ w13[g,4096,4096] -> [m,4096]` (n=4096, k=4096)
//!
//! Geometry is the DSv4-Flash production config
//! (/data00/DeepSeek-V4-Flash-0731/config.json): hidden 4096, moe_intermediate
//! 2048, 256 routed experts, top-6. Scales are per-128×128-block f32 on both
//! operands (the kernel multiplies `a_q·b_q·sfa·sfb`); weights are f32 block
//! scales, not ue8m0 (moe/dsv4.rs:587).
//!
//! Two group-count regimes:
//! - 16-band sweep over flat / one-hot / bimodal token distributions, each with
//!   empty experts and a hot expert.
//! - a full 256-expert distribution sized to production routing:
//!   512 tokens × top-6 = 3072 routed rows, one hot 128-row expert, the rest
//!   spread over ~64 experts so ~191 bands are empty (masked_m=0 / absent
//!   segment). Every active expert gets its OWN random weights; empty bands use
//!   a zero buffer. This exercises group indices above 15 through the tile
//!   scheduler and m_indices ranges.
//!
//! Operands are generated directly as e4m3 bytes with real-magnitude per-block
//! f32 scales (random 0.03..0.15, the post-abs-max/448 range) rather than the
//! perf-probe scale=1.0. The f64 oracle checks EVERY output column
//! (`Σ_bk sfa·sfb·Σ a_q·b_q`) — a wrong N-tile store offset, swizzle, or
//! in-block column mapping leaves column 0 correct and corrupts the rest, so
//! per-block column sampling would miss it. Rows are sampled instead: all rows
//! of the hot band, first+last and a 64-row seeded sample of every other
//! active band (column bugs show per row, row-index bugs are per-row). The
//! oracle is parallelized over rows (std::thread::scope). The bound is
//! derived from the e4m3 bin error accumulated over k/128 blocks, not tuned.
//!
//! `--negative-control` perturbs each path; the run prints NEGATIVE CONTROL OK
//! and exits 0.
//!
//! Run on an SM90 pod:
//!   cargo build --release -p infer-cuda --features cuda --example deepgemm_grouped_prefill_parity
//!   target/release/examples/deepgemm_grouped_prefill_parity --kernel-build-id
//!   INFER_CUDA_DEVICE=<free-sm90-gpu> target/release/examples/deepgemm_grouped_prefill_parity
//!   INFER_CUDA_DEVICE=<free-sm90-gpu> target/release/examples/deepgemm_grouped_prefill_parity --negative-control

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
        eprintln!(
            "deepgemm_grouped_prefill_parity is a CUDA/SM90 harness; rebuild with --features cuda."
        );
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use anyhow::Result;
    use cuda_kernels::moe::{
        dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous, dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked,
    };
    use cuda_kernels::prelude::DeviceContext;
    use cuda_kernels::tensor::RawDevicePtr;
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

    use super::parity_common::{Families, Rng};

    const SEED: u64 = 0xdee9_6e44_2024_0911;

    const HIDDEN: usize = 4096;
    const INTER: usize = 2048;
    const EXPERTS: usize = 256;
    const TOP_K: usize = 6;
    const BLOCK: usize = 128; // fp8 scale block (m, n, k)

    const SWEEP_EXPERTS: usize = 16;
    const SWEEP_TOKENS: usize = 512;
    const FULL_TOKENS: usize = 512 * TOP_K; // 3072 routed rows
    const M_CAP: usize = 128; // per-group band capacity (one output tile)

    /// (n, k) for the two projections, both run on both paths.
    const GEOMS: &[(usize, usize, &str)] = &[
        (HIDDEN, INTER, "down n=4096 k=2048"),
        (HIDDEN, HIDDEN, "gate/up n=4096 k=4096"),
    ];

    // e4m3: 3 mantissa bits ⇒ per-element mid-bin rel error < 2^-4 ≈ 6.25%.
    // Two fp8 operands + one bf16 store; k/128 scale blocks accumulate. Bounds
    // pre-set from the quantization, never tuned to a run.
    const OUT_REL_L2_MAX: f64 = 8e-2;
    const OUT_ABS_SLOPE: f32 = 1.0e-1;
    const OUT_ABS_FLOOR: f32 = 2e-2;

    fn uniform(rng: &mut Rng, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * rng.unit()
    }

    /// Random e4m3 byte that is never a NaN code.
    fn fp8(rng: &mut Rng) -> u8 {
        (rng.next_u64() as u8) & 0x7e
    }

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

    /// Number of randomly sampled extra rows per active band (deterministic
    /// seed) in addition to the mandatory first and last row.
    const SAMPLE_ROWS: usize = 64;

    /// Select checked rows for one band. Column INDEXING/tiling bugs corrupt
    /// arbitrary columns of every row, so they show in any row; ROW-indexing
    /// bugs are per-row, so the hot band is checked in full and every other
    /// active band gets first + last plus a seeded sample. Sorted, unique.
    fn checked_rows(valid: usize, is_hot: bool, band_seed: u64) -> Vec<usize> {
        if valid == 0 {
            return Vec::new();
        }
        if is_hot {
            return (0..valid).collect();
        }
        let mut pick: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        pick.insert(0);
        pick.insert(valid - 1);
        let mut rng = Rng::new(band_seed);
        for _ in 0..SAMPLE_ROWS {
            pick.insert((rng.next_u64() as usize) % valid);
        }
        pick.into_iter().collect()
    }

    /// One expert band: fp8 bytes + per-block f32 scales. The f64 reference is
    /// computed lazily for the selected rows only.
    struct Band {
        a_q: Vec<u8>,      // [M_CAP, k]
        a_scale: Vec<f32>, // [k/BLOCK] (m_cap == one m-block)
        b_q: Vec<u8>,      // [n, k]; empty when zero_weights
        b_scale: Vec<f32>, // [n/BLOCK, k/BLOCK]
        valid: usize,
        zero_weights: bool,
    }

    fn build_band(rng: &mut Rng, valid: usize, k: usize, n: usize, zero_weights: bool) -> Band {
        let kb = k / BLOCK;
        let a_q: Vec<u8> = (0..M_CAP * k).map(|_| fp8(rng)).collect();
        let a_scale: Vec<f32> = (0..kb).map(|_| uniform(rng, 0.03, 0.15)).collect();
        if zero_weights {
            return Band {
                a_q,
                a_scale,
                b_q: Vec::new(),
                b_scale: Vec::new(),
                valid,
                zero_weights: true,
            };
        }
        let nb = n / BLOCK;
        let b_q: Vec<u8> = (0..n * k).map(|_| fp8(rng)).collect();
        let b_scale: Vec<f32> = (0..nb * kb).map(|_| uniform(rng, 0.03, 0.15)).collect();
        Band {
            a_q,
            a_scale,
            b_q,
            b_scale,
            valid,
            zero_weights: false,
        }
    }

    /// f64 reference for the selected rows of one band over EVERY output
    /// column: out[row,col] = Σ_bk sfa·sfb·Σ a_q·b_q. Parallelized across
    /// rows (std::thread::scope); the B matrix is decoded to f64 once per
    /// band so workers do plain f64 reads.
    fn band_ref(band: &Band, rows: &[usize], n: usize, k: usize) -> Vec<f64> {
        let kb = k / BLOCK;
        let b_f: Vec<f64> = band
            .b_q
            .iter()
            .map(|&q| f64::from(e4m3_decode(q)))
            .collect();
        let a_f: Vec<f64> = band
            .a_q
            .iter()
            .map(|&q| f64::from(e4m3_decode(q)))
            .collect();
        let threads = std::thread::available_parallelism()
            .map(|s| s.get())
            .unwrap_or(1);
        let chunk = rows.len().div_ceil(threads).max(1);
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for chunk_rows in rows.chunks(chunk) {
                let b_f = &b_f;
                let a_f = &a_f;
                let a_scale = &band.a_scale;
                let b_scale = &band.b_scale;
                handles.push(scope.spawn(move || {
                    let mut local_out = vec![0f64; chunk_rows.len() * n];
                    for (local, &m) in chunk_rows.iter().enumerate() {
                        for col in 0..n {
                            let bn = col / BLOCK;
                            let mut acc = 0f64;
                            for bk in 0..kb {
                                let sa = a_scale[bk] as f64;
                                let sb = b_scale[bn * kb + bk] as f64;
                                let k0 = bk * BLOCK;
                                let mut dot = 0f64;
                                for kk in k0..k0 + BLOCK {
                                    dot += a_f[m * k + kk] * b_f[col * k + kk];
                                }
                                acc += sa * sb * dot;
                            }
                            local_out[local * n + col] = acc;
                        }
                    }
                    local_out
                }));
            }
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect::<Vec<f64>>()
        })
    }

    // Token-count distributions: counts[band] (0..=M_CAP).

    /// 16-band sweep distributions; each includes empty bands and a hot band.
    fn sweep_distributions() -> Vec<(&'static str, Vec<usize>)> {
        let g = SWEEP_EXPERTS;
        let mut out = Vec::new();

        // Flat across the first half, second half empty.
        let mut flat = vec![0usize; g];
        let active = g / 2;
        let mut left = SWEEP_TOKENS;
        for c in flat.iter_mut().take(active) {
            let take = left.div_ceil(active).min(left).min(M_CAP);
            *c = take;
            left -= take;
        }
        if left > 0 {
            flat[0] += left;
        }
        out.push(("flat", flat));

        // One hot band, remainder sparse, rest empty.
        let mut hot = vec![0usize; g];
        hot[0] = M_CAP;
        let mut left = SWEEP_TOKENS - M_CAP;
        let mut i = 1;
        while left > 0 {
            let take = left.min(8).min(M_CAP);
            hot[i] = take;
            left -= take;
            i += 1;
        }
        out.push(("one-hot", hot));

        // Full/singleton alternating, trailing empties.
        let mut bimodal = vec![0usize; g];
        let mut left = SWEEP_TOKENS;
        let mut i = 0;
        let mut full = true;
        while left > 0 && i < g {
            let take = if full { M_CAP.min(left) } else { 1.min(left) };
            bimodal[i] = take;
            left -= take;
            i += 1;
            full = !full;
        }
        out.push(("bimodal", bimodal));
        out
    }

    /// Full 256-expert production-routing distribution: one hot 128-row band,
    /// the other 3072-128 rows spread over the next ~64 bands, leaving ~191
    /// bands empty.
    fn full_distribution() -> Vec<usize> {
        let mut counts = vec![0usize; EXPERTS];
        counts[0] = M_CAP;
        let mut left = FULL_TOKENS - M_CAP;
        let mut g = 1;
        while left > 0 && g < EXPERTS {
            let take = left.min(48).min(M_CAP);
            counts[g] = take;
            left -= take;
            g += 1;
        }
        counts
    }

    /// Compare one band's device output at the SELECTED rows and EVERY output
    /// column against the f64 ref. `is_hot` selects all rows; other bands get
    /// first/last + a seeded sample. Computes the ref in parallel.
    #[allow(clippy::too_many_arguments)]
    fn check_band(
        label: &str,
        got: &[bf16],
        band: &Band,
        n: usize,
        k: usize,
        is_hot: bool,
        band_seed: u64,
        corrupt: bool,
    ) -> bool {
        if band.valid == 0 {
            return true; // empty band: no rows to compare
        }
        let rows = checked_rows(band.valid, is_hot, band_seed);
        let reff = band_ref(band, &rows, n, k);
        let mut diff_sq = 0f64;
        let mut ref_sq = 0f64;
        let mut violators = 0usize;
        for (ri, &m) in rows.iter().enumerate() {
            for col in 0..n {
                let mut w = reff[ri * n + col];
                if corrupt && ri == 0 && col == 0 {
                    w += 5.0 * w.abs().max(1.0);
                }
                let gv = got[m * n + col].to_f32() as f64;
                let d = (gv - w).abs();
                diff_sq += d * d;
                ref_sq += w * w;
                if d > (OUT_ABS_FLOOR + OUT_ABS_SLOPE * w.abs() as f32) as f64 {
                    violators += 1;
                }
            }
        }
        let rel_l2 = (diff_sq / ref_sq.max(1e-12)).sqrt();
        let pass = rel_l2 < OUT_REL_L2_MAX && violators == 0;
        eprintln!(
            "  [{label} valid={} checked_rows={} cols={n} rel_l2={rel_l2:.2e} violators={violators} {}]",
            band.valid,
            rows.len(),
            if pass { "PASS" } else { "FAIL" }
        );
        pass
    }

    fn run_masked(
        ctx: &DeviceContext,
        n: usize,
        k: usize,
        geom: &str,
        negative: bool,
    ) -> Result<bool> {
        let kb = k / BLOCK;
        let mut overall = true;

        let mut cases: Vec<(&str, Vec<usize>)> = sweep_distributions().into_iter().collect();
        cases.push(("full-256", full_distribution()));

        for (dist_name, counts) in cases {
            let ng = counts.len();
            // Allocate full [ng,...] bands; empty bands' weights are zero
            // bytes. Every band is present in the kernel call so masked_m=0 is
            // exercised across the whole group-id range (incl ids >15).
            let mut rng = Rng::new(SEED ^ ((n as u64) << 20) ^ ((k as u64) << 8));
            let bands: Vec<Band> = counts
                .iter()
                .map(|&c| build_band(&mut rng, c, n, k, c == 0))
                .collect();

            let mut a_all = vec![0u8; ng * M_CAP * k];
            let mut sa_all = vec![0f32; ng * kb];
            let mut b_all = vec![0u8; ng * n * k];
            let mut sb_all = vec![0f32; ng * (n / BLOCK) * kb];
            let mut masked = vec![0i32; ng];
            for (i, p) in bands.iter().enumerate() {
                a_all[i * M_CAP * k..(i + 1) * M_CAP * k].copy_from_slice(&p.a_q);
                sa_all[i * kb..(i + 1) * kb].copy_from_slice(&p.a_scale);
                masked[i] = p.valid as i32;
                if !p.zero_weights {
                    b_all[i * n * k..(i + 1) * n * k].copy_from_slice(&p.b_q);
                    sb_all[i * (n / BLOCK) * kb..(i + 1) * (n / BLOCK) * kb]
                        .copy_from_slice(&p.b_scale);
                }
            }

            let a_d = ctx.stream.clone_htod(&a_all)?;
            let sa_d = ctx.stream.clone_htod(&sa_all)?;
            let b_d = ctx.stream.clone_htod(&b_all)?;
            let sb_d = ctx.stream.clone_htod(&sb_all)?;
            let masked_d = ctx.stream.clone_htod(&masked)?;
            let mut d_d = ctx.stream.alloc_zeros::<bf16>(ng * M_CAP * n)?;
            let (ap, _) = a_d.device_ptr(&ctx.stream);
            let (sap, _) = sa_d.device_ptr(&ctx.stream);
            let (bp, _) = b_d.device_ptr(&ctx.stream);
            let (sbp, _) = sb_d.device_ptr(&ctx.stream);
            let (mp, _) = masked_d.device_ptr(&ctx.stream);
            let (dp, _) = d_d.device_ptr_mut(&ctx.stream);
            // SAFETY: all slices are live buffers sized to the [ng,...] bands.
            unsafe {
                dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked(
                    RawDevicePtr::from_raw(ap),
                    RawDevicePtr::from_raw(sap),
                    RawDevicePtr::from_raw(bp),
                    RawDevicePtr::from_raw(sbp),
                    RawDevicePtr::from_raw(dp),
                    RawDevicePtr::from_raw(mp),
                    ng,
                    M_CAP,
                    n,
                    k,
                    M_CAP, // sfa_aligned_m == m_cap (one m-block per band)
                    ctx.stream.cu_stream(),
                )?;
            }
            ctx.sync()?;
            let got = ctx.stream.clone_dtoh(&d_d)?;

            let mut dist_ok = true;
            for (i, p) in bands.iter().enumerate() {
                let band = &got[i * M_CAP * n..(i + 1) * M_CAP * n];
                // Band 0 is the hot band (M_CAP rows) in every distribution.
                let is_hot = i == 0 && p.valid == M_CAP;
                let corrupt = negative && i == 0 && p.valid > 0;
                let band_seed =
                    SEED.wrapping_add((n as u64) << 24) ^ ((k as u64) << 16) ^ (i as u64);
                dist_ok &= check_band(
                    &format!("{dist_name} band{i}"),
                    band,
                    p,
                    n,
                    k,
                    is_hot,
                    band_seed,
                    corrupt,
                );
            }
            eprintln!(
                "[masked {geom} dist={dist_name} groups={ng}] {}",
                if dist_ok { "PASS" } else { "FAIL" }
            );
            overall &= dist_ok;
        }
        Ok(overall)
    }

    fn run_contiguous(
        ctx: &DeviceContext,
        n: usize,
        k: usize,
        geom: &str,
        negative: bool,
    ) -> Result<bool> {
        let kb = k / BLOCK;
        let mk_align = 128usize;
        let mut overall = true;

        let mut cases: Vec<(&str, Vec<usize>)> = sweep_distributions().into_iter().collect();
        cases.push(("full-256", full_distribution()));

        for (dist_name, counts) in cases {
            let ng = counts.len();
            let mut rng = Rng::new((SEED ^ 0x00c0_1716) ^ ((n as u64) << 20) ^ ((k as u64) << 8));
            let bands: Vec<Band> = counts
                .iter()
                .map(|&c| build_band(&mut rng, c, n, k, c == 0))
                .collect();

            // 128-aligned segment per non-empty band; total m is their sum
            // (multiple of 128). -1 on pad rows, which the kernel may compute
            // against group 0 — those outputs are excluded.
            let seg: Vec<usize> = counts
                .iter()
                .map(|&c| {
                    if c == 0 {
                        0
                    } else {
                        c.div_ceil(mk_align) * mk_align
                    }
                })
                .collect();
            let m: usize = seg.iter().sum();

            let mut m_indices = vec![-1i32; m];
            let mut cursor = 0;
            for (g, &c) in counts.iter().enumerate() {
                for r in 0..c {
                    m_indices[cursor + r] = g as i32;
                }
                cursor += seg[g];
            }

            let mut a_q = vec![0u8; m * k];
            let mut sa = vec![0f32; (m / BLOCK) * kb];
            let mut b_all = vec![0u8; ng * n * k];
            let mut sb_all = vec![0f32; ng * (n / BLOCK) * kb];
            let mut seg_start = vec![0usize; ng];
            {
                let mut acc = 0;
                for g in 0..ng {
                    seg_start[g] = acc;
                    acc += seg[g];
                }
            }
            for (g, p) in bands.iter().enumerate() {
                if !p.zero_weights {
                    b_all[g * n * k..(g + 1) * n * k].copy_from_slice(&p.b_q);
                    sb_all[g * (n / BLOCK) * kb..(g + 1) * (n / BLOCK) * kb]
                        .copy_from_slice(&p.b_scale);
                }
                let start = seg_start[g];
                for r in 0..p.valid {
                    a_q[(start + r) * k..(start + r + 1) * k]
                        .copy_from_slice(&p.a_q[r * k..(r + 1) * k]);
                }
                // One m-block per segment here (counts <= 128); copy the band's
                // single a_scale row into that tile.
                if seg[g] > 0 {
                    sa[(start / BLOCK) * kb..(start / BLOCK + 1) * kb].copy_from_slice(&p.a_scale);
                }
            }

            let a_d = ctx.stream.clone_htod(&a_q)?;
            let sa_d = ctx.stream.clone_htod(&sa)?;
            let b_d = ctx.stream.clone_htod(&b_all)?;
            let sb_d = ctx.stream.clone_htod(&sb_all)?;
            let idx_d = ctx.stream.clone_htod(&m_indices)?;
            let mut d_d = ctx.stream.alloc_zeros::<bf16>(m * n)?;
            let (ap, _) = a_d.device_ptr(&ctx.stream);
            let (sap, _) = sa_d.device_ptr(&ctx.stream);
            let (bp, _) = b_d.device_ptr(&ctx.stream);
            let (sbp, _) = sb_d.device_ptr(&ctx.stream);
            let (ip, _) = idx_d.device_ptr(&ctx.stream);
            let (dp, _) = d_d.device_ptr_mut(&ctx.stream);
            // SAFETY: all slices live; m is 128-aligned, m_indices holds one
            // band id per valid row and -1 on pad rows.
            unsafe {
                dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous(
                    RawDevicePtr::from_raw(ap),
                    RawDevicePtr::from_raw(sap),
                    RawDevicePtr::from_raw(bp),
                    RawDevicePtr::from_raw(sbp),
                    RawDevicePtr::from_raw(dp),
                    RawDevicePtr::from_raw(ip),
                    ng,
                    m,
                    n,
                    k,
                    m, // sfa_aligned_m
                    mk_align,
                    ctx.stream.cu_stream(),
                )?;
            }
            ctx.sync()?;
            let got = ctx.stream.clone_dtoh(&d_d)?;

            let mut diff_sq = 0f64;
            let mut ref_sq = 0f64;
            let mut violators = 0usize;
            let mut n_checked_rows = 0usize;
            for (g, p) in bands.iter().enumerate() {
                if p.valid == 0 {
                    continue;
                }
                let start = seg_start[g];
                let is_hot = g == 0 && p.valid == M_CAP;
                let band_seed = SEED.wrapping_add(0x1000).wrapping_add((n as u64) << 24)
                    ^ ((k as u64) << 16)
                    ^ (g as u64);
                let rows = checked_rows(p.valid, is_hot, band_seed);
                n_checked_rows += rows.len();
                let reff = band_ref(p, &rows, n, k); // every column, parallel
                for (ri, &r) in rows.iter().enumerate() {
                    for col in 0..n {
                        let mut w = reff[ri * n + col];
                        if negative && g == 0 && ri == 0 && col == 0 {
                            w += 5.0 * w.abs().max(1.0);
                        }
                        let gv = got[(start + r) * n + col].to_f32() as f64;
                        let d = (gv - w).abs();
                        diff_sq += d * d;
                        ref_sq += w * w;
                        if d > (OUT_ABS_FLOOR + OUT_ABS_SLOPE * w.abs() as f32) as f64 {
                            violators += 1;
                        }
                    }
                }
            }
            let rel_l2 = (diff_sq / ref_sq.max(1e-12)).sqrt();
            let pass = n_checked_rows > 0 && rel_l2 < OUT_REL_L2_MAX && violators == 0;
            eprintln!(
                "[contiguous {geom} dist={dist_name} m={m} groups={ng}] rows={n_checked_rows} \
                 cols={n} rel_l2={rel_l2:.2e} violators={violators} {}",
                if pass { "PASS" } else { "FAIL" }
            );
            overall &= pass;
        }
        Ok(overall)
    }

    pub(super) fn run(negative: bool) -> Result<()> {
        let ctx = DeviceContext::new()?;
        eprintln!(
            "[deepgemm-grouped-parity] device={} build={}{}",
            ctx.ordinal(),
            cuda_kernels::KERNEL_BUILD_ID,
            if negative { " NEGATIVE-CONTROL" } else { "" }
        );

        let mut masked_fail = false;
        let mut contiguous_fail = false;
        for &(n, k, geom) in GEOMS {
            masked_fail |= !run_masked(&ctx, n, k, geom, negative)?;
            contiguous_fail |= !run_contiguous(&ctx, n, k, geom, negative)?;
        }

        let mut families = Families::new();
        families.record("masked path", masked_fail);
        families.record("contiguous path", contiguous_fail);
        families.finish("deepgemm-grouped-parity", negative)
    }
}
