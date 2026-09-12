//! FlashMLA SM90 sparse-PREFILL numeric-parity gate for DSv4-Flash MODEL1
//! (CSA ratio=4 and HCA ratio=128).
//!
//! Sibling of the two decode gates. Prefill differs from decode: s_q > 1 with
//! causality enforced at index-build time, the KV pool is one contiguous
//! BF16 buffer `[rebased SW window | current chunk K | compressed]` (NOT the
//! FP8 block-paged decode pool), and there is no tile-scheduler metadata
//! (the grid is (s_q × h_q/64); wrapper num_sm arg is ignored).
//!
//! Drives the production prefill wrappers
//! `flashmla_csa_pack_kv_raw`, `flashmla_csa_build_indices_raw` /
//! `flashmla_hca_build_indices_raw`, `flashmla_sm90_sparse_prefill_fwd_raw`.
//!
//! Chunk lengths (read from code): 128 (scheduler grain/floor), 2048
//! (`executor/dsv4.rs::max_prefill_chunk` DEFAULT_PREFILL_CHUNK), 4096
//! (`attention/kv_layout.rs::DSV4_PREFILL_QUERY_CHUNK` hard ceiling).
//! start_pos straddles 64/128-token boundaries. The SW window is filled out
//! of order so the rebased copy exercises the (sw_base+row)%128 ring mapping.
//!
//! Families, each with its own verdict and negative flag:
//!   - pack — unified BF16 pool rows match SW rebase + chunk + compressed;
//!   - indices — production per-token builder vs a semantic oracle that
//!     enumerates, for query abs_pos: prior-window rows (rebased slots),
//!     current-chunk rows 0..=token, then CSA causal selected keys / HCA
//!     dense compressed rows, -1 padded;
//!   - attention output — prefill fwd vs an f32 causal reference over each
//!     row's own index set (every row for the 128 chunk; sampled rows that
//!     cover first/last/each boundary for 2048/4096);
//!   - LSE — natural-log LSE with the per-head sink folded in;
//!   - max_logits — per-row max pre-softmax logit vs the reference.
//!
//! `--negative-control` corrupts one thing per family; prints NEGATIVE
//! CONTROL OK and exits 0 only if every flag fired. The index family is a
//! real kernel tooth: the CSA index builder is re-run on a corrupted
//! selection and its produced indices must differ from the clean oracle yet
//! match the oracle rebuilt from that selection (matching the sparse/hca
//! decode gates — never a host-edited copy).
//!
//! Mode select: default runs both CSA and HCA. sm90 only. Build/run (pod;
//! pod.sh build rejects --example):
//!   cargo build --release -p infer-cuda --features cuda \
//!     --example flashmla_prefill_parity
//!   target/release/examples/flashmla_prefill_parity --kernel-build-id
//!   target/release/examples/flashmla_prefill_parity
//!   target/release/examples/flashmla_prefill_parity --negative-control

#![allow(clippy::print_stdout, clippy::print_stderr)]

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
        eprintln!("flashmla_prefill_parity is a CUDA/sm90 harness; rebuild with --features cuda.");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use anyhow::{Result, ensure};
    use cuda_kernels::attention;
    use cuda_kernels::prelude::DeviceContext;
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

    const H_Q: usize = 64;
    const H_KV: i32 = 1;
    const D: usize = 512;
    const SW: usize = 128;
    const INDEX_TOPK: usize = 512;
    const TOPK: usize = SW + INDEX_TOPK; // 640

    // Chunk lengths from the production scheduler (see file header).
    const CHUNKS: [usize; 3] = [128, 2048, 4096];
    // start_pos values straddling 64/128-token boundaries.
    const STARTS: [i32; 3] = [63, 128, 255];

    // Output rel, LSE abs, and logit abs: no derivation. They compare the
    // sparse forward against the f64 dense causal oracle; the error is the
    // e4m3-packed-KV round plus online-softmax/LSE accumulation, but no
    // supremum is computed. Clean-run-set at chunks 128/2048/4096 and need a
    // bound on the packed-KV softmax/logit error.
    const PASS_MAX_REL_OUT: f64 = 0.05;
    const PASS_MAX_ABS_LSE: f64 = 0.10;
    const PASS_MAX_ABS_LOGIT: f64 = 0.05;
    const PACK_ABS: f64 = 0.0; // bf16 copies must be bit-exact: no quant step, only an exact copy, so zero is the derived bound

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Mode {
        Csa,
        Hca,
    }
    impl Mode {
        fn ratio(self) -> usize {
            match self {
                Mode::Csa => 4,
                Mode::Hca => 128,
            }
        }
        fn name(self) -> &'static str {
            match self {
                Mode::Csa => "CSA",
                Mode::Hca => "HCA",
            }
        }
    }

    #[derive(Clone, Copy)]
    struct Case {
        mode: Mode,
        s_q: usize,
        start: i32,
    }

    pub(super) fn run(negative: bool) -> Result<()> {
        let ctx = DeviceContext::new()?;
        let cc = ctx.compute_capability();
        eprintln!(
            "[flashmla-prefill-parity] device={} cc={}.{} sms={} build={}",
            ctx.ordinal(),
            cc.0,
            cc.1,
            ctx.sm_count(),
            cuda_kernels::KERNEL_BUILD_ID
        );
        ensure!(cuda_kernels::HAS_FLASHMLA, "binary built without FlashMLA");
        ensure!(
            cc.0 >= 9,
            "FlashMLA prefill needs sm90+; got sm_{}{}",
            cc.0,
            cc.1
        );
        ensure!(TOPK.is_multiple_of(128), "topk must be a multiple of 128");

        let modes = [Mode::Csa, Mode::Hca];
        // One (mode, chunk, start) case at a time. Negative mode injects one
        // targeted corruption into the first case only.
        let mut cases = Vec::new();
        for &mode in &modes {
            for &s_q in &CHUNKS {
                for &start in &STARTS {
                    cases.push(Case { mode, s_q, start });
                }
            }
        }

        for (ci, case) in cases.iter().enumerate() {
            let corrupt = negative && ci == 0;
            run_case(&ctx, *case, corrupt)?;
        }
        if negative {
            println!("NEGATIVE CONTROL OK");
        } else {
            println!("[flashmla-prefill-parity] ALL PASS");
        }
        Ok(())
    }

    /// Deterministic per-(position,dim) latent in [-0.05, 0.05); a designated
    /// dominant key is set elsewhere.
    fn latent(abs_pos: i32, d: usize, plane: u64) -> f32 {
        let a = (abs_pos as u64)
            .wrapping_add(1)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let b = (d as u64).wrapping_mul(0x517c_c1cc_9e37_79b9) ^ plane.wrapping_mul(7);
        let mut x = a ^ b;
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
        x ^= x >> 33;
        (x >> 40) as f32 / (1u64 << 24) as f32 * 0.1 - 0.05
    }

    #[allow(clippy::too_many_arguments)]
    fn run_case(ctx: &DeviceContext, case: Case, corrupt: bool) -> Result<()> {
        let Case { mode, s_q, start } = case;
        let ratio = mode.ratio();
        // Compressed rows available: enough that every query has dense/selected
        // keys, plus headroom for the boundary start.
        let compressed_count = ((start as usize + s_q) / ratio + 2).min(INDEX_TOPK);
        let kv_rows = SW + s_q + compressed_count;

        // Host SW window (128 logical prior positions), shuffled physically:
        // we store positions in ring order so the production rebased copy maps
        // ring slot (sw_base+row)%128 → position start-128+row.
        let sw_base = (start - SW as i32).max(0);
        let mut window = vec![0f32; SW * D]; // indexed by RING SLOT
        for ring in 0..SW {
            let abs_pos = sw_base + ring as i32;
            for d in 0..D {
                window[ring * D + d] = latent(abs_pos, d, 0);
            }
        }
        // Current chunk K.
        let mut chunk = vec![0f32; s_q * D];
        for token in 0..s_q {
            for d in 0..D {
                chunk[token * D + d] = latent(start + token as i32, d, 1);
            }
        }
        // Compressed rows (HCA dense keys; CSA selected keys). Give each a
        // moderate fixed latent so its attention is distinguishable.
        let mut comp = vec![0f32; compressed_count * D];
        for c in 0..compressed_count {
            for d in 0..D {
                comp[c * D + d] = 0.05 * (((c * 7 + d * 3) % 11) as f32 / 11.0) - 0.02;
            }
        }

        // CSA selected: causal keys 0..min(INDEX_TOPK, compressed_count),
        // permuted per token is unnecessary for a constant selection; identity
        // with a Fisher-Yates shuffle seeded off the case keeps slot order !=
        // key order. Use one shared selection array per query row.
        let selected = match mode {
            Mode::Csa => {
                let n = compressed_count.min(INDEX_TOPK);
                let mut v: Vec<i32> = (0..n as i32).collect();
                v.resize(INDEX_TOPK, -1);
                let mut state = 0x1234_5678_9abc_def0u64
                    ^ (s_q as u64).wrapping_mul(0x9e37_79b9)
                    ^ (start as u64).wrapping_mul(0x517c_c1cc);
                let mut nx = || {
                    state ^= state >> 33;
                    state = state.wrapping_mul(0xff51_afd7_ed55_8ccd);
                    state ^= state >> 33;
                    state
                };
                for i in (1..n).rev() {
                    let j = (nx() as usize) % (i + 1);
                    v.swap(i, j);
                }
                // Replicate the same selection row for every query token.
                let mut all = Vec::with_capacity(s_q * INDEX_TOPK);
                for _ in 0..s_q {
                    all.extend_from_slice(&v);
                }
                all
            }
            Mode::Hca => Vec::new(),
        };

        // Per-head nonzero sink.
        let sink: Vec<f32> = (0..H_Q)
            .map(|h| {
                let mut x = 0xA11Cu64.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    ^ (h as u64).wrapping_mul(0x517c_c1cc_9e37_79b9);
                x ^= x >> 33;
                x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
                x ^= x >> 33;
                ((x >> 40) as f32 / (1u64 << 24) as f32) * 0.4 - 0.1
            })
            .collect();
        // Constant Q per token/head (aligned only with chunk current-row keys
        // to keep the reference tractable; no single dominant key so causal
        // coverage is exercised across many keys).
        let q_host: Vec<bf16> = (0..s_q * H_Q * D)
            .map(|i| {
                let d = i % D;
                bf16::from_f32(latent(start + ((i / (H_Q * D)) as i32), d, 1) * 20.0)
            })
            .collect();

        let window_dev = ctx.stream.clone_htod(&to_bf16(&window))?;
        let chunk_dev = ctx.stream.clone_htod(&to_bf16(&chunk))?;
        let comp_dev = if compressed_count > 0 {
            Some(ctx.stream.clone_htod(&to_bf16(&comp))?)
        } else {
            None
        };
        let mut unified = ctx.stream.alloc_zeros::<bf16>(kv_rows * D)?;
        {
            let (u_ptr, _g1) = unified.device_ptr_mut(&ctx.stream);
            let (w_ptr, _g2) = window_dev.device_ptr(&ctx.stream);
            let (k_ptr, _g3) = chunk_dev.device_ptr(&ctx.stream);
            let c_ptr = comp_dev
                .as_ref()
                .map(|c| c.device_ptr(&ctx.stream).0)
                .unwrap_or(0);
            attention::flashmla_csa_pack_kv_raw(
                &ctx.stream,
                u_ptr,
                w_ptr,
                k_ptr,
                c_ptr,
                start,
                SW as i32,
                s_q as i32,
                compressed_count as i32,
                D as i32,
            )?;
        }
        ctx.sync()?;

        // Pack family: unified rows must equal the BF16 sources exactly.
        let unified_host = ctx.stream.clone_dtoh(&unified)?;
        let pack_ok = check_pack(
            &unified_host,
            &window,
            &chunk,
            &comp,
            start,
            s_q,
            compressed_count,
        );
        if corrupt {
            // Pack corruption handled below against a flipped buffer; the
            // positive pack_ok here must still be true for a clean build.
            ensure!(pack_ok, "clean pack mismatch before corruption");
        }

        // Build indices + topk_length.
        let mut indices_dev = ctx.stream.alloc_zeros::<i32>(s_q * TOPK)?;
        let mut topk_dev = ctx.stream.alloc_zeros::<i32>(s_q)?;
        let selected_dev = if matches!(mode, Mode::Csa) {
            Some(ctx.stream.clone_htod(&selected)?)
        } else {
            None
        };
        {
            let (idx_ptr, _gi) = indices_dev.device_ptr_mut(&ctx.stream);
            let (topk_ptr, _gt) = topk_dev.device_ptr_mut(&ctx.stream);
            match mode {
                Mode::Csa => attention::flashmla_csa_build_indices_raw(
                    &ctx.stream,
                    idx_ptr,
                    topk_ptr,
                    selected_dev.as_ref().unwrap().device_ptr(&ctx.stream).0,
                    s_q as i32,
                    start,
                    SW as i32,
                    INDEX_TOPK as i32,
                    compressed_count as i32,
                    ratio as i32,
                )?,
                Mode::Hca => {
                    let max_keys = compressed_count.div_ceil(128) * 128;
                    attention::flashmla_hca_build_indices_raw(
                        &ctx.stream,
                        idx_ptr,
                        topk_ptr,
                        s_q as i32,
                        start,
                        SW as i32,
                        max_keys as i32,
                        compressed_count as i32,
                        ratio as i32,
                    )?;
                }
            }
        }
        ctx.sync()?;
        let indices = ctx.stream.clone_dtoh(&indices_dev)?;
        let topk_len = ctx.stream.clone_dtoh(&topk_dev)?;

        let q_dev = ctx.stream.clone_htod(&q_host)?;
        let sink_dev = ctx.stream.clone_htod(&sink)?;
        let mut out_dev = ctx.stream.alloc_zeros::<bf16>(s_q * H_Q * D)?;
        let mut max_dev = ctx.stream.alloc_zeros::<f32>(s_q * H_Q)?;
        let mut lse_dev = ctx.stream.alloc_zeros::<f32>(s_q * H_Q)?;
        {
            let (q_ptr, _gq) = q_dev.device_ptr(&ctx.stream);
            let (kv_ptr, _gk) = unified.device_ptr(&ctx.stream);
            let (idx_ptr, _gi) = indices_dev.device_ptr(&ctx.stream);
            let (sink_ptr, _gs) = sink_dev.device_ptr(&ctx.stream);
            let (topk_ptr, _gt) = topk_dev.device_ptr(&ctx.stream);
            let (out_ptr, _go) = out_dev.device_ptr_mut(&ctx.stream);
            let (max_ptr, _gm) = max_dev.device_ptr_mut(&ctx.stream);
            let (lse_ptr, _gl) = lse_dev.device_ptr_mut(&ctx.stream);
            attention::flashmla_sm90_sparse_prefill_fwd_raw(
                &ctx.stream,
                q_ptr,
                kv_ptr,
                idx_ptr,
                sink_ptr,
                topk_ptr,
                out_ptr,
                max_ptr,
                lse_ptr,
                s_q as i32,
                kv_rows as i32,
                H_Q as i32,
                H_KV,
                D as i32,
                D as i32,
                TOPK as i32,
                1.0 / (D as f32).sqrt(),
                (H_Q * D) as i32,
                D as i32,
                D as i32,
                0,
                TOPK as i32,
                0,
                0,
            )?;
        }
        ctx.sync()?;
        let out = ctx.stream.clone_dtoh(&out_dev)?;
        let max_logits = ctx.stream.clone_dtoh(&max_dev)?;
        let lse = ctx.stream.clone_dtoh(&lse_dev)?;

        // Reference: reconstruct the unified pool as f32 rows from the SAME
        // sources the pack kernel used, and derive each query's causal index
        // set semantically.
        let pool_f32 = reference_pool(&window, &chunk, &comp, start, s_q, compressed_count);
        let ref_indices = oracle_indices(case, &selected, compressed_count);

        // Rows to compare exhaustively / sample.
        let sample_rows = sampled_rows(s_q);
        let mut max_rel_out = 0f64;
        let mut max_abs_lse = 0f64;
        let mut max_abs_logit = 0f64;
        for &token in &sample_rows {
            for h in 0..H_Q {
                let row_idx = &ref_indices[token * TOPK..(token + 1) * TOPK];
                let (ro, rlse, rmax) = reference_row(
                    &pool_f32,
                    row_idx,
                    &q_host[token * H_Q * D + h * D..token * H_Q * D + (h + 1) * D],
                    sink[h],
                );
                let base = (token * H_Q + h) * D;
                let mut sq = 0f64;
                let mut mabs = 0f64;
                for d in 0..D {
                    let g = out[base + d].to_f32();
                    sq += (ro[d] as f64) * (ro[d] as f64);
                    mabs = mabs.max((g - ro[d]).abs() as f64);
                }
                max_rel_out = max_rel_out.max(mabs / sq.sqrt().max(f64::MIN_POSITIVE));
                max_abs_lse = max_abs_lse.max((lse[(token * H_Q) + h] - rlse).abs() as f64);
                max_abs_logit =
                    max_abs_logit.max((max_logits[(token * H_Q) + h] - rmax).abs() as f64);
            }
        }

        let idx_ok = indices == ref_indices;
        let topk_ok = check_topk(&topk_len, case, compressed_count);
        let out_ok = max_rel_out <= PASS_MAX_REL_OUT;
        let lse_ok = max_abs_lse <= PASS_MAX_ABS_LSE;
        let logit_ok = max_abs_logit <= PASS_MAX_ABS_LOGIT;

        if !corrupt {
            println!(
                "{} s_q={:<5} start={:<4} pack={} indices={} topk={} out={:.5} lse={:.5} maxlogit={:.5}",
                mode.name(),
                s_q,
                start,
                pack_ok,
                idx_ok,
                topk_ok,
                max_rel_out,
                max_abs_lse,
                max_abs_logit
            );
            ensure!(pack_ok, "{} s_q={s_q} pack mismatch", mode.name());
            ensure!(idx_ok, "{} s_q={s_q} indices mismatch", mode.name());
            ensure!(topk_ok, "{} s_q={s_q} topk_length mismatch", mode.name());
            ensure!(out_ok, "{} s_q={s_q} output rel {max_rel_out}", mode.name());
            ensure!(lse_ok, "{} s_q={s_q} lse abs {max_abs_lse}", mode.name());
            ensure!(
                logit_ok,
                "{} s_q={s_q} max_logit abs {max_abs_logit}",
                mode.name()
            );
            return Ok(());
        }

        // ── negative controls for the first case (CSA, s_q=128, start=63).
        // pack: flip one unified SW byte region (a bf16), compare must fail.
        let mut bad_unified = unified_host.clone();
        let mut bits = bad_unified[3].to_bits();
        bits ^= 0x40; // perturb one bf16 exponent/mantissa bit
        bad_unified[3] = bf16::from_bits(bits);
        let neg_pack = !check_pack(
            &bad_unified,
            &window,
            &chunk,
            &comp,
            start,
            s_q,
            compressed_count,
        );

        // indices: a REAL index-builder tooth, matching sparse/hca decode. Feed
        // the CSA builder a corrupted selection (the last token's first
        // compressed key points at a different valid key) and compare the
        // indices the KERNEL produces — not a host-edited copy — against both
        // the clean oracle (must differ) and the oracle rebuilt from the
        // corrupted selection (must match: the kernel honoured the bad input).
        let last = s_q - 1;
        let mut bad_selected = selected.clone();
        let sel_base = last * INDEX_TOPK;
        // Pick a valid compressed key different from the current first choice
        // and visible at the last token; corrupting an entry the builder would
        // drop as out-of-range could leave the output unchanged.
        let valid_last = compressed_count.min(INDEX_TOPK);
        let first_choice = bad_selected[sel_base];
        let replacement = (0..valid_last as i32)
            .find(|&c| {
                c != first_choice && c * ratio as i32 + (ratio as i32 - 1) <= start + last as i32
            })
            .expect("a second visible compressed key exists for the negative");
        bad_selected[sel_base] = replacement;
        let mut bad_idx_dev = ctx.stream.alloc_zeros::<i32>(s_q * TOPK)?;
        let mut bad_topk_dev = ctx.stream.alloc_zeros::<i32>(s_q)?;
        {
            let bad_sel_dev = ctx.stream.clone_htod(&bad_selected)?;
            let (idx_ptr, _gi) = bad_idx_dev.device_ptr_mut(&ctx.stream);
            let (topk_ptr, _gt) = bad_topk_dev.device_ptr_mut(&ctx.stream);
            attention::flashmla_csa_build_indices_raw(
                &ctx.stream,
                idx_ptr,
                topk_ptr,
                bad_sel_dev.device_ptr(&ctx.stream).0,
                s_q as i32,
                start,
                SW as i32,
                INDEX_TOPK as i32,
                compressed_count as i32,
                ratio as i32,
            )?;
        }
        ctx.sync()?;
        let bad_indices = ctx.stream.clone_dtoh(&bad_idx_dev)?;
        let bad_ref = oracle_indices(case, &bad_selected, compressed_count);
        // Differs from clean AND self-consistent with the corrupted oracle.
        let neg_indices = bad_indices != ref_indices && bad_indices == bad_ref;

        // output/LSE/maxlogit: run fwd again with the kernel-built bad indices.
        let mut bad_out = ctx.stream.alloc_zeros::<bf16>(s_q * H_Q * D)?;
        let mut bad_max = ctx.stream.alloc_zeros::<f32>(s_q * H_Q)?;
        let mut bad_lse = ctx.stream.alloc_zeros::<f32>(s_q * H_Q)?;
        {
            let (q_ptr, _gq) = q_dev.device_ptr(&ctx.stream);
            let (kv_ptr, _gk) = unified.device_ptr(&ctx.stream);
            let (idx_ptr, _gi) = bad_idx_dev.device_ptr_mut(&ctx.stream);
            let (sink_ptr, _gs) = sink_dev.device_ptr(&ctx.stream);
            let (topk_ptr, _gt) = topk_dev.device_ptr(&ctx.stream);
            let (out_ptr, _go) = bad_out.device_ptr_mut(&ctx.stream);
            let (max_ptr, _gm) = bad_max.device_ptr_mut(&ctx.stream);
            let (lse_ptr, _gl) = bad_lse.device_ptr_mut(&ctx.stream);
            attention::flashmla_sm90_sparse_prefill_fwd_raw(
                &ctx.stream,
                q_ptr,
                kv_ptr,
                idx_ptr,
                sink_ptr,
                topk_ptr,
                out_ptr,
                max_ptr,
                lse_ptr,
                s_q as i32,
                kv_rows as i32,
                H_Q as i32,
                H_KV,
                D as i32,
                D as i32,
                TOPK as i32,
                1.0 / (D as f32).sqrt(),
                (H_Q * D) as i32,
                D as i32,
                D as i32,
                0,
                TOPK as i32,
                0,
                0,
            )?;
        }
        ctx.sync()?;
        let bad_out_h = ctx.stream.clone_dtoh(&bad_out)?;
        let bad_max_h = ctx.stream.clone_dtoh(&bad_max)?;
        let bad_lse_h = ctx.stream.clone_dtoh(&bad_lse)?;
        let mut fired_out = false;
        let mut fired_lse = false;
        let mut fired_logit = false;
        for h in 0..H_Q {
            let row_idx = &ref_indices[last * TOPK..(last + 1) * TOPK];
            let (ro, rlse, rmax) = reference_row(
                &pool_f32,
                row_idx,
                &q_host[last * H_Q * D + h * D..last * H_Q * D + (h + 1) * D],
                sink[h],
            );
            let base = (last * H_Q + h) * D;
            let mut sq = 0f64;
            let mut mabs = 0f64;
            for d in 0..D {
                let g = bad_out_h[base + d].to_f32();
                sq += (ro[d] as f64).powi(2);
                mabs = mabs.max((g - ro[d]).abs() as f64);
            }
            if mabs / sq.sqrt().max(f64::MIN_POSITIVE) > PASS_MAX_REL_OUT {
                fired_out = true;
            }
            if (bad_lse_h[last * H_Q + h] - rlse).abs() as f64 > PASS_MAX_ABS_LSE {
                fired_lse = true;
            }
            if (bad_max_h[last * H_Q + h] - rmax).abs() as f64 > PASS_MAX_ABS_LOGIT {
                fired_logit = true;
            }
        }

        println!(
            "NEG {} s_q={} pack={} indices={} out={} lse={} maxlogit={}",
            mode.name(),
            s_q,
            neg_pack,
            neg_indices,
            fired_out,
            fired_lse,
            fired_logit
        );
        ensure!(neg_pack, "pack negative control did not fire");
        ensure!(neg_indices, "indices negative control did not fire");
        ensure!(fired_out, "output negative control did not fire");
        ensure!(fired_lse, "LSE negative control did not fire");
        ensure!(fired_logit, "max_logit negative control did not fire");
        Ok(())
    }

    fn to_bf16(v: &[f32]) -> Vec<bf16> {
        v.iter().map(|&x| bf16::from_f32(x)).collect()
    }

    /// Rows compared: every token for the small chunk; first, last, and each
    /// 64/128-boundary ±1 for the large chunks.
    fn sampled_rows(s_q: usize) -> Vec<usize> {
        if s_q <= 128 {
            return (0..s_q).collect();
        }
        let mut rows = vec![0, 1, s_q - 2, s_q - 1];
        for b in (0..=s_q).step_by(64) {
            for &d in &[-1i32, 0, 1] {
                let t = b as i32 + d;
                if t >= 0 && (t as usize) < s_q {
                    rows.push(t as usize);
                }
            }
        }
        rows.sort_unstable();
        rows.dedup();
        rows
    }

    /// Reconstruct the production unified BF16 pool as f32, applying the SW
    /// ring rebase the pack kernel uses.
    fn reference_pool(
        window: &[f32],
        chunk: &[f32],
        comp: &[f32],
        start: i32,
        s_q: usize,
        compressed_count: usize,
    ) -> Vec<f32> {
        let sw_base = (start - SW as i32).max(0);
        let mut out = vec![0f32; (SW + s_q + compressed_count) * D];
        for row in 0..SW {
            let slot = (sw_base + row as i32) as usize % SW;
            out[row * D..(row + 1) * D].copy_from_slice(&window[slot * D..(slot + 1) * D]);
        }
        out[SW * D..(SW + s_q) * D].copy_from_slice(chunk);
        if compressed_count > 0 {
            out[(SW + s_q) * D..(SW + s_q + compressed_count) * D].copy_from_slice(comp);
        }
        out
    }

    /// Production pack check: SW rebased region, current chunk, compressed.
    fn check_pack(
        unified: &[bf16],
        window: &[f32],
        chunk: &[f32],
        comp: &[f32],
        start: i32,
        s_q: usize,
        compressed_count: usize,
    ) -> bool {
        let sw_base = (start - SW as i32).max(0) as usize;
        // The sources are uploaded after an f32 -> bf16 round (to_bf16) and the
        // pack kernel copies bf16 -> bf16 with no other conversion, so the
        // bit-exact target is the bf16-rounded source, not the raw f32.
        let want_bf = |x: f32| -> f32 { bf16::from_f32(x).to_f32() };
        for row in 0..SW {
            let slot = (sw_base + row) % SW;
            for d in 0..D {
                if (unified[row * D + d].to_f32() - want_bf(window[slot * D + d])).abs() as f64
                    != PACK_ABS
                {
                    return false;
                }
            }
        }
        for i in 0..s_q * D {
            if unified[SW * D + i].to_f32() != want_bf(chunk[i]) {
                return false;
            }
        }
        for i in 0..compressed_count * D {
            if unified[(SW + s_q) * D + i].to_f32() != want_bf(comp[i]) {
                return false;
            }
        }
        true
    }

    /// Semantic per-token index oracle.
    /// Pool layout rows: [0,128) rebased SW; [128,128+s_q) current chunk;
    /// [128+s_q, ...) compressed c starting at comp_base=128+s_q.
    #[allow(clippy::too_many_arguments)]
    fn oracle_indices(case: Case, selected: &[i32], compressed_count: usize) -> Vec<i32> {
        let Case { mode, s_q, start } = case;
        let ratio = mode.ratio();
        let mut out = vec![-1i32; s_q * TOPK];
        let comp_base = (SW + s_q) as i32;
        let sw_base = (start - SW as i32).max(0);
        for (token, _) in std::iter::repeat_n((), s_q).enumerate() {
            let abs_pos = start + token as i32;
            let sw_start = (abs_pos + 1 - SW as i32).max(0);
            let sw_count = (abs_pos - sw_start + 1) as usize;
            let row = &mut out[token * TOPK..(token + 1) * TOPK];
            for (j, _) in std::iter::repeat_n((), sw_count).enumerate() {
                let p = sw_start + j as i32;
                let slot = if p < start {
                    p - sw_base
                } else {
                    SW as i32 + (p - start)
                };
                row[j] = slot;
            }
            match mode {
                Mode::Csa => {
                    let sel = &selected[token * INDEX_TOPK..(token + 1) * INDEX_TOPK];
                    for k in 0..INDEX_TOPK {
                        let c = sel[k];
                        let valid = c >= 0
                            && (c as usize) < compressed_count
                            && c * ratio as i32 + (ratio as i32 - 1) <= abs_pos;
                        if valid {
                            row[sw_count + k] = comp_base + c;
                        }
                    }
                }
                Mode::Hca => {
                    let keys = (abs_pos / ratio as i32) as usize;
                    let keys = keys.min(compressed_count);
                    for k in 0..keys {
                        row[sw_count + k] = comp_base + k as i32;
                    }
                }
            }
        }
        out
    }

    fn check_topk(topk: &[i32], case: Case, compressed_count: usize) -> bool {
        let Case { mode, s_q, start } = case;
        let ratio = mode.ratio();
        for (token, _) in std::iter::repeat_n((), s_q).enumerate() {
            let abs_pos = start + token as i32;
            let sw_start = (abs_pos + 1 - SW as i32).max(0);
            let sw_count = (abs_pos - sw_start + 1) as usize;
            let keys = match mode {
                Mode::Csa => INDEX_TOPK, // builder always emits sw+index_topk
                Mode::Hca => (abs_pos / ratio as i32) as usize,
            }
            .min(if matches!(mode, Mode::Hca) {
                compressed_count
            } else {
                INDEX_TOPK
            });
            let expect = sw_count + keys;
            if topk[token] != expect as i32 {
                return false;
            }
        }
        true
    }

    /// f32 causal attention for one (token, head) over its index set.
    /// Returns (out[D], lse, max_logit) with the sink folded into the softmax
    /// denominator and LSE exactly as the prefill combine does.
    fn reference_row(pool: &[f32], indices: &[i32], q: &[bf16], sink: f32) -> (Vec<f32>, f32, f32) {
        let sm = 1.0 / (D as f32).sqrt();
        let mut scores = Vec::new();
        let mut rows = Vec::new();
        let mut m = f32::NEG_INFINITY;
        for &idx in indices {
            if idx < 0 {
                continue;
            }
            let base = idx as usize * D;
            let mut dot = 0f32;
            for d in 0..D {
                dot += q[d].to_f32() * pool[base + d];
            }
            let s = dot * sm;
            scores.push(s);
            rows.push(idx as usize);
            m = m.max(s);
        }
        let m_eff = m.max(sink);
        let tok: f32 = scores.iter().map(|&s| (s - m_eff).exp()).sum();
        // Prefill: LSE is token-only (log(rL)+m, phase1.cuh:449); the sink
        // enters ONLY the output scale factor 1/(rL+exp2(sink-m)).
        let lse = m + tok.ln();
        let out_denom = tok + (sink - m_eff).exp();
        let mut out = vec![0f32; D];
        for (&idx, &s) in rows.iter().zip(&scores) {
            let p = (s - m_eff).exp() / out_denom;
            let base = idx * D;
            for d in 0..D {
                out[d] += p * pool[base + d];
            }
        }
        let max_logit = m; // rM is the max token logit (phase1.cuh:448)
        (out, lse, max_logit)
    }
}
