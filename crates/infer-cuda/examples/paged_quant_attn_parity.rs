//! sm80+ native 1-byte paged attention numeric-parity gate.
//!
//! Standalone kernel harness — NO engine, NO serve — for
//! `paged_attention_quantized_fa3_cuda`
//! (`crates/cuda-kernels/csrc/attention/paged_attention_quantized_fa3.cu`):
//! the production `--kv-cache-dtype int8|fp8` decode / short-verify kernel on
//! A100-tier (sm_80..sm_89) hosts where the FA3 hd256 shim does not exist.
//! head_dim 256 is a hard template instantiation in the shipped launcher
//! (`LAUNCH_PARTIAL(256, …)` at paged_attention_quantized_fa3.cu:585, merge
//! `<256>` at :600, alongside the `<128>` rows); the template `__trap`s below
//! sm_80 (same file:116).
//! It reads the 1-byte pool directly (no dequant temp): one CTA per
//! (kv head, split, batch row, q-tile), bf16 `mma.sync` tensor cores with
//! the per-token K scale folded into the score columns and the per-token V
//! scale into the probability operand, then an f32 online-softmax merge.
//!
//! Geometry is Qwen3.5-4B full attention per its Hugging Face config.json
//! (huggingface.co/Qwen/Qwen3.5-4B): 16 query heads / 4 KV heads (GQA 4),
//! head_dim 256, 8 full-attention layers, page_size 16, B in {1,8}.
//!
//! Coverage:
//! - INT8 and FP8 e4m3 pools (the two production dtypes), one form each;
//! - per-(token, kv_head) f32 scales planted as the production quantizer's
//!   token absmax / qmax (127 INT8, 448 FP8);
//! - DISCONTINUOUS LCG-scattered and ROTATED non-identity page tables over a
//!   junk-filled pool;
//! - mixed per-row kv lengths including 1 and the 15/16/17, 127/128/129, 257
//!   page boundaries;
//! - GQA: the oracle maps all 16 q heads to kv head h/4;
//! - decode rows (qlen 1) and spec-verify rows up to the kernel's
//!   PAF3_MAX_QLEN=8 packed through `cu_seqlens_q` in one B=8 launch;
//! - split-KV: num_splits 1 (single-CTA accumulation) and 8 (partial +
//!   merge, the production decode configuration).
//!
//! The f64 oracle shares the durable pool BYTES with the kernel — e4m3 and
//! int8 values are exact bf16 subsets, so there is no dequant-rounding term
//! (unlike the FA3 dequant shim): it decodes the bytes exactly, multiplies
//! the f32 per-token scale in f64, runs bottom-right causal softmax, and
//! rounds the output to bf16 once. The remaining gap is f32 tile
//! accumulation, split merge, and fast `__expf`, so the band is close to the
//! BF16-pool gate's. Every output column is compared on every query row.
//!
//! Negative control is per pool family: `--negative-control=int8|fp8`
//! corrupts one family's expectations and requires the other to stay clean;
//! the bare flag corrupts both.
//!
//! Run on a pod (sm_80 or newer; builds on T1, traps below sm_80):
//!   cargo build --release -p infer-cuda --features cuda \
//!     --example paged_quant_attn_parity
//!   target/release/examples/paged_quant_attn_parity --kernel-build-id
//!   INFER_CUDA_DEVICE=<free-gpu> target/release/examples/paged_quant_attn_parity
//!   ... --negative-control=int8
//!   ... --negative-control=fp8

#[cfg(feature = "cuda")]
#[path = "support/attn_common.rs"]
mod attn_common;

fn main() -> anyhow::Result<()> {
    use parity_common::Parsed;
    match parity_common::cli() {
        Parsed::BuildIdPrinted => Ok(()),
        Parsed::Run(cli) => {
            // The two production pool dtypes are the two gate families.
            let sel = match cli.negative_value.as_deref() {
                None if cli.negative => None,
                Some("int8") => Some(Family::Int8),
                Some("fp8") => Some(Family::Fp8),
                Some(other) => anyhow::bail!("unknown --negative-control family {other:?}"),
                None => None,
            };
            let negative = if cli.negative { Some(sel) } else { None };
            real::run(negative)
        }
    }
}

/// The two production pool dtypes are the two gate families.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Family {
    Int8,
    Fp8,
}

#[allow(dead_code)] // shared harness; each gate uses only the subset it needs
#[path = "support/parity_common.rs"]
mod parity_common;

#[cfg(not(feature = "cuda"))]
mod real {
    use super::Family;

    pub(super) fn run(_negative: Option<Option<Family>>) -> anyhow::Result<()> {
        eprintln!("paged_quant_attn_parity is a CUDA harness; rebuild with --features cuda.");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use super::Family;
    use super::attn_common::{
        Rng, Tol, attention_row, bf, compare_rows, e4m3_decode, e4m3_encode, i8_encode, lcg_perm,
        metrics_pass,
    };
    use anyhow::{Result, ensure};
    use cuda_kernels::prelude::DeviceContext;
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

    // Qwen3.5-4B full-attention geometry (HF config.json).
    const H: usize = 16;
    const HK: usize = 4;
    const G: usize = H / HK;
    const D: usize = 256;
    const PAGE: usize = 16;
    const SM_SCALE: f64 = 1.0 / 16.0; // 1/sqrt(256)
    const SEED: u64 = 0x5a80_1071_7125_0a31;
    const MAX_QLEN: usize = 8; // PAF3_MAX_QLEN in the .cu (validated server-side)

    // The kernel and oracle consume the SAME decoded bytes (e4m3/int8 are
    // exact bf16 subsets); the gap is f32 tile accumulation, per-split merge,
    // fast __expf, and the one bf16 output round. Because both sides read the
    // same bytes there is NO quantization-error term: the floor is one bf16
    // output store (rms 2^-8/sqrt(3) ≈ 2.3e-3) plus unclosed-form __expf and
    // merge-order differences. The 8e-2/7e-2/3e-2 tuple sits far above that
    // floor and is clean-run-set, not a supremum; it needs the __expf/merge
    // bound computed and is a Phase-2 tightening candidate.
    const TOL: Tol = Tol {
        rel_l2: 8e-2,
        slope: 7e-2,
        floor: 3e-2,
        max_viol_frac: 5e-3,
    };

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum TableKind {
        Discont,
        Rotated,
    }

    struct Case {
        label: &'static str,
        is_fp8: bool,
        table_kind: TableKind,
        splits: usize,
        kvlens: Vec<usize>,
        cu_q: Vec<i32>,
        table: Vec<i32>,
        table_stride: usize,
        q_packed: Vec<bf16>,
        /// NHD 1-byte pools `[phys, off, hk, d]`, K and V separate.
        k_pool: Vec<u8>,
        v_pool: Vec<u8>,
        k_scales: Vec<f32>,
        v_scales: Vec<f32>,
    }

    impl Case {
        fn batch(&self) -> usize {
            self.kvlens.len()
        }
        fn total_q(&self) -> usize {
            *self.cu_q.last().unwrap() as usize
        }
        fn max_qlen(&self) -> usize {
            self.cu_q
                .windows(2)
                .map(|w| (w[1] - w[0]) as usize)
                .max()
                .unwrap()
        }
        fn family(&self) -> Family {
            if self.is_fp8 {
                Family::Fp8
            } else {
                Family::Int8
            }
        }
        fn row_of_token(&self, t: usize) -> usize {
            self.cu_q.partition_point(|&e| (e as usize) <= t) - 1
        }
        /// Exact decoded K/V byte value at logical (b, t, hk, d); the kernel
        /// dequantizes bytes into bf16 with no rounding.
        fn byte_value(&self, is_v: bool, b: usize, t: usize, hk: usize, d: usize) -> f64 {
            let phys = self.table[b * self.table_stride + t / PAGE] as usize;
            let off = t % PAGE;
            let pool = if is_v { &self.v_pool } else { &self.k_pool };
            let byte = pool[(phys * PAGE + off) * HK * D + hk * D + d];
            if self.is_fp8 {
                f64::from(e4m3_decode(byte))
            } else {
                f64::from(byte as i8)
            }
        }
        fn scale(&self, is_v: bool, b: usize, t: usize, hk: usize) -> f64 {
            let phys = self.table[b * self.table_stride + t / PAGE] as usize;
            let off = t % PAGE;
            let s = if is_v { &self.v_scales } else { &self.k_scales };
            f64::from(s[(phys * PAGE + off) * HK + hk])
        }
    }

    fn build_case(
        label: &'static str,
        is_fp8: bool,
        table_kind: TableKind,
        splits: usize,
        kvlens: Vec<usize>,
        qlens: Vec<usize>,
        case_seed: u64,
    ) -> Case {
        let batch = kvlens.len();
        assert_eq!(qlens.len(), batch);
        assert!(qlens.iter().all(|&q| q <= MAX_QLEN));
        let mut rng = Rng::new(
            SEED ^ ((is_fp8 as u64) << 40)
                ^ ((table_kind as usize as u64) << 36)
                ^ ((kvlens.iter().sum::<usize>() as u64) << 8)
                ^ ((qlens[0] as u64) << 20)
                ^ case_seed,
        );
        let qmax: f32 = if is_fp8 { 448.0 } else { 127.0 };

        let pages_per: Vec<usize> = kvlens.iter().map(|l| l.div_ceil(PAGE)).collect();
        let total_pages: usize = pages_per.iter().sum();
        let junk_pages = batch + 8;
        let num_phys = total_pages + junk_pages;
        let table_stride = *pages_per.iter().max().unwrap();

        let mut table = vec![-1i32; batch * table_stride];
        match table_kind {
            TableKind::Discont => {
                let perm = lcg_perm(num_phys, rng.next_u64());
                let mut slot = 0usize;
                for b in 0..batch {
                    for j in 0..pages_per[b] {
                        table[b * table_stride + j] = perm[slot] as i32;
                        slot += 1;
                    }
                }
            }
            TableKind::Rotated => {
                let mut base = 0usize;
                for b in 0..batch {
                    let n = pages_per[b].max(1);
                    let rot = 1 + (rng.next_u64() as usize % n);
                    for j in 0..pages_per[b] {
                        table[b * table_stride + j] = (base + (j + rot) % n) as i32;
                    }
                    base += n + 1;
                }
                assert!(base <= num_phys);
            }
        }

        // Junk everywhere first, so a missing page-table walk reads garbage.
        let page_elems = PAGE * HK * D;
        let mut k_pool = vec![0u8; num_phys * page_elems];
        let mut v_pool = vec![0u8; num_phys * page_elems];
        for x in k_pool.iter_mut().chain(v_pool.iter_mut()) {
            *x = (rng.next_u64() & 0xff) as u8;
        }
        let mut k_scales = vec![0f32; num_phys * PAGE * HK];
        let mut v_scales = vec![0f32; num_phys * PAGE * HK];
        for s in k_scales.iter_mut().chain(v_scales.iter_mut()) {
            *s = 5e-2 + rng.unit() * 0.2;
        }

        // Mapped content: normal*0.3, per-(token, hk) scale = absmax/qmax.
        for b in 0..batch {
            for t in 0..kvlens[b] {
                let phys = table[b * table_stride + t / PAGE] as usize;
                let off = t % PAGE;
                for hk in 0..HK {
                    for (pool, scales) in
                        [(&mut k_pool, &mut k_scales), (&mut v_pool, &mut v_scales)]
                    {
                        let vals: Vec<f32> = (0..D).map(|_| rng.normal() * 0.3).collect();
                        let amax = vals.iter().fold(0f32, |m, x| m.max(x.abs()));
                        let scale = (amax / qmax).max(1e-6);
                        scales[(phys * PAGE + off) * HK + hk] = scale;
                        for d in 0..D {
                            pool[(phys * PAGE + off) * HK * D + hk * D + d] = if is_fp8 {
                                e4m3_encode(vals[d] / scale)
                            } else {
                                i8_encode(vals[d] / scale)
                            };
                        }
                    }
                }
            }
        }

        let total_q = qlens.iter().sum::<usize>();
        let mut cu_q = vec![0i32];
        for q in &qlens {
            let last = *cu_q.last().unwrap();
            cu_q.push(last + *q as i32);
        }
        let q_packed: Vec<bf16> = (0..total_q * H * D)
            .map(|_| bf(rng.normal() * 0.3))
            .collect();

        Case {
            label,
            is_fp8,
            table_kind,
            splits,
            kvlens,
            cu_q,
            table,
            table_stride,
            q_packed,
            k_pool,
            v_pool,
            k_scales,
            v_scales,
        }
    }

    // ── f64 oracle ──────────────────────────────────────────────────────────

    /// Bottom-right causal bound of query token `t` in row `b`; a decode row
    /// (qlen 1) sees the full kv extent (the shim demotes at qlen 1).
    fn row_lim(case: &Case, b: usize, t: usize) -> usize {
        let qlen = (case.cu_q[b + 1] - case.cu_q[b]) as usize;
        let kv_len = case.kvlens[b];
        if qlen > 1 {
            kv_len - qlen + 1 + t
        } else {
            kv_len
        }
    }

    fn attention_head(case: &Case, token: usize, h: usize) -> Vec<f64> {
        let b = case.row_of_token(token);
        let t = token - case.cu_q[b] as usize;
        let hk = h / G;
        let lim = row_lim(case, b, t);
        let q: Vec<f64> = (0..D)
            .map(|d| f64::from(case.q_packed[(token * H + h) * D + d].to_f32()))
            .collect();
        // Kernel math: S columns carry the per-token K scale times sm_scale,
        // probabilities carry the per-token V scale.
        let mut k = vec![0f64; lim * D];
        let mut v = vec![0f64; lim * D];
        for j in 0..lim {
            let ks = case.scale(false, b, j, hk) * SM_SCALE;
            let vs = case.scale(true, b, j, hk);
            for d in 0..D {
                k[j * D + d] = case.byte_value(false, b, j, hk, d) * ks;
                v[j * D + d] = case.byte_value(true, b, j, hk, d) * vs;
            }
        }
        attention_row(&q, &k, &v, lim, D, 1.0)
    }

    // ── device run ──────────────────────────────────────────────────────────

    fn run_case(ctx: &DeviceContext, case: &Case, corrupt: bool) -> Result<bool> {
        use cuda_kernels::ffi;
        let batch = case.batch();
        let total_q = case.total_q();
        let max_q = case.max_qlen();

        let q_d = ctx.stream.clone_htod(&case.q_packed)?;
        let mut o_d = ctx.stream.alloc_zeros::<bf16>(total_q * H * D)?;
        let k_d = ctx.stream.clone_htod(&case.k_pool)?;
        let v_d = ctx.stream.clone_htod(&case.v_pool)?;
        let ks_d = ctx.stream.clone_htod(&case.k_scales)?;
        let vs_d = ctx.stream.clone_htod(&case.v_scales)?;
        let cu_d = ctx.stream.clone_htod(&case.cu_q)?;
        let seqused: Vec<i32> = case.kvlens.iter().map(|&l| l as i32).collect();
        let seq_d = ctx.stream.clone_htod(&seqused)?;
        let tab_d = ctx.stream.clone_htod(&case.table)?;

        let ws_bytes = cuda_kernels::kv_quant::paged_attention_quantized_fa3_workspace_bytes(
            total_q,
            H,
            D,
            case.splits,
        );
        let ws_d = ctx.stream.alloc_zeros::<u8>(ws_bytes)?;

        let (qp, _) = q_d.device_ptr(&ctx.stream);
        let (op, _) = o_d.device_ptr_mut(&ctx.stream);
        let (kp, _) = k_d.device_ptr(&ctx.stream);
        let (vp, _) = v_d.device_ptr(&ctx.stream);
        let (ksp, _) = ks_d.device_ptr(&ctx.stream);
        let (vsp, _) = vs_d.device_ptr(&ctx.stream);
        let (cup, _) = cu_d.device_ptr(&ctx.stream);
        let (seqp, _) = seq_d.device_ptr(&ctx.stream);
        let (tabp, _) = tab_d.device_ptr(&ctx.stream);
        let (wsp, _) = ws_d.device_ptr(&ctx.stream);

        // SAFETY: every pointer is a live stream-ordered allocation sized to
        // the dims passed; the wrapper forwards the kernel's pointer contract.
        unsafe {
            ffi::attention::paged_attention_quantized_fa3_cuda(
                qp as *const ffi::Half,
                kp as *const u8,
                vp as *const u8,
                ksp as *const f32,
                vsp as *const f32,
                tabp as *const i32,
                cup as *const i32,
                seqp as *const i32,
                op as *mut ffi::Half,
                H as i32,
                HK as i32,
                D as i32,
                PAGE as i32,
                case.table_stride as i32,
                batch as i32,
                total_q as i32,
                max_q as i32,
                SM_SCALE as f32,
                case.is_fp8,
                case.splits as i32,
                ctx.stream.cu_stream(),
                wsp as *mut u8,
                ws_bytes,
            )
        }
        .result()
        .map_err(|e| anyhow::anyhow!("paged_attention_quantized_fa3_cuda failed: {e}"))?;
        ctx.sync()?;
        let got = ctx
            .stream
            .clone_dtoh(&o_d)
            .map_err(|e| anyhow::anyhow!("clone_dtoh failed: {e}"))?;

        // All cases are short (decode / <=8-token verify), so every query row
        // is checked; K/V are rebuilt per (row, kv head) and shared by the G
        // group implicitly here (each head reads the same bytes).
        let rows: Vec<(usize, usize)> = (0..total_q)
            .flat_map(|t| (0..H).map(move |h| (t, h)))
            .collect();
        let mut wants: Vec<Vec<f64>> = vec![Vec::new(); rows.len()];
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (i, (token, h)) in rows.iter().enumerate() {
                let case_ref = &*case;
                handles.push(scope.spawn(move || (i, attention_head(case_ref, *token, *h))));
            }
            for h in handles {
                let (i, w) = h.join().unwrap();
                wants[i] = w;
            }
        });

        let m = compare_rows(&got, &wants, rows.len(), D, H, &rows, &TOL, corrupt);
        // Clean: global metrics. Corrupt: the tooth is measured on the
        // corrupted row alone so additional checked rows/heads cannot dilute
        // the violation fraction below max_viol_frac.
        let pass = if corrupt {
            m.corrupted_row_fails(&TOL)
        } else {
            metrics_pass(&m, &TOL)
        };
        eprintln!(
            "[{} {} B={} total_q={} kv={:?} table={:?} splits={}] rel_l2={:.2e} \
             viol_frac={:.2e} max_dev={:.2e} {}",
            case.label,
            if case.is_fp8 { "fp8" } else { "int8" },
            batch,
            total_q,
            case.kvlens,
            case.table_kind,
            case.splits,
            m.rel_l2,
            m.viol_frac,
            m.max_dev,
            if pass { "PASS" } else { "FAIL" }
        );
        Ok(pass)
    }

    pub(super) fn run(negative: Option<Option<Family>>) -> Result<()> {
        let ctx = DeviceContext::new()?;
        eprintln!(
            "[paged-quant-attn-parity] device={} build={}{}",
            ctx.ordinal(),
            cuda_kernels::KERNEL_BUILD_ID,
            if negative.is_some() {
                " NEGATIVE-CONTROL"
            } else {
                ""
            }
        );

        // Mixed decode KV lengths: kv_len 1 and page boundaries ±1.
        let decode_kv = [1usize, 15, 16, 17, 127, 128, 129, 257];

        let cases = vec![
            // INT8. The B=8 decode rows run at BOTH split configs: the
            // single-CTA accumulation and the split partial+merge are
            // distinct code, so each gets its own negative tooth.
            build_case(
                "B1 decode kv33",
                false,
                TableKind::Discont,
                1,
                vec![33],
                vec![1],
                1,
            ),
            build_case(
                "B8 decode mixed-kv splits1",
                false,
                TableKind::Rotated,
                1,
                decode_kv.to_vec(),
                vec![1; 8],
                2,
            ),
            build_case(
                "B8 decode mixed-kv splits8",
                false,
                TableKind::Discont,
                8,
                decode_kv.to_vec(),
                vec![1; 8],
                3,
            ),
            build_case(
                "B8 mixed-q causal",
                false,
                TableKind::Discont,
                8,
                decode_kv.to_vec(),
                vec![1, 2, 1, 4, 1, 8, 2, 1],
                4,
            ),
            // FP8: same split-1/split-8 B=8 decode pair, plus verify rows.
            build_case(
                "B1 decode kv33",
                true,
                TableKind::Rotated,
                8,
                vec![33],
                vec![1],
                5,
            ),
            build_case(
                "B8 decode mixed-kv splits1",
                true,
                TableKind::Discont,
                1,
                decode_kv.to_vec(),
                vec![1; 8],
                6,
            ),
            build_case(
                "B8 decode mixed-kv splits8",
                true,
                TableKind::Rotated,
                8,
                decode_kv.to_vec(),
                vec![1; 8],
                7,
            ),
            build_case(
                "B8 mixed-q causal",
                true,
                TableKind::Rotated,
                16,
                decode_kv.to_vec(),
                // Row i must satisfy kv_len >= qlen (bottom-right causal);
                // the qlen-8 row pairs with kv=15.
                vec![1, 8, 1, 4, 1, 2, 3, 1],
                8,
            ),
        ];

        // Teeth are (pool, split configuration) pairs: the split-merge path
        // is separate code from single-split, so each combination must fail
        // independently under corruption.
        #[derive(Clone, Copy, PartialEq, Eq)]
        struct Tooth {
            fam: Family,
            splits1: bool,
            seen: bool,
            ok: bool,
        }
        let mut teeth = [
            Tooth {
                fam: Family::Int8,
                splits1: true,
                seen: false,
                ok: true,
            },
            Tooth {
                fam: Family::Int8,
                splits1: false,
                seen: false,
                ok: true,
            },
            Tooth {
                fam: Family::Fp8,
                splits1: true,
                seen: false,
                ok: true,
            },
            Tooth {
                fam: Family::Fp8,
                splits1: false,
                seen: false,
                ok: true,
            },
        ];
        for case in &cases {
            // Mixed-q verify rows are splits=8/16, so they belong to the
            // merge tooth; only the B=8 decode rows carry splits1.
            let splits1 = case.splits == 1;
            let corrupt = match negative {
                None => false,
                Some(None) => true,
                Some(Some(fam)) => fam == case.family(),
            };
            let pass = run_case(&ctx, case, corrupt)?;
            for t in &mut teeth {
                if t.fam == case.family() && t.splits1 == splits1 {
                    t.seen = true;
                    t.ok &= pass;
                }
            }
        }

        let name = |f: Family| if f == Family::Int8 { "int8" } else { "fp8" };
        let split_name = |s1: bool| if s1 { "num_splits=1" } else { "split-merge" };
        match negative {
            None => {
                for t in &teeth {
                    ensure!(
                        t.seen && t.ok,
                        "paged_quant_attn_parity clean run FAILED for {} {}",
                        name(t.fam),
                        split_name(t.splits1)
                    );
                }
                eprintln!("[paged-quant-attn-parity] ALL PASS");
            }
            Some(maybe_fam) => {
                for t in &teeth {
                    let targeted = maybe_fam.is_none_or(|f| f == t.fam);
                    ensure!(
                        t.seen,
                        "missing case for {} {}",
                        name(t.fam),
                        split_name(t.splits1)
                    );
                    if targeted {
                        ensure!(
                            !t.ok,
                            "paged_quant_attn_parity negative control did NOT fail the {} {} comparator",
                            name(t.fam),
                            split_name(t.splits1)
                        );
                    } else {
                        ensure!(
                            t.ok,
                            "paged_quant_attn_parity collateral failure in {} {}",
                            name(t.fam),
                            split_name(t.splits1)
                        );
                    }
                }
                eprintln!("[paged-quant-attn-parity] NEGATIVE CONTROL OK");
            }
        }
        Ok(())
    }
}
