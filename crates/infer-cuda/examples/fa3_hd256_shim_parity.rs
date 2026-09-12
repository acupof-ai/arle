//! FA3 hd256 paged-shim numeric-parity gate.
//!
//! Standalone kernel harness — NO engine, NO serve. Covers the vendored FA3
//! sm90 forward shims the Qwen3.6 paged attention path calls
//! (`crates/cuda-kernels/csrc/attention/arle_fa3_shim.cu`):
//!
//! - `arle_fa3_fwd_hd256_bf16_cuda` over a paged BF16 HND pool: decode rows
//!   (split-KV + PackGQA + combine) and causal varlen prefill rows,
//! - `arle_fa3_fwd_hd256_quant_cuda` over the production 1-byte pools: FP8
//!   e4m3 and INT8 with per-(token, kv_head) f32 scales. Decode-length rows
//!   take the shim's dequant-to-bf16 form; one long case (qlen 256, kv_len
//!   65537) forces the requantized-e4m3 form with one descale per
//!   (batch row, kv_head).
//!
//! Geometry is Qwen3.6-27B full attention: 24 query heads / 4 KV heads
//! (GQA ratio 6), head_dim 256, 16 full-attention layers — from the
//! Qwen3.6-27B-FP8 and ThinkingCap-27B host `config.json` (read on the H20
//! box); page_size 16 is the shim's non-TMA paged lane
//! (`arle_fa3_shim.cu`, "qwen35's page_size (16)").
//!
//! The four shim responsibilities named in the plan get independent coverage:
//! - page-table translation: every case uses a non-identity table, either
//!   DISCONTINUOUS (an LCG permutation scatters a row's logical pages across
//!   a pool that holds junk pages between them) or ROTATED (per-row contiguous
//!   run addressed through a rotated window that wraps inside the run, with
//!   junk pages between rows);
//! - KV dequant scale plumbing: scales vary per (token, kv_head); the oracle
//!   applies exactly the shim's chains — bf16 form `bf16(decode(byte)*s)`
//!   (Q untouched), fp8 form a second e4m3 requant of Q/K/V under the device
//!   kernels' per-(row, kv_head) descales;
//! - GQA head mapping: the oracle maps q head h to kv head h/6 on all 24;
//! - varlen offsets: B=8 batches pack mixed q lens through `cu_seqlens_q` and
//!   mixed per-row KV extents through `seqused_k`, including kv_len 1 and the
//!   page boundaries 15/16/17, 127/128/129, 257.
//!
//! The requantized-e4m3 case gets TWO oracles. The mirrored oracle models the
//! device chain (descale -> e4m3 round) exactly; it cannot see a descale that
//! saturates or loses range, because kernel and oracle requantize identically.
//! The unmirrored anchor is f64 attention over the raw dequantized KV
//! (`e4m3_decode(byte) * scale`) and the unrequantized bf16 Q, compared under
//! a band widened for one extra e4m3 round of Q/K/V.
//!
//! The oracles are f64 softmax written from the kernel contract, independent
//! of the CUDA sources. Every output column is compared on sampled (long case)
//! or all (short cases) query rows, L2-relative plus elementwise floor+slope;
//! the bound comes from the KV quantization error and the single final bf16
//! rounding against FA3's f32 accumulation.
//!
//! Negative control is per FORM family — bf16, int8, fp8-dequant (the decode
//! dequant-to-bf16 form), fp8-requant (the q256/kv65537 form):
//! `--negative-control=<family>` corrupts one family and the other three MUST
//! still pass; the bare flag corrupts all four.
//!
//! Run on a pod (sm_90):
//!   cargo build --release -p infer-cuda --features cuda \
//!     --example fa3_hd256_shim_parity
//!   target/release/examples/fa3_hd256_shim_parity --kernel-build-id
//!   INFER_CUDA_DEVICE=<free-gpu> target/release/examples/fa3_hd256_shim_parity
//!   INFER_CUDA_DEVICE=<free-gpu> \
//!     target/release/examples/fa3_hd256_shim_parity --negative-control
//!   ... --negative-control=bf16|int8|fp8-dequant|fp8-requant

#[cfg(feature = "cuda")]
#[path = "support/attn_common.rs"]
mod attn_common;

#[allow(dead_code)] // shared harness; each gate uses only the subset it needs
#[path = "support/parity_common.rs"]
mod parity_common;

fn main() -> anyhow::Result<()> {
    use parity_common::Parsed;
    match parity_common::cli() {
        Parsed::BuildIdPrinted => Ok(()),
        Parsed::Run(cli) => {
            let sel = match cli.negative_value.as_deref() {
                Some("bf16") => Family::Bf16,
                Some("int8") => Family::Int8,
                Some("fp8-dequant") => Family::Fp8Dequant,
                Some("fp8-requant") => Family::Fp8Requant,
                Some(other) => anyhow::bail!("unknown --negative-control family {other:?}"),
                None => Family::Bf16,
            };
            // Bare flag corrupts every family; `=form` selects just that one.
            let negative = if cli.negative {
                Some((cli.negative_value.is_some()).then_some(sel))
            } else {
                None
            };
            real::run(negative)
        }
    }
}

/// Negative-control families are the four shim FORMS. The fp8 pool has two
/// forms (decode dequant-to-bf16 and the long-context requantized-e4m3 form),
/// so pool-level grouping would let a dead requant comparator hide behind the
/// decode case.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Family {
    Bf16,
    Int8,
    Fp8Dequant,
    Fp8Requant,
}

#[cfg(not(feature = "cuda"))]
mod real {
    use super::Family;

    pub(super) fn run(_negative: Option<Option<Family>>) -> anyhow::Result<()> {
        eprintln!("fa3_hd256_shim_parity is a CUDA harness; rebuild with --features cuda.");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use super::Family;
    use super::attn_common::{self, Rng, Tol, bf, e4m3_decode, e4m3_encode, i8_encode, lcg_perm};
    use anyhow::{Result, ensure};
    use cuda_kernels::attention::{fa3_fwd_hd256_bf16, fa3_fwd_hd256_quant};
    use cuda_kernels::ffi;
    use cuda_kernels::prelude::DeviceContext;
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

    // Qwen3.6-27B full-attention geometry.
    const H: usize = 24;
    const HK: usize = 4;
    const G: usize = H / HK;
    const D: usize = 256;
    const PAGE: usize = 16;
    const SM_SCALE: f64 = 1.0 / 16.0; // 1/sqrt(256)
    const SEED: u64 = 0xfa32_568d_2560_1a23;
    const E4M3_MAX: f32 = 448.0;
    const I8_MAX: f32 = 127.0;

    // FA3 accumulates S/P/O in f32 and rounds the output to bf16 once. The
    // gap to the f64 oracle is the KV quantization error plus that rounding:
    // bf16 pool has no KV error; the bf16 form carries one 1-byte round
    // (e4m3 ~3%/elem, int8 ~0.4%/elem, averaged over the softmax weights);
    // the fp8 form carries a second e4m3 round, hence the widest band.
    const TOL_BF16_POOL: Tol = Tol {
        rel_l2: 6e-2,
        slope: 5e-2,
        floor: 3e-2,
        max_viol_frac: 2e-3,
    };
    const TOL_INT8_FORM: Tol = Tol {
        rel_l2: 7e-2,
        slope: 6e-2,
        floor: 3e-2,
        max_viol_frac: 2e-3,
    };
    const TOL_FP8_BF16_FORM: Tol = Tol {
        rel_l2: 1.0e-1,
        slope: 8e-2,
        floor: 4e-2,
        max_viol_frac: 5e-3,
    };
    const TOL_FP8_FORM: Tol = Tol {
        rel_l2: 1.4e-1,
        slope: 1.1e-1,
        floor: 4e-2,
        max_viol_frac: 5e-3,
    };
    // Unmirrored anchor for the requant form: f64 attention over the raw
    // dequantized KV and the unrequantized Q. The gap is the ORIGINAL pool
    // e4m3 round plus one extra e4m3 round each of Q/K/V and the bf16 output,
    // so the band is wider than the mirrored comparator's.
    const TOL_FP8_ANCHOR: Tol = Tol {
        rel_l2: 2.2e-1,
        slope: 1.6e-1,
        floor: 5e-2,
        max_viol_frac: 5e-3,
    };

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum TableKind {
        Discont,
        Rotated,
    }

    /// What the shim's math units actually consume after the input chain.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Form {
        /// BF16 HND paged pool, bf16 FA3 kernel.
        Bf16Pool,
        /// 1-byte pool -> shim dequant temp rounded to bf16 -> bf16 kernel.
        QuantBf16 { is_fp8: bool },
        /// e4m3 pool -> per-row descale requant to e4m3 (Q too) -> fp8 kernel.
        QuantFp8,
    }

    impl Form {
        fn family(self) -> Family {
            match self {
                Form::Bf16Pool => Family::Bf16,
                Form::QuantBf16 { is_fp8: true } => Family::Fp8Dequant,
                Form::QuantBf16 { is_fp8: false } => Family::Int8,
                Form::QuantFp8 => Family::Fp8Requant,
            }
        }
        fn tol(self) -> &'static Tol {
            match self {
                Form::Bf16Pool => &TOL_BF16_POOL,
                Form::QuantBf16 { is_fp8: false } => &TOL_INT8_FORM,
                Form::QuantBf16 { is_fp8: true } => &TOL_FP8_BF16_FORM,
                Form::QuantFp8 => &TOL_FP8_FORM,
            }
        }
    }

    /// One built case: durable host pools, rectangular page table, and the
    /// per-row descales the fp8 form's device kernels derive.
    struct Case {
        label: String,
        form: Form,
        table_kind: TableKind,
        is_causal: bool,
        splits: i32,
        kvlens: Vec<usize>,
        cu_q: Vec<i32>,
        /// `[batch, table_stride]`, -1 in tail slots.
        table: Vec<i32>,
        table_stride: usize,
        num_phys: usize,
        q_packed: Vec<bf16>,
        /// HND bf16 `[phys, hk, off, d]`, K and V pools separate.
        k_bf: Vec<bf16>,
        v_bf: Vec<bf16>,
        /// NHD 1-byte `[phys, off, hk, d]`, K and V pools separate.
        k_q: Vec<u8>,
        v_q: Vec<u8>,
        /// Per-(physical token, kv_head) scales.
        k_scales: Vec<f32>,
        v_scales: Vec<f32>,
        /// fp8 form only: f32-value descales `[batch, hk]` (device formula).
        descale_q: Vec<f32>,
        descale_k: Vec<f32>,
        descale_v: Vec<f32>,
        checked_tokens: Vec<usize>,
    }

    impl Case {
        fn batch(&self) -> usize {
            self.kvlens.len()
        }
        fn total_q(&self) -> usize {
            *self.cu_q.last().unwrap() as usize
        }
        fn qlen_of(&self, b: usize) -> usize {
            (self.cu_q[b + 1] - self.cu_q[b]) as usize
        }
        fn family(&self) -> Family {
            self.form.family()
        }
        fn row_of_token(&self, t: usize) -> usize {
            self.cu_q.partition_point(|&e| (e as usize) <= t) - 1
        }

        /// Logical K/V value (`is_v`) the FA3 math units consume at row
        /// `b`, token `t`, kv head `hk`, dim `d`.
        fn kv_value(&self, is_v: bool, b: usize, t: usize, hk: usize, d: usize) -> f64 {
            let phys = self.table[b * self.table_stride + t / PAGE] as usize;
            let off = t % PAGE;
            match self.form {
                Form::Bf16Pool => {
                    let pool = if is_v { &self.v_bf } else { &self.k_bf };
                    let idx = (phys * HK * PAGE + hk * PAGE + off) * D + d;
                    f64::from(pool[idx].to_f32())
                }
                Form::QuantBf16 { is_fp8 } => {
                    let pool = if is_v { &self.v_q } else { &self.k_q };
                    let scales = if is_v { &self.v_scales } else { &self.k_scales };
                    let byte = pool[(phys * PAGE + off) * HK * D + hk * D + d];
                    let raw = if is_fp8 {
                        f64::from(e4m3_decode(byte))
                    } else {
                        f64::from(byte as i8)
                    };
                    // dequantize_paged_kv.cu: f32 multiply, then one bf16 store.
                    let v = (raw as f32) * scales[(phys * PAGE + off) * HK + hk];
                    f64::from(bf(v).to_f32())
                }
                Form::QuantFp8 => {
                    let pool = if is_v { &self.v_q } else { &self.k_q };
                    let scales = if is_v { &self.v_scales } else { &self.k_scales };
                    let descales = if is_v {
                        &self.descale_v
                    } else {
                        &self.descale_k
                    };
                    let byte = pool[(phys * PAGE + off) * HK * D + hk * D + d];
                    // requant kernel: raw * (scale / descale) in f32, e4m3 RN;
                    // FA3 then multiplies by the descale.
                    let v0 = e4m3_decode(byte)
                        * (scales[(phys * PAGE + off) * HK + hk] / descales[b * HK + hk]);
                    f64::from(e4m3_decode(e4m3_encode(v0))) * f64::from(descales[b * HK + hk])
                }
            }
        }

        /// Raw dequantized K/V value: pool byte decoded and multiplied by its
        /// per-token scale, with NO second e4m3 round and NO bf16 round. This
        /// is the unmirrored anchor's KV for the requant form: it sees descale
        /// saturation or range loss that the mirrored chain hides.
        fn kv_value_raw(&self, is_v: bool, b: usize, t: usize, hk: usize, d: usize) -> f64 {
            let phys = self.table[b * self.table_stride + t / PAGE] as usize;
            let off = t % PAGE;
            let pool = if is_v { &self.v_q } else { &self.k_q };
            let scales = if is_v { &self.v_scales } else { &self.k_scales };
            let byte = pool[(phys * PAGE + off) * HK * D + hk * D + d];
            f64::from(e4m3_decode(byte)) * f64::from(scales[(phys * PAGE + off) * HK + hk])
        }

        /// Unrequantized Q value (the bf16 Q the device q_quant kernel reads).
        fn q_value_raw(&self, t: usize, h: usize, d: usize) -> f64 {
            f64::from(self.q_packed[(t * H + h) * D + d].to_f32())
        }

        /// Logical Q value the FA3 math units consume at global token `t`.
        fn q_value(&self, t: usize, h: usize, d: usize) -> f64 {
            let v = f64::from(self.q_packed[(t * H + h) * D + d].to_f32());
            if self.form == Form::QuantFp8 {
                let b = self.row_of_token(t);
                let hk = h / G;
                let descale = self.descale_q[b * HK + hk];
                // q_quant_kernel does the divide in f32.
                let v0 = self.q_packed[(t * H + h) * D + d].to_f32() / descale;
                f64::from(e4m3_decode(e4m3_encode(v0))) * f64::from(descale)
            } else {
                v
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_case(
        label: &str,
        form: Form,
        table_kind: TableKind,
        is_causal: bool,
        splits: i32,
        kvlens: Vec<usize>,
        qlens: Vec<usize>,
        case_seed: u64,
    ) -> Case {
        let batch = kvlens.len();
        assert_eq!(qlens.len(), batch);
        let mut rng = Rng::new(
            SEED ^ ((form.family() as u64) << 40)
                ^ ((table_kind as usize as u64) << 36)
                ^ ((kvlens.iter().sum::<usize>() as u64) << 8)
                ^ ((qlens[0] as u64) << 20)
                ^ case_seed,
        );
        let pages_per: Vec<usize> = kvlens.iter().map(|l| l.div_ceil(PAGE)).collect();
        let total_pages: usize = pages_per.iter().sum();
        let junk_pages = batch + 8;
        let num_phys = total_pages + junk_pages;
        let table_stride = *pages_per.iter().max().unwrap();

        // ── page table ──────────────────────────────────────────────────────
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

        let page_elems = PAGE * HK * D;
        let is_fp8 = matches!(form, Form::QuantBf16 { is_fp8: true } | Form::QuantFp8);
        let qmax = if is_fp8 { E4M3_MAX } else { I8_MAX };

        // Pools start as junk: bytes/bf16 values in EVERY physical page, so a
        // missing page-table walk reads visibly wrong data.
        let junk = |rng: &mut Rng| -> f32 { rng.normal() * 0.5 };
        let mut k_bf: Vec<bf16> = Vec::new();
        let mut v_bf: Vec<bf16> = Vec::new();
        let mut k_q = vec![0u8; num_phys * page_elems];
        let mut v_q = vec![0u8; num_phys * page_elems];
        for _ in 0..num_phys * page_elems {
            k_bf.push(bf(junk(&mut rng)));
            v_bf.push(bf(junk(&mut rng)));
        }
        for x in k_q.iter_mut().chain(v_q.iter_mut()) {
            *x = (rng.next_u64() & 0xff) as u8;
        }
        let mut k_scales = vec![0f32; num_phys * PAGE * HK];
        let mut v_scales = vec![0f32; num_phys * PAGE * HK];
        for s in k_scales.iter_mut().chain(v_scales.iter_mut()) {
            *s = 5e-2 + rng.unit() * 0.2;
        }

        // ── mapped logical content: normal*0.3, scale = token absmax/qmax ──
        for b in 0..batch {
            for t in 0..kvlens[b] {
                let phys = table[b * table_stride + t / PAGE] as usize;
                let off = t % PAGE;
                for hk in 0..HK {
                    let mut kvals = vec![0f32; D];
                    let mut vvals = vec![0f32; D];
                    for d in 0..D {
                        kvals[d] = rng.normal() * 0.3;
                        vvals[d] = rng.normal() * 0.3;
                    }
                    let kamax = kvals.iter().fold(0f32, |m, x| m.max(x.abs()));
                    let vamax = vvals.iter().fold(0f32, |m, x| m.max(x.abs()));
                    let sk = (kamax / qmax).max(1e-6);
                    let sv = (vamax / qmax).max(1e-6);
                    k_scales[(phys * PAGE + off) * HK + hk] = sk;
                    v_scales[(phys * PAGE + off) * HK + hk] = sv;
                    for d in 0..D {
                        match form {
                            Form::Bf16Pool => {
                                k_bf[(phys * HK * PAGE + hk * PAGE + off) * D + d] = bf(kvals[d]);
                                v_bf[(phys * HK * PAGE + hk * PAGE + off) * D + d] = bf(vvals[d]);
                            }
                            Form::QuantBf16 { .. } | Form::QuantFp8 => {
                                k_q[(phys * PAGE + off) * HK * D + hk * D + d] = if is_fp8 {
                                    e4m3_encode(kvals[d] / sk)
                                } else {
                                    i8_encode(kvals[d] / sk)
                                };
                                v_q[(phys * PAGE + off) * HK * D + hk * D + d] = if is_fp8 {
                                    e4m3_encode(vvals[d] / sv)
                                } else {
                                    i8_encode(vvals[d] / sv)
                                };
                            }
                        }
                    }
                }
            }
        }

        // ── Q packed [total_q, H, D] ───────────────────────────────────────
        let total_q = qlens.iter().sum::<usize>();
        let mut cu_q = vec![0i32];
        for q in &qlens {
            let last = *cu_q.last().unwrap();
            cu_q.push(last + *q as i32);
        }
        let q_packed: Vec<bf16> = (0..total_q * H * D)
            .map(|_| bf(rng.normal() * 0.3))
            .collect();

        // ── fp8-form descales, mirroring dequant_paged_kv.cu: kv descale is
        //    f32 max of per-token scale * qmax/448 (qmax=448 for the e4m3
        //    source pool, so it equals the row's max scale), q descale is the
        //    f32 group absmax / 448. ────────────────────────────────────────
        let mut descale_q = vec![0f32; batch * HK];
        let mut descale_k = vec![0f32; batch * HK];
        let mut descale_v = vec![0f32; batch * HK];
        if form == Form::QuantFp8 {
            for b in 0..batch {
                for hk in 0..HK {
                    let mut mk = f32::MIN;
                    let mut mv = f32::MIN;
                    for t in 0..kvlens[b] {
                        let phys = table[b * table_stride + t / PAGE] as usize;
                        let off = t % PAGE;
                        mk = mk.max(k_scales[(phys * PAGE + off) * HK + hk]);
                        mv = mv.max(v_scales[(phys * PAGE + off) * HK + hk]);
                    }
                    descale_k[b * HK + hk] = (mk * qmax / E4M3_MAX).max(f32::MIN_POSITIVE);
                    descale_v[b * HK + hk] = (mv * qmax / E4M3_MAX).max(f32::MIN_POSITIVE);
                    let mut mq = 0f32;
                    for t in (cu_q[b] as usize)..(cu_q[b + 1] as usize) {
                        for h in hk * G..(hk + 1) * G {
                            for d in 0..D {
                                mq = mq.max(q_packed[(t * H + h) * D + d].to_f32().abs());
                            }
                        }
                    }
                    descale_q[b * HK + hk] = (mq / E4M3_MAX).max(f32::MIN_POSITIVE);
                }
            }
        }

        // Long cases sample the first and last query token (each group's
        // dequantized K/V matrices are ~268 MB/hk at 65537); short cases
        // check every row.
        let checked_tokens = if total_q <= 64 {
            (0..total_q).collect()
        } else {
            vec![0, total_q - 1]
        };

        Case {
            label: label.to_string(),
            form,
            table_kind,
            is_causal,
            splits,
            kvlens,
            cu_q,
            table,
            table_stride,
            num_phys,
            q_packed,
            k_bf,
            v_bf,
            k_q,
            v_q,
            k_scales,
            v_scales,
            descale_q,
            descale_k,
            descale_v,
            checked_tokens,
        }
    }

    // ── f64 oracle ──────────────────────────────────────────────────────────

    /// Dequantized K/V matrices for one checked row and one kv head; shared by
    /// the G query heads of its GQA group (builds once per group).
    struct RowKV {
        k: Vec<f64>, // [row_kv_len, D]
        v: Vec<f64>, // [row_kv_len, D]
    }

    /// `lim` (causal bound) is a property of (row, query token); same for all
    /// kv heads.
    fn row_lim(case: &Case, token: usize) -> usize {
        let b = case.row_of_token(token);
        let t = token - case.cu_q[b] as usize;
        let qlen = case.qlen_of(b);
        let kv_len = case.kvlens[b];
        if case.is_causal && qlen > 1 {
            kv_len - qlen + 1 + t
        } else {
            kv_len
        }
    }

    fn build_row_kv(case: &Case, b: usize, hk: usize, lim: usize) -> RowKV {
        let mut k = vec![0f64; lim * D];
        let mut v = vec![0f64; lim * D];
        for j in 0..lim {
            for d in 0..D {
                k[j * D + d] = case.kv_value(false, b, j, hk, d);
                v[j * D + d] = case.kv_value(true, b, j, hk, d);
            }
        }
        RowKV { k, v }
    }

    /// Unmirrored anchor matrices: raw dequantized values, no second round.
    fn build_row_kv_raw(case: &Case, b: usize, hk: usize, lim: usize) -> RowKV {
        let mut k = vec![0f64; lim * D];
        let mut v = vec![0f64; lim * D];
        for j in 0..lim {
            for d in 0..D {
                k[j * D + d] = case.kv_value_raw(false, b, j, hk, d);
                v[j * D + d] = case.kv_value_raw(true, b, j, hk, d);
            }
        }
        RowKV { k, v }
    }

    #[allow(clippy::needless_range_loop)]
    fn attention_with_kv(case: &Case, token: usize, h: usize, kv: &RowKV, lim: usize) -> Vec<f64> {
        attention_with_kv_q(case, token, h, kv, lim, false)
    }

    /// Anchor variant uses the raw (unrequantized) Q.
    fn attention_with_kv_raw_q(
        case: &Case,
        token: usize,
        h: usize,
        kv: &RowKV,
        lim: usize,
    ) -> Vec<f64> {
        attention_with_kv_q(case, token, h, kv, lim, true)
    }

    #[allow(clippy::needless_range_loop)]
    fn attention_with_kv_q(
        case: &Case,
        token: usize,
        h: usize,
        kv: &RowKV,
        lim: usize,
        raw_q: bool,
    ) -> Vec<f64> {
        let q: Vec<f64> = (0..D)
            .map(|d| {
                if raw_q {
                    case.q_value_raw(token, h, d)
                } else {
                    case.q_value(token, h, d)
                }
            })
            .collect();
        attn_common::attention_row(&q, &kv.k, &kv.v, lim, D, SM_SCALE)
    }

    /// Verdict of one comparator pair: per-comparator pass flags and metrics.
    struct CmpOutcome {
        mirror_ok: bool,
        anchor_ok: Option<bool>,
        /// Negative tooth measured on the single corrupted row alone, so the
        /// number of other checked rows/heads cannot dilute it.
        mirror_tooth_ok: bool,
        anchor_tooth_ok: Option<bool>,
        mirror: attn_common::Metrics,
        anchor: Option<attn_common::Metrics>,
    }

    /// All three evaluations of one case from a single oracle build: the clean
    /// run, a mirror-expectation corruption (anchor stays clean in the same
    /// evaluation), and an anchor-expectation corruption (mirror stays clean).
    /// Negative control uses the last two as independent teeth.
    struct CompareAll {
        clean: CmpOutcome,
        mirror_neg: CmpOutcome,
        anchor_neg: Option<CmpOutcome>,
    }

    fn eval_pair(
        case: &Case,
        got: &[bf16],
        rows: &[(usize, usize)],
        wants: &[Vec<f64>],
        wants_anchor: Option<&[Vec<f64>]>,
        corrupt_mirror: bool,
        corrupt_anchor: bool,
    ) -> CmpOutcome {
        let tol = case.form.tol();
        let m = attn_common::compare_rows(got, wants, rows.len(), D, H, rows, tol, corrupt_mirror);
        let mirror_ok = attn_common::metrics_pass(&m, tol);
        let Some(wants_anchor) = wants_anchor else {
            return CmpOutcome {
                mirror_ok,
                anchor_ok: None,
                mirror_tooth_ok: m.corrupted_row_fails(tol),
                anchor_tooth_ok: None,
                mirror: m,
                anchor: None,
            };
        };
        let ma = attn_common::compare_rows(
            got,
            wants_anchor,
            rows.len(),
            D,
            H,
            rows,
            &TOL_FP8_ANCHOR,
            corrupt_anchor,
        );
        let anchor_ok = attn_common::metrics_pass(&ma, &TOL_FP8_ANCHOR);
        CmpOutcome {
            mirror_ok,
            anchor_ok: Some(anchor_ok),
            mirror_tooth_ok: m.corrupted_row_fails(tol),
            anchor_tooth_ok: Some(ma.corrupted_row_fails(&TOL_FP8_ANCHOR)),
            mirror: m,
            anchor: Some(ma),
        }
    }

    fn compare(case: &Case, got: &[bf16]) -> CompareAll {
        let rows: Vec<(usize, usize)> = case
            .checked_tokens
            .iter()
            .flat_map(|&t| (0..H).map(move |h| (t, h)))
            .collect();

        // Build each checked ROW's K/V matrices once: two checked tokens in
        // one prefill row share the same KV, and the attention call slices its
        // own causal prefix. GQA-shared across the G query heads; parallelized
        // across scoped threads because the long case's matrices are large.
        let rows_checked: Vec<usize> = {
            let mut v: Vec<usize> = case
                .checked_tokens
                .iter()
                .map(|&t| case.row_of_token(t))
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        let build_matrices = |raw: bool| -> Vec<Vec<RowKV>> {
            let mut by_row: Vec<Vec<RowKV>> = (0..rows_checked.len())
                .map(|_| Vec::with_capacity(HK))
                .collect();
            let built = std::thread::scope(|scope| {
                let mut handles = Vec::new();
                for (ri, &b) in rows_checked.iter().enumerate() {
                    for hk in 0..HK {
                        let case_ref = &*case;
                        let lim = case.kvlens[b];
                        handles.push(scope.spawn(move || {
                            if raw {
                                (ri, hk, build_row_kv_raw(case_ref, b, hk, lim))
                            } else {
                                (ri, hk, build_row_kv(case_ref, b, hk, lim))
                            }
                        }));
                    }
                }
                let mut built: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
                built.sort_by_key(|(ri, hk, _)| (*ri, *hk));
                built
            });
            for (ri, hk, kv) in built {
                if hk == 0 {
                    by_row[ri] = Vec::with_capacity(HK);
                }
                by_row[ri].push(kv);
            }
            by_row
        };
        let kv_by_row = build_matrices(false);
        // The requant form gets a second, unmirrored anchor over the raw
        // dequantized KV and unrequantized Q.
        let kv_by_row_raw = if case.form == Form::QuantFp8 {
            build_matrices(true)
        } else {
            Vec::new()
        };

        let build_wants = |raw_q: bool, matrices: &[Vec<RowKV>]| -> Vec<Vec<f64>> {
            let mut wants: Vec<Vec<f64>> = (0..rows.len()).map(|_| Vec::new()).collect();
            std::thread::scope(|scope| {
                let mut handles = Vec::new();
                for (i, (token, h)) in rows.iter().enumerate() {
                    let b = case.row_of_token(*token);
                    let ri = rows_checked.iter().position(|&rb| rb == b).unwrap();
                    let hk = h / G;
                    let kv = &matrices[ri][hk];
                    let lim = row_lim(case, *token);
                    let case_ref = &*case;
                    handles.push(scope.spawn(move || {
                        if raw_q {
                            (i, attention_with_kv_raw_q(case_ref, *token, *h, kv, lim))
                        } else {
                            (i, attention_with_kv(case_ref, *token, *h, kv, lim))
                        }
                    }));
                }
                for handle in handles {
                    let (i, v) = handle.join().unwrap();
                    wants[i] = v;
                }
            });
            wants
        };
        let wants = build_wants(false, &kv_by_row);
        let wants_anchor = if case.form == Form::QuantFp8 {
            Some(build_wants(true, &kv_by_row_raw))
        } else {
            None
        };

        let clean = eval_pair(
            case,
            got,
            &rows,
            &wants,
            wants_anchor.as_deref(),
            false,
            false,
        );
        // The mirror corruption also lands in the anchor's first row, so for
        // the requant family evaluate the anchor separately with its own flag.
        let mirror_neg = eval_pair(
            case,
            got,
            &rows,
            &wants,
            wants_anchor.as_deref(),
            true,
            false,
        );
        let anchor_neg = wants_anchor
            .as_ref()
            .map(|wa| eval_pair(case, got, &rows, &wants, Some(wa), false, true));
        CompareAll {
            clean,
            mirror_neg,
            anchor_neg,
        }
    }

    // ── device run ──────────────────────────────────────────────────────────

    fn run_kernel(ctx: &DeviceContext, case: &Case) -> Result<Vec<bf16>> {
        let batch = case.batch();
        let total_q = case.total_q();
        let max_q = case
            .cu_q
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap();
        let max_kv = *case.kvlens.iter().max().unwrap();
        let page_elems = PAGE * HK * D;

        let q_d = ctx.stream.clone_htod(&case.q_packed)?;
        let mut o_d = ctx.stream.alloc_zeros::<bf16>(total_q * H * D)?;
        let cu_d = ctx.stream.clone_htod(&case.cu_q)?;
        let seqused_d: Vec<i32> = case.kvlens.iter().map(|&l| l as i32).collect();
        let seqused_d = ctx.stream.clone_htod(&seqused_d)?;
        let table_d = ctx.stream.clone_htod(&case.table)?;

        let mut lse_d = ctx.stream.alloc_zeros::<f32>(H * total_q)?;
        let meta_cap = (batch.div_ceil(4) * 4 * 4 + 1) as i32;
        let mut sem_d = ctx.stream.alloc_zeros::<i32>(meta_cap as usize)?;
        let (mut accum_d, mut lseaccum_d) = (None, None);
        if case.splits > 1 {
            accum_d = Some(
                ctx.stream
                    .alloc_zeros::<f32>(case.splits as usize * H * total_q * D)?,
            );
            lseaccum_d = Some(
                ctx.stream
                    .alloc_zeros::<f32>(case.splits as usize * H * total_q)?,
            );
        }

        let (q_ptr, _) = q_d.device_ptr(&ctx.stream);
        let (o_ptr, _) = o_d.device_ptr_mut(&ctx.stream);
        let (cu_ptr, _) = cu_d.device_ptr(&ctx.stream);
        let (seq_ptr, _) = seqused_d.device_ptr(&ctx.stream);
        let (tab_ptr, _) = table_d.device_ptr(&ctx.stream);
        let (lse_ptr, _) = lse_d.device_ptr_mut(&ctx.stream);
        let (sem_ptr, _) = sem_d.device_ptr_mut(&ctx.stream);
        let (accum_ptr, lseaccum_ptr) = if case.splits > 1 {
            let (a, _) = accum_d.as_mut().unwrap().device_ptr_mut(&ctx.stream);
            let (l, _) = lseaccum_d.as_mut().unwrap().device_ptr_mut(&ctx.stream);
            (a, l)
        } else {
            (0u64, 0u64)
        };

        let args = ffi::ArleFa3FwdHd256Args {
            q: q_ptr as *const ffi::Half,
            k: std::ptr::null(),
            v: std::ptr::null(),
            o: o_ptr as *mut ffi::Half,
            softmax_lse: lse_ptr as *mut f32,
            out_accum: if case.splits > 1 {
                accum_ptr as *mut f32
            } else {
                std::ptr::null_mut()
            },
            softmax_lse_accum: if case.splits > 1 {
                lseaccum_ptr as *mut f32
            } else {
                std::ptr::null_mut()
            },
            tile_count_semaphore: sem_ptr as *mut i32,
            metadata_capacity: meta_cap,
            cu_seqlens_q: cu_ptr as *const i32,
            seqused_k: seq_ptr as *const i32,
            batch: batch as i32,
            total_q: total_q as i32,
            seqlen_q: max_q as i32,
            seqlen_k: max_kv as i32,
            num_heads: H as i32,
            num_heads_k: HK as i32,
            head_dim: D as i32,
            q_row_stride: (H * D) as i64,
            // HND per-page pool [phys, hk, off, d].
            k_row_stride: D as i64,
            v_row_stride: D as i64,
            o_row_stride: (H * D) as i64,
            q_head_stride: D as i64,
            k_head_stride: (PAGE * D) as i64,
            v_head_stride: (PAGE * D) as i64,
            o_head_stride: D as i64,
            softmax_scale: SM_SCALE as f32,
            is_causal: if case.is_causal { 1 } else { 0 },
            num_splits: case.splits,
            page_table: tab_ptr as *const i32,
            page_table_batch_stride: case.table_stride as i64,
            page_size: PAGE as i32,
            num_pages: case.num_phys as i32,
            k_page_stride: page_elems as i64,
            v_page_stride: page_elems as i64,
        };

        match case.form {
            Form::Bf16Pool => {
                let k_d = ctx.stream.clone_htod(&case.k_bf)?;
                let v_d = ctx.stream.clone_htod(&case.v_bf)?;
                let (kp, _) = k_d.device_ptr(&ctx.stream);
                let (vp, _) = v_d.device_ptr(&ctx.stream);
                let mut args = args;
                args.k = kp as *const ffi::Half;
                args.v = vp as *const ffi::Half;
                fa3_fwd_hd256_bf16(&ctx.stream, &args)?;
            }
            Form::QuantBf16 { .. } | Form::QuantFp8 => {
                let is_fp8 = matches!(case.form, Form::QuantBf16 { is_fp8: true } | Form::QuantFp8);
                let kd_d = ctx.stream.clone_htod(&case.k_q)?;
                let vd_d = ctx.stream.clone_htod(&case.v_q)?;
                let ks_d = ctx.stream.clone_htod(&case.k_scales)?;
                let vs_d = ctx.stream.clone_htod(&case.v_scales)?;
                let (kp, _) = kd_d.device_ptr(&ctx.stream);
                let (vp, _) = vd_d.device_ptr(&ctx.stream);
                let (ksp, _) = ks_d.device_ptr(&ctx.stream);
                let (vsp, _) = vs_d.device_ptr(&ctx.stream);
                let quant_args = ffi::ArleFa3FwdHd256QuantArgs {
                    base: args,
                    k_data: kp as *const u8,
                    v_data: vp as *const u8,
                    k_scales: ksp as *const f32,
                    v_scales: vsp as *const f32,
                    is_fp8: if is_fp8 { 1 } else { 0 },
                };
                fa3_fwd_hd256_quant(&ctx.stream, &quant_args)?;
            }
        }
        ctx.sync()?;
        ctx.stream
            .clone_dtoh(&o_d)
            .map_err(|e| anyhow::anyhow!("clone_dtoh output failed: {e}"))
    }

    fn print_verdict(case: &Case, out: &CmpOutcome) {
        let mirror_pass = out.mirror_ok && out.anchor_ok.unwrap_or(true);
        if let Some(ma) = &out.anchor {
            eprintln!(
                "[{} B={} kv={:?} table={:?} splits={}] mirrored rel_l2={:.2e} viol={:.2e} \
                 max={:.2e} | unmirrored-anchor rel_l2={:.2e} viol={:.2e} max={:.2e} {}",
                case.label,
                case.batch(),
                case.kvlens,
                case.table_kind,
                case.splits,
                out.mirror.rel_l2,
                out.mirror.viol_frac,
                out.mirror.max_dev,
                ma.rel_l2,
                ma.viol_frac,
                ma.max_dev,
                if mirror_pass { "PASS" } else { "FAIL" }
            );
        } else {
            eprintln!(
                "[{} B={} kv={:?} table={:?} splits={}] rel_l2={:.2e} viol_frac={:.2e} \
                 max_dev={:.2e} {}",
                case.label,
                case.batch(),
                case.kvlens,
                case.table_kind,
                case.splits,
                out.mirror.rel_l2,
                out.mirror.viol_frac,
                out.mirror.max_dev,
                if mirror_pass { "PASS" } else { "FAIL" }
            );
        }
    }

    pub(super) fn run(negative: Option<Option<Family>>) -> Result<()> {
        let ctx = DeviceContext::new()?;
        // SAFETY: marker is an argumentless C ABI probe.
        let marker = unsafe { ffi::attention::arle_fa3_real_kernel_marker_cuda() };
        ensure!(
            marker == 1,
            "FA3 shim is the stub build (marker=0); gate needs sm_90"
        );
        eprintln!(
            "[fa3-hd256-shim-parity] device={} build={}{}",
            ctx.ordinal(),
            cuda_kernels::KERNEL_BUILD_ID,
            if negative.is_some() {
                " NEGATIVE-CONTROL"
            } else {
                ""
            }
        );

        // Mixed decode KV lengths: kv_len 1 and page boundaries ±1
        // (15/16/17, 127/128/129, 257 at stride 17).
        let decode_kv = [1usize, 15, 16, 17, 127, 128, 129, 257];

        let cases: Vec<Case> = vec![
            // BF16 pool: B=1 split decode, B=8 split decode, mixed-q causal
            // varlen, B=1 short prefill.
            build_case(
                "bf16 B1 decode kv33",
                Form::Bf16Pool,
                TableKind::Discont,
                true,
                8,
                vec![33],
                vec![1],
                1,
            ),
            build_case(
                "bf16 B8 decode mixed-kv",
                Form::Bf16Pool,
                TableKind::Rotated,
                true,
                8,
                decode_kv.to_vec(),
                vec![1; 8],
                2,
            ),
            build_case(
                "bf16 B8 mixed-q causal",
                Form::Bf16Pool,
                TableKind::Discont,
                true,
                8,
                decode_kv.to_vec(),
                vec![1, 2, 1, 3, 1, 1, 2, 1],
                3,
            ),
            build_case(
                "bf16 B1 q8 prefill kv41",
                Form::Bf16Pool,
                TableKind::Rotated,
                true,
                1,
                vec![41],
                vec![8],
                4,
            ),
            // FP8 pool: B=8 decode takes the dequant-bf16 form; the long row
            // (q256, kv 65537 = 64K+1) forces the requant-e4m3 form.
            build_case(
                "fp8 B8 decode mixed-kv",
                Form::QuantBf16 { is_fp8: true },
                TableKind::Discont,
                true,
                8,
                decode_kv.to_vec(),
                vec![1; 8],
                5,
            ),
            build_case(
                "fp8 B1 q256 kv65537",
                Form::QuantFp8,
                TableKind::Rotated,
                true,
                1,
                vec![65537],
                vec![256],
                6,
            ),
            // INT8 pool: B=8 decode dequant-bf16 form plus B=1 causal prefill.
            build_case(
                "int8 B8 decode mixed-kv",
                Form::QuantBf16 { is_fp8: false },
                TableKind::Rotated,
                true,
                8,
                decode_kv.to_vec(),
                vec![1; 8],
                7,
            ),
            build_case(
                "int8 B1 q8 prefill kv49",
                Form::QuantBf16 { is_fp8: false },
                TableKind::Discont,
                true,
                1,
                vec![49],
                vec![8],
                8,
            ),
        ];

        // Per-family state over that family's cases: the clean verdict plus
        // the negative teeth (the requant form's anchor has its own tooth).
        struct FamState {
            clean: bool,
            any_case: bool,
            mirror_tooth: bool,
            anchor_tooth: bool,
            has_anchor: bool,
        }
        let mut fam = std::array::from_fn::<FamState, 4, _>(|_| FamState {
            clean: true,
            any_case: false,
            mirror_tooth: true,
            anchor_tooth: true,
            has_anchor: false,
        });
        let fam_idx = |f: Family| match f {
            Family::Bf16 => 0,
            Family::Int8 => 1,
            Family::Fp8Dequant => 2,
            Family::Fp8Requant => 3,
        };

        let fam_name = |f: Family| -> &'static str {
            match f {
                Family::Bf16 => "bf16",
                Family::Int8 => "int8",
                Family::Fp8Dequant => "fp8-dequant",
                Family::Fp8Requant => "fp8-requant",
            }
        };

        for case in &cases {
            // One kernel launch and one oracle build per case; compare()
            // derives the clean verdict and both negative teeth from it.
            let got = run_kernel(&ctx, case)?;
            let out = compare(case, &got);
            let st = &mut fam[fam_idx(case.family())];
            st.any_case = true;
            st.has_anchor |= out.clean.anchor_ok.is_some();
            st.clean &= out.clean.mirror_ok && out.clean.anchor_ok.unwrap_or(true);
            // Mirror tooth: the corrupted mirror expectation MUST fail,
            // measured on the corrupted row itself (not the global violation
            // fraction, which more checked rows would dilute); the anchor
            // comparator, evaluated uncorrupted in the same pair, MUST still
            // pass for the requant family.
            st.mirror_tooth &=
                out.mirror_neg.mirror_tooth_ok && out.mirror_neg.anchor_ok.is_none_or(|ok| ok);
            if let Some(an) = &out.anchor_neg {
                // Anchor tooth: corrupted anchor MUST fail on its own row
                // while the uncorrupted mirror in that pair still passes.
                st.anchor_tooth &= an.mirror_ok && an.anchor_tooth_ok.unwrap_or(false);
            }
            print_verdict(case, &out.clean);
        }

        let families = [
            Family::Bf16,
            Family::Int8,
            Family::Fp8Dequant,
            Family::Fp8Requant,
        ];
        match negative {
            None => {
                for (i, f) in families.iter().enumerate() {
                    ensure!(
                        fam[i].any_case && fam[i].clean,
                        "fa3_hd256_shim_parity clean run FAILED for the {} family",
                        fam_name(*f)
                    );
                }
                eprintln!("[fa3-hd256-shim-parity] ALL PASS");
            }
            Some(maybe_fam) => {
                for (i, f) in families.iter().enumerate() {
                    let targeted = maybe_fam.is_none_or(|tf| tf == *f);
                    ensure!(fam[i].any_case, "family {} has no cases", fam_name(*f));
                    if !targeted {
                        // An untargeted family runs its CLEAN expectations even
                        // under a single-family negative flag.
                        ensure!(
                            fam[i].clean,
                            "fa3_hd256_shim_parity collateral failure in the {} family",
                            fam_name(*f)
                        );
                    } else {
                        ensure!(
                            fam[i].mirror_tooth,
                            "fa3_hd256_shim_parity negative control did NOT fail the {} mirror comparator",
                            fam_name(*f)
                        );
                        if fam[i].has_anchor {
                            ensure!(
                                fam[i].anchor_tooth,
                                "fa3_hd256_shim_parity negative control did NOT fail the {} unmirrored anchor",
                                fam_name(*f)
                            );
                        }
                    }
                }
                eprintln!("[fa3-hd256-shim-parity] NEGATIVE CONTROL OK");
            }
        }
        Ok(())
    }
}
