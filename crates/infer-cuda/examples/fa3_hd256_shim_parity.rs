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
//! (GQA ratio 6), head_dim 256, paged pool page_size 16
//! (`docs/research/2026-07-27-longctx-decode-attention-plan.md:44`,
//! `docs/experience/wins/2026-08-04-fa3-decode-splits-fill-the-sms.md:37`).
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
//! The oracle is f64 softmax written from the kernel contract, independent of
//! the CUDA sources. Every output column is compared on sampled (long case) or
//! all (short cases) query rows, L2-relative plus elementwise floor+slope; the
//! bound comes from the KV quantization error and the single final bf16
//! rounding against FA3's f32 accumulation.
//!
//! `--negative-control[=bf16|fp8|int8]` corrupts one path family's
//! expectations; that family MUST fail and the other two MUST still pass
//! (bare flag corrupts all three).
//!
//! Run on a pod (sm_90):
//!   INFER_CUDA_DEVICE=<free-gpu> target/release/examples/fa3_hd256_shim_parity
//!   INFER_CUDA_DEVICE=<free-gpu> target/release/examples/fa3_hd256_shim_parity --negative-control=fp8

fn main() -> anyhow::Result<()> {
    if std::env::args().nth(1).as_deref() == Some("--kernel-build-id") {
        println!("{}", cuda_kernels::KERNEL_BUILD_ID);
        return Ok(());
    }
    let mut negative: Option<Option<PoolSel>> = None;
    for arg in std::env::args().skip(1) {
        if arg == "--negative-control" {
            negative = Some(None);
        } else if let Some(v) = arg.strip_prefix("--negative-control=") {
            let sel = match v {
                "bf16" => PoolSel::Bf16,
                "fp8" => PoolSel::Fp8,
                "int8" => PoolSel::Int8,
                other => anyhow::bail!("unknown --negative-control family {other:?}"),
            };
            negative = Some(Some(sel));
        } else {
            anyhow::bail!("unknown argument {arg:?}");
        }
    }
    real::run(negative)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PoolSel {
    Bf16,
    Fp8,
    Int8,
}

#[cfg(not(feature = "cuda"))]
mod real {
    use super::PoolSel;

    pub(super) fn run(_negative: Option<Option<PoolSel>>) -> anyhow::Result<()> {
        eprintln!("fa3_hd256_shim_parity is a CUDA harness; rebuild with --features cuda.");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use super::PoolSel;
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
    struct Tol {
        rel_l2: f64,
        slope: f64,
        floor: f64,
        max_viol_frac: f64,
    }
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
        fn family(self) -> PoolSel {
            match self {
                Form::Bf16Pool => PoolSel::Bf16,
                Form::QuantBf16 { is_fp8: true } | Form::QuantFp8 => PoolSel::Fp8,
                Form::QuantBf16 { is_fp8: false } => PoolSel::Int8,
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
        let e2 = v.log2().floor() as i32;
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
            let sig = round_even(v / 2f64.powi(-9)).clamp(0.0, 7.0) as i32;
            sign | sig as u8
        }
    }

    fn e4m3_decode(b: u8) -> f32 {
        let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
        let field = (b >> 3) & 0x0f;
        let mant = f32::from(b & 0x07);
        if field == 0 {
            sign * mant * 2f32.powi(-9)
        } else {
            sign * (8.0 + mant) * 2f32.powi(i32::from(field) - 10)
        }
    }

    fn i8_encode(x: f32) -> u8 {
        round_even(f64::from(x)).clamp(-127.0, 127.0) as i8 as u8
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
        fn family(&self) -> PoolSel {
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

    /// Permutation of 0..n from an LCG over a power-of-two modulus (multiplier
    /// 5 mod 8 gives a full cycle); out-of-range draws are rejected.
    fn lcg_perm(n: usize, seed: u64) -> Vec<usize> {
        let modulus = n.next_power_of_two() as u64;
        let mut state = seed | 1;
        let mut out = Vec::with_capacity(n);
        let mut guard = 0usize;
        while out.len() < n {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let v = (state >> 40) % modulus;
            guard += 1;
            assert!(guard < 10_000_000, "lcg_perm stall");
            if (v as usize) < n && !out.contains(&(v as usize)) {
                out.push(v as usize);
            }
        }
        out
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

    #[allow(clippy::needless_range_loop)]
    fn attention_with_kv(case: &Case, token: usize, h: usize, kv: &RowKV, lim: usize) -> Vec<f64> {
        let mut scores = vec![0f64; lim];
        for (j, s) in scores.iter_mut().enumerate() {
            let mut dot = 0f64;
            for d in 0..D {
                dot += case.q_value(token, h, d) * kv.k[j * D + d];
            }
            *s = dot * SM_SCALE;
        }
        let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
        let denom: f64 = exps.iter().sum();

        let mut out = vec![0f64; D];
        for d in 0..D {
            let mut acc = 0f64;
            for j in 0..lim {
                acc += (exps[j] / denom) * kv.v[j * D + d];
            }
            out[d] = f64::from(bf(acc as f32).to_f32());
        }
        out
    }

    struct Metrics {
        rel_l2: f64,
        viol_frac: f64,
        max_dev: f64,
    }

    fn compare(case: &Case, got: &[bf16], corrupt_family: bool) -> (bool, Metrics) {
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
        let mut kv_by_row: Vec<Vec<RowKV>> = (0..rows_checked.len())
            .map(|_| Vec::with_capacity(HK))
            .collect();
        let built = std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (ri, &b) in rows_checked.iter().enumerate() {
                for hk in 0..HK {
                    let case_ref = &*case;
                    let lim = case.kvlens[b];
                    handles.push(scope.spawn(move || (ri, hk, build_row_kv(case_ref, b, hk, lim))));
                }
            }
            let mut built: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
            built.sort_by_key(|(ri, hk, _)| (*ri, *hk));
            built
        });
        for (ri, hk, kv) in built {
            if hk == 0 {
                kv_by_row[ri] = Vec::with_capacity(HK);
            }
            kv_by_row[ri].push(kv);
        }

        let mut wants: Vec<Vec<f64>> = (0..rows.len()).map(|_| Vec::new()).collect();
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (i, (token, h)) in rows.iter().enumerate() {
                let b = case.row_of_token(*token);
                let ri = rows_checked.iter().position(|&rb| rb == b).unwrap();
                let hk = h / G;
                let kv = &kv_by_row[ri][hk];
                let lim = row_lim(case, *token);
                let case_ref = &*case;
                handles.push(
                    scope.spawn(move || (i, attention_with_kv(case_ref, *token, *h, kv, lim))),
                );
            }
            for handle in handles {
                let (i, v) = handle.join().unwrap();
                wants[i] = v;
            }
        });

        let tol = case.form.tol();
        let mut diff_sq = 0f64;
        let mut ref_sq = 0f64;
        let mut violators = 0usize;
        let mut total = 0usize;
        let mut max_dev = 0f64;
        for (i, (token, h)) in rows.iter().enumerate() {
            for d in 0..D {
                let mut w = wants[i][d];
                // Corrupt the whole first checked row so the violation-rate
                // gate trips at every geometry (one element would dilute below
                // max_viol_frac on the long/batched cases).
                if corrupt_family && i == 0 {
                    w += 0.5;
                }
                let g = f64::from(got[(*token * H + h) * D + d].to_f32());
                let dev = (g - w).abs();
                diff_sq += dev.powi(2);
                ref_sq += w.powi(2);
                max_dev = max_dev.max(dev);
                total += 1;
                if dev > tol.floor + tol.slope * w.abs() {
                    violators += 1;
                }
            }
        }
        let rel_l2 = (diff_sq / ref_sq.max(1e-12)).sqrt();
        let viol_frac = violators as f64 / total as f64;
        let pass = rel_l2 < tol.rel_l2 && viol_frac <= tol.max_viol_frac;
        (
            pass,
            Metrics {
                rel_l2,
                viol_frac,
                max_dev,
            },
        )
    }

    // ── device run ──────────────────────────────────────────────────────────

    fn run_case(ctx: &DeviceContext, case: &Case, corrupt_family: bool) -> Result<bool> {
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
        let got = ctx.stream.clone_dtoh(&o_d)?;
        let (pass, m) = compare(case, &got, corrupt_family);
        eprintln!(
            "[{} B={} total_q={} kv={:?} table={:?} causal={} splits={}] rel_l2={:.2e} \
             viol_frac={:.2e} max_dev={:.2e} {}",
            case.label,
            batch,
            total_q,
            case.kvlens,
            case.table_kind,
            case.is_causal,
            case.splits,
            m.rel_l2,
            m.viol_frac,
            m.max_dev,
            if pass { "PASS" } else { "FAIL" }
        );
        Ok(pass)
    }

    pub(super) fn run(negative: Option<Option<PoolSel>>) -> Result<()> {
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

        let mut per_family = [
            (PoolSel::Bf16, true),
            (PoolSel::Fp8, true),
            (PoolSel::Int8, true),
        ];
        for case in &cases {
            let corrupt = match negative {
                None => false,
                Some(None) => true,
                Some(Some(fam)) => fam == case.family(),
            };
            let pass = run_case(&ctx, case, corrupt)?;
            for entry in &mut per_family {
                if entry.0 == case.family() {
                    entry.1 &= pass;
                }
            }
        }

        match negative {
            None => {
                let all = per_family.iter().all(|(_, ok)| *ok);
                ensure!(all, "fa3_hd256_shim_parity FAILED — see violations above");
                eprintln!("[fa3-hd256-shim-parity] ALL PASS");
            }
            Some(maybe_fam) => {
                for (fam, ok) in per_family {
                    let corrupted = maybe_fam.is_none_or(|f| f == fam);
                    if corrupted {
                        ensure!(
                            !ok,
                            "fa3_hd256_shim_parity negative control did NOT fail the {fam:?} family"
                        );
                    } else {
                        ensure!(
                            ok,
                            "fa3_hd256_shim_parity collateral failure in the {fam:?} family"
                        );
                    }
                }
                eprintln!("[fa3-hd256-shim-parity] NEGATIVE CONTROL OK");
            }
        }
        Ok(())
    }
}
