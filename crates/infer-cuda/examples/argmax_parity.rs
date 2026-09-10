//! Argmax / argmax_batch numeric-parity gate.
//!
//! Standalone kernel harness — NO engine, NO serve. Drives the production
//! greedy-sampler selectors (`argmax_cuda`, `argmax_batch_cuda`) over BF16
//! rows and compares against an independent CPU scan that pins the kernel's
//! actual contract (sampling.cu):
//!   - tie at the maximum → LOWEST index
//!   - NaN never selected (all comparisons false); all-NaN row → index 0
//!   - -inf is a real value: finite max wins; all -inf row → index 0
//!
//! Geometry: vocab sweep across the 1024-thread / bf16x2-vectorized boundary,
//! including an odd length (tail-element lane) and a production-scale vocab;
//! batch 1 and 8 mixing normal / tie / NaN / -inf / mixed rows.
//!
//! `--negative-control` expects the wrong index for row 0 of EACH selector
//! family (batch and singular); both MUST independently FAIL.
//!
//! Run on a pod: `INFER_CUDA_DEVICE=<free-gpu> target/release/examples/argmax_parity`

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
        eprintln!("argmax_parity is a CUDA harness; rebuild with --features cuda.");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use anyhow::{Result, ensure};
    use cuda_kernels::prelude::DeviceContext;
    use cuda_kernels::sampling::{argmax, argmax_batch};
    use half::bf16;

    // Covers: <2048 elems (one bf16x2 pass), 2048 boundary, odd tail,
    // production-scale vocab.
    const VOCABS: &[usize] = &[7, 128, 1023, 2048, 2049, 100_000, 151_936];
    const BATCHES: &[usize] = &[1, 8];
    const SEED: u64 = 0x2545_f491_4f6c_dd1d;

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

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Kind {
        Normal,
        /// Two equal maxima, lower index must win.
        Tie,
        /// NaN entries sprinkled in, including at index 0; a finite max wins.
        Nan,
        /// Entire row NaN → kernel contract yields 0.
        AllNan,
        /// -inf entries with one finite max.
        NegInf,
        /// Entire row -inf → kernel contract yields 0.
        AllNegInf,
    }

    /// CPU reference pinned to the kernel contract: strict `>` scan from
    /// index 0 gives lowest-index ties for free; NaN fails every comparison.
    fn cpu_argmax(row: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best = i;
            }
        }
        best
    }

    fn build_row(rng: &mut Rng, n: usize, kind: Kind) -> Vec<f32> {
        let mut row: Vec<f32> = (0..n).map(|_| rng.normal() * 2.0).collect();
        match kind {
            Kind::Normal => {}
            Kind::Tie => {
                // Identical BF16 value at two positions, above every other.
                let v = 7.5f32;
                let i1 = n / 3;
                let i2 = (2 * n) / 3;
                for x in row.iter_mut() {
                    *x = (*x * 0.2).clamp(-2.0, 2.0);
                }
                row[i1] = v;
                row[i2] = v;
            }
            Kind::Nan => {
                let winner = n / 2;
                row[winner] = 9.0;
                for i in (0..n).step_by(7) {
                    if i != winner {
                        row[i] = f32::NAN;
                    }
                }
                row[0] = f32::NAN;
            }
            Kind::AllNan => row.fill(f32::NAN),
            Kind::NegInf => {
                let winner = 4 * n / 5;
                row[winner] = 3.0;
                for i in (0..n).step_by(5) {
                    if i != winner {
                        row[i] = f32::NEG_INFINITY;
                    }
                }
            }
            Kind::AllNegInf => row.fill(f32::NEG_INFINITY),
        }
        row
    }

    fn kinds_for(b: usize, v: usize) -> Vec<Kind> {
        use Kind::*;
        let mut ks = vec![Normal, Tie, Nan, AllNan, NegInf, AllNegInf];
        // Deterministic extra random-normal rows for batch width.
        ks.extend((6..b).map(|_| Normal));
        let _ = v;
        ks
    }

    pub(super) fn run(negative: bool) -> Result<()> {
        let ctx = DeviceContext::new()?;
        eprintln!(
            "[argmax-parity] device={} build={}{}",
            ctx.ordinal(),
            cuda_kernels::KERNEL_BUILD_ID,
            if negative { " NEGATIVE-CONTROL" } else { "" }
        );

        let mut batch_fail = false;
        let mut singular_fail = false;
        for &vocab in VOCABS {
            for &batch in BATCHES {
                let mut rng = Rng::new(SEED ^ ((vocab as u64) << 20) ^ ((batch as u64) << 8));
                let kinds = kinds_for(batch, vocab);
                let mut rows_f32 = Vec::with_capacity(batch);
                let mut rows_bf16 = Vec::with_capacity(batch * vocab);
                let mut expect = Vec::with_capacity(batch);
                for k in &kinds {
                    let r = build_row(&mut rng, vocab, *k);
                    expect.push(cpu_argmax(&r));
                    rows_bf16.extend(r.iter().map(|v| bf16::from_f32(*v)));
                    rows_f32.push(r);
                }
                if negative {
                    expect[0] = if expect[0] == 0 { 1 } else { 0 };
                }

                let logits_d = ctx.stream.clone_htod(&rows_bf16)?;
                let mut ids_d = ctx.stream.alloc_zeros::<i32>(batch)?;

                if batch == 1 {
                    argmax(&ctx, &logits_d, &mut ids_d, vocab)?;
                } else {
                    argmax_batch(&ctx, &logits_d, &mut ids_d, batch, vocab)?;
                }
                ctx.sync()?;
                let got = ctx.stream.clone_dtoh(&ids_d)?;

                for lane in 0..batch {
                    let pass = got[lane] as usize == expect[lane];
                    if !pass {
                        batch_fail = true;
                    }
                    eprintln!(
                        "[vocab={vocab:>6} B={batch} lane={lane} kind={:?}] got={} want={} {}",
                        kinds[lane],
                        got[lane],
                        expect[lane],
                        if pass { "PASS" } else { "FAIL" }
                    );
                }

                // Every row also through the singular selector: exercises the
                // single-row ABI on tie/NaN/-inf rows, not just the batch
                // kernel's row stride.
                for lane in 0..batch {
                    let row: Vec<bf16> = rows_bf16[lane * vocab..(lane + 1) * vocab].to_vec();
                    let row_d = ctx.stream.clone_htod(&row)?;
                    let mut one_d = ctx.stream.alloc_zeros::<i32>(1)?;
                    argmax(&ctx, &row_d, &mut one_d, vocab)?;
                    ctx.sync()?;
                    let one = ctx.stream.clone_dtoh(&one_d)?;
                    let mut want = cpu_argmax(&rows_f32[lane]);
                    if negative && lane == 0 {
                        want = if want == 0 { 1 } else { 0 };
                    }
                    let pass = one[0] as usize == want;
                    if !pass {
                        singular_fail = true;
                    }
                    eprintln!(
                        "[vocab={vocab:>6} B={batch} lane={lane} kind={:?} singular] got={} want={} {}",
                        kinds[lane],
                        one[0],
                        want,
                        if pass { "PASS" } else { "FAIL" }
                    );
                }
            }
        }

        if negative {
            ensure!(
                batch_fail,
                "argmax_parity negative control did NOT fail family: batch selector"
            );
            ensure!(
                singular_fail,
                "argmax_parity negative control did NOT fail family: singular selector"
            );
            eprintln!(
                "[argmax-parity] NEGATIVE CONTROL OK (both selector families failed as required)"
            );
            return Ok(());
        }
        ensure!(
            !batch_fail && !singular_fail,
            "argmax_parity FAILED — see violations above"
        );
        eprintln!("[argmax-parity] ALL PASS");
        Ok(())
    }
}
