//! Shared HARNESS for the standalone numeric-parity examples.
//!
//! This module holds only test plumbing that is identical across every gate:
//!   - [`Rng`]: the xorshift* normal/uniform generator every example uses;
//!   - [`cli`]: the `--kernel-build-id` / `--negative-control` argument parse;
//!   - [`Families`]: the "every comparator family must independently trip under
//!     `--negative-control`, and none may fail on a clean run" aggregator;
//!   - [`compare`] / [`rel_l2`]: the rel-L2 + elementwise slope/floor comparator.
//!
//! It deliberately contains NO reference math. Each example keeps its own
//! independent host oracle: a shared oracle would couple the gates to one
//! implementation and let a single math error pass every gate. Helpers that
//! only shuffle or tolerate numbers are harness, not oracle.
//!
//! Host-only, no CUDA dependency; each example gates the `mod` include behind
//! its own `#[cfg(feature = "cuda")]` block (mirroring `attn_common.rs`).


/// xorshift* generator, bit-for-bit identical to the inline Rng each parity
/// example used to carry. Deterministic across platforms; `normal` is
/// Box-Muller on the same 24-bit unit draws.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn normal(&mut self) -> f32 {
        let u1 = self.unit().max(1e-7);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Result of the common CLI parse.
pub struct Cli {
    /// `--negative-control` was passed.
    pub negative: bool,
}

/// Outcome of the common CLI parse: either run the gate, or `--kernel-build-id`
/// was handled and its (already printed) id should terminate the process.
pub enum Parsed {
    Run(Cli),
    BuildIdPrinted,
}

/// Parse the two flags every parity example shares. `--kernel-build-id` prints
/// the embedded bundle id (it must answer without a CUDA context) and returns
/// [`Parsed::BuildIdPrinted`], which the example's `main` turns into an early
/// `Ok(())`; `--negative-control` is reported inside [`Parsed::Run`].
pub fn cli() -> Parsed {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().skip(1).any(|a| a == "--kernel-build-id") {
        println!("{}", cuda_kernels::KERNEL_BUILD_ID);
        return Parsed::BuildIdPrinted;
    }
    let negative = args.iter().any(|a| a == "--negative-control");
    Parsed::Run(Cli { negative })
}

/// One bit per comparator family. A family is any independently-named
/// correctness claim the gate makes (one kernel ABI, one pool dtype, one output
/// tensor, …). Under `--negative-control` EVERY family must fail; on a clean
/// run NONE may. Recording them by name makes the negative-control assertion
/// prove each tooth separately and report the dead one instead of passing.
#[derive(Default)]
pub struct Families {
    failed: Vec<(String, bool)>,
}

impl Families {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record family `name`: `failed` is this run's verdict for that family.
    pub fn record(&mut self, name: impl Into<String>, failed: bool) {
        self.failed.push((name.into(), failed));
    }

    /// True if any recorded family failed on this run.
    pub fn any_failed(&self) -> bool {
        self.failed.iter().any(|(_, f)| *f)
    }

    /// Final verdict shared by every gate.
    ///
    /// Clean run (`negative == false`): every family must be green; one red
    /// family is an error naming it.
    ///
    /// Negative run (`negative == true`): every family must be red; a green
    /// family means that tooth is dead and is an error naming it.
    ///
    /// Prints the single terminal line (`ALL PASS` / `NEGATIVE CONTROL OK`) and
    /// returns an `Err` (already formatted) on failure, `Ok(())` on success.
    pub fn finish(self, tag: &str, negative: bool) -> anyhow::Result<()> {
        if negative {
            let alive: Vec<&str> = self.failed.iter().filter(|(_, f)| !f).map(|(n, _)| n.as_str()).collect();
            if !alive.is_empty() {
                anyhow::bail!(
                    "{tag} negative control did NOT fail {}: {}",
                    if alive.len() == 1 { "family" } else { "families" },
                    alive.join(", ")
                );
            }
            eprintln!("[{tag}] NEGATIVE CONTROL OK");
            Ok(())
        } else {
            let dead: Vec<&str> = self.failed.iter().filter(|(_, f)| *f).map(|(n, _)| n.as_str()).collect();
            if !dead.is_empty() {
                anyhow::bail!(
                    "{tag} FAILED {}: {}",
                    if dead.len() == 1 { "family" } else { "families" },
                    dead.join(", ")
                );
            }
            eprintln!("[{tag}] ALL PASS");
            Ok(())
        }
    }
}

/// Relative L2 distance `||got-want||_2 / ||want||_2` over f64 pairs.
pub fn rel_l2(got: &[f64], want: &[f64]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let d = g - w;
        num += d * d;
        den += w * w;
    }
    (num / den.max(1e-12)).sqrt()
}

/// The single element furthest outside its tolerance envelope.
#[derive(Clone, Copy, Debug)]
pub struct Worst {
    pub index: usize,
    pub got: f64,
    pub want: f64,
    /// Signed amount by which the absolute error exceeds its slope/floor band.
    pub excess: f64,
}

#[derive(Clone, Debug)]
pub struct Verdict {
    pub rel_l2: f64,
    /// Number of elements outside the absolute slope/floor band.
    pub violations: usize,
    pub worst: Option<Worst>,
}

impl Verdict {
    /// Green when rel-L2 is finite and within `rel_max` AND no element breaches
    /// its absolute band.
    /// Green when rel-L2 is finite and within `rel_max` (inclusive, matching
    /// the dominant gate form) AND no element breaches its absolute band.
    pub fn is_ok(&self, rel_max: f64) -> bool {
        self.rel_l2.is_finite()
            && self.rel_l2 <= rel_max
            && self.violations == 0
            && self.worst.as_ref().is_none_or(|w| w.excess.is_finite())
    }

    /// One-line worst-element report for FAIL diagnostics.
    pub fn worst_line(&self) -> String {
        match self.worst {
            Some(w) => format!(
                "relL2={:.3e} viol={} worst[{}] got={:.6} want={:.6} excess={:.3e}",
                self.rel_l2, self.violations, w.index, w.got, w.want, w.excess
            ),
            None => format!("relL2={:.3e} viol={}", self.rel_l2, self.violations),
        }
    }
}

/// Compare two equal-length numeric sequences under the standard gate bound:
/// - relative L2 distance below `rel_max`;
/// - every element within `abs_floor + abs_slope * |want|` absolute.
///
/// Inputs are taken as f64 so the same comparator serves bf16/f32/u8 `got`
/// (convert at the call site) against f64/f32 `want`. The oracle is computed
/// by the caller; this only measures distance.
pub fn compare(got: &[f64], want: &[f64], abs_floor: f64, abs_slope: f64) -> Verdict {
    let r = rel_l2(got, want);
    let mut violations = 0usize;
    let mut worst: Option<Worst> = None;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let excess = (g - w).abs() - (abs_floor + abs_slope * w.abs());
        if excess > 0.0 {
            violations += 1;
        }
        if worst.as_ref().is_none_or(|cur| excess > cur.excess) {
            worst = Some(Worst {
                index: i,
                got: *g,
                want: *w,
                excess,
            });
        }
    }
    Verdict {
        rel_l2: r,
        violations,
        worst,
    }
}
