//! Decode-shape W4 GEMM probe: Marlin (sm_80 `mma.sync`) against the CUTLASS
//! sm_90 mixed-input collective (`wgmma` + TMA), at the shapes decode actually
//! runs. Answers whether a Hopper-native kernel is worth wiring before any
//! dispatch work exists.
//!
//! Dev tooling — not a serving path. Buffers are zero-filled: this measures
//! launch configuration and byte movement, not numerics.
//!
//! ```text
//! cargo build -p cuda-kernels --release --features cuda --bin decode_gemm_probe
//! ```
use anyhow::{Result, anyhow};
use cuda_kernels::prelude::DeviceContext;
use cuda_kernels::quant_linear as ql;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use std::time::Instant;

const WARMUP: usize = 10;
const ITERS: usize = 100;

/// Weight bytes one pass over B moves: 4-bit values plus the group scales.
fn weight_bytes(n: usize, k: usize, group_size: usize) -> usize {
    n * k / 2 + n * k / group_size
}

fn sync(ctx: &DeviceContext) -> Result<()> {
    ctx.stream
        .synchronize()
        .map_err(|e| anyhow!("stream sync: {e}"))
}

/// Median of `ITERS` per-launch times, in ms. Timed as a batch after a warmup,
/// so the launch cost is amortised the same way decode amortises it.
fn time_ms(ctx: &DeviceContext, mut run: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        run()?;
    }
    sync(ctx)?;
    let t = Instant::now();
    for _ in 0..ITERS {
        run()?;
    }
    sync(ctx)?;
    Ok(t.elapsed().as_secs_f64() * 1e3 / ITERS as f64)
}

struct MarlinArm {
    input: CudaSlice<bf16>,
    packed: CudaSlice<u8>,
    global: CudaSlice<u16>,
    output: CudaSlice<bf16>,
    c_tmp: CudaSlice<f32>,
    workspace: CudaSlice<i32>,
}

impl MarlinArm {
    fn new(ctx: &DeviceContext, m: usize, n: usize, k: usize, gs: usize) -> Result<Self> {
        let sms = ctx.sm_count();
        let s = &ctx.stream;
        Ok(Self {
            input: s.alloc_zeros(m * k)?,
            packed: s.alloc_zeros(weight_bytes(n, k, gs))?,
            // bf16(1.0): the repack folds the 2^119 dequant bias in here, but a
            // zero global scale is a value the kernel may shortcut on.
            global: s.clone_htod(&[bf16::from_f32(1.0).to_bits()])?,
            output: s.alloc_zeros(m * n)?,
            c_tmp: s.alloc_zeros(ql::marlin_c_tmp_floats(64, sms)?)?,
            workspace: s.alloc_zeros(ql::marlin_workspace_ints(sms)?)?,
        })
    }

    fn run(&mut self, ctx: &DeviceContext, m: usize, n: usize, k: usize, gs: usize) -> Result<()> {
        ql::marlin_fp4_gemm(
            ctx,
            &self.input,
            &self.packed,
            &self.global,
            &mut self.output,
            &self.c_tmp,
            &self.workspace,
            m,
            n,
            k,
            gs,
        )
    }
}

/// The DSv4 MoE grouped GEMM driven with one group, which is a dense GEMM. The
/// dispatch inside picks its tile off `total_m / topk`, so decode M lands on
/// `SM90_CO<128, 16, 512>` — an M-tile of 128 against an M of 1..16, which is
/// the waste this probe is measuring the cost of.
struct CutlassArm {
    act: CudaSlice<u8>,
    weights: CudaSlice<u8>,
    a_scale: CudaSlice<f32>,
    b_scales: CudaSlice<u16>,
    expert_offsets: CudaSlice<i32>,
    problem_sizes: CudaSlice<i32>,
    workspace: CudaSlice<u8>,
    output: CudaSlice<bf16>,
}

impl CutlassArm {
    const WORKSPACE_BYTES: usize = 64 << 20;

    fn new(ctx: &DeviceContext, m: usize, n: usize, k: usize) -> Result<Self> {
        let s = &ctx.stream;
        Ok(Self {
            act: s.alloc_zeros(m * k)?,
            weights: s.alloc_zeros(n * k / 2)?,
            a_scale: s.clone_htod(&[1.0f32])?,
            // [E, K/512, N*4] bf16, the layout the scale pointer formula assumes.
            b_scales: s.alloc_zeros((k / 512) * n * 4)?,
            expert_offsets: s.clone_htod(&[0i32])?,
            // SGLang order: (N, M, K).
            problem_sizes: s.clone_htod(&[n as i32, m as i32, k as i32])?,
            workspace: s.alloc_zeros(Self::WORKSPACE_BYTES)?,
            output: s.alloc_zeros(m * n)?,
        })
    }

    fn run(&mut self, ctx: &DeviceContext, m: usize, n: usize, k: usize) -> Result<()> {
        let s = &ctx.stream;
        let (out, _g0) = self.output.device_ptr_mut(s);
        let (act, _g1) = self.act.device_ptr(s);
        let (w, _g2) = self.weights.device_ptr(s);
        let (a_scale, _g3) = self.a_scale.device_ptr(s);
        let (b_scales, _g4) = self.b_scales.device_ptr(s);
        let (offsets, _g5) = self.expert_offsets.device_ptr(s);
        let (sizes, _g6) = self.problem_sizes.device_ptr(s);
        let (ws, _g7) = self.workspace.device_ptr_mut(s);
        if std::env::var_os("PROBE_DUMP_PTRS").is_some() {
            for (name, p) in [
                ("out", out),
                ("act", act),
                ("weights", w),
                ("a_scale", a_scale),
                ("b_scales", b_scales),
                ("offsets", offsets),
                ("sizes", sizes),
                ("workspace", ws),
            ] {
                println!("  {name:<10} 0x{p:x}  %16={}", p % 16);
            }
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
        // SAFETY: every pointer comes from a live CudaSlice pinned by its guard,
        // sized above to the layout the C ABI documents; one group, topk 1.
        let rc = unsafe {
            cuda_kernels::ffi::moe::w4a8_moe_grouped_gemm_sm90(
                out as *mut u16,
                act as *const u8,
                w as *const u8,
                a_scale as *const f32,
                b_scales as *const u16,
                offsets as *const i32,
                sizes as *const i32,
                1,
                n as i32,
                k as i32,
                m as i32,
                1,
                ws as *mut u8,
                Self::WORKSPACE_BYTES,
                s.cu_stream(),
            )
        };
        if rc != 0 {
            return Err(anyhow!("w4a8_moe_grouped_gemm_sm90 returned {rc}"));
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let (major, minor) = ctx.compute_capability();
    println!("device sm_{major}{minor}, {} SMs\n", ctx.sm_count());

    // The two Qwen3.8-27B MLP shapes, as [n, k]. `argv` overrides with
    // `n k m` so a failing configuration can be bisected without a rebuild.
    let argv: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let (shapes, rows): (Vec<(usize, usize, &str)>, Vec<usize>) = match argv.as_slice() {
        [n, k, m] => (vec![(*n, *k, "argv")], vec![*m]),
        _ => (
            vec![(34816usize, 5120usize, "gate_up"), (5120, 17408, "down")],
            vec![1usize, 4, 8, 16],
        ),
    };
    let gs = 16;

    println!(
        "{:<9} {:>6} {:>7} {:>6} {:>10} {:>10} {:>10} {:>8}",
        "weight", "m", "n", "k", "marlin ms", "cutlass ms", "marlin GB/s", "speedup"
    );
    for &(n, k, label) in &shapes {
        let bytes = weight_bytes(n, k, gs) as f64;
        for &m in &rows {
            let mut marlin = MarlinArm::new(&ctx, m, n, k, gs)?;
            let marlin_ms = time_ms(&ctx, || marlin.run(&ctx, m, n, k, gs))?;

            let cutlass_ms = match CutlassArm::new(&ctx, m, n, k) {
                Ok(mut arm) => match time_ms(&ctx, || arm.run(&ctx, m, n, k)) {
                    Ok(ms) => Some(ms),
                    Err(e) => {
                        println!("  cutlass {label} m={m}: {e}");
                        None
                    }
                },
                Err(e) => {
                    println!("  cutlass {label} m={m} alloc: {e}");
                    None
                }
            };

            println!(
                "{label:<9} {m:>6} {n:>7} {k:>6} {marlin_ms:>10.4} {:>10} {:>10.1} {:>8}",
                cutlass_ms.map_or("-".to_owned(), |v| format!("{v:.4}")),
                bytes / (marlin_ms * 1e-3) / 1e9,
                cutlass_ms.map_or("-".to_owned(), |v| format!("{:.2}x", marlin_ms / v)),
            );
        }
    }
    Ok(())
}
