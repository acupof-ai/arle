//! CPU matmul micro-bench at Qwen3-0.6B production shapes.
//!
//! Codex's B2 phase profile (2026-05-19) attributes 96% of OPD step wall-clock
//! to forward + backward at the tiny smoke shape (hidden=16, vocab=16). Per
//! CLAUDE.md §0, tiny-window framing is not wall-clock ground truth for
//! production workloads; the real shape is Qwen3-0.6B (hidden=1024,
//! intermediate=3072, vocab=151936).
//!
//! This bench measures `cpu_matmul_forward` GFLOPs/s at every distinct matmul
//! shape an OPD forward pass touches on Qwen3-0.6B (seq=4 = 3 prompt + 1
//! rollout). Output establishes the ceiling for any drop-in matmul kernel swap
//! (matrixmultiply / gemm crate / rayon-parallel cache-blocked).
//!
//! Run:
//!   cargo run -p autograd --example cpu_matmul_microbench --release

use std::time::Instant;

use autograd::backend::cpu_matmul_forward;

const WARMUP: usize = 1;
const RUNS: usize = 5;
const TARGET_FMAS_PER_RUN: usize = 1_000_000_000;

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    m: usize,
    k: usize,
    n: usize,
    /// How many times this shape appears per single full forward
    /// (per-layer counts × num_hidden_layers, or lm_head=1).
    per_forward_count: usize,
}

const QWEN3_06B_SHAPES: &[Shape] = &[
    // Per-layer (×28 layers).
    Shape {
        name: "q_proj      [4,1024] @ [1024,2048]",
        m: 4,
        k: 1024,
        n: 2048,
        per_forward_count: 28,
    },
    Shape {
        name: "k_proj      [4,1024] @ [1024,1024]",
        m: 4,
        k: 1024,
        n: 1024,
        per_forward_count: 28,
    },
    Shape {
        name: "v_proj      [4,1024] @ [1024,1024]",
        m: 4,
        k: 1024,
        n: 1024,
        per_forward_count: 28,
    },
    Shape {
        name: "o_proj      [4,2048] @ [2048,1024]",
        m: 4,
        k: 2048,
        n: 1024,
        per_forward_count: 28,
    },
    Shape {
        name: "gate_proj   [4,1024] @ [1024,3072]",
        m: 4,
        k: 1024,
        n: 3072,
        per_forward_count: 28,
    },
    Shape {
        name: "up_proj     [4,1024] @ [1024,3072]",
        m: 4,
        k: 1024,
        n: 3072,
        per_forward_count: 28,
    },
    Shape {
        name: "down_proj   [4,3072] @ [3072,1024]",
        m: 4,
        k: 3072,
        n: 1024,
        per_forward_count: 28,
    },
    // Head (×1).
    Shape {
        name: "lm_head     [4,1024] @ [1024,151936]",
        m: 4,
        k: 1024,
        n: 151936,
        per_forward_count: 1,
    },
];

fn deterministic_fill(buf: &mut [f32], seed: u64) {
    let mut state = seed;
    for slot in buf.iter_mut() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = ((state >> 32) as f32) / (u32::MAX as f32);
        *slot = unit - 0.5;
    }
}

fn time_one(shape: Shape) -> (f64, f64, f64) {
    let a_len = shape.m * shape.k;
    let b_len = shape.k * shape.n;
    let mut a = vec![0.0f32; a_len];
    let mut b = vec![0.0f32; b_len];
    deterministic_fill(&mut a, 0x00A1_10C5);
    deterministic_fill(&mut b, 0x00B7_70C5);
    let a_shape = [shape.m, shape.k];
    let b_shape = [shape.k, shape.n];
    let fmas = shape.m * shape.k * shape.n;
    let iters_per_run = (TARGET_FMAS_PER_RUN / fmas).max(1);

    // Warm caches.
    for _ in 0..WARMUP {
        for _ in 0..iters_per_run {
            let _ = cpu_matmul_forward(&a, &a_shape, &b, &b_shape).expect("warmup");
        }
    }

    let mut secs_runs = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let started = Instant::now();
        for _ in 0..iters_per_run {
            let (out, _) = cpu_matmul_forward(&a, &a_shape, &b, &b_shape).expect("run");
            std::hint::black_box(out);
        }
        let elapsed = started.elapsed().as_secs_f64();
        secs_runs.push(elapsed / iters_per_run as f64);
    }
    secs_runs.sort_by(f64::total_cmp);
    let median = secs_runs[secs_runs.len() / 2];
    let mean = secs_runs.iter().sum::<f64>() / secs_runs.len() as f64;
    let variance = secs_runs
        .iter()
        .map(|secs| {
            let delta = secs - mean;
            delta * delta
        })
        .sum::<f64>()
        / secs_runs.len() as f64;
    let sigma = variance.sqrt();
    let sigma_pct = if mean == 0.0 {
        0.0
    } else {
        sigma / mean * 100.0
    };
    (median, mean, sigma_pct)
}

fn main() {
    println!(
        "backend=cpu_matmul_forward shape_count={} warmup={} runs={} target_fmas_per_run={}",
        QWEN3_06B_SHAPES.len(),
        WARMUP,
        RUNS,
        TARGET_FMAS_PER_RUN
    );
    println!(
        "{:<45} {:>10} {:>12} {:>12} {:>10} {:>14} {:>10} {:>10} {:>14}",
        "shape",
        "fmas",
        "median_s",
        "mean_s",
        "sigma_pct",
        "gflops_s",
        "iters/run",
        "x/fwd",
        "fwd_total_s"
    );

    let mut total_per_forward_secs = 0.0_f64;
    for shape in QWEN3_06B_SHAPES {
        let fmas = shape.m * shape.k * shape.n;
        let (median, mean, sigma_pct) = time_one(*shape);
        let gflops = (2.0 * fmas as f64 / median) / 1.0e9;
        let per_fwd_secs = median * shape.per_forward_count as f64;
        total_per_forward_secs += per_fwd_secs;
        println!(
            "{:<45} {:>10.3e} {:>12.6} {:>12.6} {:>10.3} {:>14.3} {:>10} {:>10} {:>14.6}",
            shape.name,
            fmas as f64,
            median,
            mean,
            sigma_pct,
            gflops,
            (TARGET_FMAS_PER_RUN / fmas).max(1),
            shape.per_forward_count,
            per_fwd_secs
        );
    }

    println!();
    println!(
        "estimated matmul cost per full forward (seq=4, all 28 layers + lm_head) = {:.6} s",
        total_per_forward_secs
    );
    println!(
        "estimated matmul cost per OPD step (≈3 forwards, ignores backward) = {:.6} s",
        total_per_forward_secs * 3.0
    );
}
