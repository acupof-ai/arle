//! CPU matmul **backward** micro-bench at Qwen3-0.6B production shapes.
//!
//! Sister bench to `cpu_matmul_microbench`. Verifies that codex's `499bfc0`
//! row-major locality rewrite (`cpu_matmul_forward`) flows through to
//! `cpu_matmul_backward` — which calls `cpu_matmul_forward` twice on
//! transposed operands. Per CLAUDE.md §0, the inheritance has to be measured,
//! not assumed: the host `transpose_last_two_ref` step is still a naive scalar
//! double loop, and if its share blows past a few percent the wall-clock win
//! shrinks accordingly.
//!
//! Run:
//!   cargo run -p autograd --example cpu_matmul_backward_microbench --release

use std::time::Instant;

use autograd::backend::cpu_matmul_backward;

const WARMUP: usize = 1;
const RUNS: usize = 5;
const TARGET_FMAS_PER_RUN: usize = 1_000_000_000;

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    m: usize,
    k: usize,
    n: usize,
    per_forward_count: usize,
}

const QWEN3_06B_SHAPES: &[Shape] = &[
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
    let out_len = shape.m * shape.n;
    let mut a = vec![0.0f32; a_len];
    let mut b = vec![0.0f32; b_len];
    let mut grad_out = vec![0.0f32; out_len];
    deterministic_fill(&mut a, 0x00A1_10C5);
    deterministic_fill(&mut b, 0x00B7_70C5);
    deterministic_fill(&mut grad_out, 0x0009_AD07);
    let a_shape = [shape.m, shape.k];
    let b_shape = [shape.k, shape.n];
    let grad_shape = [shape.m, shape.n];

    let fmas = (shape.m * shape.k * shape.n) as f64;
    let inner_iters = ((TARGET_FMAS_PER_RUN as f64) / (2.0 * fmas)).max(1.0) as usize;

    for _ in 0..WARMUP {
        let _ = cpu_matmul_backward(
            &a,
            &a_shape,
            &b,
            &b_shape,
            &grad_out,
            &grad_shape,
            true,
            true,
        )
        .expect("warmup");
    }

    let mut secs_runs = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let started = Instant::now();
        for _ in 0..inner_iters {
            let (ga, gb) = cpu_matmul_backward(
                &a,
                &a_shape,
                &b,
                &b_shape,
                &grad_out,
                &grad_shape,
                true,
                true,
            )
            .expect("run");
            std::hint::black_box((ga, gb));
        }
        let elapsed = started.elapsed().as_secs_f64() / inner_iters as f64;
        secs_runs.push(elapsed);
    }
    secs_runs.sort_by(f64::total_cmp);
    let median = secs_runs[secs_runs.len() / 2];
    let mean = secs_runs.iter().sum::<f64>() / secs_runs.len() as f64;
    let variance =
        secs_runs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / secs_runs.len() as f64;
    let sigma = variance.sqrt();
    (median, mean, sigma / mean.max(f64::EPSILON))
}

fn main() {
    println!(
        "backend=cpu_matmul_backward (forward inherits 499bfc0 row-major) shape_count={} warmup={} runs={} target_fmas_per_run={}",
        QWEN3_06B_SHAPES.len(),
        WARMUP,
        RUNS,
        TARGET_FMAS_PER_RUN
    );
    println!(
        "{:<45} {:>10} {:>12} {:>12} {:>10} {:>14} {:>10} {:>14}",
        "shape", "fmas", "median_s", "mean_s", "sigma_pct", "gflops_s", "×/fwd", "fwd_total_s"
    );

    let mut total_per_forward_secs = 0.0_f64;
    for shape in QWEN3_06B_SHAPES {
        let fmas = (shape.m * shape.k * shape.n) as f64;
        let (median, mean, sigma_frac) = time_one(*shape);
        // Backward = 2 sgemms (grad_a = grad_out @ B^T, grad_b = A^T @ grad_out)
        // so effective FMAs = 2 × forward FMAs.
        let gflops = (2.0 * (2.0 * fmas) / median) / 1.0e9;
        let per_fwd_secs = median * shape.per_forward_count as f64;
        total_per_forward_secs += per_fwd_secs;
        println!(
            "{:<45} {:>10.3e} {:>12.6} {:>12.6} {:>9.3}% {:>14.3} {:>10} {:>14.6}",
            shape.name,
            fmas,
            median,
            mean,
            sigma_frac * 100.0,
            gflops,
            shape.per_forward_count,
            per_fwd_secs
        );
    }

    println!();
    println!(
        "estimated backward matmul cost per full forward (seq=4, all 28 layers + lm_head) = {:.6} s",
        total_per_forward_secs
    );
    println!(
        "estimated backward matmul cost per OPD step (1 backward pass through the student) = {:.6} s",
        total_per_forward_secs
    );
}
