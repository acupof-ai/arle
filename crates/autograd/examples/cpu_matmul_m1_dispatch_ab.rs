//! Single-variable A/B for the post-OPD `lm_head` M=1 dispatch question.
//!
//! Shapes:
//! - M=1, K=1024, N=151936: rollout-last-row `lm_head` candidate regime.
//! - M=4, K=1024, N=151936: existing Qwen3-0.6B full-row `lm_head` regime.
//!
//! Compares current `cpu_matmul_forward` dispatch against explicit saxpy and
//! explicit `matrixmultiply::sgemm` routes. This is intentionally an example
//! harness so the hot path stays untouched until a measured route is licensed.

use std::time::Instant;

use autograd::backend::cpu_matmul_forward;

const K: usize = 1024;
const N: usize = 151_936;
const WARMUP: usize = 1;
const RUNS: usize = 5;

#[derive(Clone, Copy)]
struct Shape {
    label: &'static str,
    m: usize,
}

const SHAPES: &[Shape] = &[
    Shape {
        label: "lm_head_m1",
        m: 1,
    },
    Shape {
        label: "lm_head_m4",
        m: 4,
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

fn saxpy_row_major(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        let a_row = &a[row * k..(row + 1) * k];
        let out_row = &mut out[row * n..(row + 1) * n];
        for inner in 0..k {
            let a_value = a_row[inner];
            let b_row = &b[inner * n..(inner + 1) * n];
            for col in 0..n {
                out_row[col] += a_value * b_row[col];
            }
        }
    }
    out
}

fn matrixmultiply_row_major(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }
    out
}

fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
}

fn bench_route<F>(mut route: F) -> (f64, f64, f64)
where
    F: FnMut() -> Vec<f32>,
{
    for _ in 0..WARMUP {
        std::hint::black_box(route());
    }

    let mut times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let started = Instant::now();
        std::hint::black_box(route());
        times.push(started.elapsed().as_secs_f64());
    }
    times.sort_by(f64::total_cmp);
    let median = times[times.len() / 2];
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let variance = times
        .iter()
        .map(|time| {
            let delta = time - mean;
            delta * delta
        })
        .sum::<f64>()
        / times.len() as f64;
    let sigma_pct = if mean > 0.0 {
        variance.sqrt() / mean * 100.0
    } else {
        0.0
    };
    (median, mean, sigma_pct)
}

fn print_route<F>(shape: Shape, label: &str, reference: &[f32], mut route: F)
where
    F: FnMut() -> Vec<f32>,
{
    let sample = route();
    let diff = max_abs_diff(reference, &sample);
    let (median, mean, sigma_pct) = bench_route(route);
    let fmas = shape.m * K * N;
    let gflops = (2.0 * fmas as f64 / median) / 1.0e9;
    println!(
        "{:<12} {:>2} {:>14} {:>10.3} {:>12.6} {:>12.6} {:>10.3} {:>12.6e}",
        shape.label, shape.m, label, gflops, median, mean, sigma_pct, diff,
    );
}

fn main() {
    println!("bench=cpu_matmul_m1_dispatch_ab k={K} n={N} runs={RUNS} warmup={WARMUP}");
    println!(
        "{:<12} {:>2} {:>14} {:>10} {:>12} {:>12} {:>10} {:>12}",
        "shape", "m", "route", "gflops/s", "median_s", "mean_s", "sigma_pct", "max_abs_diff",
    );

    for shape in SHAPES {
        let a_len = shape.m * K;
        let b_len = K * N;
        let mut a = vec![0.0f32; a_len];
        let mut b = vec![0.0f32; b_len];
        deterministic_fill(&mut a, 0x00A1_1CE5);
        deterministic_fill(&mut b, 0xB00C_5EED);

        let reference = saxpy_row_major(shape.m, K, N, &a, &b);

        print_route(*shape, "current", &reference, || {
            cpu_matmul_forward(&a, &[shape.m, K], &b, &[K, N])
                .expect("current route")
                .0
        });
        print_route(*shape, "saxpy", &reference, || {
            saxpy_row_major(shape.m, K, N, &a, &b)
        });
        print_route(*shape, "matrixmultiply", &reference, || {
            matrixmultiply_row_major(shape.m, K, N, &a, &b)
        });
    }
}
