//! A/B for train-side `linear_forward`: current `transpose(weight) + matmul`
//! versus direct `A @ weight^T` using matrixmultiply strided-B views.
//!
//! This isolates the physical weight-transpose hypothesis before adding a
//! real autograd `matmul_bt` op.

use std::time::Instant;

use autograd::backend::cpu_matmul_forward;

const WARMUP: usize = 1;
const RUNS: usize = 5;

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    m: usize,
    k: usize,
    n: usize,
}

const SHAPES: &[Shape] = &[
    Shape {
        name: "q_proj",
        m: 4,
        k: 1024,
        n: 2048,
    },
    Shape {
        name: "k_proj",
        m: 4,
        k: 1024,
        n: 1024,
    },
    Shape {
        name: "v_proj",
        m: 4,
        k: 1024,
        n: 1024,
    },
    Shape {
        name: "o_proj",
        m: 4,
        k: 2048,
        n: 1024,
    },
    Shape {
        name: "gate_proj",
        m: 4,
        k: 1024,
        n: 3072,
    },
    Shape {
        name: "up_proj",
        m: 4,
        k: 1024,
        n: 3072,
    },
    Shape {
        name: "down_proj",
        m: 4,
        k: 3072,
        n: 1024,
    },
    Shape {
        name: "lm_head",
        m: 4,
        k: 1024,
        n: 151_936,
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

fn transpose_weight(weight: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; weight.len()];
    for row in 0..n {
        for col in 0..k {
            out[col * n + row] = weight[row * k + col];
        }
    }
    out
}

fn current_transpose_then_matmul(shape: Shape, a: &[f32], weight: &[f32]) -> Vec<f32> {
    let weight_t = transpose_weight(weight, shape.n, shape.k);
    cpu_matmul_forward(a, &[shape.m, shape.k], &weight_t, &[shape.k, shape.n])
        .expect("transpose + matmul")
        .0
}

fn direct_bt_matrixmultiply(shape: Shape, a: &[f32], weight: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; shape.m * shape.n];
    unsafe {
        matrixmultiply::sgemm(
            shape.m,
            shape.k,
            shape.n,
            1.0,
            a.as_ptr(),
            shape.k as isize,
            1,
            weight.as_ptr(),
            1,
            shape.k as isize,
            0.0,
            out.as_mut_ptr(),
            shape.n as isize,
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

fn time_route<F>(mut route: F) -> (f64, f64, f64)
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
    let var = times
        .iter()
        .map(|time| {
            let delta = time - mean;
            delta * delta
        })
        .sum::<f64>()
        / times.len() as f64;
    let sigma_pct = if mean > 0.0 {
        var.sqrt() / mean * 100.0
    } else {
        0.0
    };
    (median, mean, sigma_pct)
}

fn main() {
    println!("bench=cpu_linear_transpose_bt_ab runs={RUNS} warmup={WARMUP}");
    println!(
        "{:<10} {:>2} {:>5} {:>7} {:>12} {:>12} {:>10} {:>12} {:>12} {:>10} {:>9} {:>12}",
        "shape",
        "m",
        "k",
        "n",
        "current_s",
        "direct_s",
        "speedup",
        "cur_sigma",
        "bt_sigma",
        "diff",
        "cur_GF/s",
        "bt_GF/s",
    );

    for shape in SHAPES {
        let mut a = vec![0.0f32; shape.m * shape.k];
        let mut weight = vec![0.0f32; shape.n * shape.k];
        deterministic_fill(&mut a, 0x00A1_1CE5);
        deterministic_fill(&mut weight, 0xB00C_5EED);

        let current_sample = current_transpose_then_matmul(*shape, &a, &weight);
        let direct_sample = direct_bt_matrixmultiply(*shape, &a, &weight);
        let diff = max_abs_diff(&current_sample, &direct_sample);

        let (current_median, _, current_sigma) =
            time_route(|| current_transpose_then_matmul(*shape, &a, &weight));
        let (direct_median, _, direct_sigma) =
            time_route(|| direct_bt_matrixmultiply(*shape, &a, &weight));
        let fmas = shape.m * shape.k * shape.n;
        let current_gflops = (2.0 * fmas as f64 / current_median) / 1.0e9;
        let direct_gflops = (2.0 * fmas as f64 / direct_median) / 1.0e9;
        println!(
            "{:<10} {:>2} {:>5} {:>7} {:>12.6} {:>12.6} {:>10.3} {:>12.3} {:>12.3} {:>10.3e} {:>9.3} {:>12.3}",
            shape.name,
            shape.m,
            shape.k,
            shape.n,
            current_median,
            direct_median,
            current_median / direct_median,
            current_sigma,
            direct_sigma,
            diff,
            current_gflops,
            direct_gflops,
        );
    }
}
