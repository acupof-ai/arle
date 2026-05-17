//! Host AdamW micro-bench for training-step optimizer work.
//!
//! This isolates `AdamW::step` on the default host path with preallocated
//! params and fixed gradients. It is intentionally small and dependency-free:
//! run it before and after an optimizer edit to get a matched A/B line.
//!
//! Usage:
//!   cargo run --release -p autograd --example bench_adamw_host -- \
//!       --params 256 --len 4096 --iters 100 --wd 0.01

use autograd::{Tensor, TensorId, TensorStore, optim::AdamW};
use std::time::Instant;

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    if let Some(pos) = args.iter().position(|arg| arg == flag) {
        args.get(pos + 1)
            .and_then(|value| value.parse::<T>().ok())
            .unwrap_or(default)
    } else {
        default
    }
}

fn deterministic_values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 32) as u32 as f32) / (u32::MAX as f32);
            (unit - 0.5) * 0.02
        })
        .collect()
}

fn build_store(params: usize, len: usize) -> (TensorStore, Vec<TensorId>) {
    let mut store = TensorStore::default();
    let mut param_ids = Vec::with_capacity(params);

    for index in 0..params {
        let param_data = deterministic_values(len, (index as u64) + 1);
        let grad_data = deterministic_values(len, (index as u64) + 10_001);
        let param_id = store.alloc(
            Tensor::new(param_data, vec![len], true).expect("param tensor should be well-formed"),
        );
        let grad_id = store.alloc(
            Tensor::new(grad_data, vec![len], false).expect("grad tensor should be well-formed"),
        );
        store.get_mut(param_id).expect("param exists").grad = Some(grad_id);
        param_ids.push(param_id);
    }

    (store, param_ids)
}

fn checksum(store: &TensorStore, params: &[TensorId]) -> f64 {
    params
        .iter()
        .filter_map(|&id| store.get(id))
        .flat_map(|tensor| tensor.data.iter())
        .map(|&value| value as f64)
        .sum()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let params: usize = parse_arg(&args, "--params", 256);
    let len: usize = parse_arg(&args, "--len", 4096);
    let iters: usize = parse_arg(&args, "--iters", 100);
    let lr: f32 = parse_arg(&args, "--lr", 3.0e-4);
    let wd: f32 = parse_arg(&args, "--wd", 0.01);

    let (mut store, param_ids) = build_store(params, len);
    let mut optim = AdamW::new(lr, (0.9, 0.999), 1.0e-8, wd);

    optim.step(&param_ids, &mut store);

    let t0 = Instant::now();
    for _ in 0..iters {
        optim.step(&param_ids, &mut store);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let total_elements = params * len * iters;
    let ns_per_element = (elapsed * 1.0e9) / (total_elements as f64);
    let steps_per_s = (iters as f64) / elapsed;

    println!(
        "params={params} len={len} iters={iters} lr={lr} wd={wd} \
         wall={elapsed:.6}s step_ms={:.6} steps/s={steps_per_s:.3} \
         ns_per_element={ns_per_element:.3} checksum={:.6}",
        (elapsed / iters as f64) * 1000.0,
        checksum(&store, &param_ids)
    );
}
