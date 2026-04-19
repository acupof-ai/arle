//! TinyLM-shape training-step micro-bench for the M5.3a acceptance check.
//!
//! Plan: `docs/plans/m5.3-device-resident-tensor.md` §5 requires Metal ≥ 1.1×
//! CPU at d_model=128 on a forward-matmul + sum + backward loop (the
//! shape that regressed to 0.52× under the pre-M5.3a 1-op-per-eval
//! path). This bench isolates that loop — no tokenizer, no sampler, no
//! dataset — so numbers are reproducible and the ratio is directly
//! comparable to the 2026-04-18 TinyLM baseline.
//!
//! Usage:
//!   cargo run --release -p autograd --example bench_step_matmul \
//!       -- --backend cpu   --d 128 --iters 200
//!   cargo run --release -p autograd --example bench_step_matmul \
//!       --features metal -- --backend metal --d 128 --iters 200
//!
//! Prints a single tabular line per run:
//!   backend=metal d=128 iters=200 wall=0.612s step_ms=3.06 steps/s=326.7

use autograd::{
    CpuBackend, Tape, TensorStore,
    ops::{matmul, sum},
};
use std::sync::Arc;
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

fn deterministic_row(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 32) as u32 as f32) / (u32::MAX as f32);
            (unit - 0.5) * 0.1 // small amplitude to keep matmul outputs moderate
        })
        .collect()
}

enum BackendKind {
    Cpu,
    #[cfg(feature = "metal")]
    Metal,
}

impl BackendKind {
    fn parse(flag: &str) -> Result<Self, String> {
        match flag {
            "cpu" => Ok(BackendKind::Cpu),
            #[cfg(feature = "metal")]
            "metal" => Ok(BackendKind::Metal),
            other => Err(format!(
                "unknown backend '{other}' — rebuild with --features metal or pass --backend cpu"
            )),
        }
    }

    fn label(&self) -> &'static str {
        match self {
            BackendKind::Cpu => "cpu",
            #[cfg(feature = "metal")]
            BackendKind::Metal => "metal",
        }
    }
}

fn run_one_backend(kind: BackendKind, d: usize, iters: usize, batch: usize) -> f64 {
    // Shape: x:[batch, d] @ w:[d, d] → y:[batch, d]; loss = sum(y); backward.
    // Mirrors the single-matmul hot path inside TinyLM (a Linear + reduction).
    let x_data = deterministic_row(batch * d, 1);
    let w_data = deterministic_row(d * d, 2);

    let mut store = match kind {
        BackendKind::Cpu => TensorStore::with_backend(Arc::new(CpuBackend)),
        #[cfg(feature = "metal")]
        BackendKind::Metal => {
            TensorStore::with_backend(Arc::new(autograd::backend_metal::MetalBackend))
        }
    };

    // One-off warmup step so we don't count first-call FFI init + JIT paths.
    {
        let mut tape = Tape::new();
        let x = store
            .from_slice(&x_data, &[batch, d])
            .expect("alloc x warmup");
        let w = store.from_slice(&w_data, &[d, d]).expect("alloc w warmup");
        store.get_mut(w).expect("w").requires_grad = true;
        let y = matmul(x, w, &mut store, &mut tape).expect("matmul warmup");
        let loss = sum(y, &mut store, &mut tape).expect("sum warmup");
        let _grads = tape.backward(loss, &mut store).expect("backward warmup");
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        let mut tape = Tape::new();
        // New tensors every iter to model a fresh mini-batch; matches the RL
        // trainer's per-step allocation pattern (weights are fresh because we
        // discard each tape — parity to the 2026-04-18 bench).
        let x = store.from_slice(&x_data, &[batch, d]).expect("alloc x");
        let w = store.from_slice(&w_data, &[d, d]).expect("alloc w");
        store.get_mut(w).expect("w").requires_grad = true;
        let y = matmul(x, w, &mut store, &mut tape).expect("matmul");
        let loss = sum(y, &mut store, &mut tape).expect("sum");
        let _grads = tape.backward(loss, &mut store).expect("backward");
    }
    t0.elapsed().as_secs_f64()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let backend_flag: String = parse_arg(&args, "--backend", "cpu".to_string());
    let d: usize = parse_arg(&args, "--d", 128);
    let iters: usize = parse_arg(&args, "--iters", 200);
    let batch: usize = parse_arg(&args, "--batch", 4);

    let kind = BackendKind::parse(&backend_flag).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(1)
    });

    let label = kind.label();
    let wall = run_one_backend(kind, d, iters, batch);
    let step_ms = (wall / iters as f64) * 1000.0;
    let steps_per_s = iters as f64 / wall;
    println!(
        "backend={label} d={d} batch={batch} iters={iters} wall={wall:.3}s \
         step_ms={step_ms:.3} steps/s={steps_per_s:.1}"
    );
}
