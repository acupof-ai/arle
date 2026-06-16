use std::{sync::Arc, time::Instant};

use autograd::{
    Backend, Result, Tape, Tensor, TensorStore,
    ops::{causal_sdpa_recompute, mul, sum},
};

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
use autograd::backend_cuda::CudaBackend;

fn main() -> Result<()> {
    let backend_label = env_string("ARLE_A1_BENCH_BACKEND", "cpu");
    let batch = env_usize("ARLE_A1_BENCH_BATCH", 1);
    let heads = env_usize("ARLE_A1_BENCH_HEADS", 2);
    let seq_len = env_usize("ARLE_A1_BENCH_SEQ", 128);
    let head_dim = env_usize("ARLE_A1_BENCH_HEAD_DIM", 256);
    let warmup = env_usize("ARLE_A1_BENCH_WARMUP", 1);
    let repeats = env_usize("ARLE_A1_BENCH_REPEATS", 3);
    let backend = make_backend(&backend_label)?;

    for _ in 0..warmup {
        let _ = run_once(
            backend.clone(),
            &backend_label,
            batch,
            heads,
            seq_len,
            head_dim,
        )?;
    }

    let mut forward_total = 0.0_f64;
    let mut backward_total = 0.0_f64;
    let mut attention_total = 0.0_f64;
    for iter in 0..repeats {
        let result = run_once(
            backend.clone(),
            &backend_label,
            batch,
            heads,
            seq_len,
            head_dim,
        )?;
        forward_total += result.forward_seconds;
        backward_total += result.backward_seconds;
        attention_total += result.attention_backward_seconds;
        eprintln!(
            "a1_attention_bench_iter iter={iter} backend={backend_label} batch={batch} \
             heads={heads} seq={seq_len} head_dim={head_dim} \
             forward_seconds={:.6} backward_seconds={:.6} \
             attention_backward_seconds={:.6}",
            result.forward_seconds, result.backward_seconds, result.attention_backward_seconds
        );
    }

    let denom = repeats.max(1) as f64;
    println!(
        "a1_attention_bench backend={backend_label} batch={batch} heads={heads} \
         seq={seq_len} head_dim={head_dim} warmup={warmup} repeats={repeats} \
         avg_forward_seconds={:.6} avg_backward_seconds={:.6} \
         avg_attention_backward_seconds={:.6}",
        forward_total / denom,
        backward_total / denom,
        attention_total / denom
    );
    Ok(())
}

struct BenchResult {
    forward_seconds: f64,
    backward_seconds: f64,
    attention_backward_seconds: f64,
}

fn run_once(
    backend: Arc<dyn Backend>,
    backend_label: &str,
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<BenchResult> {
    let mut store = TensorStore::with_backend(backend);
    let shape = vec![batch, heads, seq_len, head_dim];
    let size = shape.iter().product();
    let q = store.alloc(Tensor::new(
        deterministic_vec(size, 0x4d59_5df4, 0.05),
        shape.clone(),
        true,
    )?);
    let k = store.alloc(Tensor::new(
        deterministic_vec(size, 0x7a35_2b19, 0.05),
        shape.clone(),
        true,
    )?);
    let v = store.alloc(Tensor::new(
        deterministic_vec(size, 0x1f12_d9e7, 0.25),
        shape.clone(),
        true,
    )?);
    let probe = store.alloc(Tensor::new(
        deterministic_vec(size, 0x29c6_a413, 0.40),
        shape,
        false,
    )?);
    store.ensure_device(q)?;
    store.ensure_device(k)?;
    store.ensure_device(v)?;
    store.ensure_device(probe)?;
    store.backend().eval(&[])?;

    let mut tape = Tape::new();
    let forward_started = Instant::now();
    let out = causal_sdpa_recompute(q, k, v, &mut store, &mut tape)?;
    let weighted = mul(out, probe, &mut store, &mut tape)?;
    let loss = sum(weighted, &mut store, &mut tape)?;
    store.backend().eval(&[])?;
    let forward_seconds = forward_started.elapsed().as_secs_f64();

    let backward_started = Instant::now();
    let (grads, profile) = tape.backward_profiled(loss, &mut store)?;
    store.backend().eval(&[])?;
    let backward_seconds = backward_started.elapsed().as_secs_f64();

    if backend_label == "cuda" {
        for (name, id) in [("q", q), ("k", k), ("v", v)] {
            let grad_id = grads.get(&id).copied().expect("gradient exists");
            let grad = store.get(grad_id).expect("gradient tensor exists");
            assert!(
                grad.dirty != autograd::tensor::Dirty::Host && grad.device_handle.is_some(),
                "A1 attention CUDA gradient for {name} fell back to host"
            );
        }
    }

    let attention_backward_seconds = profile
        .op_totals
        .get(&autograd::BackwardOp::CausalSdpaRecompute)
        .map(|stats| stats.duration.as_secs_f64())
        .unwrap_or(0.0);
    print_profile(&profile);
    Ok(BenchResult {
        forward_seconds,
        backward_seconds,
        attention_backward_seconds,
    })
}

fn print_profile(profile: &autograd::BackwardProfile) {
    let total = profile.total_duration.as_secs_f64();
    let mut rows = profile
        .op_totals
        .iter()
        .map(|(&op, stats)| (op, stats.count, stats.duration.as_secs_f64()))
        .collect::<Vec<_>>();
    rows.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    for (rank, (op, count, seconds)) in rows.into_iter().take(8).enumerate() {
        let pct = if total == 0.0 {
            0.0
        } else {
            seconds / total * 100.0
        };
        eprintln!(
            "a1_attention_bench_profile rank={} op={} count={} seconds={seconds:.6} \
             pct_backward={pct:.3}",
            rank + 1,
            op.name(),
            count
        );
    }
}

fn make_backend(label: &str) -> Result<Arc<dyn Backend>> {
    match label {
        "cpu" => Ok(Arc::new(autograd::CpuBackend)),
        "cuda" => make_cuda_backend(),
        _ => Err(autograd::AutogradError::TapeInvariant(
            "ARLE_A1_BENCH_BACKEND must be cpu or cuda",
        )),
    }
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
fn make_cuda_backend() -> Result<Arc<dyn Backend>> {
    let ordinal = env_usize("ARLE_CUDA_TEST_DEVICE", 0);
    Ok(Arc::new(CudaBackend::new(ordinal)?))
}

#[cfg(not(all(feature = "cuda", not(feature = "no-cuda"))))]
fn make_cuda_backend() -> Result<Arc<dyn Backend>> {
    Err(autograd::AutogradError::TapeInvariant(
        "cuda backend requested but this binary was not built with executable CUDA",
    ))
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_string(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn deterministic_vec(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = ((state >> 40) as f32) / ((1u64 << 24) as f32);
        out.push((unit - 0.5) * scale);
    }
    out
}
