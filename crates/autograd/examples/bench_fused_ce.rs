//! Micro-bench for `fused_linear_ce_loss_indexed` — the agent-OPD masked-CE
//! writeback core. Times the GPU device path (CUDA) against the host scalar
//! loop on the production shape (vocab=248320, hidden=5120), reporting the
//! per-target wall-clock for each so the GPU speedup over the ~20 s/target host
//! loop is measurable.
//!
//! Usage (on the H20 pod, GPU pinned via CUDA_VISIBLE_DEVICES):
//!   cargo run --release -p autograd --features cuda --example bench_fused_ce \
//!       -- --vocab 248320 --hidden 5120 --gpu-targets 512 --host-targets 4 \
//!          --chunk 256
//!
//! Host runs few targets (the loop is per-target linear, so per-target time is
//! representative); GPU runs the full --gpu-targets count. Each backend forward
//! + backward is timed; per-target = wall / targets.

use autograd::{CpuBackend, Tape, TensorId, TensorStore};
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

fn deterministic_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 33) as f32) / ((1u64 << 31) as f32);
            (u - 1.0) * 0.5
        })
        .collect()
}

/// One forward+backward of the fused masked CE on the given store/backend.
/// Returns (loss, elapsed_seconds). `seq == targets` (every row is a target).
fn run_once(
    mut store: TensorStore,
    targets: usize,
    hidden_dim: usize,
    vocab: usize,
    chunk: usize,
    seed: u64,
) -> (f32, f64) {
    let hidden_data = deterministic_vec(targets * hidden_dim, seed);
    let weight_data = deterministic_vec(vocab * hidden_dim, seed ^ 0xABCD);
    let positions: Vec<i32> = (0..targets as i32).collect();
    let targets_v: Vec<i32> = (0..targets).map(|i| ((i * 7 + 3) % vocab) as i32).collect();

    let mut tape = Tape::new();
    let hidden = store
        .from_slice(&hidden_data, &[1, targets, hidden_dim])
        .expect("hidden");
    store.get_mut(hidden).expect("hidden").requires_grad = true;
    let weight = store
        .from_slice(&weight_data, &[vocab, hidden_dim])
        .expect("weight");

    let start = Instant::now();
    let loss = autograd::ops::fused_linear_distill::fused_linear_ce_loss_indexed(
        hidden, weight, &positions, &targets_v, chunk, None, &mut store, &mut tape,
    )
    .expect("fused ce");
    let loss_value = store.to_host(loss).expect("loss")[0];
    let grads = tape.backward(loss, &mut store).expect("backward");
    let _d_hidden: TensorId = *grads.get(&hidden).expect("d_hidden");
    // Force the grad to host so device work is fully realized before timing ends.
    let _ = store.to_host(_d_hidden).expect("d_hidden host");
    let elapsed = start.elapsed().as_secs_f64();
    (loss_value, elapsed)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let vocab = parse_arg(&args, "--vocab", 248320usize);
    let hidden_dim = parse_arg(&args, "--hidden", 5120usize);
    let gpu_targets = parse_arg(&args, "--gpu-targets", 512usize);
    let host_targets = parse_arg(&args, "--host-targets", 4usize);
    let chunk = parse_arg(&args, "--chunk", 256usize);

    println!(
        "bench_fused_ce: vocab={vocab} hidden={hidden_dim} chunk={chunk} \
         gpu_targets={gpu_targets} host_targets={host_targets}"
    );

    // Host scalar loop (CPU backend) — few targets, per-target is linear.
    let (host_loss, host_secs) = run_once(
        TensorStore::with_backend(Arc::new(CpuBackend)),
        host_targets,
        hidden_dim,
        vocab,
        chunk,
        7,
    );
    let host_per_target = host_secs / host_targets as f64;
    println!(
        "HOST   targets={host_targets} wall={host_secs:.3}s per_target={host_per_target:.4}s \
         loss={host_loss:.4}"
    );

    // GPU device path (CUDA backend) — full target count.
    #[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
    {
        use autograd::backend_cuda::CudaBackend;
        let backend = CudaBackend::new(0).expect("cuda ctx");
        let (gpu_loss, gpu_secs) = run_once(
            TensorStore::with_backend(Arc::new(backend)),
            gpu_targets,
            hidden_dim,
            vocab,
            chunk,
            7,
        );
        let gpu_per_target = gpu_secs / gpu_targets as f64;
        println!(
            "GPU    targets={gpu_targets} wall={gpu_secs:.3}s per_target={gpu_per_target:.4}s \
             loss={gpu_loss:.4}"
        );
        let speedup = host_per_target / gpu_per_target;
        println!(
            "SPEEDUP per_target host/gpu = {host_per_target:.4}s / {gpu_per_target:.6}s = {speedup:.1}x"
        );
    }
    #[cfg(not(all(feature = "cuda", not(feature = "no-cuda"))))]
    {
        let _ = gpu_targets;
        println!("GPU    skipped (build with --features cuda on a GPU box)");
    }
}
