//! A2 gate: differentiable NCCL all-reduce inside autograd.
//!
//! Coordinator mode spawns `world` child ranks for three cases
//! (`delta=-eps,0,+eps`). Each rank computes:
//!
//!   y = all_reduce_sum(x_rank)
//!   loss_rank = sum(y * y)
//!   backward(loss_rank)
//!
//! The coordinator sums per-rank losses and compares the central-difference
//! derivative of the distributed total loss against the analytic gradient on
//! the probed rank. This catches the common TP bug where forward all-reduces
//! but backward forgets the adjoint all-reduce (or applies the wrong scale).
//!
//! Run on a GPU host:
//!   ARLE_A2_WORLD=2 ARLE_A2_CUDA_DEVICES=4,5 \
//!     cargo run -p autograd --release --no-default-features \
//!       --features cuda,nccl --example a2_nccl_all_reduce_autograd

#[cfg(all(feature = "cuda", feature = "nccl"))]
use anyhow::{Context, Result, anyhow, ensure};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use autograd::{Tensor, TensorStore, backend_cuda::CudaBackend, ops, tape::Tape};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use cuda_kernels::ffi::nccl;

#[cfg(all(feature = "cuda", feature = "nccl"))]
const DEFAULT_EPS: f32 = 1.0e-3;
#[cfg(all(feature = "cuda", feature = "nccl"))]
const REL_TOL: f32 = 1.0e-2;
#[cfg(all(feature = "cuda", feature = "nccl"))]
const PROBE_RANK: usize = 0;
#[cfg(all(feature = "cuda", feature = "nccl"))]
const PROBE_INDEX: usize = 2;

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn main() -> Result<()> {
    if let Ok(rank) = std::env::var("ARLE_A2_RANK") {
        let rank = rank.parse().context("ARLE_A2_RANK parse")?;
        return rank_main(rank);
    }
    coordinator_main()
}

#[cfg(not(all(feature = "cuda", feature = "nccl")))]
fn main() {
    eprintln!("a2_nccl_all_reduce_autograd requires --features cuda,nccl");
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn coordinator_main() -> Result<()> {
    let world = env_usize("ARLE_A2_WORLD", 2)?;
    ensure!(world >= 2, "ARLE_A2_WORLD must be >= 2");
    let devices = parse_devices(world)?;
    let eps = env_f32("ARLE_A2_EPS", DEFAULT_EPS)?;
    let probe_rank = env_usize("ARLE_A2_PROBE_RANK", PROBE_RANK)?;
    let probe_index = env_usize("ARLE_A2_PROBE_INDEX", PROBE_INDEX)?;
    ensure!(
        probe_rank < world,
        "probe_rank {probe_rank} >= world {world}"
    );
    ensure!(
        probe_index < seed_values(0).len(),
        "probe_index out of range"
    );

    let root = std::env::var("ARLE_A2_DIR").unwrap_or_else(|_| {
        format!(
            "/tmp/arle_a2_nccl_allreduce_{}_{}",
            std::process::id(),
            unix_millis()
        )
    });
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).with_context(|| format!("create {root}"))?;

    let minus = run_case(
        &root,
        "minus",
        world,
        &devices,
        probe_rank,
        probe_index,
        -eps,
    )?;
    let base = run_case(&root, "base", world, &devices, probe_rank, probe_index, 0.0)?;
    let plus = run_case(&root, "plus", world, &devices, probe_rank, probe_index, eps)?;

    let numeric = (plus.total_loss - minus.total_loss) / (2.0 * eps);
    let analytic = base.grad[probe_index];
    let denom = analytic.abs().max(numeric.abs()).max(1.0e-6);
    let rel_err = (analytic - numeric).abs() / denom;

    println!(
        "a2_nccl_all_reduce_autograd world={world} devices={devices:?} probe=rank{probe_rank}[{probe_index}] eps={eps:.1e}"
    );
    println!(
        "loss_minus={:.9e} loss_base={:.9e} loss_plus={:.9e}",
        minus.total_loss, base.total_loss, plus.total_loss
    );
    println!(
        "analytic={analytic:.9e} numeric={numeric:.9e} rel_err={rel_err:.3e} tol={REL_TOL:.1e}"
    );
    ensure!(
        rel_err <= REL_TOL,
        "finite-diff mismatch: analytic={analytic} numeric={numeric} rel_err={rel_err}"
    );
    println!("PASS");
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
#[derive(Debug)]
struct CaseResult {
    total_loss: f32,
    grad: Vec<f32>,
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn run_case(
    root: &str,
    label: &str,
    world: usize,
    devices: &[usize],
    probe_rank: usize,
    probe_index: usize,
    delta: f32,
) -> Result<CaseResult> {
    let case_dir = format!("{root}/{label}");
    let _ = std::fs::remove_dir_all(&case_dir);
    std::fs::create_dir_all(&case_dir).with_context(|| format!("create {case_dir}"))?;

    let exe = std::env::current_exe()?;
    let children: Vec<(usize, std::process::Child)> = devices
        .iter()
        .copied()
        .enumerate()
        .take(world)
        .map(|(rank, device)| {
            let child = std::process::Command::new(&exe)
                .env("ARLE_A2_RANK", rank.to_string())
                .env("ARLE_A2_WORLD", world.to_string())
                .env("ARLE_A2_DIR", &case_dir)
                .env("ARLE_A2_CUDA_DEVICE", device.to_string())
                .env("ARLE_A2_PROBE_RANK", probe_rank.to_string())
                .env("ARLE_A2_PROBE_INDEX", probe_index.to_string())
                .env("ARLE_A2_PROBE_DELTA", format!("{delta:.9e}"))
                .spawn()
                .with_context(|| format!("spawn A2 rank {rank} case {label}"))?;
            Ok((rank, child))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    for (rank, mut child) in children {
        let status = child.wait()?;
        ensure!(
            status.success(),
            "A2 rank {rank} case {label} exited {status:?}"
        );
    }

    let mut total_loss = 0.0f32;
    let mut probe_grad = None;
    for rank in 0..world {
        let result = read_rank_result(&case_dir, rank)?;
        total_loss += result.total_loss;
        if rank == probe_rank {
            probe_grad = Some(result.grad);
        }
    }
    Ok(CaseResult {
        total_loss,
        grad: probe_grad.ok_or_else(|| anyhow!("missing probe rank {probe_rank} result"))?,
    })
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn rank_main(rank: usize) -> Result<()> {
    let world = env_usize("ARLE_A2_WORLD", 2)?;
    let dir = std::env::var("ARLE_A2_DIR").context("ARLE_A2_DIR missing")?;
    let device = env_usize("ARLE_A2_CUDA_DEVICE", rank)?;
    let probe_rank = env_usize("ARLE_A2_PROBE_RANK", PROBE_RANK)?;
    let probe_index = env_usize("ARLE_A2_PROBE_INDEX", PROBE_INDEX)?;
    let delta = env_f32("ARLE_A2_PROBE_DELTA", 0.0)?;

    let unique_id = nccl_rendezvous(rank, &dir)?;
    let backend = CudaBackend::new_with_nccl(device, unique_id, world, rank)
        .with_context(|| format!("rank {rank} CudaBackend::new_with_nccl device {device}"))?;
    let mut store = TensorStore::with_backend(std::sync::Arc::new(backend));
    let mut tape = Tape::new();

    let mut x_data = seed_values(rank);
    if rank == probe_rank {
        x_data[probe_index] += delta;
    }
    let x = store.alloc(Tensor::new(x_data, vec![4], true)?);
    let y = ops::all_reduce_sum(x, &mut store, &mut tape)?;
    let yy = ops::mul(y, y, &mut store, &mut tape)?;
    let loss = ops::sum(yy, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;

    let total_loss = *store
        .to_host(loss)?
        .first()
        .ok_or_else(|| anyhow!("rank {rank} loss readback empty"))?;
    let grad = store.to_host(grads[&x])?;
    write_rank_result(&dir, rank, total_loss, &grad)?;
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn seed_values(rank: usize) -> Vec<f32> {
    let r = rank as f32;
    vec![1.0 + 0.25 * r, -2.0 + 0.5 * r, 0.5 - 0.75 * r, 3.0 + r]
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn nccl_rendezvous(rank: usize, dir: &str) -> Result<nccl::ncclUniqueId> {
    use std::time::{Duration, Instant};

    let path = format!("{dir}/nccl_id.bin");
    let mut id = nccl::ncclUniqueId {
        internal: [0i8; 128],
    };
    if rank == 0 {
        nccl::check(unsafe { nccl::ncclGetUniqueId(&mut id) })?;
        let bytes: Vec<u8> = id.internal.iter().map(|&b| b as u8).collect();
        let tmp = format!("{path}.tmp");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, &path)?;
        return Ok(id);
    }

    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        if let Ok(bytes) = std::fs::read(&path) {
            if bytes.len() == 128 {
                for (dst, src) in id.internal.iter_mut().zip(bytes) {
                    *dst = src as i8;
                }
                return Ok(id);
            }
        }
        ensure!(
            Instant::now() < deadline,
            "rank {rank} timed out waiting for NCCL id at {path}"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn write_rank_result(dir: &str, rank: usize, loss: f32, grad: &[f32]) -> Result<()> {
    let grad_csv = grad
        .iter()
        .map(|v| format!("{v:.9e}"))
        .collect::<Vec<_>>()
        .join(",");
    let body = format!("loss={loss:.9e}\ngrad={grad_csv}\n");
    let path = format!("{dir}/rank_{rank}.txt");
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, body)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn read_rank_result(dir: &str, rank: usize) -> Result<CaseResult> {
    let path = format!("{dir}/rank_{rank}.txt");
    let text = std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
    let mut loss = None;
    let mut grad = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("loss=") {
            loss = Some(rest.parse::<f32>()?);
        } else if let Some(rest) = line.strip_prefix("grad=") {
            let values = rest
                .split(',')
                .map(|part| part.parse::<f32>().map_err(Into::into))
                .collect::<Result<Vec<_>>>()?;
            grad = Some(values);
        }
    }
    Ok(CaseResult {
        total_loss: loss.ok_or_else(|| anyhow!("{path} missing loss"))?,
        grad: grad.ok_or_else(|| anyhow!("{path} missing grad"))?,
    })
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn parse_devices(world: usize) -> Result<Vec<usize>> {
    let raw = std::env::var("ARLE_A2_CUDA_DEVICES").unwrap_or_else(|_| {
        (0..world)
            .map(|idx| idx.to_string())
            .collect::<Vec<_>>()
            .join(",")
    });
    let devices = raw
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.trim().parse::<usize>().map_err(Into::into))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        devices.len() >= world,
        "ARLE_A2_CUDA_DEVICES has {} entries, need {world}: {raw}",
        devices.len()
    );
    Ok(devices)
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn env_usize(name: &str, default: usize) -> Result<usize> {
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .with_context(|| format!("{name} parse: {value}")),
        Err(_) => Ok(default),
    }
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn env_f32(name: &str, default: f32) -> Result<f32> {
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .with_context(|| format!("{name} parse: {value}")),
        Err(_) => Ok(default),
    }
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn unix_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}
