//! A2 model gate: Qwen train-side TP row-output all-reduce participates in autograd.
//!
//! Coordinator mode spawns `world` child ranks for three cases
//! (`delta=-eps,0,+eps`). Each rank builds a tiny full-attention Qwen35 LoRA
//! model with tensor-parallel local q/k/v, gate/up, o/down weights. The loss is
//! the sum of squared logits after the row-parallel all-reduces. The central
//! difference of the distributed total loss is compared against the analytic
//! gradient for one rank-local `mlp.down_proj.lora_b` element.
//!
//! Run on a GPU host:
//!   ARLE_A2_WORLD=2 ARLE_A2_CUDA_DEVICES=4,5 \
//!     cargo run -p train --release --no-default-features \
//!       --features cuda,nccl --example a2_qwen35_tp_lora_fd

#[cfg(all(feature = "cuda", feature = "nccl"))]
use anyhow::{Context, Result, anyhow, ensure};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use autograd::{TensorStore, backend_cuda::CudaBackend, ops, tape::Tape};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use cuda_kernels::ffi::nccl;
#[cfg(all(feature = "cuda", feature = "nccl"))]
use qwen35_spec::{LayerType, Qwen35Config};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use train::{LoraConfig, LoraTargetSet, qwen35::Qwen35Model, tensor_parallel::TpContext};

#[cfg(all(feature = "cuda", feature = "nccl"))]
const DEFAULT_EPS: f32 = 2.0e-3;
#[cfg(all(feature = "cuda", feature = "nccl"))]
const REL_TOL: f32 = 1.0e-2;
#[cfg(all(feature = "cuda", feature = "nccl"))]
const PROBE_RANK: usize = 0;
#[cfg(all(feature = "cuda", feature = "nccl"))]
const PROBE_INDEX: usize = 8;
#[cfg(all(feature = "cuda", feature = "nccl"))]
const PROBE_SUFFIX: &str = ".layers.0.mlp.down_proj.weight.lora_b";

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
    eprintln!("a2_qwen35_tp_lora_fd requires --features cuda,nccl");
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

    let root = std::env::var("ARLE_A2_DIR").unwrap_or_else(|_| {
        format!(
            "/tmp/arle_a2_qwen35_tp_lora_fd_{}_{}",
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
        "a2_qwen35_tp_lora_fd world={world} devices={devices:?} probe=rank{probe_rank}[{probe_index}] eps={eps:.1e}"
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
    let mut children = Vec::with_capacity(world);
    for (rank, device) in devices.iter().copied().enumerate().take(world) {
        let child = std::process::Command::new(&exe)
            .env("ARLE_A2_RANK", rank.to_string())
            .env("ARLE_A2_WORLD", world.to_string())
            .env("ARLE_A2_DIR", &case_dir)
            .env("ARLE_A2_CUDA_DEVICE", device.to_string())
            .env("ARLE_A2_PROBE_RANK", probe_rank.to_string())
            .env("ARLE_A2_PROBE_INDEX", probe_index.to_string())
            .env("ARLE_A2_PROBE_DELTA", format!("{delta:.9e}"))
            .spawn()
            .with_context(|| format!("spawn A2 qwen35 rank {rank} case {label}"))?;
        children.push((rank, child));
    }
    for (rank, mut child) in children {
        let status = child.wait()?;
        ensure!(
            status.success(),
            "A2 qwen35 rank {rank} case {label} exited {status:?}"
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

    let cfg = tiny_qwen35_config();
    let lora = LoraConfig {
        rank: 2,
        alpha: 4.0,
    };
    let tp = TpContext::new(rank, world);
    let model = Qwen35Model::new_with_lora_targets_and_tp(
        &cfg,
        lora,
        LoraTargetSet::AllLinear,
        tp,
        &mut store,
    )?;
    let adapters = model.adapter_name_map();
    let (probe_name, &probe_id) = adapters
        .iter()
        .find(|(name, _)| name.ends_with(PROBE_SUFFIX))
        .ok_or_else(|| anyhow!("missing adapter ending with {PROBE_SUFFIX}"))?;
    let probe_len = store
        .get(probe_id)
        .ok_or_else(|| anyhow!("missing probe tensor {probe_id:?}"))?
        .data
        .len();
    ensure!(
        probe_index < probe_len,
        "probe index {probe_index} out of range for {probe_name} len {probe_len}"
    );
    if rank == probe_rank {
        let probe = store
            .get_mut(probe_id)
            .ok_or_else(|| anyhow!("missing mutable probe tensor {probe_id:?}"))?;
        probe.data[probe_index] += delta;
    }

    let input_ids = [1_u32, 2, 4];
    let position_ids = [0_u32, 1, 2];
    let logits = model.forward_batch(&mut store, &mut tape, &input_ids, &position_ids, 1, 3)?;
    let squared = ops::mul(logits, logits, &mut store, &mut tape)?;
    let loss = ops::sum(squared, &mut store, &mut tape)?;
    let grads = tape.backward(loss, &mut store)?;

    let total_loss = *store
        .to_host(loss)?
        .first()
        .ok_or_else(|| anyhow!("rank {rank} loss readback empty"))?;
    let grad_id = *grads
        .get(&probe_id)
        .ok_or_else(|| anyhow!("rank {rank} missing grad for {probe_name}"))?;
    let grad = store.to_host(grad_id)?;
    write_rank_result(&dir, rank, total_loss, &grad)?;
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn tiny_qwen35_config() -> Qwen35Config {
    Qwen35Config {
        hidden_size: 8,
        intermediate_size: 8,
        num_hidden_layers: 1,
        vocab_size: 13,
        rms_norm_eps: 1.0e-6,
        stop_token_ids: vec![12],
        bos_token_id: Some(1),
        eos_token_id: 12,
        tie_word_embeddings: false,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 2,
        linear_num_key_heads: 2,
        linear_key_head_dim: 2,
        linear_num_value_heads: 2,
        linear_value_head_dim: 2,
        linear_conv_kernel_dim: 4,
        rope_theta: 10_000.0,
        rope_scaling: None,
        partial_rotary_factor: 1.0,
        rotary_dim: 2,
        rope_cache_len_hint: Some(8),
        layer_types: vec![LayerType::FullAttention],
        num_experts: 0,
        num_experts_per_tok: 0,
        decoder_sparse_step: 1,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: 0,
        norm_topk_prob: true,
        mlp_only_layers: Vec::new(),
        full_attn_gated: true,
            output_gate_type: "sigmoid".to_string(),
    }
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
        if let Ok(bytes) = std::fs::read(&path)
            && bytes.len() == 128
        {
            for (dst, src) in id.internal.iter_mut().zip(bytes) {
                *dst = src as i8;
            }
            return Ok(id);
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
