//! N-D parallel parity gate: context-parallel (CP) writeback vs single card.
//!
//! Exercises the REAL CP path on GPU — `all_gather_seq` KV forward + its
//! `reduce_scatter` backward + the post-backward `all_reduce_cp_grads` — which no
//! local (CPU) test can reach. The single-card reference and each CP rank run the
//! same name-seeded (deterministic) LoRA model over the same trajectory; CP shards
//! the sequence, so the returned loss is each rank's shard contribution scaled by
//! `1/global_targets`. Summed over ranks it must equal the single-card global-mean
//! CE within the correct-inference envelope — a wrong gather / shard filter /
//! position rebase / inv_n silently breaks this.
//!
//! Full-attention only: option-B CP shards full-attention; linear-attn CP is the
//! deferred/unmeasured piece (see docs/nd-parallel-training-design.md), so it is
//! out of this gate's scope.
//!
//! Run on a GPU host (>= 2 GPUs):
//!   ARLE_ND_CUDA_DEVICES=4,5 \
//!     cargo run -p train --release --no-default-features \
//!       --features cuda,nccl --example nd_parallel_parity

#[cfg(all(feature = "cuda", feature = "nccl"))]
use anyhow::{Context, Result, ensure};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use autograd::{TensorId, TensorStore, backend_cuda::CudaBackend, optim::AdamW};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use cuda_kernels::ffi::nccl;
#[cfg(all(feature = "cuda", feature = "nccl"))]
use qwen35_spec::{LayerType, Qwen35Config};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use train::{
    context_parallel::CpContext,
    lora::{LoraConfig, LoraTargetSet},
    opd::{WritebackLoss, masked_writeback_step},
    qwen35::Qwen35Model,
    tensor_parallel::TpContext,
};

#[cfg(all(feature = "cuda", feature = "nccl"))]
const CP_SIZE: usize = 2;
#[cfg(all(feature = "cuda", feature = "nccl"))]
const REL_TOL: f32 = 1.0e-3;
#[cfg(all(feature = "cuda", feature = "nccl"))]
const WINDOW: usize = 64;

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn main() -> Result<()> {
    if let Ok(rank) = std::env::var("ARLE_ND_RANK") {
        return rank_main(rank.parse().context("ARLE_ND_RANK parse")?);
    }
    coordinator_main()
}

#[cfg(not(all(feature = "cuda", feature = "nccl")))]
fn main() {
    eprintln!("nd_parallel_parity requires --features cuda,nccl");
}

// Prompt/response shared by the reference and every CP rank. Default seq 16 splits
// into 2*CP_SIZE=4 evenly (the zigzag precondition; opd.rs also pads up, but an
// already-divisible default keeps the reference trivially aligned). A SHORT prompt
// (4) so masked targets (predicting positions [prompt_len-1 .. seq-2]) straddle BOTH
// shards. `ARLE_ND_SEQ` overrides the total length (must be >= 6) to drive a
// >65535-local-seq case (131072 → local 65536, the ring path): the prompt stays 4,
// the response fills the rest with a deterministic vocab-cycling pattern so every
// rank builds the identical trajectory and targets still cross the shard boundary.
#[cfg(all(feature = "cuda", feature = "nccl"))]
fn nd_seq() -> usize {
    std::env::var("ARLE_ND_SEQ")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(16)
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn trajectory() -> (Vec<u32>, Vec<u32>, Vec<u8>) {
    let seq = nd_seq();
    let prompt: Vec<u32> = vec![1, 3, 8, 2];
    // Vocab is 16 (tiny_full_attn_config); cycle non-special ids 4..=13 so targets
    // are deterministic and never hit the eos (15) / bos (1). Response length = seq
    // - prompt_len.
    let response: Vec<u32> = (0..seq - prompt.len())
        .map(|i| 4 + (i % 10) as u32)
        .collect();
    let mask = vec![1u8; response.len()];
    (prompt, response, mask)
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn coordinator_main() -> Result<()> {
    let devices = parse_devices(CP_SIZE)?;
    let root = std::env::var("ARLE_ND_DIR")
        .unwrap_or_else(|_| format!("/tmp/arle_nd_parallel_parity_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).with_context(|| format!("create {root}"))?;

    // Single-card reference (no NCCL), in-process on the first device.
    let loss_single = run_writeback(devices[0], CpContext::single())?;

    // CP ranks: spawn CP_SIZE workers, each runs the same step under its shard.
    let exe = std::env::current_exe()?;
    let mut children = Vec::with_capacity(CP_SIZE);
    for (rank, &device) in devices.iter().enumerate().take(CP_SIZE) {
        let child = std::process::Command::new(&exe)
            .env("ARLE_ND_RANK", rank.to_string())
            .env("ARLE_ND_DIR", &root)
            .env("ARLE_ND_CUDA_DEVICE", device.to_string())
            .spawn()
            .with_context(|| format!("spawn CP rank {rank}"))?;
        children.push((rank, child));
    }
    for (rank, mut child) in children {
        let status = child.wait()?;
        ensure!(status.success(), "CP rank {rank} exited {status:?}");
    }

    // Each rank's returned loss is its shard's CE sum / global_targets; summed over
    // ranks that is the global-mean CE the single card computes directly.
    let mut loss_cp = 0.0f32;
    for rank in 0..CP_SIZE {
        loss_cp += read_loss(&root, rank)?;
    }

    let denom = loss_single.abs().max(loss_cp.abs()).max(1.0e-6);
    let rel_err = (loss_single - loss_cp).abs() / denom;
    println!("nd_parallel_parity cp_size={CP_SIZE} devices={devices:?}");
    println!(
        "loss_single={loss_single:.9e} loss_cp_sum={loss_cp:.9e} rel_err={rel_err:.3e} tol={REL_TOL:.1e}"
    );
    ensure!(
        rel_err <= REL_TOL,
        "CP loss parity mismatch: single={loss_single} cp_sum={loss_cp} rel_err={rel_err}"
    );
    println!("PASS");
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn rank_main(rank: usize) -> Result<()> {
    let dir = std::env::var("ARLE_ND_DIR").context("ARLE_ND_DIR missing")?;
    let device = env_usize("ARLE_ND_CUDA_DEVICE", rank)?;
    let unique_id = nccl_rendezvous(rank, &dir)?;
    let backend = CudaBackend::new_with_nccl(device, unique_id, CP_SIZE, rank)
        .with_context(|| format!("rank {rank} CudaBackend::new_with_nccl device {device}"))?;
    let store = TensorStore::with_backend(std::sync::Arc::new(backend));
    // Drive the CP context through the mesh (from_mesh → RankCoord), the path the
    // convergence installed, not a bespoke {rank, size}.
    let loss = run_writeback_in(store, CpContext::from_mesh(1, CP_SIZE, rank))?;
    write_loss(&dir, rank, loss)
}

// Single-card path builds its own store on `device`; CP ranks pass an NCCL store.
#[cfg(all(feature = "cuda", feature = "nccl"))]
fn run_writeback(device: usize, cp: CpContext) -> Result<f32> {
    let backend =
        CudaBackend::new(device).with_context(|| format!("CudaBackend::new({device})"))?;
    run_writeback_in(TensorStore::with_backend(std::sync::Arc::new(backend)), cp)
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn run_writeback_in(mut store: TensorStore, cp: CpContext) -> Result<f32> {
    let cfg = tiny_full_attn_config();
    let model = Qwen35Model::new_with_lora_targets_and_tp(
        &cfg,
        LoraConfig {
            rank: 2,
            alpha: 4.0,
        },
        LoraTargetSet::AllLinear,
        TpContext::single(),
        &mut store,
    )?;
    let all_params = model.all_parameter_ids();
    let trainable: Vec<TensorId> = all_params
        .iter()
        .copied()
        .filter(|&id| store.get(id).is_some_and(|t| t.requires_grad))
        .collect();
    ensure!(
        !trainable.is_empty(),
        "LoRA student must have trainable adapters"
    );
    let mut optimizer = AdamW::new(1.0e-3, (0.9, 0.999), 1.0e-8, 0.0);
    let (prompt, response, mask) = trajectory();

    masked_writeback_step(
        WritebackLoss::Ce,
        &model,
        &all_params,
        &trainable,
        &mut optimizer,
        true,
        &prompt,
        &response,
        &mask,
        cfg.vocab_size,
        WINDOW,
        cp,
        train::context_parallel::DpContext::single(),
        &mut store,
    )
    .map(|(loss, _)| loss)
    .context("masked writeback step")
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn tiny_full_attn_config() -> Qwen35Config {
    Qwen35Config {
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 2,
        vocab_size: 16,
        rms_norm_eps: 1.0e-6,
        stop_token_ids: vec![15],
        bos_token_id: Some(1),
        eos_token_id: 15,
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
        // RoPE cache must cover every absolute position: seq (from ARLE_ND_SEQ,
        // padded up by opd.rs) plus headroom. seq=131072 needs 131072 rows, not 16.
        rope_cache_len_hint: Some(nd_seq().next_power_of_two().max(16)),
        layer_types: vec![LayerType::FullAttention, LayerType::FullAttention],
        num_experts: 0,
        num_experts_per_tok: 0,
        decoder_sparse_step: 1,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: 0,
        norm_topk_prob: true,
        mlp_only_layers: Vec::new(),
        full_attn_gated: true,
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
            "rank {rank} timed out on NCCL id at {path}"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn write_loss(dir: &str, rank: usize, loss: f32) -> Result<()> {
    let path = format!("{dir}/rank_{rank}.txt");
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, format!("{loss:.9e}\n"))?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn read_loss(dir: &str, rank: usize) -> Result<f32> {
    let path = format!("{dir}/rank_{rank}.txt");
    let text = std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
    text.trim()
        .parse::<f32>()
        .with_context(|| format!("parse loss in {path}"))
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn parse_devices(need: usize) -> Result<Vec<usize>> {
    let raw = std::env::var("ARLE_ND_CUDA_DEVICES").unwrap_or_else(|_| {
        (0..need)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",")
    });
    let devices = raw
        .split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| p.trim().parse::<usize>().map_err(Into::into))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        devices.len() >= need,
        "ARLE_ND_CUDA_DEVICES has {} entries, need {need}",
        devices.len()
    );
    Ok(devices)
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn env_usize(name: &str, default: usize) -> Result<usize> {
    match std::env::var(name) {
        Ok(v) => v.parse().with_context(|| format!("{name} parse: {v}")),
        Err(_) => Ok(default),
    }
}
