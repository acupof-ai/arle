//! MoE-TP parity gate: tensor-parallel MoE forward must equal the single-rank MoE.
//!
//! Weights are name-seeded (deterministic), so a world=N run builds column/row
//! shards of the SAME logical experts + shared expert the world=1 run builds
//! whole. `forward_sparse_mlp` all-reduces the routed+shared partials over the TP
//! group, so every rank's final logits must equal the single-rank logits within
//! the correct-inference envelope — a wrong expert shard, a mis-split shared
//! expert, or a dropped all-reduce breaks it. Router is replicated (full
//! num_experts) so routing is bit-identical across ranks by construction.
//!
//! Run on a GPU host (≥2 GPUs):
//!   ARLE_MOE_TP_CUDA_DEVICES=4,5 \
//!     cargo run -p train --release --no-default-features \
//!       --features cuda,nccl --example moe_tp_parity

#[cfg(all(feature = "cuda", feature = "nccl"))]
use anyhow::{Context, Result, anyhow, ensure};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use autograd::{TensorStore, backend_cuda::CudaBackend, tape::Tape};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use cuda_kernels::ffi::nccl;
#[cfg(all(feature = "cuda", feature = "nccl"))]
use qwen35_spec::{LayerType, Qwen35Config};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use train::{
    lora::{LoraConfig, LoraTargetSet},
    qwen35::{Qwen35Model, Qwen35TensorParallelConfig},
};

#[cfg(all(feature = "cuda", feature = "nccl"))]
const WORLD: usize = 2;
#[cfg(all(feature = "cuda", feature = "nccl"))]
const REL_TOL: f32 = 2.0e-3; // MoE nondeterminism envelope, not byte-identity

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn main() -> Result<()> {
    if let Ok(rank) = std::env::var("ARLE_MOE_TP_RANK") {
        return rank_main(rank.parse().context("ARLE_MOE_TP_RANK parse")?);
    }
    coordinator_main()
}

#[cfg(not(all(feature = "cuda", feature = "nccl")))]
fn main() {
    eprintln!("moe_tp_parity requires --features cuda,nccl");
}

// Tiny MoE config: moe_intermediate_size + shared_expert_intermediate_size must
// divide WORLD (column/row-parallel), num_experts > 0, one sparse layer.
#[cfg(all(feature = "cuda", feature = "nccl"))]
fn tiny_moe_config() -> Qwen35Config {
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
        num_experts: 4,
        num_experts_per_tok: 2,
        decoder_sparse_step: 1,
        moe_intermediate_size: 8, // /WORLD=2 → 4 per rank (column/row)
        shared_expert_intermediate_size: 8, // /WORLD=2 → 4 per rank
        norm_topk_prob: true,
        mlp_only_layers: Vec::new(),
        full_attn_gated: true,
    }
}

// Forward the tiny MoE model at a given TP config, return the mean-abs logit (a
// cheap scalar fingerprint of the whole forward; TP-reduce errors move it).
#[cfg(all(feature = "cuda", feature = "nccl"))]
fn forward_fingerprint(store: &mut TensorStore, tp: Qwen35TensorParallelConfig) -> Result<f32> {
    let cfg = tiny_moe_config();
    let model = Qwen35Model::new_with_lora_targets_and_tp(
        &cfg,
        LoraConfig {
            rank: 2,
            alpha: 4.0,
        },
        LoraTargetSet::AllLinear,
        tp,
        store,
    )?;
    let mut tape = Tape::new();
    let input_ids = [1u32, 2, 4];
    let position_ids = [0u32, 1, 2];
    let logits = model.forward_batch(store, &mut tape, &input_ids, &position_ids, 1, 3)?;
    let host = store.to_host(logits)?;
    ensure!(!host.is_empty(), "empty logits");
    Ok(host.iter().map(|v| v.abs()).sum::<f32>() / host.len() as f32)
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn coordinator_main() -> Result<()> {
    let devices = parse_devices(WORLD)?;
    let root = std::env::var("ARLE_MOE_TP_DIR")
        .unwrap_or_else(|_| format!("/tmp/arle_moe_tp_parity_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).with_context(|| format!("create {root}"))?;

    // Single-rank reference (no NCCL) on the first device.
    let ref_fp = {
        let backend = CudaBackend::new(devices[0]).context("CudaBackend::new ref")?;
        let mut store = TensorStore::with_backend(std::sync::Arc::new(backend));
        forward_fingerprint(&mut store, Qwen35TensorParallelConfig::single())?
    };

    // TP ranks.
    let exe = std::env::current_exe()?;
    let mut children = Vec::with_capacity(WORLD);
    for (rank, &device) in devices.iter().enumerate().take(WORLD) {
        let child = std::process::Command::new(&exe)
            .env("ARLE_MOE_TP_RANK", rank.to_string())
            .env("ARLE_MOE_TP_DIR", &root)
            .env("ARLE_MOE_TP_CUDA_DEVICE", device.to_string())
            .spawn()
            .with_context(|| format!("spawn MoE-TP rank {rank}"))?;
        children.push((rank, child));
    }
    for (rank, mut child) in children {
        ensure!(child.wait()?.success(), "MoE-TP rank {rank} failed");
    }

    // Every rank's post-all-reduce fingerprint must match the single-rank ref.
    for rank in 0..WORLD {
        let fp = read_fp(&root, rank)?;
        let denom = ref_fp.abs().max(fp.abs()).max(1.0e-6);
        let rel = (ref_fp - fp).abs() / denom;
        println!("rank {rank}: fp={fp:.9e} ref={ref_fp:.9e} rel_err={rel:.3e}");
        ensure!(
            rel <= REL_TOL,
            "MoE-TP rank {rank} logits diverge from single-rank: rel_err={rel}"
        );
    }
    println!("PASS");
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn rank_main(rank: usize) -> Result<()> {
    let dir = std::env::var("ARLE_MOE_TP_DIR").context("ARLE_MOE_TP_DIR missing")?;
    let device = env_usize("ARLE_MOE_TP_CUDA_DEVICE", rank)?;
    let unique_id = nccl_rendezvous(rank, &dir)?;
    let backend = CudaBackend::new_with_nccl(device, unique_id, WORLD, rank)
        .with_context(|| format!("rank {rank} CudaBackend::new_with_nccl device {device}"))?;
    let mut store = TensorStore::with_backend(std::sync::Arc::new(backend));
    let fp = forward_fingerprint(&mut store, Qwen35TensorParallelConfig::new(rank, WORLD))?;
    write_fp(&dir, rank, fp)
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
fn write_fp(dir: &str, rank: usize, fp: f32) -> Result<()> {
    let path = format!("{dir}/rank_{rank}.txt");
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, format!("{fp:.9e}\n"))?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn read_fp(dir: &str, rank: usize) -> Result<f32> {
    let path = format!("{dir}/rank_{rank}.txt");
    let text = std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
    text.trim()
        .parse::<f32>()
        .with_context(|| format!("parse fp in {path}"))
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn parse_devices(need: usize) -> Result<Vec<usize>> {
    let raw = std::env::var("ARLE_MOE_TP_CUDA_DEVICES").unwrap_or_else(|_| {
        (0..need)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",")
    });
    let devices = raw
        .split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| p.trim().parse::<usize>().map_err(|e| anyhow!("{e}")))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        devices.len() >= need,
        "need {need} devices, got {}",
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
