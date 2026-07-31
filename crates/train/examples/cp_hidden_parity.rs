//! CP hidden-state parity: does the CP forward produce the SAME per-position
//! hidden state as the single card, BEFORE the loss? The device ring kernel and
//! its NCCL transport are already exonerated (see the kernel + transport parity
//! tests), and the end-to-end masked-writeback still FAILs ~5% concentrated on
//! rank0's non-contiguous zigzag targets. This bisects forward-wiring vs the loss
//! path: run the FULL model forward (RoPE at shard positions, q/k/v norm, gate,
//! MLP — everything the transport test skipped) on each rank's zigzag shard and
//! compare its output rows against the single-card full-seq hidden. Mismatch =>
//! forward wiring bug (cos/sin, gate, position plumbing). Match => the divergence
//! is in the masked-CE target indexing, not the forward.
//!
//! Run on a GPU host (>= 2 GPUs):
//!   ARLE_CPH_CUDA_DEVICES=1,3 \
//!     cargo run -p train --release --no-default-features \
//!       --features cuda,nccl --example cp_hidden_parity

#[cfg(all(feature = "cuda", feature = "nccl"))]
use anyhow::{Context, Result, ensure};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use autograd::{TensorStore, backend_cuda::CudaBackend, tape::Tape};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use cuda_kernels::ffi::nccl;
#[cfg(all(feature = "cuda", feature = "nccl"))]
use qwen35_spec::{LayerType, Qwen35Config};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use train::{
    context_parallel::CpContext,
    lora::{LoraConfig, LoraTargetSet},
    qwen35::Qwen35Model,
    tensor_parallel::TpContext,
};

#[cfg(all(feature = "cuda", feature = "nccl"))]
const CP_SIZE: usize = 2;
#[cfg(all(feature = "cuda", feature = "nccl"))]
const SEQ: usize = 16;
#[cfg(all(feature = "cuda", feature = "nccl"))]
const REL_TOL: f32 = 2.0e-2;

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn main() -> Result<()> {
    if let Ok(rank) = std::env::var("ARLE_CPH_RANK") {
        return rank_main(rank.parse().context("ARLE_CPH_RANK parse")?);
    }
    coordinator_main()
}

#[cfg(not(all(feature = "cuda", feature = "nccl")))]
fn main() {
    eprintln!("cp_hidden_parity requires --features cuda,nccl");
}

// Same tiny config + trajectory as nd_parallel_parity, so this shares the exact
// name-seeded model and token stream the failing gate uses.
#[cfg(all(feature = "cuda", feature = "nccl"))]
fn tiny_cfg() -> Qwen35Config {
    Qwen35Config {
        hidden_size: 512,
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
        head_dim: 128,
        linear_num_key_heads: 2,
        linear_key_head_dim: 2,
        linear_num_value_heads: 2,
        linear_value_head_dim: 2,
        linear_conv_kernel_dim: 4,
        rope_theta: 10_000.0,
        rope_scaling: None,
        partial_rotary_factor: 1.0,
        rotary_dim: 128,
        rope_cache_len_hint: Some(SEQ.next_power_of_two().max(16)),
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

// Deterministic full sequence: prompt [1,3,8,2] then vocab-cycling 4..=13.
#[cfg(all(feature = "cuda", feature = "nccl"))]
fn full_tokens() -> Vec<u32> {
    let prompt = [1u32, 3, 8, 2];
    let mut ids = prompt.to_vec();
    for i in 0..SEQ - prompt.len() {
        ids.push(4 + (i % 10) as u32);
    }
    ids
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn build_model(store: &mut TensorStore) -> Result<Qwen35Model> {
    let cfg = tiny_cfg();
    Ok(Qwen35Model::new_with_lora_targets_and_tp(
        &cfg,
        LoraConfig {
            rank: 2,
            alpha: 4.0,
        },
        LoraTargetSet::AllLinear,
        TpContext::single(),
        store,
    )?)
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn rank_main(rank: usize) -> Result<()> {
    let dir = std::env::var("ARLE_CPH_DIR").context("ARLE_CPH_DIR missing")?;
    let device = env_usize("ARLE_CPH_CUDA_DEVICE", rank)?;
    let unique_id = nccl_rendezvous(rank, &dir)?;
    let backend = CudaBackend::new_with_nccl(device, unique_id, CP_SIZE, rank)
        .with_context(|| format!("rank {rank} CudaBackend::new_with_nccl device {device}"))?;
    let mut store = TensorStore::with_backend(std::sync::Arc::new(backend));
    let mut model = build_model(&mut store)?;
    model.set_cp(CpContext::from_mesh(1, CP_SIZE, rank));

    let full = full_tokens();
    let cp = CpContext::from_mesh(1, CP_SIZE, rank);
    let shard = cp.shard(SEQ);
    let rows = shard.local_rows();
    let ids: Vec<u32> = rows.iter().map(|&r| full[r]).collect();
    let pos: Vec<u32> = rows.iter().map(|&r| r as u32).collect();

    let mut tape = Tape::new();
    tape.set_enabled(false);
    let hidden = model
        .forward_hidden_states(&mut store, &mut tape, &ids, &pos)
        .context("CP forward_hidden_states")?;
    let host = store.to_host(hidden)?; // [1, local_len, hidden]
    // Write "row0=..,row1=.." mapping global row -> flat hidden vector.
    write_rows(&dir, rank, &rows, &host, tiny_cfg().hidden_size)
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn coordinator_main() -> Result<()> {
    let devices = parse_devices(CP_SIZE)?;
    let root = std::env::var("ARLE_CPH_DIR")
        .unwrap_or_else(|_| format!("/tmp/arle_cp_hidden_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).with_context(|| format!("create {root}"))?;

    // Single-card CUDA reference: full-seq forward, in-process on the first device.
    let ref_hidden = {
        let backend = CudaBackend::new(devices[0])
            .with_context(|| format!("CudaBackend::new({})", devices[0]))?;
        let mut store = TensorStore::with_backend(std::sync::Arc::new(backend));
        let model = build_model(&mut store)?; // cp = single by default
        let full = full_tokens();
        let pos: Vec<u32> = (0..SEQ as u32).collect();
        let mut tape = Tape::new();
        tape.set_enabled(false);
        let h = model
            .forward_hidden_states(&mut store, &mut tape, &full, &pos)
            .context("single-card forward_hidden_states")?;
        store.to_host(h)?
    };
    // CPU-f32 GROUND TRUTH: same name-seeded model on the default (CPU) backend.
    // Disambiguates a real CP bug from a bf16-vs-f32 artifact — the single-card
    // CUDA ref uses the bf16 prefill kernel (lossy output), the CP path is f32, so
    // "CP != single-card" alone can't tell which is wrong. Whichever CUDA path is
    // CLOSER to this f32 reference is the more correct one.
    let cpu_hidden = {
        let mut store = TensorStore::default();
        let model = build_model(&mut store)?;
        let full = full_tokens();
        let pos: Vec<u32> = (0..SEQ as u32).collect();
        let mut tape = Tape::new();
        tape.set_enabled(false);
        let h = model
            .forward_hidden_states(&mut store, &mut tape, &full, &pos)
            .context("CPU-f32 forward_hidden_states")?;
        store.to_host(h)?
    };
    let hidden_size = tiny_cfg().hidden_size;
    // Single-card CUDA vs CPU-f32: the precision floor the CP path is judged against.
    let single_vs_cpu = (0..SEQ)
        .map(|r| {
            let a = &ref_hidden[r * hidden_size..(r + 1) * hidden_size];
            let b = &cpu_hidden[r * hidden_size..(r + 1) * hidden_size];
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        })
        .fold(0.0f32, f32::max);
    println!("single_card_cuda_vs_cpu_f32 max_diff={single_vs_cpu:.6e}");

    // Spawn CP ranks.
    let exe = std::env::current_exe()?;
    let mut children = Vec::with_capacity(CP_SIZE);
    for (rank, &device) in devices.iter().enumerate().take(CP_SIZE) {
        let child = std::process::Command::new(&exe)
            .env("ARLE_CPH_RANK", rank.to_string())
            .env("ARLE_CPH_DIR", &root)
            .env("ARLE_CPH_CUDA_DEVICE", device.to_string())
            .spawn()
            .with_context(|| format!("spawn CP rank {rank}"))?;
        children.push((rank, child));
    }
    for (rank, mut child) in children {
        let status = child.wait()?;
        ensure!(status.success(), "CP rank {rank} exited {status:?}");
    }

    // Compare each CP shard row against BOTH the single-card CUDA row and the
    // CPU-f32 ground truth at the same global pos. worst = CP-vs-single (the gate);
    // cp_vs_cpu = CP-vs-f32 (does CP track f32 better than single-card does?).
    let mut worst = 0.0f32;
    let mut worst_row = 0usize;
    let mut cp_vs_cpu = 0.0f32;
    for rank in 0..CP_SIZE {
        let (rows, data) = read_rows(&root, rank, hidden_size)?;
        for (local, &g) in rows.iter().enumerate() {
            let got = &data[local * hidden_size..(local + 1) * hidden_size];
            let want = &ref_hidden[g * hidden_size..(g + 1) * hidden_size];
            let d = got
                .iter()
                .zip(want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            if d > worst {
                worst = d;
                worst_row = g;
            }
            let cpu = &cpu_hidden[g * hidden_size..(g + 1) * hidden_size];
            let dc = got
                .iter()
                .zip(cpu)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            cp_vs_cpu = cp_vs_cpu.max(dc);
        }
        // Per-rank worst (vs single-card) so the zigzag asymmetry is visible.
        let rank_worst = rows
            .iter()
            .enumerate()
            .map(|(local, &g)| {
                let got = &data[local * hidden_size..(local + 1) * hidden_size];
                let want = &ref_hidden[g * hidden_size..(g + 1) * hidden_size];
                got.iter()
                    .zip(want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max)
            })
            .fold(0.0f32, f32::max);
        println!("rank {rank} shard_rows={rows:?} max_diff_vs_single={rank_worst:.6e}");
        // Per-TARGET-ROW diff vs f32: the loss reads hidden ONLY at target positions
        // (predict p in prompt_len-1..=seq-2). A global max-diff hides a bias
        // concentrated there — this is the diff the CE actually sees. prompt_len=4.
        for (local, &g) in rows.iter().enumerate() {
            if (3..=SEQ - 2).contains(&g) {
                let got = &data[local * hidden_size..(local + 1) * hidden_size];
                let cpu = &cpu_hidden[g * hidden_size..(g + 1) * hidden_size];
                let single = &ref_hidden[g * hidden_size..(g + 1) * hidden_size];
                let d_f32 = got
                    .iter()
                    .zip(cpu)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let d_single = got
                    .iter()
                    .zip(single)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let single_f32 = single
                    .iter()
                    .zip(cpu)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                println!(
                    "  target row g{g}: cp_vs_f32={d_f32:.4e} cp_vs_single={d_single:.4e} single_vs_f32={single_f32:.4e}"
                );
            }
        }
    }

    println!("cp_hidden_parity cp_size={CP_SIZE} devices={devices:?} seq={SEQ}");
    println!(
        "cp_vs_single={worst:.6e} (worst_row={worst_row})  cp_vs_cpu_f32={cp_vs_cpu:.6e}  single_vs_cpu_f32={single_vs_cpu:.6e}  tol={REL_TOL:.1e}"
    );
    // The real correctness question: is CP as close to f32 as single-card is? If
    // cp_vs_cpu <= single_vs_cpu (within tol), CP is not the wrong one — the
    // cp_vs_single gap is just single-card's bf16-prefill lossiness.
    ensure!(
        cp_vs_cpu <= single_vs_cpu.max(REL_TOL),
        "CP diverges from f32 ground truth MORE than single-card does: cp_vs_cpu={cp_vs_cpu} single_vs_cpu={single_vs_cpu}"
    );
    println!("PASS (CP tracks f32 ground truth at least as well as single-card)");
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn write_rows(dir: &str, rank: usize, rows: &[usize], data: &[f32], hidden: usize) -> Result<()> {
    let mut body = format!("rows={rows:?}\n");
    for (local, &g) in rows.iter().enumerate() {
        let vec = &data[local * hidden..(local + 1) * hidden];
        let csv = vec
            .iter()
            .map(|v| format!("{v:.9e}"))
            .collect::<Vec<_>>()
            .join(",");
        body.push_str(&format!("g{g}={csv}\n"));
    }
    let path = format!("{dir}/rank_{rank}.txt");
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, body)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn read_rows(dir: &str, rank: usize, hidden: usize) -> Result<(Vec<usize>, Vec<f32>)> {
    let path = format!("{dir}/rank_{rank}.txt");
    let text = std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
    let mut rows = Vec::new();
    let mut data = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("rows=") {
            rows = rest
                .trim_matches(|c| c == '[' || c == ']')
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().parse::<usize>().map_err(Into::into))
                .collect::<Result<Vec<_>>>()?;
        } else if let Some((_, csv)) = line.split_once('=') {
            let vals = csv
                .split(',')
                .map(|s| s.parse::<f32>().map_err(Into::into))
                .collect::<Result<Vec<_>>>()?;
            ensure!(vals.len() == hidden, "row width {} != {hidden}", vals.len());
            data.extend(vals);
        }
    }
    Ok((rows, data))
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
fn parse_devices(need: usize) -> Result<Vec<usize>> {
    let raw = std::env::var("ARLE_CPH_CUDA_DEVICES").unwrap_or_else(|_| {
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
        "ARLE_CPH_CUDA_DEVICES has {} entries, need {need}",
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
