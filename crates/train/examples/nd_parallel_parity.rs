//! N-D parallel parity gate: context-parallel (CP) writeback vs single card.
//!
//! Exercises the REAL CP path on GPU — the ring KV forward, its backward, and the
//! post-backward CP grad all-reduce — which no local (CPU) test can reach. The
//! single-card reference and each CP rank run the same name-seeded (deterministic)
//! LoRA model over the same trajectory; CP shards the sequence, and the writeback
//! world-sums the shard partials before returning, so EVERY rank already reports
//! the global-mean CE — the gate reads rank 0, it does not sum.
//!
//! The gate anchors on a CPU-f32 ground truth, NOT single-card byte-identity: both
//! CUDA paths run bf16 attention (chunked-prefill kernel for single card, the ring
//! kernel for CP) and at this tiny random-weight scale sit ~8% off f32. Comparing
//! two bf16-lossy losses to each other at a near-identity tolerance is a
//! miscalibrated gate — it fails on rounding, not on bugs (learned 2026-08-01: the
//! device ring kernel + NCCL transport both parity-PASS against their references,
//! and CP tracks f32 BETTER than single-card, yet a 1e-3 single-card tol "failed"
//! at 5%). So the real invariant is: CP must track f32 at least as well as the
//! single card — a wrong gather / shard filter / position rebase / inv_n drifts CP
//! away from f32 far past `TRACK_MARGIN`; symmetric bf16 noise passes.
//!
//! The gate covers the BACKWARD too, on the same three-way (f32 / single / CP)
//! shape: post-reduce every rank's grad IS the single card's global grad, so their
//! L2 norms must agree. A pod 27B run had matching losses (spread 8.2e-4) but grad
//! norms 3.744990 vs 1.984009 — a 1.89x break the loss-only gate could not see.
//!
//! Full-attention by default; `ARLE_ND_HYBRID=1` makes every layer but the last
//! linear attention, and `ARLE_ND_HYBRID=k` (k>=2) interleaves full attention every
//! k-th layer — `4` is the real 27B's shape (48 of its 64 layers are GDN).
//! `ARLE_ND_LAYERS=n` sets depth, `ARLE_ND_CP_SIZE=n` the world size
//! (= the ring step count) — the two axes a bias has to scale on to matter.
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

// Ring step count is the CP world size, not the sequence length — `ARLE_ND_CP_SIZE`
// is the only axis that varies it (default 2).
#[cfg(all(feature = "cuda", feature = "nccl"))]
fn cp_size() -> usize {
    std::env::var("ARLE_ND_CP_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(2)
}
// CP is CORRECT when it tracks the f32 ground truth at least as well as the
// single card does — NOT when it byte-matches the single card. Both CUDA paths
// run bf16 attention (chunked-prefill kernel vs the ring kernel), which at this
// tiny random-weight scale sits ~8% off f32; comparing two bf16-lossy paths at a
// near-identity tolerance is a miscalibrated gate, not a correctness one. So the
// gate anchors on a CPU-f32 reference and allows CP to be at most `TRACK_MARGIN`
// (loss-relative) farther from f32 than the single card — CE projection +
// f32-reduction-order slack. A real gather/shard/position bug drifts CP from f32
// far past this; symmetric bf16 noise passes.
#[cfg(all(feature = "cuda", feature = "nccl"))]
const TRACK_MARGIN: f32 = 2.0e-2;
// Same tracking rule for the post-reduce grad norm, looser because the backward
// accumulates bf16 error over every layer: 5% swallows that, while a dropped or
// duplicated CP contribution shows up as a RATIO (the pod run: cp=1 3.744990 vs
// cp=2 1.984009, 1.89x = 89% relative) — three orders above the noise.
#[cfg(all(feature = "cuda", feature = "nccl"))]
const GRAD_TRACK_MARGIN: f64 = 5.0e-2;
// Above this local seq the CPU-f32 reference is infeasible (its O(seq^2) attention
// is the very host-RAM wall the >65535 ring case exists to avoid), so that case is
// a LIVENESS gate (the step completes, no OOM) plus a loose bf16-vs-bf16 bound.
#[cfg(all(feature = "cuda", feature = "nccl"))]
const F32_REF_MAX_SEQ: usize = 4096;
#[cfg(all(feature = "cuda", feature = "nccl"))]
const BF16_REL_TOL: f32 = 1.0e-1;
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
// into 2*cp_size()=4 evenly (the zigzag precondition; opd.rs also pads up, but an
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

// `=1` is one full-attn layer atop an all-GDN stack, so no ring gradient crosses a
// layer below it — blind to a defect that compounds down the backward. `=k` (k>=2)
// interleaves every k-th layer; 4 is the 27B's shape.
#[cfg(all(feature = "cuda", feature = "nccl"))]
fn nd_layer_types() -> Vec<LayerType> {
    let depth = std::env::var("ARLE_ND_LAYERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2)
        .max(1);
    let period = std::env::var("ARLE_ND_HYBRID")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(0);
    if period == 0 {
        return vec![LayerType::FullAttention; depth];
    }
    let mut types: Vec<LayerType> = (0..depth)
        .map(|i| {
            if period >= 2 && (i + 1) % period == 0 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();
    types[depth - 1] = LayerType::FullAttention;
    types
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
    let devices = parse_devices(cp_size())?;
    let root = std::env::var("ARLE_ND_DIR")
        .unwrap_or_else(|_| format!("/tmp/arle_nd_parallel_parity_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).with_context(|| format!("create {root}"))?;

    // Single-card reference (no NCCL), in-process on the first device.
    let single = run_writeback(devices[0], CpContext::single())?;
    let loss_single = single.loss;

    // CP ranks: spawn cp_size() workers, each runs the same step under its shard.
    let exe = std::env::current_exe()?;
    let mut children = Vec::with_capacity(cp_size());
    for (rank, &device) in devices.iter().enumerate().take(cp_size()) {
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

    // Neither loss nor grad norm sums across ranks: the writeback world-sums the
    // shard partials before returning (fe6cc2346) and `all_reduce_cp_grads` sums the
    // grads, so every rank already holds the identical global value. A spread across
    // ranks is itself a reduce bug — assert it, then use rank 0 as "the" CP value.
    let mut losses_cp = Vec::with_capacity(cp_size());
    let mut norms_cp = Vec::with_capacity(cp_size());
    for rank in 0..cp_size() {
        let step = read_loss(&root, rank)?;
        losses_cp.push(step.loss);
        norms_cp.push(step.norm);
    }
    let loss_cp = losses_cp[0];
    let norm_cp = norms_cp[0];
    let spread = |vals: &[f64], base: f64| {
        vals.iter()
            .map(|v| (v - base).abs() / base.abs().max(1.0e-12))
            .fold(0.0f64, f64::max)
    };
    let loss_spread = spread(
        &losses_cp.iter().map(|&l| l as f64).collect::<Vec<_>>(),
        loss_cp as f64,
    );
    let rank_spread = spread(&norms_cp, norm_cp);
    ensure!(
        loss_spread <= TRACK_MARGIN as f64,
        "post-reduce CP losses differ across ranks ({losses_cp:?}) — the world sum did not converge"
    );

    println!(
        "nd_parallel_parity cp_size={} devices={devices:?} seq={}",
        cp_size(),
        nd_seq()
    );
    println!("grad_norm_cp_per_rank={norms_cp:?} rank_spread={rank_spread:.3e}");
    ensure!(
        rank_spread <= GRAD_TRACK_MARGIN,
        "post-reduce CP grad norms differ across ranks ({norms_cp:?}) — the all-reduce did not converge"
    );

    // Small seq: anchor on the f32 ground truth. CP is correct iff it tracks f32 at
    // least as well as the single card does (both are bf16 CUDA, ~equally lossy).
    // Comparing the two bf16 losses to each other would false-fail on rounding.
    if nd_seq() <= F32_REF_MAX_SEQ {
        let f32_ref = run_writeback_cpu_f32()?;
        let loss_f32 = f32_ref.loss;
        let denom = loss_f32.abs().max(1.0e-6);
        let cp_vs_f32 = (loss_cp - loss_f32).abs() / denom;
        let single_vs_f32 = (loss_single - loss_f32).abs() / denom;
        println!("loss_f32={loss_f32:.9e} loss_single={loss_single:.9e} loss_cp={loss_cp:.9e}");
        println!(
            "cp_vs_f32={cp_vs_f32:.3e} single_vs_f32={single_vs_f32:.3e} margin={TRACK_MARGIN:.1e}"
        );
        ensure!(
            cp_vs_f32 <= single_vs_f32 + TRACK_MARGIN,
            "CP diverges from f32 more than single-card + margin: cp_vs_f32={cp_vs_f32} single_vs_f32={single_vs_f32}"
        );

        // Same three-way shape for the BACKWARD: single-card and CP run different
        // attention backwards (fused SDPA recompute vs the ring), so when the two
        // bf16 arms disagree only the f32 arm says which one drifted.
        let norm_f32 = f32_ref.norm;
        // A ~zero anchor makes both ratios ~1/eps and the comparison passes on
        // nothing — refuse to certify instead (2026-08-04: the f32 arm returned
        // exactly 0 for full-attention-only configs, see docs/experience/errors/).
        ensure!(
            norm_f32.abs() > 1.0e-9,
            "f32 grad-norm anchor is degenerate ({norm_f32:e}) — the reference backward produced no gradient, so this gate cannot certify CP"
        );
        let n_denom = norm_f32.abs();
        let norm_cp_vs_f32 = (norm_cp - norm_f32).abs() / n_denom;
        let norm_single_vs_f32 = (single.norm - norm_f32).abs() / n_denom;
        println!(
            "grad_norm_f32={norm_f32:.9e} grad_norm_single={:.9e} grad_norm_cp={norm_cp:.9e}",
            single.norm
        );
        println!(
            "grad_cp_vs_f32={norm_cp_vs_f32:.3e} grad_single_vs_f32={norm_single_vs_f32:.3e} margin={GRAD_TRACK_MARGIN:.1e}"
        );
        ensure!(
            norm_cp_vs_f32 <= norm_single_vs_f32 + GRAD_TRACK_MARGIN,
            "CP grad norm diverges from f32 more than single-card + margin: cp_vs_f32={norm_cp_vs_f32} single_vs_f32={norm_single_vs_f32} (cp={norm_cp} single={} f32={norm_f32})",
            single.norm
        );
        println!("PASS (CP loss AND grad norm track the f32 ground truth as well as single-card)");
        return Ok(());
    }

    // Large seq (>F32_REF_MAX_SEQ): the f32 reference is infeasible (O(seq^2) host
    // RAM — the wall the ring avoids). Reaching here already proves LIVENESS (every
    // rank completed a step, no OOM). Fall back to a loose bf16-vs-bf16 bound to
    // catch only a gross gather/shard break, not fine numerics.
    let denom = loss_single.abs().max(loss_cp.abs()).max(1.0e-6);
    let rel_err = (loss_single - loss_cp).abs() / denom;
    println!(
        "loss_single={loss_single:.9e} loss_cp={loss_cp:.9e} rel_err={rel_err:.3e} bf16_tol={BF16_REL_TOL:.1e} (liveness + gross-error gate; f32 ref skipped at seq>{F32_REF_MAX_SEQ})"
    );
    ensure!(
        rel_err <= BF16_REL_TOL,
        "CP loss gross mismatch at large seq: single={loss_single} cp_sum={loss_cp} rel_err={rel_err}"
    );
    // No f32 arm here, so the grad norm falls back to a direct cp-vs-single bound —
    // it can flag a divergence but not attribute it.
    let n_denom = single.norm.abs().max(norm_cp.abs()).max(1.0e-12);
    let norm_rel_err = (single.norm - norm_cp).abs() / n_denom;
    println!(
        "grad_norm_single={:.9e} grad_norm_cp={norm_cp:.9e} rel_err={norm_rel_err:.3e} tol={:.1e}",
        single.norm,
        f64::from(BF16_REL_TOL)
    );
    ensure!(
        norm_rel_err <= f64::from(BF16_REL_TOL),
        "CP grad norm gross mismatch at large seq: single={} cp={norm_cp} rel_err={norm_rel_err}",
        single.norm
    );
    println!("PASS (liveness + gross-error; f32 parity gated at seq<={F32_REF_MAX_SEQ})");
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn rank_main(rank: usize) -> Result<()> {
    let dir = std::env::var("ARLE_ND_DIR").context("ARLE_ND_DIR missing")?;
    let device = env_usize("ARLE_ND_CUDA_DEVICE", rank)?;
    let unique_id = nccl_rendezvous(rank, &dir)?;
    let backend = CudaBackend::new_with_nccl(device, unique_id, cp_size(), rank)
        .with_context(|| format!("rank {rank} CudaBackend::new_with_nccl device {device}"))?;
    let store = TensorStore::with_backend(std::sync::Arc::new(backend));
    // Drive the CP context through the mesh (from_mesh → RankCoord), the path the
    // convergence installed, not a bespoke {rank, size}.
    let step = run_writeback_in(store, CpContext::from_mesh(1, cp_size(), rank))?;
    write_loss(&dir, rank, &step)
}

// One writeback step's two observables: the returned loss and the post-reduce,
// pre-optimizer global grad L2 norm over the trainable params.
#[cfg(all(feature = "cuda", feature = "nccl"))]
struct Step {
    loss: f32,
    norm: f64,
}

// Single-card path builds its own store on `device`; CP ranks pass an NCCL store.
#[cfg(all(feature = "cuda", feature = "nccl"))]
fn run_writeback(device: usize, cp: CpContext) -> Result<Step> {
    let backend =
        CudaBackend::new(device).with_context(|| format!("CudaBackend::new({device})"))?;
    run_writeback_in(TensorStore::with_backend(std::sync::Arc::new(backend)), cp)
}

// The f32 ground truth: the same name-seeded model + trajectory on the CPU
// backend (single card, no NCCL). This is the true loss both bf16 CUDA paths
// approximate; a real CP bug drifts CP away from it, bf16 rounding does not.
#[cfg(all(feature = "cuda", feature = "nccl"))]
fn run_writeback_cpu_f32() -> Result<Step> {
    run_writeback_in(TensorStore::default(), CpContext::single())
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn run_writeback_in(mut store: TensorStore, cp: CpContext) -> Result<Step> {
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
    .context("masked writeback step")
    .and_then(|(loss, _, norm)| {
        let norm = norm.context("writeback returned no grad norm (step_optimizer off?)")?;
        Ok(Step { loss, norm })
    })
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn tiny_full_attn_config() -> Qwen35Config {
    Qwen35Config {
        hidden_size: 512,
        intermediate_size: 16,
        num_hidden_layers: nd_layer_types().len(),
        vocab_size: 16,
        rms_norm_eps: 1.0e-6,
        stop_token_ids: vec![15],
        bos_token_id: Some(1),
        eos_token_id: 15,
        tie_word_embeddings: false,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        // head_dim 256 = the production 27B full-attn dim, and the ONLY dim where
        // `ring_fa3_route` engages — at 128 this gate silently exercised the scalar
        // ring fallback only, so its FA3-vs-scalar A/B was hollow. (Toy dim 2 is also
        // out: the single-card ref would fall back to f32, making parity f32-vs-bf16.)
        head_dim: 256,
        // 8 GDN heads, not 2: linear-attn CP is head-parallel, so the transport
        // refuses cp_size > heads — at 2 the gate could never reach the cp=8 that
        // 256K training targets.
        linear_num_key_heads: 8,
        linear_key_head_dim: 2,
        linear_num_value_heads: 8,
        linear_value_head_dim: 2,
        linear_conv_kernel_dim: 4,
        rope_theta: 10_000.0,
        rope_scaling: None,
        partial_rotary_factor: 1.0,
        rotary_dim: 256,
        // RoPE cache must cover every absolute position: seq (from ARLE_ND_SEQ,
        // padded up by opd.rs) plus headroom. seq=131072 needs 131072 rows, not 16.
        rope_cache_len_hint: Some(nd_seq().next_power_of_two().max(16)),
        layer_types: nd_layer_types(),
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
fn write_loss(dir: &str, rank: usize, step: &Step) -> Result<()> {
    let path = format!("{dir}/rank_{rank}.txt");
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, format!("{:.9e} {:.9e}\n", step.loss, step.norm))?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn read_loss(dir: &str, rank: usize) -> Result<Step> {
    let path = format!("{dir}/rank_{rank}.txt");
    let text = std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
    let mut parts = text.split_whitespace();
    let loss = parts
        .next()
        .context("missing loss")?
        .parse::<f32>()
        .with_context(|| format!("parse loss in {path}"))?;
    let norm = parts
        .next()
        .context("missing grad norm")?
        .parse::<f64>()
        .with_context(|| format!("parse grad norm in {path}"))?;
    Ok(Step { loss, norm })
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
