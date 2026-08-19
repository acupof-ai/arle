//! CP ring TRANSPORT parity: the NCCL ring in `cp_causal_sdpa` vs a full-seq
//! causal SDPA ground truth. The single-GPU kernel test
//! (`device_ring_two_blocks_matches_host_reference_gqa_hd128`) already proved the
//! per-block kernel + merge + GQA + position mask are correct; this isolates what
//! that test bypassed — the live `ring_send_recv_kv` rotation of k/v AND their
//! positions across ranks. Each rank builds the SAME global q/k/v (name-seeded),
//! shards the sequence with the zigzag `SeqShard`, runs the real ring on its
//! shard, and checks its output rows against the full-seq reference it computes
//! locally. A wrong rotation/position transport shows up as a per-rank row
//! mismatch — and reproduces the pod's zigzag-rank-asymmetric 5.2% in seconds,
//! without the 20-min model build.
//!
//! Run on a GPU host (>= 2 GPUs):
//!   ARLE_CP_CUDA_DEVICES=1,3 \
//!     cargo run -p train --release --no-default-features \
//!       --features cuda,nccl --example cp_ring_transport_parity

#[cfg(all(feature = "cuda", feature = "nccl"))]
use anyhow::{Context, Result, ensure};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use autograd::{
    Tensor, TensorStore, backend_cuda::CudaBackend, ops, ops::ring_attention, tape::Tape,
};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use cuda_kernels::ffi::nccl;
#[cfg(all(feature = "cuda", feature = "nccl"))]
use train::context_parallel::CpContext;

#[cfg(all(feature = "cuda", feature = "nccl"))]
const CP_SIZE: usize = 2;
#[cfg(all(feature = "cuda", feature = "nccl"))]
const HEADS: usize = 4;
#[cfg(all(feature = "cuda", feature = "nccl"))]
const HEAD_DIM: usize = 128;
#[cfg(all(feature = "cuda", feature = "nccl"))]
const SEQ: usize = 16; // 2*CP_SIZE divides it (zigzag precondition)
#[cfg(all(feature = "cuda", feature = "nccl"))]
const REL_TOL: f32 = 5.0e-3;

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn main() -> Result<()> {
    if let Ok(rank) = std::env::var("ARLE_CP_RANK") {
        return rank_main(rank.parse().context("ARLE_CP_RANK parse")?);
    }
    coordinator_main()
}

#[cfg(not(all(feature = "cuda", feature = "nccl")))]
fn main() {
    eprintln!("cp_ring_transport_parity requires --features cuda,nccl");
}

// Deterministic global q/k/v as exact bf16 (top 16 mantissa bits) so the device
// round-trip is lossless and the tolerance flags a real transport bug.
#[cfg(all(feature = "cuda", feature = "nccl"))]
fn synth(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let raw = (((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5) * 0.5;
            f32::from_bits(raw.to_bits() & 0xffff_0000)
        })
        .collect()
}

// [1, HEADS, SEQ, HEAD_DIM] global tensors, identical on every rank.
#[cfg(all(feature = "cuda", feature = "nccl"))]
fn global_qkv() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = HEADS * SEQ * HEAD_DIM;
    (synth(n, 1), synth(n, 2), synth(n, 3))
}

// Full-seq causal SDPA over the WHOLE sequence (MHA: kv_heads == heads). This is
// the ground truth every rank compares its shard rows against.
#[cfg(all(feature = "cuda", feature = "nccl"))]
fn reference_out(store: &mut TensorStore) -> Result<Vec<f32>> {
    let (q, k, v) = global_qkv();
    let shape = vec![1, HEADS, SEQ, HEAD_DIM];
    let mut tape = Tape::new();
    tape.set_enabled(false);
    let qid = store.alloc(Tensor::new(q, shape.clone(), false)?);
    let kid = store.alloc(Tensor::new(k, shape.clone(), false)?);
    let vid = store.alloc(Tensor::new(v, shape, false)?);
    let out = ops::causal_sdpa(qid, kid, vid, store, &mut tape)?;
    Ok(store.to_host(out)?)
}

// Slice global row `r`'s vector for head `h` out of a [1,H,S,D] row-major buffer.
#[cfg(all(feature = "cuda", feature = "nccl"))]
fn row_slice(buf: &[f32], h: usize, r: usize) -> &[f32] {
    let base = (h * SEQ + r) * HEAD_DIM;
    &buf[base..base + HEAD_DIM]
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn rank_main(rank: usize) -> Result<()> {
    let dir = std::env::var("ARLE_CP_DIR").context("ARLE_CP_DIR missing")?;
    let device = env_usize("ARLE_CP_CUDA_DEVICE", rank)?;
    let unique_id = nccl_rendezvous(rank, &dir)?;
    let backend = CudaBackend::new_with_nccl(device, unique_id, CP_SIZE, rank)
        .with_context(|| format!("rank {rank} CudaBackend::new_with_nccl device {device}"))?;
    let mut store = TensorStore::with_backend(std::sync::Arc::new(backend));

    // This rank's zigzag shard of the sequence + the absolute position of each
    // local row (the exact data opd.rs threads into the ring).
    let cp = CpContext::from_mesh(1, CP_SIZE, rank);
    let shard = cp.shard(SEQ);
    let rows = shard.local_rows();
    let positions: Vec<usize> = rows.clone();

    // Build this rank's LOCAL q/k/v shard: gather the global rows it owns.
    let (gq, gk, gv) = global_qkv();
    let local_len = rows.len();
    let mut q = vec![0.0f32; HEADS * local_len * HEAD_DIM];
    let mut k = vec![0.0f32; HEADS * local_len * HEAD_DIM];
    let mut v = vec![0.0f32; HEADS * local_len * HEAD_DIM];
    for (local, &r) in rows.iter().enumerate() {
        for h in 0..HEADS {
            let dst = (h * local_len + local) * HEAD_DIM;
            q[dst..dst + HEAD_DIM].copy_from_slice(row_slice(&gq, h, r));
            k[dst..dst + HEAD_DIM].copy_from_slice(row_slice(&gk, h, r));
            v[dst..dst + HEAD_DIM].copy_from_slice(row_slice(&gv, h, r));
        }
    }

    let shape = vec![1, HEADS, local_len, HEAD_DIM];
    let mut tape = Tape::new();
    tape.set_enabled(false);
    let qid = store.alloc(Tensor::new(q, shape.clone(), false)?);
    let kid = store.alloc(Tensor::new(k, shape.clone(), false)?);
    let vid = store.alloc(Tensor::new(v, shape, false)?);

    // The real device ring over NCCL — the path under test.
    let out = ring_attention::cp_causal_sdpa(
        qid,
        kid,
        vid,
        CP_SIZE,
        rank,
        Some(&positions),
        None,
        &mut store,
        &mut tape,
    )
    .context("cp_causal_sdpa")?;
    let got = store.to_host(out)?;

    // Ground truth: full-seq reference, then pick this rank's rows.
    let mut ref_store = TensorStore::default();
    let reference = reference_out(&mut ref_store)?;

    let mut max_diff = 0.0f32;
    let mut worst = (0usize, 0usize);
    for (local, &r) in rows.iter().enumerate() {
        for h in 0..HEADS {
            let g = &got[(h * local_len + local) * HEAD_DIM..][..HEAD_DIM];
            let want = row_slice(&reference, h, r);
            for (a, b) in g.iter().zip(want) {
                let d = (a - b).abs();
                if d > max_diff {
                    max_diff = d;
                    worst = (h, r);
                }
            }
        }
    }
    // Report per-rank so the coordinator (and the log) shows the zigzag asymmetry.
    println!(
        "rank {rank} shard_rows={rows:?} max_diff={max_diff:.6e} worst=head{}row{}",
        worst.0, worst.1
    );
    write_result(&dir, rank, max_diff)
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn coordinator_main() -> Result<()> {
    let devices = parse_devices(CP_SIZE)?;
    let root = std::env::var("ARLE_CP_DIR")
        .unwrap_or_else(|_| format!("/tmp/arle_cp_ring_transport_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).with_context(|| format!("create {root}"))?;

    let exe = std::env::current_exe()?;
    let mut children = Vec::with_capacity(CP_SIZE);
    for (rank, &device) in devices.iter().enumerate().take(CP_SIZE) {
        let child = std::process::Command::new(&exe)
            .env("ARLE_CP_RANK", rank.to_string())
            .env("ARLE_CP_DIR", &root)
            .env("ARLE_CP_CUDA_DEVICE", device.to_string())
            .spawn()
            .with_context(|| format!("spawn CP rank {rank}"))?;
        children.push((rank, child));
    }
    for (rank, mut child) in children {
        let status = child.wait()?;
        ensure!(status.success(), "CP rank {rank} exited {status:?}");
    }

    let mut worst = 0.0f32;
    for rank in 0..CP_SIZE {
        worst = worst.max(read_result(&root, rank)?);
    }
    println!("cp_ring_transport_parity cp_size={CP_SIZE} devices={devices:?} seq={SEQ}");
    println!("worst_rank_max_diff={worst:.6e} tol={REL_TOL:.1e}");
    ensure!(
        worst <= REL_TOL,
        "CP ring transport mismatch: worst rank max_diff={worst} exceeds {REL_TOL}"
    );
    println!("PASS");
    Ok(())
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
fn write_result(dir: &str, rank: usize, max_diff: f32) -> Result<()> {
    let path = format!("{dir}/rank_{rank}.txt");
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, format!("{max_diff:.9e}\n"))?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn read_result(dir: &str, rank: usize) -> Result<f32> {
    let path = format!("{dir}/rank_{rank}.txt");
    let text = std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
    text.trim()
        .parse::<f32>()
        .with_context(|| format!("parse {path}"))
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn parse_devices(need: usize) -> Result<Vec<usize>> {
    let raw = std::env::var("ARLE_CP_CUDA_DEVICES").unwrap_or_else(|_| {
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
        "ARLE_CP_CUDA_DEVICES has {} entries, need {need}",
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
