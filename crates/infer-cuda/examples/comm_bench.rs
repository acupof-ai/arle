//! Isolated small-message collective bench — one-shot custom AR/AG vs NCCL.
//!
//! Measures the EXPOSED latency of the collectives sitting on the DSv4 decode
//! critical path (2× allreduce + 1× Q-allgather per layer; bf16, 14–450 KB).
//!
//! Arms:
//!   - `nccl`      — plain ncclAllReduce / ncclAllGather (B0 baseline)
//!   - `nccl_sym`  — same calls over `ncclMemAlloc`+`ncclCommWindowRegister`
//!                   symmetric windows (NCCL ≥ 2.27 low-latency kernels, C4)
//!   - `car_1stage`/`car_2stage` — vendored sgl-kernel/vLLM custom allreduce (C2)
//!   - `car_ag`    — one-shot all-gather on the same IPC framework
//!
//! Timing: per arm × shape, K-iteration stream loop bracketed by CUDA events,
//! in two modes — `exposed` (a 1-thread chain kernel makes iteration k+1's
//! input depend on iteration k's output; chain cost measured separately and
//! subtracted) and `b2b` (back-to-back issue). R repeats; p50 of repeats.
//!
//! Correctness per arm: output vs the NCCL reference (tolerance), plus a
//! cross-rank FNV-1a digest allgather asserting every rank holds IDENTICAL
//! bytes (the lockstep requirement one-shot guarantees by fixed-order
//! accumulation).
//!
//! Launch (coordinator spawns ranks 1..N, runs rank 0 itself):
//!   WORLD_SIZE=8 ./comm_bench
//! Env: `ARLE_COMM_BENCH_DIR` rendezvous dir (default /tmp/arle_comm_bench),
//! `ARLE_COMM_BENCH_ITERS` (default 200), `ARLE_COMM_BENCH_REPEATS` (default 5).

#![cfg(all(feature = "cuda", feature = "nccl"))]

use anyhow::{Context, Result, anyhow, ensure};
use cuda_kernels::collective::{CollectiveBackend, DType, NcclBackend, ReduceOp};
use cuda_kernels::ffi::{comm as car, nccl};
use cuda_kernels::prelude::DeviceContext;
use cudarc::driver::sys::CUevent_flags;

const MAX_BYTES: usize = 1 << 20; // 1 MiB registered region (≥ all shapes)
const HIDDEN: usize = 7168;

fn main() -> Result<()> {
    let world: usize = std::env::var("WORLD_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    match std::env::var("ARLE_COMM_BENCH_RANK") {
        Ok(r) => rank_main(r.parse().context("ARLE_COMM_BENCH_RANK parse")?, world),
        Err(_) => coordinator_main(world),
    }
}

/// Spawn ranks 1..N as children (same exe), then run rank 0 in-process so the
/// bench is one command on the pod. Children inherit the rendezvous dir.
fn coordinator_main(world: usize) -> Result<()> {
    let dir = bench_dir();
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).with_context(|| format!("create {dir}"))?;

    let exe = std::env::current_exe()?;
    let mut children = Vec::new();
    for rank in 1..world {
        let child = std::process::Command::new(&exe)
            .env("ARLE_COMM_BENCH_RANK", rank.to_string())
            .env("WORLD_SIZE", world.to_string())
            .env("INFER_CUDA_DEVICE", rank.to_string())
            .spawn()
            .with_context(|| format!("spawn bench rank {rank}"))?;
        children.push((rank, child));
    }
    // SAFETY: single-threaded coordinator, pre-CUDA-init.
    unsafe {
        std::env::set_var("INFER_CUDA_DEVICE", "0");
    }
    let result = rank_main(0, world);
    for (rank, mut child) in children {
        let status = child.wait()?;
        ensure!(status.success(), "bench rank {rank} exited {status:?}");
    }
    result
}

fn bench_dir() -> String {
    std::env::var("ARLE_COMM_BENCH_DIR").unwrap_or_else(|_| "/tmp/arle_comm_bench".to_string())
}

struct RankCtx {
    rank: usize,
    world: usize,
    ctx: DeviceContext,
    backend: NcclBackend,
    car: *mut std::os::raw::c_void,
    car_in: u64,
    car_out: u64,
    sym: Option<SymPair>,
}

struct SymPair {
    input: *mut std::ffi::c_void,
    output: *mut std::ffi::c_void,
    win_in: nccl::ncclWindow_t,
    win_out: nccl::ncclWindow_t,
}

fn rank_main(rank: usize, world: usize) -> Result<()> {
    let dir = bench_dir();
    let ctx = DeviceContext::new()?; // honours INFER_CUDA_DEVICE
    // NCCL rendezvous: rank 0 mints the unique id into the dir; others poll.
    let unique_id = nccl_rendezvous(rank, &dir)?;
    let backend = NcclBackend::init_rank(unique_id, world, rank)?;

    // Custom-AR bootstrap (IPC handle exchange through the same dir).
    let cdir = std::ffi::CString::new(dir.clone())?;
    let car_handle =
        unsafe { car::arle_car_bootstrap(rank as i32, world as i32, cdir.as_ptr(), MAX_BYTES) };
    ensure!(
        !car_handle.is_null(),
        "rank {rank} custom-AR bootstrap failed"
    );
    let car_in = unsafe { car::arle_car_input_ptr(car_handle) };
    let car_out = unsafe { car::arle_car_output_ptr(car_handle) };

    // NCCL symmetric windows (C4) — optional: skip arm if registration fails
    // (e.g. runtime NCCL < 2.27). COLLECTIVE: all ranks attempt together.
    let sym = match unsafe { backend.mem_alloc_symmetric_window(MAX_BYTES) } {
        Ok((input, win_in)) => match unsafe { backend.mem_alloc_symmetric_window(MAX_BYTES) } {
            Ok((output, win_out)) => Some(SymPair {
                input,
                output,
                win_in,
                win_out,
            }),
            Err(err) => {
                eprintln!("[comm-bench rank={rank}] symmetric output window failed: {err:#}");
                unsafe { backend.free_symmetric_window(input, win_in)? };
                None
            }
        },
        Err(err) => {
            eprintln!("[comm-bench rank={rank}] symmetric window unavailable: {err:#}");
            None
        }
    };

    let mut rc = RankCtx {
        rank,
        world,
        ctx,
        backend,
        car: car_handle,
        car_in,
        car_out,
        sym,
    };

    run_matrix(&mut rc)?;

    if let Some(sym) = rc.sym.take() {
        unsafe {
            rc.backend.free_symmetric_window(sym.input, sym.win_in)?;
            rc.backend.free_symmetric_window(sym.output, sym.win_out)?;
        }
    }
    unsafe { car::arle_car_destroy(rc.car) };
    Ok(())
}

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

#[derive(Clone, Copy, PartialEq)]
enum Op {
    AllReduce,
    AllGather,
}

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Nccl,
    NcclSym,
    Car1Stage,
    Car2Stage,
    CarAg,
}

impl Arm {
    fn label(self) -> &'static str {
        match self {
            Arm::Nccl => "nccl",
            Arm::NcclSym => "nccl_sym",
            Arm::Car1Stage => "car_1stage",
            Arm::Car2Stage => "car_2stage",
            Arm::CarAg => "car_ag",
        }
    }
}

fn run_matrix(rc: &mut RankCtx) -> Result<()> {
    let iters: usize = std::env::var("ARLE_COMM_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let repeats: usize = std::env::var("ARLE_COMM_BENCH_REPEATS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    if rc.rank == 0 {
        println!(
            "# comm_bench world={} iters={iters} repeats={repeats} (exposed = chain-corrected p50 µs/op; b2b = back-to-back)",
            rc.world
        );
        println!(
            "| op | shape | bytes | arm | exposed p50 µs | b2b p50 µs | vs nccl | identical-across-ranks | max|Δ| vs nccl |"
        );
        println!("|---|---|---|---|---|---|---|---|---|");
    }

    // Decode AR shapes: [N, 7168] bf16, N = batch rows.
    let ar_shapes: Vec<(String, usize)> = [1usize, 2, 4, 8, 16, 32]
        .iter()
        .map(|n| (format!("[{n},{HIDDEN}]"), n * HIDDEN))
        .collect();
    // Q-allgather shapes: per-rank contribution (FlashMLA: local q heads).
    let ag_shapes: Vec<(String, usize)> = vec![
        ("ag[2048/rank]".to_string(), 2048),
        ("ag[9216/rank]".to_string(), 9216),
    ];

    for (label, elems) in &ar_shapes {
        let mut nccl_p50 = f64::NAN;
        let mut nccl_ref: Vec<u16> = Vec::new();
        for arm in [Arm::Nccl, Arm::NcclSym, Arm::Car1Stage, Arm::Car2Stage] {
            if arm == Arm::NcclSym && rc.sym.is_none() {
                continue;
            }
            let r = bench_one(rc, Op::AllReduce, arm, *elems, iters, repeats, &nccl_ref)?;
            if arm == Arm::Nccl {
                nccl_p50 = r.exposed_p50_us;
                nccl_ref = r.output.clone();
            }
            print_row(rc.rank, Op::AllReduce, label, *elems * 2, arm, &r, nccl_p50);
        }
    }
    for (label, per_rank) in &ag_shapes {
        let mut nccl_p50 = f64::NAN;
        let mut nccl_ref: Vec<u16> = Vec::new();
        for arm in [Arm::Nccl, Arm::NcclSym, Arm::CarAg] {
            if arm == Arm::NcclSym && rc.sym.is_none() {
                continue;
            }
            let r = bench_one(rc, Op::AllGather, arm, *per_rank, iters, repeats, &nccl_ref)?;
            if arm == Arm::Nccl {
                nccl_p50 = r.exposed_p50_us;
                nccl_ref = r.output.clone();
            }
            print_row(
                rc.rank,
                Op::AllGather,
                label,
                *per_rank * 2,
                arm,
                &r,
                nccl_p50,
            );
        }
    }
    Ok(())
}

struct ArmResult {
    exposed_p50_us: f64,
    b2b_p50_us: f64,
    identical_across_ranks: bool,
    max_abs_delta_vs_ref: f32,
    output: Vec<u16>,
}

#[allow(clippy::too_many_arguments)]
fn print_row(
    rank: usize,
    op: Op,
    shape: &str,
    bytes: usize,
    arm: Arm,
    r: &ArmResult,
    nccl_p50: f64,
) {
    if rank != 0 {
        return;
    }
    let opname = match op {
        Op::AllReduce => "AR",
        Op::AllGather => "AG",
    };
    let speedup = if nccl_p50.is_nan() || arm == Arm::Nccl {
        "1.00×".to_string()
    } else {
        format!("{:.2}×", nccl_p50 / r.exposed_p50_us)
    };
    println!(
        "| {opname} | {shape} | {bytes} | {} | {:.1} | {:.1} | {speedup} | {} | {:.4} |",
        arm.label(),
        r.exposed_p50_us,
        r.b2b_p50_us,
        if r.identical_across_ranks {
            "yes"
        } else {
            "**NO**"
        },
        r.max_abs_delta_vs_ref,
    );
}

/// Pointers an arm reads/writes for one op invocation.
fn arm_ptrs(rc: &RankCtx, arm: Arm) -> (u64, u64) {
    match arm {
        Arm::Nccl | Arm::Car1Stage | Arm::Car2Stage | Arm::CarAg => (rc.car_in, rc.car_out),
        Arm::NcclSym => {
            let sym = rc.sym.as_ref().expect("sym arm gated above");
            (sym.input as u64, sym.output as u64)
        }
    }
}

fn issue_op(rc: &RankCtx, op: Op, arm: Arm, elems: usize) -> Result<()> {
    let stream = rc.ctx.stream.cu_stream() as *mut std::ffi::c_void;
    let (input, output) = arm_ptrs(rc, arm);
    match (op, arm) {
        (Op::AllReduce, Arm::Nccl) | (Op::AllReduce, Arm::NcclSym) => unsafe {
            rc.backend.all_reduce_out_of_place(
                input as *const _,
                output as *mut _,
                elems,
                DType::BF16,
                ReduceOp::Sum,
                stream,
            )
        },
        (Op::AllReduce, Arm::Car1Stage) | (Op::AllReduce, Arm::Car2Stage) => {
            let force = if arm == Arm::Car1Stage { 1 } else { 2 };
            let res = unsafe {
                car::arle_car_allreduce_bf16(
                    rc.car,
                    rc.ctx.stream.cu_stream(),
                    elems as i32,
                    512,
                    36,
                    force,
                )
            };
            ensure!(
                res == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                "custom AR failed: {res:?}"
            );
            Ok(())
        }
        (Op::AllGather, Arm::Nccl) | (Op::AllGather, Arm::NcclSym) => unsafe {
            rc.backend.all_gather(
                input as *const _,
                output as *mut _,
                elems,
                DType::BF16,
                stream,
            )
        },
        (Op::AllGather, Arm::CarAg) => {
            let res = unsafe {
                car::arle_car_allgather_bf16(
                    rc.car,
                    rc.ctx.stream.cu_stream(),
                    elems as i32,
                    512,
                    36,
                )
            };
            ensure!(
                res == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                "custom AG failed: {res:?}"
            );
            Ok(())
        }
        _ => Err(anyhow!("unsupported op/arm combination")),
    }
}

fn bench_one(
    rc: &mut RankCtx,
    op: Op,
    arm: Arm,
    elems: usize,
    iters: usize,
    repeats: usize,
    nccl_ref: &[u16],
) -> Result<ArmResult> {
    let out_elems = match op {
        Op::AllReduce => elems,
        Op::AllGather => elems * rc.world,
    };
    ensure!(
        out_elems * 2 <= MAX_BYTES,
        "shape exceeds registered region"
    );
    let (input, output) = arm_ptrs(rc, arm);
    let stream = &rc.ctx.stream;

    // Deterministic fill (per-rank seed), then one correctness invocation.
    let res =
        unsafe { car::arle_car_fill_bf16(stream.cu_stream(), input, elems as i32, rc.rank as i32) };
    ensure!(
        res == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
        "fill failed"
    );
    issue_op(rc, op, arm, elems)?;
    stream.synchronize().map_err(|e| anyhow!("sync: {e}"))?;
    let mut host: Vec<u16> = vec![0; out_elems];
    unsafe {
        cudarc::driver::result::memcpy_dtoh_sync(&mut host, output)
            .map_err(|e| anyhow!("dtoh: {e}"))?;
    }

    // Cross-rank identity: allgather an FNV-1a digest of the output bytes.
    let digest = fnv1a(&host);
    let digests = rc
        .backend
        .all_gather_bytes(&rc.ctx, &digest.to_le_bytes(), 8)?;
    let identical = digests.chunks(8).all(|c| c == digest.to_le_bytes());

    // Numeric delta vs the NCCL reference output (arm `nccl` passes empty ref).
    let max_abs_delta = if nccl_ref.len() == host.len() {
        host.iter()
            .zip(nccl_ref)
            .map(|(&a, &b)| (bf16_to_f32(a) - bf16_to_f32(b)).abs())
            .fold(0.0f32, f32::max)
    } else {
        0.0
    };

    // Timed loops. Chain kernel cost is measured separately and subtracted
    // from the exposed mode so the number is the op's own exposed latency.
    let exposed = timed_loop(rc, op, arm, elems, iters, repeats, true)?;
    let chain_only = timed_chain_only(rc, iters, repeats)?;
    let b2b = timed_loop(rc, op, arm, elems, iters, repeats, false)?;

    Ok(ArmResult {
        exposed_p50_us: (exposed - chain_only).max(0.0),
        b2b_p50_us: b2b,
        identical_across_ranks: identical,
        max_abs_delta_vs_ref: max_abs_delta,
        output: host,
    })
}

fn timed_loop(
    rc: &RankCtx,
    op: Op,
    arm: Arm,
    elems: usize,
    iters: usize,
    repeats: usize,
    chained: bool,
) -> Result<f64> {
    let (input, output) = arm_ptrs(rc, arm);
    let mut per_op_us: Vec<f64> = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        // Warmup outside the events.
        for _ in 0..10 {
            issue_op(rc, op, arm, elems)?;
        }
        let (start, stop) = events(&rc.ctx)?;
        start
            .record(&rc.ctx.stream)
            .map_err(|e| anyhow!("record start: {e}"))?;
        for _ in 0..iters {
            issue_op(rc, op, arm, elems)?;
            if chained {
                let res =
                    unsafe { car::arle_car_chain_touch(rc.ctx.stream.cu_stream(), output, input) };
                ensure!(
                    res == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                    "chain failed"
                );
            }
        }
        stop.record(&rc.ctx.stream)
            .map_err(|e| anyhow!("record stop: {e}"))?;
        stop.synchronize().map_err(|e| anyhow!("sync stop: {e}"))?;
        let ms = start
            .elapsed_ms(&stop)
            .map_err(|e| anyhow!("elapsed: {e}"))? as f64;
        per_op_us.push(ms * 1000.0 / iters as f64);
    }
    per_op_us.sort_by(|a, b| a.total_cmp(b));
    Ok(per_op_us[per_op_us.len() / 2])
}

/// Chain-kernel-only loop: the dependency hook's own cost, subtracted from the
/// chained timing so `exposed` isolates the collective.
fn timed_chain_only(rc: &RankCtx, iters: usize, repeats: usize) -> Result<f64> {
    let mut per_op_us: Vec<f64> = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let (start, stop) = events(&rc.ctx)?;
        start
            .record(&rc.ctx.stream)
            .map_err(|e| anyhow!("record start: {e}"))?;
        for _ in 0..iters {
            let res = unsafe {
                car::arle_car_chain_touch(rc.ctx.stream.cu_stream(), rc.car_out, rc.car_in)
            };
            ensure!(
                res == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                "chain failed"
            );
        }
        stop.record(&rc.ctx.stream)
            .map_err(|e| anyhow!("record stop: {e}"))?;
        stop.synchronize().map_err(|e| anyhow!("sync stop: {e}"))?;
        let ms = start
            .elapsed_ms(&stop)
            .map_err(|e| anyhow!("elapsed: {e}"))? as f64;
        per_op_us.push(ms * 1000.0 / iters as f64);
    }
    per_op_us.sort_by(|a, b| a.total_cmp(b));
    Ok(per_op_us[per_op_us.len() / 2])
}

fn events(ctx: &DeviceContext) -> Result<(cudarc::driver::CudaEvent, cudarc::driver::CudaEvent)> {
    let start = ctx
        .ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|e| anyhow!("create start event: {e}"))?;
    let stop = ctx
        .ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|e| anyhow!("create stop event: {e}"))?;
    Ok((start, stop))
}

fn fnv1a(data: &[u16]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in data {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
