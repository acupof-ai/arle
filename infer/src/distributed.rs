//! Distributed coordination primitives for multi-GPU / multi-host execution.
//!
//! F0/F1 foundation: TCP rendezvous (`init_method`), NCCL smoke (`nccl`),
//! pure parallel-state group layout, and group coordinator metadata — see
//! [`docs/plans/2026-04-28-single-node-multi-gpu.md`](../../docs/plans/2026-04-28-single-node-multi-gpu.md).

pub mod group_coordinator;
pub mod init_method;
#[cfg(feature = "nccl")]
pub mod nccl;
pub mod parallel_state;

pub use init_method::{EnvRendezvousConfig, RendezvousClient, RendezvousServer, UNIQUE_ID_BYTES};

#[cfg(feature = "nccl")]
pub fn smoke_2_thread_all_reduce() -> anyhow::Result<()> {
    use std::net::TcpListener;
    use std::thread;

    use anyhow::{Context, bail};
    use nccl::{NcclGroup, NcclInitMethod};

    let device_count = cudarc::driver::CudaContext::device_count()
        .context("failed to query CUDA device count for NCCL smoke")?
        as usize;
    if device_count < 2 {
        bail!("NCCL smoke requires at least 2 CUDA devices, found {device_count}");
    }

    let probe = TcpListener::bind("127.0.0.1:0").context("failed to reserve NCCL TCP port")?;
    let addr = probe
        .local_addr()
        .context("failed to read reserved NCCL TCP addr")?;
    drop(probe);

    let mut handles = Vec::with_capacity(2);
    for rank in 0..2 {
        handles.push(thread::spawn(move || -> anyhow::Result<Vec<f32>> {
            let group = NcclGroup::new(rank, 2, NcclInitMethod::TcpStore(addr))?;
            let input = if rank == 0 {
                vec![1.0_f32, 2.0, 3.0]
            } else {
                vec![4.0_f32, 5.0, 6.0]
            };
            group.all_reduce_smoke(&input)
        }));
    }

    for (rank, handle) in handles.into_iter().enumerate() {
        let output = handle
            .join()
            .map_err(|_| anyhow::anyhow!("NCCL smoke rank {rank} thread panicked"))??;
        if output != [5.0_f32, 7.0, 9.0] {
            bail!("NCCL smoke rank {rank} returned {output:?}, expected [5.0, 7.0, 9.0]");
        }
    }

    Ok(())
}
