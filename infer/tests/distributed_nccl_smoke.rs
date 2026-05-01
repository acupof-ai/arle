#![cfg(feature = "nccl")]

#[test]
fn distributed_nccl_smoke() -> anyhow::Result<()> {
    let Ok(gpus) = cudarc::driver::CudaContext::device_count() else {
        eprintln!("skipping NCCL smoke: CUDA device count unavailable");
        return Ok(());
    };
    if gpus < 2 {
        eprintln!("skipping NCCL smoke: requires >=2 GPUs, found {gpus}");
        return Ok(());
    }

    infer::distributed::smoke_2_thread_all_reduce()
}
