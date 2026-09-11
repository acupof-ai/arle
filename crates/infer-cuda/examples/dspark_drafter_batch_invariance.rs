//! DSpark whole-drafter-step batch-invariance gate (dev-only).
//!
//! Runs the SAME target sequence through the real DSpark drafter once alone
//! (`dspark_draft_block`, GEMM rows = block) and once in an 8-slot batch
//! (`dspark_draft_blocks`, GEMM rows = 8*block) with seven other LIVE slots
//! holding different contexts, at target batch index 0 and 3. Compares the
//! target's draft logits (rel-L2) and greedy draft tokens at every draft row.
//!
//! The head is synthetic at the served geometry (40 q / 8 kv heads, head_dim
//! 128, block 7 by default) — batch invariance is independent of weight values,
//! so no checkpoint or trunk is loaded. Pass `--config <path>` to read q/kv
//! heads, head_dim, block_size, sliding_window and max_position_embeddings
//! from a served DSpark config.json.
//!
//! `--negative-control` adds two families that MUST move the target beyond the
//! bound: swap two slots' start positions, and separately exchange their K/V
//! ring base pointers.
//!
//! Run on a pod:
//! `INFER_CUDA_DEVICE=<free-gpu> target/release/examples/dspark_drafter_batch_invariance`

fn main() -> anyhow::Result<()> {
    if std::env::args().any(|a| a == "--kernel-build-id") {
        println!("{}", cuda_kernels::KERNEL_BUILD_ID);
        return Ok(());
    }
    let negative = std::env::args().any(|a| a == "--negative-control");
    let config = std::env::args()
        .position(|a| a == "--config")
        .and_then(|i| std::env::args().nth(i + 1))
        .map(std::path::PathBuf::from);

    let executor = infer_cuda::CudaExecutor::new();
    let report = executor.dspark_drafter_batch_invariance(config.as_deref(), negative)?;
    for line in report.lines {
        println!("{line}");
    }
    if !report.ok {
        std::process::exit(1);
    }
    Ok(())
}
