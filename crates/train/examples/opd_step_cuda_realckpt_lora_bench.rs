#![cfg_attr(
    not(all(feature = "cuda", not(feature = "no-cuda"))),
    allow(dead_code, unused_imports)
)]

#[cfg_attr(all(feature = "cuda", not(feature = "no-cuda")), allow(dead_code))]
#[path = "opd_step_cuda_realckpt_train.rs"]
mod realckpt_train;

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    realckpt_train::app::main_lora_rank16()
}

#[cfg(not(all(feature = "cuda", not(feature = "no-cuda"))))]
fn main() -> std::process::ExitCode {
    eprintln!(
        "opd_step_cuda_realckpt_lora_bench requires CUDA. Run with: \
         cargo run -p train --example opd_step_cuda_realckpt_lora_bench --release --features cuda"
    );
    std::process::ExitCode::FAILURE
}
