//! FFI for the vendored one-shot custom collective kernels
//! (`csrc/comm/custom_all_reduce.cu` — sgl-kernel/vLLM lineage + ARLE C ABI).
//!
//! Bench-scoped surface (the isolated comm bench drives these directly);
//! production integration goes through a `CollectiveBackend` impl later.

use cudarc::driver::sys::{CUresult, CUstream};

unsafe extern "C" {
    /// File-rendezvous bootstrap of one rank's custom-AR context (allocates the
    /// IPC-shared signal/input regions, exchanges `cudaIpcMemHandle_t`s through
    /// `dir`, opens peers, registers the shared input set). Null on failure.
    pub fn arle_car_bootstrap(
        rank: i32,
        world: i32,
        dir: *const std::os::raw::c_char,
        max_bytes: usize,
    ) -> *mut std::os::raw::c_void;

    /// Raw device pointer of this rank's registered input buffer.
    pub fn arle_car_input_ptr(handle: *mut std::os::raw::c_void) -> u64;
    /// Raw device pointer of this rank's local output buffer.
    pub fn arle_car_output_ptr(handle: *mut std::os::raw::c_void) -> u64;

    /// bf16 sum allreduce input→output. `force_algo`: 0 auto, 1 one-shot,
    /// 2 two-shot.
    pub fn arle_car_allreduce_bf16(
        handle: *mut std::os::raw::c_void,
        stream: CUstream,
        elems: i32,
        threads: i32,
        block_limit: i32,
        force_algo: i32,
    ) -> CUresult;

    /// bf16 one-shot all-gather: output = [world × per_rank_elems] in rank
    /// order, reading every rank's registered input via P2P.
    pub fn arle_car_allgather_bf16(
        handle: *mut std::os::raw::c_void,
        stream: CUstream,
        per_rank_elems: i32,
        threads: i32,
        block_limit: i32,
    ) -> CUresult;

    /// Deterministic per-seed bf16 fill (correctness reference inputs).
    pub fn arle_car_fill_bf16(stream: CUstream, ptr: u64, elems: i32, seed: i32) -> CUresult;

    /// 1-thread dependency hook: `input[0] = output[0]` — serializes timed
    /// iterations so the loop measures exposed latency, not pipelined issue.
    pub fn arle_car_chain_touch(stream: CUstream, output: u64, input: u64) -> CUresult;

    pub fn arle_car_destroy(handle: *mut std::os::raw::c_void);
}
