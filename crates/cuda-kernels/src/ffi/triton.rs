use super::{CUresult, CUstream, Half};

// ============================================================================
// Triton AOT GDN decode lane (opt-in, `ARLE_QWEN35_SGL_GDN`). Three SGLang
// GDN decode kernels compiled to cubins at build time and linked into
// `libtriton_kernels_aot.a`. On non-sm90 / no-triton / DSv4 builds these
// resolve to `CUDA_ERROR_NOT_SUPPORTED` stubs (the runtime loud-fails when the
// flag is on and a stub is hit).
//
// Launch convention: grid (gX, gY, gZ) FIRST, stream LAST. Block dims and
// dynamic shared memory are baked into the generated wrapper (num_warps*32).
// Each kernel also exports a `*_load_cuda` symbol: an idempotent
// cuModuleLoadData + cuModuleGetFunction called once at executor init (where a
// CUDA context is current) so no module load happens inside a CUDA-graph
// capture.
// ============================================================================
#[allow(dead_code)]
unsafe extern "C" {
    /// K1 — `arle_gdn_fused_recurrent.py::arle_gdn_fused_recurrent_decode`.
    /// Grid (V/BV=4, B*HV, 1), block num_warps=1. `state_ptrs` is a u64 device
    /// table of one `[HV, K, V]` f32 state base per batch row (in-place update).
    /// `scale` = key_dim^-0.5.
    pub fn arle_gdn_fused_recurrent_decode_cuda(
        g_x: u32,
        g_y: u32,
        g_z: u32,
        mixed_qkv: *const Half,
        a_proj: *const Half,
        b_proj: *const Half,
        a_log: *const f32,
        dt_bias: *const Half,
        o: *mut Half,
        state_ptrs: *const u64,
        scale: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn arle_gdn_fused_recurrent_decode_load_cuda() -> CUresult;

    /// K2 — `arle_gdn_conv1d_update.py::arle_gdn_conv1d_update`.
    /// Grid (batch, dim/BLOCK_N=32, 1), block num_warps=4. `conv_state_ptrs` is
    /// a u64 device table of one `[dim, state_len]` bf16 ring per batch row;
    /// x/o are `[batch, dim]` token-contiguous bf16.
    pub fn arle_gdn_conv1d_update_cuda(
        g_x: u32,
        g_y: u32,
        g_z: u32,
        x: *const Half,
        w: *const Half,
        conv_state_ptrs: *const u64,
        o: *mut Half,
        batch: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn arle_gdn_conv1d_update_load_cuda() -> CUresult;

    /// K3 — `arle_gdn_rms_norm_gated.py::arle_gdn_rms_norm_gated`.
    /// Grid (cdiv(M, ROWS_PER_BLOCK=4), ngroups=1, 1), block num_warps=1.
    /// M = tokens*HV rows over `[M, 128]` views. `weight` is f32 (our norm
    /// weight buffer dtype). `rstd` is a preallocated `[>=M]` f32 scratch.
    pub fn arle_gdn_rms_norm_gated_cuda(
        g_x: u32,
        g_y: u32,
        g_z: u32,
        x: *const Half,
        y: *mut Half,
        weight: *const f32,
        z: *const Half,
        rstd: *mut f32,
        m: i32,
        eps: f32,
        stream: CUstream,
    ) -> CUresult;

    pub fn arle_gdn_rms_norm_gated_load_cuda() -> CUresult;
}
