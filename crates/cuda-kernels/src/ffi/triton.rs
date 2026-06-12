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

// ============================================================================
// Triton AOT fused-MoE decode lane (opt-in, `ARLE_QWEN35_MOE_FUSED_SGLANG`).
// Adopts SGLang's `fused_moe` triton stack for the Qwen3.6-35B-A3B MoE FFN:
//   moe_align (native, see ffi/moe.rs) -> GEMM1 (gate||up grouped GEMM) ->
//   act_and_mul (silu(gate)*up) -> GEMM2 (down, carries MUL_ROUTED_WEIGHT) ->
//   sum_reduce (top_k partial sum * routed_scaling_factor).
// One `fused_moe_kernel` source compiles to two cubins: GEMM1 (top_k=8,
// MUL_ROUTED_WEIGHT=0) and GEMM2 (top_k=1, MUL_ROUTED_WEIGHT=1). bf16-only
// for now; fp8 is a 1-shape signature add. Same launch + `*_load_cuda`
// convention as the GDN lane. Stubs return CUDA_ERROR_NOT_SUPPORTED off
// sm90 / no-triton / DSv4 builds.
// ============================================================================
#[allow(dead_code)]
unsafe extern "C" {
    /// GEMM1 — `arle_fused_moe.py::fused_moe_kernel` (top_k=8,
    /// MUL_ROUTED_WEIGHT=0, compute_type="bf16"). Grid
    /// (cdiv(EM, 64)*cdiv(N, 64), 1, 1), block num_warps=4, num_stages=3.
    /// `a` = hidden `[M, K]`, `b` = stacked w1 `[E, 2N, K]`, `c` = cache1
    /// `[M*top_k, 2N]`. `topk_weights` is unused in GEMM1 (MUL_ROUTED_WEIGHT=0)
    /// but kept in the signature for the shared kernel.
    pub fn arle_fused_moe_gemm1_cuda(
        g_x: u32,
        g_y: u32,
        g_z: u32,
        a_ptr: *const Half,
        b_ptr: *const Half,
        c_ptr: *mut Half,
        topk_weights_ptr: *const f32,
        sorted_token_ids_ptr: *const i32,
        expert_ids_ptr: *const i32,
        num_tokens_post_padded_ptr: *const i32,
        n: i32,
        k: i32,
        em: i32,
        num_valid_tokens: i32,
        stride_am: i32,
        stride_ak: i32,
        stride_be: i32,
        stride_bk: i32,
        stride_bn: i32,
        stride_cm: i32,
        stride_cn: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn arle_fused_moe_gemm1_load_cuda() -> CUresult;

    /// GEMM2 — `arle_fused_moe.py::fused_moe_kernel` (top_k=1,
    /// MUL_ROUTED_WEIGHT=1, compute_type="bf16"). `a` = act `[M*top_k, N]`,
    /// `b` = stacked w2 `[E, K, N]`, `c` = cache3 `[M*top_k, K]`.
    /// `topk_weights` is applied here (MUL_ROUTED_WEIGHT=1).
    pub fn arle_fused_moe_gemm2_cuda(
        g_x: u32,
        g_y: u32,
        g_z: u32,
        a_ptr: *const Half,
        b_ptr: *const Half,
        c_ptr: *mut Half,
        topk_weights_ptr: *const f32,
        sorted_token_ids_ptr: *const i32,
        expert_ids_ptr: *const i32,
        num_tokens_post_padded_ptr: *const i32,
        n: i32,
        k: i32,
        em: i32,
        num_valid_tokens: i32,
        stride_am: i32,
        stride_ak: i32,
        stride_be: i32,
        stride_bk: i32,
        stride_bn: i32,
        stride_cm: i32,
        stride_cn: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn arle_fused_moe_gemm2_load_cuda() -> CUresult;

    /// act_and_mul — `arle_fused_moe_act.py::act_and_mul_kernel`
    /// (silu, expert_step=1, BLOCK_SIZE=512). Grid (M*top_k, 1, 1),
    /// block num_warps=4. `gateup_output` = cache1 `[M*top_k, 2N]`,
    /// `down_input` = act `[M*top_k, N]`, `hidden_size` = 2N. `expert_ids_ptr`
    /// is the token-major topk_ids row buffer (pad-row skip; never fires on
    /// the decode bf16 path but kept byte-for-byte).
    pub fn arle_fused_moe_act_cuda(
        g_x: u32,
        g_y: u32,
        g_z: u32,
        gateup_output: *const Half,
        down_input: *mut Half,
        hidden_size: i32,
        expert_ids_ptr: *const i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn arle_fused_moe_act_load_cuda() -> CUresult;

    /// sum_reduce — `arle_fused_moe_sum.py::_moe_sum_reduce_kernel`
    /// (BLOCK_M=1, BLOCK_DIM=2048, NUM_STAGE=1, num_warps=16,
    /// routed_scaling_factor baked). Grid (cdiv(M, 1), cdiv(K, 2048), 1).
    /// `input` = cache3 viewed `[M, top_k, K]`, `output` = out `[M, K]`.
    pub fn arle_fused_moe_sum_cuda(
        g_x: u32,
        g_y: u32,
        g_z: u32,
        input_ptr: *const Half,
        input_stride_0: i32,
        input_stride_1: i32,
        input_stride_2: i32,
        output_ptr: *mut Half,
        output_stride_0: i32,
        output_stride_1: i32,
        token_num: i32,
        topk_num: i32,
        hidden_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn arle_fused_moe_sum_load_cuda() -> CUresult;
}
