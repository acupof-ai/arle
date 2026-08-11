//! HIP kernel build + FFI layer for the AIPC DSv4-Flash 2-bit lane
//! (#76/#77).
//!
//! Kernel objects only exist when built on a ROCm box (`--features hip`
//! with hipcc installed — pending-remote); everywhere else the crate is
//! the always-compiled layout layer (row-bytes helpers below) plus stubs
//! that return [`NOT_COMPILED`]. With `--features hip` but no hipcc,
//! build.rs warns and skips so `cargo check --features hip` stays green
//! on the Mac (mirror of the `cuda,no-cuda` lane).
//!
//! Kernel corpus: `csrc/iq2_mmvq.cu` (IQ2_XXS / Q2_K mat-vec + q8_1
//! activation quantization, adapted from llama.cpp @ d2462f8f) and the six
//! shim-portable DSv4 sources from `crates/cuda-kernels/csrc/`, compiled
//! by hipcc with `csrc/arle_hip_shim.h` force-included.

/// Per-block byte sizes, pinned against the vendored
/// `vendor/llama.cpp/ggml-common.h` struct definitions and mirrored by the
/// `static_assert`s in `csrc/iq2_mmvq.cu`:
/// - `block_iq2_xxs` (ggml-common.h:368-375): half d + (QK_K/8)*u16 qs = 66
/// - `block_q2_K` (ggml-common.h:286-299): (QK_K/16) scales + (QK_K/4) qs + 2*half = 84
/// - `block_q8_1` (ggml-common.h:249-259): half2 ds + QK8_1 i8 qs = 36
pub const BLOCK_IQ2_XXS_BYTES: usize = 66;
pub const BLOCK_Q2_K_BYTES: usize = 84;
pub const BLOCK_Q8_1_BYTES: usize = 36;

/// Weights per super-block for the two 2-bit formats (ggml-common.h:89).
pub const QK_K: usize = 256;
/// Values per q8_1 activation block (ggml-common.h:248).
pub const QK8_1: usize = 32;

/// Bytes of one IQ2_XXS weight row. `None` unless `ncols % QK_K == 0`
/// (Option over assert: callers size buffers from untrusted model dims).
pub const fn iq2_xxs_row_bytes(ncols: usize) -> Option<usize> {
    if ncols == 0 || !ncols.is_multiple_of(QK_K) {
        return None;
    }
    Some(ncols / QK_K * BLOCK_IQ2_XXS_BYTES)
}

/// Bytes of one Q2_K weight row. `None` unless `ncols % QK_K == 0`.
pub const fn q2_k_row_bytes(ncols: usize) -> Option<usize> {
    if ncols == 0 || !ncols.is_multiple_of(QK_K) {
        return None;
    }
    Some(ncols / QK_K * BLOCK_Q2_K_BYTES)
}

/// Bytes of one q8_1 activation row. `None` unless `ncols % QK8_1 == 0`.
pub const fn q8_1_row_bytes(ncols: usize) -> Option<usize> {
    if ncols == 0 || !ncols.is_multiple_of(QK8_1) {
        return None;
    }
    Some(ncols / QK8_1 * BLOCK_Q8_1_BYTES)
}

/// Raw HIP error code from a kernel launcher. `0` is `hipSuccess`;
/// [`NOT_COMPILED`] is our stub-build sentinel, never produced by the
/// runtime. Local on purpose — no dependency on `hip-sys`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelError(pub i32);

/// Sentinel returned by every stub entry point when built without `hip`.
pub const NOT_COMPILED: KernelError = KernelError(-1);

pub type Result<T> = std::result::Result<T, KernelError>;

impl std::fmt::Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if *self == NOT_COMPILED {
            return write!(
                f,
                "HIP kernels not compiled (build with --features hip on a ROCm box)"
            );
        }
        write!(f, "HIP kernel error {}", self.0)
    }
}

impl std::error::Error for KernelError {}

/// Basic-op launcher declarations shared by the real and stub `ffi` modules
/// (signatures mirrored 1:1 from the cuda-kernels sources named per block;
/// `__nv_bfloat16*` crosses as `*const/mut u16`, `cudaStream_t`/`CUstream`
/// as `*mut c_void`, `cudaError_t`/`CUresult` as `i32`).
macro_rules! basic_op_launchers {
    ($decl:ident) => {
        // crates/cuda-kernels/csrc/misc/elementwise_basic.cu
        $decl!(silu_mul_cuda(
            gate: *const u16, up: *const u16, out: *mut u16, n: i32,
            stream: *mut c_void,
        ));
        $decl!(dsv4_swiglu_clamped_cuda(
            gate: *const u16, up: *const u16, out: *mut u16, n: i32,
            limit: f32, stream: *mut c_void,
        ));
        $decl!(add_scaled_row_cuda(
            row: *const u16, out: *mut u16, hidden_dim: i32, token_idx: i32,
            scale: f32, stream: *mut c_void,
        ));
        $decl!(add_scaled_row_segment_cuda(
            row: *const u16, out: *mut u16, row_len: i32, out_hidden_dim: i32,
            token_idx: i32, segment_offset: i32, scale: f32, stream: *mut c_void,
        ));
        $decl!(add_cuda(
            a: *const u16, b: *const u16, out: *mut u16, n: i32,
            stream: *mut c_void,
        ));
        $decl!(add_assign_cuda(
            a: *mut u16, b: *const u16, n: i32, stream: *mut c_void,
        ));
        $decl!(embedding_decode_cuda(
            table: *const u16, token_id: *const i32, out: *mut u16,
            hidden_dim: i32, stream: *mut c_void,
        ));
        $decl!(embedding_batched_cuda(
            table: *const u16, token_ids: *const i32, out: *mut u16,
            hidden_dim: i32, batch_size: i32, stream: *mut c_void,
        ));

        // crates/cuda-kernels/csrc/misc/norm.cu
        $decl!(rms_norm_cuda(
            x: *const u16, weight: *const u16, out: *mut u16, n: i32,
            eps: f32, stream: *mut c_void,
        ));
        $decl!(fused_add_rms_norm_cuda(
            hidden: *mut u16, residual: *const u16, weight: *const u16,
            out: *mut u16, n: i32, eps: f32, stream: *mut c_void,
        ));
        $decl!(fused_add_rms_norm_batched_cuda(
            hidden: *mut u16, residual: *const u16, weight: *const u16,
            out: *mut u16, hidden_dim: i32, seq_len: i32, eps: f32,
            stream: *mut c_void,
        ));
        $decl!(rms_norm_batched_cuda(
            x: *const u16, weight: *const u16, out: *mut u16,
            hidden_dim: i32, seq_len: i32, eps: f32, stream: *mut c_void,
        ));
        $decl!(rms_norm_batched_f32_in_cuda(
            x: *const f32, weight: *const u16, out: *mut u16,
            hidden_dim: i32, seq_len: i32, eps: f32, stream: *mut c_void,
        ));
        $decl!(add_bf16_into_f32_cuda(
            out: *mut f32, input: *const u16, n: i32, stream: *mut c_void,
        ));
        $decl!(cast_bf16_to_f32_cuda(
            input: *const u16, out: *mut f32, n: i32, stream: *mut c_void,
        ));
        $decl!(cast_f32_to_bf16_cuda(
            input: *const f32, out: *mut u16, n: i32, stream: *mut c_void,
        ));
        $decl!(rms_norm_offset_cuda(
            x: *const u16, weight: *const u16, out: *mut u16, n: i32,
            eps: f32, stream: *mut c_void,
        ));
        $decl!(fused_add_rms_norm_offset_cuda(
            hidden: *mut u16, residual: *const u16, weight: *const u16,
            out: *mut u16, n: i32, eps: f32, stream: *mut c_void,
        ));
        $decl!(rms_norm_batched_offset_cuda(
            x: *const u16, weight: *const u16, out: *mut u16,
            hidden_dim: i32, seq_len: i32, eps: f32, stream: *mut c_void,
        ));
        $decl!(rms_norm_gated_cuda(
            x: *const u16, weight: *const f32, gate: *const u16, out: *mut u16,
            num_heads: i32, head_dim: i32, eps: f32, stream: *mut c_void,
        ));

        // crates/cuda-kernels/csrc/misc/sampling.cu
        $decl!(argmax_logprob_cuda(
            x: *const u16, out_idx: *mut i32, out_logprob: *mut f32, n: i32,
            stream: *mut c_void,
        ));
        $decl!(argmax_cuda(
            x: *const u16, out: *mut i32, n: i32, stream: *mut c_void,
        ));
        $decl!(argmax_batch_logprob_cuda(
            logits: *const u16, token_ids: *mut i32, logprobs: *mut f32,
            batch_size: i32, vocab_size: i32, stream: *mut c_void,
        ));
        $decl!(argmax_batch_cuda(
            logits: *const u16, token_ids: *mut i32, batch_size: i32,
            vocab_size: i32, stream: *mut c_void,
        ));
        $decl!(gpu_sample_cuda(
            logits: *const u16, probs_scratch: *mut f32, output: *mut i32,
            vocab_size: i32, inv_temperature: f32, top_k: i32, top_p: f32,
            random_val: f32, stream: *mut c_void,
        ));

        // crates/cuda-kernels/csrc/gemm/quantized_gemv.cu — the GGUF
        // Q4_K/Q5_K/Q6_K subset infer-hip serves; the fp8/fp4 + w{2,4,8}a16
        // launchers also compile into the archive but get Rust decls with
        // their first caller (interfaces trail callers).
        $decl!(q4k_gemv_cuda(
            weight: *const u8, input: *const u16, output: *mut u16,
            n: i32, k: i32, stream: *mut c_void,
        ));
        $decl!(q4k_gemv_batch_cuda(
            weight: *const u8, input: *const u16, output: *mut u16,
            b: i32, n: i32, k: i32, stream: *mut c_void,
        ));
        $decl!(q4k_dequant_chunk_cuda(
            weight: *const u8, out: *mut u16, n: i32, k: i32,
            k_start: i32, k_len: i32, stream: *mut c_void,
        ));
        $decl!(q5k_gemv_cuda(
            weight: *const u8, input: *const u16, output: *mut u16,
            n: i32, k: i32, stream: *mut c_void,
        ));
        $decl!(q5k_gemv_batch_cuda(
            weight: *const u8, input: *const u16, output: *mut u16,
            b: i32, n: i32, k: i32, stream: *mut c_void,
        ));
        $decl!(q5k_dequant_chunk_cuda(
            weight: *const u8, out: *mut u16, n: i32, k: i32,
            k_start: i32, k_len: i32, stream: *mut c_void,
        ));
        $decl!(q6k_gemv_cuda(
            weight: *const u8, input: *const u16, output: *mut u16,
            n: i32, k: i32, stream: *mut c_void,
        ));
        $decl!(q6k_gemv_batch_cuda(
            weight: *const u8, input: *const u16, output: *mut u16,
            b: i32, n: i32, k: i32, stream: *mut c_void,
        ));
        $decl!(q6k_dequant_chunk_cuda(
            weight: *const u8, out: *mut u16, n: i32, k: i32,
            k_start: i32, k_len: i32, stream: *mut c_void,
        ));
        $decl!(q4k_embedding_batched_cuda(
            weight: *const u8, token_ids: *const i32, out: *mut u16,
            hidden_dim: i32, batch_size: i32, stream: *mut c_void,
        ));
        $decl!(q5k_embedding_batched_cuda(
            weight: *const u8, token_ids: *const i32, out: *mut u16,
            hidden_dim: i32, batch_size: i32, stream: *mut c_void,
        ));
        $decl!(q6k_embedding_batched_cuda(
            weight: *const u8, token_ids: *const i32, out: *mut u16,
            hidden_dim: i32, batch_size: i32, stream: *mut c_void,
        ));
        $decl!(q4k_embedding_decode_cuda(
            weight: *const u8, token_id: *const i32, out: *mut u16,
            hidden_dim: i32, stream: *mut c_void,
        ));
        $decl!(q5k_embedding_decode_cuda(
            weight: *const u8, token_id: *const i32, out: *mut u16,
            hidden_dim: i32, stream: *mut c_void,
        ));
        $decl!(q6k_embedding_decode_cuda(
            weight: *const u8, token_id: *const i32, out: *mut u16,
            hidden_dim: i32, stream: *mut c_void,
        ));

        // crates/cuda-kernels/csrc/misc/dsv4_mhc.cu (compiled into the archive
        // since stage A; externs land with their first caller, the infer-hip
        // DSv4 forward). pre/post/comb are f32 sinkhorn weights; everything
        // else is bf16.
        $decl!(dsv4_mhc_expand_cuda(
            embeddings: *const u16, out: *mut u16, num_tokens: i32,
            hidden_size: i32, hc_mult: i32, stream: *mut c_void,
        ));
        $decl!(dsv4_mhc_params_cuda(
            residual: *const u16, mixes: *const u16, base: *const u16,
            scale: *const u16, pre: *mut f32, post: *mut f32, comb: *mut f32,
            num_tokens: i32, residual_hidden_dim: i32, mix_dim: i32,
            hc_mult: i32, eps: f32, sinkhorn_iters: i32, stream: *mut c_void,
        ));
        $decl!(dsv4_mhc_pre_cuda(
            residual: *const u16, pre: *const f32, out: *mut u16,
            num_tokens: i32, hidden_size: i32, hc_mult: i32,
            stream: *mut c_void,
        ));
        $decl!(dsv4_mhc_pre_rms_norm_cuda(
            residual: *const u16, pre: *const f32, weight: *const u16,
            out: *mut u16, num_tokens: i32, hidden_size: i32, hc_mult: i32,
            eps: f32, stream: *mut c_void,
        ));
        $decl!(dsv4_mhc_post_cuda(
            new_x: *const u16, residual: *const u16, post: *const f32,
            comb: *const f32, out: *mut u16, num_tokens: i32,
            hidden_size: i32, hc_mult: i32, stream: *mut c_void,
        ));
        $decl!(dsv4_mhc_head_pre_cuda(
            residual_row: *const u16, mixes: *const u16, base: *const u16,
            scale: *const u16, out: *mut u16, residual_hidden_dim: i32,
            hidden_size: i32, hc_mult: i32, eps: f32, stream: *mut c_void,
        ));
    };
}

/// Raw C-ABI launcher declarations (status = `hipError_t` as `i32`, stream
/// last as `*mut c_void`). Public so the future `infer-hip` executor can
/// call the DSv4 launchers directly — wrappers trail callers.
#[cfg(feature = "hip")]
pub mod ffi {
    use std::ffi::c_void;

    macro_rules! declare_launcher {
        ($name:ident($($arg:ident: $ty:ty,)*)) => {
            unsafe extern "C" {
                pub fn $name($($arg: $ty),*) -> i32;
            }
        };
    }
    basic_op_launchers!(declare_launcher);

    // csrc/iq2_mmvq.cu launchers (status = hipError_t, stream last).
    unsafe extern "C" {
        pub fn arle_quantize_row_q8_1_cuda(
            x: *const f32,
            y_q8_1: *mut c_void,
            ncols: i64,
            nrows: i64,
            stream: *mut c_void,
        ) -> i32;
        pub fn arle_mmvq_iq2_xxs_cuda(
            w_blocks: *const c_void,
            x_q8_1: *const c_void,
            y: *mut f32,
            ncols: i32,
            nrows: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn arle_mmvq_q2_k_cuda(
            w_blocks: *const c_void,
            x_q8_1: *const c_void,
            y: *mut f32,
            ncols: i32,
            nrows: i32,
            stream: *mut c_void,
        ) -> i32;
    }

    // DSv4 graph-safe (`*_start_pos_ptr_cuda`) launchers exported by the
    // shim-compiled cuda-kernels sources — signatures mirrored 1:1 from
    // crates/cuda-kernels/csrc/misc/dsv4_attention.cu. Extern decls only;
    // wrappers land with the infer-hip executor (interfaces trail callers).
    unsafe extern "C" {
        pub fn dsv4_prepare_qk_start_pos_ptr_cuda(
            q_raw: *const u16,
            k_raw: *const u16,
            q_out: *mut u16,
            k_out: *mut u16,
            num_tokens: i32,
            local_heads: i32,
            head_dim: i32,
            rope_dim: i32,
            start_pos_ptr: *const i32,
            rms_eps: f32,
            rope_base: f32,
            original_seq_len: i32,
            factor: f32,
            beta_fast: f32,
            beta_slow: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn dsv4_prepare_qk_fused_start_pos_ptr_cuda(
            q_raw: *const u16,
            k_raw: *const u16,
            q_out: *mut u16,
            k_out: *mut u16,
            num_tokens: i32,
            local_heads: i32,
            head_dim: i32,
            rope_dim: i32,
            start_pos_ptr: *const i32,
            rms_eps: f32,
            rope_base: f32,
            original_seq_len: i32,
            factor: f32,
            beta_fast: f32,
            beta_slow: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn dsv4_swa_attention_start_pos_ptr_cuda(
            q: *const u16,
            k_new: *const u16,
            window_cache: *mut u16,
            attn_sink: *const u16,
            out: *mut u16,
            num_tokens: i32,
            local_heads: i32,
            head_dim: i32,
            sliding_window: i32,
            start_pos_ptr: *const i32,
            sink_offset: i32,
            scale_value: f32,
            rope_dim: i32,
            rope_base: f32,
            original_seq_len: i32,
            factor: f32,
            beta_fast: f32,
            beta_slow: f32,
            write_window_cache: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn dsv4_update_window_cache_start_pos_ptr_cuda(
            k_new: *const u16,
            window_cache: *mut u16,
            num_tokens: i32,
            start_pos_ptr: *const i32,
            sliding_window: i32,
            head_dim: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn dsv4_compressor_update_start_pos_ptr_cuda(
            kv_raw: *const u16,
            score_raw: *const u16,
            ape: *const u16,
            norm: *const u16,
            pending_kv: *mut u16,
            pending_score: *mut u16,
            prev_overlap_kv: *mut u16,
            prev_overlap_score: *mut u16,
            compressed: *mut u16,
            num_tokens: i32,
            start_pos: *const i32,
            head_dim: i32,
            ratio: i32,
            width: i32,
            overlap: i32,
            eps: f32,
            rope_dim: i32,
            rope_base: f32,
            original_seq_len: i32,
            factor: f32,
            beta_fast: f32,
            beta_slow: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn dsv4_hybrid_attention_start_pos_ptr_cuda(
            q: *const u16,
            k_new: *const u16,
            window_cache: *mut u16,
            compressed: *const u16,
            selected: *const i32,
            attn_sink: *const u16,
            out: *mut u16,
            num_tokens: i32,
            local_heads: i32,
            head_dim: i32,
            sliding_window: i32,
            start_pos_ptr: *const i32,
            sink_offset: i32,
            scale_value: f32,
            rope_dim: i32,
            rope_base: f32,
            original_seq_len: i32,
            factor: f32,
            beta_fast: f32,
            beta_slow: f32,
            mode: i32,
            compress_ratio: i32,
            compressed_count: i32,
            selected_topk: i32,
            write_window_cache: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn arle_dsv4_output_inverse_rope_start_pos_ptr_cuda(
            out: *mut u16,
            token_count: i32,
            local_heads: i32,
            head_dim: i32,
            rope_dim: i32,
            start_pos_ptr: *const i32,
            rope_base: f32,
            original_seq_len: i32,
            factor: f32,
            beta_fast: f32,
            beta_slow: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn dsv4_csa_select_start_pos_ptr_cuda(
            q: *const u16,
            weights: *const u16,
            keys: *const u16,
            selected: *mut i32,
            num_tokens: i32,
            q_width: i32,
            local_heads: i32,
            index_dim: i32,
            key_count: i32,
            ratio: i32,
            topk: i32,
            score_scale: f32,
            start_pos: *const i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

#[cfg(feature = "hip")]
mod real {
    use super::{KernelError, Result, ffi};
    use std::ffi::c_void;

    fn check(code: i32) -> Result<()> {
        if code == 0 {
            Ok(())
        } else {
            Err(KernelError(code))
        }
    }

    /// Quantize `nrows` contiguous rows of `ncols` f32 activations into
    /// q8_1 blocks at `y_q8_1`. `ncols` must be a multiple of [`super::QK8_1`].
    ///
    /// # Safety
    /// `x` must point to `nrows * ncols` device f32; `y_q8_1` to at least
    /// `nrows * q8_1_row_bytes(ncols)` device bytes; `stream` must be a
    /// valid HIP stream (or null for the default stream).
    pub unsafe fn quantize_row_q8_1(
        x: *const f32,
        y_q8_1: *mut c_void,
        ncols: i64,
        nrows: i64,
        stream: *mut c_void,
    ) -> Result<()> {
        check(unsafe { ffi::arle_quantize_row_q8_1_cuda(x, y_q8_1, ncols, nrows, stream) })
    }

    /// y = W·x for a row-major IQ2_XXS weight matrix `[nrows, ncols]`
    /// against one q8_1-quantized activation vector (B=1 decode shape).
    ///
    /// # Safety
    /// `w_blocks` must hold `nrows * iq2_xxs_row_bytes(ncols)` device
    /// bytes; `x_q8_1` one q8_1 row of `ncols` values; `y` `nrows` device
    /// f32; `stream` a valid HIP stream (or null).
    pub unsafe fn mmvq_iq2_xxs(
        w_blocks: *const c_void,
        x_q8_1: *const c_void,
        y: *mut f32,
        ncols: i32,
        nrows: i32,
        stream: *mut c_void,
    ) -> Result<()> {
        check(unsafe { ffi::arle_mmvq_iq2_xxs_cuda(w_blocks, x_q8_1, y, ncols, nrows, stream) })
    }

    /// y = W·x for a row-major Q2_K weight matrix — same shape contract as
    /// [`mmvq_iq2_xxs`].
    ///
    /// # Safety
    /// As [`mmvq_iq2_xxs`], with `q2_k_row_bytes(ncols)` weight rows.
    pub unsafe fn mmvq_q2_k(
        w_blocks: *const c_void,
        x_q8_1: *const c_void,
        y: *mut f32,
        ncols: i32,
        nrows: i32,
        stream: *mut c_void,
    ) -> Result<()> {
        check(unsafe { ffi::arle_mmvq_q2_k_cuda(w_blocks, x_q8_1, y, ncols, nrows, stream) })
    }
}

#[cfg(feature = "hip")]
pub use real::{mmvq_iq2_xxs, mmvq_q2_k, quantize_row_q8_1};

/// Stub `ffi` with the same basic-op launcher names/signatures as the real
/// module — `infer-hip` compiles against `hip_kernels::ffi::*` on any machine
/// and every call answers [`NOT_COMPILED`].
#[cfg(not(feature = "hip"))]
pub mod ffi {
    use std::ffi::c_void;

    macro_rules! declare_launcher {
        ($name:ident($($arg:ident: $ty:ty,)*)) => {
            /// Stub — see the `hip` build for the real contract.
            ///
            /// # Safety
            /// No-op; never dereferences its arguments.
            pub unsafe fn $name($($arg: $ty),*) -> i32 {
                $(let _ = $arg;)*
                super::NOT_COMPILED.0
            }
        };
    }
    basic_op_launchers!(declare_launcher);
}

#[cfg(not(feature = "hip"))]
mod stub {
    use super::{NOT_COMPILED, Result};
    use std::ffi::c_void;

    /// Stub — see the `hip` build for the real contract.
    ///
    /// # Safety
    /// No-op; never dereferences its arguments.
    pub unsafe fn quantize_row_q8_1(
        _x: *const f32,
        _y_q8_1: *mut c_void,
        _ncols: i64,
        _nrows: i64,
        _stream: *mut c_void,
    ) -> Result<()> {
        Err(NOT_COMPILED)
    }

    /// Stub — see the `hip` build for the real contract.
    ///
    /// # Safety
    /// No-op; never dereferences its arguments.
    pub unsafe fn mmvq_iq2_xxs(
        _w_blocks: *const c_void,
        _x_q8_1: *const c_void,
        _y: *mut f32,
        _ncols: i32,
        _nrows: i32,
        _stream: *mut c_void,
    ) -> Result<()> {
        Err(NOT_COMPILED)
    }

    /// Stub — see the `hip` build for the real contract.
    ///
    /// # Safety
    /// No-op; never dereferences its arguments.
    pub unsafe fn mmvq_q2_k(
        _w_blocks: *const c_void,
        _x_q8_1: *const c_void,
        _y: *mut f32,
        _ncols: i32,
        _nrows: i32,
        _stream: *mut c_void,
    ) -> Result<()> {
        Err(NOT_COMPILED)
    }
}

#[cfg(not(feature = "hip"))]
pub use stub::{mmvq_iq2_xxs, mmvq_q2_k, quantize_row_q8_1};
