//! Recurrent-family launch wrappers: Qwen3.5 conv1d + gated-delta-rule (GDR)
//! serving/training kernels and the FlashQLA prep stage.
//!
//! Raw u64 device addresses throughout — consumers apply pointer offsets
//! (chunk bases, head shifts, staged pointer-table slots) before the call and
//! keep every owning buffer alive and stream-ordered on `stream`. State
//! mutation, chunk replay, accepted-length rollback, and backward semantics
//! stay with the calling model / autograd policy; only launch mechanics live
//! here. The `gdr_fq_{cumsum,kkt,fwd,prepare_h,bwd}` per-geometry symbols stay
//! fn-pointer tables in their consumers (see `ffi::FLASHQLA_GDR_TABLE`).

use anyhow::{Result, anyhow};
use cudarc::driver::CudaStream;

use crate::ffi;

fn rec_i32(v: usize, what: &'static str) -> Result<i32> {
    i32::try_from(v).map_err(|_| anyhow!("{what} {v} exceeds i32"))
}

/// Batched decode conv1d: `x`/`out` token-major `[batch, num_channels]`;
/// `conv_state_tbl` holds `batch` pointers at live `[num_channels, kernel-1]`
/// conv rings, each advanced in place.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_decode_batch_raw(
    stream: &CudaStream,
    x_ptr: u64,
    weight_ptr: u64,
    conv_state_tbl: u64,
    out_ptr: u64,
    num_channels: usize,
    kernel_size: usize,
    batch: usize,
) -> Result<()> {
    // SAFETY: caller passes live device addresses sized to the dims below,
    // stream-ordered on `stream`.
    unsafe {
        ffi::conv1d_decode_batch_cuda(
            x_ptr as *const ffi::Half,
            weight_ptr as *const ffi::Half,
            conv_state_tbl as *mut *mut ffi::Half,
            out_ptr as *mut ffi::Half,
            rec_i32(num_channels, "conv1d_decode_batch channels")?,
            rec_i32(kernel_size, "conv1d_decode_batch kernel")?,
            rec_i32(batch, "conv1d_decode_batch batch")?,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("conv1d_decode_batch_cuda failed at batch={batch}: {e}"))
    }
}

/// Single-sequence prefill conv1d over `seq_len` rows; `conv_state` is the
/// `[num_channels, kernel-1]` ring, advanced in place.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_prefill_raw(
    stream: &CudaStream,
    x_ptr: u64,
    weight_ptr: u64,
    conv_state_ptr: u64,
    out_ptr: u64,
    num_channels: usize,
    seq_len: usize,
    kernel_size: usize,
) -> Result<()> {
    // SAFETY: caller passes live device addresses sized to the dims below,
    // stream-ordered on `stream`.
    unsafe {
        ffi::conv1d_prefill_cuda(
            x_ptr as *const ffi::Half,
            weight_ptr as *const ffi::Half,
            conv_state_ptr as *mut ffi::Half,
            out_ptr as *mut ffi::Half,
            rec_i32(num_channels, "conv1d_prefill channels")?,
            rec_i32(seq_len, "conv1d_prefill seq_len")?,
            rec_i32(kernel_size, "conv1d_prefill kernel")?,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("conv1d_prefill_cuda failed at seq={seq_len}: {e}"))
    }
}

/// Varlen prefill conv1d: slot `s` reads `x_tbl[s]`, runs `row_len[s]` rows
/// into `state_tbl[s]`, writes at its `s * max_len` block of `out`.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_prefill_varlen_raw(
    stream: &CudaStream,
    x_tbl: u64,
    weight_ptr: u64,
    state_tbl: u64,
    row_len_ptr: u64,
    out_ptr: u64,
    num_channels: usize,
    max_len: usize,
    kernel_size: usize,
    batch: usize,
) -> Result<()> {
    // SAFETY: caller passes live device addresses (tables hold `batch` live
    // pointers) sized to the dims below, stream-ordered on `stream`.
    unsafe {
        ffi::conv1d_prefill_varlen_cuda(
            x_tbl as *const *const ffi::Half,
            weight_ptr as *const ffi::Half,
            state_tbl as *const *mut ffi::Half,
            row_len_ptr as *const i32,
            out_ptr as *mut ffi::Half,
            rec_i32(num_channels, "conv1d_varlen channels")?,
            rec_i32(max_len, "conv1d_varlen max_len")?,
            rec_i32(kernel_size, "conv1d_varlen kernel")?,
            rec_i32(batch, "conv1d_varlen batch")?,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("conv1d_prefill_varlen_cuda failed at batch={batch}: {e}"))
    }
}

/// Single-token GDR decode step; `state` is the `[v_heads, key_dim, val_dim]`
/// f32 recurrent state, advanced in place.
#[allow(clippy::too_many_arguments)]
pub fn gdr_decode_raw(
    stream: &CudaStream,
    qkv_ptr: u64,
    b_ptr: u64,
    a_ptr: u64,
    dt_bias_ptr: u64,
    a_log_ptr: u64,
    state_ptr: u64,
    out_ptr: u64,
    num_key_heads: usize,
    num_value_heads: usize,
    key_dim: usize,
    val_dim: usize,
) -> Result<()> {
    // SAFETY: caller passes live device addresses sized to the dims below,
    // stream-ordered on `stream`.
    unsafe {
        ffi::gated_delta_rule_decode_cuda(
            qkv_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            a_ptr as *const ffi::Half,
            dt_bias_ptr as *const ffi::Half,
            a_log_ptr as *const f32,
            state_ptr as *mut f32,
            out_ptr as *mut ffi::Half,
            rec_i32(num_key_heads, "gdr_decode k_heads")?,
            rec_i32(num_value_heads, "gdr_decode v_heads")?,
            rec_i32(key_dim, "gdr_decode key_dim")?,
            rec_i32(val_dim, "gdr_decode val_dim")?,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("gated_delta_rule_decode_cuda failed: {e}"))
    }
}

/// Sequential GDR prefill over `seq_len` rows; same state contract as
/// [`gdr_decode_raw`].
#[allow(clippy::too_many_arguments)]
pub fn gdr_prefill_recurrent_raw(
    stream: &CudaStream,
    qkv_ptr: u64,
    b_ptr: u64,
    a_ptr: u64,
    dt_bias_ptr: u64,
    a_log_ptr: u64,
    state_ptr: u64,
    out_ptr: u64,
    num_key_heads: usize,
    num_value_heads: usize,
    key_dim: usize,
    val_dim: usize,
    seq_len: usize,
) -> Result<()> {
    // SAFETY: caller passes live device addresses sized to the dims below,
    // stream-ordered on `stream`.
    unsafe {
        ffi::gated_delta_rule_prefill_recurrent_cuda(
            qkv_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            a_ptr as *const ffi::Half,
            dt_bias_ptr as *const ffi::Half,
            a_log_ptr as *const f32,
            state_ptr as *mut f32,
            out_ptr as *mut ffi::Half,
            rec_i32(num_key_heads, "gdr_prefill k_heads")?,
            rec_i32(num_value_heads, "gdr_prefill v_heads")?,
            rec_i32(key_dim, "gdr_prefill key_dim")?,
            rec_i32(val_dim, "gdr_prefill val_dim")?,
            rec_i32(seq_len, "gdr_prefill seq_len")?,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("gated_delta_rule_prefill_recurrent_cuda failed at seq={seq_len}: {e}")
        })
    }
}

/// Varlen GDR replay: slot `s` reads qkv / writes output at its `s * max_len`
/// block, running `row_len[s]` rows into `state_tbl[s]`; `b_tbl`/`a_tbl` hold
/// per-slot input pointers.
#[allow(clippy::too_many_arguments)]
pub fn gdr_prefill_recurrent_varlen_raw(
    stream: &CudaStream,
    qkv_ptr: u64,
    b_tbl: u64,
    a_tbl: u64,
    dt_bias_ptr: u64,
    a_log_ptr: u64,
    state_tbl: u64,
    row_len_ptr: u64,
    out_ptr: u64,
    num_key_heads: usize,
    num_value_heads: usize,
    key_dim: usize,
    val_dim: usize,
    max_len: usize,
    batch: usize,
) -> Result<()> {
    // SAFETY: caller passes live device addresses (tables hold `batch` live
    // pointers) sized to the dims below, stream-ordered on `stream`.
    unsafe {
        ffi::gated_delta_rule_prefill_recurrent_varlen_cuda(
            qkv_ptr as *const ffi::Half,
            b_tbl as *const *const ffi::Half,
            a_tbl as *const *const ffi::Half,
            dt_bias_ptr as *const ffi::Half,
            a_log_ptr as *const f32,
            state_tbl as *const *mut f32,
            row_len_ptr as *const i32,
            out_ptr as *mut ffi::Half,
            rec_i32(num_key_heads, "gdr_varlen k_heads")?,
            rec_i32(num_value_heads, "gdr_varlen v_heads")?,
            rec_i32(key_dim, "gdr_varlen key_dim")?,
            rec_i32(val_dim, "gdr_varlen val_dim")?,
            rec_i32(max_len, "gdr_varlen max_len")?,
            rec_i32(batch, "gdr_varlen batch")?,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("gated_delta_rule_prefill_recurrent_varlen_cuda failed at batch={batch}: {e}")
        })
    }
}

/// Batched single-token GDR decode; `state_tbl` holds `batch` pointers at live
/// `[v_heads, key_dim, val_dim]` f32 states, each advanced in place.
#[allow(clippy::too_many_arguments)]
pub fn gdr_decode_batch_raw(
    stream: &CudaStream,
    qkv_ptr: u64,
    b_ptr: u64,
    a_ptr: u64,
    dt_bias_ptr: u64,
    a_log_ptr: u64,
    state_tbl: u64,
    out_ptr: u64,
    num_key_heads: usize,
    num_value_heads: usize,
    key_dim: usize,
    val_dim: usize,
    batch: usize,
) -> Result<()> {
    // SAFETY: caller passes live device addresses (table holds `batch` live
    // state pointers) sized to the dims below, stream-ordered on `stream`.
    unsafe {
        ffi::gdr_decode_batch_cuda(
            qkv_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            a_ptr as *const ffi::Half,
            dt_bias_ptr as *const ffi::Half,
            a_log_ptr as *const f32,
            state_tbl as *mut *mut f32,
            out_ptr as *mut ffi::Half,
            rec_i32(num_key_heads, "gdr_decode_batch k_heads")?,
            rec_i32(num_value_heads, "gdr_decode_batch v_heads")?,
            rec_i32(key_dim, "gdr_decode_batch key_dim")?,
            rec_i32(val_dim, "gdr_decode_batch val_dim")?,
            rec_i32(batch, "gdr_decode_batch batch")?,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("gdr_decode_batch_cuda failed at batch={batch}: {e}"))
    }
}

/// FlashQLA prep: fused conv-out `[seq, qkv]` -> l2norm'd q/k `[seq, Hg, key]`,
/// v `[seq, H, val]`, g/beta `[seq, H]` for the chunked GDR pipeline.
#[allow(clippy::too_many_arguments)]
pub fn gdr_fq_prep_raw(
    stream: &CudaStream,
    qkv_ptr: u64,
    b_ptr: u64,
    a_ptr: u64,
    dt_bias_ptr: u64,
    a_log_ptr: u64,
    q_out_ptr: u64,
    k_out_ptr: u64,
    v_out_ptr: u64,
    g_out_ptr: u64,
    beta_out_ptr: u64,
    num_key_heads: usize,
    num_value_heads: usize,
    key_dim: usize,
    val_dim: usize,
    seq_len: usize,
) -> Result<()> {
    // SAFETY: caller passes live device addresses sized to the dims below,
    // stream-ordered on `stream`.
    unsafe {
        ffi::gdr_fq_prep_cuda(
            qkv_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            a_ptr as *const ffi::Half,
            dt_bias_ptr as *const ffi::Half,
            a_log_ptr as *const f32,
            q_out_ptr as *mut ffi::Half,
            k_out_ptr as *mut ffi::Half,
            v_out_ptr as *mut ffi::Half,
            g_out_ptr as *mut f32,
            beta_out_ptr as *mut f32,
            rec_i32(num_key_heads, "gdr_fq_prep k_heads")?,
            rec_i32(num_value_heads, "gdr_fq_prep v_heads")?,
            rec_i32(key_dim, "gdr_fq_prep key_dim")?,
            rec_i32(val_dim, "gdr_fq_prep val_dim")?,
            rec_i32(seq_len, "gdr_fq_prep seq_len")?,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("gdr_fq_prep_cuda failed at seq={seq_len}: {e}"))
    }
}

/// Chunked-prefill prepare: splits fused qkv into q/k/v (value-head axis) plus
/// g/beta for the training chunk kernel; note `qkv_dim` replaces the decode
/// path's key/val split.
#[allow(clippy::too_many_arguments)]
pub fn gdr_prefill_chunk_prepare_raw(
    stream: &CudaStream,
    qkv_ptr: u64,
    b_ptr: u64,
    a_ptr: u64,
    dt_bias_ptr: u64,
    a_log_ptr: u64,
    q_out_ptr: u64,
    k_out_ptr: u64,
    v_out_ptr: u64,
    g_out_ptr: u64,
    beta_out_ptr: u64,
    num_key_heads: usize,
    num_value_heads: usize,
    qkv_dim: usize,
    seq_len: usize,
) -> Result<()> {
    // SAFETY: caller passes live device addresses sized to the dims below,
    // stream-ordered on `stream`.
    unsafe {
        ffi::gated_delta_rule_prefill_chunk_prepare_cuda(
            qkv_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            a_ptr as *const ffi::Half,
            dt_bias_ptr as *const ffi::Half,
            a_log_ptr as *const f32,
            q_out_ptr as *mut ffi::Half,
            k_out_ptr as *mut ffi::Half,
            v_out_ptr as *mut ffi::Half,
            g_out_ptr as *mut f32,
            beta_out_ptr as *mut f32,
            rec_i32(num_key_heads, "gdr_chunk_prepare k_heads")?,
            rec_i32(num_value_heads, "gdr_chunk_prepare v_heads")?,
            rec_i32(qkv_dim, "gdr_chunk_prepare qkv_dim")?,
            rec_i32(seq_len, "gdr_chunk_prepare seq_len")?,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("gated_delta_rule_prefill_chunk_prepare_cuda failed at seq={seq_len}: {e}")
        })
    }
}

/// `count` equal-or-varied-size D2D copies in one launch. `bytes` (uniform
/// size) must be a 16B multiple; `words_each_ptr == 0` selects the uniform
/// path, otherwise it names `count` per-copy 16B word counts with
/// `max_words` their maximum.
#[allow(clippy::too_many_arguments)]
pub fn batched_copy_uniform_raw(
    stream: &CudaStream,
    dst_tbl: u64,
    src_tbl: u64,
    words_each_ptr: u64,
    bytes: usize,
    max_words: usize,
    count: usize,
) -> Result<()> {
    // SAFETY: tables hold `count` live dst/src addresses, each buffer at least
    // its byte size and cudaMalloc-aligned, stream-ordered on `stream`.
    unsafe {
        ffi::batched_copy_uniform_cuda(
            dst_tbl as *const *mut std::ffi::c_void,
            src_tbl as *const *const std::ffi::c_void,
            words_each_ptr as *const i32,
            bytes,
            max_words,
            rec_i32(count, "batched_copy count")?,
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("batched_copy_uniform_cuda failed at count={count}: {e}"))
    }
}
