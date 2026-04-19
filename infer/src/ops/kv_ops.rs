use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr};

use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, HiddenStates};

/// Scatter-write prefill K/V from contiguous GEMM output to a token-level KV pool.
///
/// After QKV projection produces contiguous `k_batch` and `v_batch` (each
/// `[seq_len, num_kv_heads * head_dim]`), this copies each token's K and V to
/// `k_pool`/`v_pool` at the positions specified by `token_indices`.
///
/// No norm or RoPE is applied — the downstream attention kernel (Triton prefill
/// or FlashInfer) handles those internally.
///
/// # Arguments
/// * `k_batch`, `v_batch` — contiguous GEMM outputs, shape `[seq_len, kv_dim]`
/// * `k_pool`, `v_pool` — token-level KV pool, shape `[max_tokens, kv_dim]`
/// * `token_indices` — `[seq_len]` i32 on GPU, destination index per token
/// * `num_kv_heads`, `head_dim` — such that `kv_dim = num_kv_heads * head_dim`
#[allow(clippy::too_many_arguments)]
pub fn scatter_write_kv(
    ctx: &DeviceContext,
    k_batch: &HiddenStates,
    v_batch: &HiddenStates,
    k_pool_ptr: u64,
    v_pool_ptr: u64,
    token_indices: &CudaSlice<i32>,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let seq_len = k_batch.seq_len;
    let kv_dim = num_kv_heads * head_dim;

    assert_eq!(
        k_batch.hidden_dim, kv_dim,
        "k_batch hidden_dim ({}) != kv_dim ({})",
        k_batch.hidden_dim, kv_dim
    );
    assert_eq!(
        v_batch.hidden_dim, kv_dim,
        "v_batch hidden_dim ({}) != kv_dim ({})",
        v_batch.hidden_dim, kv_dim
    );
    assert_eq!(
        v_batch.seq_len, seq_len,
        "v_batch seq_len ({}) != k_batch seq_len ({})",
        v_batch.seq_len, seq_len
    );

    let (k_ptr, _gk) = k_batch.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v_batch.data.device_ptr(&ctx.stream);
    let (ti_ptr, _gti) = token_indices.device_ptr(&ctx.stream);

    unsafe {
        ffi::scatter_write_kv_cuda(
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            k_pool_ptr as *mut ffi::Half,
            v_pool_ptr as *mut ffi::Half,
            ti_ptr as *const i32,
            seq_len as i32,
            num_kv_heads as i32,
            head_dim as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }

    Ok(())
}
