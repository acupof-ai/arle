//! GPU sampling ops: argmax, top-k, nucleus, temperature, penalties.
//!
//! Two sampling paths:
//!   - **Greedy** (`argmax_batch_launch`): single kernel, batched across requests
//!   - **Full** (`gpu_sample_launch`): softmax + top-k filter + multinomial via
//!     parallel prefix sum (Blelloch scan)
//!
//! All kernels are CUDA Graph safe (no allocation, fixed pointers).

use anyhow::{Result, anyhow};
use cudarc::driver::sys::CUstream;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::backend::cuda::ffi;
use crate::backend::cuda::tensor::{DeviceContext, DeviceVec};

/// Argmax — returns the index of the maximum element.
///
/// Allocates a temporary output buffer. Used by benchmarks; model code uses
/// `gpu_sample_into` for both greedy and non-greedy paths.
pub fn argmax(ctx: &DeviceContext, x: &DeviceVec) -> Result<u32> {
    let mut out_gpu: CudaSlice<i32> = ctx
        .stream
        .alloc_zeros(1)
        .map_err(|e| anyhow!("Alloc failed: {}", e))?;

    {
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out_gpu.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::argmax_cuda(
                x_ptr as *const ffi::Half,
                out_ptr as *mut i32,
                x.len as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }

    ctx.sync()?;

    let result = ctx
        .stream
        .clone_dtoh(&out_gpu)
        .map_err(|e| anyhow!("D2H copy failed: {}", e))?;

    Ok(result[0] as u32)
}

/// Launch batched argmax on a HiddenStates [B, vocab] buffer. No sync, no readback.
pub(crate) fn argmax_batch_launch(
    ctx: &DeviceContext,
    logits: &crate::backend::cuda::tensor::HiddenStates,
    out: &mut CudaSlice<i32>,
    batch_size: usize,
) -> Result<()> {
    let vocab_size = logits.hidden_dim;
    let (l_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::argmax_batch_cuda(
            l_ptr as *const ffi::Half,
            o_ptr as *mut i32,
            batch_size as i32,
            vocab_size as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Batched argmax + logprob launch (no sync). Returns token IDs + logprobs.
pub(crate) fn argmax_batch_logprob_launch(
    ctx: &DeviceContext,
    logits: &crate::backend::cuda::tensor::HiddenStates,
    out_ids: &mut CudaSlice<i32>,
    out_logprobs: &mut CudaSlice<f32>,
    batch_size: usize,
) -> Result<()> {
    let vocab_size = logits.hidden_dim;
    let (l_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
    let (i_ptr, _gi) = out_ids.device_ptr_mut(&ctx.stream);
    let (lp_ptr, _glp) = out_logprobs.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::argmax_batch_logprob_cuda(
            l_ptr as *const ffi::Half,
            i_ptr as *mut i32,
            lp_ptr as *mut f32,
            batch_size as i32,
            vocab_size as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Read back B argmax results into a pre-allocated host slice. Call after sync.
pub(crate) fn argmax_batch_readback_into(
    ctx: &DeviceContext,
    out: &CudaSlice<i32>,
    dst: &mut [i32],
    batch_size: usize,
) -> Result<()> {
    let tmp = ctx
        .stream
        .clone_dtoh(&out.slice(0..batch_size))
        .map_err(|e| anyhow!("D2H batch argmax readback: {e}"))?;
    dst[..batch_size].copy_from_slice(&tmp);
    Ok(())
}

/// GPU sampling: temperature → softmax → top-k → top-p → multinomial.
/// Allocates a temporary output buffer — use `gpu_sample_into` for the decode loop.
pub fn gpu_sample(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    probs_scratch: &mut CudaSlice<f32>,
    params: &crate::sampler::SamplingParams,
    random_val: f32,
) -> Result<u32> {
    let mut out_gpu: CudaSlice<i32> = ctx
        .stream
        .alloc_zeros(1)
        .map_err(|e| anyhow!("Alloc failed: {}", e))?;

    gpu_sample_core(ctx, logits, probs_scratch, &mut out_gpu, params, random_val)
}

/// GPU sampling into pre-allocated buffers — zero allocation, suitable for decode loop.
///
/// Greedy dispatch: argmax kernel. Non-greedy: full sampling kernel.
pub fn gpu_sample_into(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    probs_scratch: &mut CudaSlice<f32>,
    out: &mut CudaSlice<i32>,
    params: &crate::sampler::SamplingParams,
    random_val: f32,
) -> Result<u32> {
    gpu_sample_core(ctx, logits, probs_scratch, out, params, random_val)
}

/// Greedy argmax + logprob: returns (token_id, log_probability) in a single kernel.
///
/// Uses fused argmax+logsoftmax kernel — same cost as argmax alone (one scan
/// over logits, with an extra sum_exp accumulator). Zero additional memory traffic.
pub fn argmax_with_logprob(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    out_idx: &mut CudaSlice<i32>,
    out_logprob: &mut CudaSlice<f32>,
) -> Result<(u32, f32)> {
    // Launch kernel
    {
        let (l_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
        let (i_ptr, _gi) = out_idx.device_ptr_mut(&ctx.stream);
        let (lp_ptr, _glp) = out_logprob.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::argmax_logprob_cuda(
                l_ptr as *const ffi::Half,
                i_ptr as *mut i32,
                lp_ptr as *mut f32,
                logits.len as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }

    ctx.sync()?;

    // Readback (borrows released from kernel launch scope)
    let idx = ctx
        .stream
        .clone_dtoh(out_idx)
        .map_err(|e| anyhow!("D2H argmax idx: {}", e))?;
    let lp = ctx
        .stream
        .clone_dtoh(out_logprob)
        .map_err(|e| anyhow!("D2H logprob: {}", e))?;

    Ok((idx[0] as u32, lp[0]))
}

fn gpu_sample_core(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    probs_scratch: &mut CudaSlice<f32>,
    out: &mut CudaSlice<i32>,
    params: &crate::sampler::SamplingParams,
    random_val: f32,
) -> Result<u32> {
    gpu_sample_launch(ctx, logits, probs_scratch, out, params, random_val)?;

    ctx.sync()?;

    let result = ctx
        .stream
        .clone_dtoh(out)
        .map_err(|e| anyhow!("D2H sample read failed: {}", e))?;

    Ok(result[0] as u32)
}

/// Core sampling kernel dispatch -- used by all public launch variants.
///
/// Greedy (temperature=0): launches argmax kernel.
/// Non-greedy: launches full sampling kernel (temperature + softmax + top-k/p).
///
/// # Safety
/// All pointers must be valid device pointers on the given stream.
unsafe fn launch_sample_kernel_inner(
    logits_ptr: *const ffi::Half,
    logits_len: i32,
    probs_ptr: *mut f32,
    out_ptr: *mut i32,
    params: &crate::sampler::SamplingParams,
    random_val: f32,
    stream: CUstream,
) {
    if params.is_greedy() {
        unsafe { ffi::argmax_cuda(logits_ptr, out_ptr, logits_len, stream) }
            .result()
            .expect("argmax_cuda failed");
    } else {
        unsafe {
            ffi::gpu_sample_cuda(
                logits_ptr,
                probs_ptr,
                out_ptr,
                logits_len,
                1.0 / params.temperature,
                params.top_k,
                params.top_p,
                random_val,
                stream,
            )
        }
        .result()
        .expect("gpu_sample_cuda failed");
    }
}

/// Launch the sampling kernel without syncing. Call `ctx.sync()` separately.
pub fn gpu_sample_launch(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    probs_scratch: &mut CudaSlice<f32>,
    out: &mut CudaSlice<i32>,
    params: &crate::sampler::SamplingParams,
    random_val: f32,
) -> Result<()> {
    let (l_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
    let (p_ptr, _gp) = probs_scratch.device_ptr_mut(&ctx.stream);
    let (o_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    unsafe {
        launch_sample_kernel_inner(
            l_ptr as *const ffi::Half,
            logits.len as i32,
            p_ptr as *mut f32,
            o_ptr as *mut i32,
            params,
            random_val,
            ctx.stream.cu_stream(),
        );
    }
    Ok(())
}

/// Launch sampling kernel using pre-cached raw device pointers.
/// Eliminates device_ptr() overhead on every call.
pub(crate) fn gpu_sample_launch_raw(
    ctx: &DeviceContext,
    logits_ptr: crate::backend::cuda::tensor::RawDevicePtr<half::bf16>,
    logits_len: usize,
    probs_ptr: crate::backend::cuda::tensor::RawDevicePtr<f32>,
    out_ptr: crate::backend::cuda::tensor::RawDevicePtr<i32>,
    params: &crate::sampler::SamplingParams,
    random_val: f32,
) -> Result<()> {
    unsafe {
        launch_sample_kernel_inner(
            logits_ptr.as_ptr() as *const ffi::Half,
            logits_len as i32,
            probs_ptr.as_mut_ptr(),
            out_ptr.as_mut_ptr(),
            params,
            random_val,
            ctx.stream.cu_stream(),
        );
    }
    Ok(())
}

/// Read back the sampling result after a prior `gpu_sample_launch` + `ctx.sync()`.
pub fn gpu_sample_readback(ctx: &DeviceContext, out: &CudaSlice<i32>) -> Result<u32> {
    let result = ctx
        .stream
        .clone_dtoh(out)
        .map_err(|e| anyhow!("D2H sample read failed: {}", e))?;
    Ok(result[0] as u32)
}
