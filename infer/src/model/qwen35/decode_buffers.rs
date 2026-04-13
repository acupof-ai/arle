//! Sampling buffers for Qwen3.5 token selection.

use anyhow::Result;

use cudarc::driver::CudaSlice;

use super::config::Config35;
use crate::backend::cuda::tensor::{DeviceContext, DeviceVec, RawDevicePtr};

/// Cached raw pointers for hot-path sampling ops (avoids cudarc device_ptr overhead).
pub(crate) struct DecodeBufferPtrs35 {
    pub logits_ptr: RawDevicePtr<half::bf16>,
    pub logits_len: usize,
    pub sample_probs_ptr: RawDevicePtr<f32>,
    pub sample_out_ptr: RawDevicePtr<i32>,
}

/// Pre-allocated GPU buffers for token sampling (softmax + multinomial).
pub(crate) struct DecodeBuffers35 {
    /// FP32 scratch buffer for GPU sampling softmax (vocab_size)
    pub(crate) sample_probs: CudaSlice<f32>,
    /// Pre-allocated sampling output (1 element, token id)
    pub(crate) sample_out: CudaSlice<i32>,
    /// Cached raw device pointers for hot-path sampling ops.
    pub(crate) ptrs: DecodeBufferPtrs35,
    /// Per-slot logits buffer for batched decode logit scatter.
    pub(crate) logits_scratch: DeviceVec,
}

impl DecodeBuffers35 {
    pub(crate) fn new(ctx: &DeviceContext, config: &Config35, logits: &DeviceVec) -> Result<Self> {
        let sample_probs: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(config.vocab_size)
            .map_err(|e| anyhow::anyhow!("Alloc sample_probs failed: {}", e))?;
        let sample_out: CudaSlice<i32> = ctx
            .stream
            .alloc_zeros(1)
            .map_err(|e| anyhow::anyhow!("Alloc sample_out failed: {}", e))?;

        let ptrs = DecodeBufferPtrs35 {
            logits_ptr: crate::backend::cuda::tensor::cache_ptr(&logits.data, ctx),
            logits_len: config.vocab_size,
            sample_probs_ptr: crate::backend::cuda::tensor::cache_ptr(&sample_probs, ctx),
            sample_out_ptr: crate::backend::cuda::tensor::cache_ptr(&sample_out, ctx),
        };

        let logits_scratch = DeviceVec::zeros(ctx, config.vocab_size)?;

        Ok(Self {
            sample_probs,
            sample_out,
            ptrs,
            logits_scratch,
        })
    }
}
