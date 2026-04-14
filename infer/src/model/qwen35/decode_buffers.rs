//! Sampling buffers for Qwen3.5 token selection.

use anyhow::Result;

use cudarc::driver::CudaSlice;

use super::config::Config35;
use infer_cuda_kernels::prelude::{DeviceContext, DeviceVec, RawDevicePtr};
use infer_cuda_kernels::tensor::cache_ptr;

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
    /// Tracks whether the latest decode logits live in `logits_scratch`.
    using_logits_scratch: bool,
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
            logits_ptr: cache_ptr(&logits.data, ctx),
            logits_len: config.vocab_size,
            sample_probs_ptr: cache_ptr(&sample_probs, ctx),
            sample_out_ptr: cache_ptr(&sample_out, ctx),
        };

        let logits_scratch = DeviceVec::zeros(ctx, config.vocab_size)?;

        Ok(Self {
            sample_probs,
            sample_out,
            ptrs,
            logits_scratch,
            using_logits_scratch: false,
        })
    }

    /// Point raw sampling kernels at the state's single-token logits buffer.
    pub(crate) fn bind_single_token_logits(&mut self, ctx: &DeviceContext, logits: &DeviceVec) {
        self.ptrs.logits_ptr = cache_ptr(&logits.data, ctx);
        self.ptrs.logits_len = logits.len;
        self.using_logits_scratch = false;
    }

    /// Point raw sampling kernels at the per-request batched decode logits copy.
    pub(crate) fn bind_logits_scratch(&mut self, ctx: &DeviceContext) {
        self.ptrs.logits_ptr = cache_ptr(&self.logits_scratch.data, ctx);
        self.ptrs.logits_len = self.logits_scratch.len;
        self.using_logits_scratch = true;
    }

    pub(crate) fn current_logits<'a>(
        &'a self,
        single_token_logits: &'a DeviceVec,
    ) -> &'a DeviceVec {
        if self.using_logits_scratch {
            &self.logits_scratch
        } else {
            single_token_logits
        }
    }

    pub(crate) fn current_logits_and_sampling_bufs<'a>(
        &'a mut self,
        single_token_logits: &'a DeviceVec,
    ) -> (
        &'a DeviceVec,
        &'a mut CudaSlice<f32>,
        &'a mut CudaSlice<i32>,
    ) {
        let Self {
            sample_probs,
            sample_out,
            logits_scratch,
            using_logits_scratch,
            ..
        } = self;
        let logits = if *using_logits_scratch {
            &*logits_scratch
        } else {
            single_token_logits
        };
        (logits, sample_probs, sample_out)
    }
}
