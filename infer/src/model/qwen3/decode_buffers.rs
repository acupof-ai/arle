//! Pre-allocated GPU buffers for zero-allocation decode steps.

use anyhow::Result;

use cudarc::driver::CudaSlice;

use super::config::Config;
use infer_cuda_kernels::prelude::{DeviceContext, DeviceVec, RawDevicePtr};
use infer_cuda_kernels::tensor::cache_ptr;

/// Cached raw pointers for hot-path sampling ops (avoids cudarc device_ptr overhead).
pub(crate) struct DecodeBufferPtrs {
    pub logits_ptr: RawDevicePtr<half::bf16>,
    pub logits_len: usize,
    pub sample_probs_ptr: RawDevicePtr<f32>,
    pub sample_out_ptr: RawDevicePtr<i32>,
}

/// Pre-allocated temporary buffers for the single-token decode path.
///
/// All buffer dimensions are determined by the model config and remain fixed
/// for the entire generation. Reusing these across decode steps eliminates
/// ~292 cudaMalloc/cudaFree calls per token.
pub(crate) struct DecodeBuffers {
    /// RMSNorm output / general scratch (hidden_size)
    pub(crate) normed: DeviceVec,
    /// Q projection output (num_attention_heads * head_dim)
    pub(crate) q: DeviceVec,
    /// K projection output (num_key_value_heads * head_dim)
    pub(crate) k: DeviceVec,
    /// V projection output (num_key_value_heads * head_dim)
    pub(crate) v: DeviceVec,
    /// Fused attention output (num_attention_heads * head_dim)
    pub(crate) attn_out: DeviceVec,
    /// O projection output (hidden_size)
    pub(crate) attn_proj: DeviceVec,
    /// Fused MLP intermediate activation (intermediate_size)
    pub(crate) mlp_act: DeviceVec,
    /// Scratch buffer for `up_proj` output when the unfused LoRA decode MLP
    /// path is used (size `intermediate_size`). Unused when no LoRA adapter
    /// touches the MLP; still allocated up-front to preserve zero-alloc on
    /// the hot path once LoRA is attached.
    pub(crate) mlp_up_scratch: DeviceVec,
    /// Fused MLP output (hidden_size)
    pub(crate) mlp_out: DeviceVec,
    /// Current hidden state, persists across layers (hidden_size)
    pub(crate) hidden: DeviceVec,
    /// LM head logits (vocab_size)
    pub(crate) logits: DeviceVec,
    /// Decode metadata on GPU: [token_id, current_pos, seq_len] as i32
    pub(crate) decode_meta: CudaSlice<i32>,
    /// FP32 scratch buffer for GPU sampling softmax (vocab_size)
    pub(crate) sample_probs: CudaSlice<f32>,
    /// Pre-allocated sampling output (1 element, token id)
    pub(crate) sample_out: CudaSlice<i32>,
    /// Split-KV partial output accumulator: [num_qheads * NUM_KV_SPLITS * HEAD_DIM] f32
    pub(crate) partial_out: CudaSlice<f32>,
    /// Split-KV partial max: [num_qheads * NUM_KV_SPLITS] f32
    pub(crate) partial_m: CudaSlice<f32>,
    /// Split-KV partial sum: [num_qheads * NUM_KV_SPLITS] f32
    pub(crate) partial_l: CudaSlice<f32>,
    /// Cached raw device pointers for hot-path sampling ops.
    pub(crate) ptrs: DecodeBufferPtrs,
}

impl DecodeBuffers {
    /// NUM_KV_SPLITS must match the Triton AOT compile-time constant.
    const NUM_KV_SPLITS: usize = 4;

    pub(crate) fn new(ctx: &DeviceContext, config: &Config) -> Result<Self> {
        let h = config.hidden_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let num_qheads = config.num_attention_heads;

        let logits = DeviceVec::zeros(ctx, config.vocab_size)?;
        let sample_probs: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(config.vocab_size)
            .map_err(|e| anyhow::anyhow!("Alloc sample_probs failed: {}", e))?;
        let sample_out: CudaSlice<i32> = ctx
            .stream
            .alloc_zeros(1)
            .map_err(|e| anyhow::anyhow!("Alloc sample_out failed: {}", e))?;

        let ptrs = DecodeBufferPtrs {
            logits_ptr: cache_ptr(&logits.data, ctx),
            logits_len: config.vocab_size,
            sample_probs_ptr: cache_ptr(&sample_probs, ctx),
            sample_out_ptr: cache_ptr(&sample_out, ctx),
        };

        Ok(Self {
            normed: DeviceVec::zeros(ctx, h)?,
            q: DeviceVec::zeros(ctx, q_dim)?,
            k: DeviceVec::zeros(ctx, kv_dim)?,
            v: DeviceVec::zeros(ctx, kv_dim)?,
            attn_out: DeviceVec::zeros(ctx, q_dim)?,
            attn_proj: DeviceVec::zeros(ctx, h)?,
            mlp_act: DeviceVec::zeros(ctx, config.intermediate_size)?,
            mlp_up_scratch: DeviceVec::zeros(ctx, config.intermediate_size)?,
            mlp_out: DeviceVec::zeros(ctx, h)?,
            hidden: DeviceVec::zeros(ctx, h)?,
            logits,
            decode_meta: ctx
                .stream
                .alloc_zeros(3)
                .map_err(|e| anyhow::anyhow!("Alloc decode_meta failed: {}", e))?,
            sample_probs,
            sample_out,
            partial_out: ctx
                .stream
                .alloc_zeros(num_qheads * Self::NUM_KV_SPLITS * config.head_dim)
                .map_err(|e| anyhow::anyhow!("Alloc partial_out failed: {}", e))?,
            partial_m: ctx
                .stream
                .alloc_zeros(num_qheads * Self::NUM_KV_SPLITS)
                .map_err(|e| anyhow::anyhow!("Alloc partial_m failed: {}", e))?,
            partial_l: ctx
                .stream
                .alloc_zeros(num_qheads * Self::NUM_KV_SPLITS)
                .map_err(|e| anyhow::anyhow!("Alloc partial_l failed: {}", e))?,
            ptrs,
        })
    }
}
