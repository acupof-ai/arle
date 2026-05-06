//! GPU operations on device tensors.

use anyhow::Result;

use crate::sampler::SamplingParams;

/// Backend tag for opaque op tensors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpsBackendKind {
    Cuda,
    Metal,
}

/// Opaque activation tensor for cross-backend op plumbing.
///
/// Hot paths should prefer [`OpsBackend`] associated types when monomorphized
/// dispatch is needed; this owned wrapper is for backend-neutral boundaries
/// that need to carry a tensor without exposing cudarc or MLX handles.
pub struct Tensor {
    repr: TensorRepr,
}

#[allow(dead_code)]
enum TensorRepr {
    #[cfg(feature = "cuda")]
    Cuda(CudaTensor),
    #[cfg(feature = "metal")]
    Metal(crate::backend::metal::mlx::MlxArray),
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
enum CudaTensor {
    Vector(cuda_kernels::prelude::DeviceVec),
    Batch(cuda_kernels::prelude::HiddenStates),
}

#[allow(dead_code)]
impl Tensor {
    pub fn backend(&self) -> OpsBackendKind {
        match &self.repr {
            #[cfg(feature = "cuda")]
            TensorRepr::Cuda(_) => OpsBackendKind::Cuda,
            #[cfg(feature = "metal")]
            TensorRepr::Metal(_) => OpsBackendKind::Metal,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn from_cuda_vec(value: cuda_kernels::prelude::DeviceVec) -> Self {
        Self {
            repr: TensorRepr::Cuda(CudaTensor::Vector(value)),
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn from_cuda_batch(value: cuda_kernels::prelude::HiddenStates) -> Self {
        Self {
            repr: TensorRepr::Cuda(CudaTensor::Batch(value)),
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn as_cuda_vec(&self) -> Option<&cuda_kernels::prelude::DeviceVec> {
        match &self.repr {
            TensorRepr::Cuda(CudaTensor::Vector(value)) => Some(value),
            TensorRepr::Cuda(CudaTensor::Batch(_)) => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn as_cuda_vec_mut(&mut self) -> Option<&mut cuda_kernels::prelude::DeviceVec> {
        match &mut self.repr {
            TensorRepr::Cuda(CudaTensor::Vector(value)) => Some(value),
            TensorRepr::Cuda(CudaTensor::Batch(_)) => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn as_cuda_batch(&self) -> Option<&cuda_kernels::prelude::HiddenStates> {
        match &self.repr {
            TensorRepr::Cuda(CudaTensor::Batch(value)) => Some(value),
            TensorRepr::Cuda(CudaTensor::Vector(_)) => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn as_cuda_batch_mut(&mut self) -> Option<&mut cuda_kernels::prelude::HiddenStates> {
        match &mut self.repr {
            TensorRepr::Cuda(CudaTensor::Batch(value)) => Some(value),
            TensorRepr::Cuda(CudaTensor::Vector(_)) => None,
        }
    }

    #[cfg(feature = "metal")]
    pub(crate) fn from_metal_array(value: crate::backend::metal::mlx::MlxArray) -> Self {
        Self {
            repr: TensorRepr::Metal(value),
        }
    }

    #[cfg(feature = "metal")]
    pub(crate) fn as_metal_array(&self) -> Option<&crate::backend::metal::mlx::MlxArray> {
        match &self.repr {
            TensorRepr::Metal(value) => Some(value),
            #[cfg(feature = "cuda")]
            TensorRepr::Cuda(_) => None,
        }
    }

    #[cfg(feature = "metal")]
    pub(crate) fn as_metal_array_mut(
        &mut self,
    ) -> Option<&mut crate::backend::metal::mlx::MlxArray> {
        match &mut self.repr {
            TensorRepr::Metal(value) => Some(value),
            #[cfg(feature = "cuda")]
            TensorRepr::Cuda(_) => None,
        }
    }
}

/// Monomorphized op surface shared by CUDA and Metal implementations.
///
/// Associated types keep backend handles out of the trait interface while
/// allowing hot-path call sites to compile to direct backend calls.
pub trait OpsBackend {
    type Tensor;
    type TensorBatch;
    type Matrix;
    type TokenIds;
    type SamplingScratch;
    type SamplingOutput;

    fn backend_kind(&self) -> OpsBackendKind;

    fn rms_norm_into(
        &self,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
        out: &mut Self::Tensor,
    ) -> Result<()>;

    fn fused_add_rms_norm_into(
        &self,
        hidden: &mut Self::Tensor,
        residual: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
        out: &mut Self::Tensor,
    ) -> Result<()>;

    fn rms_norm_batch_into(
        &self,
        x: &Self::TensorBatch,
        weight: &Self::Tensor,
        eps: f32,
        out: &mut Self::TensorBatch,
    ) -> Result<()>;

    fn fused_add_rms_norm_batch_into(
        &self,
        hidden: &mut Self::TensorBatch,
        residual: &Self::TensorBatch,
        weight: &Self::Tensor,
        eps: f32,
        out: &mut Self::TensorBatch,
    ) -> Result<()>;

    fn linear_vec_into(
        &self,
        weight: &Self::Matrix,
        input: &Self::Tensor,
        output: &mut Self::Tensor,
    ) -> Result<()>;

    fn linear_batch_into(
        &self,
        weight: &Self::Matrix,
        input: &Self::TensorBatch,
        output: &mut Self::TensorBatch,
    ) -> Result<()>;

    fn fused_mlp_into(
        &self,
        input: &Self::Tensor,
        gate_proj: &Self::Matrix,
        up_proj: &Self::Matrix,
        down_proj: &Self::Matrix,
        act: &mut Self::Tensor,
        out: &mut Self::Tensor,
    ) -> Result<()>;

    fn add_batch_into(
        &self,
        a: &Self::TensorBatch,
        b: &Self::TensorBatch,
        out: &mut Self::TensorBatch,
    ) -> Result<()>;

    fn silu_mul_batch_into(
        &self,
        gate: &Self::TensorBatch,
        up: &Self::TensorBatch,
        out: &mut Self::TensorBatch,
    ) -> Result<()>;

    fn extract_vec_into(
        &self,
        batch: &Self::TensorBatch,
        token_idx: usize,
        out: &mut Self::Tensor,
    ) -> Result<()>;

    fn embedding_decode_into(
        &self,
        embed: &Self::Matrix,
        token_ids: &Self::TokenIds,
        out: &mut Self::Tensor,
    ) -> Result<()>;

    fn embedding_batch_into(
        &self,
        embed: &Self::Matrix,
        token_ids: &Self::TokenIds,
        out: &mut Self::TensorBatch,
    ) -> Result<()>;

    fn sample_token_into(
        &self,
        logits: &Self::Tensor,
        scratch: &mut Self::SamplingScratch,
        out: &mut Self::SamplingOutput,
        params: &SamplingParams,
        random_val: f32,
    ) -> Result<u32>;
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
pub struct CudaOpsBackend<'ctx> {
    ctx: &'ctx cuda_kernels::prelude::DeviceContext,
}

#[cfg(feature = "cuda")]
impl<'ctx> CudaOpsBackend<'ctx> {
    pub fn new(ctx: &'ctx cuda_kernels::prelude::DeviceContext) -> Self {
        Self { ctx }
    }

    pub fn context(&self) -> &'ctx cuda_kernels::prelude::DeviceContext {
        self.ctx
    }
}

#[cfg(feature = "cuda")]
impl OpsBackend for CudaOpsBackend<'_> {
    type Tensor = cuda_kernels::prelude::DeviceVec;
    type TensorBatch = cuda_kernels::prelude::HiddenStates;
    type Matrix = cuda_kernels::prelude::DeviceMatrix;
    type TokenIds = cudarc::driver::CudaSlice<i32>;
    type SamplingScratch = cudarc::driver::CudaSlice<f32>;
    type SamplingOutput = cudarc::driver::CudaSlice<i32>;

    fn backend_kind(&self) -> OpsBackendKind {
        OpsBackendKind::Cuda
    }

    fn rms_norm_into(
        &self,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
        out: &mut Self::Tensor,
    ) -> Result<()> {
        norm::rms_norm_into(self.ctx, x, weight, eps, out)
    }

    fn fused_add_rms_norm_into(
        &self,
        hidden: &mut Self::Tensor,
        residual: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
        out: &mut Self::Tensor,
    ) -> Result<()> {
        norm::fused_add_rms_norm_into(self.ctx, hidden, residual, weight, eps, out)
    }

    fn rms_norm_batch_into(
        &self,
        x: &Self::TensorBatch,
        weight: &Self::Tensor,
        eps: f32,
        out: &mut Self::TensorBatch,
    ) -> Result<()> {
        norm::rms_norm_batch_into(self.ctx, x, weight, eps, out);
        Ok(())
    }

    fn fused_add_rms_norm_batch_into(
        &self,
        hidden: &mut Self::TensorBatch,
        residual: &Self::TensorBatch,
        weight: &Self::Tensor,
        eps: f32,
        out: &mut Self::TensorBatch,
    ) -> Result<()> {
        norm::fused_add_rms_norm_batch_into(self.ctx, hidden, residual, weight, eps, out);
        Ok(())
    }

    fn linear_vec_into(
        &self,
        weight: &Self::Matrix,
        input: &Self::Tensor,
        output: &mut Self::Tensor,
    ) -> Result<()> {
        linear::gemv(self.ctx, weight, input, output)
    }

    fn linear_batch_into(
        &self,
        weight: &Self::Matrix,
        input: &Self::TensorBatch,
        output: &mut Self::TensorBatch,
    ) -> Result<()> {
        linear::gemm_into(self.ctx, weight, input, output);
        Ok(())
    }

    fn fused_mlp_into(
        &self,
        input: &Self::Tensor,
        gate_proj: &Self::Matrix,
        up_proj: &Self::Matrix,
        down_proj: &Self::Matrix,
        act: &mut Self::Tensor,
        out: &mut Self::Tensor,
    ) -> Result<()> {
        linear::fused_mlp_into(self.ctx, input, gate_proj, up_proj, down_proj, act, out)
    }

    fn add_batch_into(
        &self,
        a: &Self::TensorBatch,
        b: &Self::TensorBatch,
        out: &mut Self::TensorBatch,
    ) -> Result<()> {
        elementwise::add_batch_into(self.ctx, a, b, out)
    }

    fn silu_mul_batch_into(
        &self,
        gate: &Self::TensorBatch,
        up: &Self::TensorBatch,
        out: &mut Self::TensorBatch,
    ) -> Result<()> {
        elementwise::silu_mul_batch_into(self.ctx, gate, up, out)
    }

    fn extract_vec_into(
        &self,
        batch: &Self::TensorBatch,
        token_idx: usize,
        out: &mut Self::Tensor,
    ) -> Result<()> {
        elementwise::extract_vec_into(self.ctx, batch, token_idx, out)
    }

    fn embedding_decode_into(
        &self,
        embed: &Self::Matrix,
        token_ids: &Self::TokenIds,
        out: &mut Self::Tensor,
    ) -> Result<()> {
        embedding::embedding_decode_into(self.ctx, embed, token_ids, out)
    }

    fn embedding_batch_into(
        &self,
        embed: &Self::Matrix,
        token_ids: &Self::TokenIds,
        out: &mut Self::TensorBatch,
    ) -> Result<()> {
        embedding::embedding_batch(self.ctx, embed, token_ids, out)
    }

    fn sample_token_into(
        &self,
        logits: &Self::Tensor,
        scratch: &mut Self::SamplingScratch,
        out: &mut Self::SamplingOutput,
        params: &SamplingParams,
        random_val: f32,
    ) -> Result<u32> {
        sampling::gpu_sample_into(self.ctx, logits, scratch, out, params, random_val)
    }
}

#[cfg(feature = "cuda")]
#[path = "ops/attention.rs"]
mod attention;
#[cfg(feature = "cuda")]
#[path = "ops/elementwise.rs"]
mod elementwise;
#[cfg(feature = "cuda")]
#[path = "ops/embedding.rs"]
mod embedding;
#[cfg(feature = "cuda")]
#[path = "ops/kv_ops.rs"]
mod kv_ops;
#[cfg(feature = "cuda")]
#[path = "ops/linear.rs"]
mod linear;
#[cfg(feature = "cuda")]
#[path = "ops/norm.rs"]
mod norm;
#[cfg(feature = "cuda")]
#[path = "ops/recurrent.rs"]
mod recurrent;
#[cfg(feature = "cuda")]
#[path = "ops/sampling.rs"]
mod sampling;

#[cfg(all(test, feature = "cuda"))]
#[path = "ops/tests.rs"]
mod tests;

// pub re-exports
#[cfg(feature = "cuda")]
pub use attention::{
    TileLangHeadConfig, fused_attention_decode_batched_into, fused_attention_decode_into,
    tilelang_tc_run_layer,
};
#[cfg(feature = "cuda")]
pub(crate) use attention::{
    decode_prep_paged, prefill_attention_batch, prefill_attention_hd256_batch,
    prefill_attention_hd256_batch_with_scratch, prefill_attention_paged_batch,
    prefill_attention_paged_run_hd256,
};
#[cfg(feature = "cuda")]
pub use elementwise::{add_batch, silu_mul_batch};
#[cfg(feature = "cuda")]
pub use embedding::{embedding_batch, embedding_decode_into};
#[cfg(feature = "cuda")]
pub use kv_ops::scatter_write_kv;
#[cfg(feature = "cuda")]
pub use linear::{
    apply_lora_gemm_add, apply_lora_gemv_add, fused_mlp_into, gemm, gemv, mlp_decode_with_lora_into,
};
#[cfg(feature = "cuda")]
pub use norm::{
    fused_add_rms_norm_into, fused_add_rms_norm_offset_into, rms_norm_batch_offset_into,
    rms_norm_gated_into, rms_norm_into, rms_norm_offset_into,
};
#[cfg(feature = "cuda")]
pub use recurrent::{GdrHeadConfig, GdrWeights, gated_delta_rule_prefill_chunkwise_into};
#[cfg(feature = "cuda")]
pub use sampling::{
    argmax, argmax_with_logprob, gpu_sample, gpu_sample_into, gpu_sample_launch,
    gpu_sample_readback,
};

// pub(crate) re-exports
#[cfg(all(test, feature = "cuda"))]
pub(crate) use attention::nonpaged_prefill_hd256_into;
#[cfg(feature = "cuda")]
pub(crate) use attention::{
    HeadConfig, NormRopeParams, PagedKVMeta, PagedPrefillForward, PagedPrefillMeta,
    PagedPrefillSequence, attention_gate_paged_hd256, decode_prep_paged_hd256,
    tilelang_run_layer_hd256,
};
#[cfg(feature = "cuda")]
pub(crate) use elementwise::{add_batch_into, extract_vec, extract_vec_into, silu_mul_batch_into};
#[cfg(feature = "cuda")]
pub(crate) use linear::{gemm_graphsafe_batched_into, gemm_into, linear};
#[cfg(feature = "cuda")]
#[allow(unused_imports)]
pub(crate) use norm::{
    add_bf16_into_f32, cast_bf16_to_f32, cast_f32_to_bf16, fused_add_rms_norm_batch_into, rms_norm,
    rms_norm_batch_f32_in_into, rms_norm_batch_into, rms_norm_gated_batch_into,
};
#[cfg(feature = "cuda")]
pub(crate) use recurrent::{
    Conv1dPrefillBatchLaunch, GdrPrefillBatchLaunch, conv1d_decode_batch_into,
    conv1d_prefill_batch_into, conv1d_prefill_packed_batch_into, gated_delta_rule_decode_into,
    gated_delta_rule_prefill_chunkwise_batch_into, gdr_decode_batch_into,
};
#[cfg(feature = "cuda")]
pub(crate) use sampling::{
    argmax_batch_logprob_launch, argmax_batch_readback_into, gpu_sample_launch_raw,
};
