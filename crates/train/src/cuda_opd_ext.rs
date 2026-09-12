//! OPD control surface over an [`LoadedInferenceEngine`] running the CUDA backend.
//!
//! These were methods on `LoadedInferenceEngine`; they live in train because
//! only OPD teacher/student code calls them. infer-api exposes the two generic
//! downcast seams (`with_cuda_executor` / `with_cuda_engine`); this module
//! names the CUDA executor methods and their types.

use anyhow::Result;
use cuda_kernels::prelude::{DeviceContext, DeviceVec};
use cudarc::driver::DevicePtr;
use infer_api::LoadedInferenceEngine;
use infer_cuda::{
    CudaExecutor, SharedBf16BaseProjection, SharedFp4BaseProjection, SharedFp8BaseProjection,
    StudentLoraUpdate,
};
use infer_seam::BackendExecutor;

/// Raw `[seq_len, vocab]` teacher logits carried back from the engine thread.
pub struct RawLogits {
    pub logits: DeviceVec,
    pub shape: [usize; 2],
    pub device: DeviceContext,
}

impl RawLogits {
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.shape[0]
    }

    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.shape[1]
    }

    pub fn to_host_f32(&self) -> Result<Vec<f32>> {
        self.logits.to_host(&self.device)
    }

    /// Run `f` with the raw logits device pointer; valid only for its duration.
    pub fn with_logits_device_ptr<T>(&self, f: impl FnOnce(u64) -> T) -> T {
        let (ptr, _guard) = self.logits.data.device_ptr(&self.device.stream);
        f(ptr)
    }
}

// SAFETY: produced on the engine thread and handed to one OPD-teacher caller.
unsafe impl Send for RawLogits {}

/// OPD teacher/student operations on a CUDA-backed loaded engine.
pub trait CudaInferenceEngineExt {
    /// Full `[seq_len, vocab]` teacher forward without sampling.
    fn forward_token_logits(&self, input_ids: &[u32], positions: &[u32]) -> Result<RawLogits>;

    /// Trunk taps + final hidden states for offline DSpark draft training.
    fn forward_training_taps(
        &self,
        input_ids: &[u32],
        target_layer_ids: &[i64],
    ) -> Result<(Vec<f32>, Vec<f32>)>;

    /// Re-merge a student LoRA update, invalidate the decode graph and prefix cache.
    fn remerge_student_lora(&self, update: StudentLoraUpdate) -> Result<()>;

    /// Read-only NVFP4 resident base projection pointers.
    fn frozen_base_fp4_pointers(&self) -> Result<Vec<SharedFp4BaseProjection>>;

    /// Read-only FP8 resident base projection pointers.
    fn frozen_base_fp8_pointers(&self) -> Result<Vec<SharedFp8BaseProjection>>;

    /// Read-only dense-BF16 resident base projection pointers.
    fn frozen_base_bf16_pointers(&self) -> Result<Vec<SharedBf16BaseProjection>>;

    /// Hot-swap the DSpark Markov head weights, then drop the prefix cache.
    fn update_dspark_markov_weights(&self, w1: &[f32], w2: &[f32]) -> Result<()>;
}

impl CudaInferenceEngineExt for LoadedInferenceEngine {
    fn forward_token_logits(&self, input_ids: &[u32], positions: &[u32]) -> Result<RawLogits> {
        let input_ids = input_ids.to_vec();
        let positions = positions.to_vec();
        self.with_cuda_executor(move |executor: &mut CudaExecutor| {
            let (logits, shape, device) = executor.forward_token_logits(&input_ids, &positions)?;
            Ok(RawLogits {
                logits,
                shape,
                device,
            })
        })
    }

    fn forward_training_taps(
        &self,
        input_ids: &[u32],
        target_layer_ids: &[i64],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let input_ids = input_ids.to_vec();
        let target_layer_ids = target_layer_ids.to_vec();
        self.with_cuda_executor(move |executor| {
            executor.forward_training_taps(&input_ids, &target_layer_ids)
        })
    }

    fn remerge_student_lora(&self, update: StudentLoraUpdate) -> Result<()> {
        self.with_cuda_engine(move |engine| {
            let executor = engine
                .executor_mut()
                .as_any_mut()
                .downcast_mut::<CudaExecutor>()
                .ok_or_else(|| anyhow::anyhow!("engine backend is not cuda"))?;
            executor.remerge_student_lora(update)?;
            executor.invalidate_decode_graph();
            engine.invalidate_prefix_cache();
            Ok(())
        })
    }

    fn frozen_base_fp4_pointers(&self) -> Result<Vec<SharedFp4BaseProjection>> {
        self.with_cuda_executor(|executor| executor.frozen_base_fp4_pointers())
    }

    fn frozen_base_fp8_pointers(&self) -> Result<Vec<SharedFp8BaseProjection>> {
        self.with_cuda_executor(|executor| executor.frozen_base_fp8_pointers())
    }

    fn frozen_base_bf16_pointers(&self) -> Result<Vec<SharedBf16BaseProjection>> {
        self.with_cuda_executor(|executor| executor.frozen_base_bf16_pointers())
    }

    fn update_dspark_markov_weights(&self, w1: &[f32], w2: &[f32]) -> Result<()> {
        let w1 = w1.to_vec();
        let w2 = w2.to_vec();
        self.with_cuda_engine(move |engine| {
            let executor = engine
                .executor_mut()
                .as_any_mut()
                .downcast_mut::<CudaExecutor>()
                .ok_or_else(|| anyhow::anyhow!("engine backend is not cuda"))?;
            executor.update_dspark_markov_weights(&w1, &w2)?;
            engine.invalidate_prefix_cache();
            Ok(())
        })
    }
}
