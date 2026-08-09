//! Weak-to-strong (w2s) online distillation.
//!
//! Core idea: do NOT distill the weak teacher's final output directly.
//! Distill the **policy shift** ΔT = log π_post-RL − log π_pre-RL, build a
//! proxy teacher from the student's own logits + the averaged shift, and
//! train the student with reverse KL.
//!
//! Two auxiliary models (each with a pre-RL base and post-RL instruct
//! checkpoint) supply ΔT₁ and ΔT₂. Their cosine similarity gates the update;
//! the student's own max-probability filters samples it is already confident
//! about. Local (vs previous adapter) and global (vs base) KL regularizers
//! prevent catastrophic forgetting.

use std::collections::HashSet;

use anyhow::{Result, anyhow};
use autograd::{
    Optimizer, Tape, TensorId, TensorStore,
    ops::{add, log_softmax, mul_scalar, softmax},
};

use crate::loss::{KlDirection, kl_distill_loss};
use crate::qwen35::Qwen35Model;
#[cfg(feature = "cuda")]
use crate::teacher_infer::{InferTeacher, TeacherForward};

/// Copy host data into a tensor and mark it dirty for device upload.
fn write_tensor_data(store: &mut TensorStore, id: TensorId, data: &[f32]) -> Result<()> {
    {
        let tensor = store
            .get_mut(id)
            .ok_or_else(|| anyhow!("write_tensor_data: tensor {id} not found"))?;
        if tensor.data.len() != data.len() {
            return Err(anyhow!(
                "write_tensor_data: length mismatch for {id}: tensor has {}, data has {}",
                tensor.data.len(),
                data.len()
            ));
        }
        tensor.data.copy_from_slice(data);
        tensor.dirty = autograd::tensor::Dirty::Host;
    }
    store.ensure_device(id)?;
    Ok(())
}

/// Copy all LoRA adapter weights from `src` to `dst` (shadow → serving or
/// serving → shadow). Both models must share the same adapter layout (created
/// from the same base).
pub fn sync_lora_adapters(
    src: &Qwen35Model,
    dst: &Qwen35Model,
    store: &mut TensorStore,
) -> Result<()> {
    let src_map = src.adapter_name_map();
    let dst_map = dst.adapter_name_map();
    for (name, src_id) in &src_map {
        let dst_id = dst_map
            .get(name)
            .ok_or_else(|| anyhow!("sync_lora_adapters: dst missing adapter {name:?}"))?;
        let data = store.to_host(*src_id)?;
        write_tensor_data(store, *dst_id, &data)?;
    }
    Ok(())
}

/// A single auxiliary model: holds the pre-RL (base) and post-RL (instruct)
/// checkpoints. Both are frozen — only the student trains.
///
/// Two backends:
/// - `InProcess`: autograd-loaded `Qwen35Model`, for small models that fit in
///   the training process's GPU memory alongside the student.
/// - `Infer`: `LoadedInferenceEngine`-backed, for large models that need TP or
///   would otherwise OOM the training process. The logits are imported into
///   the train store as constants (no gradient flows into the aux models).
pub enum W2sAuxModel {
    InProcess {
        pre_rl: Qwen35Model,
        post_rl: Qwen35Model,
    },
    #[cfg(feature = "cuda")]
    Infer {
        pre_rl: InferTeacher,
        post_rl: InferTeacher,
    },
}

impl W2sAuxModel {
    pub fn new_in_process(pre_rl: Qwen35Model, post_rl: Qwen35Model) -> Self {
        Self::InProcess { pre_rl, post_rl }
    }

    #[cfg(feature = "cuda")]
    pub fn new_infer(pre_rl: InferTeacher, post_rl: InferTeacher) -> Self {
        Self::Infer { pre_rl, post_rl }
    }

    /// Pre-RL model parameter IDs (in-process only; infer teachers have no
    /// train-store params).
    pub fn pre_rl_param_ids(&self) -> Vec<TensorId> {
        match self {
            Self::InProcess { pre_rl, .. } => pre_rl.all_parameter_ids(),
            #[cfg(feature = "cuda")]
            Self::Infer { .. } => vec![],
        }
    }

    /// Post-RL model parameter IDs.
    pub fn post_rl_param_ids(&self) -> Vec<TensorId> {
        match self {
            Self::InProcess { post_rl, .. } => post_rl.all_parameter_ids(),
            #[cfg(feature = "cuda")]
            Self::Infer { .. } => vec![],
        }
    }

    /// Policy shift ΔT = log_softmax(post_rl) − log_softmax(pre_rl).
    ///
    /// Both forwards run with the tape disabled — the aux models supply a
    /// constant distillation signal; no gradient flows into them.
    pub fn forward_delta(
        &self,
        input_ids: &[u32],
        positions: &[u32],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        match self {
            Self::InProcess { pre_rl, post_rl } => {
                let was_enabled = tape.enabled;
                tape.set_enabled(false);
                let post_logits = post_rl
                    .forward(store, tape, input_ids, positions)
                    .map_err(|e| anyhow!("aux post-RL forward: {e}"))?;
                let pre_logits = pre_rl
                    .forward(store, tape, input_ids, positions)
                    .map_err(|e| anyhow!("aux pre-RL forward: {e}"))?;
                let lp_post = log_softmax(post_logits, store, tape)?;
                let lp_pre = log_softmax(pre_logits, store, tape)?;
                let neg_pre = mul_scalar(lp_pre, -1.0, store, tape)?;
                let delta = add(lp_post, neg_pre, store, tape)?;
                tape.set_enabled(was_enabled);
                Ok(delta)
            }
            #[cfg(feature = "cuda")]
            Self::Infer { pre_rl, post_rl } => {
                // InferTeacher imports logits as device constants into the store;
                // the tape is not involved (no grad flows into aux models).
                let post_logits = post_rl
                    .forward_logits_device(input_ids, positions, store, tape)
                    .map_err(|e| anyhow!("aux post-RL infer forward: {e}"))?;
                let pre_logits = pre_rl
                    .forward_logits_device(input_ids, positions, store, tape)
                    .map_err(|e| anyhow!("aux pre-RL infer forward: {e}"))?;
                let lp_post = log_softmax(post_logits.tensor_id, store, tape)?;
                let lp_pre = log_softmax(pre_logits.tensor_id, store, tape)?;
                let neg_pre = mul_scalar(lp_pre, -1.0, store, tape)?;
                let delta = add(lp_post, neg_pre, store, tape)?;
                Ok(delta)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct W2sConfig {
    /// Distillation strength α. Tune so α·T·‖ΔT‖ is comparable to ‖z_student‖.
    pub alpha: f32,
    /// Softmax temperature T (2–4).
    pub temperature: f32,
    /// Skip samples where the student's max probability exceeds this (it is
    /// already confident; the weak signal is likely noise).
    pub confidence_threshold: f32,
    /// Skip samples where cos(ΔT₁, ΔT₂) is below this (the two weak models
    /// disagree on the direction of the shift).
    pub consistency_threshold: f32,
    /// Local KL regularizer weight β₁ — pulls π_new toward π_old.
    pub beta_local: f32,
    /// Global KL regularizer weight β₂ — pulls π_new toward π_base.
    pub beta_global: f32,
    pub grad_clip: f32,
}

impl Default for W2sConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            temperature: 2.0,
            confidence_threshold: 0.9,
            consistency_threshold: 0.0,
            beta_local: 0.01,
            beta_global: 0.001,
            grad_clip: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// Student's max probability exceeded the confidence threshold.
    Confidence,
    /// cos(ΔT₁, ΔT₂) was below the consistency threshold.
    Consistency,
}

#[derive(Debug, Clone, Copy)]
pub struct W2sStepOutcome {
    pub loss: f32,
    pub skipped: bool,
    pub skip_reason: Option<SkipReason>,
    pub max_prob: f32,
    pub consistency: f32,
}

/// One w2s training step.
///
/// Flow:
/// 1. Student forward → z_s
/// 2. Confidence filter: max(softmax(z_s)) > threshold → skip
/// 3. Aux ΔT: ΔT₁, ΔT₂
/// 4. Consistency gate: cos(ΔT₁, ΔT₂) < threshold → skip
/// 5. z_proxy = z_s.detach() + α·T·(ΔT₁ + ΔT₂)/2
/// 6. L = T²·KL(softmax(z_proxy/T) ‖ softmax(z_s/T))
///      + β₁·KL(π_new ‖ π_old) + β₂·KL(π_new ‖ π_base)
/// 7. Backward → shadow adapter
#[allow(clippy::too_many_arguments)]
pub fn w2s_step<O: Optimizer>(
    student: &Qwen35Model,
    student_old: &Qwen35Model,
    student_base: &Qwen35Model,
    aux1: &W2sAuxModel,
    aux2: &W2sAuxModel,
    input_ids: &[u32],
    cfg: &W2sConfig,
    trainable_params: &[TensorId],
    optimizer: &mut O,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<W2sStepOutcome> {
    let positions: Vec<u32> = (0..input_ids.len() as u32).collect();
    let num_positions = input_ids.len();

    // Cleanup: keep all model params, free everything else. Grads are freed by
    // `zero_grad` after the optimizer step — keeping them here would silently
    // accumulate across steps (backward adds to existing grads).
    let cleanup = |store: &mut TensorStore, tape: &mut Tape| {
        let mut keep = HashSet::new();
        for model in [
            student.all_parameter_ids(),
            student_old.all_parameter_ids(),
            student_base.all_parameter_ids(),
            aux1.pre_rl_param_ids(),
            aux1.post_rl_param_ids(),
            aux2.pre_rl_param_ids(),
            aux2.post_rl_param_ids(),
        ] {
            keep.extend(model);
        }
        store.retain_ids(&keep);
        tape.entries.clear();
    };

    // 1. Student forward → z_s
    let z_s = student
        .forward(store, tape, input_ids, &positions)
        .map_err(|e| anyhow!("student forward: {e}"))?;

    // 2. Confidence filter (host-side for Phase 1).
    let student_probs = softmax(z_s, store, tape)?;
    let probs_host = store.to_host(student_probs)?;
    let max_prob = probs_host.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_prob > cfg.confidence_threshold {
        cleanup(store, tape);
        return Ok(W2sStepOutcome {
            loss: 0.0,
            skipped: true,
            skip_reason: Some(SkipReason::Confidence),
            max_prob,
            consistency: 0.0,
        });
    }

    // 3. Aux policy shifts ΔT₁, ΔT₂.
    let delta1 = aux1.forward_delta(input_ids, &positions, store, tape)?;
    let delta2 = aux2.forward_delta(input_ids, &positions, store, tape)?;

    // 4. Consistency gate (host-side for Phase 1).
    let d1_host = store.to_host(delta1)?;
    let d2_host = store.to_host(delta2)?;
    let consistency = cosine_similarity(&d1_host, &d2_host);
    if consistency < cfg.consistency_threshold {
        cleanup(store, tape);
        return Ok(W2sStepOutcome {
            loss: 0.0,
            skipped: true,
            skip_reason: Some(SkipReason::Consistency),
            max_prob,
            consistency,
        });
    }

    // 5. Build proxy teacher: z_proxy = z_s.detach() + α·T·(ΔT₁ + ΔT₂)/2.
    let z_s_detached = store.detach(z_s)?;
    let avg_delta = mul_scalar(add(delta1, delta2, store, tape)?, 0.5, store, tape)?;
    let scaled_delta = mul_scalar(avg_delta, cfg.alpha * cfg.temperature, store, tape)?;
    let z_proxy = add(z_s_detached, scaled_delta, store, tape)?;

    // 6a. Reverse KL distillation loss.
    let loss_kd = kl_distill_loss(
        z_s,
        z_proxy,
        num_positions,
        cfg.temperature,
        KlDirection::Reverse,
        store,
        tape,
    )?;

    // 6b. Local KL regularizer: π_new vs π_old (previous adapter).
    let z_old = student_old
        .forward(store, tape, input_ids, &positions)
        .map_err(|e| anyhow!("student_old forward: {e}"))?;
    let z_old_detached = store.detach(z_old)?;
    let loss_local = kl_distill_loss(
        z_s,
        z_old_detached,
        num_positions,
        1.0,
        KlDirection::Reverse,
        store,
        tape,
    )?;

    // 6c. Global KL regularizer: π_new vs π_base (no adapter).
    let z_base = student_base
        .forward(store, tape, input_ids, &positions)
        .map_err(|e| anyhow!("student_base forward: {e}"))?;
    let z_base_detached = store.detach(z_base)?;
    let loss_global = kl_distill_loss(
        z_s,
        z_base_detached,
        num_positions,
        1.0,
        KlDirection::Reverse,
        store,
        tape,
    )?;

    let loss_local_scaled = mul_scalar(loss_local, cfg.beta_local, store, tape)?;
    let loss_global_scaled = mul_scalar(loss_global, cfg.beta_global, store, tape)?;
    let loss_reg = add(loss_local_scaled, loss_global_scaled, store, tape)?;
    let loss = add(loss_kd, loss_reg, store, tape)?;

    // 7. Backward + optimizer step.
    let loss_value = store.to_host(loss)?[0];
    tape.backward(loss, store)?;
    if cfg.grad_clip > 0.0 {
        crate::grad_clip::clip_grad_norm(trainable_params, cfg.grad_clip, store);
    }
    optimizer.step(store, trainable_params)?;
    optimizer.zero_grad(store, trainable_params);

    cleanup(store, tape);

    Ok(W2sStepOutcome {
        loss: loss_value,
        skipped: false,
        skip_reason: None,
        max_prob,
        consistency,
    })
}

/// Host-side cosine similarity of two flat f32 vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine_similarity: length mismatch");
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

#[cfg(test)]
mod tests {
    use super::cosine_similarity;

    #[test]
    fn cosine_identical_vectors_is_one() {
        let a = [1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "identical vectors should have cosine 1.0, got {sim}"
        );
    }

    #[test]
    fn cosine_orthogonal_vectors_is_zero() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-6,
            "orthogonal vectors should have cosine 0.0, got {sim}"
        );
    }

    #[test]
    fn cosine_opposite_vectors_is_negative_one() {
        let a = [1.0, 2.0, 3.0];
        let b = [-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim + 1.0).abs() < 1e-6,
            "opposite vectors should have cosine -1.0, got {sim}"
        );
    }

    #[test]
    fn cosine_zero_vector_returns_zero() {
        let a = [0.0, 0.0];
        let b = [1.0, 2.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "zero vector should return 0.0");
    }
}
