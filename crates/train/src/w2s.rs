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
    grad_clip::finite_optimizer_step,
    ops::{add, mul_scalar, softmax},
};

use crate::loss::{KlDirection, kl_distill_loss};
use crate::qwen35::Qwen35Model;
#[cfg(feature = "cuda")]
use crate::teacher_infer::{InferTeacher, TeacherForward};
use crate::trainer::cleanup_after_backward;

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
        let dst_id = *dst_map
            .get(name)
            .ok_or_else(|| anyhow!("sync_lora_adapters: dst missing adapter {name:?}"))?;
        let data = store.to_host(*src_id)?;
        let tensor = store
            .get_mut(dst_id)
            .ok_or_else(|| anyhow!("sync_lora_adapters: tensor {dst_id} not found"))?;
        if tensor.data.len() != data.len() {
            return Err(anyhow!(
                "sync_lora_adapters: length mismatch for {dst_id}: tensor has {}, data has {}",
                tensor.data.len(),
                data.len()
            ));
        }
        tensor.data.copy_from_slice(&data);
        tensor.dirty = autograd::tensor::Dirty::Host;
        store.ensure_device(dst_id)?;
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
#[derive(Clone)]
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

    /// Policy shift ΔT = post_rl_logits − pre_rl_logits (logit space).
    ///
    /// Using the logit-space shift (not log-prob) keeps the proxy teacher
    /// `z_proxy = z_s + α·T·ΔT` in a single space — adding a log-prob
    /// difference to logits mixes coordinate systems and distorts the
    /// softmax. Both forwards run with the tape disabled — the aux models
    /// supply a constant distillation signal; no gradient flows into them.
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
                let neg_pre = mul_scalar(pre_logits, -1.0, store, tape)?;
                let delta = add(post_logits, neg_pre, store, tape)?;
                tape.set_enabled(was_enabled);
                Ok(delta)
            }
            #[cfg(feature = "cuda")]
            Self::Infer { pre_rl, post_rl } => {
                // InferTeacher imports logits as device constants into the store;
                // the tape is not involved (no grad flows into aux models).
                // Reload each teacher's weights onto GPU just-in-time and offload
                // immediately after, so only one aux engine is resident at a time.
                post_rl
                    .reload_engine_weights()
                    .map_err(|e| anyhow!("reload aux post-RL engine: {e}"))?;
                let post_logits = post_rl
                    .forward_logits_device(input_ids, positions, store, tape)
                    .map_err(|e| anyhow!("aux post-RL infer forward: {e}"))?;
                post_rl
                    .offload_engine_weights()
                    .map_err(|e| anyhow!("offload aux post-RL engine: {e}"))?;

                pre_rl
                    .reload_engine_weights()
                    .map_err(|e| anyhow!("reload aux pre-RL engine: {e}"))?;
                let pre_logits = pre_rl
                    .forward_logits_device(input_ids, positions, store, tape)
                    .map_err(|e| anyhow!("aux pre-RL infer forward: {e}"))?;
                pre_rl
                    .offload_engine_weights()
                    .map_err(|e| anyhow!("offload aux pre-RL engine: {e}"))?;

                let neg_pre = mul_scalar(pre_logits.tensor_id, -1.0, store, tape)?;
                let delta = add(post_logits.tensor_id, neg_pre, store, tape)?;
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

#[derive(Debug, Clone)]
pub struct W2sStepOutcome {
    pub loss: f32,
    pub skipped: bool,
    pub skip_reason: Option<SkipReason>,
    pub max_prob: f32,
    pub consistency: f32,
    pub stages: Vec<(&'static str, f64)>,
}

/// Wall-clock per stage, appended in execution order. Every device op the stage
/// enqueued is fenced before the timer stops, so the numbers are GPU time and
/// not launch time.
struct StageTimer {
    stages: Vec<(&'static str, f64)>,
    mark: std::time::Instant,
}

impl StageTimer {
    fn new() -> Self {
        Self {
            stages: Vec::new(),
            mark: std::time::Instant::now(),
        }
    }

    fn stage(&mut self, label: &'static str, store: &TensorStore) {
        let _ = store.backend().device_synchronize();
        self.stages.push((label, self.mark.elapsed().as_secs_f64()));
        self.mark = std::time::Instant::now();
    }
}

/// One w2s training step.
///
/// Flow:
/// 1. Student forward → z_s
/// 2. Confidence filter: max(softmax(z_s)) > threshold → skip
/// 3. Aux ΔT: ΔT₁, ΔT₂
/// 4. Consistency gate: cos(ΔT₁, ΔT₂) < threshold → skip
/// 5. z_proxy = z_s.detach() + α·T·(ΔT₁ + ΔT₂)/2
/// 6. L = T²·KL(softmax(z_s/T) ‖ softmax(z_proxy/T))
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
    share_aux: bool,
) -> Result<W2sStepOutcome> {
    let positions: Vec<u32> = (0..input_ids.len() as u32).collect();
    let num_positions = input_ids.len();
    let mut timer = StageTimer::new();

    // Cleanup keeps every model param and frees the rest. Grads are already gone
    // by then: `zero_grad` frees the tensor and clears the param's grad slot.
    let keep_params: Vec<TensorId> = [
        student.all_parameter_ids(),
        student_old.all_parameter_ids(),
        student_base.all_parameter_ids(),
        aux1.pre_rl_param_ids(),
        aux1.post_rl_param_ids(),
        aux2.pre_rl_param_ids(),
        aux2.post_rl_param_ids(),
    ]
    .concat();
    let keep_extra = HashSet::new();

    // 1. Student forward → z_s
    let z_s = student
        .forward(store, tape, input_ids, &positions)
        .map_err(|e| anyhow!("student forward: {e}"))?;
    timer.stage("student_fwd", store);

    // 2. Confidence filter (host-side for Phase 1).
    // Only the last position matters for generation — prompt positions are
    // always high-confidence (the next token is the fixed prompt text), so
    // max-prob over the whole sequence would skip every sample.
    let student_probs = softmax(z_s, store, tape)?;
    let probs_host = store.to_host(student_probs)?;
    let vocab = store
        .get(z_s)
        .and_then(|t| t.shape.last().copied())
        .unwrap_or(0);
    let last_pos_start = probs_host.len().saturating_sub(vocab);
    let max_prob = probs_host[last_pos_start..]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    if max_prob > cfg.confidence_threshold {
        cleanup_after_backward(store, tape, &keep_params, &keep_extra);
        return Ok(W2sStepOutcome {
            loss: 0.0,
            skipped: true,
            skip_reason: Some(SkipReason::Confidence),
            max_prob,
            consistency: 0.0,
            stages: timer.stages,
        });
    }

    timer.stage("confidence", store);

    // 3. Aux policy shifts ΔT₁, ΔT₂.
    let delta1 = aux1.forward_delta(input_ids, &positions, store, tape)?;
    let delta2 = if share_aux {
        // Both aux models are identical — reuse ΔT₁ instead of running the
        // same forward pass twice (saves one pre-RL + one post-RL reload).
        delta1
    } else {
        aux2.forward_delta(input_ids, &positions, store, tape)?
    };

    timer.stage("aux_delta", store);

    // 4. Consistency gate (host-side for Phase 1).
    let d1_host = store.to_host(delta1)?;
    let d2_host = store.to_host(delta2)?;
    let consistency = cosine_similarity(&d1_host, &d2_host);
    // NaN consistency means the aux forward produced NaN logits (e.g. INT8
    // quantization overflow on some inputs). Skip rather than poisoning the
    // student's adapter weights with a NaN gradient.
    if consistency.is_nan() || consistency < cfg.consistency_threshold {
        cleanup_after_backward(store, tape, &keep_params, &keep_extra);
        return Ok(W2sStepOutcome {
            loss: 0.0,
            skipped: true,
            skip_reason: Some(SkipReason::Consistency),
            max_prob,
            consistency,
            stages: timer.stages,
        });
    }

    timer.stage("consistency", store);

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

    timer.stage("kd_loss", store);

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

    timer.stage("local_kl", store);

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

    timer.stage("global_kl", store);

    // 7. Backward + optimizer step.
    let loss_value = store.to_host(loss)?[0];
    tape.backward(loss, store)?;
    timer.stage("backward", store);
    finite_optimizer_step(
        loss_value,
        trainable_params,
        cfg.grad_clip,
        optimizer,
        store,
    )?;
    timer.stage("optimizer", store);

    cleanup_after_backward(store, tape, &keep_params, &keep_extra);
    timer.stage("cleanup", store);

    Ok(W2sStepOutcome {
        loss: loss_value,
        skipped: false,
        skip_reason: None,
        max_prob,
        consistency,
        stages: timer.stages,
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
