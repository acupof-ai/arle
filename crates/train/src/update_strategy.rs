//! Data-driven OPD policy-update seam. An algorithm is a [`UpdatePreset`] —
//! six orthogonal fields over ONE update path — not a new enum arm; the named
//! constructors below are the algorithm table. The rollout/scoring harness is
//! preset-agnostic: it only reads [`UpdatePreset::needs`].

use autograd::ops::fused_linear_distill::{PgStats, WeightForm, pg_token_weight};
use autograd::{TensorId, TensorStore, optim::Optimizer};

use crate::grad_clip::finite_optimizer_step;
use crate::opd::{
    OpdError, Result, ValueCritic, WritebackLoss, capture_rollout_logprobs,
    masked_writeback_ce_step_dispatch, masked_writeback_step,
};
use crate::qwen35::Qwen35Model;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RolloutNeeds {
    pub keep_failing: bool,
    pub behavior_logprobs: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ScoredTrajectory {
    pub prompt_ids: Vec<u32>,
    pub response_ids: Vec<u32>,
    pub response_mask: Vec<u8>,
    pub reward: f32,
    pub behavior_logprobs: Option<Vec<f32>>,
    pub group_id: usize,
    pub truncated: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleFilter {
    PassOnly,
    KeepAll,
    DropZeroAdvGroup,
    DropTruncated,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    Batch,
    Group,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Advantage {
    None,
    Mean { scope: Scope, std_norm: bool },
    ValueGae { gamma: f32, lam: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RatioGrain {
    None,
    PerToken,
    PerSequence,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClipForm {
    HardGate { lo: f32, hi: f32 },
    DetachedSoftClamp { lo: f32, hi: f32 },
    PpoClip { lo: f32, hi: f32 },
}

impl ClipForm {
    fn weight_form(self) -> WeightForm {
        match self {
            Self::HardGate { lo, hi } => WeightForm::HardGate { lo, hi },
            Self::DetachedSoftClamp { lo, hi } => WeightForm::DetachedSoftClamp { lo, hi },
            Self::PpoClip { lo, hi } => WeightForm::PpoClip { lo, hi },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Aggregation {
    PerSeqTokenMean,
    GlobalTokenMean { norm_const: Option<usize> },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct UpdateReport {
    pub loss: f32,
    pub trained: usize,
    pub tokens: usize,
    pub stats: PgStats,
    pub critic_mse: f32,
    pub adv_mean: f32,
    pub adv_std: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UpdatePreset {
    pub filter: SampleFilter,
    pub advantage: Advantage,
    pub ratio: RatioGrain,
    pub clip: ClipForm,
    pub agg: Aggregation,
}

impl UpdatePreset {
    pub fn rejection_ce() -> Self {
        Self {
            filter: SampleFilter::PassOnly,
            advantage: Advantage::None,
            ratio: RatioGrain::None,
            // Unused: ratio None routes to the fused CE op.
            clip: ClipForm::HardGate { lo: 0.0, hi: 0.0 },
            agg: Aggregation::PerSeqTokenMean,
        }
    }

    pub fn sao_dis(lo: f32, hi: f32) -> Self {
        Self {
            filter: SampleFilter::KeepAll,
            advantage: Advantage::Mean {
                scope: Scope::Batch,
                std_norm: false,
            },
            ratio: RatioGrain::PerToken,
            clip: ClipForm::HardGate { lo, hi },
            agg: Aggregation::PerSeqTokenMean,
        }
    }

    pub fn sao_value(lo: f32, hi: f32, gamma: f32, lam: f32) -> Self {
        Self {
            advantage: Advantage::ValueGae { gamma, lam },
            ..Self::sao_dis(lo, hi)
        }
    }

    pub fn grpo() -> Self {
        Self {
            filter: SampleFilter::KeepAll,
            advantage: Advantage::Mean {
                scope: Scope::Group,
                std_norm: true,
            },
            ratio: RatioGrain::PerToken,
            clip: ClipForm::PpoClip { lo: 0.2, hi: 0.2 },
            agg: Aggregation::PerSeqTokenMean,
        }
    }

    pub fn dapo() -> Self {
        Self {
            filter: SampleFilter::DropZeroAdvGroup,
            advantage: Advantage::Mean {
                scope: Scope::Group,
                std_norm: false,
            },
            clip: ClipForm::PpoClip { lo: 0.2, hi: 0.28 },
            agg: Aggregation::GlobalTokenMean { norm_const: None },
            ..Self::grpo()
        }
    }

    pub fn dr_grpo(norm_const: usize) -> Self {
        Self {
            advantage: Advantage::Mean {
                scope: Scope::Group,
                std_norm: false,
            },
            agg: Aggregation::GlobalTokenMean {
                norm_const: Some(norm_const),
            },
            ..Self::grpo()
        }
    }

    pub fn gspo() -> Self {
        Self {
            ratio: RatioGrain::PerSequence,
            clip: ClipForm::PpoClip { lo: 3e-4, hi: 4e-4 },
            ..Self::grpo()
        }
    }

    pub fn cispo() -> Self {
        Self {
            clip: ClipForm::DetachedSoftClamp { lo: 1.0, hi: 3.0 },
            agg: Aggregation::GlobalTokenMean { norm_const: None },
            ..Self::grpo()
        }
    }

    pub fn needs(&self) -> RolloutNeeds {
        RolloutNeeds {
            keep_failing: self.filter != SampleFilter::PassOnly,
            behavior_logprobs: self.ratio != RatioGrain::None,
        }
    }

    pub fn needs_value_critic(&self) -> bool {
        matches!(self.advantage, Advantage::ValueGae { .. })
    }

    fn filter_batch<'a>(&self, batch: &'a [ScoredTrajectory]) -> Vec<&'a ScoredTrajectory> {
        match self.filter {
            SampleFilter::PassOnly => batch.iter().filter(|t| t.reward >= 1.0).collect(),
            SampleFilter::KeepAll => batch.iter().collect(),
            SampleFilter::DropTruncated => batch.iter().filter(|t| !t.truncated).collect(),
            SampleFilter::DropZeroAdvGroup => {
                let live = |t: &ScoredTrajectory| {
                    batch
                        .iter()
                        .any(|o| o.group_id == t.group_id && o.reward != t.reward)
                };
                batch.iter().filter(|t| !t.truncated && live(t)).collect()
            }
        }
    }

    fn validated_training<'a>(
        &self,
        batch: &'a [ScoredTrajectory],
        vocab: usize,
        window: usize,
    ) -> Result<Vec<&'a ScoredTrajectory>> {
        validate_trajectory_structure(batch)?;
        if window == 0 {
            return Err(OpdError::InvalidInput(
                "masked writeback window_size must be > 0".to_owned(),
            ));
        }
        if vocab > i32::MAX as usize {
            return Err(OpdError::InvalidInput(format!(
                "masked writeback vocab {vocab} exceeds i32::MAX"
            )));
        }
        let training: Vec<_> = self
            .filter_batch(batch)
            .into_iter()
            .filter(|traj| !over_max_seq(traj) && masked_target_count(traj) > 0)
            .collect();
        validate_training_inputs(&training, vocab, self.needs().behavior_logprobs)?;
        Ok(training)
    }

    pub fn planned_training_count(&self, batch: &[ScoredTrajectory]) -> usize {
        self.filter_batch(batch)
            .into_iter()
            .filter(|traj| !over_max_seq(traj) && masked_target_count(traj) > 0)
            .count()
    }

    pub fn preflight(&self, batch: &[ScoredTrajectory], vocab: usize, window: usize) -> Result<()> {
        self.validated_training(batch, vocab, window).map(drop)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update<O: Optimizer>(
        &self,
        batch: &[ScoredTrajectory],
        student: &Qwen35Model,
        all_params: &[TensorId],
        trainable: &[TensorId],
        opt: &mut O,
        critic: Option<&mut ValueCritic>,
        vocab: usize,
        window: usize,
        store: &mut TensorStore,
    ) -> Result<UpdateReport> {
        let survivors = self.validated_training(batch, vocab, window)?;
        if survivors.is_empty() {
            return Ok(UpdateReport::default());
        }
        if self.ratio == RatioGrain::None {
            return update_ce(
                &survivors, student, all_params, trainable, opt, vocab, window, store,
            );
        }
        self.update_pg(
            &survivors, student, all_params, trainable, opt, critic, vocab, window, store,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn update_pg<O: Optimizer>(
        &self,
        survivors: &[&ScoredTrajectory],
        student: &Qwen35Model,
        all_params: &[TensorId],
        trainable: &[TensorId],
        opt: &mut O,
        mut critic: Option<&mut ValueCritic>,
        vocab: usize,
        window: usize,
        store: &mut TensorStore,
    ) -> Result<UpdateReport> {
        let scalar_advs: Vec<f32> = match self.advantage {
            Advantage::Mean { scope, std_norm } => centered_advantages(survivors, scope, std_norm),
            Advantage::ValueGae { .. } => Vec::new(),
            Advantage::None => vec![1.0; survivors.len()],
        };
        if self.needs_value_critic() && critic.is_none() {
            return Err(OpdError::InvalidInput(
                "ValueGae update requires a ValueCritic (caller must pass Some)".to_owned(),
            ));
        }
        // Critic weights aren't student params; keep them so cleanup_after_backward doesn't free them.
        let all_with_critic: Vec<TensorId> = match &critic {
            Some(c) => all_params
                .iter()
                .copied()
                .chain(c.param_ids().iter().copied())
                .collect(),
            None => Vec::new(),
        };
        let params = if critic.is_some() {
            all_with_critic.as_slice()
        } else {
            all_params
        };

        let batch_tokens: usize = survivors.iter().map(|t| masked_target_count(t)).sum();
        let step_each = matches!(self.agg, Aggregation::PerSeqTokenMean);

        let mut loss_sum = 0.0f32;
        let mut steps = 0usize;
        let mut stats = PgStats::default();
        let mut mse_sum = 0.0f32;
        let mut adv_abs_sum = 0.0f32;
        let mut adv_tokens = 0usize;
        let mut adv_sum = 0.0f64;
        let mut adv_sq = 0.0f64;
        let mut adv_n = 0usize;
        let accumulate = (|| -> std::result::Result<(), OpdError> {
            for (i, traj) in survivors.iter().enumerate() {
                let behavior_logprobs = traj
                    .behavior_logprobs
                    .as_deref()
                    .expect("ratio-weighted batch validated before policy/critic forward");
                let (advantages, returns) = match critic.as_deref_mut() {
                    Some(c) => c.advantages(
                        student,
                        &traj.prompt_ids,
                        &traj.response_ids,
                        &traj.response_mask,
                        traj.reward,
                        store,
                    )?,
                    None => (vec![scalar_advs[i]; behavior_logprobs.len()], Vec::new()),
                };
                if advantages.is_empty() {
                    continue;
                }
                if critic.is_none() && scalar_advs[i] == 0.0 {
                    continue;
                }
                for &a in &advantages {
                    adv_sum += f64::from(a);
                    adv_sq += f64::from(a) * f64::from(a);
                }
                adv_n += advantages.len();

                let norm = token_mean_norm(self.agg, advantages.len(), batch_tokens);
                let mut weights: Vec<f32> = advantages.iter().map(|a| a / norm).collect();

                let form = match self.ratio {
                    RatioGrain::PerToken => self.clip.weight_form(),
                    RatioGrain::PerSequence => {
                        let current_lp = capture_rollout_logprobs(
                            student,
                            &traj.prompt_ids,
                            &traj.response_ids,
                            &traj.response_mask,
                            store,
                        )?;
                        let s = sequence_ratio(&current_lp, behavior_logprobs)?;
                        let mut clipped = false;
                        for w in &mut weights {
                            let (effective, token_clipped) =
                                pg_token_weight(self.clip.weight_form(), *w, s, 0.0);
                            *w = effective;
                            clipped |= token_clipped;
                        }
                        WeightForm::Precomputed { ratio: s, clipped }
                    }
                    RatioGrain::None => unreachable!("CE path handled in update()"),
                };

                let (loss, traj_stats, _) = masked_writeback_step(
                    WritebackLoss::Pg {
                        rollout_logprobs: behavior_logprobs,
                        weight: &weights,
                        form,
                        kl_coef: 0.0,
                    },
                    student,
                    params,
                    trainable,
                    opt,
                    step_each,
                    &traj.prompt_ids,
                    &traj.response_ids,
                    &traj.response_mask,
                    vocab,
                    window,
                    crate::context_parallel::CpContext::from_env(),
                    crate::context_parallel::DpContext::from_env(),
                    store,
                )?;
                stats.merge(traj_stats);
                if let Some(c) = critic.as_deref_mut() {
                    mse_sum += c.update(
                        student,
                        &traj.prompt_ids,
                        &traj.response_ids,
                        &traj.response_mask,
                        &returns,
                        store,
                    )?;
                    adv_abs_sum += advantages.iter().map(|a| a.abs()).sum::<f32>();
                    adv_tokens += advantages.len();
                }
                loss_sum += loss;
                if !loss_sum.is_finite() {
                    return Err(OpdError::InvalidInput(format!(
                        "policy loss became non-finite while accumulating trajectory {i}: {loss_sum}"
                    )));
                }
                steps += 1;
            }
            Ok(())
        })();
        if let Err(e) = accumulate {
            opt.zero_grad(store, trainable);
            return Err(e);
        }

        if !step_each && steps > 0 {
            finite_optimizer_step(loss_sum, trainable, 0.0, opt, store)?;
        }
        let critic_mse = if critic.is_some() && steps > 0 {
            mse_sum / steps as f32
        } else {
            0.0
        };
        if steps > 0 {
            let critic_suffix = if critic.is_some() {
                format!(
                    " mean_critic_mse={critic_mse:.4e} mean_adv_abs={:.4e}",
                    adv_abs_sum / adv_tokens.max(1) as f32,
                )
            } else {
                String::new()
            };
            eprintln!(
                "[update] trained={steps} mean_policy_loss={:.4} kl={:.4e} clip_frac={:.3}{critic_suffix}",
                loss_sum / steps as f32,
                stats.kl_mean(),
                stats.clip_frac(),
            );
        }
        let adv_mean = adv_sum / adv_n.max(1) as f64;
        let adv_std = (adv_sq / adv_n.max(1) as f64 - adv_mean * adv_mean).max(0.0);
        Ok(UpdateReport {
            loss: if steps > 0 {
                loss_sum / steps as f32
            } else {
                0.0
            },
            trained: steps,
            tokens: stats.tokens,
            stats,
            critic_mse,
            adv_mean: adv_mean as f32,
            adv_std: adv_std.sqrt() as f32,
        })
    }
}

fn validate_trajectory_structure(batch: &[ScoredTrajectory]) -> Result<()> {
    for (idx, traj) in batch.iter().enumerate() {
        if !traj.reward.is_finite() {
            return Err(OpdError::InvalidInput(format!(
                "trajectory {idx}: reward must be finite, got {}",
                traj.reward
            )));
        }
        if traj.response_ids.len() != traj.response_mask.len() {
            return Err(OpdError::InvalidInput(format!(
                "trajectory {idx}: response_ids len {} != response_mask len {}",
                traj.response_ids.len(),
                traj.response_mask.len()
            )));
        }
        if let Some((position, mask)) = traj
            .response_mask
            .iter()
            .copied()
            .enumerate()
            .find(|(_, mask)| *mask > 1)
        {
            return Err(OpdError::InvalidInput(format!(
                "trajectory {idx}: response_mask[{position}]={mask}, expected 0 or 1"
            )));
        }
    }
    Ok(())
}

fn masked_target_count(traj: &ScoredTrajectory) -> usize {
    traj.response_mask.iter().filter(|&&mask| mask == 1).count()
}

fn validate_training_inputs(
    training: &[&ScoredTrajectory],
    vocab: usize,
    needs_behavior: bool,
) -> Result<()> {
    for (idx, traj) in training.iter().enumerate() {
        if traj.prompt_ids.is_empty() {
            return Err(OpdError::InvalidInput(format!(
                "training trajectory {idx}: masked writeback requires a non-empty prompt"
            )));
        }
        let seq_len = traj
            .prompt_ids
            .len()
            .checked_add(traj.response_ids.len())
            .ok_or_else(|| {
                OpdError::InvalidInput(format!(
                    "training trajectory {idx}: prompt + response length overflow"
                ))
            })?;
        if seq_len > i32::MAX as usize {
            return Err(OpdError::InvalidInput(format!(
                "training trajectory {idx}: length {seq_len} exceeds i32::MAX position indices"
            )));
        }
        for (position, (&token, &mask)) in traj
            .response_ids
            .iter()
            .zip(&traj.response_mask)
            .enumerate()
        {
            if mask == 1 && token as usize >= vocab {
                return Err(OpdError::InvalidInput(format!(
                    "training trajectory {idx}: masked target token {token} at response index {position} exceeds vocab={vocab}"
                )));
            }
        }
        if needs_behavior {
            let behavior = traj.behavior_logprobs.as_deref().ok_or_else(|| {
                OpdError::InvalidInput(format!(
                    "training trajectory {idx}: ratio-weighted update requires generation-time behavior_logprobs sidecar"
                ))
            })?;
            let targets = masked_target_count(traj);
            if behavior.len() != targets {
                return Err(OpdError::InvalidInput(format!(
                    "training trajectory {idx}: behavior_logprobs len {} != masked token count {targets}",
                    behavior.len()
                )));
            }
            if let Some(position) = behavior.iter().position(|value| !value.is_finite()) {
                return Err(OpdError::InvalidInput(format!(
                    "training trajectory {idx}: behavior_logprobs[{position}] is not finite ({})",
                    behavior[position]
                )));
            }
        }
    }
    Ok(())
}

fn centered_advantages(survivors: &[&ScoredTrajectory], scope: Scope, std_norm: bool) -> Vec<f32> {
    let group_of = |t: &ScoredTrajectory| match scope {
        Scope::Batch => 0usize,
        Scope::Group => t.group_id,
    };
    let mut stats: Vec<(usize, f32, f32, usize)> = Vec::new();
    for t in survivors {
        let key = group_of(t);
        match stats.iter_mut().find(|(k, ..)| *k == key) {
            Some((_, sum, sumsq, n)) => {
                *sum += t.reward;
                *sumsq += t.reward * t.reward;
                *n += 1;
            }
            None => stats.push((key, t.reward, t.reward * t.reward, 1)),
        }
    }
    survivors
        .iter()
        .map(|t| {
            let &(_, sum, sumsq, n) = stats
                .iter()
                .find(|(k, ..)| *k == group_of(t))
                .expect("group key inserted above");
            let mean = sum / n as f32;
            let adv = t.reward - mean;
            if std_norm {
                let var = (sumsq / n as f32 - mean * mean).max(0.0);
                adv / (var.sqrt() + 1e-6)
            } else {
                adv
            }
        })
        .collect()
}

fn token_mean_norm(agg: Aggregation, seq_tokens: usize, batch_tokens: usize) -> f32 {
    match agg {
        Aggregation::PerSeqTokenMean => seq_tokens,
        Aggregation::GlobalTokenMean { norm_const } => norm_const.unwrap_or(batch_tokens),
    }
    .max(1) as f32
}

fn sequence_ratio(current_lp: &[f32], behavior_lp: &[f32]) -> Result<f32> {
    if current_lp.len() != behavior_lp.len() {
        return Err(OpdError::InvalidInput(format!(
            "seq-ratio capture len {} != behavior logprobs len {}",
            current_lp.len(),
            behavior_lp.len()
        )));
    }
    if current_lp.is_empty() {
        return Err(OpdError::InvalidInput(
            "seq-ratio capture must be non-empty".to_owned(),
        ));
    }
    let mean_delta = current_lp
        .iter()
        .zip(behavior_lp)
        .map(|(&current, &behavior)| f64::from(current) - f64::from(behavior))
        .sum::<f64>()
        / current_lp.len() as f64;
    let ratio = mean_delta.exp();
    if !ratio.is_finite() || ratio <= 0.0 || ratio > f64::from(f32::MAX) {
        return Err(OpdError::InvalidInput(
            "sequence importance ratio must be finite, positive, and fit f32".to_owned(),
        ));
    }
    Ok(ratio as f32)
}

fn over_max_seq(traj: &ScoredTrajectory) -> bool {
    let cap = crate::runtime_flags::max_update_seq();
    cap != 0 && traj.prompt_ids.len() + traj.response_ids.len() > cap
}

#[allow(clippy::too_many_arguments)]
fn update_ce<O: Optimizer>(
    survivors: &[&ScoredTrajectory],
    student: &Qwen35Model,
    all_params: &[TensorId],
    trainable: &[TensorId],
    opt: &mut O,
    vocab: usize,
    window: usize,
    store: &mut TensorStore,
) -> Result<UpdateReport> {
    let mut loss_sum = 0.0f32;
    let mut tokens = 0usize;
    let mut trained = 0usize;
    for traj in survivors {
        loss_sum += masked_writeback_ce_step_dispatch(
            student,
            all_params,
            trainable,
            opt,
            &traj.prompt_ids,
            &traj.response_ids,
            &traj.response_mask,
            vocab,
            window,
            crate::context_parallel::CpContext::from_env(),
            crate::context_parallel::DpContext::from_env(),
            store,
        )?;
        tokens += traj.response_mask.iter().filter(|&&m| m == 1).count();
        trained += 1;
    }
    Ok(UpdateReport {
        loss: if trained > 0 {
            loss_sum / trained as f32
        } else {
            0.0
        },
        trained,
        tokens,
        ..UpdateReport::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn traj(reward: f32, group_id: usize) -> ScoredTrajectory {
        ScoredTrajectory {
            prompt_ids: vec![1],
            response_ids: vec![2],
            response_mask: vec![1],
            reward,
            behavior_logprobs: None,
            group_id,
            truncated: false,
        }
    }

    #[test]
    fn preset_needs_derive_mechanically() {
        let cases = [
            (UpdatePreset::rejection_ce(), false, false),
            (UpdatePreset::sao_dis(0.8, 3.0), true, true),
            (UpdatePreset::sao_value(0.8, 3.0, 1.0, 0.95), true, true),
            (UpdatePreset::grpo(), true, true),
            (UpdatePreset::dapo(), true, true),
            (UpdatePreset::dr_grpo(4096), true, true),
            (UpdatePreset::gspo(), true, true),
            (UpdatePreset::cispo(), true, true),
        ];
        for (preset, keep_failing, behavior_logprobs) in cases {
            assert_eq!(
                preset.needs(),
                RolloutNeeds {
                    keep_failing,
                    behavior_logprobs
                },
                "{preset:?}"
            );
        }
        assert!(UpdatePreset::sao_value(0.8, 3.0, 1.0, 0.95).needs_value_critic());
        assert!(!UpdatePreset::grpo().needs_value_critic());
    }

    #[test]
    fn group_vs_batch_advantage_scope() {
        let batch = [traj(1.0, 0), traj(0.0, 0), traj(1.0, 1), traj(1.0, 1)];
        let refs: Vec<&ScoredTrajectory> = batch.iter().collect();
        assert_eq!(
            centered_advantages(&refs, Scope::Batch, false),
            vec![0.25, -0.75, 0.25, 0.25]
        );
        assert_eq!(
            centered_advantages(&refs, Scope::Group, false),
            vec![0.5, -0.5, 0.0, 0.0]
        );
        let gn = centered_advantages(&refs, Scope::Group, true);
        assert!((gn[0] - 0.5 / (0.5 + 1e-6)).abs() < 1e-6, "{gn:?}");
        assert_eq!(gn[2], 0.0);
    }

    #[test]
    fn dr_grpo_norm_is_fixed_constant_independent_of_length() {
        let dr = Aggregation::GlobalTokenMean {
            norm_const: Some(4096),
        };
        assert_eq!(token_mean_norm(dr, 10, 999), 4096.0);
        assert_eq!(token_mean_norm(dr, 4000, 999), 4096.0);
        let grpo = Aggregation::GlobalTokenMean { norm_const: None };
        assert_eq!(token_mean_norm(grpo, 10, 512), 512.0);
        assert_eq!(token_mean_norm(Aggregation::PerSeqTokenMean, 10, 512), 10.0);
        assert_eq!(token_mean_norm(Aggregation::PerSeqTokenMean, 0, 0), 1.0);
    }

    #[test]
    fn weight_builder_distinguishes_hard_cispo_and_ppo() {
        let hard = WeightForm::HardGate { lo: 0.2, hi: 0.5 };
        let cispo = WeightForm::DetachedSoftClamp { lo: 0.2, hi: 0.5 };
        let ppo = WeightForm::PpoClip { lo: 0.2, hi: 0.5 };
        assert_eq!(pg_token_weight(hard, 2.0, 1.0, 0.0), (2.0, false));
        assert_eq!(pg_token_weight(hard, 2.0, 1.6, 0.0), (0.0, true));

        assert_eq!(pg_token_weight(cispo, 2.0, 1.2, 0.0), (2.4, false));
        assert_eq!(pg_token_weight(cispo, 2.0, 3.0, 0.0), (3.0, true));
        assert_eq!(pg_token_weight(cispo, -2.0, 0.5, 0.0), (-1.6, true));

        let cases = [
            (2.0, 0.5, 1.0, false),
            (2.0, 1.0, 2.0, false),
            (2.0, 1.6, 0.0, true),
            (-2.0, 0.5, 0.0, true),
            (-2.0, 1.0, -2.0, false),
            (-2.0, 1.6, -3.2, false),
        ];
        for (base, ratio, expected, clipped) in cases {
            assert_eq!(
                pg_token_weight(ppo, base, ratio, 0.0),
                (expected, clipped),
                "base={base} ratio={ratio}"
            );
        }

        let (w, clipped) = pg_token_weight(
            WeightForm::Precomputed {
                ratio: 2.0,
                clipped: false,
            },
            1.0,
            2.0,
            0.1,
        );
        assert!((w - 0.9).abs() < 1e-6);
        assert!(!clipped);

        let s = sequence_ratio(&[0.0, 0.0], &[-0.2, 0.2]).unwrap();
        assert!((s - 1.0).abs() < 1e-6, "mean Δ=0 → s=1, got {s}");
        let s = sequence_ratio(&[0.5, 0.5], &[0.0, 0.0]).unwrap();
        assert!((s - 0.5f32.exp()).abs() < 1e-6, "got {s}");
        assert_eq!(pg_token_weight(ppo, 1.0, 1.6, 0.0), (0.0, true));
        assert_eq!(pg_token_weight(ppo, -1.0, 1.6, 0.0), (-1.6, false));
    }

    #[test]
    fn sequence_ratio_uses_f64_and_fails_closed() {
        let ratio = sequence_ratio(&[1.0e20, -1.0e20], &[0.0, 0.0]).unwrap();
        assert_eq!(ratio, 1.0);

        let overflow = sequence_ratio(&[1000.0], &[0.0]).unwrap_err();
        assert!(overflow.to_string().contains("sequence importance ratio"));
        let empty = sequence_ratio(&[], &[]).unwrap_err();
        assert!(empty.to_string().contains("must be non-empty"));
    }

    #[test]
    fn behavior_sidecars_fail_closed() {
        let mut batch = vec![traj(1.0, 0)];
        let err = UpdatePreset::grpo().preflight(&batch, 100, 1).unwrap_err();
        assert!(
            err.to_string()
                .contains("requires generation-time behavior_logprobs")
        );

        batch[0].behavior_logprobs = Some(vec![f32::NAN]);
        let err = UpdatePreset::grpo().preflight(&batch, 100, 1).unwrap_err();
        assert!(err.to_string().contains("is not finite"));

        batch[0].behavior_logprobs = Some(vec![]);
        let err = UpdatePreset::grpo().preflight(&batch, 100, 1).unwrap_err();
        assert!(err.to_string().contains("masked token count 1"));
    }

    #[test]
    fn trajectory_structure_accepts_mixed_mask_and_rejects_mismatch() {
        let mut batch = vec![traj(1.0, 0)];
        batch[0].response_ids = vec![2, 3];
        batch[0].response_mask = vec![1, 0];
        assert!(validate_trajectory_structure(&batch).is_ok());

        batch[0].response_mask.pop();
        let err = validate_trajectory_structure(&batch).unwrap_err();
        assert!(
            err.to_string()
                .contains("response_ids len 2 != response_mask len 1")
        );

        batch[0].response_mask = vec![1, 2];
        let err = validate_trajectory_structure(&batch).unwrap_err();
        assert!(err.to_string().contains("expected 0 or 1"));
    }

    #[test]
    fn dapo_filtered_malformed_sidecars_do_not_block() {
        let mut batch = vec![traj(1.0, 0), traj(0.0, 0), traj(1.0, 1), traj(1.0, 1)];
        batch[0].truncated = true;
        batch[1].behavior_logprobs = Some(vec![-0.5]);
        batch[2].behavior_logprobs = Some(vec![f32::NAN]);
        UpdatePreset::dapo().preflight(&batch, 100, 1).unwrap();
    }

    #[test]
    fn preflight_rejects_deterministic_training_inputs() {
        for reward in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut batch = vec![traj(reward, 0)];
            batch[0].behavior_logprobs = Some(vec![-0.5]);
            assert!(
                UpdatePreset::grpo()
                    .preflight(&batch, 100, 1)
                    .unwrap_err()
                    .to_string()
                    .contains("reward must be finite")
            );
        }

        let mut batch = vec![traj(1.0, 0)];
        batch[0].behavior_logprobs = Some(vec![-0.5]);
        batch[0].prompt_ids.clear();
        assert!(
            UpdatePreset::grpo()
                .preflight(&batch, 100, 1)
                .unwrap_err()
                .to_string()
                .contains("non-empty prompt")
        );

        let mut batch = vec![traj(1.0, 0)];
        batch[0].behavior_logprobs = Some(vec![-0.5]);
        batch[0].response_ids[0] = 100;
        assert!(
            UpdatePreset::grpo()
                .preflight(&batch, 100, 1)
                .unwrap_err()
                .to_string()
                .contains("exceeds vocab=100")
        );
        assert!(
            UpdatePreset::grpo()
                .preflight(&batch, 101, 0)
                .unwrap_err()
                .to_string()
                .contains("window_size must be > 0")
        );
    }

    #[test]
    fn preflight_validates_second_training_trajectory_before_execution() {
        let mut batch = vec![traj(1.0, 0), traj(0.0, 0)];
        batch[0].behavior_logprobs = Some(vec![-0.5]);
        batch[1].behavior_logprobs = Some(vec![f32::NAN]);
        let err = UpdatePreset::grpo().preflight(&batch, 100, 1).unwrap_err();
        assert!(err.to_string().contains("training trajectory 1"));
    }

    #[test]
    fn skipped_trajectories_do_not_require_valid_sidecars() {
        let mut batch = vec![traj(1.0, 0), traj(0.0, 0), traj(1.0, 1), traj(1.0, 1)];
        batch[0].truncated = true;
        batch[1].behavior_logprobs = Some(vec![-0.5]);
        batch[2].behavior_logprobs = Some(vec![f32::NAN]);
        batch[3].response_mask[0] = 0;
        UpdatePreset::dapo().preflight(&batch, 100, 1).unwrap();

        let mut zero_target = traj(1.0, 0);
        zero_target.response_mask[0] = 0;
        zero_target.behavior_logprobs = Some(vec![f32::NAN]);
        UpdatePreset::grpo()
            .preflight(&[zero_target], 100, 1)
            .unwrap();
    }

    #[test]
    fn dapo_filter_drops_zero_variance_groups_and_truncated() {
        let mut batch = vec![traj(1.0, 0), traj(0.0, 0), traj(1.0, 1), traj(1.0, 1)];
        batch[0].truncated = true;
        let kept: Vec<f32> = UpdatePreset::dapo()
            .filter_batch(&batch)
            .iter()
            .map(|t| t.reward)
            .collect();
        assert_eq!(kept, vec![0.0]);
    }
}
