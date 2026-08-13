//! Skip-Obs GAE and the frozen-attention linear value critic for PG writeback.

use std::collections::HashSet;

use autograd::{
    AutogradError, Tape, TensorId, TensorStore,
    ops::{add, embedding, matmul_bt, mean, mul, reshape},
    optim::AdamW,
};

use crate::{grad_clip::finite_optimizer_step, qwen35::Qwen35Model};

use super::{
    OpdError, Result, loss::build_masked_loss_targets, map_qwen35_forward_error,
    validation::validate_loss_value, writeback::GenSegment,
};

/// `values` holds V(s_t) at the masked (LLM-generated) positions ONLY, in
/// trajectory order — so the recursion's "next value" is the next LLM token's,
/// and environment/tool observation tokens are skipped for free.
/// `terminal_reward` lands on the final generated token (agentic reward is
/// trajectory-terminal). Returns `(advantages, returns)`,
/// `returns = advantages + values` (the MSE target for the critic).
/// γ=discount, λ=GAE trace.
pub fn skip_obs_gae(
    values: &[f32],
    terminal_reward: f32,
    gamma: f32,
    lam: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = values.len();
    let mut advantages = vec![0.0f32; n];
    let mut gae = 0.0f32;
    for t in (0..n).rev() {
        let reward = if t == n - 1 { terminal_reward } else { 0.0 };
        let next_value = if t == n - 1 { 0.0 } else { values[t + 1] };
        let delta = reward + gamma * next_value - values[t];
        gae = delta + gamma * lam * gae;
        advantages[t] = gae;
    }
    let returns = advantages.iter().zip(values).map(|(a, v)| a + v).collect();
    (advantages, returns)
}

/// **Frozen-Attention**: both the GAE value read and the MSE update project a
/// DETACHED copy of the masked hidden rows (host round-trip), so gradient
/// reaches `weight` only — never the base. Zero-init → V₀(s)=0 → round-0 GAE =
/// discounted reward-to-go (MC), a stable cold start (no separate
/// value-pretraining phase).
///
/// ponytail: one 27B forward per trajectory (`masked_hidden`) feeds BOTH the GAE
/// values and the MSE update; K=1 update/policy-step. Upgrade to K=2 / a fused
/// value-in-writeback forward only if the critic's fit lags or the writeback
/// wall dominates rollout — neither observed yet.
pub struct ValueCritic {
    weight: TensorId,
    params: [TensorId; 1],
    opt: AdamW,
    gamma: f32,
    lam: f32,
}

impl ValueCritic {
    pub fn new(
        hidden_dim: usize,
        lr: f32,
        gamma: f32,
        lam: f32,
        store: &mut TensorStore,
    ) -> Result<Self> {
        let weight = store
            .from_slice(&vec![0.0f32; hidden_dim], &[1, hidden_dim])
            .map_err(OpdError::from)?;
        store
            .get_mut(weight)
            .ok_or(AutogradError::InvalidTensorId(weight))?
            .requires_grad = true;
        Ok(Self {
            weight,
            params: [weight],
            opt: AdamW::new(lr, (0.9, 0.999), 1.0e-8, 0.0),
            gamma,
            lam,
        })
    }

    /// The critic's parameter ids — the policy writeback must keep these in its
    /// cleanup set (`cleanup_after_backward` frees everything not in
    /// `all_model_params`, and the critic weight is deliberately NOT a student
    /// param), else the next `update` hits a freed weight.
    pub fn param_ids(&self) -> &[TensorId] {
        &self.params
    }

    /// The masked (LLM-generated) hidden rows as a DETACHED host copy, in
    /// `build_masked_loss_targets` order. One checkpointed forward (tape-on but
    /// never backwarded, like `capture_rollout_logprobs`).
    fn masked_hidden(
        &self,
        student: &Qwen35Model,
        prompt_ids: &[u32],
        response_ids: &[u32],
        response_mask: &[u8],
        store: &mut TensorStore,
    ) -> Result<(Vec<f32>, usize)> {
        let prompt_len = prompt_ids.len();
        let full: Vec<u32> = prompt_ids
            .iter()
            .copied()
            .chain(response_ids.iter().copied())
            .collect();
        let seq_len = full.len();
        let loss_targets = build_masked_loss_targets(&full, prompt_len, response_mask);
        if loss_targets.is_empty() {
            return Ok((Vec::new(), 0));
        }
        let positions: Vec<u32> = (0..seq_len as u32).collect();
        // Frozen-prompt-KV: gen-segment forward, gather masked rows rebased by
        // -gen_start. gen_start=0 keeps the full path byte-identical. Returned
        // (rows_flat, n) semantics are unchanged.
        let frozen = crate::runtime_flags::writeback_frozen_prompt_kv() && prompt_len > 1;
        let gen_start = if frozen { prompt_len - 1 } else { 0 };
        let hidden_rows = seq_len - gen_start;
        let keep_ids: HashSet<TensorId> = store.live_ids().into_iter().collect();
        let mut tape = Tape::new();
        tape.set_enabled(true);
        let hidden = if frozen {
            let seg = GenSegment::split(&full, prompt_len);
            student
                .forward_hidden_states_gen_segment(
                    store,
                    &mut tape,
                    &seg.prompt_prefix,
                    &seg.gen_ids,
                    &seg.prompt_positions,
                    &seg.gen_positions,
                    crate::context_parallel::CpContext::single(),
                )
                .map_err(|err| {
                    map_qwen35_forward_error("value-critic frozen-prompt-KV student hidden", err)
                })?
        } else {
            student
                .forward_hidden_states(
                    store,
                    &mut tape,
                    &full,
                    &positions,
                    crate::context_parallel::CpContext::single(),
                )
                .map_err(|err| map_qwen35_forward_error("value-critic student hidden", err))?
        };
        let hidden_dim = *store
            .get(hidden)
            .ok_or(AutogradError::InvalidTensorId(hidden))?
            .shape
            .last()
            .ok_or_else(|| OpdError::InvalidInput("value-critic: empty hidden shape".to_owned()))?;
        let hidden_2d = reshape(hidden, &[hidden_rows, hidden_dim], store, &mut tape)
            .map_err(OpdError::from)?;
        let rows: Vec<usize> = loss_targets.iter().map(|&(p, _)| p - gen_start).collect();
        let n = rows.len();
        let rows_hidden_3d =
            embedding(hidden_2d, &rows, store, &mut tape).map_err(OpdError::from)?;
        let rows_hidden =
            reshape(rows_hidden_3d, &[n, hidden_dim], store, &mut tape).map_err(OpdError::from)?;
        let rows_flat = store.to_host(rows_hidden).map_err(OpdError::from)?;
        store.retain_ids(&keep_ids);
        Ok((rows_flat, n))
    }

    /// Skip-Obs GAE advantages + MSE-target returns for one trajectory, from the
    /// current (detached) critic values. `advantages`/`returns` are empty when
    /// the trajectory has no LLM tokens (caller skips it).
    pub fn advantages(
        &self,
        student: &Qwen35Model,
        prompt_ids: &[u32],
        response_ids: &[u32],
        response_mask: &[u8],
        terminal_reward: f32,
        store: &mut TensorStore,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let (rows_flat, n) =
            self.masked_hidden(student, prompt_ids, response_ids, response_mask, store)?;
        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let hidden_dim = rows_flat.len() / n;
        let w = store.to_host(self.weight).map_err(OpdError::from)?;
        let values: Vec<f32> = (0..n)
            .map(|i| {
                rows_flat[i * hidden_dim..(i + 1) * hidden_dim]
                    .iter()
                    .zip(&w)
                    .map(|(h, wj)| h * wj)
                    .sum()
            })
            .collect();
        Ok(skip_obs_gae(&values, terminal_reward, self.gamma, self.lam))
    }

    /// Frozen-attention: the masked hidden rows are a detached constant,
    /// so backward accumulates grad on `weight` only.
    pub fn update(
        &mut self,
        student: &Qwen35Model,
        prompt_ids: &[u32],
        response_ids: &[u32],
        response_mask: &[u8],
        returns: &[f32],
        store: &mut TensorStore,
    ) -> Result<f32> {
        let (rows_flat, n) =
            self.masked_hidden(student, prompt_ids, response_ids, response_mask, store)?;
        if n == 0 {
            return Ok(0.0);
        }
        if returns.len() != n {
            return Err(OpdError::InvalidInput(format!(
                "value-critic returns len {} != masked targets {n}",
                returns.len()
            )));
        }
        let hidden_dim = rows_flat.len() / n;
        // Free the MSE graph after the step (rows is [n, hidden] — tens of MB);
        // `weight` + optimizer moments live in `keep_ids` and survive.
        let keep_ids: HashSet<TensorId> = store.live_ids().into_iter().collect();
        let mut tape = Tape::new();
        tape.set_enabled(true);
        // Detached hidden rows (rg=false) → grad flows to `weight` only.
        let rows = store
            .from_slice(&rows_flat, &[n, hidden_dim])
            .map_err(OpdError::from)?;
        let value = matmul_bt(rows, self.weight, store, &mut tape).map_err(OpdError::from)?; // [n,1]
        let value_1d = reshape(value, &[n], store, &mut tape).map_err(OpdError::from)?;
        let neg_returns: Vec<f32> = returns.iter().map(|r| -r).collect();
        let neg_returns_id = store
            .from_slice(&neg_returns, &[n])
            .map_err(OpdError::from)?;
        let diff = add(value_1d, neg_returns_id, store, &mut tape).map_err(OpdError::from)?;
        let sq = mul(diff, diff, store, &mut tape).map_err(OpdError::from)?;
        let mse = mean(sq, store, &mut tape).map_err(OpdError::from)?;
        let mse_value = store.to_host(mse).map_err(OpdError::from)?[0];
        validate_loss_value(mse_value)?;
        tape.backward(mse, store).map_err(OpdError::from)?;
        finite_optimizer_step(mse_value, &self.params, 0.0, &mut self.opt, store)?;
        store.retain_ids(&keep_ids);
        Ok(mse_value)
    }
}
