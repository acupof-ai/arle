use std::collections::{HashMap, VecDeque};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use infer_plan::{
    DiffusionBlockModel, DiffusionGenerationConfig, FinishReason, ForwardPlan, MultimodalImage,
    SamplingParams, SlotToken, StepOutput, generate_diffusion_with_cancel,
};

use crate::{BackendExecutor, KvPool, PollResult};

#[derive(Debug, Clone)]
struct BufferedToken {
    token: u32,
    finish: Option<FinishReason>,
}

#[derive(Debug, Clone, Default)]
struct SlotState {
    epoch: u64,
    prompt: Vec<u32>,
    generated: VecDeque<BufferedToken>,
}

/// Adapter that lets a block-diffusion model run behind the normal
/// autoregressive [`BackendExecutor`] seam.
///
/// The shared `Engine` still sees one token per step. Internally, the adapter
/// accumulates chunked prompt-prefill rows, runs
/// [`generate_diffusion_with_cancel`] once on the final prompt chunk, returns
/// the first generated token from that prefill
/// step, then serves the remaining generated tokens from a per-slot buffer on
/// subsequent decode rows. The wrapped model only needs to implement
/// [`DiffusionBlockModel`].
pub struct BufferedDiffusionExecutor<M> {
    model: M,
    base_config: DiffusionGenerationConfig,
    slots: HashMap<usize, SlotState>,
    cancel: Option<Arc<AtomicBool>>,
}

impl<M> BufferedDiffusionExecutor<M> {
    #[must_use]
    pub fn new_with_cancel(
        model: M,
        base_config: DiffusionGenerationConfig,
        cancel: Arc<AtomicBool>,
    ) -> Self {
        Self {
            model,
            base_config,
            slots: HashMap::new(),
            cancel: Some(cancel),
        }
    }

    #[must_use]
    pub fn into_inner(self) -> M {
        self.model
    }

    fn config_for_row(
        base: &DiffusionGenerationConfig,
        params: &infer_plan::SamplingParams,
    ) -> DiffusionGenerationConfig {
        let mut config = base.clone();
        if let Some(max_new_tokens) = params.max_new_tokens {
            config.max_new_tokens = max_new_tokens;
        }
        if let Some(seed) = params.seed {
            config.seed = seed;
        }
        if params.ignore_eos {
            config.stop_token_ids.clear();
        } else if !params.stop_token_ids.is_empty() {
            config.stop_token_ids = params.stop_token_ids.clone();
        }
        config
    }

    fn slot_mut(&mut self, slot: usize, epoch: u64, reset_prompt: bool) -> &mut SlotState {
        let state = self.slots.entry(slot).or_insert_with(|| SlotState {
            epoch,
            ..SlotState::default()
        });
        if state.epoch != epoch || reset_prompt {
            state.epoch = epoch;
            state.prompt.clear();
            state.generated.clear();
        }
        state
    }

    fn next_buffered_token(&mut self, slot: usize, epoch: u64) -> anyhow::Result<StepOutput> {
        let state = self.slot_mut(slot, epoch, false);
        let Some(token) = state.generated.pop_front() else {
            anyhow::bail!(
                "diffusion buffered executor received decode for slot {slot} with no generated token buffered"
            );
        };
        if state.generated.is_empty() {
            state.prompt.clear();
        }
        Ok(StepOutput {
            tokens: vec![SlotToken {
                slot,
                token: token.token,
                logprob: None,
                top_logprobs: Vec::new(),
                finish: token.finish,
            }],
        })
    }

    fn submit_prefill(
        &mut self,
        row: &infer_plan::PrefillRow,
        kv: &mut dyn KvPool,
    ) -> anyhow::Result<StepOutput>
    where
        M: DiffusionBlockModel,
    {
        let epoch = kv.slot_epoch(row.slot);
        let reset_prompt = row.start_pos == 0;
        let prompt = {
            let state = self.slot_mut(row.slot, epoch, reset_prompt);
            if row.start_pos != state.prompt.len() {
                anyhow::bail!(
                    "diffusion prefill chunk mismatch for slot {}: row.start_pos={}, buffered_prompt_len={}",
                    row.slot,
                    row.start_pos,
                    state.prompt.len()
                );
            }
            state.prompt.extend_from_slice(&row.tokens);
            if state.prompt.len() < row.total_tokens {
                return Ok(StepOutput { tokens: Vec::new() });
            }
            state.prompt.clone()
        };

        let config = Self::config_for_row(&self.base_config, &row.params);
        let output = generate_diffusion_with_cancel(
            &mut self.model,
            &prompt,
            &config,
            self.cancel.as_deref(),
        )
        .map_err(|err| anyhow::anyhow!("{err}"))?;
        if std::env::var_os("ARLE_DIFFUSION_TRACE").is_some() {
            eprintln!(
                "diffusion generate complete: prompt_tokens={} generated_tokens={} blocks={} denoise_steps={} forced_commits={} adaptive_commits={} finish={:?}",
                prompt.len(),
                output.generated_tokens.len(),
                output.stats.blocks,
                output.stats.denoise_steps,
                output.stats.forced_commits,
                output.stats.adaptive_commits,
                output.finish
            );
        }
        let tokens = output.generated_tokens;
        anyhow::ensure!(
            !tokens.is_empty(),
            "diffusion generation produced no tokens for non-empty request"
        );

        let finish = output.finish;
        let mut buffered: VecDeque<BufferedToken> = tokens
            .into_iter()
            .map(|token| BufferedToken {
                token,
                finish: None,
            })
            .collect();
        if !matches!(finish, FinishReason::Length)
            && let Some(last) = buffered.back_mut()
        {
            last.finish = Some(finish);
        }

        let state = self.slot_mut(row.slot, epoch, false);
        state.generated = buffered;
        self.next_buffered_token(row.slot, epoch)
    }
}

impl<M> BackendExecutor for BufferedDiffusionExecutor<M>
where
    M: DiffusionBlockModel,
{
    type Inflight = StepOutput;

    fn submit(
        &mut self,
        plan: &ForwardPlan,
        kv: &mut dyn KvPool,
    ) -> anyhow::Result<Self::Inflight> {
        if plan.is_idle() {
            return Ok(StepOutput { tokens: Vec::new() });
        }
        anyhow::ensure!(
            !self
                .cancel
                .as_ref()
                .is_some_and(|flag| flag.load(Ordering::Acquire)),
            "diffusion generation cancelled"
        );
        let row_count = plan.prefill_rows.len() + plan.decode_rows.len();
        anyhow::ensure!(
            row_count == 1,
            "diffusion buffered executor supports exactly one prefill or decode row, got {row_count}"
        );
        if let Some(row) = plan.prefill_rows.first() {
            return self.submit_prefill(row, kv);
        }
        if let Some(row) = plan.decode_rows.first() {
            return self.next_buffered_token(row.slot, kv.slot_epoch(row.slot));
        }
        anyhow::bail!("diffusion buffered executor received a non-idle plan with no rows")
    }

    fn poll(&mut self, inflight: Self::Inflight) -> anyhow::Result<PollResult<Self::Inflight>> {
        Ok(PollResult::Ready(inflight))
    }

    fn model_stop_token_ids(&self) -> Vec<u32> {
        self.base_config.stop_token_ids.clone()
    }

    fn step_limits(&self) -> crate::StepLimits {
        crate::StepLimits {
            max_rows_per_step: 1,
            max_live_requests: 1,
            ..crate::StepLimits::default()
        }
    }

    fn multimodal(&mut self) -> Option<&mut dyn crate::MultimodalGenerate> {
        Some(self)
    }
}

impl<M> crate::MultimodalGenerate for BufferedDiffusionExecutor<M>
where
    M: DiffusionBlockModel,
{
    fn generate_multimodal(
        &mut self,
        prompt_tokens: &[u32],
        images: &[MultimodalImage],
        max_tokens: usize,
        sampling: &SamplingParams,
    ) -> anyhow::Result<Option<infer_plan::DiffusionGenerateOutput>> {
        let mut row_sampling = sampling.clone();
        row_sampling.max_new_tokens = Some(max_tokens);
        let config = Self::config_for_row(&self.base_config, &row_sampling);
        self.model
            .generate_multimodal_with_cancel(prompt_tokens, images, &config, self.cancel.as_deref())
            .map_err(|err| anyhow::anyhow!("{err}"))
    }

    fn multimodal_kind(&self) -> Option<infer_plan::MultimodalKind> {
        self.model.multimodal_kind()
    }
}
