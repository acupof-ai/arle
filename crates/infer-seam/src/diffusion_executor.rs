use std::collections::{HashMap, VecDeque};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use infer_plan::{
    DiffusionBlockModel, DiffusionGenerationConfig, FinishReason, ForwardPlan, MultimodalImage,
    SamplingParams, SlotToken, StepOutput, generate_diffusion_with_cancel,
};

use crate::{BackendExecutor, KvPool, PollResult, PrefixBlock};

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
    pub fn new(model: M, base_config: DiffusionGenerationConfig) -> Self {
        Self {
            model,
            base_config,
            slots: HashMap::new(),
            cancel: None,
        }
    }

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

    fn max_rows_per_step(&self) -> usize {
        1
    }

    fn multimodal_kind(&self) -> Option<infer_plan::MultimodalKind> {
        self.model.multimodal_kind()
    }

    fn max_live_requests(&self) -> usize {
        1
    }

    fn reusable_prefix_blocks(&self, _blocks: &[PrefixBlock]) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use infer_plan::{DiffusionCanvasPrediction, DiffusionModelError, PrefillRow, SamplingParams};

    use super::*;
    use crate::{HostPagedKvPool, KvQuery};

    #[derive(Default)]
    struct FakeDiffusionModel {
        prompts: Vec<Vec<u32>>,
        commits: Vec<Vec<u32>>,
        predictions: Vec<DiffusionCanvasPrediction>,
        calls: usize,
    }

    impl DiffusionBlockModel for FakeDiffusionModel {
        fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<(), DiffusionModelError> {
            self.prompts.push(prompt_tokens.to_vec());
            Ok(())
        }

        fn predict_canvas(
            &mut self,
            _canvas: &[u32],
            _valid_len: usize,
            _step: usize,
            _temperature: f32,
        ) -> Result<DiffusionCanvasPrediction, DiffusionModelError> {
            let idx = self.calls.min(self.predictions.len().saturating_sub(1));
            self.calls += 1;
            self.predictions
                .get(idx)
                .cloned()
                .ok_or_else(|| DiffusionModelError::new("no prediction"))
        }

        fn commit(&mut self, tokens: &[u32]) -> Result<(), DiffusionModelError> {
            self.commits.push(tokens.to_vec());
            Ok(())
        }
    }

    fn prediction(tokens: &[u32], canvas_len: usize) -> DiffusionCanvasPrediction {
        let mut sampled_tokens = vec![0; canvas_len];
        let mut argmax_tokens = vec![0; canvas_len];
        let entropies = vec![0.0; canvas_len];
        for (idx, &token) in tokens.iter().enumerate() {
            sampled_tokens[idx] = token;
            argmax_tokens[idx] = token;
        }
        DiffusionCanvasPrediction {
            sampled_tokens,
            argmax_tokens,
            entropies,
        }
    }

    fn config(max_new_tokens: usize) -> DiffusionGenerationConfig {
        DiffusionGenerationConfig {
            canvas_length: 4,
            max_denoising_steps: 1,
            max_new_tokens,
            vocab_size: 128,
            stop_token_ids: vec![99],
            pad_token_id: 0,
            entropy_bound: 0.1,
            confidence_threshold: 0.01,
            t_min: 0.4,
            t_max: 0.8,
            stability_threshold: 1,
            seed: 0,
        }
    }

    fn prefill_row(
        slot: usize,
        tokens: &[u32],
        start_pos: usize,
        total_tokens: usize,
        max_new_tokens: usize,
    ) -> PrefillRow {
        PrefillRow {
            slot,
            tokens: tokens.to_vec(),
            start_pos,
            total_tokens,
            params: SamplingParams {
                max_new_tokens: Some(max_new_tokens),
                ..SamplingParams::default()
            },
        }
    }

    #[test]
    fn buffers_generated_tokens_after_final_prefill_chunk() {
        let model = FakeDiffusionModel {
            predictions: vec![prediction(&[10, 11, 12], 4)],
            ..FakeDiffusionModel::default()
        };
        let mut executor = BufferedDiffusionExecutor::new(model, config(3));
        let mut kv = HostPagedKvPool::new(1, 8, 4);

        let first = executor
            .submit_prefill(&prefill_row(0, &[1, 2], 0, 4, 3), &mut kv)
            .unwrap();
        assert!(first.tokens.is_empty());

        let second = executor
            .submit_prefill(&prefill_row(0, &[3, 4], 2, 4, 3), &mut kv)
            .unwrap();
        assert_eq!(second.tokens[0].token, 10);
        assert_eq!(
            executor
                .next_buffered_token(0, kv.slot_epoch(0))
                .unwrap()
                .tokens[0]
                .token,
            11
        );
        assert_eq!(
            executor
                .next_buffered_token(0, kv.slot_epoch(0))
                .unwrap()
                .tokens[0]
                .token,
            12
        );

        let model = executor.into_inner();
        assert_eq!(model.prompts, vec![vec![1, 2, 3, 4]]);
        assert_eq!(model.commits, vec![vec![10, 11, 12]]);
    }

    #[test]
    fn carries_stop_finish_on_final_buffered_token() {
        let model = FakeDiffusionModel {
            predictions: vec![prediction(&[10, 99, 12], 4)],
            ..FakeDiffusionModel::default()
        };
        let mut executor = BufferedDiffusionExecutor::new(model, config(3));
        let mut kv = HostPagedKvPool::new(1, 8, 4);

        let first = executor
            .submit_prefill(&prefill_row(0, &[1], 0, 1, 3), &mut kv)
            .unwrap();
        assert_eq!(first.tokens[0].token, 10);
        let terminal = executor.next_buffered_token(0, kv.slot_epoch(0)).unwrap();
        assert_eq!(terminal.tokens[0].token, 99);
        assert_eq!(terminal.tokens[0].finish, Some(FinishReason::Stop));
    }

    #[test]
    fn request_params_override_seed_stops_and_length() {
        let model = FakeDiffusionModel {
            predictions: vec![prediction(&[42, 43], 4)],
            ..FakeDiffusionModel::default()
        };
        let mut executor = BufferedDiffusionExecutor::new(model, config(4));
        let mut kv = HostPagedKvPool::new(1, 8, 4);
        let mut row = prefill_row(0, &[1], 0, 1, 2);
        row.params.seed = Some(123);
        row.params.stop_token_ids = vec![43];

        let out = executor.submit_prefill(&row, &mut kv).unwrap();
        assert_eq!(out.tokens[0].token, 42);
        let terminal = executor.next_buffered_token(0, kv.slot_epoch(0)).unwrap();
        assert_eq!(terminal.tokens[0].token, 43);
        assert_eq!(terminal.tokens[0].finish, Some(FinishReason::Stop));
    }
}
