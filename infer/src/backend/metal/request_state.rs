use std::collections::VecDeque;

use anyhow::{Context, Result, bail, ensure};

use super::KV_CACHE_CHUNK;
use super::config::{MetalModelArch, MetalModelConfig};
use super::dflash::{self, MetalDflashRuntime};
use super::forward::build_forward_graph;
use super::gdr::MetalRecurrentState;
use super::kv_pool::MetalKVPool;
use super::mlx::{MlxArray, concatenate_axis, eval, slice, take_axis, zeros};
use super::ops::{clear_metal_cache, extend_kv_cache};
use super::qwen35::{CppQwen35Model, Qwen35MetalWeights, qwen35_forward_step};
use super::sampling::{gpu_sample_token, validate_metal_sampling_params};
use super::weights::{MetalWeights, StandardMetalWeights};
use crate::sampler::SamplingParams;

const METAL_REQUEST_STATE_ID: usize = 0;

fn round_up_kv_capacity(tokens: i32) -> i32 {
    ((tokens + KV_CACHE_CHUNK - 1) / KV_CACHE_CHUNK) * KV_CACHE_CHUNK
}

fn left_pad_kv_cache_row(
    array: &MlxArray,
    left_pad: i32,
    cache_len: i32,
    target_kv_capacity: i32,
) -> MlxArray {
    let shape = array.shape();
    debug_assert_eq!(shape.len(), 4);
    debug_assert_eq!(shape[0], 1);
    debug_assert!(left_pad >= 0);
    debug_assert!(cache_len >= 0);
    debug_assert!(left_pad + cache_len <= target_kv_capacity);

    let n_kv = shape[1];
    let head_dim = shape[3];
    let mut padded = zeros(&[1, n_kv, target_kv_capacity, head_dim], array.dtype());
    if cache_len == 0 {
        return padded;
    }

    let valid = slice(
        array,
        &[0, 0, 0, 0],
        &[1, n_kv, cache_len, head_dim],
        &[1, 1, 1, 1],
    );
    padded = super::mlx::slice_update(
        &mut padded,
        &valid,
        &[0, 0, left_pad, 0],
        &[1, n_kv, left_pad + cache_len, head_dim],
    );
    padded
}

fn strip_left_padding_from_packed_row(
    array: &MlxArray,
    row: i32,
    left_pad: i32,
    batch_cache_len: i32,
    row_kv_capacity: i32,
) -> MlxArray {
    let row_slice = slice_row(array, row);
    let shape = row_slice.shape();
    debug_assert_eq!(shape.len(), 4);
    debug_assert_eq!(shape[0], 1);
    debug_assert!(left_pad >= 0);
    debug_assert!(batch_cache_len >= left_pad);

    let n_kv = shape[1];
    let head_dim = shape[3];
    let valid_len = batch_cache_len - left_pad;
    let mut unpadded = zeros(&[1, n_kv, row_kv_capacity, head_dim], row_slice.dtype());
    if valid_len == 0 {
        return unpadded;
    }

    let valid = slice(
        &row_slice,
        &[0, 0, left_pad, 0],
        &[1, n_kv, batch_cache_len, head_dim],
        &[1, 1, 1, 1],
    );
    unpadded = super::mlx::slice_update(
        &mut unpadded,
        &valid,
        &[0, 0, 0, 0],
        &[1, n_kv, valid_len, head_dim],
    );
    unpadded
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalRequestPhase {
    Prefill,
    Decode,
    Finished,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillChunkResult {
    pub processed_tokens: usize,
    pub emitted_token: Option<u32>,
    pub phase: MetalRequestPhase,
    pub finish_reason: Option<&'static str>,
}

trait StepDriver {
    fn prefill_token(&mut self, token: u32, terminal_prompt: bool) -> Result<Option<u32>>;
    fn prefill_tokens(&mut self, tokens: &[u32], terminal_prompt: bool) -> Result<Option<u32>> {
        let mut emitted = None;
        for (idx, &token) in tokens.iter().enumerate() {
            let is_terminal = terminal_prompt && idx + 1 == tokens.len();
            let sampled = self.prefill_token(token, is_terminal)?;
            if is_terminal {
                emitted = sampled;
            } else if sampled.is_some() {
                bail!("non-terminal prefill step unexpectedly emitted a sampled token");
            }
        }
        Ok(emitted)
    }
    fn decode_token(&mut self, token: u32) -> Result<u32>;
    fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}

struct ResumableRequestState<D: StepDriver> {
    driver: D,
    prompt_tokens: Vec<u32>,
    prompt_cursor: usize,
    max_new_tokens: usize,
    generated_tokens: usize,
    last_token: Option<u32>,
    stop_token_ids: Vec<u32>,
    eos_token_id: u32,
    ignore_eos: bool,
    phase: MetalRequestPhase,
    finish_reason: Option<&'static str>,
    cleaned_up: bool,
}

impl<D: StepDriver> ResumableRequestState<D> {
    fn new(
        driver: D,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
        stop_token_ids: Vec<u32>,
        eos_token_id: u32,
        ignore_eos: bool,
    ) -> Result<Self> {
        ensure!(
            !prompt_tokens.is_empty(),
            "Metal request state requires at least one prompt token"
        );
        ensure!(max_new_tokens > 0, "max_new_tokens must be >= 1");
        Ok(Self {
            driver,
            prompt_tokens,
            prompt_cursor: 0,
            max_new_tokens,
            generated_tokens: 0,
            last_token: None,
            stop_token_ids,
            eos_token_id,
            ignore_eos,
            phase: MetalRequestPhase::Prefill,
            finish_reason: None,
            cleaned_up: false,
        })
    }

    fn phase(&self) -> MetalRequestPhase {
        self.phase
    }

    fn prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }

    fn prompt_progress(&self) -> usize {
        self.prompt_cursor
    }

    fn generated_tokens(&self) -> usize {
        self.generated_tokens
    }

    fn finish_reason(&self) -> Option<&'static str> {
        self.finish_reason
    }

    fn prefill_chunk(&mut self, budget: usize) -> Result<PrefillChunkResult> {
        ensure!(budget > 0, "prefill budget must be >= 1");
        ensure!(
            self.phase == MetalRequestPhase::Prefill,
            "prefill_chunk requires Prefill phase, got {:?}",
            self.phase
        );

        let remaining = self.prompt_tokens.len() - self.prompt_cursor;
        let processed = budget.min(remaining);
        let prompt_end = self.prompt_cursor + processed;
        let terminal_prompt = prompt_end == self.prompt_tokens.len();
        let sampled = self.driver.prefill_tokens(
            &self.prompt_tokens[self.prompt_cursor..prompt_end],
            terminal_prompt,
        )?;
        self.prompt_cursor = prompt_end;

        if terminal_prompt {
            let sampled_token =
                sampled.context("terminal prefill step did not emit a sampled token")?;
            self.record_sampled_token(sampled_token)?;
            return Ok(PrefillChunkResult {
                processed_tokens: processed,
                emitted_token: Some(sampled_token),
                phase: self.phase,
                finish_reason: self.finish_reason,
            });
        }

        Ok(PrefillChunkResult {
            processed_tokens: processed,
            emitted_token: None,
            phase: self.phase,
            finish_reason: self.finish_reason,
        })
    }

    fn decode_step(&mut self) -> Result<Option<u32>> {
        ensure!(
            self.phase == MetalRequestPhase::Decode,
            "decode_step requires Decode phase, got {:?}",
            self.phase
        );
        let input_token = self
            .last_token
            .context("decode_step requires a committed prefill token")?;
        let sampled_token = self.driver.decode_token(input_token)?;
        self.record_sampled_token(sampled_token)?;
        Ok(Some(sampled_token))
    }

    fn cancel(&mut self) -> Result<()> {
        if self.phase != MetalRequestPhase::Finished {
            self.phase = MetalRequestPhase::Finished;
            self.finish_reason = Some("cancelled");
            self.cleanup_once()?;
        }
        Ok(())
    }

    fn record_sampled_token(&mut self, sampled_token: u32) -> Result<()> {
        self.last_token = Some(sampled_token);
        self.generated_tokens += 1;

        if self.should_stop(sampled_token) {
            self.phase = MetalRequestPhase::Finished;
            self.finish_reason = Some("stop");
            self.cleanup_once()?;
        } else if self.generated_tokens >= self.max_new_tokens {
            self.phase = MetalRequestPhase::Finished;
            self.finish_reason = Some("length");
            self.cleanup_once()?;
        } else {
            self.phase = MetalRequestPhase::Decode;
        }

        Ok(())
    }

    fn should_stop(&self, token: u32) -> bool {
        (!self.ignore_eos && token == self.eos_token_id) || self.stop_token_ids.contains(&token)
    }

    fn cleanup_once(&mut self) -> Result<()> {
        if self.cleaned_up {
            return Ok(());
        }
        self.driver.cleanup()?;
        self.cleaned_up = true;
        Ok(())
    }
}

impl<D: StepDriver> Drop for ResumableRequestState<D> {
    fn drop(&mut self) {
        if self.cleaned_up {
            return;
        }
        if let Err(err) = self.driver.cleanup() {
            log::warn!("Metal request state cleanup failed during drop: {err:#}");
        }
        self.cleaned_up = true;
    }
}

pub struct MetalRequestState<'a> {
    inner: MetalRequestStateInner<'a>,
}

enum MetalRequestStateInner<'a> {
    Qwen3(ResumableRequestState<Qwen3StepDriver<'a>>),
    Qwen35(ResumableRequestState<Qwen35StepDriver<'a>>),
}

pub(crate) struct Qwen3PrefixSnapshot {
    pub token_ids: Vec<u32>,
    pub k_rows_by_layer: Vec<MlxArray>,
    pub v_rows_by_layer: Vec<MlxArray>,
}

pub(crate) struct Qwen35PrefixSnapshot {
    pub token_ids: Vec<u32>,
    pub kv_flat: Vec<MlxArray>,
    pub gdr_flat: Vec<MlxArray>,
    pub cache_len: i32,
    pub kv_capacity: i32,
}

pub(crate) struct Qwen35PackedDecodeBatch<'a> {
    weights: &'a Qwen35MetalWeights,
    config: &'a MetalModelConfig,
    arch: &'a super::config::MetalQwen35ArchConfig,
    batch_cache_len: i32,
    kv_capacity: i32,
    left_padding: Vec<i32>,
    n_kv_per_request: i32,
    n_gdr_per_request: i32,
    packed_kv_flat: Vec<MlxArray>,
    packed_gdr_flat: Vec<MlxArray>,
}

impl<'a> Qwen35PackedDecodeBatch<'a> {
    pub(crate) fn batch_size(&self) -> usize {
        self.packed_kv_flat
            .first()
            .or_else(|| self.packed_gdr_flat.first())
            .map(|array| array.shape().first().copied().unwrap_or(0).max(0) as usize)
            .unwrap_or(0)
    }

    /// Shared column cursor for this packed batch. All rows write their next
    /// decode token at this column; rows that joined late are left-padded so
    /// their valid KV data sits in `[left_padding[i], batch_cache_len)`.
    pub(crate) fn batch_cache_len(&self) -> i32 {
        self.batch_cache_len
    }

    fn matches_driver(&self, driver: &Qwen35StepDriver<'a>) -> bool {
        std::ptr::eq(self.weights, driver.weights)
            && std::ptr::eq(self.config, driver.config)
            && std::ptr::eq(self.arch, driver.arch)
            && matches!(driver.mode, Qwen35StepMode::Cpp(_))
    }

    fn ensure_capacity_for_states(
        &mut self,
        states: &mut [&mut ResumableRequestState<Qwen35StepDriver<'a>>],
        needed_tokens: i32,
    ) {
        for state in states.iter_mut() {
            state.driver.kv_capacity = state.driver.kv_capacity.max(self.kv_capacity);
        }
        while needed_tokens > self.kv_capacity {
            let new_cap = self.kv_capacity + KV_CACHE_CHUNK;
            for cache in &mut self.packed_kv_flat {
                extend_kv_cache(
                    cache,
                    self.config.num_key_value_heads as i32,
                    self.config.head_dim as i32,
                    new_cap,
                );
            }
            self.kv_capacity = new_cap;
            for state in states.iter_mut() {
                state.driver.kv_capacity = new_cap;
            }
        }
    }

    pub(crate) fn retain_rows(&mut self, row_indices: &[usize]) -> Result<()> {
        if row_indices.len() == self.batch_size()
            && row_indices.iter().enumerate().all(|(idx, row)| idx == *row)
        {
            return Ok(());
        }

        let indices: Vec<i32> = row_indices
            .iter()
            .map(|&row| i32::try_from(row).context("Qwen3.5 packed batch row index overflow"))
            .collect::<Result<_>>()?;
        let index_arr = MlxArray::from_slice_i32(
            &indices,
            &[i32::try_from(indices.len()).context("Qwen3.5 packed batch row count overflow")?],
        );

        for tensor in &mut self.packed_kv_flat {
            let old = std::mem::replace(tensor, take_axis(tensor, &index_arr, 0));
            drop(old);
        }
        for tensor in &mut self.packed_gdr_flat {
            let old = std::mem::replace(tensor, take_axis(tensor, &index_arr, 0));
            drop(old);
        }
        self.left_padding = row_indices
            .iter()
            .map(|&row| self.left_padding[row])
            .collect();

        let mut eval_refs =
            Vec::with_capacity(self.packed_kv_flat.len() + self.packed_gdr_flat.len());
        eval_refs.extend(self.packed_kv_flat.iter());
        eval_refs.extend(self.packed_gdr_flat.iter());
        let eval_refs: Vec<&MlxArray> = eval_refs.into_iter().collect();
        eval(&eval_refs);
        Ok(())
    }

    pub(crate) fn admit_rows(
        &mut self,
        states: &mut [&mut MetalRequestState<'a>],
        new_indices: &[usize],
    ) -> Result<()> {
        if new_indices.is_empty() {
            return Ok(());
        }

        let mut target_kv_capacity = self.kv_capacity;
        for &idx in new_indices {
            let Some(state) = states.get(idx) else {
                bail!("Qwen3.5 packed batch admit_rows index {idx} out of range");
            };
            let state_ref: &MetalRequestState<'a> = state;
            let MetalRequestStateInner::Qwen35(qwen35) = &state_ref.inner else {
                bail!("Qwen3.5 packed batch admit_rows received mixed model batch");
            };
            if qwen35.phase() != MetalRequestPhase::Decode {
                bail!("Qwen3.5 packed batch admit_rows requires Decode phase");
            }
            target_kv_capacity = target_kv_capacity.max(qwen35.driver.kv_capacity);
        }
        target_kv_capacity = round_up_kv_capacity(target_kv_capacity);
        while target_kv_capacity > self.kv_capacity {
            let new_cap = self.kv_capacity + KV_CACHE_CHUNK;
            for cache in &mut self.packed_kv_flat {
                extend_kv_cache(
                    cache,
                    self.config.num_key_value_heads as i32,
                    self.config.head_dim as i32,
                    new_cap,
                );
            }
            self.kv_capacity = new_cap;
        }

        let mut new_kv_rows: Vec<Vec<MlxArray>> = (0..self.n_kv_per_request)
            .map(|_| Vec::with_capacity(new_indices.len()))
            .collect();
        let mut new_gdr_rows: Vec<Vec<MlxArray>> = (0..self.n_gdr_per_request)
            .map(|_| Vec::with_capacity(new_indices.len()))
            .collect();
        let mut new_left_padding = Vec::with_capacity(new_indices.len());

        for state in states.iter_mut() {
            let state_ref: &mut MetalRequestState<'a> = state;
            if let MetalRequestStateInner::Qwen35(qwen35) = &mut state_ref.inner {
                if self.matches_driver(&qwen35.driver) {
                    qwen35.driver.kv_capacity = qwen35.driver.kv_capacity.max(self.kv_capacity);
                }
            }
        }

        for &idx in new_indices {
            let state_ref: &mut MetalRequestState<'a> = states.get_mut(idx).ok_or_else(|| {
                anyhow::anyhow!("Qwen3.5 packed batch admit_rows index {idx} out of range")
            })?;
            let MetalRequestStateInner::Qwen35(qwen35) = &mut state_ref.inner else {
                bail!("Qwen3.5 packed batch admit_rows received mixed model batch");
            };
            ensure!(
                qwen35.phase() == MetalRequestPhase::Decode,
                "Qwen3.5 packed batch admit_rows requires Decode phase"
            );
            ensure!(
                std::ptr::eq(qwen35.driver.weights, self.weights)
                    && std::ptr::eq(qwen35.driver.config, self.config)
                    && std::ptr::eq(qwen35.driver.arch, self.arch),
                "Qwen3.5 packed batch admit_rows requires matching model handles"
            );

            qwen35.driver.ensure_capacity(self.kv_capacity);
            qwen35.driver.kv_capacity = self.kv_capacity;
            let left_pad = self.batch_cache_len - qwen35.driver.cache_len;
            ensure!(
                left_pad >= 0,
                "Qwen3.5 packed batch cannot admit cache_len {} into batch_cache_len {}",
                qwen35.driver.cache_len,
                self.batch_cache_len
            );

            match &mut qwen35.driver.mode {
                Qwen35StepMode::Cpp(cpp) => {
                    ensure!(
                        i32::try_from(cpp.kv_flat.len())
                            .context("Qwen3.5 packed batch admit_rows kv count overflow")?
                            == self.n_kv_per_request
                            && i32::try_from(cpp.gdr_flat.len())
                                .context("Qwen3.5 packed batch admit_rows gdr count overflow")?
                                == self.n_gdr_per_request,
                        "Qwen3.5 packed batch admit_rows requires matching state vector counts"
                    );
                    for (slot_idx, slot) in cpp.kv_flat.iter().enumerate() {
                        new_kv_rows[slot_idx].push(left_pad_kv_cache_row(
                            slot,
                            left_pad,
                            qwen35.driver.cache_len,
                            self.kv_capacity,
                        ));
                    }
                    for (slot_idx, slot) in cpp.gdr_flat.iter().enumerate() {
                        new_gdr_rows[slot_idx].push(slot.clone());
                    }
                }
                Qwen35StepMode::Rust(_) => {
                    bail!("Qwen3.5 packed batch admit_rows requires compiled Qwen3.5 state")
                }
            }

            new_left_padding.push(left_pad);
        }

        for (slot_idx, appended_rows) in new_kv_rows.iter_mut().enumerate() {
            let mut concatenated = Vec::with_capacity(1 + appended_rows.len());
            concatenated.push(self.packed_kv_flat[slot_idx].clone());
            concatenated.append(appended_rows);
            let old = std::mem::replace(
                &mut self.packed_kv_flat[slot_idx],
                concatenate_axis(&concatenated, 0),
            );
            drop(old);
        }
        for (slot_idx, appended_rows) in new_gdr_rows.iter_mut().enumerate() {
            let mut concatenated = Vec::with_capacity(1 + appended_rows.len());
            concatenated.push(self.packed_gdr_flat[slot_idx].clone());
            concatenated.append(appended_rows);
            let old = std::mem::replace(
                &mut self.packed_gdr_flat[slot_idx],
                concatenate_axis(&concatenated, 0),
            );
            drop(old);
        }
        self.left_padding.extend(new_left_padding);

        let mut eval_refs =
            Vec::with_capacity(self.packed_kv_flat.len() + self.packed_gdr_flat.len());
        eval_refs.extend(self.packed_kv_flat.iter());
        eval_refs.extend(self.packed_gdr_flat.iter());
        let eval_refs: Vec<&MlxArray> = eval_refs.into_iter().collect();
        eval(&eval_refs);
        Ok(())
    }
}

impl<'a> MetalRequestState<'a> {
    pub(super) fn new(
        weights: &'a MetalWeights,
        config: &'a MetalModelConfig,
        prompt_tokens: Vec<u32>,
        params: &SamplingParams,
        use_kv_pool: bool,
        max_new_tokens: usize,
        dflash_runtime: Option<(&'static MetalDflashRuntime, &'static MetalModelConfig)>,
    ) -> Result<Self> {
        validate_metal_sampling_params(params)?;

        let inner = match weights {
            MetalWeights::Qwen3(weights) => {
                // DFlash needs direct KV cache access — disable pool when DFlash is active.
                let effective_kv_pool = use_kv_pool && dflash_runtime.is_none();
                let driver = Qwen3StepDriver::new(
                    weights,
                    config,
                    params,
                    effective_kv_pool,
                    &prompt_tokens,
                    max_new_tokens,
                    dflash_runtime,
                )?;
                let state = ResumableRequestState::new(
                    driver,
                    prompt_tokens,
                    max_new_tokens,
                    params.stop_token_ids.clone(),
                    config.eos_token_id,
                    params.ignore_eos,
                )?;
                MetalRequestStateInner::Qwen3(state)
            }
            MetalWeights::Qwen35(weights) => {
                let driver = Qwen35StepDriver::new(
                    weights,
                    config,
                    params,
                    &prompt_tokens,
                    max_new_tokens,
                    dflash_runtime,
                )?;
                let state = ResumableRequestState::new(
                    driver,
                    prompt_tokens,
                    max_new_tokens,
                    params.stop_token_ids.clone(),
                    config.eos_token_id,
                    params.ignore_eos,
                )?;
                MetalRequestStateInner::Qwen35(state)
            }
        };

        Ok(Self { inner })
    }

    pub fn phase(&self) -> MetalRequestPhase {
        match &self.inner {
            MetalRequestStateInner::Qwen3(state) => state.phase(),
            MetalRequestStateInner::Qwen35(state) => state.phase(),
        }
    }

    pub fn prompt_len(&self) -> usize {
        match &self.inner {
            MetalRequestStateInner::Qwen3(state) => state.prompt_len(),
            MetalRequestStateInner::Qwen35(state) => state.prompt_len(),
        }
    }

    pub fn prompt_progress(&self) -> usize {
        match &self.inner {
            MetalRequestStateInner::Qwen3(state) => state.prompt_progress(),
            MetalRequestStateInner::Qwen35(state) => state.prompt_progress(),
        }
    }

    pub fn generated_tokens(&self) -> usize {
        match &self.inner {
            MetalRequestStateInner::Qwen3(state) => state.generated_tokens(),
            MetalRequestStateInner::Qwen35(state) => state.generated_tokens(),
        }
    }

    pub fn finish_reason(&self) -> Option<&'static str> {
        match &self.inner {
            MetalRequestStateInner::Qwen3(state) => state.finish_reason(),
            MetalRequestStateInner::Qwen35(state) => state.finish_reason(),
        }
    }

    pub fn kv_pool_usage(&self) -> Option<(usize, usize)> {
        match &self.inner {
            MetalRequestStateInner::Qwen3(state) => state
                .driver
                .kv_pool
                .as_ref()
                .map(|pool| (pool.total_tokens_used(), pool.max_total_tokens())),
            MetalRequestStateInner::Qwen35(_) => None,
        }
    }

    pub fn prefill_chunk(&mut self, budget: usize) -> Result<PrefillChunkResult> {
        match &mut self.inner {
            MetalRequestStateInner::Qwen3(state) => state.prefill_chunk(budget),
            MetalRequestStateInner::Qwen35(state) => state.prefill_chunk(budget),
        }
    }

    pub fn decode_step(&mut self) -> Result<Option<u32>> {
        match &mut self.inner {
            MetalRequestStateInner::Qwen3(state) => state.decode_step(),
            MetalRequestStateInner::Qwen35(state) => state.decode_step(),
        }
    }

    /// Try to execute one cross-request Qwen3 decode batch.
    ///
    /// Returns `Ok(None)` when the batch is not eligible for the Qwen3 batched
    /// path (for example Qwen3.5 requests or non-decode phases), so the caller
    /// can fall back to per-request decode. Returns sampled tokens in the same
    /// order as the input slice on success.
    pub fn decode_batch(states: &mut [&mut MetalRequestState<'a>]) -> Result<Option<Vec<u32>>> {
        if states.len() < 2 {
            return Ok(None);
        }

        match &mut states[0].inner {
            MetalRequestStateInner::Qwen3(_) => {
                let mut qwen3_states = Vec::with_capacity(states.len());
                for state in states.iter_mut() {
                    let state_ref: &mut MetalRequestState<'a> = state;
                    match &mut state_ref.inner {
                        MetalRequestStateInner::Qwen3(qwen3) => {
                            if qwen3.phase() != MetalRequestPhase::Decode {
                                return Ok(None);
                            }
                            qwen3_states.push(qwen3);
                        }
                        MetalRequestStateInner::Qwen35(_) => return Ok(None),
                    }
                }

                let first_cache_len = qwen3_states[0].driver.cache_len;
                if qwen3_states
                    .iter()
                    .any(|state| state.driver.cache_len != first_cache_len)
                {
                    return Ok(None);
                }

                let sampled = decode_qwen3_batch(&mut qwen3_states)?;
                Ok(Some(sampled))
            }
            MetalRequestStateInner::Qwen35(_) => {
                let mut qwen35_states = Vec::with_capacity(states.len());
                for state in states.iter_mut() {
                    let state_ref: &mut MetalRequestState<'a> = state;
                    match &mut state_ref.inner {
                        MetalRequestStateInner::Qwen35(qwen35) => {
                            if qwen35.phase() != MetalRequestPhase::Decode {
                                return Ok(None);
                            }
                            qwen35_states.push(qwen35);
                        }
                        MetalRequestStateInner::Qwen3(_) => return Ok(None),
                    }
                }

                let first_cache_len = qwen35_states[0].driver.cache_len;
                let first_kv_capacity = qwen35_states[0].driver.kv_capacity;
                let cpp_mode = matches!(qwen35_states[0].driver.mode, Qwen35StepMode::Cpp(_));
                if !cpp_mode
                    || qwen35_states.iter().any(|state| {
                        state.driver.cache_len != first_cache_len
                            || state.driver.kv_capacity != first_kv_capacity
                            || !matches!(state.driver.mode, Qwen35StepMode::Cpp(_))
                    })
                {
                    return Ok(None);
                }

                let sampled = decode_qwen35_batch(&mut qwen35_states)?;
                Ok(Some(sampled))
            }
        }
    }

    /// Return this request's Qwen3.5 decode cursor (`cache_len`) if it is a
    /// Qwen3.5 request currently in the `Decode` phase. Used by the scheduler
    /// runtime to decide whether a freshly prefilled row can join an existing
    /// packed batch without forcing a full cache rebuild.
    pub(crate) fn qwen35_decode_cursor(&self) -> Option<i32> {
        match &self.inner {
            MetalRequestStateInner::Qwen35(state) if state.phase() == MetalRequestPhase::Decode => {
                Some(state.driver.cache_len)
            }
            _ => None,
        }
    }

    /// Whether this request has DFlash speculative decode enabled.
    pub(crate) fn is_dflash_enabled(&self) -> bool {
        match &self.inner {
            MetalRequestStateInner::Qwen3(state) => state.driver.dflash.is_some(),
            MetalRequestStateInner::Qwen35(state) => state.driver.dflash.is_some(),
        }
    }

    /// DFlash acceptance rate for this request: fraction of generated tokens
    /// that came from draft predictions (matches reference metric).
    /// Returns None if not DFlash-enabled or no blocks executed yet.
    pub(crate) fn dflash_acceptance_rate(&self) -> Option<f64> {
        match &self.inner {
            MetalRequestStateInner::Qwen3(state) => {
                let d = state.driver.dflash.as_ref()?;
                if d.acceptance_lengths.is_empty() {
                    return None;
                }
                let total_accepted: usize = d.acceptance_lengths.iter().sum();
                if total_accepted == 0 {
                    return Some(0.0);
                }
                let blocks = d.acceptance_lengths.len();
                let from_draft = total_accepted.saturating_sub(blocks);
                Some(from_draft as f64 / total_accepted as f64)
            }
            _ => None,
        }
    }

    /// DFlash per-block acceptance lengths + block size for metrics flush.
    /// Returns `(block_count, acceptance_lengths, block_size)` or `None` if
    /// not a DFlash request.
    pub(crate) fn dflash_block_stats(&self) -> Option<(usize, &[usize], usize)> {
        match &self.inner {
            MetalRequestStateInner::Qwen3(state) => {
                let d = state.driver.dflash.as_ref()?;
                Some((
                    d.acceptance_lengths.len(),
                    &d.acceptance_lengths,
                    d.runtime.block_size(),
                ))
            }
            MetalRequestStateInner::Qwen35(state) => {
                let d = state.driver.dflash.as_ref()?;
                Some((
                    d.acceptance_lengths.len(),
                    &d.acceptance_lengths,
                    d.runtime.block_size(),
                ))
            }
        }
    }

    pub(crate) fn try_build_qwen35_packed_decode_batch(
        states: &mut [&mut MetalRequestState<'a>],
    ) -> Result<Option<Qwen35PackedDecodeBatch<'a>>> {
        if states.len() < 2 {
            return Ok(None);
        }

        let mut qwen35_states = Vec::with_capacity(states.len());
        for state in states.iter_mut() {
            let state_ref: &mut MetalRequestState<'a> = state;
            match &mut state_ref.inner {
                MetalRequestStateInner::Qwen35(qwen35) => {
                    if qwen35.phase() != MetalRequestPhase::Decode {
                        return Ok(None);
                    }
                    qwen35_states.push(qwen35);
                }
                MetalRequestStateInner::Qwen3(_) => return Ok(None),
            }
        }

        try_build_qwen35_packed_decode_batch(&mut qwen35_states)
    }

    pub(crate) fn try_decode_qwen35_packed_batch(
        states: &mut [&mut MetalRequestState<'a>],
        batch: &mut Qwen35PackedDecodeBatch<'a>,
    ) -> Result<Option<Vec<u32>>> {
        if states.len() < 2 {
            return Ok(None);
        }

        let mut qwen35_states = Vec::with_capacity(states.len());
        for state in states.iter_mut() {
            let state_ref: &mut MetalRequestState<'a> = state;
            match &mut state_ref.inner {
                MetalRequestStateInner::Qwen35(qwen35) => {
                    if qwen35.phase() != MetalRequestPhase::Decode
                        || !batch.matches_driver(&qwen35.driver)
                    {
                        return Ok(None);
                    }
                    qwen35_states.push(qwen35);
                }
                MetalRequestStateInner::Qwen3(_) => return Ok(None),
            }
        }

        if qwen35_states.len() != batch.batch_size() {
            return Ok(None);
        }

        let sampled = decode_qwen35_packed_batch(&mut qwen35_states, batch)?;
        Ok(Some(sampled))
    }

    pub(crate) fn sync_qwen35_packed_decode_batch(
        states: &mut [&mut MetalRequestState<'a>],
        batch: &Qwen35PackedDecodeBatch<'a>,
    ) -> Result<()> {
        let mut qwen35_states = Vec::with_capacity(states.len());
        for state in states.iter_mut() {
            let state_ref: &mut MetalRequestState<'a> = state;
            match &mut state_ref.inner {
                MetalRequestStateInner::Qwen35(qwen35) => qwen35_states.push(qwen35),
                MetalRequestStateInner::Qwen3(_) => {
                    bail!("sync_qwen35_packed_decode_batch received mixed model batch")
                }
            }
        }

        sync_qwen35_packed_decode_batch(&mut qwen35_states, batch)
    }

    pub fn cancel(&mut self) -> Result<()> {
        match &mut self.inner {
            MetalRequestStateInner::Qwen3(state) => state.cancel(),
            MetalRequestStateInner::Qwen35(state) => state.cancel(),
        }
    }

    pub(crate) fn import_qwen3_prefix_from_pool(
        &mut self,
        shared_pool: &MetalKVPool,
        matched_len: usize,
        slot_indices: &[u32],
    ) -> Result<bool> {
        match &mut self.inner {
            MetalRequestStateInner::Qwen3(state) => {
                ensure!(
                    state.phase == MetalRequestPhase::Prefill,
                    "Qwen3 prefix import requires Prefill phase, got {:?}",
                    state.phase
                );
                ensure!(
                    state.prompt_cursor == 0 && state.generated_tokens == 0,
                    "Qwen3 prefix import requires a fresh request state"
                );
                ensure!(
                    matched_len == slot_indices.len(),
                    "Qwen3 prefix import length {} does not match {} slot indices",
                    matched_len,
                    slot_indices.len()
                );
                if matched_len == 0 || matched_len >= state.prompt_tokens.len() {
                    return Ok(false);
                }

                state
                    .driver
                    .import_prefix_from_pool(shared_pool, slot_indices)
                    .context("import Qwen3 cached prefix into request state")?;
                state.prompt_cursor = matched_len;
                Ok(true)
            }
            MetalRequestStateInner::Qwen35(_) => Ok(false),
        }
    }

    pub(crate) fn export_qwen3_prompt_prefix(
        &self,
        token_count: usize,
    ) -> Result<Option<Qwen3PrefixSnapshot>> {
        match &self.inner {
            MetalRequestStateInner::Qwen3(state) => {
                if token_count == 0 {
                    return Ok(None);
                }
                ensure!(
                    token_count <= state.prompt_tokens.len(),
                    "Qwen3 prefix export requested {} prompt tokens but prompt only has {}",
                    token_count,
                    state.prompt_tokens.len()
                );
                ensure!(
                    token_count <= state.prompt_cursor,
                    "Qwen3 prefix export requested {} tokens but only {} prompt tokens are materialized",
                    token_count,
                    state.prompt_cursor
                );
                let (k_rows_by_layer, v_rows_by_layer) = state
                    .driver
                    .export_prefix_rows(token_count)
                    .context("export Qwen3 cached prefix rows")?;
                Ok(Some(Qwen3PrefixSnapshot {
                    token_ids: state.prompt_tokens[..token_count].to_vec(),
                    k_rows_by_layer,
                    v_rows_by_layer,
                }))
            }
            MetalRequestStateInner::Qwen35(_) => Ok(None),
        }
    }

    pub(crate) fn import_qwen35_prefix_snapshot(
        &mut self,
        snapshot: &Qwen35PrefixSnapshot,
        matched_len: usize,
    ) -> Result<bool> {
        match &mut self.inner {
            MetalRequestStateInner::Qwen35(state) => {
                ensure!(
                    state.phase == MetalRequestPhase::Prefill,
                    "Qwen3.5 prefix import requires Prefill phase, got {:?}",
                    state.phase
                );
                ensure!(
                    state.prompt_cursor == 0 && state.generated_tokens == 0,
                    "Qwen3.5 prefix import requires a fresh request state"
                );
                ensure!(
                    matched_len == snapshot.token_ids.len(),
                    "Qwen3.5 prefix import length {} does not match snapshot {}",
                    matched_len,
                    snapshot.token_ids.len()
                );
                ensure!(
                    snapshot.cache_len == matched_len as i32,
                    "Qwen3.5 prefix snapshot cache_len {} does not match {} tokens",
                    snapshot.cache_len,
                    matched_len
                );
                if matched_len == 0 || matched_len >= state.prompt_tokens.len() {
                    return Ok(false);
                }

                state
                    .driver
                    .import_prefix_snapshot(snapshot)
                    .context("import Qwen3.5 cached prefix snapshot")?;
                state.prompt_cursor = matched_len;
                Ok(true)
            }
            MetalRequestStateInner::Qwen3(_) => Ok(false),
        }
    }

    pub(crate) fn export_qwen35_prompt_prefixes(
        &self,
        block_size: usize,
    ) -> Result<Vec<Qwen35PrefixSnapshot>> {
        match &self.inner {
            MetalRequestStateInner::Qwen35(state) => {
                if block_size == 0 {
                    return Ok(Vec::new());
                }
                let aligned_len = (state.prompt_cursor / block_size) * block_size;
                if aligned_len == 0 {
                    return Ok(Vec::new());
                }
                state
                    .driver
                    .build_prefix_snapshots(&state.prompt_tokens[..aligned_len], block_size)
            }
            MetalRequestStateInner::Qwen3(_) => Ok(Vec::new()),
        }
    }
}

fn decode_qwen3_batch(
    states: &mut [&mut ResumableRequestState<Qwen3StepDriver<'_>>],
) -> Result<Vec<u32>> {
    use super::mlx::{concatenate_axis, eval, rms_norm, slice, take_axis, transpose_axes};
    use super::ops::linear;

    ensure!(
        !states.is_empty(),
        "decode_qwen3_batch requires at least one request state"
    );

    let batch = i32::try_from(states.len()).context("decode_qwen3_batch batch size overflow")?;
    let first = &states[0].driver;
    let weights = first.weights;
    let n_heads = first.n_heads;
    let n_kv_heads = first.n_kv_heads;
    let head_dim = first.head_dim;
    let attn_scale = first.attn_scale;
    let rope_base = first.rope_base;
    let eps = first.eps;
    let kv_dim = n_kv_heads * head_dim;
    let cache_len = first.cache_len;
    let end_pos = cache_len + 1;

    for state in states.iter() {
        ensure!(
            std::ptr::eq(state.driver.weights, weights),
            "decode_qwen3_batch requires identical Qwen3 weight handles"
        );
        ensure!(
            state.driver.n_heads == n_heads
                && state.driver.n_kv_heads == n_kv_heads
                && state.driver.head_dim == head_dim,
            "decode_qwen3_batch requires identical Qwen3 geometry"
        );
    }

    let input_tokens: Vec<u32> = states
        .iter()
        .map(|state| {
            state
                .last_token
                .context("decode_qwen3_batch requires a committed prefill token")
        })
        .collect::<Result<_>>()?;

    if states.iter().any(|state| {
        let cache_len = state.driver.cache_len;
        cache_len > 0 && cache_len % KV_CACHE_CHUNK == 0
    }) {
        clear_metal_cache();
    }

    for state in states.iter_mut() {
        let driver = &mut state.driver;
        if let Some(pool) = driver.kv_pool.as_mut() {
            pool.alloc_tokens(METAL_REQUEST_STATE_ID, 1)
                .context("alloc MetalKVPool slot for batched decode")?;
        } else {
            driver.ensure_capacity(driver.cache_len + 1);
        }
    }

    let token_values: Vec<i32> = input_tokens.iter().map(|&token| token as i32).collect();
    let token_arr = MlxArray::from_slice_i32(&token_values, &[batch]);
    let mut x = take_axis(&weights.embed_tokens, &token_arr, 0);

    // MLX 0.31.1 scalar-rope `[B>1, H, S=1, D]` workaround: always feed an
    // int32[B] offsets array so the `fast::rope(..., const array&)` overload
    // is used. Same-length batch here, so every entry is `cache_len`; when
    // this path grows to varlen the values diverge.
    // See docs/experience/errors/2026-04-16-metal-varlen-rope-blocker.md.
    let rope_offsets_data: Vec<i32> = vec![cache_len; states.len()];
    let rope_offsets = MlxArray::from_slice_i32(&rope_offsets_data, &[batch]);

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let residual = x.clone();
        let x_norm = rms_norm(&x, &layer.input_layernorm, eps);
        let (q_raw, k_raw, v_raw) = layer.attention_inputs.project(&x_norm);

        let q = super::mlx::reshape(&q_raw, &[batch, 1, n_heads, head_dim]);
        let q = rms_norm(&q, &layer.q_norm, eps);
        let q = transpose_axes(&q, &[0, 2, 1, 3]);
        let q = super::mlx::rope_dynamic(&q, head_dim, false, rope_base, 1.0f32, &rope_offsets);

        let k = super::mlx::reshape(&k_raw, &[batch, 1, n_kv_heads, head_dim]);
        let k = rms_norm(&k, &layer.k_norm, eps);
        let k = transpose_axes(&k, &[0, 2, 1, 3]);
        let k = super::mlx::rope_dynamic(&k, head_dim, false, rope_base, 1.0f32, &rope_offsets);

        let v = super::mlx::reshape(&v_raw, &[batch, 1, n_kv_heads, head_dim]);
        let v = transpose_axes(&v, &[0, 2, 1, 3]);

        let k_rows = transpose_axes(&k, &[0, 2, 1, 3]);
        let k_rows = super::mlx::reshape(&k_rows, &[batch, kv_dim]);
        let v_rows = transpose_axes(&v, &[0, 2, 1, 3]);
        let v_rows = super::mlx::reshape(&v_rows, &[batch, kv_dim]);

        let mut batch_k = Vec::with_capacity(states.len());
        let mut batch_v = Vec::with_capacity(states.len());
        for (row_idx, state) in states.iter_mut().enumerate() {
            let row = i32::try_from(row_idx).context("decode_qwen3_batch row index overflow")?;
            let row_k = slice(&k_rows, &[row, 0], &[row + 1, kv_dim], &[1, 1]);
            let row_v = slice(&v_rows, &[row, 0], &[row + 1, kv_dim], &[1, 1]);

            let (k_full, v_full) = if let Some(pool) = state.driver.kv_pool.as_mut() {
                pool.write_kv(layer_idx, METAL_REQUEST_STATE_ID, &row_k, &row_v)
                    .context("write MetalKVPool during batched decode")?;
                pool.gather_kv(layer_idx, METAL_REQUEST_STATE_ID)
                    .context("gather MetalKVPool during batched decode")?
            } else {
                let k_token = slice(
                    &k,
                    &[row, 0, 0, 0],
                    &[row + 1, n_kv_heads, 1, head_dim],
                    &[1, 1, 1, 1],
                );
                let v_token = slice(
                    &v,
                    &[row, 0, 0, 0],
                    &[row + 1, n_kv_heads, 1, head_dim],
                    &[1, 1, 1, 1],
                );
                state.driver.k_caches[layer_idx] = super::mlx::slice_update(
                    &mut state.driver.k_caches[layer_idx],
                    &k_token,
                    &[0, 0, state.driver.cache_len, 0],
                    &[1, n_kv_heads, end_pos, head_dim],
                );
                state.driver.v_caches[layer_idx] = super::mlx::slice_update(
                    &mut state.driver.v_caches[layer_idx],
                    &v_token,
                    &[0, 0, state.driver.cache_len, 0],
                    &[1, n_kv_heads, end_pos, head_dim],
                );
                let k_full = slice(
                    &state.driver.k_caches[layer_idx],
                    &[0, 0, 0, 0],
                    &[1, n_kv_heads, end_pos, head_dim],
                    &[1, 1, 1, 1],
                );
                let v_full = slice(
                    &state.driver.v_caches[layer_idx],
                    &[0, 0, 0, 0],
                    &[1, n_kv_heads, end_pos, head_dim],
                    &[1, 1, 1, 1],
                );
                (k_full, v_full)
            };

            batch_k.push(k_full);
            batch_v.push(v_full);
        }

        let k_full = concatenate_axis(&batch_k, 0);
        let v_full = concatenate_axis(&batch_v, 0);
        let attn_out =
            super::mlx::scaled_dot_product_attention(&q, &k_full, &v_full, attn_scale, None);
        let attn_out = transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = super::mlx::reshape(&attn_out, &[batch, n_heads * head_dim]);
        let attn_out = linear(&attn_out, &layer.o_proj);
        x = super::mlx::add(&residual, &attn_out);

        let residual2 = x.clone();
        let xn = rms_norm(&x, &layer.post_attention_layernorm, eps);
        let (gate_raw, up) = layer.mlp_inputs.project(&xn);
        let gate = super::mlx::silu(&gate_raw);
        let mlp = linear(&super::mlx::multiply(&gate, &up), &layer.down_proj);
        x = super::mlx::add(&residual2, &mlp);
    }

    let logits = linear(&rms_norm(&x, &weights.norm, eps), &weights.lm_head);
    let mut sampled_tokens = Vec::with_capacity(states.len());
    let mut sampled_arrays = Vec::with_capacity(states.len());
    for (row_idx, state) in states.iter().enumerate() {
        let row = i32::try_from(row_idx).context("decode_qwen3_batch sample row overflow")?;
        let row_logits = slice(&logits, &[row, 0], &[row + 1, logits.shape()[1]], &[1, 1]);
        sampled_arrays.push(gpu_sample_token(&row_logits, &state.driver.sample_params));
    }
    let sample_refs: Vec<&MlxArray> = sampled_arrays.iter().collect();
    eval(&sample_refs);

    for (state, sampled) in states.iter_mut().zip(sampled_arrays.iter()) {
        let token = sampled.item_i32() as u32;
        state.driver.cache_len += 1;
        state.record_sampled_token(token)?;
        sampled_tokens.push(token);
    }

    Ok(sampled_tokens)
}

fn try_build_qwen35_packed_decode_batch<'a>(
    states: &mut [&mut ResumableRequestState<Qwen35StepDriver<'a>>],
) -> Result<Option<Qwen35PackedDecodeBatch<'a>>> {
    ensure!(
        !states.is_empty(),
        "try_build_qwen35_packed_decode_batch requires at least one request state"
    );

    let first = &states[0].driver;
    let weights = first.weights;
    let config = first.config;
    let arch = first.arch;

    let (n_kv_per_request, n_gdr_per_request) = match &first.mode {
        Qwen35StepMode::Cpp(state) => (
            i32::try_from(state.kv_flat.len())
                .context("Qwen3.5 packed decode batch kv count overflow")?,
            i32::try_from(state.gdr_flat.len())
                .context("Qwen3.5 packed decode batch gdr count overflow")?,
        ),
        Qwen35StepMode::Rust(_) => return Ok(None),
    };

    // Shape / model identity check only. `cache_len` and `kv_capacity` are
    // allowed to differ across rows — we unify them below via a shared
    // `batch_cache_len` cursor + per-row `left_padding` (mlx-lm BatchKVCache
    // pattern). Correctness of variable-length batching depends on both the
    // attention mask (columns [0, left_pad) zeroed) AND per-row RoPE offsets
    // (each row's Q/K rotated at its own logical position). The rope offsets
    // ride through the bridge via `current_rope_offsets` on
    // `Qwen35CompiledModel`; see `decode_qwen35_packed_batch` below.
    for state in states.iter() {
        if !std::ptr::eq(state.driver.weights, weights)
            || !std::ptr::eq(state.driver.config, config)
            || !std::ptr::eq(state.driver.arch, arch)
        {
            return Ok(None);
        }
        match &state.driver.mode {
            Qwen35StepMode::Cpp(cpp) => {
                if i32::try_from(cpp.kv_flat.len())
                    .context("Qwen3.5 packed decode batch kv count overflow")?
                    != n_kv_per_request
                    || i32::try_from(cpp.gdr_flat.len())
                        .context("Qwen3.5 packed decode batch gdr count overflow")?
                        != n_gdr_per_request
                {
                    return Ok(None);
                }
            }
            Qwen35StepMode::Rust(_) => return Ok(None),
        }
    }

    // Shared batch cursor = max of all per-row cache_lens. Rows with shorter
    // caches get left-padded up to this cursor so every row writes its next
    // decode token at the same column (`batch_cache_len`).
    let mut batch_cache_len: i32 = 0;
    let mut target_kv_capacity: i32 = 0;
    for state in states.iter() {
        batch_cache_len = batch_cache_len.max(state.driver.cache_len);
        target_kv_capacity = target_kv_capacity.max(state.driver.kv_capacity);
    }
    // Capacity must fit the next decode write (batch_cache_len + 1) rounded up
    // to KV_CACHE_CHUNK so future grow steps stay aligned.
    target_kv_capacity = target_kv_capacity.max(round_up_kv_capacity(batch_cache_len + 1));

    // Normalize every state's own storage up to target_kv_capacity before we
    // read its KV arrays — concatenate_axis requires identical trailing shapes.
    for state in states.iter_mut() {
        state.driver.ensure_capacity(target_kv_capacity);
        state.driver.kv_capacity = target_kv_capacity;
    }

    let mut left_padding = Vec::with_capacity(states.len());
    for state in states.iter() {
        let pad = batch_cache_len - state.driver.cache_len;
        debug_assert!(pad >= 0);
        left_padding.push(pad);
    }

    let mut packed_kv_flat = Vec::with_capacity(n_kv_per_request as usize);
    for kv_idx in 0..n_kv_per_request as usize {
        let mut per_request = Vec::with_capacity(states.len());
        for (row_idx, state) in states.iter().enumerate() {
            let Qwen35StepMode::Cpp(cpp) = &state.driver.mode else {
                unreachable!("checked above");
            };
            let pad = left_padding[row_idx];
            if pad == 0 {
                per_request.push(cpp.kv_flat[kv_idx].clone());
            } else {
                per_request.push(left_pad_kv_cache_row(
                    &cpp.kv_flat[kv_idx],
                    pad,
                    state.driver.cache_len,
                    target_kv_capacity,
                ));
            }
        }
        packed_kv_flat.push(concatenate_axis(&per_request, 0));
    }

    // GDR state is per-request recurrent state (not a time-series cache), so
    // it does NOT get left-padded — just stacked along the batch axis.
    let mut packed_gdr_flat = Vec::with_capacity(n_gdr_per_request as usize);
    for gdr_idx in 0..n_gdr_per_request as usize {
        let mut per_request = Vec::with_capacity(states.len());
        for state in states.iter() {
            let Qwen35StepMode::Cpp(cpp) = &state.driver.mode else {
                unreachable!("checked above");
            };
            per_request.push(cpp.gdr_flat[gdr_idx].clone());
        }
        packed_gdr_flat.push(concatenate_axis(&per_request, 0));
    }

    let mut eval_refs = Vec::with_capacity(packed_kv_flat.len() + packed_gdr_flat.len());
    eval_refs.extend(packed_kv_flat.iter());
    eval_refs.extend(packed_gdr_flat.iter());
    let eval_refs: Vec<&MlxArray> = eval_refs.into_iter().collect();
    eval(&eval_refs);

    Ok(Some(Qwen35PackedDecodeBatch {
        weights,
        config,
        arch,
        batch_cache_len,
        kv_capacity: target_kv_capacity,
        left_padding,
        n_kv_per_request,
        n_gdr_per_request,
        packed_kv_flat,
        packed_gdr_flat,
    }))
}

fn sync_qwen35_packed_decode_batch<'a>(
    states: &mut [&mut ResumableRequestState<Qwen35StepDriver<'a>>],
    batch: &Qwen35PackedDecodeBatch<'a>,
) -> Result<()> {
    ensure!(
        states.len() == batch.batch_size(),
        "sync_qwen35_packed_decode_batch expected {} states, got {}",
        batch.batch_size(),
        states.len()
    );

    for (row_idx, state) in states.iter_mut().enumerate() {
        ensure!(
            batch.matches_driver(&state.driver),
            "sync_qwen35_packed_decode_batch state mismatch at row {row_idx}"
        );
        let row = i32::try_from(row_idx).context("sync_qwen35_packed_decode_batch row overflow")?;
        let left_pad = batch.left_padding[row_idx];
        match &mut state.driver.mode {
            Qwen35StepMode::Cpp(cpp) => {
                // KV caches carry per-column valid-mask positions; strip the
                // left pad so each row's own cache is left-aligned again.
                for (slot, packed) in cpp.kv_flat.iter_mut().zip(batch.packed_kv_flat.iter()) {
                    let new_slot = if left_pad == 0 {
                        slice_row(packed, row)
                    } else {
                        strip_left_padding_from_packed_row(
                            packed,
                            row,
                            left_pad,
                            batch.batch_cache_len,
                            batch.kv_capacity,
                        )
                    };
                    let old = std::mem::replace(slot, new_slot);
                    drop(old);
                }
                // GDR recurrent state is not time-series — no pad to strip.
                for (slot, packed) in cpp.gdr_flat.iter_mut().zip(batch.packed_gdr_flat.iter()) {
                    let old = std::mem::replace(slot, slice_row(packed, row));
                    drop(old);
                }
                state.driver.kv_capacity = batch.kv_capacity;
                state.driver.cache_len = batch.batch_cache_len - left_pad;
            }
            Qwen35StepMode::Rust(_) => {
                bail!("sync_qwen35_packed_decode_batch requires compiled Qwen3.5 state")
            }
        }
    }

    Ok(())
}

fn decode_qwen35_packed_batch<'a>(
    states: &mut [&mut ResumableRequestState<Qwen35StepDriver<'a>>],
    batch: &mut Qwen35PackedDecodeBatch<'a>,
) -> Result<Vec<u32>> {
    ensure!(
        !states.is_empty(),
        "decode_qwen35_packed_batch requires at least one request state"
    );
    ensure!(
        states.len() == batch.batch_size(),
        "decode_qwen35_packed_batch expected {} states, got {}",
        batch.batch_size(),
        states.len()
    );

    if batch.batch_cache_len > 0 && batch.batch_cache_len % KV_CACHE_CHUNK == 0 {
        clear_metal_cache();
    }

    batch.ensure_capacity_for_states(states, batch.batch_cache_len + 1);

    let input_tokens: Vec<u32> = states
        .iter()
        .map(|state| {
            state
                .last_token
                .context("decode_qwen35_packed_batch requires a committed prefill token")
        })
        .collect::<Result<_>>()?;
    let token_values: Vec<i32> = input_tokens.iter().map(|&token| token as i32).collect();
    let token_arr =
        MlxArray::from_slice_i32(
            &token_values,
            &[i32::try_from(states.len())
                .context("decode_qwen35_packed_batch batch size overflow")?],
        );

    // Only materialize the additive attention mask when at least one row is
    // left-padded; same-length batches take the no-mask fast path (identical
    // to pre-varlen behavior).
    let needs_mask = batch.left_padding.iter().any(|&pad| pad != 0);
    let mask_opt: Option<MlxArray> = if needs_mask {
        Some(super::mlx::build_varlen_decode_mask(
            &batch.left_padding,
            batch.batch_cache_len,
        ))
    } else {
        None
    };

    // ALWAYS build per-row RoPE offsets for the packed (batched) decode.
    //
    // Two reasons:
    //   1. Varlen correctness — each row's new Q/K must rotate at its own
    //      logical position `batch_cache_len - left_padding[row]`, not at
    //      the shared `batch_cache_len`.
    //   2. MLX 0.31.1 bug workaround — `fast::rope(..., int offset)` on a
    //      `[B, H, S=1, D]` tensor with `B > 1` silently zeroes out batch
    //      rows > 0. The array-offset overload works for both B=1 and B>1.
    //      So even same-length batches (all offsets equal) must go through
    //      the array path to stay correct. See
    //      docs/experience/errors/2026-04-16-metal-varlen-rope-blocker.md.
    let rope_offsets_data: Vec<i32> = batch
        .left_padding
        .iter()
        .map(|&pad| batch.batch_cache_len - pad)
        .collect();
    let rope_offsets = MlxArray::from_slice_i32(
        &rope_offsets_data,
        &[i32::try_from(rope_offsets_data.len())
            .context("decode_qwen35_packed_batch rope offsets overflow")?],
    );

    let cpp_model = batch
        .weights
        .cpp_model
        .as_ref()
        .context("decode_qwen35_packed_batch requires the compiled Qwen3.5 path")?;
    let logits = cpp_model.step_batch_packed(
        &token_arr,
        i32::try_from(states.len()).context("decode_qwen35_packed_batch batch size overflow")?,
        batch.batch_cache_len,
        &mut batch.packed_kv_flat,
        batch.n_kv_per_request,
        &mut batch.packed_gdr_flat,
        batch.n_gdr_per_request,
        mask_opt.as_ref(),
        Some(&rope_offsets),
    )?;

    let mut eval_refs: Vec<&MlxArray> =
        Vec::with_capacity(1 + batch.packed_kv_flat.len() + batch.packed_gdr_flat.len());
    eval_refs.push(&logits);
    eval_refs.extend(batch.packed_kv_flat.iter());
    eval_refs.extend(batch.packed_gdr_flat.iter());
    eval(&eval_refs);

    let logits_shape = logits.shape().to_vec();
    ensure!(
        !logits_shape.is_empty()
            && logits_shape[0]
                == i32::try_from(states.len())
                    .context("decode_qwen35_packed_batch batch shape overflow")?,
        "decode_qwen35_packed_batch expected batched logits, got shape {:?}",
        logits_shape
    );

    batch.batch_cache_len += 1;

    let sampled_tokens = if qwen35_can_batch_sample(states) {
        let sampled = gpu_sample_token(&logits, &states[0].driver.params);
        let sampled = super::mlx::contiguous(&super::mlx::reshape(
            &sampled,
            &[i32::try_from(states.len())
                .context("decode_qwen35_packed_batch reshape overflow")?],
        ));
        eval(&[&sampled]);
        (0..states.len())
            .map(|row_idx| {
                let row =
                    i32::try_from(row_idx).context("decode_qwen35_packed_batch row overflow")?;
                Ok(slice_row(&sampled, row).item_i32() as u32)
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        let mut sampled_arrays = Vec::with_capacity(states.len());
        for row_idx in 0..states.len() {
            let row = i32::try_from(row_idx).context("decode_qwen35_packed_batch row overflow")?;
            sampled_arrays.push(gpu_sample_token(
                &slice_row(&logits, row),
                &states[row_idx].driver.params,
            ));
        }
        let sample_refs: Vec<&MlxArray> = sampled_arrays.iter().collect();
        eval(&sample_refs);
        sampled_arrays
            .iter()
            .map(|sampled| sampled.item_i32() as u32)
            .collect::<Vec<_>>()
    };

    let mut sampled_tokens_out = Vec::with_capacity(states.len());
    for (row_idx, (state, token)) in states
        .iter_mut()
        .zip(sampled_tokens.into_iter())
        .enumerate()
    {
        // Each row's own logical length = batch_cursor - its own pad, so a
        // row that joined late stays at its shorter length.
        state.driver.cache_len = batch.batch_cache_len - batch.left_padding[row_idx];
        state.driver.kv_capacity = batch.kv_capacity;
        state.record_sampled_token(token)?;
        sampled_tokens_out.push(token);
    }

    Ok(sampled_tokens_out)
}

fn decode_qwen35_batch(
    states: &mut [&mut ResumableRequestState<Qwen35StepDriver<'_>>],
) -> Result<Vec<u32>> {
    ensure!(
        !states.is_empty(),
        "decode_qwen35_batch requires at least one request state"
    );

    let batch = i32::try_from(states.len()).context("decode_qwen35_batch batch size overflow")?;
    let first = &states[0].driver;
    let weights = first.weights;
    let config = first.config;
    let arch = first.arch;
    let cache_len = first.cache_len;
    let kv_capacity = first.kv_capacity;
    let cpp_model = weights
        .cpp_model
        .as_ref()
        .context("decode_qwen35_batch requires the compiled Qwen3.5 path")?;

    for state in states.iter() {
        ensure!(
            std::ptr::eq(state.driver.weights, weights)
                && std::ptr::eq(state.driver.config, config)
                && std::ptr::eq(state.driver.arch, arch),
            "decode_qwen35_batch requires identical Qwen3.5 model handles"
        );
        ensure!(
            state.driver.cache_len == cache_len && state.driver.kv_capacity == kv_capacity,
            "decode_qwen35_batch requires identical cache_len and kv_capacity"
        );
        ensure!(
            matches!(state.driver.mode, Qwen35StepMode::Cpp(_)),
            "decode_qwen35_batch requires compiled Qwen3.5 state"
        );
    }

    if cache_len > 0 && cache_len % KV_CACHE_CHUNK == 0 {
        clear_metal_cache();
    }

    for state in states.iter_mut() {
        state.driver.ensure_capacity(state.driver.cache_len + 1);
    }

    let input_tokens: Vec<u32> = states
        .iter()
        .map(|state| {
            state
                .last_token
                .context("decode_qwen35_batch requires a committed prefill token")
        })
        .collect::<Result<_>>()?;
    let token_values: Vec<i32> = input_tokens.iter().map(|&token| token as i32).collect();
    let token_arr = MlxArray::from_slice_i32(&token_values, &[batch]);

    let n_kv_per_request = match &states[0].driver.mode {
        Qwen35StepMode::Cpp(state) => {
            i32::try_from(state.kv_flat.len()).context("decode_qwen35_batch kv count overflow")?
        }
        Qwen35StepMode::Rust(_) => unreachable!("checked above"),
    };
    let n_gdr_per_request = match &states[0].driver.mode {
        Qwen35StepMode::Cpp(state) => {
            i32::try_from(state.gdr_flat.len()).context("decode_qwen35_batch gdr count overflow")?
        }
        Qwen35StepMode::Rust(_) => unreachable!("checked above"),
    };

    let mut flat_kv = Vec::with_capacity(states.len() * n_kv_per_request as usize);
    let mut flat_gdr = Vec::with_capacity(states.len() * n_gdr_per_request as usize);
    for state in states.iter() {
        match &state.driver.mode {
            Qwen35StepMode::Cpp(cpp) => {
                flat_kv.extend(cpp.kv_flat.iter().cloned());
                flat_gdr.extend(cpp.gdr_flat.iter().cloned());
            }
            Qwen35StepMode::Rust(_) => unreachable!("checked above"),
        }
    }

    // Same MLX 0.31.1 scalar-rope `[B>1, H, S=1, D]` bug workaround as
    // `decode_qwen35_packed_batch`: always feed a per-row rope offsets
    // array. This is a same-length batch so every row shares `cache_len`,
    // but we still need the array path to stay correct for B > 1.
    let rope_offsets_data: Vec<i32> = vec![cache_len; states.len()];
    let rope_offsets = MlxArray::from_slice_i32(&rope_offsets_data, &[batch]);

    let logits = cpp_model.step_batch(
        &token_arr,
        batch,
        cache_len,
        &mut flat_kv,
        n_kv_per_request,
        &mut flat_gdr,
        n_gdr_per_request,
        None,
        Some(&rope_offsets),
    )?;

    let mut step_outputs: Vec<&MlxArray> = Vec::with_capacity(1 + flat_kv.len() + flat_gdr.len());
    step_outputs.push(&logits);
    step_outputs.extend(flat_kv.iter());
    step_outputs.extend(flat_gdr.iter());
    eval(&step_outputs);

    let logits_shape = logits.shape().to_vec();
    ensure!(
        !logits_shape.is_empty() && logits_shape[0] == batch,
        "decode_qwen35_batch expected batched logits, got shape {:?}",
        logits_shape
    );

    let mut sampled_arrays = Vec::with_capacity(states.len());
    for row_idx in 0..states.len() {
        let row = i32::try_from(row_idx).context("decode_qwen35_batch row overflow")?;
        let mut start = vec![0; logits_shape.len()];
        let mut end = logits_shape.clone();
        let strides = vec![1; logits_shape.len()];
        start[0] = row;
        end[0] = row + 1;
        let row_logits = slice(&logits, &start, &end, &strides);
        sampled_arrays.push(gpu_sample_token(
            &row_logits,
            &states[row_idx].driver.params,
        ));
    }
    let sample_refs: Vec<&MlxArray> = sampled_arrays.iter().collect();
    eval(&sample_refs);

    let mut kv_iter = flat_kv.into_iter();
    let mut gdr_iter = flat_gdr.into_iter();
    let mut sampled_tokens = Vec::with_capacity(states.len());

    for (state, sampled) in states.iter_mut().zip(sampled_arrays.iter()) {
        match &mut state.driver.mode {
            Qwen35StepMode::Cpp(cpp) => {
                for slot in cpp.kv_flat.iter_mut() {
                    let old = std::mem::replace(
                        slot,
                        kv_iter
                            .next()
                            .context("decode_qwen35_batch missing KV output")?,
                    );
                    drop(old);
                }
                for slot in cpp.gdr_flat.iter_mut() {
                    let old = std::mem::replace(
                        slot,
                        gdr_iter
                            .next()
                            .context("decode_qwen35_batch missing GDR output")?,
                    );
                    drop(old);
                }
            }
            Qwen35StepMode::Rust(_) => unreachable!("checked above"),
        }

        let token = sampled.item_i32() as u32;
        state.driver.cache_len += 1;
        state.record_sampled_token(token)?;
        sampled_tokens.push(token);
    }

    ensure!(
        kv_iter.next().is_none() && gdr_iter.next().is_none(),
        "decode_qwen35_batch produced unexpected extra state outputs"
    );

    Ok(sampled_tokens)
}

fn slice_row(array: &MlxArray, row: i32) -> MlxArray {
    let mut start = vec![0; array.shape().len()];
    let mut end = array.shape().to_vec();
    let strides = vec![1; array.shape().len()];
    start[0] = row;
    end[0] = row + 1;
    slice(array, &start, &end, &strides)
}

fn qwen35_can_batch_sample(states: &[&mut ResumableRequestState<Qwen35StepDriver<'_>>]) -> bool {
    let _ = states;
    // `gpu_sample_token` is still a scalar path on Metal. Do not treat it as a
    // batched sampler until the MLX bridge exposes a real batched sampling
    // kernel/result shape.
    false
}

/// Optional DFlash speculative-decode state attached to a `Qwen3StepDriver`.
///
/// When present, `decode_token` runs full DFlash speculative blocks and
/// buffers the accepted tokens; the scheduler still sees one token per step.
/// The DFlash state OWNS the target model's KV cache (`target_state`) instead
/// of the driver's `k_caches`/`v_caches`, because `dflash_speculative_block`
/// needs `&mut ContiguousKvState` for both target and draft.
struct Qwen3DFlashState {
    runtime: &'static MetalDflashRuntime,
    config: &'static MetalModelConfig,
    /// Target model KV state — owned by DFlash, replaces the driver's k/v caches.
    target_state: dflash::ContiguousKvState,
    /// Draft model KV state — owned, separate from target.
    draft_state: dflash::ContiguousKvState,
    /// Target-layer hidden states from the last verified block. Bootstrapped
    /// during prefill via `qwen3_forward_with_hidden_states`.
    target_hidden: Option<MlxArray>,
    /// Multi-token buffer: accepted tokens from the latest speculative block.
    /// `decode_token` pops from here until empty, then runs a new block.
    token_buffer: VecDeque<u32>,
    /// Which target-model layers to capture hidden states from.
    target_layer_ids: Vec<usize>,
    // ── Metrics accumulators (flushed on request completion) ──
    acceptance_lengths: Vec<usize>,
}

struct Qwen3StepDriver<'a> {
    weights: &'a StandardMetalWeights,
    sample_params: SamplingParams,
    prefill_params: SamplingParams,
    kv_capacity: i32,
    k_caches: Vec<MlxArray>,
    v_caches: Vec<MlxArray>,
    kv_pool: Option<MetalKVPool>,
    cache_len: i32,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    attn_scale: f32,
    rope_base: f32,
    eps: f32,
    /// DFlash speculative decode state. When `Some`, `decode_token` runs
    /// DFlash blocks and the driver's `k_caches`/`v_caches` are empty stubs
    /// (all KV management goes through `dflash.target_state`).
    dflash: Option<Qwen3DFlashState>,
}

impl<'a> Qwen3StepDriver<'a> {
    fn new(
        weights: &'a StandardMetalWeights,
        config: &'a MetalModelConfig,
        params: &SamplingParams,
        use_kv_pool: bool,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        dflash_runtime: Option<(&'static MetalDflashRuntime, &'static MetalModelConfig)>,
    ) -> Result<Self> {
        let n_layers = config.num_hidden_layers;
        let n_heads = config.num_attention_heads as i32;
        let n_kv_heads = config.num_key_value_heads as i32;
        let head_dim = config.head_dim as i32;
        let prefill_len = prompt_tokens.len() as i32;
        let initial_cap =
            ((prefill_len + KV_CACHE_CHUNK - 1) / KV_CACHE_CHUNK + 1) * KV_CACHE_CHUNK;
        let total_tokens_needed = std::cmp::max(
            initial_cap as usize,
            prompt_tokens.len().saturating_add(max_new_tokens),
        );
        let kv_dtype = weights.layers[0].attention_inputs.kv_dtype();
        // When DFlash is enabled, the driver's k/v caches are unused —
        // DFlash owns its own target_state ContiguousKvState. Skip the
        // allocation to avoid wasting memory on empty cache tensors.
        let is_dflash = dflash_runtime.is_some();
        let (k_caches, v_caches) = if is_dflash {
            (Vec::new(), Vec::new())
        } else {
            let cache_shape = [1i32, n_kv_heads, initial_cap, head_dim];
            let k: Vec<MlxArray> = (0..n_layers)
                .map(|_| zeros(&cache_shape, kv_dtype))
                .collect();
            let v: Vec<MlxArray> = (0..n_layers)
                .map(|_| zeros(&cache_shape, kv_dtype))
                .collect();
            (k, v)
        };

        Ok(Self {
            weights,
            sample_params: params.clone(),
            prefill_params: SamplingParams {
                temperature: 0.0,
                top_k: 1,
                top_p: 1.0,
                min_p: 0.0,
                repetition_penalty: 1.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                ignore_eos: params.ignore_eos,
                stop_token_ids: params.stop_token_ids.clone(),
                seed: None,
                max_new_tokens: None,
            },
            kv_capacity: initial_cap,
            k_caches,
            v_caches,
            kv_pool: if use_kv_pool {
                Some(
                    MetalKVPool::new(
                        n_layers,
                        n_kv_heads as usize,
                        head_dim as usize,
                        total_tokens_needed,
                        kv_dtype,
                    )
                    .context("pre-alloc MetalKVPool for request state")?,
                )
            } else {
                None
            },
            cache_len: 0,
            n_heads,
            n_kv_heads,
            head_dim,
            attn_scale: 1.0f32 / (head_dim as f32).sqrt(),
            rope_base: config.rope_theta as f32,
            eps: config.rms_norm_eps as f32,
            dflash: match dflash_runtime {
                Some((runtime, static_config)) => {
                    let total_cap = prompt_tokens.len() + max_new_tokens;
                    let target_state = dflash::ContiguousKvState::from_dtype(
                        n_layers, n_kv_heads, head_dim, total_cap, kv_dtype,
                    );
                    let draft_state = dflash::ContiguousKvState::new(
                        runtime.draft_num_hidden_layers(),
                        runtime.draft_n_kv_heads(),
                        runtime.draft_head_dim(),
                        total_cap,
                    );
                    Some(Qwen3DFlashState {
                        runtime,
                        config: static_config,
                        target_state,
                        draft_state,
                        target_hidden: None,
                        token_buffer: VecDeque::new(),
                        target_layer_ids: runtime.target_layer_ids().to_vec(),
                        acceptance_lengths: Vec::new(),
                    })
                }
                None => None,
            },
        })
    }

    fn run_step(&mut self, token: u32, params: &SamplingParams) -> Result<MlxArray> {
        self.ensure_capacity(self.cache_len + 1);
        let sampled = build_forward_graph(
            &[token],
            self.weights,
            &mut self.k_caches,
            &mut self.v_caches,
            self.cache_len,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.attn_scale,
            self.rope_base,
            self.eps,
            self.kv_pool.as_mut(),
            METAL_REQUEST_STATE_ID,
            params,
        )?;
        self.cache_len += 1;
        Ok(sampled)
    }

    fn ensure_capacity(&mut self, needed_tokens: i32) {
        if self.kv_pool.is_some() {
            return;
        }
        while needed_tokens > self.kv_capacity {
            let new_cap = self.kv_capacity + KV_CACHE_CHUNK;
            for li in 0..self.k_caches.len() {
                extend_kv_cache(
                    &mut self.k_caches[li],
                    self.n_kv_heads,
                    self.head_dim,
                    new_cap,
                );
                extend_kv_cache(
                    &mut self.v_caches[li],
                    self.n_kv_heads,
                    self.head_dim,
                    new_cap,
                );
            }
            self.kv_capacity = new_cap;
        }
    }

    fn import_prefix_from_pool(
        &mut self,
        shared_pool: &MetalKVPool,
        slot_indices: &[u32],
    ) -> Result<()> {
        use super::mlx::{reshape, slice_update, transpose_axes};

        if slot_indices.is_empty() {
            return Ok(());
        }
        ensure!(
            self.cache_len == 0,
            "Qwen3 prefix import requires an empty cache"
        );
        ensure!(
            shared_pool.num_layers() == self.k_caches.len()
                && shared_pool.num_kv_heads() == self.n_kv_heads as usize
                && shared_pool.head_dim() == self.head_dim as usize,
            "Qwen3 prefix import requires matching KV geometry"
        );

        let prefix_len =
            i32::try_from(slot_indices.len()).context("Qwen3 prefix import length overflow")?;
        self.ensure_capacity(prefix_len);
        if let Some(pool) = self.kv_pool.as_mut() {
            pool.alloc_tokens(METAL_REQUEST_STATE_ID, slot_indices.len())
                .context("alloc request-local MetalKVPool slots for imported prefix")?;
        }

        for layer_idx in 0..self.k_caches.len() {
            let (k_rows, v_rows) = shared_pool
                .gather_kv_rows(layer_idx, slot_indices)
                .with_context(|| {
                    format!("gather shared Metal prefix rows for layer {layer_idx}")
                })?;

            if let Some(pool) = self.kv_pool.as_mut() {
                pool.write_kv(layer_idx, METAL_REQUEST_STATE_ID, &k_rows, &v_rows)
                    .with_context(|| {
                        format!(
                            "write imported Qwen3 prefix into request-local pool layer {layer_idx}"
                        )
                    })?;
            } else {
                let k = reshape(&k_rows, &[1, prefix_len, self.n_kv_heads, self.head_dim]);
                let k = transpose_axes(&k, &[0, 2, 1, 3]);
                let v = reshape(&v_rows, &[1, prefix_len, self.n_kv_heads, self.head_dim]);
                let v = transpose_axes(&v, &[0, 2, 1, 3]);
                self.k_caches[layer_idx] = slice_update(
                    &mut self.k_caches[layer_idx],
                    &k,
                    &[0, 0, 0, 0],
                    &[1, self.n_kv_heads, prefix_len, self.head_dim],
                );
                self.v_caches[layer_idx] = slice_update(
                    &mut self.v_caches[layer_idx],
                    &v,
                    &[0, 0, 0, 0],
                    &[1, self.n_kv_heads, prefix_len, self.head_dim],
                );
            }
        }

        self.cache_len = prefix_len;
        Ok(())
    }

    fn export_prefix_rows(&self, token_count: usize) -> Result<(Vec<MlxArray>, Vec<MlxArray>)> {
        use super::mlx::{reshape, slice, transpose_axes};

        ensure!(
            token_count > 0,
            "Qwen3 prefix export requires at least one token"
        );
        ensure!(
            token_count <= self.cache_len as usize,
            "Qwen3 prefix export requested {} tokens but cache_len is {}",
            token_count,
            self.cache_len
        );

        let prefix_len =
            i32::try_from(token_count).context("Qwen3 prefix export length overflow")?;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let mut k_rows_by_layer = Vec::with_capacity(self.k_caches.len());
        let mut v_rows_by_layer = Vec::with_capacity(self.v_caches.len());

        for layer_idx in 0..self.k_caches.len() {
            let (k_rows, v_rows) = if let Some(pool) = self.kv_pool.as_ref() {
                let request_slots = pool
                    .token_indices(METAL_REQUEST_STATE_ID)
                    .context("Qwen3 prefix export missing request-local MetalKVPool slots")?;
                ensure!(
                    token_count <= request_slots.len(),
                    "Qwen3 prefix export requested {} tokens but pool only tracks {} slots",
                    token_count,
                    request_slots.len()
                );
                pool.gather_kv_rows(layer_idx, &request_slots[..token_count])
                    .with_context(|| {
                        format!("gather request-local Qwen3 prefix rows for layer {layer_idx}")
                    })?
            } else {
                let k = slice(
                    &self.k_caches[layer_idx],
                    &[0, 0, 0, 0],
                    &[1, self.n_kv_heads, prefix_len, self.head_dim],
                    &[1, 1, 1, 1],
                );
                let k = transpose_axes(&k, &[0, 2, 1, 3]);
                let k = reshape(&k, &[prefix_len, kv_dim]);
                let v = slice(
                    &self.v_caches[layer_idx],
                    &[0, 0, 0, 0],
                    &[1, self.n_kv_heads, prefix_len, self.head_dim],
                    &[1, 1, 1, 1],
                );
                let v = transpose_axes(&v, &[0, 2, 1, 3]);
                let v = reshape(&v, &[prefix_len, kv_dim]);
                (k, v)
            };
            k_rows_by_layer.push(k_rows);
            v_rows_by_layer.push(v_rows);
        }

        Ok((k_rows_by_layer, v_rows_by_layer))
    }
}

impl StepDriver for Qwen3StepDriver<'_> {
    fn prefill_token(&mut self, token: u32, terminal_prompt: bool) -> Result<Option<u32>> {
        let params = if terminal_prompt {
            self.sample_params.clone()
        } else {
            self.prefill_params.clone()
        };
        let sampled = self.run_step(token, &params)?;
        eval(&[&sampled]);
        if terminal_prompt {
            Ok(Some(sampled.item_i32() as u32))
        } else {
            Ok(None)
        }
    }

    fn prefill_tokens(&mut self, tokens: &[u32], terminal_prompt: bool) -> Result<Option<u32>> {
        // DFlash path: run the full prompt through qwen3_forward_with_hidden_states
        // on the terminal chunk to capture target-layer hidden states for the
        // first speculative block. The budget override in execute_prefill_chunk
        // ensures the entire prompt arrives as one terminal chunk.
        if terminal_prompt && self.dflash.is_some() {
            let dflash = self.dflash.as_mut().unwrap();
            let (norm_hidden, target_hidden) = dflash::qwen3_forward_with_hidden_states_on_state(
                tokens,
                self.weights,
                dflash.config,
                &dflash.target_layer_ids,
                &mut dflash.target_state,
            )?;
            dflash.target_hidden = Some(target_hidden);
            let logits = super::ops::linear(&norm_hidden, &self.weights.lm_head);
            let sampled = gpu_sample_token(&logits, &self.sample_params);
            eval(&[&sampled]);
            return Ok(Some(sampled.item_i32() as u32));
        }

        // Default: per-token prefill
        let mut emitted = None;
        for (idx, &token) in tokens.iter().enumerate() {
            let is_terminal = terminal_prompt && idx + 1 == tokens.len();
            let sampled = self.prefill_token(token, is_terminal)?;
            if is_terminal {
                emitted = sampled;
            } else if sampled.is_some() {
                bail!("non-terminal prefill step unexpectedly emitted a sampled token");
            }
        }
        Ok(emitted)
    }

    fn decode_token(&mut self, token: u32) -> Result<u32> {
        // ── DFlash speculative path ──────────────────────────────────────
        if let Some(dflash) = self.dflash.as_mut() {
            // 1. Drain buffer first — cheap, no GPU work.
            if let Some(buffered) = dflash.token_buffer.pop_front() {
                return Ok(buffered);
            }

            // 2. Buffer empty → run one full speculative block.
            let target_hidden = dflash
                .target_hidden
                .take()
                .context("DFlash decode_token: target_hidden not set (prefill incomplete?)")?;

            let block = dflash::dflash_speculative_block(
                dflash.runtime,
                token,
                &target_hidden,
                self.weights,
                dflash.config,
                &self.sample_params,
                &mut dflash.target_state,
                &mut dflash.draft_state,
            )?;

            // 3. Update state.
            dflash.acceptance_lengths.push(block.accepted_inputs);
            dflash.target_hidden = Some(block.updated_target_hidden);

            // 4. Push accepted tokens into buffer, pop the first one.
            for &t in &block.accepted_tokens {
                dflash.token_buffer.push_back(t);
            }
            return Ok(dflash
                .token_buffer
                .pop_front()
                .context("DFlash speculative block produced zero tokens")?);
        }

        // ── Standard single-token path ───────────────────────────────────
        if self.cache_len > 0 && self.cache_len % KV_CACHE_CHUNK == 0 {
            clear_metal_cache();
        }
        let sampled = self.run_step(token, &self.sample_params.clone())?;
        eval(&[&sampled]);
        Ok(sampled.item_i32() as u32)
    }

    fn cleanup(&mut self) -> Result<()> {
        if let Some(pool) = self.kv_pool.as_mut() {
            pool.free_request(METAL_REQUEST_STATE_ID);
        }
        Ok(())
    }
}

/// Innovation tape from one GDR layer during a speculative verify block.
/// Used for O(accepted) rollback instead of O(N) re-forward.
struct GdrTape {
    /// Innovation delta at each timestep: [1, N, Hv, Dv]
    innovation_tape: MlxArray,
    /// Key projections: [1, N, Hk, Dk]
    k: MlxArray,
    /// Gating values: [1, N, Hv]
    g: MlxArray,
    /// Raw QKV for conv state rebuild: [1, N, qkv_dim]
    qkv: MlxArray,
}

/// DFlash speculative decode state for Qwen3.5 hybrid models.
/// Extends the Qwen3 DFlash pattern with GDR recurrent rollback.
struct Qwen35DFlashState {
    runtime: &'static MetalDflashRuntime,
    config: &'static MetalModelConfig,
    /// Draft model KV state (pure transformer, same as Qwen3 DFlash).
    draft_state: dflash::ContiguousKvState,
    /// Target-layer hidden states for the next draft block.
    target_hidden: Option<MlxArray>,
    /// Multi-token buffer from speculative acceptance.
    token_buffer: VecDeque<u32>,
    /// Which target-model layers to capture hidden states from.
    target_layer_ids: Vec<usize>,
    /// Per-block acceptance lengths for metrics.
    acceptance_lengths: Vec<usize>,
    /// GDR state snapshot taken before each verify (for rollback).
    gdr_snapshot: Option<Vec<MlxArray>>,
    /// Per-layer tapes recorded during verify (for partial replay).
    gdr_tapes: Vec<GdrTape>,
}

struct Qwen35CppState {
    kv_flat: Vec<MlxArray>,
    gdr_flat: Vec<MlxArray>,
}

struct Qwen35RustState {
    k_caches: Vec<MlxArray>,
    v_caches: Vec<MlxArray>,
    recurrent: MetalRecurrentState,
}

enum Qwen35StepMode {
    Cpp(Qwen35CppState),
    Rust(Qwen35RustState),
}

struct Qwen35StepDriver<'a> {
    weights: &'a Qwen35MetalWeights,
    config: &'a MetalModelConfig,
    arch: &'a super::config::MetalQwen35ArchConfig,
    params: SamplingParams,
    kv_capacity: i32,
    cache_len: i32,
    mode: Qwen35StepMode,
    /// DFlash speculative decode state (None = standard decode).
    dflash: Option<Qwen35DFlashState>,
}

impl<'a> Qwen35StepDriver<'a> {
    fn new(
        weights: &'a Qwen35MetalWeights,
        config: &'a MetalModelConfig,
        params: &SamplingParams,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        dflash_runtime: Option<(&'static MetalDflashRuntime, &'static MetalModelConfig)>,
    ) -> Result<Self> {
        let MetalModelArch::Qwen35(arch) = &config.arch else {
            bail!("Qwen3.5 request state requires a Qwen3.5 config");
        };

        let num_full_layers = arch.num_full_attention_layers();
        let prefill_len = prompt_tokens.len() as i32;
        let total_tokens_needed =
            std::cmp::max(1, prompt_tokens.len().saturating_add(max_new_tokens));
        let initial_cap =
            ((total_tokens_needed as i32 + KV_CACHE_CHUNK - 1) / KV_CACHE_CHUNK) * KV_CACHE_CHUNK;
        let cache_shape = [
            1i32,
            config.num_key_value_heads as i32,
            initial_cap,
            config.head_dim as i32,
        ];
        let k_caches: Vec<MlxArray> = (0..num_full_layers)
            .map(|_| zeros(&cache_shape, super::mlx::Dtype::Bfloat16))
            .collect();
        let v_caches: Vec<MlxArray> = (0..num_full_layers)
            .map(|_| zeros(&cache_shape, super::mlx::Dtype::Bfloat16))
            .collect();
        let recurrent = MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);

        let mode = if weights.cpp_model.is_some() {
            let kv_flat: Vec<MlxArray> = k_caches
                .iter()
                .zip(v_caches.iter())
                .flat_map(|(k, v)| [k.clone(), v.clone()])
                .collect();
            let gdr_flat: Vec<MlxArray> = recurrent
                .states
                .iter()
                .zip(recurrent.conv_states.iter())
                .flat_map(|(s, c)| [s.clone(), c.clone()])
                .collect();
            Qwen35StepMode::Cpp(Qwen35CppState { kv_flat, gdr_flat })
        } else {
            Qwen35StepMode::Rust(Qwen35RustState {
                k_caches,
                v_caches,
                recurrent,
            })
        };

        let _ = prefill_len;

        let dflash = if let Some((runtime, dflash_config)) = dflash_runtime {
            Some(Qwen35DFlashState {
                runtime,
                config: dflash_config,
                draft_state: dflash::ContiguousKvState::new(
                    runtime.draft_num_hidden_layers(),
                    runtime.draft_n_kv_heads(),
                    runtime.draft_head_dim(),
                    total_tokens_needed,
                ),
                target_hidden: None,
                token_buffer: VecDeque::new(),
                target_layer_ids: runtime.target_layer_ids().to_vec(),
                acceptance_lengths: Vec::new(),
                gdr_snapshot: None,
                gdr_tapes: Vec::new(),
            })
        } else {
            None
        };

        Ok(Self {
            weights,
            config,
            arch,
            params: params.clone(),
            kv_capacity: initial_cap,
            cache_len: 0,
            mode,
            dflash,
        })
    }

    fn run_step(&mut self, token: u32) -> Result<MlxArray> {
        self.ensure_capacity(self.cache_len + 1);
        let token_arr = MlxArray::from_slice_i32(&[token as i32], &[1]);
        let weights = self.weights;
        let config = self.config;
        let arch = self.arch;
        let cache_len = self.cache_len;
        let logits = match &mut self.mode {
            Qwen35StepMode::Cpp(state) => {
                Self::run_cpp_step(weights, cache_len, &token_arr, state)?
            }
            Qwen35StepMode::Rust(state) => {
                Self::run_rust_step(weights, config, arch, cache_len, &token_arr, state)
            }
        };
        self.cache_len += 1;
        Ok(logits)
    }

    fn run_cpp_step(
        weights: &Qwen35MetalWeights,
        cache_len: i32,
        token_arr: &MlxArray,
        state: &mut Qwen35CppState,
    ) -> Result<MlxArray> {
        let cpp_model: &CppQwen35Model = weights
            .cpp_model
            .as_ref()
            .context("Qwen3.5 C++ step path missing compiled model")?;
        let logits = cpp_model.step(
            token_arr,
            cache_len,
            &mut state.kv_flat,
            &mut state.gdr_flat,
        )?;
        let mut step_outputs: Vec<&MlxArray> =
            Vec::with_capacity(1 + state.kv_flat.len() + state.gdr_flat.len());
        step_outputs.push(&logits);
        step_outputs.extend(state.kv_flat.iter());
        step_outputs.extend(state.gdr_flat.iter());
        eval(&step_outputs);
        Ok(logits)
    }

    fn run_rust_step(
        weights: &Qwen35MetalWeights,
        config: &MetalModelConfig,
        arch: &super::config::MetalQwen35ArchConfig,
        cache_len: i32,
        token_arr: &MlxArray,
        state: &mut Qwen35RustState,
    ) -> MlxArray {
        let logits = qwen35_forward_step(
            token_arr,
            weights,
            config,
            arch,
            &mut state.k_caches,
            &mut state.v_caches,
            &mut state.recurrent,
            cache_len,
        );

        let mut step_outputs: Vec<&MlxArray> = Vec::with_capacity(
            1 + state.k_caches.len()
                + state.v_caches.len()
                + state.recurrent.states.len()
                + state.recurrent.conv_states.len(),
        );
        step_outputs.push(&logits);
        step_outputs.extend(state.k_caches.iter());
        step_outputs.extend(state.v_caches.iter());
        step_outputs.extend(state.recurrent.states.iter());
        step_outputs.extend(state.recurrent.conv_states.iter());
        eval(&step_outputs);

        state.recurrent.seq_len = (cache_len + 1) as usize;
        logits
    }

    fn ensure_capacity(&mut self, needed_tokens: i32) {
        while needed_tokens > self.kv_capacity {
            let new_cap = self.kv_capacity + KV_CACHE_CHUNK;
            match &mut self.mode {
                Qwen35StepMode::Cpp(state) => {
                    for li in 0..(state.kv_flat.len() / 2) {
                        extend_kv_cache(
                            &mut state.kv_flat[2 * li],
                            self.config.num_key_value_heads as i32,
                            self.config.head_dim as i32,
                            new_cap,
                        );
                        extend_kv_cache(
                            &mut state.kv_flat[2 * li + 1],
                            self.config.num_key_value_heads as i32,
                            self.config.head_dim as i32,
                            new_cap,
                        );
                    }
                }
                Qwen35StepMode::Rust(state) => {
                    for li in 0..state.k_caches.len() {
                        extend_kv_cache(
                            &mut state.k_caches[li],
                            self.config.num_key_value_heads as i32,
                            self.config.head_dim as i32,
                            new_cap,
                        );
                        extend_kv_cache(
                            &mut state.v_caches[li],
                            self.config.num_key_value_heads as i32,
                            self.config.head_dim as i32,
                            new_cap,
                        );
                    }
                }
            }
            self.kv_capacity = new_cap;
        }
    }

    fn import_prefix_snapshot(&mut self, snapshot: &Qwen35PrefixSnapshot) -> Result<()> {
        ensure!(
            snapshot.cache_len > 0,
            "Qwen3.5 prefix import requires a non-empty snapshot"
        );
        ensure!(
            snapshot.kv_capacity >= snapshot.cache_len,
            "Qwen3.5 prefix snapshot capacity {} is smaller than cache_len {}",
            snapshot.kv_capacity,
            snapshot.cache_len
        );
        match &mut self.mode {
            Qwen35StepMode::Cpp(state) => {
                state.kv_flat = snapshot.kv_flat.clone();
                state.gdr_flat = snapshot.gdr_flat.clone();
                self.kv_capacity = snapshot.kv_capacity;
                self.cache_len = snapshot.cache_len;
                Ok(())
            }
            Qwen35StepMode::Rust(_) => {
                bail!("Qwen3.5 live prefix reuse currently requires the compiled C++ step path")
            }
        }
    }

    fn export_current_cpp_snapshot(&self, token_ids: Vec<u32>) -> Result<Qwen35PrefixSnapshot> {
        let cache_len = self.cache_len;
        ensure!(
            cache_len > 0,
            "Qwen3.5 prefix export requires a non-empty cache"
        );
        match &self.mode {
            Qwen35StepMode::Cpp(state) => Ok(Qwen35PrefixSnapshot {
                token_ids,
                kv_flat: state.kv_flat.clone(),
                gdr_flat: state.gdr_flat.clone(),
                cache_len,
                kv_capacity: self.kv_capacity,
            }),
            Qwen35StepMode::Rust(_) => {
                bail!("Qwen3.5 live prefix export currently requires the compiled C++ step path")
            }
        }
    }

    fn build_prefix_snapshots(
        &self,
        prompt_tokens: &[u32],
        block_size: usize,
    ) -> Result<Vec<Qwen35PrefixSnapshot>> {
        ensure!(
            block_size > 0,
            "Qwen3.5 prefix snapshot block size must be > 0"
        );
        ensure!(
            prompt_tokens.len().is_multiple_of(block_size),
            "Qwen3.5 prefix snapshot build requires a block-aligned prompt"
        );
        if !matches!(self.mode, Qwen35StepMode::Cpp(_)) || prompt_tokens.is_empty() {
            return Ok(Vec::new());
        }

        let mut replay = Qwen35StepDriver::new(
            self.weights,
            self.config,
            &self.params,
            prompt_tokens,
            1,
            None,
        )
        .context("build replay driver for Qwen3.5 prefix snapshots")?;
        let mut snapshots = Vec::with_capacity(prompt_tokens.len() / block_size);

        for chunk in prompt_tokens.chunks(block_size) {
            replay
                .prefill_tokens(chunk, false)
                .context("replay Qwen3.5 prompt chunk for prefix snapshot")?;
            let materialized = replay.cache_len as usize;
            snapshots.push(
                replay
                    .export_current_cpp_snapshot(prompt_tokens[..materialized].to_vec())
                    .context("export replayed Qwen3.5 prefix snapshot")?,
            );
        }

        Ok(snapshots)
    }
}

impl StepDriver for Qwen35StepDriver<'_> {
    fn prefill_token(&mut self, token: u32, terminal_prompt: bool) -> Result<Option<u32>> {
        let logits = self.run_step(token)?;
        if terminal_prompt {
            if let Some(ref mut dflash) = self.dflash {
                if dflash.target_hidden.is_none() {
                    if let MetalModelArch::Qwen35(arch) = &self.config.arch {
                        let num_full = arch.num_full_attention_layers();
                        let cs = [
                            1i32,
                            self.config.num_key_value_heads as i32,
                            KV_CACHE_CHUNK,
                            self.config.head_dim as i32,
                        ];
                        let mut tk: Vec<MlxArray> = (0..num_full)
                            .map(|_| zeros(&cs, super::mlx::Dtype::Bfloat16))
                            .collect();
                        let mut tv: Vec<MlxArray> = (0..num_full)
                            .map(|_| zeros(&cs, super::mlx::Dtype::Bfloat16))
                            .collect();
                        let mut tr = MetalRecurrentState::new(
                            arch.num_linear_attention_layers(),
                            &arch.linear,
                        );
                        let (_, th) = super::qwen35::qwen35_forward_with_hidden_states(
                            &[token],
                            self.weights,
                            self.config,
                            arch,
                            &mut tk,
                            &mut tv,
                            &mut tr,
                            self.cache_len - 1,
                            &dflash.target_layer_ids,
                        );
                        eval(&[&th]);
                        dflash.target_hidden = Some(th);
                    }
                }
            }
            let sampled = gpu_sample_token(&logits, &self.params);
            eval(&[&sampled]);
            Ok(Some(sampled.item_i32() as u32))
        } else {
            Ok(None)
        }
    }

    fn prefill_tokens(&mut self, tokens: &[u32], terminal_prompt: bool) -> Result<Option<u32>> {
        if tokens.is_empty() {
            return Ok(None);
        }

        let use_cpp_batch_prefill = matches!(self.mode, Qwen35StepMode::Cpp(_)) && tokens.len() > 1;
        if use_cpp_batch_prefill {
            self.ensure_capacity(self.cache_len + tokens.len() as i32);
        }

        match &mut self.mode {
            Qwen35StepMode::Cpp(state) if tokens.len() > 1 => {
                let token_values: Vec<i32> = tokens.iter().map(|&token| token as i32).collect();
                let token_arr = MlxArray::from_slice_i32(&token_values, &[tokens.len() as i32]);
                let cpp_model: &CppQwen35Model = self
                    .weights
                    .cpp_model
                    .as_ref()
                    .context("Qwen3.5 C++ prefill path missing compiled model")?;
                let logits = cpp_model.prefill(
                    &token_arr,
                    tokens.len() as i32,
                    self.cache_len,
                    &mut state.kv_flat,
                    &mut state.gdr_flat,
                )?;
                let mut step_outputs: Vec<&MlxArray> =
                    Vec::with_capacity(1 + state.kv_flat.len() + state.gdr_flat.len());
                step_outputs.push(&logits);
                step_outputs.extend(state.kv_flat.iter());
                step_outputs.extend(state.gdr_flat.iter());
                eval(&step_outputs);
                self.cache_len += tokens.len() as i32;

                if terminal_prompt {
                    if let Some(ref mut dflash) = self.dflash {
                        if dflash.target_hidden.is_none() {
                            if let MetalModelArch::Qwen35(arch) = &self.config.arch {
                                let num_full = arch.num_full_attention_layers();
                                let cs = [
                                    1i32,
                                    self.config.num_key_value_heads as i32,
                                    (tokens.len() as i32 + KV_CACHE_CHUNK) / KV_CACHE_CHUNK
                                        * KV_CACHE_CHUNK,
                                    self.config.head_dim as i32,
                                ];
                                let mut tk: Vec<MlxArray> = (0..num_full)
                                    .map(|_| zeros(&cs, super::mlx::Dtype::Bfloat16))
                                    .collect();
                                let mut tv: Vec<MlxArray> = (0..num_full)
                                    .map(|_| zeros(&cs, super::mlx::Dtype::Bfloat16))
                                    .collect();
                                let mut tr = MetalRecurrentState::new(
                                    arch.num_linear_attention_layers(),
                                    &arch.linear,
                                );
                                let (_, th) = super::qwen35::qwen35_forward_with_hidden_states(
                                    tokens,
                                    self.weights,
                                    self.config,
                                    arch,
                                    &mut tk,
                                    &mut tv,
                                    &mut tr,
                                    0,
                                    &dflash.target_layer_ids,
                                );
                                eval(&[&th]);
                                dflash.target_hidden = Some(th);
                            }
                        }
                    }
                    let sampled = gpu_sample_token(&logits, &self.params);
                    eval(&[&sampled]);
                    Ok(Some(sampled.item_i32() as u32))
                } else {
                    Ok(None)
                }
            }
            _ => {
                let mut emitted = None;
                for (idx, &token) in tokens.iter().enumerate() {
                    let is_terminal = terminal_prompt && idx + 1 == tokens.len();
                    let sampled = self.prefill_token(token, is_terminal)?;
                    if is_terminal {
                        emitted = sampled;
                    } else if sampled.is_some() {
                        bail!("non-terminal prefill step unexpectedly emitted a sampled token");
                    }
                }
                Ok(emitted)
            }
        }
    }

    fn decode_token(&mut self, token: u32) -> Result<u32> {
        // ── DFlash speculative path (Qwen3.5) ────────────────────────────
        if let Some(dflash) = self.dflash.as_mut() {
            // 1. Drain buffer first — cheap, no GPU work.
            if let Some(buffered) = dflash.token_buffer.pop_front() {
                return Ok(buffered);
            }

            // 2. Buffer empty → run one full speculative block.
            if let Qwen35StepMode::Cpp(ref mut cpp_state) = self.mode {
                let Some(target_hidden) = dflash.target_hidden.take() else {
                    // First decode after prefill — target_hidden not captured yet.
                    // Fall through to standard decode.
                    drop(dflash);
                    let logits = self.run_step(token)?;
                    let sampled = gpu_sample_token(&logits, &self.params);
                    eval(&[&sampled]);
                    return Ok(sampled.item_i32() as u32);
                };

                let cpp_model = self
                    .weights
                    .cpp_model
                    .as_ref()
                    .context("Qwen3.5 DFlash requires C++ compiled model")?;

                let block = dflash::qwen35_dflash_speculative_block(
                    dflash.runtime,
                    token,
                    &target_hidden,
                    &self.weights.embed_tokens,
                    &self.weights.lm_head,
                    dflash.config,
                    cpp_model,
                    &self.params,
                    &mut cpp_state.kv_flat,
                    &mut cpp_state.gdr_flat,
                    &mut self.cache_len,
                    &mut dflash.draft_state,
                )?;

                dflash.acceptance_lengths.push(block.accepted_inputs);
                dflash.target_hidden = Some(block.updated_target_hidden);

                for &t in &block.accepted_tokens {
                    dflash.token_buffer.push_back(t);
                }
                return Ok(dflash
                    .token_buffer
                    .pop_front()
                    .context("Qwen3.5 DFlash block produced zero tokens")?);
            }
            // Rust mode fallback: fall through to standard decode
        }

        // ── Standard single-token decode ─────────────────────────────────
        if self.cache_len > 0 && self.cache_len % KV_CACHE_CHUNK == 0 {
            clear_metal_cache();
        }
        let logits = self.run_step(token)?;
        let sampled = gpu_sample_token(&logits, &self.params);
        eval(&[&sampled]);
        Ok(sampled.item_i32() as u32)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct FakeDriver {
        prefill_outputs: VecDeque<Option<u32>>,
        decode_outputs: VecDeque<u32>,
        cleanup_calls: Arc<AtomicUsize>,
    }

    impl FakeDriver {
        fn new(
            prefill_outputs: impl IntoIterator<Item = Option<u32>>,
            decode_outputs: impl IntoIterator<Item = u32>,
            cleanup_calls: Arc<AtomicUsize>,
        ) -> Self {
            Self {
                prefill_outputs: prefill_outputs.into_iter().collect(),
                decode_outputs: decode_outputs.into_iter().collect(),
                cleanup_calls,
            }
        }
    }

    impl StepDriver for FakeDriver {
        fn prefill_token(&mut self, _token: u32, _terminal_prompt: bool) -> Result<Option<u32>> {
            self.prefill_outputs
                .pop_front()
                .context("missing fake prefill output")
        }

        fn decode_token(&mut self, _token: u32) -> Result<u32> {
            self.decode_outputs
                .pop_front()
                .context("missing fake decode output")
        }

        fn cleanup(&mut self) -> Result<()> {
            self.cleanup_calls.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    #[test]
    fn prefill_chunk_only_emits_a_token_once_the_prompt_finishes() {
        let cleanup_calls = Arc::new(AtomicUsize::new(0));
        let mut state = ResumableRequestState::new(
            FakeDriver::new([None, None, None, Some(41)], [], cleanup_calls.clone()),
            vec![1, 2, 3, 4],
            3,
            vec![],
            99,
            false,
        )
        .expect("state");

        let first = state.prefill_chunk(2).expect("chunk 1");
        assert_eq!(
            first,
            PrefillChunkResult {
                processed_tokens: 2,
                emitted_token: None,
                phase: MetalRequestPhase::Prefill,
                finish_reason: None,
            }
        );
        assert_eq!(state.prompt_progress(), 2);
        assert_eq!(state.generated_tokens(), 0);
        assert_eq!(cleanup_calls.load(Ordering::Relaxed), 0);

        let second = state.prefill_chunk(2).expect("chunk 2");
        assert_eq!(second.processed_tokens, 2);
        assert_eq!(second.emitted_token, Some(41));
        assert_eq!(second.phase, MetalRequestPhase::Decode);
        assert_eq!(state.prompt_progress(), 4);
        assert_eq!(state.generated_tokens(), 1);
        assert_eq!(cleanup_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn length_finish_cleans_up_after_decode_step() {
        let cleanup_calls = Arc::new(AtomicUsize::new(0));
        let mut state = ResumableRequestState::new(
            FakeDriver::new([Some(7)], [8], cleanup_calls.clone()),
            vec![1],
            2,
            vec![],
            99,
            false,
        )
        .expect("state");

        let prefill = state.prefill_chunk(1).expect("prefill");
        assert_eq!(prefill.emitted_token, Some(7));
        assert_eq!(state.phase(), MetalRequestPhase::Decode);
        assert_eq!(state.generated_tokens(), 1);
        assert_eq!(cleanup_calls.load(Ordering::Relaxed), 0);

        let decoded = state.decode_step().expect("decode");
        assert_eq!(decoded, Some(8));
        assert_eq!(state.phase(), MetalRequestPhase::Finished);
        assert_eq!(state.finish_reason(), Some("length"));
        assert_eq!(state.generated_tokens(), 2);
        assert_eq!(cleanup_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn stop_token_from_prefill_finishes_immediately() {
        let cleanup_calls = Arc::new(AtomicUsize::new(0));
        let mut state = ResumableRequestState::new(
            FakeDriver::new([Some(42)], [], cleanup_calls.clone()),
            vec![5],
            4,
            vec![42],
            99,
            false,
        )
        .expect("state");

        let prefill = state.prefill_chunk(1).expect("prefill");
        assert_eq!(prefill.emitted_token, Some(42));
        assert_eq!(prefill.phase, MetalRequestPhase::Finished);
        assert_eq!(prefill.finish_reason, Some("stop"));
        assert_eq!(state.finish_reason(), Some("stop"));
        assert_eq!(cleanup_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn dropping_incomplete_state_still_runs_cleanup_once() {
        let cleanup_calls = Arc::new(AtomicUsize::new(0));
        {
            let state = ResumableRequestState::new(
                FakeDriver::new([None, None], [], cleanup_calls.clone()),
                vec![1, 2],
                4,
                vec![],
                99,
                false,
            )
            .expect("state");
            assert_eq!(state.phase(), MetalRequestPhase::Prefill);
        }

        assert_eq!(cleanup_calls.load(Ordering::Relaxed), 1);
    }
}
