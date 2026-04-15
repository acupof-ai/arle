use anyhow::{Context, Result, bail, ensure};

use super::KV_CACHE_CHUNK;
use super::config::{MetalModelArch, MetalModelConfig};
use super::forward::build_forward_graph;
use super::gdr::MetalRecurrentState;
use super::kv_pool::MetalKVPool;
use super::mlx::{MlxArray, eval, zeros};
use super::ops::{clear_metal_cache, extend_kv_cache};
use super::qwen35::{CppQwen35Model, Qwen35MetalWeights, qwen35_forward_step};
use super::sampling::{gpu_sample_token, validate_metal_sampling_params};
use super::weights::{MetalWeights, StandardMetalWeights};
use crate::sampler::SamplingParams;

const METAL_REQUEST_STATE_ID: usize = 0;

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

        let mut processed = 0usize;
        while processed < budget && self.prompt_cursor < self.prompt_tokens.len() {
            let token = self.prompt_tokens[self.prompt_cursor];
            let terminal_prompt = self.prompt_cursor + 1 == self.prompt_tokens.len();
            let sampled = self.driver.prefill_token(token, terminal_prompt)?;
            self.prompt_cursor += 1;
            processed += 1;

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

            if sampled.is_some() {
                bail!("non-terminal prefill step unexpectedly emitted a sampled token");
            }
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

impl<'a> MetalRequestState<'a> {
    pub(super) fn new(
        weights: &'a MetalWeights,
        config: &'a MetalModelConfig,
        prompt_tokens: Vec<u32>,
        params: &SamplingParams,
        use_kv_pool: bool,
        max_new_tokens: usize,
    ) -> Result<Self> {
        validate_metal_sampling_params(params)?;

        let inner = match weights {
            MetalWeights::Qwen3(weights) => {
                let driver = Qwen3StepDriver::new(
                    weights,
                    config,
                    params,
                    use_kv_pool,
                    &prompt_tokens,
                    max_new_tokens,
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
                let driver =
                    Qwen35StepDriver::new(weights, config, params, &prompt_tokens, max_new_tokens)?;
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

    pub fn cancel(&mut self) -> Result<()> {
        match &mut self.inner {
            MetalRequestStateInner::Qwen3(state) => state.cancel(),
            MetalRequestStateInner::Qwen35(state) => state.cancel(),
        }
    }
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
}

impl<'a> Qwen3StepDriver<'a> {
    fn new(
        weights: &'a StandardMetalWeights,
        config: &'a MetalModelConfig,
        params: &SamplingParams,
        use_kv_pool: bool,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
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
        let cache_shape = [1i32, n_kv_heads, initial_cap, head_dim];
        let k_caches: Vec<MlxArray> = (0..n_layers)
            .map(|_| zeros(&cache_shape, kv_dtype))
            .collect();
        let v_caches: Vec<MlxArray> = (0..n_layers)
            .map(|_| zeros(&cache_shape, kv_dtype))
            .collect();

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

    fn decode_token(&mut self, token: u32) -> Result<u32> {
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
}

impl<'a> Qwen35StepDriver<'a> {
    fn new(
        weights: &'a Qwen35MetalWeights,
        config: &'a MetalModelConfig,
        params: &SamplingParams,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
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

        Ok(Self {
            weights,
            config,
            arch,
            params: params.clone(),
            kv_capacity: initial_cap,
            cache_len: 0,
            mode,
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
}

impl StepDriver for Qwen35StepDriver<'_> {
    fn prefill_token(&mut self, token: u32, terminal_prompt: bool) -> Result<Option<u32>> {
        let logits = self.run_step(token)?;
        if terminal_prompt {
            let sampled = gpu_sample_token(&logits, &self.params);
            eval(&[&sampled]);
            Ok(Some(sampled.item_i32() as u32))
        } else {
            Ok(None)
        }
    }

    fn decode_token(&mut self, token: u32) -> Result<u32> {
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
