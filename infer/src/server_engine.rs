#[cfg(feature = "cuda")]
use std::time::Instant;

use anyhow::Result;
#[cfg(feature = "cuda")]
use fastrace::local::LocalSpan;
#[cfg(feature = "cuda")]
use log::{debug, info};
#[cfg(feature = "cuda")]
use rand::SeedableRng;
#[cfg(feature = "cuda")]
use rand::rngs::StdRng;
use tokio::sync::mpsc::UnboundedSender;

#[cfg(feature = "cuda")]
pub use crate::backend::cuda::bootstrap::{
    EngineOptions, ModelType, ServerRuntimeConfig, detect_model_type, model_id_from_path,
};
#[cfg(feature = "cuda")]
use crate::model::{GLM4Model, GenerationState, ModelForward, Qwen3Model, Qwen35Model};
use crate::sampler::SamplingParams;
#[cfg(feature = "cuda")]
use crate::tokenizer::Tokenizer;

/// Truncate at the first occurrence of any stop string (OpenAI-compatible).
/// Returns the prefix of `text` up to (but not including) the earliest stop.
#[cfg(feature = "cuda")]
fn truncate_at_first_stop(text: &str, stops: &[String]) -> Option<String> {
    let mut earliest = None::<usize>;
    for s in stops {
        let s = s.as_str();
        if s.is_empty() {
            continue;
        }
        if let Some(pos) = text.find(s) {
            earliest = Some(match earliest {
                None => pos,
                Some(e) => std::cmp::min(e, pos),
            });
        }
    }
    earliest.map(|pos| text[..pos].to_string())
}

/// If `new_full` (accumulated text) ends with any of `stops`, return the delta to send
/// (from `sent_len` up to but not including the stop) and the matching stop.
/// Prefers the longest matching stop when several match at the end.
#[cfg(feature = "cuda")]
fn truncate_at_stop<'a>(
    new_full: &str,
    sent_len: usize,
    stops: &[&'a str],
) -> Option<(String, &'a str)> {
    let mut best: Option<(usize, &'a str)> = None;
    for s in stops {
        if s.is_empty() {
            continue;
        }
        if new_full.ends_with(s) {
            let len = s.len();
            if best.is_none_or(|(l, _)| len > l) {
                best = Some((len, *s));
            }
        }
    }
    best.map(|(stop_len, stop)| {
        let end = new_full.len().saturating_sub(stop_len);
        let to_send = if end >= sent_len {
            new_full[sent_len..end].to_string()
        } else {
            String::new()
        };
        (to_send, stop)
    })
}

pub struct CompleteRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    /// Stop generation when output ends with any of these strings (OpenAI-compatible).
    pub stop: Option<Vec<String>>,
    /// Return per-token log-probabilities (greedy sampling only).
    pub logprobs: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    Length,
    Stop,
}

impl FinishReason {
    pub(crate) fn as_openai_str(self) -> &'static str {
        match self {
            Self::Length => "length",
            Self::Stop => "stop",
        }
    }
}

pub struct CompleteOutput {
    pub text: String,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    /// Per-token log-probabilities (greedy only). Empty if logprobs not requested.
    pub token_logprobs: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

pub struct StreamDelta {
    pub text_delta: String,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
    /// Log-probability of the generated token (greedy only, None otherwise).
    #[allow(dead_code)]
    pub logprob: Option<f32>,
}

impl StreamDelta {
    /// Create a text delta (no finish, no logprob).
    pub fn text(s: String) -> Self {
        Self {
            text_delta: s,
            finish_reason: None,
            usage: None,
            logprob: None,
        }
    }
}

pub trait ServerEngine: Send {
    /// Returns the model identifier (e.g. `"Qwen3-8B"`).
    fn model_id(&self) -> &str;

    /// Run a complete generation request synchronously and return the full output.
    fn complete(&mut self, req: CompleteRequest) -> Result<CompleteOutput>;

    /// Run a generation request, streaming token deltas through `tx` as they are produced.
    fn complete_stream(
        &mut self,
        req: CompleteRequest,
        tx: UnboundedSender<StreamDelta>,
    ) -> Result<()>;
}

// ============================================================================
// Shared generation loop — uses ModelForward trait
// ============================================================================

#[cfg(feature = "cuda")]
struct StreamingStats {
    emitted_tokens: usize,
    hit_eos: bool,
    consumer_dropped: bool,
}

#[cfg(feature = "cuda")]
fn generate<M: ModelForward>(
    model: &M,
    state: &mut M::State,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    params: &SamplingParams,
    rng: &mut StdRng,
) -> Result<Vec<u32>> {
    anyhow::ensure!(!prompt_tokens.is_empty(), "prompt_tokens must not be empty");
    let _span = LocalSpan::enter_with_local_parent("generate").with_properties(|| {
        [
            ("prompt_len", prompt_tokens.len().to_string()),
            ("max_new_tokens", max_new_tokens.to_string()),
        ]
    });

    let mut tokens = prompt_tokens.to_vec();

    let ttft_start = Instant::now();
    model.forward_prefill(prompt_tokens, state)?;
    let next_token = model.select_token(state, params, rng)?;
    let ttft = ttft_start.elapsed();

    LocalSpan::add_property(|| ("ttft_ms", format!("{:.2}", ttft.as_secs_f64() * 1000.0)));
    debug!(
        "TTFT: {:.2}ms (prompt_len={})",
        ttft.as_secs_f64() * 1000.0,
        prompt_tokens.len()
    );

    if !params.ignore_eos && model.is_stop_token(next_token) {
        return Ok(tokens);
    }
    tokens.push(next_token);

    let tpot_start = Instant::now();
    let mut generated_count = 0;
    for i in 1..max_new_tokens {
        let _span = LocalSpan::enter_with_local_parent("decode_step")
            .with_property(|| ("step", i.to_string()));
        model.forward_decode(*tokens.last().unwrap(), state)?;
        let next_token = model.select_token(state, params, rng)?;

        if !params.ignore_eos && model.is_stop_token(next_token) {
            break;
        }
        tokens.push(next_token);
        generated_count += 1;
    }

    if generated_count > 0 {
        let tpot_total = tpot_start.elapsed();
        let tpot_avg = tpot_total.as_secs_f64() / generated_count as f64;
        LocalSpan::add_properties(|| {
            [
                ("tpot_avg_ms", format!("{:.2}", tpot_avg * 1000.0)),
                ("generated_tokens", generated_count.to_string()),
                (
                    "tok_per_sec",
                    format!("{:.1}", generated_count as f64 / tpot_total.as_secs_f64()),
                ),
            ]
        });
        debug!(
            "TPOT: {:.2}ms/tok (generated {} tokens in {:.2}ms, {:.1} tok/s)",
            tpot_avg * 1000.0,
            generated_count,
            tpot_total.as_secs_f64() * 1000.0,
            generated_count as f64 / tpot_total.as_secs_f64()
        );
    }

    Ok(tokens)
}

/// Same as `generate_tokens_inner` but also returns per-token logprobs (greedy only).
#[cfg(feature = "cuda")]
fn generate_tokens_with_logprobs_inner<M: ModelForward>(
    model: &M,
    state: &mut M::State,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    params: &SamplingParams,
    rng: &mut StdRng,
) -> Result<(Vec<u32>, Vec<f32>)> {
    anyhow::ensure!(!prompt_tokens.is_empty(), "prompt_tokens must not be empty");

    let mut tokens = prompt_tokens.to_vec();
    let mut logprobs = Vec::new();

    model.forward_prefill(prompt_tokens, state)?;
    let (next_token, lp) = model.select_token_with_logprob(state, params, rng)?;
    if let Some(lp) = lp {
        logprobs.push(lp);
    }

    if !params.ignore_eos && model.is_stop_token(next_token) {
        return Ok((tokens, logprobs));
    }
    tokens.push(next_token);

    for _i in 1..max_new_tokens {
        model.forward_decode(*tokens.last().unwrap(), state)?;
        let (next_token, lp) = model.select_token_with_logprob(state, params, rng)?;
        if let Some(lp) = lp {
            logprobs.push(lp);
        }

        if !params.ignore_eos && model.is_stop_token(next_token) {
            break;
        }
        tokens.push(next_token);
    }

    Ok((tokens, logprobs))
}

#[cfg(feature = "cuda")]
fn generate_streaming_with_callback<M: ModelForward>(
    model: &M,
    state: &mut M::State,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    params: &SamplingParams,
    rng: &mut StdRng,
    mut on_token: impl FnMut(u32) -> bool,
) -> Result<StreamingStats> {
    anyhow::ensure!(!prompt_tokens.is_empty(), "prompt_tokens must not be empty");
    let _span = LocalSpan::enter_with_local_parent("generate_streaming").with_properties(|| {
        [
            ("prompt_len", prompt_tokens.len().to_string()),
            ("max_new_tokens", max_new_tokens.to_string()),
        ]
    });

    let mut tokens = prompt_tokens.to_vec();

    let ttft_start = Instant::now();
    model.forward_prefill(prompt_tokens, state)?;
    let next_token = model.select_token(state, params, rng)?;
    let ttft = ttft_start.elapsed();
    debug!(
        "TTFT: {:.2}ms (prompt_len={})",
        ttft.as_secs_f64() * 1000.0,
        prompt_tokens.len()
    );

    if !params.ignore_eos && model.is_stop_token(next_token) {
        return Ok(StreamingStats {
            emitted_tokens: 0,
            hit_eos: true,
            consumer_dropped: false,
        });
    }

    tokens.push(next_token);
    let mut emitted_tokens = 1usize;
    if !on_token(next_token) {
        return Ok(StreamingStats {
            emitted_tokens,
            hit_eos: false,
            consumer_dropped: true,
        });
    }

    let tpot_start = Instant::now();
    let mut generated_count = 0;
    let mut hit_eos = false;
    for i in 1..max_new_tokens {
        let _span = LocalSpan::enter_with_local_parent("decode_step")
            .with_property(|| ("step", i.to_string()));
        model.forward_decode(*tokens.last().unwrap(), state)?;
        let next_token = model.select_token(state, params, rng)?;

        if !params.ignore_eos && model.is_stop_token(next_token) {
            hit_eos = true;
            break;
        }

        tokens.push(next_token);
        generated_count += 1;
        emitted_tokens += 1;

        if !on_token(next_token) {
            return Ok(StreamingStats {
                emitted_tokens,
                hit_eos: false,
                consumer_dropped: true,
            });
        }
    }

    if generated_count > 0 {
        let tpot_total = tpot_start.elapsed();
        let tpot_avg = tpot_total.as_secs_f64() / generated_count as f64;
        debug!(
            "TPOT: {:.2}ms/tok (generated {} tokens in {:.2}ms, {:.1} tok/s)",
            tpot_avg * 1000.0,
            generated_count,
            tpot_total.as_secs_f64() * 1000.0,
            generated_count as f64 / tpot_total.as_secs_f64()
        );
    }

    Ok(StreamingStats {
        emitted_tokens,
        hit_eos,
        consumer_dropped: false,
    })
}

// ============================================================================
// Generic server engine — shared complete/complete_stream logic
// ============================================================================

/// Legacy single-request inference engine.
///
/// **Deprecated**: Use [`crate::scheduler::Scheduler`] for production serving.
/// `GenericServerEngine` processes one request at a time with no batching or
/// continuous scheduling. It remains for the agent CLI REPL and E2E tests.
/// New features should target the `Scheduler` path.
#[cfg(feature = "cuda")]
pub struct GenericServerEngine<M: ModelForward> {
    model_id: String,
    model: M,
    state: M::State,
    tokenizer: Tokenizer,
    rng: StdRng,
    /// Cached prompt tokens from the last request for prefix reuse.
    cached_prompt: Vec<u32>,
}

#[cfg(feature = "cuda")]
impl<M: ModelForward> GenericServerEngine<M> {
    fn from_model_components(
        components: crate::backend::cuda::bootstrap::ModelComponents<M>,
        seed: u64,
    ) -> Result<Self> {
        let crate::backend::cuda::bootstrap::ModelComponents {
            model_id,
            tokenizer,
            model,
        } = components;
        let state = model.create_state()?;
        let rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            model_id,
            model,
            state,
            tokenizer,
            rng,
            cached_prompt: Vec::new(),
        })
    }

    /// Set the maximum KV cache tokens to keep on GPU.
    /// Tokens beyond this are offloaded to CPU. Used to simulate memory pressure.
    pub fn set_max_gpu_kv(&mut self, max_tokens: usize) {
        self.state.set_max_gpu_kv(max_tokens);
    }

    /// Prepare state for a new request, reusing cached KV prefix where possible.
    ///
    /// Returns the tokens that still need to be processed (the non-cached suffix)
    /// and the prefix length that was reused.
    ///
    /// Key optimizations:
    /// - **Full prefix hit**: Reuse entire cached KV, only prefill the new suffix.
    /// - **Partial prefix hit**: Truncate KV to the common prefix (no full reset),
    ///   then prefill from the divergence point. Saves re-computing the shared part.
    /// - **KV offload prefetch**: If KV data was offloaded to CPU, automatically
    ///   prefetch it back to GPU before the prefill starts.
    fn prepare_with_prefix_cache(&mut self, prompt_tokens: &[u32]) -> Result<(Vec<u32>, usize)> {
        let prefix_len = self
            .cached_prompt
            .iter()
            .zip(prompt_tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();

        if prefix_len > 0 && prefix_len == self.cached_prompt.len() {
            // Full prefix hit — reuse all cached KV.
            let suffix = prompt_tokens[prefix_len..].to_vec();
            info!(
                "KV prefix cache HIT: reusing {}/{} tokens (saving {:.1}% prefill)",
                prefix_len,
                prompt_tokens.len(),
                prefix_len as f64 / prompt_tokens.len() as f64 * 100.0
            );
            Ok((suffix, prefix_len))
        } else if prefix_len > 0 {
            // Partial prefix hit — truncate KV to common prefix and reuse it.
            // This avoids re-computing the shared prefix entirely.
            info!(
                "KV prefix cache PARTIAL: reusing {}/{} common tokens, truncating {} stale tokens",
                prefix_len,
                prompt_tokens.len(),
                self.cached_prompt.len() - prefix_len,
            );
            self.state.truncate_to(prefix_len)?;
            self.cached_prompt.truncate(prefix_len);
            let suffix = prompt_tokens[prefix_len..].to_vec();
            Ok((suffix, prefix_len))
        } else {
            // No prefix match — full reset.
            info!("KV prefix cache MISS: resetting");
            self.state.reset()?;
            self.cached_prompt.clear();
            Ok((prompt_tokens.to_vec(), 0))
        }
    }
}

#[cfg(feature = "cuda")]
impl<M: ModelForward> ServerEngine for GenericServerEngine<M> {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn complete(&mut self, req: CompleteRequest) -> Result<CompleteOutput> {
        let prompt_tokens = self.tokenizer.encode(&req.prompt)?;
        let (tokens_to_process, _prefix_len) = self.prepare_with_prefix_cache(&prompt_tokens)?;
        // tokens_to_process is the suffix that still needs prefill.
        // If empty (full cache hit), use just the last token to get logits.
        let effective = if tokens_to_process.is_empty() {
            vec![*prompt_tokens.last().unwrap()]
        } else {
            tokens_to_process
        };
        let (output_tokens, token_logprobs) = if req.logprobs && req.sampling.is_greedy() {
            generate_tokens_with_logprobs_inner(
                &self.model,
                &mut self.state,
                &effective,
                req.max_tokens,
                &req.sampling,
                &mut self.rng,
            )?
        } else {
            let tokens = generate(
                &self.model,
                &mut self.state,
                &effective,
                req.max_tokens,
                &req.sampling,
                &mut self.rng,
            )?;
            (tokens, vec![])
        };
        // Update cached prompt and offload excess KV for next request
        self.cached_prompt = prompt_tokens.clone();
        self.state.offload_kv_if_needed()?;
        // output_tokens = effective_prompt + generated tokens
        let completion_tokens = output_tokens.len().saturating_sub(effective.len());
        let mut text = self.tokenizer.decode(&output_tokens[effective.len()..])?;
        let mut finish_reason = if completion_tokens >= req.max_tokens {
            FinishReason::Length
        } else {
            FinishReason::Stop
        };
        if let Some(ref stops) = req.stop
            && let Some(truncated) = truncate_at_first_stop(&text, stops)
        {
            text = truncated;
            finish_reason = FinishReason::Stop;
        }
        let usage = Usage {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens,
            total_tokens: output_tokens.len(),
        };
        Ok(CompleteOutput {
            text,
            finish_reason,
            usage,
            token_logprobs,
        })
    }

    fn complete_stream(
        &mut self,
        req: CompleteRequest,
        tx: UnboundedSender<StreamDelta>,
    ) -> Result<()> {
        let prompt_tokens = self.tokenizer.encode(&req.prompt)?;
        let (tokens_to_process, _prefix_len) = self.prepare_with_prefix_cache(&prompt_tokens)?;
        let effective_prompt = if tokens_to_process.is_empty() {
            vec![*prompt_tokens.last().unwrap()]
        } else {
            tokens_to_process
        };
        let mut decoder = self.tokenizer.incremental_decoder();
        let mut decode_error = None;
        let stops: Option<Vec<&str>> = req.stop.as_ref().map(|v| {
            v.iter()
                .map(String::as_str)
                .filter(|s| !s.is_empty())
                .collect()
        });
        let mut sent_len: usize = 0;
        let stopped_by_stop_sequence = std::cell::Cell::new(false);

        let stats = generate_streaming_with_callback(
            &self.model,
            &mut self.state,
            &effective_prompt,
            req.max_tokens,
            &req.sampling,
            &mut self.rng,
            |token_id| match decoder.step(token_id) {
                Ok(Some(text_delta)) => {
                    if let Some(ref stop_list) = stops {
                        let new_full = {
                            let emitted = decoder.emitted_text();
                            emitted.to_string()
                        };
                        if let Some((to_send, stopped)) =
                            truncate_at_stop(&new_full, sent_len, stop_list)
                        {
                            if !to_send.is_empty()
                                && tx
                                    .send(StreamDelta {
                                        text_delta: to_send,
                                        finish_reason: None,
                                        usage: None,
                                        logprob: None,
                                    })
                                    .is_err()
                            {
                                return false;
                            }
                            sent_len = new_full.len() - stopped.len();
                            stopped_by_stop_sequence.set(true);
                            return false;
                        }
                        let to_send = &new_full[sent_len..];
                        sent_len = new_full.len();
                        tx.send(StreamDelta {
                            text_delta: to_send.to_string(),
                            finish_reason: None,
                            usage: None,
                            logprob: None,
                        })
                        .is_ok()
                    } else {
                        tx.send(StreamDelta {
                            text_delta,
                            finish_reason: None,
                            usage: None,
                            logprob: None,
                        })
                        .is_ok()
                    }
                }
                Ok(None) => true,
                Err(err) => {
                    decode_error = Some(err);
                    false
                }
            },
        )?;

        if let Some(err) = decode_error {
            return Err(err);
        }

        if stats.consumer_dropped && !stopped_by_stop_sequence.get() {
            return Ok(());
        }

        if !stopped_by_stop_sequence.get()
            && let Some(text_delta) = decoder.finish()?
        {
            if let Some(ref stop_list) = stops {
                let new_full = decoder.emitted_text().to_string();
                if let Some((to_send, _)) = truncate_at_stop(&new_full, sent_len, stop_list) {
                    if !to_send.is_empty() {
                        let _ = tx.send(StreamDelta {
                            text_delta: to_send,
                            finish_reason: None,
                            usage: None,
                            logprob: None,
                        });
                    }
                } else {
                    let to_send = &new_full[sent_len..];
                    if !to_send.is_empty() {
                        let _ = tx.send(StreamDelta {
                            text_delta: to_send.to_string(),
                            finish_reason: None,
                            usage: None,
                            logprob: None,
                        });
                    }
                }
            } else {
                let _ = tx.send(StreamDelta {
                    text_delta,
                    finish_reason: None,
                    usage: None,
                    logprob: None,
                });
            }
        }

        let finish_reason = if stopped_by_stop_sequence.get() || stats.hit_eos {
            FinishReason::Stop
        } else {
            FinishReason::Length
        };

        let _ = tx.send(StreamDelta {
            text_delta: String::new(),
            finish_reason: Some(finish_reason),
            usage: Some(Usage {
                prompt_tokens: prompt_tokens.len(),
                completion_tokens: stats.emitted_tokens,
                total_tokens: prompt_tokens.len() + stats.emitted_tokens,
            }),
            logprob: None,
        });

        // Update cached prompt and offload excess KV for next request
        self.cached_prompt = prompt_tokens;
        self.state.offload_kv_if_needed()?;

        Ok(())
    }
}

// ============================================================================
// Public engine constructors
// ============================================================================

#[cfg(feature = "cuda")]
pub type RealServerEngine = GenericServerEngine<Qwen3Model>;
#[cfg(feature = "cuda")]
pub type Qwen35ServerEngine = GenericServerEngine<Qwen35Model>;
#[cfg(feature = "cuda")]
pub type GLM4ServerEngine = GenericServerEngine<GLM4Model>;

#[cfg(feature = "cuda")]
pub enum LoadedServerEngine {
    Qwen3(RealServerEngine),
    Qwen35(Qwen35ServerEngine),
    GLM4(GLM4ServerEngine),
}

#[cfg(feature = "cuda")]
impl RealServerEngine {
    pub fn load(model_path: &str, seed: u64) -> Result<Self> {
        Self::load_with_options(model_path, seed, EngineOptions::default())
    }

    pub fn load_with_options(model_path: &str, seed: u64, options: EngineOptions) -> Result<Self> {
        let components =
            crate::backend::cuda::bootstrap::load_qwen3_components(model_path, options)?;
        Self::from_model_components(components, seed)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }
}

#[cfg(feature = "cuda")]
impl Qwen35ServerEngine {
    pub fn load(model_path: &str, seed: u64) -> Result<Self> {
        Self::load_with_options(model_path, seed, EngineOptions::default())
    }

    pub fn load_with_options(model_path: &str, seed: u64, options: EngineOptions) -> Result<Self> {
        let components =
            crate::backend::cuda::bootstrap::load_qwen35_components(model_path, options)?;
        Self::from_model_components(components, seed)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }
}

#[cfg(feature = "cuda")]
impl GLM4ServerEngine {
    pub fn load(model_path: &str, seed: u64) -> Result<Self> {
        Self::load_with_options(model_path, seed, EngineOptions::default())
    }

    pub fn load_with_options(model_path: &str, seed: u64, options: EngineOptions) -> Result<Self> {
        let components =
            crate::backend::cuda::bootstrap::load_glm4_components(model_path, options)?;
        Self::from_model_components(components, seed)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }
}

#[cfg(feature = "cuda")]
impl LoadedServerEngine {
    pub fn load(model_path: &str, seed: u64) -> Result<Self> {
        Self::load_with_options(model_path, seed, EngineOptions::default())
    }

    pub fn load_with_options(model_path: &str, seed: u64, options: EngineOptions) -> Result<Self> {
        match detect_model_type(model_path)? {
            ModelType::Qwen3 => Ok(Self::Qwen3(RealServerEngine::load_with_options(
                model_path, seed, options,
            )?)),
            ModelType::Qwen35 => Ok(Self::Qwen35(Qwen35ServerEngine::load_with_options(
                model_path, seed, options,
            )?)),
            ModelType::GLM4 => Ok(Self::GLM4(GLM4ServerEngine::load_with_options(
                model_path, seed, options,
            )?)),
        }
    }

    pub fn model_type(&self) -> ModelType {
        match self {
            Self::Qwen3(_) => ModelType::Qwen3,
            Self::Qwen35(_) => ModelType::Qwen35,
            Self::GLM4(_) => ModelType::GLM4,
        }
    }

    pub fn set_max_gpu_kv(&mut self, max_tokens: usize) {
        match self {
            Self::Qwen3(engine) => engine.set_max_gpu_kv(max_tokens),
            Self::Qwen35(engine) => engine.set_max_gpu_kv(max_tokens),
            Self::GLM4(engine) => engine.set_max_gpu_kv(max_tokens),
        }
    }
}

#[cfg(feature = "cuda")]
impl ServerEngine for LoadedServerEngine {
    fn model_id(&self) -> &str {
        match self {
            Self::Qwen3(engine) => engine.model_id(),
            Self::Qwen35(engine) => engine.model_id(),
            Self::GLM4(engine) => engine.model_id(),
        }
    }

    fn complete(&mut self, req: CompleteRequest) -> Result<CompleteOutput> {
        match self {
            Self::Qwen3(engine) => engine.complete(req),
            Self::Qwen35(engine) => engine.complete(req),
            Self::GLM4(engine) => engine.complete(req),
        }
    }

    fn complete_stream(
        &mut self,
        req: CompleteRequest,
        tx: UnboundedSender<StreamDelta>,
    ) -> Result<()> {
        match self {
            Self::Qwen3(engine) => engine.complete_stream(req, tx),
            Self::Qwen35(engine) => engine.complete_stream(req, tx),
            Self::GLM4(engine) => engine.complete_stream(req, tx),
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::{truncate_at_first_stop, truncate_at_stop};

    #[test]
    fn test_truncate_at_first_stop() {
        let stops: Vec<String> = vec!["\n\n".into(), "END".into()];
        assert_eq!(
            truncate_at_first_stop("4\n\nand more", &stops),
            Some("4".to_string())
        );
        assert_eq!(
            truncate_at_first_stop("helloEND", &stops),
            Some("hello".to_string())
        );
        assert_eq!(truncate_at_first_stop("hello", &stops), None);
        assert_eq!(truncate_at_first_stop("", &stops), None);
        assert_eq!(
            truncate_at_first_stop("a\n\nbEND", &stops),
            Some("a".to_string())
        );
        let stops_nl: Vec<String> = vec!["\n".into()];
        assert_eq!(
            truncate_at_first_stop("hello\nworld", &stops_nl),
            Some("hello".to_string())
        );
        assert_eq!(
            truncate_at_first_stop("ab", &["ab".to_string()]),
            Some(String::new())
        );
    }

    #[test]
    fn test_truncate_at_stop() {
        let stops = ["\n"];
        assert_eq!(
            truncate_at_stop("hello\n", 0, &stops),
            Some(("hello".to_string(), "\n"))
        );
        assert_eq!(truncate_at_stop("hello", 0, &stops), None);
        assert_eq!(
            truncate_at_stop("ab\n", 2, &stops),
            Some((String::new(), "\n"))
        );
    }
}
