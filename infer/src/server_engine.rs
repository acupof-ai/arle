#[cfg(any(feature = "metal", feature = "cpu"))]
use std::panic::{AssertUnwindSafe, catch_unwind};
#[cfg(any(feature = "metal", feature = "cpu", test))]
use std::path::Path;
#[cfg(feature = "cuda")]
use std::time::Instant;

use anyhow::Result;
#[cfg(feature = "cuda")]
use fastrace::local::LocalSpan;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use log::warn;
#[cfg(feature = "cuda")]
use log::{debug, info};
#[cfg(feature = "cuda")]
use rand::SeedableRng;
#[cfg(feature = "cuda")]
use rand::rngs::StdRng;
use tokio::sync::mpsc::UnboundedSender;

#[cfg(any(feature = "metal", feature = "cpu"))]
use crate::backend::InferenceBackend;
#[cfg(feature = "cpu")]
use crate::backend::cpu::CpuBackend;
#[cfg(feature = "cuda")]
pub use crate::backend::cuda::bootstrap::{
    InferenceEngineOptions, ModelType, ServerRuntimeConfig, detect_model_type,
};
#[cfg(feature = "metal")]
use crate::backend::metal::MetalBackend;
#[cfg(feature = "cuda")]
use crate::model::{GLM4Model, GenerationState, ModelForward, Qwen3Model, Qwen35Model};
use crate::sampler::SamplingParams;
#[cfg(feature = "cuda")]
use crate::tokenizer::Tokenizer;

/// Truncate at the first occurrence of any stop string (OpenAI-compatible).
/// Returns the prefix of `text` up to (but not including) the earliest stop.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu", test))]
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

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PrefixReuseAction {
    Miss,
    FullRecompute,
    ReplayFinalToken { replay_from: usize },
    ReuseFullCachedPrefix { prefix_len: usize },
    PartialReuse { prefix_len: usize },
}

#[cfg(any(feature = "cuda", test))]
fn choose_prefix_reuse_action(
    prompt_len: usize,
    cached_len: usize,
    prefix_len: usize,
    supports_partial_prefix: bool,
) -> PrefixReuseAction {
    if prefix_len == 0 {
        return PrefixReuseAction::Miss;
    }

    let exact_full_match = prefix_len == prompt_len && prefix_len == cached_len;
    if exact_full_match {
        return if supports_partial_prefix {
            PrefixReuseAction::ReplayFinalToken {
                replay_from: prefix_len.saturating_sub(1),
            }
        } else {
            PrefixReuseAction::FullRecompute
        };
    }

    let prompt_is_strict_prefix_of_cached = prefix_len == prompt_len;
    if prompt_is_strict_prefix_of_cached {
        return PrefixReuseAction::FullRecompute;
    }

    let prompt_extends_cached_exactly = prefix_len == cached_len;
    if prompt_extends_cached_exactly {
        return PrefixReuseAction::ReuseFullCachedPrefix { prefix_len };
    }

    if supports_partial_prefix {
        PrefixReuseAction::PartialReuse { prefix_len }
    } else {
        PrefixReuseAction::FullRecompute
    }
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

#[cfg(any(feature = "metal", feature = "cpu", test))]
fn model_id_from_path(model_path: &str) -> String {
    Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_string()
}

#[cfg(any(feature = "metal", feature = "cpu"))]
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(msg) => *msg,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(msg) => (*msg).to_string(),
            Err(_) => "unknown panic payload".to_string(),
        },
    }
}

#[cfg(any(feature = "metal", feature = "cpu", test))]
fn parse_finish_reason(finish_reason: &str) -> FinishReason {
    match finish_reason {
        "length" => FinishReason::Length,
        _ => FinishReason::Stop,
    }
}

pub struct CompletionRequest {
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

pub struct CompletionOutput {
    pub text: String,
    pub finish_reason: FinishReason,
    pub usage: TokenUsage,
    /// Per-token log-probabilities (greedy only). Empty if logprobs not requested.
    pub token_logprobs: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

pub struct CompletionStreamDelta {
    pub text_delta: String,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<TokenUsage>,
    /// Log-probability of the generated token (greedy only, None otherwise).
    #[allow(dead_code)]
    pub logprob: Option<f32>,
}

impl CompletionStreamDelta {
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

pub trait InferenceEngine: Send {
    /// Returns the model identifier (e.g. `"Qwen3-8B"`).
    fn model_id(&self) -> &str;

    /// Run a complete generation request synchronously and return the full output.
    fn complete(&mut self, req: CompletionRequest) -> Result<CompletionOutput>;

    /// Run a generation request, streaming token deltas through `tx` as they are produced.
    fn complete_stream(
        &mut self,
        req: CompletionRequest,
        tx: UnboundedSender<CompletionStreamDelta>,
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
    if let Err(e) = state.save_prefix_snapshot() {
        warn!(
            "KV prefix cache snapshot save failed after prefill: {} (exact-hit reuse may fall back later)",
            e
        );
    }
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
    if let Err(e) = state.save_prefix_snapshot() {
        warn!(
            "KV prefix cache snapshot save failed after prefill: {} (exact-hit reuse may fall back later)",
            e
        );
    }
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
    if let Err(e) = state.save_prefix_snapshot() {
        warn!(
            "KV prefix cache snapshot save failed after prefill: {} (exact-hit reuse may fall back later)",
            e
        );
    }
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
// Model inference engine — shared complete/complete_stream logic
// ============================================================================

/// Legacy single-request inference engine.
///
/// **Deprecated**: Use [`crate::scheduler::Scheduler`] for production serving.
/// `ModelInferenceEngine` processes one request at a time with no batching or
/// continuous scheduling. It remains for the agent CLI REPL and E2E tests.
/// New features should target the `Scheduler` path.
#[cfg(feature = "cuda")]
pub struct ModelInferenceEngine<M: ModelForward> {
    model_id: String,
    model: M,
    state: M::State,
    tokenizer: Tokenizer,
    rng: StdRng,
    /// Cached prompt tokens from the last request for prefix reuse.
    cached_prompt: Vec<u32>,
}

#[cfg(feature = "cuda")]
impl<M: ModelForward> ModelInferenceEngine<M> {
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

    /// Compatibility no-op. Legacy contiguous CPU KV offload has been retired.
    pub fn set_max_gpu_kv(&mut self, max_tokens: usize) {
        warn!(
            "Ignoring set_max_gpu_kv({}): legacy contiguous CPU KV offload has been retired",
            max_tokens
        );
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
    fn prepare_with_prefix_cache(&mut self, prompt_tokens: &[u32]) -> Result<(Vec<u32>, usize)> {
        let cached_len = self.cached_prompt.len();
        let prefix_len = self
            .cached_prompt
            .iter()
            .zip(prompt_tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();
        match choose_prefix_reuse_action(
            prompt_tokens.len(),
            cached_len,
            prefix_len,
            self.state.supports_partial_prefix(),
        ) {
            PrefixReuseAction::Miss => {
                info!("KV prefix cache MISS: resetting");
                self.state.reset()?;
                self.cached_prompt.clear();
                Ok((prompt_tokens.to_vec(), 0))
            }
            PrefixReuseAction::FullRecompute => {
                let reason = if prefix_len > 0 && prefix_len == prompt_tokens.len() {
                    "cached prompt has extra suffix"
                } else {
                    "non-truncatable state"
                };
                info!(
                    "KV prefix cache FULL/PARTIAL HIT: {} , falling back to full prefill",
                    reason
                );
                self.state.reset()?;
                self.cached_prompt.clear();
                Ok((prompt_tokens.to_vec(), 0))
            }
            PrefixReuseAction::ReplayFinalToken { replay_from: _ } => {
                // Exact prompt match. Rewinding to N-1 and replaying the final
                // token via a single-token forward_prefill is not numerically
                // identical to a cold batch prefill of the same N tokens —
                // FlashInfer and the attention prep kernels produce
                // slightly different position-N-1 logits at batch=1 vs batch=N,
                // and greedy argmax flips. See
                // `docs/experience/errors/2026-04-15-e2e-phase3-replay-drift.md`.
                // Fall back to a full recompute: correctness over the ~one
                // prefill saved, which only ever helps exact-same-prompt
                // retries in the single-request engine.
                info!(
                    "KV prefix cache FULL HIT: exact match, falling back to full recompute for numerical consistency"
                );
                self.state.reset()?;
                self.cached_prompt.clear();
                Ok((prompt_tokens.to_vec(), 0))
            }
            PrefixReuseAction::ReuseFullCachedPrefix { prefix_len } => {
                if self.state.supports_partial_prefix() {
                    self.state.truncate_to(prefix_len)?;
                }
                match self.state.restore_prefix_snapshot() {
                    Ok(true) => {
                        info!(
                            "KV prefix cache HIT: restored clean prompt snapshot for {}/{} reused tokens",
                            prefix_len,
                            prompt_tokens.len()
                        );
                    }
                    Ok(false) if self.state.supports_partial_prefix() => {
                        info!(
                            "KV prefix cache HIT: reusing {}/{} tokens (saving {:.1}% prefill)",
                            prefix_len,
                            prompt_tokens.len(),
                            prefix_len as f64 / prompt_tokens.len() as f64 * 100.0
                        );
                    }
                    Ok(false) => {
                        warn!(
                            "KV prefix cache HIT: snapshot missing for non-truncatable state, falling back to full prefill"
                        );
                        self.state.reset()?;
                        self.cached_prompt.clear();
                        return Ok((prompt_tokens.to_vec(), 0));
                    }
                    Err(e) if self.state.supports_partial_prefix() => {
                        warn!(
                            "KV prefix cache HIT: snapshot restore failed ({}), continuing with truncation-only reuse",
                            e
                        );
                    }
                    Err(e) => {
                        warn!(
                            "KV prefix cache HIT: snapshot restore failed for non-truncatable state ({}), falling back to full prefill",
                            e
                        );
                        self.state.reset()?;
                        self.cached_prompt.clear();
                        return Ok((prompt_tokens.to_vec(), 0));
                    }
                }
                Ok((prompt_tokens[prefix_len..].to_vec(), prefix_len))
            }
            PrefixReuseAction::PartialReuse { prefix_len } => {
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
                Ok((prompt_tokens[prefix_len..].to_vec(), prefix_len))
            }
        }
    }
}

#[cfg(feature = "cuda")]
impl<M: ModelForward> InferenceEngine for ModelInferenceEngine<M> {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn complete(&mut self, req: CompletionRequest) -> Result<CompletionOutput> {
        let prompt_tokens = self.tokenizer.encode(&req.prompt)?;
        let (effective, _prefix_len) = self.prepare_with_prefix_cache(&prompt_tokens)?;
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
        // Update cached prompt for next request.
        self.cached_prompt = prompt_tokens.clone();
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
        let usage = TokenUsage {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens,
            total_tokens: output_tokens.len(),
        };
        Ok(CompletionOutput {
            text,
            finish_reason,
            usage,
            token_logprobs,
        })
    }

    fn complete_stream(
        &mut self,
        req: CompletionRequest,
        tx: UnboundedSender<CompletionStreamDelta>,
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
                                    .send(CompletionStreamDelta {
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
                        tx.send(CompletionStreamDelta {
                            text_delta: to_send.to_string(),
                            finish_reason: None,
                            usage: None,
                            logprob: None,
                        })
                        .is_ok()
                    } else {
                        tx.send(CompletionStreamDelta {
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
                        let _ = tx.send(CompletionStreamDelta {
                            text_delta: to_send,
                            finish_reason: None,
                            usage: None,
                            logprob: None,
                        });
                    }
                } else {
                    let to_send = &new_full[sent_len..];
                    if !to_send.is_empty() {
                        let _ = tx.send(CompletionStreamDelta {
                            text_delta: to_send.to_string(),
                            finish_reason: None,
                            usage: None,
                            logprob: None,
                        });
                    }
                }
            } else {
                let _ = tx.send(CompletionStreamDelta {
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

        let _ = tx.send(CompletionStreamDelta {
            text_delta: String::new(),
            finish_reason: Some(finish_reason),
            usage: Some(TokenUsage {
                prompt_tokens: prompt_tokens.len(),
                completion_tokens: stats.emitted_tokens,
                total_tokens: prompt_tokens.len() + stats.emitted_tokens,
            }),
            logprob: None,
        });

        // Update cached prompt for next request.
        self.cached_prompt = prompt_tokens;

        Ok(())
    }
}

// ============================================================================
// Public engine constructors
// ============================================================================

#[cfg(feature = "cuda")]
pub type Qwen3InferenceEngine = ModelInferenceEngine<Qwen3Model>;
#[cfg(feature = "cuda")]
pub type Qwen35InferenceEngine = ModelInferenceEngine<Qwen35Model>;
#[cfg(feature = "cuda")]
pub type GLM4InferenceEngine = ModelInferenceEngine<GLM4Model>;

#[cfg(any(feature = "metal", feature = "cpu"))]
pub struct BackendInferenceEngine<B: InferenceBackend> {
    model_id: String,
    backend: B,
}

#[cfg(feature = "metal")]
impl BackendInferenceEngine<MetalBackend> {
    fn load(model_path: &str) -> Result<Self> {
        let mut backend = MetalBackend::new();
        backend.load(Path::new(model_path))?;
        Ok(Self {
            model_id: model_id_from_path(model_path),
            backend,
        })
    }
}

#[cfg(feature = "cpu")]
impl BackendInferenceEngine<CpuBackend> {
    fn load(model_path: &str) -> Result<Self> {
        let mut backend = CpuBackend::new();
        backend.load(Path::new(model_path))?;
        Ok(Self {
            model_id: model_id_from_path(model_path),
            backend,
        })
    }
}

#[cfg(any(feature = "metal", feature = "cpu"))]
impl<B: InferenceBackend> InferenceEngine for BackendInferenceEngine<B> {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn complete(&mut self, req: CompletionRequest) -> Result<CompletionOutput> {
        let mut sampling = req.sampling;
        sampling.max_new_tokens = Some(req.max_tokens);

        let generated = catch_unwind(AssertUnwindSafe(|| {
            self.backend.generate(&req.prompt, &sampling)
        }))
        .map_err(|panic| {
            anyhow::anyhow!(
                "{} backend panicked during completion: {}",
                self.backend.name(),
                panic_message(panic)
            )
        })??;

        let mut text = generated.text;
        let mut finish_reason = parse_finish_reason(&generated.finish_reason);

        if let Some(stops) = req.stop
            && let Some(truncated) = truncate_at_first_stop(&text, &stops)
        {
            text = truncated;
            finish_reason = FinishReason::Stop;
        }

        Ok(CompletionOutput {
            text,
            finish_reason,
            usage: TokenUsage {
                prompt_tokens: generated.prompt_tokens,
                completion_tokens: generated.completion_tokens,
                total_tokens: generated.prompt_tokens + generated.completion_tokens,
            },
            token_logprobs: Vec::new(),
        })
    }

    fn complete_stream(
        &mut self,
        req: CompletionRequest,
        tx: UnboundedSender<CompletionStreamDelta>,
    ) -> Result<()> {
        let output = self.complete(req)?;
        if !output.text.is_empty() {
            let _ = tx.send(CompletionStreamDelta {
                text_delta: output.text.clone(),
                finish_reason: None,
                usage: None,
                logprob: None,
            });
        }
        let _ = tx.send(CompletionStreamDelta {
            text_delta: String::new(),
            finish_reason: Some(output.finish_reason),
            usage: Some(output.usage),
            logprob: None,
        });
        Ok(())
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
pub enum LoadedInferenceEngine {
    #[cfg(feature = "cuda")]
    Qwen3(Qwen3InferenceEngine),
    #[cfg(feature = "cuda")]
    Qwen35(Qwen35InferenceEngine),
    #[cfg(feature = "cuda")]
    GLM4(GLM4InferenceEngine),
    #[cfg(feature = "metal")]
    Metal(BackendInferenceEngine<MetalBackend>),
    #[cfg(feature = "cpu")]
    Cpu(BackendInferenceEngine<CpuBackend>),
}

#[cfg(feature = "cuda")]
impl Qwen3InferenceEngine {
    pub fn load(model_path: &str, seed: u64) -> Result<Self> {
        Self::load_with_options(model_path, seed, InferenceEngineOptions::default())
    }

    pub fn load_with_options(
        model_path: &str,
        seed: u64,
        options: InferenceEngineOptions,
    ) -> Result<Self> {
        let components =
            crate::backend::cuda::bootstrap::load_qwen3_components(model_path, options)?;
        Self::from_model_components(components, seed)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }
}

#[cfg(feature = "cuda")]
impl Qwen35InferenceEngine {
    pub fn load(model_path: &str, seed: u64) -> Result<Self> {
        Self::load_with_options(model_path, seed, InferenceEngineOptions::default())
    }

    pub fn load_with_options(
        model_path: &str,
        seed: u64,
        options: InferenceEngineOptions,
    ) -> Result<Self> {
        let components =
            crate::backend::cuda::bootstrap::load_qwen35_components(model_path, options)?;
        Self::from_model_components(components, seed)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }
}

#[cfg(feature = "cuda")]
impl GLM4InferenceEngine {
    pub fn load(model_path: &str, seed: u64) -> Result<Self> {
        Self::load_with_options(model_path, seed, InferenceEngineOptions::default())
    }

    pub fn load_with_options(
        model_path: &str,
        seed: u64,
        options: InferenceEngineOptions,
    ) -> Result<Self> {
        let components =
            crate::backend::cuda::bootstrap::load_glm4_components(model_path, options)?;
        Self::from_model_components(components, seed)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
impl LoadedInferenceEngine {
    pub fn load(model_path: &str, enable_cuda_graph: bool) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            return Self::load_with_options(
                model_path,
                42,
                InferenceEngineOptions { enable_cuda_graph },
            );
        }

        #[cfg(all(not(feature = "cuda"), feature = "metal"))]
        {
            let _ = enable_cuda_graph;
            return Ok(Self::Metal(BackendInferenceEngine::load(model_path)?));
        }

        #[cfg(all(not(feature = "cuda"), not(feature = "metal"), feature = "cpu"))]
        {
            let _ = enable_cuda_graph;
            return Ok(Self::Cpu(BackendInferenceEngine::load(model_path)?));
        }

        #[allow(unreachable_code)]
        {
            let _ = (model_path, enable_cuda_graph);
            anyhow::bail!("no inference backend enabled")
        }
    }

    #[cfg(feature = "cuda")]
    pub fn load_with_options(
        model_path: &str,
        seed: u64,
        options: InferenceEngineOptions,
    ) -> Result<Self> {
        match detect_model_type(model_path)? {
            ModelType::Qwen3 => Ok(Self::Qwen3(Qwen3InferenceEngine::load_with_options(
                model_path, seed, options,
            )?)),
            ModelType::Qwen35 => Ok(Self::Qwen35(Qwen35InferenceEngine::load_with_options(
                model_path, seed, options,
            )?)),
            ModelType::GLM4 => Ok(Self::GLM4(GLM4InferenceEngine::load_with_options(
                model_path, seed, options,
            )?)),
        }
    }

    pub fn backend_name(&self) -> &'static str {
        match self {
            #[cfg(feature = "cuda")]
            Self::Qwen3(_) | Self::Qwen35(_) | Self::GLM4(_) => "cuda",
            #[cfg(feature = "metal")]
            Self::Metal(_) => "metal",
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => "cpu",
        }
    }

    #[cfg(feature = "cuda")]
    pub fn model_type(&self) -> ModelType {
        match self {
            Self::Qwen3(_) => ModelType::Qwen3,
            Self::Qwen35(_) => ModelType::Qwen35,
            Self::GLM4(_) => ModelType::GLM4,
            #[cfg(feature = "metal")]
            Self::Metal(_) => unreachable!("model_type is only defined for CUDA engines"),
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => unreachable!("model_type is only defined for CUDA engines"),
        }
    }

    pub fn set_max_gpu_kv(&mut self, max_tokens: usize) {
        match self {
            #[cfg(feature = "cuda")]
            Self::Qwen3(engine) => engine.set_max_gpu_kv(max_tokens),
            #[cfg(feature = "cuda")]
            Self::Qwen35(engine) => engine.set_max_gpu_kv(max_tokens),
            #[cfg(feature = "cuda")]
            Self::GLM4(engine) => engine.set_max_gpu_kv(max_tokens),
            #[cfg(feature = "metal")]
            Self::Metal(_) => {
                warn!(
                    "Ignoring set_max_gpu_kv({}): legacy contiguous CPU KV offload was CUDA-only and has been retired",
                    max_tokens
                );
            }
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => {
                warn!(
                    "Ignoring set_max_gpu_kv({}): legacy contiguous CPU KV offload has been retired",
                    max_tokens
                );
            }
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
impl InferenceEngine for LoadedInferenceEngine {
    fn model_id(&self) -> &str {
        match self {
            #[cfg(feature = "cuda")]
            Self::Qwen3(engine) => engine.model_id(),
            #[cfg(feature = "cuda")]
            Self::Qwen35(engine) => engine.model_id(),
            #[cfg(feature = "cuda")]
            Self::GLM4(engine) => engine.model_id(),
            #[cfg(feature = "metal")]
            Self::Metal(engine) => engine.model_id(),
            #[cfg(feature = "cpu")]
            Self::Cpu(engine) => engine.model_id(),
        }
    }

    fn complete(&mut self, req: CompletionRequest) -> Result<CompletionOutput> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Qwen3(engine) => engine.complete(req),
            #[cfg(feature = "cuda")]
            Self::Qwen35(engine) => engine.complete(req),
            #[cfg(feature = "cuda")]
            Self::GLM4(engine) => engine.complete(req),
            #[cfg(feature = "metal")]
            Self::Metal(engine) => engine.complete(req),
            #[cfg(feature = "cpu")]
            Self::Cpu(engine) => engine.complete(req),
        }
    }

    fn complete_stream(
        &mut self,
        req: CompletionRequest,
        tx: UnboundedSender<CompletionStreamDelta>,
    ) -> Result<()> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Qwen3(engine) => engine.complete_stream(req, tx),
            #[cfg(feature = "cuda")]
            Self::Qwen35(engine) => engine.complete_stream(req, tx),
            #[cfg(feature = "cuda")]
            Self::GLM4(engine) => engine.complete_stream(req, tx),
            #[cfg(feature = "metal")]
            Self::Metal(engine) => engine.complete_stream(req, tx),
            #[cfg(feature = "cpu")]
            Self::Cpu(engine) => engine.complete_stream(req, tx),
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    use super::truncate_at_stop;
    use super::{FinishReason, model_id_from_path, parse_finish_reason, truncate_at_first_stop};
    #[cfg(any(feature = "cuda", test))]
    use super::{PrefixReuseAction, choose_prefix_reuse_action};

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

    #[cfg(feature = "cuda")]
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

    #[test]
    fn model_id_uses_final_path_segment() {
        assert_eq!(
            model_id_from_path("mlx-community/Qwen3-0.6B-4bit"),
            "Qwen3-0.6B-4bit"
        );
        assert_eq!(model_id_from_path("/tmp/models/Qwen3-4B"), "Qwen3-4B");
    }

    #[test]
    fn parse_finish_reason_defaults_to_stop() {
        assert_eq!(parse_finish_reason("length"), FinishReason::Length);
        assert_eq!(parse_finish_reason("stop"), FinishReason::Stop);
        assert_eq!(parse_finish_reason("tool_calls"), FinishReason::Stop);
    }

    #[test]
    fn choose_prefix_reuse_action_covers_scheduler_aligned_cases() {
        assert_eq!(
            choose_prefix_reuse_action(4, 4, 4, true),
            PrefixReuseAction::ReplayFinalToken { replay_from: 3 }
        );
        assert_eq!(
            choose_prefix_reuse_action(4, 4, 4, false),
            PrefixReuseAction::FullRecompute
        );
        assert_eq!(
            choose_prefix_reuse_action(6, 4, 4, true),
            PrefixReuseAction::ReuseFullCachedPrefix { prefix_len: 4 }
        );
        assert_eq!(
            choose_prefix_reuse_action(6, 5, 3, true),
            PrefixReuseAction::PartialReuse { prefix_len: 3 }
        );
        assert_eq!(
            choose_prefix_reuse_action(6, 5, 3, false),
            PrefixReuseAction::FullRecompute
        );
        assert_eq!(
            choose_prefix_reuse_action(3, 5, 3, true),
            PrefixReuseAction::FullRecompute
        );
        assert_eq!(
            choose_prefix_reuse_action(5, 5, 0, true),
            PrefixReuseAction::Miss
        );
    }
}
