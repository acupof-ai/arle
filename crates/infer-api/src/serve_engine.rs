//! Generic adapter wrapping a [`ServeHandle`] + tokenizer as an
//! [`InferenceEngine`]: `tokenize -> submit -> collect -> detokenize`, projecting
//! the rewrite `CompletedRequest` back into [`CompletionOutput`].

use std::sync::Arc;

use anyhow::{Result, anyhow};
use infer_core::CompletedRequest;
use infer_seam::{BackendExecutor, KvPool};
use infer_server::{OpenAiTokenizer, ServeHandle, StreamItem};
use tokio::sync::mpsc::UnboundedSender;

use crate::types::{
    ChatPromptMessage, CompletionOutput, CompletionRequest, CompletionStreamDelta, EngineTelemetry,
    FinishReason, InferenceEngine, MultimodalChatRequest, TokenUsage,
};

/// Adapter over one running [`ServeHandle`] + its tokenizer, generic over the
/// executor / KV pool so the same wiring serves every backend. The shared body
/// behind each [`crate::LoadedInferenceEngine`] variant's impl.
pub struct ServeInferenceEngine<E: BackendExecutor, K: KvPool> {
    model_id: String,
    tokenizer: OpenAiTokenizer,
    serve: Arc<ServeHandle<E, K>>,
}

impl<E, K> ServeInferenceEngine<E, K>
where
    E: BackendExecutor + 'static,
    K: KvPool + 'static,
{
    /// Adopt a spawned [`ServeHandle`] plus the matching tokenizer.
    #[must_use]
    pub fn new(model_id: String, tokenizer: OpenAiTokenizer, serve: ServeHandle<E, K>) -> Self {
        Self {
            model_id,
            tokenizer,
            serve: Arc::new(serve),
        }
    }

    /// Shared handle to the running engine, for wiring an HTTP router over an
    /// already-loaded engine ([`crate::LoadedInferenceEngine::local_router`]).
    #[cfg(feature = "cuda")]
    pub(crate) fn serve_arc(&self) -> Arc<ServeHandle<E, K>> {
        Arc::clone(&self.serve)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn tokenizer(&self) -> &OpenAiTokenizer {
        &self.tokenizer
    }

    /// Offload the engine's device weights to host RAM (OPD teacher weight
    /// time-share), returning the device bytes freed. Threads down to the backend
    /// executor on the engine thread via the [`ServeHandle`] control channel.
    pub fn offload_engine_weights(&self) -> Result<usize> {
        self.serve.offload_engine_weights()
    }

    /// Reload the engine's device weights from the host snapshot (OPD teacher
    /// weight time-share).
    pub fn reload_engine_weights(&self) -> Result<()> {
        self.serve.reload_engine_weights()
    }

    /// Release the engine's inference forward scratch WITHOUT offloading weights or
    /// evicting KV (OPD rollout->writeback VRAM reclaim). Threads down to the backend
    /// executor on the engine thread via the [`ServeHandle`] control channel.
    pub fn release_inference_scratch(&self) -> Result<()> {
        self.serve.release_inference_scratch()
    }

    /// Drop the engine's KV pool WITHOUT offloading weights (OPD writeback
    /// headroom: the writeback's fresh autograd forward never reads this engine's
    /// KV). Threads to the backend executor on the engine thread via the
    /// [`ServeHandle`] control channel.
    pub fn release_kv_pool(&self) -> Result<()> {
        self.serve.release_kv_pool()
    }

    /// Re-acquire the KV pool dropped by [`Self::release_kv_pool`].
    pub fn ensure_kv_pool(&self) -> Result<()> {
        self.serve.ensure_kv_pool()
    }

    /// Re-acquire the KV pool, then resume admission only after success.
    pub fn ensure_kv_pool_and_resume_admissions(&self) -> Result<()> {
        self.serve.ensure_kv_pool_and_resume_admissions()
    }

    /// Quiesce engine admission (the serve loop defers new admission) and cancel
    /// every in-flight (waiting + active) request, returning how many were
    /// cancelled. The OPD round-loop writeback bracket; pairs with
    /// [`Self::resume_admissions`].
    pub fn quiesce_admissions(&self) -> Result<usize> {
        self.serve.quiesce_admissions()
    }

    /// Re-arm admission after the OPD writeback bracket (KV pool re-acquired).
    pub fn resume_admissions(&self) -> Result<()> {
        self.serve.resume_admissions()
    }

    /// Generate token ids from an already-tokenized prompt through the serving
    /// scheduler. This is the programmatic OPD rollout surface: unlike
    /// `forward_token_logits`, it keeps one request alive in infer-core, so the
    /// backend uses its normal KV-cache incremental decode path.
    pub fn generate_token_ids(
        &self,
        prompt_token_ids: &[u32],
        max_tokens: usize,
        sampling: infer_plan::SamplingParams,
    ) -> Result<Vec<u32>> {
        if max_tokens == 0 {
            return Ok(Vec::new());
        }
        let ticket = self
            .serve
            .submit(prompt_token_ids.to_vec(), max_tokens, sampling)
            .map_err(|err| anyhow!("request submission failed: {err}"))?;
        let completed: CompletedRequest = ticket.collect()?;
        Ok(completed.generated_tokens)
    }

    /// Batched sibling of [`generate_token_ids`]: submit ALL requests first
    /// (non-blocking) so the `ServeHandle`'s continuous-batching engine thread
    /// decodes them together, then collect each in order. OPD eval/rollout used
    /// submit→collect→submit→collect (one in-flight request at a time), which
    /// left the batcher idle and made decode memory-bandwidth-bound at B=1;
    /// batching amortizes the weight reads across the set. Each `(prompt,
    /// sampling)` pair is an independent request (own KV slot, own seed), so the
    /// per-request outputs are identical to the serial path.
    #[cfg(feature = "cuda")]
    pub fn generate_token_ids_batch(
        &self,
        requests: &[(Vec<u32>, infer_plan::SamplingParams)],
        max_tokens: usize,
    ) -> Result<Vec<Vec<u32>>> {
        if max_tokens == 0 {
            return Ok(vec![Vec::new(); requests.len()]);
        }
        let tickets = requests
            .iter()
            .map(|(prompt, sampling)| {
                self.serve
                    .submit(prompt.clone(), max_tokens, sampling.clone())
                    .map_err(|err| anyhow!("batch request submission failed: {err}"))
            })
            .collect::<Result<Vec<_>>>()?;
        tickets
            .into_iter()
            .map(|ticket| {
                let completed: CompletedRequest = ticket.collect()?;
                Ok(completed.generated_tokens)
            })
            .collect::<Result<Vec<_>>>()
    }

    /// Shared `tokenize -> submit -> collect -> detokenize` body returning the
    /// projected [`CompletionOutput`]; both `complete` and `complete_stream`
    /// build on it.
    fn run(&self, req: &CompletionRequest) -> Result<CompletionOutput> {
        let prompt_token_ids = self
            .tokenizer
            .encode(&req.prompt)
            .map_err(|err| anyhow!("tokenize prompt failed: {err}"))?;

        let ticket = self
            .serve
            .submit(
                prompt_token_ids.clone(),
                req.max_tokens,
                req.sampling.clone(),
            )
            .map_err(|err| anyhow!("request submission failed: {err}"))?;
        let completed: CompletedRequest = ticket.collect()?;
        self.project_completion(prompt_token_ids, req, completed)
    }

    /// Batched sibling of [`run`]: submit ALL completion requests first
    /// (non-blocking) so the continuous-batching engine thread decodes them
    /// together, then collect + post-process each in order. The OPD judge ran
    /// one verdict at a time (submit→collect per rollout), which left the
    /// batcher idle and made the judge decode memory-bandwidth-bound at B=1;
    /// batching amortizes the weight reads across the rollout set. Each result
    /// is byte-for-byte what `run(&req)` would produce for that request.
    #[cfg(feature = "cuda")]
    pub fn complete_batch(&self, reqs: Vec<CompletionRequest>) -> Result<Vec<CompletionOutput>> {
        // Pass 1: tokenize + submit every request without waiting (fills the
        // batcher). Keep the request + its prompt token ids alongside the ticket
        // so pass 2 reproduces `run`'s post-processing exactly.
        let pending = reqs
            .into_iter()
            .map(|req| -> Result<_> {
                let prompt_token_ids = self
                    .tokenizer
                    .encode(&req.prompt)
                    .map_err(|err| anyhow!("tokenize prompt failed: {err}"))?;
                let ticket = self
                    .serve
                    .submit(
                        prompt_token_ids.clone(),
                        req.max_tokens,
                        req.sampling.clone(),
                    )
                    .map_err(|err| anyhow!("request submission failed: {err}"))?;
                Ok((ticket, prompt_token_ids, req))
            })
            .collect::<Result<Vec<_>>>()?;
        // Pass 2: collect each ticket and project to `CompletionOutput` with the
        // same stop-string truncation + finish-reason logic as `run`.
        pending
            .into_iter()
            .map(|(ticket, prompt_token_ids, req)| {
                let completed: CompletedRequest = ticket.collect()?;
                self.project_completion(prompt_token_ids, &req, completed)
            })
            .collect()
    }

    /// Project a collected [`CompletedRequest`] into the public
    /// [`CompletionOutput`]: decode the response, apply host-side stop-string
    /// truncation, and resolve the finish reason. Shared by `run` and
    /// `complete_batch` so both paths produce identical outputs.
    fn project_completion(
        &self,
        prompt_token_ids: Vec<u32>,
        req: &CompletionRequest,
        completed: CompletedRequest,
    ) -> Result<CompletionOutput> {
        let response_token_ids = completed.generated_tokens.clone();
        let raw_text = self
            .tokenizer
            .decode(&response_token_ids)
            .map_err(|err| anyhow!("decode generated tokens failed: {err}"))?;

        // Host-side stop-string truncation (the engine stops on token ids; OpenAI
        // `stop` strings are applied here).
        let (text, stop_truncated) = match req.stop.as_deref() {
            Some(stops) => match truncate_at_first_stop(&raw_text, stops) {
                Some(truncated) => (truncated, true),
                None => (raw_text, false),
            },
            None => (raw_text, false),
        };

        let finish_reason = finish_reason_from(stop_truncated, &completed);

        let prompt_tokens = completed.prompt_tokens.len();
        let completion_tokens = response_token_ids.len();
        Ok(CompletionOutput {
            text,
            finish_reason,
            usage: TokenUsage::new(prompt_tokens, completion_tokens),
            token_logprobs: Vec::new(), // not yet surfaced
            prompt_token_ids,
            response_token_ids,
        })
    }

    fn run_multimodal_chat(&self, req: &MultimodalChatRequest) -> Result<CompletionOutput> {
        if req.messages.is_empty() {
            anyhow::bail!("messages must contain at least one message");
        }
        let multimodal_kind = self
            .serve
            .run_on_executor(|executor| executor.multimodal_kind())
            .unwrap_or(None);
        let images = req
            .messages
            .iter()
            .flat_map(|msg| &msg.images)
            .map(|image| {
                match multimodal_kind {
                    Some(infer_plan::MultimodalKind::DeepseekOcr) => {
                        infer_server::multimodal::preprocess_deepseek_ocr_image(&image.data)
                    }
                    _ => infer_server::multimodal::preprocess_gemma4_image(&image.data),
                }
                .map_err(|err| anyhow!("preprocess image {} failed: {err}", image.source))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let server_messages = server_chat_messages(req.messages.as_slice());
        anyhow::ensure!(
            !images.is_empty(),
            "multimodal chat request must include at least one image"
        );
        // DeepSeek-OCR's chat template is trivial and emits no BOS; rendering it
        // here is what produced the non-stopping `{"text":"image"}…` loop on the
        // in-process path. Build the prompt via the shared BOS-prefixed builder
        // (same source of truth as the HTTP server) instead of `render_chat`.
        let prompt = match multimodal_kind {
            Some(infer_plan::MultimodalKind::DeepseekOcr) => {
                let prompt = infer_server::multimodal::build_deepseek_ocr_prompt(&server_messages);
                infer_server::multimodal::expand_deepseek_ocr_image_markers(&prompt, &images)
            }
            _ => {
                let prompt = self
                    .tokenizer
                    .render_chat(&server_messages)
                    .map_err(|err| anyhow!("render multimodal chat prompt failed: {err}"))?;
                infer_server::multimodal::expand_gemma4_image_markers(&prompt, &images)
            }
        }
        .map_err(|err| anyhow!("expand image prompt markers failed: {err}"))?;
        let prompt_token_ids = self
            .tokenizer
            .encode(&prompt)
            .map_err(|err| anyhow!("tokenize prompt failed: {err}"))?;
        let exec_prompt = prompt_token_ids.clone();
        let exec_images = images;
        let sampling = req.sampling.clone();
        let max_tokens = req.max_tokens;
        let output = self.serve.run_on_executor(move |executor| {
            executor.generate_multimodal(&exec_prompt, &exec_images, max_tokens, &sampling)
        })??;
        let output =
            output.ok_or_else(|| anyhow!("backend does not expose multimodal chat completion"))?;
        let response_token_ids = output.generated_tokens;
        let text = self
            .tokenizer
            .decode(&response_token_ids)
            .map_err(|err| anyhow!("decode generated tokens failed: {err}"))?;
        Ok(CompletionOutput {
            text,
            finish_reason: FinishReason::from_plan(&output.finish),
            usage: TokenUsage::new(prompt_token_ids.len(), response_token_ids.len()),
            token_logprobs: Vec::new(),
            prompt_token_ids,
            response_token_ids,
        })
    }
}

/// OPD-teacher raw-logits surface (CUDA only).
///
/// `forward_token_logits` runs the full `[seq_len, vocab]` teacher forward on the
/// engine-thread-owned [`CudaExecutor`] (no sampling) via the [`ServeHandle`]
/// out-of-band control channel, then returns the device logits as [`RawLogits`].
/// The closure builds `RawLogits` on the engine thread so the device buffer +
/// context cross back to the caller as a single `Send` value.
#[cfg(feature = "cuda")]
impl ServeInferenceEngine<infer_cuda::CudaExecutor, infer_cuda::CudaKvPool> {
    pub fn forward_token_logits(
        &self,
        input_ids: &[u32],
        positions: &[u32],
    ) -> Result<crate::types::RawLogits> {
        let input_ids = input_ids.to_vec();
        let positions = positions.to_vec();
        self.serve.run_on_executor(move |executor| {
            let (logits, shape, device) = executor.forward_token_logits(&input_ids, &positions)?;
            Ok(crate::types::RawLogits {
                logits,
                shape,
                device,
            })
        })?
    }

    /// Trunk taps + final hidden states for offline DSpark draft training.
    /// Runs on the engine thread like `forward_token_logits`; the results are
    /// host `Vec<f32>`, so nothing device-bound crosses back.
    pub fn forward_training_taps(
        &self,
        input_ids: &[u32],
        target_layer_ids: &[i64],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let input_ids = input_ids.to_vec();
        let target_layer_ids = target_layer_ids.to_vec();
        self.serve.run_on_executor(move |executor| {
            executor.forward_training_taps(&input_ids, &target_layer_ids)
        })?
    }

    /// Fold a fresh student LoRA update into the resident projection weights
    /// (OPD per-step re-merge), then drop the now-stale prefix cache. Runs both
    /// on the engine-thread-owned [`Engine`] via the [`ServeHandle`] out-of-band
    /// control channel, so the resident weight mutation never races an in-flight
    /// forward step.
    ///
    /// The re-merge changes resident weights, so cached hidden/KV values are now
    /// prior-epoch values; serving a post-merge request from that cache is a
    /// silent correctness bug. Both steps run in **one** control closure so no
    /// scheduler step can interleave between the weight change and the cache
    /// drop.
    pub fn remerge_student_lora(&self, update: infer_cuda::StudentLoraUpdate) -> Result<()> {
        self.serve.run_on_engine(move |engine| {
            engine.executor_mut().remerge_student_lora(update)?;
            engine.invalidate_prefix_cache();
            Ok(())
        })?
    }

    /// Read-only borrow of resident FP8 block-scaled base projection pointers
    /// for train-infer weight sharing (`--share-frozen-base`). Runs on the
    /// engine thread via the control seam (exclusive `&mut E`) and returns the
    /// pointer table — raw `u64` device pointers + dims, all `Send`. The borrow
    /// is read-only; resident weights are not mutated, so no prefix-cache drop.
    pub fn frozen_base_fp8_pointers(&self) -> Result<Vec<infer_cuda::SharedFp8BaseProjection>> {
        self.serve
            .run_on_executor(|executor| executor.frozen_base_fp8_pointers())?
    }

    /// Non-owning views of every resident dense-BF16 base projection's device
    /// pointer, for refreshing the train student's frozen base AFTER a LoRA
    /// re-merge.
    pub fn frozen_base_bf16_pointers(&self) -> Result<Vec<infer_cuda::SharedBf16BaseProjection>> {
        self.serve
            .run_on_executor(|executor| executor.frozen_base_bf16_pointers())?
    }

    /// Free the retired FP8 qweight/scales buffers for every projection that
    /// has been promoted to dense BF16. Call ONLY after the train student has
    /// re-aliased its frozen base to the BF16 `data` pointer.
    pub fn free_retired_fp8_buffers(&self) {
        let _ = self.serve.run_on_executor(|executor| -> Result<()> {
            executor.free_retired_fp8_buffers();
            Ok(())
        });
    }

    /// Hot-swap the DSpark Markov head weights from a host f32 snapshot, then
    /// drop the now-stale prefix cache. Runs on the engine thread via the
    /// control seam so the resident weight mutation never races an in-flight
    /// forward step.
    pub fn update_dspark_markov_weights(&self, w1: Vec<f32>, w2: Vec<f32>) -> Result<()> {
        self.serve.run_on_engine(move |engine| {
            engine
                .executor_mut()
                .update_dspark_markov_weights(&w1, &w2)?;
            engine.invalidate_prefix_cache();
            Ok(())
        })?
    }
}

// `Send` holds via the auto-trait: `ServeInferenceEngine` stores only a
// `ServeHandle<E, K>` (Send channel + JoinHandle + PhantomData), never an `E`/`K`
// value, so a `!Send` executor (e.g. MLX `MetalExecutor`, which lives on the
// engine thread) is fine.
impl<E, K> InferenceEngine for ServeInferenceEngine<E, K>
where
    E: BackendExecutor + 'static,
    K: KvPool + 'static,
{
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn complete(&mut self, req: CompletionRequest) -> Result<CompletionOutput> {
        self.run(&req)
    }

    fn complete_multimodal_chat(&mut self, req: MultimodalChatRequest) -> Result<CompletionOutput> {
        self.run_multimodal_chat(&req)
    }

    fn complete_stream(
        &mut self,
        req: CompletionRequest,
        tx: UnboundedSender<CompletionStreamDelta>,
    ) -> Result<()> {
        let prompt_token_ids = match self.tokenizer.encode(&req.prompt) {
            Ok(ids) => ids,
            Err(err) => {
                let _ = tx.send(CompletionStreamDelta::error(
                    "tokenize_failed",
                    vec![err.to_string()],
                ));
                return Ok(());
            }
        };
        let (ticket, stream_rx) = match self.serve.submit_streaming(
            prompt_token_ids.clone(),
            req.max_tokens,
            req.sampling.clone(),
        ) {
            Ok(v) => v,
            Err(err) => {
                let _ = tx.send(CompletionStreamDelta::error(
                    "inference_failed",
                    vec![err.to_string()],
                ));
                return Ok(());
            }
        };

        // Stream tokens live. Decode the accumulated ids each step (so multi-byte
        // / BPE-merge boundaries resolve correctly) and emit only the newly-stable
        // prefix, holding back a tail that could be the start of a stop string.
        let holdback = req.stop.as_deref().map_or(0, stop_holdback);
        let mut acc_ids: Vec<u32> = Vec::new();
        // Cursor into acc_ids: ids already reported on a prior delta. Every
        // emitted delta carries the ids committed since the last report, so the
        // union of streamed token_ids equals generated_tokens with no duplicate
        // (held-back / non-boundary tokens ride the delta that finally flushes).
        let mut reported_upto = 0usize;
        let mut emitted = String::new();
        let mut completed = None;
        while let Ok(item) = stream_rx.recv() {
            match item {
                StreamItem::Token { token, .. } => {
                    acc_ids.push(token);
                    let full = self.tokenizer.decode(&acc_ids).unwrap_or_default();
                    if let Some(delta) = deliverable_delta(&full, &emitted, holdback) {
                        emitted.push_str(&delta);
                        let token_ids = acc_ids[reported_upto..].to_vec();
                        reported_upto = acc_ids.len();
                        let _ = tx.send(CompletionStreamDelta {
                            text_delta: delta,
                            finish_reason: None,
                            usage: None,
                            logprob: None,
                            token_ids,
                            error: None,
                        });
                    }
                }
                StreamItem::Done(c) => {
                    completed = Some(c);
                    break;
                }
            }
        }

        // Terminal: re-decode the full output, apply host-side stop truncation,
        // flush any text not yet emitted (the held-back tail, up to the stop),
        // then a finish delta with the reason + usage.
        let completed = match completed {
            Some(c) => c,
            None => ticket.collect()?,
        };
        let full = self
            .tokenizer
            .decode(&completed.generated_tokens)
            .unwrap_or_default();
        let (final_text, stop_truncated) = match req.stop.as_deref() {
            Some(stops) => match truncate_at_first_stop(&full, stops) {
                Some(truncated) => (truncated, true),
                None => (full, false),
            },
            None => (full, false),
        };
        // The held-back tail's ids were never reported (their bytes stayed inside
        // the holdback window); attach them to the flush so the streamed token_ids
        // union equals generated_tokens.
        let remaining_ids = acc_ids[reported_upto..].to_vec();
        let text_delta = if final_text.len() > emitted.len() && final_text.starts_with(&emitted) {
            final_text[emitted.len()..].to_string()
        } else {
            String::new()
        };
        if !text_delta.is_empty() || !remaining_ids.is_empty() {
            let _ = tx.send(CompletionStreamDelta {
                text_delta,
                finish_reason: None,
                usage: None,
                logprob: None,
                token_ids: remaining_ids,
                error: None,
            });
        }
        let finish_reason = finish_reason_from(stop_truncated, &completed);
        let prompt_tokens = completed.prompt_tokens.len();
        let completion_tokens = completed.generated_tokens.len();
        let _ = tx.send(CompletionStreamDelta {
            text_delta: String::new(),
            finish_reason: Some(finish_reason),
            usage: Some(TokenUsage::new(prompt_tokens, completion_tokens)),
            logprob: None,
            token_ids: Vec::new(),
            error: None,
        });
        Ok(())
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text)
            .map_err(|err| anyhow!("tokenize failed: {err}"))
    }

    fn render_chat_prompt(&self, messages: &[ChatPromptMessage]) -> Result<String> {
        let rows = server_chat_messages(messages);
        self.tokenizer.render_chat(&rows)
    }

    fn telemetry(&self) -> EngineTelemetry {
        // Real scheduler occupancy from the engine's live counters. Latency /
        // batch-occupancy / spec metrics need per-request timestamps the engine
        // does not track yet, so they stay at their "unavailable" defaults.
        let counters = self.serve.counters();
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64);
        EngineTelemetry {
            queue_depth: counters.queue_depth as u32,
            active_requests: counters.active_requests as u32,
            timestamp_ms,
            ..EngineTelemetry::default()
        }
    }
}

fn server_chat_messages(messages: &[ChatPromptMessage]) -> Vec<infer_server::ChatMessage> {
    messages
        .iter()
        .map(|message| infer_server::ChatMessage {
            role: message.role.clone(),
            content: Some(server_chat_content(message)),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        })
        .collect()
}

fn server_chat_content(message: &ChatPromptMessage) -> infer_server::ChatContent {
    if message.images.is_empty() {
        return infer_server::ChatContent::Text(message.content.clone());
    }
    let parts = (!message.content.is_empty())
        .then(|| infer_server::ChatContentPart {
            kind: "text".to_string(),
            text: Some(message.content.clone()),
            image_url: None,
            input_image: None,
            extra: serde_json::Map::new(),
        })
        .into_iter()
        .chain(
            message
                .images
                .iter()
                .map(|_| infer_server::ChatContentPart {
                    kind: "image".to_string(),
                    text: None,
                    image_url: None,
                    input_image: None,
                    extra: serde_json::Map::new(),
                }),
        )
        .collect();
    infer_server::ChatContent::Parts(parts)
}

/// Truncate at the first occurrence of any non-empty stop string, returning the
/// prefix before it (or `None` if none matched).
fn truncate_at_first_stop(text: &str, stops: &[String]) -> Option<String> {
    let pos = stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|stop| text.find(stop.as_str()))
        .min()?;
    Some(text[..pos].to_string())
}

/// Map the host-side stop-truncation outcome + engine finish to the public
/// [`FinishReason`]: an applied stop string wins (`Stop`), else the engine's
/// finish (`Length` when none). Shared by `run` and `complete_stream`.
fn finish_reason_from(stop_truncated: bool, completed: &CompletedRequest) -> FinishReason {
    if stop_truncated {
        FinishReason::Stop
    } else {
        completed
            .finish
            .as_ref()
            .map_or(FinishReason::Length, FinishReason::from_plan)
    }
}

/// Bytes to hold back from the live stream so a stop string spanning token
/// boundaries is detected before any of its prefix is emitted: one less than the
/// longest stop (0 when there are no stops).
fn stop_holdback(stops: &[String]) -> usize {
    stops
        .iter()
        .map(String::len)
        .max()
        .unwrap_or(0)
        .saturating_sub(1)
}

/// The newly-stable text to emit given the full decode so far, what has already
/// been emitted, and the stop-boundary `holdback`. Returns the slice of `full`
/// after `emitted` up to `full.len() - holdback` (rounded to a char boundary), or
/// `None` if nothing new is stable yet. Returns `None` if `full` no longer starts
/// with `emitted` (a rare retroactive decode change — wait for the next token).
fn deliverable_delta(full: &str, emitted: &str, holdback: usize) -> Option<String> {
    if !full.starts_with(emitted) {
        return None;
    }
    let mut end = full.len().saturating_sub(holdback).max(emitted.len());
    while end > emitted.len() && !full.is_char_boundary(end) {
        end -= 1;
    }
    if end <= emitted.len() {
        return None;
    }
    Some(full[emitted.len()..end].to_string())
}

/// Derive a model id from the final path segment of a model path or HF id.
/// Used by the backend variants; unused on a no-backend lib build.
#[must_use]
#[cfg_attr(
    not(any(
        feature = "metal",
        feature = "cuda",
        feature = "hip",
        feature = "vulkan",
        feature = "cpu"
    )),
    allow(dead_code)
)]
pub(crate) fn model_id_from_path(model_path: &str) -> String {
    std::path::Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::{deliverable_delta, model_id_from_path, stop_holdback, truncate_at_first_stop};

    #[test]
    fn truncate_at_first_stop_picks_earliest() {
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
        assert_eq!(
            truncate_at_first_stop("a\n\nbEND", &stops),
            Some("a".to_string())
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
    fn stop_holdback_is_longest_stop_minus_one() {
        assert_eq!(stop_holdback(&[]), 0);
        assert_eq!(stop_holdback(&["END".into(), "\n\n".into()]), 2); // max len 3 -> 2
        assert_eq!(stop_holdback(&["x".into()]), 0);
    }

    #[test]
    fn deliverable_delta_emits_new_stable_prefix() {
        // No holdback: every newly-decoded char is deliverable immediately.
        assert_eq!(deliverable_delta("Paris", "", 0).as_deref(), Some("Paris"));
        assert_eq!(
            deliverable_delta("Paris.", "Paris", 0).as_deref(),
            Some(".")
        );
        // Nothing new beyond what's emitted.
        assert_eq!(deliverable_delta("Paris", "Paris", 0), None);
    }

    #[test]
    fn deliverable_delta_holds_back_stop_boundary() {
        // holdback=2 ("END" stop): hold back the last 2 bytes so "EN" of a
        // potential "END" is never emitted before the full stop is seen.
        assert_eq!(deliverable_delta("hello", "", 2).as_deref(), Some("hel"));
        // After "hel" emitted, "hello" still holds back "lo".
        assert_eq!(deliverable_delta("hello", "hel", 2), None);
        // The completion grows; "lo" becomes safe once 2 more bytes follow
        // ("XY" is now the held-back tail).
        assert_eq!(
            deliverable_delta("helloXY", "hel", 2).as_deref(),
            Some("lo")
        );
    }

    #[test]
    fn deliverable_delta_waits_on_retroactive_change() {
        // `full` no longer starts with `emitted` (a decode that changed under us):
        // emit nothing and wait for the next token.
        assert_eq!(deliverable_delta("Pari", "Paris", 0), None);
    }
}
