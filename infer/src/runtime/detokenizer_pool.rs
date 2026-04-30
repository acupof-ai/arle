#![allow(unreachable_pub)]
#![allow(warnings)]
/*!
 * Parallel detokenizer pool for incremental token-to-text conversion
 */

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use tokio::task::JoinHandle;

use super::channels::{DetokenizeTask, RequestId};
use crate::server_engine::{CompletionStreamDelta, TokenUsage};
use crate::tokenizer::Tokenizer;

/// Pool of detokenizer workers for parallel token-to-text conversion
pub struct DetokenizerPool {
    workers: Vec<JoinHandle<()>>,
    task_sender: mpsc::Sender<DetokenizeTask>,
    worker_count: usize,
    shutdown_tx: Option<mpsc::Sender<()>>,
    // Per-request decode state for incremental decoding
    decode_states: Arc<Mutex<HashMap<RequestId, DecodeState>>>,
}

/// Incremental decode state for each request
#[derive(Debug)]
struct DecodeState {
    /// All tokens generated so far for this request
    accumulated_tokens: Vec<u32>,
    /// Text generated so far
    accumulated_text: String,
    /// Partial token buffer for incomplete UTF-8 sequences
    partial_buffer: Vec<u8>,
    /// Token count for usage tracking
    token_count: usize,
}

impl DecodeState {
    fn new() -> Self {
        Self {
            accumulated_tokens: Vec::new(),
            accumulated_text: String::new(),
            partial_buffer: Vec::new(),
            token_count: 0,
        }
    }
}

impl DetokenizerPool {
    pub async fn new(
        tokenizer: Arc<Tokenizer>,
        worker_count: usize,
        task_receiver: mpsc::Receiver<DetokenizeTask>,
    ) -> Result<Self> {
        let (task_sender, _) = mpsc::channel(1024);
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);

        // Shared task receiver and decode state
        let task_receiver = Arc::new(Mutex::new(task_receiver));
        let shutdown_rx = Arc::new(Mutex::new(shutdown_rx));
        let decode_states = Arc::new(Mutex::new(HashMap::new()));

        let mut workers = Vec::new();

        for worker_id in 0..worker_count {
            let worker = DetokenizerWorker::new(
                worker_id,
                tokenizer.clone(),
                task_receiver.clone(),
                shutdown_rx.clone(),
                decode_states.clone(),
            );

            let handle = tokio::spawn(async move {
                worker.run().await;
            });

            workers.push(handle);
        }

        log::info!("Created detokenizer pool with {} workers", worker_count);

        Ok(Self {
            workers,
            task_sender,
            worker_count,
            shutdown_tx: Some(shutdown_tx),
            decode_states,
        })
    }

    pub async fn start(&self) -> Result<()> {
        log::info!(
            "Detokenizer pool started with {} workers",
            self.worker_count
        );
        Ok(())
    }

    pub fn queue_depth(&self) -> usize {
        // Approximate queue depth
        0
    }

    pub async fn shutdown(&self) -> Result<()> {
        if let Some(shutdown_tx) = &self.shutdown_tx {
            let _ = shutdown_tx.send(()).await;
        }

        // Clean up decode states
        self.decode_states.lock().await.clear();

        // Wait for all workers to complete
        for handle in &self.workers {
            if !handle.is_finished() {
                handle.abort();
            }
        }

        log::info!("Detokenizer pool shutdown complete");
        Ok(())
    }
}

/// Individual detokenizer worker
struct DetokenizerWorker {
    worker_id: usize,
    tokenizer: Arc<Tokenizer>,
    task_receiver: Arc<Mutex<mpsc::Receiver<DetokenizeTask>>>,
    shutdown_receiver: Arc<Mutex<mpsc::Receiver<()>>>,
    decode_states: Arc<Mutex<HashMap<RequestId, DecodeState>>>,
}

impl DetokenizerWorker {
    fn new(
        worker_id: usize,
        tokenizer: Arc<Tokenizer>,
        task_receiver: Arc<Mutex<mpsc::Receiver<DetokenizeTask>>>,
        shutdown_receiver: Arc<Mutex<mpsc::Receiver<()>>>,
        decode_states: Arc<Mutex<HashMap<RequestId, DecodeState>>>,
    ) -> Self {
        Self {
            worker_id,
            tokenizer,
            task_receiver,
            shutdown_receiver,
            decode_states,
        }
    }

    async fn run(&self) {
        log::debug!("Detokenizer worker {} starting", self.worker_id);

        loop {
            tokio::select! {
                // Check for shutdown signal
                _ = async {
                    let mut shutdown_rx = self.shutdown_receiver.lock().await;
                    shutdown_rx.recv().await
                } => {
                    log::debug!("Detokenizer worker {} received shutdown signal", self.worker_id);
                    break;
                }

                // Process detokenization tasks
                task = async {
                    let mut task_rx = self.task_receiver.lock().await;
                    task_rx.recv().await
                } => {
                    match task {
                        Some(task) => {
                            if let Err(e) = self.process_task(task).await {
                                log::error!("Detokenizer worker {} error: {}", self.worker_id, e);
                            }
                        }
                        None => {
                            log::debug!("Detokenizer worker {} task channel closed", self.worker_id);
                            break;
                        }
                    }
                }
            }
        }

        log::debug!("Detokenizer worker {} stopping", self.worker_id);
    }

    async fn process_task(&self, task: DetokenizeTask) -> Result<()> {
        let start = std::time::Instant::now();

        // Get or create decode state for this request
        let mut states = self.decode_states.lock().await;
        let state = states
            .entry(task.request_id)
            .or_insert_with(DecodeState::new);

        // Add new tokens to accumulated tokens
        state.accumulated_tokens.extend(&task.new_tokens);
        state.token_count += task.new_tokens.len();

        // Perform incremental decoding
        let delta = self.decode_incremental(state, &task.new_tokens).await?;

        let finish_reason = if task.finished {
            // Clean up state for finished requests
            states.remove(&task.request_id);
            task.finish_reason
        } else {
            None
        };

        drop(states); // Release the lock

        // Create completion delta
        let completion_delta = CompletionStreamDelta {
            text_delta: delta.text_delta,
            finish_reason,
            usage: if task.finished {
                Some(TokenUsage {
                    prompt_tokens: delta.prompt_tokens,
                    completion_tokens: delta.completion_tokens,
                    total_tokens: delta.prompt_tokens + delta.completion_tokens,
                })
            } else {
                None
            },
            logprob: delta.logprob,
            token_ids: task.new_tokens.clone(),
        };

        // Save values before move
        let tokens_len = task.new_tokens.len();

        // Send response
        if let Err(e) = task.response_tx.send(completion_delta) {
            log::debug!("Worker {} client disconnected: {}", self.worker_id, e);
        }

        let duration = start.elapsed();

        log::trace!(
            "Worker {} detokenized {} tokens in {:?}",
            self.worker_id,
            tokens_len,
            duration
        );

        Ok(())
    }

    async fn decode_incremental(
        &self,
        state: &mut DecodeState,
        new_tokens: &[u32],
    ) -> Result<IncrementalDelta> {
        // Decode only the new tokens to get the delta text
        let new_text = if !new_tokens.is_empty() {
            match self.tokenizer.decode(new_tokens) {
                Ok(text) => {
                    // Handle potential UTF-8 boundary issues
                    self.handle_utf8_boundaries(state, text)?
                }
                Err(e) => {
                    log::warn!("Worker {} decode error: {}", self.worker_id, e);
                    String::new()
                }
            }
        } else {
            String::new()
        };

        // Update accumulated text
        state.accumulated_text.push_str(&new_text);

        // Check for stop sequences
        let (final_text, _stopped) = self.check_stop_sequences(&new_text, &state.accumulated_text);

        Ok(IncrementalDelta {
            text_delta: final_text,
            prompt_tokens: 0, // This would be set by the scheduler
            completion_tokens: state.token_count,
            logprob: None, // Would be populated if logprobs requested
        })
    }

    fn handle_utf8_boundaries(&self, state: &mut DecodeState, mut text: String) -> Result<String> {
        // Handle partial UTF-8 sequences at token boundaries
        // This is a simplified version - real implementation would be more robust

        if !state.partial_buffer.is_empty() {
            // Prepend partial buffer to new text bytes
            let mut combined = state.partial_buffer.clone();
            combined.extend(text.as_bytes());

            match String::from_utf8(combined) {
                Ok(combined_text) => {
                    state.partial_buffer.clear();
                    text = combined_text;
                }
                Err(_) => {
                    // Still incomplete, keep accumulating
                    state.partial_buffer.extend(text.as_bytes());
                    return Ok(String::new());
                }
            }
        }

        // Check if current text ends with incomplete UTF-8
        let bytes = text.as_bytes();
        if let Some(last_char_start) = text.char_indices().next_back() {
            let (pos, _) = last_char_start;
            if pos < bytes.len() - 1 {
                // Potential incomplete character at the end
                state.partial_buffer = bytes[pos..].to_vec();
                text.truncate(pos);
            }
        }

        Ok(text)
    }

    fn check_stop_sequences(&self, new_text: &str, accumulated_text: &str) -> (String, bool) {
        // Simple stop sequence checking
        // In practice, this would be more sophisticated and configurable

        let common_stops = ["\n\n", "```", "<|endoftext|>", "<|end|>"];

        for stop in &common_stops {
            if let Some(pos) = accumulated_text.rfind(stop) {
                if pos + stop.len() >= accumulated_text.len() - new_text.len() {
                    // Stop sequence appears in the new text
                    let stop_pos_in_new =
                        pos + stop.len() - (accumulated_text.len() - new_text.len());
                    if stop_pos_in_new < new_text.len() {
                        return (new_text[..stop_pos_in_new].to_string(), true);
                    }
                }
            }
        }

        (new_text.to_string(), false)
    }
}

/// Result of incremental decoding
#[derive(Debug)]
struct IncrementalDelta {
    text_delta: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    logprob: Option<f32>,
}

/// Statistics for detokenizer pool monitoring
#[derive(Debug, Clone)]
pub struct DetokenizerStats {
    pub worker_count: usize,
    pub active_requests: usize,
    pub tasks_processed: u64,
    pub avg_processing_time_ms: f64,
    pub queue_depth: usize,
}

impl DetokenizerStats {
    pub fn new() -> Self {
        Self {
            worker_count: 0,
            active_requests: 0,
            tasks_processed: 0,
            avg_processing_time_ms: 0.0,
            queue_depth: 0,
        }
    }
}
