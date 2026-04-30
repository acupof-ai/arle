/*!
 * Multi-threaded inference runtime infrastructure
 *
 * This module provides a Rust-native multi-threaded alternative to SGLang's
 * multi-process architecture, leveraging Rust's ownership system for safe
 * concurrency without Python's GIL limitations.
 */

mod backend_worker;
mod channels;
mod config;
mod detokenizer_pool;
mod error_recovery;
mod gpu_executor;
mod http_integration;
mod memory_pool_coordinator;
mod scheduler_actor;
mod thread_safe_radix_cache;
mod tokenizer_pool;

#[cfg(test)]
mod tests;

pub use channels::RuntimeChannels;
pub use config::{ChannelConfig, RuntimeConfig, RuntimeMode, ThreadingConfig};

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::backend::InferenceBackend;
use crate::server_engine::{
    CompletionOutput, CompletionRequest, CompletionStreamDelta, InferenceEngine,
};

use self::detokenizer_pool::DetokenizerPool;
use self::gpu_executor::GpuExecutorPool;
use self::scheduler_actor::SchedulerActor;
use self::tokenizer_pool::TokenizerPool;

/// Multi-threaded inference runtime that coordinates specialized thread roles
/// for high-throughput LLM inference
pub struct MultiThreadRuntime {
    // Core thread actors
    tokenizer_pool: Arc<TokenizerPool>,
    scheduler: Arc<SchedulerActor>,
    gpu_executor: Arc<GpuExecutorPool>,
    detokenizer_pool: Arc<DetokenizerPool>,

    // Communication infrastructure
    channels: RuntimeChannels,
    config: RuntimeConfig,

    // Model information
    model_id: String,
}

impl MultiThreadRuntime {
    pub async fn new(
        backend: Box<dyn InferenceBackend>,
        tokenizer: Arc<crate::tokenizer::Tokenizer>,
        config: RuntimeConfig,
    ) -> Result<Self> {
        let model_id = backend.model_id().to_string();

        // Create communication channels
        let channels = RuntimeChannels::new(&config.channel_config)?;

        // Initialize thread pools and actors
        let tokenizer_pool = Arc::new(
            TokenizerPool::new(
                tokenizer.clone(),
                config.threading.tokenizer_workers,
                channels.tokenizer_tx.clone(),
            )
            .await?,
        );

        let detokenizer_pool = Arc::new(
            DetokenizerPool::new(
                tokenizer.clone(),
                config.threading.detokenizer_workers,
                channels.detokenizer_rx.clone(),
            )
            .await?,
        );

        let gpu_executor = Arc::new(
            GpuExecutorPool::new(
                backend,
                config.threading.gpu_workers_per_device,
                channels.execute_rx.clone(),
                channels.gpu_result_tx.clone(),
            )
            .await?,
        );

        let scheduler = Arc::new(
            SchedulerActor::new(
                config.clone(),
                channels.schedule_rx.clone(),
                channels.execute_tx.clone(),
                channels.gpu_result_rx.clone(),
                channels.detokenizer_tx.clone(),
            )
            .await?,
        );

        Ok(Self {
            tokenizer_pool,
            scheduler,
            gpu_executor,
            detokenizer_pool,
            channels,
            config,
            model_id,
        })
    }

    /// Start all background actors and thread pools
    pub async fn start(&self) -> Result<()> {
        // Start thread pools
        self.tokenizer_pool.start().await?;
        self.detokenizer_pool.start().await?;
        self.gpu_executor.start().await?;

        // Start scheduler actor
        self.scheduler.start().await?;

        log::info!(
            "Multi-threaded runtime started with {} tokenizer workers, {} GPU workers, {} detokenizer workers",
            self.config.threading.tokenizer_workers,
            self.config.threading.gpu_workers_per_device,
            self.config.threading.detokenizer_workers
        );

        Ok(())
    }

    /// Gracefully shutdown all components
    pub async fn shutdown(&self) -> Result<()> {
        log::info!("Shutting down multi-threaded runtime");

        // Stop scheduler first to prevent new work
        self.scheduler.shutdown().await?;

        // Stop thread pools
        self.gpu_executor.shutdown().await?;
        self.detokenizer_pool.shutdown().await?;
        self.tokenizer_pool.shutdown().await?;

        Ok(())
    }

    /// Submit a completion request to the runtime
    pub async fn submit_request(
        &self,
        req: CompletionRequest,
    ) -> Result<mpsc::UnboundedReceiver<CompletionStreamDelta>> {
        let (response_tx, response_rx) = mpsc::unbounded_channel();

        let task = channels::TokenizeTask {
            request_id: uuid::Uuid::new_v4(),
            request: req,
            response_tx,
        };

        self.tokenizer_pool.submit(task).await?;

        Ok(response_rx)
    }

    /// Get runtime statistics
    pub fn stats(&self) -> RuntimeStats {
        RuntimeStats {
            tokenizer_queue_depth: self.tokenizer_pool.queue_depth(),
            scheduler_pending_requests: self.scheduler.pending_count(),
            gpu_active_batches: self.gpu_executor.active_count(),
            detokenizer_queue_depth: self.detokenizer_pool.queue_depth(),
        }
    }
}

// TODO: Fix thread safety issues before enabling InferenceEngine implementation
/*
impl InferenceEngine for MultiThreadRuntime {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn complete(&mut self, req: CompletionRequest) -> Result<CompletionOutput> {
        // For blocking complete, we create a temporary runtime and block on async
        let rt = tokio::runtime::Handle::current();

        rt.block_on(async {
            let mut stream = self.submit_request(req).await?;

            let mut output = CompletionOutput {
                text: String::new(),
                finish_reason: crate::server_engine::FinishReason::Length,
                usage: crate::server_engine::TokenUsage {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    total_tokens: 0,
                },
                token_logprobs: Vec::new(),
                prompt_token_ids: Vec::new(),
                response_token_ids: Vec::new(),
            };

            while let Some(delta) = stream.recv().await {
                output.text.push_str(&delta.text_delta);
                output.response_token_ids.extend(delta.token_ids);

                if let Some(finish_reason) = delta.finish_reason {
                    output.finish_reason = finish_reason;
                    break;
                }

                if let Some(usage) = delta.usage {
                    output.usage = usage;
                }
            }

            Ok(output)
        })
    }

    fn complete_stream(
        &mut self,
        req: CompletionRequest,
        tx: tokio::sync::mpsc::UnboundedSender<CompletionStreamDelta>,
    ) -> Result<()> {
        let rt = tokio::runtime::Handle::current();

        rt.block_on(async {
            let mut stream = self.submit_request(req).await?;

            while let Some(delta) = stream.recv().await {
                if tx.send(delta).is_err() {
                    // Receiver dropped, client disconnected
                    break;
                }
            }

            Ok(())
        })
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        // Delegate to tokenizer pool for consistency
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async { self.tokenizer_pool.tokenize_sync(text).await })
    }
}
*/

/// Runtime performance and utilization statistics
#[derive(Debug, Clone)]
pub struct RuntimeStats {
    pub tokenizer_queue_depth: usize,
    pub scheduler_pending_requests: usize,
    pub gpu_active_batches: usize,
    pub detokenizer_queue_depth: usize,
}
