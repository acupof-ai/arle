/*!
 * Parallel tokenizer pool for CPU-bound text processing
 */

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use uuid::Uuid;

use super::channels::{RequestId, ScheduleTask, TokenizeTask};
use crate::server_engine::CompletionRequest;
use crate::tokenizer::Tokenizer;

/// Pool of tokenizer workers for parallel text processing
pub struct TokenizerPool {
    workers: Vec<JoinHandle<()>>,
    task_sender: mpsc::Sender<TokenizeTask>,
    schedule_sender: mpsc::Sender<ScheduleTask>,
    worker_count: usize,
    shutdown_tx: Option<mpsc::Sender<()>>,
}

impl TokenizerPool {
    pub async fn new(
        tokenizer: Arc<Tokenizer>,
        worker_count: usize,
        schedule_sender: mpsc::Sender<ScheduleTask>,
    ) -> Result<Self> {
        let (task_sender, task_receiver) = mpsc::channel(1024);
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);

        // Shared task receiver among workers
        let task_receiver = Arc::new(tokio::sync::Mutex::new(task_receiver));
        let shutdown_rx = Arc::new(tokio::sync::Mutex::new(shutdown_rx));

        let mut workers = Vec::new();

        for worker_id in 0..worker_count {
            let worker = TokenizerWorker::new(
                worker_id,
                tokenizer.clone(),
                task_receiver.clone(),
                schedule_sender.clone(),
                shutdown_rx.clone(),
            );

            let handle = tokio::spawn(async move {
                worker.run().await;
            });

            workers.push(handle);
        }

        log::info!("Created tokenizer pool with {} workers", worker_count);

        Ok(Self {
            workers,
            task_sender,
            schedule_sender,
            worker_count,
            shutdown_tx: Some(shutdown_tx),
        })
    }

    pub async fn start(&self) -> Result<()> {
        log::info!("Tokenizer pool started with {} workers", self.worker_count);
        Ok(())
    }

    pub async fn submit(&self, task: TokenizeTask) -> Result<()> {
        self.task_sender
            .send(task)
            .await
            .map_err(|_| anyhow::anyhow!("Tokenizer pool shutdown"))?;
        Ok(())
    }

    pub async fn tokenize_sync(&self, text: &str) -> Result<Vec<u32>> {
        // For sync tokenization, we can use the tokenizer directly
        // This bypasses the worker pool for efficiency
        match &self.worker_count {
            0 => Err(anyhow::anyhow!("No tokenizer workers available")),
            _ => {
                // We need access to the tokenizer, but it's owned by workers
                // For now, create a temporary tokenizer instance
                // In practice, we'd either expose a direct tokenize method or use a shared tokenizer

                // Simple direct tokenization - this is a temporary solution
                // The actual tokenizer is held by the workers, so we can't access it directly here
                // For a proper implementation, we'd either:
                // 1. Keep a separate tokenizer instance for sync calls
                // 2. Use a worker-based approach with result collection
                // 3. Expose a direct tokenize method on the pool

                Err(anyhow::anyhow!(
                    "Sync tokenization requires refactoring - use async submit instead"
                ))
            }
        }
    }

    pub fn queue_depth(&self) -> usize {
        // Approximate queue depth - in practice we'd track this
        0
    }

    pub async fn shutdown(&self) -> Result<()> {
        if let Some(shutdown_tx) = &self.shutdown_tx {
            let _ = shutdown_tx.send(()).await;
        }

        // Wait for all workers to complete
        for handle in &self.workers {
            if !handle.is_finished() {
                handle.abort();
            }
        }

        log::info!("Tokenizer pool shutdown complete");
        Ok(())
    }
}

/// Individual tokenizer worker
struct TokenizerWorker {
    worker_id: usize,
    tokenizer: Arc<Tokenizer>,
    task_receiver: Arc<tokio::sync::Mutex<mpsc::Receiver<TokenizeTask>>>,
    schedule_sender: mpsc::Sender<ScheduleTask>,
    shutdown_receiver: Arc<tokio::sync::Mutex<mpsc::Receiver<()>>>,
}

impl TokenizerWorker {
    fn new(
        worker_id: usize,
        tokenizer: Arc<Tokenizer>,
        task_receiver: Arc<tokio::sync::Mutex<mpsc::Receiver<TokenizeTask>>>,
        schedule_sender: mpsc::Sender<ScheduleTask>,
        shutdown_receiver: Arc<tokio::sync::Mutex<mpsc::Receiver<()>>>,
    ) -> Self {
        Self {
            worker_id,
            tokenizer,
            task_receiver,
            schedule_sender,
            shutdown_receiver,
        }
    }

    async fn run(&self) {
        log::debug!("Tokenizer worker {} starting", self.worker_id);

        loop {
            tokio::select! {
                // Check for shutdown signal
                _ = async {
                    let mut shutdown_rx = self.shutdown_receiver.lock().await;
                    shutdown_rx.recv().await
                } => {
                    log::debug!("Tokenizer worker {} received shutdown signal", self.worker_id);
                    break;
                }

                // Process tokenization tasks
                task = async {
                    let mut task_rx = self.task_receiver.lock().await;
                    task_rx.recv().await
                } => {
                    match task {
                        Some(task) => {
                            if let Err(e) = self.process_task(task).await {
                                log::error!("Tokenizer worker {} error: {}", self.worker_id, e);
                            }
                        }
                        None => {
                            log::debug!("Tokenizer worker {} task channel closed", self.worker_id);
                            break;
                        }
                    }
                }
            }
        }

        log::debug!("Tokenizer worker {} stopping", self.worker_id);
    }

    async fn process_task(&self, task: TokenizeTask) -> Result<()> {
        let start = std::time::Instant::now();

        // Handle multimodal content if present
        let tokens =
            if task.request.prompt.contains("<image>") || task.request.prompt.contains("<video>") {
                self.process_multimodal_content(&task.request.prompt)
                    .await?
            } else {
                // Standard text tokenization - this is the actual implementation
                match self.tokenizer.encode(&task.request.prompt) {
                    Ok(tokens) => tokens,
                    Err(e) => {
                        log::error!("Worker {} tokenization failed: {}", self.worker_id, e);
                        // Send error response
                        let _ =
                            task.response_tx
                                .send(crate::server_engine::CompletionStreamDelta {
                                    text_delta: format!("Tokenization error: {}", e),
                                    finish_reason: Some(crate::server_engine::FinishReason::Stop),
                                    usage: None,
                                    logprob: None,
                                    token_ids: Vec::new(),
                                });
                        return Err(e);
                    }
                }
            };

        let duration = start.elapsed();

        // Save values before move
        let prompt_len = task.request.prompt.len();
        let tokens_len = tokens.len();

        // Forward to scheduler
        let schedule_task = ScheduleTask {
            request_id: task.request_id,
            tokens,
            request: task.request,
            response_tx: task.response_tx,
        };

        if let Err(e) = self.schedule_sender.send(schedule_task).await {
            log::error!(
                "Worker {} failed to send to scheduler: {}",
                self.worker_id,
                e
            );
            return Err(anyhow::anyhow!("Schedule channel closed"));
        }

        log::trace!(
            "Worker {} tokenized {} chars to {} tokens in {:?}",
            self.worker_id,
            prompt_len,
            tokens_len,
            duration
        );

        Ok(())
    }

    async fn process_multimodal_content(&self, prompt: &str) -> Result<Vec<u32>> {
        // Placeholder for multimodal processing
        // In a real implementation, this would:
        // 1. Extract image/video URLs or base64 data
        // 2. Download and preprocess media
        // 3. Generate embeddings via vision encoder
        // 4. Combine text and media tokens

        log::debug!("Worker {} processing multimodal content", self.worker_id);

        // For now, just tokenize the text parts
        let text_only = prompt
            .replace("<image>", "[IMAGE]")
            .replace("<video>", "[VIDEO]");

        self.tokenizer.encode(&text_only)
    }
}

/// Statistics for tokenizer pool monitoring
#[derive(Debug, Clone)]
pub struct TokenizerStats {
    pub worker_count: usize,
    pub tasks_processed: u64,
    pub avg_processing_time_ms: f64,
    pub queue_depth: usize,
    pub workers_busy: usize,
}

impl TokenizerStats {
    pub fn new() -> Self {
        Self {
            worker_count: 0,
            tasks_processed: 0,
            avg_processing_time_ms: 0.0,
            queue_depth: 0,
            workers_busy: 0,
        }
    }
}
