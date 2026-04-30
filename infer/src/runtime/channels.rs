#![allow(unreachable_pub)]
#![allow(warnings)]
/*!
 * Inter-thread communication channels for multi-threaded runtime
 */

use anyhow::Result;
use tokio::sync::mpsc::{self, Receiver, Sender, UnboundedSender};
use uuid::Uuid;

use super::config::ChannelConfig;
use crate::server_engine::{CompletionRequest, CompletionStreamDelta};

/// Request ID for tracking requests through the pipeline
pub type RequestId = Uuid;

/// Task for tokenization worker
#[derive(Debug)]
pub struct TokenizeTask {
    pub request_id: RequestId,
    pub request: CompletionRequest,
    pub response_tx: UnboundedSender<CompletionStreamDelta>,
}

/// Task for scheduler
#[derive(Debug)]
pub struct ScheduleTask {
    pub request_id: RequestId,
    pub tokens: Vec<u32>,
    pub request: CompletionRequest,
    pub response_tx: UnboundedSender<CompletionStreamDelta>,
}

/// Task for GPU execution
#[derive(Debug)]
pub struct ExecuteTask {
    pub batch_id: Uuid,
    pub requests: Vec<BatchedRequest>,
    pub result_tx: UnboundedSender<BatchResult>,
}

/// Individual request within a batch
#[derive(Debug)]
pub struct BatchedRequest {
    pub request_id: RequestId,
    pub tokens: Vec<u32>,
    pub prompt: String, // Keep original prompt
    pub max_tokens: usize,
    pub sampling_params: crate::sampler::SamplingParams,
    pub response_tx: UnboundedSender<CompletionStreamDelta>,
}

/// Result from GPU execution
#[derive(Debug)]
pub struct BatchResult {
    pub batch_id: Uuid,
    pub results: Vec<RequestResult>,
}

/// Result for individual request
#[derive(Debug)]
pub struct RequestResult {
    pub request_id: RequestId,
    pub new_tokens: Vec<u32>,
    pub logprobs: Vec<f32>,
    pub finished: bool,
    pub finish_reason: Option<crate::server_engine::FinishReason>,
}

/// Task for detokenization
#[derive(Debug)]
pub struct DetokenizeTask {
    pub request_id: RequestId,
    pub new_tokens: Vec<u32>,
    pub finished: bool,
    pub finish_reason: Option<crate::server_engine::FinishReason>,
    pub response_tx: UnboundedSender<CompletionStreamDelta>,
}

/// Central communication channels for the runtime
pub struct RuntimeChannels {
    // Tokenizer channels
    pub tokenizer_tx: Sender<TokenizeTask>,
    pub tokenizer_rx: Receiver<TokenizeTask>,

    // Scheduler channels
    pub schedule_tx: Sender<ScheduleTask>,
    pub schedule_rx: Receiver<ScheduleTask>,

    // GPU executor channels
    pub execute_tx: Sender<ExecuteTask>,
    pub execute_rx: Receiver<ExecuteTask>,

    // GPU result channels
    pub gpu_result_tx: Sender<BatchResult>,
    pub gpu_result_rx: Receiver<BatchResult>,

    // Detokenizer channels
    pub detokenizer_tx: Sender<DetokenizeTask>,
    pub detokenizer_rx: Receiver<DetokenizeTask>,
}

impl RuntimeChannels {
    /// Create all communication channels with configured buffer sizes
    pub fn new(config: &ChannelConfig) -> Result<Self> {
        let (tokenizer_tx, tokenizer_rx) = mpsc::channel(config.buffer_sizes.tokenizer_buffer);
        let (schedule_tx, schedule_rx) = mpsc::channel(config.buffer_sizes.scheduler_buffer);
        let (execute_tx, execute_rx) = mpsc::channel(config.buffer_sizes.gpu_executor_buffer);
        let (gpu_result_tx, gpu_result_rx) = mpsc::channel(config.buffer_sizes.gpu_executor_buffer);
        let (detokenizer_tx, detokenizer_rx) =
            mpsc::channel(config.buffer_sizes.detokenizer_buffer);

        Ok(Self {
            tokenizer_tx,
            tokenizer_rx,
            schedule_tx,
            schedule_rx,
            execute_tx,
            execute_rx,
            gpu_result_tx,
            gpu_result_rx,
            detokenizer_tx,
            detokenizer_rx,
        })
    }

    /// Get channel statistics for monitoring
    pub fn stats(&self) -> ChannelStats {
        ChannelStats {
            tokenizer_queue_size: self.tokenizer_tx.capacity() - self.tokenizer_tx.max_capacity(),
            scheduler_queue_size: self.schedule_tx.capacity() - self.schedule_tx.max_capacity(),
            executor_queue_size: self.execute_tx.capacity() - self.execute_tx.max_capacity(),
            detokenizer_queue_size: self.detokenizer_tx.capacity()
                - self.detokenizer_tx.max_capacity(),
        }
    }
}

/// Channel utilization statistics
#[derive(Debug, Clone)]
pub struct ChannelStats {
    pub tokenizer_queue_size: usize,
    pub scheduler_queue_size: usize,
    pub executor_queue_size: usize,
    pub detokenizer_queue_size: usize,
}

/// Lock-free communication using crossbeam for high-throughput paths
pub struct LockFreeChannels {
    /// High-throughput tokenizer task queue
    pub tokenizer_queue: crossbeam_queue::SegQueue<TokenizeTask>,
    /// Scheduler task queue
    pub scheduler_queue: crossbeam_queue::SegQueue<ScheduleTask>,
    /// GPU executor task queue
    pub executor_queue: crossbeam_queue::SegQueue<ExecuteTask>,
    /// Detokenizer task queue
    pub detokenizer_queue: crossbeam_queue::SegQueue<DetokenizeTask>,
}

impl LockFreeChannels {
    pub fn new() -> Self {
        Self {
            tokenizer_queue: crossbeam_queue::SegQueue::new(),
            scheduler_queue: crossbeam_queue::SegQueue::new(),
            executor_queue: crossbeam_queue::SegQueue::new(),
            detokenizer_queue: crossbeam_queue::SegQueue::new(),
        }
    }

    pub fn stats(&self) -> LockFreeStats {
        LockFreeStats {
            tokenizer_queue_len: self.tokenizer_queue.len(),
            scheduler_queue_len: self.scheduler_queue.len(),
            executor_queue_len: self.executor_queue.len(),
            detokenizer_queue_len: self.detokenizer_queue.len(),
        }
    }
}

impl Default for LockFreeChannels {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct LockFreeStats {
    pub tokenizer_queue_len: usize,
    pub scheduler_queue_len: usize,
    pub executor_queue_len: usize,
    pub detokenizer_queue_len: usize,
}

/// Channel wrapper that can use either tokio channels or lock-free queues
pub enum ChannelMode {
    /// Use tokio mpsc channels for simplicity and backpressure
    Tokio(RuntimeChannels),
    /// Use lock-free queues for maximum throughput
    LockFree(LockFreeChannels),
}

impl ChannelMode {
    pub fn new_tokio(config: &ChannelConfig) -> Result<Self> {
        Ok(Self::Tokio(RuntimeChannels::new(config)?))
    }

    pub fn new_lock_free() -> Self {
        Self::LockFree(LockFreeChannels::new())
    }

    /// Submit tokenize task
    pub async fn submit_tokenize(&self, task: TokenizeTask) -> Result<()> {
        match self {
            Self::Tokio(channels) => {
                channels
                    .tokenizer_tx
                    .send(task)
                    .await
                    .map_err(|_| anyhow::anyhow!("Tokenizer channel closed"))?;
            }
            Self::LockFree(channels) => {
                channels.tokenizer_queue.push(task);
            }
        }
        Ok(())
    }

    /// Submit schedule task
    pub async fn submit_schedule(&self, task: ScheduleTask) -> Result<()> {
        match self {
            Self::Tokio(channels) => {
                channels
                    .schedule_tx
                    .send(task)
                    .await
                    .map_err(|_| anyhow::anyhow!("Schedule channel closed"))?;
            }
            Self::LockFree(channels) => {
                channels.scheduler_queue.push(task);
            }
        }
        Ok(())
    }

    /// Submit execute task
    pub async fn submit_execute(&self, task: ExecuteTask) -> Result<()> {
        match self {
            Self::Tokio(channels) => {
                channels
                    .execute_tx
                    .send(task)
                    .await
                    .map_err(|_| anyhow::anyhow!("Execute channel closed"))?;
            }
            Self::LockFree(channels) => {
                channels.executor_queue.push(task);
            }
        }
        Ok(())
    }

    /// Submit detokenize task
    pub async fn submit_detokenize(&self, task: DetokenizeTask) -> Result<()> {
        match self {
            Self::Tokio(channels) => {
                channels
                    .detokenizer_tx
                    .send(task)
                    .await
                    .map_err(|_| anyhow::anyhow!("Detokenizer channel closed"))?;
            }
            Self::LockFree(channels) => {
                channels.detokenizer_queue.push(task);
            }
        }
        Ok(())
    }

    /// Get channel statistics
    pub fn stats(&self) -> ChannelStatsEnum {
        match self {
            Self::Tokio(channels) => ChannelStatsEnum::Tokio(channels.stats()),
            Self::LockFree(channels) => ChannelStatsEnum::LockFree(channels.stats()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ChannelStatsEnum {
    Tokio(ChannelStats),
    LockFree(LockFreeStats),
}
