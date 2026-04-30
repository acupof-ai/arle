/*!
 * Hybrid scheduler actor with specialized sub-schedulers to solve bottleneck issues
 */

use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock, mpsc};
use tokio::task::JoinHandle;
use tokio::time::{Duration, Instant, interval};
use uuid::Uuid;

use super::channels::{
    BatchResult, BatchedRequest, DetokenizeTask, ExecuteTask, RequestResult, ScheduleTask,
};
use super::config::RuntimeConfig;
use crate::server_engine::CompletionRequest;

/// Hybrid scheduler that coordinates multiple specialized sub-schedulers
pub struct SchedulerActor {
    // Core scheduler state
    state: Arc<RwLock<SchedulerState>>,
    config: RuntimeConfig,

    // Sub-schedulers for different responsibilities
    admission_scheduler: Arc<AdmissionScheduler>,
    batch_scheduler: Arc<BatchScheduler>,
    memory_scheduler: Arc<MemoryScheduler>,

    // Communication channels
    schedule_rx: Arc<Mutex<mpsc::Receiver<ScheduleTask>>>,
    execute_tx: mpsc::Sender<ExecuteTask>,
    gpu_result_rx: Arc<Mutex<mpsc::Receiver<BatchResult>>>,
    detokenizer_tx: mpsc::Sender<DetokenizeTask>,

    // Background tasks
    scheduler_handle: Option<JoinHandle<()>>,
    shutdown_tx: Option<mpsc::Sender<()>>,
}

/// Central scheduler state
#[derive(Debug)]
struct SchedulerState {
    /// Requests waiting for admission
    pending_requests: VecDeque<ScheduleTask>,
    /// Currently executing requests
    running_requests: HashMap<Uuid, RunningRequest>,
    /// Request to batch mapping
    request_to_batch: HashMap<Uuid, Uuid>,
    /// Current memory usage
    memory_stats: MemoryStats,
    /// Performance metrics
    metrics: SchedulerMetrics,
}

/// Information about a running request
#[derive(Debug)]
struct RunningRequest {
    request_id: Uuid,
    tokens_generated: usize,
    max_tokens: usize,
    start_time: Instant,
    last_update: Instant,
    kv_blocks: Vec<u32>,
    response_tx: mpsc::UnboundedSender<crate::server_engine::CompletionStreamDelta>,
}

/// Memory utilization statistics
#[derive(Debug, Clone)]
struct MemoryStats {
    kv_cache_used: usize,
    kv_cache_total: usize,
    gpu_memory_used: usize,
    gpu_memory_total: usize,
    fragmentation: f64,
}

/// Scheduler performance metrics
#[derive(Debug)]
struct SchedulerMetrics {
    requests_admitted: u64,
    requests_completed: u64,
    batches_executed: u64,
    avg_batch_size: f64,
    avg_scheduling_latency_ms: f64,
    cache_hit_rate: f64,
}

impl SchedulerActor {
    pub async fn new(
        config: RuntimeConfig,
        schedule_rx: mpsc::Receiver<ScheduleTask>,
        execute_tx: mpsc::Sender<ExecuteTask>,
        gpu_result_rx: mpsc::Receiver<BatchResult>,
        detokenizer_tx: mpsc::Sender<DetokenizeTask>,
    ) -> Result<Self> {
        let state = Arc::new(RwLock::new(SchedulerState {
            pending_requests: VecDeque::new(),
            running_requests: HashMap::new(),
            request_to_batch: HashMap::new(),
            memory_stats: MemoryStats {
                kv_cache_used: 0,
                kv_cache_total: 0,
                gpu_memory_used: 0,
                gpu_memory_total: 0,
                fragmentation: 0.0,
            },
            metrics: SchedulerMetrics {
                requests_admitted: 0,
                requests_completed: 0,
                batches_executed: 0,
                avg_batch_size: 0.0,
                avg_scheduling_latency_ms: 0.0,
                cache_hit_rate: 0.0,
            },
        }));

        // Create specialized sub-schedulers
        let admission_scheduler = Arc::new(AdmissionScheduler::new(config.clone()).await?);
        let batch_scheduler = Arc::new(BatchScheduler::new(config.clone()).await?);
        let memory_scheduler = Arc::new(MemoryScheduler::new(config.clone()).await?);

        Ok(Self {
            state,
            config,
            admission_scheduler,
            batch_scheduler,
            memory_scheduler,
            schedule_rx: Arc::new(Mutex::new(schedule_rx)),
            execute_tx,
            gpu_result_rx: Arc::new(Mutex::new(gpu_result_rx)),
            detokenizer_tx,
            scheduler_handle: None,
            shutdown_tx: None,
        })
    }

    pub async fn start(&self) -> Result<()> {
        let (_shutdown_tx, shutdown_rx) = mpsc::channel(1);

        // Start sub-schedulers
        self.admission_scheduler.start().await?;
        self.batch_scheduler.start().await?;
        self.memory_scheduler.start().await?;

        // Start main scheduler loop
        let _scheduler_handle = {
            let state = self.state.clone();
            let config = self.config.clone();
            let admission = self.admission_scheduler.clone();
            let batch = self.batch_scheduler.clone();
            let memory = self.memory_scheduler.clone();
            let schedule_rx = self.schedule_rx.clone();
            let execute_tx = self.execute_tx.clone();
            let gpu_result_rx = self.gpu_result_rx.clone();
            let detokenizer_tx = self.detokenizer_tx.clone();

            tokio::spawn(async move {
                Self::scheduler_loop(
                    state,
                    config,
                    admission,
                    batch,
                    memory,
                    schedule_rx,
                    execute_tx,
                    gpu_result_rx,
                    detokenizer_tx,
                    shutdown_rx,
                )
                .await;
            })
        };

        log::info!("Scheduler actor started");

        // Store handles (note: we can't store them in self due to &self)
        // In practice, we'd use Arc<Mutex<Option<JoinHandle<()>>>> or similar
        // self.scheduler_handle = Some(scheduler_handle);
        // self.shutdown_tx = Some(shutdown_tx);

        Ok(())
    }

    async fn scheduler_loop(
        state: Arc<RwLock<SchedulerState>>,
        config: RuntimeConfig,
        admission: Arc<AdmissionScheduler>,
        batch: Arc<BatchScheduler>,
        memory: Arc<MemoryScheduler>,
        schedule_rx: Arc<Mutex<mpsc::Receiver<ScheduleTask>>>,
        execute_tx: mpsc::Sender<ExecuteTask>,
        gpu_result_rx: Arc<Mutex<mpsc::Receiver<BatchResult>>>,
        detokenizer_tx: mpsc::Sender<DetokenizeTask>,
        mut shutdown_rx: mpsc::Receiver<()>,
    ) {
        let mut tick_interval = interval(Duration::from_millis(config.threading.scheduler_tick_ms));

        loop {
            tokio::select! {
                // Shutdown signal
                _ = shutdown_rx.recv() => {
                    log::info!("Scheduler received shutdown signal");
                    break;
                }

                // Periodic scheduling tick
                _ = tick_interval.tick() => {
                    if let Err(e) = Self::handle_scheduling_tick(
                        &state,
                        &config,
                        &admission,
                        &batch,
                        &memory,
                        &execute_tx,
                    ).await {
                        log::error!("Scheduling tick error: {}", e);
                    }
                }

                // New requests from tokenizer
                new_request = async {
                    let mut rx = schedule_rx.lock().await;
                    rx.recv().await
                } => {
                    if let Some(request) = new_request {
                        if let Err(e) = Self::handle_new_request(&state, &admission, request).await {
                            log::error!("Failed to handle new request: {}", e);
                        }
                    }
                }

                // GPU execution results
                gpu_result = async {
                    let mut rx = gpu_result_rx.lock().await;
                    rx.recv().await
                } => {
                    if let Some(result) = gpu_result {
                        if let Err(e) = Self::handle_gpu_result(
                            &state,
                            &memory,
                            &detokenizer_tx,
                            result,
                        ).await {
                            log::error!("Failed to handle GPU result: {}", e);
                        }
                    }
                }
            }
        }

        log::info!("Scheduler loop terminated");
    }

    async fn handle_new_request(
        state: &Arc<RwLock<SchedulerState>>,
        admission: &Arc<AdmissionScheduler>,
        request: ScheduleTask,
    ) -> Result<()> {
        // Admission control check
        if admission.can_admit(&request).await? {
            let mut state = state.write().await;
            state.pending_requests.push_back(request);
            state.metrics.requests_admitted += 1;
        } else {
            log::warn!(
                "Request {} rejected by admission control",
                request.request_id
            );
            // Send rejection response
            let _ = request
                .response_tx
                .send(crate::server_engine::CompletionStreamDelta {
                    text_delta: "Request rejected due to capacity limits".to_string(),
                    finish_reason: Some(crate::server_engine::FinishReason::Length),
                    usage: None,
                    logprob: None,
                    token_ids: Vec::new(),
                });
        }

        Ok(())
    }

    async fn handle_scheduling_tick(
        state: &Arc<RwLock<SchedulerState>>,
        config: &RuntimeConfig,
        _admission: &Arc<AdmissionScheduler>,
        batch: &Arc<BatchScheduler>,
        memory: &Arc<MemoryScheduler>,
        execute_tx: &mpsc::Sender<ExecuteTask>,
    ) -> Result<()> {
        let batch_candidates = {
            let mut state = state.write().await;

            // Update memory statistics
            memory.update_memory_stats(&mut state.memory_stats).await?;

            // Get candidates for batching
            let memory_stats = state.memory_stats.clone();
            let candidates = batch
                .select_batch_candidates(
                    &mut state.pending_requests,
                    &memory_stats,
                    config.scheduling.max_batch_size,
                )
                .await?;

            candidates
        };

        if !batch_candidates.is_empty() {
            // Build and execute batch
            let execute_task = batch.build_execute_task(batch_candidates).await?;

            if let Err(e) = execute_tx.send(execute_task).await {
                log::error!("Failed to submit batch for execution: {}", e);
            } else {
                let mut state = state.write().await;
                state.metrics.batches_executed += 1;
            }
        }

        Ok(())
    }

    async fn handle_gpu_result(
        state: &Arc<RwLock<SchedulerState>>,
        memory: &Arc<MemoryScheduler>,
        detokenizer_tx: &mpsc::Sender<DetokenizeTask>,
        result: BatchResult,
    ) -> Result<()> {
        for request_result in result.results {
            Self::process_request_result(state, memory, detokenizer_tx, request_result).await?;
        }

        Ok(())
    }

    async fn process_request_result(
        state: &Arc<RwLock<SchedulerState>>,
        memory: &Arc<MemoryScheduler>,
        detokenizer_tx: &mpsc::Sender<DetokenizeTask>,
        result: RequestResult,
    ) -> Result<()> {
        let (response_tx, finished) = {
            let mut state = state.write().await;

            if let Some(running_request) = state.running_requests.get_mut(&result.request_id) {
                running_request.tokens_generated += result.new_tokens.len();
                running_request.last_update = Instant::now();

                let finished = result.finished
                    || running_request.tokens_generated >= running_request.max_tokens;

                let response_tx = running_request.response_tx.clone();

                if finished {
                    // Clean up finished request
                    let completed = state.running_requests.remove(&result.request_id);
                    if let Some(completed) = completed {
                        // Release memory blocks
                        memory.release_blocks(&completed.kv_blocks).await?;
                        state.metrics.requests_completed += 1;
                    }
                }

                (response_tx, finished)
            } else {
                log::warn!("Received result for unknown request: {}", result.request_id);
                return Ok(());
            }
        };

        // Send to detokenizer
        let detokenize_task = DetokenizeTask {
            request_id: result.request_id,
            new_tokens: result.new_tokens,
            finished,
            finish_reason: result.finish_reason,
            response_tx,
        };

        detokenizer_tx
            .send(detokenize_task)
            .await
            .map_err(|_| anyhow::anyhow!("Detokenizer channel closed"))?;

        Ok(())
    }

    pub fn pending_count(&self) -> usize {
        // This would need to be implemented with proper async access
        0
    }

    pub async fn shutdown(&self) -> Result<()> {
        if let Some(shutdown_tx) = &self.shutdown_tx {
            let _ = shutdown_tx.send(()).await;
        }

        // Shutdown sub-schedulers
        self.admission_scheduler.shutdown().await?;
        self.batch_scheduler.shutdown().await?;
        self.memory_scheduler.shutdown().await?;

        log::info!("Scheduler actor shutdown complete");
        Ok(())
    }
}

/// Specialized scheduler for admission control
struct AdmissionScheduler {
    config: RuntimeConfig,
    // RadixCache for prefix matching would be integrated here
}

impl AdmissionScheduler {
    async fn new(config: RuntimeConfig) -> Result<Self> {
        Ok(Self { config })
    }

    async fn start(&self) -> Result<()> {
        Ok(())
    }

    async fn can_admit(&self, _request: &ScheduleTask) -> Result<bool> {
        // Implement admission control logic:
        // - Check memory availability
        // - Check request queue depth
        // - Apply rate limiting
        // - Check prefix cache hit potential
        Ok(true)
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }
}

/// Specialized scheduler for batch formation
struct BatchScheduler {
    config: RuntimeConfig,
}

impl BatchScheduler {
    async fn new(config: RuntimeConfig) -> Result<Self> {
        Ok(Self { config })
    }

    async fn start(&self) -> Result<()> {
        Ok(())
    }

    async fn select_batch_candidates(
        &self,
        pending_requests: &mut VecDeque<ScheduleTask>,
        memory_stats: &MemoryStats,
        max_batch_size: usize,
    ) -> Result<Vec<ScheduleTask>> {
        let mut candidates = Vec::new();
        let mut to_remove = Vec::new();

        // Simple FIFO batching for now
        // In practice, this would implement sophisticated batching strategies:
        // - Prefix matching for RadixAttention
        // - Sequence length compatibility
        // - Memory requirements

        for (idx, request) in pending_requests.iter().enumerate() {
            if candidates.len() >= max_batch_size {
                break;
            }

            // Simple memory check - allow batching if memory is not initialized or below threshold
            if memory_stats.kv_cache_total == 0
                || memory_stats.kv_cache_used < memory_stats.kv_cache_total * 8 / 10
            {
                to_remove.push(idx);
                if candidates.len() < max_batch_size {
                    // Would clone the request here in real implementation
                    candidates.push(ScheduleTask {
                        request_id: request.request_id,
                        tokens: request.tokens.clone(),
                        request: CompletionRequest {
                            prompt: request.request.prompt.clone(),
                            max_tokens: request.request.max_tokens,
                            sampling: request.request.sampling.clone(),
                            stop: request.request.stop.clone(),
                            logprobs: request.request.logprobs,
                            session_id: request.request.session_id.clone(),
                            trace_context: request.request.trace_context,
                        },
                        response_tx: request.response_tx.clone(),
                    });
                }
            }
        }

        // Remove selected requests from pending queue
        for &idx in to_remove.iter().rev() {
            pending_requests.remove(idx);
        }

        Ok(candidates)
    }

    async fn build_execute_task(&self, candidates: Vec<ScheduleTask>) -> Result<ExecuteTask> {
        let batch_id = Uuid::new_v4();
        let mut requests = Vec::new();

        for candidate in candidates {
            requests.push(BatchedRequest {
                request_id: candidate.request_id,
                tokens: candidate.tokens,
                prompt: candidate.request.prompt.clone(),
                max_tokens: candidate.request.max_tokens,
                sampling_params: candidate.request.sampling,
                response_tx: candidate.response_tx,
            });
        }

        let (result_tx, _result_rx) = mpsc::unbounded_channel();

        Ok(ExecuteTask {
            batch_id,
            requests,
            result_tx,
        })
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }
}

/// Specialized scheduler for memory management
struct MemoryScheduler {
    config: RuntimeConfig,
}

impl MemoryScheduler {
    async fn new(config: RuntimeConfig) -> Result<Self> {
        Ok(Self { config })
    }

    async fn start(&self) -> Result<()> {
        Ok(())
    }

    async fn update_memory_stats(&self, _stats: &mut MemoryStats) -> Result<()> {
        // Update memory statistics from actual allocators
        // This would integrate with the KV cache and GPU memory management
        Ok(())
    }

    async fn release_blocks(&self, _blocks: &[u32]) -> Result<()> {
        // Release KV cache blocks
        Ok(())
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }
}
