# ARLE Multi-Threading Implementation Plan

## Overview

Based on SGLang's multi-threaded architecture analysis, this document outlines a concrete implementation plan for ARLE's Rust-native multi-threading system that achieves similar performance benefits without Python's limitations.

## Core Architecture

### Thread Role Separation

```rust
// infer/src/runtime/mod.rs
pub struct MultiThreadRuntime {
    // Core thread actors
    tokenizer_pool: TokenizerPool,
    scheduler: SchedulerActor, 
    gpu_executor: GpuExecutorPool,
    detokenizer_pool: DetokenizerPool,
    
    // Communication channels
    channels: RuntimeChannels,
    config: RuntimeConfig,
}

pub struct RuntimeChannels {
    // Request flow: HTTP -> Tokenizer -> Scheduler -> GPU -> Detokenizer -> HTTP
    tokenize_tx: Sender<TokenizeTask>,
    schedule_tx: Sender<ScheduleTask>, 
    execute_tx: Sender<ExecuteTask>,
    detokenize_tx: Sender<DetokenizeTask>,
    response_tx: Sender<ResponseTask>,
}
```

### Implementation Strategy

#### 1. Tokenizer Pool (CPU I/O Bound)
```rust
// infer/src/runtime/tokenizer_pool.rs
pub struct TokenizerPool {
    workers: Vec<JoinHandle<()>>,
    task_queue: Arc<SegQueue<TokenizeTask>>,
    result_sender: Sender<ScheduleTask>,
    config: TokenizerConfig,
}

#[derive(Debug)]
pub struct TokenizeTask {
    request_id: RequestId,
    prompt: String,
    options: TokenizeOptions,
    response_sender: oneshot::Sender<TokenizeResult>,
}

impl TokenizerPool {
    pub fn new(config: TokenizerConfig) -> Self {
        let (workers, task_queue, result_sender) = Self::spawn_workers(&config);
        Self { workers, task_queue, result_sender, config }
    }
    
    fn spawn_workers(config: &TokenizerConfig) -> (Vec<JoinHandle<()>>, Arc<SegQueue<TokenizeTask>>, Sender<ScheduleTask>) {
        let task_queue = Arc::new(SegQueue::new());
        let (result_tx, result_rx) = mpsc::channel(1024);
        
        let workers: Vec<_> = (0..config.worker_count)
            .map(|worker_id| {
                let queue = task_queue.clone();
                let sender = result_tx.clone();
                let tokenizer = config.tokenizer.clone();
                
                tokio::spawn(async move {
                    Self::worker_loop(worker_id, queue, sender, tokenizer).await
                })
            })
            .collect();
            
        (workers, task_queue, result_tx)
    }
    
    async fn worker_loop(
        worker_id: usize,
        task_queue: Arc<SegQueue<TokenizeTask>>,
        result_sender: Sender<ScheduleTask>,
        tokenizer: Arc<dyn Tokenizer>,
    ) {
        loop {
            if let Some(task) = task_queue.pop() {
                let start = Instant::now();
                
                let tokens = tokenizer.encode(&task.prompt);
                let schedule_task = ScheduleTask {
                    request_id: task.request_id,
                    tokens,
                    options: task.options.into(),
                };
                
                if let Err(e) = result_sender.send(schedule_task).await {
                    log::error!("Worker {}: failed to send result: {}", worker_id, e);
                }
                
                let duration = start.elapsed();
                log::debug!("Worker {}: tokenized {} chars in {:?}", worker_id, task.prompt.len(), duration);
            }
            
            // Yield control to allow other tasks
            tokio::task::yield_now().await;
        }
    }
}
```

#### 2. Scheduler Actor (Latency Critical)
```rust
// infer/src/runtime/scheduler.rs
pub struct SchedulerActor {
    state: Arc<RwLock<SchedulerState>>,
    kv_cache: Arc<RadixCache>,
    gpu_pool: Arc<GpuExecutorPool>,
    config: SchedulerConfig,
}

pub struct SchedulerState {
    pending_requests: VecDeque<ScheduleTask>,
    running_requests: HashMap<RequestId, RunningRequest>,
    memory_usage: MemoryStats,
    batch_builder: BatchBuilder,
}

impl SchedulerActor {
    pub async fn run(&self, mut task_rx: Receiver<ScheduleTask>) {
        let mut ticker = tokio::time::interval(Duration::from_millis(self.config.tick_interval_ms));
        
        loop {
            tokio::select! {
                // Process incoming requests
                Some(task) = task_rx.recv() => {
                    self.handle_new_request(task).await;
                }
                
                // Periodic scheduling tick
                _ = ticker.tick() => {
                    self.schedule_tick().await;
                }
                
                // Handle completed batches
                result = self.gpu_pool.recv_completed() => {
                    if let Ok(batch_result) = result {
                        self.handle_batch_completion(batch_result).await;
                    }
                }
            }
        }
    }
    
    async fn schedule_tick(&self) {
        let mut state = self.state.write().await;
        
        // Build next batch using RadixCache for prefix matching
        let batch = state.batch_builder.build_batch(
            &mut state.pending_requests,
            &self.kv_cache,
            &state.memory_usage
        );
        
        if !batch.is_empty() {
            // Dispatch to least loaded GPU worker
            let worker = self.gpu_pool.least_loaded_worker().await;
            if let Err(e) = worker.submit_batch(batch).await {
                log::error!("Failed to submit batch: {}", e);
            }
        }
    }
}
```

#### 3. GPU Executor with Stream Overlap
```rust
// infer/src/runtime/gpu_executor.rs
pub struct GpuExecutorPool {
    workers: Vec<Arc<GpuWorker>>,
    load_balancer: LoadBalancer,
    completed_rx: Receiver<BatchResult>,
}

pub struct GpuWorker {
    worker_id: usize,
    backend: Arc<dyn BackendEngine>,
    batch_queue: Arc<SegQueue<ExecuteTask>>,
    current_load: Arc<AtomicUsize>,
    
    // Stream overlap support
    compute_stream: ComputeStream,
    copy_stream: CopyStream,
    result_sender: Sender<BatchResult>,
}

impl GpuWorker {
    pub async fn run(&self) {
        loop {
            if let Some(task) = self.batch_queue.pop() {
                self.current_load.fetch_add(task.batch.size(), Ordering::Relaxed);
                
                // Execute with stream overlap
                let result = self.execute_with_overlap(task).await;
                
                self.current_load.fetch_sub(result.batch_size, Ordering::Relaxed);
                
                if let Err(e) = self.result_sender.send(result).await {
                    log::error!("Worker {}: failed to send result: {}", self.worker_id, e);
                }
            }
            
            tokio::task::yield_now().await;
        }
    }
    
    async fn execute_with_overlap(&self, task: ExecuteTask) -> BatchResult {
        // Launch async compute on GPU
        let compute_future = self.launch_compute(task.batch.clone());
        
        // Concurrently handle previous results (if any)
        let copy_future = self.handle_pending_copies();
        
        // Wait for both to complete
        let (forward_result, _) = tokio::join!(compute_future, copy_future);
        
        // Queue for async D2H copy
        self.queue_result_copy(forward_result).await
    }
    
    async fn launch_compute(&self, batch: Batch) -> ForwardResult {
        // Use dedicated compute stream to avoid blocking
        let _stream_guard = self.compute_stream.acquire().await;
        
        match &self.backend {
            #[cfg(feature = "cuda")]
            Backend::Cuda(engine) => engine.forward(&batch).await,
            
            #[cfg(feature = "metal")]
            Backend::Metal(engine) => engine.forward(&batch).await,
            
            #[cfg(feature = "no-cuda")]
            _ => todo!("GPU required: CUDA or Metal backend needed for inference"),
        }
    }
}
```

#### 4. Backend Abstraction
```rust
// infer/src/backend/engine.rs
#[async_trait::async_trait]
pub trait BackendEngine: Send + Sync {
    async fn forward(&self, batch: &Batch) -> Result<ForwardResult>;
    fn memory_stats(&self) -> MemoryStats;
    fn supports_overlap(&self) -> bool;
    fn max_batch_size(&self) -> usize;
}

// CUDA implementation
#[cfg(feature = "cuda")]
pub struct CudaEngine {
    context: CudaContext,
    model: Arc<CudaModel>,
    memory_pool: Arc<CudaMemoryPool>,
    streams: CudaStreamPool,
}

#[cfg(feature = "cuda")]
#[async_trait::async_trait]
impl BackendEngine for CudaEngine {
    async fn forward(&self, batch: &Batch) -> Result<ForwardResult> {
        let stream = self.streams.acquire().await?;
        
        // Copy input to GPU
        let gpu_batch = self.copy_to_device(batch, &stream).await?;
        
        // Execute forward pass
        let gpu_result = self.model.forward(&gpu_batch, &stream).await?;
        
        // Async copy result back
        let cpu_result = self.copy_to_host(&gpu_result, &stream).await?;
        
        Ok(cpu_result)
    }
}

// Metal implementation
#[cfg(feature = "metal")]
pub struct MetalEngine {
    device: MTLDevice,
    model: Arc<MetalModel>,
    command_queues: MetalCommandQueuePool,
    memory_pool: Arc<MetalMemoryPool>,
}
```

## Configuration & Tuning

```rust
// infer/src/runtime/config.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    // Threading
    pub tokenizer_workers: usize,
    pub detokenizer_workers: usize,
    pub gpu_workers_per_device: usize,
    
    // Scheduling
    pub scheduler_tick_ms: u64,
    pub max_batch_size: usize,
    pub max_sequence_length: usize,
    pub batch_timeout_ms: u64,
    
    // Memory management
    pub kv_cache_fraction: f32,
    pub memory_pool_strategy: MemoryPoolStrategy,
    
    // Backend-specific
    #[cfg(feature = "cuda")]
    pub cuda: CudaConfig,
    
    #[cfg(feature = "metal")]
    pub metal: MetalConfig,
}

impl RuntimeConfig {
    pub fn auto_tune() -> Self {
        let num_cpus = num_cpus::get();
        let gpu_count = detect_gpu_count();
        
        Self {
            tokenizer_workers: (num_cpus / 4).max(2),
            detokenizer_workers: (num_cpus / 4).max(2),
            gpu_workers_per_device: 2,
            scheduler_tick_ms: 1,  // 1ms for low latency
            max_batch_size: 64,
            // ... other auto-tuned values
        }
    }
}
```

## Integration with Existing ARLE

### 1. HTTP Server Integration
```rust
// infer/src/http_server/mod.rs
impl HttpServer {
    pub async fn handle_request(&self, request: InferRequest) -> Result<InferResponse> {
        // Create response channel
        let (tx, rx) = oneshot::channel();
        
        // Submit to tokenizer pool
        let task = TokenizeTask {
            request_id: request.id,
            prompt: request.prompt,
            options: request.options,
            response_sender: tx,
        };
        
        self.runtime.tokenizer_pool.submit(task).await?;
        
        // Wait for completion
        let result = rx.await?;
        Ok(result.into())
    }
}
```

### 2. Backward Compatibility
```rust
// infer/src/lib.rs
pub enum RuntimeMode {
    SingleThreaded,  // Current implementation
    MultiThreaded,   // New implementation
}

pub fn create_runtime(config: RuntimeConfig) -> Box<dyn InferenceRuntime> {
    match config.mode {
        RuntimeMode::SingleThreaded => Box::new(SingleThreadRuntime::new(config)),
        RuntimeMode::MultiThreaded => Box::new(MultiThreadRuntime::new(config)),
    }
}
```

## Implementation Timeline

### Phase 1 (Week 1-2): Foundation
- [ ] Basic thread pool abstractions
- [ ] Channel-based communication infrastructure  
- [ ] Configuration system
- [ ] Basic tokenizer pool implementation

### Phase 2 (Week 3-4): Core Scheduling
- [ ] Scheduler actor with batching
- [ ] GPU executor pool
- [ ] Backend abstraction layer
- [ ] Memory management integration

### Phase 3 (Week 5-6): Optimization
- [ ] Stream overlap implementation
- [ ] RadixCache integration
- [ ] Load balancing and auto-tuning
- [ ] Performance benchmarking

### Phase 4 (Week 7-8): Production Ready
- [ ] Fault tolerance and recovery
- [ ] Monitoring and observability
- [ ] Documentation and examples
- [ ] Integration testing

## Success Criteria

### Performance Targets
- **Throughput**: 2x improvement over single-threaded baseline
- **Latency**: <10ms P99 scheduling overhead
- **GPU Utilization**: >90% on compute-bound workloads
- **Memory Efficiency**: <20% overhead vs single-threaded

### Quality Gates
- [ ] All tests pass under high concurrency
- [ ] No race conditions detected by ThreadSanitizer
- [ ] Memory usage remains bounded under load
- [ ] Graceful degradation on resource exhaustion

### Benchmarking Protocol
```bash
# Regression test suite
cargo test --release --features multi-thread
cargo bench --features multi-thread baseline

# Performance comparison
scripts/bench_multithread_vs_single.sh
scripts/bench_multithread_vs_sglang.sh
```

This implementation leverages Rust's strengths in safe concurrency while achieving the same architectural benefits as SGLang's multi-process design, but with better performance characteristics and operational simplicity.