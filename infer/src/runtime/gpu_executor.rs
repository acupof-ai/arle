/*!
 * GPU executor pool with stream overlap for high-throughput inference
 */

use anyhow::Result;
// use rand::{Rng, thread_rng};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::{Mutex, mpsc};
use tokio::task::JoinHandle;
use uuid::Uuid;

use super::backend_worker::BackendWorker;
use super::channels::{BatchResult, ExecuteTask, RequestResult};
use crate::backend::InferenceBackend;

/// Pool of GPU workers with load balancing and stream overlap
pub struct GpuExecutorPool {
    workers: Vec<Arc<GpuWorker>>,
    worker_count: usize,
    load_balancer: LoadBalancer,
    task_receiver: Arc<Mutex<mpsc::Receiver<ExecuteTask>>>,
    result_sender: mpsc::Sender<BatchResult>,
    shutdown_tx: Option<mpsc::Sender<()>>,
}

/// Individual GPU worker with stream overlap capabilities
pub struct GpuWorker {
    worker_id: usize,
    backend_worker: BackendWorker,
    current_load: Arc<AtomicUsize>,

    // Stream overlap infrastructure
    compute_stream: ComputeStream,
    copy_stream: CopyStream,

    // Task coordination
    active_batches: Arc<Mutex<Vec<ActiveBatch>>>,
    result_sender: mpsc::Sender<BatchResult>,

    // Worker control
    worker_handle: Option<JoinHandle<()>>,
}

/// Active batch being processed
#[derive(Debug)]
struct ActiveBatch {
    batch_id: Uuid,
    start_time: std::time::Instant,
    request_count: usize,
}

/// Compute stream abstraction for different backends
#[derive(Debug)]
pub enum ComputeStream {
    #[cfg(feature = "cuda")]
    Cuda(CudaComputeStream),
    #[cfg(feature = "metal")]
    Metal(MetalComputeStream),
    #[cfg(feature = "cpu")]
    Cpu,
}

/// Copy stream for async data transfers
#[derive(Debug)]
pub enum CopyStream {
    #[cfg(feature = "cuda")]
    Cuda(CudaCopyStream),
    #[cfg(feature = "metal")]
    Metal(MetalCopyStream),
    #[cfg(feature = "cpu")]
    Cpu,
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct CudaComputeStream {
    stream: cudarc::driver::CudaStream,
    device: Arc<cudarc::driver::CudaDevice>,
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct CudaCopyStream {
    stream: cudarc::driver::CudaStream,
    device: Arc<cudarc::driver::CudaDevice>,
}

#[cfg(feature = "metal")]
#[derive(Debug)]
struct MetalComputeStream {
    // MLX handles device/command queue internally - no explicit handle needed
    _placeholder: (),
}

#[cfg(feature = "metal")]
#[derive(Debug)]
struct MetalCopyStream {
    // MLX handles device/command queue internally - no explicit handle needed
    _placeholder: (),
}

/// Load balancer for distributing work across GPU workers
pub struct LoadBalancer {
    workers: Vec<Arc<GpuWorker>>,
    round_robin_counter: AtomicUsize,
    strategy: LoadBalancingStrategy,
}

#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    Random,
}

impl GpuExecutorPool {
    pub async fn new(
        backend: Box<dyn InferenceBackend>,
        workers_per_device: usize,
        task_receiver: mpsc::Receiver<ExecuteTask>,
        result_sender: mpsc::Sender<BatchResult>,
    ) -> Result<Self> {
        let (shutdown_tx, _shutdown_rx) = mpsc::channel(1);

        // Create backend workers using the Backend Pool approach
        let mut workers = Vec::new();

        // For now, just create one worker to avoid backend cloning issues
        let worker_id = 0;
        if workers_per_device > 0 {
            let backend_worker = BackendWorker::new(backend, worker_id)?;

            // Create streams for this worker
            let compute_stream = Self::create_compute_stream(worker_id).await?;
            let copy_stream = Self::create_copy_stream(worker_id).await?;

            let worker = Arc::new(
                GpuWorker::new(
                    worker_id,
                    backend_worker,
                    compute_stream,
                    copy_stream,
                    result_sender.clone(),
                )
                .await?,
            );

            workers.push(worker);
        }

        let load_balancer = LoadBalancer::new(workers.clone(), LoadBalancingStrategy::LeastLoaded);

        log::info!(
            "Created GPU executor pool with {} workers",
            workers_per_device
        );

        Ok(Self {
            workers,
            worker_count: workers_per_device,
            load_balancer,
            task_receiver: Arc::new(Mutex::new(task_receiver)),
            result_sender,
            shutdown_tx: Some(shutdown_tx),
        })
    }

    async fn create_compute_stream(_worker_id: usize) -> Result<ComputeStream> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::CudaDevice;
            let device = CudaDevice::new(0)?; // Use GPU 0, would be configurable
            let stream = device.fork_default_stream()?;
            Ok(ComputeStream::Cuda(CudaComputeStream {
                stream,
                device: Arc::new(device),
            }))
        }

        #[cfg(feature = "metal")]
        {
            // MLX handles device/command queue internally - no explicit setup needed
            Ok(ComputeStream::Metal(MetalComputeStream {
                _placeholder: (),
            }))
        }

        #[cfg(feature = "cpu")]
        {
            Ok(ComputeStream::Cpu)
        }

        #[cfg(not(any(feature = "cuda", feature = "metal", feature = "cpu")))]
        {
            Err(anyhow::anyhow!("No backend feature enabled"))
        }
    }

    async fn create_copy_stream(_worker_id: usize) -> Result<CopyStream> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::CudaDevice;
            let device = CudaDevice::new(0)?;
            let stream = device.fork_default_stream()?;
            Ok(CopyStream::Cuda(CudaCopyStream {
                stream,
                device: Arc::new(device),
            }))
        }

        #[cfg(feature = "metal")]
        {
            // MLX handles device/command queue internally - no explicit setup needed
            Ok(CopyStream::Metal(MetalCopyStream { _placeholder: () }))
        }

        #[cfg(feature = "cpu")]
        {
            Ok(CopyStream::Cpu)
        }

        #[cfg(not(any(feature = "cuda", feature = "metal", feature = "cpu")))]
        {
            Err(anyhow::anyhow!("No backend feature enabled"))
        }
    }

    pub async fn start(&self) -> Result<()> {
        // Start all workers
        for worker in &self.workers {
            worker.start().await?;
        }

        // TODO: Fix thread safety issues with InferenceBackend
        // Start task distribution loop
        // let workers = self.workers.clone();
        // let load_balancer = self.load_balancer.clone();
        // let task_receiver = self.task_receiver.clone();

        // tokio::spawn(async move {
        //     Self::task_distribution_loop(workers, load_balancer, task_receiver).await;
        // });

        log::info!(
            "GPU executor pool started with {} workers",
            self.worker_count
        );
        Ok(())
    }

    async fn task_distribution_loop(
        _workers: Vec<Arc<GpuWorker>>,
        load_balancer: LoadBalancer,
        task_receiver: Arc<Mutex<mpsc::Receiver<ExecuteTask>>>,
    ) {
        loop {
            let task = {
                let mut rx = task_receiver.lock().await;
                rx.recv().await
            };

            match task {
                Some(task) => {
                    // Select worker using load balancer
                    if let Some(worker) = load_balancer.select_worker().await {
                        if let Err(e) = worker.submit_task(task).await {
                            log::error!(
                                "Failed to submit task to worker {}: {}",
                                worker.worker_id,
                                e
                            );
                        }
                    } else {
                        log::error!("No available workers for task");
                    }
                }
                None => {
                    log::info!("Task distribution loop terminating");
                    break;
                }
            }
        }
    }

    pub fn active_count(&self) -> usize {
        self.workers
            .iter()
            .map(|w| w.current_load.load(Ordering::Relaxed))
            .sum()
    }

    pub async fn shutdown(&self) -> Result<()> {
        if let Some(shutdown_tx) = &self.shutdown_tx {
            let _ = shutdown_tx.send(()).await;
        }

        // Shutdown all workers
        for worker in &self.workers {
            worker.shutdown().await?;
        }

        log::info!("GPU executor pool shutdown complete");
        Ok(())
    }
}

impl GpuWorker {
    async fn new(
        worker_id: usize,
        backend_worker: BackendWorker,
        compute_stream: ComputeStream,
        copy_stream: CopyStream,
        result_sender: mpsc::Sender<BatchResult>,
    ) -> Result<Self> {
        Ok(Self {
            worker_id,
            backend_worker,
            current_load: Arc::new(AtomicUsize::new(0)),
            compute_stream,
            copy_stream,
            active_batches: Arc::new(Mutex::new(Vec::new())),
            result_sender,
            worker_handle: None,
        })
    }

    pub async fn start(&self) -> Result<()> {
        log::debug!("GPU worker {} starting", self.worker_id);
        Ok(())
    }

    pub async fn submit_task(&self, task: ExecuteTask) -> Result<()> {
        // Update load tracking
        self.current_load
            .fetch_add(task.requests.len(), Ordering::Relaxed);

        // Track active batch
        {
            let mut active_batches = self.active_batches.lock().await;
            active_batches.push(ActiveBatch {
                batch_id: task.batch_id,
                start_time: std::time::Instant::now(),
                request_count: task.requests.len(),
            });
        }

        // Execute with stream overlap
        self.execute_with_overlap(task).await
    }

    async fn execute_with_overlap(&self, task: ExecuteTask) -> Result<()> {
        let batch_id = task.batch_id;
        let request_count = task.requests.len();

        // Convert to backend format
        let batch_input = self.convert_to_backend_batch(&task).await?;

        // Launch compute on GPU (async)
        let compute_future = self.launch_compute(&batch_input);

        // Concurrently handle any pending copies from previous batches
        let copy_future = self.handle_pending_copies();

        // Wait for both to complete
        let (forward_result, _) = tokio::join!(compute_future, copy_future);

        match forward_result {
            Ok(backend_result) => {
                // Convert backend result to our format
                let batch_result = self
                    .convert_from_backend_result(batch_id, backend_result)
                    .await?;

                // Send result
                if let Err(e) = self.result_sender.send(batch_result).await {
                    log::error!("Worker {} failed to send result: {}", self.worker_id, e);
                }
            }
            Err(e) => {
                log::error!("Worker {} compute failed: {}", self.worker_id, e);

                // Send error result
                let error_result = BatchResult {
                    batch_id,
                    results: task
                        .requests
                        .iter()
                        .map(|req| RequestResult {
                            request_id: req.request_id,
                            new_tokens: Vec::new(),
                            logprobs: Vec::new(),
                            finished: true,
                            finish_reason: Some(crate::server_engine::FinishReason::Length),
                        })
                        .collect(),
                };

                let _ = self.result_sender.send(error_result).await;
            }
        }

        // Update load tracking
        self.current_load
            .fetch_sub(request_count, Ordering::Relaxed);

        // Remove from active batches
        {
            let mut active_batches = self.active_batches.lock().await;
            active_batches.retain(|b| b.batch_id != batch_id);
        }

        Ok(())
    }

    async fn launch_compute(&self, _batch_input: &BackendBatch) -> Result<BackendResult> {
        // Use dedicated compute stream to avoid blocking
        match &self.compute_stream {
            #[cfg(feature = "cuda")]
            ComputeStream::Cuda(cuda_stream) => {
                self.backend_worker
                    .execute_cuda(batch_input, &cuda_stream.stream)
                    .await
            }
            #[cfg(feature = "metal")]
            ComputeStream::Metal(_metal_stream) => {
                // MLX doesn't expose command queues, use unit type
                self.backend_worker.execute_metal(batch_input, &()).await
            }
            #[cfg(feature = "cpu")]
            ComputeStream::Cpu => self.backend_worker.execute_cpu(batch_input).await,
            #[cfg(not(any(feature = "cuda", feature = "metal", feature = "cpu")))]
            _ => unreachable!("No backend features enabled"),
        }
    }

    async fn handle_pending_copies(&self) -> Result<()> {
        // Handle any pending D2H copies from previous batches
        // This runs concurrently with the current compute
        match &self.copy_stream {
            #[cfg(feature = "cuda")]
            CopyStream::Cuda(_cuda_stream) => {
                // Implement async copy operations
                Ok(())
            }
            #[cfg(feature = "metal")]
            CopyStream::Metal(_metal_stream) => {
                // Implement async copy operations
                Ok(())
            }
            #[cfg(feature = "cpu")]
            CopyStream::Cpu => {
                // No-op for CPU
                Ok(())
            }
            #[cfg(not(any(feature = "cuda", feature = "metal", feature = "cpu")))]
            _ => unreachable!("No backend features enabled"),
        }
    }

    async fn convert_to_backend_batch(&self, task: &ExecuteTask) -> Result<BackendBatch> {
        // Convert our ExecuteTask to backend-specific format
        Ok(BackendBatch {
            requests: task
                .requests
                .iter()
                .map(|req| super::backend_worker::BackendRequest {
                    id: req.request_id,
                    tokens: req.tokens.clone(),
                    prompt: req.prompt.clone(),
                    max_tokens: req.max_tokens,
                    sampling_params: req.sampling_params.clone(),
                })
                .collect(),
        })
    }

    async fn convert_from_backend_result(
        &self,
        batch_id: Uuid,
        result: BackendResult,
    ) -> Result<BatchResult> {
        let results = result
            .outputs
            .into_iter()
            .map(|output| RequestResult {
                request_id: output.request_id,
                new_tokens: output.new_tokens,
                logprobs: output.logprobs,
                finished: output.finished,
                finish_reason: output.finish_reason,
            })
            .collect();

        Ok(BatchResult { batch_id, results })
    }

    pub async fn shutdown(&self) -> Result<()> {
        log::debug!("GPU worker {} shutting down", self.worker_id);

        if let Some(handle) = &self.worker_handle {
            handle.abort();
        }

        Ok(())
    }
}

impl LoadBalancer {
    fn new(workers: Vec<Arc<GpuWorker>>, strategy: LoadBalancingStrategy) -> Self {
        Self {
            workers,
            round_robin_counter: AtomicUsize::new(0),
            strategy,
        }
    }

    async fn select_worker(&self) -> Option<Arc<GpuWorker>> {
        if self.workers.is_empty() {
            return None;
        }

        match self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                let index =
                    self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % self.workers.len();
                Some(self.workers[index].clone())
            }
            LoadBalancingStrategy::LeastLoaded => {
                let mut best_worker = None;
                let mut min_load = usize::MAX;

                for worker in &self.workers {
                    let load = worker.current_load.load(Ordering::Relaxed);
                    if load < min_load {
                        min_load = load;
                        best_worker = Some(worker.clone());
                    }
                }

                best_worker
            }
            LoadBalancingStrategy::Random => {
                // TODO: Fix rand dependency issue
                // let mut rng = thread_rng();
                // let index = rng.gen_range(0..self.workers.len());
                // Use first worker for now
                Some(self.workers[0].clone())
            }
        }
    }
}

impl Clone for LoadBalancer {
    fn clone(&self) -> Self {
        Self {
            workers: self.workers.clone(),
            round_robin_counter: AtomicUsize::new(0),
            strategy: self.strategy.clone(),
        }
    }
}

// Use types from backend_worker module
use super::backend_worker::{BackendBatch, BackendResult};
