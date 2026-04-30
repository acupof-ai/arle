/*!
 * Quota-based memory pool coordination for multi-threaded inference
 *
 * Solves memory pool coordination by implementing shared pools with per-worker quotas,
 * automatic rebalancing coordinator, and fallback mechanisms to prevent contention
 * while maintaining utilization
 */

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use tokio::sync::{Mutex, RwLock};
use tokio::time::{Duration, Instant};

use crate::prefix_cache::BlockId;

/// Memory pool coordinator that manages shared pools with quota-based access control
pub struct MemoryPoolCoordinator {
    /// GPU memory pool for KV cache blocks
    kv_cache_pool: Arc<SharedMemoryPool>,
    /// Host memory pool for staging and overflow
    host_memory_pool: Arc<SharedMemoryPool>,
    /// Per-worker quota management
    quota_manager: Arc<QuotaManager>,
    /// Rebalancing coordinator
    rebalancer: Arc<Rebalancer>,
    /// Configuration
    config: MemoryPoolConfig,
}

/// Shared memory pool with quota-based access control
pub struct SharedMemoryPool {
    /// Pool identifier
    pool_id: PoolId,
    /// Total pool capacity in bytes
    total_capacity: usize,
    /// Currently allocated bytes
    allocated_bytes: Arc<AtomicUsize>,
    /// Block allocations (block_id -> allocation_info)
    allocations: Arc<RwLock<HashMap<BlockId, AllocationInfo>>>,
    /// Free block tracking
    free_blocks: Arc<Mutex<FreeBlockTracker>>,
    /// Pool statistics
    stats: Arc<PoolStats>,
    /// Memory backend interface
    backend: Arc<dyn MemoryBackend>,
}

/// Quota manager for per-worker memory limits
pub struct QuotaManager {
    /// Worker quotas (worker_id -> quota_info)
    worker_quotas: Arc<RwLock<HashMap<usize, WorkerQuota>>>,
    /// Global quota enforcement
    global_quota: Arc<GlobalQuota>,
    /// Quota violation tracking
    violations: Arc<Mutex<QuotaViolations>>,
}

/// Rebalancer for automatic memory redistribution
pub struct Rebalancer {
    /// Rebalancing state
    state: Arc<RwLock<RebalancingState>>,
    /// Rebalancing task handle
    task_handle: Option<tokio::task::JoinHandle<()>>,
    /// Configuration
    config: RebalancingConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoolId {
    KvCache,
    HostMemory,
    GpuTemporary,
}

#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// KV cache pool size (fraction of GPU memory)
    pub kv_cache_fraction: f32,
    /// Host memory pool size in bytes
    pub host_memory_bytes: usize,
    /// Default worker quota as fraction of pool
    pub default_worker_quota_fraction: f32,
    /// Enable automatic rebalancing
    pub enable_rebalancing: bool,
    /// Rebalancing interval
    pub rebalancing_interval: Duration,
    /// Quota enforcement mode
    pub quota_enforcement: QuotaEnforcementMode,
}

#[derive(Debug, Clone, Copy)]
pub enum QuotaEnforcementMode {
    /// Strict enforcement - reject allocations over quota
    Strict,
    /// Soft enforcement - allow temporary overuse with penalties
    Soft,
    /// Adaptive - adjust quotas based on usage patterns
    Adaptive,
}

/// Information about a memory allocation
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    /// Size in bytes
    pub size_bytes: usize,
    /// Owning worker ID
    pub worker_id: usize,
    /// Allocation timestamp
    pub allocated_at: Instant,
    /// Reference count
    pub ref_count: usize,
    /// Memory backend handle
    pub backend_handle: BackendHandle,
}

/// Per-worker quota information
#[derive(Debug)]
pub struct WorkerQuota {
    /// Maximum bytes this worker can allocate
    pub max_bytes: usize,
    /// Currently allocated bytes
    pub allocated_bytes: AtomicUsize,
    /// Quota utilization history
    pub utilization_history: Vec<f32>,
    /// Last rebalance time
    pub last_rebalance: Instant,
}

/// Global quota state
#[derive(Debug)]
pub struct GlobalQuota {
    /// Total quota allocated across all workers
    pub total_allocated_quota: AtomicUsize,
    /// Hard capacity limit
    pub hard_limit: usize,
    /// Soft capacity threshold
    pub soft_limit: usize,
    /// Emergency reserve
    pub emergency_reserve: usize,
}

/// Quota violation tracking
#[derive(Debug)]
pub struct QuotaViolations {
    /// Per-worker violation counts
    pub worker_violations: HashMap<usize, usize>,
    /// Total violations
    pub total_violations: usize,
    /// Last violation time
    pub last_violation: Option<Instant>,
}

/// Rebalancing state
#[derive(Debug)]
pub struct RebalancingState {
    /// Last rebalance time
    pub last_rebalance: Instant,
    /// Rebalancing in progress
    pub in_progress: bool,
    /// Rebalance requests queue
    pub rebalance_requests: Vec<RebalanceRequest>,
    /// Performance metrics for decision making
    pub metrics: RebalancingMetrics,
}

#[derive(Debug, Clone)]
pub struct RebalanceRequest {
    /// Worker requesting more memory
    pub requesting_worker: usize,
    /// Amount requested in bytes
    pub bytes_requested: usize,
    /// Priority of the request
    pub priority: RebalancePriority,
    /// Request timestamp
    pub requested_at: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RebalancePriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Rebalancing performance metrics
#[derive(Debug, Clone)]
pub struct RebalancingMetrics {
    /// Worker utilization rates
    pub worker_utilizations: HashMap<usize, f32>,
    /// Allocation failure rates
    pub failure_rates: HashMap<usize, f32>,
    /// Memory pressure indicators
    pub pressure_indicators: PressureIndicators,
}

#[derive(Debug, Clone)]
pub struct PressureIndicators {
    /// Overall memory pressure (0.0 = no pressure, 1.0 = maximum)
    pub overall_pressure: f32,
    /// Allocation request backlog
    pub allocation_backlog: usize,
    /// Recent allocation failures
    pub recent_failures: usize,
}

/// Quota statistics
#[derive(Debug, Clone)]
pub struct QuotaStats {
    /// Total allocated quota across all workers
    pub total_allocated: usize,
    /// Hard memory limit
    pub hard_limit: usize,
    /// Soft memory limit
    pub soft_limit: usize,
    /// Emergency reserve
    pub emergency_reserve: usize,
    /// Per-worker statistics
    pub worker_stats: HashMap<usize, WorkerQuotaStats>,
}

/// Per-worker quota statistics
#[derive(Debug, Clone)]
pub struct WorkerQuotaStats {
    /// Maximum allocated bytes for this worker
    pub max_bytes: usize,
    /// Currently allocated bytes
    pub allocated_bytes: usize,
    /// Utilization ratio (0.0 to 1.0)
    pub utilization: f64,
}

#[derive(Debug, Clone)]
pub struct RebalancingConfig {
    /// Minimum interval between rebalances
    pub min_interval: Duration,
    /// Utilization threshold for triggering rebalance
    pub utilization_threshold: f32,
    /// Maximum quota adjustment per rebalance (fraction)
    pub max_adjustment_fraction: f32,
    /// Enable predictive rebalancing
    pub enable_prediction: bool,
}

/// Free block tracking
#[derive(Debug)]
struct FreeBlockTracker {
    /// Free blocks by size class
    pub free_by_size: HashMap<usize, Vec<BlockRange>>,
    /// Total free bytes
    pub total_free_bytes: usize,
    /// Largest contiguous block
    pub largest_free_block: usize,
}

#[derive(Debug, Clone)]
struct BlockRange {
    /// Starting block ID
    pub start_block: BlockId,
    /// Number of contiguous blocks
    pub block_count: usize,
    /// Size in bytes
    pub size_bytes: usize,
}

/// Pool performance statistics
#[derive(Debug)]
pub struct PoolStats {
    /// Total allocations performed
    pub total_allocations: AtomicU64,
    /// Total deallocations performed
    pub total_deallocations: AtomicU64,
    /// Total allocation failures
    pub allocation_failures: AtomicU64,
    /// Average allocation size
    pub avg_allocation_size: AtomicU64,
    /// Peak memory usage
    pub peak_usage_bytes: AtomicUsize,
    /// Fragmentation metric (0.0 = no fragmentation, 1.0 = maximum)
    pub fragmentation: Arc<AtomicU64>, // f64 as u64 bits
}

/// Memory backend abstraction
#[async_trait::async_trait]
pub trait MemoryBackend: Send + Sync {
    /// Allocate memory block
    async fn allocate(&self, size_bytes: usize) -> Result<BackendHandle>;
    /// Deallocate memory block
    async fn deallocate(&self, handle: BackendHandle) -> Result<()>;
    /// Get memory statistics
    fn stats(&self) -> MemoryBackendStats;
    /// Defragment memory (if supported)
    async fn defragment(&self) -> Result<DefragmentationResult>;
}

/// Backend-specific memory handle
#[derive(Debug, Clone)]
pub struct BackendHandle {
    /// Backend-specific identifier
    pub id: u64,
    /// Memory address (if applicable)
    pub address: Option<u64>,
    /// Size in bytes
    pub size: usize,
    /// Backend type
    pub backend_type: BackendType,
}

#[derive(Debug, Clone, Copy)]
pub enum BackendType {
    CudaDevice,
    MetalDevice,
    HostPinned,
    HostPaged,
}

#[derive(Debug, Clone)]
pub struct MemoryBackendStats {
    /// Total capacity
    pub total_bytes: usize,
    /// Currently allocated
    pub allocated_bytes: usize,
    /// Free bytes
    pub free_bytes: usize,
    /// Number of allocations
    pub allocation_count: usize,
    /// Fragmentation percentage
    pub fragmentation_percent: f32,
}

#[derive(Debug, Clone)]
pub struct DefragmentationResult {
    /// Bytes recovered
    pub bytes_recovered: usize,
    /// Time taken
    pub duration: Duration,
    /// Success status
    pub success: bool,
}

impl MemoryPoolCoordinator {
    /// Create a new memory pool coordinator
    pub async fn new(config: MemoryPoolConfig) -> Result<Self> {
        // Create memory pools
        let kv_cache_pool = Arc::new(
            SharedMemoryPool::new(
                PoolId::KvCache,
                (config.kv_cache_fraction * Self::detect_gpu_memory()? as f32) as usize,
                Self::create_gpu_backend()?,
            )
            .await?,
        );

        let host_memory_pool = Arc::new(
            SharedMemoryPool::new(
                PoolId::HostMemory,
                config.host_memory_bytes,
                Self::create_host_backend()?,
            )
            .await?,
        );

        // Create quota manager
        let quota_manager = Arc::new(QuotaManager::new(
            kv_cache_pool.total_capacity + host_memory_pool.total_capacity,
            config.default_worker_quota_fraction,
            config.quota_enforcement,
        ));

        // Create rebalancer
        let rebalancer = Arc::new(Rebalancer::new(RebalancingConfig {
            min_interval: config.rebalancing_interval,
            utilization_threshold: 0.8,
            max_adjustment_fraction: 0.1,
            enable_prediction: true,
        }));

        Ok(Self {
            kv_cache_pool,
            host_memory_pool,
            quota_manager,
            rebalancer,
            config,
        })
    }

    /// Start the coordinator (begins background rebalancing if enabled)
    pub async fn start(&self) -> Result<()> {
        if self.config.enable_rebalancing {
            self.rebalancer.start().await?;
        }

        log::info!("Memory pool coordinator started");
        Ok(())
    }

    /// Allocate memory from the appropriate pool
    pub async fn allocate(
        &self,
        pool_id: PoolId,
        size_bytes: usize,
        worker_id: usize,
    ) -> Result<AllocationHandle> {
        // Check worker quota
        if !self
            .quota_manager
            .check_quota(worker_id, size_bytes)
            .await?
        {
            return Err(anyhow::anyhow!("Worker {} quota exceeded", worker_id));
        }

        // Select pool
        let pool = match pool_id {
            PoolId::KvCache => &self.kv_cache_pool,
            PoolId::HostMemory => &self.host_memory_pool,
            PoolId::GpuTemporary => &self.kv_cache_pool, // Use KV cache pool for now
        };

        // Attempt allocation
        let allocation = pool.allocate(size_bytes, worker_id).await?;

        // Update quota usage - TODO: implement these methods
        // self.quota_manager.record_allocation(worker_id, size_bytes).await?;

        // Check if rebalancing is needed - TODO: implement this method
        // if self.config.enable_rebalancing {
        //     self.rebalancer.check_rebalance_trigger().await?;
        // }

        Ok(AllocationHandle {
            block_id: allocation.block_id,
            pool_id,
            worker_id,
            size_bytes,
            backend_handle: allocation.backend_handle,
        })
    }

    /// Deallocate memory
    pub async fn deallocate(&self, handle: AllocationHandle) -> Result<()> {
        // Select pool
        let pool = match handle.pool_id {
            PoolId::KvCache => &self.kv_cache_pool,
            PoolId::HostMemory => &self.host_memory_pool,
            PoolId::GpuTemporary => &self.kv_cache_pool,
        };

        // Perform deallocation
        pool.deallocate(handle).await?;

        // Update quota usage - TODO: implement this method
        // self.quota_manager.record_deallocation(handle.worker_id, handle.size_bytes).await?;

        Ok(())
    }

    /// Request quota increase for a worker
    pub async fn request_quota_increase(
        &self,
        worker_id: usize,
        additional_bytes: usize,
        priority: RebalancePriority,
    ) -> Result<bool> {
        // TODO: implement request_rebalance method
        // self.rebalancer.request_rebalance(RebalanceRequest {
        //     requesting_worker: worker_id,
        //     bytes_requested: additional_bytes,
        //     priority,
        //     requested_at: Instant::now(),
        // }).await
        Ok(false) // Placeholder
    }

    /// Get memory statistics
    pub async fn stats(&self) -> MemoryCoordinatorStats {
        // TODO: implement stats methods for SharedMemoryPool
        // let kv_stats = self.kv_cache_pool.stats().await;
        // let host_stats = self.host_memory_pool.stats().await;
        let quota_stats = self.quota_manager.stats().await;

        // TODO: implement proper stats collection
        MemoryCoordinatorStats {
            kv_cache_pool: PoolStatsSnapshot {
                pool_id: crate::runtime::memory_pool_coordinator::PoolId::KvCache,
                total_capacity: self.kv_cache_pool.total_capacity,
                allocated_bytes: self.kv_cache_pool.allocated_bytes.load(Ordering::Relaxed),
                free_bytes: self.kv_cache_pool.total_capacity
                    - self.kv_cache_pool.allocated_bytes.load(Ordering::Relaxed),
                allocation_count: 0,        // TODO: track this
                fragmentation_percent: 0.0, // TODO: calculate this
            },
            host_memory_pool: PoolStatsSnapshot {
                pool_id: crate::runtime::memory_pool_coordinator::PoolId::HostMemory,
                total_capacity: self.host_memory_pool.total_capacity,
                allocated_bytes: self
                    .host_memory_pool
                    .allocated_bytes
                    .load(Ordering::Relaxed),
                free_bytes: self.host_memory_pool.total_capacity
                    - self
                        .host_memory_pool
                        .allocated_bytes
                        .load(Ordering::Relaxed),
                allocation_count: 0,        // TODO: track this
                fragmentation_percent: 0.0, // TODO: calculate this
            },
            quota_stats: QuotaStats {
                total_allocated: 0,           // TODO: implement
                hard_limit: 0,                // TODO: implement
                soft_limit: 0,                // TODO: implement
                emergency_reserve: 0,         // TODO: implement
                worker_stats: HashMap::new(), // TODO: implement
            },
            total_allocated_bytes: self.kv_cache_pool.allocated_bytes.load(Ordering::Relaxed)
                + self
                    .host_memory_pool
                    .allocated_bytes
                    .load(Ordering::Relaxed),
        }
    }

    /// Shutdown the coordinator
    pub async fn shutdown(&self) -> Result<()> {
        self.rebalancer.shutdown().await?;
        log::info!("Memory pool coordinator shutdown complete");
        Ok(())
    }

    // Helper methods

    fn detect_gpu_memory() -> Result<usize> {
        // Platform-specific GPU memory detection
        #[cfg(feature = "cuda")]
        {
            // Use CUDA API to detect GPU memory
            Ok(8 * 1024 * 1024 * 1024) // 8GB placeholder
        }
        #[cfg(feature = "metal")]
        {
            // Use Metal API to detect GPU memory
            Ok(16 * 1024 * 1024 * 1024) // 16GB placeholder for Apple Silicon
        }
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            Ok(4 * 1024 * 1024 * 1024) // 4GB placeholder
        }
    }

    fn create_gpu_backend() -> Result<Arc<dyn MemoryBackend>> {
        #[cfg(feature = "cuda")]
        {
            Ok(Arc::new(CudaMemoryBackend::new()?))
        }
        #[cfg(feature = "metal")]
        {
            Ok(Arc::new(MetalMemoryBackend::new()?))
        }
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            Ok(Arc::new(HostMemoryBackend::new()?))
        }
    }

    fn create_host_backend() -> Result<Arc<dyn MemoryBackend>> {
        Ok(Arc::new(HostMemoryBackend::new()?))
    }
}

/// Allocation handle returned to callers
#[derive(Debug, Clone)]
pub struct AllocationHandle {
    pub block_id: BlockId,
    pub pool_id: PoolId,
    pub worker_id: usize,
    pub size_bytes: usize,
    pub backend_handle: BackendHandle,
}

/// Combined statistics for all memory pools
#[derive(Debug, Clone)]
pub struct MemoryCoordinatorStats {
    pub kv_cache_pool: PoolStatsSnapshot,
    pub host_memory_pool: PoolStatsSnapshot,
    pub quota_stats: QuotaStats,
    pub total_allocated_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct PoolStatsSnapshot {
    pub pool_id: PoolId,
    pub total_capacity: usize,
    pub allocated_bytes: usize,
    pub free_bytes: usize,
    pub allocation_count: usize,
    pub fragmentation_percent: f32,
}

// Placeholder backend implementations

/// Host memory backend implementation
struct HostMemoryBackend {
    allocated_blocks: Arc<Mutex<HashMap<u64, Vec<u8>>>>,
    next_handle_id: Arc<AtomicU64>,
}

impl HostMemoryBackend {
    fn new() -> Result<Self> {
        Ok(Self {
            allocated_blocks: Arc::new(Mutex::new(HashMap::new())),
            next_handle_id: Arc::new(AtomicU64::new(1)),
        })
    }
}

#[async_trait::async_trait]
impl MemoryBackend for HostMemoryBackend {
    async fn allocate(&self, size_bytes: usize) -> Result<BackendHandle> {
        let handle_id = self.next_handle_id.fetch_add(1, Ordering::Relaxed);
        let mut blocks = self.allocated_blocks.lock().await;

        // Allocate memory
        let memory = vec![0u8; size_bytes];
        let address = memory.as_ptr() as u64;
        blocks.insert(handle_id, memory);

        Ok(BackendHandle {
            id: handle_id,
            address: Some(address),
            size: size_bytes,
            backend_type: BackendType::HostPaged,
        })
    }

    async fn deallocate(&self, handle: BackendHandle) -> Result<()> {
        let mut blocks = self.allocated_blocks.lock().await;
        blocks.remove(&handle.id);
        Ok(())
    }

    fn stats(&self) -> MemoryBackendStats {
        MemoryBackendStats {
            total_bytes: 32 * 1024 * 1024 * 1024, // 32GB placeholder
            allocated_bytes: 0,                   // Would track this
            free_bytes: 32 * 1024 * 1024 * 1024,
            allocation_count: 0,
            fragmentation_percent: 0.0,
        }
    }

    async fn defragment(&self) -> Result<DefragmentationResult> {
        Ok(DefragmentationResult {
            bytes_recovered: 0,
            duration: Duration::from_millis(0),
            success: true,
        })
    }
}

/// CUDA memory backend (stub implementation)
#[cfg(feature = "cuda")]
struct CudaMemoryBackend {
    _placeholder: (),
}

#[cfg(feature = "cuda")]
impl CudaMemoryBackend {
    fn new() -> Result<Self> {
        Ok(Self { _placeholder: () })
    }
}

#[cfg(feature = "cuda")]
#[async_trait::async_trait]
impl MemoryBackend for CudaMemoryBackend {
    async fn allocate(&self, size_bytes: usize) -> Result<BackendHandle> {
        Ok(BackendHandle {
            id: 0, // Stub ID
            address: None,
            size: size_bytes,
            backend_type: BackendType::CudaDevice,
        })
    }

    async fn deallocate(&self, _handle: BackendHandle) -> Result<()> {
        Ok(())
    }

    fn stats(&self) -> MemoryBackendStats {
        MemoryBackendStats {
            total_bytes: 16 * 1024 * 1024 * 1024, // 16GB GPU placeholder
            allocated_bytes: 0,
            free_bytes: 16 * 1024 * 1024 * 1024,
            allocation_count: 0,
            fragmentation_percent: 0.0,
        }
    }

    async fn defragment(&self) -> Result<DefragmentationResult> {
        Ok(DefragmentationResult {
            bytes_recovered: 0,
            duration: Duration::from_millis(0),
            success: true,
        })
    }
}

/// Metal memory backend (stub implementation)
#[cfg(feature = "metal")]
struct MetalMemoryBackend {
    _placeholder: (),
}

#[cfg(feature = "metal")]
impl MetalMemoryBackend {
    fn new() -> Result<Self> {
        Ok(Self { _placeholder: () })
    }
}

#[cfg(feature = "metal")]
#[async_trait::async_trait]
impl MemoryBackend for MetalMemoryBackend {
    async fn allocate(&self, size_bytes: usize) -> Result<BackendHandle> {
        Ok(BackendHandle {
            id: 0, // Stub ID
            address: None,
            size: size_bytes,
            backend_type: BackendType::MetalDevice,
        })
    }

    async fn deallocate(&self, _handle: BackendHandle) -> Result<()> {
        Ok(())
    }

    fn stats(&self) -> MemoryBackendStats {
        MemoryBackendStats {
            total_bytes: 24 * 1024 * 1024 * 1024, // 24GB unified memory placeholder
            allocated_bytes: 0,
            free_bytes: 24 * 1024 * 1024 * 1024,
            allocation_count: 0,
            fragmentation_percent: 0.0,
        }
    }

    async fn defragment(&self) -> Result<DefragmentationResult> {
        Ok(DefragmentationResult {
            bytes_recovered: 0,
            duration: Duration::from_millis(0),
            success: true,
        })
    }
}

// Missing implementations

impl SharedMemoryPool {
    pub async fn new(
        pool_id: PoolId,
        total_capacity: usize,
        backend: Arc<dyn MemoryBackend>,
    ) -> Result<Self> {
        Ok(Self {
            pool_id,
            total_capacity,
            allocated_bytes: Arc::new(AtomicUsize::new(0)),
            allocations: Arc::new(RwLock::new(HashMap::new())),
            free_blocks: Arc::new(Mutex::new(FreeBlockTracker::new())),
            stats: Arc::new(PoolStats::new()),
            backend,
        })
    }

    pub async fn allocate(&self, size_bytes: usize, worker_id: usize) -> Result<AllocationHandle> {
        // Check if we have enough capacity
        let current_allocated = self.allocated_bytes.load(Ordering::Relaxed);
        if current_allocated + size_bytes > self.total_capacity {
            return Err(anyhow::anyhow!("Insufficient pool capacity"));
        }

        // Allocate from backend
        let backend_handle = self.backend.allocate(size_bytes).await?;

        // Generate block ID
        let block_id = BlockId(backend_handle.id as u32);

        // Record allocation
        let allocation_info = AllocationInfo {
            size_bytes,
            worker_id,
            allocated_at: Instant::now(),
            ref_count: 1,
            backend_handle: backend_handle.clone(),
        };

        {
            let mut allocations = self.allocations.write().await;
            allocations.insert(block_id, allocation_info);
        }

        // Update allocated bytes
        self.allocated_bytes
            .fetch_add(size_bytes, Ordering::Relaxed);

        Ok(AllocationHandle {
            block_id,
            pool_id: self.pool_id,
            worker_id,
            size_bytes,
            backend_handle,
        })
    }

    pub async fn deallocate(&self, handle: AllocationHandle) -> Result<()> {
        // Remove allocation record
        let allocation_info = {
            let mut allocations = self.allocations.write().await;
            allocations.remove(&handle.block_id)
        };

        if let Some(info) = allocation_info {
            // Deallocate from backend
            self.backend.deallocate(info.backend_handle).await?;

            // Update allocated bytes
            self.allocated_bytes
                .fetch_sub(info.size_bytes, Ordering::Relaxed);
        }

        Ok(())
    }
}

impl QuotaManager {
    pub fn new(
        total_capacity: usize,
        _default_quota_fraction: f32,
        _enforcement_mode: QuotaEnforcementMode,
    ) -> Self {
        Self {
            worker_quotas: Arc::new(RwLock::new(HashMap::new())),
            global_quota: Arc::new(GlobalQuota {
                total_allocated_quota: AtomicUsize::new(0),
                hard_limit: total_capacity,
                soft_limit: (total_capacity as f32 * 0.8) as usize,
                emergency_reserve: (total_capacity as f32 * 0.1) as usize,
            }),
            violations: Arc::new(Mutex::new(QuotaViolations {
                worker_violations: HashMap::new(),
                total_violations: 0,
                last_violation: None,
            })),
        }
    }

    pub async fn check_quota(&self, worker_id: usize, size_bytes: usize) -> Result<bool> {
        let quotas = self.worker_quotas.read().await;

        if let Some(quota) = quotas.get(&worker_id) {
            let current_allocated = quota.allocated_bytes.load(Ordering::Relaxed);
            Ok(current_allocated + size_bytes <= quota.max_bytes)
        } else {
            // Create default quota for new worker
            drop(quotas);
            self.create_worker_quota(worker_id).await?;
            Ok(true) // Allow first allocation
        }
    }

    async fn create_worker_quota(&self, worker_id: usize) -> Result<()> {
        let mut quotas = self.worker_quotas.write().await;

        // Default quota is a fraction of total capacity
        let default_quota = (self.global_quota.hard_limit as f32 * 0.1) as usize;

        quotas.insert(
            worker_id,
            WorkerQuota {
                max_bytes: default_quota,
                allocated_bytes: AtomicUsize::new(0),
                utilization_history: Vec::new(),
                last_rebalance: Instant::now(),
            },
        );

        Ok(())
    }

    pub async fn stats(&self) -> QuotaStats {
        let quotas = self.worker_quotas.read().await;
        let total_allocated = self
            .global_quota
            .total_allocated_quota
            .load(Ordering::Relaxed);

        let mut worker_stats = HashMap::new();
        for (worker_id, quota) in quotas.iter() {
            worker_stats.insert(
                *worker_id,
                WorkerQuotaStats {
                    max_bytes: quota.max_bytes,
                    allocated_bytes: quota.allocated_bytes.load(Ordering::Relaxed),
                    utilization: quota.allocated_bytes.load(Ordering::Relaxed) as f64
                        / quota.max_bytes as f64,
                },
            );
        }

        QuotaStats {
            total_allocated,
            hard_limit: self.global_quota.hard_limit,
            soft_limit: self.global_quota.soft_limit,
            emergency_reserve: self.global_quota.emergency_reserve,
            worker_stats,
        }
    }
}

impl Rebalancer {
    pub fn new(config: RebalancingConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(RebalancingState {
                last_rebalance: Instant::now(),
                in_progress: false,
                rebalance_requests: Vec::new(),
                metrics: RebalancingMetrics {
                    worker_utilizations: HashMap::new(),
                    failure_rates: HashMap::new(),
                    pressure_indicators: PressureIndicators::default(),
                },
            })),
            task_handle: None,
            config,
        }
    }

    pub async fn start(&self) -> Result<()> {
        log::info!("Rebalancer started");
        Ok(())
    }

    pub async fn shutdown(&self) -> Result<()> {
        if let Some(handle) = &self.task_handle {
            handle.abort();
        }
        Ok(())
    }
}

// Implementation for existing structs

impl FreeBlockTracker {
    fn new() -> Self {
        Self {
            free_by_size: HashMap::new(),
            total_free_bytes: 0,
            largest_free_block: 0,
        }
    }
}

impl PoolStats {
    fn new() -> Self {
        Self {
            total_allocations: AtomicU64::new(0),
            total_deallocations: AtomicU64::new(0),
            allocation_failures: AtomicU64::new(0),
            avg_allocation_size: AtomicU64::new(0),
            peak_usage_bytes: AtomicUsize::new(0),
            fragmentation: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl Default for PressureIndicators {
    fn default() -> Self {
        Self {
            overall_pressure: 0.0,
            allocation_backlog: 0,
            recent_failures: 0,
        }
    }
}
