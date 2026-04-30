#![allow(warnings)]
/*!
 * Runtime configuration for multi-threaded inference
 */

use serde::{Deserialize, Serialize};

/// Runtime execution mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeMode {
    /// Single-threaded execution (current implementation)
    SingleThreaded,
    /// Multi-threaded execution (new actor-based implementation)
    MultiThreaded,
}

impl Default for RuntimeMode {
    fn default() -> Self {
        Self::SingleThreaded
    }
}

/// Threading configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadingConfig {
    /// Number of tokenizer worker threads
    pub tokenizer_workers: usize,
    /// Number of detokenizer worker threads
    pub detokenizer_workers: usize,
    /// Number of GPU workers per device
    pub gpu_workers_per_device: usize,
    /// Scheduler tick interval in milliseconds
    pub scheduler_tick_ms: u64,
}

impl ThreadingConfig {
    /// Auto-tune thread counts based on system resources
    pub fn auto_tune() -> Self {
        let num_cpus = num_cpus::get();

        Self {
            tokenizer_workers: (num_cpus / 4).max(2),
            detokenizer_workers: (num_cpus / 4).max(2),
            gpu_workers_per_device: 2,
            scheduler_tick_ms: 1, // 1ms for low latency
        }
    }

    /// Conservative configuration for limited resources
    pub fn minimal() -> Self {
        Self {
            tokenizer_workers: 1,
            detokenizer_workers: 1,
            gpu_workers_per_device: 1,
            scheduler_tick_ms: 5,
        }
    }

    /// High-throughput configuration for powerful machines
    pub fn high_throughput() -> Self {
        let num_cpus = num_cpus::get();

        Self {
            tokenizer_workers: (num_cpus / 2).max(4),
            detokenizer_workers: (num_cpus / 2).max(4),
            gpu_workers_per_device: 4,
            scheduler_tick_ms: 1,
        }
    }
}

/// Scheduling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingConfig {
    /// Maximum batch size for GPU execution
    pub max_batch_size: usize,
    /// Maximum sequence length supported
    pub max_sequence_length: usize,
    /// Batch formation timeout in milliseconds
    pub batch_timeout_ms: u64,
    /// Maximum number of pending requests
    pub max_pending_requests: usize,
    /// Prefill chunk size for long sequences
    pub prefill_chunk_size: usize,
}

impl Default for SchedulingConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            max_sequence_length: 8192,
            batch_timeout_ms: 5, // 5ms max wait for batching
            max_pending_requests: 1024,
            prefill_chunk_size: 512,
        }
    }
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Fraction of GPU VRAM to allocate for KV cache
    pub kv_cache_fraction: f32,
    /// Memory pool strategy
    pub memory_pool_strategy: MemoryPoolStrategy,
    /// Worker memory quotas (per-worker allocation limits)
    pub worker_quotas: Option<WorkerMemoryQuotas>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryPoolStrategy {
    /// Shared pools with quota-based access control
    SharedWithQuotas,
    /// Per-worker dedicated pools with coordinator
    PerWorkerPools,
    /// Global shared pools (higher contention)
    GlobalShared,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerMemoryQuotas {
    /// Memory quota per tokenizer worker (MB)
    pub tokenizer_quota_mb: usize,
    /// Memory quota per GPU worker (GPU memory MB)
    pub gpu_quota_mb: usize,
    /// Memory quota per detokenizer worker (MB)
    pub detokenizer_quota_mb: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            kv_cache_fraction: 0.85,
            memory_pool_strategy: MemoryPoolStrategy::SharedWithQuotas,
            worker_quotas: Some(WorkerMemoryQuotas {
                tokenizer_quota_mb: 64,
                gpu_quota_mb: 512,
                detokenizer_quota_mb: 32,
            }),
        }
    }
}

/// Channel configuration for inter-thread communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    /// Channel buffer sizes
    pub buffer_sizes: ChannelBufferSizes,
    /// Channel timeout configuration
    pub timeouts: ChannelTimeouts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelBufferSizes {
    /// Tokenizer task queue buffer size
    pub tokenizer_buffer: usize,
    /// Scheduler task queue buffer size
    pub scheduler_buffer: usize,
    /// GPU executor task queue buffer size
    pub gpu_executor_buffer: usize,
    /// Detokenizer task queue buffer size
    pub detokenizer_buffer: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelTimeouts {
    /// Task submission timeout (ms)
    pub submit_timeout_ms: u64,
    /// Worker response timeout (ms)
    pub response_timeout_ms: u64,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            buffer_sizes: ChannelBufferSizes {
                tokenizer_buffer: 1024,
                scheduler_buffer: 1024,
                gpu_executor_buffer: 128, // Smaller for GPU work
                detokenizer_buffer: 1024,
            },
            timeouts: ChannelTimeouts {
                submit_timeout_ms: 1000,
                response_timeout_ms: 30000,
            },
        }
    }
}

/// Error handling and recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorHandlingConfig {
    /// Enable graceful degradation on worker failures
    pub enable_graceful_degradation: bool,
    /// Automatic worker restart on panic
    pub auto_restart_workers: bool,
    /// Maximum number of restart attempts
    pub max_restart_attempts: usize,
    /// Worker health check interval (ms)
    pub health_check_interval_ms: u64,
}

impl Default for ErrorHandlingConfig {
    fn default() -> Self {
        Self {
            enable_graceful_degradation: true,
            auto_restart_workers: true,
            max_restart_attempts: 3,
            health_check_interval_ms: 5000,
        }
    }
}

/// Comprehensive runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Runtime execution mode
    pub mode: RuntimeMode,
    /// Threading configuration
    pub threading: ThreadingConfig,
    /// Scheduling configuration
    pub scheduling: SchedulingConfig,
    /// Memory management configuration
    pub memory: MemoryConfig,
    /// Channel configuration
    pub channel_config: ChannelConfig,
    /// Error handling configuration
    pub error_handling: ErrorHandlingConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::default(),
            threading: ThreadingConfig::auto_tune(),
            scheduling: SchedulingConfig::default(),
            memory: MemoryConfig::default(),
            channel_config: ChannelConfig::default(),
            error_handling: ErrorHandlingConfig::default(),
        }
    }
}

impl RuntimeConfig {
    /// Create configuration optimized for development/testing
    pub fn development() -> Self {
        Self {
            mode: RuntimeMode::MultiThreaded,
            threading: ThreadingConfig::minimal(),
            scheduling: SchedulingConfig {
                max_batch_size: 16,
                max_sequence_length: 2048,
                max_pending_requests: 64,
                ..Default::default()
            },
            memory: MemoryConfig {
                kv_cache_fraction: 0.7,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create configuration optimized for production high-throughput
    pub fn production_high_throughput() -> Self {
        Self {
            mode: RuntimeMode::MultiThreaded,
            threading: ThreadingConfig::high_throughput(),
            scheduling: SchedulingConfig {
                max_batch_size: 128,
                max_sequence_length: 16384,
                max_pending_requests: 2048,
                ..Default::default()
            },
            memory: MemoryConfig {
                kv_cache_fraction: 0.9,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create configuration that maintains backward compatibility
    pub fn single_threaded_compat() -> Self {
        Self {
            mode: RuntimeMode::SingleThreaded,
            ..Default::default()
        }
    }

    /// Validate configuration for consistency and feasibility
    pub fn validate(&self) -> Result<(), String> {
        if self.threading.tokenizer_workers == 0 {
            return Err("tokenizer_workers must be > 0".to_string());
        }

        if self.threading.detokenizer_workers == 0 {
            return Err("detokenizer_workers must be > 0".to_string());
        }

        if self.threading.gpu_workers_per_device == 0 {
            return Err("gpu_workers_per_device must be > 0".to_string());
        }

        if self.scheduling.max_batch_size == 0 {
            return Err("max_batch_size must be > 0".to_string());
        }

        if self.scheduling.max_sequence_length == 0 {
            return Err("max_sequence_length must be > 0".to_string());
        }

        if self.memory.kv_cache_fraction <= 0.0 || self.memory.kv_cache_fraction > 1.0 {
            return Err("kv_cache_fraction must be between 0.0 and 1.0".to_string());
        }

        // Check for reasonable resource limits
        let total_workers = self.threading.tokenizer_workers
            + self.threading.detokenizer_workers
            + self.threading.gpu_workers_per_device;

        if total_workers > num_cpus::get() * 4 {
            return Err(format!(
                "Total workers ({}) exceeds 4x CPU count ({})",
                total_workers,
                num_cpus::get()
            ));
        }

        Ok(())
    }
}
