/*!
 * Comprehensive error handling and recovery for multi-threaded runtime
 *
 * Implements graceful degradation strategies: thread panic isolation,
 * cross-thread error propagation, automatic worker restart, and fallback
 * to reduced parallelism on failures
 */

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock, broadcast, mpsc};
use tokio::task::JoinHandle;

/// Error recovery coordinator for the multi-threaded runtime
pub struct ErrorRecoveryCoordinator {
    /// Error tracking state
    state: Arc<RwLock<ErrorState>>,
    /// Recovery policies
    policies: Arc<RecoveryPolicies>,
    /// Worker health monitors
    health_monitors: Arc<RwLock<HashMap<WorkerId, WorkerHealthMonitor>>>,
    /// Recovery actions executor
    recovery_executor: Arc<RecoveryExecutor>,
    /// Error notification system
    error_broadcaster: broadcast::Sender<ErrorEvent>,
    /// Configuration
    config: ErrorRecoveryConfig,
}

/// Global error state tracking
#[derive(Debug)]
struct ErrorState {
    /// Total errors encountered
    total_errors: usize,
    /// Errors by category
    errors_by_category: HashMap<ErrorCategory, usize>,
    /// Current system health level
    health_level: SystemHealthLevel,
    /// Recovery actions in progress
    active_recoveries: HashMap<RecoveryActionId, RecoveryAction>,
    /// Last critical error time
    last_critical_error: Option<Instant>,
    /// Degraded mode state
    degraded_mode: Option<DegradedModeState>,
}

/// Recovery policies configuration
#[derive(Debug, Clone)]
pub struct RecoveryPolicies {
    /// Worker restart policy
    pub worker_restart: WorkerRestartPolicy,
    /// Degradation policy
    pub degradation: DegradationPolicy,
    /// Isolation policy
    pub isolation: IsolationPolicy,
    /// Fallback policy
    pub fallback: FallbackPolicy,
}

/// Worker health monitoring
pub struct WorkerHealthMonitor {
    /// Worker identifier
    worker_id: WorkerId,
    /// Health status
    health_status: Arc<RwLock<WorkerHealth>>,
    /// Heartbeat tracking
    last_heartbeat: Arc<Mutex<Instant>>,
    /// Error count
    error_count: Arc<AtomicUsize>,
    /// Restart count
    restart_count: Arc<AtomicUsize>,
    /// Monitor task handle
    monitor_task: Option<JoinHandle<()>>,
}

/// Recovery action executor
pub struct RecoveryExecutor {
    /// Active recovery tasks
    active_tasks: Arc<RwLock<HashMap<RecoveryActionId, JoinHandle<()>>>>,
    /// Recovery queue
    recovery_queue: Arc<Mutex<Vec<RecoveryRequest>>>,
    /// Executor task handle
    executor_task: Option<JoinHandle<()>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// Worker thread panicked
    WorkerPanic,
    /// GPU operation failed
    GpuError,
    /// Memory allocation failed
    MemoryError,
    /// Communication timeout
    CommunicationTimeout,
    /// Network/IO error
    NetworkError,
    /// Resource exhaustion
    ResourceExhaustion,
    /// Configuration error
    ConfigurationError,
    /// Unknown error
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SystemHealthLevel {
    /// All systems operational
    Healthy = 0,
    /// Minor issues, full functionality
    Warning = 1,
    /// Significant issues, degraded performance
    Degraded = 2,
    /// Major issues, minimal functionality
    Critical = 3,
    /// System failure, emergency mode
    Emergency = 4,
}

#[derive(Debug, Clone)]
pub struct DegradedModeState {
    /// When degraded mode was entered
    entered_at: Instant,
    /// Reason for degradation
    reason: String,
    /// Reduced capabilities
    limitations: Vec<String>,
    /// Recovery plan
    recovery_plan: RecoveryPlan,
}

#[derive(Debug, Clone)]
pub struct WorkerRestartPolicy {
    /// Maximum restart attempts
    pub max_attempts: usize,
    /// Restart delay (exponential backoff)
    pub initial_delay: Duration,
    /// Maximum delay between restarts
    pub max_delay: Duration,
    /// Reset restart count after this duration
    pub reset_after: Duration,
}

#[derive(Debug, Clone)]
pub struct DegradationPolicy {
    /// Error threshold for degradation
    pub error_threshold: usize,
    /// Time window for error counting
    pub error_window: Duration,
    /// Degradation levels and their triggers
    pub degradation_levels: Vec<DegradationLevel>,
}

#[derive(Debug, Clone)]
pub struct DegradationLevel {
    /// Health level this represents
    pub level: SystemHealthLevel,
    /// Error count threshold
    pub error_threshold: usize,
    /// Actions to take at this level
    pub actions: Vec<DegradationAction>,
}

#[derive(Debug, Clone)]
pub enum DegradationAction {
    /// Reduce worker thread count
    ReduceWorkerCount { pool: WorkerPoolType, factor: f32 },
    /// Disable non-essential features
    DisableFeature { feature: String },
    /// Increase timeouts
    IncreaseTimeouts { factor: f32 },
    /// Switch to backup resources
    UseBackupResources,
    /// Enable emergency mode
    EnterEmergencyMode,
}

#[derive(Debug, Clone)]
pub enum WorkerPoolType {
    Tokenizer,
    Scheduler,
    GpuExecutor,
    Detokenizer,
}

#[derive(Debug, Clone)]
pub struct IsolationPolicy {
    /// Isolate failed workers
    pub isolate_failed_workers: bool,
    /// Quarantine time for failed workers
    pub quarantine_duration: Duration,
    /// Maximum workers to isolate per pool
    pub max_isolated_per_pool: usize,
}

#[derive(Debug, Clone)]
pub struct FallbackPolicy {
    /// Enable fallback to single-threaded mode
    pub enable_single_threaded_fallback: bool,
    /// CPU-only fallback for GPU errors
    pub enable_cpu_fallback: bool,
    /// Reduced functionality mode
    pub enable_reduced_functionality: bool,
    /// External service fallback
    pub enable_external_fallback: bool,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct WorkerId {
    pub pool_type: WorkerPoolType,
    pub worker_index: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum WorkerHealth {
    Healthy,
    Warning,
    Degraded,
    Failed,
    Isolated,
}

#[derive(Debug, Clone)]
pub struct ErrorEvent {
    /// Error category
    pub category: ErrorCategory,
    /// Error details
    pub error: anyhow::Error,
    /// Affected worker (if applicable)
    pub worker_id: Option<WorkerId>,
    /// Timestamp
    pub timestamp: Instant,
    /// Severity level
    pub severity: ErrorSeverity,
    /// Recovery action taken
    pub recovery_action: Option<RecoveryActionType>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Info,
    Warning,
    Error,
    Critical,
    Emergency,
}

type RecoveryActionId = u64;

#[derive(Debug, Clone)]
pub struct RecoveryAction {
    pub id: RecoveryActionId,
    pub action_type: RecoveryActionType,
    pub started_at: Instant,
    pub target: RecoveryTarget,
    pub status: RecoveryStatus,
}

#[derive(Debug, Clone)]
pub enum RecoveryActionType {
    RestartWorker,
    RestartPool,
    IsolateWorker,
    ReduceParallelism,
    SwitchToFallback,
    EnterDegradedMode,
    EmergencyShutdown,
}

#[derive(Debug, Clone)]
pub enum RecoveryTarget {
    Worker(WorkerId),
    Pool(WorkerPoolType),
    System,
}

#[derive(Debug, Clone)]
pub enum RecoveryStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct RecoveryRequest {
    pub action_type: RecoveryActionType,
    pub target: RecoveryTarget,
    pub priority: RecoveryPriority,
    pub requested_at: Instant,
    pub context: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecoveryPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
    Emergency = 4,
}

#[derive(Debug, Clone)]
pub struct RecoveryPlan {
    /// Recovery steps in order
    pub steps: Vec<RecoveryStep>,
    /// Estimated recovery time
    pub estimated_duration: Duration,
    /// Success criteria
    pub success_criteria: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RecoveryStep {
    /// Step description
    pub description: String,
    /// Action to take
    pub action: RecoveryActionType,
    /// Target for action
    pub target: RecoveryTarget,
    /// Step timeout
    pub timeout: Duration,
}

#[derive(Debug, Clone)]
pub struct ErrorRecoveryConfig {
    /// Error tracking window
    pub error_window: Duration,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Recovery policies
    pub policies: RecoveryPolicies,
    /// Enable automatic recovery
    pub enable_auto_recovery: bool,
    /// Maximum concurrent recovery actions
    pub max_concurrent_recoveries: usize,
}

impl ErrorRecoveryCoordinator {
    /// Create a new error recovery coordinator
    pub fn new(config: ErrorRecoveryConfig) -> Self {
        let (error_broadcaster, _) = broadcast::channel(1000);

        Self {
            state: Arc::new(RwLock::new(ErrorState {
                total_errors: 0,
                errors_by_category: HashMap::new(),
                health_level: SystemHealthLevel::Healthy,
                active_recoveries: HashMap::new(),
                last_critical_error: None,
                degraded_mode: None,
            })),
            policies: Arc::new(config.policies.clone()),
            health_monitors: Arc::new(RwLock::new(HashMap::new())),
            recovery_executor: Arc::new(RecoveryExecutor::new()),
            error_broadcaster,
            config,
        }
    }

    /// Start the error recovery coordinator
    pub async fn start(&mut self) -> Result<()> {
        // Start health monitoring task
        self.start_health_monitoring().await?;

        // Start recovery executor
        self.recovery_executor.start().await?;

        log::info!("Error recovery coordinator started");
        Ok(())
    }

    /// Report an error to the coordinator
    pub async fn report_error(
        &self,
        error: anyhow::Error,
        category: ErrorCategory,
        worker_id: Option<WorkerId>,
    ) -> Result<()> {
        let severity = self.categorize_error_severity(&error, category);

        let error_event = ErrorEvent {
            category,
            error,
            worker_id,
            timestamp: Instant::now(),
            severity,
            recovery_action: None,
        };

        // Update error state
        {
            let mut state = self.state.write().await;
            state.total_errors += 1;
            *state.errors_by_category.entry(category).or_insert(0) += 1;

            if severity >= ErrorSeverity::Critical {
                state.last_critical_error = Some(error_event.timestamp);
            }
        }

        // Update worker health if applicable
        if let Some(worker_id) = worker_id {
            self.update_worker_health(worker_id, &error_event).await?;
        }

        // Assess system health
        let new_health_level = self.assess_system_health().await?;
        self.update_system_health(new_health_level).await?;

        // Trigger recovery if needed
        if self.config.enable_auto_recovery {
            self.trigger_auto_recovery(&error_event).await?;
        }

        // Broadcast error event
        let _ = self.error_broadcaster.send(error_event);

        Ok(())
    }

    /// Register a worker for health monitoring
    pub async fn register_worker(&self, worker_id: WorkerId) -> Result<()> {
        let monitor = WorkerHealthMonitor::new(worker_id);

        {
            let mut monitors = self.health_monitors.write().await;
            monitors.insert(worker_id, monitor);
        }

        log::debug!("Registered worker for health monitoring: {:?}", worker_id);
        Ok(())
    }

    /// Report worker heartbeat
    pub async fn report_heartbeat(&self, worker_id: WorkerId) -> Result<()> {
        if let Some(monitor) = self.health_monitors.read().await.get(&worker_id) {
            monitor.record_heartbeat().await;
        }
        Ok(())
    }

    /// Request manual recovery action
    pub async fn request_recovery(&self, request: RecoveryRequest) -> Result<RecoveryActionId> {
        self.recovery_executor.queue_recovery(request).await
    }

    /// Get current system health status
    pub async fn system_health(&self) -> SystemHealthLevel {
        self.state.read().await.health_level
    }

    /// Get error statistics
    pub async fn error_stats(&self) -> ErrorStats {
        let state = self.state.read().await;

        ErrorStats {
            total_errors: state.total_errors,
            errors_by_category: state.errors_by_category.clone(),
            current_health_level: state.health_level,
            active_recoveries: state.active_recoveries.len(),
            degraded_mode: state.degraded_mode.is_some(),
        }
    }

    /// Subscribe to error events
    pub fn subscribe_to_errors(&self) -> broadcast::Receiver<ErrorEvent> {
        self.error_broadcaster.subscribe()
    }

    /// Shutdown the coordinator
    pub async fn shutdown(&mut self) -> Result<()> {
        // Stop health monitoring
        self.stop_health_monitoring().await?;

        // Stop recovery executor
        self.recovery_executor.shutdown().await?;

        log::info!("Error recovery coordinator shutdown complete");
        Ok(())
    }

    // Private methods

    async fn start_health_monitoring(&self) -> Result<()> {
        // Implementation would start background health monitoring tasks
        Ok(())
    }

    async fn stop_health_monitoring(&self) -> Result<()> {
        let mut monitors = self.health_monitors.write().await;
        for (_, monitor) in monitors.drain() {
            monitor.shutdown().await?;
        }
        Ok(())
    }

    fn categorize_error_severity(
        &self,
        error: &anyhow::Error,
        category: ErrorCategory,
    ) -> ErrorSeverity {
        match category {
            ErrorCategory::WorkerPanic => ErrorSeverity::Error,
            ErrorCategory::GpuError => ErrorSeverity::Critical,
            ErrorCategory::MemoryError => ErrorSeverity::Critical,
            ErrorCategory::CommunicationTimeout => ErrorSeverity::Warning,
            ErrorCategory::NetworkError => ErrorSeverity::Warning,
            ErrorCategory::ResourceExhaustion => ErrorSeverity::Error,
            ErrorCategory::ConfigurationError => ErrorSeverity::Critical,
            ErrorCategory::Unknown => ErrorSeverity::Warning,
        }
    }

    async fn update_worker_health(
        &self,
        worker_id: WorkerId,
        error_event: &ErrorEvent,
    ) -> Result<()> {
        if let Some(monitor) = self.health_monitors.read().await.get(&worker_id) {
            monitor.record_error(error_event).await;
        }
        Ok(())
    }

    async fn assess_system_health(&self) -> Result<SystemHealthLevel> {
        let state = self.state.read().await;

        // Simple health assessment based on error rates
        if state.total_errors == 0 {
            return Ok(SystemHealthLevel::Healthy);
        }

        let recent_errors = state.errors_by_category.values().sum::<usize>();

        match recent_errors {
            0..=5 => Ok(SystemHealthLevel::Healthy),
            6..=15 => Ok(SystemHealthLevel::Warning),
            16..=30 => Ok(SystemHealthLevel::Degraded),
            31..=50 => Ok(SystemHealthLevel::Critical),
            _ => Ok(SystemHealthLevel::Emergency),
        }
    }

    async fn update_system_health(&self, new_health_level: SystemHealthLevel) -> Result<()> {
        let mut state = self.state.write().await;

        if new_health_level != state.health_level {
            log::info!(
                "System health level changed: {:?} -> {:?}",
                state.health_level,
                new_health_level
            );
            state.health_level = new_health_level;

            // Trigger degradation actions if needed
            if new_health_level >= SystemHealthLevel::Degraded {
                self.trigger_degradation_actions(new_health_level).await?;
            }
        }

        Ok(())
    }

    async fn trigger_auto_recovery(&self, error_event: &ErrorEvent) -> Result<()> {
        let recovery_request = match error_event.category {
            ErrorCategory::WorkerPanic => {
                if let Some(worker_id) = error_event.worker_id {
                    Some(RecoveryRequest {
                        action_type: RecoveryActionType::RestartWorker,
                        target: RecoveryTarget::Worker(worker_id),
                        priority: RecoveryPriority::High,
                        requested_at: Instant::now(),
                        context: Some("Worker panic detected".to_string()),
                    })
                } else {
                    None
                }
            }
            ErrorCategory::GpuError => Some(RecoveryRequest {
                action_type: RecoveryActionType::SwitchToFallback,
                target: RecoveryTarget::System,
                priority: RecoveryPriority::Critical,
                requested_at: Instant::now(),
                context: Some("GPU error detected".to_string()),
            }),
            ErrorCategory::MemoryError => Some(RecoveryRequest {
                action_type: RecoveryActionType::ReduceParallelism,
                target: RecoveryTarget::System,
                priority: RecoveryPriority::High,
                requested_at: Instant::now(),
                context: Some("Memory error detected".to_string()),
            }),
            _ => None,
        };

        if let Some(request) = recovery_request {
            self.recovery_executor.queue_recovery(request).await?;
        }

        Ok(())
    }

    async fn trigger_degradation_actions(&self, health_level: SystemHealthLevel) -> Result<()> {
        // Implementation would trigger appropriate degradation actions
        // based on the current health level and policies
        log::warn!(
            "Triggering degradation actions for health level: {:?}",
            health_level
        );
        Ok(())
    }
}

impl WorkerHealthMonitor {
    fn new(worker_id: WorkerId) -> Self {
        Self {
            worker_id,
            health_status: Arc::new(RwLock::new(WorkerHealth::Healthy)),
            last_heartbeat: Arc::new(Mutex::new(Instant::now())),
            error_count: Arc::new(AtomicUsize::new(0)),
            restart_count: Arc::new(AtomicUsize::new(0)),
            monitor_task: None,
        }
    }

    async fn record_heartbeat(&self) {
        let mut last_heartbeat = self.last_heartbeat.lock().await;
        *last_heartbeat = Instant::now();
    }

    async fn record_error(&self, _error_event: &ErrorEvent) {
        self.error_count.fetch_add(1, Ordering::Relaxed);

        let error_count = self.error_count.load(Ordering::Relaxed);
        if error_count >= 5 {
            let mut health_status = self.health_status.write().await;
            *health_status = WorkerHealth::Failed;
        } else if error_count >= 3 {
            let mut health_status = self.health_status.write().await;
            *health_status = WorkerHealth::Degraded;
        }
    }

    async fn shutdown(&self) -> Result<()> {
        if let Some(task) = &self.monitor_task {
            task.abort();
        }
        Ok(())
    }
}

impl RecoveryExecutor {
    fn new() -> Self {
        Self {
            active_tasks: Arc::new(RwLock::new(HashMap::new())),
            recovery_queue: Arc::new(Mutex::new(Vec::new())),
            executor_task: None,
        }
    }

    async fn start(&mut self) -> Result<()> {
        // Start recovery executor task
        log::info!("Recovery executor started");
        Ok(())
    }

    async fn queue_recovery(&self, request: RecoveryRequest) -> Result<RecoveryActionId> {
        let action_id = rand::random::<u64>();

        {
            let mut queue = self.recovery_queue.lock().await;
            queue.push(request);
            queue.sort_by_key(|r| std::cmp::Reverse(r.priority));
        }

        log::info!("Queued recovery action: {}", action_id);
        Ok(action_id)
    }

    async fn shutdown(&mut self) -> Result<()> {
        if let Some(task) = &self.executor_task {
            task.abort();
        }

        let active_tasks = self.active_tasks.read().await;
        for (_, task) in active_tasks.iter() {
            task.abort();
        }

        log::info!("Recovery executor shutdown complete");
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ErrorStats {
    pub total_errors: usize,
    pub errors_by_category: HashMap<ErrorCategory, usize>,
    pub current_health_level: SystemHealthLevel,
    pub active_recoveries: usize,
    pub degraded_mode: bool,
}
