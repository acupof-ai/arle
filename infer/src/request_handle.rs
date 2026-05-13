use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use std::fmt;

use crate::scheduler::{IncomingRequest, SchedulerHandle};

/// Error returned when a request cannot be submitted to a runtime handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubmitError;

impl fmt::Display for SubmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "request submission failed")
    }
}

impl std::error::Error for SubmitError {}

/// DFlash runtime status exposed through the `/v1/models` endpoint.
///
/// `draft_model` and `speculative_tokens` are the DFlash init-time constants
/// (one process → one draft pair). `acceptance_rate` is the rolling rate read
/// from `ServerMetrics::dflash_acceptance_rate()` at response time — it is
/// `None` until at least one speculative block has executed.
#[derive(Clone, Debug, PartialEq)]
pub struct DflashStatus {
    pub draft_model: String,
    pub speculative_tokens: usize,
}

/// Unified request-submission interface used by the HTTP layer.
pub trait RequestHandle: Send + Sync {
    fn submit(&self, req: IncomingRequest) -> Result<(), SubmitError>;
    fn model_id(&self) -> &str;
    fn tokenizer_clone(&self) -> Option<crate::tokenizer::Tokenizer> {
        None
    }

    /// DFlash init-time metadata, if speculative decode is active for this
    /// runtime. Default `None` — CUDA and non-DFlash Metal paths return it
    /// unchanged. The Metal scheduler wrappers override this when a draft
    /// model was successfully loaded.
    fn dflash_status(&self) -> Option<DflashStatus> {
        None
    }

    /// Borrow the rolling `ServerMetrics` instance that the scheduler
    /// thread is writing into, if the handle owns one. Used by
    /// `InferenceEngine::telemetry()` to project the unified
    /// `EngineTelemetry` snapshot. Default `None` for handles that do
    /// not run through a scheduler thread (mocks, tests).
    fn server_metrics(&self) -> Option<&crate::metrics::ServerMetrics> {
        None
    }
}

impl RequestHandle for SchedulerHandle {
    fn submit(&self, req: IncomingRequest) -> Result<(), SubmitError> {
        SchedulerHandle::submit(self, req).map_err(|_| SubmitError)
    }

    fn model_id(&self) -> &str {
        SchedulerHandle::model_id(self)
    }

    fn tokenizer_clone(&self) -> Option<crate::tokenizer::Tokenizer> {
        SchedulerHandle::tokenizer_clone(self)
    }

    fn server_metrics(&self) -> Option<&crate::metrics::ServerMetrics> {
        SchedulerHandle::server_metrics(self)
    }
}

#[derive(Clone)]
pub struct NumaSchedulerWorker {
    pub handle: SchedulerHandle,
    pub placement: crate::runtime_topology::WorkerPlacement,
}

pub struct NumaSchedulerRouter {
    topology: crate::runtime_topology::RuntimeTopology,
    workers: Vec<NumaSchedulerWorker>,
    model_id: Arc<str>,
    tokenizer: Option<crate::tokenizer::Tokenizer>,
    metrics: crate::metrics::ServerMetrics,
    session_routes: Mutex<HashMap<crate::types::SessionId, usize>>,
    rebalance_threshold: usize,
}

impl NumaSchedulerRouter {
    pub fn single(
        handle: SchedulerHandle,
        topology: crate::runtime_topology::RuntimeTopology,
        placement: crate::runtime_topology::WorkerPlacement,
        metrics: crate::metrics::ServerMetrics,
    ) -> Self {
        Self::new(
            topology,
            vec![NumaSchedulerWorker { handle, placement }],
            metrics,
        )
    }

    pub fn new(
        topology: crate::runtime_topology::RuntimeTopology,
        workers: Vec<NumaSchedulerWorker>,
        metrics: crate::metrics::ServerMetrics,
    ) -> Self {
        assert!(
            !workers.is_empty(),
            "NUMA scheduler router requires at least one worker"
        );
        let model_id = Arc::from(workers[0].handle.model_id());
        let tokenizer = workers[0].handle.tokenizer_clone();
        Self {
            topology,
            workers,
            model_id,
            tokenizer,
            metrics,
            session_routes: Mutex::new(HashMap::new()),
            rebalance_threshold: 2,
        }
    }

    fn ranked_workers(&self, req: &IncomingRequest) -> Vec<usize> {
        if self.workers.len() == 1 {
            return vec![0];
        }

        let min_waiting = self
            .workers
            .iter()
            .map(|worker| worker.handle.waiting_count())
            .min()
            .unwrap_or(0);
        let mut ranked = self.ranked_workers_by_score(req, min_waiting);
        if let Some(session_id) = req.session_id.as_ref()
            && let Some(&sticky_idx) = self
                .session_routes
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(session_id)
            && let Some(worker) = self.workers.get(sticky_idx)
            && worker.handle.waiting_count() <= min_waiting + self.rebalance_threshold
        {
            ranked.retain(|idx| *idx != sticky_idx);
            ranked.insert(0, sticky_idx);
        }
        ranked
    }

    fn ranked_workers_by_score(&self, req: &IncomingRequest, min_waiting: usize) -> Vec<usize> {
        let mut ranked = self
            .workers
            .iter()
            .enumerate()
            .map(|(idx, worker)| {
                let cost = self
                    .topology
                    .route_cost_from_numa(&worker.placement, req.ingress_numa_node);
                let waiting = worker.handle.waiting_count();
                (route_score(cost, waiting, min_waiting), waiting, idx)
            })
            .collect::<Vec<_>>();
        ranked.sort_unstable();
        ranked
            .into_iter()
            .map(|(_, _, idx)| idx)
            .collect::<Vec<_>>()
    }

    fn record_worker_selection(
        &self,
        selected: usize,
        ingress_numa_node: Option<i32>,
        session_id: Option<&crate::types::SessionId>,
    ) {
        if let Some(session_id) = session_id {
            let mut routes = self
                .session_routes
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let previous = routes.insert(session_id.clone(), selected);
            if previous.is_some_and(|idx| idx != selected) {
                self.metrics.record_numa_migration();
            }
            if previous != Some(selected) {
                self.metrics.record_numa_rebalance();
            }
        } else if self.workers.len() > 1 {
            self.metrics.record_numa_rebalance();
        }
        let worker = &self.workers[selected];
        let cost = self
            .topology
            .route_cost_from_numa(&worker.placement, ingress_numa_node);
        self.metrics.record_numa_route(
            cost,
            locality(worker.placement.numa_node, ingress_numa_node),
        );
    }
}

fn locality(worker_numa: Option<i32>, ingress_numa: Option<i32>) -> Option<bool> {
    match (worker_numa, ingress_numa) {
        (Some(worker), Some(ingress)) => Some(worker == ingress),
        _ => None,
    }
}

fn route_score(numa_cost: u32, waiting: usize, min_waiting: usize) -> u32 {
    let load_delta = waiting.saturating_sub(min_waiting) as u32;
    numa_cost.saturating_add(load_delta.saturating_mul(50))
}

impl RequestHandle for NumaSchedulerRouter {
    fn submit(&self, req: IncomingRequest) -> Result<(), SubmitError> {
        let ranked = self.ranked_workers(&req);
        let ingress_numa_node = req.ingress_numa_node;
        let session_id = req.session_id.clone();
        let mut req = req;
        for selected in ranked {
            let Ok(permit) = self.workers[selected].handle.reserve_submission() else {
                continue;
            };
            match permit.submit(req) {
                Ok(()) => {
                    self.record_worker_selection(selected, ingress_numa_node, session_id.as_ref());
                    return Ok(());
                }
                Err(failure) => {
                    req = failure.into_request();
                }
            }
        }
        Err(SubmitError)
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn tokenizer_clone(&self) -> Option<crate::tokenizer::Tokenizer> {
        self.tokenizer.clone()
    }

    fn server_metrics(&self) -> Option<&crate::metrics::ServerMetrics> {
        Some(&self.metrics)
    }
}

impl<T> RequestHandle for Arc<T>
where
    T: RequestHandle + ?Sized,
{
    fn submit(&self, req: IncomingRequest) -> Result<(), SubmitError> {
        (**self).submit(req)
    }

    fn model_id(&self) -> &str {
        (**self).model_id()
    }

    fn tokenizer_clone(&self) -> Option<crate::tokenizer::Tokenizer> {
        (**self).tokenizer_clone()
    }

    fn dflash_status(&self) -> Option<DflashStatus> {
        (**self).dflash_status()
    }

    fn server_metrics(&self) -> Option<&crate::metrics::ServerMetrics> {
        (**self).server_metrics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_topology::{NumaNodeTopology, RuntimeTopology, WorkerPlacement};
    use crate::sampler::SamplingParams;
    use crate::server_engine::CompletionStreamDelta;
    use crate::types::SessionId;
    use tokio::sync::mpsc;

    fn test_request(session: Option<&str>, ingress_numa_node: Option<i32>) -> IncomingRequest {
        let (delta_tx, _rx) = mpsc::unbounded_channel::<CompletionStreamDelta>();
        IncomingRequest {
            prompt: "hello".to_string(),
            prompt_tokens: Some(vec![1]),
            max_tokens: 1,
            sampling: SamplingParams::default(),
            stop: None,
            speculative: None,
            priority: crate::scheduler::RequestPriority::Normal,
            session_id: session.map(SessionId::from),
            ingress_numa_node,
            delta_tx,
            trace_context: None,
        }
    }

    fn test_topology() -> RuntimeTopology {
        RuntimeTopology {
            numa_nodes: vec![
                NumaNodeTopology {
                    node: 0,
                    cpus: vec![0, 1],
                },
                NumaNodeTopology {
                    node: 1,
                    cpus: vec![2, 3],
                },
            ],
            gpus: Vec::new(),
            nics: Vec::new(),
            fallback_cpus: vec![0, 1, 2, 3],
        }
    }

    fn placement(worker_id: usize, numa_node: i32) -> WorkerPlacement {
        WorkerPlacement {
            worker_id,
            gpu_ordinal: worker_id,
            numa_node: Some(numa_node),
            cpus: if numa_node == 0 {
                vec![0, 1]
            } else {
                vec![2, 3]
            },
            nics: Vec::new(),
            route_cost: 0,
        }
    }

    #[test]
    fn numa_router_prefers_local_worker() {
        let (tx0, mut rx0) = mpsc::unbounded_channel();
        let (tx1, mut rx1) = mpsc::unbounded_channel();
        let worker0 = SchedulerHandle::from_parts(tx0, "model");
        let worker1 = SchedulerHandle::from_parts(tx1, "model");
        let router = NumaSchedulerRouter::new(
            test_topology(),
            vec![
                NumaSchedulerWorker {
                    handle: worker0,
                    placement: placement(0, 0),
                },
                NumaSchedulerWorker {
                    handle: worker1,
                    placement: placement(1, 1),
                },
            ],
            crate::metrics::ServerMetrics::new("model"),
        );

        router.submit(test_request(None, Some(1))).unwrap();
        assert!(rx0.try_recv().is_err());
        assert_eq!(rx1.try_recv().unwrap().ingress_numa_node, Some(1));
    }

    #[test]
    fn numa_router_migrates_sticky_session_when_worker_is_overloaded() {
        let (tx0, mut rx0) = mpsc::unbounded_channel();
        let (tx1, mut rx1) = mpsc::unbounded_channel();
        let worker0 = SchedulerHandle::from_parts(tx0, "model");
        let worker1 = SchedulerHandle::from_parts(tx1, "model");
        let metrics = crate::metrics::ServerMetrics::new("model");
        let router = NumaSchedulerRouter::new(
            test_topology(),
            vec![
                NumaSchedulerWorker {
                    handle: worker0.clone(),
                    placement: placement(0, 0),
                },
                NumaSchedulerWorker {
                    handle: worker1,
                    placement: placement(1, 1),
                },
            ],
            metrics.clone(),
        );

        router.submit(test_request(Some("s1"), Some(0))).unwrap();
        assert_eq!(rx0.try_recv().unwrap().session_id.unwrap().as_str(), "s1");
        for _ in 0..4 {
            worker0.submit(test_request(None, Some(0))).unwrap();
        }

        router.submit(test_request(Some("s1"), Some(0))).unwrap();
        assert_eq!(rx1.try_recv().unwrap().session_id.unwrap().as_str(), "s1");
        let snapshot = metrics.runtime_topology_snapshot();
        assert_eq!(snapshot.numa_migration_total, 1);
        assert_eq!(snapshot.numa_rebalance_total, 2);
    }

    #[test]
    fn numa_router_retries_next_worker_when_first_queue_is_full() {
        let (tx0, mut rx0) = mpsc::unbounded_channel();
        let (tx1, mut rx1) = mpsc::unbounded_channel();
        let worker0 = SchedulerHandle::with_max_waiting(tx0, "model", 1);
        let worker1 = SchedulerHandle::with_max_waiting(tx1, "model", 1);
        worker0.submit(test_request(None, Some(0))).unwrap();
        assert_eq!(worker0.waiting_count(), 1);

        let router = NumaSchedulerRouter::new(
            test_topology(),
            vec![
                NumaSchedulerWorker {
                    handle: worker0,
                    placement: placement(0, 0),
                },
                NumaSchedulerWorker {
                    handle: worker1,
                    placement: placement(1, 1),
                },
            ],
            crate::metrics::ServerMetrics::new("model"),
        );

        router.submit(test_request(None, Some(0))).unwrap();
        assert_eq!(rx0.try_recv().unwrap().ingress_numa_node, Some(0));
        assert_eq!(rx1.try_recv().unwrap().ingress_numa_node, Some(0));
    }
}
