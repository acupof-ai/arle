//! NUMA-aware HTTP prompt preprocessing workers.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Result, anyhow};
use tokio::sync::{OwnedSemaphorePermit, oneshot};

use crate::runtime_topology::{NumaWorkerGroup, bind_current_thread_to_cpus};
use crate::tokenizer::Tokenizer;

pub(super) struct PreprocessPermitGuard {
    permit: Option<OwnedSemaphorePermit>,
    permits: Arc<tokio::sync::Semaphore>,
    capacity: usize,
    metrics: crate::metrics::ServerMetrics,
    wait_us: u64,
    tokenize_started_at: std::time::Instant,
    tokenize_us: u64,
}

impl PreprocessPermitGuard {
    pub(super) fn new(
        permits: Arc<tokio::sync::Semaphore>,
        capacity: usize,
        metrics: crate::metrics::ServerMetrics,
        permit: OwnedSemaphorePermit,
        wait_us: u64,
        tokenize_started_at: std::time::Instant,
    ) -> Self {
        Self {
            permit: Some(permit),
            permits,
            capacity,
            metrics,
            wait_us,
            tokenize_started_at,
            tokenize_us: 0,
        }
    }

    fn record_tokenize_elapsed(&mut self) {
        self.tokenize_us = self.tokenize_started_at.elapsed().as_micros() as u64;
    }

    fn active_depth(&self) -> u64 {
        self.capacity
            .saturating_sub(self.permits.available_permits()) as u64
    }
}

impl Drop for PreprocessPermitGuard {
    fn drop(&mut self) {
        drop(self.permit.take());
        self.metrics
            .set_preprocess_stage(self.active_depth(), self.wait_us, self.tokenize_us);
    }
}

pub(super) struct PreprocessOutput {
    pub(super) prompt: String,
    pub(super) prompt_tokens: Vec<u32>,
    pub(super) numa_node: Option<i32>,
}

pub(super) struct PreprocessWorkerPool {
    groups: Vec<PreprocessGroup>,
    cursor: AtomicUsize,
    capacity: usize,
}

struct PreprocessGroup {
    numa_node: Option<i32>,
    workers: Vec<std::sync::mpsc::Sender<TokenizeJob>>,
    cursor: AtomicUsize,
}

struct TokenizeJob {
    prompt: String,
    permit_guard: PreprocessPermitGuard,
    reply_tx: oneshot::Sender<std::result::Result<PreprocessOutput, String>>,
}

impl PreprocessWorkerPool {
    pub(super) fn spawn(tokenizer: Tokenizer, groups: Vec<NumaWorkerGroup>) -> Self {
        let mut preprocess_groups = Vec::new();
        let mut capacity = 0usize;
        for (group_idx, group) in groups.into_iter().enumerate() {
            let worker_count = group.worker_count.max(1);
            let mut workers = Vec::with_capacity(worker_count);
            for worker_idx in 0..worker_count {
                let (tx, rx) = std::sync::mpsc::channel::<TokenizeJob>();
                let worker_tokenizer = tokenizer.clone();
                let cpus = group.cpus.clone();
                let numa_node = group.numa_node;
                let thread_name = format!("infer-tokenize-n{:?}-{worker_idx}", numa_node);
                std::thread::Builder::new()
                    .name(thread_name)
                    .spawn(move || {
                        let affinity = bind_current_thread_to_cpus(
                            &cpus,
                            format!("tokenizer-numa-{numa_node:?}-{worker_idx}"),
                        );
                        let reported_numa_node = affinity.applied.then_some(numa_node).flatten();
                        log::info!(
                            "Tokenizer worker ready: group={} worker={} numa={:?} cpus={} affinity_applied={} reason={}",
                            group_idx,
                            worker_idx,
                            numa_node,
                            cpus.len(),
                            affinity.applied,
                            affinity.reason,
                        );
                        while let Ok(job) = rx.recv() {
                            let mut permit_guard = job.permit_guard;
                            let result = worker_tokenizer
                                .encode(&job.prompt)
                                .map(|prompt_tokens| PreprocessOutput {
                                    prompt: job.prompt,
                                    prompt_tokens,
                                    numa_node: reported_numa_node,
                                })
                                .map_err(|err| format!("{err:#}"));
                            permit_guard.record_tokenize_elapsed();
                            let _ = job.reply_tx.send(result);
                        }
                    })
                    .expect("tokenizer worker thread");
                workers.push(tx);
            }
            capacity += worker_count;
            preprocess_groups.push(PreprocessGroup {
                numa_node: group.numa_node,
                workers,
                cursor: AtomicUsize::new(0),
            });
        }

        Self {
            groups: preprocess_groups,
            cursor: AtomicUsize::new(0),
            capacity,
        }
    }

    pub(super) fn group_count(&self) -> usize {
        self.groups.len()
    }

    pub(super) fn capacity(&self) -> usize {
        self.capacity.max(1)
    }

    pub(super) async fn encode(
        &self,
        prompt: String,
        permit_guard: PreprocessPermitGuard,
    ) -> Result<PreprocessOutput> {
        let group = self.next_group()?;
        let worker_idx = group.cursor.fetch_add(1, Ordering::Relaxed) % group.workers.len();
        let (reply_tx, reply_rx) = oneshot::channel();
        group.workers[worker_idx]
            .send(TokenizeJob {
                prompt,
                permit_guard,
                reply_tx,
            })
            .map_err(|err| {
                anyhow!(
                    "tokenizer worker queue closed for NUMA group {:?}: {err}",
                    group.numa_node
                )
            })?;
        reply_rx
            .await
            .map_err(|err| anyhow!("tokenizer worker dropped response: {err}"))?
            .map_err(|err| anyhow!("{err}"))
    }

    fn next_group(&self) -> Result<&PreprocessGroup> {
        if self.groups.is_empty() {
            return Err(anyhow!("tokenizer worker pool has no groups"));
        }
        let group_idx = self.cursor.fetch_add(1, Ordering::Relaxed) % self.groups.len();
        Ok(&self.groups[group_idx])
    }
}

impl std::fmt::Debug for PreprocessWorkerPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreprocessWorkerPool")
            .field("groups", &self.groups.len())
            .field("capacity", &self.capacity)
            .finish_non_exhaustive()
    }
}
