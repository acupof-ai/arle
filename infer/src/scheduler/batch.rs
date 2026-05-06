use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use crate::block_manager::{BlockId, BlockManager};
use crate::events::{EngineEvent, EventSink, NoopEventSink};
use crate::scheduler::policy::{ChunkingPolicy, DecodeAwareChunking, SchedulerSignals};
use crate::scheduler::{LogicalDecodeRow, LogicalPrefillRow, LogicalServePlan};
use crate::types::{InferenceMode, RequestEventKind, RequestId};

use super::RequestPriority;

/// A request waiting to begin (or resume) prefill.
pub struct PendingRequest {
    pub req_id: RequestId,
    /// Tokenized prompt. Caller must tokenize before calling `add_request`.
    pub prompt_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub priority: RequestPriority,
    /// Tokens already prefilled in previous chunks (chunked prefill progress).
    pub prefill_progress: usize,
    /// KV blocks already allocated for earlier prefill chunks.
    pub allocated_blocks: Vec<BlockId>,
}

/// A request that finished prefill and is now in the decode (generation) phase.
pub struct RunningRequest {
    pub req_id: RequestId,
    /// Kept for recompute-mode preemption (re-queue without re-tokenizing).
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: usize,
    pub max_tokens: usize,
    /// KV blocks currently allocated on GPU.
    pub blocks: Vec<BlockId>,
    /// Total KV tokens committed (prompt_len + generated_tokens).
    pub kv_tokens: usize,
}

impl RunningRequest {
    pub fn prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }
}

/// Configuration for `BatchScheduler`.
#[derive(Clone, Debug)]
pub struct BatchSchedulerConfig {
    /// Maximum tokens processed per step (decode + prefill combined).
    pub max_tokens_per_step: usize,
    /// Maximum tokens in a single prefill chunk.
    pub prefill_chunk_size: usize,
    /// Policy for adapting prefill chunk size under decode pressure.
    pub chunking_policy: DecodeAwareChunking,
}

impl Default for BatchSchedulerConfig {
    fn default() -> Self {
        Self {
            max_tokens_per_step: 2048,
            prefill_chunk_size: 512,
            chunking_policy: DecodeAwareChunking {
                decode_active_chunk: 64,
                idle_chunk: 512,
            },
        }
    }
}

/// Continuous batching scheduler.
///
/// This is a pure CPU accounting layer. It decides *what* to run each step
/// (decode + optional prefill chunk) using block-level KV accounting, but does
/// not touch GPU memory. The caller is responsible for:
/// - Tokenizing prompts before calling `add_request`.
/// - Running the model forward pass with the returned `LogicalServePlan`.
/// - Calling `advance_decode` after each successful decode step.
/// - Calling `finish_request` when a request reaches EOS or max_tokens.
pub struct BatchScheduler {
    config: BatchSchedulerConfig,
    /// Requests waiting to be prefilled, ordered by arrival (FCFS).
    waiting: VecDeque<PendingRequest>,
    /// Requests currently in the decode phase.
    pub(crate) running: HashMap<RequestId, RunningRequest>,
    block_manager: BlockManager,
    next_req_id: u64,
    event_sink: Arc<dyn EventSink>,
}

impl BatchScheduler {
    pub fn new(config: BatchSchedulerConfig, block_manager: BlockManager) -> Self {
        Self::with_event_sink(config, block_manager, Arc::new(NoopEventSink))
    }

    pub fn with_event_sink(
        config: BatchSchedulerConfig,
        block_manager: BlockManager,
        event_sink: Arc<dyn EventSink>,
    ) -> Self {
        Self {
            config,
            waiting: VecDeque::new(),
            running: HashMap::new(),
            block_manager,
            next_req_id: 0,
            event_sink,
        }
    }

    fn emit_event(
        &self,
        request_id: RequestId,
        kind: RequestEventKind,
        mode: Option<InferenceMode>,
    ) {
        self.event_sink.emit(&EngineEvent {
            request_id,
            kind,
            mode,
        });
    }

    /// Submit a new request. Returns its assigned `req_id`.
    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        priority: RequestPriority,
    ) -> RequestId {
        let req_id = RequestId(self.next_req_id);
        self.next_req_id += 1;
        self.waiting.push_back(PendingRequest {
            req_id,
            prompt_tokens,
            max_tokens,
            priority,
            prefill_progress: 0,
            allocated_blocks: Vec::new(),
        });
        self.emit_event(req_id, RequestEventKind::Enqueued, None);
        req_id
    }

    /// Call after the model generates one more token for a decode request.
    /// Allocates additional KV blocks if the sequence has grown into a new block.
    pub fn advance_decode(&mut self, req_id: RequestId) -> bool {
        let Some(req) = self.running.get_mut(&req_id) else {
            return false;
        };
        req.generated_tokens += 1;
        req.kv_tokens += 1;
        let blocks_needed = self.block_manager.blocks_for_tokens(req.kv_tokens);
        if blocks_needed > req.blocks.len() {
            match self
                .block_manager
                .allocate_gpu(blocks_needed - req.blocks.len())
            {
                Ok(new_blocks) => req.blocks.extend(new_blocks),
                Err(_) => return false,
            }
        }
        self.emit_event(
            req_id,
            RequestEventKind::DecodeStep,
            Some(InferenceMode::Decode),
        );
        true
    }

    /// Mark a request as finished and free its KV blocks.
    pub fn finish_request(&mut self, req_id: RequestId) {
        if let Some(req) = self.running.remove(&req_id) {
            self.block_manager.free(&req.blocks);
            self.emit_event(
                req_id,
                RequestEventKind::Completed,
                Some(InferenceMode::Decode),
            );
        }
        self.waiting.retain(|r| r.req_id != req_id);
    }

    /// Schedule the next forward pass.
    ///
    /// Returns an idle `LogicalServePlan` when there is no work to do.
    pub fn schedule_step(&mut self) -> LogicalServePlan {
        let preempted = self.ensure_decode_memory();

        let has_decode = !self.running.is_empty();
        let has_waiting = !self.waiting.is_empty();

        if !has_decode && !has_waiting {
            return LogicalServePlan::idle();
        }

        let decode_rows = if has_decode {
            self.build_decode_rows()
        } else {
            Vec::new()
        };
        let decode_tokens = self.running.len();
        let budget = self
            .config
            .max_tokens_per_step
            .saturating_sub(decode_tokens);
        let prefill_rows = (!preempted && has_waiting && budget > 0)
            .then(|| self.try_admit_prefill_chunk(budget, decode_rows.len()))
            .flatten()
            .into_iter()
            .collect();

        LogicalServePlan::new(decode_rows, prefill_rows)
    }

    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    pub fn free_gpu_blocks(&self) -> usize {
        self.block_manager.free_gpu_blocks()
    }

    pub fn is_running(&self, req_id: RequestId) -> bool {
        self.running.contains_key(&req_id)
    }

    pub fn is_waiting(&self, req_id: RequestId) -> bool {
        self.waiting.iter().any(|r| r.req_id == req_id)
    }

    /// Preempt running requests (highest req_id first — FCFS reverse / newest
    /// first) until the block manager has enough free blocks for all running
    /// requests to advance by one decode token.
    ///
    /// Returns `true` if any request was preempted (caller skips prefill this step).
    fn ensure_decode_memory(&mut self) -> bool {
        let mut preempted = false;
        loop {
            let new_blocks_needed: usize = self
                .running
                .values()
                .map(|r| {
                    let next_kv = r.kv_tokens + 1;
                    self.block_manager
                        .blocks_for_tokens(next_kv)
                        .saturating_sub(r.blocks.len())
                })
                .sum();

            if new_blocks_needed <= self.block_manager.free_gpu_blocks() {
                break;
            }

            let Some(&preempt_id) = self.running.keys().max() else {
                break;
            };

            let req = self.running.remove(&preempt_id).expect("checked above");
            self.block_manager.free(&req.blocks);
            preempted = true;

            self.emit_event(
                preempt_id,
                RequestEventKind::Evicted,
                Some(InferenceMode::Decode),
            );

            self.waiting.push_front(PendingRequest {
                req_id: preempt_id,
                prompt_tokens: req.prompt_tokens,
                max_tokens: req.max_tokens,
                priority: RequestPriority::Normal,
                prefill_progress: 0,
                allocated_blocks: Vec::new(),
            });
            self.emit_event(preempt_id, RequestEventKind::Requeued, None);
        }
        preempted
    }

    fn build_decode_rows(&self) -> Vec<LogicalDecodeRow> {
        let n = self.running.len();
        let mut rows = Vec::with_capacity(n);

        let mut ids: Vec<RequestId> = self.running.keys().copied().collect();
        ids.sort_unstable();

        for id in ids {
            let req = &self.running[&id];
            let logical_kv_offset = req.prompt_len() + req.generated_tokens.saturating_sub(1);
            rows.push(LogicalDecodeRow::new(
                rows.len(),
                req.req_id,
                0,
                logical_kv_offset,
            ));
        }

        rows
    }

    fn prefill_chunk_budget(&self) -> usize {
        let signals = SchedulerSignals::queue_state(self.waiting.len(), self.running.len());
        self.config
            .chunking_policy
            .next_chunk_size(InferenceMode::Prefill, signals)
            .max(1)
            .min(self.config.prefill_chunk_size)
    }

    /// Attempt to emit one prefill chunk for the head of `waiting`.
    ///
    /// Allocates new KV blocks for the chunk. Returns `None` if the queue is
    /// empty or KV memory is too full to admit even one token.
    fn try_admit_prefill_chunk(
        &mut self,
        token_budget: usize,
        row_index: usize,
    ) -> Option<LogicalPrefillRow> {
        let policy_chunk_budget = self.prefill_chunk_budget();
        let should_drop_empty = self
            .waiting
            .front()
            .is_some_and(|pending| pending.prefill_progress >= pending.prompt_tokens.len());
        if should_drop_empty {
            self.waiting.pop_front();
            return None;
        }

        let (
            chunk,
            req_id,
            prompt_start,
            prompt_end,
            is_last_chunk,
            total_tokens,
            emit_prefill_started,
        ) = {
            let pending = self.waiting.front_mut()?;

            let total_tokens = pending.prompt_tokens.len();
            let prompt_start = pending.prefill_progress;
            let is_first_chunk = prompt_start == 0;
            let remaining = total_tokens.saturating_sub(prompt_start);
            let chunk_tokens = remaining.min(policy_chunk_budget).min(token_budget);
            if chunk_tokens == 0 {
                return None;
            }
            let prompt_end = prompt_start + chunk_tokens;
            let blocks_needed = self.block_manager.blocks_for_tokens(prompt_end);
            let have = pending.allocated_blocks.len();
            let extra_needed = blocks_needed.saturating_sub(have);

            if extra_needed > 0 {
                match self.block_manager.allocate_gpu(extra_needed) {
                    Ok(new_blocks) => pending.allocated_blocks.extend(new_blocks),
                    Err(_) => return None,
                }
            }

            let chunk = pending.prompt_tokens[prompt_start..prompt_end].to_vec();
            let req_id = pending.req_id;
            let is_last_chunk = prompt_end >= total_tokens;

            if !is_last_chunk {
                pending.prefill_progress = prompt_end;
            }

            (
                chunk,
                req_id,
                prompt_start,
                prompt_end,
                is_last_chunk,
                total_tokens,
                is_first_chunk,
            )
        };

        if emit_prefill_started {
            // Lifecycle signal, not a per-chunk notification.
            self.emit_event(
                req_id,
                RequestEventKind::PrefillStarted,
                Some(InferenceMode::Prefill),
            );
        }

        if is_last_chunk {
            let done = self.waiting.pop_front().expect("checked above");
            self.running.insert(
                req_id,
                RunningRequest {
                    req_id,
                    prompt_tokens: done.prompt_tokens,
                    generated_tokens: 0,
                    max_tokens: done.max_tokens,
                    blocks: done.allocated_blocks,
                    kv_tokens: total_tokens,
                },
            );
        }

        Some(LogicalPrefillRow::new(
            row_index,
            req_id,
            chunk,
            prompt_start,
            prompt_end,
            total_tokens,
        ))
    }
}
