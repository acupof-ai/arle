use super::super::core::PrefetchTicketState;
use super::super::{CompletionStreamDelta, FinishReason, TokenUsage};
use crate::kv_tier::ReadmissionSource;

#[derive(Clone)]
pub(super) struct FetchWaiter {
    pub(super) slot_idx: usize,
    pub(super) request_id: u64,
    pub(super) prompt_tokens: Vec<u32>,
    pub(super) session_id: Option<crate::types::SessionId>,
    pub(super) staged_prefix: crate::kv_tier::ReadmissionPlan,
}

#[derive(Clone)]
pub(super) struct PrefixAdmissionPlan {
    pub(super) radix_blocks: Vec<crate::prefix_cache::BlockId>,
    pub(super) lookup: crate::kv_tier::LookupOutcome,
    pub(super) reusable: Option<(usize, usize, usize)>,
    pub(super) direct_gpu_attach: bool,
    pub(super) attached_prefix_blocks: Vec<crate::prefix_cache::BlockId>,
    pub(super) staged_prefix_plan: Option<crate::kv_tier::ReadmissionPlan>,
}

pub(super) struct QueuedAdmissionCandidate {
    pub(super) incoming: super::super::IncomingRequest,
    pub(super) prompt_tokens: Vec<u32>,
    pub(super) plan: PrefixAdmissionPlan,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct WaitingRequestHint {
    pub(super) immediate_reuse_tokens: usize,
    pub(super) total_reuse_tokens: usize,
}

impl WaitingRequestHint {
    pub(super) fn from_plan(plan: &PrefixAdmissionPlan, reusable_prefix_len: usize) -> Self {
        let immediate_reuse_tokens = if plan.direct_gpu_attach {
            plan.lookup.matched_len
        } else {
            reusable_prefix_len
        };
        let total_reuse_tokens = if immediate_reuse_tokens > 0 {
            immediate_reuse_tokens
        } else if let Some(staged_prefix_plan) = plan.staged_prefix_plan.as_ref() {
            staged_prefix_plan.matched_len
        } else if !plan.lookup.recompute_advised {
            plan.lookup.matched_len
        } else {
            0
        };
        Self {
            immediate_reuse_tokens,
            total_reuse_tokens,
        }
    }
}

pub(super) struct DeferredWaitingRequest {
    pub(super) incoming: super::super::IncomingRequest,
    pub(super) prompt_tokens: Vec<u32>,
    pub(super) hint: WaitingRequestHint,
}

pub(super) fn staged_prefix_direct_host_blocks(
    plan: &crate::kv_tier::ReadmissionPlan,
) -> Option<Vec<crate::kv_tier::FetchedBlock>> {
    let mut fetched_blocks = Vec::new();
    let mut host_blocks = 0usize;
    for block in &plan.blocks {
        match block.source.as_ref() {
            None => {}
            Some(ReadmissionSource::HostPinned { region }) => {
                host_blocks += 1;
                fetched_blocks.push(crate::kv_tier::FetchedBlock {
                    block_id: block.block_id,
                    host_region: *region,
                    byte_len: region.len,
                    release_after_promote: false,
                });
            }
            Some(ReadmissionSource::Disk { .. } | ReadmissionSource::Remote { .. }) => {
                return None;
            }
        }
    }
    (host_blocks > 0).then_some(fetched_blocks)
}

pub(super) fn staged_prefix_prefetch_state(
    plan: &crate::kv_tier::ReadmissionPlan,
) -> Option<PrefetchTicketState> {
    let (host_blocks, disk_blocks, remote_blocks) = plan.source_counts();
    (disk_blocks + remote_blocks > 0).then_some(PrefetchTicketState {
        host_blocks,
        disk_blocks,
        remote_blocks,
    })
}

pub(super) fn best_reusable_slot_for_radix_hit(
    matched_blocks: &[crate::prefix_cache::BlockId],
    free_slots: &[usize],
    block_owner_slots: &std::collections::HashMap<crate::prefix_cache::BlockId, usize>,
    slot_materialized_prompt_lens: &[usize],
    block_size: usize,
) -> Option<(usize, usize, usize)> {
    for (idx, &bid) in matched_blocks.iter().enumerate().rev() {
        let Some(&slot_idx) = block_owner_slots.get(&bid) else {
            continue;
        };
        if !free_slots.contains(&slot_idx) {
            continue;
        }
        let reusable_prefix_len = (idx + 1) * block_size;
        let cached_prompt_len = slot_materialized_prompt_lens
            .get(slot_idx)
            .copied()
            .unwrap_or_default();
        if cached_prompt_len >= reusable_prefix_len && reusable_prefix_len > 0 {
            return Some((slot_idx, reusable_prefix_len, cached_prompt_len));
        }
    }
    None
}

pub(super) fn lookup_blocks_ready_on_gpu(blocks: &[crate::kv_tier::LookupBlock]) -> bool {
    blocks
        .iter()
        .filter(|block| !matches!(block.hit_kind, crate::kv_tier::HitKind::Miss))
        .all(|block| matches!(block.hit_kind, crate::kv_tier::HitKind::ReadyOnGpu))
}

pub(super) fn matched_sealed_lookup_blocks(blocks: &[crate::kv_tier::LookupBlock]) -> usize {
    blocks
        .iter()
        .filter(|block| !matches!(block.hit_kind, crate::kv_tier::HitKind::Miss))
        .count()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::scheduler::cuda) enum WaitingInsertBias {
    BeforeEqual,
    AfterEqual,
}

pub(super) fn waiting_request_precedes(
    incoming_priority: super::super::RequestPriority,
    incoming_hint: WaitingRequestHint,
    queued_priority: super::super::RequestPriority,
    queued_hint: WaitingRequestHint,
    bias: WaitingInsertBias,
) -> bool {
    match incoming_priority.cmp(&queued_priority) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => match (
            incoming_hint.immediate_reuse_tokens,
            incoming_hint.total_reuse_tokens,
        )
            .cmp(&(
                queued_hint.immediate_reuse_tokens,
                queued_hint.total_reuse_tokens,
            )) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Less => false,
            std::cmp::Ordering::Equal => matches!(bias, WaitingInsertBias::BeforeEqual),
        },
    }
}

pub(super) fn waiting_insert_position<T>(
    waiting: &std::collections::VecDeque<T>,
    incoming_priority: super::super::RequestPriority,
    incoming_hint: WaitingRequestHint,
    bias: WaitingInsertBias,
    queued_key: impl Fn(&T) -> (super::super::RequestPriority, WaitingRequestHint),
) -> usize {
    waiting
        .iter()
        .position(|queued| {
            let (queued_priority, queued_hint) = queued_key(queued);
            waiting_request_precedes(
                incoming_priority,
                incoming_hint,
                queued_priority,
                queued_hint,
                bias,
            )
        })
        .unwrap_or(waiting.len())
}

pub(super) fn insert_waiting_request_by_priority(
    waiting: &mut std::collections::VecDeque<super::super::IncomingRequest>,
    incoming: super::super::IncomingRequest,
    bias: WaitingInsertBias,
) {
    let insert_at = waiting_insert_position(
        waiting,
        incoming.priority,
        WaitingRequestHint::default(),
        bias,
        |queued| (queued.priority, WaitingRequestHint::default()),
    );
    waiting.insert(insert_at, incoming);
}

pub(super) fn insert_deferred_waiting_request(
    waiting: &mut std::collections::VecDeque<DeferredWaitingRequest>,
    incoming: DeferredWaitingRequest,
    bias: WaitingInsertBias,
) {
    let insert_at = waiting_insert_position(
        waiting,
        incoming.incoming.priority,
        incoming.hint,
        bias,
        |queued| (queued.incoming.priority, queued.hint),
    );
    waiting.insert(insert_at, incoming);
}

pub(super) fn finish_rejected_request(
    delta_tx: &tokio::sync::mpsc::UnboundedSender<CompletionStreamDelta>,
    reason: FinishReason,
    prompt_tokens: usize,
) {
    let _ = delta_tx.send(CompletionStreamDelta {
        text_delta: String::new(),
        finish_reason: Some(reason),
        usage: Some(TokenUsage {
            prompt_tokens,
            completion_tokens: 0,
            total_tokens: prompt_tokens,
        }),
        logprob: None,
    });
}
