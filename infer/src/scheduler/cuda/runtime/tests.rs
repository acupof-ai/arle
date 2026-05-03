//! Tests for the CUDA scheduler runtime.
//!
//! Split out of `runtime.rs` (pure structural refactor — no behavior change).

#[cfg(test)]
mod tests {
    use super::super::helpers::{
        DeferredWaitingRequest, PrefixAdmissionPlan, QueuedAdmissionCandidate, WaitingInsertBias,
        WaitingRequestHint, best_reusable_slot_for_radix_hit, choose_session_affinity_candidate,
        finish_rejected_request, insert_deferred_waiting_request,
        insert_waiting_request_by_priority, lookup_blocks_ready_on_gpu,
        matched_sealed_lookup_blocks, session_affinity_tokens_for_plan,
        staged_prefix_direct_host_blocks, staged_prefix_prefetch_state,
    };
    use crate::kv_tier::{HostPinnedRegion, ReadmissionBlock, ReadmissionPlan, ReadmissionSource};
    use crate::prefix_cache::BlockId;
    use crate::scheduler::cuda::budget::{PageBudget, estimated_request_target};
    use crate::scheduler::cuda::core::{PrefetchTicketState, is_full_sealed_prefix};
    use crate::scheduler::{IncomingRequest, RequestPriority};
    use crate::server_engine::FinishReason;
    use crate::types::BlockFingerprint;
    use std::collections::{HashMap, VecDeque};
    use tokio::sync::mpsc;

    #[test]
    fn best_reusable_slot_prefers_deepest_free_owned_block() {
        let matched_blocks = vec![BlockId(10), BlockId(20), BlockId(30)];
        let free_slots = vec![1, 2];
        let mut owners = HashMap::new();
        owners.insert(BlockId(10), 0);
        owners.insert(BlockId(20), 1);
        owners.insert(BlockId(30), 2);

        let reusable = best_reusable_slot_for_radix_hit(
            &matched_blocks,
            &free_slots,
            &owners,
            &[0, 32, 48],
            16,
        );
        assert_eq!(reusable, Some((2, 48, 48)));
    }

    #[test]
    fn best_reusable_slot_skips_busy_or_stale_slots() {
        let matched_blocks = vec![BlockId(10), BlockId(20)];
        let free_slots = vec![1];
        let mut owners = HashMap::new();
        owners.insert(BlockId(10), 1);
        owners.insert(BlockId(20), 0);

        let reusable =
            best_reusable_slot_for_radix_hit(&matched_blocks, &free_slots, &owners, &[0, 8], 16);
        assert_eq!(reusable, None);
    }

    #[test]
    fn rejected_request_emits_terminal_delta() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        finish_rejected_request(&tx, FinishReason::Length, 17);
        let delta = rx.try_recv().expect("terminal delta");
        assert_eq!(delta.text_delta, "");
        assert_eq!(delta.finish_reason, Some(FinishReason::Length));
        let usage = delta.usage.expect("usage");
        assert_eq!(usage.prompt_tokens, 17);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 17);
    }

    #[test]
    fn direct_host_blocks_require_host_only_sources() {
        let region = HostPinnedRegion {
            offset: 4096,
            len: 2048,
        };
        let host_only = ReadmissionPlan::new(
            16,
            vec![ReadmissionBlock {
                block_id: BlockId(7),
                fingerprint: BlockFingerprint([7; 16]),
                source: Some(ReadmissionSource::HostPinned { region }),
            }],
        );
        let fetched = staged_prefix_direct_host_blocks(&host_only).expect("host-only fetch blocks");
        assert_eq!(fetched.len(), 1);
        assert_eq!(fetched[0].block_id, BlockId(7));
        assert_eq!(fetched[0].host_region, region);
        assert!(!fetched[0].release_after_promote);

        let mixed = ReadmissionPlan::new(
            16,
            vec![
                ReadmissionBlock {
                    block_id: BlockId(8),
                    fingerprint: BlockFingerprint([8; 16]),
                    source: Some(ReadmissionSource::HostPinned { region }),
                },
                ReadmissionBlock {
                    block_id: BlockId(9),
                    fingerprint: BlockFingerprint([9; 16]),
                    source: Some(ReadmissionSource::Disk {
                        fingerprint: BlockFingerprint([9; 16]),
                        payload_len: 4096,
                    }),
                },
            ],
        );
        assert!(staged_prefix_direct_host_blocks(&mixed).is_none());
    }

    #[test]
    fn prefetch_state_only_exists_for_slower_tier_blocks() {
        let host_only = ReadmissionPlan::new(
            16,
            vec![ReadmissionBlock {
                block_id: BlockId(1),
                fingerprint: BlockFingerprint([1; 16]),
                source: Some(ReadmissionSource::HostPinned {
                    region: HostPinnedRegion {
                        offset: 0,
                        len: 4096,
                    },
                }),
            }],
        );
        assert!(staged_prefix_prefetch_state(&host_only).is_none());

        let disk_plan = ReadmissionPlan::new(
            32,
            vec![
                ReadmissionBlock {
                    block_id: BlockId(2),
                    fingerprint: BlockFingerprint([2; 16]),
                    source: None,
                },
                ReadmissionBlock {
                    block_id: BlockId(3),
                    fingerprint: BlockFingerprint([3; 16]),
                    source: Some(ReadmissionSource::Disk {
                        fingerprint: BlockFingerprint([3; 16]),
                        payload_len: 4096,
                    }),
                },
            ],
        );
        assert_eq!(
            staged_prefix_prefetch_state(&disk_plan),
            Some(PrefetchTicketState {
                host_blocks: 0,
                disk_blocks: 1,
                remote_blocks: 0,
            })
        );
    }

    fn queued_request(label: &str, priority: RequestPriority) -> IncomingRequest {
        let (tx, _rx) = mpsc::unbounded_channel();
        IncomingRequest {
            prompt: label.to_string(),
            prompt_tokens: None,
            max_tokens: 1,
            sampling: crate::sampler::SamplingParams::default(),
            stop: None,
            speculative: None,
            priority,
            session_id: None,
            delta_tx: tx,
            trace_context: None,
        }
    }

    #[test]
    fn waiting_insert_after_equal_preserves_fifo_for_same_priority() {
        let mut waiting = VecDeque::from(vec![
            queued_request("high", RequestPriority::High),
            queued_request("first-normal", RequestPriority::Normal),
        ]);

        insert_waiting_request_by_priority(
            &mut waiting,
            queued_request("second-normal", RequestPriority::Normal),
            WaitingInsertBias::AfterEqual,
        );

        assert_eq!(
            waiting
                .into_iter()
                .map(|req| req.prompt)
                .collect::<Vec<_>>(),
            vec!["high", "first-normal", "second-normal"]
        );
    }

    #[test]
    fn waiting_insert_before_equal_gives_requeues_equal_priority_preference() {
        let mut waiting = VecDeque::from(vec![
            queued_request("high", RequestPriority::High),
            queued_request("queued-normal", RequestPriority::Normal),
        ]);

        insert_waiting_request_by_priority(
            &mut waiting,
            queued_request("requeued-normal", RequestPriority::Normal),
            WaitingInsertBias::BeforeEqual,
        );

        assert_eq!(
            waiting
                .into_iter()
                .map(|req| req.prompt)
                .collect::<Vec<_>>(),
            vec!["high", "requeued-normal", "queued-normal"]
        );
    }

    fn deferred_request(
        label: &str,
        priority: RequestPriority,
        hint: WaitingRequestHint,
    ) -> DeferredWaitingRequest {
        DeferredWaitingRequest {
            incoming: queued_request(label, priority),
            prompt_tokens: vec![1, 2, 3],
            hint,
        }
    }

    fn hint_plan(
        matched_len: usize,
        reusable_prefix_len: usize,
        direct_gpu_attach: bool,
        staged_prefix_len: Option<usize>,
        recompute_advised: bool,
    ) -> WaitingRequestHint {
        let staged_prefix_plan = staged_prefix_len
            .map(|matched_len| crate::kv_tier::ReadmissionPlan::new(matched_len, Vec::new()));
        WaitingRequestHint::from_plan(
            &PrefixAdmissionPlan {
                radix_blocks: Vec::new(),
                lookup: crate::kv_tier::LookupOutcome::new(
                    matched_len,
                    Vec::new(),
                    recompute_advised,
                ),
                session_resume_tokens: 0,
                reusable: None,
                direct_gpu_attach,
                attached_prefix_blocks: Vec::new(),
                staged_prefix_plan,
                session_slot_hold: None,
            },
            reusable_prefix_len,
        )
    }

    fn lookup_block(
        block_id: u32,
        hit_kind: crate::kv_tier::HitKind,
    ) -> crate::kv_tier::LookupBlock {
        crate::kv_tier::LookupBlock {
            block_id: Some(BlockId(block_id)),
            hit_kind,
        }
    }

    #[test]
    fn session_affinity_tokens_require_matching_session_prefix_blocks() {
        let session_a = crate::types::SessionId::from("session-a");
        let session_b = crate::types::SessionId::from("session-b");
        let plan = PrefixAdmissionPlan {
            radix_blocks: Vec::new(),
            lookup: crate::kv_tier::LookupOutcome::new(
                48,
                vec![
                    lookup_block(1, crate::kv_tier::HitKind::ReadyOnGpu),
                    lookup_block(2, crate::kv_tier::HitKind::ReadyOnGpu),
                    lookup_block(3, crate::kv_tier::HitKind::ReadyOnGpu),
                ],
                false,
            ),
            session_resume_tokens: 0,
            reusable: None,
            direct_gpu_attach: true,
            attached_prefix_blocks: Vec::new(),
            staged_prefix_plan: None,
            session_slot_hold: None,
        };

        let tokens =
            session_affinity_tokens_for_plan(
                &plan,
                Some(&session_a),
                16,
                |block_id| match block_id {
                    BlockId(1) | BlockId(2) => Some(session_a.clone()),
                    BlockId(3) => Some(session_b.clone()),
                    _ => None,
                },
            );

        assert_eq!(tokens, 32);
    }

    #[test]
    fn session_affinity_tokens_ignore_recompute_advised_hits() {
        let session = crate::types::SessionId::from("session");
        let plan = PrefixAdmissionPlan {
            radix_blocks: Vec::new(),
            lookup: crate::kv_tier::LookupOutcome::new(
                16,
                vec![lookup_block(1, crate::kv_tier::HitKind::StagingFromDisk)],
                true,
            ),
            session_resume_tokens: 0,
            reusable: None,
            direct_gpu_attach: false,
            attached_prefix_blocks: Vec::new(),
            staged_prefix_plan: None,
            session_slot_hold: None,
        };

        assert_eq!(
            session_affinity_tokens_for_plan(&plan, Some(&session), 16, |_| {
                Some(session.clone())
            }),
            0
        );
    }

    #[test]
    fn session_affinity_tokens_use_session_resume_lookup_when_shared_ancestor_differs() {
        let session = crate::types::SessionId::from("session-a");
        let other_session = crate::types::SessionId::from("session-b");
        let plan = PrefixAdmissionPlan {
            radix_blocks: Vec::new(),
            lookup: crate::kv_tier::LookupOutcome::new(
                48,
                vec![
                    lookup_block(1, crate::kv_tier::HitKind::ReadyOnGpu),
                    lookup_block(2, crate::kv_tier::HitKind::ReadyOnGpu),
                    lookup_block(3, crate::kv_tier::HitKind::ReadyOnGpu),
                ],
                false,
            ),
            session_resume_tokens: 48,
            reusable: None,
            direct_gpu_attach: true,
            attached_prefix_blocks: Vec::new(),
            staged_prefix_plan: None,
            session_slot_hold: None,
        };

        let tokens =
            session_affinity_tokens_for_plan(
                &plan,
                Some(&session),
                16,
                |block_id| match block_id {
                    BlockId(1) => Some(other_session.clone()),
                    BlockId(2) | BlockId(3) => Some(session.clone()),
                    _ => None,
                },
            );

        assert_eq!(tokens, 48);
    }

    fn queued_candidate(
        label: &str,
        priority: RequestPriority,
        hint: WaitingRequestHint,
    ) -> QueuedAdmissionCandidate {
        QueuedAdmissionCandidate {
            incoming: queued_request(label, priority),
            prompt_tokens: vec![1, 2, 3],
            plan: PrefixAdmissionPlan {
                radix_blocks: Vec::new(),
                lookup: crate::kv_tier::LookupOutcome::new(0, Vec::new(), false),
                session_resume_tokens: 0,
                reusable: None,
                direct_gpu_attach: false,
                attached_prefix_blocks: Vec::new(),
                staged_prefix_plan: None,
                session_slot_hold: None,
            },
            hint,
        }
    }

    #[test]
    fn session_affinity_candidate_overtakes_same_priority_cold_head() {
        let candidates = vec![
            queued_candidate(
                "cold",
                RequestPriority::Normal,
                WaitingRequestHint::default(),
            ),
            queued_candidate(
                "warm",
                RequestPriority::Normal,
                WaitingRequestHint::default().with_session_affinity_tokens(32),
            ),
        ];

        assert_eq!(choose_session_affinity_candidate(&candidates), Some(1));
    }

    #[test]
    fn session_affinity_candidate_does_not_overtake_higher_priority_head() {
        let candidates = vec![
            queued_candidate(
                "high-cold",
                RequestPriority::High,
                WaitingRequestHint::default(),
            ),
            queued_candidate(
                "normal-warm",
                RequestPriority::Normal,
                WaitingRequestHint::default().with_session_affinity_tokens(32),
            ),
        ];

        assert_eq!(choose_session_affinity_candidate(&candidates), Some(0));
    }

    #[test]
    fn deferred_waiting_keeps_priority_primary_over_prefix_hint() {
        let mut waiting = std::collections::VecDeque::from(vec![deferred_request(
            "high-cold",
            RequestPriority::High,
            WaitingRequestHint::default(),
        )]);

        insert_deferred_waiting_request(
            &mut waiting,
            deferred_request(
                "normal-gpu-ready",
                RequestPriority::Normal,
                hint_plan(48, 48, false, None, false),
            ),
            WaitingInsertBias::BeforeEqual,
        );

        assert_eq!(
            waiting
                .into_iter()
                .map(|req| req.incoming.prompt)
                .collect::<Vec<_>>(),
            vec!["high-cold", "normal-gpu-ready"]
        );
    }

    #[test]
    fn deferred_waiting_prefers_gpu_ready_then_larger_prefix_within_same_priority() {
        let mut waiting = std::collections::VecDeque::from(vec![
            deferred_request(
                "queued-staged",
                RequestPriority::Normal,
                hint_plan(64, 0, false, Some(64), false),
            ),
            deferred_request(
                "queued-cold",
                RequestPriority::Normal,
                WaitingRequestHint::default(),
            ),
        ]);

        insert_deferred_waiting_request(
            &mut waiting,
            deferred_request(
                "gpu-ready",
                RequestPriority::Normal,
                hint_plan(16, 16, false, None, false),
            ),
            WaitingInsertBias::BeforeEqual,
        );

        assert_eq!(
            waiting
                .into_iter()
                .map(|req| req.incoming.prompt)
                .collect::<Vec<_>>(),
            vec!["gpu-ready", "queued-staged", "queued-cold"]
        );
    }

    #[test]
    fn deferred_waiting_before_equal_still_precedes_same_hint_peer() {
        let mut waiting = std::collections::VecDeque::from(vec![deferred_request(
            "queued",
            RequestPriority::Normal,
            hint_plan(32, 32, false, None, false),
        )]);

        insert_deferred_waiting_request(
            &mut waiting,
            deferred_request(
                "requeued",
                RequestPriority::Normal,
                hint_plan(32, 32, false, None, false),
            ),
            WaitingInsertBias::BeforeEqual,
        );

        assert_eq!(
            waiting
                .into_iter()
                .map(|req| req.incoming.prompt)
                .collect::<Vec<_>>(),
            vec!["requeued", "queued"]
        );
    }

    #[test]
    fn admission_budget_accounts_for_prior_reservations_in_same_pass() {
        let mut budget = PageBudget::new(3, vec![0, 0], 4, true);

        assert!(budget.can_fit_target(estimated_request_target(0, 8, 4, 0)));
        budget.reserve_target(estimated_request_target(0, 8, 4, 0));

        assert!(
            !budget.can_fit_target(estimated_request_target(1, 8, 8, 0)),
            "later admissions must see pages reserved by earlier ones in the same assign pass",
        );
    }

    #[test]
    fn admission_budget_honors_attached_prefix_as_existing_seq() {
        let mut budget = PageBudget::new(1, vec![0], 4, true);

        assert!(budget.can_fit_target(estimated_request_target(0, 4, 4, 4)));
        budget.reserve_target(estimated_request_target(0, 4, 4, 4));
        assert_eq!(budget.remaining_free_pages(), 0);
        assert_eq!(budget.planned_seq_len(0), 8);
    }

    #[test]
    fn admission_budget_keeps_estimated_request_headroom_for_active_slots() {
        let mut budget = PageBudget::new(2, vec![4, 0], 4, true);

        // An already-admitted slot must keep its estimated decode tail
        // reserved across scheduler iterations; new admissions cannot borrow
        // it away.
        budget.reserve_target(estimated_request_target(0, 4, 4, 0));
        assert_eq!(budget.remaining_free_pages(), 1);

        assert!(
            !budget.can_fit_target(estimated_request_target(1, 4, 4, 0)),
            "later admissions must respect estimated headroom held by active slots",
        );
        assert!(budget.can_fit_target(estimated_request_target(1, 4, 0, 0)));
    }

    #[test]
    fn matched_sealed_lookup_blocks_ignore_trailing_tombstone() {
        let blocks = vec![
            crate::kv_tier::LookupBlock {
                block_id: Some(crate::prefix_cache::BlockId(10)),
                hit_kind: crate::kv_tier::HitKind::ReadyOnGpu,
            },
            crate::kv_tier::LookupBlock {
                block_id: Some(crate::prefix_cache::BlockId(20)),
                hit_kind: crate::kv_tier::HitKind::ReadyOnGpu,
            },
            crate::kv_tier::LookupBlock {
                block_id: None,
                hit_kind: crate::kv_tier::HitKind::Miss,
            },
        ];

        assert_eq!(matched_sealed_lookup_blocks(&blocks), 2);
        assert!(lookup_blocks_ready_on_gpu(&blocks));
        assert!(is_full_sealed_prefix(
            8,
            4,
            matched_sealed_lookup_blocks(&blocks)
        ));
    }
}
