use std::collections::HashMap;

use crate::kv_tier::{BlockLocation, HitKind, LookupBlock, LookupHeuristics, LookupOutcome, Tier};
use crate::prefix_cache::BlockId;
use crate::scheduler::cuda::{ModelForward, Scheduler};
use crate::types::SessionId;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::scheduler::cuda) struct SessionSlot {
    pub(in crate::scheduler::cuda) blocks: Vec<BlockId>,
    pub(in crate::scheduler::cuda) committed_len: usize,
    pub(in crate::scheduler::cuda) ref_count: u32,
    pub(in crate::scheduler::cuda) last_access_tick: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::scheduler) struct SessionSlotHold {
    pub(in crate::scheduler::cuda) session_id: SessionId,
}

pub(in crate::scheduler::cuda) struct SessionSlotLookup {
    pub(in crate::scheduler::cuda) lookup: LookupOutcome,
    pub(in crate::scheduler::cuda) hold: SessionSlotHold,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::scheduler::cuda) enum PressureMode {
    Soft,
    Hard,
}

impl PressureMode {
    fn min_idle_ticks(self, keepalive_ticks: u64) -> u64 {
        match self {
            Self::Soft => keepalive_ticks,
            Self::Hard => 0,
        }
    }
}

impl<M: ModelForward> Scheduler<M> {
    pub(super) fn publish_session_slot(
        &mut self,
        session_id: &SessionId,
        blocks: Vec<BlockId>,
        committed_len: usize,
    ) {
        if blocks.is_empty() || committed_len == 0 {
            return;
        }
        if let Some(old) = self.session_slots.remove(session_id) {
            self.remove_session_block_refs(&old.blocks);
        }
        self.add_session_block_refs(&blocks);
        self.session_slots.insert(
            session_id.clone(),
            SessionSlot {
                blocks,
                committed_len,
                ref_count: 0,
                last_access_tick: self.prefix_cache.logical_clock(),
            },
        );
    }

    pub(in crate::scheduler::cuda) fn lookup_session_slot_or_stage(
        &mut self,
        session_id: &SessionId,
        prompt_len: usize,
        heuristics: LookupHeuristics,
    ) -> Option<SessionSlotLookup> {
        let block_size = self.prefix_cache.block_size();
        let (slot_blocks, committed_len) = {
            let slot = self.session_slots.get(session_id)?;
            (slot.blocks.clone(), slot.committed_len)
        };
        let matched_len = committed_len.min(prompt_len);
        let block_count = matched_len / block_size;
        if block_count == 0 {
            return None;
        }
        let selected = slot_blocks
            .into_iter()
            .take(block_count)
            .collect::<Vec<_>>();
        let rounded_len = block_count * block_size;
        let lookup = self.lookup_session_blocks(&selected, rounded_len, heuristics);
        if lookup.matched_len == 0 {
            self.drop_session_slot(session_id);
            return None;
        }
        let slot = self.session_slots.get_mut(session_id)?;
        slot.ref_count = slot.ref_count.saturating_add(1);
        slot.last_access_tick = self.prefix_cache.logical_clock();
        Some(SessionSlotLookup {
            lookup,
            hold: SessionSlotHold {
                session_id: session_id.clone(),
            },
        })
    }

    pub(in crate::scheduler::cuda) fn release_session_slot_hold(
        &mut self,
        hold: Option<&SessionSlotHold>,
    ) {
        let Some(hold) = hold else {
            return;
        };
        if let Some(slot) = self.session_slots.get_mut(&hold.session_id) {
            slot.ref_count = slot.ref_count.saturating_sub(1);
        }
    }

    pub(in crate::scheduler::cuda) fn session_slot_gpu_ready_plan(
        &self,
        session_id: &SessionId,
        matched_len: usize,
    ) -> Option<crate::kv_tier::ReadmissionPlan> {
        let block_size = self.prefix_cache.block_size();
        let slot = self.session_slots.get(session_id)?;
        let block_count = matched_len / block_size;
        if block_count == 0 || block_count > slot.blocks.len() {
            return None;
        }
        let lookup = self.lookup_session_blocks(
            &slot.blocks[..block_count],
            block_count * block_size,
            LookupHeuristics::default(),
        );
        if lookup.matched_len != matched_len
            || !lookup.blocks.iter().all(|block| {
                matches!(block.hit_kind, HitKind::ReadyOnGpu) && block.block_id.is_some()
            })
        {
            return None;
        }
        self.build_staged_prefix_plan(&lookup)
    }

    pub(in crate::scheduler::cuda) fn retag_session_slot_block(
        &mut self,
        old: BlockId,
        new: BlockId,
    ) {
        let Some(count) = self.session_block_refs.remove(&old) else {
            return;
        };
        *self.session_block_refs.entry(new).or_insert(0) += count;
        for slot in self.session_slots.values_mut() {
            for block in &mut slot.blocks {
                if *block == old {
                    *block = new;
                }
            }
        }
    }

    pub(super) fn block_has_active_session_ref(&self, block_id: BlockId) -> bool {
        self.session_slots
            .values()
            .any(|slot| slot.ref_count > 0 && slot.blocks.contains(&block_id))
    }

    pub(super) fn session_protected_blocks(
        &self,
    ) -> std::collections::HashSet<crate::prefix_cache::BlockId> {
        self.session_block_refs.keys().copied().collect()
    }

    pub(super) fn evict_inactive_session_slots_for_pressure(
        &mut self,
        mode: PressureMode,
        max_slots: usize,
        tier_filter: Option<Tier>,
    ) -> usize {
        let candidates = inactive_session_slot_eviction_candidates(
            &self.session_slots,
            self.prefix_cache.logical_clock(),
            mode.min_idle_ticks(self.config.prefix_cache_keepalive_ticks),
            max_slots,
            |slot| self.session_slot_has_tier_blocks(slot, tier_filter),
        );
        if candidates.is_empty() {
            return 0;
        }

        let mut released_blocks = 0usize;
        let mut released_slots = 0usize;
        for session_id in candidates {
            let Some(slot) = self.session_slots.remove(&session_id) else {
                continue;
            };
            released_slots += 1;
            released_blocks += slot.blocks.len();
            self.remove_session_block_refs(&slot.blocks);
        }

        if released_slots > 0 {
            if mode == PressureMode::Hard {
                self.metrics
                    .record_session_slot_pressure_evictions_hard(released_slots);
            }
            log::info!(
                "session slot pressure eviction: mode={:?} released {} inactive slots ({} block refs)",
                mode,
                released_slots,
                released_blocks
            );
        }
        released_blocks
    }

    fn session_slot_has_tier_blocks(&self, slot: &SessionSlot, tier_filter: Option<Tier>) -> bool {
        tier_filter.is_none_or(|tier| {
            slot.blocks.iter().any(|block_id| {
                self.block_metadata(*block_id)
                    .map(|metadata| block_metadata_tier(metadata.location.as_ref()))
                    == Some(tier)
            })
        })
    }

    fn lookup_session_blocks(
        &self,
        blocks: &[BlockId],
        matched_len: usize,
        heuristics: LookupHeuristics,
    ) -> LookupOutcome {
        let block_size = self.prefix_cache.block_size();
        let mut lookup_blocks = Vec::with_capacity(blocks.len());
        let mut reusable_blocks = 0usize;
        let mut recompute_advised = false;

        for &block_id in blocks {
            let Some(metadata) = self.block_metadata(block_id) else {
                lookup_blocks.push(LookupBlock {
                    block_id: Some(block_id),
                    hit_kind: HitKind::Miss,
                });
                break;
            };
            let hit_kind = match metadata.location {
                Some(BlockLocation::HostPinned { .. }) => HitKind::StagingFromHost,
                Some(BlockLocation::Disk { .. } | BlockLocation::Remote { .. }) => {
                    HitKind::StagingFromDisk
                }
                Some(BlockLocation::Gpu { .. }) | None => HitKind::ReadyOnGpu,
            };
            let hit_kind = if metadata.entry_state == crate::kv_tier::IndexEntryState::Ready {
                hit_kind
            } else {
                HitKind::Miss
            };
            if matches!(hit_kind, HitKind::Miss) {
                lookup_blocks.push(LookupBlock {
                    block_id: Some(block_id),
                    hit_kind,
                });
                break;
            }
            if matches!(
                hit_kind,
                HitKind::StagingFromHost | HitKind::StagingFromDisk
            ) {
                let byte_len = metadata.byte_len.max(block_size as u32);
                recompute_advised |=
                    heuristics.advise_recompute(hit_kind, block_size, byte_len as u64);
            }
            reusable_blocks += 1;
            lookup_blocks.push(LookupBlock {
                block_id: Some(block_id),
                hit_kind,
            });
        }

        LookupOutcome::new(
            matched_len.min(reusable_blocks * block_size),
            lookup_blocks,
            recompute_advised,
        )
    }

    fn add_session_block_refs(&mut self, blocks: &[BlockId]) {
        for &block_id in blocks {
            *self.session_block_refs.entry(block_id).or_insert(0) += 1;
        }
    }

    fn remove_session_block_refs(&mut self, blocks: &[BlockId]) {
        for &block_id in blocks {
            let Some(count) = self.session_block_refs.get_mut(&block_id) else {
                continue;
            };
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.session_block_refs.remove(&block_id);
            }
        }
    }

    fn drop_session_slot(&mut self, session_id: &SessionId) {
        if let Some(slot) = self.session_slots.remove(session_id) {
            self.remove_session_block_refs(&slot.blocks);
        }
    }
}

fn block_metadata_tier(location: Option<&BlockLocation>) -> Tier {
    location.map(BlockLocation::tier).unwrap_or(Tier::Gpu)
}

fn inactive_session_slot_eviction_candidates(
    slots: &HashMap<SessionId, SessionSlot>,
    now: u64,
    min_idle_ticks: u64,
    max_slots: usize,
    mut slot_matches_pressure: impl FnMut(&SessionSlot) -> bool,
) -> Vec<SessionId> {
    if max_slots == 0 {
        return Vec::new();
    }

    let mut candidates = slots
        .iter()
        .filter(|(_, slot)| {
            slot.ref_count == 0 && slot.last_access_tick.saturating_add(min_idle_ticks) <= now
        })
        .filter(|(_, slot)| slot_matches_pressure(slot))
        .map(|(session_id, slot)| (slot.last_access_tick, session_id.clone()))
        .collect::<Vec<_>>();
    candidates.sort_by_key(|(last_access_tick, session_id)| {
        (*last_access_tick, session_id.as_str().to_owned())
    });
    candidates
        .into_iter()
        .take(max_slots)
        .map(|(_, session_id)| session_id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_slot_retains_blocks_outside_radix_refs() {
        let slot = SessionSlot {
            blocks: vec![BlockId(1), BlockId(2)],
            committed_len: 32,
            ref_count: 0,
            last_access_tick: 7,
        };

        assert_eq!(slot.blocks, vec![BlockId(1), BlockId(2)]);
        assert_eq!(slot.committed_len, 32);
        assert_eq!(slot.ref_count, 0);
    }

    #[test]
    fn inactive_session_slot_candidates_are_lru_and_skip_active_holds() {
        let mut slots = HashMap::new();
        slots.insert(
            SessionId::from("old"),
            SessionSlot {
                blocks: vec![BlockId(1)],
                committed_len: 16,
                ref_count: 0,
                last_access_tick: 10,
            },
        );
        slots.insert(
            SessionId::from("active"),
            SessionSlot {
                blocks: vec![BlockId(2)],
                committed_len: 16,
                ref_count: 1,
                last_access_tick: 5,
            },
        );
        slots.insert(
            SessionId::from("newer"),
            SessionSlot {
                blocks: vec![BlockId(3)],
                committed_len: 16,
                ref_count: 0,
                last_access_tick: 20,
            },
        );

        assert_eq!(
            inactive_session_slot_eviction_candidates(&slots, 100, 64, 8, |_| true),
            vec![SessionId::from("old"), SessionId::from("newer")]
        );
    }

    #[test]
    fn inactive_session_slot_candidates_respect_idle_threshold_and_limit() {
        let mut slots = HashMap::new();
        slots.insert(
            SessionId::from("oldest"),
            SessionSlot {
                blocks: vec![BlockId(1)],
                committed_len: 16,
                ref_count: 0,
                last_access_tick: 1,
            },
        );
        slots.insert(
            SessionId::from("too-fresh"),
            SessionSlot {
                blocks: vec![BlockId(2)],
                committed_len: 16,
                ref_count: 0,
                last_access_tick: 50,
            },
        );
        slots.insert(
            SessionId::from("middle"),
            SessionSlot {
                blocks: vec![BlockId(3)],
                committed_len: 16,
                ref_count: 0,
                last_access_tick: 10,
            },
        );

        assert_eq!(
            inactive_session_slot_eviction_candidates(&slots, 70, 32, 1, |_| true),
            vec![SessionId::from("oldest")]
        );
    }

    #[test]
    fn inactive_session_slot_candidates_can_filter_by_pressure_target() {
        let mut slots = HashMap::new();
        slots.insert(
            SessionId::from("old-wrong-tier"),
            SessionSlot {
                blocks: vec![BlockId(1)],
                committed_len: 16,
                ref_count: 0,
                last_access_tick: 1,
            },
        );
        slots.insert(
            SessionId::from("newer-target-tier"),
            SessionSlot {
                blocks: vec![BlockId(2)],
                committed_len: 16,
                ref_count: 0,
                last_access_tick: 10,
            },
        );

        assert_eq!(
            inactive_session_slot_eviction_candidates(&slots, 100, 64, 8, |slot| {
                slot.blocks.contains(&BlockId(2))
            }),
            vec![SessionId::from("newer-target-tier")]
        );
    }

    #[test]
    fn missing_block_location_counts_as_gpu_resident() {
        assert_eq!(block_metadata_tier(None), Tier::Gpu);
        assert_eq!(
            block_metadata_tier(Some(&BlockLocation::HostPinned { offset: 4096 })),
            Tier::HostPinned
        );
    }

    #[test]
    fn pressure_mode_soft_preserves_keepalive_hard_drops_idle_threshold() {
        let mut slots = HashMap::new();
        slots.insert(
            SessionId::from("fresh-inactive"),
            SessionSlot {
                blocks: vec![BlockId(1)],
                committed_len: 16,
                ref_count: 0,
                last_access_tick: 69,
            },
        );

        let now = 70;
        let keepalive_ticks = 64;
        assert!(
            inactive_session_slot_eviction_candidates(
                &slots,
                now,
                PressureMode::Soft.min_idle_ticks(keepalive_ticks),
                1,
                |_| true,
            )
            .is_empty()
        );
        assert_eq!(
            inactive_session_slot_eviction_candidates(
                &slots,
                now,
                PressureMode::Hard.min_idle_ticks(keepalive_ticks),
                1,
                |_| true,
            ),
            vec![SessionId::from("fresh-inactive")]
        );
    }

    #[test]
    fn pressure_modes_respect_ref_count_and_tier_filter() {
        let mut slots = HashMap::new();
        slots.insert(
            SessionId::from("active-target-tier"),
            SessionSlot {
                blocks: vec![BlockId(1)],
                committed_len: 16,
                ref_count: 1,
                last_access_tick: 1,
            },
        );
        slots.insert(
            SessionId::from("inactive-wrong-tier"),
            SessionSlot {
                blocks: vec![BlockId(2)],
                committed_len: 16,
                ref_count: 0,
                last_access_tick: 1,
            },
        );
        slots.insert(
            SessionId::from("inactive-target-tier"),
            SessionSlot {
                blocks: vec![BlockId(3)],
                committed_len: 16,
                ref_count: 0,
                last_access_tick: 2,
            },
        );

        for mode in [PressureMode::Soft, PressureMode::Hard] {
            assert_eq!(
                inactive_session_slot_eviction_candidates(
                    &slots,
                    100,
                    mode.min_idle_ticks(64),
                    8,
                    |slot| slot.blocks.contains(&BlockId(3)),
                ),
                vec![SessionId::from("inactive-target-tier")]
            );
        }
    }
}
