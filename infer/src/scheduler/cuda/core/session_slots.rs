use crate::kv_tier::{BlockLocation, HitKind, LookupBlock, LookupHeuristics, LookupOutcome};
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
pub(in crate::scheduler::cuda) struct SessionSlotHold {
    pub(in crate::scheduler::cuda) session_id: SessionId,
}

pub(in crate::scheduler::cuda) struct SessionSlotLookup {
    pub(in crate::scheduler::cuda) lookup: LookupOutcome,
    pub(in crate::scheduler::cuda) hold: SessionSlotHold,
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

    pub(super) fn lookup_session_slot_or_stage(
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

    pub(super) fn release_session_slot_hold(&mut self, hold: Option<&SessionSlotHold>) {
        let Some(hold) = hold else {
            return;
        };
        if let Some(slot) = self.session_slots.get_mut(&hold.session_id) {
            slot.ref_count = slot.ref_count.saturating_sub(1);
        }
    }

    pub(super) fn session_slot_gpu_ready_plan(
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

    pub(super) fn retag_session_slot_block(&mut self, old: BlockId, new: BlockId) {
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
}
