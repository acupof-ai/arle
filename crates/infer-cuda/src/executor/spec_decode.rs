//! DSv4 MTP speculative-decode orchestration.
//!
//! MTP drafts a top-1 chain, records top-k candidates at each draft row, verifies
//! the chain in one target pass, then matches target top-1 against those
//! candidates. `topk` does not add verify rows on this path.

use anyhow::{Result, anyhow, ensure};

use infer_plan::DraftChain;

use super::{DeviceVec, Dsv4CudaExecutor};

// Re-exported for `executor::qwen35`, which routes through the same
// step-level scheduling types.
pub(crate) use infer_plan::{
    DecodeDispatch, DecodeKvClass, SpecChain, SpecKind, assign_row_offsets, decide_decode,
    dspark_draft_plan, flatten_chains, greedy_seeded_indices, pages_covering,
    qwen_spec_decode_compatible, spec_accept_greedy, spec_accept_totals, speculative_chain_fits,
};

impl Dsv4CudaExecutor {
    /// Returns the committed tokens (accepted drafts + the bonus) and advances
    /// the per-slot spec state (`pending` / `hidden`).
    pub(crate) fn spec_step(
        &mut self,
        slot_idx: usize,
        start_pos: usize,
        position: u64,
    ) -> Result<Vec<u32>> {
        let depth = self.spec_depth();
        let topk = self.spec_topk();
        let pending = self.spec_slots[slot_idx]
            .pending
            .ok_or_else(|| anyhow!("DSv4 MTP decode missing pending token"))?;
        let hidden = self.spec_slots[slot_idx]
            .hidden
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 MTP decode missing previous hidden"))?
            .clone();

        // Snapshot the ring slots the draft will overwrite BEFORE any
        // speculative write (the draft writes the frozen target layer's
        // SW/FP8 ring; the batched verify itself is pure).
        self.model.capture_spec_rings(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            start_pos,
            depth,
        )?;

        // `topk` samples extra candidates from each existing draft logits row;
        // siblings are verify-only candidates, not additional MTP forwards.
        let chain = self.draft_chain(slot_idx, pending, &hidden, depth, topk, start_pos)?;
        chain.validate()?;

        let tokens = chain.tokens();
        let sched = chain.verify_schedule(start_pos);
        crate::attention::set_dsv4_verify_frozen(true);
        let res = self.model.forward_tokens_verify_scheduled(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &tokens,
            start_pos,
            position,
            &sched,
        );
        crate::attention::set_dsv4_verify_frozen(false);
        let mut verify = res?;
        ensure!(
            verify.argmax.len() == chain.len()
                && verify.hiddens.len() == chain.len()
                && verify.logits.seq_len == chain.len(),
            "DSv4 MTP verify expected {} rows, got argmax={} hidden={} logits={}",
            chain.len(),
            verify.argmax.len(),
            verify.hiddens.len(),
            verify.logits.seq_len
        );

        // A non-chain top-k hit is still a valid bonus token, but the path
        // stops at its parent because no later chain row was conditioned on
        // that token.
        let (path, bonus, bonus_parent_row, topk_bonus_hit) = chain.accept_path(&verify.argmax)?;
        let accepted = path.len() - 1;

        self.mtp_accepts += accepted;
        self.mtp_rejects += depth - accepted;
        self.mtp_chains += 1;
        if self.model.tp.config().rank == 0 {
            log::debug!(
                "[dsv4-mtp] depth={} topk={} draft_rows={} verify_rows={} accepted={accepted} topk_bonus_hit={topk_bonus_hit} accept_total={} reject_total={} bonus={bonus}",
                depth,
                topk,
                chain.len().saturating_sub(1),
                chain.len(),
                self.mtp_accepts,
                self.mtp_rejects
            );
        }

        self.model
            .truncate_slot(&mut self.slots[slot_idx], &mut self.kv_adapter, start_pos)?;
        self.model.restore_spec_ring_tail(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            start_pos,
            accepted,
            depth,
        )?;

        self.model.commit_accepted_fold(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            path.iter().copied(),
            start_pos,
        )?;
        {
            let spec = &mut self.spec_slots[slot_idx];
            spec.pending = Some(bonus);
            spec.hidden = Some(verify.hiddens.swap_remove(bonus_parent_row));
        }

        let accepted_tokens = chain.accepted_tokens(&path);
        let mut out = accepted_tokens;
        out.push(bonus);
        Ok(out)
    }

    /// The CLI flag is the single source of truth; the clamp to the snapshot
    /// ceiling keeps an over-large request safe-by-construction rather than
    /// overflowing the per-slot spec-ring buffers.
    fn spec_depth(&self) -> usize {
        self.spec_draft_tokens
            .unwrap_or(crate::dsv4::DEFAULT_SPEC_DRAFT_DEPTH)
            .clamp(1, crate::dsv4::MAX_SPEC_DRAFT_DEPTH)
    }

    pub(super) fn spec_topk(&self) -> usize {
        self.spec_draft_topk
            .unwrap_or(crate::dsv4::DEFAULT_SPEC_DRAFT_TOPK)
            .max(1)
    }

    pub(super) fn spec_requested(&self) -> bool {
        self.spec_draft_tokens.is_some() || self.spec_draft_topk.is_some() || self.dspark.is_some()
    }

    fn draft_chain(
        &mut self,
        slot_idx: usize,
        pending: u32,
        trunk_hidden: &DeviceVec,
        depth: usize,
        topk: usize,
        start_pos: usize,
    ) -> Result<DraftChain> {
        let mut chain = DraftChain::start(pending, depth);
        let mut chain_token = pending;
        let mut chain_hidden: Option<DeviceVec> = None;
        for level in 0..depth {
            let row = crate::dsv4::MtpDraftRow { token: chain_token };
            let h_prev = if level == 0 {
                trunk_hidden
            } else {
                chain_hidden.as_ref().ok_or_else(|| {
                    anyhow!("DSv4 MTP draft chain hidden missing at level {level}")
                })?
            };
            let mut expanded = self.model.mtp_forward_level(
                &mut self.slots,
                &mut self.kv_adapter,
                &[slot_idx],
                std::slice::from_ref(&row),
                &[h_prev],
                &[(start_pos + level) as u64],
                topk,
            )?;
            let (candidates, stream) = expanded
                .pop()
                .ok_or_else(|| anyhow!("DSv4 MTP draft chain level {level} returned no row"))?;
            ensure!(
                !candidates.is_empty(),
                "DSv4 MTP draft chain level {level} produced no candidates"
            );
            let next = candidates[0];
            chain.push_candidates(candidates);
            chain.add_chain_child(next)?;
            chain_token = next;
            chain_hidden = Some(stream);
        }
        Ok(chain)
    }
}
