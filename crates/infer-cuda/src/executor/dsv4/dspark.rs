//! One DSpark block decode step: draft, TP-lockstep the block shape, verify,
//! commit the accepted prefix.

use super::*;

impl Dsv4CudaExecutor {
    /// Rebase `slot`'s DSpark draft latent cache to absolute trunk position
    /// `pos` (fresh prefill → 0, restored prefix → the restored frontier).
    pub(super) fn reset_dspark_slot(&mut self, slot: usize, pos: usize) {
        if let Some(ds) = self.dspark.as_mut() {
            ds.slots[slot].df.rebase(pos);
        }
    }

    /// Hot-swap the DSpark Markov head weights from a host f32 snapshot.
    /// `w1` is `[vocab * rank]`, `w2` is `[rank * vocab]`.
    pub(crate) fn update_dspark_markov_weights(&mut self, w1: &[f32], w2: &[f32]) -> Result<()> {
        let dspark = self
            .dspark
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("DSpark head not loaded"))?;
        let stage = dspark
            .draft
            .stages
            .last_mut()
            .ok_or_else(|| anyhow::anyhow!("DSpark draft has no stages"))?;
        let (Some(mw1), Some(mw2)) = (&mut stage.markov_w1, &mut stage.markov_w2) else {
            anyhow::bail!("DSpark exit stage has no Markov head weights");
        };
        let w1_len = mw1.rows * mw1.cols;
        let w2_len = mw2.rows * mw2.cols;
        ensure!(
            w1.len() == w1_len,
            "DSpark markov w1 size mismatch: got {}, expected {w1_len}",
            w1.len()
        );
        ensure!(
            w2.len() == w2_len,
            "DSpark markov w2 size mismatch: got {}, expected {w2_len}",
            w2.len()
        );
        let w1_bf16: Vec<half::bf16> = w1.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let w2_bf16: Vec<half::bf16> = w2.iter().map(|&x| half::bf16::from_f32(x)).collect();
        mw1.data = self
            .model
            .ctx
            .stream
            .clone_htod(&w1_bf16)
            .map_err(|e| anyhow::anyhow!("DSpark markov w1 H2D upload failed: {e}"))?;
        mw2.data = self
            .model
            .ctx
            .stream
            .clone_htod(&w2_bf16)
            .map_err(|e| anyhow::anyhow!("DSpark markov w2 H2D upload failed: {e}"))?;
        self.model.ctx.sync()?;
        Ok(())
    }

    /// Append the prefill forward's stashed multi-row taps to the DSpark draft
    /// context at absolute trunk positions `start_abs..`.
    pub(super) fn seed_dspark_prompt(&mut self, slot: usize, start_abs: usize) -> Result<()> {
        let Self {
            model,
            slots,
            dspark,
            ..
        } = self;
        let Some(ds) = dspark.as_mut() else {
            return Ok(());
        };
        let Some(taps) = slots[slot].take_dspark_prompt_taps() else {
            return Ok(());
        };
        model.dspark_append_context(
            &ds.draft,
            &mut ds.slots[slot].df,
            &taps.bufs,
            taps.rows,
            start_abs,
        )
    }

    fn tp_lockstep_proposal(
        model: &crate::dsv4::Dsv4Model,
        proposal: &mut crate::dsv4::dspark::Dsv4DsparkProposal,
    ) -> Result<()> {
        if model.tp.config().world_size <= 1 {
            return Ok(());
        }
        let block = model.config.dspark_block_size;
        let mut payload = Vec::with_capacity(2 + block);
        payload.push(proposal.draft_len as i32);
        payload.extend(proposal.chain.iter().map(|&t| t as i32));
        payload.resize(2 + block, 0);
        let r0 = model.tp.broadcast_rank0_i32(&model.ctx, &payload)?;
        let r0_chain: Vec<u32> = r0[1..2 + r0[0] as usize]
            .iter()
            .map(|&v| v as u32)
            .collect();
        if r0_chain != proposal.chain {
            proposal.draft_len = r0[0] as usize;
            proposal.chain = r0_chain;
        }
        Ok(())
    }

    fn tp_lockstep_accept(
        model: &crate::dsv4::Dsv4Model,
        accepted: usize,
        bonus: u32,
    ) -> Result<(usize, u32)> {
        if model.tp.config().world_size <= 1 {
            return Ok((accepted, bonus));
        }
        let r0 = model
            .tp
            .broadcast_rank0_i32(&model.ctx, &[accepted as i32, bonus as i32])?;
        Ok((r0[0] as usize, r0[1] as u32))
    }

    /// One DSpark block decode step (greedy). Every committed token is a target
    /// greedy argmax (anchor + verified accepted drafts + bonus).
    pub(super) fn dspark_decode_tokens(
        &mut self,
        slot_idx: usize,
        last_token: u32,
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
    ) -> Result<Vec<u32>> {
        // Request normalization caps EMITTED tokens but NOT the speculative
        // chain, so near `max_seq_len` a full block would overflow the KV/ring
        // and abort an otherwise-valid request: take one non-spec token instead.
        let ds = self
            .dspark
            .as_ref()
            .expect("dspark_decode_tokens without dspark");
        let block = self.model.config.dspark_block_size;
        if start_pos + 1 + block + 1 > ds.max_seq_len {
            let token = self.model.forward_tokens(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                &[last_token],
                start_pos,
                params,
                position,
                infer_plan::PenaltyHistory::default(),
            )?;
            return Ok(vec![token]);
        }
        // Under `is_dspark()` this also populates `slot.dspark_taps()` at the
        // target layers — no arming flag.
        let anchor = self.model.forward_tokens(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &[last_token],
            start_pos,
            params,
            position,
            infer_plan::PenaltyHistory::default(),
        )?;
        let verify_pos = start_pos + 1;

        // Split-borrow: model, slot taps, and the draft + this slot's caches are
        // disjoint fields.
        let mut proposal = {
            let Self {
                model,
                slots,
                dspark,
                ..
            } = self;
            let Dsv4DsparkExec {
                draft,
                sps,
                slots: ds_slots,
                ..
            } = dspark
                .as_mut()
                .expect("dspark_decode_tokens without dspark");
            let rt = &mut ds_slots[slot_idx];
            // The anchor token must enter the draft context BEFORE the block
            // forward reads it.
            model.dspark_append_context(
                draft,
                &mut rt.df,
                slots[slot_idx].dspark_taps(),
                1,
                start_pos,
            )?;
            let block_hidden = model.dspark_forward_block(
                draft,
                &mut rt.df,
                &mut rt.scratch,
                &mut rt.attn_states,
                anchor,
                verify_pos,
            )?;
            model.dspark_build_proposal(draft, &block_hidden, anchor, params.temperature, *sps)?
        };
        ensure!(
            proposal.chain.len() == proposal.draft_len + 1 && proposal.chain[0] == anchor,
            "DSpark proposal chain {} != draft_len {} + 1 (anchor {anchor})",
            proposal.chain.len(),
            proposal.draft_len,
        );

        // The confidence-truncated draft_len drifts by FP across ranks, and a
        // divergent length feeds verify a different token count per rank →
        // collective-count mismatch → deadlock. Adopt rank 0's shape.
        Self::tp_lockstep_proposal(&self.model, &mut proposal)?;
        let draft_len = proposal.draft_len;

        self.model.capture_spec_rings(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            verify_pos,
            draft_len,
        )?;
        let verify = self.model.forward_tokens_verify(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &proposal.chain,
            verify_pos,
            position,
        )?;
        ensure!(
            verify.argmax.len() == proposal.chain.len(),
            "DSpark verify rows {} != chain {}",
            verify.argmax.len(),
            proposal.chain.len()
        );

        self.dspark_commit_block(slot_idx, &proposal, &verify.argmax, verify_pos)
    }

    /// Batched DSpark block decode: drafts per slot, then verifies ALL chains
    /// in ONE batched target forward, amortizing the heaviest phase.
    /// Currently unreachable under the DSv4 B=1 gate pin; preserved for the
    /// batched-draft lever (#230).
    pub(super) fn dspark_decode_tokens_batched(
        &mut self,
        rows: &[Dsv4DecodeBatchRow],
    ) -> Result<Vec<Vec<u32>>> {
        let n = rows.len();
        let max_seq_len = self
            .dspark
            .as_ref()
            .expect("dspark_decode_tokens_batched without dspark")
            .max_seq_len;
        let block = self.model.config.dspark_block_size;

        let mut proposals: Vec<Option<crate::dsv4::dspark::Dsv4DsparkProposal>> =
            (0..n).map(|_| None).collect();
        let mut verify_positions: Vec<usize> = vec![0; n];
        let mut fallback_tokens: Vec<Option<Vec<u32>>> = (0..n).map(|_| None).collect();
        let mut dspark_idxs: Vec<usize> = Vec::with_capacity(n);
        for (i, row) in rows.iter().enumerate() {
            if row.start_pos + 1 + block + 1 > max_seq_len {
                let token = self.model.forward_tokens(
                    &mut self.slots[row.slot],
                    &mut self.kv_adapter,
                    &[row.last_token],
                    row.start_pos,
                    &row.params,
                    row.position,
                    infer_plan::PenaltyHistory::default(),
                )?;
                fallback_tokens[i] = Some(vec![token]);
            } else {
                dspark_idxs.push(i);
                verify_positions[i] = row.start_pos + 1;
            }
        }

        if !dspark_idxs.is_empty() {
            let slot_ids: Vec<usize> = dspark_idxs.iter().map(|&i| rows[i].slot).collect();
            let tokens: Vec<u32> = dspark_idxs.iter().map(|&i| rows[i].last_token).collect();
            let starts: Vec<usize> = dspark_idxs.iter().map(|&i| rows[i].start_pos).collect();
            let positions: Vec<u64> = dspark_idxs.iter().map(|&i| rows[i].position).collect();
            let params: Vec<SamplingParams> = dspark_idxs
                .iter()
                .map(|&i| rows[i].params.clone())
                .collect();
            let penalties = vec![infer_plan::PenaltyHistory::default(); params.len()];
            let anchors = self.model.forward_decode_batch(
                &mut self.slots,
                &mut self.kv_adapter,
                &slot_ids,
                &tokens,
                &starts,
                &positions,
                &params,
                &penalties,
                true, // anchor: the draft reads these taps
            )?;
            // The draft stays per-slot: each slot has independent latent_kv
            // context, so batching it would mean unifying variable-length
            // context for a phase the batched verify already dominates.
            for (k, &i) in dspark_idxs.iter().enumerate() {
                let anchor = anchors[k];
                let verify_pos = verify_positions[i];
                let proposal = {
                    let Self {
                        model,
                        slots,
                        dspark,
                        ..
                    } = self;
                    let Dsv4DsparkExec {
                        draft,
                        sps,
                        slots: ds_slots,
                        ..
                    } = dspark
                        .as_mut()
                        .expect("dspark_decode_tokens_batched without dspark");
                    let rt = &mut ds_slots[rows[i].slot];
                    model.dspark_append_context(
                        draft,
                        &mut rt.df,
                        slots[rows[i].slot].dspark_taps(),
                        1,
                        rows[i].start_pos,
                    )?;
                    let block_hidden = model.dspark_forward_block(
                        draft,
                        &mut rt.df,
                        &mut rt.scratch,
                        &mut rt.attn_states,
                        anchor,
                        verify_pos,
                    )?;
                    model.dspark_build_proposal(
                        draft,
                        &block_hidden,
                        anchor,
                        rows[i].params.temperature,
                        *sps,
                    )?
                };
                proposals[i] = Some(proposal);
            }
        }

        // Divergent chain lengths deadlock the verify collective — rank 0 wins.
        for prop in proposals.iter_mut() {
            let Some(p) = prop else { continue };
            Self::tp_lockstep_proposal(&self.model, p)?;
        }

        let (mut slot_ids, mut chains, mut starts) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            if fallback_tokens[i].is_some() {
                continue;
            }
            slot_ids.push(rows[i].slot);
            chains.push(proposals[i].as_ref().unwrap().chain.clone());
            starts.push(verify_positions[i]);
        }
        if slot_ids.is_empty() {
            return Ok(fallback_tokens.into_iter().map(|t| t.unwrap()).collect());
        }
        let scheds: Vec<crate::dsv4::SpecVerifySchedule> = chains
            .iter()
            .zip(&starts)
            .map(|(chain, &start)| {
                let positions: Vec<usize> = (0..chain.len()).map(|j| start + j).collect();
                let ancestors: Vec<Vec<usize>> =
                    (0..chain.len()).map(|j| (0..j).collect()).collect();
                crate::dsv4::SpecVerifySchedule {
                    positions,
                    ancestors,
                }
            })
            .collect();

        // Must run before verify overwrites the speculative KV band.
        for (s, &slot_idx) in slot_ids.iter().enumerate() {
            self.model.capture_spec_rings(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                starts[s],
                chains[s].len() - 1,
            )?;
        }

        let (verified, _verify_logits) = self.model.forward_decode_batch_verify(
            &mut self.slots,
            &mut self.kv_adapter,
            &slot_ids,
            &chains,
            &starts,
            &scheds,
        )?;

        let mut out: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut verified_iter = verified.into_iter();
        for i in 0..n {
            if let Some(toks) = &fallback_tokens[i] {
                out[i] = toks.clone();
                continue;
            }
            let slot_idx = rows[i].slot;
            let verify_pos = verify_positions[i];
            let proposal = proposals[i].as_ref().unwrap();
            let (argmax, _hiddens) = verified_iter.next().unwrap();
            ensure!(
                argmax.len() == proposal.chain.len(),
                "DSpark batched verify slot {slot_idx} argmax {} != chain {}",
                argmax.len(),
                proposal.chain.len()
            );
            out[i] = self.dspark_commit_block(slot_idx, proposal, &argmax, verify_pos)?;
        }
        Ok(out)
    }

    /// Accept the verified chain prefix and commit it: `argmax[i]` is the target
    /// greedy token AFTER `chain[i]`. Returns the committed tokens (anchor +
    /// accepted drafts + bonus).
    fn dspark_commit_block(
        &mut self,
        slot_idx: usize,
        proposal: &crate::dsv4::dspark::Dsv4DsparkProposal,
        argmax: &[u32],
        verify_pos: usize,
    ) -> Result<Vec<u32>> {
        let draft_len = proposal.draft_len;
        let mut accepted = 0usize;
        while accepted < draft_len && argmax[accepted] == proposal.chain[accepted + 1] {
            accepted += 1;
        }
        let mut bonus = argmax[accepted];

        // accepted/bonus are rank-local FP; adopt rank 0's so every rank commits
        // an identical KV tail — next tick's combined attention reads all ranks'
        // shards, and an inconsistent commit corrupts it.
        (accepted, bonus) = Self::tp_lockstep_accept(&self.model, accepted, bonus)?;

        self.model
            .truncate_slot(&mut self.slots[slot_idx], &mut self.kv_adapter, verify_pos)?;
        self.model.restore_spec_ring_tail(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            verify_pos,
            accepted,
            draft_len,
        )?;
        self.model.commit_accepted_fold(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            0..accepted + 1,
            verify_pos,
        )?;

        // The verify taps cover [anchor, drafts..]; only the committed prefix
        // belongs in the next block's context.
        {
            let Self {
                model,
                slots,
                dspark,
                ..
            } = self;
            let Dsv4DsparkExec {
                draft,
                slots: ds_slots,
                ..
            } = dspark.as_mut().expect("dspark_commit_block without dspark");
            model.dspark_append_context(
                draft,
                &mut ds_slots[slot_idx].df,
                slots[slot_idx].dspark_taps(),
                accepted + 1,
                verify_pos,
            )?;
        }

        self.mtp_accepts += accepted;
        self.mtp_rejects += draft_len - accepted;
        self.mtp_chains += 1;
        if self.model.tp.config().rank == 0 {
            log::debug!(
                "[dsv4-dspark] slot={slot_idx} block={draft_len} accepted={accepted} \
                 bonus={bonus} accept_total={} reject_total={}",
                self.mtp_accepts,
                self.mtp_rejects
            );
        }

        // The bonus is the next block's anchor, committed by the next step's
        // anchor forward.
        self.spec_slots[slot_idx] = Dsv4SpecSlotState::default();
        let mut out = proposal.chain[..accepted + 1].to_vec();
        out.push(bonus);
        Ok(out)
    }
}
