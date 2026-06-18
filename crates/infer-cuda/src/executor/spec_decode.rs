//! DSv4 MTP speculative-decode orchestration.
//!
//! `topk == 1` keeps the linear chain: verify `[pending, d0, d1, ...]` and
//! return accepted drafts plus the free target bonus. `topk > 1` uses the D2
//! branch verifier: verify `[pending, root_topk...]` so D2/T2 stays at 3 target
//! rows while off-chain first-token hits become commit-safe.

use anyhow::{Result, anyhow, ensure};

use crate::dsv4::SpecVerifySchedule;

use super::{DeviceVec, Dsv4CudaExecutor};

/// The draft chain for one step: `tokens[0]` is the already-committed
/// `pending`, `tokens[1..]` the top-1 draft path, depth = drafts. `candidates`
/// is the draft logits matrix after top-k extraction: one candidate row per
/// draft level, highest-first, with `candidates[level][0] == tokens[level + 1]`.
struct DraftChain {
    tokens: Vec<u32>,
    candidates: Vec<Vec<u32>>,
}

impl DraftChain {
    fn depth(&self) -> usize {
        self.tokens.len() - 1
    }

    /// The verify-forward schedule: row `i` at `start_pos + i`, attending the
    /// committed KV + the chain prefix rows `[0, i)` through plain causal attention.
    fn verify_schedule(&self, start_pos: usize) -> SpecVerifySchedule {
        SpecVerifySchedule::prefix_chain(self.tokens.len(), start_pos)
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            !self.tokens.is_empty() && self.candidates.len() + 1 == self.tokens.len(),
            "DSv4 MTP draft matrix shape mismatch: tokens={} candidate_rows={}",
            self.tokens.len(),
            self.candidates.len()
        );
        for (level, candidates) in self.candidates.iter().enumerate() {
            ensure!(
                !candidates.is_empty(),
                "DSv4 MTP draft matrix level {level} returned no candidates"
            );
            ensure!(
                candidates[0] == self.tokens[level + 1],
                "DSv4 MTP draft matrix level {level} top1 {} != chain token {}",
                candidates[0],
                self.tokens[level + 1]
            );
        }
        Ok(())
    }
}

/// D2 top-k branch verifier. `tokens[0]` is pending; `tokens[1..]` are root
/// top-k candidates, each verified at `start_pos + 1` with row 0 as ancestor.
/// The accepted branch token is committed; the target argmax after that branch
/// becomes the pending/bonus token for the next step.
struct DraftBranch {
    tokens: Vec<u32>,
}

struct BranchAccept {
    accepted: usize,
    bonus: u32,
    pending_row: usize,
    rows: Vec<usize>,
    out: Vec<u32>,
}

impl DraftBranch {
    fn verify_schedule(&self, start_pos: usize) -> SpecVerifySchedule {
        SpecVerifySchedule::branch_root(self.tokens.len() - 1, start_pos)
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.tokens.len() >= 2,
            "DSv4 MTP branch verify needs pending + at least one candidate, got {} rows",
            self.tokens.len()
        );
        Ok(())
    }

    fn accept(&self, argmax: &[u32]) -> Result<BranchAccept> {
        ensure!(
            argmax.len() == self.tokens.len(),
            "DSv4 MTP branch argmax rows {} != tokens {}",
            argmax.len(),
            self.tokens.len()
        );
        if let Some(child_row) = self.tokens[1..]
            .iter()
            .position(|&token| token == argmax[0])
            .map(|idx| idx + 1)
        {
            let branch = self.tokens[child_row];
            let bonus = argmax[child_row];
            Ok(BranchAccept {
                accepted: 1,
                bonus,
                pending_row: child_row,
                rows: vec![0, child_row],
                out: vec![branch, bonus],
            })
        } else {
            let bonus = argmax[0];
            Ok(BranchAccept {
                accepted: 0,
                bonus,
                pending_row: 0,
                rows: vec![0],
                out: vec![bonus],
            })
        }
    }
}

/// Longest prefix whose target top-1 is present in the draft top-k matrix.
/// This is diagnostic for the top-k candidate hit rate; only hits that stay on the
/// verified top-1 chain can be committed by the current fold path.
fn longest_candidate_hit_prefix(candidates: &[Vec<u32>], argmax: &[u32]) -> usize {
    let mut hits = 0;
    let max_hits = candidates.len().min(argmax.len());
    while hits < max_hits && candidates[hits].contains(&argmax[hits]) {
        hits += 1;
    }
    hits
}

/// Longest committable prefix. The verifier only ran the top-1 chain, so a
/// target hit on an off-chain top-k candidate becomes the divergence bonus; it
/// is not folded into KV as an accepted draft token.
fn longest_accepted_prefix(chain: &DraftChain, argmax: &[u32]) -> usize {
    let mut accepted = 0;
    let max_accept = chain.candidates.len().min(argmax.len());
    while accepted < max_accept
        && chain.candidates[accepted].contains(&argmax[accepted])
        && chain.tokens[accepted + 1] == argmax[accepted]
    {
        accepted += 1;
    }
    accepted
}

impl Dsv4CudaExecutor {
    /// One speculative decode step: draft a chain, verify it in a single
    /// frozen forward, accept the longest matching prefix, commit. Returns
    /// the committed tokens (accepted drafts + the bonus) and advances the
    /// per-slot spec state (`pending` / `hidden`).
    pub(crate) fn spec_step(
        &mut self,
        slot_idx: usize,
        start_pos: usize,
        position: u64,
    ) -> Result<Vec<u32>> {
        let depth = self.spec_depth();
        let topk = self.spec_topk();
        if topk > 1 {
            return self.spec_step_branch(slot_idx, start_pos, position, depth, topk);
        }
        let pending = self.spec_slots[slot_idx]
            .pending
            .ok_or_else(|| anyhow!("DSv4 MTP decode missing pending token"))?;
        let hidden = self.spec_slots[slot_idx]
            .hidden
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 MTP decode missing previous hidden"))?
            .clone();

        // Frozen-KV P1-2: snapshot the ring slots the draft will overwrite
        // BEFORE any speculative write (the draft writes the frozen target
        // layer's SW/FP8 ring; the batched verify itself is pure).
        self.model.capture_spec_rings(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            start_pos,
            depth,
        )?;

        // 1. Draft the top-1 chain and retain each level's top-k candidate row.
        let chain = self.draft_chain(slot_idx, pending, &hidden, depth, topk, start_pos)?;
        chain.validate()?;

        // 2. Verify the whole chain in ONE frozen forward.
        let sched = chain.verify_schedule(start_pos);
        crate::attention::set_dsv4_verify_frozen(true);
        let res = self.model.forward_tokens_verify_scheduled(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &chain.tokens,
            start_pos,
            position,
            &sched,
        );
        crate::attention::set_dsv4_verify_frozen(false);
        let mut verify = res?;
        ensure!(
            verify.argmax.len() == chain.tokens.len()
                && verify.hiddens.len() == chain.tokens.len()
                && verify.logits.seq_len == chain.tokens.len(),
            "DSv4 MTP verify expected {} rows, got argmax={} hidden={} logits={}",
            chain.tokens.len(),
            verify.argmax.len(),
            verify.hiddens.len(),
            verify.logits.seq_len
        );

        // 3. The verify logits are still the chain matrix. Top-k only changes the
        //    candidate membership test; off-chain hits stop at the free bonus.
        let candidate_hits = longest_candidate_hit_prefix(&chain.candidates, &verify.argmax);
        let accepted = longest_accepted_prefix(&chain, &verify.argmax);
        let bonus = verify.argmax[accepted];

        self.mtp_accepts += accepted;
        self.mtp_rejects += chain.depth() - accepted;
        if self.model.tp.config().rank == 0 {
            eprintln!(
                "[dsv4-mtp] depth={} topk={} draft_rows={} verify_rows={} candidate_hits={} accepted={accepted} accept_total={} reject_total={} bonus={bonus}",
                depth,
                topk,
                chain.depth(),
                chain.tokens.len(),
                candidate_hits,
                self.mtp_accepts,
                self.mtp_rejects
            );
        }

        // 4. Commit: truncate, restore the rejected ring tail (draft layer-0
        //    writes), then fold from the persisted verify rows.
        self.model
            .truncate_slot(&mut self.slots[slot_idx], start_pos)?;
        self.model.restore_spec_ring_tail(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            start_pos,
            accepted,
            depth,
        )?;

        let accepted_tokens: Vec<u32> = chain.tokens[1..=accepted].to_vec();
        // Commit the accepted prefix by folding the persisted verify rows.
        let rows: Vec<usize> = (0..=accepted).collect();
        self.model.commit_accepted_fold(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &rows,
            start_pos,
        )?;
        {
            let spec = &mut self.spec_slots[slot_idx];
            spec.pending = Some(bonus);
            spec.hidden = Some(verify.hiddens.swap_remove(accepted));
        }

        let mut out = accepted_tokens;
        out.push(bonus);
        Ok(out)
    }

    fn spec_step_branch(
        &mut self,
        slot_idx: usize,
        start_pos: usize,
        position: u64,
        depth: usize,
        topk: usize,
    ) -> Result<Vec<u32>> {
        ensure!(
            depth == 2,
            "DSv4 top-k branch verify currently supports D2 only; got depth={depth}"
        );
        ensure!(
            topk < crate::dsv4::MAX_SPEC_VERIFY_ROWS,
            "DSv4 top-k branch verify topk={topk} needs {} verify rows, max {}",
            topk + 1,
            crate::dsv4::MAX_SPEC_VERIFY_ROWS
        );
        let pending = self.spec_slots[slot_idx]
            .pending
            .ok_or_else(|| anyhow!("DSv4 MTP branch decode missing pending token"))?;
        let hidden = self.spec_slots[slot_idx]
            .hidden
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 MTP branch decode missing previous hidden"))?
            .clone();

        self.model.capture_spec_rings(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            start_pos,
            depth,
        )?;

        let branch = self.draft_branch(slot_idx, pending, &hidden, topk, start_pos)?;
        branch.validate()?;
        let sched = branch.verify_schedule(start_pos);
        crate::attention::set_dsv4_verify_frozen(true);
        let res = self.model.forward_tokens_verify_scheduled(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &branch.tokens,
            start_pos,
            position,
            &sched,
        );
        crate::attention::set_dsv4_verify_frozen(false);
        let mut verify = res?;
        ensure!(
            verify.argmax.len() == branch.tokens.len()
                && verify.hiddens.len() == branch.tokens.len()
                && verify.logits.seq_len == branch.tokens.len(),
            "DSv4 MTP branch verify expected {} rows, got argmax={} hidden={} logits={}",
            branch.tokens.len(),
            verify.argmax.len(),
            verify.hiddens.len(),
            verify.logits.seq_len
        );

        let accept = branch.accept(&verify.argmax)?;
        self.mtp_accepts += accept.accepted;
        self.mtp_rejects += depth - accept.accepted;
        if self.model.tp.config().rank == 0 {
            eprintln!(
                "[dsv4-mtp-branch] depth={depth} topk={topk} draft_rows=1 verify_rows={} accepted={} accept_total={} reject_total={} bonus={}",
                branch.tokens.len(),
                accept.accepted,
                self.mtp_accepts,
                self.mtp_rejects,
                accept.bonus
            );
        }

        self.model
            .truncate_slot(&mut self.slots[slot_idx], start_pos)?;
        self.model.restore_spec_ring_tail(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            start_pos,
            accept.accepted,
            depth,
        )?;
        self.model.commit_accepted_fold(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &accept.rows,
            start_pos,
        )?;
        let spec = &mut self.spec_slots[slot_idx];
        spec.pending = Some(accept.bonus);
        spec.hidden = Some(verify.hiddens.swap_remove(accept.pending_row));
        Ok(accept.out)
    }

    /// Cross-slot batched MTP decode step. `topk == 1` batches the linear chains;
    /// `topk > 1` takes the D2 branch verifier so off-chain first-token hits do
    /// not fall back to chain-only bonus semantics. Returns one committed-token
    /// list per input slot, index-aligned with `slot_ids`.
    ///
    /// B=1 never calls this path; the executor keeps single-slot spec on
    /// `spec_step`. B>1 always calls this path so verify does not degrade into
    /// per-slot tiny GEMMs/GEMVs.
    ///
    /// §0.1 mutated state (per slot s, looped — no cross-slot aliasing; each
    /// `Dsv4SlotState` owns its rings):
    /// - `slots[s]` SW/FP8 rings: snapshot pre-draft (`capture_spec_rings`),
    ///   rejected tail restored post-accept (`restore_spec_ring_tail`) — looped
    ///   per slot, the PROVEN per-row calls (NOT a batched snapshot).
    /// - `slots[s].seq_len` + accepted KV `[start_pos .. start_pos+accepted]`:
    ///   written by the per-slot commit fold from persisted verify rows.
    /// - `spec_slots[s].pending` / `.hidden`: set to the bonus token + the
    ///   accepted chain head's MTP stream hidden.
    /// - `slot.spec_normed`: the batched verify scatters each slot's chain rows
    ///   into that slot's own fold cache.
    pub(crate) fn spec_step_batched(
        &mut self,
        slot_ids: &[usize],
        start_positions: &[usize],
    ) -> Result<Vec<Vec<u32>>> {
        let n = slot_ids.len();
        ensure!(n > 0, "DSv4 batched spec step requires at least one slot");
        ensure!(
            start_positions.len() == n,
            "DSv4 batched spec step surface length mismatch (slots {n}, starts {})",
            start_positions.len()
        );
        let depth = self.spec_depth();
        let topk = self.spec_topk();

        // ── 1. Per-slot pre-draft ring capture, then draft the N chains.
        // The ring capture is ALWAYS per-slot (cheap host-side snapshot, the
        // proven per-row call — never batched). Draft itself is depth-sequential
        // and slot-batched: each level depends on the previous token/hidden, but
        // the N slots share one `mtp_forward_level_batched` wave per level.
        let mut chains: Vec<DraftChain> = Vec::with_capacity(n);
        let mut scheds: Vec<SpecVerifySchedule> = Vec::with_capacity(n);

        // Gather per-slot pending tokens + previous hidden (h_prev) BEFORE the
        // draft, and snapshot each slot's draft rings pre-write.
        let mut pendings: Vec<u32> = Vec::with_capacity(n);
        let mut h_prevs: Vec<DeviceVec> = Vec::with_capacity(n);
        for s in 0..n {
            let slot_idx = slot_ids[s];
            let pending = self.spec_slots[slot_idx]
                .pending
                .ok_or_else(|| anyhow!("DSv4 MTP batched decode missing pending token"))?;
            let hidden = self.spec_slots[slot_idx]
                .hidden
                .as_ref()
                .ok_or_else(|| anyhow!("DSv4 MTP batched decode missing previous hidden"))?
                .clone();
            // Snapshot the ring slots the draft will overwrite BEFORE any
            // speculative write (per slot; the batched verify itself is pure).
            self.model.capture_spec_rings(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                start_positions[s],
                depth,
            )?;
            pendings.push(pending);
            h_prevs.push(hidden);
        }
        if topk > 1 {
            return self.spec_step_batched_branch(
                slot_ids,
                start_positions,
                pendings,
                h_prevs,
                depth,
                topk,
            );
        }

        let mut tokens_per_slot: Vec<Vec<u32>> = pendings.iter().map(|&p| vec![p]).collect();
        let mut candidates_per_slot: Vec<Vec<Vec<u32>>> =
            (0..n).map(|_| Vec::with_capacity(depth)).collect();
        // h_prev[s] starts as the trunk hidden; updated to level i's stream.
        let mut cur_hidden: Vec<DeviceVec> = h_prevs;
        for level in 0..depth {
            let rows: Vec<crate::dsv4::MtpDraftRow> = (0..n)
                .map(|s| crate::dsv4::MtpDraftRow {
                    token: tokens_per_slot[s][level],
                })
                .collect();
            let h_refs: Vec<&DeviceVec> = cur_hidden.iter().collect();
            let expanded = self.model.mtp_forward_level_batched(
                &mut self.slots,
                &mut self.kv_adapter,
                slot_ids,
                &rows,
                &h_refs,
                start_positions,
                level,
                topk,
            )?;
            ensure!(
                expanded.len() == n,
                "DSv4 batched MTP draft level {level} returned {} rows for {n} slots",
                expanded.len()
            );
            let mut next_hidden: Vec<DeviceVec> = Vec::with_capacity(n);
            for (s, (candidates, stream)) in expanded.into_iter().enumerate() {
                let candidate = candidates.first().copied().ok_or_else(|| {
                    anyhow!("DSv4 batched MTP draft level {level} returned no candidate")
                })?;
                tokens_per_slot[s].push(candidate);
                candidates_per_slot[s].push(candidates);
                next_hidden.push(stream);
            }
            cur_hidden = next_hidden;
        }
        for s in 0..n {
            let tokens = std::mem::take(&mut tokens_per_slot[s]);
            let candidates = std::mem::take(&mut candidates_per_slot[s]);
            let chain = DraftChain { tokens, candidates };
            chain.validate()?;
            scheds.push(chain.verify_schedule(start_positions[s]));
            chains.push(chain);
        }

        // ── 2. ONE batched verify over the N chains (MoE grouped over all rows,
        // attention per slot/row). The verify persists per-slot spec_normed for
        // the commit fold.
        let chain_tokens: Vec<Vec<u32>> = chains.iter().map(|chain| chain.tokens.clone()).collect();
        let verified = self.model.forward_decode_batch_verify(
            &mut self.slots,
            &mut self.kv_adapter,
            slot_ids,
            &chain_tokens,
            start_positions,
            &scheds,
        )?;
        ensure!(
            verified.len() == n,
            "DSv4 batched verify returned {} chains for {n} slots",
            verified.len()
        );

        // ── 3. Per-slot accept / ring-restore / fold commit. The batched verify
        // above is the amortized phase; commit stays the proven per-slot fold.
        let mut out = Vec::with_capacity(n);
        for (s, (argmax, mut hiddens)) in verified.into_iter().enumerate() {
            let slot_idx = slot_ids[s];
            let start_pos = start_positions[s];
            let chain = &chains[s];
            let tokens = &chain.tokens;
            let chain_depth = chain.depth();
            ensure!(
                argmax.len() == tokens.len() && hiddens.len() == tokens.len(),
                "DSv4 batched verify slot {slot_idx} expected {} rows, got argmax={} hidden={}",
                tokens.len(),
                argmax.len(),
                hiddens.len()
            );

            let candidate_hits = longest_candidate_hit_prefix(&chain.candidates, &argmax);
            let accepted = longest_accepted_prefix(chain, &argmax);
            let bonus = argmax[accepted];
            self.mtp_accepts += accepted;
            self.mtp_rejects += chain_depth - accepted;
            if self.model.tp.config().rank == 0 {
                eprintln!(
                    "[dsv4-mtp-batched] slot={slot_idx} depth={depth} topk={topk} \
                     draft_rows={chain_depth} verify_rows={} candidate_hits={candidate_hits} \
                     accepted={accepted} accept_total={} reject_total={} bonus={bonus}",
                    tokens.len(),
                    self.mtp_accepts,
                    self.mtp_rejects
                );
            }

            self.model
                .truncate_slot(&mut self.slots[slot_idx], start_pos)?;
            self.model.restore_spec_ring_tail(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                start_pos,
                accepted,
                depth,
            )?;
            let accepted_tokens: Vec<u32> = tokens[1..=accepted].to_vec();
            let rows: Vec<usize> = (0..=accepted).collect();
            self.model.commit_accepted_fold(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                &rows,
                start_pos,
            )?;
            let spec = &mut self.spec_slots[slot_idx];
            spec.pending = Some(bonus);
            spec.hidden = Some(hiddens.swap_remove(accepted));

            let mut slot_out = accepted_tokens;
            slot_out.push(bonus);
            out.push(slot_out);
        }
        Ok(out)
    }

    fn spec_step_batched_branch(
        &mut self,
        slot_ids: &[usize],
        start_positions: &[usize],
        pendings: Vec<u32>,
        h_prevs: Vec<DeviceVec>,
        depth: usize,
        topk: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let n = slot_ids.len();
        ensure!(
            depth == 2,
            "DSv4 top-k branch verify currently supports D2 only; got depth={depth}"
        );
        ensure!(
            topk < crate::dsv4::MAX_SPEC_VERIFY_ROWS,
            "DSv4 top-k branch verify topk={topk} needs {} verify rows, max {}",
            topk + 1,
            crate::dsv4::MAX_SPEC_VERIFY_ROWS
        );
        let rows: Vec<crate::dsv4::MtpDraftRow> = pendings
            .iter()
            .map(|&token| crate::dsv4::MtpDraftRow { token })
            .collect();
        let h_refs: Vec<&DeviceVec> = h_prevs.iter().collect();
        let expanded = self.model.mtp_forward_level_batched(
            &mut self.slots,
            &mut self.kv_adapter,
            slot_ids,
            &rows,
            &h_refs,
            start_positions,
            0,
            topk,
        )?;
        ensure!(
            expanded.len() == n,
            "DSv4 batched MTP branch draft returned {} rows for {n} slots",
            expanded.len()
        );

        let mut branches = Vec::with_capacity(n);
        let mut scheds = Vec::with_capacity(n);
        for (s, (candidates, _stream)) in expanded.into_iter().enumerate() {
            ensure!(
                candidates.len() == topk,
                "DSv4 batched MTP branch slot {s} returned {} candidates, expected {topk}",
                candidates.len()
            );
            let mut tokens = Vec::with_capacity(topk + 1);
            tokens.push(pendings[s]);
            tokens.extend(candidates);
            let branch = DraftBranch { tokens };
            branch.validate()?;
            scheds.push(branch.verify_schedule(start_positions[s]));
            branches.push(branch);
        }

        let branch_tokens: Vec<Vec<u32>> = branches
            .iter()
            .map(|branch| branch.tokens.clone())
            .collect();
        let verified = self.model.forward_decode_batch_verify(
            &mut self.slots,
            &mut self.kv_adapter,
            slot_ids,
            &branch_tokens,
            start_positions,
            &scheds,
        )?;
        ensure!(
            verified.len() == n,
            "DSv4 batched branch verify returned {} chains for {n} slots",
            verified.len()
        );

        let mut out = Vec::with_capacity(n);
        for (s, (argmax, mut hiddens)) in verified.into_iter().enumerate() {
            let slot_idx = slot_ids[s];
            let start_pos = start_positions[s];
            let branch = &branches[s];
            ensure!(
                argmax.len() == branch.tokens.len() && hiddens.len() == branch.tokens.len(),
                "DSv4 batched branch verify slot {slot_idx} expected {} rows, got argmax={} hidden={}",
                branch.tokens.len(),
                argmax.len(),
                hiddens.len()
            );
            let accept = branch.accept(&argmax)?;
            self.mtp_accepts += accept.accepted;
            self.mtp_rejects += depth - accept.accepted;
            if self.model.tp.config().rank == 0 {
                eprintln!(
                    "[dsv4-mtp-branch-batched] slot={slot_idx} depth={depth} topk={topk} \
                     draft_rows=1 verify_rows={} accepted={} accept_total={} reject_total={} bonus={}",
                    branch.tokens.len(),
                    accept.accepted,
                    self.mtp_accepts,
                    self.mtp_rejects,
                    accept.bonus
                );
            }

            self.model
                .truncate_slot(&mut self.slots[slot_idx], start_pos)?;
            self.model.restore_spec_ring_tail(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                start_pos,
                accept.accepted,
                depth,
            )?;
            self.model.commit_accepted_fold(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                &accept.rows,
                start_pos,
            )?;
            let spec = &mut self.spec_slots[slot_idx];
            spec.pending = Some(accept.bonus);
            spec.hidden = Some(hiddens.swap_remove(accept.pending_row));
            out.push(accept.out);
        }
        Ok(out)
    }

    /// The draft depth for this step: the explicit `--mtp-draft-tokens` request,
    /// clamped to `[1, MAX_SPEC_DRAFT_DEPTH]`. The CLI flag is the single source
    /// of truth; the clamp to the snapshot ceiling keeps an over-large request safe-by-construction
    /// rather than overflowing the per-slot spec-ring buffers.
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
        self.spec_draft_tokens.is_some()
            || self.spec_draft_topk.is_some()
            || crate::dsv4::dsv4_spec_decode_enabled()
    }

    /// Draft a top-1 chain and retain each level's top-k candidate row from the
    /// MTP logits matrix. Single chain row per level means no ring contention
    /// and the target verifier stays at `depth + 1` rows.
    fn draft_chain(
        &mut self,
        slot_idx: usize,
        pending: u32,
        trunk_hidden: &DeviceVec,
        depth: usize,
        topk: usize,
        start_pos: usize,
    ) -> Result<DraftChain> {
        let mut tokens = Vec::with_capacity(depth + 1);
        let mut matrix = Vec::with_capacity(depth);
        tokens.push(pending);
        let mut h_prev = trunk_hidden.clone();
        for level in 0..depth {
            let rows = [crate::dsv4::MtpDraftRow {
                token: tokens[level],
            }];
            let mut expanded = self.model.mtp_forward_level(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                &rows,
                &[&h_prev],
                (start_pos + level) as u64,
                topk,
            )?;
            let (candidates, stream) = expanded
                .pop()
                .ok_or_else(|| anyhow!("DSv4 MTP draft level returned no rows"))?;
            let candidate = candidates
                .first()
                .copied()
                .ok_or_else(|| anyhow!("DSv4 MTP draft level returned no candidate"))?;
            tokens.push(candidate);
            matrix.push(candidates);
            h_prev = stream;
        }
        Ok(DraftChain {
            tokens,
            candidates: matrix,
        })
    }

    fn draft_branch(
        &mut self,
        slot_idx: usize,
        pending: u32,
        trunk_hidden: &DeviceVec,
        topk: usize,
        start_pos: usize,
    ) -> Result<DraftBranch> {
        let rows = [crate::dsv4::MtpDraftRow { token: pending }];
        let mut expanded = self.model.mtp_forward_level(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &rows,
            &[trunk_hidden],
            start_pos as u64,
            topk,
        )?;
        let (candidates, _stream) = expanded
            .pop()
            .ok_or_else(|| anyhow!("DSv4 MTP branch draft returned no rows"))?;
        ensure!(
            candidates.len() == topk,
            "DSv4 MTP branch draft returned {} candidates, expected {topk}",
            candidates.len()
        );
        let mut tokens = Vec::with_capacity(topk + 1);
        tokens.push(pending);
        tokens.extend(candidates);
        Ok(DraftBranch { tokens })
    }
}

#[cfg(test)]
mod tests {
    use super::{DraftBranch, DraftChain, longest_accepted_prefix, longest_candidate_hit_prefix};

    fn chain(tokens: Vec<u32>, candidates: Vec<Vec<u32>>) -> DraftChain {
        let chain = DraftChain { tokens, candidates };
        chain.validate().unwrap();
        chain
    }

    /// Accept the longest matching prefix; the divergence argmax is the bonus.
    #[test]
    fn chain_longest_prefix() {
        // pending=10, drafts 11,12,13. argmax[0]=11 ✓, argmax[1]=12 ✓,
        // argmax[2]=99 ✗ → accepted=2, bonus=99.
        let chain = chain(vec![10, 11, 12, 13], vec![vec![11], vec![12], vec![13]]);
        let argmax = [11, 12, 99, 0];
        assert_eq!(longest_accepted_prefix(&chain, &argmax), 2);
        assert_eq!(argmax[2], 99);
    }

    /// Nothing accepted when the first draft mismatches — bonus still free.
    #[test]
    fn chain_reject_first() {
        let chain = chain(vec![10, 11], vec![vec![11]]);
        assert_eq!(longest_accepted_prefix(&chain, &[99, 0]), 0);
    }

    /// A top-k hit that leaves the verified top-1 chain is only the divergence
    /// bonus; it cannot be folded as an accepted draft row.
    #[test]
    fn topk_off_chain_hit_is_not_committable() {
        let chain = chain(vec![10, 11, 12], vec![vec![11, 21], vec![12, 22]]);
        let argmax = [11, 22, 0];
        assert_eq!(longest_candidate_hit_prefix(&chain.candidates, &argmax), 2);
        assert_eq!(longest_accepted_prefix(&chain, &argmax), 1);
        assert_eq!(argmax[1], 22);
    }

    #[test]
    fn topk_off_chain_first_token_stops_at_bonus() {
        let chain = chain(vec![10, 11, 12], vec![vec![11, 21], vec![12, 22]]);
        let argmax = [21, 22, 0];
        assert_eq!(longest_candidate_hit_prefix(&chain.candidates, &argmax), 2);
        assert_eq!(longest_accepted_prefix(&chain, &argmax), 0);
        assert_eq!(argmax[0], 21);
    }

    #[test]
    fn chain_schedule_prefix_ancestors() {
        let chain = chain(vec![10, 11, 12], vec![vec![11, 21], vec![12, 22]]);
        let sched = chain.verify_schedule(100);
        assert_eq!(sched.positions, vec![100, 101, 102]);
        assert_eq!(sched.ancestors, vec![vec![], vec![0], vec![0, 1]]);
        assert!(sched.is_prefix_chain_at(100));
        assert!(!sched.is_prefix_chain_at(101));
    }

    #[test]
    fn d2_t2_verify_rows_stay_chain_shaped() {
        let chain = chain(vec![10, 11, 12], vec![vec![11, 21], vec![12, 22]]);
        let sched = chain.verify_schedule(4096);
        assert_eq!(chain.depth(), 2);
        assert_eq!(chain.candidates.len(), 2);
        assert_eq!(sched.positions.len(), 3);
        assert_eq!(sched.positions, vec![4096, 4097, 4098]);
        assert_eq!(sched.ancestors.len(), 3);
    }

    #[test]
    fn d2_t2_branch_verify_rows_are_root_plus_candidates() {
        let branch = DraftBranch {
            tokens: vec![10, 11, 21],
        };
        let sched = branch.verify_schedule(100);
        assert_eq!(sched.positions, vec![100, 101, 101]);
        assert_eq!(sched.ancestors, vec![vec![], vec![0], vec![0]]);
        sched.validate_sparse_at(100).unwrap();
        assert!(!sched.is_prefix_chain_at(100));
    }

    #[test]
    fn branch_accepts_off_chain_first_token() {
        let branch = DraftBranch {
            tokens: vec![10, 11, 21],
        };
        let accept = branch.accept(&[21, 12, 22]).unwrap();
        assert_eq!(accept.accepted, 1);
        assert_eq!(accept.rows, vec![0, 2]);
        assert_eq!(accept.pending_row, 2);
        assert_eq!(accept.out, vec![21, 22]);
        assert_eq!(accept.bonus, 22);
    }

    #[test]
    fn branch_rejects_when_root_misses_topk() {
        let branch = DraftBranch {
            tokens: vec![10, 11, 21],
        };
        let accept = branch.accept(&[99, 12, 22]).unwrap();
        assert_eq!(accept.accepted, 0);
        assert_eq!(accept.rows, vec![0]);
        assert_eq!(accept.pending_row, 0);
        assert_eq!(accept.out, vec![99]);
        assert_eq!(accept.bonus, 99);
    }
}
