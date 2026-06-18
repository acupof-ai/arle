//! DSv4 MTP speculative-decode orchestration.
//!
//! The drafter always advances one top-1 chain and returns the draft logits'
//! per-level top-k candidate matrix. The target verifier still receives only
//! `[pending, d0, d1, ...]`, so verify rows stay `depth + 1`; `topk > 1` only
//! widens the candidate set used when interpreting the existing verify logits.

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
        //    writes), then either fold from the persisted verify rows or
        //    re-forward the accepted prefix.
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

    /// Cross-slot batched MTP decode step (batched-MTP Stage 1, gated OFF by
    /// `ARLE_DSV4_BATCHED_MTP`). Drive the per-row `spec_step`'s draft + verify
    /// across all N slots at once — batching the MoE/HC/norm over all verify
    /// chains (`forward_decode_batch_verify`) while attention stays per-slot —
    /// then per-slot accept / commit / ring-restore. Returns one committed-token
    /// list per input slot, index-aligned with `slot_ids`.
    ///
    /// N=1 IDENTITY: with a single slot this does structurally the same work as
    /// per-row `spec_step` — same `capture_spec_rings`, same per-slot draft
    /// (`draft_chain`), same verify forward (M=depth+1 = the single chain), same
    /// `longest_accepted_prefix` + bonus, same `truncate_slot` /
    /// `restore_spec_ring_tail` / per-slot re-forward commit.
    ///
    /// §0.1 mutated state (per slot s, looped — no cross-slot aliasing; each
    /// `Dsv4SlotState` owns its rings):
    /// - `slots[s]` SW/FP8 rings: snapshot pre-draft (`capture_spec_rings`),
    ///   rejected tail restored post-accept (`restore_spec_ring_tail`) — looped
    ///   per slot, the PROVEN per-row calls (NOT a batched snapshot).
    /// - `slots[s].seq_len` + accepted KV `[start_pos .. start_pos+accepted]`:
    ///   overwritten by the per-slot commit re-forward (`forward_tokens_verify`,
    ///   which WRITES the accepted-prefix rings + advances seq_len).
    /// - `spec_slots[s].pending` / `.hidden`: set to the bonus token + the
    ///   accepted chain head's MTP stream hidden.
    /// - `slot.spec_normed`: written only when commit-fold is enabled; the
    ///   batched verify scatters each slot's chain rows into that slot's own
    ///   fold cache. With fold disabled, the caller commits by per-slot
    ///   re-forward.
    pub(crate) fn spec_step_batched(
        &mut self,
        slot_ids: &[usize],
        start_positions: &[usize],
        positions: &[u64],
    ) -> Result<Vec<Vec<u32>>> {
        let n = slot_ids.len();
        ensure!(n > 0, "DSv4 batched spec step requires at least one slot");
        ensure!(
            start_positions.len() == n && positions.len() == n,
            "DSv4 batched spec step surface length mismatch (slots {n}, starts {}, positions {})",
            start_positions.len(),
            positions.len()
        );
        let depth = self.spec_depth();
        let topk = self.spec_topk();

        // ── 1. Per-slot pre-draft ring capture, then draft the N chains.
        // The ring capture is ALWAYS per-slot (cheap host-side snapshot, the
        // PROVEN per-row call — never batched). The DRAFT is either:
        //   - per-slot serial `draft_chain` (default — byte-identical to today), OR
        //   - lever 2a: depth-sequential, slot-batched `mtp_forward_level_batched`
        //     (gated `ARLE_DSV4_BATCHED_MTP_DRAFT`), amortizing the MTP-head MoE
        //     over the N slots while the level loop stays sequential (chaining).
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

        if crate::dsv4::dsv4_batched_mtp_draft_enabled() {
            // Lever 2a: per depth LEVEL, ONE batched MTP-head forward over the N
            // slots' current draft tokens. The level loop is sequential (level
            // i+1 chains from level i's per-slot stream + token); the N slots
            // batch WITHIN each level. With N=1 / depth identical to the per-slot
            // draft_chain (same head math, same per-slot attention, same chaining).
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
                let positions: Vec<u64> = (0..n)
                    .map(|s| (start_positions[s] + level) as u64)
                    .collect();
                let expanded = self.model.mtp_forward_level_batched(
                    &mut self.slots,
                    &mut self.kv_adapter,
                    slot_ids,
                    &rows,
                    &h_refs,
                    &positions,
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
        } else {
            // Default: per-slot serial draft (byte-identical to today).
            for s in 0..n {
                let chain = self.draft_chain(
                    slot_ids[s],
                    pendings[s],
                    &h_prevs[s],
                    depth,
                    topk,
                    start_positions[s],
                )?;
                chain.validate()?;
                scheds.push(chain.verify_schedule(start_positions[s]));
                chains.push(chain);
            }
        }

        // ── 2. ONE batched verify over the N chains (MoE grouped over all rows,
        // attention per slot/row). The verify persists per-slot spec_normed
        // (fold is always on) so the commit avoids a per-slot re-forward — the
        // fold is what makes per-row MTP fast; re-forward commit scales with c and
        // erases the verify-batching win (errors/2026-06-15-...submode2-regression).
        let chain_tokens: Vec<Vec<u32>> = chains.iter().map(|chain| chain.tokens.clone()).collect();
        let fold = crate::dsv4::dsv4_mtp_commit_fold_enabled();
        let verified = self.model.forward_decode_batch_verify(
            &mut self.slots,
            &mut self.kv_adapter,
            slot_ids,
            &chain_tokens,
            start_positions,
            &scheds,
            fold,
        )?;
        ensure!(
            verified.len() == n,
            "DSv4 batched verify returned {} chains for {n} slots",
            verified.len()
        );

        // ── 3. Per-slot accept / commit / ring-restore. The batched verify above
        // is the amortized phase; the per-slot draft (phase 0) and per-slot commit
        // (phase 2) stay sequential — the profile shows which dominates the wave.
        //
        // Commit is FOLD (default — re-ingest the per-slot spec_normed the batched
        // verify persisted; no re-forward, the cheap path per-row MTP uses) or
        // RE-FORWARD (fallback when fold is disabled). When fold is on AND lever 2b
        // (`ARLE_DSV4_BATCHED_MTP_COMMIT`) is enabled, the per-slot fold's 60-layer
        // host loop is shared across all slots via `commit_accepted_fold_batched`:
        // the cheap per-slot truncate+restore stay per-slot (host/ring ops), then
        // ONE batched fold over all slots, then the per-slot pending/hidden set.
        let batched_commit = fold && crate::dsv4::dsv4_batched_mtp_commit_enabled();
        let mut out = Vec::with_capacity(n);
        if batched_commit {
            // Phase 3a (per slot — cheap host/ring ops): accept, stats, truncate,
            // rejected-tail restore. Collect each slot's accepted count + the
            // accepted-row hidden + bonus + output tokens for after the batched fold.
            let mut accepted_per_slot: Vec<usize> = Vec::with_capacity(n);
            let mut accepted_hiddens: Vec<DeviceVec> = Vec::with_capacity(n);
            let mut bonuses: Vec<u32> = Vec::with_capacity(n);
            let mut slot_outs: Vec<Vec<u32>> = Vec::with_capacity(n);
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
                let mut slot_out: Vec<u32> = tokens[1..=accepted].to_vec();
                slot_out.push(bonus);
                accepted_per_slot.push(accepted);
                // The batched verify's per-slot hidden at the accepted row is the
                // next step's trunk (same as per-row fold's verify hidden).
                accepted_hiddens.push(hiddens.swap_remove(accepted));
                bonuses.push(bonus);
                slot_outs.push(slot_out);
            }
            // Phase 3b: ONE 60-layer pass folding the accepted prefix for every
            // slot (the amortization — per-slot rings written, no MoE).
            self.model.commit_accepted_fold_batched(
                &mut self.slots,
                &mut self.kv_adapter,
                slot_ids,
                &accepted_per_slot,
                start_positions,
            )?;
            // Phase 3c (per slot): publish pending + trunk hidden, emit tokens.
            for s in 0..n {
                let spec = &mut self.spec_slots[slot_ids[s]];
                spec.pending = Some(bonuses[s]);
                spec.hidden = Some(accepted_hiddens.remove(0));
                out.push(std::mem::take(&mut slot_outs[s]));
            }
        } else {
            for (s, (argmax, mut hiddens)) in verified.into_iter().enumerate() {
                let slot_idx = slot_ids[s];
                let start_pos = start_positions[s];
                let position = positions[s];
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

                // Truncate to the committed length, restore the rejected ring tail
                // (the draft's speculative layer-0 writes), then commit the accepted
                // prefix: FOLD (default — re-ingest the per-slot spec_normed the
                // batched verify persisted; no re-forward, the cheap path per-row MTP
                // uses) or RE-FORWARD (fallback when fold is disabled).
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
                if fold {
                    let rows: Vec<usize> = (0..=accepted).collect();
                    self.model.commit_accepted_fold(
                        &mut self.slots[slot_idx],
                        &mut self.kv_adapter,
                        &rows,
                        start_pos,
                    )?;
                    let spec = &mut self.spec_slots[slot_idx];
                    spec.pending = Some(bonus);
                    // The batched verify's per-slot hidden at the accepted row is the
                    // next step's trunk (same as per-row fold's verify hidden).
                    spec.hidden = Some(hiddens.swap_remove(accepted));
                } else {
                    let mut prefix = Vec::with_capacity(accepted + 1);
                    prefix.push(tokens[0]);
                    prefix.extend_from_slice(&accepted_tokens);
                    let mut verify = self.model.forward_tokens_verify(
                        &mut self.slots[slot_idx],
                        &mut self.kv_adapter,
                        &prefix,
                        start_pos,
                        position,
                    )?;
                    let spec = &mut self.spec_slots[slot_idx];
                    spec.pending = Some(bonus);
                    spec.hidden = Some(verify.hiddens.remove(accepted));
                    hiddens.clear();
                }

                let mut slot_out = accepted_tokens;
                slot_out.push(bonus);
                out.push(slot_out);
            }
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
}

#[cfg(test)]
mod tests {
    use super::{DraftChain, longest_accepted_prefix, longest_candidate_hit_prefix};

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
        assert!(sched.is_chain());
        assert!(sched.has_prefix_ancestors());
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
}
