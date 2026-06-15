//! DSv4 MTP speculative-decode orchestration — CHAIN-ONLY (ckl's minimal
//! scheme, 2026-06-12).
//!
//! One step: draft a top-1 chain off the MTP head (depth sequential — the
//! nextn-1 head predicts one step, so level i+1 must chain from level i's
//! stream; ~2 ms/level), verify it in ONE frozen forward (the transformer's
//! teacher-forcing property: every row's logits come out of the same pass),
//! accept the longest matching prefix + the free bonus (the argmax at the
//! divergence IS the target's true token), and commit from the persisted
//! verify rows (fold) or via the re-forward (fallback).
//!
//! WIDTH WAS DELETED (`94d91948` was its last commit): a wide candidate
//! checked WITHOUT its own forward row is exactly the bonus (membership at
//! the divergence = committing the argmax — already free), and a candidate
//! row only pays when the walk CONTINUES through it — measured on this MoE,
//! a verify row costs ~13% of a forward in distinct-expert reads, so the
//! +0.4 A a complete d2k2 tree buys costs more than it returns (33.6 tok/s
//! vs the chain-fold's projected ~37). Width returns only with a
//! depth-robust draft head (OPD axis) AND a cheaper row cost — git history
//! has the tree machinery if that day comes.

use anyhow::{Result, anyhow, ensure};

use crate::dsv4::SpecVerifySchedule;

use super::{DeviceVec, Dsv4CudaExecutor};

/// The draft chain for one step: `tokens[0]` is the already-committed
/// `pending`, `tokens[1..]` the drafts, depth = drafts.
struct DraftChain {
    tokens: Vec<u32>,
}

impl DraftChain {
    fn depth(&self) -> usize {
        self.tokens.len() - 1
    }

    /// The verify-forward schedule: row `i` at `start_pos + i`, attending the
    /// committed KV + the chain prefix rows `[0, i)` (root included).
    fn verify_schedule(&self, start_pos: usize) -> SpecVerifySchedule {
        let n = self.tokens.len();
        SpecVerifySchedule {
            positions: (0..n).map(|i| start_pos + i).collect(),
            ancestors: (0..n).map(|i| (0..i).collect()).collect(),
        }
    }
}

/// Longest accepted prefix: draft `i+1` is accepted iff it equals the
/// target's argmax after row `i`. Returns the accepted draft count.
fn longest_accepted_prefix(tokens: &[u32], argmax: &[u32]) -> usize {
    let mut accepted = 0;
    while accepted + 1 < tokens.len() && tokens[accepted + 1] == argmax[accepted] {
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

        // 1. Draft the chain off the MTP head (depth sequential head passes).
        let chain = self.draft_chain(slot_idx, pending, &hidden, depth, start_pos)?;

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
        let (argmax, mut hiddens) = res?;
        ensure!(
            argmax.len() == chain.tokens.len() && hiddens.len() == chain.tokens.len(),
            "DSv4 MTP verify expected {} rows, got argmax={} hidden={}",
            chain.tokens.len(),
            argmax.len(),
            hiddens.len()
        );

        // 3. Accept the longest matching prefix; the argmax at the divergence
        //    is the free bonus token.
        let accepted = longest_accepted_prefix(&chain.tokens, &argmax);
        let bonus = argmax[accepted];

        self.mtp_accepts += accepted;
        self.mtp_rejects += chain.depth() - accepted;
        if self.model.tp.config().rank == 0 {
            eprintln!(
                "[dsv4-mtp] depth={} nodes={} accepted={accepted} accept_total={} reject_total={} bonus={bonus}",
                depth,
                chain.depth(),
                self.mtp_accepts,
                self.mtp_rejects
            );
            if std::env::var("ARLE_DSV4_MTP_PROBE").as_deref() == Ok("1") {
                eprintln!(
                    "[dsv4-mtp-probe] pending={} target={} drafts_l1={:?}",
                    chain.tokens[0],
                    argmax[0],
                    &chain.tokens[1..2.min(chain.tokens.len())]
                );
            }
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
        // Commit the accepted prefix by folding the persisted verify rows (the
        // tree-attn + commit-fold lanes are both always-on, so there is no
        // re-forward fallback here).
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
            spec.hidden = Some(hiddens.swap_remove(accepted));
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
    /// - `slot.spec_normed`: NOT touched — commit-fold is DISABLED for the
    ///   batched path (per-slot re-forward commit only; the combined verify
    ///   `normed` must not scatter cross-slot into a fold cache — codex P2).
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

        // ── 1. Per-slot pre-draft ring capture, then draft the N chains.
        // The ring capture is ALWAYS per-slot (cheap host-side snapshot, the
        // PROVEN per-row call — never batched). The DRAFT is either:
        //   - per-slot serial `draft_chain` (default — byte-identical to today), OR
        //   - lever 2a: depth-sequential, slot-batched `mtp_forward_level_batched`
        //     (gated `ARLE_DSV4_BATCHED_MTP_DRAFT`), amortizing the MTP-head MoE
        //     over the N slots while the level loop stays sequential (chaining).
        let mut chains: Vec<Vec<u32>> = Vec::with_capacity(n);
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
                )?;
                ensure!(
                    expanded.len() == n,
                    "DSv4 batched MTP draft level {level} returned {} rows for {n} slots",
                    expanded.len()
                );
                let mut next_hidden: Vec<DeviceVec> = Vec::with_capacity(n);
                for (s, (candidate, stream)) in expanded.into_iter().enumerate() {
                    tokens_per_slot[s].push(candidate);
                    next_hidden.push(stream);
                }
                cur_hidden = next_hidden;
            }
            for s in 0..n {
                let tokens = std::mem::take(&mut tokens_per_slot[s]);
                let chain = DraftChain { tokens };
                scheds.push(chain.verify_schedule(start_positions[s]));
                chains.push(chain.tokens);
            }
        } else {
            // Default: per-slot serial draft (byte-identical to today).
            for s in 0..n {
                let chain = self.draft_chain(
                    slot_ids[s],
                    pendings[s],
                    &h_prevs[s],
                    depth,
                    start_positions[s],
                )?;
                scheds.push(chain.verify_schedule(start_positions[s]));
                chains.push(chain.tokens);
            }
        }

        // ── 2. ONE batched verify over the N chains (MoE grouped over all rows,
        // attention per-slot tree-attn). The verify persists per-slot spec_normed
        // (fold is always on) so the commit avoids a per-slot re-forward — the
        // fold is what makes per-row MTP fast; re-forward commit scales with c and
        // erases the verify-batching win (errors/2026-06-15-...submode2-regression).
        let fold = crate::dsv4::dsv4_mtp_commit_fold_enabled();
        let verified = self.model.forward_decode_batch_verify(
            &mut self.slots,
            &mut self.kv_adapter,
            slot_ids,
            &chains,
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
                let tokens = &chains[s];
                let chain_depth = tokens.len() - 1;
                ensure!(
                    argmax.len() == tokens.len() && hiddens.len() == tokens.len(),
                    "DSv4 batched verify slot {slot_idx} expected {} rows, got argmax={} hidden={}",
                    tokens.len(),
                    argmax.len(),
                    hiddens.len()
                );
                let accepted = longest_accepted_prefix(tokens, &argmax);
                let bonus = argmax[accepted];
                self.mtp_accepts += accepted;
                self.mtp_rejects += chain_depth - accepted;
                if self.model.tp.config().rank == 0 {
                    eprintln!(
                        "[dsv4-mtp-batched] slot={slot_idx} depth={depth} nodes={chain_depth} \
                         accepted={accepted} accept_total={} reject_total={} bonus={bonus}",
                        self.mtp_accepts, self.mtp_rejects
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
                let tokens = &chains[s];
                let chain_depth = tokens.len() - 1;
                ensure!(
                    argmax.len() == tokens.len() && hiddens.len() == tokens.len(),
                    "DSv4 batched verify slot {slot_idx} expected {} rows, got argmax={} hidden={}",
                    tokens.len(),
                    argmax.len(),
                    hiddens.len()
                );

                let accepted = longest_accepted_prefix(tokens, &argmax);
                let bonus = argmax[accepted];
                self.mtp_accepts += accepted;
                self.mtp_rejects += chain_depth - accepted;
                if self.model.tp.config().rank == 0 {
                    eprintln!(
                        "[dsv4-mtp-batched] slot={slot_idx} depth={depth} nodes={chain_depth} \
                         accepted={accepted} accept_total={} reject_total={} bonus={bonus}",
                        self.mtp_accepts, self.mtp_rejects
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
                    let (_, mut re_hiddens) = self.model.forward_tokens_verify(
                        &mut self.slots[slot_idx],
                        &mut self.kv_adapter,
                        &prefix,
                        start_pos,
                        position,
                    )?;
                    let spec = &mut self.spec_slots[slot_idx];
                    spec.pending = Some(bonus);
                    spec.hidden = Some(re_hiddens.remove(accepted));
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
    /// of truth — there is no env gate (the old `ARLE_DSV4_MTP_UNCLAMP` made the
    /// flag beg permission from an env var, which is backwards). The clamp to
    /// the snapshot ceiling keeps an over-large request safe-by-construction
    /// rather than overflowing the per-slot spec-ring buffers.
    fn spec_depth(&self) -> usize {
        self.spec_draft_tokens
            .unwrap_or(1)
            .clamp(1, crate::dsv4::MAX_SPEC_DRAFT_DEPTH)
    }

    /// Draft a top-1 chain: `depth` sequential MTP head passes, each chaining
    /// from the previous level's stream. Single node per level ⇒ no ring
    /// contention, no fix-ups.
    fn draft_chain(
        &mut self,
        slot_idx: usize,
        pending: u32,
        trunk_hidden: &DeviceVec,
        depth: usize,
        start_pos: usize,
    ) -> Result<DraftChain> {
        let mut tokens = Vec::with_capacity(depth + 1);
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
            )?;
            let (candidate, stream) = expanded
                .pop()
                .ok_or_else(|| anyhow!("DSv4 MTP draft level returned no rows"))?;
            tokens.push(candidate);
            h_prev = stream;
        }
        Ok(DraftChain { tokens })
    }
}

#[cfg(test)]
mod tests {
    use super::{DraftChain, longest_accepted_prefix};

    /// Accept the longest matching prefix; the divergence argmax is the bonus.
    #[test]
    fn chain_longest_prefix() {
        // pending=10, drafts 11,12,13. argmax[0]=11 ✓, argmax[1]=12 ✓,
        // argmax[2]=99 ✗ → accepted=2, bonus=99.
        let tokens = [10, 11, 12, 13];
        let argmax = [11, 12, 99, 0];
        assert_eq!(longest_accepted_prefix(&tokens, &argmax), 2);
        assert_eq!(argmax[2], 99);
    }

    /// Nothing accepted when the first draft mismatches — bonus still free.
    #[test]
    fn chain_reject_first() {
        assert_eq!(longest_accepted_prefix(&[10, 11], &[99, 0]), 0);
    }

    /// The chain schedule: strictly increasing positions, prefix ancestors
    /// (root included) — what the batched verify lane consumes.
    #[test]
    fn chain_schedule_prefix_ancestors() {
        let chain = DraftChain {
            tokens: vec![10, 11, 12],
        };
        let sched = chain.verify_schedule(100);
        assert_eq!(sched.positions, vec![100, 101, 102]);
        assert_eq!(sched.ancestors[0], Vec::<usize>::new());
        assert_eq!(sched.ancestors[1], vec![0]);
        assert_eq!(sched.ancestors[2], vec![0, 1]);
        assert!(sched.is_chain());
    }
}
