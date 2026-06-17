//! DSv4 MTP speculative-decode orchestration.
//!
//! `topk == 1` is the validated linear chain: draft depth sequentially, verify
//! `[pending, d0..]`, accept the longest prefix plus the free bonus. `topk > 1`
//! drafts a complete top-k tree, verifies every flattened tree row once, then
//! commits the longest matching root-to-leaf path. The tree path is opt-in via
//! `--mtp-draft-topk`; the default chain path stays structurally unchanged.

use anyhow::{Result, anyhow, ensure};

use crate::dsv4::SpecVerifySchedule;

use super::{DeviceVec, Dsv4CudaExecutor};

struct SpecShape {
    depth: usize,
    topk: usize,
    nodes: usize,
}

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
    /// committed KV + the chain prefix rows `[0, i)` through plain causal attention.
    fn verify_schedule(&self, start_pos: usize) -> SpecVerifySchedule {
        SpecVerifySchedule::chain(self.tokens.len(), start_pos)
    }
}

/// Flattened speculative draft tree. Node 0 is the root/pending token. Nodes are
/// stored breadth-first, so node index == verify row index.
struct DraftTree {
    tokens: Vec<u32>,
    parent: Vec<usize>,
    children: Vec<Vec<usize>>,
    depth: Vec<usize>,
}

impl DraftTree {
    fn new(pending: u32) -> Self {
        Self {
            tokens: vec![pending],
            parent: vec![0],
            children: vec![Vec::new()],
            depth: vec![0],
        }
    }

    fn push(&mut self, parent: usize, token: u32) -> usize {
        let idx = self.tokens.len();
        self.tokens.push(token);
        self.parent.push(parent);
        self.children.push(Vec::new());
        self.depth.push(self.depth[parent] + 1);
        self.children[parent].push(idx);
        idx
    }

    fn len(&self) -> usize {
        self.tokens.len()
    }

    fn token(&self, node: usize) -> u32 {
        self.tokens[node]
    }

    fn parent(&self, node: usize) -> Option<usize> {
        (node != 0).then_some(self.parent[node])
    }

    fn depth(&self, node: usize) -> usize {
        self.depth[node]
    }

    fn branch_ancestors(&self, node: usize) -> Vec<usize> {
        let mut chain = Vec::new();
        let mut cur = node;
        while let Some(parent) = self.parent(cur) {
            if parent != 0 {
                chain.push(parent);
            }
            cur = parent;
        }
        chain.reverse();
        chain
    }

    fn verify_schedule(&self, start_pos: usize) -> SpecVerifySchedule {
        let rows = self.len();
        let positions: Vec<usize> = self.depth.iter().map(|d| start_pos + d).collect();
        let max_depth = self.depth.iter().copied().max().unwrap_or(0);
        let mut owner = vec![usize::MAX; max_depth + 1];
        let mut restores = vec![Vec::new(); rows];
        let mut saves = vec![false; rows];
        for row in 0..rows {
            for ancestor in self.branch_ancestors(row) {
                let depth = self.depth(ancestor);
                if owner[depth] != ancestor {
                    restores[row].push(ancestor);
                    saves[ancestor] = true;
                    owner[depth] = ancestor;
                }
            }
            owner[self.depth(row)] = row;
        }
        SpecVerifySchedule {
            positions,
            restores,
            saves,
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

fn longest_accepted_path(tree: &DraftTree, argmax: &[u32]) -> Vec<usize> {
    let mut path = Vec::new();
    let mut cur = 0;
    while let Some(child) = tree.children[cur]
        .iter()
        .copied()
        .find(|&child| tree.tokens[child] == argmax[cur])
    {
        path.push(child);
        cur = child;
    }
    path
}

fn complete_tree_nodes(depth: usize, topk: usize) -> Option<usize> {
    let mut total = 1usize;
    let mut level = 1usize;
    for _ in 0..depth {
        level = level.checked_mul(topk)?;
        total = total.checked_add(level)?;
    }
    Some(total)
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
        if self.spec_topk() > 1 {
            return self.spec_step_tree(slot_idx, start_pos, position);
        }
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

        // 3. Greedy acceptance is the top-1 view of the verify logits matrix; the
        //    argmax at the divergence is the free bonus token.
        let accepted = longest_accepted_prefix(&chain.tokens, &verify.argmax);
        let bonus = verify.argmax[accepted];

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
                        node: level,
                        restores: Vec::new(),
                        save: false,
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
        // attention per slot/row). The verify persists per-slot spec_normed
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

    fn spec_shape(&self) -> Result<SpecShape> {
        let depth = self.spec_depth();
        let topk = self.spec_topk();
        let nodes = complete_tree_nodes(depth, topk).ok_or_else(|| {
            anyhow!("DSv4 MTP tree node count overflow for depth={depth} topk={topk}")
        })?;
        ensure!(
            nodes <= crate::dsv4::MAX_SPEC_TREE_NODES,
            "DSv4 MTP tree depth={depth} topk={topk} needs {nodes} verify rows, max {}",
            crate::dsv4::MAX_SPEC_TREE_NODES
        );
        Ok(SpecShape { depth, topk, nodes })
    }

    fn spec_step_tree(
        &mut self,
        slot_idx: usize,
        start_pos: usize,
        position: u64,
    ) -> Result<Vec<u32>> {
        let shape = self.spec_shape()?;
        let pending = self.spec_slots[slot_idx]
            .pending
            .ok_or_else(|| anyhow!("DSv4 MTP tree decode missing pending token"))?;
        let hidden = self.spec_slots[slot_idx]
            .hidden
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 MTP tree decode missing previous hidden"))?
            .clone();

        self.model.capture_spec_rings(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            start_pos,
            shape.depth,
        )?;
        let tree = self.draft_tree(slot_idx, pending, &hidden, &shape, start_pos)?;
        ensure!(
            tree.len() == shape.nodes,
            "DSv4 MTP tree built {} rows, expected {} for depth={} topk={}",
            tree.len(),
            shape.nodes,
            shape.depth,
            shape.topk
        );

        let sched = tree.verify_schedule(start_pos);
        crate::attention::set_dsv4_verify_frozen(true);
        let res = self.model.forward_tokens_verify_scheduled(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &tree.tokens,
            start_pos,
            position,
            &sched,
        );
        crate::attention::set_dsv4_verify_frozen(false);
        let mut verify = res?;
        ensure!(
            verify.argmax.len() == tree.len()
                && verify.hiddens.len() == tree.len()
                && verify.logits.seq_len == tree.len(),
            "DSv4 MTP tree verify expected {} rows, got argmax={} hidden={} logits={}",
            tree.len(),
            verify.argmax.len(),
            verify.hiddens.len(),
            verify.logits.seq_len
        );

        let path = longest_accepted_path(&tree, &verify.argmax);
        let accepted = path.len();
        let last_row = path.last().copied().unwrap_or(0);
        let bonus = verify.argmax[last_row];
        let drafts = tree.len() - 1;
        self.mtp_accepts += accepted;
        self.mtp_rejects += drafts - accepted;
        if self.model.tp.config().rank == 0 {
            let path_tokens: Vec<u32> = path.iter().map(|&row| tree.tokens[row]).collect();
            eprintln!(
                "[dsv4-mtp-tree] depth={} topk={} verify_rows={} draft_nodes={} accepted={} path={:?} accept_total={} reject_total={} bonus={bonus}",
                shape.depth,
                shape.topk,
                tree.len(),
                drafts,
                accepted,
                path_tokens,
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
            shape.depth,
        )?;

        let mut rows = Vec::with_capacity(accepted + 1);
        rows.push(0usize);
        rows.extend_from_slice(&path);
        self.model.commit_accepted_fold(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &rows,
            start_pos,
        )?;
        {
            let spec = &mut self.spec_slots[slot_idx];
            spec.pending = Some(bonus);
            spec.hidden = Some(verify.hiddens.swap_remove(last_row));
        }

        let mut out: Vec<u32> = path.iter().map(|&row| tree.tokens[row]).collect();
        out.push(bonus);
        Ok(out)
    }

    fn draft_tree(
        &mut self,
        slot_idx: usize,
        pending: u32,
        trunk_hidden: &DeviceVec,
        shape: &SpecShape,
        start_pos: usize,
    ) -> Result<DraftTree> {
        let mut tree = DraftTree::new(pending);
        let mut node_stream: Vec<Option<DeviceVec>> = vec![None];
        let mut frontier = vec![0usize];
        let mut owner = vec![usize::MAX; shape.depth + 1];
        for depth in 0..shape.depth {
            ensure!(
                !frontier.is_empty(),
                "DSv4 MTP tree frontier unexpectedly empty at depth {depth}"
            );
            let contested = frontier.len() > 1;
            let mut rows = Vec::with_capacity(frontier.len());
            for &node in &frontier {
                let mut restores = Vec::new();
                for ancestor in tree.branch_ancestors(node) {
                    let ancestor_depth = tree.depth(ancestor);
                    if owner[ancestor_depth] != ancestor {
                        restores.push((ancestor, start_pos + ancestor_depth));
                        owner[ancestor_depth] = ancestor;
                    }
                }
                owner[depth] = node;
                rows.push(crate::dsv4::MtpDraftRow {
                    token: tree.token(node),
                    node,
                    restores,
                    save: contested,
                });
            }
            let h_prevs: Vec<&DeviceVec> = frontier
                .iter()
                .map(|&node| match tree.parent(node) {
                    Some(parent) => node_stream[parent].as_ref().expect("parent MTP stream"),
                    None => trunk_hidden,
                })
                .collect();
            let expanded = self.model.mtp_forward_level(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                &rows,
                &h_prevs,
                (start_pos + depth) as u64,
                shape.topk,
            )?;
            drop(h_prevs);
            ensure!(
                expanded.len() == frontier.len(),
                "DSv4 MTP tree depth {depth} expanded {} rows for frontier {}",
                expanded.len(),
                frontier.len()
            );
            let mut next = Vec::with_capacity(frontier.len() * shape.topk);
            for (&node, (candidates, stream)) in frontier.iter().zip(expanded) {
                ensure!(
                    candidates.len() == shape.topk,
                    "DSv4 MTP tree node {node} returned {} candidates, expected {}",
                    candidates.len(),
                    shape.topk
                );
                node_stream[node] = Some(stream);
                for candidate in candidates {
                    let child = tree.push(node, candidate);
                    debug_assert_eq!(child, node_stream.len());
                    node_stream.push(None);
                    next.push(child);
                }
            }
            frontier = next;
        }
        Ok(tree)
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
                node: level,
                restores: Vec::new(),
                save: false,
            }];
            let mut expanded = self.model.mtp_forward_level(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                &rows,
                &[&h_prev],
                (start_pos + level) as u64,
                1,
            )?;
            let (candidates, stream) = expanded
                .pop()
                .ok_or_else(|| anyhow!("DSv4 MTP draft level returned no rows"))?;
            let candidate = candidates
                .first()
                .copied()
                .ok_or_else(|| anyhow!("DSv4 MTP draft level returned no candidate"))?;
            tokens.push(candidate);
            h_prev = stream;
        }
        Ok(DraftChain { tokens })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DraftChain, DraftTree, complete_tree_nodes, longest_accepted_path, longest_accepted_prefix,
    };

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

    /// The chain schedule is strictly increasing positions only.
    #[test]
    fn chain_schedule_positions() {
        let chain = DraftChain {
            tokens: vec![10, 11, 12],
        };
        let sched = chain.verify_schedule(100);
        assert_eq!(sched.positions, vec![100, 101, 102]);
        assert!(sched.is_chain());
    }

    #[test]
    fn complete_tree_node_count() {
        assert_eq!(complete_tree_nodes(2, 2), Some(7));
        assert_eq!(complete_tree_nodes(2, 4), Some(21));
        assert_eq!(complete_tree_nodes(5, 2), Some(63));
    }

    #[test]
    fn tree_longest_branch() {
        let mut tree = DraftTree::new(10);
        let a = tree.push(0, 11);
        let b = tree.push(0, 21);
        tree.push(a, 12);
        tree.push(b, 22);
        let b_alt = tree.push(b, 23);
        let mut argmax = vec![0u32; tree.len()];
        argmax[0] = 21;
        argmax[b] = 23;
        assert_eq!(longest_accepted_path(&tree, &argmax), vec![b, b_alt]);
    }

    #[test]
    fn tree_schedule_replays_same_depth_siblings() {
        let mut tree = DraftTree::new(10);
        let a = tree.push(0, 11);
        let b = tree.push(0, 21);
        let a0 = tree.push(a, 12);
        let a1 = tree.push(a, 13);
        let b0 = tree.push(b, 22);
        let b1 = tree.push(b, 23);

        let sched = tree.verify_schedule(100);
        assert_eq!(sched.positions, vec![100, 101, 101, 102, 102, 102, 102]);
        assert!(!sched.is_chain());
        assert!(sched.restores[0].is_empty());
        assert!(sched.restores[a].is_empty());
        assert!(sched.restores[b].is_empty());
        assert_eq!(sched.restores[a0], vec![a]);
        assert!(sched.restores[a1].is_empty());
        assert_eq!(sched.restores[b0], vec![b]);
        assert!(sched.restores[b1].is_empty());

        let saved: Vec<usize> = (0..tree.len()).filter(|&row| sched.saves[row]).collect();
        assert_eq!(saved, vec![a, b]);
    }
}
