//! DSv4 MTP speculative-decode orchestration.
//!
//! MTP drafts a bounded top-k tree, verifies all tree nodes in one target pass
//! with sparse ancestor masks, then walks the verified logits matrix to accept
//! the longest matching path.

use std::time::Instant;

use anyhow::{anyhow, ensure, Result};
use cuda_kernels::prelude::DeviceContext;

use crate::dsv4::SpecVerifySchedule;

use super::{DeviceVec, Dsv4CudaExecutor};

fn mtp_phase_time_enabled() -> bool {
    matches!(
        std::env::var("ARLE_DSV4_MTP_PHASE_TIME").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
    )
}

fn mtp_phase_start(ctx: &DeviceContext, enabled: bool) -> Instant {
    if enabled {
        ctx.stream.synchronize().ok();
    }
    Instant::now()
}

fn mtp_phase_mark(ctx: &DeviceContext, last: &mut Instant, enabled: bool) -> f64 {
    if !enabled {
        return 0.0;
    }
    ctx.stream.synchronize().ok();
    let now = Instant::now();
    let ms = now.duration_since(*last).as_secs_f64() * 1000.0;
    *last = now;
    ms
}

struct DraftNode {
    token: u32,
    parent: Option<usize>,
    depth: usize,
    hidden: Option<DeviceVec>,
}

struct DraftTree {
    nodes: Vec<DraftNode>,
    children: Vec<Vec<usize>>,
    depth: usize,
    topk: usize,
}

impl DraftTree {
    fn verify_schedule(&self, start_pos: usize) -> SpecVerifySchedule {
        let mut positions = Vec::with_capacity(self.nodes.len());
        let mut ancestors = Vec::with_capacity(self.nodes.len());
        for row in 0..self.nodes.len() {
            let node = &self.nodes[row];
            positions.push(start_pos + node.depth);
            let mut path = Vec::with_capacity(node.depth);
            let mut cur = node.parent;
            while let Some(parent) = cur {
                path.push(parent);
                cur = self.nodes[parent].parent;
            }
            path.reverse();
            ancestors.push(path);
        }
        SpecVerifySchedule {
            positions,
            ancestors,
        }
    }

    fn validate(&self) -> Result<()> {
        ensure!(!self.nodes.is_empty(), "DSv4 MTP draft tree is empty");
        ensure!(
            self.nodes[0].parent.is_none() && self.nodes[0].depth == 0,
            "DSv4 MTP draft tree root is malformed"
        );
        ensure!(
            self.children.len() == self.nodes.len(),
            "DSv4 MTP draft tree children len {} != nodes {}",
            self.children.len(),
            self.nodes.len()
        );
        for (idx, node) in self.nodes.iter().enumerate().skip(1) {
            let parent = node
                .parent
                .ok_or_else(|| anyhow!("DSv4 MTP draft node {idx} has no parent"))?;
            ensure!(
                parent < idx,
                "DSv4 MTP draft node {idx} parent {parent} is not earlier"
            );
            ensure!(
                node.depth == self.nodes[parent].depth + 1,
                "DSv4 MTP draft node {idx} depth {} != parent depth {} + 1",
                node.depth,
                self.nodes[parent].depth
            );
        }
        Ok(())
    }

    fn tokens(&self) -> Vec<u32> {
        self.nodes.iter().map(|node| node.token).collect()
    }

    fn accept_path(&self, argmax: &[u32]) -> Result<(Vec<usize>, u32, usize)> {
        ensure!(
            argmax.len() == self.nodes.len(),
            "DSv4 MTP draft tree argmax rows {} != nodes {}",
            argmax.len(),
            self.nodes.len()
        );
        let mut path = vec![0usize];
        let mut row = 0usize;
        loop {
            let target = argmax[row];
            let Some(&child) = self.children[row]
                .iter()
                .find(|&&child| self.nodes[child].token == target)
            else {
                return Ok((path, target, row));
            };
            path.push(child);
            row = child;
            if self.nodes[row].depth >= self.depth {
                let bonus = *argmax
                    .get(row)
                    .ok_or_else(|| anyhow!("DSv4 MTP draft tree missing bonus row {row}"))?;
                return Ok((path, bonus, row));
            }
        }
    }

    fn accepted_tokens(&self, path: &[usize]) -> Vec<u32> {
        path.iter()
            .copied()
            .skip(1)
            .map(|row| self.nodes[row].token)
            .collect()
    }

    fn add_child(&mut self, parent: usize, token: u32) -> Result<usize> {
        ensure!(
            parent < self.nodes.len(),
            "DSv4 MTP draft tree parent {parent} out of {} nodes",
            self.nodes.len()
        );
        ensure!(
            self.nodes.len() < crate::dsv4::MAX_SPEC_VERIFY_ROWS,
            "DSv4 MTP draft tree exceeds {} verify rows; reduce --mtp-draft-tokens or --mtp-draft-topk",
            crate::dsv4::MAX_SPEC_VERIFY_ROWS
        );
        let row = self.nodes.len();
        self.nodes.push(DraftNode {
            token,
            parent: Some(parent),
            depth: self.nodes[parent].depth + 1,
            hidden: None,
        });
        self.children.push(Vec::new());
        self.children[parent].push(row);
        Ok(row)
    }
}

impl Dsv4CudaExecutor {
    /// One speculative decode step: draft a bounded top-k tree, verify it in a
    /// single frozen forward, accept the longest matching path, commit. Returns
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
        let phase_time = mtp_phase_time_enabled();
        let mut phase_last = mtp_phase_start(&self.model.ctx, phase_time);
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
        let capture_ms = mtp_phase_mark(&self.model.ctx, &mut phase_last, phase_time);

        // 1. Draft a bounded top-k tree. DFS preserves ancestor KV in the draft
        // ring while avoiding sibling-as-ancestor pollution.
        let tree = self.draft_tree(slot_idx, pending, &hidden, depth, topk, start_pos)?;
        tree.validate()?;
        let draft_ms = mtp_phase_mark(&self.model.ctx, &mut phase_last, phase_time);

        // 2. Verify the whole tree in ONE frozen target forward.
        let tokens = tree.tokens();
        let sched = tree.verify_schedule(start_pos);
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
        let verify_ms = mtp_phase_mark(&self.model.ctx, &mut phase_last, phase_time);
        ensure!(
            verify.argmax.len() == tree.nodes.len()
                && verify.hiddens.len() == tree.nodes.len()
                && verify.logits.seq_len == tree.nodes.len(),
            "DSv4 MTP verify expected {} rows, got argmax={} hidden={} logits={}",
            tree.nodes.len(),
            verify.argmax.len(),
            verify.hiddens.len(),
            verify.logits.seq_len
        );

        // 3. Walk the verified tree logits: target top-1 chooses the next child
        // if that child exists; otherwise it is the free bonus.
        let (path, bonus, bonus_parent_row) = tree.accept_path(&verify.argmax)?;
        let accepted = path.len() - 1;

        self.mtp_accepts += accepted;
        self.mtp_rejects += depth - accepted;
        if self.model.tp.config().rank == 0 {
            eprintln!(
                "[dsv4-mtp] depth={} topk={} draft_rows={} verify_rows={} accepted={accepted} accept_total={} reject_total={} bonus={bonus}",
                depth,
                topk,
                tree.nodes.len().saturating_sub(1),
                tree.nodes.len(),
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

        self.model.commit_accepted_fold(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &path,
            start_pos,
        )?;
        {
            let spec = &mut self.spec_slots[slot_idx];
            spec.pending = Some(bonus);
            spec.hidden = Some(verify.hiddens.swap_remove(bonus_parent_row));
        }

        let accepted_tokens = tree.accepted_tokens(&path);
        let mut out = accepted_tokens;
        out.push(bonus);
        let commit_ms = mtp_phase_mark(&self.model.ctx, &mut phase_last, phase_time);
        if phase_time && self.model.tp.config().rank == 0 {
            eprintln!(
                "[dsv4-mtp-phase] n=1 depth={depth} topk={topk} capture={capture_ms:.3}ms draft={draft_ms:.3}ms verify={verify_ms:.3}ms commit={commit_ms:.3}ms total={:.3}ms accepted={accepted} out_tokens={}",
                capture_ms + draft_ms + verify_ms + commit_ms,
                out.len()
            );
        }
        Ok(out)
    }

    /// Cross-slot batched MTP decode step. Each slot drafts a bounded tree; all
    /// tree rows are verified in one batched target pass.
    /// Returns one committed-token list per input slot, index-aligned with
    /// `slot_ids`.
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
    ///   accepted tree row's MTP stream hidden.
    /// - `slot.spec_normed`: the batched verify scatters each slot's tree rows
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
        let phase_time = mtp_phase_time_enabled();
        let mut phase_last = mtp_phase_start(&self.model.ctx, phase_time);

        // ── 1. Per-slot pre-draft ring capture, then draft the N trees.
        // The ring capture is ALWAYS per-slot (cheap host-side snapshot, the
        // proven per-row call — never batched). Draft is per-slot DFS so each
        // branch sees only its ancestors; target verify below is the batched
        // phase.
        let mut trees: Vec<DraftTree> = Vec::with_capacity(n);
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
        let capture_ms = mtp_phase_mark(&self.model.ctx, &mut phase_last, phase_time);
        for s in 0..n {
            let tree = self.draft_tree(
                slot_ids[s],
                pendings[s],
                &h_prevs[s],
                depth,
                topk,
                start_positions[s],
            )?;
            tree.validate()?;
            scheds.push(tree.verify_schedule(start_positions[s]));
            trees.push(tree);
        }
        let draft_ms = mtp_phase_mark(&self.model.ctx, &mut phase_last, phase_time);

        // ── 2. ONE batched verify over the N trees (MoE grouped over all rows,
        // attention per slot/row). The verify persists per-slot spec_normed for
        // the commit fold.
        let tree_tokens: Vec<Vec<u32>> = trees.iter().map(DraftTree::tokens).collect();
        let verified = self.model.forward_decode_batch_verify(
            &mut self.slots,
            &mut self.kv_adapter,
            slot_ids,
            &tree_tokens,
            start_positions,
            &scheds,
        )?;
        let verify_ms = mtp_phase_mark(&self.model.ctx, &mut phase_last, phase_time);
        ensure!(
            verified.len() == n,
            "DSv4 batched verify returned {} trees for {n} slots",
            verified.len()
        );

        // ── 3. Per-slot accept / ring-restore / fold commit. The batched verify
        // above is the amortized phase; commit stays the proven per-slot fold.
        let mut out = Vec::with_capacity(n);
        for (s, (argmax, mut hiddens)) in verified.into_iter().enumerate() {
            let slot_idx = slot_ids[s];
            let start_pos = start_positions[s];
            let tree = &trees[s];
            ensure!(
                argmax.len() == tree.nodes.len() && hiddens.len() == tree.nodes.len(),
                "DSv4 batched verify slot {slot_idx} expected {} rows, got argmax={} hidden={}",
                tree.nodes.len(),
                argmax.len(),
                hiddens.len()
            );

            let (path, bonus, bonus_parent_row) = tree.accept_path(&argmax)?;
            let accepted = path.len() - 1;
            self.mtp_accepts += accepted;
            self.mtp_rejects += depth - accepted;
            if self.model.tp.config().rank == 0 {
                eprintln!(
                    "[dsv4-mtp-batched] slot={slot_idx} depth={depth} topk={topk} \
                     draft_rows={} verify_rows={} accepted={accepted} \
                     accept_total={} reject_total={} bonus={bonus}",
                    tree.nodes.len().saturating_sub(1),
                    tree.nodes.len(),
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
            self.model.commit_accepted_fold(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                &path,
                start_pos,
            )?;
            let spec = &mut self.spec_slots[slot_idx];
            spec.pending = Some(bonus);
            spec.hidden = Some(hiddens.swap_remove(bonus_parent_row));

            let mut slot_out = tree.accepted_tokens(&path);
            slot_out.push(bonus);
            out.push(slot_out);
        }
        let commit_ms = mtp_phase_mark(&self.model.ctx, &mut phase_last, phase_time);
        if phase_time && self.model.tp.config().rank == 0 {
            let out_tokens: usize = out.iter().map(Vec::len).sum();
            eprintln!(
                "[dsv4-mtp-phase] n={n} depth={depth} topk={topk} capture={capture_ms:.3}ms draft={draft_ms:.3}ms verify={verify_ms:.3}ms commit={commit_ms:.3}ms total={:.3}ms out_tokens={out_tokens}",
                capture_ms + draft_ms + verify_ms + commit_ms
            );
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

    fn draft_tree(
        &mut self,
        slot_idx: usize,
        pending: u32,
        trunk_hidden: &DeviceVec,
        depth: usize,
        topk: usize,
        start_pos: usize,
    ) -> Result<DraftTree> {
        let mut tree = DraftTree {
            nodes: vec![DraftNode {
                token: pending,
                parent: None,
                depth: 0,
                hidden: None,
            }],
            children: vec![Vec::new()],
            depth,
            topk,
        };
        self.expand_tree_node(&mut tree, slot_idx, 0, trunk_hidden, start_pos)?;
        Ok(tree)
    }

    fn expand_tree_node(
        &mut self,
        tree: &mut DraftTree,
        slot_idx: usize,
        node_idx: usize,
        h_prev: &DeviceVec,
        start_pos: usize,
    ) -> Result<()> {
        let node_depth = tree.nodes[node_idx].depth;
        if node_depth >= tree.depth {
            return Ok(());
        }
        let token = tree.nodes[node_idx].token;
        let rows = [crate::dsv4::MtpDraftRow { token }];
        let mut expanded = self.model.mtp_forward_level(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &rows,
            &[h_prev],
            (start_pos + node_depth) as u64,
            tree.topk,
        )?;
        let (candidates, stream) = expanded
            .pop()
            .ok_or_else(|| anyhow!("DSv4 MTP draft tree level returned no rows"))?;
        ensure!(
            !candidates.is_empty(),
            "DSv4 MTP draft tree node {node_idx} produced no candidates"
        );
        tree.nodes[node_idx].hidden = Some(stream);
        let child_h_prev = tree.nodes[node_idx]
            .hidden
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 MTP draft tree node {node_idx} missing stream"))?
            .clone();
        for token in candidates {
            let child = tree.add_child(node_idx, token)?;
            self.expand_tree_node(tree, slot_idx, child, &child_h_prev, start_pos)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{DraftNode, DraftTree};

    fn d2k2_tree() -> DraftTree {
        DraftTree {
            nodes: vec![
                DraftNode {
                    token: 10,
                    parent: None,
                    depth: 0,
                    hidden: None,
                },
                DraftNode {
                    token: 11,
                    parent: Some(0),
                    depth: 1,
                    hidden: None,
                },
                DraftNode {
                    token: 21,
                    parent: Some(0),
                    depth: 1,
                    hidden: None,
                },
                DraftNode {
                    token: 12,
                    parent: Some(1),
                    depth: 2,
                    hidden: None,
                },
                DraftNode {
                    token: 22,
                    parent: Some(1),
                    depth: 2,
                    hidden: None,
                },
                DraftNode {
                    token: 31,
                    parent: Some(2),
                    depth: 2,
                    hidden: None,
                },
                DraftNode {
                    token: 32,
                    parent: Some(2),
                    depth: 2,
                    hidden: None,
                },
            ],
            children: vec![
                vec![1, 2],
                vec![3, 4],
                vec![5, 6],
                vec![],
                vec![],
                vec![],
                vec![],
            ],
            depth: 2,
            topk: 2,
        }
    }

    #[test]
    fn d2k2_verify_schedule_is_tree_shaped() {
        let tree = d2k2_tree();
        tree.validate().unwrap();
        let sched = tree.verify_schedule(100);
        assert_eq!(sched.positions, vec![100, 101, 101, 102, 102, 102, 102]);
        assert_eq!(
            sched.ancestors,
            vec![
                vec![],
                vec![0],
                vec![0],
                vec![0, 1],
                vec![0, 1],
                vec![0, 2],
                vec![0, 2]
            ]
        );
    }

    #[test]
    fn d2k2_accepts_branch_verified_path() {
        let tree = d2k2_tree();
        let argmax = [21, 0, 32, 0, 0, 0, 99];
        let (path, bonus, parent) = tree.accept_path(&argmax).unwrap();
        assert_eq!(path, vec![0, 2, 6]);
        assert_eq!(tree.accepted_tokens(&path), vec![21, 32]);
        assert_eq!(bonus, 99);
        assert_eq!(parent, 6);
    }

    #[test]
    fn d2k2_reject_first_keeps_root_bonus() {
        let tree = d2k2_tree();
        let (path, bonus, parent) = tree.accept_path(&[77, 0, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(path, vec![0]);
        assert_eq!(tree.accepted_tokens(&path), Vec::<u32>::new());
        assert_eq!(bonus, 77);
        assert_eq!(parent, 0);
    }
}
