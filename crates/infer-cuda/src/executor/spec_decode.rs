//! DSv4 MTP speculative-decode orchestration.
//!
//! MTP drafts a top-1 chain, records top-k candidates at each draft row, verifies
//! the chain in one target pass, then matches target top-1 against those
//! candidates. `topk` does not add verify rows on this path.

use anyhow::{Result, anyhow, ensure};

use crate::dsv4::SpecVerifySchedule;

use super::{DeviceVec, Dsv4CudaExecutor};

/// Both CUDA executors resolve this from their own state (`dspark`/`mtp`
/// handles).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SpecKind {
    None,
    Mtp,
    Dspark,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DecodeRoute {
    Plain,
    /// c=1 / low-concurrency win.
    Mtp,
    /// c=1 / low-concurrency win.
    Dspark,
}

/// Speculate only at or below the concurrency gate. At small batch the GPU is
/// memory-bound and the B+1 verify positions are ~free, so speculation wins;
/// above the gate the target forward is compute-bound and the same verify costs
/// ~(B+1)× step time for ~2.5 committed tokens, a net loss — so fall back to the
/// plain batched path that scales. `gate` is `--spec-max-batch` (default 1).
/// Pure so the routing is unit-tested without a GPU.
///
/// `vetoed` covers request features the selected speculative implementation
/// cannot apply to every accepted token in a chain.
pub(super) fn route_decode(
    spec_kind: SpecKind,
    n_rows: usize,
    gate: usize,
    vetoed: bool,
) -> DecodeRoute {
    if vetoed || n_rows > gate {
        return DecodeRoute::Plain;
    }
    match spec_kind {
        SpecKind::Dspark => DecodeRoute::Dspark,
        SpecKind::Mtp => DecodeRoute::Mtp,
        SpecKind::None => DecodeRoute::Plain,
    }
}

struct DraftNode {
    token: u32,
    parent: Option<usize>,
    depth: usize,
}

struct DraftChain {
    nodes: Vec<DraftNode>,
    candidates: Vec<Vec<u32>>,
    depth: usize,
}

impl DraftChain {
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
        ensure!(!self.nodes.is_empty(), "DSv4 MTP draft chain is empty");
        ensure!(
            self.nodes[0].parent.is_none() && self.nodes[0].depth == 0,
            "DSv4 MTP draft chain root is malformed"
        );
        ensure!(
            self.nodes.len() == self.depth + 1,
            "DSv4 MTP draft chain rows {} != depth {} + 1",
            self.nodes.len(),
            self.depth
        );
        ensure!(
            self.candidates.len() == self.depth,
            "DSv4 MTP draft candidate rows {} != depth {}",
            self.candidates.len(),
            self.depth
        );
        for (idx, node) in self.nodes.iter().enumerate().skip(1) {
            let parent = node
                .parent
                .ok_or_else(|| anyhow!("DSv4 MTP draft node {idx} has no parent"))?;
            ensure!(
                parent + 1 == idx,
                "DSv4 MTP draft chain node {idx} parent {parent} is not previous row"
            );
            ensure!(
                node.depth == self.nodes[parent].depth + 1,
                "DSv4 MTP draft node {idx} depth {} != parent depth {} + 1",
                node.depth,
                self.nodes[parent].depth
            );
        }
        for (row, candidates) in self.candidates.iter().enumerate() {
            ensure!(
                !candidates.is_empty(),
                "DSv4 MTP draft candidate row {row} is empty"
            );
            ensure!(
                candidates[0] == self.nodes[row + 1].token,
                "DSv4 MTP draft row {row} top1 {} != chain token {}",
                candidates[0],
                self.nodes[row + 1].token
            );
        }
        Ok(())
    }

    fn tokens(&self) -> Vec<u32> {
        self.nodes.iter().map(|node| node.token).collect()
    }

    fn accept_path(&self, argmax: &[u32]) -> Result<(Vec<usize>, u32, usize, bool)> {
        ensure!(
            argmax.len() == self.nodes.len(),
            "DSv4 MTP draft chain argmax rows {} != nodes {}",
            argmax.len(),
            self.nodes.len()
        );
        let mut path = vec![0usize];
        for (row, &target) in argmax.iter().take(self.depth).enumerate() {
            let topk_hit = self.candidates[row].contains(&target);
            if topk_hit && target == self.nodes[row + 1].token {
                path.push(row + 1);
                continue;
            }
            return Ok((path, target, row, topk_hit));
        }
        let bonus = *argmax
            .get(self.depth)
            .ok_or_else(|| anyhow!("DSv4 MTP draft chain missing bonus row {}", self.depth))?;
        Ok((path, bonus, self.depth, false))
    }

    fn accepted_tokens(&self, path: &[usize]) -> Vec<u32> {
        path.iter()
            .copied()
            .skip(1)
            .map(|row| self.nodes[row].token)
            .collect()
    }

    fn add_chain_child(&mut self, token: u32) -> Result<usize> {
        let parent = self.nodes.len() - 1;
        ensure!(
            self.nodes.len() < crate::dsv4::MAX_SPEC_VERIFY_ROWS,
            "DSv4 MTP draft chain exceeds {} verify rows; reduce --mtp-draft-tokens",
            crate::dsv4::MAX_SPEC_VERIFY_ROWS
        );
        let row = self.nodes.len();
        self.nodes.push(DraftNode {
            token,
            parent: Some(parent),
            depth: self.nodes[parent].depth + 1,
        });
        Ok(row)
    }
}

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

        // Frozen-KV P1-2: snapshot the ring slots the draft will overwrite
        // BEFORE any speculative write (the draft writes the frozen target
        // layer's SW/FP8 ring; the batched verify itself is pure).
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
            verify.argmax.len() == chain.nodes.len()
                && verify.hiddens.len() == chain.nodes.len()
                && verify.logits.seq_len == chain.nodes.len(),
            "DSv4 MTP verify expected {} rows, got argmax={} hidden={} logits={}",
            chain.nodes.len(),
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
                chain.nodes.len().saturating_sub(1),
                chain.nodes.len(),
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
        let mut chain = DraftChain {
            nodes: vec![DraftNode {
                token: pending,
                parent: None,
                depth: 0,
            }],
            candidates: Vec::with_capacity(depth),
            depth,
        };
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
            chain.candidates.push(candidates);
            chain.add_chain_child(next)?;
            chain_token = next;
            chain_hidden = Some(stream);
        }
        Ok(chain)
    }
}
