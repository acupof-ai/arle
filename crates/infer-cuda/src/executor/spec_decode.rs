//! DSv4 MTP speculative-decode orchestration.
//!
//! MTP drafts a top-1 chain, records top-k candidates at each draft row, verifies
//! the chain in one target pass, then matches target top-1 against those
//! candidates. `topk` does not add verify rows on this path.

use anyhow::{Result, anyhow, ensure};

use crate::dsv4::SpecVerifySchedule;

use super::{DeviceVec, Dsv4CudaExecutor};

/// Which speculative-decode scheme a serve is configured for. Both CUDA
/// executors resolve this from their own state (`dspark`/`mtp` handles).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SpecKind {
    None,
    Mtp,
    Dspark,
}

/// The decode path a batch of `n_rows` should take this tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DecodeRoute {
    /// Plain batched (or single-row) decode: 1 token/row, scales with batch.
    Plain,
    /// MTP speculative decode — c=1 / low-concurrency win.
    Mtp,
    /// DSpark speculative decode — c=1 / low-concurrency win.
    Dspark,
}

/// Speculate only at or below the concurrency gate. At small batch the GPU is
/// memory-bound and the B+1 verify positions are ~free, so speculation wins;
/// above the gate the target forward is compute-bound and the same verify costs
/// ~(B+1)× step time for ~2.5 committed tokens, a net loss — so fall back to the
/// plain batched path that scales. `gate` is `--spec-max-batch` (default 1).
/// Pure so the routing is unit-tested without a GPU.
///
/// `any_penalty` vetoes speculation: the draft and verify lanes commit tokens
/// from device argmax / a rejection draw over raw target logits, neither of
/// which sees the host-side repetition/frequency/presence penalties.
pub(super) fn route_decode(
    spec_kind: SpecKind,
    n_rows: usize,
    gate: usize,
    any_penalty: bool,
) -> DecodeRoute {
    if any_penalty || n_rows > gate {
        return DecodeRoute::Plain;
    }
    match spec_kind {
        SpecKind::Dspark => DecodeRoute::Dspark,
        SpecKind::Mtp => DecodeRoute::Mtp,
        SpecKind::None => DecodeRoute::Plain,
    }
}

/// Adaptive MTP gate (B=1), opt-in via `--mtp-adaptive`. MTP only beats
/// no-spec when it emits more than `t_mtp/t_nospec` tok/step; below the
/// matching acceptance rate the gate runs a warm no-spec step instead so
/// typical prompts stop paying the speculation tax.
///
/// Minimum running accept-rate EMA to keep speculating. Default 0.55 = the dt=3
/// break-even on 8xH20 TP4 (t_mtp ~68ms / t_nospec ~26ms => need >2.6 tok/step =>
/// accept >~0.55). Override with `--mtp-min-accept` for other depths.
/// ponytail: a fixed depth-tuned threshold; upgrade path is to self-calibrate
/// from measured step times.
///
/// Force one real spec step after this many consecutive gated skips, to refresh
/// the acceptance EMA — else a dip below threshold never recovers (no new accept
/// data arrives while skipping).
const MTP_PROBE_INTERVAL: usize = 8;

/// EMA smoothing for the per-step accept rate (accepted/depth): higher reacts
/// faster to an acceptance shift, noisier.
const MTP_ACCEPT_EMA_ALPHA: f32 = 0.25;

/// Pure gate decision: speculate iff running acceptance clears the break-even
/// threshold, OR a probe is due (force one step to refresh the EMA). Pure so the
/// money path is unit-tested without a GPU.
fn mtp_should_speculate(
    accept_ema: f32,
    skip_streak: usize,
    min_accept: f32,
    probe_interval: usize,
) -> bool {
    accept_ema >= min_accept || skip_streak >= probe_interval
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

    fn new(root_token: u32, depth: usize) -> Self {
        Self {
            nodes: vec![DraftNode {
                token: root_token,
                parent: None,
                depth: 0,
            }],
            candidates: Vec::with_capacity(depth),
            depth,
        }
    }

    fn add_level(&mut self, candidates: Vec<u32>) -> Result<u32> {
        ensure!(
            !candidates.is_empty(),
            "DSv4 MTP draft level produced no candidates"
        );
        let next = candidates[0];
        self.candidates.push(candidates);
        self.add_chain_child(next)?;
        Ok(next)
    }
}

impl Dsv4CudaExecutor {
    /// One speculative decode step: draft a top-1 chain, verify it in a single
    /// frozen forward, accept the matching chain prefix, commit. Returns the
    /// committed tokens (accepted drafts + the bonus) and advances the per-slot
    /// spec state (`pending` / `hidden`).
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

        // 1. Draft only the top-1 chain. `topk` samples extra candidates from
        // each existing draft logits row; siblings are verify-only candidates,
        // not additional MTP forwards.
        let chain = self.draft_chain(slot_idx, pending, &hidden, depth, topk, start_pos)?;
        chain.validate()?;

        // 2. Verify the whole chain in ONE frozen target forward.
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

        // 3. Walk the verified chain logits. Target top-1 must match the chain
        // token to extend the committed prefix. A non-chain top-k hit is still a
        // valid bonus token, but the path stops at its parent because no later
        // chain row was conditioned on that token.
        let (path, bonus, bonus_parent_row, topk_bonus_hit) = chain.accept_path(&verify.argmax)?;
        let accepted = path.len() - 1;

        self.mtp_accepts += accepted;
        self.mtp_rejects += depth - accepted;
        self.mtp_chains += 1;
        self.mtp_note_accept(accepted, depth);
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

        // 4. Commit: truncate, restore the rejected ring tail (draft layer-0
        //    writes), then fold from the persisted verify rows.
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

    /// Cross-slot batched MTP decode step. Each slot drafts a top-1 chain; all
    /// chain rows are verified in one batched target pass.
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
    ///   accepted chain row's MTP stream hidden.
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
        // Ring capture is per-slot (cheap host snapshot, never batched). Draft
        // runs `depth` batched `mtp_forward_level` calls, one per level.
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
        // Batched draft: `depth` levels, each level runs ONE `mtp_forward_level`
        // over all N slots (one row per slot) instead of N×depth serial m=1 calls.
        let mut chains: Vec<DraftChain> = (0..n)
            .map(|s| DraftChain::new(pendings[s], depth))
            .collect();
        let mut cur_tokens: Vec<u32> = pendings.clone();
        let mut cur_hiddens: Vec<DeviceVec> = h_prevs.clone();
        for level in 0..depth {
            let rows: Vec<crate::dsv4::MtpDraftRow> = cur_tokens
                .iter()
                .map(|&t| crate::dsv4::MtpDraftRow { token: t })
                .collect();
            let h_refs: Vec<&DeviceVec> = cur_hiddens.iter().collect();
            let positions: Vec<u64> = (0..n)
                .map(|s| (start_positions[s] + level) as u64)
                .collect();
            let expanded = self.model.mtp_forward_level(
                &mut self.slots,
                &mut self.kv_adapter,
                slot_ids,
                &rows,
                &h_refs,
                &positions,
                topk,
            )?;
            for s in 0..n {
                let (candidates, stream) = expanded[s].clone();
                let next = chains[s].add_level(candidates)?;
                cur_tokens[s] = next;
                cur_hiddens[s] = stream;
            }
        }
        for s in 0..n {
            chains[s].validate()?;
            scheds.push(chains[s].verify_schedule(start_positions[s]));
        }

        // ── 2. ONE batched verify over the N chains (MoE grouped over all rows,
        // attention currently runs once per slot chunk, not once per row). The
        // verify persists per-slot spec_normed for the commit fold.
        let chain_tokens: Vec<Vec<u32>> = chains.iter().map(DraftChain::tokens).collect();
        let (verified, _verify_logits) = self.model.forward_decode_batch_verify(
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
            ensure!(
                argmax.len() == chain.nodes.len() && hiddens.len() == chain.nodes.len(),
                "DSv4 batched verify slot {slot_idx} expected {} rows, got argmax={} hidden={}",
                chain.nodes.len(),
                argmax.len(),
                hiddens.len()
            );

            let (path, bonus, bonus_parent_row, topk_bonus_hit) = chain.accept_path(&argmax)?;
            let accepted = path.len() - 1;
            self.mtp_accepts += accepted;
            self.mtp_rejects += depth - accepted;
            self.mtp_chains += 1;
            if self.model.tp.config().rank == 0 {
                log::debug!(
                    "[dsv4-mtp-batched] slot={slot_idx} depth={depth} topk={topk} \
                     draft_rows={} verify_rows={} accepted={accepted} \
                     topk_bonus_hit={topk_bonus_hit} accept_total={} reject_total={} bonus={bonus}",
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
            let spec = &mut self.spec_slots[slot_idx];
            spec.pending = Some(bonus);
            spec.hidden = Some(hiddens.swap_remove(bonus_parent_row));

            let mut slot_out = chain.accepted_tokens(&path);
            slot_out.push(bonus);
            out.push(slot_out);
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
        self.spec_draft_tokens.is_some() || self.spec_draft_topk.is_some() || self.dspark.is_some()
    }

    /// Fold one spec step's acceptance (accepted/`depth`) into the running EMA and
    /// clear the skip streak. Drives the adaptive gate; B=1 only — the batched
    /// path is a win and is never gated, so it does not perturb this EMA.
    fn mtp_note_accept(&mut self, accepted: usize, depth: usize) {
        let rate = if depth > 0 {
            accepted as f32 / depth as f32
        } else {
            1.0
        };
        self.mtp_accept_ema =
            MTP_ACCEPT_EMA_ALPHA * rate + (1.0 - MTP_ACCEPT_EMA_ALPHA) * self.mtp_accept_ema;
        self.mtp_skip_streak = 0;
    }

    /// Adaptive gate (B=1): true when MTP should be skipped for a warm no-spec
    /// step this decode. Off unless `--mtp-adaptive` is set.
    pub(super) fn mtp_adaptive_skip(&self) -> bool {
        crate::runtime_flags::mtp_adaptive()
            && !mtp_should_speculate(
                self.mtp_accept_ema,
                self.mtp_skip_streak,
                crate::runtime_flags::mtp_min_accept(),
                MTP_PROBE_INTERVAL,
            )
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
