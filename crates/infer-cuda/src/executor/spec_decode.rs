//! DSv4 MTP speculative-decode orchestration.
//!
//! One code path for every draft shape. A [`DraftTree`] — `topk == 1` is a linear
//! chain, `topk >= 2` branches — is drafted off the MTP head, verified in a single
//! frozen forward, and the longest accepted root→leaf path is committed. This
//! module owns the tree topology and the pure accept walk, and drives the model's
//! existing draft / verify / commit primitives (`mtp_forward`,
//! `forward_tokens_verify`, the frozen-KV ring snapshot). It does **not** touch the
//! forward kernels, the frozen-KV path, or the ring-snapshot internals — the
//! executor simply delegates [`Dsv4CudaExecutor::spec_step`] here.
//!
//! Speculative decoding wins by accepting *multiple* tokens out of one verify
//! forward (the forward is weight-read-bound, so a 1-token and a whole-tree forward
//! cost ~the same). The lever is the accepted path length, which a `topk >= 2` tree
//! raises by offering several candidates per position. `topk == 1` is the
//! degenerate one-chain case and reproduces the validated linear path exactly.

use anyhow::{Result, anyhow, ensure};

use super::{DeviceVec, Dsv4CudaExecutor};

/// The draft-tree shape for one step. `topk == 1` ⇒ a depth-`depth` linear chain
/// (the validated path). `topk >= 2` ⇒ a tree, which needs the tree-mask verify
/// kernel (not yet wired — see [`Dsv4CudaExecutor::draft_tree`]).
pub(crate) struct SpecShape {
    pub depth: usize,
    pub topk: usize,
}

/// A flattened speculative draft tree in parent-pointer form. Node `0` is the root
/// (the already-committed `pending` token); nodes are pushed in build order, so
/// `parent[i] < i`. Verify and accept treat a chain and a tree identically.
pub(crate) struct DraftTree {
    /// Per-node token; `tokens[0]` is `pending`.
    tokens: Vec<u32>,
    /// Per-node parent index; `parent[0] == 0` (root points at itself).
    parent: Vec<usize>,
    /// Per-node child indices.
    children: Vec<Vec<usize>>,
}

impl DraftTree {
    /// A tree containing only the root (`pending`).
    fn new(pending: u32) -> Self {
        Self {
            tokens: vec![pending],
            parent: vec![0],
            children: vec![Vec::new()],
        }
    }

    /// Append `token` as a child of `parent_idx`; returns the new node index.
    fn push(&mut self, parent_idx: usize, token: u32) -> usize {
        let idx = self.tokens.len();
        self.tokens.push(token);
        self.parent.push(parent_idx);
        self.children.push(Vec::new());
        self.children[parent_idx].push(idx);
        idx
    }

    /// Flattened tokens in node order — exactly what one verify forward consumes
    /// (`tokens[0] = pending`, then every draft node).
    fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Total node count (root + drafts).
    fn len(&self) -> usize {
        self.tokens.len()
    }
}

/// The longest root→leaf path whose every node's token equals its parent's verify
/// argmax. `argmax[i]` is the target's greedy argmax *after* node `i`'s token, i.e.
/// what node `i`'s accepted child must equal. Returns the accepted node indices,
/// root excluded. Pure — no device state — so it is unit-testable in isolation.
///
/// For a chain this is the classic "longest accepted prefix"; for a tree it is the
/// longest matching branch.
fn longest_accepted_path(tree: &DraftTree, argmax: &[u32]) -> Vec<usize> {
    let mut path = Vec::new();
    let mut cur = 0;
    loop {
        let want = argmax[cur];
        match tree.children[cur]
            .iter()
            .copied()
            .find(|&c| tree.tokens[c] == want)
        {
            Some(child) => {
                path.push(child);
                cur = child;
            }
            None => break,
        }
    }
    path
}

impl Dsv4CudaExecutor {
    /// One speculative decode step: draft a tree, verify it in a single frozen
    /// forward, accept the longest matching path, and commit it. Returns the
    /// committed tokens (accepted drafts + the bonus correction) and advances the
    /// per-slot spec state (`pending` / `hidden`).
    pub(crate) fn spec_step(
        &mut self,
        slot_idx: usize,
        start_pos: usize,
        position: u64,
    ) -> Result<Vec<u32>> {
        let shape = self.spec_shape();
        let pending = self.spec_slots[slot_idx]
            .pending
            .ok_or_else(|| anyhow!("DSv4 MTP decode missing pending token"))?;
        let hidden = self.spec_slots[slot_idx]
            .hidden
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 MTP decode missing previous hidden"))?
            .clone();

        // Frozen-KV P1-2: snapshot the ring slots the draft + verify will overwrite
        // BEFORE any speculative write (the draft itself writes the frozen target
        // layer's SW/FP8 ring). No-op when the spec-ring snapshot is unallocated.
        self.model.capture_spec_rings(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            start_pos,
            shape.depth,
        )?;

        // 1. Draft the tree off the MTP head.
        let tree = self.draft_tree(slot_idx, pending, &hidden, &shape, start_pos)?;

        // 2. Verify the whole tree in ONE frozen forward.
        let (argmax, hiddens) = self.verify_tree(slot_idx, &tree, start_pos, position)?;

        // 3. Accept the longest matching path.
        let path = longest_accepted_path(&tree, &argmax);

        // 4. Commit accepted drafts + the bonus correction.
        self.commit_path(
            slot_idx, &tree, &path, &argmax, hiddens, start_pos, position,
        )
    }

    /// The draft shape for this step. `depth` honours `--mtp-draft-tokens` only
    /// under `ARLE_DSV4_MTP_UNCLAMP=1`; otherwise it clamps to 1 so a stray
    /// `--mtp-draft-tokens N` can never run an un-validated depth. `topk` is 1 until
    /// the tree-mask verify kernel lands.
    fn spec_shape(&self) -> SpecShape {
        let requested = self.spec_draft_tokens.unwrap_or(1).max(1);
        let unclamp = std::env::var("ARLE_DSV4_MTP_UNCLAMP").as_deref() == Ok("1");
        SpecShape {
            depth: if unclamp { requested } else { 1 },
            topk: 1,
        }
    }

    /// Draft the tree off the MTP head. `topk == 1` chains each draft off the
    /// previous draft's wide MTP stream (`d_{i+1}` conditions on `d_i`). `topk >= 2`
    /// is not yet wired — it needs a top-k expansion of the head logits and the
    /// tree-mask verify; the [`DraftTree`] shape is already general for it.
    fn draft_tree(
        &mut self,
        slot_idx: usize,
        pending: u32,
        trunk_hidden: &DeviceVec,
        shape: &SpecShape,
        start_pos: usize,
    ) -> Result<DraftTree> {
        ensure!(
            shape.topk == 1,
            "DSv4 MTP tree draft (topk={}) not yet wired — needs top-k head expansion + the tree-mask verify kernel",
            shape.topk
        );
        let mut tree = DraftTree::new(pending);
        let mut leaf = 0;
        let mut chain_token = pending;
        let mut prev_stream: Option<DeviceVec> = None;
        for i in 0..shape.depth {
            let h_prev = prev_stream.as_ref().unwrap_or(trunk_hidden);
            let (draft, draft_stream) = self.model.mtp_forward(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                h_prev,
                chain_token,
                (start_pos + i) as u64,
            )?;
            leaf = tree.push(leaf, draft);
            chain_token = draft;
            prev_stream = Some(draft_stream);
        }
        Ok(tree)
    }

    /// Verify the whole tree in ONE forward with the compressor FROZEN, so the
    /// speculative verify mutates nothing compressed. Returns each node's target
    /// argmax (`argmax[i]` = argmax after `tokens[i]`) and its MTP stream hidden.
    /// `topk == 1` flattens to a causal sequence — the validated linear verify; a
    /// tree needs the tree mask in the verify forward (future).
    fn verify_tree(
        &mut self,
        slot_idx: usize,
        tree: &DraftTree,
        start_pos: usize,
        position: u64,
    ) -> Result<(Vec<u32>, Vec<DeviceVec>)> {
        crate::attention::set_dsv4_verify_frozen(true);
        let res = self.model.forward_tokens_verify(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            tree.tokens(),
            start_pos,
            position,
        );
        crate::attention::set_dsv4_verify_frozen(false);
        let (argmax, hiddens) = res?;
        ensure!(
            argmax.len() == tree.len() && hiddens.len() == tree.len(),
            "DSv4 MTP verify expected {} rows, got argmax={} hidden={}",
            tree.len(),
            argmax.len(),
            hiddens.len()
        );
        Ok((argmax, hiddens))
    }

    /// Commit the accepted path: truncate back to the pre-step length, restore the
    /// rejected ring tail the frozen verify wrote, then re-forward
    /// `[pending, accepted…]` NON-frozen to commit the compressor for exactly the
    /// accepted tokens and capture the next hidden. Returns the committed tokens.
    ///
    /// `topk == 1` only: the rejected nodes are the chain tail `[n+1 ..= depth]`, so
    /// the existing `restore_spec_ring_tail` covers them. A tree's rejected set is
    /// the non-path nodes; generalising the restore is the tree follow-up.
    fn commit_path(
        &mut self,
        slot_idx: usize,
        tree: &DraftTree,
        path: &[usize],
        argmax: &[u32],
        _hiddens: Vec<DeviceVec>,
        start_pos: usize,
        position: u64,
    ) -> Result<Vec<u32>> {
        let depth = tree.len() - 1;
        let accepted = path.len();
        // `argmax` at the last accepted node (root if nothing accepted) is the
        // target's correction / continuation = the next pending token.
        let bonus = argmax[path.last().copied().unwrap_or(0)];

        self.mtp_accepts += accepted;
        self.mtp_rejects += depth - accepted;
        if self.model.tp.config().rank == 0 {
            eprintln!(
                "[dsv4-mtp] depth={depth} accepted={accepted}/{depth} accept_total={} reject_total={} bonus={bonus}",
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

        let accepted_tokens: Vec<u32> = path.iter().map(|&i| tree.tokens()[i]).collect();
        let mut prefix = Vec::with_capacity(accepted + 1);
        prefix.push(tree.tokens()[0]);
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

        let mut out = accepted_tokens;
        out.push(bonus);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{DraftTree, longest_accepted_path};

    /// A linear chain: accept the longest matching prefix.
    #[test]
    fn chain_longest_prefix() {
        // pending=10, drafts d0=11 d1=12 d2=13 (chain).
        let mut tree = DraftTree::new(10);
        let mut leaf = 0;
        for &t in &[11, 12, 13] {
            leaf = tree.push(leaf, t);
        }
        // argmax after each node. d0=11 matches argmax[0]=11; d1=12 matches
        // argmax[1]=12; d2=13 does NOT match argmax[2]=99 → stop after 2.
        let argmax = vec![11, 12, 99, 0];
        let path = longest_accepted_path(&tree, &argmax);
        assert_eq!(path, vec![1, 2]); // nodes d0, d1
    }

    /// Nothing accepted when the first draft mismatches.
    #[test]
    fn chain_reject_first() {
        let mut tree = DraftTree::new(10);
        tree.push(0, 11);
        let argmax = vec![99, 0];
        assert!(longest_accepted_path(&tree, &argmax).is_empty());
    }

    /// A topk=2 tree: accept whichever branch matches, longest.
    #[test]
    fn tree_longest_branch() {
        // root=10; children a=11, b=21; a's children=12; b's children=22,23.
        let mut tree = DraftTree::new(10);
        let a = tree.push(0, 11);
        let b = tree.push(0, 21);
        tree.push(a, 12);
        tree.push(b, 22);
        let b3 = tree.push(b, 23);
        // argmax[root]=21 → take b; argmax[b]=23 → take b's child 23.
        let mut argmax = vec![0u32; tree.len()];
        argmax[0] = 21; // root → b
        argmax[b] = 23; // b → node 23
        let path = longest_accepted_path(&tree, &argmax);
        assert_eq!(path, vec![b, b3]);
    }
}
