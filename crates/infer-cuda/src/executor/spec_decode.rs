//! DSv4 MTP speculative-decode orchestration.
//!
//! One code path for every draft shape. A [`DraftTree`] — `topk == 1` is a linear
//! chain, `topk >= 2` branches — is drafted off the MTP head, verified in a single
//! frozen forward, and the longest accepted root→leaf path is committed. This
//! module owns the tree topology, the pure accept walk, and the
//! [`SpecVerifySchedule`] that maps tree rows onto ring-slot fix-ups; it drives
//! the model's existing draft / verify / commit primitives (`mtp_forward`,
//! `forward_tokens_verify_scheduled`, the frozen-KV ring snapshot + node
//! scratch). It does **not** touch the forward kernels — the executor simply
//! delegates [`Dsv4CudaExecutor::spec_step`] here.
//!
//! Speculative decoding wins by accepting *multiple* tokens out of one verify
//! forward (the forward is weight-read-bound, so a 1-token and a whole-tree
//! forward cost ~the same). The lever is the accepted path length, which a
//! `topk >= 2` tree raises by offering several candidates per position.
//! `topk == 1` is the degenerate one-chain case and reproduces the validated
//! linear path exactly — zero fix-ups, byte-identical attention.
//!
//! Why fix-ups instead of a tree-mask kernel: under the frozen verify the
//! compressed/DSA side is pinned to the committed keys (identical for every
//! row), so tree topology only matters to the position-keyed SW ring
//! (`pos % sliding_window`), where same-depth siblings fight over one slot.
//! BFS row order plus parking/replaying each contested slot per ancestor path
//! makes every row attend to exactly its own branch — reusing the existing
//! decode attention kernels untouched.

use anyhow::{Result, anyhow, ensure};

use crate::dsv4::{MAX_SPEC_TREE_NODES, SpecVerifySchedule};

use super::{DeviceVec, Dsv4CudaExecutor};

/// The draft-tree shape for one step. `topk == 1` ⇒ a depth-`depth` linear chain
/// (the validated path). `topk >= 2` ⇒ a tree (every expansion branches into the
/// MTP head's top-`topk` candidates), capped at [`MAX_SPEC_TREE_NODES`] nodes.
pub(crate) struct SpecShape {
    pub depth: usize,
    pub topk: usize,
}

/// A flattened speculative draft tree in parent-pointer form. Node `0` is the root
/// (the already-committed `pending` token); nodes are pushed level by level
/// (BFS), so `parent[i] < i` and depths are non-decreasing in node order — the
/// invariant both the draft-phase and verify-phase ring fix-ups rely on. Verify
/// and accept treat a chain and a tree identically.
pub(crate) struct DraftTree {
    /// Per-node token; `tokens[0]` is `pending`.
    tokens: Vec<u32>,
    /// Per-node parent index; `parent[0] == 0` (root points at itself).
    parent: Vec<usize>,
    /// Per-node child indices.
    children: Vec<Vec<usize>>,
    /// Per-node draft depth; `depth[0] == 0`, child = parent + 1.
    depth: Vec<usize>,
}

impl DraftTree {
    /// A tree containing only the root (`pending`).
    fn new(pending: u32) -> Self {
        Self {
            tokens: vec![pending],
            parent: vec![0],
            children: vec![Vec::new()],
            depth: vec![0],
        }
    }

    /// Append `token` as a child of `parent_idx`; returns the new node index.
    fn push(&mut self, parent_idx: usize, token: u32) -> usize {
        let idx = self.tokens.len();
        self.tokens.push(token);
        self.parent.push(parent_idx);
        self.children.push(Vec::new());
        self.depth.push(self.depth[parent_idx] + 1);
        self.children[parent_idx].push(idx);
        idx
    }

    /// Flattened tokens in node order — exactly what one verify forward consumes
    /// (`tokens[0] = pending`, then every draft node).
    fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Node `i`'s token.
    fn token(&self, i: usize) -> u32 {
        self.tokens[i]
    }

    /// Node `i`'s parent index, or `None` for the root.
    fn parent(&self, i: usize) -> Option<usize> {
        (i != 0).then(|| self.parent[i])
    }

    /// Node `i`'s draft depth (root = 0).
    fn depth(&self, i: usize) -> usize {
        self.depth[i]
    }

    /// Total node count (root + drafts).
    fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Node `i`'s ancestors strictly between the root and `i` (depths
    /// `1..depth(i)`), shallow→deep — the nodes whose ring slots `i`'s
    /// sliding-window attention must see.
    fn branch_ancestors(&self, i: usize) -> Vec<usize> {
        let mut chain = Vec::new();
        let mut a = i;
        while let Some(p) = self.parent(a) {
            if p != 0 {
                chain.push(p);
            }
            a = p;
        }
        chain.reverse();
        chain
    }

    /// The verify-forward row schedule for this tree: per-row absolute
    /// positions plus the exact ring-slot fix-ups, from simulating the
    /// per-depth slot owner in row order. A row restores an ancestor exactly
    /// when a sibling overwrote that slot since the ancestor last held it; a
    /// row is parked exactly when some later row restores it. A chain owns
    /// every depth alone — zero fix-ups, the validated per-token verify.
    fn verify_schedule(&self, start_pos: usize) -> SpecVerifySchedule {
        let n = self.len();
        let positions: Vec<usize> = self.depth.iter().map(|d| start_pos + d).collect();
        let max_depth = self.depth.iter().copied().max().unwrap_or(0);
        let mut owner = vec![usize::MAX; max_depth + 1];
        let mut restores: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut saves = vec![false; n];
        for i in 0..n {
            for anc in self.branch_ancestors(i) {
                if owner[self.depth[anc]] != anc {
                    restores[i].push(anc);
                    saves[anc] = true;
                    owner[self.depth[anc]] = anc;
                }
            }
            owner[self.depth[i]] = i;
        }
        SpecVerifySchedule {
            positions,
            restores,
            saves,
        }
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
        // layer's SW/FP8 ring). Positions cover `start_pos..=start_pos+depth` — the
        // superset every tree node writes into. No-op when the spec-ring snapshot
        // is unallocated.
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

        // 4. Commit accepted drafts + the bonus correction. The ring restore
        //    must use the depth the snapshot CAPTURED (shape.depth), not a
        //    tree-derived count.
        self.commit_path(
            slot_idx,
            &tree,
            &path,
            &argmax,
            hiddens,
            start_pos,
            position,
            shape.depth,
        )
    }

    /// The draft shape for this step. `depth` honours `--mtp-draft-tokens` only
    /// under `ARLE_DSV4_MTP_UNCLAMP=1`; otherwise it clamps to 1 so a stray
    /// `--mtp-draft-tokens N` can never run an un-validated depth. `topk` comes
    /// from `ARLE_DSV4_MTP_TOPK` (experimental knob, default 1 — promoted to a
    /// CLI flag once the tree A/B licenses a default).
    fn spec_shape(&self) -> SpecShape {
        let requested = self.spec_draft_tokens.unwrap_or(1).max(1);
        let unclamp = std::env::var("ARLE_DSV4_MTP_UNCLAMP").as_deref() == Ok("1");
        let topk = std::env::var("ARLE_DSV4_MTP_TOPK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1)
            .max(1);
        SpecShape {
            depth: if unclamp { requested } else { 1 },
            topk,
        }
    }

    /// Draft the tree off the MTP head: branch each node into the head's
    /// top-`shape.topk` candidates, `shape.depth` levels deep, capped at
    /// [`MAX_SPEC_TREE_NODES`] nodes. Each node's wide MTP stream is the `h_prev`
    /// its children chain from, so every child conditions on its parent token.
    /// BFS by depth, so every node at depth `d` is drafted at the same absolute
    /// position `start_pos + d`.
    ///
    /// Expanding a node writes the MTP target layer's ring slot at its depth
    /// (`mtp_forward` → `mla_attention` append), so sibling expansions overwrite
    /// each other; the owner walk below parks a contested slot after its
    /// expansion and replays a node's ancestors before expanding it — one layer,
    /// host-precomputable, and exactly zero copies when `topk == 1`.
    fn draft_tree(
        &mut self,
        slot_idx: usize,
        pending: u32,
        trunk_hidden: &DeviceVec,
        shape: &SpecShape,
        start_pos: usize,
    ) -> Result<DraftTree> {
        let mut tree = DraftTree::new(pending);
        // Per-node wide MTP stream from expanding that node — its children's
        // `h_prev`. Indexed in lockstep with the tree's nodes. The borrow ends at
        // the `mtp_forward` call (NLL), so the later `node_stream[node] = …` write
        // and the per-child `push` don't conflict — and topk=1 needs no clone.
        let mut node_stream: Vec<Option<DeviceVec>> = vec![None];
        let mut frontier = vec![0usize];
        // Ring-slot owner per draft depth (MTP target layer only).
        let mut owner = vec![usize::MAX; shape.depth + 1];
        for depth in 0..shape.depth {
            let mut next = Vec::with_capacity(frontier.len() * shape.topk);
            // A lone frontier node owns its slot until its children expand —
            // park only contested depths.
            let contested = frontier.len() > 1;
            for node in frontier {
                if tree.len() >= MAX_SPEC_TREE_NODES {
                    break;
                }
                for anc in tree.branch_ancestors(node) {
                    if owner[tree.depth(anc)] != anc {
                        self.model.mtp_restore_node_ring(
                            &mut self.slots[slot_idx],
                            &mut self.kv_adapter,
                            anc,
                            start_pos + tree.depth(anc),
                        )?;
                        owner[tree.depth(anc)] = anc;
                    }
                }
                let token = tree.token(node);
                let h_prev: &DeviceVec = match tree.parent(node) {
                    Some(parent) => node_stream[parent].as_ref().expect("parent stream"),
                    None => trunk_hidden,
                };
                let (candidates, stream) = self.model.mtp_forward(
                    &mut self.slots[slot_idx],
                    &mut self.kv_adapter,
                    h_prev,
                    token,
                    (start_pos + depth) as u64,
                    shape.topk,
                )?;
                owner[depth] = node;
                if contested {
                    self.model.mtp_save_node_ring(
                        &mut self.slots[slot_idx],
                        &mut self.kv_adapter,
                        node,
                        start_pos + depth,
                    )?;
                }
                node_stream[node] = Some(stream);
                for &candidate in &candidates {
                    if tree.len() >= MAX_SPEC_TREE_NODES {
                        break;
                    }
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

    /// Verify the whole tree in ONE forward with the compressor FROZEN, so the
    /// speculative verify mutates nothing compressed. The tree's
    /// [`SpecVerifySchedule`] gives every row its depth position and the ring
    /// fix-ups that make its SW attention see exactly its ancestor branch.
    /// Returns each node's target argmax (`argmax[i]` = argmax after
    /// `tokens[i]`) and its MTP stream hidden.
    fn verify_tree(
        &mut self,
        slot_idx: usize,
        tree: &DraftTree,
        start_pos: usize,
        position: u64,
    ) -> Result<(Vec<u32>, Vec<DeviceVec>)> {
        let sched = tree.verify_schedule(start_pos);
        crate::attention::set_dsv4_verify_frozen(true);
        let res = self.model.forward_tokens_verify_scheduled(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            tree.tokens(),
            start_pos,
            position,
            &sched,
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
    /// rejected ring positions the frozen verify wrote, then re-forward
    /// `[pending, accepted…]` NON-frozen to commit the compressor for exactly the
    /// accepted tokens and capture the next hidden. Returns the committed tokens.
    ///
    /// Trees need no extra restore work: every speculative write (any branch)
    /// landed in ring positions `start_pos..=start_pos+depth`. The re-forward
    /// overwrites `[start_pos ..= start_pos+accepted]` and
    /// `restore_spec_ring_tail` replays the committed contents of the rest —
    /// non-path nodes only ever lived inside that window.
    #[allow(clippy::too_many_arguments)]
    fn commit_path(
        &mut self,
        slot_idx: usize,
        tree: &DraftTree,
        path: &[usize],
        argmax: &[u32],
        _hiddens: Vec<DeviceVec>,
        start_pos: usize,
        position: u64,
        depth: usize,
    ) -> Result<Vec<u32>> {
        let drafts = tree.len() - 1;
        let accepted = path.len();
        // `argmax` at the last accepted node (root if nothing accepted) is the
        // target's correction / continuation = the next pending token.
        let bonus = argmax[path.last().copied().unwrap_or(0)];

        self.mtp_accepts += accepted;
        self.mtp_rejects += drafts - accepted;
        if self.model.tp.config().rank == 0 {
            eprintln!(
                "[dsv4-mtp] depth={depth} nodes={drafts} accepted={accepted} accept_total={} reject_total={} bonus={bonus}",
                self.mtp_accepts, self.mtp_rejects
            );
            // P0 accept-rate probe (fast-path plan): per-step (target, level-1
            // drafts) token-id pairs, detokenized offline. Near-miss rejects =
            // head quality; nonsense rejects = draft-path bug.
            if std::env::var("ARLE_DSV4_MTP_PROBE").as_deref() == Ok("1") {
                let drafts_l1: Vec<u32> =
                    tree.children[0].iter().map(|&c| tree.tokens()[c]).collect();
                eprintln!(
                    "[dsv4-mtp-probe] pending={} target={} drafts_l1={:?}",
                    tree.tokens()[0],
                    argmax[0],
                    drafts_l1
                );
            }
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

    /// A chain schedule has strictly increasing positions and ZERO fix-ups —
    /// the validated per-token verify is untouched.
    #[test]
    fn schedule_chain_no_fixups() {
        let mut tree = DraftTree::new(10);
        let mut leaf = 0;
        for &t in &[11, 12, 13] {
            leaf = tree.push(leaf, t);
        }
        let sched = tree.verify_schedule(100);
        assert_eq!(sched.positions, vec![100, 101, 102, 103]);
        assert!(sched.restores.iter().all(|r| r.is_empty()));
        assert!(sched.saves.iter().all(|&s| !s));
        assert!(sched.is_chain());
    }

    /// A full topk=2 depth=2 tree: BFS rows share positions per depth; each
    /// deeper row replays exactly the ancestors a sibling's write displaced,
    /// and every restored source row is parked.
    #[test]
    fn schedule_tree_fixups() {
        // root=0; depth1: a=1, b=2; depth2: a's c,d=3,4; b's e,f=5,6.
        let mut tree = DraftTree::new(10);
        let a = tree.push(0, 11);
        let b = tree.push(0, 21);
        let c = tree.push(a, 12);
        let d = tree.push(a, 13);
        let e = tree.push(b, 22);
        let f = tree.push(b, 23);
        let sched = tree.verify_schedule(100);
        assert_eq!(sched.positions, vec![100, 101, 101, 102, 102, 102, 102]);
        assert!(!sched.is_chain());
        // Rows 0 (root), 1 (a), 2 (b): no ancestors below the root → no restores.
        assert!(sched.restores[0].is_empty());
        assert!(sched.restores[a].is_empty());
        assert!(sched.restores[b].is_empty());
        // Row c: slot depth-1 holds b (last depth-1 row) → replay a.
        assert_eq!(sched.restores[c], vec![a]);
        // Row d: a already replayed by c and undisturbed since → nothing.
        assert!(sched.restores[d].is_empty());
        // Row e: slot depth-1 holds a → replay b; row f: undisturbed.
        assert_eq!(sched.restores[e], vec![b]);
        assert!(sched.restores[f].is_empty());
        // Parked exactly the restored sources: a and b.
        let parked: Vec<usize> = (0..tree.len()).filter(|&i| sched.saves[i]).collect();
        assert_eq!(parked, vec![a, b]);
    }

    /// Capacity cap: pushes beyond MAX_SPEC_TREE_NODES would be the draft
    /// loop's job to refuse; the schedule itself handles ragged trees (a
    /// truncated frontier leaves childless internal-depth nodes).
    #[test]
    fn schedule_ragged_tree() {
        // root; depth1: a,b; only b expands at depth2.
        let mut tree = DraftTree::new(10);
        let a = tree.push(0, 11);
        let b = tree.push(0, 21);
        let g = tree.push(b, 22);
        let sched = tree.verify_schedule(50);
        assert_eq!(sched.positions, vec![50, 51, 51, 52]);
        // g chains through b, which still owns slot depth-1 (it wrote last).
        assert!(sched.restores[g].is_empty());
        assert!(!sched.saves[a] && !sched.saves[b]);
        // Zero fix-ups, but the repeated depth-1 position still rules out the
        // per-row chain fallback (it would re-derive positions as start_pos+r).
        assert!(!sched.is_chain());
    }
}
