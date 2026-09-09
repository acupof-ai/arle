//! Step-level speculative-decode scheduling: route selection, draft-chain
//! accept/rollback, and the verify-row schedule. Backend-neutral — the CUDA
//! qwen35 and dsv4 executors both route through `route_decode`.

use anyhow::{Result, anyhow, ensure};

/// Both CUDA executors resolve this from their own state (`dspark`/`mtp`
/// handles).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecKind {
    None,
    Mtp,
    Dspark,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeRoute {
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
///
/// `vetoed` covers request features the selected speculative implementation
/// cannot apply to every accepted token in a chain.
pub fn route_decode(spec_kind: SpecKind, n_rows: usize, gate: usize, vetoed: bool) -> DecodeRoute {
    if vetoed || n_rows > gate {
        return DecodeRoute::Plain;
    }
    match spec_kind {
        SpecKind::Dspark => DecodeRoute::Dspark,
        SpecKind::Mtp => DecodeRoute::Mtp,
        SpecKind::None => DecodeRoute::Plain,
    }
}

/// `topk` widens candidate matching only; verifier rows remain chain-shaped.
pub const MAX_SPEC_DRAFT_DEPTH: usize = 8;
/// Bounded chain verifier rows per slot. MTP uses `depth + 1`; `topk` adds none.
pub const MAX_SPEC_VERIFY_ROWS: usize = 64;
pub const DEFAULT_SPEC_DRAFT_DEPTH: usize = 2;

pub const DEFAULT_SPEC_DRAFT_TOPK: usize = 1;

/// Row schedule for one speculative verify forward. `ancestors` is the prefix
/// metadata the batched FlashMLA sparse verify lane reads: row `r` attends
/// committed KV plus the listed earlier chunk rows and self.
pub struct SpecVerifySchedule {
    /// Per row: absolute position (`start_pos + node depth`).
    pub positions: Vec<usize>,
    /// Per row: chunk-row ancestors, shallow to deep, self excluded.
    pub ancestors: Vec<Vec<usize>>,
}

impl SpecVerifySchedule {
    pub fn validate_sparse_at(&self, start_pos: usize) -> Result<()> {
        ensure!(
            !self.positions.is_empty() && self.positions.len() == self.ancestors.len(),
            "DSv4 sparse verify schedule shape mismatch: positions={} ancestors={}",
            self.positions.len(),
            self.ancestors.len()
        );
        ensure!(
            self.positions.len() <= MAX_SPEC_VERIFY_ROWS,
            "DSv4 sparse verify rows {} exceed fold cache rows {MAX_SPEC_VERIFY_ROWS}",
            self.positions.len()
        );
        for (row, &pos) in self.positions.iter().enumerate() {
            ensure!(
                pos >= start_pos,
                "DSv4 sparse verify row {row} position {pos} precedes start_pos {start_pos}"
            );
            for &ancestor in &self.ancestors[row] {
                ensure!(
                    ancestor < row,
                    "DSv4 sparse verify row {row} has non-causal ancestor row {ancestor}"
                );
                ensure!(
                    self.positions[ancestor] < pos,
                    "DSv4 sparse verify row {row} position {pos} ancestor {ancestor} position {} is not earlier",
                    self.positions[ancestor]
                );
            }
        }
        Ok(())
    }
}

/// One MTP draft row's token.
pub struct MtpDraftRow {
    pub token: u32,
}

struct DraftNode {
    token: u32,
    parent: Option<usize>,
    depth: usize,
}

/// A top-1 draft chain plus the top-k candidates recorded at each draft row.
/// `accept_path` is the accept/rollback state machine: the target's argmax
/// matches the chain until the first miss, and everything after that row rolls
/// back.
pub struct DraftChain {
    nodes: Vec<DraftNode>,
    candidates: Vec<Vec<u32>>,
    depth: usize,
}

impl DraftChain {
    /// Root the chain at the pending token; `depth` children are appended with
    /// [`add_chain_child`](Self::add_chain_child).
    pub fn start(token: u32, depth: usize) -> Self {
        Self {
            nodes: vec![DraftNode {
                token,
                parent: None,
                depth: 0,
            }],
            candidates: Vec::with_capacity(depth),
            depth,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Candidates for draft row `row` (row 0 = the first draft level), highest
    /// first; `candidates[0]` must equal the chain token appended at that row.
    pub fn push_candidates(&mut self, candidates: Vec<u32>) {
        self.candidates.push(candidates);
    }

    pub fn verify_schedule(&self, start_pos: usize) -> SpecVerifySchedule {
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

    pub fn validate(&self) -> Result<()> {
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

    pub fn tokens(&self) -> Vec<u32> {
        self.nodes.iter().map(|node| node.token).collect()
    }

    /// Match the target's argmax against the chain. Returns the accepted row
    /// path (always starting at the root), the bonus token, the bonus's parent
    /// row, and whether the bonus was a non-chain top-k hit. A non-chain top-k
    /// hit is still a valid bonus token, but the path stops at its parent
    /// because no later chain row was conditioned on that token.
    pub fn accept_path(&self, argmax: &[u32]) -> Result<(Vec<usize>, u32, usize, bool)> {
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

    pub fn accepted_tokens(&self, path: &[usize]) -> Vec<u32> {
        path.iter()
            .copied()
            .skip(1)
            .map(|row| self.nodes[row].token)
            .collect()
    }

    pub fn add_chain_child(&mut self, token: u32) -> Result<usize> {
        let parent = self.nodes.len() - 1;
        ensure!(
            self.nodes.len() < MAX_SPEC_VERIFY_ROWS,
            "DSv4 MTP draft chain exceeds {} verify rows; reduce --mtp-draft-tokens",
            MAX_SPEC_VERIFY_ROWS
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

#[cfg(test)]
mod tests {
    use super::*;

    fn chain_depth_2() -> DraftChain {
        let mut chain = DraftChain::start(10, 2);
        chain.push_candidates(vec![11, 99]);
        chain.add_chain_child(11).unwrap();
        chain.push_candidates(vec![12, 98]);
        chain.add_chain_child(12).unwrap();
        chain
    }

    #[test]
    fn route_decode_respects_gate_and_veto() {
        assert_eq!(route_decode(SpecKind::Mtp, 1, 1, false), DecodeRoute::Mtp);
        assert_eq!(
            route_decode(SpecKind::Dspark, 1, 1, false),
            DecodeRoute::Dspark
        );
        assert_eq!(
            route_decode(SpecKind::None, 1, 1, false),
            DecodeRoute::Plain
        );
        assert_eq!(route_decode(SpecKind::Mtp, 2, 1, false), DecodeRoute::Plain);
        assert_eq!(route_decode(SpecKind::Mtp, 1, 1, true), DecodeRoute::Plain);
    }

    #[test]
    fn draft_chain_accepts_full_path_with_bonus() {
        let chain = chain_depth_2();
        chain.validate().unwrap();
        // argmax[row] matches draft row `row`; the last row is the bonus.
        let (path, bonus, parent_row, topk_hit) = chain.accept_path(&[11, 12, 13]).unwrap();
        assert_eq!(path, vec![0, 1, 2]);
        assert_eq!(bonus, 13);
        assert_eq!(parent_row, 2);
        assert!(!topk_hit);
        assert_eq!(chain.accepted_tokens(&path), vec![11, 12]);
        assert_eq!(chain.tokens(), vec![10, 11, 12]);
    }

    /// The rollback gate: the path stops at the first row whose target is not
    /// the chain token, no matter what later rows carry.
    #[test]
    fn draft_chain_rolls_back_at_first_mismatch() {
        let chain = chain_depth_2();
        // Top-k sibling hit without chain match: path stops at the root, the
        // sibling is still a valid bonus token.
        let (path, bonus, row, topk_hit) = chain.accept_path(&[99, 12, 13]).unwrap();
        assert_eq!(path, vec![0]);
        assert_eq!(bonus, 99);
        assert_eq!(row, 0);
        assert!(topk_hit);
        // Chain match at row 0, plain miss at row 1.
        let (path, bonus, row, topk_hit) = chain.accept_path(&[11, 99, 13]).unwrap();
        assert_eq!(path, vec![0, 1]);
        assert_eq!(bonus, 99);
        assert_eq!(row, 1);
        assert!(!topk_hit);
    }

    #[test]
    fn draft_chain_validates_shape_and_schedules_verify() {
        let mut chain = DraftChain::start(10, 2);
        assert!(chain.validate().is_err()); // no children yet
        chain.push_candidates(vec![11]);
        chain.add_chain_child(11).unwrap();
        chain.push_candidates(vec![12]);
        chain.add_chain_child(12).unwrap();
        chain.validate().unwrap();

        let sched = chain.verify_schedule(100);
        assert_eq!(sched.positions, vec![100, 101, 102]);
        assert_eq!(sched.ancestors, vec![vec![], vec![0], vec![0, 1]]);
        sched.validate_sparse_at(100).unwrap();

        // A non-causal ancestor is rejected.
        let bad = SpecVerifySchedule {
            positions: vec![100, 101],
            ancestors: vec![vec![], vec![1]],
        };
        assert!(bad.validate_sparse_at(100).is_err());

        // Wrong argmax length is rejected, not silently truncated.
        assert!(chain.accept_path(&[10]).is_err());

        // The verify-row cap is enforced while building.
        let mut deep = DraftChain::start(1, MAX_SPEC_VERIFY_ROWS);
        for _ in 0..MAX_SPEC_VERIFY_ROWS - 1 {
            deep.push_candidates(vec![2]);
            deep.add_chain_child(2).unwrap();
        }
        assert!(deep.add_chain_child(2).is_err());
    }
}
