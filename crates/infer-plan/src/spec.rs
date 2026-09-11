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
enum DecodeRoute {
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
fn route_decode(spec_kind: SpecKind, n_rows: usize, gate: usize, vetoed: bool) -> DecodeRoute {
    if vetoed || n_rows > gate {
        return DecodeRoute::Plain;
    }
    match spec_kind {
        SpecKind::Dspark => DecodeRoute::Dspark,
        SpecKind::Mtp => DecodeRoute::Mtp,
        SpecKind::None => DecodeRoute::Plain,
    }
}

/// A verify of `depth` draft rows after `start` committed rows stays under the
/// trunk cap.
pub fn speculative_chain_fits(start: usize, depth: usize, max_seq_len: usize) -> bool {
    start
        .checked_add(depth)
        .is_some_and(|last_position| last_position < max_seq_len)
}

/// Qwen MTP/DSpark can reproduce raw greedy or temperature/top-k/top-p/min-p
/// sampling. Every other token rewrite needs the plain one-token sampler.
pub fn qwen_spec_decode_compatible(params: &crate::SamplingParams) -> bool {
    params.grammar_bitmask.is_none()
        && params.logit_bias.is_empty()
        && params.top_logprobs.is_none()
        && params.force_next_token.is_none()
        && params.max_thinking_tokens.is_none()
        && !params.has_penalty()
}

/// One chain in a batched spec verify; `row0` indexes the shared logits and tap
/// features. `partial_ctx` is a ctx-ring flag only the DSpark draft sets; MTP
/// leaves it false.
pub struct SpecChain {
    /// Index of the originating row in the tick's decode rows.
    pub out: usize,
    pub slot: usize,
    pub start: usize,
    pub row0: usize,
    pub chain: Vec<u32>,
    pub partial_ctx: bool,
}

/// Lay chains out back-to-back in the shared verify logits: chain `i` owns
/// rows `[row0, row0 + chain.len())`. Returns the total row count.
pub fn assign_row_offsets(chains: &mut [SpecChain]) -> usize {
    let mut total = 0;
    for c in chains {
        c.row0 = total;
        total += c.chain.len();
    }
    total
}

pub fn flatten_chains(chains: &[SpecChain]) -> Vec<u32> {
    chains
        .iter()
        .flat_map(|c| c.chain.iter().copied())
        .collect()
}

/// Accepted/rejected draft counts for a chain of `chain_len` rows (the pending
/// token plus the drafts) with `k` drafts accepted. The pending row is never a
/// reject.
pub fn spec_accept_totals(chain_len: usize, k: usize) -> (usize, usize) {
    (k, chain_len - 1 - k)
}

/// Page count covering `len` tokens at `page_size` tokens per page.
pub fn pages_covering(len: usize, page_size: usize) -> usize {
    len.div_ceil(page_size)
}

/// The outcome of a greedy accept scan over one verified chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecAcceptOutcome {
    /// Accepted draft tokens plus the bonus token, in commit order.
    pub emitted: Vec<u32>,
    /// The first mismatched trunk argmax (or the last row's argmax on a full
    /// accept); the next pending token.
    pub bonus: u32,
    /// Number of draft tokens accepted.
    pub k: usize,
    /// The chain has unaccepted rows the caller rolls back (`k + 1 <
    /// chain.len()`).
    pub partial: bool,
}

/// Greedy accept scan: the longest prefix where each draft token equals the
/// trunk argmax at its row. `chain[0]` is the pending token, `chain[1..]` the
/// drafts; `argmax[row0..row0+chain.len()]` is the verified target. Pure host —
/// the executor's accept loop calls this and runs the device leaf calls
/// (truncate, mirror, ctx append) on the outcome.
pub fn spec_accept_greedy(chain: &[u32], argmax: &[u32], row0: usize) -> Result<SpecAcceptOutcome> {
    let depth = chain.len() - 1;
    ensure!(
        row0 + chain.len() <= argmax.len(),
        "spec accept: chain rows outside the verify argmax"
    );
    let mut k = 0usize;
    let bonus;
    loop {
        let am = argmax[row0 + k];
        if k < depth && am == chain[k + 1] {
            k += 1;
        } else {
            bonus = am;
            break;
        }
    }
    let mut emitted: Vec<u32> = chain[1..=k].to_vec();
    emitted.push(bonus);
    Ok(SpecAcceptOutcome {
        emitted,
        bonus,
        k,
        partial: k + 1 < chain.len(),
    })
}

/// The seeded rows that share one batched DSpark draft forward: slot-sorted
/// row indices with the anchor tokens and start positions the draft kernel
/// reads. `dspark_draft_plan` returns `None` when fewer than two rows seeded
/// or any row is sampled — the batched gate is greedy-only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsparkDraftPlan {
    pub idx: Vec<usize>,
    pub anchors: Vec<u32>,
    pub starts: Vec<usize>,
}

/// Pure host half of the batched DSpark draft: which rows draft together and
/// what each one drafts from. The caller gathers the per-slot device state
/// (`pick`/`dfs`) by slot and runs `dspark_draft_blocks`.
pub fn dspark_draft_plan(seeded: &[bool], rows: &[crate::DecodeRow]) -> Option<DsparkDraftPlan> {
    let mut idx: Vec<usize> = (0..seeded.len()).filter(|&i| seeded[i]).collect();
    if idx.len() < 2 || !rows.iter().all(|r| r.params.is_greedy()) {
        return None;
    }
    idx.sort_by_key(|&i| rows[i].slot);
    let anchors = idx.iter().map(|&i| rows[i].last_token).collect();
    let starts = idx.iter().map(|&i| rows[i].kv_seq_len).collect();
    Some(DsparkDraftPlan {
        idx,
        anchors,
        starts,
    })
}

impl DsparkDraftPlan {
    /// Place each drafted chain at its row in `pre` (caller-sized, one slot
    /// per decode row).
    pub fn scatter_into(&self, chains: &[Vec<u32>], pre: &mut [Option<Vec<u32>>]) {
        for (n, &i) in self.idx.iter().enumerate() {
            pre[i] = Some(chains[n].clone());
        }
    }
}

/// Slot-sorted indices of the seeded, greedy rows; empty when no row
/// qualifies — the caller then skips the batched path.
pub fn greedy_seeded_indices(seeded: &[bool], rows: &[crate::DecodeRow]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..seeded.len())
        .filter(|&i| seeded[i] && rows[i].params.is_greedy())
        .collect();
    idx.sort_by_key(|&i| rows[i].slot);
    idx
}

/// The paged KV pool's decode-relevant shape. The executor maps its pool
/// format and FA3 state onto this class; the dispatch decision reads only the
/// class, never the format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeKvClass {
    /// No paged pool (recurrent-only decode).
    NoPool,
    /// BF16 pool on the FA3 decode lane.
    Bf16Fa3,
    /// BF16 pool without FA3 (tensor-core pool kernel).
    Bf16NoFa3,
    /// FP8/INT8 pool: split-KV decode lane with a fixed grid at B=1.
    Quantized,
    /// Pool present in a format the spec paths do not batch against.
    Other,
}

/// The settled decode route for one tick — a value the caller executes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeDispatch {
    Dspark,
    Mtp,
    /// One plain row. `capturable` admits the whole-step B=1 decode graph for
    /// this pool's format; the armed flag and the seq-len gate stay with the
    /// graph slot.
    PlainSingle {
        capturable: bool,
    },
    PlainBatch,
    Empty,
}

/// The `dspark → mtp → plain` dispatch ladder as a pure decision. At or below
/// `spec_max_batch` a spec scheme drafts per row; above it spec is a
/// compute-bound loss, so decode falls to the plain batched path that scales.
/// DSpark batches only on BF16 (its quant-KV verify loses above c=1,
/// errors/2026-08-22-batched-dspark-quant-kv-verify-loses); MTP batches on any
/// paged pool.
pub fn decide_decode(
    kind: SpecKind,
    n_rows: usize,
    spec_compatible: bool,
    all_greedy: bool,
    kv_class: DecodeKvClass,
    spec_max_batch: usize,
) -> DecodeDispatch {
    let batched = spec_compatible
        && all_greedy
        && match kind {
            SpecKind::Dspark => {
                matches!(kv_class, DecodeKvClass::Bf16Fa3 | DecodeKvClass::Bf16NoFa3)
            }
            SpecKind::Mtp => !matches!(kv_class, DecodeKvClass::NoPool),
            SpecKind::None => false,
        };
    let gate = if batched { spec_max_batch } else { 1 };
    match route_decode(kind, n_rows, gate, !spec_compatible) {
        DecodeRoute::Dspark => DecodeDispatch::Dspark,
        DecodeRoute::Mtp => DecodeDispatch::Mtp,
        DecodeRoute::Plain => match n_rows {
            0 => DecodeDispatch::Empty,
            1 => DecodeDispatch::PlainSingle {
                capturable: matches!(kv_class, DecodeKvClass::Bf16Fa3 | DecodeKvClass::Quantized),
            },
            _ => DecodeDispatch::PlainBatch,
        },
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
    fn speculative_chain_boundary_falls_back_before_verify_exceeds_max_seq_len() {
        assert!(speculative_chain_fits(12, 3, 16));
        assert!(!speculative_chain_fits(13, 3, 16));
        assert!(!speculative_chain_fits(usize::MAX, 1, usize::MAX));
    }

    #[test]
    fn qwen_spec_vetoes_token_rewrites_and_thinking_budget() {
        let mut params = crate::SamplingParams::default();
        assert!(qwen_spec_decode_compatible(&params));

        params.force_next_token = Some(42);
        assert!(!qwen_spec_decode_compatible(&params));
        params.force_next_token = None;
        params.max_thinking_tokens = Some(8);
        assert!(!qwen_spec_decode_compatible(&params));
        params.max_thinking_tokens = None;
        params.logit_bias.push((42, 1.0));
        assert!(!qwen_spec_decode_compatible(&params));
        params.logit_bias.clear();
        params.grammar_bitmask = Some(vec![u32::MAX].into());
        assert!(!qwen_spec_decode_compatible(&params));
        params.grammar_bitmask = None;
        params.repetition_penalty = 1.1;
        assert!(!qwen_spec_decode_compatible(&params));
    }

    #[test]
    fn decide_decode_routes_the_ladder() {
        use DecodeDispatch::*;
        use DecodeKvClass::*;
        let d =
            |kind, n, compat, greedy, kv, gate| decide_decode(kind, n, compat, greedy, kv, gate);
        // Spec schemes draft per row at c=1 on any pool that admits them.
        assert_eq!(d(SpecKind::Dspark, 1, true, true, Bf16Fa3, 1), Dspark);
        assert_eq!(d(SpecKind::Dspark, 1, true, true, Quantized, 1), Dspark);
        assert_eq!(d(SpecKind::Mtp, 1, true, true, Quantized, 1), Mtp);
        assert_eq!(d(SpecKind::Mtp, 1, true, true, NoPool, 1), Mtp);
        // Above the gate, only the batched-capable pairings stay on spec.
        assert_eq!(d(SpecKind::Dspark, 2, true, true, Bf16Fa3, 2), Dspark);
        assert_eq!(d(SpecKind::Dspark, 2, true, true, Quantized, 2), PlainBatch);
        assert_eq!(d(SpecKind::Mtp, 2, true, true, Bf16NoFa3, 2), Mtp);
        assert_eq!(d(SpecKind::Mtp, 2, true, true, NoPool, 2), PlainBatch);
        // Vetoed rows always go plain; a single sampled DSpark row still takes
        // the per-row DSpark path (its batched gate is greedy-only).
        assert_eq!(
            d(SpecKind::Mtp, 1, false, true, Bf16Fa3, 1),
            PlainSingle { capturable: true }
        );
        assert_eq!(d(SpecKind::Dspark, 1, true, false, Bf16Fa3, 1), Dspark);
        assert_eq!(d(SpecKind::None, 0, true, true, NoPool, 1), Empty);
        // Capturable tracks the format/FA3 class only.
        assert_eq!(
            d(SpecKind::None, 1, true, true, Bf16Fa3, 1),
            PlainSingle { capturable: true }
        );
        assert_eq!(
            d(SpecKind::None, 1, true, true, Quantized, 1),
            PlainSingle { capturable: true }
        );
        assert_eq!(
            d(SpecKind::None, 1, true, true, Bf16NoFa3, 1),
            PlainSingle { capturable: false }
        );
        assert_eq!(
            d(SpecKind::None, 1, true, true, Other, 1),
            PlainSingle { capturable: false }
        );
        assert_eq!(
            d(SpecKind::None, 1, true, true, NoPool, 1),
            PlainSingle { capturable: false }
        );
    }

    #[test]
    fn spec_accept_totals_excludes_the_pending_row_from_rejects() {
        // chain of 4 (pending + 3 drafts), 2 drafts accepted.
        assert_eq!(spec_accept_totals(4, 2), (2, 1));
        assert_eq!(spec_accept_totals(4, 3), (3, 0));
    }

    #[test]
    fn spec_accept_greedy_scans_the_longest_matching_prefix() {
        // chain = [pending, d0, d1, d2]; argmax matches d0, d1, misses d2.
        let chain = [7, 11, 12, 13];
        let argmax = [11, 12, 99, 0];
        let out = spec_accept_greedy(&chain, &argmax, 0).unwrap();
        assert_eq!(out.emitted, vec![11, 12, 99]);
        assert_eq!(out.bonus, 99);
        assert_eq!(out.k, 2);
        assert!(out.partial);
        // Full accept: every draft matches, bonus is the last row's argmax.
        let argmax = [11, 12, 13, 14];
        let out = spec_accept_greedy(&chain, &argmax, 0).unwrap();
        assert_eq!(out.emitted, vec![11, 12, 13, 14]);
        assert_eq!(out.k, 3);
        assert!(!out.partial);
        // First-row miss: only the bonus commits.
        let argmax = [99, 12, 13, 14];
        let out = spec_accept_greedy(&chain, &argmax, 0).unwrap();
        assert_eq!(out.emitted, vec![99]);
        assert_eq!(out.k, 0);
        assert!(out.partial);
        // row0 offsets into a shared argmax buffer.
        let argmax = [0, 0, 11, 12, 99, 0];
        let out = spec_accept_greedy(&chain, &argmax, 2).unwrap();
        assert_eq!(out.k, 2);
        assert_eq!(out.bonus, 99);
        // Chains outside the argmax buffer are rejected, not truncated.
        assert!(spec_accept_greedy(&chain, &argmax, 4).is_err());
    }

    #[test]
    fn pages_covering_rounds_up() {
        assert_eq!(pages_covering(0, 16), 0);
        assert_eq!(pages_covering(1, 16), 1);
        assert_eq!(pages_covering(16, 16), 1);
        assert_eq!(pages_covering(17, 16), 2);
    }

    fn draft_row(slot: usize, last_token: u32, kv_seq_len: usize, temp: f32) -> crate::DecodeRow {
        crate::DecodeRow {
            slot,
            last_token,
            kv_seq_len,
            params: crate::SamplingParams {
                temperature: temp,
                ..Default::default()
            },
            penalty_history: None,
            penalty_prompt_len: 0,
        }
    }

    #[test]
    fn dspark_draft_plan_batches_two_or_more_greedy_seeded_rows() {
        let rows = vec![
            draft_row(3, 10, 100, 0.0),
            draft_row(1, 11, 200, 0.0),
            draft_row(2, 12, 300, 0.0),
        ];
        // Fewer than two seeded: no batch.
        assert_eq!(dspark_draft_plan(&[true, false, false], &rows), None);
        // Two seeded, slot-sorted: row 2 (slot 1) before row 0 (slot 3).
        let plan = dspark_draft_plan(&[true, false, true], &rows).expect("2 seeded greedy rows");
        assert_eq!(plan.idx, vec![2, 0]);
        assert_eq!(plan.anchors, vec![12, 10]);
        assert_eq!(plan.starts, vec![300, 100]);
        // Any sampled row vetoes the greedy-only batch.
        let mut sampled = rows.clone();
        sampled[1].params.temperature = 0.7;
        assert_eq!(dspark_draft_plan(&[true, true, true], &sampled), None);
        // scatter places each chain at its row.
        let mut pre = vec![None; 3];
        plan.scatter_into(&[vec![1, 2], vec![3, 4]], &mut pre);
        assert_eq!(pre[0], Some(vec![3, 4]));
        assert_eq!(pre[1], None);
        assert_eq!(pre[2], Some(vec![1, 2]));
    }

    #[test]
    fn greedy_seeded_indices_filters_and_sorts() {
        let rows = vec![
            draft_row(3, 10, 100, 0.0),
            draft_row(1, 11, 200, 0.7),
            draft_row(2, 12, 300, 0.0),
        ];
        // Row 1 is seeded but sampled; row 2 is greedy but unseeded.
        assert_eq!(greedy_seeded_indices(&[true, true, false], &rows), vec![0]);
        assert!(greedy_seeded_indices(&[false, true, false], &rows).is_empty());
        // Slot-sorted: slot 1 (row 1) is sampled, so only rows 0 and 2 qualify.
        assert_eq!(
            greedy_seeded_indices(&[true, false, true], &rows),
            vec![2, 0]
        );
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
