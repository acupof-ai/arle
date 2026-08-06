//! Anchored block construction for DSpark draft training.
//!
//! One sequence yields up to `num_anchors` independent training blocks: at each
//! anchor `a` the draft predicts positions `a+1 ..= a+block_size` from the prefix
//! `<= a`. That amplification is why offline training sees ~1.8M rows per
//! optimizer step where an online capture sees one block per decode step — it is
//! a property of the training-time attention mask, not a data-rate knob.
//!
//! Ported from deepseek-ai/DeepSpec (MIT), `deepspec/modeling/dspark/common.py`:
//! `build_anchor_candidate_mask`, `sample_anchor_positions`, `build_eval_mask`,
//! and the `label_indices` / `prev_token_ids` construction in
//! `deepspec/modeling/dspark/qwen3/modeling.py`. Deviations are called out where
//! they occur; there are none in the supervision itself.
//!
//! Everything here is host index math over one sequence. The attention plan is
//! [`crate::mask`]; the objective is [`crate::loss`].

use anyhow::{Result, ensure};

/// One anchored block's supervision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    /// Trunk position the block hangs off. The draft sees the prefix `<= anchor`.
    pub anchor: usize,
    /// Ground-truth ids at `anchor+1 ..= anchor+block_size`. Indices past the
    /// sequence clamp to the last position, exactly as the reference's
    /// `safe_label_indices` does; those rows are masked off by `eval`.
    pub targets: Vec<u32>,
    /// Bias condition per draft row: `[input_ids[anchor], targets[..block-1]]` —
    /// teacher-forced. Row `t` predicts `targets[t]` conditioned on `prev[t]`.
    pub prev: Vec<u32>,
    /// Per-row loss mask, prefix-closed (the reference's `cumprod`): once a row
    /// is invalid the rest of the block is too, because a draft chain stops at
    /// its first rejection.
    pub eval: Vec<bool>,
}

impl Block {
    /// Rows carrying loss — always a prefix of the block.
    #[must_use]
    pub fn trained_rows(&self) -> usize {
        self.eval.iter().take_while(|&&e| e).count()
    }
}

/// Positions that may anchor a block: `p` is trainable and so is `p+1`, so the
/// block has at least one supervised target.
#[must_use]
pub fn anchor_candidates(loss_mask: &[bool]) -> Vec<usize> {
    (0..loss_mask.len().saturating_sub(1))
        .filter(|&p| loss_mask[p] && loss_mask[p + 1])
        .collect()
}

/// `num_anchors` candidates sampled without replacement, returned ascending.
///
/// The reference draws a uniform key per candidate and sorts; this uses a
/// counter-based hash of `(seed, candidate)` for the same distribution with a
/// reproducible run. Returns fewer than `num_anchors` when the sequence has
/// fewer candidates — the reference's `keep_mask`, expressed as a shorter list
/// rather than a padded one.
#[must_use]
pub fn sample_anchors(loss_mask: &[bool], num_anchors: usize, seed: u64) -> Vec<usize> {
    let candidates = anchor_candidates(loss_mask);
    if candidates.is_empty() || num_anchors == 0 {
        return Vec::new();
    }
    let mut keyed: Vec<(u64, usize)> = candidates
        .iter()
        .map(|&p| {
            (
                splitmix64(seed ^ (p as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                p,
            )
        })
        .collect();
    keyed.sort_unstable();
    keyed.truncate(num_anchors);
    let mut anchors: Vec<usize> = keyed.into_iter().map(|(_, p)| p).collect();
    anchors.sort_unstable();
    anchors
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Build the supervision for one anchor.
///
/// `input_ids` and `loss_mask` are the whole sequence; `loss_mask[p]` marks `p`
/// as trainable (the reference's assistant-turn mask).
pub fn build_block(
    input_ids: &[u32],
    loss_mask: &[bool],
    anchor: usize,
    block_size: usize,
) -> Result<Block> {
    ensure!(block_size > 0, "block_size must be positive");
    ensure!(
        input_ids.len() == loss_mask.len(),
        "input_ids {} != loss_mask {}",
        input_ids.len(),
        loss_mask.len()
    );
    ensure!(
        anchor + 1 < input_ids.len(),
        "anchor {anchor} leaves nothing to predict in {} tokens",
        input_ids.len()
    );

    let last = input_ids.len() - 1;
    let mut targets = Vec::with_capacity(block_size);
    let mut eval = Vec::with_capacity(block_size);
    let mut live = true;
    for t in 0..block_size {
        let pos = anchor + 1 + t;
        let safe = pos.min(last);
        live &= pos < input_ids.len() && loss_mask[safe];
        eval.push(live);
        targets.push(input_ids[safe]);
    }

    let mut prev = Vec::with_capacity(block_size);
    prev.push(input_ids[anchor]);
    prev.extend_from_slice(&targets[..block_size - 1]);

    Ok(Block {
        anchor,
        targets,
        prev,
        eval,
    })
}

/// Draft-row input ids, flattened over blocks. Only the anchor row carries a
/// real token; the rest of the block is denoised from `mask_token_id`, so the
/// draft never sees the answer it is predicting. `prev` is the Markov head's
/// condition and is a different thing.
#[must_use]
pub fn noise_token_ids(blocks: &[Block], mask_token_id: u32) -> Vec<u32> {
    blocks
        .iter()
        .flat_map(|b| {
            std::iter::once(b.prev[0]).chain(std::iter::repeat_n(mask_token_id, b.prev.len() - 1))
        })
        .collect()
}

/// RoPE position of each draft row: `anchor + 1 + t` — the position the row
/// predicts. Matches the serve, where a block starting at `kv_seq_len = anchor+1`
/// puts row `t` at RoPE position `start + t`.
#[must_use]
pub fn draft_positions(blocks: &[Block]) -> Vec<usize> {
    blocks
        .iter()
        .flat_map(|b| (0..b.targets.len()).map(move |t| b.anchor + 1 + t))
        .collect()
}

/// Trunk hidden index whose logits are row `t`'s distillation target:
/// `anchor + t`, since hidden `p` predicts token `p+1` and row `t` predicts
/// `anchor + 1 + t`.
#[must_use]
pub fn target_hidden_positions(blocks: &[Block]) -> Vec<usize> {
    blocks
        .iter()
        .flat_map(|b| (0..b.targets.len()).map(move |t| b.anchor + t))
        .collect()
}

/// Per-row loss weight `exp(-t / gamma)`, zeroed where `eval` is false.
/// `gamma = None` weights every live row equally. The reference ships 4.0.
#[must_use]
pub fn row_weights(block: &Block, gamma: Option<f32>) -> Vec<f32> {
    block
        .eval
        .iter()
        .enumerate()
        .map(|(t, &live)| match (live, gamma) {
            (false, _) => 0.0,
            (true, Some(g)) if g > 0.0 => (-(t as f32) / g).exp(),
            (true, _) => 1.0,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(n: usize) -> (Vec<u32>, Vec<bool>) {
        ((0..n as u32).collect(), vec![true; n])
    }

    #[test]
    fn block_predicts_the_positions_after_its_anchor() {
        let (ids, mask) = seq(16);
        let b = build_block(&ids, &mask, 3, 4).unwrap();
        assert_eq!(b.targets, vec![4, 5, 6, 7]);
        // Row t predicts targets[t] conditioned on prev[t]: teacher-forced.
        assert_eq!(b.prev, vec![3, 4, 5, 6]);
        assert!(b.eval.iter().all(|&e| e));
    }

    #[test]
    fn eval_is_prefix_closed_and_survives_the_sequence_end() {
        let (ids, mut mask) = seq(16);
        mask[6] = false; // row 2 of an anchor-3 block
        let b = build_block(&ids, &mask, 3, 4).unwrap();
        assert_eq!(b.eval, vec![true, true, false, false], "must not resume");
        assert_eq!(b.trained_rows(), 2);

        let b = build_block(&ids, &[true; 16], 13, 4).unwrap();
        assert_eq!(b.eval, vec![true, true, false, false], "runs off the end");
        assert_eq!(b.targets.len(), 4, "still block_size wide");
    }

    #[test]
    fn weights_decay_over_the_block_and_zero_the_dead_rows() {
        let (ids, mut mask) = seq(16);
        mask[6] = false;
        let b = build_block(&ids, &mask, 3, 4).unwrap();
        let w = row_weights(&b, Some(4.0));
        assert!((w[0] - 1.0).abs() < 1e-6);
        assert!((w[1] - (-0.25f32).exp()).abs() < 1e-6);
        assert_eq!(&w[2..], &[0.0, 0.0]);
        assert_eq!(row_weights(&b, None), vec![1.0, 1.0, 0.0, 0.0]);
    }

    /// The reference requires BOTH `p` and `p+1` trainable — an anchor outside
    /// the assistant turn would condition the block on a prompt token.
    #[test]
    fn a_candidate_needs_its_own_position_trainable_too() {
        let mut mask = vec![false; 8];
        mask[4..].fill(true);
        assert_eq!(anchor_candidates(&mask), vec![4, 5, 6]);
        assert!(
            !anchor_candidates(&mask).contains(&3),
            "position 3 is not trainable, only its successor is"
        );
    }

    #[test]
    fn sampling_is_reproducible_ascending_and_within_the_candidates() {
        let mut mask = vec![false; 64];
        mask[16..].fill(true);
        let a = sample_anchors(&mask, 8, 42);
        assert_eq!(a.len(), 8);
        assert_eq!(a, sample_anchors(&mask, 8, 42), "same seed, same anchors");
        assert_ne!(a, sample_anchors(&mask, 8, 43), "seed actually moves it");
        assert!(a.windows(2).all(|w| w[0] < w[1]), "ascending, no repeats");
        let cands = anchor_candidates(&mask);
        assert!(a.iter().all(|p| cands.contains(p)));
    }

    #[test]
    fn only_the_anchor_row_carries_a_real_token() {
        let (ids, mask) = seq(16);
        let blocks = [
            build_block(&ids, &mask, 3, 3).unwrap(),
            build_block(&ids, &mask, 8, 3).unwrap(),
        ];
        assert_eq!(
            noise_token_ids(&blocks, 999),
            vec![3, 999, 999, 8, 999, 999]
        );
        assert_eq!(draft_positions(&blocks), vec![4, 5, 6, 9, 10, 11]);
        assert_eq!(
            target_hidden_positions(&blocks),
            vec![3, 4, 5, 8, 9, 10],
            "the trunk hidden that predicts the row's target sits one below it"
        );
    }

    #[test]
    fn fewer_candidates_than_anchors_is_not_an_error() {
        let mut mask = vec![false; 64];
        mask[60..].fill(true);
        let cands = anchor_candidates(&mask);
        assert_eq!(sample_anchors(&mask, 100, 7).len(), cands.len());
        assert!(sample_anchors(&[false; 32], 4, 7).is_empty());
        assert!(sample_anchors(&mask, 0, 7).is_empty());
    }
}
