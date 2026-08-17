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

/// RoPE position of each draft row: `anchor + t`, matching the serve. Doubles as
/// the trunk hidden index that supervises row `t` — hidden `p` predicts `p+1`,
/// and row `t` targets `anchor+1+t`.
#[must_use]
pub fn draft_positions(blocks: &[Block]) -> Vec<usize> {
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
