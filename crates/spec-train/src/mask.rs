//! Additive attention mask for one DSpark training forward.
//!
//! Ported from deepseek-ai/DeepSpec (MIT) `create_dspark_attention_mask`.
//! Keys are `[context(ctx_len) ; draft(blocks × block_size)]`; queries are the
//! draft rows alone. Row `(j, t)` sees context keys `<= anchor_j` within its own
//! sliding window and every draft key of block `j` — the block is bidirectional
//! inside itself.
//!
//! Dense because that is all it needs to be: the mask adds into scores through
//! the existing `add_broadcast`, so attention stays the eager matmul/softmax
//! chain and its backward comes free. No new kernel.

use crate::block::Block;

/// Context keys row `t` of the block at `anchor` may reach: `[lo, anchor)` with
/// `lo = (anchor+t) - (window-1)`, clamped. The serve's `start` IS the anchor —
/// `last_token` has not been forwarded when the draft runs, so its tap is not in
/// the ring (`executor/qwen35.rs` appends it only after the verify). The span
/// therefore stops strictly below `anchor`; reaching `taps[anchor]` would hand
/// the draft the residual its own distillation target is projected from.
/// The serve runs every draft layer on the sliding ring regardless of
/// `layer_types`, so training past the window would fit links inference does not
/// have.
fn ctx_span(anchor: usize, t: usize, ctx_len: usize, window: Option<usize>) -> (usize, usize) {
    let hi = anchor.min(ctx_len);
    let lo = match window {
        Some(w) if w > 0 => (anchor + t).saturating_sub(w - 1),
        _ => 0,
    };
    (lo.min(hi), hi)
}

/// `[q_len, ctx_len + q_len]` row-major, 0 where visible.
#[must_use]
pub fn additive(blocks: &[Block], ctx_len: usize, window: Option<usize>) -> Vec<f32> {
    let Some(block_size) = blocks.first().map(|b| b.targets.len()) else {
        return Vec::new();
    };
    let q_len = blocks.len() * block_size;
    let kv_len = ctx_len + q_len;
    let mut m = vec![f32::NEG_INFINITY; q_len * kv_len];
    for (j, b) in blocks.iter().enumerate() {
        for t in 0..block_size {
            let (lo, hi) = ctx_span(b.anchor, t, ctx_len, window);
            let row = (j * block_size + t) * kv_len;
            m[row + lo..row + hi].fill(0.0);
            let own = row + ctx_len + j * block_size;
            m[own..own + block_size].fill(0.0);
        }
    }
    m
}

/// Visible fraction of the score matrix.
#[must_use]
pub fn density(blocks: &[Block], ctx_len: usize, window: Option<usize>) -> f32 {
    let Some(block_size) = blocks.first().map(|b| b.targets.len()) else {
        return 0.0;
    };
    let q_len = blocks.len() * block_size;
    let visible: usize = blocks
        .iter()
        .flat_map(|b| {
            (0..block_size).map(move |t| {
                let (lo, hi) = ctx_span(b.anchor, t, ctx_len, window);
                hi - lo + block_size
            })
        })
        .sum();
    visible as f32 / (q_len * (ctx_len + q_len)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::build_block;

    fn blocks_at(anchors: &[usize], n: usize, block_size: usize) -> Vec<Block> {
        let ids: Vec<u32> = (0..n as u32).collect();
        let mask = vec![true; n];
        anchors
            .iter()
            .map(|&a| build_block(&ids, &mask, a, block_size).unwrap())
            .collect()
    }

    #[test]
    fn a_row_sees_its_prefix_and_its_whole_block() {
        let blocks = blocks_at(&[3, 6], 16, 2);
        let (ctx, q) = (16, 4);
        let m = additive(&blocks, ctx, None);
        let visible = |r: usize| -> Vec<usize> {
            (0..ctx + q)
                .filter(|&c| m[r * (ctx + q) + c] == 0.0)
                .collect()
        };
        assert_eq!(visible(0), vec![0, 1, 2, 16, 17]);
        assert_eq!(visible(1), visible(0), "no causality inside a block");
        assert_eq!(visible(2), vec![0, 1, 2, 3, 4, 5, 18, 19]);
    }

    #[test]
    fn a_block_never_sees_another_block() {
        let blocks = blocks_at(&[3, 6, 9], 16, 7);
        let (ctx, q) = (16, 21);
        let m = additive(&blocks, ctx, None);
        for r in 0..q {
            for c in ctx..ctx + q {
                let same_block = (c - ctx) / 7 == r / 7;
                assert_eq!(m[r * (ctx + q) + c] == 0.0, same_block, "row {r} key {c}");
            }
        }
    }

    /// The serve's window, written out from `qwen35/dspark.rs` rather than from
    /// `ctx_span`: `lo = (start + row).saturating_sub(sliding_window - 1)`, keys
    /// `[lo, start + block)`. `start` is the anchor's own absolute position —
    /// `last_token` at `kv_seq_len`, not yet forwarded — so the ring stops below
    /// it. The span slides per row; a block-wide one over-feeds the later rows.
    #[test]
    fn every_row_reaches_exactly_what_the_serve_gives_it() {
        let (ctx, window, anchor, block_size) = (32usize, 8usize, 20usize, 4usize);
        let blocks = blocks_at(&[anchor], ctx, block_size);
        let kv = ctx + block_size;
        let m = additive(&blocks, ctx, Some(window));

        let start = anchor;
        for row in 0..block_size {
            let lo = (start + row).saturating_sub(window - 1);
            let want: Vec<usize> = (lo..start).chain(ctx..kv).collect();
            let got: Vec<usize> = (0..kv).filter(|&c| m[row * kv + c] == 0.0).collect();
            assert_eq!(got, want, "row {row}");
        }
    }

    /// `taps[anchor]` is the residual `trainer.rs` projects through `lm_head` for
    /// row 0's target. The serve cannot supply it — the anchor is unforwarded at
    /// draft time — so reaching it here would fit a shortcut to the answer.
    #[test]
    fn the_anchor_tap_is_out_of_reach() {
        let (ctx, anchor, block_size) = (32usize, 10usize, 4usize);
        let blocks = blocks_at(&[anchor], ctx, block_size);
        for window in [None, Some(0), Some(4), Some(4096)] {
            let m = additive(&blocks, ctx, window);
            assert_eq!(m[anchor], f32::NEG_INFINITY, "window {window:?}");
            assert_eq!(m[anchor - 1], 0.0, "but the token before it is in reach");
        }
    }

    #[test]
    fn density_at_training_shape() {
        let anchors: Vec<usize> = (0..512).map(|i| 8 * i + 3).collect();
        let blocks = blocks_at(&anchors, 4096, 7);
        let d = density(&blocks, 4096, None);
        assert!(d > 0.2 && d < 0.3, "density {d}");
    }
}
