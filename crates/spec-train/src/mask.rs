//! Additive attention mask for one DSpark training forward.
//!
//! Ported from deepseek-ai/DeepSpec (MIT) `create_dspark_attention_mask`.
//! Keys are `[context(ctx_len) ; draft(blocks × block_size)]`; queries are the
//! draft rows alone. Row `(j, t)` sees context keys `< anchor_j` and every
//! draft key of block `j` — the block is bidirectional inside itself.
//!
//! Dense because that is all it needs to be: the mask adds into scores through
//! the existing `add_broadcast`, so attention stays the eager matmul/softmax
//! chain and its backward comes free. No new kernel.

use crate::block::Block;

/// `[q_len, ctx_len + q_len]` row-major, 0 where visible.
#[must_use]
pub fn additive(blocks: &[Block], ctx_len: usize) -> Vec<f32> {
    let Some(block_size) = blocks.first().map(|b| b.targets.len()) else {
        return Vec::new();
    };
    let q_len = blocks.len() * block_size;
    let kv_len = ctx_len + q_len;
    let mut m = vec![f32::NEG_INFINITY; q_len * kv_len];
    for (j, b) in blocks.iter().enumerate() {
        for t in 0..block_size {
            let row = (j * block_size + t) * kv_len;
            m[row..row + b.anchor.min(ctx_len)].fill(0.0);
            let own = row + ctx_len + j * block_size;
            m[own..own + block_size].fill(0.0);
        }
    }
    m
}

/// Visible fraction of the score matrix.
#[must_use]
pub fn density(blocks: &[Block], ctx_len: usize) -> f32 {
    let Some(block_size) = blocks.first().map(|b| b.targets.len()) else {
        return 0.0;
    };
    let q_len = blocks.len() * block_size;
    let visible: usize = blocks
        .iter()
        .map(|b| block_size * (b.anchor.min(ctx_len) + block_size))
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
        let m = additive(&blocks, ctx);
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
        let m = additive(&blocks, ctx);
        for r in 0..q {
            for c in ctx..ctx + q {
                let same_block = (c - ctx) / 7 == r / 7;
                assert_eq!(m[r * (ctx + q) + c] == 0.0, same_block, "row {r} key {c}");
            }
        }
    }

    #[test]
    fn density_at_training_shape() {
        let anchors: Vec<usize> = (0..512).map(|i| 8 * i + 3).collect();
        let blocks = blocks_at(&anchors, 4096, 7);
        let d = density(&blocks, 4096);
        assert!(d > 0.2 && d < 0.3, "density {d}");
    }
}
