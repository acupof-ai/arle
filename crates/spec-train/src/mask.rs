//! Additive attention mask for one DSpark training forward.
//!
//! Ported from deepseek-ai/DeepSpec (MIT) `create_dspark_attention_mask`.
//! Keys are `[context(ctx_len) ; draft(blocks × block_size)]`; queries are the
//! draft rows alone. Row `(j, t)` sees context keys `< anchor_j` within its own
//! sliding window and every draft key of block `j` — the block is bidirectional
//! inside itself.
//!
//! Dense because that is all it needs to be: the mask adds into scores through
//! the existing `add_broadcast`, so attention stays the eager matmul/softmax
//! chain and its backward comes free. No new kernel.

use crate::block::Block;

/// Context keys row `t` of the block at `anchor` may reach: `[lo, anchor)`,
/// clamped. Exclusive because the serve's ring stops below the anchor, and the
/// window applies to every draft layer there regardless of `layer_types` —
/// both derived in the tests below.
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
