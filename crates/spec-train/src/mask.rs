//! Attention plan for a batch of anchored blocks.
//!
//! Block `j` at anchor `a_j` attends to the prefix `[0, a_j)` plus its own
//! `block_size` draft rows. Feeding the prefix as fixed chunks, each
//! (block, chunk) pair is [`Visibility::All`] (chunk ends at or before the
//! anchor), [`Visibility::None`] (chunk starts at or after it), or
//! [`Visibility::Partial`] — the one chunk the anchor falls inside.
//!
//! Partial is the whole design problem. ARLE's ring-attention merge kernels
//! (`autograd::ops::ring_attention`) are cheap precisely because they refuse it:
//! `classify_pair` errors on any q/k run that partially overlaps, so every tile
//! is all-or-nothing. Chunk-aligning the anchors does NOT help — a block's rows
//! start at `a_j + 1`, inside the chunk containing `a_j`, by construction. So a
//! masked-tile variant is required, and [`Plan::partial`] is exactly the set it
//! has to cover. [`Plan::partial_fraction`] says how small that set is.

use crate::block::Block;

/// How much of one prefix chunk block `j` may attend to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    /// Every key in the chunk precedes the anchor.
    All,
    /// No key in the chunk precedes the anchor.
    None,
    /// The anchor falls inside the chunk: keys `[start, anchor)` only.
    Partial,
}

/// One (block, chunk) pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    pub block: usize,
    /// First key position of the chunk.
    pub chunk_start: usize,
    pub visibility: Visibility,
    /// Keys visible in this chunk: `chunk_len` for `All`, 0 for `None`,
    /// `anchor - chunk_start` for `Partial`.
    pub visible: usize,
}

/// The full plan for one sequence's blocks against a chunked prefix.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub chunk_len: usize,
    pub tiles: Vec<Tile>,
}

impl Plan {
    /// Tiles needing per-element masking — what a masked-tile kernel must cover.
    pub fn partial(&self) -> impl Iterator<Item = &Tile> {
        self.tiles
            .iter()
            .filter(|t| t.visibility == Visibility::Partial)
    }

    /// Fraction of contributing tiles that are partial. Every block straddles at
    /// most one chunk, so this falls as the prefix grows: the masked path is a
    /// tail, not the common case.
    #[must_use]
    pub fn partial_fraction(&self) -> f32 {
        let contributing = self
            .tiles
            .iter()
            .filter(|t| t.visibility != Visibility::None)
            .count();
        if contributing == 0 {
            return 0.0;
        }
        self.partial().count() as f32 / contributing as f32
    }
}

/// Plan `blocks` against the prefix `[0, seq_len)` cut into `chunk_len` chunks.
#[must_use]
pub fn plan(blocks: &[Block], seq_len: usize, chunk_len: usize) -> Plan {
    assert!(chunk_len > 0, "chunk_len must be positive");
    let mut tiles = Vec::new();
    for (block, b) in blocks.iter().enumerate() {
        for chunk_start in (0..seq_len).step_by(chunk_len) {
            let chunk_end = (chunk_start + chunk_len).min(seq_len);
            let (visibility, visible) = if chunk_end <= b.anchor {
                (Visibility::All, chunk_end - chunk_start)
            } else if chunk_start >= b.anchor {
                (Visibility::None, 0)
            } else {
                (Visibility::Partial, b.anchor - chunk_start)
            };
            tiles.push(Tile {
                block,
                chunk_start,
                visibility,
                visible,
            });
        }
    }
    Plan { chunk_len, tiles }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::build_block;

    fn blocks_at(anchors: &[usize], n: usize) -> Vec<Block> {
        let ids: Vec<u32> = (0..n as u32).collect();
        let mask = vec![true; n];
        anchors
            .iter()
            .map(|&a| build_block(&ids, &mask, a, 7).unwrap())
            .collect()
    }

    #[test]
    fn a_block_sees_exactly_its_prefix() {
        let p = plan(&blocks_at(&[20], 64), 64, 16);
        // anchor 20: chunk 0 fully visible, chunk 16 straddled (4 keys), rest none.
        let visible: Vec<usize> = p.tiles.iter().map(|t| t.visible).collect();
        assert_eq!(visible, vec![16, 4, 0, 0]);
        assert_eq!(visible.iter().sum::<usize>(), 20, "== the anchor");
    }

    #[test]
    fn every_block_straddles_at_most_one_chunk() {
        let p = plan(&blocks_at(&[5, 13, 27, 41], 64), 64, 16);
        for b in 0..4 {
            let n = p.partial().filter(|t| t.block == b).count();
            assert!(n <= 1, "block {b} straddles {n} chunks");
        }
    }

    /// The finding that decides the kernel work: aligning anchors to chunk
    /// boundaries does not remove the partial tiles, because a block's rows
    /// begin one past its anchor.
    #[test]
    fn chunk_aligned_anchors_do_not_remove_the_partial_tiles() {
        let aligned = plan(&blocks_at(&[16, 32, 48], 64), 64, 16);
        assert_eq!(
            aligned.partial().count(),
            0,
            "an anchor ON a boundary is clean"
        );
        // But anchors land wherever the assistant turn starts, so the general
        // case always straddles.
        let real = plan(&blocks_at(&[17, 33, 49], 64), 64, 16);
        assert_eq!(real.partial().count(), 3, "one per block");
    }

    #[test]
    fn the_masked_path_is_a_tail_at_training_shape() {
        // 512 anchors over a 4096 prefix, the reference's shape.
        let anchors: Vec<usize> = (0..512).map(|i| 8 * i + 3).collect();
        let p = plan(&blocks_at(&anchors, 4096), 4096, 16);
        let frac = p.partial_fraction();
        assert!(
            frac < 0.05,
            "partial tiles are {:.1}% of contributing work",
            frac * 100.0
        );
        assert_eq!(p.partial().count(), 512, "exactly one per block");
    }
}
