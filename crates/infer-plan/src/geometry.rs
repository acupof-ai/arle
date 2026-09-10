//! Pure host prefill geometry: the page-table and CP-ring layout a prefill
//! needs, computed without any device handle. The executor uploads this
//! descriptor into `PageMeta` / `Qwen35CpPrefill` on the device side.

/// The prefill layout for one row, settled before any upload.
///
/// `slices` is empty when the CP ring path is not engaged; the executor then
/// builds a plain slot page table. When engaged, rank `cp_rank` computes
/// slice `slices[cp_rank]`; `q_pos`/`k_pos` are the absolute position arrays
/// the FA3 ring route reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillGeometry {
    pub slot: usize,
    pub start_pos: usize,
    pub len: usize,
    pub cp_rank: usize,
    /// Per CP rank: (chunk-relative offset, row count).
    pub slices: Vec<(usize, usize)>,
    /// `ceil(len / cp_size)` — the fixed per-rank row count for the collectives.
    pub pad: usize,
    /// Slot page indices covering this shard's local pages, in local-index order.
    pub page_indices: Vec<i32>,
    /// This rank's absolute q positions `[off, off + rows)`.
    pub q_pos: Vec<usize>,
    /// Per-owner absolute k positions: `k_pos[owner]` covers slice `owner`.
    pub k_pos: Vec<Vec<usize>>,
}

impl PrefillGeometry {
    /// Balanced block-cyclic slices over `len` rows: the first `rem` ranks get
    /// `base + 1` rows, the rest `base`. `page_indices` is copied in only for
    /// the ring path.
    pub fn compute(
        slot: usize,
        start_pos: usize,
        len: usize,
        page_indices: &[i32],
        cp_size: usize,
        cp_rank: usize,
        ring_prefill: bool,
    ) -> Self {
        if !ring_prefill {
            return Self {
                slot,
                start_pos,
                len,
                cp_rank,
                slices: Vec::new(),
                pad: 0,
                page_indices: Vec::new(),
                q_pos: Vec::new(),
                k_pos: Vec::new(),
            };
        }
        let per = len.div_ceil(cp_size);
        let base = len / cp_size;
        let rem = len % cp_size;
        let slices: Vec<(usize, usize)> = (0..cp_size)
            .map(|p| (p * base + p.min(rem), base + usize::from(p < rem)))
            .collect();
        let (off, my_len) = slices[cp_rank];
        let q_pos: Vec<usize> = (0..my_len).map(|i| start_pos + off + i).collect();
        let k_pos: Vec<Vec<usize>> = slices
            .iter()
            .map(|&(o, l)| (0..l).map(|i| start_pos + o + i).collect())
            .collect();
        Self {
            slot,
            start_pos,
            len,
            cp_rank,
            slices,
            pad: per,
            page_indices: page_indices.to_vec(),
            q_pos,
            k_pos,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_slices_are_balanced_and_cover_the_chunk() {
        let geo = PrefillGeometry::compute(3, 100, 10, &[9, 8, 7], 4, 0, true);
        assert_eq!(geo.pad, 3);
        // base=2, rem=2: ranks 0..1 get 3, ranks 2..3 get 2.
        assert_eq!(geo.slices, vec![(0, 3), (3, 3), (6, 2), (8, 2)]);
        assert_eq!(geo.q_pos, vec![100, 101, 102]);
        assert_eq!(geo.k_pos[1], vec![103, 104, 105]);
        assert_eq!(geo.k_pos[3], vec![108, 109]);
        assert_eq!(geo.page_indices, vec![9, 8, 7]);
        // Every row belongs to exactly one slice.
        let covered: usize = geo.slices.iter().map(|&(_, l)| l).sum();
        assert_eq!(covered, 10);
    }

    #[test]
    fn non_ring_geometry_is_empty_layout() {
        let geo = PrefillGeometry::compute(3, 100, 10, &[9, 8, 7], 4, 0, false);
        assert!(geo.slices.is_empty());
        assert!(geo.page_indices.is_empty());
        assert_eq!(geo.len, 10);
    }
}
