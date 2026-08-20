//! A tiled q inside a longer zigzag k run must decompose into Full/Causal
//! rectangles that cover exactly the causal mask. Before `split_runs_at`,
//! any q tile shorter than its k run errored with "q/k runs partially
//! overlap", which blocked chunking CP full attention over q tiles.

use cuda_kernels::ring_attention::{PairClass, classify_pair, contiguous_pos_runs, split_runs_at};
use std::collections::HashSet;

/// Reimplements `ring_fa3_pairs`'s classification (the cuda-gated part is just
/// kernel launches) and expands each pair into its visible (q_abs, k_abs) set.
fn covered(q_pos: &[usize], k_pos: &[usize]) -> HashSet<(usize, usize)> {
    let q_runs = contiguous_pos_runs(q_pos);
    let mut cuts: Vec<usize> = q_runs.iter().flat_map(|q| [q.abs, q.abs + q.len]).collect();
    cuts.sort_unstable();
    cuts.dedup();
    let k_runs = split_runs_at(contiguous_pos_runs(k_pos), &cuts);
    let mut seen = HashSet::new();
    for q in &q_runs {
        for k in &k_runs {
            match classify_pair(*q, *k).expect("no partial overlap after split") {
                PairClass::Full => {
                    for qa in q.abs..q.abs + q.len {
                        for ka in k.abs..k.abs + k.len {
                            assert!(seen.insert((qa, ka)), "double-covered {qa},{ka}");
                        }
                    }
                }
                PairClass::Causal => {
                    for qa in q.abs..q.abs + q.len {
                        for ka in k.abs..=qa {
                            assert!(seen.insert((qa, ka)), "double-covered {qa},{ka}");
                        }
                    }
                }
                PairClass::Skip => {}
            }
        }
    }
    seen
}

fn causal_truth(q_pos: &[usize], k_pos: &[usize]) -> HashSet<(usize, usize)> {
    let mut truth = HashSet::new();
    for &qa in q_pos {
        for &ka in k_pos {
            if ka <= qa {
                truth.insert((qa, ka));
            }
        }
    }
    truth
}

#[test]
fn tiled_q_covers_the_causal_mask_exactly() {
    // Zigzag cp=2: rank0 owns abs [0..8) + [24..32), rank1 owns [8..24).
    let rank0: Vec<usize> = (0..8).chain(24..32).collect();
    let rank1: Vec<usize> = (8..24).collect();
    for own in [&rank0, &rank1] {
        for other in [&rank0, &rank1] {
            // Every q tile of width 3 (misaligned on purpose) and 4.
            for width in [3usize, 4] {
                for start in (0..own.len()).step_by(width) {
                    let tile = &own[start..(start + width).min(own.len())];
                    assert_eq!(
                        covered(tile, other),
                        causal_truth(tile, other),
                        "tile {tile:?} vs block of len {}",
                        other.len()
                    );
                }
            }
        }
    }
}
