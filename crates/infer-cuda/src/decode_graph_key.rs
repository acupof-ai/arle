//! Pure host arithmetic for the B=1 decode-graph capture key (CG bookkeeping).
//!
//! Feature-agnostic on purpose: the `GraphBucket` HashMap that stores captured
//! decode graphs is keyed by the page-table length the captured TileLang kernel
//! walks (batch is the fixed [`DECODE_GRAPH_BATCH`] for this B=1 landing). That key
//! derivation is plain `usize` math with no device types, so it lives here — outside
//! the `cuda` feature gate — and is CPU-unit-testable on a Mac without nvcc, even
//! though the live `GraphBucket` / capture machinery in `decode_graph.rs` is gated.

/// Decode batch size this first landing captures. B=1 is the AI-PC headline win
/// (design §7): at batch 1 the decode step is purely launch-bound, so collapsing
/// the ~250-400 per-token `cuLaunchKernel` calls into one `cuGraphLaunch` removes
/// essentially all per-token CPU launch overhead.
pub(crate) const DECODE_GRAPH_BATCH: usize = 1;

/// Identifies a captured decode graph by the shape baked into its launch args.
///
/// `batch_size` is always [`DECODE_GRAPH_BATCH`] in this B=1 landing; `num_pages`
/// is the page-table length the captured TileLang kernel walks (its `total_pages`
/// scalar launch arg). A captured graph is only replay-valid when both match the
/// current step.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct DecodeGraphKey {
    pub(crate) batch_size: usize,
    pub(crate) num_pages: usize,
}

/// Pure derivation of the B=1 decode capture key from the cache state.
///
/// `kv_seq_len` is the cache length BEFORE appending this step's token, so the new
/// total length is `kv_seq_len + 1` and the page-table length is that rounded up by
/// `page_size`. The `GraphBucket` keys captured graphs by `num_pages`, so this is
/// the lookup key the bucket bookkeeping inserts and looks up against; the
/// page-boundary recapture trigger is exactly a change in `num_pages`.
#[allow(dead_code)] // used by the cuda-gated decode_graph.rs; pure path stays testable
pub(crate) fn decode_graph_key_for(page_size: usize, kv_seq_len: usize) -> DecodeGraphKey {
    let total_len = kv_seq_len + 1;
    let num_pages = total_len.div_ceil(page_size.max(1));
    DecodeGraphKey {
        batch_size: DECODE_GRAPH_BATCH,
        num_pages,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // GraphBucket bookkeeping is keyed by the page-table length the captured
    // TileLang decode kernel walks (batch is the fixed DECODE_GRAPH_BATCH for this
    // B=1 landing). These cover the pure key derivation the bucket inserts/looks up
    // against, plus the page-boundary recapture trigger — no GPU needed. The
    // `GraphBucket` HashMap itself needs a live CudaStream and is exercised on H20.

    #[test]
    fn key_batch_is_one_and_pages_round_up() {
        // page_size=16: 0 prior tokens → total 1 token → 1 page.
        let k0 = decode_graph_key_for(16, 0);
        assert_eq!(k0.batch_size, DECODE_GRAPH_BATCH);
        assert_eq!(k0.num_pages, 1);
        // 14 prior → total 15 → still 1 page.
        assert_eq!(decode_graph_key_for(16, 14).num_pages, 1);
        // 15 prior → total 16 → exactly 1 page (no spillover yet).
        assert_eq!(decode_graph_key_for(16, 15).num_pages, 1);
    }

    #[test]
    fn page_boundary_crossing_changes_the_lookup_key() {
        // The recapture trigger: when total length crosses a page_size boundary the
        // num_pages key increments, so the bucket misses the old graph and captures
        // a new one for the longer page-table walk.
        let before = decode_graph_key_for(16, 15); // total 16 → 1 page
        let after = decode_graph_key_for(16, 16); // total 17 → 2 pages
        assert_eq!(before.num_pages, 1);
        assert_eq!(after.num_pages, 2);
        assert_ne!(
            before, after,
            "page-boundary cross must change the bucket key"
        );
    }

    #[test]
    fn same_page_run_reuses_one_key() {
        // Every step inside a 16-token page run maps to the same key, so the bucket
        // looks up (and replays) one captured graph across those steps.
        let keys: Vec<_> = (16..32).map(|n| decode_graph_key_for(16, n)).collect();
        assert!(
            keys.iter().all(|k| *k == keys[0]),
            "all steps in one page run share the bucket key: {keys:?}"
        );
        assert_eq!(keys[0].num_pages, 2);
    }

    #[test]
    fn key_is_hashmap_friendly() {
        // GraphBucket stores graphs in a HashMap keyed by num_pages; the key derives
        // deterministically, so repeated lookups for the same state hit the same slot.
        use std::collections::HashMap;
        let mut seen: HashMap<usize, usize> = HashMap::new();
        for kv_seq_len in 0..64 {
            let key = decode_graph_key_for(16, kv_seq_len);
            *seen.entry(key.num_pages).or_insert(0) += 1;
        }
        // kv_seq_len 0..=63 → total length 1..=64 → page counts 1..=4 (64/16 = 4).
        assert_eq!(seen.keys().min().copied(), Some(1));
        assert_eq!(seen.keys().max().copied(), Some(4));
        // Each of the 4 page counts covers exactly 16 of the 64 steps.
        assert!(seen.values().all(|&c| c == 16), "even split: {seen:?}");
    }
}
