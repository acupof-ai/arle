//! Pure host arithmetic for the B=1 decode-graph capture key.
//!
//! Not cuda-gated: this is plain `usize` math (no device types), so it is
//! CPU-unit-testable without nvcc even though `GraphBucket` in `decode_graph.rs`
//! is gated.

/// Decode batch size captured. B=1 is purely launch-bound, so one `cuGraphLaunch`
/// removes ~250-400 per-token `cuLaunchKernel` calls.
pub(crate) const DECODE_GRAPH_BATCH: usize = 1;

/// Identifies a captured decode graph by its baked launch shape. `batch_size` is
/// always [`DECODE_GRAPH_BATCH`]; `num_pages` is the kernel's `total_pages`
/// scalar. A graph replays only when both match.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct DecodeGraphKey {
    pub(crate) batch_size: usize,
    pub(crate) num_pages: usize,
}

/// Derive the B=1 capture key from the cache state. `kv_seq_len` is the length
/// BEFORE appending this step's token; `num_pages` is `(kv_seq_len + 1)` rounded
/// up by `page_size`, and a change in it is the recapture trigger.
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

    #[test]
    fn key_batch_is_one_and_pages_round_up() {
        let k0 = decode_graph_key_for(16, 0);
        assert_eq!(k0.batch_size, DECODE_GRAPH_BATCH);
        assert_eq!(k0.num_pages, 1);
        assert_eq!(decode_graph_key_for(16, 14).num_pages, 1);
        assert_eq!(decode_graph_key_for(16, 15).num_pages, 1);
    }

    #[test]
    fn page_boundary_crossing_changes_the_lookup_key() {
        // The recapture trigger: when total length crosses a page_size boundary the
        // num_pages key increments, so the bucket misses the old graph and captures
        // a new one for the longer page-table walk.
        let before = decode_graph_key_for(16, 15);
        let after = decode_graph_key_for(16, 16);
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
        assert_eq!(seen.keys().min().copied(), Some(1));
        assert_eq!(seen.keys().max().copied(), Some(4));
        assert!(seen.values().all(|&c| c == 16), "even split: {seen:?}");
    }
}
