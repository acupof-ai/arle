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

