//! Per-rank weight-shard byte slicing (pure-CPU, GPU-testable).
//!
//! Slices a host 2D safetensors weight (`[rows, cols]`, row-major) to one rank's
//! [`infer_topo::ShardingSpec`] before upload. HF `nn.Linear` layout: dim 0 =
//! `out_features`, dim 1 = `in_features`.
//!
//! - **Column-parallel** (`q/k/v/gate/up`): split rows → one contiguous range.
//! - **Row-parallel** (`o/down`): split cols → strided (gathered per row).

use anyhow::{Result, ensure};
use infer_topo::ShardingSpec;

/// A host-side 2D weight slice: the sliced bytes plus the new `[rows, cols]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedBytes {
    /// The sliced weight bytes (row-major, same element size as the source).
    pub bytes: Vec<u8>,
    /// Number of rows (`out_features`) after slicing.
    pub rows: usize,
    /// Number of columns (`in_features`) after slicing.
    pub cols: usize,
}

/// Slice the output dim (rows) of a `[rows, cols]` row-major weight for a
/// column-parallel layer.
///
/// `spec` shards the row count; the result is the contiguous block of whole rows
/// `spec.offset..spec.offset+spec.size`.
///
/// # Errors
/// Errors if `spec.total != rows`, the shard range exceeds `rows`, or the byte
/// buffer length is inconsistent with `rows * cols * elem_size`.
pub fn shard_column_parallel(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    elem_size: usize,
    spec: &ShardingSpec,
) -> Result<ShardedBytes> {
    ensure_buffer(bytes, rows, cols, elem_size)?;
    ensure!(
        spec.total == rows,
        "column-parallel shard total {} must equal out_features (rows) {rows}",
        spec.total
    );
    ensure!(
        spec.end() <= rows,
        "column-parallel shard range {:?} exceeds rows {rows}",
        spec.range()
    );
    let row_stride = cols * elem_size;
    let start = spec.offset * row_stride;
    let end = spec.end() * row_stride;
    Ok(ShardedBytes {
        bytes: bytes[start..end].to_vec(),
        rows: spec.size,
        cols,
    })
}

/// Slice the input dim (columns) of a `[rows, cols]` row-major weight for a
/// row-parallel layer.
///
/// `spec` shards the column count; the result gathers columns
/// `spec.offset..spec.offset+spec.size` from every one of the `rows` rows.
///
/// # Errors
/// Errors if `spec.total != cols`, the shard range exceeds `cols`, or the byte
/// buffer length is inconsistent with `rows * cols * elem_size`.
pub fn shard_row_parallel(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    elem_size: usize,
    spec: &ShardingSpec,
) -> Result<ShardedBytes> {
    ensure_buffer(bytes, rows, cols, elem_size)?;
    ensure!(
        spec.total == cols,
        "row-parallel shard total {} must equal in_features (cols) {cols}",
        spec.total
    );
    ensure!(
        spec.end() <= cols,
        "row-parallel shard range {:?} exceeds cols {cols}",
        spec.range()
    );
    let row_stride = cols * elem_size;
    let col_start = spec.offset * elem_size;
    let col_len = spec.size * elem_size;
    let mut out = Vec::with_capacity(rows * col_len);
    for r in 0..rows {
        let row_base = r * row_stride + col_start;
        out.extend_from_slice(&bytes[row_base..row_base + col_len]);
    }
    Ok(ShardedBytes {
        bytes: out,
        rows,
        cols: spec.size,
    })
}

fn ensure_buffer(bytes: &[u8], rows: usize, cols: usize, elem_size: usize) -> Result<()> {
    ensure!(elem_size > 0, "elem_size must be > 0");
    let expected = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(elem_size))
        .ok_or_else(|| anyhow::anyhow!("rows*cols*elem_size overflows"))?;
    ensure!(
        bytes.len() == expected,
        "buffer length {} != rows*cols*elem_size {expected} ({rows}x{cols}x{elem_size})",
        bytes.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use infer_topo::{TpConfig, column_shard, row_shard};

    // Build a fake [rows, cols] u16 (bf16-sized) weight where element (r, c) is
    // encoded as `r * cols + c`, so a slice's bytes are trivially checkable.
    fn fake_weight(rows: usize, cols: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(rows * cols * 2);
        for r in 0..rows {
            for c in 0..cols {
                let v = (r * cols + c) as u16;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        bytes
    }

    fn decode(bytes: &[u8]) -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect()
    }

    #[test]
    fn column_parallel_slices_contiguous_rows_per_rank() {
        // [8, 4] weight, column-parallel over TP=2 → each rank owns 4 whole rows.
        let rows = 8;
        let cols = 4;
        let w = fake_weight(rows, cols);

        let tp0 = TpConfig::new(2, 0).unwrap();
        let s0 = column_shard(rows, &tp0);
        let r0 = shard_column_parallel(&w, rows, cols, 2, &s0).unwrap();
        assert_eq!(r0.rows, 4);
        assert_eq!(r0.cols, 4);
        // Rank 0 owns rows 0..4 → values 0..16.
        assert_eq!(decode(&r0.bytes), (0u16..16).collect::<Vec<_>>());

        let tp1 = TpConfig::new(2, 1).unwrap();
        let s1 = column_shard(rows, &tp1);
        let r1 = shard_column_parallel(&w, rows, cols, 2, &s1).unwrap();
        assert_eq!(r1.rows, 4);
        // Rank 1 owns rows 4..8 → values 16..32.
        assert_eq!(decode(&r1.bytes), (16u16..32).collect::<Vec<_>>());

        // The two shards reassemble the original exactly.
        let mut joined = r0.bytes.clone();
        joined.extend_from_slice(&r1.bytes);
        assert_eq!(joined, w);
    }

    #[test]
    fn row_parallel_gathers_strided_columns_per_rank() {
        // [4, 8] weight, row-parallel over TP=2 → each rank owns 4 columns of all rows.
        let rows = 4;
        let cols = 8;
        let w = fake_weight(rows, cols);

        let tp0 = TpConfig::new(2, 0).unwrap();
        let s0 = row_shard(cols, &tp0);
        let r0 = shard_row_parallel(&w, rows, cols, 2, &s0).unwrap();
        assert_eq!(r0.rows, 4);
        assert_eq!(r0.cols, 4);
        // Rank 0 owns cols 0..4 of every row: row r → r*8 + (0..4).
        let want0: Vec<u16> = (0..rows)
            .flat_map(|r| (0..4).map(move |c| (r * cols + c) as u16))
            .collect();
        assert_eq!(decode(&r0.bytes), want0);

        let tp1 = TpConfig::new(2, 1).unwrap();
        let s1 = row_shard(cols, &tp1);
        let r1 = shard_row_parallel(&w, rows, cols, 2, &s1).unwrap();
        assert_eq!(r1.cols, 4);
        // Rank 1 owns cols 4..8 of every row: row r → r*8 + (4..8).
        let want1: Vec<u16> = (0..rows)
            .flat_map(|r| (4..8).map(move |c| (r * cols + c) as u16))
            .collect();
        assert_eq!(decode(&r1.bytes), want1);
    }

    #[test]
    fn single_gpu_column_shard_is_identity() {
        let w = fake_weight(6, 3);
        let tp = TpConfig::single();
        let s = column_shard(6, &tp);
        let r = shard_column_parallel(&w, 6, 3, 2, &s).unwrap();
        assert_eq!(r.bytes, w);
        assert_eq!((r.rows, r.cols), (6, 3));
    }

    // Head-aligned Q/K/V column shard (GQA invariant): shards land on whole-head
    // boundaries. These prove the shards tile the projection with no gap/overlap
    // and never split a head.
    fn head_aligned_spec(
        local_heads: usize,
        head_dim: usize,
        total_heads: usize,
        rank: usize,
    ) -> ShardingSpec {
        let local_rows = local_heads * head_dim;
        ShardingSpec {
            offset: rank * local_rows,
            size: local_rows,
            total: total_heads * head_dim,
        }
    }

    #[test]
    fn qkv_head_aligned_shard_tiles_full_projection_on_head_boundaries() {
        use infer_topo::head_shard;
        // Qwen3-ish Q: 32 heads, head_dim 128 → q_proj rows = 4096, cols = hidden.
        let total_heads = 32usize;
        let head_dim = 128usize;
        let rows = total_heads * head_dim; // 4096
        let cols = 8usize; // small fake hidden for a cheap byte check
        let w = fake_weight(rows, cols);

        for world in [2usize, 4, 8] {
            let mut joined = Vec::new();
            let mut covered_heads = 0usize;
            for rank in 0..world {
                let tp = TpConfig::new(world, rank).unwrap();
                // head_shard is the GQA-aware split the loader uses.
                let (local_q_heads, _local_kv) = head_shard(total_heads, 8, &tp).unwrap();
                let spec = head_aligned_spec(local_q_heads, head_dim, total_heads, rank);
                // Boundary is a whole-head multiple (no head ever split across ranks).
                assert_eq!(
                    spec.offset % head_dim,
                    0,
                    "world={world} rank={rank} offset on-head"
                );
                assert_eq!(
                    spec.size % head_dim,
                    0,
                    "world={world} rank={rank} size on-head"
                );
                let r = shard_column_parallel(&w, rows, cols, 2, &spec).unwrap();
                covered_heads += local_q_heads;
                joined.extend_from_slice(&r.bytes);
            }
            // Every head accounted for exactly once; concatenation == original.
            assert_eq!(covered_heads, total_heads, "world={world} head coverage");
            assert_eq!(joined, w, "world={world} concat reconstructs full q_proj");
        }
    }

    #[test]
    fn single_gpu_head_aligned_shard_is_identity() {
        // world_size == 1 must be byte-identical to the full load (no-op path).
        let total_heads = 16usize;
        let head_dim = 128usize;
        let rows = total_heads * head_dim;
        let cols = 4usize;
        let w = fake_weight(rows, cols);
        let tp = TpConfig::single();
        let spec = head_aligned_spec(total_heads, head_dim, total_heads, tp.rank);
        let r = shard_column_parallel(&w, rows, cols, 2, &spec).unwrap();
        assert_eq!(r.bytes, w);
        assert_eq!((r.rows, r.cols), (rows, cols));
    }

    #[test]
    fn column_shard_remainder_lands_on_last_rank() {
        // 10 rows over TP=4: ranks get 2,2,2,4. Concatenation reproduces the source.
        let rows = 10;
        let cols = 2;
        let w = fake_weight(rows, cols);
        let mut joined = Vec::new();
        let mut total_rows = 0;
        for rank in 0..4 {
            let tp = TpConfig::new(4, rank).unwrap();
            let s = column_shard(rows, &tp);
            let r = shard_column_parallel(&w, rows, cols, 2, &s).unwrap();
            total_rows += r.rows;
            joined.extend_from_slice(&r.bytes);
        }
        assert_eq!(total_rows, rows);
        assert_eq!(joined, w);
    }

    #[test]
    fn mismatched_total_is_rejected() {
        let w = fake_weight(8, 4);
        // column_shard sized for a different dimension → total mismatch.
        let bad = ShardingSpec {
            offset: 0,
            size: 4,
            total: 7,
        };
        assert!(shard_column_parallel(&w, 8, 4, 2, &bad).is_err());
    }

    #[test]
    fn wrong_buffer_length_is_rejected() {
        let w = fake_weight(8, 4);
        let tp = TpConfig::new(2, 0).unwrap();
        let s = column_shard(8, &tp);
        // Claim the wrong cols → buffer-length check fails.
        assert!(shard_column_parallel(&w, 8, 5, 2, &s).is_err());
    }
}
