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
    let out: Vec<u8> = (0..rows)
        .flat_map(|r| {
            bytes[r * row_stride + col_start..r * row_stride + col_start + col_len]
                .iter()
                .copied()
        })
        .collect();
    Ok(ShardedBytes {
        bytes: out,
        rows,
        cols: spec.size,
    })
}

/// One sub-block of a fused column-parallel projection: `heads` whole heads of
/// `head_rows` rows each. The Qwen3.5/3.6 gated-delta `in_proj_qkv` stacks
/// `[q(Kh×Kd); k(Kh×Kd); v(Vh×Vd)]` along dim 0, and its depthwise `conv1d`
/// channels mirror the same row order — a flat [`shard_column_parallel`] over
/// the fused output dim would cut across the block boundaries, so each block
/// is sharded independently on whole-head boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadBlock {
    /// Number of whole heads in this block (must divide the TP world size).
    pub heads: usize,
    /// Rows contributed by each head (e.g. `key_head_dim`, or `2*head_dim` for
    /// a gated `[query; gate]` per-head layout).
    pub head_rows: usize,
}

impl HeadBlock {
    fn rows(&self) -> usize {
        self.heads * self.head_rows
    }
}

/// Slice a fused multi-block column-parallel weight (`[Σ block rows, cols]`
/// row-major) for one TP rank: every block is head-sharded independently and
/// this rank's per-block slices are re-stacked in block order.
///
/// Rank `r` takes heads `[r·heads/world, (r+1)·heads/world)` of EVERY block,
/// preserving each block's head grouping (e.g. gated-delta k↔v head pairing).
/// Single-GPU (`world_size == 1`) is the identity slice.
///
/// # Errors
/// Errors if any block's `heads` is not divisible by the world size, the block
/// rows don't sum to the buffer's row count, or the buffer length is
/// inconsistent with `rows * cols * elem_size`.
pub fn shard_head_blocks_column_parallel(
    bytes: &[u8],
    cols: usize,
    elem_size: usize,
    blocks: &[HeadBlock],
    tp: &infer_topo::TpConfig,
) -> Result<ShardedBytes> {
    let total_rows: usize = blocks.iter().map(HeadBlock::rows).sum();
    ensure_buffer(bytes, total_rows, cols, elem_size)?;
    let world = tp.world_size;
    let row_stride = cols * elem_size;
    let mut out = Vec::new();
    let mut local_rows = 0usize;
    let mut block_start = 0usize;
    for (i, block) in blocks.iter().enumerate() {
        ensure!(
            block.heads.is_multiple_of(world),
            "fused block {i}: {} heads not divisible by world_size {world}",
            block.heads
        );
        let local_heads = block.heads / world;
        let shard_rows = local_heads * block.head_rows;
        let start = (block_start + tp.rank * shard_rows) * row_stride;
        let end = start + shard_rows * row_stride;
        out.extend_from_slice(&bytes[start..end]);
        local_rows += shard_rows;
        block_start += block.rows();
    }
    Ok(ShardedBytes {
        bytes: out,
        rows: local_rows,
        cols,
    })
}

/// Borrow expert `expert_idx`'s `[out_rows, cols]` row-major sub-block out of
/// a stacked `[num_experts, stacked_rows, cols]` expert tensor.
///
/// Row-major layout means each expert's matrix — and any whole-row range
/// inside it — is one contiguous byte block, so the returned slice borrows
/// the source with no copy. `row_offset`/`out_rows` select rows inside the
/// expert block: the fused Qwen3.6 `experts.gate_up_proj`
/// `[E, 2*moe_inter, hidden]` stores gate at rows `[0, moe_inter)` and up at
/// `[moe_inter, 2*moe_inter)`; a plain stacked tensor (e.g.
/// `experts.down_proj` `[E, hidden, moe_inter]`) is the whole block
/// (`row_offset == 0`, `out_rows == stacked_rows`).
///
/// # Errors
/// Errors if `expert_idx >= num_experts`, the row range exceeds
/// `stacked_rows`, or the buffer length is inconsistent with
/// `num_experts * stacked_rows * cols * elem_size`.
pub fn slice_stacked_expert(
    bytes: &[u8],
    num_experts: usize,
    stacked_rows: usize,
    cols: usize,
    elem_size: usize,
    expert_idx: usize,
    row_offset: usize,
    out_rows: usize,
) -> Result<&[u8]> {
    let total_rows = num_experts
        .checked_mul(stacked_rows)
        .ok_or_else(|| anyhow::anyhow!("num_experts*stacked_rows overflows"))?;
    ensure_buffer(bytes, total_rows, cols, elem_size)?;
    ensure!(
        expert_idx < num_experts,
        "stacked expert index {expert_idx} out of range (num_experts={num_experts})"
    );
    ensure!(
        row_offset + out_rows <= stacked_rows,
        "stacked expert row range [{row_offset}, {}) exceeds stacked rows {stacked_rows}",
        row_offset + out_rows
    );
    let row_stride = cols * elem_size;
    let start = (expert_idx * stacked_rows + row_offset) * row_stride;
    let end = start + out_rows * row_stride;
    Ok(&bytes[start..end])
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
    fn fused_qkv_head_blocks_shard_each_block_and_reassemble() {
        // Gated-delta in_proj_qkv shape: [q(Kh×Kd); k(Kh×Kd); v(Vh×Vd)] blocks.
        // Small analogue: Kh=4, Kd=3, Vh=8, Vd=2 → rows = 12 + 12 + 16 = 40.
        let (kh, kd, vh, vd) = (4usize, 3usize, 8usize, 2usize);
        let cols = 5usize;
        let rows = 2 * kh * kd + vh * vd;
        let w = fake_weight(rows, cols);
        let blocks = [
            HeadBlock {
                heads: kh,
                head_rows: kd,
            },
            HeadBlock {
                heads: kh,
                head_rows: kd,
            },
            HeadBlock {
                heads: vh,
                head_rows: vd,
            },
        ];

        for world in [2usize, 4] {
            let mut per_rank = Vec::new();
            for rank in 0..world {
                let tp = TpConfig::new(world, rank).unwrap();
                let r = shard_head_blocks_column_parallel(&w, cols, 2, &blocks, &tp).unwrap();
                assert_eq!(r.cols, cols);
                assert_eq!(r.rows, rows / world, "uniform per-rank row count");

                // Hand-build the expectation: rank's q rows, then k rows, then
                // v rows — each sliced inside its own block, never across the
                // block boundary.
                let q_block = 0..kh * kd;
                let k_block = kh * kd..2 * kh * kd;
                let v_block = 2 * kh * kd..rows;
                let local_k_rows = kh / world * kd;
                let local_v_rows = vh / world * vd;
                let mut want_rows = Vec::new();
                want_rows.extend(q_block.skip(rank * local_k_rows).take(local_k_rows));
                want_rows.extend(k_block.skip(rank * local_k_rows).take(local_k_rows));
                want_rows.extend(v_block.skip(rank * local_v_rows).take(local_v_rows));
                let want: Vec<u16> = want_rows
                    .iter()
                    .flat_map(|&r| (0..cols).map(move |c| (r * cols + c) as u16))
                    .collect();
                assert_eq!(decode(&r.bytes), want, "world={world} rank={rank}");
                per_rank.push(r.bytes);
            }
            // Per-block reassembly: concatenating every rank's q slices, then k
            // slices, then v slices reconstructs the original (no gap/overlap).
            let local_k_bytes = (kh / world) * kd * cols * 2;
            let local_v_bytes = (vh / world) * vd * cols * 2;
            let mut joined = Vec::new();
            for (block_idx, block_bytes) in [local_k_bytes, local_k_bytes, local_v_bytes]
                .into_iter()
                .enumerate()
            {
                for bytes in &per_rank {
                    let start: usize = [local_k_bytes, local_k_bytes, local_v_bytes][..block_idx]
                        .iter()
                        .sum();
                    joined.extend_from_slice(&bytes[start..start + block_bytes]);
                }
            }
            assert_eq!(joined, w, "world={world} blockwise concat reconstructs");
        }
    }

    #[test]
    fn gated_q_proj_head_block_shard_keeps_query_and_gate_rows_paired() {
        // The gated q_proj interleaves [query(HD); gate(HD)] PER HEAD
        // (prefill_attention_hd256.cu reads q at head*2*HD + d and the gate at
        // head*2*HD + HD + d), so the per-head row block is 2*HD. Each rank must
        // get its heads' query rows AND the matching gate rows — never a
        // contiguous slice across the q/gate boundary of a foreign head.
        let heads = 4usize;
        let hd = 3usize; // small stand-in for head_dim=256
        let cols = 2usize;
        let rows = heads * 2 * hd;
        let w = fake_weight(rows, cols);
        let blocks = [HeadBlock {
            heads,
            head_rows: 2 * hd,
        }];

        let world = 2usize;
        for rank in 0..world {
            let tp = TpConfig::new(world, rank).unwrap();
            let r = shard_head_blocks_column_parallel(&w, cols, 2, &blocks, &tp).unwrap();
            assert_eq!(r.rows, rows / world);
            let local_heads = heads / world;
            for lh in 0..local_heads {
                let global_head = rank * local_heads + lh;
                for d in 0..hd {
                    // Query row d of local head lh == global head's query row.
                    let got_q = decode(&r.bytes)[(lh * 2 * hd + d) * cols];
                    let want_q = ((global_head * 2 * hd + d) * cols) as u16;
                    assert_eq!(got_q, want_q, "rank={rank} head={lh} q row {d}");
                    // Gate row d of local head lh == global head's gate row.
                    let got_g = decode(&r.bytes)[(lh * 2 * hd + hd + d) * cols];
                    let want_g = ((global_head * 2 * hd + hd + d) * cols) as u16;
                    assert_eq!(got_g, want_g, "rank={rank} head={lh} gate row {d}");
                }
            }
        }
    }

    #[test]
    fn fused_block_shard_single_gpu_is_identity_and_indivisible_rejected() {
        let blocks = [
            HeadBlock {
                heads: 3,
                head_rows: 2,
            },
            HeadBlock {
                heads: 6,
                head_rows: 1,
            },
        ];
        let w = fake_weight(12, 4);
        let single = TpConfig::single();
        let r = shard_head_blocks_column_parallel(&w, 4, 2, &blocks, &single).unwrap();
        assert_eq!(r.bytes, w);
        assert_eq!((r.rows, r.cols), (12, 4));

        // heads=3 not divisible by world=2 → loud error, not a silent split head.
        let tp = TpConfig::new(2, 0).unwrap();
        assert!(shard_head_blocks_column_parallel(&w, 4, 2, &blocks, &tp).is_err());

        // Wrong row total (blocks don't cover the buffer) → rejected.
        let bad_blocks = [HeadBlock {
            heads: 2,
            head_rows: 2,
        }];
        assert!(shard_head_blocks_column_parallel(&w, 4, 2, &bad_blocks, &single).is_err());
    }

    // Build a fake stacked [num_experts, stacked_rows, cols] u16 (bf16-sized)
    // expert tensor where element (e, r, c) is encoded as
    // `e * stacked_rows * cols + r * cols + c` — a per-expert slice's bytes
    // are then trivially checkable.
    fn fake_stacked(num_experts: usize, stacked_rows: usize, cols: usize) -> Vec<u8> {
        fake_weight(num_experts * stacked_rows, cols)
    }

    #[test]
    fn stacked_expert_full_block_slices_tile_the_tensor() {
        // down_proj-shaped analogue: [E=4, hidden=3, mi=2], whole-block slices.
        let (e, rows, cols) = (4usize, 3usize, 2usize);
        let w = fake_stacked(e, rows, cols);
        let mut joined = Vec::new();
        for i in 0..e {
            let s = slice_stacked_expert(&w, e, rows, cols, 2, i, 0, rows).unwrap();
            // Expert i's block holds the values [i*rows*cols, (i+1)*rows*cols).
            let want: Vec<u16> = (i * rows * cols..(i + 1) * rows * cols)
                .map(|v| v as u16)
                .collect();
            assert_eq!(decode(s), want, "expert {i}");
            joined.extend_from_slice(s);
        }
        // Concatenating every expert's slice reconstructs the stacked tensor.
        assert_eq!(joined, w);
    }

    #[test]
    fn stacked_fused_gate_up_split_keeps_gate_first_row_order() {
        // gate_up_proj-shaped analogue: [E=3, 2*mi, hidden] with mi=2,
        // hidden=4. Gate = rows [0, mi) of the expert block, up = rows
        // [mi, 2*mi) — the production Qwen3.6 fused row order.
        let (e, mi, hidden) = (3usize, 2usize, 4usize);
        let stacked_rows = 2 * mi;
        let w = fake_stacked(e, stacked_rows, hidden);
        for i in 0..e {
            let gate = slice_stacked_expert(&w, e, stacked_rows, hidden, 2, i, 0, mi).unwrap();
            let up = slice_stacked_expert(&w, e, stacked_rows, hidden, 2, i, mi, mi).unwrap();
            let block_base = i * stacked_rows * hidden;
            let want_gate: Vec<u16> = (block_base..block_base + mi * hidden)
                .map(|v| v as u16)
                .collect();
            let want_up: Vec<u16> = (block_base + mi * hidden..block_base + 2 * mi * hidden)
                .map(|v| v as u16)
                .collect();
            assert_eq!(decode(gate), want_gate, "expert {i} gate rows [0, mi)");
            assert_eq!(decode(up), want_up, "expert {i} up rows [mi, 2*mi)");
            // gate ‖ up reassembles the expert's full fused block.
            let mut rejoined = gate.to_vec();
            rejoined.extend_from_slice(up);
            assert_eq!(
                rejoined,
                w[block_base * 2..(block_base + stacked_rows * hidden) * 2],
                "expert {i} gate+up == fused block"
            );
        }
    }

    #[test]
    fn stacked_expert_ep_subrange_loads_only_local_experts() {
        // EP analogue of the Qwen3.6 TP=2 split: E=8 over ep_size=4 → rank 2
        // owns experts [4, 6). The loader slices exactly that subrange.
        let split = crate::moe_config::ExpertSplit::new(8, 4, 2).unwrap();
        assert_eq!((split.local_expert_start, split.local_expert_end()), (4, 6));
        let (mi, hidden) = (2usize, 3usize);
        let stacked_rows = 2 * mi;
        let w = fake_stacked(split.num_experts, stacked_rows, hidden);
        let mut local_gates = Vec::new();
        for i in split.local_expert_start..split.local_expert_end() {
            let gate =
                slice_stacked_expert(&w, split.num_experts, stacked_rows, hidden, 2, i, 0, mi)
                    .unwrap();
            let block_base = i * stacked_rows * hidden;
            let want: Vec<u16> = (block_base..block_base + mi * hidden)
                .map(|v| v as u16)
                .collect();
            assert_eq!(decode(gate), want, "global expert {i} on ep_rank 2");
            local_gates.push(gate.to_vec());
        }
        assert_eq!(local_gates.len(), split.experts_per_rank);
        // Foreign experts' values never appear in the local slices.
        let local_lo = (split.local_expert_start * stacked_rows * hidden) as u16;
        let local_hi = (split.local_expert_end() * stacked_rows * hidden) as u16;
        for bytes in &local_gates {
            for v in decode(bytes) {
                assert!((local_lo..local_hi).contains(&v), "foreign value {v}");
            }
        }
    }

    #[test]
    fn stacked_expert_out_of_range_and_bad_buffer_rejected() {
        let w = fake_stacked(2, 4, 3);
        // expert_idx >= num_experts.
        assert!(slice_stacked_expert(&w, 2, 4, 3, 2, 2, 0, 4).is_err());
        // Row range exceeds the expert block.
        assert!(slice_stacked_expert(&w, 2, 4, 3, 2, 0, 2, 3).is_err());
        // Buffer length inconsistent with the claimed shape.
        assert!(slice_stacked_expert(&w, 2, 4, 4, 2, 0, 0, 4).is_err());
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
