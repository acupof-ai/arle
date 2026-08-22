//! Slices a host 2D safetensors weight (`[rows, cols]`, row-major) to one rank's
//! [`infer_topo::ShardingSpec`] before upload. HF `nn.Linear` layout: dim 0 =
//! `out_features`, dim 1 = `in_features`.
//!
//! - **Column-parallel** (`q/k/v/gate/up`): split rows → one contiguous range.
//! - **Row-parallel** (`o/down`): split cols → strided (gathered per row).

use anyhow::{Result, ensure};
use infer_topo::ShardingSpec;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedBytes {
    pub bytes: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
}

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
