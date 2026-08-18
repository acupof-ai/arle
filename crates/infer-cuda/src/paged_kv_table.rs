//! Shared host-side page-table math for paged-KV adapters.
//!
//! Pure host math, CPU-testable without nvcc (same discipline as
//! `decode_graph_key.rs`). Two consumers share these helpers:
//!
//! - **DSv4 FlashMLA arena** (#85): the FP8 MLA latent arena lives in a shared
//!   `cuda_kernels::TokenKVPool` (`KVFormat::PackedBytes`); each slot owns a
//!   PAGE TABLE mapping its slot-logical pages `[0, slot_pages)` to physical
//!   pool pages. Stage A allocates every slot's pages up-front in slot order,
//!   so tables are contiguous identity runs and the physical layout stays
//!   byte-identical to the pre-paging band arena; the device-side pack/index
//!   kernels still assume band contiguity and are gated by
//!   [`contiguous_page_table_byte_range`]'s identity check until they take a
//!   device-resident table.
//! - **Qwen quant-KV pool** (#68): the dense Qwen3 path already pages its KV
//!   through a real (non-identity) page table; the INT8/FP8 quant store kernels
//!   (`quantize_paged_kv_*_per_channel`) consume a host-built `new_token_indices`
//!   of PHYSICAL token rows, built by [`physical_token_rows`].
//!
//! NAMING RULE (ckl 2026-06-11): the per-page token unit is a "page" everywhere
//! in this codebase; "block" appears ONLY at the FlashMLA FFI boundary, whose
//! API calls our page a block (`block_table`, `page_block_size`).

use anyhow::{Result, anyhow, ensure};

/// Physical pool page backing `logical_page` of one slot's page table.
///
/// Table-routing invariant: callers hand the kernel this PHYSICAL page id (and
/// the pool BASE pointer), never a `slot_idx`-derived band offset, so the call
/// site stays valid when the table fragments (true paging).
pub(crate) fn physical_page(table: &[u32], logical_page: usize) -> Result<u32> {
    table.get(logical_page).copied().ok_or_else(|| {
        anyhow!(
            "paged-KV logical page {logical_page} outside slot page table len {}",
            table.len()
        )
    })
}

/// Physical token rows for `num_tokens` new tokens starting at slot-logical
/// position `start_pos`, for the quant store kernels' `new_token_indices`.
///
/// The `quantize_paged_kv_*_per_channel` kernels assume an IDENTITY page map
/// (`page_idx = token_row / page_size`), so a non-identity Qwen page table must
/// be flattened to PHYSICAL rows here: for each logical position `p`,
/// `physical_row = table[p / page_size] * page_size + (p % page_size)`. Feeding
/// the kernel these rows makes its identity arithmetic land in the right
/// physical slot. Returns `i32` rows (the kernel's index type).
// Consumer is the #68 T3 quant-KV store glue in `executor`/`attention` (pod
// build): the CPU-tested host helper lands ahead of its cuda-only caller.
#[allow(dead_code)]
pub(crate) fn physical_token_rows(
    table: &[u32],
    page_size: usize,
    start_pos: usize,
    num_tokens: usize,
) -> Result<Vec<i32>> {
    ensure!(page_size > 0, "paged-KV page_size must be non-zero");
    let rows = (0..num_tokens)
        .map(|offset| {
            let pos = start_pos + offset;
            let physical = physical_page(table, pos / page_size)? as usize;
            let row = physical
                .checked_mul(page_size)
                .and_then(|base| base.checked_add(pos % page_size))
                .ok_or_else(|| anyhow!("paged-KV physical token row overflow at pos {pos}"))?;
            i32::try_from(row).map_err(|_| anyhow!("paged-KV physical token row {row} exceeds i32"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(rows)
}

/// Byte range of the contiguous physical run covered by one slot's page
/// table.
///
/// Semantic honesty (codex review on 9d63682d): this proves CONTIGUITY —
/// which is what licenses band-base addressing — not identity placement.
/// The Stage A byte-identical-to-band claim is proven separately at pool
/// construction (`first page == slot × slot_pages`, ensure!d per slot).
///
/// Errors when the table length differs from the slot's expected page count
/// or the run has a gap: a gapped table is valid paging (Stage B), but it
/// means a caller still using contiguous band-base addressing (the device-side
/// pack/index kernels) can no longer do so - fail loudly instead of aliasing
/// another slot's pages.
pub(crate) fn contiguous_page_table_byte_range(
    table: &[u32],
    expected_pages: usize,
    page_bytes: usize,
) -> Result<std::ops::Range<usize>> {
    ensure!(
        table.len() == expected_pages && expected_pages > 0,
        "paged-KV page table len {} != expected slot pages {expected_pages}",
        table.len()
    );
    ensure!(page_bytes > 0, "paged-KV page_bytes must be non-zero");
    let first = table[0] as usize;
    for (logical, &page) in table.iter().enumerate() {
        ensure!(
            page as usize == first + logical,
            "paged-KV page table is not a contiguous identity run \
             (logical {logical} -> physical {page}, run starts at {first}); \
             band-base addressing cannot serve a fragmented table"
        );
    }
    let start = first
        .checked_mul(page_bytes)
        .ok_or_else(|| anyhow!("paged-KV page table byte offset overflow"))?;
    let end = (first + expected_pages)
        .checked_mul(page_bytes)
        .ok_or_else(|| anyhow!("paged-KV page table byte end overflow"))?;
    Ok(start..end)
}
