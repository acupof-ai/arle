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
    let mut rows = Vec::with_capacity(num_tokens);
    for offset in 0..num_tokens {
        let pos = start_pos + offset;
        let logical_page = pos / page_size;
        let token_in_page = pos % page_size;
        let physical = physical_page(table, logical_page)? as usize;
        let row = physical
            .checked_mul(page_size)
            .and_then(|base| base.checked_add(token_in_page))
            .ok_or_else(|| anyhow!("paged-KV physical token row overflow at pos {pos}"))?;
        rows.push(
            i32::try_from(row)
                .map_err(|_| anyhow!("paged-KV physical token row {row} exceeds i32"))?,
        );
    }
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

#[cfg(test)]
mod tests {
    use super::{contiguous_page_table_byte_range, physical_page, physical_token_rows};

    /// Synthetic Stage A config: 3 slots x 5 pages of 64 x 584 B each.
    const PAGE_BYTES: usize = 64 * 584;
    const SLOT_PAGES: usize = 5;
    const NUM_SLOTS: usize = 3;

    fn identity_table(slot: usize) -> Vec<u32> {
        let first = (slot * SLOT_PAGES) as u32;
        (first..first + SLOT_PAGES as u32).collect()
    }

    /// Stage A identity property: the table-routed byte range equals the
    /// pre-paging band arithmetic (`slot_idx * slot_bytes ..`).
    #[test]
    fn identity_table_range_matches_band_arithmetic() {
        let slot_bytes = SLOT_PAGES * PAGE_BYTES;
        for slot in 0..NUM_SLOTS {
            let table = identity_table(slot);
            let range = contiguous_page_table_byte_range(&table, SLOT_PAGES, PAGE_BYTES)
                .expect("identity run");
            assert_eq!(range, slot * slot_bytes..(slot + 1) * slot_bytes);
        }
    }

    /// Stage A identity property for per-page translation: the physical page
    /// is the slot-logical page plus the slot's first page.
    #[test]
    fn identity_physical_page_matches_band_arithmetic() {
        for slot in 0..NUM_SLOTS {
            let table = identity_table(slot);
            for logical in 0..SLOT_PAGES {
                assert_eq!(
                    physical_page(&table, logical).expect("in range") as usize,
                    slot * SLOT_PAGES + logical
                );
            }
        }
    }

    #[test]
    fn physical_page_rejects_out_of_range_logical_page() {
        let table = identity_table(1);
        let err = physical_page(&table, SLOT_PAGES).unwrap_err().to_string();
        assert!(err.contains("outside slot page table"), "got: {err}");
    }

    #[test]
    fn gapped_table_is_rejected_for_band_addressing() {
        // Valid Stage B paging (pages 0,1,7,8,9), invalid for band-base math.
        let table = vec![0, 1, 7, 8, 9];
        let err = contiguous_page_table_byte_range(&table, SLOT_PAGES, PAGE_BYTES)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not a contiguous identity run"), "got: {err}");
    }

    #[test]
    fn wrong_table_length_is_rejected() {
        let table = identity_table(0);
        let err = contiguous_page_table_byte_range(&table, SLOT_PAGES + 1, PAGE_BYTES)
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected slot pages"), "got: {err}");
        let err = contiguous_page_table_byte_range(&[], 0, PAGE_BYTES)
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected slot pages"), "got: {err}");
    }

    /// Identity table (Qwen Stage-A-equivalent): physical rows are just the
    /// contiguous logical rows — `page * page_size + in_page`.
    #[test]
    fn physical_token_rows_identity_is_contiguous() {
        let page_size = 16;
        // 3 identity pages → logical rows 0..48 map to physical 0..48.
        let table: Vec<u32> = (0..3).collect();
        let rows = physical_token_rows(&table, page_size, 5, 20).expect("rows");
        let expected: Vec<i32> = (5..25).collect();
        assert_eq!(rows, expected);
    }

    /// Non-identity (fragmented) table: physical rows follow the table's
    /// physical pages, NOT the logical positions — the whole point of the
    /// flattening for the identity-assuming quant kernel.
    #[test]
    fn physical_token_rows_fragmented_follows_table() {
        let page_size = 16;
        // Logical pages 0,1,2 → physical pages 7,3,9.
        let table = vec![7u32, 3, 9];
        // Tokens at logical pos 14,15 (page 0 → phys 7) and 16,17 (page 1 → phys 3).
        let rows = physical_token_rows(&table, page_size, 14, 4).expect("rows");
        assert_eq!(
            rows,
            vec![
                7 * 16 + 14,
                7 * 16 + 15,
                3 * 16, // pos 16 → logical page 1 → phys 3, in-page 0
                3 * 16 + 1,
            ]
        );
    }

    #[test]
    fn physical_token_rows_rejects_position_past_table() {
        let page_size = 16;
        let table = vec![0u32, 1]; // logical pages 0,1 only (positions 0..32)
        let err = physical_token_rows(&table, page_size, 30, 5)
            .unwrap_err()
            .to_string();
        assert!(err.contains("outside slot page table"), "got: {err}");
    }

    #[test]
    fn physical_token_rows_rejects_zero_page_size() {
        let err = physical_token_rows(&[0u32], 0, 0, 1)
            .unwrap_err()
            .to_string();
        assert!(err.contains("page_size must be non-zero"), "got: {err}");
    }
}
