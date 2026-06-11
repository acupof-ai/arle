//! Host-side page-table math for the DSv4 paged-KV adapter (#85 P2).
//!
//! Pure host math, CPU-testable without nvcc (same discipline as
//! `decode_graph_key.rs`). The DSv4 FP8 MLA latent arena lives in a shared
//! `cuda_kernels::TokenKVPool` (`KVFormat::PackedBytes`); each slot owns a
//! PAGE TABLE mapping its slot-logical pages `[0, slot_pages)` to physical
//! pool pages. NAMING RULE (ckl 2026-06-11): the 64-token unit is a "page"
//! everywhere in this codebase; "block" appears ONLY at the FlashMLA FFI
//! boundary, whose API calls our page a block (`block_table`,
//! `page_block_size`). Stage A (#85 P2)
//! allocates every slot's pages up-front in slot order, so tables are
//! contiguous identity runs and the physical layout stays byte-identical to
//! the pre-paging band arena. These helpers are the single place a table is
//! turned into byte offsets, so Stage B (true paging) changes table CONTENTS,
//! not the consumers - except the device-side pack/index kernels, which still
//! assume band contiguity and are gated by [`contiguous_page_table_byte_range`]'s
//! identity check until they take a device-resident table.

use anyhow::{Result, anyhow, ensure};

/// Physical pool page backing `logical_page` of one slot's page table.
///
/// Table-routing invariant: callers hand the kernel this PHYSICAL page id (and
/// the pool BASE pointer), never a `slot_idx`-derived band offset, so the call
/// site stays valid when Stage B fragments the table.
pub(crate) fn physical_page(table: &[u32], logical_page: usize) -> Result<u32> {
    table.get(logical_page).copied().ok_or_else(|| {
        anyhow!(
            "DSv4 FlashMLA logical page {logical_page} outside slot page table len {}",
            table.len()
        )
    })
}

/// Byte range of the contiguous physical run covered by one slot's block
/// table (#85 P2 Stage A identity layout).
///
/// Errors when the table length differs from the slot's expected block count
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
        "DSv4 FlashMLA page table len {} != expected slot pages {expected_pages}",
        table.len()
    );
    ensure!(page_bytes > 0, "DSv4 FlashMLA page_bytes must be non-zero");
    let first = table[0] as usize;
    for (logical, &page) in table.iter().enumerate() {
        ensure!(
            page as usize == first + logical,
            "DSv4 FlashMLA page table is not a contiguous identity run \
             (logical {logical} -> physical {page}, run starts at {first}); \
             Stage A band-base addressing cannot serve a fragmented table"
        );
    }
    let start = first
        .checked_mul(page_bytes)
        .ok_or_else(|| anyhow!("DSv4 FlashMLA page table byte offset overflow"))?;
    let end = (first + expected_pages)
        .checked_mul(page_bytes)
        .ok_or_else(|| anyhow!("DSv4 FlashMLA page table byte end overflow"))?;
    Ok(start..end)
}

#[cfg(test)]
mod tests {
    use super::{contiguous_page_table_byte_range, physical_page};

    /// Synthetic Stage A config: 3 slots x 5 blocks of 64 x 584 B pages.
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

    /// Stage A identity property for per-block translation: the physical page
    /// is the band-relative block plus the slot's first page.
    #[test]
    fn identity_physical_block_matches_band_arithmetic() {
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
    fn physical_block_rejects_out_of_range_logical_block() {
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
}
