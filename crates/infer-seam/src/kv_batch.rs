//! Host-only KV batch descriptors.
//!
//! The engine builds this from a [`ForwardPlan`](infer_plan::ForwardPlan) and a
//! [`KvPool`](crate::KvPool) after logical KV allocation. Backends lower the
//! descriptor into device-specific page tables and model-specific KV views.

use std::ops::Range;

use anyhow::{Result, ensure};
use infer_plan::{DecodeRow, ForwardMode, ForwardPlan, PrefillRow};

use crate::KvPool;

/// Host-only batch view over the rows scheduled in one forward step.
#[derive(Debug, Clone)]
pub struct KvBatchDescriptor {
    pub mode: ForwardMode,
    /// Per-row metadata in commit order: prefill rows first, then decode rows.
    pub rows: Vec<KvBatchRow>,
    /// Flattened row input tokens. Rows point into this buffer with
    /// [`KvBatchRow::token_range`].
    pub flat_token_ids: Vec<u32>,
    /// Flattened logical page ids. Rows point into this buffer with
    /// [`KvBatchRow::page_range`].
    pub flat_page_ids: Vec<u32>,
    /// Flattened whole-slot page tables. Rows point into this buffer with
    /// [`KvBatchRow::slot_page_range`]. Sequential models usually read only
    /// `page_range`; fixed-band models (DSv4) need the full slot table.
    pub flat_slot_page_ids: Vec<u32>,
}

/// One row in a [`KvBatchDescriptor`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvBatchRow {
    pub slot: usize,
    pub kind: KvBatchRowKind,
    pub seq_len: usize,
    pub append_pos: usize,
    pub append_len: usize,
    /// The request's KNOWN final logical length for this phase (prefill: the
    /// full prompt; decode: `append_pos + append_len`). Demand-paged backends
    /// reserve device pages for the whole known span at the first chunk
    /// instead of growing per chunk (#154 Phase 3b).
    pub total_tokens: usize,
    pub slot_epoch: u64,
    pub token_range: Range<usize>,
    pub page_range: Range<usize>,
    pub slot_page_range: Range<usize>,
}

/// Forward row kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvBatchRowKind {
    Prefill,
    Decode,
}

impl KvBatchDescriptor {
    /// Build a host-only descriptor for `plan`.
    ///
    /// The engine calls this after [`KvPool::alloc`](crate::KvAllocator::alloc)
    /// has reserved the row's output span, so the pool length must cover
    /// `append_pos + append_len`.
    pub fn from_plan(plan: &ForwardPlan, kv: &dyn KvPool) -> Result<Self> {
        let mut desc = Self {
            mode: plan.mode.clone(),
            rows: Vec::with_capacity(plan.prefill_rows.len() + plan.decode_rows.len()),
            flat_token_ids: Vec::new(),
            flat_page_ids: Vec::new(),
            flat_slot_page_ids: Vec::new(),
        };

        for row in &plan.prefill_rows {
            desc.push_prefill(row, kv)?;
        }
        for row in &plan.decode_rows {
            desc.push_decode(row, kv)?;
        }

        Ok(desc)
    }

    fn push_prefill(&mut self, row: &PrefillRow, kv: &dyn KvPool) -> Result<()> {
        self.push_row(
            row.slot,
            KvBatchRowKind::Prefill,
            row.start_pos,
            row.start_pos,
            row.tokens.len(),
            row.total_tokens,
            &row.tokens,
            kv,
        )
    }

    fn push_decode(&mut self, row: &DecodeRow, kv: &dyn KvPool) -> Result<()> {
        self.push_row(
            row.slot,
            KvBatchRowKind::Decode,
            row.kv_seq_len,
            row.kv_seq_len,
            1,
            row.kv_seq_len + 1,
            &[row.last_token],
            kv,
        )
    }

    /// Re-rooted sub-descriptor over `rows[range]`, with token/page ranges
    /// rebased so the result is indistinguishable from a descriptor built for
    /// just those rows. Backends that split a mixed plan into per-mode
    /// sub-steps use this to keep single-mode validation invariants (e.g.
    /// "a prefill descriptor has exactly one row") intact.
    pub fn subset(&self, range: Range<usize>) -> Result<Self> {
        ensure!(
            range.start < range.end && range.end <= self.rows.len(),
            "KV batch subset {range:?} outside {} rows",
            self.rows.len()
        );
        let mut desc = Self {
            mode: ForwardMode::Idle,
            rows: Vec::with_capacity(range.len()),
            flat_token_ids: Vec::new(),
            flat_page_ids: Vec::new(),
            flat_slot_page_ids: Vec::new(),
        };
        let mut has_prefill = false;
        let mut has_decode = false;
        for row in &self.rows[range] {
            match row.kind {
                KvBatchRowKind::Prefill => has_prefill = true,
                KvBatchRowKind::Decode => has_decode = true,
            }
            let token_start = desc.flat_token_ids.len();
            desc.flat_token_ids
                .extend_from_slice(&self.flat_token_ids[row.token_range.clone()]);
            let page_start = desc.flat_page_ids.len();
            desc.flat_page_ids
                .extend_from_slice(&self.flat_page_ids[row.page_range.clone()]);
            let slot_page_start = desc.flat_slot_page_ids.len();
            desc.flat_slot_page_ids
                .extend_from_slice(&self.flat_slot_page_ids[row.slot_page_range.clone()]);
            desc.rows.push(KvBatchRow {
                token_range: token_start..desc.flat_token_ids.len(),
                page_range: page_start..desc.flat_page_ids.len(),
                slot_page_range: slot_page_start..desc.flat_slot_page_ids.len(),
                ..row.clone()
            });
        }
        desc.mode = match (has_prefill, has_decode) {
            (true, false) => ForwardMode::Prefill,
            (false, true) => ForwardMode::Decode,
            _ => ForwardMode::Mixed,
        };
        Ok(desc)
    }

    #[allow(clippy::too_many_arguments)]
    fn push_row(
        &mut self,
        slot: usize,
        kind: KvBatchRowKind,
        seq_len: usize,
        append_pos: usize,
        append_len: usize,
        total_tokens: usize,
        tokens: &[u32],
        kv: &dyn KvPool,
    ) -> Result<()> {
        ensure!(
            append_len == tokens.len(),
            "KV batch row append_len {append_len} != token count {}",
            tokens.len()
        );
        let append_end = append_pos
            .checked_add(append_len)
            .ok_or_else(|| anyhow::anyhow!("KV batch append span overflow"))?;
        let materialized = kv.seq_len(slot);
        ensure!(
            materialized >= append_end,
            "KV batch row slot {slot} needs materialized len {append_end}, got {materialized}"
        );

        let token_start = self.flat_token_ids.len();
        self.flat_token_ids.extend_from_slice(tokens);
        let token_range = token_start..self.flat_token_ids.len();

        let page_start = self.flat_page_ids.len();
        if append_end > 0 {
            self.flat_page_ids
                .extend_from_slice(kv.page_indices_for_token_range(slot, 0, append_end));
        }
        let page_range = page_start..self.flat_page_ids.len();
        let slot_page_start = self.flat_slot_page_ids.len();
        self.flat_slot_page_ids
            .extend_from_slice(kv.page_indices(slot));
        let slot_page_range = slot_page_start..self.flat_slot_page_ids.len();

        self.rows.push(KvBatchRow {
            slot,
            kind,
            seq_len,
            append_pos,
            append_len,
            total_tokens,
            slot_epoch: kv.slot_epoch(slot),
            token_range,
            page_range,
            slot_page_range,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Result, bail};
    use infer_plan::{DecodeRow, ForwardMode, ForwardPlan, PrefillRow, SamplingParams};

    use super::*;
    use crate::{KvAllocator, KvPrefixStore, KvQuery};

    #[derive(Debug, Clone)]
    struct MockSlot {
        seq_len: usize,
        epoch: u64,
        pages: Vec<u32>,
    }

    #[derive(Debug, Clone)]
    struct MockKvPool {
        page_size: usize,
        slots: Vec<MockSlot>,
    }

    impl MockKvPool {
        fn new(page_size: usize, slots: Vec<MockSlot>) -> Self {
            Self { page_size, slots }
        }
    }

    impl KvQuery for MockKvPool {
        fn is_active(&self) -> bool {
            true
        }

        fn page_size(&self) -> usize {
            self.page_size
        }

        fn free_pages(&self) -> usize {
            128
        }

        fn free_tokens(&self) -> usize {
            128 * self.page_size
        }

        fn seq_len(&self, slot: usize) -> usize {
            self.slots[slot].seq_len
        }

        fn slot_epoch(&self, slot: usize) -> u64 {
            self.slots[slot].epoch
        }

        fn append_pages_needed(&self, _slot: usize, _tokens: usize) -> usize {
            0
        }

        fn page_indices(&self, slot: usize) -> &[u32] {
            &self.slots[slot].pages
        }

        fn page_indices_for_token_range(&self, slot: usize, start: usize, len: usize) -> &[u32] {
            if len == 0 {
                return &self.slots[slot].pages[0..0];
            }
            let first = start / self.page_size;
            let last = (start + len - 1) / self.page_size;
            &self.slots[slot].pages[first..=last]
        }
    }

    impl KvAllocator for MockKvPool {
        fn alloc(&mut self, _slot: usize, _tokens: usize) -> Result<()> {
            Ok(())
        }

        fn alloc_detached_pages(&mut self, pages: usize) -> Result<Vec<u32>> {
            Ok((0..pages as u32).collect())
        }

        fn free_detached_pages(&mut self, _pages: &[u32]) {}

        fn free_slot(&mut self, _slot: usize) {}

        fn truncate_slot(&mut self, slot: usize, new_len: usize) -> Result<()> {
            let Some(slot) = self.slots.get_mut(slot) else {
                bail!("bad slot")
            };
            slot.seq_len = new_len;
            slot.epoch += 1;
            Ok(())
        }

        fn migrate(&mut self, _slot: usize, _start: usize, _len: usize) -> Result<()> {
            Ok(())
        }
    }

    impl KvPrefixStore for MockKvPool {
        fn retain_pages(&mut self, _pages: &[u32]) {}

        fn release_pages(&mut self, _pages: &[u32]) {}

        fn retained_count(&self) -> usize {
            0
        }

        fn attach_pages(
            &mut self,
            _slot: usize,
            _pages: &[u32],
            _token_count: usize,
        ) -> Result<()> {
            Ok(())
        }
    }

    fn params() -> SamplingParams {
        SamplingParams::default()
    }

    #[test]
    fn decode_descriptor_flattens_n_rows_pages_and_epochs() {
        let kv = MockKvPool::new(
            4,
            vec![
                MockSlot {
                    seq_len: 5,
                    epoch: 7,
                    pages: vec![10, 11],
                },
                MockSlot {
                    seq_len: 9,
                    epoch: 8,
                    pages: vec![20, 21, 22],
                },
            ],
        );
        let plan = ForwardPlan {
            mode: ForwardMode::Decode,
            decode_rows: vec![
                DecodeRow {
                    slot: 0,
                    last_token: 101,
                    kv_seq_len: 4,
                    params: params(),
                },
                DecodeRow {
                    slot: 1,
                    last_token: 202,
                    kv_seq_len: 8,
                    params: params(),
                },
            ],
            prefill_rows: Vec::new(),
            microbatch: None,
            spec: None,
        };

        let desc = KvBatchDescriptor::from_plan(&plan, &kv).unwrap();
        assert_eq!(desc.flat_token_ids, vec![101, 202]);
        assert_eq!(desc.flat_page_ids, vec![10, 11, 20, 21, 22]);
        assert_eq!(desc.rows.len(), 2);
        assert_eq!(desc.rows[0].kind, KvBatchRowKind::Decode);
        assert_eq!(desc.rows[0].slot, 0);
        assert_eq!(desc.rows[0].seq_len, 4);
        assert_eq!(desc.rows[0].append_pos, 4);
        assert_eq!(desc.rows[0].append_len, 1);
        assert_eq!(desc.rows[0].slot_epoch, 7);
        assert_eq!(desc.rows[0].token_range, 0..1);
        assert_eq!(desc.rows[0].page_range, 0..2);
        assert_eq!(desc.rows[0].slot_page_range, 0..2);
        assert_eq!(desc.rows[1].slot, 1);
        assert_eq!(desc.rows[1].seq_len, 8);
        assert_eq!(desc.rows[1].append_pos, 8);
        assert_eq!(desc.rows[1].append_len, 1);
        assert_eq!(desc.rows[1].slot_epoch, 8);
        assert_eq!(desc.rows[1].token_range, 1..2);
        assert_eq!(desc.rows[1].page_range, 2..5);
        assert_eq!(desc.rows[1].slot_page_range, 2..5);
        assert_eq!(desc.flat_slot_page_ids, vec![10, 11, 20, 21, 22]);
    }

    #[test]
    fn prefill_descriptor_uses_start_pos_as_append_position() {
        let kv = MockKvPool::new(
            4,
            vec![MockSlot {
                seq_len: 7,
                epoch: 3,
                pages: vec![30, 31],
            }],
        );
        let plan = ForwardPlan {
            mode: ForwardMode::Prefill,
            decode_rows: Vec::new(),
            prefill_rows: vec![PrefillRow {
                slot: 0,
                tokens: vec![11, 12, 13],
                start_pos: 4,
                total_tokens: 7,
                params: params(),
            }],
            microbatch: None,
            spec: None,
        };

        let desc = KvBatchDescriptor::from_plan(&plan, &kv).unwrap();
        assert_eq!(desc.flat_token_ids, vec![11, 12, 13]);
        assert_eq!(desc.flat_page_ids, vec![30, 31]);
        assert_eq!(desc.rows.len(), 1);
        let row = &desc.rows[0];
        assert_eq!(row.kind, KvBatchRowKind::Prefill);
        assert_eq!(row.seq_len, 4);
        assert_eq!(row.append_pos, 4);
        assert_eq!(row.append_len, 3);
        assert_eq!(row.slot_epoch, 3);
        assert_eq!(row.token_range, 0..3);
        assert_eq!(row.page_range, 0..2);
        assert_eq!(row.slot_page_range, 0..2);
        assert_eq!(desc.flat_slot_page_ids, vec![30, 31]);
    }

    #[test]
    fn subset_rebases_mixed_descriptor_into_single_mode_descriptors() {
        let kv = MockKvPool::new(
            4,
            vec![
                MockSlot {
                    seq_len: 7,
                    epoch: 3,
                    pages: vec![30, 31],
                },
                MockSlot {
                    seq_len: 5,
                    epoch: 7,
                    pages: vec![10, 11],
                },
                MockSlot {
                    seq_len: 9,
                    epoch: 8,
                    pages: vec![20, 21, 22],
                },
            ],
        );
        let plan = ForwardPlan {
            mode: ForwardMode::Mixed,
            decode_rows: vec![
                DecodeRow {
                    slot: 1,
                    last_token: 101,
                    kv_seq_len: 4,
                    params: params(),
                },
                DecodeRow {
                    slot: 2,
                    last_token: 202,
                    kv_seq_len: 8,
                    params: params(),
                },
            ],
            prefill_rows: vec![PrefillRow {
                slot: 0,
                tokens: vec![11, 12, 13],
                start_pos: 4,
                total_tokens: 7,
                params: params(),
            }],
            microbatch: None,
            spec: None,
        };
        let desc = KvBatchDescriptor::from_plan(&plan, &kv).unwrap();
        assert_eq!(desc.rows.len(), 3);

        // Prefill sub-descriptor: identical to a prefill-only tick's descriptor.
        let prefill = desc.subset(0..1).unwrap();
        assert!(matches!(prefill.mode, ForwardMode::Prefill));
        assert_eq!(prefill.flat_token_ids, vec![11, 12, 13]);
        assert_eq!(prefill.flat_page_ids, vec![30, 31]);
        assert_eq!(prefill.rows.len(), 1);
        assert_eq!(prefill.rows[0].kind, KvBatchRowKind::Prefill);
        assert_eq!(prefill.rows[0].slot, 0);
        assert_eq!(prefill.rows[0].token_range, 0..3);
        assert_eq!(prefill.rows[0].page_range, 0..2);
        assert_eq!(prefill.rows[0].slot_page_range, 0..2);
        assert_eq!(prefill.flat_slot_page_ids, vec![30, 31]);

        // Decode sub-descriptor: ranges rebased to its own flat buffers.
        let decode = desc.subset(1..3).unwrap();
        assert!(matches!(decode.mode, ForwardMode::Decode));
        assert_eq!(decode.flat_token_ids, vec![101, 202]);
        assert_eq!(decode.flat_page_ids, vec![10, 11, 20, 21, 22]);
        assert_eq!(decode.rows.len(), 2);
        assert_eq!(decode.rows[0].kind, KvBatchRowKind::Decode);
        assert_eq!(decode.rows[0].slot, 1);
        assert_eq!(decode.rows[0].token_range, 0..1);
        assert_eq!(decode.rows[0].page_range, 0..2);
        assert_eq!(decode.rows[0].slot_page_range, 0..2);
        assert_eq!(decode.rows[1].slot, 2);
        assert_eq!(decode.rows[1].token_range, 1..2);
        assert_eq!(decode.rows[1].page_range, 2..5);
        assert_eq!(decode.rows[1].slot_page_range, 2..5);
        assert_eq!(decode.flat_slot_page_ids, vec![10, 11, 20, 21, 22]);

        assert!(desc.subset(0..0).is_err());
        assert!(desc.subset(2..4).is_err());
    }

    #[test]
    fn descriptor_rejects_unallocated_append_span() {
        let kv = MockKvPool::new(
            4,
            vec![MockSlot {
                seq_len: 4,
                epoch: 1,
                pages: vec![40],
            }],
        );
        let plan = ForwardPlan {
            mode: ForwardMode::Decode,
            decode_rows: vec![DecodeRow {
                slot: 0,
                last_token: 9,
                kv_seq_len: 4,
                params: params(),
            }],
            prefill_rows: Vec::new(),
            microbatch: None,
            spec: None,
        };

        let err = KvBatchDescriptor::from_plan(&plan, &kv).unwrap_err();
        assert!(err.to_string().contains("needs materialized len 5"));
    }
}
