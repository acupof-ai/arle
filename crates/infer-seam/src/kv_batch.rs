//! Host-only KV batch descriptors.
//!
//! The engine builds this from a [`ForwardPlan`](infer_plan::ForwardPlan) and a
//! [`KvPool`](crate::KvPool) after logical KV allocation; backends lower the
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvBatchRowKind {
    Prefill,
    Decode,
}

impl KvBatchDescriptor {
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
                .extend_from_slice(kv.page_indices_for_token_range(slot, append_end));
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
