use super::*;

#[derive(Clone)]
pub(super) struct Dsv4DecodeBatchRow {
    pub(super) slot: usize,
    pub(super) last_token: u32,
    pub(super) start_pos: usize,
    pub(super) position: u64,
    pub(super) params: SamplingParams,
    pub(super) penalty_history: Option<std::sync::Arc<[u32]>>,
    pub(super) penalty_prompt_len: usize,
}

pub(super) struct Dsv4DecodeBatch {
    pub(super) slot_ids: Vec<usize>,
    pub(super) tokens: Vec<u32>,
    pub(super) start_positions: Vec<usize>,
    pub(super) positions: Vec<u64>,
    pub(super) rows: Vec<Dsv4DecodeBatchRow>,
}

impl Dsv4DecodeBatch {
    pub(super) fn from_rows(
        rows: &[DecodeRow],
        slots: &[crate::dsv4::Dsv4SlotState],
        num_slots: usize,
    ) -> Result<Self> {
        let mut seen = vec![false; num_slots];
        let mut slot_ids = Vec::with_capacity(rows.len());
        let mut tokens = Vec::with_capacity(rows.len());
        let mut start_positions = Vec::with_capacity(rows.len());
        let mut positions = Vec::with_capacity(rows.len());
        let mut batch_rows = Vec::with_capacity(rows.len());
        for row in rows {
            ensure!(
                row.slot < num_slots,
                "decode slot {} outside DSv4 executor slots {}",
                row.slot,
                num_slots
            );
            ensure!(
                !seen[row.slot],
                "DSv4 decode batch contains duplicate slot {}",
                row.slot
            );
            seen[row.slot] = true;
            ensure!(
                slots[row.slot].seq_len() == row.kv_seq_len,
                "DSv4 materialized state len {} != DecodeRow.kv_seq_len {} for slot {}",
                slots[row.slot].seq_len(),
                row.kv_seq_len,
                row.slot
            );
            let position = row.kv_seq_len.saturating_add(1) as u64;
            slot_ids.push(row.slot);
            tokens.push(row.last_token);
            start_positions.push(row.kv_seq_len);
            positions.push(position);
            batch_rows.push(Dsv4DecodeBatchRow {
                slot: row.slot,
                last_token: row.last_token,
                start_pos: row.kv_seq_len,
                position,
                params: row.params.clone(),
                penalty_history: row.penalty_history.clone(),
                penalty_prompt_len: row.penalty_prompt_len,
            });
        }
        Ok(Self {
            slot_ids,
            tokens,
            start_positions,
            positions,
            rows: batch_rows,
        })
    }
}

pub(super) struct KvRowExpect<'a> {
    kind: KvBatchRowKind,
    pub(super) slot: usize,
    pos: usize,
    append_len: usize,
    tokens: &'a [u32],
}

impl<'a> KvRowExpect<'a> {
    pub(super) fn decode(row: &'a DecodeRow) -> Self {
        Self {
            kind: KvBatchRowKind::Decode,
            slot: row.slot,
            pos: row.kv_seq_len,
            append_len: 1,
            tokens: std::slice::from_ref(&row.last_token),
        }
    }

    pub(super) fn prefill(row: &'a infer_plan::PrefillRow) -> Self {
        Self {
            kind: KvBatchRowKind::Prefill,
            slot: row.slot,
            pos: row.start_pos,
            append_len: row.tokens.len(),
            tokens: &row.tokens,
        }
    }
}

pub(super) fn validate_dsv4_kv_batch(
    expect: &[KvRowExpect<'_>],
    kv_batch: &KvBatchDescriptor,
) -> Result<()> {
    ensure!(
        kv_batch.rows.len() == expect.len(),
        "DSv4 KV batch row count {} != plan rows {}",
        kv_batch.rows.len(),
        expect.len()
    );
    for (idx, (want, kv_row)) in expect.iter().zip(&kv_batch.rows).enumerate() {
        ensure!(
            kv_row.kind == want.kind,
            "DSv4 KV batch row {idx} has kind {:?}, plan says {:?}",
            kv_row.kind,
            want.kind
        );
        ensure!(
            kv_row.slot == want.slot,
            "DSv4 KV batch row {idx} slot {} != plan slot {}",
            kv_row.slot,
            want.slot
        );
        ensure!(
            kv_row.seq_len == want.pos && kv_row.append_pos == want.pos,
            "DSv4 KV batch row {idx} seq/append ({},{}) != plan pos {}",
            kv_row.seq_len,
            kv_row.append_pos,
            want.pos
        );
        ensure!(
            kv_row.append_len == want.append_len,
            "DSv4 KV batch row {idx} append_len {} != plan token count {}",
            kv_row.append_len,
            want.append_len
        );
        ensure!(
            kv_row.page_range.start < kv_row.page_range.end,
            "DSv4 KV batch row {idx} has empty page range"
        );
        let tokens = &kv_batch.flat_token_ids[kv_row.token_range.clone()];
        ensure!(
            tokens == want.tokens,
            "DSv4 KV batch row {idx} tokens do not match plan tokens"
        );
    }
    Ok(())
}

pub(super) fn validate_dsv4_kv_view(
    expect: &[KvRowExpect<'_>],
    view: &crate::attention::Dsv4KvBatchView,
) -> Result<()> {
    ensure!(
        view.rows.len() == expect.len(),
        "DSv4 KV adapter view row count {} != plan rows {}",
        view.rows.len(),
        expect.len()
    );
    for (idx, (want, view_row)) in expect.iter().zip(&view.rows).enumerate() {
        ensure!(
            view_row.kind == want.kind,
            "DSv4 KV adapter row {idx} has kind {:?}, plan says {:?}",
            view_row.kind,
            want.kind
        );
        ensure!(
            view_row.slot == want.slot
                && view_row.seq_len == want.pos
                && view_row.append_pos == want.pos
                && view_row.append_len == want.append_len,
            "DSv4 KV adapter row {idx} slot/seq/append ({},{},{},{}) != plan ({},{},{})",
            view_row.slot,
            view_row.seq_len,
            view_row.append_pos,
            view_row.append_len,
            want.slot,
            want.pos,
            want.append_len
        );
        ensure!(
            view_row.page_range.end <= view.flat_page_ids.len(),
            "DSv4 KV adapter row {idx} page range {:?} outside flat page len {}",
            view_row.page_range,
            view.flat_page_ids.len()
        );
        ensure!(
            !view.flat_page_ids[view_row.page_range.clone()].is_empty(),
            "DSv4 KV adapter row {idx} has no page ids"
        );
        ensure!(
            view_row.slot_page_range.end <= view.flat_slot_page_ids.len(),
            "DSv4 KV adapter row {idx} slot page range {:?} outside flat slot page len {}",
            view_row.slot_page_range,
            view.flat_slot_page_ids.len()
        );
        ensure!(
            !view.flat_slot_page_ids[view_row.slot_page_range.clone()].is_empty(),
            "DSv4 KV adapter row {idx} has no slot page ids"
        );
    }
    Ok(())
}
