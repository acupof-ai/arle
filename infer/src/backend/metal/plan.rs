use crate::types::RequestId;

/// Logical per-row decode work selected by the Metal scheduler.
///
/// This is intentionally backend-local and CPU-only. Runtime-owned request
/// state remains the authority for MLX cache objects; the scheduler records the
/// logical row shape it selected so later Metal serving work can consume one
/// plan instead of re-deriving batch structure from legacy DTO fields.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalLogicalDecodeRow {
    /// Row position inside the scheduler-selected logical batch.
    pub row_index: usize,
    pub req_id: RequestId,
    pub input_token: u32,
    pub logical_offset: usize,
}

impl MetalLogicalDecodeRow {
    pub fn new(
        row_index: usize,
        req_id: RequestId,
        input_token: u32,
        logical_offset: usize,
    ) -> Self {
        Self {
            row_index,
            req_id,
            input_token,
            logical_offset,
        }
    }
}

/// Logical prefill work selected for a scheduler tick.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalLogicalPrefillRow {
    /// Row position inside the scheduler-selected logical batch.
    pub row_index: usize,
    pub req_id: RequestId,
    pub input_tokens: Vec<u32>,
    pub prompt_start: usize,
    pub prompt_end: usize,
    pub prompt_len: usize,
    pub logical_start: usize,
    pub logical_end: usize,
}

impl MetalLogicalPrefillRow {
    pub fn new(
        row_index: usize,
        req_id: RequestId,
        input_tokens: Vec<u32>,
        prompt_start: usize,
        prompt_end: usize,
        prompt_len: usize,
    ) -> Self {
        Self {
            row_index,
            req_id,
            input_tokens,
            prompt_start,
            prompt_end,
            prompt_len,
            logical_start: prompt_start,
            logical_end: prompt_end,
        }
    }
}

/// CPU-visible logical shape for one Metal scheduler tick.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MetalLogicalBatchShape {
    pub decode_rows: usize,
    pub prefill_rows: usize,
    pub total_rows: usize,
    pub decode_tokens: usize,
    pub prefill_tokens: usize,
    pub scheduled_tokens: usize,
}

impl MetalLogicalBatchShape {
    fn from_rows(
        decode_rows: &[MetalLogicalDecodeRow],
        prefill_rows: &[MetalLogicalPrefillRow],
    ) -> Self {
        let decode_tokens = decode_rows.len();
        let prefill_tokens = prefill_rows
            .iter()
            .map(|row| row.input_tokens.len())
            .sum::<usize>();
        Self {
            decode_rows: decode_rows.len(),
            prefill_rows: prefill_rows.len(),
            total_rows: decode_rows.len() + prefill_rows.len(),
            decode_tokens,
            prefill_tokens,
            scheduled_tokens: decode_tokens + prefill_tokens,
        }
    }
}

/// Metal-local logical serving plan for one scheduler tick.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MetalLogicalServePlan {
    pub decode_rows: Vec<MetalLogicalDecodeRow>,
    pub prefill_rows: Vec<MetalLogicalPrefillRow>,
    pub batch_shape: MetalLogicalBatchShape,
}

impl MetalLogicalServePlan {
    pub fn new(
        decode_rows: Vec<MetalLogicalDecodeRow>,
        prefill_rows: Vec<MetalLogicalPrefillRow>,
    ) -> Self {
        let batch_shape = MetalLogicalBatchShape::from_rows(&decode_rows, &prefill_rows);
        Self {
            decode_rows,
            prefill_rows,
            batch_shape,
        }
    }

    pub fn is_idle(&self) -> bool {
        self.decode_rows.is_empty() && self.prefill_rows.is_empty()
    }
}
